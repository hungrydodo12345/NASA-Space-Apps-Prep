"""Whole-flight tracker with full homographies (the helicopter tilts while it accelerates, so the ground is seen in perspective and a similarity
model rejects most good links; see track_flight.py for why the fixed-pattern removal and the pose graph are needed).

1. Frames: flattened, fixed pattern (temporal median) subtracted.
2. Pairs (i, i+1), (i, i+2), (i, i+3): SIFT (5000 features, ratio 0.85) -> RANSAC homography (4 px). Accepted with >= 40 inliers spread over the frame.
3. Pose graph over per-frame homographies H_i (frame i -> reference frame); residuals are measured in frame a's own pixels,
   T_a^-1 T_b vs the measured map at five frame points, robust (soft-L1). Initialised by chaining the lag-1 links, bridging gaps with lag-2/3 links.
Usage: python track_flight_h.py <scratch_dir> <first> <last> <ref> <out_npz>
"""
import sys, json, numpy as np, cv2
from scipy.optimize import least_squares
S, I0, I1, REF, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
order = json.load(open(f"{S}/ingenuity/nav_order.json")); med = np.load(f"{S}/ingenuity/fixed_pattern.npy")
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__))); from rectify import rectify, OUT_W, OUT_H
idx = list(range(I0, I1 + 1)); n = len(idx)
def clean(i):
    a = cv2.imread(f"{S}/ingenuity/{order[i]}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32); f = rectify(a / (cv2.GaussianBlur(a, (0, 0), 45) + 1e-3) - med)
    m = rectify(np.ones_like(a)) > 0.99; m = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0; return np.where(m, f, 0.0).astype(np.float32)
F = [clean(i) for i in idx]; h, w = F[0].shape
u8 = lambda a: np.clip((a + 0.7) / 1.4 * 255, 0, 255).astype(np.uint8)
sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.005); cl = cv2.createCLAHE(3.0, (4, 4)); K = [sift.detectAndCompute(cl.apply(u8(f)), None) for f in F]
MASK = cv2.erode((rectify(np.ones((480, 640), np.float32)) > 0.99).astype(np.uint8), np.ones((15, 15), np.uint8)) * 255
sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.005); K = [sift.detectAndCompute(cl.apply(u8(f)), MASK) for f in F]
def measure(a, b):                                           # homography mapping frame b pixels -> frame a pixels
    (k1, d1), (k2, d2) = K[a], K[b]
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10: return None
    mt = [x for x, y in (z for z in cv2.BFMatcher().knnMatch(d2, d1, k=2) if len(z) == 2) if x.distance < 0.85 * y.distance]
    if len(mt) < 20: return None
    p2 = np.float32([k2[m.queryIdx].pt for m in mt]); p1 = np.float32([k1[m.trainIdx].pt for m in mt]); H, inl = cv2.findHomography(p2, p1, cv2.RANSAC, 4.0, maxIters=5000)
    if H is None or inl.sum() < 40: return None
    q = p2[inl.ravel() > 0]
    if q[:, 0].std() < 50 or q[:, 1].std() < 40: return None
    return H / H[2, 2], int(inl.sum())
meas = []
for lag in (1, 2, 3):
    for a in range(n - lag):
        r = measure(a, a + lag)
        if r is not None: meas.append((a, a + lag, r[0], r[1]))
print("measurements:", len(meas), "(lag1 %d, lag2 %d, lag3 %d) for %d frames" % tuple([sum(m[1] - m[0] == l for m in meas) for l in (1, 2, 3)] + [n]), flush=True)
pts = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1], [w / 2, h / 2, 1]], float).T
SC = np.array([1, 1, 1, 1, 1, 1, 1000., 1000.])              # scale the perspective terms so all parameters are O(1)
def mat(p): q = p / SC; return np.array([[q[0], q[1], q[2]], [q[3], q[4], q[5]], [q[6], q[7], 1.0]])
def proj(Hm, X): Y = Hm @ X; return Y[:2] / Y[2]
ref_i = REF - I0
def resid(x):
    P = x.reshape(n, 8); T = [mat(P[i]) for i in range(n)]; Ti = [np.linalg.inv(t) for t in T]; r = []
    for a, b, M, ni in meas: r.append((proj(Ti[a] @ T[b], pts) - proj(M, pts)).ravel())
    r.append((P[ref_i] - np.array([1, 0, 0, 0, 1, 0, 0, 0]) * SC * np.array([1, 1, 1, 1, 1, 1, 1, 1])) * 10)
    return np.concatenate(r)
# initialisation: chain from the reference using the best available link (lag 1, else 2, else 3)
byb = {}
for a, b, M, ni in meas: byb.setdefault(b, []).append((b - a, a, M))
T0 = {ref_i: np.eye(3)}
for j in range(ref_i + 1, n):
    c = sorted(byb.get(j, [])); c = [x for x in c if x[1] in T0]
    if not c: continue
    _, a, M = c[0]; T0[j] = T0[a] @ M
bya = {}
for a, b, M, ni in meas: bya.setdefault(a, []).append((b - a, b, M))
for j in range(ref_i - 1, -1, -1):
    c = sorted(bya.get(j, [])); c = [x for x in c if x[1] in T0]
    if not c: continue
    _, b, M = c[0]; T0[j] = T0[b] @ np.linalg.inv(M)
x0 = np.tile(np.array([1, 0, 0, 0, 1, 0, 0, 0.]) * SC, (n, 1))
for j, Tj in T0.items(): Tj = Tj / Tj[2, 2]; x0[j] = np.array([Tj[0, 0], Tj[0, 1], Tj[0, 2], Tj[1, 0], Tj[1, 1], Tj[1, 2], Tj[2, 0], Tj[2, 1]]) * SC
print("initial chain reaches", len(T0), "of", n, "frames", flush=True)
sol = least_squares(resid, x0.ravel(), loss="soft_l1", f_scale=3.0, max_nfev=60, x_scale="jac")
for it in range(3):
    P = sol.x.reshape(n, 8); T = [mat(P[i]) for i in range(n)]; Ti = [np.linalg.inv(t) for t in T]
    err = np.array([np.abs(proj(Ti[a] @ T[b], pts) - proj(M, pts)).max() for a, b, M, ni in meas]); keep = [m for m, e in zip(meas, err) if e < 10.0]
    print(f"pass {it}: {len(meas) - len(keep)} of {len(meas)} measurements rejected (median err {np.median(err):.1f} px)", flush=True)
    if len(keep) == len(meas): break
    meas = keep; sol = least_squares(resid, sol.x, loss="soft_l1", f_scale=3.0, max_nfev=60, x_scale="jac")
P = sol.x.reshape(n, 8); res = np.abs(resid(sol.x)[:-8]); print("pose graph residual: median %.2f px, 95th %.2f px" % (np.median(res), np.percentile(res, 95)))
par = list(range(n)); f_ = lambda x: par[x] if par[x] == x else f_(par[x])
for a, b, M, ni in meas: par[f_(a)] = f_(b)
comp = {}
for i in range(n): comp.setdefault(f_(i), []).append(i)
big = sorted(comp.values(), key=len, reverse=True)[:3]; print("largest connected groups (frame idx ranges):", [(min(c) + I0, max(c) + I0, len(c)) for c in big])
np.savez_compressed(OUT, params=P, idx=np.array(idx), ref=REF, crop=np.array([0, OUT_H, 0, OUT_W]), kept=np.array([(a, b, ni) for a, b, M, ni in meas]), groups=np.array([f_(i) for i in range(n)]))

"""Frame-to-frame motion of Ingenuity's downward camera over a whole flight, robust to camera-fixed features and featureless ground.

Why a new tracker: matching consecutive frames with SIFT alone returns zero motion when the ground is smooth, because the helicopter's own
shadow, lens marks and vignette are fixed in every frame. Here:
  1. Each frame is flattened (vignette divided out) and the fixed pattern (temporal median over the flight) is subtracted.
  2. For frame pairs (i, i+1), (i, i+2), (i, i+3): SIFT matches on the cleaned frame -> RANSAC 4-parameter similarity (a camera looking
     straight down sees the ground through a similarity); inliers must be >= 15 and spread over the frame. Where SIFT fails, phase correlation
     of the cleaned high-passed frames gives a translation (accepted only with a clear peak).
  3. All pairwise measurements are combined in one robust least-squares pose graph (per-frame similarity: tx, ty, rotation, log scale), so a bad link is
     out-voted by the redundant ones and drift is spread instead of accumulated.
Output: poses (frame -> reference-frame coordinates), per-pair diagnostics. Usage: python track_flight.py <scratch_dir> <first> <last> <ref> <out_npz>
"""
import sys, json, numpy as np, cv2
from scipy.optimize import least_squares
S, I0, I1, REF, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
order = json.load(open(f"{S}/ingenuity/nav_order.json")); med = np.load(f"{S}/ingenuity/fixed_pattern.npy")
CROP = (slice(40, 440), slice(40, 600)); idx = list(range(I0, I1 + 1)); n = len(idx)
def clean(i):
    a = cv2.imread(f"{S}/ingenuity/{order[i]}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32); f = a / (cv2.GaussianBlur(a, (0, 0), 45) + 1e-3) - med
    f = f[CROP]; return f
F = [clean(i) for i in idx]; h, w = F[0].shape
u8 = lambda a: np.clip((a + 0.7) / 1.4 * 255, 0, 255).astype(np.uint8)
sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.008); cl = cv2.createCLAHE(3.0, (4, 4)); K = [sift.detectAndCompute(cl.apply(u8(f)), None) for f in F]
hpf = lambda a: (a - cv2.GaussianBlur(a, (0, 0), 5)).astype(np.float32); win = cv2.createHanningWindow((w, h), cv2.CV_32F)
def measure(a, b):
    """similarity mapping frame b's pixel coords -> frame a's pixel coords; returns (2x3 matrix, n_inliers, kind) or None"""
    (k1, d1), (k2, d2) = K[a], K[b]
    if d1 is not None and d2 is not None and len(k1) > 8 and len(k2) > 8:
        mt = [x for x, y in (z for z in cv2.BFMatcher().knnMatch(d2, d1, k=2) if len(z) == 2) if x.distance < 0.8 * y.distance]
        if len(mt) >= 12:
            p2 = np.float32([k2[m.queryIdx].pt for m in mt]); p1 = np.float32([k1[m.trainIdx].pt for m in mt])
            M, inl = cv2.estimateAffinePartial2D(p2, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=3000)
            if M is not None and inl.sum() >= 15:
                q = p2[inl.ravel() > 0]
                if q[:, 0].std() > 40 and q[:, 1].std() > 30 and 0.8 < np.hypot(M[0, 0], M[1, 0]) < 1.25: return M, int(inl.sum()), "sift"
    (sx, sy), resp = cv2.phaseCorrelate(hpf(F[a]), hpf(F[b]), win)                      # b(x) = a(x - s)  =>  b -> a is x - s
    if resp > 0.12: return np.array([[1, 0, -sx], [0, 1, -sy]], float), int(resp * 100), "phase"
    return None
meas = []
for lag in (1, 2, 3):
    for a in range(n - lag):
        r = measure(a, a + lag)
        if r is not None: meas.append((a, a + lag, r[0], r[1], r[2]))
kinds = {k: sum(m[4] == k for m in meas) for k in ("sift", "phase")}; print("measurements:", len(meas), kinds, "for", n, "frames", flush=True)
def mat(p):                       # per-frame parameters (tx, ty, theta, log s) -> 3x3 mapping frame coords -> reference coords
    c, s = np.exp(p[3]) * np.cos(p[2]), np.exp(p[3]) * np.sin(p[2]); return np.array([[c, -s, p[0]], [s, c, p[1]], [0, 0, 1]])
ref_i = REF - I0; pts = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1], [w / 2, h / 2, 1]], float).T
def resid(x):
    """Residuals are measured in frame a's own pixels (T_a^-1 T_b applied to frame-b points vs the measured map), so shrinking the whole chain
    to a point cannot lower them; a weak prior keeps the scale (flight height) near its initial value, with a slow-change term."""
    P = x.reshape(n, 4); T = [mat(P[i]) for i in range(n)]; Ti = [np.linalg.inv(t) for t in T]; r = []
    for a, b, M, ni, kind in meas:
        pred = (Ti[a] @ T[b] @ pts)[:2]; obs = (np.vstack([M, [0, 0, 1]]) @ pts)[:2]; wgt = 1.0 if kind == "sift" else 0.5
        r.append(((pred - obs) * wgt).ravel())
    r.append(P[ref_i]); r.append(30.0 * (P[:, 3] - 0.0)); r.append(200.0 * np.diff(P[:, 3]))                       # scale prior and smoothness
    return np.concatenate(r)
x0 = np.zeros((n, 4)); T = {ref_i: np.eye(3)}                                               # initial guess: chain the lag-1 measurements
lag1 = {m[1]: m for m in meas if m[1] - m[0] == 1}
for j in range(ref_i + 1, n):
    if j in lag1: T[j] = T[j - 1] @ np.vstack([lag1[j][2], [0, 0, 1]])
    else: break
for j in range(ref_i - 1, -1, -1):
    if j + 1 in lag1: T[j] = T[j + 1] @ np.linalg.inv(np.vstack([lag1[j + 1][2], [0, 0, 1]]))
    else: break
for j, Tj in T.items(): x0[j] = [Tj[0, 2], Tj[1, 2], np.arctan2(Tj[1, 0], Tj[0, 0]), np.log(np.hypot(Tj[0, 0], Tj[1, 0]))]
def solve(meas_, x_init):
    global meas
    meas = meas_; return least_squares(resid, x_init, loss="soft_l1", f_scale=3.0, max_nfev=80, verbose=0)
sol = solve(meas, x0.ravel()); all_meas = list(meas)
for it in range(3):                                                  # drop measurements the solution cannot reconcile (>12 px at the frame corners) and re-solve
    P = sol.x.reshape(n, 4); T = [mat(P[i]) for i in range(n)]; Ti = [np.linalg.inv(t) for t in T]
    err = np.array([np.abs((Ti[a] @ T[b] @ pts)[:2] - (np.vstack([M, [0, 0, 1]]) @ pts)[:2]).max() for a, b, M, ni, kind in all_meas])
    keep = [m for m, e in zip(all_meas, err) if e < 12.0]; print(f"pass {it}: {len(all_meas) - len(keep)} of {len(all_meas)} measurements rejected (median error {np.median(err):.1f} px)", flush=True)
    if len(keep) == len(all_meas): break
    all_meas = keep; sol = solve(all_meas, sol.x)
meas = all_meas; P = sol.x.reshape(n, 4)
res = resid(sol.x)[:-(4 + n + n - 1)]; print("pose graph: median residual %.2f px, 95th %.2f px" % (np.median(np.abs(res)), np.percentile(np.abs(res), 95)), flush=True)
tr = P[:, :2]; step = np.hypot(*np.diff(tr, axis=0).T); print("path length %.0f px (~%.0f m at 2.7 cm/px); scale range %.3f-%.3f; rotation range %.1f deg" % (step.sum(), step.sum() * 0.027, np.exp(P[:, 3]).min(), np.exp(P[:, 3]).max(), np.degrees(np.ptp(np.unwrap(P[:, 2])))))
np.savez_compressed(OUT, params=P, kept=np.array([(a, b) for a, b, *_ in meas]), idx=np.array(idx), ref=REF, crop=np.array([40, 440, 40, 600]), meas=np.array([(a, b, ni) for a, b, _, ni, _ in meas]), kinds=np.array([k for *_, k in meas]))

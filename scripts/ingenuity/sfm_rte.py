"""Structure from motion on the 10 colour frames (HELI_RTE) of Ingenuity flight 26.

Camera model (rte_camera.py): in the pinhole frame (x/z, y/z) the CAHVOR distortion reduces to a radial model x_d = x_u (1 + R0 + R1 r^2 + R2 r^4)
(O is 0.74 deg from A: ignored), pixel = (fx x_d + cx, fy y_d + cy), with f ~ 1780 px, R1 = 0.47, R2 = -0.053.
Pipeline: SIFT (half resolution) -> all-pairs matching -> essential-matrix verification -> incremental reconstruction (best pair, PnP) -> bundle adjustment
(robust, sparse) with the radial coefficient R1 and the focal length free. Output: poses (camera from world), 3D points, per-point colour, and statistics.
The reconstruction has an arbitrary scale (|t| between the first two cameras = 1); the metric scale is set later from the HiRISE geometry.
Usage: python sfm_rte.py <scratch_dir> <out_npz>
"""
import sys, json, numpy as np, cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__))); from rte_camera import parse
S, OUT = sys.argv[1:3]
meta = json.load(open(f"{S}/ingenuity/sol414_meta.json")); rte = sorted([i for i in meta if i["camera"]["instrument"] == "HELI_RTE"], key=lambda i: float(i["imageid"].split("_")[2]))
C, A, H, V, O, R, E = parse(rte[0]["camera"]["camera_model_component_list"]); fx = np.linalg.norm(np.cross(A, H)); fy = np.linalg.norm(np.cross(A, V)); cx = A @ H; cy = A @ V
K0 = np.array([fx, fy, cx, cy, R[0], R[1], R[2]])                                        # intrinsics vector
def project(X, rv, tv, K):
    Rm = cv2.Rodrigues(rv)[0]; Xc = X @ Rm.T + tv; xu = Xc[:, :2] / Xc[:, 2:3]; r2 = (xu ** 2).sum(1)[:, None]; xd = xu * (1 + K[4] + K[5] * r2 + K[6] * r2 ** 2)
    return np.stack([K[0] * xd[:, 0] + K[2], K[1] * xd[:, 1] + K[3]], 1), Xc[:, 2]
def undist(px, K):
    xd = np.stack([(px[:, 0] - K[2]) / K[0], (px[:, 1] - K[3]) / K[1]], 1); xu = xd.copy()
    for _ in range(40): r2 = (xu ** 2).sum(1)[:, None]; xu = xd / (1 + K[4] + K[5] * r2 + K[6] * r2 ** 2)
    return xu
imgs = [cv2.imread(f"{S}/ingenuity/{i['imageid']}.png") for i in rte]; n = len(imgs); print("images", n, imgs[0].shape, flush=True)
sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.01); cl = cv2.createCLAHE(3.0, (8, 8)); kps, des = [], []
for im in imgs:
    g = cv2.cvtColor(cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY); g = cl.apply(g); k, d = sift.detectAndCompute(g, None); kps.append(np.float32([p.pt for p in k]) * 2.0); des.append(d)
print("features", [len(k) for k in kps], flush=True)
pairs = {}
bf = cv2.BFMatcher()
for a in range(n):
    for b in range(a + 1, n):
        mt = [x for x, y in (z for z in bf.knnMatch(des[a], des[b], k=2) if len(z) == 2) if x.distance < 0.8 * y.distance]
        if len(mt) < 30: continue
        p1 = kps[a][[m.queryIdx for m in mt]]; p2 = kps[b][[m.trainIdx for m in mt]]; x1, x2 = undist(p1, K0), undist(p2, K0)
        E_, inl = cv2.findEssentialMat(x1.astype(np.float64), x2.astype(np.float64), np.eye(3), cv2.RANSAC, 0.999, 3.0 / 1780)
        if E_ is None or inl is None or inl.sum() < 30: continue
        inl = inl.ravel() > 0; pairs[(a, b)] = (np.array([[m.queryIdx, m.trainIdx] for m in mt])[inl], int(inl.sum()))
print("verified pairs:", {k: v[1] for k, v in sorted(pairs.items(), key=lambda t: -t[1][1])[:12]}, "... total", len(pairs), flush=True)
# tracks by union-find over (image, keypoint)
par = {}
def find(x):
    while par.setdefault(x, x) != x: par[x] = par[par[x]]; x = par[x]
    return x
for (a, b), (mm, _) in pairs.items():
    for i, j in mm: par[find((a, int(i)))] = find((b, int(j)))
tr = {}
for x in list(par): tr.setdefault(find(x), []).append(x)
tracks = [t for t in tr.values() if len({im for im, _ in t}) == len(t) and len(t) >= 2]; print("tracks:", len(tracks), "with >=3 views:", sum(len(t) >= 3 for t in tracks), flush=True)
# initial pair
(a0, b0) = max(pairs, key=lambda k: pairs[k][1] * (1 if True else 0)); best = None
for (a, b), (mm, ni) in sorted(pairs.items(), key=lambda t: -t[1][1])[:8]:
    x1, x2 = undist(kps[a][mm[:, 0]], K0), undist(kps[b][mm[:, 1]], K0); E_, inl = cv2.findEssentialMat(x1, x2, np.eye(3), cv2.RANSAC, 0.999, 3.0 / 1780); _, Rr, tt, mk = cv2.recoverPose(E_, x1, x2, np.eye(3), mask=inl.copy())
    ang = np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1)))
    if best is None or (mk > 0).sum() * min(1.0, 0.3 + abs(tt[2, 0])) > best[0]: best = ((mk > 0).sum() * min(1.0, 0.3 + abs(tt[2, 0])), a, b, Rr, tt.ravel(), ang)
_, a0, b0, Rr, tt, ang = best; print("initial pair", a0, b0, "rotation %.1f deg" % ang, flush=True)
cam = {a0: (np.zeros(3), np.zeros(3)), b0: (cv2.Rodrigues(Rr)[0].ravel(), tt)}
def tri(i, j, xi, xj):
    P1 = np.hstack([cv2.Rodrigues(cam[i][0])[0], cam[i][1][:, None]]); P2 = np.hstack([cv2.Rodrigues(cam[j][0])[0], cam[j][1][:, None]]); X = cv2.triangulatePoints(P1, P2, xi.T.astype(np.float64), xj.T.astype(np.float64)); return (X[:3] / X[3]).T
pts = {}; obs = {}
def add_tracks():
    new = 0
    for ti, t in enumerate(tracks):
        views = [(im, k) for im, k in t if im in cam]
        if len(views) < 2 or ti in pts: continue
        (i, ki), (j, kj) = views[0], views[1]; X = tri(i, j, undist(kps[i][[ki]], K0), undist(kps[j][[kj]], K0))[0]
        ok = True; rr = []
        for im, k in views:
            p, z = project(X[None], cam[im][0], cam[im][1], K0)
            if z[0] <= 0: ok = False; break
            rr.append(np.linalg.norm(p[0] - kps[im][k]))
        if ok and max(rr) < 6.0: pts[ti] = X; new += 1
    return new
print("initial points:", add_tracks(), flush=True)
rest = [i for i in range(n) if i not in cam]
while rest:
    cnt = {}
    for i in rest:
        c = [(ti, k) for ti, t in enumerate(tracks) if ti in pts for im, k in t if im == i]; cnt[i] = c
    i = max(rest, key=lambda r: len(cnt[r]))
    if len(cnt[i]) < 12: print("cannot register remaining images", rest, flush=True); break
    X3 = np.float64([pts[ti] for ti, _ in cnt[i]]); x2 = np.float64([kps[i][k] for _, k in cnt[i]])
    ok, rv, tv, inl = cv2.solvePnPRansac(X3, x2, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.array([K0[5], K0[6], 0, 0], float), reprojectionError=8.0, iterationsCount=2000)
    if not ok or inl is None or len(inl) < 10: rest.remove(i); print("PnP failed for", i, flush=True); continue
    cam[i] = (rv.ravel(), tv.ravel()); rest.remove(i); print("registered image", i, "with", len(inl), "of", len(cnt[i]), "correspondences;  +", add_tracks(), "points", flush=True)
ids = sorted(cam); pid = sorted(pts); print("cameras", ids, "points", len(pid), flush=True)
# bundle adjustment: camera poses, points, focal length (shared) and R1 (shared)
cidx = {c: k for k, c in enumerate(ids)}; pidx = {p: k for k, p in enumerate(pid)}; ob = []
for ti in pid:
    for im, k in tracks[ti]:
        if im in cam: ob.append((cidx[im], pidx[ti], kps[im][k]))
oc = np.array([o[0] for o in ob]); op = np.array([o[1] for o in ob]); ox = np.array([o[2] for o in ob])
x0 = np.concatenate([np.concatenate([cam[c][0], cam[c][1]]) for c in ids] + [np.concatenate([pts[p] for p in pid]), [fx, K0[5], K0[6]]])
nc, npnt = len(ids), len(pid)
def unpack(x): return x[:6 * nc].reshape(nc, 6), x[6 * nc:6 * nc + 3 * npnt].reshape(npnt, 3), x[6 * nc + 3 * npnt:]
def resid(x):
    C_, X_, g = unpack(x); K = K0.copy(); K[0] = g[0]; K[1] = g[0] * fy / fx; K[5] = g[1]; K[6] = g[2]; r = np.empty((len(ob), 2))
    for c in range(nc):
        m = oc == c
        if m.any(): p, z = project(X_[op[m]], C_[c, :3], C_[c, 3:], K); r[m] = p - ox[m]
    return np.concatenate([r.ravel(), [(np.linalg.norm(C_[1, 3:]) - np.linalg.norm(cam[ids[1]][1])) * 1e3]])
sp = lil_matrix((2 * len(ob) + 1, len(x0)), dtype=int)
for k in range(len(ob)):
    for q in range(2):
        r_ = 2 * k + q; sp[r_, 6 * oc[k]:6 * oc[k] + 6] = 1; sp[r_, 6 * nc + 3 * op[k]:6 * nc + 3 * op[k] + 3] = 1; sp[r_, 6 * nc + 3 * npnt:] = 1
sp[2 * len(ob), 6 + 3:6 + 6] = 1
sol = least_squares(resid, x0, jac_sparsity=sp, loss="soft_l1", f_scale=2.0, x_scale="jac", max_nfev=40, verbose=0)
C_, X_, g = unpack(sol.x); rr = np.abs(resid(sol.x)[:-1]); print("BA: focal %.1f (metadata %.1f), R1 %.3f (%.3f), R2 %.3f; reprojection median %.2f px, 95th %.2f px, %d observations" % (g[0], fx, g[1], K0[5], g[2], np.median(rr), np.percentile(rr, 95), len(ob)))
good = np.ones(npnt, bool)
np.savez_compressed(OUT, cams=C_, ids=np.array(ids), points=X_, intr=np.array([g[0], g[0] * fy / fx, cx, cy, K0[4], g[1], g[2]]), image_ids=np.array([r["imageid"] for r in rte]))

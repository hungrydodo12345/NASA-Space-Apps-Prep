"""Metric top-down mosaic from the homography pose graph (track_flight_h.py).
Every rectified, flattened, fixed-pattern-free frame is warped into the reference frame's pinhole plane with its pose-graph homography and blended with a
weight that favours the frame centre. The reference frame is taken as level (the helicopter hovers there); the ground sampling is nominal altitude / focal =
8 m / 275.4 px = 0.0291 m per pixel (to be refined by the HiRISE registration). A tilt of the reference frame appears as a small perspective error in the mosaic.
Usage: python mosaic_from_poses.py <scratch_dir> <track_npz> <first_idx> <last_idx> <out_npz> [max_side_px]
"""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from rectify import rectify, OUT_W, OUT_H
S, TR, A0, A1, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]; MAXS = int(sys.argv[6]) if len(sys.argv) > 6 else 4500
order = json.load(open(f"{S}/ingenuity/nav_order.json")); med = np.load(f"{S}/ingenuity/fixed_pattern.npy"); d = np.load(TR); P = d["params"]; idx0 = int(d["idx"][0]); ref = int(d["ref"]); grp = d["groups"]
SC = np.array([1, 1, 1, 1, 1, 1, 1000., 1000.])
def T(i): q = P[i - idx0] / SC; return np.array([[q[0], q[1], q[2]], [q[3], q[4], q[5]], [q[6], q[7], 1.0]])
rg = grp[ref - idx0]; use = [i for i in range(A0, A1 + 1) if grp[i - idx0] == rg]; print("frames in the reference's connected group:", len(use), "of", A1 - A0 + 1)
cs = np.array([[0, 0, 1], [OUT_W, 0, 1], [OUT_W, OUT_H, 1], [0, OUT_H, 1]], float).T
pts = []
for i in use:
    Y = T(i) @ cs; pts.append(Y[:2] / Y[2])
pts = np.concatenate(pts, 1); (x0, y0), (x1, y1) = np.percentile(pts, 1, axis=1), np.percentile(pts, 99, axis=1); W = int(x1 - x0); H = int(y1 - y0); sc = min(1.0, MAXS / max(W, H))
print("mosaic %d x %d px at reference scale, downscaled by %.2f" % (W, H, sc)); W2, H2 = int(W * sc), int(H * sc)
Sh = np.array([[sc, 0, -x0 * sc], [0, sc, -y0 * sc], [0, 0, 1]], float); acc = np.zeros((H2, W2), np.float32); wt = np.zeros((H2, W2), np.float32)
yy, xx = np.mgrid[0:OUT_H, 0:OUT_W]; r = np.hypot(xx - OUT_W / 2, yy - OUT_H / 2); wmap = np.clip(1 - r / 230.0, 0, 1) ** 2 * (rectify(np.ones((480, 640), np.float32)) > 0.99)
for i in use:
    a = cv2.imread(f"{S}/ingenuity/{order[i]}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32); f = rectify(a / (cv2.GaussianBlur(a, (0, 0), 45) + 1e-3) - med) + 1.0
    M = Sh @ T(i); acc += cv2.warpPerspective((f * wmap).astype(np.float32), M, (W2, H2)); wt += cv2.warpPerspective(wmap.astype(np.float32), M, (W2, H2))
mos = np.where(wt > 0.02, acc / np.maximum(wt, 1e-6), np.nan).astype(np.float32); mpp = 8.0 / 275.4 / sc
np.savez_compressed(OUT, mosaic=mos, weight=wt, used=np.array(use), ref=ref, m_per_px=mpp, origin=np.array([x0, y0]), scale=sc)
v = np.nan_to_num(mos, nan=1.0); lo, hi = np.percentile(v[wt > 0.02], [1, 99.7]); cv2.imwrite(OUT.replace(".npz", ".png"), (np.clip((v - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)); print("wrote", OUT, "m/px", round(mpp, 4))

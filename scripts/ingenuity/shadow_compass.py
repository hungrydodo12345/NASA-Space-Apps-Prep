"""Robust helicopter-shadow compass for the backshell stretch of the flight.

Per frame (rectified, flattened) the shadow is a small dark cross about 150 px from the nadir point. A single darkest-blob detector locks onto rocks in several frames,
so here: (1) take the 8 best candidates per frame inside the radius window (120-190 px) from a cross-shaped detector (dark at scale 2-3 px, small bright-free surroundings);
(2) carry every candidate's direction into the mosaic frame with the pose-graph map; (3) the sun is fixed, so the true shadow has one direction in the mosaic frame for all frames:
find it by a weighted circular vote over all candidates, then keep per frame only the candidate within 20 deg of the consensus; (4) report the consensus, its spread and the
rotation of the mosaic to north-up given the computed sun azimuth.
Usage: python shadow_compass.py <scratch_dir> <track_npz> <first> <last> <sun_azimuth_deg> <out_json>"""
import sys, os, json, numpy as np, cv2
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from rectify import rectify, OUT_W, OUT_H
S, TR, A0, A1, SUN, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
order = json.load(open(f"{S}/ingenuity/nav_order.json")); d = np.load(TR); P = d["params"]; idx0 = int(d["idx"][0]); SCL = np.array([1, 1, 1, 1, 1, 1, 1000., 1000.])
def T(i): q = P[i - idx0] / SCL; return np.array([[q[0], q[1], q[2]], [q[3], q[4], q[5]], [q[6], q[7], 1.0]])
valid = cv2.erode((rectify(np.ones((480, 640), np.float32)) > 0.99).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0; cx, cy = OUT_W / 2, OUT_H / 2
yy, xx = np.mgrid[0:OUT_H, 0:OUT_W]; rr = np.hypot(xx - cx, yy - cy); ring = (rr > 120) & (rr < 195) & valid
# cross template: two thin dark bars (the rotor shadows) ~30 px long, 4 px wide, at any rotation: correlate with a bank of rotations and take the max
def cross(ang, L=30, w=2.0, size=41):
    t = np.zeros((size, size), np.float32); c = size // 2; a = np.radians(ang)
    for k in (0, 90):
        dx, dy = np.cos(a + np.radians(k)), np.sin(a + np.radians(k)); cv2.line(t, (int(c - L / 2 * dx), int(c - L / 2 * dy)), (int(c + L / 2 * dx), int(c + L / 2 * dy)), 1.0, int(w))
    t = cv2.GaussianBlur(t, (0, 0), 1.2); return t - t.mean()
bank = [cross(a) for a in range(0, 90, 15)]
cand = {}
for i in range(A0, A1 + 1):
    a = cv2.imread(f"{S}/ingenuity/{order[i]}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32); r = rectify(a / (cv2.GaussianBlur(a, (0, 0), 45) + 1e-3)); loc = r - cv2.GaussianBlur(r, (0, 0), 10)
    dark = np.maximum(-loc, 0); resp = np.max([cv2.filter2D(dark, -1, t) for t in bank], 0); resp = np.where(ring, resp, 0)
    pk = (resp == ndi.maximum_filter(resp, 31)) & (resp > 0); ys, xs = np.nonzero(pk); o = np.argsort(-resp[ys, xs])[:8]; Ti = T(i); p = Ti @ np.array([cx, cy, 1.0]); lst = []
    for k in o:
        v = np.array([xs[k] - cx, ys[k] - cy]); q = Ti @ np.array([xs[k], ys[k], 1.0]); w_ = (q[:2] / q[2]) - (p[:2] / p[2]); lst.append((float(resp[ys[k], xs[k]]), float(np.degrees(np.arctan2(v[1], v[0]))), float(np.degrees(np.arctan2(w_[1], w_[0]))), int(xs[k]), int(ys[k])))
    cand[i] = lst
allc = [(c[0], c[2]) for l in cand.values() for c in l]; w = np.array([a for a, _ in allc]); ang = np.radians([b for _, b in allc])
grid = np.arange(-180, 180, 1.0); score = np.array([(w * np.exp(-0.5 * ((((np.radians(g) - ang) + np.pi) % (2 * np.pi) - np.pi) / np.radians(12)) ** 2)).sum() for g in grid]); cons = grid[np.argmax(score)]
print("consensus shadow direction in the mosaic frame: %.0f deg (vote peak %.1f vs median %.1f)" % (cons, score.max(), np.median(score)))
keep = {}
for i, l in cand.items():
    ok = [(c[0], c[2], c[3], c[4]) for c in l if abs((c[2] - cons + 180) % 360 - 180) < 20]
    if ok: keep[i] = max(ok)
vals = np.array([keep[i][1] for i in keep]); m = np.degrees(np.arctan2(np.mean(np.sin(np.radians(vals))), np.mean(np.cos(np.radians(vals))))); sd = np.degrees(np.sqrt(-2 * np.log(np.hypot(np.mean(np.sin(np.radians(vals))), np.mean(np.cos(np.radians(vals)))))))
print("frames with a consistent candidate: %d of %d; mean direction %.1f deg, circular sd %.1f deg" % (len(keep), A1 - A0 + 1, m, sd))
shadow_az = (SUN + 180) % 360; rho = ((shadow_az - 90) - m + 180) % 360 - 180; theta_ccw = (-rho) % 360
print("sun azimuth %.1f -> shadow azimuth %.1f -> map angle %.1f; mosaic-to-north rotation (clockwise-positive) %.1f deg = %.1f deg counter-clockwise (OpenCV)" % (SUN, shadow_az, shadow_az - 90, rho, theta_ccw))
json.dump(dict(consensus=float(cons), mean=float(m), circ_sd=float(sd), n=len(keep), rho_cw=float(rho), theta_ccw=float(theta_ccw), kept={int(i): [float(keep[i][1]), keep[i][2], keep[i][3]] for i in keep}), open(OUT, "w"), indent=1)

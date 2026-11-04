"""Sun direction from the rocks' own shadows, in the mosaic frame.
A rock lit from the sun side has a dark shadow on the opposite side; across the shadow-to-rock edge the brightness increases toward the sun.
Per frame (rectified, flattened) the magnitude-weighted mean gradient over strong edges of bright objects gives the direction toward the sun in the image;
the pose-graph map carries it into the mosaic frame. A consistent direction across frames is a compass independent of the helicopter's shadow and of HiRISE.
Usage: python rock_shadow_sun.py <scratch_dir> <track_npz> <first> <last>"""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from rectify import rectify, OUT_W, OUT_H
S, TR, A0, A1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
order = json.load(open(f"{S}/ingenuity/nav_order.json")); d = np.load(TR); P = d["params"]; idx0 = int(d["idx"][0]); SCL = np.array([1, 1, 1, 1, 1, 1, 1000., 1000.])
def T(i): q = P[i - idx0] / SCL; return np.array([[q[0], q[1], q[2]], [q[3], q[4], q[5]], [q[6], q[7], 1.0]])
valid = cv2.erode((rectify(np.ones((480, 640), np.float32)) > 0.99).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0; cx, cy = OUT_W / 2, OUT_H / 2
res = {}
for i in range(A0, A1 + 1):
    a = cv2.imread(f"{S}/ingenuity/{order[i]}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32); r = rectify(a / (cv2.GaussianBlur(a, (0, 0), 45) + 1e-3)); g = cv2.GaussianBlur(r, (0, 0), 1.6)
    loc = g - cv2.GaussianBlur(g, (0, 0), 14); sd = 1.4826 * np.median(np.abs(loc[valid] - np.median(loc[valid])))
    Bm = (loc > 2.5 * sd) & valid; Dm = (loc < -2.0 * sd) & valid; nb, lab, st, cen = cv2.connectedComponentsWithStats(Bm.astype(np.uint8), connectivity=8); vs = []; ws = []
    for k in range(1, nb):
        if not (25 <= st[k, cv2.CC_STAT_AREA] <= 3000): continue
        mk = (lab == k).astype(np.uint8); near = cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0; dk = Dm & near & (lab != k)
        if dk.sum() < 15: continue
        yy, xx = np.nonzero(dk); vec = np.array([xx.mean() - cen[k][0], yy.mean() - cen[k][1]]); vs.append(vec / (np.linalg.norm(vec) + 1e-9)); ws.append(min(dk.sum(), 400))
    if not vs: continue
    v = -(np.array(vs) * np.array(ws)[:, None]).sum(0); coh = np.linalg.norm(v) / (np.array(ws).sum() + 1e-9); v /= np.linalg.norm(v) + 1e-12                                 # toward the sun = opposite to rock->shadow
    mag = np.ones(1); m = np.ones(1, bool); gx = gy = np.zeros(1)
    Ti = T(i); p = Ti @ np.array([cx, cy, 1.0]); q = Ti @ np.array([cx + 40 * v[0], cy + 40 * v[1], 1.0]); w = (q[:2] / q[2]) - (p[:2] / p[2])
    res[i] = (np.degrees(np.arctan2(v[1], v[0])), np.degrees(np.arctan2(w[1], w[0])), float(coh))
ks = sorted(res); print('frames with rock/shadow pairs:', len(ks)); ar = np.array([res[k][1] for k in ks]); ai = np.array([res[k][0] for k in ks]); co = np.array([res[k][2] for k in ks])
def circ(x): m = np.degrees(np.arctan2(np.median(np.sin(np.radians(x))), np.median(np.cos(np.radians(x))))); dv = (x - m + 180) % 360 - 180; return m, 1.4826 * np.median(np.abs(dv)), int((np.abs(dv) < 20).sum())
m, mad, n20 = circ(ar); print(f"frames {A0}-{A1}: direction toward the sun in the mosaic frame (image angle, clockwise from +x, y down): median {m:.1f} deg, MAD {mad:.1f}, {n20}/{len(ks)} within 20 deg; mean coherence {co.mean():.2f}")
print("by frame (every 2nd):", {k: round(res[k][1]) for k in ks[::2]})
print("(helicopter-shadow compass gave: shadow direction %.1f deg in this frame => direction toward the sun %.1f deg)" % (-71.6, (-71.6 + 180 + 180) % 360 - 180))

"""Top-down mosaic of Ingenuity flight-26 navigation frames over the parachute/backshell field.

Frames: HELI_NAV (nadir-pointing, 640x480, CAHVOR camera; distortion terms are <0.2 % of the field and are ignored, only the central
crop is used). Each frame is flattened (vignetting divided out), matched to its neighbours (SIFT + ratio test + RANSAC homography on the
ground plane) and chained to one reference frame. For a camera that looks straight down, ground-plane -> image is a similarity, so the
mosaic in the reference image plane is metric up to one scale (set from the nominal 8 m flight height and focal length 294 px,
or fitted against HiRISE later) and one rotation. Tilt of the reference frame is not modelled (documented limit).
Usage: python build_mosaic.py <scratch_dir> <first_idx> <last_idx> <ref_idx> <out_npz>
"""
import sys, json, numpy as np, cv2
S, I0, I1, REF, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
order = json.load(open(f"{S}/ingenuity/nav_order.json")); idx = list(range(I0, I1 + 1))
CROP = (slice(60, 420), slice(80, 560))                                   # central 480 x 360 of 640 x 480 (+-27 deg horizontally)
def load(i):
    a = cv2.imread(f"{S}/ingenuity/{order[i]}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    flat = a / (cv2.GaussianBlur(a, (0, 0), 45) + 1e-3)                    # divide out vignetting and slow illumination
    return flat[CROP]
F = {i: load(i) for i in idx}
u8 = lambda a: np.clip((a - 0.4) / (1.8 - 0.4) * 255, 0, 255).astype(np.uint8)
sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.01); cl = cv2.createCLAHE(3.0, (4, 4))
K = {i: sift.detectAndCompute(cl.apply(u8(F[i])), None) for i in idx}
def pair(i, j):
    (k1, d1), (k2, d2) = K[i], K[j]
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8: return None, 0
    m = [a for a, b in (x for x in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(x) == 2) if a.distance < 0.8 * b.distance]
    if len(m) < 10: return None, len(m)
    p1 = np.float32([k1[a.queryIdx].pt for a in m]); p2 = np.float32([k2[a.trainIdx].pt for a in m])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    return (H, int(inl.sum())) if H is not None else (None, 0)
# H[j] maps frame j pixels into frame j-1 pixels  (so chaining towards the reference multiplies)
Hn = {}
for i, j in zip(idx[:-1], idx[1:]):
    H, n = pair(j, i); Hn[j] = (H, n); print(f"{j - I0:3d} inliers {n:4d}", flush=True) if n < 40 else None
good = [j for j in idx[1:] if Hn[j][0] is not None and Hn[j][1] >= 20]
print("pairs with >=20 inliers:", len(good), "of", len(idx) - 1)
T = {REF: np.eye(3)}
for j in range(REF + 1, I1 + 1):                                           # frames after the reference: T[j] = T[j-1] @ Hn[j]
    if Hn[j][0] is None or Hn[j][1] < 20: break
    T[j] = T[j - 1] @ Hn[j][0]
for j in range(REF - 1, I0 - 1, -1):                                       # frames before: need Hn[j+1]^-1
    if Hn[j + 1][0] is None or Hn[j + 1][1] < 20: break
    T[j] = T[j + 1] @ np.linalg.inv(Hn[j + 1][0])
used = sorted(T); print("frames chained:", used[0] - I0, "to", used[-1] - I0, "of", len(idx))
h, w = F[REF].shape; cs = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float).T
pts = np.concatenate([(T[i] @ cs)[:2] / (T[i] @ cs)[2] for i in used], 1); x0, y0 = np.floor(pts.min(1)).astype(int); x1, y1 = np.ceil(pts.max(1)).astype(int)
W, Hh = int(x1 - x0), int(y1 - y0); print("mosaic size", W, Hh)
if W * Hh > 40e6: raise SystemExit("mosaic too large - frames drifted; narrow the range")
sh = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float); acc = np.zeros((Hh, W), np.float32); wt = np.zeros((Hh, W), np.float32)
fy, fx = np.mgrid[0:h, 0:w]; feather = np.minimum.reduce([fx, w - 1 - fx, fy, h - 1 - fy]).astype(np.float32).clip(0, 40) / 40
for i in used:
    M = sh @ T[i]; acc += cv2.warpPerspective(F[i] * feather, M, (W, Hh)); wt += cv2.warpPerspective(feather, M, (W, Hh))
mos = np.where(wt > 0.05, acc / np.maximum(wt, 1e-6), np.nan).astype(np.float32)
np.savez_compressed(OUT, mosaic=mos, weight=wt, used=np.array(used), ref=REF, i0=I0)
cv2.imwrite(OUT.replace(".npz", ".png"), u8(np.nan_to_num(mos, nan=1.0)))
print("wrote", OUT)

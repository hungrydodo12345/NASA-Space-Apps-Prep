"""New bright objects between the 2016 pass (ESP_045994) and the post-landing pass (ESP_068294): coarse phase-correlation alignment, high-pass difference,
local maxima above 8 robust sigma. The tight cluster ~1.1 km W / ~200 m N of the landing spot is the parachute + backshell. Usage: python find_new_objects.py <scratch_dir>"""
import sys, numpy as np, cv2, json
from scipy import ndimage as ndi
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep")
from pipeline.hirise_io import Rdr
S = sys.argv[1]; g0l, g0s = -4371755.17190706, -23476249.22764247; cl, cs = g0l + 64, g0s + 64; Hh = 12000; k = 4
A = np.load(f"{S}/pass_068294_12km_k4.npy")                                            # 068294 (post-landing)
lock = json.load(open(f"{S}/anywhere_site1_v2/summary.json"))["locks"]                  # per-pass integer locks relative to PSP_002387 at the no-rover site
R = Rdr(f"{S}/hirise_rdr/ESP_045994_1985_RED"); w = R.read_if(cl - Hh // 2, cs - Hh // 2, Hh, Hh); w = np.nan_to_num(w, nan=np.nanmedian(w))
Bm = w[:Hh // k * k, :Hh // k * k].reshape(Hh // k, k, Hh // k, k).mean((1, 3)).astype(np.float32); np.save(f"{S}/pass_045994_12km_k4.npy", Bm)
hp = lambda a: (a - cv2.GaussianBlur(a, (0, 0), 6)).astype(np.float32)
cen = (slice(500, 2500), slice(500, 2500))
(sx, sy), resp = cv2.phaseCorrelate(hp(A[cen]), hp(Bm[cen]), cv2.createHanningWindow((2000, 2000), cv2.CV_32F))
print("coarse shift (px at 1 m):", round(sx, 2), round(sy, 2), "response", round(resp, 3), flush=True)
Bs = ndi.shift(Bm, (-sy, -sx), order=1, mode="nearest")                                # 045994 brought onto 068294's grid
D = hp(A) - hp(Bs); D[:100] = D[-100:] = 0; D[:, :100] = D[:, -100:] = 0
sd = 1.4826 * np.median(np.abs(D - np.median(D)))
pk = (D == ndi.maximum_filter(D, 9)) & (D > 8 * sd)
ys, xs = np.nonzero(pk); order = np.argsort(-D[ys, xs])[:25]
print("sigma", round(float(sd), 5)); out = []
for i in order:
    y, x = ys[i], xs[i]; dn = (y - Hh / k / 2) * k * 0.2524; de = (x - Hh / k / 2) * k * 0.2524
    out.append(dict(y=int(y), x=int(x), z=float(D[y, x] / sd), north_m=float(-dn), east_m=float(de))); print(f"z {D[y,x]/sd:6.1f}  at ({x},{y})  {-dn:+7.0f} m N, {de:+7.0f} m E of the rover")
json.dump(out, open(f"{S}/new_objects.json", "w"), indent=1)

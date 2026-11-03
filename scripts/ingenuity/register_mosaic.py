"""Place the Ingenuity mosaic in HiRISE by distinct physical artifacts.

1. Anchor: the backshell is the brightest compact object in both datasets; its centroids are made to coincide.
2. The remaining debris pieces and rocks around it decide rotation (0-359 deg, with and without a mirror) and scale (metres per mosaic pixel,
   around the nominal 8 m / 294 px = 0.027 m): for each, the mosaic is resampled to the HiRISE 6.31 cm grid, blurred by the measured blur of the
   reference pass, and correlated (zero-mean, masked normalised cross-correlation) with the high-passed reference HiRISE pass (raw, so no
   restoration method is favoured), searching translations of +-0.8 m around the anchor.
3. Reports the best placement, the runner-up and the mirrored control; a placement is accepted only if it beats the best mirrored/rotated
   alternative by a clear margin.
Usage: python register_mosaic.py <mosaic_npz> <chain_dir> <out_json>
"""
import sys, json, numpy as np, cv2
from scipy import ndimage as ndi
MOS, CH, OUT = sys.argv[1:4]; HR = 0.0631
m = np.load(MOS); M = m["mosaic"].astype(np.float32); valid0 = np.isfinite(M)
d = np.load(f"{CH}/chain.npz"); meta = json.load(open(f"{CH}/chain.json")); fr = d["frames"][0].astype(np.float32)
H0 = cv2.resize(fr, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC); hp = lambda a, s=20: a - cv2.GaussianBlur(a, (0, 0), s)
psf = json.load(open("data_products/pass_blur.json"))[meta["passes_used"][0]]["sigma_px"] * 4
Hh = hp(H0)
def blob(a, s):
    b = cv2.GaussianBlur(np.nan_to_num(a, nan=np.nanmedian(a)).astype(np.float32), (0, 0), s); y, x = np.unravel_index(np.argmax(b), b.shape); return y, x
by, bx = blob(H0, 3); print("HiRISE backshell (hr px):", by, bx, "tile", H0.shape)
Mn = np.nan_to_num(M, nan=np.nanmedian(M)); my, mx = blob(Mn, 12); print("mosaic backshell (px):", my, mx, "mosaic", M.shape)
def place(scale_mpp, theta, flip):
    r = scale_mpp / HR; A = M[:, ::-1] if flip else M; V = valid0[:, ::-1] if flip else valid0
    cx_ = (M.shape[1] - 1 - mx) if flip else mx; cy_ = my
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta)); Rm = np.array([[r * c, -r * s], [r * s, r * c]])
    # affine mapping mosaic px (x, y) -> HR px, with the backshell landing on the HiRISE backshell
    t = np.array([bx, by]) - Rm @ np.array([cx_, cy_]); W = np.hstack([Rm, t[:, None]])
    a = cv2.warpAffine(np.nan_to_num(A, nan=0.0), W, H0.shape[::-1], flags=cv2.INTER_AREA); v = cv2.warpAffine(V.astype(np.float32), W, H0.shape[::-1]) > 0.9
    return a, v, W
def score(a, v, dy, dx):
    a = ndi.shift(a, (dy, dx), order=1); v = ndi.shift(v.astype(np.float32), (dy, dx), order=0) > 0.5
    a = cv2.GaussianBlur(a, (0, 0), psf); a = hp(a); v = ndi.binary_erosion(v, iterations=int(2 * psf)); 
    if v.sum() < 3000: return -1
    return float(np.corrcoef(a[v], Hh[v])[0, 1])
res = []
for flip in (0, 1):
    for scale in (0.022, 0.025, 0.0272, 0.030, 0.033):
        for theta in range(0, 360, 6):
            a, v, W = place(scale, theta, flip); best = max((score(a, v, dy, dx), dy, dx) for dy in range(-6, 7, 3) for dx in range(-6, 7, 3))
            res.append(dict(flip=flip, scale=scale, theta=theta, corr=best[0], dy=best[1], dx=best[2]))
res.sort(key=lambda r: -r["corr"])
for r in res[:8]: print(r)
best = res[0]; alt = [r for r in res if r["flip"] == 1][0]; far = [r for r in res if abs(((r["theta"] - best["theta"] + 180) % 360) - 180) > 24 or r["flip"] != best["flip"]][0]
# refine the best
cands = []
for scale in np.linspace(best["scale"] - 0.002, best["scale"] + 0.002, 5):
    for theta in np.arange(best["theta"] - 6, best["theta"] + 6.1, 2):
        a, v, W = place(scale, theta, best["flip"]); sc = max((score(a, v, best["dy"] + dy, best["dx"] + dx), best["dy"] + dy, best["dx"] + dx) for dy in range(-3, 4) for dx in range(-3, 4))
        cands.append((sc[0], scale, theta, sc[1], sc[2]))
f = max(cands); print("refined", f)
out = dict(best=dict(corr=f[0], scale_mpp=float(f[1]), theta=float(f[2]), flip=best["flip"], dy=f[3], dx=f[4]), coarse_top=res[:8], best_mirror_alternative=alt, best_far_alternative=far,
           margin_over_far_alternative=f[0] - far["corr"], hirise_backshell_hr_px=[int(by), int(bx)], mosaic_backshell_px=[int(my), int(mx)])
json.dump(out, open(OUT, "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k not in ("coarse_top",)}, indent=1))

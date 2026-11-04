"""Register the Ingenuity strip mosaic to HiRISE by masked normalised cross-correlation over rotation, scale and (optionally) mirror, with NO location prior.

HiRISE: the raw reference pass (ESP_068294) in a window around the debris field, band-passed (difference of Gaussians, 1-4 native px). Mosaic: resampled to the
HiRISE grid at each trial scale/rotation, blurred by the pass's measured PSF, same band-pass; strip shape handled by a mask (Padfield's masked NCC, by FFT).
Validation: the peak position is compared with the known backshell position, and the peak is compared with the distribution of the best score over all
other rotations/scales (a real match stands out; a chance match does not).
Usage: python register_strip.py <scratch_dir> <mosaic_npz> <out_json>
"""
import sys, json, numpy as np, cv2
from scipy.signal import fftconvolve
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep")
from pipeline.hirise_io import Rdr
S, MOS, OUT = sys.argv[1:4]; PXH = 0.2524; PSF = 0.889
site = json.load(open(f"{S}/debris_site/summary.json")); bl, bs = site["backshell_global"]; N = 760
R = Rdr(f"{S}/hirise_rdr/ESP_068294_1985_RED"); I = R.read_if(bl - N // 2, bs - N // 2, N, N); I = np.nan_to_num(I, nan=float(np.nanmedian(I))).astype(np.float32)
import os
B0, B1 = float(os.environ.get("BAND_LO", 1.0)), float(os.environ.get("BAND_HI", 4.0))          # band-pass in native HiRISE pixels
band = lambda a: (cv2.GaussianBlur(a, (0, 0), B0) - cv2.GaussianBlur(a, (0, 0), B1)).astype(np.float32)
Ib = band(I); print("HiRISE window", I.shape, "backshell at the window centre (%d, %d) by construction of the window" % (N // 2, N // 2), flush=True)
m = np.load(MOS); M0 = m["mosaic"].astype(np.float32); V0 = np.isfinite(M0); M0 = np.nan_to_num(M0, nan=1.0); mpp0 = float(m["m_per_px"])
v = cv2.GaussianBlur(M0, (0, 0), 12); v[~V0] = -1; by, bx = np.unravel_index(np.argmax(v), v.shape); print("mosaic backshell (largest bright blob) at px", bx, by, flush=True)
I2 = Ib * Ib
def mncc(T, Mk):
    Tm = T * Mk; n = Mk.sum()
    if n < 2000: return None
    s_tm = Tm.sum(); denT = (Tm * T).sum() - s_tm ** 2 / n
    c = lambda A, B: fftconvolve(A, B[::-1, ::-1], mode="valid")
    cI = c(Ib, Tm); cM = c(Ib, Mk); cI2 = c(I2, Mk); num = cI - cM * s_tm / n; den = np.sqrt(np.maximum(cI2 - cM ** 2 / n, 1e-9) * max(denT, 1e-9))
    return num / den
def trial(scale, theta, flip):
    A, Vv = (M0[:, ::-1], V0[:, ::-1]) if flip else (M0, V0); f = scale * mpp0 / PXH
    h, w = A.shape; a = cv2.resize(A, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA); vm = cv2.resize(Vv.astype(np.float32), a.shape[::-1], interpolation=cv2.INTER_AREA) > 0.95
    H, W = a.shape; c = (W / 2, H / 2); Rm = cv2.getRotationMatrix2D(c, theta, 1.0); cs, sn = abs(Rm[0, 0]), abs(Rm[0, 1]); W2, H2 = int(H * sn + W * cs), int(H * cs + W * sn)
    Rm[0, 2] += W2 / 2 - c[0]; Rm[1, 2] += H2 / 2 - c[1]
    ar = cv2.warpAffine(a, Rm, (W2, H2), flags=cv2.INTER_LINEAR); vr = cv2.warpAffine(vm.astype(np.float32), Rm, (W2, H2)) > 0.99
    ar = cv2.GaussianBlur(np.where(vr, ar, float(np.median(a[vm]))), (0, 0), PSF); T = band(ar); vr = cv2.erode(vr.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if T.shape[0] >= Ib.shape[0] - 40 or T.shape[1] >= Ib.shape[1] - 40: return None
    r = mncc(T, vr.astype(np.float32))
    if r is None: return None
    y, x = np.unravel_index(np.argmax(r), r.shape); return float(r[y, x]), x + T.shape[1] / 2, y + T.shape[0] / 2, Rm, (W2, H2)
def rng_(spec):
    a, b, c = (float(x) for x in spec.split(":")); return list(np.arange(a, b + 1e-9, c))
THETAS = rng_(os.environ.get("THETAS", "0:356:4")); SCALES = [float(x) for x in os.environ.get("SCALES", "0.90,0.95,1.0,1.05,1.10").split(",")]; FLIPS = [int(x) for x in os.environ.get("FLIPS", "0,1").split(",")]
res = []
for flip in FLIPS:
    for scale in SCALES:
        for theta in THETAS:
            t = trial(scale, theta, flip)
            if t is not None: res.append(dict(flip=flip, scale=scale, theta=theta, ncc=t[0], cx=t[1], cy=t[2]))
    print("flip", flip, "done, best so far", round(max(r["ncc"] for r in res), 3), flush=True)
MOSBS = os.environ.get("MOSAIC_BACKSHELL")
res.sort(key=lambda r: -r["ncc"]); nc = np.array([r["ncc"] for r in res])
print("top placements:"); [print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}) for r in res[:6]]
print("score distribution over all placements: median %.3f, 99th pct %.3f, best %.3f, best minus 2nd-best distinct rotation %.3f" % (np.median(nc), np.percentile(nc, 99), nc.max(), nc.max() - max(r["ncc"] for r in res if abs(((r["theta"] - res[0]["theta"] + 180) % 360) - 180) > 12 or r["flip"] != res[0]["flip"])))
json.dump(dict(top=res[:12], median=float(np.median(nc)), p99=float(np.percentile(nc, 99)), window=N, mosaic_backshell_px=[int(bx), int(by)]), open(OUT, "w"), indent=1)

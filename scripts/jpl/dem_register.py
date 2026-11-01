"""Register each JPL Navcam DEM (band 3, 0.5 m sampling, +-45 m) to the HiRISE DTM (1.01 m posts) by terrain shape.
Tries all 8 orientations; search +-60 m around the corrected waypoint; detrended heights, normalised cross-correlation."""
import sys, glob, os, json, numpy as np
from scipy import ndimage as ndi
S = sys.argv[1]; sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep")
from pipeline import groundtruth as gt
from pipeline.dtm import Dtm
D = Dtm(f"{S}/hirise_jezero/DTEEC_045994_1985_046060_1985_U01.IMG"); P = gt.ORTHO["scale"]
wp = {f["properties"]["RMC"]: f["properties"] for f in json.load(open(f"{S}/rover/waypoints.json"))["features"]}
DL, DS = -547, 74                                     # global waypoint -> ortho correction (from the parked rover, sols 1-11)
ops = {"as-is": lambda a: a, "rot90": lambda a: np.rot90(a, 1), "rot180": lambda a: np.rot90(a, 2), "rot270": lambda a: np.rot90(a, 3),
       "flipud": np.flipud, "fliplr": np.fliplr, "transpose": lambda a: a.T, "anti-transpose": lambda a: np.rot90(a, 2).T}
def detrend(a, m):
    yy, xx = np.nonzero(m); A = np.stack([np.ones(len(yy)), yy, xx, yy * yy, xx * xx, yy * xx], 1)
    c = np.linalg.lstsq(A, a[m], rcond=None)[0]; Y, X = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return a - (c[0] + c[1] * Y + c[2] * X + c[3] * Y * Y + c[4] * X * X + c[5] * Y * X)
res = {}
for f in sorted(glob.glob(f"{S}/jpl_mosaic/dem/*.npy")):
    rmc = os.path.basename(f)[:-4]; z = np.load(f)[2].astype(float); m = (z != 0) & np.isfinite(z) & (np.abs(z) < 50)
    if m.mean() < 0.2: print(rmc, "too little DEM"); continue
    zr = ndi.zoom(np.where(m, z, 0), 0.5 / 1.01, order=1); mr = ndi.zoom(m.astype(float), 0.5 / 1.01, order=1) > 0.9   # to DTM posts (~1 m)
    w = wp[rmc]; l0, s0 = gt.latlon_to_ortho(w["lat"], w["lon"]); l0, s0 = l0 + DL, s0 + DS
    R = 40; half = zr.shape[0] // 2 + R + 2
    H, _ = D.heights_for_ortho(int(l0 - 4 * half), int(s0 - 4 * half), 8 * half, 8 * half, step=4)
    best = None
    for nm, op in ops.items():
        a = op(zr); mm = op(mr)
        a = np.where(mm, detrend(a, mm), 0.0); a = (a - a[mm].mean()) * mm; a /= np.sqrt((a ** 2).sum()) + 1e-12; t = a.shape[0]
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                y0, x0 = half - t // 2 + dy, half - t // 2 + dx
                h = H[y0:y0 + t, x0:x0 + t]
                if h.shape != a.shape: continue
                hh = detrend(h, mm)
                hv = hh[mm]; hv = hv - hv.mean(); c = float((a[mm] * hv).sum() / (np.sqrt((hv ** 2).sum()) + 1e-12))
                if best is None or c > best[0]: best = (c, nm, dy, dx)
    res[rmc] = dict(corr=best[0], orientation=best[1], dS_m=best[2] * 1.01, dE_m=best[3] * 1.01)
    print(f"{rmc:8s} best {best[1]:14s} corr {best[0]:.2f}  residual offset {best[2]*1.01:+6.1f} m S, {best[3]*1.01:+6.1f} m E", flush=True)
json.dump(res, open(f"{S}/jpl_mosaic/dem_register.json", "w"), indent=1)

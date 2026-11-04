"""Second Ingenuity truth site: JPL's orthorectified downward-camera mosaic of flights 4-5 (sol 76, 1 cm/px, site frame 3, north-aligned) resampled onto the
6.25 cm restoration grid of a HiRISE chain tile.

Placement: site frame 3 is level and north-aligned, so the mosaic differs from the HiRISE map grid only by a translation (plus the map's east-west scale,
cos 15 deg / cos lat). The translation was found by band-passed correlation at 1 m (post-landing passes agree; landing-blast patches visible in both) and
refined at 25 cm against ESP_068294 (correlation 0.69 in the 25 cm-1 m band, runner-up 0.08); see benchmark/results/ingenuity_truth/site2_placement.json.
Usage: python truth_patch_site2.py <mosaic.IMG> <chain_dir> <placement.json> <out_npz>"""
import sys, json, numpy as np
from scipy import ndimage as ndi
IMG, CH, PL, OUT = sys.argv[1:5]
p = json.load(open(PL)); glNW, gsNW = p["nw_corner_global_ref"]          # LR (25 cm) grid position of the mosaic's north-west corner, reference-pass grid
meta = json.load(open(f"{CH}/chain.json")); gl0, gs0 = meta["origin_global"]; SC = meta["scale"]; n = meta["tile_native_px"] * SC
ysc = np.cos(np.radians(15)) / np.cos(np.radians(p["lat_deg"]))          # map samples per ground metre, relative to lines
A = np.fromfile(IMG, dtype=">i2", offset=p["img_offset"], count=p["lines"] * p["samples"]).reshape(p["lines"], p["samples"]).astype(np.float32)
v = A > 0; A[~v] = np.nan
i, j = np.mgrid[0:n, 0:n].astype(float)
Gl = gl0 + (i + 0.5) / SC - 0.5; Gs = gs0 + (j + 0.5) / SC - 0.5         # HR pixel centres in LR (global) coordinates
u = (Gl - glNW + 0.5) * 25 - 0.5; w = ((Gs - gsNW + 0.5) / ysc) * 25 - 0.5  # mosaic 1 cm pixel coordinates (25 mosaic px per LR px north-south)
y0, y1 = int(max(0, u.min() - 20)), int(min(A.shape[0], u.max() + 20)); x0, x1 = int(max(0, w.min() - 20)), int(min(A.shape[1], w.max() + 20))
a = A[y0:y1, x0:x1]; m = np.isfinite(a); af = np.where(m, a, np.nanmean(a))
sig = 6.25 / 2.5                                                            # pre-filter to the 6.25 cm footprint (box ~ Gaussian sigma = width / 2.5)
num = ndi.gaussian_filter(af * m, sig); den = ndi.gaussian_filter(m.astype(np.float32), sig)
T = ndi.map_coordinates(num / np.maximum(den, 1e-6), [u - y0, w - x0], order=1, mode="constant", cval=np.nan)
V = ndi.map_coordinates(den, [u - y0, w - x0], order=1, mode="constant", cval=0) > 0.98
T[~V] = np.nan
np.savez_compressed(OUT, truth=T.astype(np.float32), valid=V, placement=json.dumps(p), chain_origin=np.array([gl0, gs0]))
print("truth patch", T.shape, "valid fraction %.2f" % V.mean())

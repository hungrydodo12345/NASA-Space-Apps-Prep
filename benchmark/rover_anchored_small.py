"""Small rover-anchored check (docs/METHODS.md Section 9): how well does each restoration match the rover's own top-down view?

Landing tile (rover parked at the centre, passes from sols 1-11). Truth: Navcam ground projection of the stop at the rover's
landing spot (rover_truth_traverse/3_0), placed with the rover at the tile centre and NO search. Scored inside the largest
fully covered area, outside a disc around the rover, in difference-of-Gaussian bands. Controls: the same truth shifted by
+-1.5 m (wrong place). Methods restore the same 3 passes (benchmark/run_benchmark.py settings); nothing is tuned here.
Usage: python benchmark/rover_anchored_small.py <chain_dir> <truth_npz> <out_json>
"""
import sys, os, json, numpy as np
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline import pde_tv
CH, TR, OUT = sys.argv[1:4]; SC = 4; PX = 6.25
d = np.load(f"{CH}/chain.npz"); m = json.load(open(f"{CH}/chain.json")); b = json.load(open(os.path.join(ROOT, "data_products", "pass_blur.json")))
fr = list(d["frames"].astype(float)); fl = list(zip(d["flows_dx"].astype(float), d["flows_dy"].astype(float))); sig = [b[o]["sigma_px"] * SC for o in m["passes_used"]]; n = fr[0].shape[0] * SC
P = {"raw": np.kron(fr[0], np.ones((SC, SC))),
     "ucl": pde_tv.pde_tv(fr, fl, scale=SC, psf_sigma=1.0 * SC, lam=0.005, iters=300, weighted=False)[0],
     "ucl+blur": pde_tv.pde_tv(fr, fl, scale=SC, psf_sigma=sig, lam=0.005, iters=300, weighted=False)[0],
     "this work": pde_tv.pde_tv(fr, fl, scale=SC, psf_sigma=sig, lam=0.005, iters=300)[0]}
z = np.load(TR)["z"].astype(float); k = 5.0 / PX
zz = ndi.zoom(np.nan_to_num(z), k, order=1); vv = ndi.zoom(np.isfinite(z).astype(float), k, order=0) > 0.5
T = np.zeros((n, n)); V = np.zeros((n, n), bool); o = (n - zz.shape[0]) // 2; T[o:o + zz.shape[0], o:o + zz.shape[0]] = zz; V[o:o + zz.shape[0], o:o + zz.shape[0]] = vv
yy, xx = np.mgrid[0:n, 0:n]; rover = np.hypot(yy - n / 2, xx - n / 2) < 12 * SC / 1.0 * 0 + 30          # ~1.9 m radius disc around the rover
bands = {"2-1 m": (16, 32), "1 m-50 cm": (8, 16), "50-25 cm": (4, 8)}
band = lambda a, lo, hi: ndi.gaussian_filter(a, lo) - ndi.gaussian_filter(a, hi)
def score(img, dy=0, dx=0):
    t = np.roll(np.roll(T, dy, 0), dx, 1); v = np.roll(np.roll(ndi.binary_erosion(V, iterations=10), dy, 0), dx, 1) & ~rover
    out = {}
    for kk, (lo, hi) in bands.items(): out[kk] = float(np.corrcoef(band(img, lo, hi)[v], band(t, lo, hi)[v])[0, 1])
    out["_px"] = int(v.sum()); return out
res = {"scored_area_m2": float(score(P["raw"])["_px"] * (PX / 100) ** 2), "truth_file": os.path.basename(TR), "methods": {}, "control_shifted_1.5m": {}}
for name, img in P.items():
    res["methods"][name] = score(img)
    res["control_shifted_1.5m"][name] = {kk: float(np.mean([score(img, dy, dx)[kk] for dy, dx in ((24, 0), (-24, 0), (0, 24), (0, -24))])) for kk in bands}
    print(name, {kk: round(v, 3) for kk, v in res["methods"][name].items() if kk != "_px"}, "control", {kk: round(v, 3) for kk, v in res["control_shifted_1.5m"][name].items()}, flush=True)
print("scored area", round(res["scored_area_m2"], 1), "m2")
json.dump(res, open(OUT, "w"), indent=1)

"""Restore one chain tile with a grid of methods and settings (CPU, multiprocessing) for the truth-based parameter search.
Usage: python benchmark/restore_variants.py <chain_dir> <out_npz> [iters]
Methods: gpt (assumed blur 1 px, unweighted) | gpt+blur (measured per-pass blur, unweighted) | full (measured blur, noise weights, change masks) |
         full-equal (measured blur, equal pass weights, change masks) | full-b0.8 / full-b1.2 (measured blur x0.8 / x1.2).  lambda in LAMS."""
import sys, os, json, numpy as np
from multiprocessing import Pool
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline import pde_tv
CH, OUT = sys.argv[1:3]; ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 300; SC = 4; LAMS = tuple(float(x) for x in os.environ.get("LAMS", "0.002,0.005,0.015,0.04,0.1").split(","))
d = np.load(f"{CH}/chain.npz"); meta = json.load(open(f"{CH}/chain.json")); B = json.load(open(os.path.join(ROOT, "data_products", "pass_blur.json")))
FR = list(d["frames"].astype(float)); FL = list(zip(d["flows_dx"].astype(float), d["flows_dy"].astype(float))); SIG = [B[o]["sigma_px"] * SC for o in meta["passes_used"]]
METHODS = os.environ.get("METHODS", "gpt,gpt+blur,full,full-equal,full-b0.8,full-b1.2").split(",")
def job(a):
    m, lam = a
    if m == "gpt": X = pde_tv.pde_tv(FR, FL, scale=SC, psf_sigma=1.0 * SC, lam=lam, iters=ITERS, weighted=False)[0]
    elif m == "gpt+blur": X = pde_tv.pde_tv(FR, FL, scale=SC, psf_sigma=SIG, lam=lam, iters=ITERS, weighted=False)[0]
    elif m == "full": X = pde_tv.pde_tv(FR, FL, scale=SC, psf_sigma=SIG, lam=lam, iters=ITERS, pass_weights="noise")[0]
    elif m == "full-equal": X = pde_tv.pde_tv(FR, FL, scale=SC, psf_sigma=SIG, lam=lam, iters=ITERS, pass_weights="equal")[0]
    else: k = float(m.split("-b")[1]); X = pde_tv.pde_tv(FR, FL, scale=SC, psf_sigma=[s * k for s in SIG], lam=lam, iters=ITERS, pass_weights="noise")[0]
    return m, lam, X.astype(np.float32)
if __name__ == "__main__":
    with Pool(4) as p: res = p.map(job, [(m, l) for m in METHODS for l in LAMS], chunksize=1)
    import scipy.ndimage as ndi
    out = {f"{m}__{l}": X for m, l, X in res}
    if os.environ.get("WITH_RAW", "1") == "1": out["raw"] = ndi.zoom(FR[0], SC, order=3, grid_mode=True, mode="nearest").astype(np.float32)
    np.savez_compressed(OUT, **out); print("wrote", OUT, len(out), "arrays")

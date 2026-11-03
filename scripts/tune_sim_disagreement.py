"""Tune the simulation's pass-to-pass disagreement to real HiRISE (docs/METHODS.md Section 7b).

The real chain tile gives the split-half disagreement per rock size (pipeline/confidence.disagreement): how much two
restorations from disjoint halves of the real passes disagree, relative to the scene's blob-response spread. The
simulation of experiments/exp7_confidence_calibration.py is run over a grid of white noise and smooth illumination
mismatch, and the setting whose disagreement best matches the real one (least squares in log ratio over sizes) is kept.
Usage: python scripts/tune_sim_disagreement.py <chain_dir> <out_json>
"""
import sys, os, json, itertools, numpy as np
from multiprocessing import Pool
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline import confidence as cf, pde_tv
CH, OUT = sys.argv[1:3]
GRID = list(itertools.product([0.03, 0.1, 0.2, 0.35], [0.0, 0.1, 0.25]))


def real():
    d = np.load(f"{CH}/chain.npz"); m = json.load(open(f"{CH}/chain.json")); b = json.load(open(os.path.join(ROOT, "data_products", "pass_blur.json")))
    fr = list(d["frames"].astype(float)); fl = list(zip(d["flows_dx"].astype(float), d["flows_dy"].astype(float)))
    h = cf.split_halves(fr, fl, [b[o]["sigma_px"] * 4 for o in m["passes_used"]], scale=4, lam=m["lam"], iters=m["iters"])
    return cf.disagreement(d["restored"].astype(float), h)


def sim(p):
    os.environ["SIM_NOISE"], os.environ["SIM_LIGHT"] = str(p[0]), str(p[1])
    import importlib, experiments.exp7_confidence_calibration as E; importlib.reload(E)
    fr, fl, _ = E.scene(20); sg = [s * 4 for s in E.SIG]
    X, _ = pde_tv.pde_tv(fr, fl, scale=4, psf_sigma=sg, lam=0.005, iters=E.ITERS)
    return p, cf.disagreement(X, cf.split_halves(fr, fl, sg, scale=4, lam=0.005, iters=E.ITERS))


if __name__ == "__main__":
    with Pool(4) as pool:
        rj = pool.apply_async(real); sims = pool.map(sim, GRID); R = rj.get()
    rows = []
    for (n, l), D in sims:
        cost = float(np.mean([(np.log(D[k]) - np.log(R[k])) ** 2 for k in R])); rows.append(dict(noise=n, light=l, disagreement=D, cost=cost))
        print(f"noise {n:.2f} light {l:.2f} cost {cost:.3f} " + " ".join(f"{k}:{v:.3f}" for k, v in D.items()), flush=True)
    best = min(rows, key=lambda r: r["cost"])
    json.dump(dict(real=R, grid=rows, best=best), open(OUT, "w"), indent=1)
    print("real", {k: round(v, 3) for k, v in R.items()}); print("best", best["noise"], best["light"], round(best["cost"], 3))

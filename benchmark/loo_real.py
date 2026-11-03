"""Leave-one-pass-out accuracy on real HiRISE (docs/METHODS.md Section 9): the real-data test of accuracy that needs no
truth, no registration and no prior.

For each non-reference pass k: restore from the other passes, then PREDICT pass k with the forward model (its own Gotcha
motion, its own measured blur, decimation) and compare with the pass actually recorded. Error = RMS of (prediction - recorded)
after a least-squares gain/offset, as a share of the recorded pass's standard deviation; the noise floor of pass k (Section 5
noise estimator) is subtracted in quadrature, so the number is the error of the restoration itself. Every method is scored
through the SAME forward model (measured blur of the held-out pass), so the assumed blur of a method cannot flatter it.
lambda is chosen per method on the tuning tile; the test tile is scored once. Paired by held-out pass.
Usage: python benchmark/loo_real.py <tune_chain> <test_chain> <out_json>
"""
import sys, os, json, numpy as np
from multiprocessing import Pool
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline import pde_tv
from pipeline.fuse_flow import FlowModel
TUNE, TEST, OUT = sys.argv[1:4]; SC = 4; ITERS = 200; LAMS = (0.005, 0.015, 0.04)
METHODS = ["bicubic", "gpt", "gpt+blur", "full"]
B = json.load(open(os.path.join(ROOT, "data_products", "pass_blur.json")))
_cache = {}
def load(ch):
    if ch not in _cache:
        d = np.load(f"{ch}/chain.npz"); m = json.load(open(f"{ch}/chain.json"))
        _cache[ch] = (list(d["frames"].astype(float)), list(zip(d["flows_dx"].astype(float), d["flows_dy"].astype(float))), [B[o]["sigma_px"] * SC for o in m["passes_used"]])
    return _cache[ch]
def restore(method, fr, fl, sg, lam):
    if method == "bicubic": return ndi.zoom(fr[0], SC, order=3, grid_mode=True, mode="nearest")
    if method == "gpt": return pde_tv.pde_tv(fr, fl, scale=SC, psf_sigma=1.0 * SC, lam=lam, iters=ITERS, weighted=False)[0]
    if method == "gpt+blur": return pde_tv.pde_tv(fr, fl, scale=SC, psf_sigma=sg, lam=lam, iters=ITERS, weighted=False)[0]
    return pde_tv.pde_tv(fr, fl, scale=SC, psf_sigma=sg, lam=lam, iters=ITERS, pass_weights="noise")[0]
def job(a):
    ch, method, lam, k = a; fr, fl, sg = load(ch); idx = [i for i in range(len(fr)) if i != k]
    X = restore(method, [fr[i] for i in idx], [fl[i] for i in idx], [sg[i] for i in idx], lam)
    pred = FlowModel(fr[0].shape, fl, SC, sg).forward(X, k); y = fr[k]; c = (slice(10, -10), slice(10, -10))
    A = np.stack([pred[c].ravel(), np.ones(pred[c].size)], 1); co = np.linalg.lstsq(A, y[c].ravel(), rcond=None)[0]
    rms = float(np.sqrt(np.mean((A @ co - y[c].ravel()) ** 2))); sd = float(y[c].std()); nz = pde_tv.pass_noise(y)
    return ch, method, lam, k, 100 * rms / sd, 100 * np.sqrt(max(rms ** 2 - nz ** 2, 0)) / sd, 100 * nz / sd
if __name__ == "__main__":
    n = len(load(TUNE)[0]); ks = list(range(1, n))
    with Pool(4) as p:
        tune = p.map(job, [(TUNE, m, l, k) for m in METHODS if m != "bicubic" for l in LAMS for k in ks] + [(TUNE, "bicubic", 0, k) for k in ks])
        best = {"bicubic": 0}
        for m in METHODS[1:]:
            best[m] = min(LAMS, key=lambda l: np.mean([r[4] for r in tune if r[1] == m and r[2] == l])); print("tuning tile: best lambda", m, best[m], flush=True)
        test = p.map(job, [(TEST, m, best[m], k) for m in METHODS for k in ks])
    out = dict(best_lambda=best, tune=[list(r) for r in tune], test=[list(r) for r in test], summary={})
    for m in METHODS:
        e = np.array([r[4] for r in test if r[1] == m]); ex = np.array([r[5] for r in test if r[1] == m]); out["summary"][m] = dict(error_pct=float(e.mean()), excess_over_noise_pct=float(ex.mean()), per_pass=e.tolist())
    g = np.array(out["summary"]["gpt"]["per_pass"]); out["paired_vs_gpt"] = {}
    rng = np.random.default_rng(0)
    for m in METHODS:
        if m == "gpt": continue
        dl = np.array(out["summary"][m]["per_pass"]) - g; bs = [dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(4000)]
        out["paired_vs_gpt"][m] = dict(mean=float(dl.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], passes_better=int((dl < 0).sum()), passes=len(dl))
    json.dump(out, open(OUT, "w"), indent=1)
    for m, s in out["summary"].items(): print(f"{m:10s} error {s['error_pct']:.2f}% of pass contrast (excess over noise {s['excess_over_noise_pct']:.2f}%)  lambda {best[m]}")
    for m, d in out["paired_vs_gpt"].items(): print(f"{m} - gpt: {d['mean']:+.3f} points, 95% CI {d['ci95'][0]:+.3f}..{d['ci95'][1]:+.3f}, better in {d['passes_better']}/{d['passes']} passes")

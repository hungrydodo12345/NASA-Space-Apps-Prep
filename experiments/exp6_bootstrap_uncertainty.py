"""Experiment 6: is the pass-subset bootstrap uncertainty (pipeline/uncertainty.py) honest, and at what scale?

Simulation with known truth. Fractal terrain, 8 passes, each rendered with its own measured blur
(data_products/pass_blur.json values), 3 % noise, 0.03 px registration error, and one pass carrying a planted
2 m bright change. Two scenes: every calibration factor is fitted on scene A and tested on scene B (held out).

The error is evaluated at three scales: the 6.25 cm output pixel, 12.5 cm, and the 25 cm native HiRISE pixel
(restoration, truth and every bootstrap member smoothed by a Gaussian of 0, 2 or 4 output pixels).
At each scale it reports:
  - coverage of +-1 sigma and +-2 sigma (Gaussian target 68.3 % / 95.4 %), raw and after the scale factor kappa;
  - whether sigma ranks errors: Spearman correlation of sigma and |err| over 4 x 4 output-pixel blocks, and the mean
    error in the 10 % most- and least-confident blocks;
  - the two-component model (bootstrap + local texture, pipeline/uncertainty.fit_two_component) at 6.25 cm.
Output: results/bootstrap_uncertainty_simulation.json
Environment overrides for quick checks: N_LR, BOOT, ITERS.
"""
import sys, os, json, time, numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from experiments import scenes
from pipeline import uncertainty as U

SC = 4
SIG_LR = [0.889, 0.587, 0.828, 1.336, 1.426, 1.381, 1.16, 1.203]       # measured per-pass blur (pass_blur.json)
N_LR = int(os.environ.get("N_LR", 64)); B = int(os.environ.get("BOOT", 16)); ITERS = int(os.environ.get("ITERS", 200))
SCALES = {"6.25 cm (output pixel)": 0, "12.5 cm": 2, "25 cm (native HiRISE pixel)": 4}


def simulate(truth, rng, noise_rel=0.03, reg_err=0.03):
    frames, flows = [], []
    for k, s in enumerate(SIG_LR):
        dx, dy = (0.0, 0.0) if k == 0 else rng.uniform(-2, 2, 2)
        lr = ndi.gaussian_filter(ndi.shift(truth, (-SC * dy, -SC * dx), order=3, mode="reflect"), s * SC)[::SC, ::SC]
        frames.append(lr); H, W = lr.shape
        ex, ey = (0.0, 0.0) if k == 0 else rng.normal(0, reg_err, 2)
        flows.append((np.full((H, W), dx + ex), np.full((H, W), dy + ey)))
    s0 = frames[0].std(); frames = [f + rng.normal(0, noise_rel * s0, f.shape) for f in frames]
    H, W = frames[3].shape; cy, cx = H // 3, W // 3; yy, xx = np.mgrid[0:H, 0:W]
    frames[3] = frames[3] + 4 * s0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2))
    return frames, flows, (cy * SC, cx * SC)


def run(seed):
    truth = scenes.fractal(seed, n=N_LR * SC); truth = truth[0] if isinstance(truth, tuple) else truth
    frames, flows, ch = simulate(truth, np.random.default_rng(seed))
    t = time.time()
    bs = U.bootstrap(frames, flows, [s * SC for s in SIG_LR], B=B, seed=seed, scale=SC, lam=0.005, iters=ITERS)
    m = 6 * SC
    return dict(truth=truth, members=bs["members"], c=(slice(m, -m), slice(m, -m)), change=ch, seconds=time.time() - t)


def at_scale(r, sr):
    f = (lambda a: ndi.gaussian_filter(a, sr)) if sr else (lambda a: a)
    M = np.stack([f(x) for x in r["members"]]); X = M.mean(0); sb = M.std(0, ddof=1)
    return (X - f(r["truth"]))[r["c"]], sb[r["c"]], X, sb


def blocks(a, k=4):
    H, W = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:H, :W].reshape(H // k, k, W // k, k).mean((1, 3))


def ranking(err, sd, s0):
    be, bs = blocks(np.abs(err)).ravel(), blocks(sd).ravel(); o = np.argsort(bs); n10 = max(1, len(o) // 10)
    return dict(spearman_sigma_vs_error=float(spearmanr(bs, be).correlation),
                error_pct_most_confident_10pct=100 * float(be[o[:n10]].mean() / s0),
                error_pct_least_confident_10pct=100 * float(be[o[-n10:]].mean() / s0))


if __name__ == "__main__":
    A, Bs = run(2001), run(2002)
    out = dict(setup=dict(n_lr=N_LR, scale=SC, passes=len(SIG_LR), bootstrap_replicates=B, iters=ITERS, per_pass_sigma_lr=SIG_LR,
                          noise_rel=0.03, reg_err_px=0.03, lam=0.005, seconds_per_bootstrap=[A["seconds"], Bs["seconds"]]),
               scales={})
    for name, sr in SCALES.items():
        eA, sA, _, _ = at_scale(A, sr); eB, sB, _, sbB = at_scale(Bs, sr); kappa = U.fit_kappa(eA, sA); res = dict(kappa_fitted_on_A=kappa)
        for lab, r, e, s in (("scene_A_fit", A, eA, sA), ("scene_B_heldout", Bs, eB, sB)):
            s0 = r["truth"].std()
            res[lab] = dict(rms_error_pct=100 * float(np.sqrt(np.mean(e ** 2)) / s0), median_sigma_boot_pct=100 * float(np.median(s) / s0),
                            coverage_raw=U.coverage(e, s), coverage_calibrated=U.coverage(e, kappa * s), **ranking(e, s, s0))
        cy, cx = Bs["change"]; w = 8; _, _, _, sbB_full = at_scale(Bs, sr); _, _, _, sbA_full = at_scale(A, sr)
        res["sigma_at_planted_change_over_median"] = {
            "A": float(sbA_full[A["change"][0] - w:A["change"][0] + w, A["change"][1] - w:A["change"][1] + w].mean() / np.median(sA)),
            "B": float(sbB_full[cy - w:cy + w, cx - w:cx + w].mean() / np.median(sB))}
        out["scales"][name] = res
    # two-component model (bootstrap + local texture) at the output pixel
    eA, sA, XA, _ = at_scale(A, 0); eB, sB, XB, _ = at_scale(Bs, 0)
    tA, tB = U.local_texture(XA, SC)[A["c"]], U.local_texture(XB, SC)[Bs["c"]]
    k, b = U.fit_two_component(eA, sA, tA); sdB = U.sigma_total(sB, tB, k, b)
    out["two_component_6.25cm"] = dict(kappa=k, beta=b, scene_B_coverage=U.coverage(eB, sdB), **ranking(eB, sdB, Bs["truth"].std()))
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", "bootstrap_uncertainty_simulation.json"), "w"), indent=1)

"""Experiment 7: calibrate P(rock) for the confidence-scored fill on simulations with exact truth, and test it on
held-out simulations.

Scene (32 x 32 m, 6.25 cm truth grid): fractal ground; rocks of 25/40/60/100/150 cm (counts 40/30/20/10/8) with random
brightness 0.4-1.2 x ground std above the ground and a shadow per pass sun geometry (shadow length (D/2) tan i);
decoys - flat bright patches of the same sizes and brightness, no height, no shadow (15 per size).
Passes: the 8 restoration passes' real sun geometry and measured blur (Section 3), random sub-pixel offsets, 3 % noise,
0.03 px registration error. Restoration: weighted PDE-TV (Section 2). Shadow evidence on the aligned passes (Section 4).
Candidates: blob maxima in the restoration (pipeline/confidence.blob_candidates). Label 1 if within max(3 px, D/2) of a
real rock's centre (decoys and ground are 0). Logistic model on [z, log D, S (D >= 100 cm), D >= 100 cm] fitted on
scenes 0-9, tested on scenes 10-15.
Output: results/confidence_calibration_simulation.json, data_products/confidence_calibrator.json
"""
import sys, os, json, math, time, numpy as np
from multiprocessing import Pool
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from experiments import scenes
from pipeline import pde_tv, shadow_evidence as se, confidence as cf

SC = 4; N_LR = int(os.environ.get("N_LR", 128)); ITERS = int(os.environ.get("ITERS", 200))
GEOMS = [dict(sun_az=a, incidence=i) for a, i in [(177.4, 47.6), (176.8, 49.9), (165.3, 45.8), (167.0, 46.0), (168.7, 46.2), (177.7, 47.7), (167.9, 58.9), (169.8, 55.6)]]
SIG = [1.336, 1.426, 0.889, 0.587, 0.828, 1.381, 1.16, 1.203]
COUNTS = {25: 40, 40: 30, 60: 20, 100: 10, 150: 8}; DECOYS = 15
N_TRAIN = int(os.environ.get("N_TRAIN", 10)); N_TEST = int(os.environ.get("N_TEST", 6))


def scene(seed):
    rng = np.random.default_rng(seed); n = N_LR * SC
    bg = scenes.fractal(1000 + seed, n=n); bg = bg[0] if isinstance(bg, tuple) else bg; bg = (bg - bg.mean()) / bg.std()
    objs = []
    def place(D, rock):
        for _ in range(500):
            y, x = rng.uniform(10, N_LR - 10, 2)
            if all(math.hypot(y - o[0], x - o[1]) > (D + o[2]) / 50 + 3 for o in objs): objs.append((y, x, D, rock, rng.uniform(0.4, 1.2))); return
    for D, c in sorted(COUNTS.items(), reverse=True):
        for _ in range(c): place(D, True)
        for _ in range(DECOYS): place(D, False)
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) / SC
    frames, flows = [], []
    for k, (g, s) in enumerate(zip(GEOMS, SIG)):
        sc = bg.copy(); sy, sx = se.shadow_vector(g["sun_az"])
        for y, x, D, rock, b in objs:
            r = D / 2 / 25; body = np.hypot(yy - y, xx - x) <= r; sc[body] += b
            if rock:
                L = (D / 2) * math.tan(math.radians(g["incidence"])) / 25; cy, cx = y + sy * (r + L / 2), x + sx * (r + L / 2)
                u = (yy - cy) * sy + (xx - cx) * sx; v = -(yy - cy) * sx + (xx - cx) * sy
                sc[((u / max(L / 2, .25)) ** 2 + (v / max(r, .25)) ** 2 <= 1) & ~body] -= 1.3 * b
        if k == 0: truth_img = sc
        dx, dy = (0.0, 0.0) if k == 0 else rng.uniform(-2, 2, 2)
        lr = ndi.gaussian_filter(ndi.shift(sc, (-SC * dy, -SC * dx), order=3, mode="reflect"), s * SC)[::SC, ::SC]
        frames.append(lr); ex, ey = (0.0, 0.0) if k == 0 else rng.normal(0, 0.03, 2)
        flows.append((np.full(lr.shape, dx + ex), np.full(lr.shape, dy + ey)))
    s0 = frames[0].std(); frames = [f + rng.normal(0, 0.03 * s0, f.shape) for f in frames]
    return frames, flows, objs


def run(seed):
    t = time.time(); frames, flows, objs = scene(seed)
    X, _ = pde_tv.pde_tv(frames, flows, scale=SC, psf_sigma=[s * SC for s in SIG], lam=0.005, iters=ITERS)
    aligned = cf.align_passes(frames, flows)
    shadow = se.evidence(aligned, GEOMS, SIG, sizes_cm=cf.SHADOW_SIZES)
    cands = [q for q in cf.blob_candidates(X) if 8 * SC <= q[0] < X.shape[0] - 8 * SC and 8 * SC <= q[1] < X.shape[1] - 8 * SC]
    F = cf.features(cands, shadow, SC)
    rocks = [(y * SC, x * SC, D) for y, x, D, r, _ in objs if r]
    def match(q):
        best = None
        for i, (y, x, D) in enumerate(rocks):
            d = math.hypot(q[0] - y, q[1] - x)
            if d <= max(3.0, D / 2 / cf.PX_HR_CM) and (best is None or d < best[1]): best = (i, d)
        return -1 if best is None else best[0]
    m = [match(q) for q in cands]
    return dict(seed=seed, F=F, y=np.array([int(i >= 0) for i in m]), match=np.array(m), D=np.array([q[2] for q in cands]),
                rockD=np.array([D for _, _, D in rocks]), seconds=time.time() - t)


def summarise(res, cal):
    F = np.concatenate([r["F"] for r in res]); y = np.concatenate([r["y"] for r in res]); D = np.concatenate([r["D"] for r in res])
    P = cf.predict(cal, F); out = dict(candidates=int(len(y)), real_rocks=int(sum(len(r["rockD"]) for r in res)),
                                       auc_all=cf.auc(P, y), auc_blob_z_only=cf.auc(F[:, 0], y), reliability=cf.reliability(P, y), by_size={}, sweep=[])
    for lo, hi, nm in ((0, 30, "25 cm"), (30, 50, "40 cm"), (50, 80, "60 cm"), (80, 120, "100 cm"), (120, 1e9, "150 cm")):
        m = (D >= lo) & (D < hi); out["by_size"][nm] = dict(n=int(m.sum()), auc=cf.auc(P[m], y[m]), mean_pred=float(P[m].mean()) if m.any() else None, observed=float(y[m].mean()) if m.any() else None)
    for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        shown = P >= th; found = 0; total = 0; rb = {}
        for r in res:
            Pr = cf.predict(cal, r["F"]); hit = set(r["match"][(Pr >= th) & (r["match"] >= 0)].tolist()); found += len(hit); total += len(r["rockD"])
            for i, Dt in enumerate(r["rockD"]): rb.setdefault(int(Dt), [0, 0]); rb[int(Dt)][0] += i in hit; rb[int(Dt)][1] += 1
        out["sweep"].append(dict(threshold=th, shown=int(shown.sum()), precision_pct=100 * float(y[shown].mean()) if shown.any() else None,
                                 recall_pct=100 * found / max(total, 1), recall_by_size_pct={k: 100 * v[0] / v[1] for k, v in sorted(rb.items())}))
    return out


if __name__ == "__main__":
    with Pool(int(os.environ.get("PROCS", 4))) as pool: res = pool.map(run, range(N_TRAIN + N_TEST))
    tr, te = res[:N_TRAIN], res[N_TRAIN:]
    F = np.concatenate([r["F"] for r in tr]); y = np.concatenate([r["y"] for r in tr])
    cal = cf.fit_logistic(F, y)
    F_ns = F.copy(); F_ns[:, 2] = 0; cal_noshadow = cf.fit_logistic(F_ns, y)
    te_ns = [dict(r, F=np.where(np.arange(4) == 2, 0, r["F"])) for r in te]
    out = dict(setup=dict(n_lr=N_LR, scale=SC, iters=ITERS, train_scenes=N_TRAIN, test_scenes=N_TEST, counts=COUNTS, decoys_per_size=DECOYS,
                          sigma_lr=SIG, geoms=GEOMS, seconds_per_scene=float(np.mean([r["seconds"] for r in res]))),
               model=cal, train=summarise(tr, cal), heldout=summarise(te, cal),
               heldout_without_shadow_feature=dict(model=cal_noshadow, **summarise(te_ns, cal_noshadow)))
    json.dump(out, open(os.path.join(ROOT, "results", "confidence_calibration_simulation.json"), "w"), indent=1)
    cf.save(dict(rock=cal, texture_confidence=None, sigma_limit_rel=None,
                 note="P(rock) = logistic(coef . [z, log(D/25), S*(D>=100), D>=100] + intercept); fitted on simulations (experiments/exp7_confidence_calibration.py)"),
            os.path.join(ROOT, "data_products", "confidence_calibrator.json"))
    h = out["heldout"]; print(json.dumps(dict(model=cal, auc=h["auc_all"], auc_z=h["auc_blob_z_only"], ece=h["reliability"]["ece"], by_size=h["by_size"],
                                              sweep=[(s["threshold"], s["shown"], s["precision_pct"], round(s["recall_pct"], 1)) for s in h["sweep"]],
                                              no_shadow_auc=out["heldout_without_shadow_feature"]["auc_all"]), indent=1))

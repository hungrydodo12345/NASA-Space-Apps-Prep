"""Tune each algorithm's regularisation on TUNING scenes (seeds >= 1000, never used for reporting), then freeze."""
import sys, json, itertools, numpy as np
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep")
from pipeline import synth
from experiments import algos, common as C, runner

GRID = dict(
    ibp=[dict(iters=i) for i in (10, 30, 60)],
    cg_tikhonov=[dict(mu=m) for m in (1e-4, 1e-3, 1e-2, 1e-1)],
    irls_tv=[dict(mu=m, eps=e) for m in (1e-3, 1e-2, 1e-1) for e in (0.02, 0.05)],
    chambolle_pock_tv=[dict(lam=l) for l in (5e-4, 2e-3, 8e-3)],
    richardson_lucy=[dict(iters=i) for i in (10, 20, 40, 80)],
    btv_l1=[dict(lam=l, beta=b) for l in (0.005, 0.02, 0.08) for b in (0.002, 0.004)],
)
SCEN = [(4, 0.01), (16, 0.01), (4, 0.05), (16, 0.05)]       # (frames, relative noise)


def job(j):
    rng = np.random.default_rng(j["seed"])
    truth = synth.fractal_terrain(160, seed=j["seed"] % 997, octaves=7, persistence=0.8)
    frames, est, _ = C.simulate(truth, j["n"], rng, noise_rel=j["noise"])
    bic = algos.reconstruct("bicubic", frames, est, C.SCALE, C.PSF_HR)
    hr = algos.reconstruct(j["algo"], frames, est, C.SCALE, C.PSF_HR, **j["hp"])
    return dict(algo=j["algo"], hp=j["hp"], n=j["n"], noise=j["noise"], **C.evaluate(hr, truth, bic))


if __name__ == "__main__":
    out = "/home/user/NASA-Space-Apps-Prep/results/tune.jsonl"
    jobs = []
    for a, grid in GRID.items():
        for hp in grid:
            for (n, noise), seed in itertools.product(SCEN, (1000, 1001)):
                jobs.append(dict(id=f"{a}|{json.dumps(hp, sort_keys=True)}|{n}|{noise}|{seed}", algo=a, hp=hp, n=n, noise=noise, seed=seed))
    runner.run(job, jobs, out)
    rows = [r for r in C.read(out) if r["ok"]]
    best = {}
    for a in GRID:
        by = {}
        for r in rows:
            if r["algo"] == a: by.setdefault(json.dumps(r["hp"], sort_keys=True), []).append(r["rmse_reduction_pct"])
        sc = {k: float(np.mean(v)) for k, v in by.items()}
        k = max(sc, key=sc.get); best[a] = json.loads(k)
        print(f"{a:20s} best {k}  -> {sc[k]:.1f}% error reduction | all: " + ", ".join(f"{kk}:{vv:.1f}" for kk, vv in sc.items()))
    json.dump(best, open("/home/user/NASA-Space-Apps-Prep/results/tuned_hp.json", "w"), indent=1)

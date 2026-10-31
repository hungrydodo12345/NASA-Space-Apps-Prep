"""Experiment 1: how much real detail does N-frame super-resolution recover?  (algorithm x frames x scene)

Scenes: fractal terrain (known statistics) and real rover ground texture (semi-real). Imaging model uses the
rover-calibrated HiRISE blur (sigma 0.78 px). Registration error 0.05 px, noise 2% of image std unless varied.
"""
import sys, json, itertools, os
import numpy as np
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep")
from experiments import algos, common as C, runner, scenes

TRUTH = sys.argv[1] if len(sys.argv) > 1 else None
HP = json.load(open("/home/user/NASA-Space-Apps-Prep/results/tuned_hp.json"))
SCENES = {}


def get_scene(kind, idx):
    if kind == "fractal": return scenes.fractal(2000 + idx)
    if "rover" not in SCENES: SCENES["rover"] = scenes.load_rover_scenes(TRUTH)
    s = SCENES["rover"][idx % len(SCENES["rover"])]
    return s[1], s[2]


def job(j):
    truth, valid = get_scene(j["scene"], j["idx"])
    rng = np.random.default_rng(10_000 * j["idx"] + j["n"] + (7 if j["scene"] == "rover" else 0))
    frames, est, _ = C.simulate(truth, j["n"], rng, noise_rel=j["noise"], reg_err=j["reg"], light=j["light"], psf_lr=j["psf"])
    bic = algos.reconstruct("bicubic", frames, est, C.SCALE, j["psf"] * C.SCALE)
    hr = algos.reconstruct(j["algo"], frames, est, C.SCALE, j["psf"] * C.SCALE, **HP.get(j["algo"], {}))
    r = C.evaluate(hr, truth, bic, valid)
    return dict({k: j[k] for k in ("algo", "scene", "idx", "n", "noise", "reg", "light", "psf")}, **(r or {}))


def build(kind, n_scenes, ns, algs, **kw):
    base = dict(noise=0.02, reg=0.05, light=0.0, psf=C.PSF_LR); base.update(kw)
    jobs = []
    for idx, n, a in itertools.product(range(n_scenes), ns, algs):
        if a == "bicubic" and n > 1: continue
        j = dict(scene=kind, idx=idx, n=n, algo=a, **base)
        j["id"] = "|".join(f"{k}={j[k]}" for k in sorted(j))
        jobs.append(j)
    return jobs


if __name__ == "__main__":
    out = "/home/user/NASA-Space-Apps-Prep/results/exp1_scaling.jsonl"
    A = [a for a in algos.NAMES]
    jobs = build("fractal", 5, [1, 2, 4, 8, 16, 32], A)
    if TRUTH:
        nr = len(scenes.load_rover_scenes(TRUTH)); print("real rover scenes usable:", nr)
        jobs += build("rover", min(nr, 12), [1, 2, 4, 8, 16, 32], A)
    runner.run(job, jobs, out)

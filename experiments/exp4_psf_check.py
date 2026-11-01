"""Experiment 4: which blur estimator is right?  Recover a KNOWN blur from real rover ground texture.

Two estimators disagreed on real data (spectral: 0.78 px; alignment/NCC: pushed to its 2 px grid edge). Here HiRISE-like
images are made from real rover mosaics with a known Gaussian blur, 25 cm pixel footprint and noise, then both
estimators are run exactly as on real data.
"""
import sys, os, glob, json
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline import blur_calib as bc
from experiments import runner, common as C

TRUTH = sys.argv[1]
TRUE = [0.4, 0.6, 0.78, 1.0, 1.4, 2.0]
GRID = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0]
FILES = sorted(glob.glob(f"{TRUTH}/*.npz"))


def job(j):
    mos = np.load(FILES[j["i"]])["z"].astype(float)
    if np.isfinite(mos).mean() < 0.4: return dict(skip=True)
    rng = np.random.default_rng(j["i"] * 31 + int(j["sigma"] * 100))
    clean = bc.degrade(mos, 0.05, j["sigma"]); n = clean.shape[0]
    hi = clean + rng.normal(0, 0.02 * np.nanstd(clean), clean.shape)
    # spectral (registration-free) estimator, as scripts/calibrate_psf.py
    ref = bc.box_to_hirise(mos, 0.05)[:n, :n]
    f, ph = bc.radial_psd(np.where(np.isfinite(ref), hi, np.nan)); _, pr = bc.radial_psd(ref)
    ratio = (ph - np.median(ph[f > 0.42])) / pr; ratio /= np.median(ratio[f < 0.08])
    s_spec = bc.fit_sigma(f, ratio)
    # alignment (NCC) estimator, as the first real-data attempt: shifted HiRISE window, search shifts per trial blur
    S = 6; dy, dx = rng.integers(-3, 4, 2)
    big = np.pad(np.nan_to_num(hi, nan=np.nanmean(hi)), S, mode="reflect"); big = np.roll(np.roll(big, dy, 0), dx, 1)
    hp_big = bc.hp(big)
    nccs = [bc.best_ncc(bc.hp(bc.degrade(mos, 0.05, g)[:n, :n]), hp_big, S)[0] for g in GRID]
    s_ncc = GRID[int(np.argmax(nccs))]
    return dict(i=j["i"], sigma=j["sigma"], spectral=s_spec, ncc=s_ncc, ncc_curve=nccs)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exp4_psf_check.jsonl")
    jobs = [dict(id=f"{i}|{s}", i=i, sigma=s) for i in range(0, len(FILES), 2) for s in TRUE]
    runner.run(job, jobs, out)
    rows = [r for r in C.read(out) if r.get("ok") and not r.get("skip")]
    print(f"{'true sigma':>10s} | {'spectral estimate (median, IQR)':>34s} | {'alignment/NCC estimate (median)':>32s}")
    for s in TRUE:
        sp = [r["spectral"] for r in rows if r["sigma"] == s]; nc = [r["ncc"] for r in rows if r["sigma"] == s]
        print(f"{s:10.2f} | {np.median(sp):10.2f} ({np.percentile(sp,25):.2f}-{np.percentile(sp,75):.2f}) n={len(sp):<3d}        | {np.median(nc):10.2f} (share at grid max {np.mean(np.array(nc)==GRID[-1]):.0%})")

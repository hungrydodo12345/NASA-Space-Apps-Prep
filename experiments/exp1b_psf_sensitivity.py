"""Experiment 1b: re-run the rover-texture scaling at the blur implied by Experiment 4.

Exp 4 showed the spectral estimator reads low for wide blur; inverting its calibration curve, the real-data reading of
0.78 px corresponds to a true sigma of ~1.07 px (IQR 0.84-1.36). Same jobs as Exp 1 (rover scenes), at sigma 1.07 and 1.36.
Hyperparameters are those tuned at 0.78 px (not re-tuned).
Usage: python experiments/exp1b_psf_sensitivity.py <rover_truth_dir>
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments import exp1_scaling as X, scenes, runner

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exp1b_psf_sensitivity.jsonl")
    nr = min(len(scenes.load_rover_scenes(X.TRUTH)), 12)
    A = ["bicubic", "average", "ibp", "cg_tikhonov", "btv_l1", "richardson_lucy"]
    jobs = []
    for s in (1.07, 1.36): jobs += X.build("rover", nr, [1, 4, 8, 16], A, psf=s)
    runner.run(X.job, jobs, out)

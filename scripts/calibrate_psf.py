"""Measure the HiRISE point-spread function against rover ground truth (registration-free, spectral).

Usage: python scripts/calibrate_psf.py <rover_truth_dir> <ortho_jp2> <out_json>
"""
import sys, os, json, glob
import numpy as np, glymur
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline import groundtruth as gt, blur_calib as bc

TRUTH, ORTHO, OUT = sys.argv[1:4]
J = glymur.Jp2k(ORTHO)
sig, curves = [], []
for f in sorted(glob.glob(f"{TRUTH}/*.npz")):
    d = np.load(f); mos = d["z"].astype(float)
    if np.isfinite(mos).mean() < 0.4: continue
    ref = bc.box_to_hirise(mos, 0.05); n = ref.shape[0]
    L0, S0 = gt.latlon_to_ortho(float(d["lat"]), float(d["lon"]))
    hi = np.asarray(J[int(L0) - n // 2:int(L0) - n // 2 + n, int(S0) - n // 2:int(S0) - n // 2 + n]).astype(float)
    hi = np.where(np.isfinite(ref), hi, np.nan)
    fr, ph = bc.radial_psd(hi); _, pr = bc.radial_psd(ref)
    ratio = (ph - np.median(ph[fr > 0.42])) / pr; ratio /= np.median(ratio[fr < 0.08])
    sig.append(bc.fit_sigma(fr, ratio)); curves.append(ratio)
    print(f"{os.path.basename(f)[:-4]:8s}: sigma {sig[-1]:.2f} px")
sig = np.array(sig)
res = dict(sigma_px_median=float(np.nanmedian(sig)), sigma_px_iqr=[float(np.nanpercentile(sig, 25)), float(np.nanpercentile(sig, 75))],
           n_stops=int(len(sig)), f=fr.tolist(), mtf2_median=np.nanmedian(curves, 0).tolist())
json.dump(res, open(OUT, "w"), indent=1)
print(f"HiRISE PSF sigma {res['sigma_px_median']:.2f} px (IQR {res['sigma_px_iqr'][0]:.2f}-{res['sigma_px_iqr'][1]:.2f}) over {len(sig)} rover stops")

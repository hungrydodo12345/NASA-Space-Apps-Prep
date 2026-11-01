"""Shared pieces: measured imaging parameters, observation simulator, metrics, statistics."""
import json, os, time
import numpy as np
from scipy import ndimage as ndi
from pipeline import metrics as pm, synth

HIRISE_PX_CM = 25.241583171779
SCALE = 4                                   # 25.24 cm -> 6.31 cm
OUT_PX_CM = HIRISE_PX_CM / SCALE
_cal = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data_products", "psf_calibration.json")))
PSF_LR = _cal["sigma_px_median"]            # measured against rover ground truth (HiRISE px)
PSF_HR = PSF_LR * SCALE


def simulate(truth, n, rng, scale=SCALE, psf_lr=PSF_LR, noise_rel=0.02, reg_err=0.05, max_shift=3.0, light=0.0):
    """Observation model of repeat-pass imaging. truth at output grid; returns (frames, est_shifts, true_shifts).

    noise_rel: noise std relative to the std of the clean LR image. reg_err: registration error std (LR px) fed to the
    solver. light: amplitude of smooth multiplicative illumination mismatch between passes (shadows / sun angle).
    """
    sig = psf_lr * scale
    true = [(0.0, 0.0)] + [tuple(rng.uniform(-max_shift, max_shift, 2)) for _ in range(n - 1)]
    frames = []
    for dx, dy in true:
        sh = ndi.shift(truth, (-scale * dy, -scale * dx), order=3, mode="reflect")
        lr = ndi.gaussian_filter(sh, sig)[::scale, ::scale]
        if light > 0:
            f = ndi.gaussian_filter(rng.standard_normal(lr.shape), 6.0); f = f / (f.std() + 1e-9)
            lr = lr * (1 + light * f) * (1 + 0.5 * light * rng.standard_normal())
        frames.append(lr)
    s0 = np.std(frames[0])
    frames = [f + rng.normal(0, noise_rel * s0, f.shape) for f in frames]
    est = [(0.0, 0.0)] + [(dx + rng.normal(0, reg_err), dy + rng.normal(0, reg_err)) for dx, dy in true[1:]]
    return frames, est, true


def welch_coherence(a, b, valid=None, size=64, stride=16, nb=32):
    """Radial magnitude-squared coherence of a vs b, Welch-averaged over fully valid size x size blocks."""
    H, W = a.shape; win = np.outer(np.hanning(size), np.hanning(size))
    fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij"); fr = np.hypot(fx, fy)
    edges = np.linspace(0.0, 0.5, nb + 1)
    sab = np.zeros((size, size), complex); saa = np.zeros((size, size)); sbb = np.zeros((size, size)); cnt = 0
    for y in range(0, H - size + 1, stride):
        for x in range(0, W - size + 1, stride):
            if valid is not None and not valid[y:y + size, x:x + size].all(): continue
            A = np.fft.fft2((a[y:y + size, x:x + size] - a[y:y + size, x:x + size].mean()) * win)
            B = np.fft.fft2((b[y:y + size, x:x + size] - b[y:y + size, x:x + size].mean()) * win)
            sab += A * np.conj(B); saa += np.abs(A) ** 2; sbb += np.abs(B) ** 2; cnt += 1
    if cnt == 0: return None, None
    coh = np.array([np.abs(sab[(fr >= edges[i]) & (fr < edges[i + 1])].sum()) ** 2 /
                    max(saa[(fr >= edges[i]) & (fr < edges[i + 1])].sum() * sbb[(fr >= edges[i]) & (fr < edges[i + 1])].sum(), 1e-30) for i in range(nb)])
    return 0.5 * (edges[1:] + edges[:-1]), coh


def evaluate(hr, truth, bic, valid=None, margin=12, block=64):
    """Scores of a reconstruction against ground truth (all percentages / cm).

    rmse_reduction_pct : 100*(1 - RMSE/RMSE_bicubic)   (positive = better than bicubic)
    beyond_nyquist_pct : mean coherence (x100) in the band above the single-frame sampling limit (the detail only
                         multi-frame reconstruction can recover)
    eff_res_cm         : smallest feature (half period) at which coherence with truth stays >= 50%
    """
    sl = (slice(margin, -margin), slice(margin, -margin))
    v = np.ones_like(truth, bool) if valid is None else valid
    v = ndi.binary_erosion(v, iterations=1)[sl]
    e = lambda x: np.sqrt(np.mean((x[sl][v] - truth[sl][v]) ** 2))
    vv = None if valid is None else valid[sl]
    f, c = welch_coherence(hr[sl], truth[sl], vv, size=block, stride=max(4, block // 4), nb=block // 2)
    if c is None: return None
    lr_nyq = 0.5 / SCALE
    band = f > lr_nyq
    below = np.where(c < 0.5)[0]
    fc = f[below[0]] if len(below) else 0.5
    return dict(rmse_reduction_pct=float(100 * (1 - e(hr) / e(bic))), beyond_nyquist_pct=float(100 * c[band].mean()),
                eff_res_cm=float(OUT_PX_CM * 0.5 / max(fc, 1e-6)), rmse=float(e(hr)))


def boot_ci(x, n=2000, seed=0, q=(2.5, 97.5)):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    r = np.random.default_rng(seed); m = r.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, q[0])), float(np.percentile(m, q[1]))


def log(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f: f.write(json.dumps(rec) + "\n")


def read(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []

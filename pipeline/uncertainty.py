"""Per-pixel uncertainty of the multi-pass restoration by a pass-subset bootstrap.

The restoration X = R(Y_1..Y_K) (pipeline.pde_tv) is rebuilt B times, each time from K passes drawn with replacement
from the K available (a pass drawn twice counts twice). The spread of the B rebuilds at each pixel,
    sigma_boot(p) = std_b X_b(p),
measures how much the answer depends on which passes happened to be taken: noise, registration error, per-pass
blur and changes between passes all show up in it. What it cannot see is error shared by every pass (the regulariser's
smoothing of detail below the HiRISE pixel, a wrong blur model common to all passes). That shared part is measured
in simulation, where the truth is known, and absorbed by one calibration factor kappa:
    sigma(p) = kappa * sigma_boot(p),
chosen so that the observed share of |X - truth| <= sigma matches the Gaussian 68.3 %. kappa and the coverage it gives
at 1 and 2 sigma are reported (experiments/exp6_bootstrap_uncertainty.py); sigma feeds the confidence model (Section 6).
"""
import numpy as np
from .pde_tv import pde_tv


def bootstrap(frames, flows, psf_sigma, B=12, seed=0, **kw):
    """frames, flows as for pde_tv; psf_sigma scalar or per-pass list (HR px).
    Returns dict(mean, std, members (B x H x W float32), draws (list of pass-index lists))."""
    rng = np.random.default_rng(seed); n = len(frames)
    sig = list(psf_sigma) if np.ndim(psf_sigma) else [psf_sigma] * n
    members, draws = [], []
    for b in range(B):
        idx = sorted(rng.integers(0, n, n).tolist())
        while len(set(idx)) < 2: idx = sorted(rng.integers(0, n, n).tolist())       # need >=2 distinct passes
        X, _ = pde_tv([frames[i] for i in idx], [flows[i] for i in idx], psf_sigma=[sig[i] for i in idx], **kw)
        members.append(X.astype(np.float32)); draws.append(idx)
    M = np.stack(members)
    return dict(mean=M.mean(0), std=M.std(0, ddof=1), members=M, draws=draws)


def coverage(err, sigma, ks=(1.0, 2.0)):
    """Share of pixels with |err| <= k sigma, for each k."""
    return {k: float(np.mean(np.abs(err) <= k * sigma)) for k in ks}


def fit_kappa(err, sigma, target=0.6827):
    """Smallest kappa with share(|err| <= kappa sigma) >= target (the quantile of |err| / sigma)."""
    return float(np.quantile(np.abs(err) / np.maximum(sigma, 1e-12), target))


def local_texture(X, scale=4, size=None):
    """Local amplitude of restored detail: RMS over a window of X minus its blur at one native pixel. Fine detail the
    passes could not resolve is error shared by every bootstrap member; it is largest where the ground is textured,
    so this map stands in for it (scaled by beta, fitted in simulation)."""
    from scipy import ndimage as ndi
    hp = X - ndi.gaussian_filter(X, scale)
    return np.sqrt(ndi.uniform_filter(hp * hp, size or 2 * scale + 1))


def fit_two_component(err, s_boot, tex):
    """sigma^2 = (kappa s_boot)^2 + (beta tex)^2, fitted by Gaussian maximum likelihood (grid search, log-spaced)."""
    e2 = err.ravel() ** 2; a = s_boot.ravel() ** 2; t = tex.ravel() ** 2; best = (np.inf, 1.0, 0.0)
    for k in np.geomspace(0.3, 30, 41):
        for b in np.concatenate([[0], np.geomspace(0.01, 10, 61)]):
            v = k * k * a + b * b * t + 1e-12; nll = float(np.mean(np.log(v) + e2 / v))
            if nll < best[0]: best = (nll, float(k), float(b))
    return best[1], best[2]


def sigma_total(s_boot, tex, kappa, beta):
    return np.sqrt((kappa * s_boot) ** 2 + (beta * tex) ** 2)

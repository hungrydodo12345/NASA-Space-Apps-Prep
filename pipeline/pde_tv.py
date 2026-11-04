"""PDE-TV MAP multi-pass restoration (re-implementation of Tao & Muller 2016, Eqs. 5-12), with quality-weighted passes and
change masking.

Energy minimised by steepest descent:
    E(X) = sum_k w_k * sum_p m_kp * (A_k X - Y_k)_p^2  +  lam * sum_p sqrt(|lap X|_p^2 + mu^2)
A_k = D H_k F_k (motion from Gotcha, blur H_k per pass, decimation D). The regulariser's gradient is the fourth-order PDE
term  lap( lap X / sqrt(|lap X|^2 + mu^2) )  of Eq. 8, which avoids TV's staircasing.
Weighting (not in the original): w_k = inverse white-noise variance of pass k, measured from the pass's own finest-scale
pixel differences (pass_noise; pass_weights="noise", default). The earlier variant pass_weights="residual" used the
robust residual variance against the current estimate, which on real HiRISE mistakes lighting differences between passes
for noise and collapses onto a single pass (docs/METHODS.md Section 2). m_kp are
iteratively re-weighted (Tukey biweight) per-pixel weights, so pixels where a pass disagrees with the rest - moving sand,
new rover tracks, the rover itself, dust-devil streaks - stop pulling the solution. Both are re-estimated every
`reweight_every` iterations. Set weighted=False for the original, unweighted GPT data term.
"""
import numpy as np
from scipy import ndimage as ndi
from .fuse_flow import FlowModel


def _lap(x): return ndi.laplace(x, mode="nearest")


def pass_noise(f):
    """Per-pass white-noise level from the finest-scale pixel differences (Immerkaer 1996: a Laplacian-difference
    kernel that cancels smooth structure, robust MAD). Slow lighting or season differences between passes do not enter."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    r = ndi.convolve(f.astype(np.float64), k, mode="nearest")[2:-2, 2:-2]
    return float(1.4826 * np.median(np.abs(r - np.median(r))) / 6.0) + 1e-12


def pde_tv(frames, flows, scale=4, psf_sigma=4.28, lam=0.02, mu=None, iters=300, step=0.9, weighted=True, reweight_every=20,
           tukey_c=4.685, exclude=None, pass_weights="noise", init="ref", mask_mode="tukey", consensus_c=3.0, consensus_smooth=1.0):
    """frames: list of LR passes (pass 0 = reference); flows: per-pass (dx, dy) in LR px (pass(x) ~= ref(x + d)).
    psf_sigma: HR px, scalar or per-pass list. exclude: optional LR mask of pixels to ignore in every pass (e.g. the rover).
    Returns (X, info) with per-pass weights and per-pixel weight maps."""
    n = len(frames); m = FlowModel(frames[0].shape, flows, scale, psf_sigma); HH, WW = m.hr
    Y = [f.astype(np.float64) for f in frames]; s0 = float(np.std(Y[0])) + 1e-12
    mu = 0.2 * s0 if mu is None else mu
    if init == "median":                                       # robust start: per-pixel median of all passes on the reference grid (a rover present in a minority of passes is not the start point)
        yy, xx = np.mgrid[0:Y[0].shape[0], 0:Y[0].shape[1]].astype(float)
        base = np.median([ndi.map_coordinates(Y[k], [yy - flows[k][1], xx - flows[k][0]], order=3, mode="nearest") for k in range(n)], 0)
    else: base = Y[0]
    X = ndi.zoom(base, scale, order=3, grid_mode=True, mode="nearest"); Xp = X.copy()
    wk = np.ones(n); M = [np.ones_like(Y[0]) for _ in range(n)]
    nz = np.array([pass_noise(y) for y in Y]); wk_noise = (nz.min() / nz) ** 2                # inverse noise variance, max 1
    if weighted and pass_weights == "noise": wk = wk_noise.copy()
    if pass_weights == "equal": wk_noise = np.ones(n)                                       # change masks only
    if exclude is not None: M = [mm * (~exclude) for mm in M]
    if weighted and mask_mode == "consensus":
        # change masks fixed once against the per-pixel median of ALL passes (majority vote, no feedback through X).
        # The difference is smoothed before thresholding: real changes are spatially coherent, white noise is not.
        h, w_ = Y[0].shape; yy, xx = np.mgrid[0:h, 0:w_].astype(float)
        al = [ndi.map_coordinates(Y[k], [yy - flows[k][1], xx - flows[k][0]], order=3, mode="nearest") for k in range(n)]
        med = np.median(al, 0)
        for k in range(n):
            mk = ndi.map_coordinates(med, [yy + flows[k][1], xx + flows[k][0]], order=3, mode="nearest")   # median seen on pass k's own grid
            dsm = ndi.gaussian_filter(Y[k] - mk, consensus_smooth); sc = 1.4826 * np.median(np.abs(dsm - np.median(dsm))) + 1e-12
            u = dsm / (consensus_c * sc); M[k] = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
            if exclude is not None: M[k] = M[k] * (~exclude)
    # Lipschitz-safe step, measured rather than assumed: the largest eigenvalue of A^T A is found by power iteration
    # (it depends on the blur and sampling, e.g. 0.69 for sigma = 4.28 HR px at scale 4); regulariser: |lap|^2 <= 64, slope s0/mu
    v = np.random.default_rng(0).standard_normal((HH, WW)); Ld = 0.0
    for _ in range(12):
        u = sum(m.adjoint(m.forward(v, k), k) for k in range(n)) / n; Ld = float(np.linalg.norm(u) / np.linalg.norm(v)); v = u / (np.linalg.norm(u) + 1e-30)
    gamma = step / (2.0 * Ld * 1.1 + lam * s0 * 64.0 / mu)
    for it in range(iters):
        if weighted and mask_mode == "tukey" and it % reweight_every == 0 and it > 0:
            R = [m.forward(X, k) - Y[k] for k in range(n)]
            sig = np.array([1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9 for r in R])
            if pass_weights == "residual": wk = (sig.min() / sig) ** 2         # original variant: lighting differences count as noise
            else: wk = wk_noise
            for k in range(n):
                u = R[k] / (tukey_c * sig[k]); M[k] = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
                if exclude is not None: M[k] = M[k] * (~exclude)
        Z = X + (it / (it + 3.0)) * (X - Xp)                   # Nesterov momentum
        g = np.zeros((HH, WW))
        for k in range(n): g += wk[k] * m.adjoint(M[k] * (m.forward(Z, k) - Y[k]), k)
        g *= 2.0 / max(wk.sum(), 1e-9)                           # weighted AVERAGE over passes (matches the step bound)
        lz = _lap(Z); g += lam * s0 * _lap(lz / np.sqrt(lz * lz + mu * mu))
        Xp = X; X = Z - gamma * g
    return X, dict(pass_weights=wk.tolist(), pixel_weights=M, model=m)

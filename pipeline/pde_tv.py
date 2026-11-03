"""PDE-TV MAP multi-pass restoration (re-implementation of Tao & Muller 2016, Eqs. 5-12), with quality-weighted passes and
change masking.

Energy minimised by steepest descent:
    E(X) = sum_k w_k * sum_p m_kp * (A_k X - Y_k)_p^2  +  lam * sum_p sqrt(|lap X|_p^2 + mu^2)
A_k = D H_k F_k (motion from Gotcha, blur H_k per pass, decimation D). The regulariser's gradient is the fourth-order PDE
term  lap( lap X / sqrt(|lap X|^2 + mu^2) )  of Eq. 8, which avoids TV's staircasing.
Weighting (not in the original): w_k = 1 / robust residual variance of pass k against the current estimate; m_kp are
iteratively re-weighted (Tukey biweight) per-pixel weights, so pixels where a pass disagrees with the rest - moving sand,
new rover tracks, the rover itself, dust-devil streaks - stop pulling the solution. Both are re-estimated every
`reweight_every` iterations. Set weighted=False for the original, unweighted GPT data term.
"""
import numpy as np
from scipy import ndimage as ndi
from .fuse_flow import FlowModel


def _lap(x): return ndi.laplace(x, mode="nearest")


def pde_tv(frames, flows, scale=4, psf_sigma=4.28, lam=0.02, mu=None, iters=300, step=0.9, weighted=True, reweight_every=20,
           tukey_c=4.685, exclude=None):
    """frames: list of LR passes (pass 0 = reference); flows: per-pass (dx, dy) in LR px (pass(x) ~= ref(x + d)).
    psf_sigma: HR px, scalar or per-pass list. exclude: optional LR mask of pixels to ignore in every pass (e.g. the rover).
    Returns (X, info) with per-pass weights and per-pixel weight maps."""
    n = len(frames); m = FlowModel(frames[0].shape, flows, scale, psf_sigma); HH, WW = m.hr
    Y = [f.astype(np.float64) for f in frames]; s0 = float(np.std(Y[0])) + 1e-12
    mu = 0.2 * s0 if mu is None else mu
    X = ndi.zoom(Y[0], scale, order=3, grid_mode=True, mode="nearest"); Xp = X.copy()
    wk = np.ones(n); M = [np.ones_like(Y[0]) for _ in range(n)]
    if exclude is not None: M = [mm * (~exclude) for mm in M]
    # Lipschitz-safe step, measured rather than assumed: the largest eigenvalue of A^T A is found by power iteration
    # (it depends on the blur and sampling, e.g. 0.69 for sigma = 4.28 HR px at scale 4); regulariser: |lap|^2 <= 64, slope s0/mu
    v = np.random.default_rng(0).standard_normal((HH, WW)); Ld = 0.0
    for _ in range(12):
        u = sum(m.adjoint(m.forward(v, k), k) for k in range(n)) / n; Ld = float(np.linalg.norm(u) / np.linalg.norm(v)); v = u / (np.linalg.norm(u) + 1e-30)
    gamma = step / (2.0 * Ld * 1.1 + lam * s0 * 64.0 / mu)
    for it in range(iters):
        if weighted and it % reweight_every == 0 and it > 0:
            R = [m.forward(X, k) - Y[k] for k in range(n)]
            sig = np.array([1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9 for r in R])
            wk = (sig.min() / sig) ** 2
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

"""Edge-preserving (Huber-TV) regularised multi-frame SR via iteratively reweighted CG.

Still non-generative: a convex, deterministic estimator of the measured frames. The prior only
penalises gradient magnitude (it adds no texture), so it can sharpen edges but never invent detail.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from .fuse_cg import _Model


def fuse_tv(frames, shifts, scale=4, psf_sigma=1.6, mu=1e-2, eps=0.02, outer=6, inner=30, lam=1e-6):
    m = _Model(frames[0].shape, shifts, scale, psf_sigma)
    n, (H, W) = len(frames), m.shape
    b = np.zeros((H, W), complex)
    for k in range(n):
        b += m.adjoint(frames[k].astype(np.float64), k)
    b = np.fft.ifft2(b).real.ravel()
    x = np.full(H * W, float(np.mean([f.mean() for f in frames])))
    w = np.ones((H, W))                                  # edge weights
    for _ in range(outer):
        def normal(v, w=w):
            V = v.reshape(H, W)
            X = np.fft.fft2(V)
            acc = np.zeros((H, W), complex)
            for k in range(n):
                acc += m.adjoint(m.forward(X, k), k)
            data = np.fft.ifft2(acc).real
            gx = np.roll(V, -1, 1) - V; gy = np.roll(V, -1, 0) - V
            div = (w * gx - np.roll(w * gx, 1, 1)) + (w * gy - np.roll(w * gy, 1, 0))
            return (data - mu * div + lam * V).ravel()
        op = LinearOperator((H * W, H * W), matvec=normal, dtype=np.float64)
        x, _ = cg(op, b, x0=x, maxiter=inner, rtol=1e-7)
        V = x.reshape(H, W)
        gx = np.roll(V, -1, 1) - V; gy = np.roll(V, -1, 0) - V
        w = 1.0 / np.sqrt(gx ** 2 + gy ** 2 + eps ** 2) * eps   # Huber/TV reweighting, w=1 on flat areas
    return x.reshape(H, W)

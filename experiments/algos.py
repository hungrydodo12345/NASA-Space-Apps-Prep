"""Super-resolution reconstruction algorithms under one interface (all deterministic, none generative).

    reconstruct(name, frames, shifts, scale, psf_hr, **hp) -> HR image   (frames are LR arrays, shifts in LR px,
    img(x) = ref(x + d); psf_hr = Gaussian sigma in HR pixels). Operators are the exact FFT models of pipeline.fuse_cg
    (periodic boundary - callers crop a margin).
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.linalg import LinearOperator, cg
from pipeline import fuse, fuse_cg, fuse_tv
from pipeline.fuse_cg import _Model

NAMES = ["bicubic", "average", "ibp", "cg_tikhonov", "irls_tv", "chambolle_pock_tv", "richardson_lucy", "btv_l1"]
LABEL = {"bicubic": "Bicubic (1 frame)", "average": "Shift-and-average", "ibp": "Iterative back-projection",
         "cg_tikhonov": "CG + Tikhonov", "irls_tv": "IRLS Huber-TV", "chambolle_pock_tv": "Chambolle-Pock TV",
         "richardson_lucy": "Richardson-Lucy", "btv_l1": "Bilateral-TV L1 (robust)"}
# regularisation weights tuned ONCE on separate tuning scenes (seeds >= 1000) by experiments/tune.py; frozen afterwards
HP = dict(cg_tikhonov=dict(mu=1e-2, iters=60), ibp=dict(iters=30), irls_tv=dict(mu=1e-2, eps=0.05, outer=4, inner=25),
          chambolle_pock_tv=dict(lam=2e-3, iters=250), richardson_lucy=dict(iters=40), btv_l1=dict(lam=0.02, beta=0.004, iters=150))


def upsample(a, s):
    return ndi.zoom(a.astype(np.float64), s, order=3, grid_mode=True, mode="nearest")


def _average(frames, shifts, s):
    acc = np.zeros((frames[0].shape[0] * s, frames[0].shape[1] * s))
    yy, xx = np.mgrid[0:acc.shape[0], 0:acc.shape[1]].astype(float)
    for f, (dx, dy) in zip(frames, shifts):
        up = upsample(f, s)
        acc += ndi.map_coordinates(up, [yy - s * dy, xx - s * dx], order=1, mode="nearest")   # f(x)=ref(x+d): ref(y)=f(y-d)
    return acc / len(frames)


def _chambolle_pock(frames, shifts, s, psf, lam, iters):
    m = _Model(frames[0].shape, shifts, s, psf); n = len(frames); H, W = m.hr_shape if hasattr(m, "hr_shape") else m.shape
    y = [f.astype(np.float64) for f in frames]
    x = upsample(frames[0], s); xb = x.copy()
    p = [np.zeros_like(f) for f in y]; q = np.zeros((2, H, W))
    sig = tau = 0.99 / np.sqrt(n + 8.0)
    grad = lambda v: np.stack([np.diff(v, axis=1, append=v[:, -1:]), np.diff(v, axis=0, append=v[-1:, :])])
    def divT(g):                                   # exact adjoint of grad
        dx = np.zeros((H, W)); dx[:, :-1] -= g[0][:, :-1]; dx[:, 1:] += g[0][:, :-1]
        dy = np.zeros((H, W)); dy[:-1, :] -= g[1][:-1, :]; dy[1:, :] += g[1][:-1, :]
        return dx + dy
    for _ in range(iters):
        X = np.fft.fft2(xb)
        for k in range(n): p[k] = (p[k] + sig * (m.forward(X, k) - y[k])) / (1 + sig)
        q = q + sig * grad(xb); nrm = np.maximum(1.0, np.hypot(q[0], q[1]) / lam); q = q / nrm
        back = np.zeros((H, W), complex)
        for k in range(n): back += m.adjoint(p[k], k)
        xn = x - tau * (np.fft.ifft2(back).real + divT(q))
        xb = 2 * xn - x; x = xn
    return x


def _richardson_lucy(frames, shifts, s, psf, iters):
    m = _Model(frames[0].shape, shifts, s, psf); n = len(frames); H, W = m.shape
    off = 1.0 - min(f.min() for f in frames); y = [f.astype(np.float64) + off for f in frames]
    x = np.maximum(upsample(frames[0], s) + off, 1e-3)
    norm = [np.fft.ifft2(m.adjoint(np.ones_like(f), k)).real for k, f in enumerate(y)]
    for _ in range(iters):
        X = np.fft.fft2(x); acc = np.zeros((H, W))
        for k in range(n):
            ratio = y[k] / np.maximum(m.forward(X, k), 1e-6)
            acc += np.fft.ifft2(m.adjoint(ratio, k)).real / np.maximum(norm[k], 1e-9)
        x = np.maximum(x * acc / n, 1e-6)
    return x - off


def _btv_l1(frames, shifts, s, psf, lam, beta, iters, P=2, alpha=0.7):
    m = _Model(frames[0].shape, shifts, s, psf); n = len(frames); H, W = m.shape
    y = [f.astype(np.float64) for f in frames]; x = upsample(frames[0], s)
    for it in range(iters):
        X = np.fft.fft2(x); acc = np.zeros((H, W), complex)
        for k in range(n): acc += m.adjoint(np.sign(m.forward(X, k) - y[k]), k)
        g = (s * s / n) * np.fft.ifft2(acc).real
        for l in range(-P, P + 1):
            for mm in range(0, P + 1):
                if (l == 0 and mm == 0) or (mm == 0 and l < 0): continue
                sh = np.roll(np.roll(x, l, 1), mm, 0); sg = np.sign(x - sh)
                g += lam * alpha ** (abs(l) + mm) * (sg - np.roll(np.roll(sg, -l, 1), -mm, 0))
        x = x - beta * (1 - it / (1.2 * iters)) * g
    return x


def reconstruct(name, frames, shifts, scale, psf_hr, **over):
    hp = {**HP.get(name, {}), **over}
    if name == "bicubic": return upsample(frames[0], scale)
    if name == "average": return _average(frames, shifts, scale)
    if name == "ibp": return fuse.fuse(frames, shifts, scale=scale, psf_sigma=psf_hr, iters=hp["iters"])[0]
    if name == "cg_tikhonov": return fuse_cg.fuse_cg(frames, shifts, scale=scale, psf_sigma=psf_hr, mu=hp["mu"], iters=hp["iters"])[0]
    if name == "irls_tv": return fuse_tv.fuse_tv(frames, shifts, scale=scale, psf_sigma=psf_hr, mu=hp["mu"], eps=hp["eps"], outer=hp["outer"], inner=hp["inner"])
    if name == "chambolle_pock_tv": return _chambolle_pock(frames, shifts, scale, psf_hr, hp["lam"], hp["iters"])
    if name == "richardson_lucy": return _richardson_lucy(frames, shifts, scale, psf_hr, hp["iters"])
    if name == "btv_l1": return _btv_l1(frames, shifts, scale, psf_hr, hp["lam"], hp["beta"], hp["iters"])
    raise KeyError(name)

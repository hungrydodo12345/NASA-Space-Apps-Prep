"""Dense-motion multi-frame super-resolution (deterministic, non-generative).

Observation model per frame k:   y_k = D( G( W_k x ) ) + noise
  W_k : per-pixel sub-pixel warp  (bilinear, sparse; shifted(p) = x(p + s*d_k(p)))
  G   : Gaussian PSF,  D : decimation by `scale`.
Solved as regularised least squares by CG, with exact sparse adjoints (W^T) so the system is well-posed.
Terrain parallax / pointing error make d_k vary across the image, so a single rigid shift per frame is wrong.
"""
import numpy as np
import scipy.sparse as sp
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import LinearOperator, cg
from . import register


def dense_shift_field(ref, img, tile=96, step=32):
    """Per-pixel (dx, dy) in LR px, img(x) ~= ref(x + d(x)); tile-wise ECC, smoothed and interpolated."""
    H, W = ref.shape
    hp = lambda a: (a - ndi.gaussian_filter(a, 5)).astype(np.float32)
    rh, ih = hp(ref), hp(img)
    cy = np.arange(tile // 2, H - tile // 2 + 1, step); cx = np.arange(tile // 2, W - tile // 2 + 1, step)
    fx = np.zeros((len(cy), len(cx))); fy = fx.copy()
    for i, y in enumerate(cy):
        for j, x in enumerate(cx):
            a = rh[y - tile // 2:y + tile // 2, x - tile // 2:x + tile // 2]; b = ih[y - tile // 2:y + tile // 2, x - tile // 2:x + tile // 2]
            try: fx[i, j], fy[i, j] = register.estimate_shift(a, b)
            except Exception: fx[i, j] = fy[i, j] = np.nan
    for f in (fx, fy):
        bad = ~np.isfinite(f); f[bad] = np.nanmedian(f)
    fx, fy = ndi.median_filter(fx, 3, mode="nearest"), ndi.median_filter(fy, 3, mode="nearest")
    yy, xx = np.mgrid[0:H, 0:W]
    out = []
    for f in (fx, fy):
        itp = RegularGridInterpolator((cy, cx), f, bounds_error=False, fill_value=None)
        out.append(itp(np.stack([yy.ravel(), xx.ravel()], 1)).reshape(H, W))
    return out[0], out[1]


def _warp_matrix(dx_lr, dy_lr, scale):
    H, W = dx_lr.shape
    HH, WW = H * scale, W * scale
    up = lambda f: ndi.zoom(f, scale, order=1, grid_mode=True, mode="nearest")
    dX, dY = up(dx_lr) * scale, up(dy_lr) * scale
    yy, xx = np.mgrid[0:HH, 0:WW].astype(np.float64)
    qy, qx = np.clip(yy + dY, 0, HH - 1.001), np.clip(xx + dX, 0, WW - 1.001)
    y0, x0 = np.floor(qy).astype(int), np.floor(qx).astype(int)
    ty, tx = qy - y0, qx - x0
    rows = np.arange(HH * WW)
    idx = lambda y, x: (y * WW + x).ravel()
    data = np.concatenate([((1 - ty) * (1 - tx)).ravel(), ((1 - ty) * tx).ravel(), (ty * (1 - tx)).ravel(), (ty * tx).ravel()])
    r = np.tile(rows, 4)
    c = np.concatenate([idx(y0, x0), idx(y0, x0 + 1), idx(y0 + 1, x0), idx(y0 + 1, x0 + 1)])
    return sp.csr_matrix((data, (r, c)), shape=(HH * WW, HH * WW))


class FlowModel:
    def __init__(self, shape_lr, flows, scale, psf_sigma):
        self.s, self.lr, self.sig = scale, shape_lr, psf_sigma
        self.hr = (shape_lr[0] * scale, shape_lr[1] * scale)
        self.W = [_warp_matrix(dx, dy, scale) for dx, dy in flows]
        self.WT = [w.T.tocsr() for w in self.W]

    def forward(self, x, k):
        z = (self.W[k] @ x.ravel()).reshape(self.hr)
        return ndi.gaussian_filter(z, self.sig, mode="nearest")[::self.s, ::self.s]

    def adjoint(self, r, k):
        up = np.zeros(self.hr); up[::self.s, ::self.s] = r
        return (self.WT[k] @ ndi.gaussian_filter(up, self.sig, mode="nearest").ravel()).reshape(self.hr)


def _grad(v):
    return np.diff(v, axis=1, append=v[:, -1:]), np.diff(v, axis=0, append=v[-1:, :])


def _gradT(gx, gy):
    dx = -np.diff(gx, axis=1, prepend=0 * gx[:, :1]); dx[:, -1] += gx[:, -1] * 0 + 0
    dy = -np.diff(gy, axis=0, prepend=0 * gy[:1, :]); dy[-1, :] += 0
    # exact adjoint of forward diff with replicated last row/col (last diff = 0):
    dx = np.zeros_like(gx); dx[:, :-1] -= gx[:, :-1]; dx[:, 1:] += gx[:, :-1]
    dy = np.zeros_like(gy); dy[:-1, :] -= gy[:-1, :]; dy[1:, :] += gy[:-1, :]
    return dx + dy


def fuse_flow(frames, flows, scale=2, psf_sigma=1.4, mu=2e-2, lam=1e-6, iters=40):
    m = FlowModel(frames[0].shape, flows, scale, psf_sigma)
    HH, WW = m.hr
    b = sum(m.adjoint(f.astype(np.float64), k) for k, f in enumerate(frames)).ravel()

    def normal(v):
        x = v.reshape(HH, WW)
        acc = sum(m.adjoint(m.forward(x, k), k) for k in range(len(frames)))
        gx, gy = _grad(x)
        return (acc + mu * _gradT(gx, gy) + lam * x).ravel()
    x0 = ndi.zoom(frames[0].astype(np.float64), scale, order=1, grid_mode=True, mode="nearest").ravel()
    op = LinearOperator((HH * WW, HH * WW), matvec=normal, dtype=np.float64)
    x, _ = cg(op, b, x0=x0, maxiter=iters, rtol=1e-6)
    return x.reshape(HH, WW), m

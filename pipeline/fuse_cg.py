"""Non-generative multi-frame super-resolution: regularised least squares solved by conjugate gradient.

    minimise  sum_k || D B S_k x - y_k ||^2  +  mu * || grad x ||^2  +  lam * || x ||^2

S_k = sub-pixel shift (Fourier phase ramp), B = Gaussian PSF, D = decimation by `scale`.
Every operator has an exact adjoint (FFT-domain), so CG is well-posed. No learned prior, no GAN:
the output is a deterministic function of the measured frames and the (measured) PSF.
Boundary: periodic - discard a border of ~3*sigma_psf px of the output.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


class _Model:
    def __init__(self, shape_lr, shifts, scale, psf_sigma):
        self.s = scale
        self.shape_lr = shape_lr
        self.shape = (shape_lr[0] * scale, shape_lr[1] * scale)
        fy = np.fft.fftfreq(self.shape[0])[:, None]
        fx = np.fft.fftfreq(self.shape[1])[None, :]
        self.psf = np.exp(-2 * np.pi ** 2 * psf_sigma ** 2 * (fx ** 2 + fy ** 2))
        # shifted(p) = x(p + s*d)  ->  multiply spectrum by exp(+2*pi*i*(fx*sdx + fy*sdy))
        self.ramps = [np.exp(2j * np.pi * (fx * scale * dx + fy * scale * dy)) for dx, dy in shifts]
        self.grad2 = (2 * np.pi) ** 2 * (fx ** 2 + fy ** 2)

    def forward(self, X, k):
        y = np.fft.ifft2(X * self.ramps[k] * self.psf).real
        return y[:: self.s, :: self.s]

    def adjoint(self, r, k):
        up = np.zeros(self.shape)
        up[:: self.s, :: self.s] = r
        return np.conj(self.ramps[k]) * self.psf * np.fft.fft2(up)


def fuse_cg(frames, shifts, scale=4, psf_sigma=1.6, mu=1e-3, lam=1e-6, iters=60, tol=1e-6):
    m = _Model(frames[0].shape, shifts, scale, psf_sigma)
    n = len(frames)
    H, W = m.shape

    def normal(xvec):
        X = np.fft.fft2(xvec.reshape(H, W))
        acc = np.zeros((H, W), complex)
        for k in range(n):
            acc += m.adjoint(m.forward(X, k), k)
        acc += (mu * m.grad2 + lam) * X
        return np.fft.ifft2(acc).real.ravel()

    b = np.zeros((H, W), complex)
    for k in range(n):
        b += m.adjoint(frames[k].astype(np.float64), k)
    b = np.fft.ifft2(b).real.ravel()
    # warm start: mean of frames upsampled (cheap, keeps CG from wandering at DC)
    x0 = np.full(H * W, float(np.mean([f.mean() for f in frames])))
    op = LinearOperator((H * W, H * W), matvec=normal, dtype=np.float64)
    resid = []
    x, info = cg(op, b, x0=x0, maxiter=iters, rtol=tol, callback=lambda xk: resid.append(1))
    return x.reshape(H, W), len(resid)

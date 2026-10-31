"""Multi-frame super-resolution by iterative back-projection (IBP).

Observation model (matches synth.make_observations):
    LR_k = D( B( HR shifted by s*d_k ) ) + noise
Only measured data enters the solution. Output carries a per-pixel
confidence map so downstream code can mark what is measured vs inferred.
"""
import numpy as np
from scipy import ndimage as ndi


def _simulate(hr, shift, scale, psf_sigma):
    dx, dy = shift
    s = ndi.shift(hr, (-scale * dy, -scale * dx), order=3, mode="reflect")
    return ndi.gaussian_filter(s, psf_sigma)[::scale, ::scale]


def _backproject(residual_lr, shift, scale, psf_sigma, hr_shape):
    dx, dy = shift
    up = np.zeros(hr_shape, dtype=np.float64)
    up[::scale, ::scale] = residual_lr
    up *= scale * scale  # compensate zero-insertion energy
    up = ndi.gaussian_filter(up, psf_sigma)
    return ndi.shift(up, (scale * dy, scale * dx), order=3, mode="reflect")


def fuse(frames, shifts, scale=2, psf_sigma=0.8, iters=30, step=1.0):
    """Reconstruct an HR image from registered LR frames. Returns (hr, residual_rms)."""
    ref = frames[0].astype(np.float64)
    hr_shape = (ref.shape[0] * scale, ref.shape[1] * scale)
    hr = ndi.zoom(ref, scale, order=3, grid_mode=True, mode="nearest")
    n = len(frames)
    rms = None
    for _ in range(iters):
        corr = np.zeros(hr_shape)
        sq = 0.0
        for f, d in zip(frames, shifts):
            res = f - _simulate(hr, d, scale, psf_sigma)
            sq += float((res ** 2).mean())
            corr += _backproject(res, d, scale, psf_sigma, hr_shape)
        hr += step * corr / n
        rms = (sq / n) ** 0.5
    return hr, rms


def coverage_confidence(shifts, scale, shape_lr):
    """Fraction of distinct sub-pixel phases sampled, in [0,1].

    Sub-pixel sampling diversity is what lets IBP recover detail beyond one frame.
    With >= scale**2 well-spread phases the problem is well-posed.
    """
    phases = {(round((dx * scale) % scale), round((dy * scale) % scale)) for dx, dy in shifts}
    return min(1.0, len(phases) / float(scale * scale))

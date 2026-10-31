"""Synthetic terrain + multi-frame observations with KNOWN ground truth.

Used to verify the super-resolution and stereo code without multi-GB HiRISE files.
"""
import numpy as np
from scipy import ndimage as ndi


def fractal_terrain(n=256, seed=0, octaves=6, persistence=0.55):
    """Band-limited fractal heightfield in [0, 1]."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n))
    amp = 1.0
    for o in range(octaves):
        cells = 4 * 2 ** o
        base = rng.standard_normal((cells, cells))
        layer = ndi.zoom(base, n / cells, order=3, mode="wrap")[:n, :n]
        out += amp * layer
        amp *= persistence
    out -= out.min()
    return out / out.max()


def make_observations(hr, scale=2, n_frames=8, psf_sigma=0.8, noise=0.01, seed=1, max_shift=3.0):
    """Simulate low-res frames: frame_k(x) = D(B(HR shifted by s*d_k)) + noise.

    Returns (frames, shifts) where shifts[k] = (dx, dy) in LR pixels, frame 0 = (0, 0).
    """
    rng = np.random.default_rng(seed)
    shifts = [(0.0, 0.0)] + [tuple(rng.uniform(-max_shift, max_shift, 2)) for _ in range(n_frames - 1)]
    frames = []
    for dx, dy in shifts:
        shifted = ndi.shift(hr, (-scale * dy, -scale * dx), order=3, mode="reflect")
        blurred = ndi.gaussian_filter(shifted, psf_sigma)
        lr = blurred[::scale, ::scale] + rng.normal(0, noise, blurred[::scale, ::scale].shape)
        frames.append(lr.astype(np.float32))
    return frames, shifts


def make_stereo_pair(height_m, texture, px_per_metre=1.0, noise=0.005, seed=2):
    """Left/right images of a textured surface. right(x) = left(x + d(x)), d = k * height.

    Returns (left, right, true_disparity).
    """
    rng = np.random.default_rng(seed)
    d = height_m * px_per_metre
    h, w = texture.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    right = ndi.map_coordinates(texture, [ys, xs + d], order=3, mode="reflect")
    left = texture + rng.normal(0, noise, texture.shape)
    right = right + rng.normal(0, noise, texture.shape)
    return left.astype(np.float32), right.astype(np.float32), d

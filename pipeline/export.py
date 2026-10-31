"""Write terrain tiles in the format the rover prototype loads.

Elevation PNG : 12-bit packed, R = high 8 bits, G = high nibble holds low 4 bits (G = low4 << 4), B = 0.
                JS decode: (R << 4) | (G >> 4); metres = emin + v * (emax - emin) / 4095.
Ortho PNG/JPG : 8-bit grayscale.
Confidence PNG: 255 = measured (stereo-verified), 0 = inferred/filled. Never silently merged into geometry.
"""
import numpy as np
import cv2
from scipy import ndimage as ndi


def fill_gaps(h, valid):
    """Fill unmeasured pixels by normalised convolution. Returns (filled, was_filled_mask)."""
    w = valid.astype(np.float64)
    base = np.where(valid, h, 0.0)
    out = base.copy()
    for sigma in (2, 4, 8, 16, 32, 64):
        num = ndi.gaussian_filter(base * w, sigma)
        den = ndi.gaussian_filter(w, sigma)
        est = num / np.maximum(den, 1e-9)
        todo = (~valid) & (out == 0) & (den > 1e-3)
        out = np.where(todo, est, out)
    out = np.where(valid, h, out)
    if not np.all(np.isfinite(out)) or (out == 0).all():
        out = np.nan_to_num(out, nan=float(np.nanmean(h[valid])) if valid.any() else 0.0)
    return out, ~valid


def pack_elevation(h, emin, emax):
    v = np.clip(np.rint((h - emin) / (emax - emin) * 4095), 0, 4095).astype(np.uint16)
    rgb = np.zeros(h.shape + (3,), np.uint8)
    rgb[..., 0] = (v >> 4).astype(np.uint8)          # R
    rgb[..., 1] = ((v & 15) << 4).astype(np.uint8)   # G
    return rgb


def unpack_elevation(rgb, emin, emax):
    """Python twin of the prototype's fillElev + makeElevSampler maths."""
    v = (rgb[..., 0].astype(np.uint16) << 4) | (rgb[..., 1].astype(np.uint16) >> 4)
    return emin + v.astype(np.float64) * (emax - emin) / 4095.0


def write_png(path, rgb_or_gray):
    a = rgb_or_gray
    if a.ndim == 3:
        a = a[..., ::-1]  # cv2 writes BGR
    if not cv2.imwrite(str(path), np.ascontiguousarray(a)):
        raise IOError(f"could not write {path}")


def read_png(path):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return a[..., ::-1] if a.ndim == 3 else a

"""Calibrate the HiRISE point-spread function (PSF) against rover ground truth.

Rover orthos (cm-scale, measured) are degraded with a candidate Gaussian PSF and the HiRISE pixel
footprint, aligned to the HiRISE ortho, and correlated. The PSF width that maximises agreement across
many rover stops is the measured blur. Lighting differs between the two cameras, so both images are
high-passed (illumination gradients removed) and compared by normalised correlation.
"""
import numpy as np
from scipy import ndimage as ndi

HIRISE_PX = 0.25241583171779   # m


def degrade(rover, res, sigma_px, out_px=HIRISE_PX):
    """Rover ortho (res m/px) -> what HiRISE would record: PSF (sigma in HiRISE px) + pixel box average.
    Returns an image on a grid of pitch `out_px`, aligned so output pixel (0,0) centre = rover pixel (0,0) region."""
    k = out_px / res
    valid = np.isfinite(rover).astype(float)
    a = np.nan_to_num(rover)
    blur = lambda x: ndi.uniform_filter(ndi.gaussian_filter(x, sigma_px * k), int(round(k)))
    num, den = blur(a * valid), blur(valid)
    img = np.where(den > 0.6, num / np.maximum(den, 1e-9), np.nan)
    n = int(rover.shape[0] / k)
    c = (np.arange(n) + 0.5) * k - 0.5
    yy, xx = np.meshgrid(c, c, indexing="ij")
    out = ndi.map_coordinates(np.nan_to_num(img), [yy, xx], order=1)
    ok = ndi.map_coordinates(np.isfinite(img).astype(float), [yy, xx], order=1) > 0.99
    return np.where(ok, out, np.nan)


def hp(x, s=6.0):
    """High-pass with NaN support (removes illumination / albedo gradients > ~ s px)."""
    v = np.isfinite(x).astype(float); a = np.nan_to_num(x)
    low = ndi.gaussian_filter(a * v, s) / np.maximum(ndi.gaussian_filter(v, s), 1e-9)
    return np.where(v > 0, a - low, np.nan)


def best_ncc(sim, hi, search=12):
    """Max masked NCC of `sim` (with NaNs) placed over `hi` (larger by 2*search) over integer shifts,
    refined to sub-pixel by parabola. Returns (ncc, dy, dx)."""
    n = sim.shape[0]
    m = np.isfinite(sim); s = np.where(m, sim, 0.0)
    best = (-2, 0, 0); grid = np.full((2 * search + 1, 2 * search + 1), -2.0)
    for i, dy in enumerate(range(-search, search + 1)):
        for j, dx in enumerate(range(-search, search + 1)):
            h = hi[search + dy:search + dy + n, search + dx:search + dx + n]
            mm = m & np.isfinite(h)
            if mm.sum() < 0.3 * n * n: continue
            a = s[mm] - s[mm].mean(); b = h[mm] - h[mm].mean()
            r = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
            grid[i, j] = r
            if r > best[0]: best = (r, dy, dx)
    return best[0], best[1], best[2], grid


# ---------------------------------------------------------------------------------------------
# Registration-free calibration: compare radial power spectra (rover truth vs HiRISE) per stop.
#   P_hirise(f) ~= g * MTF(f)^2 * P_rover_boxed(f) + N        (MTF Gaussian: exp(-2 pi^2 sigma^2 f^2))
# The unknown gain g (lighting/contrast differences) is removed by normalising at low frequency,
# the noise floor N comes from the flat high-frequency tail. No pixel alignment is needed.
# ---------------------------------------------------------------------------------------------

def radial_psd(img, nb=24):
    """Radial PSD of an image with NaNs (masked, mean-filled, Hann-windowed); f in cycles/pixel."""
    v = np.isfinite(img); a = np.where(v, img - np.nanmean(img), 0.0)
    n = a.shape[0]; w = np.outer(np.hanning(n), np.hanning(n)) * v
    F = np.abs(np.fft.fftshift(np.fft.fft2(a * w))) ** 2 / max(w.sum(), 1)
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(n)), np.fft.fftshift(np.fft.fftfreq(n)), indexing="ij")
    fr = np.hypot(fx, fy)
    edges = np.linspace(0.02, 0.5, nb + 1)
    f = 0.5 * (edges[1:] + edges[:-1])
    p = np.array([F[(fr >= edges[i]) & (fr < edges[i + 1])].mean() for i in range(nb)])
    return f, p


def box_to_hirise(rover, res, out_px=HIRISE_PX):
    """Pixel-footprint average only (no optical blur) - the 'perfect camera' reference."""
    return degrade(rover, res, 0.0, out_px)


def fit_sigma(f, ratio, fmax=0.35):
    """Fit ratio(f) = exp(-4 pi^2 sigma^2 f^2) (MTF^2) by least squares in log space over f <= fmax."""
    sel = (f <= fmax) & np.isfinite(ratio) & (ratio > 0)
    if sel.sum() < 4: return np.nan
    y = np.log(ratio[sel]); x = 4 * np.pi ** 2 * f[sel] ** 2
    return float(np.sqrt(max(-np.sum(x * y) / np.sum(x * x), 0)))

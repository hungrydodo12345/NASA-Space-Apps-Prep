"""Registration-free quality scores (percent) against rover ground truth.

Rover photos and orbital pixels cannot be aligned to the cm, so scores compare *what* is there
(spectra, texture statistics) and check that nothing measured was altered (consistency).
"""
import numpy as np
from scipy import ndimage as ndi
from .blur_calib import radial_psd


def _norm_psd(img, coarse_band, nb=48):
    f, p = radial_psd(img, nb=nb)
    c = (f >= coarse_band[0]) & (f <= coarse_band[1])
    return f, p / np.mean(p[c])


def spectral_match(out, truth, fine_band, coarse_band):
    """100 * exp(-mean |ln(P_out / P_truth)|) over the fine band, after matching coarse-band power.
    100% = same amount of detail at every fine scale as the rover saw; 10% = ten times off."""
    f, po = _norm_psd(np.where(np.isfinite(truth), out, np.nan), coarse_band)
    _, pt = _norm_psd(truth, coarse_band)
    s = (f >= fine_band[0]) & (f <= fine_band[1]) & (po > 0) & (pt > 0)
    return float(100 * np.exp(-np.mean(np.abs(np.log(po[s] / pt[s])))))


def texture_match(out, truth, coarse_sigma, bins=40):
    """Histogram overlap (%) of gradient magnitude, each normalised by its own coarse-band contrast."""
    def g(x):
        v = np.isfinite(truth)
        a = np.where(v, x, np.nanmean(x[v]))
        gy, gx = np.gradient(a)
        coarse = ndi.gaussian_filter(a, coarse_sigma) - ndi.gaussian_filter(a, 4 * coarse_sigma)
        return (np.hypot(gx, gy) / (np.std(coarse[v]) + 1e-12))[v]
    a, b = g(out), g(truth)
    hi = np.percentile(np.r_[a, b], 99.5)
    ha, _ = np.histogram(a, bins, (0, hi)); hb, _ = np.histogram(b, bins, (0, hi))
    return float(100 * np.minimum(ha / ha.sum(), hb / hb.sum()).sum())


def consistency(out_fine, measured, scale, psf_sigma_hr):
    """Does the output, seen through the measured HiRISE PSF and pixel, reproduce the measurement? (%)"""
    seen = ndi.uniform_filter(ndi.gaussian_filter(out_fine, psf_sigma_hr), scale)[scale // 2::scale, scale // 2::scale]
    ref = ndi.uniform_filter(ndi.gaussian_filter(ndi.zoom(measured, scale, order=3, grid_mode=True, mode="nearest"), psf_sigma_hr), scale)[scale // 2::scale, scale // 2::scale]
    h, w = min(seen.shape[0], ref.shape[0]), min(seen.shape[1], ref.shape[1])
    d = seen[:h, :w] - ref[:h, :w]; r = ref[:h, :w] - ref[:h, :w].mean()
    return float(100 * max(0.0, 1 - np.sqrt(np.mean(d ** 2)) / (np.sqrt(np.mean(r ** 2)) + 1e-12)))

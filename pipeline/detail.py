"""Rover-texture detail layer: measured-grounded fine detail for scales HiRISE cannot record.

    output = upsample(measured HiRISE)  +  amplitude * sum_c p_c * T_c

T_c   : real rover-ortho texture of surface class c, restricted to the band HiRISE removes
        (truth - PSF_blur(truth)) using the PSF measured against rover ground truth.
p_c   : fuzzy class membership from the orbital surface classifier.
amp   : rover-measured fine/coarse contrast ratio for that class x local contrast of the measured image.
Only the fine band is synthetic; everything HiRISE resolves is kept from the measurement.
"""
import numpy as np
from scipy import ndimage as ndi

CLASSES = ["soil", "bedrock", "sand", "big_rock"]
# AI4Mars NAV values -> class index; GEO values -> nearest NAV-like class
NAV_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
GEO_MAP = {**{v: 1 for v in range(0, 7)}, **{v: 3 for v in range(10, 18)}, **{v: 2 for v in (20, 21, 22)}, 30: 0}


def label_to_class(label, kind):
    m = NAV_MAP if kind == "NAV" else GEO_MAP
    out = np.full(label.shape, np.nan)
    for v, c in m.items(): out[label == v] = c
    return out


def class_mosaic(class_maps):
    """Majority vote of per-image class maps (NaN = unknown)."""
    stack = np.stack(class_maps)
    counts = np.stack([(stack == c).sum(0) for c in range(len(CLASSES))])
    tot = counts.sum(0)
    out = counts.argmax(0).astype(float); out[tot == 0] = np.nan
    return out


def bands(truth, psf_px):
    """Split rover truth (z-scored, NaN-masked) into fine band (what HiRISE removes) and coarse band."""
    v = np.isfinite(truth).astype(float); a = np.nan_to_num(truth)
    g = lambda x, s: ndi.gaussian_filter(x * v, s) / np.maximum(ndi.gaussian_filter(v, s), 1e-9)
    blur = g(a, psf_px); broad = g(a, 4 * psf_px)
    fine = np.where(v > 0, a - blur, np.nan); coarse = np.where(v > 0, blur - broad, np.nan)
    return fine, coarse


def extract_patches(truth, classes, psf_px, size, min_frac=0.8):
    """Square patches dominated (>= min_frac) by one class, broad structure (> ~4 PSF widths) removed.
    The HiRISE complement (1 - PSF) is applied once, later, by the consistency lock. Returns {class: [unit-std patch]}."""
    v = np.isfinite(truth).astype(float); a = np.nan_to_num(truth)
    broad = ndi.gaussian_filter(a * v, 4 * psf_px) / np.maximum(ndi.gaussian_filter(v, 4 * psf_px), 1e-9)
    fine = np.where(v > 0, a - broad, np.nan)
    out = {c: [] for c in range(len(CLASSES))}
    n = truth.shape[0]
    for y in range(0, n - size + 1, size // 2):
        for x in range(0, n - size + 1, size // 2):
            p = fine[y:y + size, x:x + size]; c = classes[y:y + size, x:x + size]
            if np.isfinite(p).mean() < 0.97: continue
            vals = c[np.isfinite(c)]
            if vals.size < 0.5 * size * size: continue
            cls = int(np.bincount(vals.astype(int), minlength=len(CLASSES)).argmax())
            if (vals == cls).mean() < min_frac: continue
            q = np.nan_to_num(p, nan=0.0); q = (q - q.mean()) / (q.std() + 1e-9)
            out[cls].append(q.astype(np.float32))
    return out


def ratio_fine_coarse(truth, psf_px):
    fine, coarse = bands(truth, psf_px)
    return float(np.nanstd(fine) / (np.nanstd(coarse) + 1e-12))


def synthesize(base_up, probs, tiles, amp_ratio, psf_px, seed=0, lock=True):
    """base_up: measured image upsampled to the detail grid; probs: (C,H,W) memberships on that grid;
    tiles: {class: [unit-std patches]}; amp_ratio: {class: fine/coarse}; psf_px: PSF sigma on this grid."""
    rng = np.random.default_rng(seed)
    H, W = base_up.shape
    detail = np.zeros((H, W)); var_area = np.zeros((H, W)); var_mix = np.zeros((H, W))
    for c in range(len(CLASSES)):
        if not tiles.get(c): continue
        pt = tiles[c]; s = pt[0].shape[0]
        canvas = np.zeros((H + s, W + s)); wsum = np.zeros_like(canvas)
        win = np.outer(np.hanning(s), np.hanning(s))
        for y in range(0, H + 1, s // 2):              # overlapped, Hann-blended placement (no seams)
            for x in range(0, W + 1, s // 2):
                t = pt[rng.integers(len(pt))]
                t = np.rot90(t, rng.integers(4)) if rng.random() < 0.5 else t[::-1] if rng.random() < 0.5 else t
                canvas[y:y + s, x:x + s] += t * win; wsum[y:y + s, x:x + s] += win ** 2
        d = canvas / np.sqrt(np.maximum(wsum, 1e-9))
        d = d[s // 2:s // 2 + H, s // 2:s // 2 + W]
        a_c = amp_ratio.get(c, 0.0)
        detail += probs[c] * a_c * d
        var_area += probs[c] * a_c ** 2          # a patch that is p_c class c has variance sum p_c a_c^2 ...
        var_mix += probs[c] ** 2 * a_c ** 2      # ... but a weighted sum of independent textures has sum p_c^2 a_c^2
    # (fuzzy blend variance is restored by the local renormalisation below)
    # local contrast of the measured coarse band sets the physical amplitude
    blur = ndi.gaussian_filter(base_up, psf_px); broad = ndi.gaussian_filter(base_up, 4 * psf_px)
    local = np.sqrt(ndi.gaussian_filter((blur - broad) ** 2, 8 * psf_px))
    target = local * np.sqrt(var_area)                 # wanted rms of the final fine band
    # consistency lock: remove any component the HiRISE PSF would see (keeps the measurement exact) ...
    if lock: detail = detail - ndi.gaussian_filter(detail, psf_px)
    # ... then renormalise locally so the remaining fine band has exactly the target contrast
    rms = np.sqrt(ndi.gaussian_filter(detail ** 2, 8 * psf_px))
    detail = detail * target / np.maximum(rms, 1e-12)
    if lock: detail = detail - ndi.gaussian_filter(detail, psf_px)   # final lock: strip low-frequency leak from the renormalisation
    return base_up + detail, detail


def contrast_attenuation(truth, psf_px, scale=4):
    """How much of the coarse-band contrast survives the HiRISE imaging chain (PSF + pixel + resampling).

    The detail amplitude is scaled by the *measured* image's coarse-band contrast, which HiRISE has already
    attenuated; dividing by this factor (estimated from rover truth on training stops) restores the true level.
    """
    v = np.isfinite(truth); a = np.where(v, truth, np.nanmean(truth))
    seen = ndi.uniform_filter(ndi.gaussian_filter(a, psf_px), scale)[scale // 2::scale, scale // 2::scale]
    up = ndi.zoom(seen, scale, order=3, grid_mode=True, mode="nearest")[:a.shape[0], :a.shape[1]]
    vv = v[:up.shape[0], :up.shape[1]]
    band = lambda x: ndi.gaussian_filter(x, psf_px) - ndi.gaussian_filter(x, 4 * psf_px)
    return float(np.std(band(up)[vv]) / (np.std(band(a[:up.shape[0], :up.shape[1]])[vv]) + 1e-12))


FALLBACK = {3: 1, 2: 0, 1: 0, 0: 1}   # big_rock -> bedrock, sand -> soil when the rover saw too few clean examples


def shape_tiles(tiles, f, ln_ratio):
    """Radial spectral correction of the texture library: multiply each tile's spectrum by exp(-ln_ratio(f)/2),
    where ln_ratio(f) = ln(P_out/P_truth) measured on TRAINING stops (cycles per detail px)."""
    out = {}
    for c, lst in tiles.items():
        new = []
        for t in lst:
            n = t.shape[0]
            fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
            g = np.exp(-np.interp(np.hypot(fx, fy), f, ln_ratio, left=0.0, right=ln_ratio[-1]) / 2)
            q = np.real(np.fft.ifft2(np.fft.fft2(t) * g)); q = (q - q.mean()) / (q.std() + 1e-9)
            new.append(q.astype(np.float32))
        out[c] = new
    return out

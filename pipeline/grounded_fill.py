"""Stage 2 of the grounded pipeline: FILL in rover-scale detail only where and as Stage 1 recognised it.

output = measured restoration (6.31 cm)  +  locked texture detail  +  locked rock detail

* texture detail: real rover-photo patches of the recognised surface class (pipeline.detail.synthesize), amplitude set
  by the rover-measured fine/coarse contrast ratio;
* rock detail: real rover-photographed rocks of the recognised size band, placed only at rocks Stage 1 detected;
* lock: every added layer has the part the HiRISE blur would see removed (x - G_psf * x), so the output, seen through
  HiRISE's blur, is the measurement. Only scales HiRISE cannot record are filled, and only from real rover imagery.
Libraries must be built from TRAINING stops only (the caller passes them per cross-validation fold).
"""
import numpy as np
from scipy import ndimage as ndi
from pipeline import detail as dt, recognizer as rz


px_cm_default = 25.24 / 4


def to_grid(truth, px_cm, n):
    k = 5.0 / px_cm; z5 = truth["z"].astype(float)
    zt = ndi.zoom(np.nan_to_num(z5), k, order=1); vt = ndi.zoom(np.isfinite(z5).astype(float), k, order=1) > 0.99
    ct = ndi.zoom(np.nan_to_num(truth["cls"].astype(float), nan=-1), k, order=0)
    t = zt.shape[0]; o = (n - t) // 2
    Z = np.full((n, n), np.nan); Z[o:o + t, o:o + t] = np.where(vt, zt, np.nan)
    Cc = np.full((n, n), np.nan); Cc[o:o + t, o:o + t] = np.where(vt & (ct >= 0), ct, np.nan)
    return Z, Cc


def build_library(truths, rock_lists, px_cm, psf_px, n=512, tile=32):
    """Texture tiles per class, rock sprites per size band, and amplitude statistics, from training stops' rover truth."""
    tiles = {c: [] for c in range(rz.NCLS)}; sprites = {b: [] for b in range(len(rz.BANDS))}; ratios = []; peak_rel = []
    for tr, rocks in zip(truths, rock_lists):
        Z, Cc = to_grid(tr, px_cm, n)
        cls = np.where(np.isfinite(Cc), Cc, np.where(np.isfinite(Z), 0.0, np.nan))     # unlabeled ground -> soil
        for c, lst in dt.extract_patches(Z, cls, psf_px, tile).items(): tiles[c] += lst
        if np.isfinite(Z).sum() > 5000: ratios.append(dt.ratio_fine_coarse(Z, psf_px))
        fine, _ = dt.bands(Z, psf_px); frms = np.nanstd(fine)
        a = np.nan_to_num(Z); v = np.isfinite(Z).astype(float)
        broad = ndi.gaussian_filter(a * v, 4 * psf_px) / np.maximum(ndi.gaussian_filter(v, 4 * psf_px), 1e-9)
        for y, x, D in rocks:
            b = 0 if D <= rz.BANDS[0][1] else 1; r = int(np.ceil(0.9 * D / px_cm)) + 2
            if y - r < 0 or x - r < 0 or y + r + 1 > n or x + r + 1 > n: continue
            p = Z[y - r:y + r + 1, x - r:x + r + 1]
            if not np.isfinite(p).all(): continue
            p = p - broad[y - r:y + r + 1, x - r:x + r + 1]
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; w = np.clip((0.9 * D / px_cm - np.hypot(yy, xx)) / 2.0, 0, 1)
            sprites[b].append((p * w).astype(np.float32)); peak_rel.append(float(np.abs(p[r - 1:r + 2, r - 1:r + 2]).mean() / (frms + 1e-9)))
    for c in range(rz.NCLS):
        if not tiles[c]: tiles[c] = tiles[dt.FALLBACK.get(c, 0)] or tiles[0]
    amp = {c: float(np.median(ratios)) if ratios else 0.5 for c in range(rz.NCLS)}
    return dict(tiles=tiles, sprites=sprites, amp=amp, rock_peak_rel=float(np.median(peak_rel)) if peak_rel else 2.0)


def observe(img, psf_px, scale=4):
    """What HiRISE records of an image on the 6.31 cm grid: blur, then 25 cm pixels (scale x scale box average)."""
    b = ndi.gaussian_filter(img, psf_px, mode="wrap"); H, W = b.shape
    return b[:H // scale * scale, :W // scale * scale].reshape(H // scale, scale, W // scale, scale).mean((1, 3))


def observe_T(r, psf_px, shape, scale=4):
    """Adjoint of observe: spread each HiRISE pixel back over its scale x scale footprint, then blur (Gaussian is self-adjoint)."""
    up = np.zeros(shape); h, w = r.shape
    up[:h * scale, :w * scale] = np.repeat(np.repeat(r, scale, 0), scale, 1) / scale ** 2
    return ndi.gaussian_filter(up, psf_px, mode="wrap")


def lock(detail, psf_px, scale=4, eps=1e-8):
    """Exact projection of an added detail layer onto the null space of the HiRISE observation A = observe:
    d' = d - A^T (A A^T)^-1 A d. A A^T is a convolution on the 25 cm grid (periodic boundaries), so it is inverted
    exactly with one FFT; observe(base + d') then equals the measurement, with the smallest change to the detail."""
    h, w = detail.shape[0] // scale, detail.shape[1] // scale
    delta = np.zeros((h, w)); delta[0, 0] = 1.0
    kern = observe(observe_T(delta, psf_px, detail.shape, scale), psf_px, scale)
    K = np.real(np.fft.fft2(kern)); b = observe(detail, psf_px, scale)
    x = np.real(np.fft.ifft2(np.fft.fft2(b) * K / (K * K + eps * K.max() ** 2)))
    return detail - observe_T(x, psf_px, detail.shape, scale)


def fine_coarse(img, psf_px, valid=None):
    """Fine-band (below the HiRISE blur) over coarse-band (1-4 blur widths) contrast."""
    v = np.isfinite(img) if valid is None else valid; a = np.where(v, img, 0.0); w = v.astype(float)
    g = lambda x, s: ndi.gaussian_filter(x * w, s) / np.maximum(ndi.gaussian_filter(w, s), 1e-9)
    a = g(a, 0.7)                                                          # light denoise (Navcam noise / JPEG is not texture)
    fine = a - g(a, psf_px); coarse = g(a, psf_px) - g(a, 4 * psf_px)
    m = ndi.binary_erosion(v, iterations=int(4 * psf_px))
    return float(np.std(fine[m]) / (np.std(coarse[m]) + 1e-12)) if m.sum() > 1000 else np.nan


def shadow_kernel(images, psf_px, D_px=8, r=12):
    """Mean HiRISE appearance around bright rock-like blobs (measured, from the restorations themselves): its dark lobe
    gives the shadow direction at the HiRISE sun geometry. Returns (unit shadow vector (dy, dx), kernel)."""
    acc = np.zeros((2 * r + 1, 2 * r + 1)); n = 0
    for im in images:
        z = (im - ndi.gaussian_filter(im, 4 * psf_px)) / (im.std() + 1e-9)
        lg = -ndi.gaussian_laplace(z, D_px / 2 / np.sqrt(2))
        pk = (lg == ndi.maximum_filter(lg, 9)) & (lg > np.percentile(lg, 99.5))
        for y, x in zip(*np.nonzero(pk)):
            if r <= y < z.shape[0] - r and r <= x < z.shape[1] - r: acc += z[y - r:y + r + 1, x - r:x + r + 1]; n += 1
    k = acc / max(n, 1); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    neg = np.clip(-k, 0, None) * (np.hypot(yy, xx) > 2)
    v = np.array([(neg * yy).sum(), (neg * xx).sum()]); v = v / (np.linalg.norm(v) + 1e-12)
    return v, k


def calibrate(lib, restored, truths, psf_px, px_cm, n=512):
    """Global texture gain so the fill's fine/coarse contrast matches the rover truth on TRAINING stops, plus the measured
    HiRISE shadow direction. Adds lib['gain'] and lib['shadow_dir']."""
    rt = [fine_coarse(to_grid(t, px_cm, n)[0], psf_px) for t in truths]; target = float(np.nanmedian(rt))
    lib["gain"] = 1.0
    cp = None; vals = {}
    for g in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        lib["gain"] = g; rr = []
        for r in restored[:12]:
            b = r["restored_multi"].astype(float); cp = np.zeros((rz.NCLS,) + b.shape); cp[0] = 1
            out, _ = fill(b, cp, [], lib, psf_px); rr.append(fine_coarse(out, psf_px, np.ones_like(b, bool)))
        vals[g] = float(np.nanmedian(rr))
    gs = sorted(vals); ys = [vals[g] for g in gs]
    lib["gain"] = float(np.interp(target, ys, gs)) if ys[0] <= target <= ys[-1] else (gs[0] if target < ys[0] else gs[-1])
    lib["shadow_dir"], lib["shadow_kernel"] = shadow_kernel([r["restored_multi"].astype(float) for r in restored[:30]], psf_px)
    lib["calibration"] = dict(truth_fine_coarse=target, fill_fine_coarse_by_gain=vals)
    return lib


def fill(base_up, class_prob, rocks, lib, psf_px, seed=0):
    """base_up: measured restoration; class_prob: (4,H,W) from Stage 1; rocks: [(y, x, D_cm, score)] from Stage 1."""
    rng = np.random.default_rng(seed)
    _, tex = dt.synthesize(base_up, class_prob, lib["tiles"], lib["amp"], psf_px, seed=seed, lock=True)
    tex = tex * lib.get("gain", 1.0)
    rl = np.zeros_like(base_up); H, W = base_up.shape
    frms = np.sqrt(ndi.gaussian_filter(tex ** 2, 8 * psf_px)); sd = lib.get("shadow_dir")
    # place a rock only where the MEASURED restoration shows a bright rock-scale blob (scale-normalised LoG above
    # k x robust noise); a rock HiRISE does not support would have to be cancelled by the lock, leaving a ring.
    zb = (base_up - ndi.gaussian_filter(base_up, 4 * psf_px)); zb = zb / (1.4826 * np.median(np.abs(zb - np.median(zb))) + 1e-12)
    resp = -(psf_px ** 2) * ndi.gaussian_laplace(zb, psf_px); resp = ndi.maximum_filter(resp, 5)
    rocks = [q for q in rocks if resp[int(q[0]), int(q[1])] > lib.get("support_k", 2.5)]
    for y, x, D, _ in rocks:
        b = 0 if D <= rz.BANDS[0][1] else 1; pool = lib["sprites"][b] or lib["sprites"][1 - b]
        if not pool: continue
        s = pool[rng.integers(len(pool))]; s = s[:, ::-1] if rng.random() < 0.5 else s
        r = s.shape[0] // 2
        y0, y1, x0, x1 = y - r, y + r + 1, x - r, x + r + 1
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W: continue
        c = s[r - 1:r + 2, r - 1:r + 2].mean(); sc = lib["rock_peak_rel"] * frms[y, x] / (abs(c) + 1e-9)
        rl[y0:y1, x0:x1] += s * sc
        if sd is not None:                                      # cast shadow along the measured HiRISE shadow direction
            rad = D / 2 / px_cm_default; yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
            cy, cx = y + sd[0] * rad * 1.1, x + sd[1] * rad * 1.1
            u = (yy - cy) * sd[0] + (xx - cx) * sd[1]; v = -(yy - cy) * sd[1] + (xx - cx) * sd[0]
            sh = 1 / (1 + np.exp((np.sqrt((u / max(0.6 * rad, 0.8)) ** 2 + (v / max(rad, 0.8)) ** 2) - 1) / 0.15))
            rl[y0:y1, x0:x1] -= 1.3 * lib["rock_peak_rel"] * frms[y, x] * sh
    added = lock(tex + rl, psf_px)                            # lock: nothing HiRISE could record is added
    out = base_up + added
    ob = observe(base_up, psf_px)
    resid = float(np.std(observe(out, psf_px) - ob) / (np.std(ob - ob.mean()) + 1e-12))
    # The filled image contains pasted rover rocks: it is for display only and must never be passed to a rock detector
    # or scored (that would re-detect what Stage 1 placed). Scoring code checks this flag via assert_not_synthetic().
    return out, dict(texture=tex, rocks=rl, added=added, lock_residual_rel=resid, rocks_placed=len(rocks), synthetic=True)


SYNTHETIC_KEYS = {"grounded", "grounded_fill", "fill"}


def assert_not_synthetic(name):
    """Guard used by every scoring script: refuse products that came out of the grounded fill (Stage 2)."""
    if any(k in str(name).lower() for k in SYNTHETIC_KEYS):
        raise ValueError(f"'{name}' is a grounded-fill (synthetic) product; it may be displayed but never scored or detected on")

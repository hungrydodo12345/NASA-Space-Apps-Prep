"""Height from a rectified stereo pair (parallax -> elevation) with a measured-confidence mask.

Convention: left(x) matches right(x - d), d >= 0 and d = k * height (k = px of parallax per metre).
Output pixels that fail the left-right consistency check are marked INVALID (not guessed);
callers decide whether to fill them, and filled pixels must be tagged "inferred".
"""
import numpy as np
import cv2


def _to_u8(img):
    lo, hi = np.percentile(img, [0.5, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)


def _sgbm(left, right, max_disp, block):
    nd = int(np.ceil(max_disp / 16.0)) * 16
    m = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        uniquenessRatio=5, speckleWindowSize=60, speckleRange=2,
        disp12MaxDiff=-1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    d = m.compute(np.ascontiguousarray(left), np.ascontiguousarray(right)).astype(np.float32) / 16.0
    d[d < 0] = np.nan
    return d


def disparity(left, right, max_disp=32, block=5, lr_tol=0.5, med_tol=0.75, med_size=7, ncc_min=0.85, ncc_win=9):
    """Return (disp, valid).

    Invalid = no match, fails left-right consistency (> lr_tol px), or disagrees with the
    local median disparity by more than med_tol px (kills isolated false matches).
    """
    l8, r8 = _to_u8(left), _to_u8(right)
    dl = _sgbm(l8, r8, max_disp, block)
    # right->left disparity via horizontal flip (flipped right acts as the "left" image)
    dr = _sgbm(r8[:, ::-1], l8[:, ::-1], max_disp, block)[:, ::-1]
    h, w = dl.shape
    xs = np.arange(w)[None, :].repeat(h, 0)
    ys = np.arange(h)[:, None].repeat(w, 1)
    # left x matches right x - d. In right-image disparity terms, the same point has disparity at x-d.
    xr = np.clip(np.rint(xs - np.nan_to_num(dl, nan=0)).astype(int), 0, w - 1)
    # in the flipped pipeline dr is the disparity of the RIGHT image, measured toward +x of left image
    back = dr[ys, xr]
    valid = np.isfinite(dl) & np.isfinite(back) & (np.abs(dl - back) <= lr_tol)
    # local-smoothness gate: median of currently-valid neighbours
    from scipy import ndimage as ndi
    filled = np.where(valid, dl, np.nan)
    med = ndi.generic_filter(filled, np.nanmedian, size=med_size, mode="nearest") if valid.any() else filled
    valid &= np.isfinite(med) & (np.abs(dl - med) <= med_tol)
    # photometric gate: right warped by the disparity must look like left (local NCC)
    ncc = photometric_ncc(left, right, np.nan_to_num(dl, nan=0.0), ncc_win)
    valid &= ncc >= ncc_min
    return dl, valid


def photometric_ncc(left, right, disp, win=9):
    """Local normalized cross-correlation between left and right-warped-to-left (left(x) = right(x - d))."""
    from scipy import ndimage as ndi
    h, w = left.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    warped = ndi.map_coordinates(right.astype(np.float64), [ys, xs - disp], order=1, mode="nearest")
    a = left.astype(np.float64)
    mu = lambda z: ndi.uniform_filter(z, win)
    ma, mb = mu(a), mu(warped)
    cov = mu(a * warped) - ma * mb
    va = np.maximum(mu(a * a) - ma ** 2, 1e-12)
    vb = np.maximum(mu(warped * warped) - mb ** 2, 1e-12)
    return cov / np.sqrt(va * vb)


def height_from_disparity(disp, valid, px_per_metre):
    """Metres of relief; NaN where not measured."""
    h = disp / float(px_per_metre)
    h = np.where(valid, h, np.nan)
    return h

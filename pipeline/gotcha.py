"""Gotcha-style dense sub-pixel matching between HiRISE passes (re-implementation of Tao & Muller 2016, Sec. 3.2).

No reference code is available; this follows the paper's description:
  1. Seeds (MSA-SIFT): SIFT keypoints matched between the reference and the other pass (ratio test), outliers removed with
     RANSAC, each seed refined by adaptive least-squares correlation (ALSC).
  2. ALSC: Gauss-Newton fit of an affine geometric transform (6 parameters) plus a linear radiometric one (gain, offset)
     between a window of the reference and the other pass, iterated until the update is below 1e-3 px (Eq. 10).
  3. Region growing (Gotcha): matches are propagated from the best seeds to neighbouring grid points, each new point
     starting from its neighbour's affine; a point is accepted only if the fitted window correlation is high enough.
Output follows the convention of pipeline.fuse_flow: a dense field d (dx, dy) with  pass(x) ~= ref(x + d(x)),
plus a per-pixel quality (fitted correlation) and a validity mask. Pixels inside `exclude` (e.g. the rover, which moves
between passes) are never matched; their motion is interpolated from the surrounding ground.
"""
import heapq
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.interpolate import griddata


def _u8(a):
    lo, hi = np.percentile(a, [0.5, 99.5]); return (np.clip((a - lo) / (hi - lo + 1e-12), 0, 1) * 255).astype(np.uint8)


def alsc(ref, img, gx, gy, cy, cx, A0, half=7, iters=15, tol=1e-3):
    """Adaptive least-squares correlation at (cy, cx) of ref. Model: ref(p) ~= g * img(A [p-c] + c + t) + o.
    A0 = [a11, a12, a21, a22, ty, tx]. Returns (params, ncc) or (None, -1) if the fit fails."""
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    r = ref[cy - half:cy + half + 1, cx - half:cx + half + 1].astype(np.float64).ravel()
    if r.std() < 1e-6: return None, -1.0
    p = np.array(A0, float); g, o = 1.0, 0.0
    H, W = img.shape
    for _ in range(iters):
        a11, a12, a21, a22, ty, tx = p
        qy = cy + a11 * yy + a12 * xx + ty; qx = cx + a21 * yy + a22 * xx + tx
        if qy.min() < 1 or qx.min() < 1 or qy.max() > H - 2 or qx.max() > W - 2: return None, -1.0
        v = ndi.map_coordinates(img, [qy, qx], order=1).ravel()
        dvy = ndi.map_coordinates(gy, [qy, qx], order=1).ravel(); dvx = ndi.map_coordinates(gx, [qy, qx], order=1).ravel()
        res = r - (g * v + o)
        J = np.stack([g * dvy * yy.ravel(), g * dvy * xx.ravel(), g * dvx * yy.ravel(), g * dvx * xx.ravel(), g * dvy, g * dvx, v, np.ones_like(v)], 1)
        try: step = np.linalg.lstsq(J, res, rcond=None)[0]
        except np.linalg.LinAlgError: return None, -1.0
        p += step[:6]; g += step[6]; o += step[7]
        if abs(step[4]) < tol and abs(step[5]) < tol: break
    a11, a12, a21, a22, ty, tx = p
    if not (0.7 < a11 < 1.3 and 0.7 < a22 < 1.3 and abs(a12) < 0.3 and abs(a21) < 0.3): return None, -1.0
    qy = cy + a11 * yy + a12 * xx + ty; qx = cx + a21 * yy + a22 * xx + tx
    v = ndi.map_coordinates(img, [qy, qx], order=1).ravel()
    ncc = float(np.corrcoef(r, v)[0, 1]) if v.std() > 1e-9 else -1.0
    return p, ncc


def seeds_msa_sift(ref, img, exclude=None, max_seeds=300, ratio=0.75):
    """SIFT matches -> RANSAC (affine) -> list of (y, x, A0) in reference coordinates."""
    s = cv2.SIFT_create(nfeatures=4000)
    m = None if exclude is None else (~exclude).astype(np.uint8) * 255
    k1, d1 = s.detectAndCompute(_u8(ref), m); k2, d2 = s.detectAndCompute(_u8(img), None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8: return []
    mt = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [a for a, b in (x for x in mt if len(x) == 2) if a.distance < ratio * b.distance]
    if len(good) < 6: return []
    p1 = np.float32([k1[g.queryIdx].pt for g in good]); p2 = np.float32([k2[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffine2D(p1, p2, ransacReprojThreshold=1.5, maxIters=4000)
    if M is None: return []
    out = []
    for (x1, y1), (x2, y2), ok in zip(p1, p2, inl.ravel()):
        if ok: out.append((int(round(y1)), int(round(x1)), [M[1, 1], M[1, 0], M[0, 1], M[0, 0], y2 - y1, x2 - x1]))
    return out[:max_seeds]


def seeds_phase(ref, img, exclude=None, spacing=24, half=7):
    """Fallback seeds when SIFT finds too few matches (e.g. passes years apart under different sun angles): one global
    shift from phase correlation of high-passed images, then a regular grid of seeds carrying that shift (ALSC refines
    each one, including gain/offset)."""
    hp = lambda a: (a - ndi.gaussian_filter(a, 6)).astype(np.float32)
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    (sx, sy), resp = cv2.phaseCorrelate(hp(ref), hp(img), win)        # img(x) ~= ref(x - s)  =>  ref(p) ~= img(p + s)
    H, W = ref.shape; m = half + 4 + int(max(abs(sx), abs(sy))) + 1; out = []
    for y in range(m, H - m, spacing):
        for x in range(m, W - m, spacing):
            if exclude is None or not exclude[y, x]: out.append((y, x, [1.0, 0.0, 0.0, 1.0, sy, sx]))
    return out


def gotcha(ref, img, exclude=None, step=4, half=7, ncc_min=0.6, max_jump=0.5, seeds=None, min_sift_seeds=10):
    """Dense Gotcha matching. Returns dx, dy (pass(x) ~= ref(x + d)), quality (ncc), valid mask, number of seeds."""
    ref = ref.astype(np.float64); img = img.astype(np.float64)
    gy, gx = np.gradient(img)
    H, W = ref.shape; m = half + 3
    ys = np.arange(m, H - m, step); xs = np.arange(m, W - m, step)
    gi = {(y, x): (i, j) for i, y in enumerate(ys) for j, x in enumerate(xs)}
    P = np.full((len(ys), len(xs), 6), np.nan); Q = np.full((len(ys), len(xs)), -1.0)
    ex = np.zeros((H, W), bool) if exclude is None else exclude
    if seeds is None:
        seeds = seeds_msa_sift(ref, img, exclude)
        if len(seeds) < min_sift_seeds: seeds = seeds + seeds_phase(ref, img, exclude, half=half)
    heap = []
    for y, x, A0 in seeds:                                   # snap seeds to the grid and refine them by ALSC
        i, j = int(np.argmin(np.abs(ys - y))), int(np.argmin(np.abs(xs - x)))
        yy, xx = ys[i], xs[j]
        if ex[yy, xx]: continue
        A = list(A0); A[4] += (A0[0] - 1) * (yy - y) + A0[1] * (xx - x); A[5] += A0[2] * (yy - y) + (A0[3] - 1) * (xx - x)
        p, c = alsc(ref, img, gx, gy, yy, xx, A, half)
        if p is not None and c >= ncc_min and c > Q[i, j]: P[i, j] = p; Q[i, j] = c; heapq.heappush(heap, (-c, i, j))
    while heap:                                              # best-first region growing
        _, i, j = heapq.heappop(heap)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if not (0 <= a < len(ys) and 0 <= b < len(xs)) or Q[a, b] >= ncc_min or ex[ys[a], xs[b]]: continue
            A0 = P[i, j].copy()
            dyc, dxc = ys[a] - ys[i], xs[b] - xs[j]            # carry the affine to the neighbour
            A0[4] += (A0[0] - 1) * dyc + A0[1] * dxc; A0[5] += A0[2] * dyc + (A0[3] - 1) * dxc
            p, c = alsc(ref, img, gx, gy, ys[a], xs[b], A0, half)
            if p is None or c < ncc_min: Q[a, b] = max(Q[a, b], -0.5); continue
            if abs(p[4] - A0[4]) > max_jump or abs(p[5] - A0[5]) > max_jump: continue
            P[a, b] = p; Q[a, b] = c; heapq.heappush(heap, (-c, a, b))
    ok = Q >= ncc_min
    if ok.sum() < 4: return None
    YY, XX = np.meshgrid(ys, xs, indexing="ij"); yy, xx = np.mgrid[0:H, 0:W]
    pts = np.stack([YY[ok], XX[ok]], 1)
    # ALSC gives ref(p) ~= img(p + t)  ->  in the fuse_flow convention pass(x) ~= ref(x + d) with d = -t
    dense = []
    for comp in (4, 5):
        v = griddata(pts, P[..., comp][ok], (yy, xx), method="linear")
        vn = griddata(pts, P[..., comp][ok], (yy, xx), method="nearest"); v = np.where(np.isfinite(v), v, vn)
        dense.append(-ndi.gaussian_filter(v, step / 2))
    q = griddata(pts, Q[ok], (yy, xx), method="nearest"); valid = ndi.binary_dilation(ok, iterations=1)
    valid = ndi.zoom(valid.astype(float), (H / len(ys), W / len(xs)), order=0)[:H, :W] > 0.5 if valid.shape != (H, W) else valid
    return dict(dx=dense[1], dy=dense[0], quality=q, valid=np.pad(valid, ((0, max(0, H - valid.shape[0])), (0, max(0, W - valid.shape[1]))))[:H, :W],
                n_seeds=len(seeds), n_matched=int(ok.sum()), n_grid=int(ok.size))

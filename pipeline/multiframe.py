"""Real multi-observation HiRISE stack: coarse-to-fine registration, photometric matching,
dense-motion fusion and leave-one-out validation reported as percentages."""
import numpy as np, cv2
from scipy import ndimage as ndi
from .hirise_io import Rdr
from . import fuse_flow, metrics


def _prep(x, ds):
    x = np.nan_to_num(x, nan=np.nanmean(x))
    s = ndi.zoom(ndi.gaussian_filter(x, ds / 2), 1 / ds, order=1)
    return (s - ndi.gaussian_filter(s, 6)).astype(np.float32)


def coarse_offsets(rdrs, gl, gs, n, nc=2048, ds=4):
    """Integer (dx, dy) per frame so that frame content at y appears at y - d relative to frame 0."""
    pad = (nc - n) // 2
    cw = [r.read_if(gl - pad, gs - pad, nc, nc) for r in rdrs]
    ref = _prep(cw[0], ds); win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    out, resp = [(0, 0)], [1.0]
    for w in cw[1:]:
        (sx, sy), r = cv2.phaseCorrelate(ref.copy(), _prep(w, ds).copy(), win)
        out.append((int(round(-sx * ds)), int(round(-sy * ds)))); resp.append(r)
    return out, resp


def load_stack(ids, D, gl, gs, n):
    rdrs = [Rdr(f"{D}/{i}_RED") for i in ids]
    off, resp = coarse_offsets(rdrs, gl, gs, n)
    frames = [r.read_if(gl - o[1], gs - o[0], n, n) for r, o in zip(rdrs, off)]
    return frames, off, resp


def dense_flow_2pass(ref, img):
    fx, fy = fuse_flow.dense_shift_field(ref, img, tile=96, step=24)
    H, W = ref.shape; yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    warped = ndi.map_coordinates(img, [yy - fy, xx - fx], order=3, mode="nearest")
    rx, ry = fuse_flow.dense_shift_field(ref, warped, tile=64, step=16)
    return fx + rx, fy + ry


def photometric_match(ref, img, flow, m=24):
    H, W = ref.shape; yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    w = ndi.map_coordinates(img, [yy - flow[1], xx - flow[0]], order=3, mode="nearest")
    k, b = np.polyfit(w[m:-m, m:-m].ravel(), ref[m:-m, m:-m].ravel(), 1)
    return img * k + b


def detail_pct(a, b):
    n = a.shape[0]; w = np.outer(np.hanning(n), np.hanning(n))
    return 100 * metrics.effective_cutoff(metrics.radial_coherence((a - a.mean()) * w, (b - b.mean()) * w, bins=20))


def hp(x): return x - ndi.gaussian_filter(x, 5)

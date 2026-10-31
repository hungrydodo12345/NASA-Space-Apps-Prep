"""Rover ground truth as top-down imagery: project a ground grid into Navcam photos.

For each ground point (north, east, 0) near the rover we compute its image position with the
CAHV camera model (forward projection) and sample the photo. Result: a north-up orthoimage at
a few cm/px - the finest *measured* view of that ground - used to calibrate HiRISE blur, to
score our outputs, and as real detail along the traverse.
"""
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from . import rover_project as rp


def load_gray(path):
    a = np.asarray(Image.open(path)).astype(np.float64)
    if a.ndim == 3: a = a[..., :3].mean(2)
    return a


def project_ground(meta, img, half=8.0, res=0.05, rmin=1.5, rmax=8.0, edge_frac=0.9, order=1):
    """Return (ortho, valid) north-up grid of size (2*half/res)^2 centred on the rover origin.

    Row 0 = north edge, column 0 = west edge. NaN outside the usable footprint.
    """
    cam = rp.parse_cahv(meta)
    R = rp.quat_to_R(rp.attitude(meta))          # rover -> local level
    n = int(round(2 * half / res))
    north = half - (np.arange(n) + 0.5) * res     # row -> north offset
    east = -half + (np.arange(n) + 0.5) * res     # col -> east offset
    NN, EE = np.meshgrid(north, east, indexing="ij")
    P_ll = np.stack([NN, EE, np.zeros_like(NN)], -1).reshape(-1, 3)
    P = P_ll @ R                                  # local level -> rover frame  (R^T applied via right-multiply)
    d = P - cam["C"][None, :]
    za = d @ cam["A"]
    xs = (d @ cam["H"]) / za; ys = (d @ cam["V"]) / za
    # The API camera model is already expressed in the delivered image's pixel frame
    # (downsampling and subframe offsets applied), so model coords ARE image coords.
    Hh, Ww = img.shape
    dw, dh = [float(t) for t in meta["extended"]["dimension"].strip("()").split(",")]
    u = xs * (Ww / dw); v = ys * (Hh / dh)
    rng = np.hypot(NN, EE).ravel()
    cu, cv = (Ww - 1) / 2, (Hh - 1) / 2
    ok = (za > 0) & (rng >= rmin) & (rng <= rmax) & (np.abs(u - cu) <= edge_frac * cu) & (np.abs(v - cv) <= edge_frac * cv)
    out = np.full(n * n, np.nan)
    out[ok] = ndi.map_coordinates(img, [v[ok], u[ok]], order=order, mode="nearest")
    return out.reshape(n, n)


def mosaic(orthos):
    """Average several co-located rover orthos (same stop), normalising each to zero-mean/unit-std first."""
    acc = None; cnt = None
    for o in orthos:
        ok = np.isfinite(o)
        if ok.sum() < 100: continue
        z = (o - np.nanmean(o)) / (np.nanstd(o) + 1e-9)
        z = np.where(ok, z, 0.0)
        acc = z if acc is None else acc + z
        cnt = ok.astype(float) if cnt is None else cnt + ok
    if acc is None: return None
    m = acc / np.maximum(cnt, 1); m[cnt == 0] = np.nan
    return m

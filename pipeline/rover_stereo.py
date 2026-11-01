"""Terrain-aware rover ground truth: Navcam stereo height by plane sweep, with the full CAHVORE lens model.

The earlier truth (pipeline/rover_ortho.py) projected each left-Navcam image onto a flat plane using only the C, A, H, V
terms of the camera model, ignoring lens distortion (O, R) and entrance-pupil shift (E). Here:

* `project` implements the CAHVORE 3D -> 2D model (JPL CAHVORE, fisheye/general types) so pixels land where the lens
  actually put them.
* `sweep_pair` finds, for every cell of a north-up local-level grid, the ground height at which the left and right images
  agree best (zero-mean normalised cross-correlation over a small window). Heights are searched around a prior surface
  (the HiRISE DTM, or a plane), so slopes, rock tops and range stretch are handled, and anything that is not ground (rover
  hardware, sky, deep shadow) fails to correlate and is dropped.
* `fuse_stop` merges all stereo pairs at a rover stop into one top-down image + height map + confidence.

Frames: camera models are in the rover-nav frame (x fwd, y right, z down); the attitude quaternion maps rover-nav to local
level (x north, y east, z down). Grid row 0 = north edge, column 0 = west edge, centred on the rover-nav origin, the same
convention as rover_ortho.project_ground, so downstream code is unchanged.
"""
import re
import numpy as np
from scipy import ndimage as ndi
from pipeline import rover_project as rp


def parse_cahvore(meta):
    parts = meta["camera"]["camera_model_component_list"].split(";")
    vec = [np.array([float(t) for t in p.strip("() ").split(",")]) for p in parts if "(" in p]
    nums = [float(p) for p in parts if "(" not in p and p.strip()]
    cam = dict(C=vec[0], A=vec[1], H=vec[2], V=vec[3])
    if len(vec) >= 6: cam.update(O=vec[4], R=vec[5])
    if len(vec) >= 7: cam["E"] = vec[6]
    cam["mtype"] = int(nums[0]) if nums else 1
    cam["mparm"] = nums[1] if len(nums) > 1 else 0.0
    return cam


def _linearity(cam):
    if "E" not in cam: return None
    return {1: 1.0, 2: 0.0}.get(cam["mtype"], cam["mparm"])     # 1 perspective, 2 fisheye, 3 general (parameter)


def project(cam, P, iters=12):
    """Rover-frame points (N, 3) -> image (x = sample, y = line), plus a 'in front of camera' flag. CAHV, CAHVOR or CAHVORE."""
    C, A, H, V = cam["C"], cam["A"], cam["H"], cam["V"]
    p = P - C
    if "O" not in cam:                                       # plain CAHV
        a = p @ A; return (p @ H) / a, (p @ V) / a, a > 0
    O, R = cam["O"], cam["R"]
    zeta = p @ O
    lamv = p - zeta[:, None] * O
    lam = np.linalg.norm(lamv, axis=1)
    L = _linearity(cam)
    if L is None:                                            # CAHVOR: radial distortion only
        tau = lam ** 2 / np.maximum(zeta ** 2, 1e-12)
        mu = R[0] + R[1] * tau + R[2] * tau ** 2
        q = p + mu[:, None] * lamv
    else:                                                    # CAHVORE: Newton solve for the field angle with pupil shift
        E = cam["E"]
        th = np.arctan2(lam, zeta)
        for _ in range(iters):
            c, s = np.cos(th), np.sin(th); t2 = th * th; t3 = t2 * th; t4 = t2 * t2
            ups = zeta * c + lam * s - (1 - c) * (E[0] + E[1] * t2 + E[2] * t4) - (th - s) * (2 * E[1] * th + 4 * E[2] * t3)
            th = th - ((zeta * s - lam * c) - (th - s) * (E[0] + E[1] * t2 + E[2] * t4)) / ups
        if abs(L) < 1e-12: chi = th
        elif L < 0: chi = np.sin(L * th) / L
        else: chi = np.tan(np.minimum(L * th, 1.5)) / L
        chi2 = chi * chi
        zeta2 = np.where(chi > 1e-9, lam / np.maximum(chi, 1e-12), zeta)
        mu = R[0] + R[1] * chi2 + R[2] * chi2 * chi2
        q = zeta2[:, None] * O + (1 + mu)[:, None] * lamv
    a = q @ A
    return (q @ H) / a, (q @ V) / a, (a > 0) & (zeta > 0)


def grid(half=8.0, res=0.05):
    n = int(round(2 * half / res))
    north = half - (np.arange(n) + 0.5) * res
    east = -half + (np.arange(n) + 0.5) * res
    return np.meshgrid(north, east, indexing="ij")


def _sample(img, cam, P_rov, edge=4):
    x, y, ok = project(cam, P_rov)
    h, w = img.shape
    ok &= (x >= edge) & (x <= w - 1 - edge) & (y >= edge) & (y <= h - 1 - edge)
    out = np.full(len(P_rov), np.nan)
    out[ok] = ndi.map_coordinates(img, [y[ok], x[ok]], order=1)
    return out


def _local_ncc(a, b, win):
    """Zero-mean NCC of two grids over win x win windows; NaN cells are excluded from each window."""
    m = np.isfinite(a) & np.isfinite(b)
    a0 = np.where(m, a, 0.0); b0 = np.where(m, b, 0.0); w = m.astype(float)
    f = lambda z: ndi.uniform_filter(z, win, mode="constant")
    n = f(w); n_ = np.maximum(n, 1e-9)
    ma, mb = f(a0) / n_, f(b0) / n_
    va = f(a0 * a0) / n_ - ma * ma; vb = f(b0 * b0) / n_ - mb * mb; cab = f(a0 * b0) / n_ - ma * mb
    r = cab / np.sqrt(np.maximum(va * vb, 1e-12))
    r[(n < 0.7) | ~m] = np.nan
    return r, np.sqrt(np.maximum(va, 0))


def sweep_pair(left, right, mL, mR, prior_z, NN, EE, dz=np.arange(-0.45, 0.451, 0.015), win=7, rmin=1.5, rmax=8.0):
    """Plane sweep around prior_z (local-level z, down positive) for one stereo pair.

    Returns best height z, its NCC score, the left-image texture at that height, and the texture contrast (for masking
    flat/featureless cells where stereo is ill-posed).
    """
    camL, camR = parse_cahvore(mL), parse_cahvore(mR)
    RL = rp.quat_to_R(rp.attitude(mL)); RR = rp.quat_to_R(rp.attitude(mR))
    rng = np.hypot(NN, EE)
    use = (rng >= rmin) & (rng <= rmax) & np.isfinite(prior_z)
    shp = NN.shape
    best = np.full(shp, -np.inf); bz = np.full(shp, np.nan); btex = np.full(shp, np.nan); bsd = np.full(shp, np.nan)
    # cheap footprint test at the prior height: skip cells the left camera cannot see
    P0 = np.stack([NN[use], EE[use], prior_z[use]], -1)
    seen = np.isfinite(_sample(left, camL, P0 @ RL))
    idx = np.flatnonzero(use)[seen]
    if len(idx) < 500: return None
    for d in dz:
        Pll = np.stack([NN.ravel()[idx], EE.ravel()[idx], prior_z.ravel()[idx] + d], -1)
        a = np.full(NN.size, np.nan); b = np.full(NN.size, np.nan)
        a[idx] = _sample(left, camL, Pll @ RL); b[idx] = _sample(right, camR, Pll @ RR)
        a = a.reshape(shp); b = b.reshape(shp)
        r, sd = _local_ncc(a, b, win)
        upd = np.isfinite(r) & (r > best)
        best[upd] = r[upd]; bz[upd] = (prior_z + d)[upd]; btex[upd] = a[upd]; bsd[upd] = sd[upd]
    best[~np.isfinite(bz)] = np.nan
    # reject heights stuck at the edge of the search range (true surface lies outside it)
    edge = np.isfinite(bz) & ((np.abs(bz - prior_z - dz[0]) < 1e-6) | (np.abs(bz - prior_z - dz[-1]) < 1e-6))
    best[edge] = np.nan
    return dict(z=bz, ncc=best, tex=btex, sd=bsd)


def fuse_stop(results, ncc_min=0.6, z_tol=0.06):
    """Merge per-pair sweeps: keep confident cells, take the confidence-weighted median height, and average texture
    (each pair normalised to zero mean / unit std first, as in rover_ortho.mosaic). Cells where pairs disagree on height
    by more than z_tol are dropped."""
    Z = []; T = []; W = []
    for r in results:
        if r is None: continue
        ok = np.isfinite(r["ncc"]) & (r["ncc"] >= ncc_min)
        if ok.sum() < 200: continue
        t = r["tex"]; t = (t - np.nanmean(t[ok])) / (np.nanstd(t[ok]) + 1e-9)
        Z.append(np.where(ok, r["z"], np.nan)); T.append(np.where(ok, t, np.nan)); W.append(np.where(ok, r["ncc"], np.nan))
    if not Z: return None
    Z, T, W = np.stack(Z), np.stack(T), np.stack(W)
    zmed = np.nanmedian(Z, 0)
    agree = np.abs(Z - zmed[None]) <= z_tol
    T = np.where(agree, T, np.nan); W = np.where(agree, W, np.nan)
    wsum = np.nansum(W, 0)
    tex = np.nansum(T * np.nan_to_num(W), 0) / np.maximum(wsum, 1e-9)
    n = np.isfinite(T).sum(0)
    tex[n == 0] = np.nan; zmed[n == 0] = np.nan
    conf = np.where(n > 0, wsum / np.maximum(n, 1), np.nan)
    return dict(z=tex, height=zmed, conf=conf, n_pairs=n)

"""Project rover-image label pixels onto the ground and onto the HiRISE ortho grid.

Camera: CAHV(OR) model from the M2020 raw-image API (rover-nav frame: x fwd, y right, z down).
Pose:   attitude quaternion (w, x, y, z), rover-nav -> local level (x north, y east, z down).
Ground: locally flat plane through the rover-nav origin (valid near-field; range-limited).
"""
import math, re
import numpy as np


def parse_cahv(meta):
    comps = re.findall(r"\(([^)]*)\)", meta["camera"]["camera_model_component_list"])
    v = [np.array([float(t) for t in c.split(",")]) for c in comps]
    return dict(C=v[0], A=v[1], H=v[2], V=v[3])


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def attitude(meta):
    return np.array([float(t) for t in meta["attitude"].strip("()").split(",")])


def az_el(vec):
    """Azimuth (deg, from +x toward +y) and elevation (deg, + is up for z-down frames)."""
    return math.degrees(math.atan2(vec[1], vec[0])) % 360, math.degrees(math.atan2(-vec[2], math.hypot(vec[0], vec[1])))


def rays(cam, xs, ys):
    """Unit ray directions for full-res image coords (x = sample, y = line)."""
    A, H, V = cam["A"], cam["H"], cam["V"]
    a = V[None, :] - ys[:, None] * A[None, :]
    b = H[None, :] - xs[:, None] * A[None, :]
    r = np.cross(a, b)
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    r *= np.sign(r @ A)[:, None]
    return r


def ground_points(meta, u, v, label_shape, max_range=12.0):
    """Label pixel coords (u=col, v=row) -> (north, east) metres from rover origin; NaN if no hit/too far."""
    cam = parse_cahv(meta)
    # Camera model coords = delivered image pixel coords (see rover_ortho); labels may be resized.
    dw, dh = [float(t) for t in meta["extended"]["dimension"].strip("()").split(",")]
    Lh, Lw = label_shape
    xs = (u + 0.5) * (dw / Lw) - 0.5
    ys = (v + 0.5) * (dh / Lh) - 0.5
    R = quat_to_R(attitude(meta))
    d = rays(cam, xs.astype(float), ys.astype(float)) @ R.T          # local-level directions
    C = R @ cam["C"]
    t = -C[2] / d[:, 2]                                               # hit plane z = 0 (z down; C[2] < 0)
    ok = (d[:, 2] > 1e-3) & (t > 0)
    P = C[None, :] + t[:, None] * d
    rng = np.hypot(P[:, 0], P[:, 1])
    ok &= rng <= max_range
    P[~ok] = np.nan
    return P[:, 0], P[:, 1]

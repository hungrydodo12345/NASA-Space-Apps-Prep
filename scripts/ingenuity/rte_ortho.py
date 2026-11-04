"""Orthorectify Ingenuity colour frames onto a horizontal ground plane under nominal pose: camera 8 m above the ground, axis 22 deg below the horizon
(from the image caption), zero roll, heading arbitrary. The camera model is rte_camera.py's (f ~ 1780 px, radial R1 = 0.47). World: X right, Y forward, Z up.
Output grid: X in [-xw, xw], Y in [y0, y1] metres at `res` m per pixel. Nominal pose errors (a few degrees of tilt, 1-2 m of height) distort each view
modestly; the views are then matched to each other with a similarity (rotation-invariant features), which is why this is usable.
"""
import numpy as np, cv2
FX, FY, CX, CY, R0, R1, R2 = 1784.4, 1776.6, 2125.1, 1521.2, 1.44295e-4, 0.468504, -0.0528464
def ortho(img, h=8.0, pitch_deg=22.0, xw=22.0, y0=3.0, y1=46.0, res=0.03):
    p = np.radians(pitch_deg); xc = np.array([1.0, 0, 0]); zc = np.array([0, np.cos(p), -np.sin(p)]); yc = np.cross(zc, xc)
    X, Y = np.meshgrid(np.arange(-xw, xw, res), np.arange(y1, y0, -res)); P = np.stack([X, Y, np.full_like(X, -h)], -1)          # ground point minus camera centre
    xcam, ycam, zcam = P @ xc, P @ yc, P @ zc; ok = zcam > 0.1; xu, yu = xcam / np.where(ok, zcam, 1), ycam / np.where(ok, zcam, 1)
    r2 = xu ** 2 + yu ** 2; k = 1 + R0 + R1 * r2 + R2 * r2 ** 2; u = FX * xu * k + CX; v = FY * yu * k + CY
    out = cv2.remap(img, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = ok & (u > 20) & (u < img.shape[1] - 20) & (v > 20) & (v < img.shape[0] - 20) & (r2 < 1.0)
    return out, valid

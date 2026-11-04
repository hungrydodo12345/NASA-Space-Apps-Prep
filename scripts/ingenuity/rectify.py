"""Rectify Ingenuity HELI_NAV frames from the equidistant fisheye of the published camera model to a virtual pinhole camera.

CAHVORE parameters in the image metadata: A (axis), H, V give fx = |A x H| = 275.4 px, fy = |A x V| = 275.5 px, principal point
(A.H, A.V) = (327.2, 249.7) px; the trailing '3.0; 0.0' is CAHVORE type 3 with linearity 0 = equidistant fisheye (r = f * theta), which matches the
133 deg field of view quoted for the camera. A virtual pinhole (r' = f * tan(theta)) of OUT_W x OUT_H px (+-45 deg horizontally) is rendered by inverse mapping.
On a flat ground plane the result obeys plane homographies, which the raw fisheye does not.
"""
import numpy as np, cv2
FX = 275.4; CX, CY = 327.2, 249.7; OUT_W, OUT_H = 560, 420
def maps():
    u, v = np.meshgrid(np.arange(OUT_W, dtype=np.float32) - OUT_W / 2, np.arange(OUT_H, dtype=np.float32) - OUT_H / 2)
    rp = np.hypot(u, v) + 1e-9; th = np.arctan(rp / FX); r = FX * th; k = r / rp
    return (CX + u * k).astype(np.float32), (CY + v * k).astype(np.float32)
MX, MY = maps()
def rectify(img):
    return cv2.remap(img.astype(np.float32), MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

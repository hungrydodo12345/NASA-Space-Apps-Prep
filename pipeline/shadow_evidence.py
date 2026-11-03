"""Multi-illumination rock evidence: a real rock casts a shadow whose direction and length follow each pass's sun.

For each pass k (sun azimuth a_k, incidence i_k, blur sigma_k) and rock diameter D, a template is rendered: a bright
disc (the sunlit rock) plus a dark ellipse (its shadow) displaced away from the sun by (D/2 + L/2), with shadow length
L = h tan(i_k) and rock height h = D/2. The template is blurred by the pass's PSF and the 25 cm pixel. Evidence for a rock
of size D at pixel p in pass k is the local normalised correlation z_k(p, D) of the (aligned) pass with that template.
A rock is consistent across passes - each shows the shadow on its own side - whereas a bright flat patch only matches the
bright part and noise does not repeat. Passes are combined by Stouffer's method, S(p, D) = sum_k z_k / sqrt(K), plus the
per-pass agreement (share of passes with z_k > 0); S is later mapped to a calibrated probability (Section 7).
Sun azimuth convention (HiRISE map-projected RDR, NORTH_AZIMUTH = 270): SUB_SOLAR_AZIMUTH is measured clockwise from
image-right; north is up. The shadow falls opposite the sun.
"""
import re, math
import numpy as np
from scipy import ndimage as ndi

PX_CM = 25.0


def pass_geometry(lbl_path):
    t = open(lbl_path, errors="replace").read()
    g = lambda k: float(re.search(rf"{k}\s*=\s*([-+0-9.]+)", t).group(1))
    return dict(incidence=g("INCIDENCE_ANGLE"), sun_az=g("SUB_SOLAR_AZIMUTH"), emission=g("EMISSION_ANGLE"))


def shadow_vector(sun_az_deg):
    """Unit vector (dy, dx) in image coordinates (y down) pointing from a rock to its shadow."""
    a = math.radians(sun_az_deg)                     # clockwise from image-right, y down: sun direction = (sin a, cos a)
    return -math.sin(a), -math.cos(a)


def template(D_cm, sun_az, incidence, sigma_px, contrast_body=1.0, contrast_shadow=1.6, sub=8, size=None, return_norm=False):
    """Rock template on the 25 cm grid (rendered at 1/sub px then integrated over pixels), zero-mean, unit-norm."""
    r = D_cm / 2 / PX_CM; L = (D_cm / 2) * math.tan(math.radians(incidence)) / PX_CM; sy, sx = shadow_vector(sun_az)
    half = int(math.ceil(r + L + 3 * sigma_px + 2)); size = size or 2 * half + 1; n = size * sub
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) / sub - size / 2
    body = (np.hypot(yy, xx) <= r).astype(float)
    cy, cx = sy * (r + L / 2), sx * (r + L / 2)
    u = (yy - cy) * sy + (xx - cx) * sx; v = -(yy - cy) * sx + (xx - cx) * sy
    shadow = ((u / max(L / 2, 0.25)) ** 2 + (v / max(r, 0.25)) ** 2 <= 1) & (body == 0)
    img = contrast_body * body - contrast_shadow * shadow
    img = img.reshape(size, sub, size, sub).mean((1, 3))           # pixel integration
    img = ndi.gaussian_filter(img, sigma_px)
    img = img - img.mean(); nrm = float(np.linalg.norm(img))
    return (img / (nrm + 1e-12), nrm) if return_norm else img / (nrm + 1e-12)


def local_ncc(img, tpl):
    """Normalised cross-correlation of img with a zero-mean unit-norm template, at every pixel (FFT, same size)."""
    from scipy.signal import fftconvolve
    k = tpl[::-1, ::-1]; ones = np.ones_like(tpl)
    num = fftconvolve(img, k, mode="same")
    s1 = fftconvolve(img, ones, mode="same"); s2 = fftconvolve(img * img, ones, mode="same"); n = tpl.size
    var = np.maximum(s2 - s1 * s1 / n, 1e-12)
    return num / np.sqrt(var)


def evidence(aligned, geom, sigmas, sizes_cm=(40, 60, 100, 150)):
    """aligned: list of passes on a common 25 cm grid; geom: list of dicts (sun_az, incidence); sigmas: per-pass PSF (px).
    Passes are combined with matched-filter weights w_k = (expected rock signal after pass k's blur) / (pass k's noise),
    S = sum_k w_k z_k / sqrt(sum_k w_k^2): a blurred or noisy pass that cannot show a rock of this size gets little say.
    Returns dict D -> (S, agreement, z_stack, weights)."""
    out = {}
    noise = []
    for img in aligned:
        hp = img - ndi.gaussian_filter(img, 1.0); noise.append(1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-9)
    for D in sizes_cm:
        zs, ws = [], []
        for img, g, s, nz in zip(aligned, geom, sigmas, noise):
            hp = img - ndi.gaussian_filter(img, 4 * max(D / PX_CM, 1))
            t, nrm = template(D, g["sun_az"], g["incidence"], s, return_norm=True)
            zs.append(local_ncc(hp, t)); ws.append(nrm / nz)
        Z = np.stack(zs); w = np.array(ws); w = w / w.max()
        out[D] = (np.tensordot(w, Z, 1) / math.sqrt((w * w).sum()), (Z > 0).mean(0), Z, w.tolist())
    return out


def peaks(S, thr, min_dist=3):
    pk = (S == ndi.maximum_filter(S, size=2 * min_dist + 1)) & (S > thr)
    return [(int(y), int(x), float(S[y, x])) for y, x in zip(*np.nonzero(pk))]

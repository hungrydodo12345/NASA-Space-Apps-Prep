"""Small discriminative model: HiRISE 25 cm patch -> cm-scale surface composition (measured by the rover).

Discovers sub-pixel surface TYPE from texture statistics; never synthesises pixels.
"""
import numpy as np
from scipy import ndimage as ndi

CLASSES_GEO = ["bedrock", "float_rock", "sand_bedform", "sand_plain", "pebbles"]


def features(p):
    """Physically motivated texture features of a reflectance patch (float array)."""
    p = p.astype(np.float64)
    p = (p - p.mean()) / (p.std() + 1e-9) * 0 + p          # keep absolute brightness too
    f = []
    m, s = p.mean(), p.std() + 1e-9
    z = (p - m) / s
    f += [m, s, *np.percentile(z, [1, 5, 25, 50, 75, 95, 99]), float((z < -1.5).mean()), float((z > 1.5).mean())]
    # multi-scale band energy (DoG pyramid) -> roughness spectrum
    prev = p
    for sig in (0.7, 1.4, 2.8, 5.6, 11.2):
        g = ndi.gaussian_filter(p, sig)
        f.append(float(np.std(prev - g) / s)); prev = g
    # gradient statistics + structure tensor coherence (ripples are strongly oriented)
    gy, gx = np.gradient(ndi.gaussian_filter(p, 0.7))
    mag = np.hypot(gx, gy)
    f += [float(mag.mean() / s), float(np.percentile(mag, 95) / s)]
    for sig in (2.0, 6.0):
        jxx, jyy, jxy = (ndi.gaussian_filter(a, sig) for a in (gx * gx, gy * gy, gx * gy))
        coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-12)
        f.append(float(coh.mean()))
    # local variance heterogeneity (patchy rocks vs uniform sand)
    lv = ndi.uniform_filter(z ** 2, 9) - ndi.uniform_filter(z, 9) ** 2
    f += [float(np.std(lv)), float(np.percentile(lv, 90))]
    # radial power-spectrum slope
    F = np.abs(np.fft.fftshift(np.fft.fft2(z * np.outer(np.hanning(z.shape[0]), np.hanning(z.shape[1]))))) ** 2
    n = z.shape[0]; yy, xx = np.mgrid[0:n, 0:n]; r = np.hypot(yy - n // 2, xx - n // 2)
    rb = np.arange(3, n // 2 - 1); pw = np.array([F[(r >= a) & (r < a + 1)].mean() for a in rb])
    f.append(float(np.polyfit(np.log(rb), np.log(pw + 1e-12), 1)[0]))
    return np.array(f)

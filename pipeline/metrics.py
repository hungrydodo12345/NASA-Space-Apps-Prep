"""Resolution metrics that don't reward blur-then-compare."""
import numpy as np


def radial_coherence(a, b, bins=16):
    """Magnitude-squared coherence-like score per radial frequency band (0..1), a vs b.

    Band k covers normalised frequency [k/bins, (k+1)/bins) * Nyquist_HR. 1 = fully recovered, 0 = absent.
    """
    fa, fb = np.fft.fftshift(np.fft.fft2(a - a.mean())), np.fft.fftshift(np.fft.fft2(b - b.mean()))
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((yy - h // 2) / (h / 2), (xx - w // 2) / (w / 2))
    out = []
    for k in range(bins):
        m = (r >= k / bins) & (r < (k + 1) / bins)
        num = np.abs((fa[m] * np.conj(fb[m])).sum()) ** 2
        den = (np.abs(fa[m]) ** 2).sum() * (np.abs(fb[m]) ** 2).sum()
        out.append(float(num / den) if den > 0 else 0.0)
    return np.array(out)


def effective_cutoff(coh, thresh=0.5):
    """Highest band index (as fraction of HR Nyquist) before coherence falls under thresh."""
    bad = np.where(coh < thresh)[0]
    k = bad[0] if len(bad) else len(coh)
    return k / len(coh)

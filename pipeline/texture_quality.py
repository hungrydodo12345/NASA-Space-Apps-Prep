"""Quality measures that separate real ground texture from rover hardware, photo seams, shadows and projection streaks."""
import numpy as np
from scipy import ndimage as ndi


def coherence(p, s=2.0, pre=0.8):
    """Mean structure-tensor coherence (0 isotropic speckle ... 1 all edges one direction)."""
    gy, gx = np.gradient(ndi.gaussian_filter(p, pre))
    jxx, jyy, jxy = (ndi.gaussian_filter(a, s) for a in (gx * gx, gy * gy, gx * gy))
    return float((np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-12)).mean())


def step_ratio(p, s=6.0):
    """Share of the variation that is a slow step (shadow edge, photo seam, exposure change)."""
    return float(np.std(ndi.gaussian_filter(p, s)) / (np.std(p) + 1e-9))


def features(p_raw):
    """All quality features of one raw-intensity patch."""
    p = p_raw.astype(np.float64); q = (p - p.mean()) / (p.std() + 1e-9)
    return dict(peak=float(np.abs(q).max()), kurt=float((q ** 4).mean()), step=step_ratio(p), coh_fine=coherence(q, 2.0),
                coh_coarse=coherence(q, 6.0, 2.0), satur=float(((p_raw >= 245) | (p_raw <= 6)).mean()), mean=float(p.mean()), std=float(p.std()))


def fine_coarse_ratio(p, sf):
    fine = p - ndi.gaussian_filter(p, sf, mode="reflect"); coarse = ndi.gaussian_filter(p, sf, mode="reflect") - ndi.gaussian_filter(p, 2 * sf, mode="reflect")
    return float(np.std(fine) / (np.std(coarse) + 1e-12))


def radial_shape(p, nb=10, fmin=0.06, fmax=0.45):
    """Fine-band radial power spectrum, normalised to mean 1 (shape only)."""
    n = p.shape[0]; w = np.outer(np.hanning(n), np.hanning(n)); q = (p - p.mean()) * w
    F = np.abs(np.fft.fft2(q)) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij"); fr = np.hypot(fx, fy)
    e = np.linspace(fmin, fmax, nb + 1)
    out = np.array([F[(fr >= e[i]) & (fr < e[i + 1])].mean() for i in range(nb)])
    return out / out.mean()


def flat_fraction(p, s=1.0, rel=0.25):
    """Share of pixels sitting in featureless plateaus (metal panels, smooth shadow) - real regolith is speckled everywhere."""
    g = np.hypot(*np.gradient(ndi.gaussian_filter(p.astype(np.float64), s)))
    return float((g < rel * (np.median(g) + 1e-9)).mean())


RULES = dict(max_step=0.50, max_coh_coarse=0.50, max_kurt=8.0, max_peak=6.0, max_flat=0.12, max_satur=0.01)


def accept(f, rules=RULES):
    """(ok, first_failed_rule) for a feature dict from features() + flat."""
    for k, key, hi in (("step", "step", rules["max_step"]), ("coh_coarse", "coh_coarse", rules["max_coh_coarse"]), ("kurt", "kurt", rules["max_kurt"]),
                       ("peak", "peak", rules["max_peak"]), ("flat", "flat", rules["max_flat"]), ("satur", "satur", rules["max_satur"])):
        if f[key] > hi: return False, k
    return True, None


def edge_sparsity(p, s=1.0):
    """Share of gradient energy held by the top 5% of pixels (sparse sharp edges = hardware; dense speckle = regolith)."""
    g = np.hypot(*np.gradient(ndi.gaussian_filter(p.astype(np.float64), s))) ** 2
    t = np.sort(g.ravel())[::-1]; return float(t[: max(1, t.size // 20)].sum() / (t.sum() + 1e-12))


def texture_density(p, s=1.0, rel=0.5):
    g = np.hypot(*np.gradient(ndi.gaussian_filter(p.astype(np.float64), s)))
    return float((g > rel * (np.median(g) + 1e-9)).mean())


FEATURE_NAMES = ["peak", "kurt", "step", "coh_fine", "coh_coarse", "satur", "flat", "sparsity", "density", "range_ratio"]


def feature_vector(p_raw, base):
    """Fixed-order feature vector (see FEATURE_NAMES) from a raw patch and its stored features() dict."""
    q = (p_raw - p_raw.mean()) / (p_raw.std() + 1e-9)
    lo, hi = np.percentile(q, [5, 95])
    return np.array([base["peak"], np.log1p(base["kurt"]), base["step"], base["coh_fine"], base["coh_coarse"], base["satur"],
                     flat_fraction(p_raw, rel=0.5), edge_sparsity(q), texture_density(q), (hi - lo)])


def load_filter(path):
    import json
    return json.load(open(path))


def ground_score(p, filt):
    """Probability (0..1) that a patch is real regolith (not hardware / seam / shadow band). p: any-scale intensity patch."""
    p = np.asarray(p, float); z = (p - p.mean()) / (p.std() + 1e-9)
    raw = np.clip(128 + 30 * z, 8, 247)                      # pseudo 8-bit so the saturation feature is comparable
    base = features(raw)
    x = (feature_vector(raw, base) - np.array(filt["mean"])) / np.array(filt["scale"])
    return float(1 / (1 + np.exp(-(x @ np.array(filt["coef"]) + filt["intercept"]))))

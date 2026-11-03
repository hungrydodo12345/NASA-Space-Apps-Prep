"""Confidence-scored fill: detail is added only where it carries a calibrated probability of being right, and the user
chooses how sure it must be (docs/METHODS.md Sections 6-7).

What carries a confidence, and where it comes from:
  * a rock (position + diameter): measured evidence only -
      - z: the blob response (negative Laplacian of Gaussian at the rock's scale) in the MEASURED restoration, in units
        of that scale's robust spread over the image;
      - S: the multi-pass shadow evidence (pipeline/shadow_evidence.py) at the nearest evaluated size, credited only for
        D >= 100 cm where it separates rocks from flat bright patches (Section 4);
      - log D.
    P(rock) = logistic(w . features), fitted on simulations with exact truth and checked on held-out simulations
    (experiments/exp7_confidence_calibration.py). The recogniser's own rock score is NOT used: against the rover answer
    key it ranks real and false candidates at chance (Section 7).
  * the base: pixels whose calibrated restoration uncertainty (pipeline/uncertainty.py, native 25 cm) exceeds a limit
    get no detail.
  * texture: real rover-photo exemplars of the recogniser's surface class. The class prediction could not be verified
    against the rover answer key (Section 7), so texture is shown only at thresholds at or below the texture confidence
    stored in the calibrator (by default the share of rover-verified ground that is soil, i.e. what the class is worth).
Everything added is locked to HiRISE (grounded_fill.lock): at every threshold the output, seen through HiRISE's blur,
is the measurement.
"""
import math, json
import numpy as np
from scipy import ndimage as ndi
from . import grounded_fill as gf, shadow_evidence as se

SIZES_CM = (25, 40, 60, 100, 150)
SHADOW_SIZES = (40, 60, 100, 150)
PX_HR_CM = 6.25


def fit_isotonic(score, label):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(score, label)
    return dict(x=[float(v) for v in ir.X_thresholds_], y=[float(v) for v in ir.y_thresholds_])


def apply_isotonic(cal, s): return np.interp(s, cal["x"], cal["y"])


def reliability(p, y, bins=10):
    """Binned reliability: per bin mean predicted, observed frequency, count; expected calibration error; Brier score."""
    p, y = np.asarray(p, float), np.asarray(y, float); e = np.linspace(0, 1, bins + 1); out = []; ece = 0.0
    for lo, hi in zip(e[:-1], e[1:]):
        m = (p >= lo) & ((p < hi) | ((hi == 1.0) & (p <= hi)))
        if m.sum():
            out.append(dict(lo=float(lo), hi=float(hi), mean_pred=float(p[m].mean()), observed=float(y[m].mean()), n=int(m.sum())))
            ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return dict(bins=out, ece=float(ece), brier=float(np.mean((p - y) ** 2)) if len(p) else None, n=int(len(p)),
                base_rate=float(y.mean()) if len(y) else None)


def auc(score, label):
    from sklearn.metrics import roc_auc_score
    label = np.asarray(label)
    return float(roc_auc_score(label, score)) if 0 < label.sum() < len(label) else None


def blob_candidates(X, floor=2.0, sizes_cm=SIZES_CM, min_dist=3):
    """Bright blob candidates in the measured restoration X (6.25 cm grid): local maxima over position and scale of the
    negative LoG, each scale in units of its own robust spread over the image (median / MAD), so z says how unusual
    the blob is for this ground at this size. Returns list of (y, x, D_cm, z)."""
    def robust_z(a): return (a - np.median(a)) / (1.4826 * np.median(np.abs(a - np.median(a))) + 1e-12)
    R = np.stack([robust_z(-ndi.gaussian_laplace(X, s)) for s in [D / 2 / PX_HR_CM / math.sqrt(2) for D in sizes_cm]])
    best = R.max(0); arg = R.argmax(0)
    pk = (best == ndi.maximum_filter(best, size=2 * min_dist + 1)) & (best > floor)
    return [(int(y), int(x), sizes_cm[arg[y, x]], float(best[y, x])) for y, x in zip(*np.nonzero(pk))]


def align_passes(frames, flows):
    """Bring every pass onto the reference grid with its Gotcha field (pass(x) ~= ref(x + d)  =>  ref(y) ~= pass(y - d))."""
    out = []
    for f, (dx, dy) in zip(frames, flows):
        yy, xx = np.mgrid[0:f.shape[0], 0:f.shape[1]].astype(float)
        out.append(ndi.map_coordinates(f, [yy - dy, xx - dx], order=3, mode="nearest"))
    return out


def features(cands, shadow, scale=4):
    """Feature rows for candidates. shadow: dict D -> S map on the native grid (from shadow_evidence.evidence) or None."""
    rows = []
    for y, x, D, z in cands:
        s = 0.0
        if shadow is not None and D >= 100:
            Dn = min(SHADOW_SIZES, key=lambda d: abs(d - D)); S = shadow[Dn][0]
            yl, xl = int(round(y / scale)), int(round(x / scale))
            s = float(S[max(yl - 1, 0):yl + 2, max(xl - 1, 0):xl + 2].max())
        rows.append([z, math.log(D / 25.0), s, float(D >= 100)])
    return np.array(rows, float).reshape(-1, 4)


FEATURE_NAMES = ["blob z (restoration)", "log(D / 25 cm)", "shadow evidence S (D >= 100 cm only)", "D >= 100 cm"]


def fit_logistic(F, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=2000).fit(F, y)
    return dict(coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]), features=FEATURE_NAMES)


def predict(cal, F):
    w = np.array(cal["coef"]); return 1.0 / (1.0 + np.exp(-(F @ w + cal["intercept"])))


def score(X, cal, aligned=None, geoms=None, sigmas=None, sigma_native=None, sigma_limit=None, scale=4):
    """Calibrated rock list [(y, x, D, P)] on the 6.25 cm grid and the base-trust mask."""
    cands = blob_candidates(X)
    shadow = se.evidence(aligned, geoms, sigmas, sizes_cm=SHADOW_SIZES) if aligned is not None else None
    P = predict(cal["rock"], features(cands, shadow, scale)) if cands else np.zeros(0)
    trust = np.ones(X.shape, bool)
    if sigma_native is not None and sigma_limit is not None: trust = sigma_native <= sigma_limit
    rocks = [(y, x, D, float(p)) for (y, x, D, _), p in zip(cands, P) if trust[y, x]]
    return dict(rocks=rocks, trust=trust, shadow=shadow)


def render(base, class_prob, layers, lib, psf_px, threshold, texture_conf, seed=0):
    """Measured base + locked detail with confidence >= threshold. Returns (image, info)."""
    keep = [q for q in layers["rocks"] if q[3] >= threshold]
    lib = dict(lib, support_k=-1e9)          # rocks are already selected by calibrated probability
    _, inf = gf.fill(base, class_prob, keep, lib, psf_px, seed=seed)
    show_tex = threshold <= texture_conf
    tex_mask = ndi.gaussian_filter(layers["trust"].astype(float), 2 * psf_px) * float(show_tex)
    added = gf.lock(inf["texture"] * tex_mask + inf["rocks"], psf_px)
    out = base + added
    ob = gf.observe(base, psf_px); resid = float(np.std(gf.observe(out, psf_px) - ob) / (np.std(ob - ob.mean()) + 1e-12))
    return out, dict(threshold=threshold, rocks_shown=len(keep), texture_shown=bool(show_tex), lock_residual_rel=resid,
                     added=added, synthetic=True)


def save(cal, path): json.dump(cal, open(path, "w"), indent=1)
def load(path): return json.load(open(path))

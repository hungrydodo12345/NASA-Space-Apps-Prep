"""Real-data reliability of the RECOGNISER's own confidences against the rover answer key (docs/METHODS.md Section 7).

Outcome: they are not informative against this answer key (held-out ECE worse than raw; class confidence collapses
to the base rate), so the fill's confidence is taken from measured evidence calibrated in simulation instead
(experiments/exp7_confidence_calibration.py). This script is kept as the record of that test.

Answer key: the rover's own top-down imagery at each traverse stop (rover truth projected onto the HiRISE grid), with
rocks >= 25 cm found in it and AI4Mars surface classes. Predictions: the recogniser's OUT-OF-FOLD outputs (each stop
predicted by a model whose spatial fold never contained it; scripts/train_recognizer.py).

1. Rocks. Every candidate the recogniser proposes (score >= 0.05) inside rover-verified ground is labelled 1 if a
   truth rock lies within max(2 px, D) of it, else 0. Isotonic regression maps score -> P(rock). The calibrator is
   CROSS-FITTED: candidates of fold f are calibrated by a map fitted on the other folds only, and the reliability
   reported is of those held-out probabilities. The final calibrator (all folds) is saved for use.
   Also: for each probability threshold t, the share of shown rocks that are real (precision) and the share of truth
   rocks shown (recall) - the trade-off the user's slider moves along.
2. Surface class. Pixels with an AI4Mars class in the truth (every 4th pixel): top class probability -> P(correct),
   same cross-fitting.
Usage: python scripts/recogniser_reliability_traverse.py <restored_dir> <truth_dir> <pred_dir> <out_json> <isotonic_json>
  (as run: out_json = results/recogniser_reliability_traverse.json)
"""
import sys, os, glob, json, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline import recognizer as rz, confidence as cf
sv = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[2]]
from experiments import exp3_rocks as E
sys.argv = sv
RD, TRUTH, PRED, OUT, CAL = sys.argv[1:6]
PX = E.PX


def near(p, ds, tol=1.0): return any(np.hypot(p[0] - q[0], p[1] - q[1]) <= max(2.0, tol * max(p[2], q[2]) / PX) for q in ds)


R = []                                     # rock candidates: (fold, stop, score, label, D)
K = []                                     # class pixels: (fold, conf, correct)
T = []                                     # truth rocks: (fold, stop, best candidate score within reach or 0)
for f in sorted(glob.glob(f"{PRED}/pred_*.npz")):
    stop = os.path.basename(f)[5:-4]
    if not (os.path.exists(f"{RD}/{stop}.npz") and os.path.exists(f"{TRUTH}/{stop}.npz")): continue
    p = np.load(f); fold = int(p["fold"]); rp = p["rock_prob"].astype(np.float32); cp = p["class_prob"].astype(np.float32)
    r = dict(np.load(f"{RD}/{stop}.npz")); t = np.load(f"{TRUTH}/{stop}.npz")
    _, _, C, M, rocks = rz.stop_arrays(r, t, E.detect, PX)
    if M.sum() < 2000: continue
    cand = [q for q in cf.candidates(rp) if M[q[0], q[1]]]
    for q in cand: R.append((fold, stop, q[3], int(near(q, rocks)), q[2]))
    for q in rocks:
        s = [c[3] for c in cand if near(c, [q])]; T.append((fold, stop, max(s) if s else 0.0))
    yy, xx = np.nonzero((C >= 0) & M); sel = slice(None, None, 4)
    if len(yy):
        y, x = yy[sel], xx[sel]; K += list(zip([fold] * len(y), cp.max(0)[y, x].tolist(), (cp.argmax(0)[y, x] == C[y, x]).astype(int).tolist()))
    print(stop, "fold", fold, "candidates", len(cand), "truth rocks", len(rocks), flush=True)

STOPS = sorted({x[1] for x in R})
R = np.array([(a, s, l, D) for a, _, s, l, D in R], float); K = np.array(K, float); T = np.array([(a, s) for a, _, s in T], float)
folds = sorted(set(R[:, 0].astype(int)))


def crossfit(fold_col, score, label):
    out = np.zeros_like(score)
    for f in folds:
        tr, te = fold_col != f, fold_col == f
        out[te] = cf.apply(cf.fit_isotonic(score[tr], label[tr]), score[te])
    return out


pr = crossfit(R[:, 0], R[:, 1], R[:, 2]); pk = crossfit(K[:, 0], K[:, 1], K[:, 2])
# truth-rock recall at each calibrated threshold: map each truth rock's best candidate score through its fold's calibrator
pt = np.zeros(len(T))
for f in folds:
    cal = cf.fit_isotonic(R[R[:, 0] != f, 1], R[R[:, 0] != f, 2]); m = T[:, 0] == f; pt[m] = np.where(T[m, 1] > 0, cf.apply(cal, T[m, 1]), 0.0)
sweep = []
for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    shown = pr >= th
    sweep.append(dict(threshold=th, rocks_shown=int(shown.sum()), precision_pct=100 * float(R[shown, 2].mean()) if shown.any() else None,
                      recall_pct=100 * float((pt >= th).mean()) if len(pt) else None,
                      pixels_with_texture_pct=100 * float((pk >= th).mean()), texture_class_correct_pct=100 * float(K[pk >= th, 2].mean()) if (pk >= th).any() else None))
by_band = {}
for name, (lo, hi) in {"25-40 cm": (0, 45), ">=50 cm": (45, 1e9)}.items():
    m = (R[:, 3] >= lo) & (R[:, 3] < hi); by_band[name] = cf.reliability(pr[m], R[m, 2])
out = dict(stops=len(STOPS), stop_ids=STOPS,
           rock=dict(raw_score=cf.reliability(np.clip(R[:, 1], 0, 1), R[:, 2]), calibrated_heldout=cf.reliability(pr, R[:, 2]), by_band_calibrated_heldout=by_band,
                     candidates=int(len(R)), truth_rocks=int(len(T))),
           surface_class=dict(raw_score=cf.reliability(K[:, 1], K[:, 2]), calibrated_heldout=cf.reliability(pk, K[:, 2]), pixels=int(len(K))),
           threshold_sweep=sweep, folds=[int(f) for f in folds])
cal = dict(rock=cf.fit_isotonic(R[:, 1], R[:, 2]), **{"class": cf.fit_isotonic(K[:, 1], K[:, 2])},
           note="isotonic, fitted on all out-of-fold recogniser outputs vs rover truth; apply with np.interp(score, x, y)")
json.dump(out, open(OUT, "w"), indent=1); json.dump(cal, open(CAL, "w"), indent=1)
print(json.dumps(dict(rock_ece_raw=out["rock"]["raw_score"]["ece"], rock_ece_cal=out["rock"]["calibrated_heldout"]["ece"],
                      class_ece_raw=out["surface_class"]["raw_score"]["ece"], class_ece_cal=out["surface_class"]["calibrated_heldout"]["ece"], sweep=sweep), indent=1))

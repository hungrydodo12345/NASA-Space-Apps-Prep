"""Real-data reliability check of the confidence model against the rover answer key (docs/METHODS.md Section 7).

For every traverse stop with rover truth: rock candidates and P(rock) from the MEASURED restoration
(pipeline/confidence.score with the simulation-fitted calibrator; no per-pass stack is stored for these stops, so the
shadow feature is 0), compared with rocks >= 25 cm found in the rover's top-down imagery inside rover-verified ground.
Same comparison for the recogniser's out-of-fold rock scores. Reported: reliability (binned, ECE), AUC pooled and
per stop, and AUC for a 'chance' control (candidates' labels taken against the truth shifted by 40 px).
If the measured evidence and the recogniser are both at chance while the chance control is too, the answer key cannot
resolve rock positions at this scale (registration), and calibration has to come from simulation.
Usage: python scripts/anchored_reliability.py <restored_dir> <truth_dir> <pred_dir> <calibrator_json> <out_json>
"""
import sys, os, glob, json, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline import recognizer as rz, confidence as cf
RD, TRUTH, PRED, CAL, OUT = sys.argv[1:6]
sys.argv = [sys.argv[0], TRUTH]
from experiments import exp3_rocks as E
PX = E.PX; cal = cf.load(CAL)


def near(p, ds, tol=1.0): return any(np.hypot(p[0] - q[0], p[1] - q[1]) <= max(2.0, tol * max(p[2], q[2]) / PX) for q in ds)


rows = {"physics P(rock)": [], "recogniser score": []}; per_stop = []
for f in sorted(glob.glob(f"{PRED}/pred_*.npz")):
    stop = os.path.basename(f)[5:-4]
    if not (os.path.exists(f"{RD}/{stop}.npz") and os.path.exists(f"{TRUTH}/{stop}.npz")): continue
    r = dict(np.load(f"{RD}/{stop}.npz")); t = np.load(f"{TRUTH}/{stop}.npz")
    _, _, _, M, rocks = rz.stop_arrays(r, t, E.detect, PX)
    if M.sum() < 2000 or not rocks: continue
    n = M.shape[0]; shifted = [((y + 40) % n, (x + 40) % n, D) for y, x, D in rocks]
    phys = [q for q in cf.score(r["restored_multi"].astype(float), cal)["rocks"] if M[q[0], q[1]]]
    reco = [q for q in rz.decode(np.load(f)["rock_prob"].astype(np.float32), 0.05) if M[q[0], q[1]]]
    st = dict(stop=stop, truth_rocks=len(rocks))
    for name, cands in (("physics P(rock)", phys), ("recogniser score", reco)):
        lab = [int(near(q, rocks)) for q in cands]; lab_c = [int(near(q, shifted)) for q in cands]
        rows[name] += [(q[3], l, lc) for q, l, lc in zip(cands, lab, lab_c)]
        st[name] = dict(candidates=len(cands), auc=cf.auc([q[3] for q in cands], lab), auc_shifted_truth=cf.auc([q[3] for q in cands], lab_c))
    per_stop.append(st); print(stop, {k: (v["candidates"], v["auc"]) for k, v in st.items() if isinstance(v, dict)}, flush=True)
out = dict(stops=len(per_stop), per_stop=per_stop, pooled={})
for name, R in rows.items():
    R = np.array(R, float)
    if not len(R): continue
    p = R[:, 0] if name.startswith("physics") else np.clip(R[:, 0], 0, 1)
    aucs = [s[name]["auc"] for s in per_stop if s[name]["auc"] is not None]
    out["pooled"][name] = dict(candidates=int(len(R)), share_matching_truth=float(R[:, 1].mean()), share_matching_shifted_truth=float(R[:, 2].mean()),
                               auc=cf.auc(p, R[:, 1]), auc_shifted_truth=cf.auc(p, R[:, 2]), median_per_stop_auc=float(np.median(aucs)) if aucs else None,
                               reliability=cf.reliability(p, R[:, 1]))
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "reliability"} for k, v in out["pooled"].items()}, indent=1))

"""Render Stage 1 + Stage 2 for held-out stops: raw HiRISE | restoration | recognised rocks & surface | grounded fill | rover truth.

Libraries for each stop come from the training stops of that stop's cross-validation fold only.
Usage: python scripts/render_grounded.py <restored_dir> <truth_dir> <recognizer_dir> <out_png> [stops comma list]
"""
import sys, os, glob, json
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline import recognizer as rz, grounded_fill as gf
sv = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[2]]
from experiments import exp3_rocks as E
sys.argv = sv
RD, TRUTH, REC, PNG = sys.argv[1:5]
PX = E.PX; PSF_PX = 1.07 * 4
preds = {os.path.basename(f)[5:-4]: f for f in glob.glob(f"{REC}/pred_*.npz")}
fold = {s: int(np.load(f)["fold"]) for s, f in preds.items()}
stops = sys.argv[5].split(",") if len(sys.argv) > 5 else None
if stops is None:   # default: the held-out stops with the most rover rocks
    sc = json.load(open(glob.glob(f"{REC}/ucl_yardstick_thr*.json")[0]))["per_stop"]
    stops = [r["stop"] for r in sorted(sc, key=lambda r: -sum(r["truth"].values()))[:3]]
libs, stats = {}, {}
GPT_DIR = os.environ.get("GPT_DIR")
NC = 6 if GPT_DIR else 5
fig, ax = plt.subplots(len(stops), NC, figsize=(4.2 * NC, 4.6 * len(stops) + 1.2)); ax = np.atleast_2d(ax)
CL = np.array([[0.80, 0.68, 0.50], [0.55, 0.45, 0.40], [0.93, 0.80, 0.45], [0.35, 0.25, 0.25]])
for r_, stop in enumerate(stops):
    f = fold[stop]
    if f not in libs:
        tr_stops = [s for s in fold if fold[s] != f]
        T = [np.load(f"{TRUTH}/{s}.npz") for s in tr_stops]
        R = [rz.stop_arrays(dict(np.load(f"{RD}/{s}.npz")), t, E.detect, PX)[4] for s, t in zip(tr_stops, T)]
        libs[f] = gf.calibrate(gf.build_library(T, R, PX, PSF_PX), [dict(np.load(f"{RD}/{s}.npz")) for s in tr_stops], T, PSF_PX, PX)
    lib = libs[f]
    r = dict(np.load(f"{RD}/{stop}.npz")); t = np.load(f"{TRUTH}/{stop}.npz"); p = np.load(preds[stop])
    X, H, C, M, truth_rocks = rz.stop_arrays(r, t, E.detect, PX)
    rp, cp, thr = p["rock_prob"].astype(float), p["class_prob"].astype(float), float(p["thr"])
    rocks = [q for q in rz.decode(rp, thr)]
    base = r["restored_multi"].astype(float)
    out, info = gf.fill(base, cp, rocks, lib, PSF_PX, seed=1)
    stats[stop] = dict(lock_residual_rel=info["lock_residual_rel"], rocks_recognised=len(rocks), rocks_placed=info.get("rocks_placed"), rover_rocks=len(truth_rocks))
    Z, _ = gf.to_grid(t, PX, base.shape[0])
    ys, xs = np.nonzero(np.isfinite(Z)); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = lambda a: a[y0:y1, x0:x1]
    show = [("Raw HiRISE (one pass)", r["bicubic"])]
    if GPT_DIR:
        gp = f"{GPT_DIR}/{stop}_gptstyle.npz"
        show.append(("UCL-style TV restoration\n(our reimplementation of GPT's approach,\nnot UCL's product)", np.load(gp)["gpt_style"] if os.path.exists(gp) else np.full_like(base, np.nan)))
    show += [("Our multi-frame restoration (8 passes)", base), ("Stage 1: recognised surface + rocks", None), ("Stage 2: grounded fill", out), ("Rover photos (truth)", Z)]
    for j, (ti, im) in enumerate(show):
        a = ax[r_, j]
        if im is None:
            rgb = np.tensordot(np.moveaxis(cp, 0, -1), CL, axes=([2], [0])); shade = (base - base.mean()) / (base.std() * 4) + 0.5
            a.imshow(np.clip(crop(rgb) * np.clip(crop(shade), 0.3, 1.2)[..., None], 0, 1))
            for y, x, D, s in rocks:
                if y0 <= y < y1 and x0 <= x < x1: a.add_patch(Circle((x - x0, y - y0), D / 2 / PX, fill=False, ec="#00e5ff" if D < 50 else "#ff1744", lw=1.5))
        else:
            c = crop(im); lo, hi = np.nanpercentile(c, [1, 99]); a.imshow(c, cmap="gray", vmin=lo, vmax=hi)
            if j == NC - 1:
                for y, x, D in truth_rocks: a.add_patch(Circle((x - x0, y - y0), D / 2 / PX, fill=False, ec="#ffb300" if D < 50 else "#e53935", lw=1.3))
        a.set_xticks([]); a.set_yticks([]); a.set_title(ti if r_ == 0 else "", fontsize=10)
        a.plot([6, 6 + 100 / PX], [(y1 - y0) - 8] * 2, color="w", lw=3)
    ax[r_, 0].set_ylabel(f"held-out stop {stop}", fontsize=10)
    ax[r_, NC - 3].set_xlabel(f"{len(rocks)} rocks recognised (cyan 25-40 cm, red >= 50 cm)", fontsize=8)
    ax[r_, NC - 1].set_xlabel(f"{len(truth_rocks)} rover rocks >= 25 cm (amber 25-40, red >= 50)", fontsize=8)
    ax[r_, NC - 2].set_xlabel(f"seen at HiRISE resolution, differs from the measurement by {100 * info['lock_residual_rel']:.2f}%", fontsize=8)
fig.text(0.5, 0.005, "Surface colours: soil sandy-beige, bedrock grey-brown, sand yellow, big rock dark. Scale bar 1 m. Every model shown never saw its stop.", ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(PNG, dpi=105)
json.dump(stats, open(PNG.replace(".png", "_stats.json"), "w"), indent=1); print(json.dumps(stats, indent=1))

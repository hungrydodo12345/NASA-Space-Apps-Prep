"""Confidence-scored fill on a chain product (scripts/run_chain.py), rendered at user-chosen confidence thresholds, plus
a self-contained HTML page with a threshold slider.

Usage:
  python scripts/confidence_fill.py <chain_dir> <hirise_rdr_dir> <recognizer_dir> <out_dir>
         [--calibrator data_products/confidence_calibrator.json] [--confidence 0.5] [--thresholds 0.1,0.2,...]
         [--sigma-limit 0.5]
--confidence renders one threshold (CLI use); --thresholds renders several and writes the slider page (default
0.1 ... 0.9). --sigma-limit: no detail where the calibrated native-scale uncertainty exceeds this share of the scene's
standard deviation.

Steps:
 1. Recogniser inputs from the chain's own passes: bicubic reference, average of aligned passes, single-pass PDE-TV,
    multi-pass PDE-TV (the chain's restoration). The three cross-validation recogniser models are averaged; the class
    map chooses which rover-photo texture is drawn (pipeline/grounded_fill.fill).
 2. Rock candidates and P(rock) from measured evidence (pipeline/confidence.score: blob repeatability against two
    restorations from disjoint pass halves, multi-pass shadow evidence for >= 100 cm, size), with the
    simulation-fitted calibrator.
 3. For each threshold t: base + lock(texture if t <= texture confidence, rocks with P >= t). Every output, seen
    through HiRISE's blur, equals the measurement (lock residual reported).
Outputs: fill_t{t}.png (8-bit, one fixed contrast stretch for all thresholds), measured.png, rocks.csv
(x_m, y_m from the tile's top-left, D_cm, P), hazard.png + hazard.csv (pipeline/hazard.py; --d-hazard, --p-alarm,
--cell), fill_meta.json, confidence_viewer.html (when several thresholds).
"""
import sys, os, glob, json, argparse, base64, io, pickle
import numpy as np, torch
from scipy import ndimage as ndi
from PIL import Image
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline import recognizer as rz, confidence as cf, pde_tv, shadow_evidence as se, hazard as hz

ap = argparse.ArgumentParser()
ap.add_argument("chain"); ap.add_argument("rdr"); ap.add_argument("rec"); ap.add_argument("out")
ap.add_argument("--calibrator", default=os.path.join(ROOT, "data_products", "confidence_calibrator.json"))
ap.add_argument("--confidence", type=float, default=None); ap.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
ap.add_argument("--sigma-limit", type=float, default=0.5); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--d-hazard", type=int, default=40, help="hazard rock diameter (cm)"); ap.add_argument("--p-alarm", type=float, default=0.5)
ap.add_argument("--cell", type=float, default=1.0, help="hazard map cell (m)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
d = np.load(f"{a.chain}/chain.npz"); meta = json.load(open(f"{a.chain}/chain.json")); SC = meta["scale"]
cal = cf.load(a.calibrator); blur = json.load(open(os.path.join(ROOT, "data_products", "pass_blur.json")))
base = d["restored"].astype(float); s0 = float(base.std())
frames = list(d["frames"].astype(float)); flows = list(zip(d["flows_dx"].astype(float), d["flows_dy"].astype(float)))
used = meta["passes_used"]; sig_lr = [blur[o]["sigma_px"] for o in used]
psf = sig_lr[0] * SC                                                       # reference pass's measured blur (HR px)

# 1. recogniser
aligned = cf.align_passes(frames, flows)
up = lambda z: ndi.zoom(z, SC, order=3, grid_mode=True, mode="nearest")
single, _ = pde_tv.pde_tv([frames[0]], [flows[0]], scale=SC, psf_sigma=psf, lam=meta["lam"], iters=meta["iters"])
P = dict(bicubic=up(frames[0]), average=up(np.mean(aligned, 0)), deconv_1frame=single, restored_multi=base)
X = np.stack([rz._std(P[k]) for k in rz.INPUTS])
models = []
for f in sorted(glob.glob(f"{a.rec}/recognizer_fold*.pt")):
    m = rz.UNet(); m.load_state_dict(torch.load(f, map_location="cpu")); m.eval(); models.append(m)
cp = np.mean([rz.predict(m, X)[1] for m in models], 0)
lib, _ = pickle.load(open(f"{a.rec}/library_all.pkl", "rb"))

# 2. calibrated rocks
geoms = [se.pass_geometry(f"{a.rdr}/{o}_RED.LBL") for o in used]
sn = ndi.zoom(d["sigma_native_cal"].astype(float), base.shape[0] / d["sigma_native_cal"].shape[0], order=1)
halves = cf.split_halves(frames, flows, [s_ * SC for s_ in sig_lr], scale=SC, lam=meta["lam"], iters=meta["iters"])
layers = cf.score(base, cal, aligned, geoms, sig_lr, sigma_native=sn, sigma_limit=a.sigma_limit * s0, scale=SC, halves=halves)
tex_conf = cal.get("texture_confidence") or 0.0

# 3. render
ths = [a.confidence] if a.confidence is not None else [float(t) for t in a.thresholds.split(",")]
lo, hi = np.percentile(base, [0.5, 99.5]); to8 = lambda im: (np.clip((im - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
Image.fromarray(to8(base)).save(f"{a.out}/measured.png")
renders, info_all = {}, []
for t in ths:
    out, info = cf.render(base, cp, layers, lib, psf, t, tex_conf, seed=a.seed)
    Image.fromarray(to8(out)).save(f"{a.out}/fill_t{t:.2f}.png"); renders[t] = f"fill_t{t:.2f}.png"
    info_all.append({k: v for k, v in info.items() if k != "added"}); print(info_all[-1], flush=True)
px_m = 0.25 / SC
with open(f"{a.out}/rocks.csv", "w") as fh:
    fh.write("x_m,y_m,diameter_cm,p_rock\n")
    for y, x, D, p in sorted(layers["rocks"], key=lambda q: -q[3]): fh.write(f"{x * px_m:.3f},{y * px_m:.3f},{D},{p:.3f}\n")
fm = dict(chain=a.chain, passes=used, psf_hr_px=psf, texture_confidence=tex_conf, sigma_limit_rel=a.sigma_limit,
          trusted_area_pct=100 * float(layers["trust"].mean()), candidates=len(layers["rocks"]),
          class_share_pct={n: 100 * float((cp.argmax(0) == i).mean()) for i, n in enumerate(["soil", "bedrock", "sand", "big rock"])},
          renders=info_all, calibrator=a.calibrator)
# hazard map (Section 8)
H = hz.cells(base.shape, layers["rocks"], layers["trust"], px_m=px_m, cell_m=a.cell, d_hazard=a.d_hazard, p_alarm=a.p_alarm)
hz.to_csv(H, f"{a.out}/hazard.csv"); fm["hazard"] = hz.summary(H)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(1, 2, figsize=(13, 6.6)); ext = (0, base.shape[1] * px_m, base.shape[0] * px_m, 0)
for k, (title, arr, cm) in enumerate((("P(at least one rock >= %d cm per %.0f m cell)" % (a.d_hazard, a.cell), H["p_hazard"], "magma"),
                                      ("Status: clear / hazard (P >= %.2f) / unknown (base not trusted)" % a.p_alarm,
                                       np.select([H["status"] == "clear", H["status"] == "hazard"], [0, 1], 2), ListedColormap(["#3a9d5d", "#d64a2f", "#9a9a9a"])))):
    ax[k].imshow(base, cmap="gray", vmin=lo, vmax=hi, extent=ext)
    im = ax[k].imshow(arr, cmap=cm, alpha=0.55, extent=ext, vmin=0, vmax=1 if k == 0 else 2, interpolation="nearest")
    ax[k].set_title(title, fontsize=10); ax[k].set_xlabel("m"); ax[k].set_ylabel("m")
    if k == 0: fig.colorbar(im, ax=ax[k], fraction=0.046)
fig.tight_layout(); fig.savefig(f"{a.out}/hazard.png", dpi=130); plt.close(fig)
json.dump(fm, open(f"{a.out}/fill_meta.json", "w"), indent=1)

# slider page
if len(ths) > 1:
    b64 = lambda p: base64.b64encode(open(f"{a.out}/{p}", "rb").read()).decode()
    imgs = {f"{t:.2f}": b64(renders[t]) for t in ths}; imgs["measured"] = b64("measured.png")
    rk = [[round(x * px_m, 3), round(y * px_m, 3), D, round(p, 3)] for y, x, D, p in layers["rocks"]]
    W = base.shape[1] * px_m
    html = open(os.path.join(ROOT, "scripts", "confidence_viewer_template.html")).read()
    html = html.replace("__IMAGES__", json.dumps(imgs)).replace("__ROCKS__", json.dumps(rk)).replace("__THS__", json.dumps([f"{t:.2f}" for t in ths]))
    html = html.replace("__WIDTH_M__", f"{W:.3f}").replace("__META__", json.dumps(dict(passes=used, texture_confidence=tex_conf, trusted_area_pct=fm["trusted_area_pct"],
                                                                                   renders=[{k: r[k] for k in ("threshold", "rocks_shown", "texture_shown", "lock_residual_rel")} for r in info_all])))
    open(f"{a.out}/confidence_viewer.html", "w").write(html); print("viewer:", f"{a.out}/confidence_viewer.html")

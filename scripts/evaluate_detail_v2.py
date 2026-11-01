"""Re-score the rover-texture detail layer on CLEAN rover ground patches (no photo seams, no hardware).

Compares texture libraries under identical calibration:  clean (filtered, 32 px)  vs  old (mosaic, unfiltered, 64 px)
vs  hardware-only (control that should score badly)  vs  no detail  vs  off-the-shelf upscalers.
Usage: python scripts/evaluate_detail_v2.py <hero_dir> <rover_truth_dir> <candidates.npz> <filter_scores.npy> <filter.json>
                                            <sr_models_dir> <cells.npz> <ortho.jp2> <out.json> [dump_dir]
"""
import sys, os, json
import numpy as np, cv2, glymur
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from pipeline import detail as dt, texture_quality as tq, groundtruth as gt, surface_classifier as scl
from experiments import runner
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

HERO, TRUTH, CAND, SCORES, FILT, MODELS, CELLS, ORTHO, OUTJ = sys.argv[1:10]
DUMP = sys.argv[10] if len(sys.argv) > 10 else None
SC, T, NCROP, THR = 4, 32, 40, 0.85
meta = json.load(open(f"{HERO}/measured_meta.json"))
fused = np.load(f"{HERO}/measured_fused.npy").astype(np.float64); single = np.load(f"{HERO}/measured_single.npy").astype(np.float64)
PSF = json.load(open(os.path.join(ROOT, "data_products", "psf_calibration.json")))["sigma_px_median"]; PSF_HR = PSF * SC
filt = tq.load_filter(FILT)
cand = {k: v for k, v in np.load(CAND).items()}; score = np.load(SCORES)      # in memory: forked workers must not share a lazy zip handle
N = 252; up = lambda a: ndi.zoom(a, SC, order=3, grid_mode=True, mode="nearest")
f_ft = np.array([(0.06 + 0.39 * (i + 0.5) / 10) for i in range(10)])           # radial_shape bin centres (cycles/px)

def to_grid(p5):                                   # 40x40 @5 cm -> 32x32 @ ~6.3 cm
    return ndi.zoom(p5.astype(float), T / p5.shape[0], order=1)

def hp_tile(p):
    q = p - ndi.gaussian_filter(p, 4 * PSF_HR, mode="reflect"); return ((q - q.mean()) / (q.std() + 1e-9)).astype(np.float32)

# ---- stops inside the hero window, with their clean truth patches
stops = []
for f in sorted(os.listdir(TRUTH)):
    d = {k: v for k, v in np.load(f"{TRUTH}/{f}").items()}; L0, S0 = gt.latlon_to_ortho(float(d["lat"]), float(d["lon"])); r, c = L0 - meta["top"], S0 - meta["left"]
    name = f[:-4]
    if not (60 < r < meta["size"] - 60 and 60 < c < meta["size"] - 60): continue
    ii = np.where((cand["stop"] == name) & (score >= THR))[0]
    stops.append(dict(stop=name, r=r, c=c, sol=int(d["sol"]), idx=ii, z=d["z"], cls=d["cls"]))
stops.sort(key=lambda s: s["sol"])
evals = [s for s in stops if len(s["idx"]) >= 2]
print(f"stops in window {len(stops)}; with >=2 clean truth patches {len(evals)}; total clean patches {sum(len(s['idx']) for s in stops)}", flush=True)
for s in stops: s["truth"] = [to_grid(cand["patches"][i].astype(float)) for i in s["idx"]]

# ---- orbital features per stop (classifier input), cells for the classifier
cz = {k: v for k, v in np.load(CELLS).items()}; X, Y, CL, CS = cz["X"], cz["Y"], cz["L"], cz["S"]
J = glymur.Jp2k(ORTHO)
for s in stops:
    Lc, Sc = int(s["r"] + meta["top"]), int(s["c"] + meta["left"]); H1, H2 = 32, 96
    img = np.asarray(J[Lc - H2:Lc + H2, Sc - H2:Sc + H2]).astype(float)
    s["feat"] = np.r_[scl.features(img[H2 - H1:H2 + H1, H2 - H1:H2 + H1]), scl.features(img)]
    h = N // SC // 2 + 2; r0, c0 = int(round(s["r"])), int(round(s["c"]))
    s["lr_f"], s["lr_s"] = fused[r0 - h:r0 + h, c0 - h:c0 + h], single[r0 - h:r0 + h, c0 - h:c0 + h]

# ---- old (v1) texture source: mosaic patches, 64 px, no filtering
def truth_grid(s):
    k = 0.05 / (meta["pixel_m"] / SC)
    z = ndi.zoom(np.nan_to_num(s["z"]), k, order=1); v = ndi.zoom(np.isfinite(s["z"]).astype(float), k, order=1) > 0.99
    c = ndi.zoom(np.nan_to_num(s["cls"], nan=-1), k, order=0); return np.where(v, z, np.nan), np.where(v & (c >= 0), c, np.nan)
for s in stops: s["old"] = dt.extract_patches(*truth_grid(s), PSF_HR, 64, min_frac=0.6)

# ---- hardware-only control library
hw_idx = np.where(score < 0.2)[0]
print(f"control library: {len(hw_idx)} clear hardware/non-ground patches", flush=True)

def library(kind, train, held_stops):
    lib = {k: [] for k in range(4)}
    if kind == "clean":
        for s in train:
            for i in s["idx"]: lib[int(cand["cls"][i])].append(hp_tile(to_grid(cand["patches"][i])))
    elif kind == "old":
        for s in train:
            for k, v in s["old"].items(): lib[k] += v
    elif kind == "hardware":
        names = {s["stop"] for s in held_stops}
        for i in hw_idx:
            if cand["stop"][i] in names: continue
            lib[int(cand["cls"][i])].append(hp_tile(to_grid(cand["patches"][i])))
    for k in range(4):
        if len(lib[k]) < 4: lib[k] = lib[k] + lib[dt.FALLBACK[k]]
    return lib

def amp_by_class(train):
    r = {k: [] for k in range(4)}
    for s in train:
        for i in s["idx"]: r[int(cand["cls"][i])].append(tq.fine_coarse_ratio(to_grid(cand["patches"][i].astype(float)), PSF_HR))
    allr = [x for v in r.values() for x in v]; m = np.mean(allr) if allr else 1.0
    out = {k: (float(np.median(v)) / m if len(v) >= 4 else 1.0) for k, v in r.items()}
    return out

# ---- metrics on 32-px patch sets (scale-free, registration-free)
def patch_stats(ps):
    rho = np.median([tq.fine_coarse_ratio(p, PSF_HR) for p in ps]); coh = np.median([tq.coherence((p - p.mean()) / (p.std() + 1e-9)) for p in ps])
    shp = np.mean([tq.radial_shape(p) for p in ps], 0)
    gm = []
    for p in ps:
        g = np.hypot(*np.gradient(ndi.gaussian_filter(p, 1.0))); gm.append(g / (np.std(ndi.gaussian_filter(p, PSF_HR) - ndi.gaussian_filter(p, 2 * PSF_HR)) + 1e-12))
    return dict(rho=rho, coh=coh, shape=shp, grad=np.concatenate([g.ravel() for g in gm]))

def crops(img, seed):
    rng = np.random.default_rng(seed); H, W = img.shape; out = []
    for _ in range(NCROP):
        y, x = rng.integers(8, H - T - 8), rng.integers(8, W - T - 8); out.append(img[y:y + T, x:x + T])
    return out

def compare(ts, os_):
    a, b = ts, os_
    hi = np.percentile(np.r_[a["grad"], b["grad"]], 99.5)
    ha, _ = np.histogram(a["grad"], 40, (0, hi)); hb, _ = np.histogram(b["grad"], 40, (0, hi))
    return dict(contrast_ratio=100 * min(a["rho"], b["rho"]) / max(a["rho"], b["rho"]), spectral_shape=100 * float(np.exp(-np.mean(np.abs(np.log(b["shape"] / a["shape"]))))),
                isotropy=100 * min(a["coh"], b["coh"]) / max(a["coh"], b["coh"]),
                texture=100 * float(np.minimum(ha / ha.sum(), hb / hb.sum()).sum()))

def ground_like(cs): return 100 * float(np.mean([tq.ground_score(c, filt) >= THR for c in cs]))

# ---- one cross-validation fold
sr_cache = {}
def offshelf(lr):
    if "m" not in sr_cache:
        m = cv2.dnn_superres.DnnSuperResImpl_create(); m.readModel(f"{MODELS}/EDSR_x4.pb"); m.setModel("edsr", 4); sr_cache["m"] = m
    lo, hi = np.percentile(lr, [0.1, 99.9]); u = np.clip((lr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return sr_cache["m"].upsample(cv2.cvtColor(u, cv2.COLOR_GRAY2BGR))[..., 0].astype(float) / 255 * (hi - lo) + lo

def synth(s, probs_rf, tiles, amp, gain, shape):
    base = up(s["lr_f"]); p = probs_rf(s); P = np.stack([np.full(base.shape, pk) for pk in p])
    t = dt.shape_tiles(tiles, *shape) if shape is not None else tiles
    return dt.synthesize(base, P, t, {k: a * gain for k, a in amp.items()}, PSF_HR, seed=s["sol"])[0][(base.shape[0] - N) // 2:(base.shape[0] - N) // 2 + N, (base.shape[1] - N) // 2:(base.shape[1] - N) // 2 + N]

def calibrate(kind, train, probs_rf, amp, held):
    tiles = library(kind, train, held); gain, shape = 1.5, None
    for it in range(2):                                   # contrast level, then spectral shape (training stops only)
        rr, ln = [], []
        for s in train:
            if len(s["idx"]) < 2: continue
            o = synth(s, probs_rf, tiles, amp, gain, shape); st = patch_stats(s["truth"]); so = patch_stats(crops(o, s["sol"]))
            rr.append(st["rho"] / so["rho"]); ln.append(np.log(so["shape"] / st["shape"]))
        if it == 0: gain *= float(np.clip(np.median(rr), 0.2, 5.0))
        else: shape = (f_ft, np.mean(ln, 0))
    return tiles, gain, shape

def fold(j):
    test = [s for s in stops if s["stop"] in j["test"]]; train = [s for s in stops if s["stop"] not in j["test"]]
    far = np.ones(len(X), bool)
    for s in test: far &= np.hypot(CL - (s["r"] + meta["top"]), CS - (s["c"] + meta["left"])) * meta["pixel_m"] > 60
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=1, random_state=0).fit(X[far], Y[far])
    def probs_rf(s):
        p = np.clip(rf.predict(s["feat"][None])[0], 0, None); return p / p.sum()
    amp = amp_by_class(train); out = {}
    cal = {k: calibrate(k, train, probs_rf, amp, test) for k in ("clean", "old", "hardware")}
    for s in test:
        if len(s["idx"]) < 2: continue
        st = patch_stats(s["truth"]); res = {}
        base = up(s["lr_f"]); cr = lambda a: a[(a.shape[0] - N) // 2:(a.shape[0] - N) // 2 + N, (a.shape[1] - N) // 2:(a.shape[1] - N) // 2 + N]
        imgs = {"bicubic, single raw frame": cr(up(s["lr_s"])), "EDSR x4, single raw frame": cr(offshelf(s["lr_s"])), "bicubic, our fused frames": cr(base)}
        for k, label in (("old", "ours with OLD texture library (mosaic, unfiltered)"), ("hardware", "CONTROL: library of rover hardware/shadows"), ("clean", "ours with CLEAN texture library")):
            tiles, gain, shape = cal[k]; imgs[label] = synth(s, probs_rf, tiles, amp, gain, shape)
        for name, img in imgs.items():
            cs = crops(img, s["sol"]); m = compare(st, patch_stats(cs)); m["ground_likeness"] = ground_like(cs); out.setdefault(name, []).append(dict(stop=s["stop"], **m))
        if DUMP:
            os.makedirs(DUMP, exist_ok=True); np.savez_compressed(f"{DUMP}/{s['stop']}.npz", **{k.split(",")[0].split(" (")[0].replace(" ", "_")[:40]: v for k, v in imgs.items()}, truth=np.stack(s["truth"][:9]))
    return dict(result=out, gains={k: float(v[1]) for k, v in cal.items()})

if __name__ == "__main__":
    names = [s["stop"] for s in stops]
    rng = np.random.default_rng(0); folds = []
    for rep in range(3):
        perm = rng.permutation(len(names))
        for f in range(6): folds.append(dict(id=f"rep{rep}_fold{f}", test=[names[i] for i in perm[f::6]]))
    res_path = OUTJ.replace(".json", ".jsonl")
    if os.path.exists(res_path): os.remove(res_path)
    runner.run(fold, folds, res_path, procs=int(os.environ.get("PROCS", 4)))
    rows = [json.loads(l) for l in open(res_path)]; rows = [r for r in rows if r.get("ok")]
    agg = {}
    for r in rows:
        for name, lst in r["result"].items():
            for m in lst: agg.setdefault(name, {}).setdefault(m["stop"], []).append(m)
    summary = {}
    for name, bystop in agg.items():
        per = {k: [float(np.mean([m[k] for m in ms])) for ms in bystop.values()] for k in ("contrast_ratio", "spectral_shape", "isotropy", "texture", "ground_likeness")}
        rb = np.random.default_rng(1)
        summary[name] = {k: dict(mean=float(np.mean(v)), lo=float(np.percentile(rb.choice(v, (2000, len(v))).mean(1), 2.5)), hi=float(np.percentile(rb.choice(v, (2000, len(v))).mean(1), 97.5)), n=len(v)) for k, v in per.items()}
    json.dump(dict(n_stops_scored=len(next(iter(agg.values()))), summary=summary), open(OUTJ, "w"), indent=1)
    print(f"\nScored on {len(next(iter(agg.values())))} rover stops (each held out in 3 random 6-fold splits; mean with 95% bootstrap CI over stops); percent, 100 = same as real clean rover ground")
    print(f"{'method':58s} {'contrast':>9s} {'spectrum':>9s} {'isotropy':>9s} {'texture':>9s} {'ground-like':>11s}")
    for name, s in summary.items():
        print(f"{name:58s} " + " ".join(f"{s[k]['mean']:8.1f}%" for k in ("contrast_ratio", "spectral_shape", "isotropy", "texture")) + f" {s['ground_likeness']['mean']:10.1f}%")

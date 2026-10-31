"""Score the measured layer and the rover-texture detail layer against HELD-OUT rover stops.

Usage: python scripts/evaluate_hero.py <hero_dir> <rover_truth_dir> <sr_models_dir> <cells_npz> <ortho_jp2> <out_json>
"""
import sys, os, json, glob
import numpy as np, cv2, glymur
from scipy import ndimage as ndi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import detail as dt, scoring as sc, groundtruth as gt, surface_classifier as scl, multiframe as mf
from sklearn.ensemble import RandomForestRegressor

HERO, TRUTH, MODELS, CELLS, ORTHO, OUTJ = sys.argv[1:7]
meta = json.load(open(f"{HERO}/measured_meta.json"))
fused = np.load(f"{HERO}/measured_fused.npy").astype(np.float64)
single = np.load(f"{HERO}/measured_single.npy").astype(np.float64)
stack = np.load(f"{HERO}/measured_stack.npy").astype(np.float64)
psf = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data_products", "psf_calibration.json")))["sigma_px_median"]
SC = 4                                          # detail grid = HiRISE ortho px / 4  (6.31 cm)
PSF_HR = psf * SC
FINE = (0.06, 0.45); COARSE = (0.012, 0.03)    # cycles per detail px: fine = below HiRISE's reach; coarse = 2-5 m scales both see
stops = []
for f in sorted(glob.glob(f"{TRUTH}/*.npz")):
    d = np.load(f); L0, S0 = gt.latlon_to_ortho(float(d["lat"]), float(d["lon"]))
    r, c = L0 - meta["top"], S0 - meta["left"]
    if 40 < r < meta["size"] - 40 and 40 < c < meta["size"] - 40:
        stops.append(dict(stop=os.path.basename(f)[:-4], sol=int(d["sol"]), r=r, c=c, z=d["z"], cls=d["cls"]))
stops.sort(key=lambda s: s["sol"])
held = [s for i, s in enumerate(stops) if i % 3 == 1]; train = [s for i, s in enumerate(stops) if i % 3 != 1]
print(f"rover stops in hero area: {len(stops)} -> train {len(train)}, held-out {len(held)}")
def to_grid(z):
    k = 0.05 / (meta["pixel_m"] / SC)
    return ndi.zoom(np.nan_to_num(z), k, order=1), ndi.zoom(np.isfinite(z).astype(float), k, order=1) > 0.99
def truth_grid(s):
    z, v = to_grid(s["z"]); c = ndi.zoom(np.nan_to_num(s["cls"], nan=-1), 0.05 / (meta["pixel_m"] / SC), order=0)
    z = np.where(v, z, np.nan); c = np.where(v & (c >= 0), c, np.nan)
    return z, c
# ---- texture library + amplitude ratios from TRAINING stops only
tiles = {k: [] for k in range(4)}; ratios = {k: [] for k in range(4)}; kappas = []
for s in train:
    z, c = truth_grid(s)
    for k, v in dt.extract_patches(z, c, PSF_HR, 64, min_frac=0.6).items(): tiles[k] += v
    kappas.append(dt.contrast_attenuation(z, PSF_HR, SC))
    fine, coarse = dt.bands(z, PSF_HR)
    for k in range(4):
        m = (c == k)
        if m.sum() > 2000: ratios[k].append(np.nanstd(fine[m]) / (np.nanstd(coarse[m]) + 1e-12))
kappa = float(np.median(kappas))
amp = {k: float(np.median(v)) / kappa if v else 0.0 for k, v in ratios.items()}
for k in range(4):                                  # classes with too few clean rover examples borrow the closest class
    if len(tiles[k]) < 4: tiles[k] = tiles[k] + tiles[dt.FALLBACK[k]]
    if amp[k] == 0.0: amp[k] = amp[dt.FALLBACK[k]]
print(f"HiRISE coarse-contrast attenuation (train stops): {kappa:.2f} -> amplitude x{1/kappa:.2f}")
print("texture library (train stops):", {dt.CLASSES[k]: len(v) for k, v in tiles.items()}, "| fine/coarse ratio:", {dt.CLASSES[k]: round(a, 2) for k, a in amp.items()})
# ---- surface classifier trained away from held-out stops (>= 60 m), predicts memberships at held-out stops
cz = np.load(CELLS); X, Y, CL, CS = cz["X"], cz["Y"], cz["L"], cz["S"]
far = np.ones(len(X), bool)
for s in held:
    far &= np.hypot(CL - (s["r"] + meta["top"]), CS - (s["c"] + meta["left"])) * meta["pixel_m"] > 60
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(X[far], Y[far])
J = glymur.Jp2k(ORTHO)
def probs_at(s):
    L, S = int(s["r"] + meta["top"]), int(s["c"] + meta["left"])
    H1, H2 = 32, 96
    img = np.asarray(J[L - H2:L + H2, S - H2:S + H2]).astype(float)
    x = np.r_[scl.features(img[H2 - H1:H2 + H1, H2 - H1:H2 + H1]), scl.features(img)]
    p = np.clip(rf.predict(x[None])[0], 0, None); return p / p.sum()
# ---- off-the-shelf upscalers
sr = {}
for name, fn in (("edsr", "EDSR_x4.pb"), ("espcn", "ESPCN_x4.pb"), ("fsrcnn", "FSRCNN_x4.pb")):
    m = cv2.dnn_superres.DnnSuperResImpl_create(); m.readModel(f"{MODELS}/{fn}"); m.setModel(name, 4); sr[name] = m
def offshelf(name, lr):
    lo, hi = np.percentile(lr, [0.1, 99.9]); u = np.clip((lr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    out = sr[name].upsample(cv2.cvtColor(u, cv2.COLOR_GRAY2BGR))[..., 0].astype(float)
    return out / 255 * (hi - lo) + lo
up = lambda a, o=3: ndi.zoom(a, SC, order=o, grid_mode=True, mode="nearest")
def stop_window(s):
    z, c = truth_grid(s)
    n = z.shape[0]; h = n // SC // 2 + 2
    r, cc = int(round(s["r"])), int(round(s["c"]))
    return z, c, n, fused[r - h:r + h, cc - h:cc + h], single[r - h:r + h, cc - h:cc + h]
def crop_to(a, n):
    return a[(a.shape[0] - n) // 2:(a.shape[0] - n) // 2 + n, (a.shape[1] - n) // 2:(a.shape[1] - n) // 2 + n]
# ---- overall gain calibrated on TRAINING stops only (leave-one-stop-out texture library); held-out never used
per_tiles = {s["stop"]: dt.extract_patches(*truth_grid(s), PSF_HR, 64, min_frac=0.6) for s in train}
def library(exclude=None):
    T = {k: [] for k in range(4)}
    for st, d in per_tiles.items():
        if st == exclude: continue
        for k, v in d.items(): T[k] += v
    for k in range(4):
        if len(T[k]) < 4: T[k] = T[k] + T[dt.FALLBACK[k]]
    return T
def train_curves(shape=None):
    """Leave-one-stop-out synthesis on TRAINING stops -> per-frequency ln(P_out/P_truth)."""
    cs = []
    for s in train:
        z, c, n, lr_f, _ = stop_window(s)
        base = up(lr_f); p = probs_at(s); P = np.stack([np.full(base.shape, pk) for pk in p])
        lib = library(s["stop"]); lib = dt.shape_tiles(lib, *shape) if shape else lib
        o, _ = dt.synthesize(base, P, lib, amp, PSF_HR, seed=s["sol"])
        f, po = sc._norm_psd(np.where(np.isfinite(z), crop_to(o, n), np.nan), COARSE); _, pt = sc._norm_psd(z, COARSE)
        cs.append(np.log(po / pt))
    return f, np.median(cs, 0)
f_tr, c0 = train_curves()
sel = (f_tr >= FINE[0]) & (f_tr <= FINE[1])
gain = float(np.exp(-np.mean(c0[sel]) / 2)); amp = {k: a * gain for k, a in amp.items()}
shape_curve = np.where(f_tr >= FINE[0] * 0.8, c0 - np.mean(c0[sel]), 0.0)     # residual shape after the level fix
f_tr, c1 = train_curves((f_tr, shape_curve))
print(f"training-stop calibration: level ln {np.mean(c0[sel]):+.2f} -> amplitude x{gain:.2f}; residual shape error {np.mean(np.abs(c0[sel]-np.mean(c0[sel]))):.2f} -> {np.mean(np.abs(c1[sel])):.2f} after shaping (train)")
tiles = dt.shape_tiles(library(), f_tr, shape_curve)
res = {}; curves = {}
for s in held:
    z, c, n, lr_f, lr_s = stop_window(s)
    crop = lambda a: crop_to(a, n)
    p = probs_at(s)
    base = up(lr_f); P = np.stack([np.full(base.shape, pk) for pk in p])
    ours, det = dt.synthesize(base, P, tiles, amp, PSF_HR, seed=s["sol"])
    lanczos = lambda a: cv2.resize(a.astype(np.float32), None, fx=SC, fy=SC, interpolation=cv2.INTER_LANCZOS4).astype(float)
    methods = {"bicubic (raw single frame)": up(lr_s), "lanczos (raw single frame)": lanczos(lr_s),
               "EDSR x4 (raw single frame)": offshelf("edsr", lr_s), "ESPCN x4 (raw single frame)": offshelf("espcn", lr_s),
               "FSRCNN x4 (raw single frame)": offshelf("fsrcnn", lr_s), "EDSR x4 (our fused frames)": offshelf("edsr", lr_f),
               "bicubic (our fused frames)": base, "OURS: fused + rover-texture detail": ours}
    if os.environ.get("DUMP_DIR"):                       # optional: save the images for figures
        os.makedirs(os.environ["DUMP_DIR"], exist_ok=True)
        np.savez_compressed(f"{os.environ['DUMP_DIR']}/{s['stop']}.npz", truth=z, raw=crop(up(lr_s, 0)), bicubic=crop(up(lr_s)),
                            edsr=crop(methods["EDSR x4 (raw single frame)"]), fused=crop(base), ours=crop(ours), sol=s["sol"], probs=p)
    for k, img in methods.items():
        o = crop(img)
        lr = lr_f if "fused" in k or "OURS" in k else lr_s
        res.setdefault(k, []).append((sc.spectral_match(o, z, FINE, COARSE), sc.texture_match(o, z, PSF_HR), sc.consistency(img, lr, SC, PSF_HR)))
        fo, po = sc._norm_psd(np.where(np.isfinite(z), o, np.nan), COARSE); _, pt = sc._norm_psd(z, COARSE)
        curves.setdefault(k, []).append(np.log(po / pt))
    print(f"held-out {s['stop']} (sol {s['sol']}): memberships " + ", ".join(f"{dt.CLASSES[i]} {100*p[i]:.0f}%" for i in range(4)), flush=True)
# ---- measured layer: leave-one-out noise (predict each held-out HiRISE frame)
loo = []
for k in range(1, stack.shape[0]):
    a = stack[k]; others = np.delete(stack, k, 0)
    m = np.isfinite(a) & np.isfinite(stack[0])
    avg = np.nanmean(others, 0)
    e1 = np.sqrt(np.nanmean((mf.hp(np.nan_to_num(stack[0])) - mf.hp(np.nan_to_num(a)))[m] ** 2))
    e2 = np.sqrt(np.nanmean((mf.hp(np.nan_to_num(avg)) - mf.hp(np.nan_to_num(a)))[m] ** 2))
    loo.append(100 * (1 - e2 / e1))
summary = {k: dict(spectral_match_pct=float(np.mean([v[0] for v in vs])), texture_match_pct=float(np.mean([v[1] for v in vs])),
                   consistency_pct=float(np.mean([v[2] for v in vs]))) for k, vs in res.items()}
out = dict(n_held_out=len(held), n_train=len(train), psf_sigma_hirise_px=psf, amp_ratio=amp, contrast_attenuation=kappa, train_gain=gain, shape_curve=dict(f=f_tr.tolist(), ln=shape_curve.tolist()),
           measured_layer=dict(noise_reduction_vs_single_frame_pct=float(np.mean(loo)), per_frame=[float(x) for x in loo], frames=stack.shape[0]),
           detail_layer=summary)
json.dump(out, open(OUTJ, "w"), indent=1)
print(f"\nMEASURED layer: prediction error of held-out HiRISE frames {np.mean(loo):.1f}% lower than a single raw frame ({stack.shape[0]} frames)")
print(f"\nDETAIL scores vs {len(held)} held-out rover stops (percent; 100 = identical to rover photo statistics):")
print(f"  {'method':38s} {'spectral':>9s} {'texture':>9s} {'consistency':>12s}")
for k, v in summary.items():
    print(f"  {k:38s} {v['spectral_match_pct']:8.1f}% {v['texture_match_pct']:8.1f}% {v['consistency_pct']:11.1f}%")
print("\nmean ln(P_out/P_truth) by frequency (cycles/detail px) - 0 is perfect, negative = too little detail:")
fo = sc._norm_psd(np.zeros((64, 64)) + np.random.default_rng(0).standard_normal((64, 64)), COARSE)[0]
for k in ("bicubic (our fused frames)", "OURS: fused + rover-texture detail"):
    cm = np.mean(curves[k], 0); print(f"  {k[:30]:30s} " + " ".join(f"{f:.2f}:{c:+.1f}" for f, c in zip(fo[::4], cm[::4])))

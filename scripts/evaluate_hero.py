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
to_grid = lambda z: ndi.zoom(np.nan_to_num(z), 0.05 / (meta["pixel_m"] / SC), order=1), ndi.zoom(np.isfinite(z).astype(float), 0.05 / (meta["pixel_m"] / SC), order=1) > 0.99
def truth_grid(s):
    z, v = to_grid(s["z"]); c = ndi.zoom(np.nan_to_num(s["cls"], nan=-1), 0.05 / (meta["pixel_m"] / SC), order=0)
    z = np.where(v, z, np.nan); c = np.where(v & (c >= 0), c, np.nan)
    return z, c
# ---- texture library + amplitude ratios from TRAINING stops only
tiles = {k: [] for k in range(4)}; ratios = {k: [] for k in range(4)}
for s in train:
    z, c = truth_grid(s)
    for k, v in dt.extract_patches(z, c, PSF_HR, 64).items(): tiles[k] += v
    fine, coarse = dt.bands(z, PSF_HR)
    for k in range(4):
        m = (c == k)
        if m.sum() > 2000: ratios[k].append(np.nanstd(fine[m]) / (np.nanstd(coarse[m]) + 1e-12))
amp = {k: float(np.median(v)) if v else 0.0 for k, v in ratios.items()}
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
res = {}
for s in held:
    z, c = truth_grid(s)
    n = z.shape[0]; h = n // SC // 2 + 2
    r, cc = int(round(s["r"])), int(round(s["c"]))
    lr_f, lr_s = fused[r - h:r + h, cc - h:cc + h], single[r - h:r + h, cc - h:cc + h]
    up = lambda a, o=3: ndi.zoom(a, SC, order=o, grid_mode=True, mode="nearest")
    crop = lambda a: a[(a.shape[0] - n) // 2:(a.shape[0] - n) // 2 + n, (a.shape[1] - n) // 2:(a.shape[1] - n) // 2 + n]
    p = probs_at(s)
    base = up(lr_f); P = np.stack([np.full(base.shape, pk) for pk in p])
    ours, det = dt.synthesize(base, P, tiles, amp, PSF_HR, seed=s["sol"])
    lanczos = lambda a: cv2.resize(a.astype(np.float32), None, fx=SC, fy=SC, interpolation=cv2.INTER_LANCZOS4).astype(float)
    methods = {"bicubic (raw single frame)": up(lr_s), "lanczos (raw single frame)": lanczos(lr_s),
               "EDSR x4 (raw single frame)": offshelf("edsr", lr_s), "ESPCN x4 (raw single frame)": offshelf("espcn", lr_s),
               "FSRCNN x4 (raw single frame)": offshelf("fsrcnn", lr_s), "EDSR x4 (our fused frames)": offshelf("edsr", lr_f),
               "bicubic (our fused frames)": base, "OURS: fused + rover-texture detail": ours}
    for k, img in methods.items():
        o = crop(img)
        lr = lr_f if "fused" in k or "OURS" in k else lr_s
        res.setdefault(k, []).append((sc.spectral_match(o, z, FINE, COARSE), sc.texture_match(o, z, PSF_HR), sc.consistency(img, lr, SC, PSF_HR)))
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
out = dict(n_held_out=len(held), n_train=len(train), psf_sigma_hirise_px=psf, amp_ratio=amp,
           measured_layer=dict(noise_reduction_vs_single_frame_pct=float(np.mean(loo)), per_frame=[float(x) for x in loo], frames=stack.shape[0]),
           detail_layer=summary)
json.dump(out, open(OUTJ, "w"), indent=1)
print(f"\nMEASURED layer: prediction error of held-out HiRISE frames {np.mean(loo):.1f}% lower than a single raw frame ({stack.shape[0]} frames)")
print(f"\nDETAIL scores vs {len(held)} held-out rover stops (percent; 100 = identical to rover photo statistics):")
print(f"  {'method':38s} {'spectral':>9s} {'texture':>9s} {'consistency':>12s}")
for k, v in summary.items():
    print(f"  {k:38s} {v['spectral_match_pct']:8.1f}% {v['texture_match_pct']:8.1f}% {v['consistency_pct']:11.1f}%")

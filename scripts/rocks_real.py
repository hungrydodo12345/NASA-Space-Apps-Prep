"""Real-data rock counts by size versus the rover's own photos (the comparison Tao and Muller made for GPT).

For each rover stop: rocks are detected with one fixed blob detector in (a) the raw HiRISE frame upsampled,
(b) shift-and-average, (c) single-frame deconvolution, (d) multi-frame restoration, and (e) the rover's top-down mosaic,
counting only inside clean ground (rover hardware / photo seams removed by the texture-quality filter). For the
restoration, a rock also counts as 'repeatable' only if the two independent half-set restorations both show it.
Usage: python scripts/rocks_real.py <restored_dir> <rover_truth_dir> <texture_filter.json> <out_json>
"""
import sys, os, glob, json
import numpy as np
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from pipeline import texture_quality as tq
sys.argv_saved = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[2]]
from experiments import exp3_rocks as E                     # same detector and size bins as the simulation
sys.argv = sys.argv_saved

RD, TRUTH, FILT, OUT = sys.argv[1:5]
filt = tq.load_filter(FILT); PX = E.PX


def clean_mask(z, valid, win=32, stride=16, thr=0.85):
    m = np.zeros_like(valid)
    for y in range(0, z.shape[0] - win + 1, stride):
        for x in range(0, z.shape[1] - win + 1, stride):
            if valid[y:y + win, x:x + win].mean() < 0.98: continue
            if tq.ground_score(z[y:y + win, x:x + win], filt) >= thr: m[y:y + win, x:x + win] = True
    return m & valid


def counts(dets, mask):
    c = {D: 0 for D in E.SIZES}
    for y, x, D, _ in dets:
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[int(y), int(x)]: c[D] += 1
    return c


def repeatable(d0, d1):
    keep = []
    for y, x, D, s in d0:
        if any(np.hypot(y - y1, x - x1) <= max(2.0, 0.3 * D / PX) and abs(np.log(D1 / D)) <= np.log(2.01) for y1, x1, D1, _ in d1): keep.append((y, x, D, s))
    return keep


tot = {}; area = 0.0; per_stop = []
for f in sorted(glob.glob(f"{RD}/halves_*.npz")):
    stop = os.path.basename(f)[7:-4]; full = f"{RD}/{stop}.npz"; tf = f"{TRUTH}/{stop}.npz"
    if not (os.path.exists(full) and os.path.exists(tf)): continue
    h = np.load(f); fr = np.load(full); z5 = np.load(tf)["z"].astype(float)
    k = 0.05 / (PX / 100)
    zt = ndi.zoom(np.nan_to_num(z5), k, order=1); vt = ndi.zoom(np.isfinite(z5).astype(float), k, order=1) > 0.99
    zt = np.where(vt, zt, np.nan)
    cm = clean_mask(np.nan_to_num(zt, nan=0.0), ndi.binary_erosion(vt, iterations=4))
    n = fr["restored_multi"].shape[0]; t = zt.shape[0]; o = (n - t) // 2
    mask = np.zeros((n, n), bool); mask[o:o + t, o:o + t] = cm
    a_m2 = mask.sum() * (PX / 100) ** 2
    if a_m2 < 2.0: continue
    area += a_m2
    imgs = {"raw HiRISE (bicubic)": fr["bicubic"], "shift-and-average": fr["average"], "single-frame deconvolution": fr["deconv_1frame"],
            "multi-frame restoration": fr["restored_multi"]}
    row = dict(stop=stop, clean_area_m2=a_m2)
    for name, img in imgs.items():
        c = counts(E.detect(img.astype(float)), mask); row[name] = c
        for D in E.SIZES: tot.setdefault(name, {}).setdefault(D, 0); tot[name][D] += c[D]
    rep = repeatable(E.detect(h["restored_0"].astype(float)), E.detect(h["restored_1"].astype(float)))
    c = counts(rep, mask); row["multi-frame restoration, repeatable in both halves"] = c
    for D in E.SIZES: tot.setdefault("multi-frame restoration, repeatable in both halves", {}).setdefault(D, 0); tot["multi-frame restoration, repeatable in both halves"][D] += c[D]
    # rover truth: same detector on the rover mosaic (resampled to the same 6.31 cm grid), clean area only
    tz = np.nan_to_num(zt, nan=float(np.nanmean(zt)))
    c = counts(E.detect(tz), cm); row["rover photos (truth)"] = c
    for D in E.SIZES: tot.setdefault("rover photos (truth)", {}).setdefault(D, 0); tot["rover photos (truth)"][D] += c[D]
    per_stop.append(row)

truth = tot.get("rover photos (truth)", {})
summary = {name: {str(D): dict(count=v[D], per_100m2=100 * v[D] / max(area, 1e-9), share_of_truth_pct=(100 * v[D] / truth[D] if truth.get(D) else None)) for D in E.SIZES} for name, v in tot.items()}
json.dump(dict(stops=len(per_stop), clean_area_m2=area, summary=summary, per_stop=per_stop), open(OUT, "w"), indent=1)
print(f"{len(per_stop)} stops, clean rover-verified ground {area:.0f} m2")
print(f"{'rocks counted (share of what the rover photos show)':52s} " + " ".join(f"{D:>6d}cm" for D in E.SIZES))
for name, v in tot.items():
    if name == "rover photos (truth)": print(f"  {name:50s} " + " ".join(f"{v[D]:7d} " for D in E.SIZES) + "  <- counts"); continue
    print(f"  {name:50s} " + " ".join((f"{100*v[D]/truth[D]:7.0f}%" if truth.get(D) else f"{v[D]:7d} ") for D in E.SIZES))

"""Score restorations against a placed centimetre-scale truth patch (Ingenuity nadir mosaic) and search the pipeline settings for the minimum error.

Truth and restorations live on the same 6.31 cm grid. Per band-pass scale (difference of Gaussians, sigma pairs in HR px) the zero-mean correlation between a restoration and the truth is
computed over the valid patch with the backshell itself excluded (it dominates the bright tail; also reported with it). Residual placement error (rotation 5 deg, translation 0.3 m) would
penalise sharper methods more, so each score is also given with a placement tolerance: the best correlation over shifts of +-8 HR px (+-0.5 m) applied equally to every method.
Primary metric: correlation in the 0.5-2 m band (sigma 6/24 HR px). Illumination differs between the truth (midday) and HiRISE (afternoon), so only the bright/dark pattern of objects and ground albedo is
comparable; the metric is a ceiling-limited ranking tool, not an absolute accuracy.
Search: settings are chosen on one spatial half of the patch and scored on the other (two folds); block-bootstrap (16 x 16 HR px blocks) intervals for differences vs the UCL re-implementation at its own best setting.
Usage: python benchmark/score_truth.py <variants_npz> <truth_npz> <out_json> [backshell_x backshell_y]"""
import sys, json, numpy as np
from scipy import ndimage as ndi
VAR, TR, OUT = sys.argv[1:4]; v = np.load(VAR); t = np.load(TR); T = t["truth"].astype(np.float64); valid = t["valid"]; N = T.shape[0]
BANDS = {"2-1 m": (16, 32), "1 m-50 cm": (8, 16), "50-25 cm": (4, 8), "0.5-2 m (primary)": (6, 24)}
yy, xx = np.mgrid[0:N, 0:N]
bs = np.array([N / 2 + t["params"][2] / (0.2524 / 4), N / 2 + t["params"][3] / (0.2524 / 4)])                                  # backshell centre in HR px (placement offset)
nb = ~(np.hypot(xx - bs[0], yy - bs[1]) < 3.2 / (0.2524 / 4)); mask = valid & ndi.binary_erosion(valid, iterations=10) & nb
band = lambda a, lo, hi: ndi.gaussian_filter(a, lo) - ndi.gaussian_filter(a, hi)
Tn = np.where(valid, T, np.nanmean(T)); TB = {k: band(Tn, *b) for k, b in BANDS.items()}
def corr(a, b, m): a = a[m] - a[m].mean(); b = b[m] - b[m].mean(); return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))
def score(X, key, m, tol=False):
    XB = band(X.astype(np.float64), *BANDS[key]); tb = TB[key]
    if not tol: return corr(XB, tb, m)
    best = -2
    for dy in range(-8, 9, 2):
        for dx in range(-8, 9, 2):
            sh = np.roll(np.roll(tb, dy, 0), dx, 1); mm = m & np.roll(np.roll(m, dy, 0), dx, 1); best = max(best, corr(XB, sh, mm))
    return best
names = [k for k in v.files]; res = {}
for k in names:
    X = v[k]; res[k] = {b: dict(zero=score(X, b, mask), tol=score(X, b, mask, True)) for b in BANDS}; res[k]["primary_with_backshell"] = score(X, "0.5-2 m (primary)", valid & ndi.binary_erosion(valid, iterations=10))
fam = lambda k: k.split("__")[0]; lam = lambda k: float(k.split("__")[1]) if "__" in k else 0.0
families = sorted({fam(k) for k in names}, key=lambda f: ["raw", "gpt", "gpt+blur", "full-equal", "full", "full-b0.8", "full-b1.2"].index(f) if f in ["raw", "gpt", "gpt+blur", "full-equal", "full", "full-b0.8", "full-b1.2"] else 9)
print("PRIMARY (0.5-2 m band correlation with the truth; zero shift / with +-0.5 m tolerance), backshell excluded")
print(f"{'method':12s} " + " ".join(f"lam={l:<6}" for l in (0.002, 0.005, 0.015, 0.04, 0.1)))
for f in families:
    if f == "raw": print(f"{'raw':12s} {res['raw']['0.5-2 m (primary)']['zero']:.3f} / {res['raw']['0.5-2 m (primary)']['tol']:.3f}"); continue
    print(f"{f:12s} " + " ".join(f"{res[f'{f}__{l}']['0.5-2 m (primary)']['zero']:.3f}/{res[f'{f}__{l}']['0.5-2 m (primary)']['tol']:.3f}" for l in (0.002, 0.005, 0.015, 0.04, 0.1)))
# two-fold search: choose the setting on half A, score on half B (and vice versa); halves split by the valid bounding box along x
ys, xs = np.nonzero(mask); cut = np.median(xs); halves = {"left": mask & (xx < cut), "right": mask & (xx >= cut)}
def best_in_family(f, half, tol):
    ks = [k for k in names if fam(k) == f]; sc = {k: score(v[k], "0.5-2 m (primary)", half, tol) for k in ks}; kb = max(sc, key=sc.get); return kb, sc[kb]
out = dict(settings=res, folds={})
for tol in (False, True):
    for a, b in (("left", "right"), ("right", "left")):
        for f in families:
            if f == "raw": ke, sc = "raw", score(v["raw"], "0.5-2 m (primary)", halves[b], tol); out["folds"][f"{'tol' if tol else 'zero'}/{a}->{b}/{f}"] = dict(chosen="raw", heldout=sc); continue
            kb, _ = best_in_family(f, halves[a], tol); out["folds"][f"{'tol' if tol else 'zero'}/{a}->{b}/{f}"] = dict(chosen=kb, heldout=score(v[kb], "0.5-2 m (primary)", halves[b], tol))
for tol in ("zero", "tol"):
    print(f"\nHELD-OUT score of each family (setting chosen on the other half), {tol}:")
    for f in families:
        s = [out["folds"][f"{tol}/{a}->{b}/{f}"] for a, b in (("left", "right"), ("right", "left"))]; print(f"  {f:12s} mean {np.mean([x['heldout'] for x in s]):.3f}   (chosen: {s[0]['chosen']}, {s[1]['chosen']})")
# block bootstrap of the difference "family best (chosen on all) - UCL best" on the primary band
bestall = {f: (max([k for k in names if fam(k) == f], key=lambda k: res[k]["0.5-2 m (primary)"]["tol"]) if f != "raw" else "raw") for f in families}
bl = 16; rng = np.random.default_rng(0); nbk = N // bl; ub = band(Tn, *BANDS["0.5-2 m (primary)"])
def blockstat(X): XB = band(X.astype(np.float64), *BANDS["0.5-2 m (primary)"]); return XB, ub
out["best_all"] = bestall; out["paired_vs_gpt"] = {}
ref = blockstat(v[bestall["gpt"]]); 
for f in families:
    if f in ("gpt",): continue
    cur = blockstat(v[bestall[f]]); diffs = []
    for _ in range(600):
        sel = rng.integers(0, nbk, (nbk * nbk // 4, 2)); m = np.zeros_like(mask)
        for by, bx in sel: m[by * bl:(by + 1) * bl, bx * bl:(bx + 1) * bl] = True
        m &= mask
        if m.sum() < 800: continue
        diffs.append(corr(cur[0], cur[1], m) - corr(ref[0], ref[1], m))
    d = np.array(diffs); out["paired_vs_gpt"][f] = dict(setting=bestall[f], mean=float(d.mean()), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], frac_positive=float((d > 0).mean()))
print("\nBEST setting per family (chosen on the whole patch, tolerance metric); paired block-bootstrap difference vs UCL best (positive = better than UCL):")
for f in families:
    if f == "gpt": print(f"  {f:12s} setting {bestall[f]}  (reference)"); continue
    p = out["paired_vs_gpt"][f]; print(f"  {f:12s} setting {p['setting']:20s} {p['mean']:+.3f}  95% CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  better in {100*p['frac_positive']:.0f}% of resamples" if f != "raw" else f"  raw          {p['mean']:+.3f}  95% CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]")
json.dump(out, open(OUT, "w"), indent=1)

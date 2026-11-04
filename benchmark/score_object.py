"""Object-centred truth test: how faithfully does each restoration reproduce a bright object of known shape (the backshell) that the Ingenuity mosaic resolves at 3 cm?
The truth's bright object (above its local threshold) is rendered as a mask on the 6.31 cm grid and compared with each restoration inside a disc around the backshell:
  - shape correlation of the (zero-mean) restoration with the truth image, in bands from 1-4 m down to 25-50 cm, at zero shift and with +-0.25 m tolerance;
  - edge sharpness and ringing: the profile of the restoration across the object's boundary, averaged over boundary points (distance from the truth boundary vs normalised brightness).
Usage: python benchmark/score_object.py <variants_npz> <truth_npz> <out_json> <radius_m>"""
import sys, json, numpy as np
from scipy import ndimage as ndi
VAR, TR, OUT, RAD = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]); HR = 0.2524 / 4
v = np.load(VAR); t = np.load(TR); T = t["truth"].astype(np.float64); valid = t["valid"]; N = T.shape[0]; yy, xx = np.mgrid[0:N, 0:N]
bs = np.array([N / 2 + t["params"][2] / HR, N / 2 + t["params"][3] / HR]); disc = (np.hypot(xx - bs[0], yy - bs[1]) < RAD / HR) & valid & ndi.binary_erosion(valid, iterations=6)
Tn = np.where(valid, T, np.nanmean(T)); thr = np.percentile(Tn[disc], 90); obj = (ndi.gaussian_filter(Tn, 1.0) > thr) & disc; obj = ndi.binary_opening(obj, iterations=2); lab, nl = ndi.label(obj); sizes = ndi.sum(obj, lab, range(1, nl + 1)); obj = lab == (1 + int(np.argmax(sizes)))
print("backshell object in the truth: %.1f m^2 (%.1f m equivalent diameter)" % (obj.sum() * HR ** 2, 2 * np.sqrt(obj.sum() * HR ** 2 / np.pi)))
band = lambda a, lo, hi: ndi.gaussian_filter(a, lo) - ndi.gaussian_filter(a, hi); BANDS = {"4-1 m": (6, 48), "2-1 m": (16, 32), "1 m-50 cm": (8, 16), "50-25 cm": (4, 8)}
def corr(a, b, m): a = a[m] - a[m].mean(); b = b[m] - b[m].mean(); return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))
TB = {k: band(Tn, *b) for k, b in BANDS.items()}
def sc(X, k, tol):
    XB = band(X.astype(np.float64), *BANDS[k]); best = -2
    for dy in (range(-4, 5, 2) if tol else [0]):
        for dx in (range(-4, 5, 2) if tol else [0]): best = max(best, corr(XB, np.roll(np.roll(TB[k], dy, 0), dx, 1), disc & np.roll(np.roll(disc, dy, 0), dx, 1)))
    return best
dist = ndi.distance_transform_edt(~obj) - ndi.distance_transform_edt(obj); edge = (np.abs(dist) < 1.5)                                     # signed distance from the object boundary (px, + outside)
def profile(X):
    X = X.astype(np.float64); inside = np.median(X[(dist < -4) & (dist > -12) & disc]); outside = np.median(X[(dist > 8) & (dist < 24) & disc]); n = (X - outside) / (inside - outside + 1e-9)
    bins = np.arange(-12, 25, 3); p = [float(np.mean(n[(dist >= a) & (dist < a + 3) & disc])) for a in bins]; return bins, p, float(max(0.0, 1 - np.min(n[(dist > 3) & (dist < 24) & disc]) if False else 0))
res = {}
for k in v.files:
    X = v[k]; r = {b: dict(zero=sc(X, b, False), tol=sc(X, b, True)) for b in BANDS}; bins, p, _ = profile(X)
    # edge width: distance over which the normalised profile goes from 0.9 to 0.1 (10-90 %), from the bin centres by interpolation
    pp = np.array(p); cx_ = bins + 1.5; hi = np.interp(0.9, pp[::-1], cx_[::-1]) if pp.min() < 0.9 < pp.max() else np.nan; lo = np.interp(0.1, pp[::-1], cx_[::-1]) if pp.min() < 0.1 < pp.max() else np.nan
    r["edge_10_90_cm"] = float(abs(lo - hi) * HR * 100) if np.isfinite(hi) and np.isfinite(lo) else None; r["overshoot_pct"] = float(100 * max(0.0, pp.max() - 1.0)); r["undershoot_pct"] = float(100 * max(0.0, -pp[cx_ > 4].min())); r["profile"] = p; res[k] = r
fam = lambda k: k.split("__")[0]
print(f"{'setting':20s} " + " ".join(f"{b:>10s}" for b in BANDS) + "   edge 10-90 cm  overshoot  undershoot")
for k in ["raw"] + sorted([x for x in v.files if x != "raw"], key=lambda s: (["gpt", "gpt+blur", "full-equal", "full", "full-b0.8", "full-b1.2"].index(fam(s)), float(s.split("__")[1]))):
    r = res[k]; print(f"{k:20s} " + " ".join(f"{r[b]['tol']:10.3f}" for b in BANDS) + f"   {r['edge_10_90_cm'] if r['edge_10_90_cm'] is None else round(r['edge_10_90_cm'])!s:>10}  {r['overshoot_pct']:8.0f}%  {r['undershoot_pct']:8.0f}%")
json.dump(res, open(OUT, "w"), indent=1)

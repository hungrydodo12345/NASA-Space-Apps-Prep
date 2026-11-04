"""Placement robustness of the object-centred truth test. The truth placement (rotation, scale, translation) has uncertainty (rotation +-8 deg, scale +-3 %, translation +-0.4 m);
the truth is re-rendered for a grid of perturbed placements and the band correlations of selected settings recomputed, so the ranking of the methods can be seen to hold (or not) over
the whole plausible placement range. Usage: python benchmark/object_robust.py <variants_npz[,variants_hi_npz]> <mosaic_npz> <out_json> <radius_m>"""
import sys, json, itertools, numpy as np, cv2
from scipy import ndimage as ndi
VARS, MOS, OUT, RAD = sys.argv[1].split(","), sys.argv[2], sys.argv[3], float(sys.argv[4]); HR = 0.2524 / 4; N = 512; BX, BY = 151, 330
V = {}; [V.update({k: np.load(p)[k] for k in np.load(p).files}) for p in VARS]
m = np.load(MOS); M = np.nan_to_num(m["mosaic"], nan=1.0).astype(np.float32); Vd = np.isfinite(m["mosaic"])
def render(TH, SC, DX, DY):
    f = SC / HR; c, s = np.cos(np.radians(TH)), np.sin(np.radians(TH)); Rm = np.array([[f * c, f * s], [-f * s, f * c]]); t = np.array([N / 2 + DX / HR, N / 2 + DY / HR]) - Rm @ np.array([BX, BY]); W = np.hstack([Rm, t[:, None]])
    a = cv2.warpAffine(M, W, (N, N), flags=cv2.INTER_AREA); v = cv2.warpAffine(Vd.astype(np.float32), W, (N, N), flags=cv2.INTER_AREA) > 0.97; return np.where(v, a, np.nan), v
band = lambda a, lo, hi: ndi.gaussian_filter(a, lo) - ndi.gaussian_filter(a, hi); BANDS = {"2-1 m": (16, 32), "1 m-50 cm": (8, 16), "50-25 cm": (4, 8)}
def corr(a, b, m): a = a[m] - a[m].mean(); b = b[m] - b[m].mean(); return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))
fams = ["raw", "gpt", "gpt+blur", "full-equal", "full"]; lams = sorted({float(k.split("__")[1]) for k in V if "__" in k}); yy, xx = np.mgrid[0:N, 0:N]
XB = {(k, b): band(V[k].astype(np.float64), *BANDS[b]) for k in V for b in BANDS}
placements = list(itertools.product((203, 211, 219), (0.0282, 0.0290, 0.0298), ((-0.4, 1.1), (0.0, 1.1), (-0.8, 1.1), (-0.4, 0.7), (-0.4, 1.5)))); rows = []
for th, sc, (dx, dy) in placements:
    T, v = render(th, sc, dx, dy); Tn = np.where(v, T, np.nanmean(T)); TB = {b: band(Tn, *BANDS[b]) for b in BANDS}; bs = np.array([N / 2 + dx / HR, N / 2 + dy / HR])
    disc = (np.hypot(xx - bs[0], yy - bs[1]) < RAD / HR) & v & ndi.binary_erosion(v, iterations=6)
    if disc.sum() < 3000: continue
    r = dict(theta=th, scale=sc, dx=dx, dy=dy)
    for k in V:
        for b in BANDS:
            best = -2
            for ddy in range(-4, 5, 2):
                for ddx in range(-4, 5, 2): best = max(best, corr(XB[(k, b)], np.roll(np.roll(TB[b], ddy, 0), ddx, 1), disc & np.roll(np.roll(disc, ddy, 0), ddx, 1)))
            r[f"{k}|{b}"] = best
    rows.append(r)
print("placements scored:", len(rows))
def best_lam(f, b, rr): return max([k for k in V if k.split("__")[0] == f], key=lambda k: np.mean([x[f"{k}|{b}"] for x in rr])) if f != "raw" else "raw"
out = dict(n_placements=len(rows), summary={})
for b in BANDS:
    bl = {f: best_lam(f, b, rows) for f in fams}; line = {}
    for f in fams: line[f] = float(np.mean([r[f"{bl[f]}|{b}"] for r in rows]))
    d = {f: np.array([r[f"{bl[f]}|{b}"] - r[f"{bl['gpt']}|{b}"] for r in rows]) for f in fams if f != "gpt"}
    out["summary"][b] = dict(best_setting=bl, mean_corr=line, vs_gpt={f: dict(mean=float(x.mean()), min=float(x.min()), max=float(x.max()), frac_better=float((x > 0).mean())) for f, x in d.items()})
    print(f"\n{b}: mean correlation over {len(rows)} placements (best setting per family): " + ", ".join(f"{f} {line[f]:.3f} [{bl[f]}]" for f in fams))
    for f, x in d.items(): print(f"   {f:11s} minus UCL: mean {x.mean():+.3f}  range [{x.min():+.3f}, {x.max():+.3f}]  better in {100*(x>0).mean():.0f}% of placements")
json.dump(out, open(OUT, "w"), indent=1)

"""Final object-centred truth result at the nominal placement: error-versus-lambda curves (the minimum of each method), the best setting per method, and block-bootstrap intervals
for the differences between methods (resampling 10 x 10 HR px blocks of the 4 m disc around the backshell; +-0.25 m placement tolerance applied equally to all).
Usage: python benchmark/object_final.py <variants_npz,...> <truth_npz> <out_json> <radius_m>"""
import sys, json, numpy as np
from scipy import ndimage as ndi
VARS, TR, OUT, RAD = sys.argv[1].split(","), sys.argv[2], sys.argv[3], float(sys.argv[4]); HR = 0.2524 / 4
V = {}; [V.update({k: np.load(p)[k] for k in np.load(p).files}) for p in VARS]; t = np.load(TR); T = t["truth"].astype(np.float64); valid = t["valid"]; N = T.shape[0]; yy, xx = np.mgrid[0:N, 0:N]
bs = np.array([N / 2 + t["params"][2] / HR, N / 2 + t["params"][3] / HR]); disc = (np.hypot(xx - bs[0], yy - bs[1]) < RAD / HR) & valid & ndi.binary_erosion(valid, iterations=6)
Tn = np.where(valid, T, np.nanmean(T)); band = lambda a, lo, hi: ndi.gaussian_filter(a, lo) - ndi.gaussian_filter(a, hi); BANDS = {"2-1 m": (16, 32), "1 m-50 cm": (8, 16), "50-25 cm": (4, 8)}
SH = [(dy, dx) for dy in range(-4, 5, 2) for dx in range(-4, 5, 2)]
def corr(a, b, m): a = a[m] - a[m].mean(); b = b[m] - b[m].mean(); return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))
XB = {(k, b): band(V[k].astype(np.float64), *BANDS[b]) for k in V for b in BANDS}; TB = {b: band(Tn, *BANDS[b]) for b in BANDS}
shifted = {(b, s): (np.roll(np.roll(TB[b], s[0], 0), s[1], 1), disc & np.roll(np.roll(disc, s[0], 0), s[1], 1)) for b in BANDS for s in SH}
def best_shift(k, b): return max(SH, key=lambda s: corr(XB[(k, b)], shifted[(b, s)][0], shifted[(b, s)][1]))
fam = lambda k: k.split("__")[0]; lam = lambda k: float(k.split("__")[1]) if "__" in k else 0.0; FAMS = ["raw", "gpt", "gpt-s0.5", "gpt-s0.75", "gpt+blur", "full-equal", "full", "full-b0.8", "full-b1.2"]
out = dict(radius_m=RAD, curves={}, best={}, bootstrap={}); sh_used = {}
for b in BANDS:
    out["curves"][b] = {}
    for f in FAMS:
        ks = sorted([k for k in V if fam(k) == f], key=lam)
        if not ks: continue
        out["curves"][b][f] = {lam(k): max(corr(XB[(k, b)], shifted[(b, s)][0], shifted[(b, s)][1]) for s in SH) for k in ks}
    out["best"][b] = {f: (max(c, key=c.get), c[max(c, key=c.get)]) for f, c in out["curves"][b].items()}
# block bootstrap on the disc
bl = 10; blocks = [(y, x) for y in range(0, N, bl) for x in range(0, N, bl) if disc[y:y + bl, x:x + bl].mean() > 0.5]; rng = np.random.default_rng(1); B = 800
for b in BANDS:
    keys = {f: (f"{f}__{out['best'][b][f][0]}" if f != "raw" else "raw") for f in out["best"][b]}
    ga = [f for f in ("gpt", "gpt-s0.5", "gpt-s0.75") if f in out["best"][b]]                     # GPT with its assumed blur tuned too
    if len(ga) > 1: fb = max(ga, key=lambda f: out["best"][b][f][1]); keys["gpt-any"] = keys[fb]; out.setdefault("best_gpt_any", {})[b] = dict(family=fb, lam=out["best"][b][fb][0], corr=out["best"][b][fb][1])
    shs = {f: best_shift(keys[f], b) for f in keys}
    # use the shift that is best for the pair (the UCL best shift for all, to apply the placement tolerance equally): the shift chosen once, on the whole disc, per method; reported as given
    def stat(f, sel):
        m = np.zeros((N, N), bool)
        for y, x in sel: m[y:y + bl, x:x + bl] = True
        m &= disc; s = shs[f]; tb, mm = shifted[(b, s)]; return corr(XB[(keys[f], b)], tb, m & mm)
    diffs = {f: [] for f in keys if f != "gpt"}
    for _ in range(B):
        sel = [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]; ref = stat("gpt", sel)
        for f in diffs: diffs[f].append(stat(f, sel) - ref)
    out["bootstrap"][b] = {f: dict(setting=keys[f], mean=float(np.mean(d)), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], frac_positive=float((np.array(d) > 0).mean())) for f, d in diffs.items()}
    if "gpt-any" in diffs:
        d = np.array(diffs["full"]) - np.array(diffs["gpt-any"]); out["bootstrap"][b]["full_minus_gpt_any"] = dict(setting=[keys["full"], keys["gpt-any"]], mean=float(d.mean()), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], frac_positive=float((d > 0).mean()))
    print(f"\n{b}: best setting per method (corr with truth, +-0.25 m tolerance): " + ", ".join(f"{f} {out['best'][b][f][1]:.3f} [lam {out['best'][b][f][0]}]" for f in out["best"][b]))
    for f, d in out["bootstrap"][b].items(): print(f"   {f:11s} minus UCL(best): {d['mean']:+.3f}  95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  better in {100*d['frac_positive']:.0f}% of resamples")
print("\nerror (1 - correlation) versus lambda, 50-25 cm band:")
for f, c in out["curves"]["50-25 cm"].items(): print(f"  {f:11s} " + "  ".join(f"{l}:{1-v:.3f}" for l, v in c.items()))
json.dump(out, open(OUT, "w"), indent=1)

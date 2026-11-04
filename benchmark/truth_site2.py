"""Score restorations of the second Ingenuity truth site (flights 4-5, sol 76, south of the landing spot) against the truth patch, the same way as the backshell test:
band-passed correlation in 2-1 m, 1 m-50 cm and 50-25 cm, a +-0.25 m placement tolerance applied equally to every method, error-versus-lambda curves,
block-bootstrap intervals for the difference to GPT (GPT's assumed blur and lambda both tuned: 'gpt-any'), and an out-of-sample check (setting chosen on one half
of the area, scored on the other; 4 splits). Area: the whole valid truth inside the 32 m tile, eroded by 10 HR px.
Usage: python benchmark/truth_site2.py <variants_npz,...> <truth_npz> <out_json>"""
import sys, json, numpy as np
from scipy import ndimage as ndi
VARS, TR, OUT = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
V = {}; [V.update({k: np.load(p)[k] for k in np.load(p).files}) for p in VARS]
t = np.load(TR); T = t["truth"].astype(np.float64); valid = t["valid"]; N = T.shape[0]; yy, xx = np.mgrid[0:N, 0:N]
area = ndi.binary_erosion(valid, iterations=10); Tn = np.where(valid, T, np.nanmean(T))
band = lambda a, lo, hi: ndi.gaussian_filter(a, lo) - ndi.gaussian_filter(a, hi); BANDS = {"2-1 m": (16, 32), "1 m-50 cm": (8, 16), "50-25 cm": (4, 8)}
SH = [(dy, dx) for dy in range(-4, 5, 2) for dx in range(-4, 5, 2)]
def corr(a, b, m): a = a[m] - a[m].mean(); b = b[m] - b[m].mean(); return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))
TB = {b: band(Tn, *BANDS[b]) for b in BANDS}; XB = {(k, b): band(V[k].astype(np.float64), *BANDS[b]) for k in V for b in BANDS}
shifted = {(b, s): (np.roll(np.roll(TB[b], s[0], 0), s[1], 1), np.roll(np.roll(area, s[0], 0), s[1], 1)) for b in BANDS for s in SH}
def sc(k, b, m): return max(corr(XB[(k, b)], shifted[(b, s)][0], m & shifted[(b, s)][1]) for s in SH)
def best_shift(k, b, m): return max(SH, key=lambda s: corr(XB[(k, b)], shifted[(b, s)][0], m & shifted[(b, s)][1]))
fam = lambda k: k.split("__")[0]; lam = lambda k: float(k.split("__")[1]) if "__" in k else 0.0
FAMS = [f for f in ["raw", "gpt", "gpt-s0.5", "gpt-s0.75", "gpt+blur", "full-equal", "full", "full-b0.8", "full-b1.2"] if any(fam(k) == f for k in V)]
member = lambda k, f: (fam(k) in ("gpt", "gpt-s0.5", "gpt-s0.75")) if f == "gpt-any" else fam(k) == f
out = dict(area_px=int(area.sum()), area_m2=float(area.sum() * 0.0625 ** 2), curves={}, best={}, bootstrap={}, heldout={})
for b in BANDS:
    out["curves"][b] = {f: {str(lam(k)): sc(k, b, area) for k in sorted([k for k in V if fam(k) == f], key=lam)} for f in FAMS}
    out["best"][b] = {f: max(c.items(), key=lambda kv: kv[1]) for f, c in out["curves"][b].items()}
    ga = max(("gpt", "gpt-s0.5", "gpt-s0.75"), key=lambda f: out["best"][b][f][1] if f in out["best"][b] else -9); out["best"][b]["gpt-any"] = [f"{ga}__{out['best'][b][ga][0]}", out["best"][b][ga][1]]
    print(f"\n{b}: best per method: " + ", ".join(f"{f} {v[1]:.3f} [{v[0]}]" for f, v in out["best"][b].items()), flush=True)
# block bootstrap of the difference at each method's best setting
bl = 10; blocks = [(y, x) for y in range(0, N, bl) for x in range(0, N, bl) if area[y:y + bl, x:x + bl].mean() > 0.5]; rng = np.random.default_rng(1); B = 600
for b in BANDS:
    key = lambda f: out["best"][b]["gpt-any"][0] if f == "gpt-any" else ("raw" if f == "raw" else f"{f}__{float(out['best'][b][f][0])}")
    keys = {f: key(f) for f in FAMS + ["gpt-any"]}; keys = {f: (k if k in V else next(kk for kk in V if fam(kk) == fam(k) and abs(lam(kk) - lam(k)) < 1e-9)) for f, k in keys.items()}
    shs = {f: best_shift(keys[f], b, area) for f in keys}
    def stat(f, sel):
        m = np.zeros((N, N), bool)
        for y, x in sel: m[y:y + bl, x:x + bl] = True
        s = shs[f]; tb, mm = shifted[(b, s)]; return corr(XB[(keys[f], b)], tb, m & area & mm)
    D = {f: [] for f in keys if f not in ("gpt-any",)}
    for _ in range(B):
        sel = [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]; ref = stat("gpt-any", sel)
        for f in D: D[f].append(stat(f, sel) - ref)
    out["bootstrap"][b] = {f: dict(setting=keys[f], mean=float(np.mean(d)), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], frac_positive=float((np.array(d) > 0).mean())) for f, d in D.items()}
    print(f"{b}: minus best GPT (blur and lambda tuned): " + ", ".join(f"{f} {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]" for f, d in out["bootstrap"][b].items()), flush=True)
# out of sample: halves by x and by y
cy, cx = np.mean(np.nonzero(area), 1); halves = {"x": (area & (xx < cx), area & (xx >= cx)), "y": (area & (yy < cy), area & (yy >= cy))}
for b in BANDS:
    out["heldout"][b] = {}
    for f in FAMS + ["gpt-any"]:
        ks = [k for k in V if member(k, f)]; r = []
        for A_, B_ in halves.values():
            for a_, b_ in ((A_, B_), (B_, A_)): kb = max(ks, key=lambda k: sc(k, b, a_)); r.append((sc(kb, b, b_), kb))
        out["heldout"][b][f] = dict(heldout=float(np.mean([x for x, _ in r])), chosen=[k for _, k in r])
    print(f"{b}: held-out: " + ", ".join(f"{f} {v['heldout']:.3f}" for f, v in out["heldout"][b].items()), flush=True)
json.dump(out, open(OUT, "w"), indent=1); print("wrote", OUT)

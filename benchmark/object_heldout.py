"""Out-of-sample check of the object-centred truth result: the setting (lambda) of every method is chosen on one half of the 4 m disc around the backshell and scored on the other half
(both directions, split by x and by y), with the same +-0.25 m placement tolerance for all. Usage: python benchmark/object_heldout.py <variants_npz,...> <truth_npz> <out_json>"""
import sys, json, numpy as np
from scipy import ndimage as ndi
VARS, TR, OUT = sys.argv[1].split(","), sys.argv[2], sys.argv[3]; HR = 0.2524 / 4; RAD = 4.0
V = {}; [V.update({k: np.load(p)[k] for k in np.load(p).files}) for p in VARS]; t = np.load(TR); T = t["truth"].astype(np.float64); valid = t["valid"]; N = T.shape[0]; yy, xx = np.mgrid[0:N, 0:N]
bs = np.array([N / 2 + t["params"][2] / HR, N / 2 + t["params"][3] / HR]); disc = (np.hypot(xx - bs[0], yy - bs[1]) < RAD / HR) & valid & ndi.binary_erosion(valid, iterations=6); Tn = np.where(valid, T, np.nanmean(T))
band = lambda a, lo, hi: ndi.gaussian_filter(a, lo) - ndi.gaussian_filter(a, hi); BANDS = {"2-1 m": (16, 32), "1 m-50 cm": (8, 16), "50-25 cm": (4, 8)}; SH = [(dy, dx) for dy in range(-4, 5, 2) for dx in range(-4, 5, 2)]
def corr(a, b, m): a = a[m] - a[m].mean(); b = b[m] - b[m].mean(); return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))
XB = {(k, b): band(V[k].astype(np.float64), *BANDS[b]) for k in V for b in BANDS}; TB = {b: band(Tn, *BANDS[b]) for b in BANDS}
def sc(k, b, m): return max(corr(XB[(k, b)], np.roll(np.roll(TB[b], s[0], 0), s[1], 1), m & np.roll(np.roll(disc, s[0], 0), s[1], 1)) for s in SH)
fam = lambda k: k.split("__")[0]; FAMS = ["raw", "gpt", "gpt+blur", "full-equal", "full"]; cx_, cy_ = bs
halves = {"x": (disc & (xx < cx_), disc & (xx >= cx_)), "y": (disc & (yy < cy_), disc & (yy >= cy_))}; out = {}
for b in BANDS:
    out[b] = {}
    for f in FAMS:
        ks = [k for k in V if fam(k) == f]; r = []
        for ax, (A, B) in halves.items():
            for a_, b_ in ((A, B), (B, A)): kb = max(ks, key=lambda k: sc(k, b, a_)); r.append((sc(kb, b, b_), kb))
        out[b][f] = dict(heldout=float(np.mean([x for x, _ in r])), chosen=[k for _, k in r])
    print(f"{b}: held-out correlation (setting chosen on the other half, 4 splits): " + ", ".join(f"{f} {out[b][f]['heldout']:.3f}" for f in FAMS) + f"  | full minus UCL {out[b]['full']['heldout']-out[b]['gpt']['heldout']:+.3f}")
json.dump(out, open(OUT, "w"), indent=1)

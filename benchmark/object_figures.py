"""Figures for the object-centred truth test: (1) error versus lambda per method (the minimum of each), (2) a labelled comparison of the backshell neighbourhood:
Ingenuity truth (3 cm native, shown at the 6.31 cm grid) beside raw HiRISE and each restoration at its truth-optimal and its default setting; second row band-passed (25 cm - 1 m).
Usage: python benchmark/object_figures.py <variants_npz,...> <truth_npz> <object_final_json> <out_prefix>"""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
VARS, TR, FIN, OUT = sys.argv[1].split(","), sys.argv[2], sys.argv[3], sys.argv[4]; HR = 0.2524 / 4
V = {}; [V.update({k: np.load(p)[k] for k in np.load(p).files}) for p in VARS]; t = np.load(TR); T = t["truth"]; valid = t["valid"]; N = T.shape[0]; fin = json.load(open(FIN))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8)); lab = {"raw": "raw (no restoration)", "gpt": "UCL re-implementation", "gpt+blur": "UCL + measured blur", "full-equal": "this work, equal pass weights", "full": "this work"}
col = {"gpt": "#c0392b", "gpt+blur": "#e67e22", "full-equal": "#2980b9", "full": "#16a085"}
for a, b in zip(ax, ("1 m-50 cm", "50-25 cm")):
    for f in ("gpt", "gpt+blur", "full-equal", "full"):
        c = fin["curves"][b][f]; l = np.array(sorted(float(x) for x in c)); e = np.array([1 - c[str(x)] if str(x) in c else 1 - c[repr(x)] for x in l]) if False else np.array([1 - v for _, v in sorted(((float(k), v) for k, v in c.items()))])
        a.plot(l, e, "-o", ms=4, color=col[f], label=lab[f]); a.plot(l[np.argmin(e)], e.min(), "*", ms=14, color=col[f], mec="k")
    a.axhline(1 - fin["curves"][b]["raw"]["0.0"], color="gray", ls="--", label=lab["raw"]); a.set_xscale("log"); a.set_xlabel("regularisation strength lambda"); a.set_ylabel("error vs truth = 1 - correlation"); a.set_title(f"{b} band (stars = each method's minimum)"); a.grid(alpha=.3)
ax[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}_error_vs_lambda.png", dpi=110)
# comparison figure
ys, xs = np.nonzero(valid); cy, cx = N / 2 + t["params"][3] / HR, N / 2 + t["params"][2] / HR; h = int(4.4 / HR); y0, x0 = int(cy - h), int(cx - h); sl = (slice(y0, y0 + 2 * h), slice(x0, x0 + 2 * h))
best = {b: {f: fin["best"][b][f][0] for f in fin["best"][b]} for b in fin["best"]}; pick = lambda f, b="50-25 cm": (f"{f}__{best[b][f]}" if f != "raw" else "raw")
cols = [("Ingenuity truth\n(3 cm, shown on the 6.3 cm grid)", T, True), ("raw HiRISE\n(one pass)", V["raw"], False), (f"UCL re-impl.\nlambda {best['50-25 cm']['gpt']} (truth-optimal)", V[pick("gpt")], False), ("UCL re-impl.\nlambda 0.005 (typical)", V["gpt__0.005"], False),
        (f"this work\nlambda {best['50-25 cm']['full']} (truth-optimal)", V[pick("full")], False), ("this work\nlambda 0.005", V["full__0.005"], False)]
Z = 6; font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15); W_ = 2 * h * Z; can = Image.new("RGB", (len(cols) * (W_ + 8), 2 * (W_ + 50)), (20, 20, 20)); d = ImageDraw.Draw(can)
for i, (label, X, is_t) in enumerate(cols):
    for row in (0, 1):
        A = np.nan_to_num(np.asarray(X, float), nan=np.nanmean(T)) if is_t else np.asarray(X, float)
        if row == 1: A = ndi.gaussian_filter(A, 2.0) - ndi.gaussian_filter(A, 8.0)
        a = A[sl]; lo, hi = np.percentile(a, [1, 99.5]); im = (np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8); im = np.kron(im, np.ones((Z, Z), np.uint8))
        if is_t: m = np.kron(valid[sl].astype(np.uint8), np.ones((Z, Z), np.uint8)) > 0; im[~m] = 25
        x, y = i * (W_ + 8), row * (W_ + 50); can.paste(Image.fromarray(im).convert("RGB"), (x, y + 44))
        if row == 0: d.multiline_text((x + 4, y + 2), label, font=font, fill=(255, 255, 0))
        else: d.text((x + 4, y + 14), "band-passed (25 cm - 1 m)", font=font, fill=(200, 200, 200))
        L = int(2 / HR * Z); d.line([(x + 14, y + 44 + W_ - 14), (x + 14 + L, y + 44 + W_ - 14)], fill=(255, 255, 255), width=4); d.text((x + 20 + L, y + 44 + W_ - 30), "2 m", font=font, fill=(255, 255, 255))
can.save(f"{OUT}_backshell_comparison.png"); print(can.size)

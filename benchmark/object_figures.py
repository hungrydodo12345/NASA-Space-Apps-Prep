"""Figures for the object-centred truth test: (1) error versus lambda per method (the minimum of each), (2) a labelled comparison of the backshell neighbourhood:
Ingenuity truth (3 cm native, shown at the 6.31 cm grid) beside raw HiRISE and each restoration at its truth-optimal and its default setting; second row band-passed (25 cm - 1 m).
Usage: python benchmark/object_figures.py <variants_npz,...> <truth_npz> <object_final_json> <out_prefix>"""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
VARS, TR, FIN, OUT = sys.argv[1].split(","), sys.argv[2], sys.argv[3], sys.argv[4]; HR = 0.2524 / 4
V = {}; [V.update({k: np.load(p)[k] for k in np.load(p).files}) for p in VARS]; t = np.load(TR); T = t["truth"]; valid = t["valid"]; N = T.shape[0]; fin = json.load(open(FIN))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8)); lab = {"raw": "Raw HiRISE, 1 pass (no restoration)", "gpt": "GPT (simplified re-impl.)", "gpt+blur": "GPT + measured blur", "full-equal": "This work, equal pass weights", "full": "This work"}
col = {"gpt": "#2a78d6", "gpt+blur": "#eb6834", "full-equal": "#6250d6", "full": "#1baf7a"}
for a, b in zip(ax, ("1 m-50 cm", "50-25 cm")):
    for f in ("gpt", "gpt+blur", "full-equal", "full"):
        c = fin["curves"][b][f]; l = np.array(sorted(float(x) for x in c)); e = np.array([1 - c[str(x)] if str(x) in c else 1 - c[repr(x)] for x in l]) if False else np.array([1 - v for _, v in sorted(((float(k), v) for k, v in c.items()))])
        a.plot(l, e, "-o", ms=4, color=col[f], label=lab[f]); a.plot(l[np.argmin(e)], e.min(), "*", ms=14, color=col[f], mec="k")
    a.axhline(1 - fin["curves"][b]["raw"]["0.0"], color="gray", ls="--", label=lab["raw"]); a.set_xscale("log"); a.set_xlabel("Regularisation strength λ (log scale)"); a.set_ylabel("Error vs Ingenuity truth (1 − correlation)"); a.set_title(f"({'ab'[('1 m-50 cm', '50-25 cm').index(b)]}) {b} band; stars = each method's minimum", loc="left"); a.grid(alpha=.3)
ax[1].legend(fontsize=8.5, frameon=False); fig.suptitle("Backshell truth test: error within 4 m of the backshell versus regularisation (lower is better)", x=0.01, ha="left"); fig.tight_layout(); fig.savefig(f"{OUT}_error_vs_lambda.png", dpi=110)
# comparison figure
ys, xs = np.nonzero(valid); cy, cx = N / 2 + t["params"][3] / HR, N / 2 + t["params"][2] / HR; h = int(4.4 / HR); y0, x0 = int(cy - h), int(cx - h); sl = (slice(y0, y0 + 2 * h), slice(x0, x0 + 2 * h))
best = {b: {f: fin["best"][b][f][0] for f in fin["best"][b]} for b in fin["best"]}; pick = lambda f, b="50-25 cm": (f"{f}__{best[b][f]}" if f != "raw" else "raw")
cols = [("(a) Ingenuity truth, flight 26\n3 cm/px, shown on the 6.3 cm grid", T, True), ("(b) Raw HiRISE ESP_068294\n1 pass, 25 cm/px", V["raw"], False), (f"(c) GPT (simplified re-impl.)\nλ = {best['50-25 cm']['gpt']} (its truth-optimal)", V[pick("gpt")], False), ("(d) GPT (simplified re-impl.)\nλ = 0.005 (typical)", V["gpt__0.005"], False),
        (f"(e) This work\nλ = {best['50-25 cm']['full']} (its truth-optimal)", V[pick("full")], False), ("(f) This work\nλ = 0.005", V["full__0.005"], False)]
Z = 6; font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30); bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36); W_ = 2 * h * Z; TOP = 70; can = Image.new("RGB", (len(cols) * (W_ + 8), TOP + 2 * (W_ + 90) + 60), (20, 20, 20)); d = ImageDraw.Draw(can)
d.text((10, 14), "Backshell of Perseverance: Ingenuity truth beside HiRISE restorations (8.8 m crops, north up)", font=bold, fill=(245, 245, 245))
for i, (label, X, is_t) in enumerate(cols):
    for row in (0, 1):
        A = np.nan_to_num(np.asarray(X, float), nan=np.nanmean(T)) if is_t else np.asarray(X, float)
        if row == 1: A = ndi.gaussian_filter(A, 2.0) - ndi.gaussian_filter(A, 8.0)
        a = A[sl]; lo, hi = np.percentile(a, [1, 99.5]); im = (np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8); im = np.kron(im, np.ones((Z, Z), np.uint8))
        if is_t: m = np.kron(valid[sl].astype(np.uint8), np.ones((Z, Z), np.uint8)) > 0; im[~m] = 25
        x, y = i * (W_ + 8), TOP + row * (W_ + 90); can.paste(Image.fromarray(im).convert("RGB"), (x, y + 84))
        if row == 0: d.multiline_text((x + 6, y + 4), label, font=font, fill=(245, 245, 245), spacing=4)
        else: d.text((x + 6, y + 40), "same, band-passed to 25 cm – 1 m", font=font, fill=(200, 200, 200))
        L = int(2 / HR * Z); d.line([(x + 14, y + 84 + W_ - 20), (x + 14 + L, y + 84 + W_ - 20)], fill=(255, 255, 255), width=6); d.text((x + 24 + L, y + 84 + W_ - 40), "2 m", font=font, fill=(255, 255, 255))
d.text((10, can.size[1] - 48), "Each panel has its own contrast stretch (1–99.5 %); grey outside the truth disc = no Ingenuity coverage. Pixels are shown as blocks, not interpolated.", font=font, fill=(190, 190, 190))
can.save(f"{OUT}_backshell_comparison.png"); print(can.size)

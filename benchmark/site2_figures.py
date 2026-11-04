"""Figures for the second Ingenuity truth site (flights 4-5, sol 76): (1) error versus lambda per method in the 1 m-50 cm and 50-25 cm bands;
(2) a labelled comparison: Ingenuity truth, raw HiRISE, the best GPT variant and this work, each at its own best setting, plus the same band-passed to 25 cm-1 m.
Usage: python benchmark/site2_figures.py <variants_npz> <truth_npz> <site2_final_json> <out_prefix>"""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
VAR, TR, FIN, OUT = sys.argv[1:5]; V = np.load(VAR); t = np.load(TR); T, valid = t["truth"], t["valid"]; fin = json.load(open(FIN)); HR = 0.0625
COL = {"gpt": "#2a78d6", "gpt-s0.75": "#86b6ef", "gpt-s0.5": "#184f95", "gpt+blur": "#eb6834", "full": "#1baf7a"}
LAB = {"gpt": "GPT, assumed blur 1 px", "gpt-s0.75": "GPT, 0.75 px", "gpt-s0.5": "GPT, 0.5 px", "gpt+blur": "GPT + measured blur", "full": "This work"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for a, b, L in zip(ax, ("1 m-50 cm", "50-25 cm"), "ab"):
    for f in COL:
        c = fin["curves"][b][f]; l = np.array([float(k) for k in c]); e = np.array([1 - v for v in c.values()]); o = np.argsort(l)
        a.plot(l[o], e[o], "-o", ms=3.5, color=COL[f], label=LAB[f], lw=2 if f == "full" else 1.4); a.plot(l[o][np.argmin(e[o])], e.min(), "*", ms=13, color=COL[f], mec="k")
    a.axhline(1 - fin["curves"][b]["raw"]["0.0"], color="#8a8985", ls="--", label="Raw HiRISE, 1 pass"); a.set_xscale("log"); a.grid(alpha=.3)
    a.set_xlabel("Regularisation strength λ (log scale)"); a.set_ylabel("Error vs Ingenuity truth (1 − correlation)"); a.set_title(f"({L}) {b} band; stars = each method's minimum", loc="left")
ax[1].legend(fontsize=8.5, frameon=False); fig.suptitle(f"Second truth site (flights 4–5, sol 76): error over {fin['area_m2']:.0f} m² versus regularisation (lower is better)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}_error_vs_lambda.png", dpi=200); plt.close(fig)
b = "50-25 cm"; bg = fin["best"][b]["gpt-any"][0]; bf = f"full__{float(fin['best'][b]['full'][0])}"
bf = bf if bf in V.files else next(k for k in V.files if k.startswith("full__") and abs(float(k.split("__")[1]) - float(fin["best"][b]["full"][0])) < 1e-9)
cols = [("(a) Ingenuity truth, flights 4–5\n1 cm/px, shown on the 6.25 cm grid", np.where(valid, T, np.nan)), ("(b) Raw HiRISE ESP_068294\n1 pass, 25 cm/px", V["raw"]),
        (f"(c) Best GPT variant\n{bg.split('__')[0].replace('gpt-s', 'blur ').replace('gpt', 'blur 1')} px, λ = {bg.split('__')[1]}", V[bg]), (f"(d) This work\nλ = {bf.split('__')[1]}", V[bf])]
ys, xs = np.nonzero(ndi.binary_erosion(valid, iterations=10)); cy, cx = int(np.median(ys)), int(np.median(xs)); h = 112; sl = (slice(cy - h, cy + h), slice(cx - h, cx + h))
fig, ax = plt.subplots(2, 4, figsize=(16, 8.6))
for j, (lab, X) in enumerate(cols):
    for r in (0, 1):
        A = np.asarray(X, float); m = np.isfinite(A); A = np.where(m, A, np.nanmean(A))
        if r == 1: A = ndi.gaussian_filter(A, 2.0) - ndi.gaussian_filter(A, 8.0)
        a_ = A[sl]; lo, hi = np.percentile(a_[m[sl]] if m[sl].any() else a_, [1, 99.5]); im = np.clip((a_ - lo) / (hi - lo + 1e-9), 0, 1); im[~m[sl]] = 0.1
        ax[r, j].imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
        L2 = 2 / HR; ax[r, j].plot([10, 10 + L2], [2 * h - 12, 2 * h - 12], color="w", lw=3); ax[r, j].text(14 + L2, 2 * h - 16, "2 m", color="w", fontsize=9)
    ax[0, j].set_title(lab, fontsize=10, loc="left"); ax[1, j].set_title("same, band-passed to 25 cm – 1 m", fontsize=9, loc="left")
fig.suptitle("Second truth site, 14 m crop south of the landing spot (north up). Each panel has its own contrast stretch; dark = no Ingenuity coverage.", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}_comparison.png", dpi=150); print("wrote", OUT)

"""Paper figures, drawn only from result files in the repository (no restoration is re-run here).
fig1_static_boulder.png      the blur/edge target in all 8 passes (benchmark/results/figdata/boulder_passes.npz)
fig2_hardware_alignment.png  parachute and backshell in 8 passes: absent before landing, present after (figdata/hardware_crops.npz)
fig3_simulation.png          test A: error at 25 cm, at 6.25 cm, and artefact at the planted change, per scene
fig4_split_half.png          test B: coherence between independent pass halves versus scale
fig5_boulder_edge.png        test C: edge width, ringing and undershoot at the boulder edge
fig6_uncertainty.png         test D: held-out coverage of the calibrated bootstrap uncertainty
Usage: python scripts/make_paper_figures.py"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); J = lambda *p: os.path.join(ROOT, *p)
OUT = J("results", "paper"); os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"raw": "#8a8985", "gpt": "#2a78d6", "gpt+blur": "#eb6834", "full": "#1baf7a"}          # fixed per method in every figure
LAB = {"raw": "Raw HiRISE (1 pass)", "gpt": "GPT (simplified re-impl.)", "gpt+blur": "GPT + measured blur", "full": "This work"}
ORDER = ["raw", "gpt", "gpt+blur", "full"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
def save(fig, name): fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote", f"results/paper/{name}")
def stretch(a, lo=1, hi=99.7): l, h = np.nanpercentile(a, [lo, hi]); return np.clip((a - l) / (h - l + 1e-12), 0, 1)
def hbars(ax, keys, vals, fmt, xlabel, dots=None):
    y = np.arange(len(keys))[::-1]
    for yi, k, v in zip(y, keys, vals):
        right = max([v] + (list(dots[k]) if dots is not None else []))
        ax.barh(yi, v, height=0.56, color=COL[k], zorder=2); ax.text(right, yi, "  " + fmt.format(v), va="center", ha="left", color=INK, fontsize=9)
        if dots is not None: ax.scatter(dots[k], np.full(len(dots[k]), yi), s=14, color=SURF, edgecolor=INK, linewidth=0.8, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([LAB[k] for k in keys]); ax.set_xlabel(xlabel); ax.grid(axis="x", color=GRID, zorder=0); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(max(vals), max(max(v) for v in dots.values()) if dots else 0) * 1.3)

# fig 1: static boulder in all passes
b = np.load(J("benchmark", "results", "figdata", "boulder_passes.npz")); idx = np.argsort(b["dates"])
fig, ax = plt.subplots(2, 4, figsize=(11, 6))
for a, i in zip(ax.flat, idx):
    a.imshow(stretch(b["aligned"][i]), cmap="gray", interpolation="nearest"); a.set_title(f"{str(b['passes'][i]).replace('_1985', '')}\n{b['dates'][i]}", fontsize=9, color=INK)
    a.annotate("", xy=(72, 55), xytext=(100, 27), arrowprops=dict(arrowstyle="->", color="#eb6834", lw=1.4)); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)
fig.suptitle("The blur and edge target: a compact bright boulder present in every pass, 2007 to 2022 (32 m tile, 25 cm pixels)", fontsize=11, color=INK, x=0.01, ha="left")
save(fig, "fig1_static_boulder.png")

# fig 2: landing hardware appears after landing (alignment check across 15 years)
h = np.load(J("benchmark", "results", "figdata", "hardware_crops.npz")); idx = np.argsort(h["dates"])
fig, ax = plt.subplots(2, 8, figsize=(16, 4.8))
for r, name in enumerate(h["names"]):
    for j, i in enumerate(idx):
        a = ax[r, j]; a.imshow(stretch(h[str(name)][i]), cmap="gray", interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(False)
        if r == 0: a.set_title(f"{str(h['passes'][i]).replace('_1985', '')}\n{h['dates'][i]}", fontsize=8.5, color=INK)
        if j == 0: a.set_ylabel(str(name), fontsize=10, color=INK)
fig.suptitle("Alignment check: the parachute (top) and backshell (bottom) are absent in 2007 and 2016 and present from sol 1; 40 m crops, the same ground in every pass", fontsize=11, color=INK, x=0.01, ha="left")
save(fig, "fig2_hardware_alignment.png")

R = json.load(open(J("benchmark", "results", "benchmark_results.json"))); A = R["A_simulation"]
# fig 3: simulation
fig, ax = plt.subplots(1, 3, figsize=(14, 3.4), gridspec_kw=dict(wspace=0.9))
for a, (k, lab, fmt) in zip(ax, (("err_25cm", "RMS error at 25 cm (% of scene contrast)", "{:.2f}%"), ("err_6cm", "RMS error at 6.25 cm (% of scene contrast)", "{:.1f}%"), ("artefact", "Largest error at the planted change (x scene std)", "{:.2f}"))):
    keys = [m for m in ORDER if m in A]; hbars(a, keys, [A[m][k]["mean"] for m in keys], fmt, lab, dots={m: A[m][k]["values"] for m in keys})
    a.set_title("(" + "abc"[list(ax).index(a)] + ")", loc="left", fontsize=10, color=INK)
    if a is not ax[0]: a.set_yticklabels([])
fig.suptitle(f"Test A, simulation with exact truth ({len(A['gpt']['err_25cm']['values'])} scenes; bars = mean, dots = scenes). Lower is better.", fontsize=11, color=INK, x=0.01, ha="left", y=1.04)
save(fig, "fig3_simulation.png")

# fig 4: split-half coherence
B = R["B_split_half_real"]; fig, a = plt.subplots(figsize=(7.5, 4))
for m in ORDER:
    if m not in B: continue
    c = np.array(B[m]["coherence_by_band"]); hp = 6.25 / ((np.arange(len(c)) + 0.5) / len(c)); a.plot(hp, c, color=COL[m], lw=2, label=LAB[m])
a.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3))); a.text(7, 0.52, "agreement threshold 0.5", color=INK2, fontsize=8.5, ha="right")
a.set_xscale("log"); a.set_xlim(400, 6); a.set_xticks([400, 200, 100, 50, 25, 12.5, 6.25]); a.set_xticklabels(["400", "200", "100", "50", "25", "12.5", "6.25"])
a.set_xlabel("Scale (half-period, cm), finer to the right"); a.set_ylabel("Coherence between independent halves"); a.set_ylim(0, 1); a.grid(color=GRID); a.legend(frameon=False, fontsize=9)
a.set_title("Test B, real HiRISE: detail that repeats in two disjoint sets of 4 passes", fontsize=11, color=INK, loc="left")
save(fig, "fig4_split_half.png")

# fig 5: boulder edge
C = R["C_rover_edge_real"]; keys = [m for m in ORDER if m in C]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.0), gridspec_kw=dict(wspace=0.9))
for a, (k, lab, sc, fmt) in zip(ax, (("edge_sigma_cm", "Edge width sigma (cm), narrower = sharper", 1, "{:.1f}"), ("ringing_rms_of_amplitude", "Ringing, RMS (% of edge step)", 100, "{:.1f}%"), ("max_undershoot_of_amplitude", "Largest undershoot (% of edge step)", 100, "{:.1f}%"))):
    hbars(a, keys, [sc * C[m][k] for m in keys], fmt, lab)
    a.set_title("(" + "abc"[list(ax).index(a)] + ")", loc="left", fontsize=10, color=INK)
    if a is not ax[0]: a.set_yticklabels([])
fig.suptitle("Test C, real HiRISE: edge of the bright boulder (3 passes, sols 1-11). Sharpness costs ringing; lower ringing is cleaner.", fontsize=11, color=INK, x=0.01, ha="left", y=1.06)
save(fig, "fig5_boulder_edge.png")

# fig 6: uncertainty coverage
U = json.load(open(J("results", "bootstrap_uncertainty_simulation.json")))["scales"]; sc = list(U)
fig, a = plt.subplots(figsize=(7.5, 3.4)); x = np.arange(len(sc)); w = 0.34
for off, (s, nom, col) in zip((-w / 2, w / 2), (("1.0", 0.683, "#2a78d6"), ("2.0", 0.954, "#eb6834"))):
    v = [U[k]["scene_B_heldout"]["coverage_calibrated"][s] for k in sc]
    a.bar(x + off, v, width=w * 0.92, color=col, label=f"±{s[0]} sigma (nominal {100 * nom:.0f}%)", zorder=2)
    for xi, vi in zip(x + off, v): a.text(xi, vi + 0.015, f"{100 * vi:.0f}%", ha="center", fontsize=8.5, color=INK)
    a.hlines(nom, x[0] - 0.6, x[-1] + 0.6, color=col, lw=1, ls=(0, (4, 3)))
a.set_xticks(x); a.set_xticklabels(sc); a.set_ylim(0, 1.32); a.set_yticks([0, .2, .4, .6, .8, 1]); a.set_ylabel("Held-out coverage"); a.grid(axis="y", color=GRID, zorder=0); a.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
a.set_title("Test D: calibrated bootstrap uncertainty on a held-out simulated scene (dashed = nominal)", fontsize=11, color=INK, loc="left")
save(fig, "fig6_uncertainty.png")

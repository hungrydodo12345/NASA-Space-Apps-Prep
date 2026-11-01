"""Aggregate experiment results into paper tables (JSON) and figures (PNG), with 95% bootstrap confidence intervals.

Usage: python experiments/report.py <out_dir>
"""
import sys, os, json, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments import common as C, algos

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ORDER = ["bicubic", "average", "ibp", "cg_tikhonov", "irls_tv", "chambolle_pock_tv", "richardson_lucy", "btv_l1"]
COL = {"bicubic": "#8a8a8a", "average": "#b07a4f", "ibp": "#1f6fb2", "cg_tikhonov": "#2a9d8f", "irls_tv": "#7a5cc4",
       "chambolle_pock_tv": "#c9628c", "richardson_lucy": "#d4a017", "btv_l1": "#3c8d3c"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
summary = {}

# ---------- Experiment 1: scaling
rows = [r for r in C.read(f"{RES}/exp1_scaling.jsonl") if r.get("ok") and "eff_res_cm" in r]
for scene in ("fractal", "rover"):
    sub = [r for r in rows if r["scene"] == scene]
    if not sub: continue
    by = collections.defaultdict(list)
    for r in sub: by[(r["algo"], r["n"])].append(r)
    ns = sorted({r["n"] for r in sub}); tab = {}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for a in ORDER:
        xs, er, ec, rr, rc = [], [], [], [], []
        for n in ns:
            v = by.get((a, n))
            if not v: continue
            m, lo, hi = C.boot_ci([x["eff_res_cm"] for x in v]); m2, lo2, hi2 = C.boot_ci([x["rmse_reduction_pct"] for x in v])
            tab.setdefault(a, {})[n] = dict(eff_res_cm=[m, lo, hi], rmse_reduction_pct=[m2, lo2, hi2], runs=len(v))
            xs.append(n); er.append(m); ec.append((m - lo, hi - m)); rr.append(m2); rc.append((m2 - lo2, hi2 - m2))
        if not xs: continue
        if a == "bicubic":
            ax[0].axhline(er[0], color=COL[a], ls="--", lw=1, label=algos.LABEL[a]); continue
        ax[0].errorbar(xs, er, yerr=np.array(ec).T, color=COL[a], marker="o", ms=3, lw=1.2, capsize=2, label=algos.LABEL[a])
        ax[1].errorbar(xs, rr, yerr=np.array(rc).T, color=COL[a], marker="o", ms=3, lw=1.2, capsize=2)
    for k, t in enumerate(("Smallest real feature (cm)", "Error vs bicubic (% lower)")):
        ax[k].set_xscale("log", base=2); ax[k].set_xticks(ns); ax[k].set_xticklabels(ns); ax[k].set_xlabel("HiRISE passes fused"); ax[k].set_ylabel(t)
    ax[0].legend(fontsize=6, frameon=False, ncol=1)
    fig.suptitle(f"Frame-count scaling, {'fractal terrain' if scene == 'fractal' else 'real rover ground texture'} (measured blur, 2% noise)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_scaling_{scene}.png"); plt.close(fig)
    summary[f"scaling_{scene}"] = tab

# ---------- Experiment 3: rocks
rows = [r for r in C.read(f"{RES}/exp3_rocks.jsonl") if r.get("ok")]
if rows:
    sizes = sorted({int(k) for r in rows for k in r["completeness"]})
    by = collections.defaultdict(list)
    for r in rows: by[(r["algo"], r["n"])].append(r)
    tab = {}
    for (a, n), v in by.items():
        tab.setdefault(a, {})[n] = dict(completeness={D: C.boot_ci([x["completeness"][str(D)] * 100 for x in v]) for D in sizes},
                                        false_per_100m2=C.boot_ci([x["false_per_100m2"] for x in v]), runs=len(v))
    summary["rocks"] = tab
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for a, n, ls in (("bicubic", 1, "--"), ("average", 6, ":"), ("ibp", 6, "-"), ("btv_l1", 6, "-"), ("ibp", 16, "-")):
        if (a, n) not in by: continue
        m = [tab[a][n]["completeness"][D] for D in sizes]
        ax.errorbar(sizes, [x[0] for x in m], yerr=np.array([(x[0] - x[1], x[2] - x[0]) for x in m]).T, color=COL[a], ls=ls, marker="o", ms=3, capsize=2,
                    lw=1.6 if n == 16 else 1.1, label=f"{algos.LABEL[a]}, {n} pass{'es' if n > 1 else ''}")
    ax.axhline(50, color="#444", lw=0.6); ax.text(sizes[-1], 52, "50% = 'most rocks'", ha="right", fontsize=7)
    ax.set_xlabel("Rock diameter (cm)"); ax.set_ylabel("Rocks found (%)"); ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels(sizes)
    ax.legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/fig_rocks.png"); plt.close(fig)

json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print("wrote", sorted(os.listdir(OUT)))

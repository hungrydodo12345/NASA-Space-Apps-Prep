"""Changed-ground test on real HiRISE: Perseverance is parked at its landing spot in three passes (sols 1, 7, 11) and absent from the others (2016, 2021-04 onward). The background
(the ground without a rover) is the per-pixel median of the aligned rover-free passes (LR grid). A method that fuses the passes should not leak the rover into the result.
Ghost fraction = mean over the rover footprint of (result - background) / mean over the footprint of (rover-present mean - background), with the result averaged down to the 25 cm grid.
0 = rover fully rejected, 1 = rover fully kept (a rover-present pass alone), ~ n_present / n_total for plain averaging. Also: RMS difference to the background outside the footprint
(does the method lose or add detail elsewhere?). Passes are chosen by (n_present, n_absent); the background always uses ALL rover-free passes. lambda = 0.04, 300 iterations.
Usage: python benchmark/changed_ground.py <chain_dir> <out_json> <out_png>"""
import sys, os, json, numpy as np
from multiprocessing import Pool
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline import pde_tv
CH, OUT, PNG = sys.argv[1:4]; SC = 4; LAM = float(os.environ.get("LAM", 0.04)); ITERS = int(os.environ.get("ITERS", 300))
d = np.load(f"{CH}/chain.npz"); meta = json.load(open(f"{CH}/chain.json")); B = json.load(open(os.path.join(ROOT, "data_products", "pass_blur.json")))
used = meta["passes_used"]; FR = list(d["frames"].astype(float)); FL = list(zip(d["flows_dx"].astype(float), d["flows_dy"].astype(float))); SIG = [B[o]["sigma_px"] * SC for o in used]; n, (H, W) = len(used), FR[0].shape
PRESENT = [i for i, o in enumerate(used) if o in ("ESP_068294_1985", "ESP_068360_1985", "ESP_068426_1985")]; ABSENT = [i for i in range(n) if i not in PRESENT]
print("rover-present passes:", [used[i] for i in PRESENT], "| rover-free passes:", [used[i] for i in ABSENT], flush=True)
yy, xx = np.mgrid[0:H, 0:W].astype(float)
al = [ndi.map_coordinates(FR[i], [yy - FL[i][1], xx - FL[i][0]], order=3, mode="nearest") for i in range(n)]                       # every pass on the reference grid
bg = np.median([al[i] for i in ABSENT], 0); pm = np.mean([al[i] for i in PRESENT], 0); diff = ndi.gaussian_filter(pm - bg, 0.8)
c = (H // 2, W // 2); win = np.zeros((H, W), bool); win[c[0] - 14:c[0] + 14, c[1] - 14:c[1] + 14] = True; pk = np.unravel_index(np.argmax(np.where(win, diff, -1e9)), diff.shape)
lab, _ = ndi.label((diff > 0.5 * diff[pk]) & win); fp = ndi.binary_dilation(lab == lab[pk], iterations=1); far = ~ndi.binary_dilation(fp, iterations=4)
print(f"rover footprint {fp.sum()} native px = {fp.sum() * 0.2524 ** 2:.1f} m2; contrast (present mean - background) over the footprint {diff[fp].mean():.3f}; background noise (rover-free passes, MAD) {1.4826 * np.median(np.abs(al[ABSENT[0]] - bg)[far]):.3f}", flush=True)
WTS = {}
METHODS = os.environ.get("METHODS", "gpt,gpt+blur,full").split(",")
def run(a):
    meth, npres, nabs = a; idx = PRESENT[:npres] + ABSENT[:nabs]; fr = [FR[i] for i in idx]; fl = [FL[i] for i in idx]; sg = [SIG[i] for i in idx]
    if meth == "gpt": X = pde_tv.pde_tv(fr, fl, scale=SC, psf_sigma=1.0 * SC, lam=LAM, iters=ITERS, weighted=False)[0]
    elif meth == "gpt+blur": X = pde_tv.pde_tv(fr, fl, scale=SC, psf_sigma=sg, lam=LAM, iters=ITERS, weighted=False)[0]
    elif meth == "full-consensus": X, info = pde_tv.pde_tv(fr, fl, scale=SC, psf_sigma=sg, lam=LAM, iters=ITERS, mask_mode="consensus"); WTS[a] = info["pass_weights"]
    elif meth == "full-median": X, info = pde_tv.pde_tv(fr, fl, scale=SC, psf_sigma=sg, lam=LAM, iters=ITERS, init="median"); WTS[a] = info["pass_weights"]
    elif meth == "full-nomask": X, info = pde_tv.pde_tv(fr, fl, scale=SC, psf_sigma=sg, lam=LAM, iters=ITERS, tukey_c=1e9); WTS[a] = info["pass_weights"]
    elif meth == "full-equal": X, info = pde_tv.pde_tv(fr, fl, scale=SC, psf_sigma=sg, lam=LAM, iters=ITERS, pass_weights="equal"); WTS[a] = info["pass_weights"]
    else: X, info = pde_tv.pde_tv(fr, fl, scale=SC, psf_sigma=sg, lam=LAM, iters=ITERS, pass_weights="noise"); WTS[a] = info["pass_weights"]
    return a, X.astype(np.float32), WTS.get(a)
nA = len(ABSENT); cfg = [(3, nA), (2, nA), (1, nA)] + [(3, k) for k in range(1, nA) if k not in (nA,)]; cfg = list(dict.fromkeys(cfg))
jobs = [(m, p, q) for (p, q) in cfg for m in METHODS]
if __name__ == "__main__":
    with Pool(4) as pool: raw = pool.map(run, jobs, chunksize=1)
    res = {a: X for a, X, w in raw}; wts = {str(a): w for a, X, w in raw if w is not None}
    lr = lambda X: X.reshape(H, SC, W, SC).mean((1, 3)); out = dict(pass_weights=wts, footprint_m2=float(fp.sum() * 0.2524 ** 2), passes_present=[used[i] for i in PRESENT], passes_absent=[used[i] for i in ABSENT], lam=LAM, configs={})
    ref_c = (pm - bg)[fp].mean()
    for (p, q) in cfg:
        row = {}
        for m in METHODS:
            X = lr(res[(m, p, q)]); row[m] = dict(ghost=float((X - bg)[fp].mean() / ref_c), background_rms_outside=float(np.sqrt(np.mean((X - bg)[far] ** 2))))
        out["configs"][f"{p}present_{q}free"] = row
    ref_pass = FR[PRESENT[0]]; out["raw_rover_pass_ghost"] = float((ref_pass - bg)[fp].mean() / ref_c); out["raw_rover_pass_background_rms_outside"] = float(np.sqrt(np.mean((ref_pass - bg)[far] ** 2)))
    out["rover_free_pass_background_rms_outside"] = float(np.sqrt(np.mean((al[ABSENT[0]] - bg)[far] ** 2)))
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nghost fraction in the rover footprint (0 = rover rejected, 1 = rover kept) | background RMS elsewhere"); print(f"{'config':18s} " + "".join(f"{m:>22s}" for m in METHODS))
    for k, row in out["configs"].items(): print(f"{k:18s} " + "".join(f"{row[m]['ghost']:>12.2f} | {row[m]['background_rms_outside']:.3f}" for m in METHODS))
    print("pass weights (order: " + ", ".join(u[:10] for u in used) + "):"); [print("  ", k, np.round(v, 2).tolist()) for k, v in list(wts.items())[:3]]
    print(f"single rover-present pass: ghost {out['raw_rover_pass_ghost']:.2f}; single rover-free pass: background RMS {out['rover_free_pass_background_rms_outside']:.3f}")
    # figure: crops around the rover for the full configuration
    if not PNG.endswith(".png") or os.environ.get("NOFIG"): sys.exit(0)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    h0 = 22; sl = (slice(c[0] - h0, c[0] + h0), slice(c[1] - h0, c[1] + h0)); p, q = cfg[0]
    panels = [("background\n(median of rover-free passes)", bg), ("rover-present passes (mean)", pm)] + [(f"{t}\n{p} present + {q} rover-free", lr(res[(m, p, q)])) for m, t in (("gpt", "UCL re-implementation"), ("gpt+blur", "UCL + measured blur"), ("full", "this work"))]
    lo, hi = np.percentile(pm[sl], [1, 99.5]); fig, ax = plt.subplots(2, 5, figsize=(20, 8.6))
    for j, (t, X) in enumerate(panels):
        ax[0, j].imshow(X[sl], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest"); ax[0, j].set_title(t, fontsize=10); ax[0, j].axis("off")
        ax[1, j].imshow((X - bg)[sl], cmap="RdBu_r", vmin=-0.6 * diff[pk], vmax=0.6 * diff[pk], interpolation="nearest"); ax[1, j].axis("off"); ax[1, j].contour(fp[sl], [0.5], colors="k", linewidths=0.8)
        if j >= 2: ax[1, j].set_title(f"ghost {out['configs'][f'{p}present_{q}free'][['gpt', 'gpt+blur', 'full'][j - 2]]['ghost']:.2f}", fontsize=11)
    ax[0, 0].text(0.0, -0.05, "25 cm pixels, 11 m wide crop centred on the landing spot; bottom row = (result - background); black outline = rover footprint", transform=ax[0, 0].transAxes, fontsize=9)
    fig.tight_layout(); fig.savefig(PNG, dpi=110); print("wrote", PNG)

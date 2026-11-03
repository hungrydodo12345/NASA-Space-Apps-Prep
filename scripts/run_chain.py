"""Measured part of the chain on real HiRISE: Gotcha alignment -> weighted PDE-TV with per-pass blur -> bootstrap uncertainty.

Usage:
  python scripts/run_chain.py <hirise_rdr_dir> <site_dir> <pass_blur.json> <out_dir> [--tile N] [--y0 Y] [--x0 X]
         [--boot B] [--iters I] [--kappa K] [--exclude-rover]
site_dir holds summary.json (per-pass integer locks to the shared grid, reference pass) and area.npz (origin_global),
as written by scripts/run_anywhere.py. The tile is N x N native 25 cm pixels starting at (y0, x0) inside the site.

Steps:
 1. Each pass is read at its integer lock plus a margin, and aligned to the reference with Gotcha (pipeline/gotcha.py).
    Passes where Gotcha matches fewer than 5 % of grid points are dropped (no reliable motion field). Each kept pass is
    then radiometrically normalised to the reference: gain and offset matching mean and standard deviation over the
    aligned overlap (passes differ by up to ~60 % in mean I/F; ground texture is ~1 %).
 2. PDE-TV MAP restoration at 4x (pipeline/pde_tv.py), weighted, with each pass's measured blur from pass_blur.json.
 3. Pass-subset bootstrap (pipeline/uncertainty.py), B rebuilds. The calibrated uncertainty at the native 25 cm scale is
    sigma = kappa * std_b(G_4 * X_b), kappa from experiments/exp6_bootstrap_uncertainty.py (25 cm row).
Output: <out_dir>/chain.npz (restored, bicubic_ref, sigma_boot_hr, sigma_native_cal, members_mean, frames, flows_dx, flows_dy)
and <out_dir>/chain.json (passes used, Gotcha statistics, pass weights, timings).
"""
import sys, os, json, time, argparse, numpy as np
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline.hirise_io import Rdr
from pipeline import gotcha, pde_tv, uncertainty as U

ap = argparse.ArgumentParser()
ap.add_argument("rdr"); ap.add_argument("site"); ap.add_argument("blur"); ap.add_argument("out")
ap.add_argument("--tile", type=int, default=128); ap.add_argument("--y0", type=int, default=None); ap.add_argument("--x0", type=int, default=None)
ap.add_argument("--boot", type=int, default=8); ap.add_argument("--iters", type=int, default=200); ap.add_argument("--lam", type=float, default=0.005)
ap.add_argument("--kappa", type=float, default=None, help="default: read from results/bootstrap_uncertainty_simulation.json")
ap.add_argument("--exclude-rover", action="store_true", help="mask a 5 m disc at the tile centre (the parked rover) from alignment and restoration")
a = ap.parse_args()
SC = 4; MARGIN = 12
os.makedirs(a.out, exist_ok=True)
summ = json.load(open(f"{a.site}/summary.json")); g0l, g0s = np.load(f"{a.site}/area.npz")["origin_global"]
blur = json.load(open(a.blur)); ref_id = summ.get("reference", summ["passes"][0])
site_n = np.load(f"{a.site}/area.npz")["bicubic"].shape[0] // SC
y0 = (site_n - a.tile) // 2 if a.y0 is None else a.y0; x0 = (site_n - a.tile) // 2 if a.x0 is None else a.x0
if a.kappa is None:
    k = json.load(open(os.path.join(ROOT, "results", "bootstrap_uncertainty_simulation.json")))
    a.kappa = k["scales"]["25 cm (native HiRISE pixel)"]["kappa_fitted_on_A"]


def read(oid, n, oy, ox):
    o = summ["locks"][oid]; w = Rdr(f"{a.rdr}/{oid}_RED").read_if(g0l + oy - o["dy"], g0s + ox - o["dx"], n, n)
    return w


t0 = time.time(); N = a.tile + 2 * MARGIN
ref = read(ref_id, N, y0 - MARGIN, x0 - MARGIN)
exclude = None
if a.exclude_rover:
    yy, xx = np.mgrid[0:N, 0:N]; exclude = np.hypot(yy - N / 2, xx - N / 2) < 10
frames, flows, used, stats = [], [], [], {}
order = [ref_id] + [p for p in summ["passes"] if p != ref_id]
for oid in order:
    img = read(oid, N, y0 - MARGIN, x0 - MARGIN)
    if np.isnan(img).mean() > 0.01: stats[oid] = "dropped: outside pass footprint"; continue
    if oid == ref_id: dx = dy = np.zeros((N, N)); st = dict(n_matched=None)
    else:
        r = gotcha.gotcha(ref, img, exclude=exclude)
        if r is None or r["n_matched"] < 0.05 * r["n_grid"]: stats[oid] = f"dropped: Gotcha matched {0 if r is None else r['n_matched']} points"; continue
        dx, dy = r["dx"], r["dy"]; st = dict(n_seeds=r["n_seeds"], n_matched=r["n_matched"], n_grid=r["n_grid"],
                                             median_quality=float(np.median(r["quality"])), mean_dx=float(dx.mean()), mean_dy=float(dy.mean()))
    # radiometric normalisation: passes differ in brightness and contrast (sun, season, dust, calibration); match the
    # aligned pass's mean and standard deviation to the reference over the jointly valid area, then apply that gain and
    # offset in the pass's own grid (the restoration model has no per-pass radiometric term)
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    wv = ndi.map_coordinates(img, [yy - dy, xx - dx], order=1, mode="nearest"); ok = np.ones((N, N), bool) if exclude is None else ~exclude
    ok[:MARGIN, :] = ok[-MARGIN:, :] = ok[:, :MARGIN] = ok[:, -MARGIN:] = False
    gain = float(ref[ok].std() / (wv[ok].std() + 1e-12)); off = float(ref[ok].mean() - gain * wv[ok].mean())
    img = gain * img + off; st.update(radiometric_gain=gain, radiometric_offset=off)
    frames.append(img); flows.append((dx, dy)); used.append(oid); stats[oid] = st; print(oid, st, flush=True)
t_align = time.time() - t0
sig = [blur[o]["sigma_px"] * SC for o in used]
t1 = time.time()
X, info = pde_tv.pde_tv(frames, flows, scale=SC, psf_sigma=sig, lam=a.lam, iters=a.iters, exclude=exclude)
t_restore = time.time() - t1; t2 = time.time()
bs = U.bootstrap(frames, flows, sig, B=a.boot, scale=SC, lam=a.lam, iters=a.iters, exclude=exclude)
t_boot = time.time() - t2
native = np.stack([ndi.gaussian_filter(m, SC) for m in bs["members"]]).std(0, ddof=1) * a.kappa
c = slice(MARGIN * SC, -MARGIN * SC)
bic = ndi.zoom(frames[0], SC, order=3, grid_mode=True, mode="nearest")
np.savez_compressed(f"{a.out}/chain.npz", restored=X[c, c].astype(np.float32), bicubic_ref=bic[c, c].astype(np.float32),
                    sigma_boot_hr=bs["std"][c, c], sigma_native_cal=native[c, c].astype(np.float32), members_mean=bs["mean"][c, c],
                    frames=np.stack(frames)[:, MARGIN:-MARGIN, MARGIN:-MARGIN].astype(np.float32),
                    flows_dx=np.stack([f[0] for f in flows])[:, MARGIN:-MARGIN, MARGIN:-MARGIN].astype(np.float32),
                    flows_dy=np.stack([f[1] for f in flows])[:, MARGIN:-MARGIN, MARGIN:-MARGIN].astype(np.float32))
s0 = float(np.std(X[c, c]))
meta = dict(site=a.site, reference=ref_id, tile_native_px=a.tile, y0=y0, x0=x0, origin_global=[int(g0l + y0), int(g0s + x0)], scale=SC,
            passes_used=used, pass_sigma_hr=sig, gotcha=stats, pass_weights=info["pass_weights"], lam=a.lam, iters=a.iters,
            bootstrap=dict(B=a.boot, draws=bs["draws"], kappa=a.kappa,
                           median_sigma_native_pct_of_scene_std=100 * float(np.median(native[c, c]) / s0),
                           p90_sigma_native_pct_of_scene_std=100 * float(np.percentile(native[c, c], 90) / s0)),
            exclude_rover=a.exclude_rover, seconds=dict(align=t_align, restore=t_restore, bootstrap=t_boot))
json.dump(meta, open(f"{a.out}/chain.json", "w"), indent=1); print(json.dumps({k: meta[k] for k in ("passes_used", "pass_weights", "bootstrap", "seconds")}, indent=1))

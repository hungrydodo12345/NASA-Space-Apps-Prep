"""CPU benchmark of four restoration methods (docs/METHODS.md Section 9). Everything runs on CPU in this repository.

Methods
  raw       one HiRISE pass, bicubic x4
  current   previous pipeline: tile-wise ECC alignment + CG Tikhonov fusion, one assumed blur (1.07 px)
  gpt       our re-implementation of Tao & Muller 2016: Gotcha alignment + unweighted PDE-TV, assumed Gaussian blur 1 px
  gpt+weights  ablation: gpt with our pass/pixel weighting only
  gpt+blur     ablation: gpt with our measured per-pass blur only
  full-resw    ablation: full with residual-based pass weights (the first version; collapses onto one pass on real data)
  full      this work: Gotcha + noise-weighted, change-masked PDE-TV with each pass's measured blur
  (oracle   full with the true motion; simulation only, an upper bound for alignment)

Tests
  A. Simulation with exact truth. Fractal ground, 8 passes with the measured per-pass blur, a smooth non-rigid jitter
     field per pass (0.3 px amplitude) on top of a random offset, 3 % noise, one pass with a planted 2 m change.
     Every method estimates its own motion from the images. Metrics: error vs truth at the 6.25 cm output pixel and at
     the 25 cm native pixel (% of truth contrast), the artefact at the planted change (x truth std), the resolved
     scale (finest scale with truth coherence >= 0.5), and CPU seconds. 3 scenes.
  B. Real HiRISE, split-half consistency. On a real chain tile (scripts/run_chain.py), the passes are split into two
     disjoint halves, each restored independently by each method. Consistency between halves per scale (Fourier ring
     coherence) gives the finest scale both halves agree on: detail that repeats across independent passes is
     measured, not invented. ('raw' compares two different single passes.)
  C. Real HiRISE, rover edge. On the landing tile (rover parked, sols 1-11), the width of the rover's edge in each
     method's restoration (blurred-bar fit, as Section 3), together with the ringing it costs (RMS residual of the
     bar fit and the largest undershoot beside the edge, both as a share of the rover's brightness step). Edge width
     alone rewards over-deconvolution (assuming more blur than the data has), so both are reported.
Usage: python benchmark/run_benchmark.py <out_json> [--site-chain DIR] [--landing-chain DIR] [--scenes 3] [--quick]
"""
import sys, os, json, time, argparse, math, numpy as np
from scipy import ndimage as ndi, optimize
from scipy.special import ndtr
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from experiments import scenes
from pipeline import fuse_flow, pde_tv, gotcha, metrics

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--site-chain"); ap.add_argument("--landing-chain")
ap.add_argument("--scenes", type=int, default=3); ap.add_argument("--quick", action="store_true")
ap.add_argument("--parts", default="A,B,C", help="which tests to run; each part is saved as it finishes and merged into <out>")
a = ap.parse_args()
SC = 4; SIG = [0.889, 0.587, 0.828, 1.336, 1.426, 1.381, 1.16, 1.203]; N_LR = 48 if a.quick else 80; ITERS = 60 if a.quick else 300
LAM = 0.005                                                  # simulation (tuned there)
LAM_REAL = float(os.environ.get("LAM_REAL", 0.04))           # real HiRISE: chosen on three independent real datasets (tuning tile split-half, leave-one-pass-out, Ingenuity truth)
METHODS_SIM = ("raw", "current", "gpt", "gpt+weights", "gpt+blur", "full-resw", "full", "oracle")
METHODS_REAL = ("raw", "current", "gpt", "gpt+weights", "gpt+blur", "full-resw", "full")


def align_current(frames):
    return [(np.zeros(frames[0].shape),) * 2] + [fuse_flow.dense_shift_field(frames[0], f, tile=32, step=16) for f in frames[1:]]


def align_gotcha(frames):
    out = [(np.zeros(frames[0].shape),) * 2]
    for f in frames[1:]:
        r = gotcha.gotcha(frames[0], f)
        out.append((r["dx"], r["dy"]) if r is not None else (np.zeros(f.shape),) * 2)
    return out


def restore(method, frames, flows, sig_lr, lam=None):
    lam = LAM if lam is None else lam
    if method == "raw": return ndi.zoom(frames[0], SC, order=3, grid_mode=True, mode="nearest")
    if method == "current": return fuse_flow.fuse_flow(frames, flows, scale=SC, psf_sigma=1.07 * SC, mu=0.3, iters=60)[0]
    if method == "gpt": return pde_tv.pde_tv(frames, flows, scale=SC, psf_sigma=1.0 * SC, lam=lam, iters=ITERS, weighted=False)[0]
    if method == "gpt+weights": return pde_tv.pde_tv(frames, flows, scale=SC, psf_sigma=1.0 * SC, lam=lam, iters=ITERS, weighted=True, pass_weights="noise")[0]
    if method == "gpt+blur": return pde_tv.pde_tv(frames, flows, scale=SC, psf_sigma=[s * SC for s in sig_lr], lam=lam, iters=ITERS, weighted=False)[0]
    if method == "full-resw": return pde_tv.pde_tv(frames, flows, scale=SC, psf_sigma=[s * SC for s in sig_lr], lam=lam, iters=ITERS, pass_weights="residual")[0]
    if method in ("full", "oracle"): return pde_tv.pde_tv(frames, flows, scale=SC, psf_sigma=[s * SC for s in sig_lr], lam=lam, iters=ITERS, pass_weights="noise")[0]


def resolved_cm(a_, b_, px_cm=6.25):
    coh = metrics.radial_coherence(a_, b_, bins=32); f = metrics.effective_cutoff(coh)
    return (px_cm / f if f > 0 else None), coh.tolist()          # half-period of the finest coherent band


# ---------------- A. simulation
def sim_scene(seed):
    rng = np.random.default_rng(seed); n = N_LR * SC
    truth = scenes.fractal(3000 + seed, n=n); truth = truth[0] if isinstance(truth, tuple) else truth
    frames, true_flows = [], []
    yy, xx = np.mgrid[0:N_LR, 0:N_LR].astype(float)
    for k, s in enumerate(SIG):
        if k == 0: dx = dy = np.zeros((N_LR, N_LR))
        else:
            ox, oy = rng.uniform(-2, 2, 2); ph = rng.uniform(0, 2 * np.pi, 2)
            dx = ox + 0.3 * np.sin(2 * np.pi * yy / N_LR + ph[0]); dy = oy + 0.3 * np.sin(2 * np.pi * xx / N_LR + ph[1])
        Y, X = np.mgrid[0:n, 0:n].astype(float)
        dxh = ndi.zoom(dx, SC, order=1, grid_mode=True, mode="nearest") * SC; dyh = ndi.zoom(dy, SC, order=1, grid_mode=True, mode="nearest") * SC
        warped = ndi.map_coordinates(truth, [Y + dyh, X + dxh], order=3, mode="reflect")     # pass(x) = ref(x + d)
        frames.append(ndi.gaussian_filter(warped, s * SC)[::SC, ::SC]); true_flows.append((dx, dy))
    s0 = frames[0].std(); frames = [f + rng.normal(0, 0.03 * s0, f.shape) for f in frames]
    cy, cx = N_LR // 3, N_LR // 3; frames[3] = frames[3] + 4 * s0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2))
    return truth, frames, true_flows, (cy * SC, cx * SC)


def test_sim():
    res = {m: dict(err_6cm=[], err_25cm=[], artefact=[], resolved_cm=[], seconds=[]) for m in METHODS_SIM}
    for sd in range(a.scenes):
        truth, frames, tf, (cy, cx) = sim_scene(sd); s0 = truth.std(); m = 8 * SC; c = (slice(m, -m), slice(m, -m))
        t = time.time(); fc = align_current(frames); t_cur = time.time() - t
        t = time.time(); fg = align_gotcha(frames); t_got = time.time() - t
        for meth in METHODS_SIM:
            fl, ta = {"raw": (tf, 0.0), "current": (fc, t_cur), "oracle": (tf, 0.0)}.get(meth, (fg, t_got))
            t = time.time(); X = restore(meth, frames, fl, SIG); el = time.time() - t + ta
            e6 = 100 * float(np.sqrt(np.mean((X - truth)[c] ** 2)) / s0)
            g = lambda z: ndi.gaussian_filter(z, SC); e25 = 100 * float(np.sqrt(np.mean((g(X) - g(truth))[c] ** 2)) / s0)
            w = 16; art = float(np.abs(X - truth)[cy - w:cy + w, cx - w:cx + w].max() / s0)
            rc, _ = resolved_cm(X[c], truth[c])
            for k, v in (("err_6cm", e6), ("err_25cm", e25), ("artefact", art), ("resolved_cm", rc), ("seconds", el)): res[meth][k].append(v)
            print(f"scene {sd} {meth:8s} err6 {e6:5.1f}% err25 {e25:5.1f}% artefact {art:.2f} resolved {rc} cm  {el:.0f}s", flush=True)
    out = {m: {k: dict(mean=float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None, values=v) for k, v in r.items()} for m, r in res.items()}
    # paired differences vs the UCL re-implementation (per scene), with a bootstrap 95 % interval over scenes
    rng = np.random.default_rng(0); out["paired_vs_gpt"] = {}
    for m in METHODS_SIM:
        if m == "gpt": continue
        for k in ("err_6cm", "err_25cm", "artefact"):
            dlt = np.array(res[m][k]) - np.array(res["gpt"][k]); bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(4000)]
            out["paired_vs_gpt"][f"{m} - gpt, {k}"] = dict(mean=float(dlt.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], per_scene=dlt.tolist(),
                                                         scenes_better=int((dlt < 0).sum()), scenes=len(dlt))
    return out


# ---------------- B. split-half on real data
def test_split(chain_dir):
    d = np.load(f"{chain_dir}/chain.npz"); meta = json.load(open(f"{chain_dir}/chain.json"))
    blur = json.load(open(os.path.join(ROOT, "data_products", "pass_blur.json")))
    frames = list(d["frames"].astype(float)); used = meta["passes_used"]; sig = [blur[o]["sigma_px"] for o in used]
    if len(frames) < 4: return dict(skipped=f"only {len(frames)} passes")
    fg = list(zip(d["flows_dx"].astype(float), d["flows_dy"].astype(float))); fc = align_current(frames)
    A, B = list(range(0, len(frames), 2)), list(range(1, len(frames), 2))
    out = dict(passes=used, half_A=[used[i] for i in A], half_B=[used[i] for i in B])
    m = 8 * SC; c = (slice(m, -m), slice(m, -m))
    for meth in METHODS_REAL:
        fl = fc if meth == "current" else fg
        pick = lambda idx: ([frames[i] for i in idx], [fl[i] for i in idx], [sig[i] for i in idx])
        if meth == "raw":                                      # two different single passes, both brought onto the reference grid
            from pipeline.confidence import align_passes
            al = align_passes(frames, fg); XA = ndi.zoom(al[A[0]], SC, order=3, grid_mode=True); XB = ndi.zoom(al[B[0]], SC, order=3, grid_mode=True)
        else:
            fa, la, sa = pick(A); fb, lb, sb = pick(B)
            # half B has no zero-motion pass; the model handles that (all flows are to the reference grid)
            XA = restore(meth, fa, la, sa, LAM_REAL); XB = restore(meth, fb, lb, sb, LAM_REAL)
        rc, coh = resolved_cm(XA[c], XB[c]); out[meth] = dict(consistent_down_to_cm=rc, coherence_by_band=coh); print("split-half", meth, rc, flush=True)
    return out


# ---------------- C. rover edge on the landing tile
def bar(t, b0, b1, amp, t1, t2, s): return b0 + b1 * t + amp * (ndtr((t - t1) / s) - ndtr((t - t2) / s))


def edge_sigma_cm(img, px_cm):
    sm = ndi.gaussian_filter(img, 1); c = img.shape[0] // 2; h = int(round(150 / px_cm))
    y0, x0 = np.unravel_index(np.argmax(sm[c - h:c + h, c - h:c + h]), (2 * h, 2 * h)); y0 += c - h; x0 += c - h
    bg = np.median(img); pk = sm[y0, x0]; lab, _ = ndi.label(sm > bg + 0.5 * (pk - bg)); obj = lab == lab[y0, x0]
    yy, xx = np.nonzero(obj); cy, cx = yy.mean(), xx.mean(); ev, evec = np.linalg.eigh(np.cov(np.stack([yy - cy, xx - cx])))
    u, v = evec[:, 0], evec[:, 1]; L = 200 / px_cm; sig, ring, under = [], [], []
    for off in np.linspace(-37.5, 37.5, 7) / px_cm:
        t = np.arange(-L, L + 1e-9, 2.5 / px_cm); p = ndi.map_coordinates(img, [cy + off * v[0] + t * u[0], cx + off * v[1] + t * u[1]], order=3)
        try:
            pr, _ = optimize.curve_fit(bar, t, p, p0=[bg, 0, pk - bg, -60 / px_cm, 60 / px_cm, 25 / px_cm], maxfev=20000)
            if pr[4] - pr[3] > 50 / px_cm and pr[5] > 0:
                sig.append(abs(pr[5]) * px_cm); res = p - bar(t, *pr)
                ring.append(float(np.sqrt(np.mean(res ** 2)) / abs(pr[2])))          # ringing / artefacts around the edge
                out_ = (t < pr[3] - 2 * abs(pr[5])) | (t > pr[4] + 2 * abs(pr[5]))
                under.append(float(max(0.0, (pr[0] + pr[1] * t[out_] - p[out_]).max()) / abs(pr[2])) if out_.any() else 0.0)
        except Exception: pass
    if not sig: return dict(edge_sigma_cm=None, profiles=0)
    return dict(edge_sigma_cm=float(np.median(sig)), spread_cm=float(np.std(sig)), profiles=len(sig),
                ringing_rms_of_amplitude=float(np.median(ring)), max_undershoot_of_amplitude=float(np.median(under)))


def test_rover(chain_dir):
    d = np.load(f"{chain_dir}/chain.npz"); meta = json.load(open(f"{chain_dir}/chain.json"))
    blur = json.load(open(os.path.join(ROOT, "data_products", "pass_blur.json")))
    frames = list(d["frames"].astype(float)); used = meta["passes_used"]; sig = [blur[o]["sigma_px"] for o in used]
    fg = list(zip(d["flows_dx"].astype(float), d["flows_dy"].astype(float))); fc = align_current(frames)
    out = dict(passes=used, note="rover masked out of alignment and of the weighted restoration's data term in run_chain; "
                                 "here every method restores the rover itself so its edge can be measured")
    for meth in METHODS_REAL:
        fl = fc if meth == "current" else fg
        X = restore(meth, frames, fl, sig, LAM_REAL); out[meth] = edge_sigma_cm(X, 6.25); print("rover edge", meth, out[meth], flush=True)
    return out


if __name__ == "__main__":
    # each part is written to <out>.<part>.json as soon as it finishes, then all available parts are merged into <out>,
    # so parts can run as separate processes and a restart loses at most the part in progress
    parts = a.parts.split(","); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    jobs = {"A": ("A_simulation", lambda: test_sim()), "B": ("B_split_half_real", lambda: test_split(a.site_chain) if a.site_chain else None),
            "C": ("C_rover_edge_real", lambda: test_rover(a.landing_chain) if a.landing_chain else None)}
    for p_ in parts:
        key, fn = jobs[p_]; t0 = time.time(); r = fn()
        if r is not None: json.dump({key: r, "seconds": time.time() - t0}, open(f"{a.out}.{p_}.json", "w"), indent=1); print("wrote part", p_, flush=True)
    R = dict(setup=dict(cpu_count=os.cpu_count(), n_lr=N_LR, iters=ITERS, scenes=a.scenes, lam_sim=LAM, lam_real=LAM_REAL, per_pass_sigma_lr=SIG, quick=a.quick), cpu_seconds_by_part={})
    for p_, (key, _) in jobs.items():
        f = f"{a.out}.{p_}.json"
        if os.path.exists(f): d = json.load(open(f)); R[key] = d[key]; R["cpu_seconds_by_part"][p_] = d["seconds"]
    json.dump(R, open(a.out, "w"), indent=1); print("wrote", a.out)

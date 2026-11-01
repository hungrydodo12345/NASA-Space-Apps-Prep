"""Real multi-frame restoration around each rover stop: native HiRISE samples (no averaging, no resampling).

For every rover stop inside the hero area:
  * reference = ESP_045994 orthoimage window; other frames = ESP_046060 ortho + RDR observations cut at their native
    0.25 m sampling (sub-pixel phase preserved), locked to the reference with a coarse template match
  * dense per-pixel motion to the reference, photometric matching, quality gate
  * restorations at 6.31 cm: bicubic (1 frame), single-frame deconvolution, shift-and-average, multi-frame dense-motion
    CG (Tikhonov) - so the gain from deconvolution and from extra frames can be separated
Usage: python scripts/restore_real.py <hirise_jezero> <hirise_rdr> <hero_dir> <rover_truth_dir> <out_dir> [frames...]
"""
import sys, os, json, glob, time
import numpy as np, cv2, glymur
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from pipeline import multiframe as mf, fuse_flow, hero, groundtruth as gt
from pipeline.hirise_io import Rdr
from experiments import runner

JZ, RDR, HERO, TRUTH, OUT = sys.argv[1:6]
FRAMES = sys.argv[6:] or ["ESP_068822_1985", "ESP_068294_1985", "ESP_068360_1985", "ESP_068426_1985"]
os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f"{HERO}/measured_meta.json"))
PSF = json.load(open(os.path.join(ROOT, "data_products", "psf_calibration.json")))["sigma_px_median"]
SC, ORTHO_PX, RDR_PX = 4, 0.25241583171779, 0.25
K = ORTHO_PX / RDR_PX                                   # native RDR px per ortho px (1.00966)
N, MG = 128, 32                                          # restored core (32 m) + margin, in ortho px
L = N + 2 * MG
OA, OB = (glymur.Jp2k(f"{JZ}/{n}_RED_A_01_ORTHO.JP2") for n in ("ESP_045994_1985", "ESP_046060_1985"))
MU = float(os.environ.get("MU", 1e-2))
MU_GRID = [1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]


def lock_frames():
    """Coarse lock of each RDR to the ortho reference over the hero centre (as build_hero_measured)."""
    path = f"{OUT}/locks.json"
    if os.path.exists(path): return json.load(open(path))
    CL, CS = meta["centre"]; NN = meta["size"] + 192
    top, left = CL - NN // 2, CS - NN // 2
    ref = np.asarray(OA[top:top + NN, left:left + NN]).astype(float)
    lat, lon = hero.ortho_to_latlon(CL, CS); gl, gs = hero.latlon_to_rdr_global(lat, lon)
    DS = 4
    prep = lambda x: (lambda s: (s - ndi.gaussian_filter(s, 6)).astype(np.float32))(ndi.zoom(ndi.gaussian_filter(np.nan_to_num(x, nan=np.nanmean(x)), 2), 1 / DS, order=1))
    locks = {}
    for oid in FRAMES:
        r = Rdr(f"{RDR}/{oid}_RED"); W = int(NN * K) + 1600
        g0l, g0s = gl - W / 2, gs - W / 2
        big = np.nan_to_num(r.read_if(g0l, g0s, W, W)); big = np.where(big == 0, np.mean(big[big > 0]), big)
        bz = ndi.zoom(big, 1 / K, order=3)                # to ortho pitch only for the coarse lock
        c0 = NN // 2 - 512
        M = cv2.matchTemplate(prep(bz), prep(ref[c0:c0 + 1024, c0:c0 + 1024]), cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(M)
        t, l = loc[1] * DS - c0, loc[0] * DS - c0         # ortho-pitch coords of ref's origin inside bz
        locks[oid] = dict(score=float(mx), g0l=float(g0l), g0s=float(g0s), t=float(t), l=float(l), top=int(top), left=int(left))
        print(f"lock {oid}: score {mx:.2f}", flush=True)
    json.dump(locks, open(path, "w"), indent=1)
    return locks


def frame_window(oid, lk, ytop, xleft):
    """Native-sampling RDR window whose corner maps onto ortho pixel (ytop, xleft); same pixel count as the reference."""
    r = Rdr(f"{RDR}/{oid}_RED")
    ny = (lk["t"] + (ytop - lk["top"])) * K; nx = (lk["l"] + (xleft - lk["left"])) * K   # native offsets inside 'big'
    a = r.read_if(lk["g0l"] + round(ny), lk["g0s"] + round(nx), L, L)
    return np.nan_to_num(a, nan=np.nanmean(a))


def job(j):
    lk = LOCKS
    ytop, xleft = j["ytop"], j["xleft"]
    ref = np.asarray(OA[ytop:ytop + L, xleft:xleft + L]).astype(float)
    frames = {"ESP_045994 (ortho)": ref, "ESP_046060 (ortho)": np.asarray(OB[ytop:ytop + L, xleft:xleft + L]).astype(float)}
    for oid in FRAMES:
        if lk[oid]["score"] >= 0.35: frames[oid] = frame_window(oid, lk[oid], ytop, xleft)
    names = list(frames); stack = [frames[n] for n in names]
    flows = [(np.zeros((L, L)), np.zeros((L, L)))]; used = [stack[0]]; kept = [names[0]]; gate = {}
    yy, xx = np.mgrid[0:L, 0:L].astype(float)
    for n, f in zip(names[1:], stack[1:]):
        fl = mf.dense_flow_2pass(ref, f)
        fm = mf.photometric_match(ref, f, fl, m=MG)
        w = ndi.map_coordinates(fm, [yy - fl[1], xx - fl[0]], order=3, mode="nearest")
        a, b = mf.hp(ref)[MG:-MG, MG:-MG], mf.hp(w)[MG:-MG, MG:-MG]
        ncc = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12)); gate[n] = round(ncc, 3)
        if ncc >= 0.5: used.append(fm); flows.append(fl); kept.append(n)
    up = lambda a: ndi.zoom(a, SC, order=3, grid_mode=True, mode="nearest")
    core = lambda a: a[MG * SC:(MG + N) * SC, MG * SC:(MG + N) * SC]
    warped = [used[0]] + [ndi.map_coordinates(f, [yy - fl[1], xx - fl[0]], order=3, mode="nearest") for f, fl in zip(used[1:], flows[1:])]
    out = dict(bicubic=core(up(ref)), average=core(up(np.mean(warped, 0))))
    out["deconv_1frame"] = core(fuse_flow.fuse_flow([used[0]], [flows[0]], scale=SC, psf_sigma=PSF * SC, mu=MU, iters=40)[0])
    out["restored_multi"] = core(fuse_flow.fuse_flow(used, flows, scale=SC, psf_sigma=PSF * SC, mu=MU, iters=40)[0])
    np.savez_compressed(f"{OUT}/{j['stop']}.npz", **{k: v.astype(np.float32) for k, v in out.items()}, frames=np.array(kept), gate=json.dumps(gate),
                        ytop=ytop + MG, xleft=xleft + MG, px_cm=ORTHO_PX * 100 / SC)
    return dict(stop=j["stop"], frames_used=len(kept), gate=gate)


# ---------------------------------------------------------------------------------------------------------------------
# Cross-validation of the regularisation on REAL data: restore from all frames but one, then predict the held-out frame
# at its own sub-pixel phase (forward model: its flow + measured PSF + decimation). Amplified noise predicts badly;
# genuine sub-pixel detail predicts better than shift-and-average.
def cv_job(j):
    ytop, xleft = j["ytop"], j["xleft"]
    ref = np.asarray(OA[ytop:ytop + L, xleft:xleft + L]).astype(float)
    stack = [ref, np.asarray(OB[ytop:ytop + L, xleft:xleft + L]).astype(float)] + [frame_window(o, LOCKS[o], ytop, xleft) for o in FRAMES if LOCKS[o]["score"] >= 0.35]
    yy, xx = np.mgrid[0:L, 0:L].astype(float)
    used, flows = [ref], [(np.zeros((L, L)), np.zeros((L, L)))]
    for f in stack[1:]:
        fl = mf.dense_flow_2pass(ref, f); fm = mf.photometric_match(ref, f, fl, m=MG)
        w = ndi.map_coordinates(fm, [yy - fl[1], xx - fl[0]], order=3, mode="nearest")
        a, b = mf.hp(ref)[MG:-MG, MG:-MG], mf.hp(w)[MG:-MG, MG:-MG]
        if float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12)) >= 0.5: used.append(fm); flows.append(fl)
    c = slice(MG, -MG); res = {}
    for k in range(1, len(used)):
        tr = [i for i in range(len(used)) if i != k]
        act = used[k][c, c]
        avg = np.mean([ndi.map_coordinates(used[i], [yy - flows[i][1], xx - flows[i][0]], order=3, mode="nearest") for i in tr], 0)
        p_avg = ndi.map_coordinates(avg, [yy + flows[k][1], xx + flows[k][0]], order=3, mode="nearest")[c, c]
        res.setdefault("average", []).append(float(np.sqrt(np.mean((mf.hp(p_avg) - mf.hp(act)) ** 2))))
        for mu in MU_GRID:
            hr, _ = fuse_flow.fuse_flow([used[i] for i in tr], [flows[i] for i in tr], scale=SC, psf_sigma=PSF * SC, mu=mu, iters=40)
            pred = fuse_flow.FlowModel(ref.shape, [flows[k]], SC, PSF * SC).forward(hr, 0)[c, c]
            res.setdefault(f"mu={mu:g}", []).append(float(np.sqrt(np.mean((mf.hp(pred) - mf.hp(act)) ** 2))))
    return dict(stop=j["stop"], frames=len(used), errors=res)


# ---------------------------------------------------------------------------------------------------------------------
# Split-half resolution (Fourier ring correlation, FRC): restore each stop twice from DISJOINT frame sets and measure
# the finest scale at which the two independent results agree (FRC >= 1/7, the cryo-EM half-set criterion).
def frc(a, b, nb=40):
    n = a.shape[0]; w = np.outer(np.hanning(n), np.hanning(n))
    A = np.fft.fft2((a - a.mean()) * w); B = np.fft.fft2((b - b.mean()) * w)
    fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij"); fr = np.hypot(fx, fy)
    e = np.linspace(0, 0.5, nb + 1); f = 0.5 * (e[1:] + e[:-1]); out = []
    for i in range(nb):
        m = (fr >= e[i]) & (fr < e[i + 1])
        out.append(float(np.real((A[m] * np.conj(B[m])).sum()) / np.sqrt((np.abs(A[m]) ** 2).sum() * (np.abs(B[m]) ** 2).sum() + 1e-30)))
    return f, np.array(out)


def frc_res_cm(f, c, px_cm, thr=1 / 7):
    below = np.where(c < thr)[0]; fc = f[below[0]] if len(below) else 0.5
    return float(px_cm / (2 * fc))                    # half-period of the cut-off frequency


def halves_job(j):
    ytop, xleft = j["ytop"], j["xleft"]
    ref = np.asarray(OA[ytop:ytop + L, xleft:xleft + L]).astype(float)
    stack = [ref, np.asarray(OB[ytop:ytop + L, xleft:xleft + L]).astype(float)] + [frame_window(o, LOCKS[o], ytop, xleft) for o in FRAMES if LOCKS[o]["score"] >= 0.35]
    yy, xx = np.mgrid[0:L, 0:L].astype(float)
    used, flows = [ref], [(np.zeros((L, L)), np.zeros((L, L)))]
    for f in stack[1:]:
        fl = mf.dense_flow_2pass(ref, f); fm = mf.photometric_match(ref, f, fl, m=MG)
        w = ndi.map_coordinates(fm, [yy - fl[1], xx - fl[0]], order=3, mode="nearest")
        a, b = mf.hp(ref)[MG:-MG, MG:-MG], mf.hp(w)[MG:-MG, MG:-MG]
        if float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12)) >= 0.5: used.append(fm); flows.append(fl)
    if len(used) < 4: return dict(stop=j["stop"], skipped=True)
    H = [list(range(0, len(used), 2)), list(range(1, len(used), 2))]           # disjoint halves
    up = lambda a: ndi.zoom(a, SC, order=3, grid_mode=True, mode="nearest")
    core = lambda a: a[MG * SC:(MG + N) * SC, MG * SC:(MG + N) * SC]
    warp = lambda i: ndi.map_coordinates(used[i], [yy - flows[i][1], xx - flows[i][0]], order=3, mode="nearest")
    out = {}
    for h, idx in enumerate(H):
        out[f"single_{h}"] = core(up(warp(idx[0])))
        out[f"average_{h}"] = core(up(np.mean([warp(i) for i in idx], 0)))
        out[f"deconv1_{h}"] = core(fuse_flow.fuse_flow([used[idx[0]]], [flows[idx[0]]], scale=SC, psf_sigma=PSF * SC, mu=MU, iters=40)[0])
        out[f"restored_{h}"] = core(fuse_flow.fuse_flow([used[i] for i in idx], [flows[i] for i in idx], scale=SC, psf_sigma=PSF * SC, mu=MU, iters=40)[0])
    px = ORTHO_PX * 100 / SC; res = {}
    for m in ("single", "average", "deconv1", "restored"):
        f, c = frc(out[f"{m}_0"], out[f"{m}_1"]); res[m] = dict(res_cm=frc_res_cm(f, c, px), curve=c.tolist())
    np.savez_compressed(f"{OUT}/halves_{j['stop']}.npz", **{k: v.astype(np.float32) for k, v in out.items()})
    return dict(stop=j["stop"], frames=len(used), halves=H, mu=MU, frc=res, f=f.tolist())


if __name__ == "__main__":
    LOCKS = lock_frames()
    jobs = []
    for f in sorted(glob.glob(f"{TRUTH}/*.npz")):
        d = np.load(f); L0, S0 = gt.latlon_to_ortho(float(d["lat"]), float(d["lon"]))
        ytop, xleft = int(round(L0)) - L // 2, int(round(S0)) - L // 2
        if not (meta["top"] + 200 < L0 < meta["top"] + meta["size"] - 200 and meta["left"] + 200 < S0 < meta["left"] + meta["size"] - 200): continue
        jobs.append(dict(id=os.path.basename(f)[:-4], stop=os.path.basename(f)[:-4], ytop=ytop, xleft=xleft))
    jobs = jobs[:int(os.environ.get("LIMIT", len(jobs)))]
    print(f"{len(jobs)} rover stops to restore with frames {FRAMES}", flush=True)
    if os.environ.get("MODE") == "halves":
        runner.run(halves_job, jobs, f"{OUT}/halves_log_mu{MU:g}.jsonl", procs=int(os.environ.get("PROCS", 4)))
    elif os.environ.get("MODE") == "cv":
        runner.run(cv_job, jobs, f"{OUT}/cv_log.jsonl", procs=int(os.environ.get("PROCS", 4)))
    else:
        runner.run(job, jobs, f"{OUT}/restore_log.jsonl", procs=int(os.environ.get("PROCS", 4)))


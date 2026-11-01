"""Real multi-frame restoration v2: any rover stop on the ortho grid, per-stop frame locks, more passes, corrected blur.

Differences from restore_real.py:
  * stops come from one or more truth directories (any stop inside the ESP_045994 ortho), not just the hero area;
  * each RDR is locked to the ortho reference separately at every stop (template match of a high-passed window), so
    pointing differences across a 2+ km traverse do not matter;
  * the frame list is configurable (FRAMES env, comma list) - default: the eight passes with sun azimuth 165-178 deg;
  * the blur defaults to the corrected 1.07 px (Section 4.8 of the paper; PSF env overrides);
  * one job writes both the full restoration ({stop}.npz) and the two disjoint half-set restorations (halves_{stop}.npz),
    with the same keys as restore_real.py, so rocks_real_matched.py and the FRC summary run unchanged.
Usage: python scripts/restore_real_v2.py <hirise_jezero> <hirise_rdr> <truth_dirs comma-separated> <out_dir>
       env: FRAMES, PSF (px, default 1.07), MU (default 0.3), PROCS, LIMIT, ONLY (comma list of stops)
"""
import sys, os, json, glob
import numpy as np, cv2, glymur
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from pipeline import multiframe as mf, fuse_flow, hero, groundtruth as gt
from pipeline.hirise_io import Rdr
from experiments import runner

JZ, RDR, TRUTHS, OUT = sys.argv[1:5]
DEFAULT = "ESP_068822_1985,ESP_068294_1985,ESP_068360_1985,ESP_068426_1985,ESP_073068_1985,PSP_002387_1985"
FRAMES = [f for f in os.environ.get("FRAMES", DEFAULT).split(",") if f]
PSF = float(os.environ.get("PSF", 1.07)); MU = float(os.environ.get("MU", 0.3))
SC, ORTHO_PX, RDR_PX = 4, 0.25241583171779, 0.25
K = ORTHO_PX / RDR_PX
N, MG = 128, 32; L = N + 2 * MG; SEARCH = 160              # lock search radius (native px)
os.makedirs(OUT, exist_ok=True)
_cache = {}


def ortho(name):
    if name not in _cache: _cache[name] = glymur.Jp2k(f"{JZ}/{name}_RED_A_01_ORTHO.JP2")
    return _cache[name]


def rdr(oid):
    if oid not in _cache: _cache[oid] = Rdr(f"{RDR}/{oid}_RED")
    return _cache[oid]


def hp(x, s=3.0):
    x = np.nan_to_num(x, nan=np.nanmean(x)); return (x - ndi.gaussian_filter(x, s)).astype(np.float32)


COARSE_T, COARSE_PAD, DS = 1024, 600, 4                     # coarse lock: 256 m template, +-150 m search, 4x decimated


def _prep_coarse(x):
    x = np.nan_to_num(x, nan=np.nanmean(x)); s = ndi.zoom(ndi.gaussian_filter(x, 2), 1 / DS, order=1)
    return (s - ndi.gaussian_filter(s, 6)).astype(np.float32)


def locked_window(oid, ref, ytop, xleft):
    """Native RDR L x L window that best matches the ortho reference window; returns (window, score).

    Stage 1 (coarse): a 1024-px ortho template around the stop is matched inside a decimated RDR window with +-600 px
    margin (the RDRs sit 16-100 m from the ortho grid). Stage 2 (fine): the L x L reference is matched at full
    resolution within +-SEARCH/4 px of the coarse position."""
    cy, cx = ytop + L // 2, xleft + L // 2
    tpl = np.asarray(ortho("ESP_045994_1985")[cy - COARSE_T // 2:cy + COARSE_T // 2, cx - COARSE_T // 2:cx + COARSE_T // 2]).astype(float)
    lat, lon = hero.ortho_to_latlon(cy, cx); gl, gs = hero.latlon_to_rdr_global(lat, lon)
    W = int(COARSE_T * K) + 2 * COARSE_PAD; g0l, g0s = gl - W / 2, gs - W / 2
    big = rdr(oid).read_if(g0l, g0s, W, W)
    if np.isfinite(big).mean() < 0.5: return None, 0.0
    big = np.where(np.isfinite(big), big, np.nanmean(big))
    bz = ndi.zoom(big, 1 / K, order=1)                                       # to ortho pitch
    M = cv2.matchTemplate(_prep_coarse(bz), _prep_coarse(tpl), cv2.TM_CCOEFF_NORMED); _, mx0, _, loc = cv2.minMaxLoc(M)
    ty, tx = loc[1] * DS, loc[0] * DS                                         # template origin inside bz (ortho px)
    # reference window origin inside bz, then a fine search around it
    ry, rx = ty + (COARSE_T // 2 - L // 2), tx + (COARSE_T // 2 - L // 2)
    s = SEARCH // 4
    y0, x0 = max(0, ry - s), max(0, rx - s)
    sub = bz[y0:ry + L + s, x0:rx + L + s]
    if sub.shape[0] < L or sub.shape[1] < L: return None, float(mx0)
    M2 = cv2.matchTemplate(hp(sub), hp(ref), cv2.TM_CCOEFF_NORMED); _, mx, _, loc2 = cv2.minMaxLoc(M2)
    oy, ox = (y0 + loc2[1]) * K, (x0 + loc2[0]) * K                           # native px offset inside 'big'
    win = rdr(oid).read_if(g0l + round(oy), g0s + round(ox), L, L)
    if not np.isfinite(win).all(): return None, float(mx)
    return win, float(mx)


def frc(a, b, nb=40):
    n = a.shape[0]; w = np.outer(np.hanning(n), np.hanning(n))
    A = np.fft.fft2((a - a.mean()) * w); B = np.fft.fft2((b - b.mean()) * w)
    fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij"); fr = np.hypot(fx, fy)
    edges = np.linspace(0, 0.5, nb + 1); c = []
    for i in range(nb):
        m = (fr >= edges[i]) & (fr < edges[i + 1])
        c.append(float(np.real((A[m] * np.conj(B[m])).sum()) / np.sqrt((np.abs(A[m]) ** 2).sum() * (np.abs(B[m]) ** 2).sum() + 1e-12)))
    return 0.5 * (edges[1:] + edges[:-1]), np.array(c)


def job(j):
    ytop, xleft = j["ytop"], j["xleft"]
    ref = np.asarray(ortho("ESP_045994_1985")[ytop:ytop + L, xleft:xleft + L]).astype(float)
    if ref.shape != (L, L) or (ref == 0).mean() > 0.01: return dict(stop=j["stop"], skipped="outside ortho")
    names = ["ESP_045994 (ortho)", "ESP_046060 (ortho)"]; stack = [ref, np.asarray(ortho("ESP_046060_1985")[ytop:ytop + L, xleft:xleft + L]).astype(float)]
    locks = {}
    for oid in FRAMES:
        try: w, sc = locked_window(oid, ref, ytop, xleft)
        except Exception as e: w, sc = None, -1.0
        locks[oid] = round(sc, 3)
        if w is not None and sc >= 0.35: names.append(oid); stack.append(w)
    yy, xx = np.mgrid[0:L, 0:L].astype(float)
    used, flows, kept, gate = [ref], [(np.zeros((L, L)), np.zeros((L, L)))], [names[0]], {}
    for n, f in zip(names[1:], stack[1:]):
        fl = mf.dense_flow_2pass(ref, f); fm = mf.photometric_match(ref, f, fl, m=MG)
        w = ndi.map_coordinates(fm, [yy - fl[1], xx - fl[0]], order=3, mode="nearest")
        a, b = mf.hp(ref)[MG:-MG, MG:-MG], mf.hp(w)[MG:-MG, MG:-MG]
        ncc = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12)); gate[n] = round(ncc, 3)
        if ncc >= 0.5: used.append(fm); flows.append(fl); kept.append(n)
    if len(used) < 4: return dict(stop=j["stop"], skipped="too few frames", locks=locks, gate=gate)
    up = lambda a: ndi.zoom(a, SC, order=3, grid_mode=True, mode="nearest")
    core = lambda a: a[MG * SC:(MG + N) * SC, MG * SC:(MG + N) * SC]
    warp = lambda i: ndi.map_coordinates(used[i], [yy - flows[i][1], xx - flows[i][0]], order=3, mode="nearest")
    fuse = lambda idx: core(fuse_flow.fuse_flow([used[i] for i in idx], [flows[i] for i in idx], scale=SC, psf_sigma=PSF * SC, mu=MU, iters=40)[0])
    allidx = list(range(len(used)))
    out = dict(bicubic=core(up(ref)), average=core(up(np.mean([warp(i) for i in allidx], 0))), deconv_1frame=fuse([0]), restored_multi=fuse(allidx))
    np.savez_compressed(f"{OUT}/{j['stop']}.npz", **{k: v.astype(np.float32) for k, v in out.items()}, frames=np.array(kept),
                        gate=json.dumps(gate), ytop=ytop + MG, xleft=xleft + MG, px_cm=ORTHO_PX * 100 / SC)
    H = [allidx[0::2], allidx[1::2]]; hv = {}
    for h, idx in enumerate(H):
        hv[f"single_{h}"] = core(up(warp(idx[0]))); hv[f"average_{h}"] = core(up(np.mean([warp(i) for i in idx], 0)))
        hv[f"deconv1_{h}"] = fuse([idx[0]]); hv[f"restored_{h}"] = fuse(idx)
    px = ORTHO_PX * 100 / SC; res = {}
    for m in ("single", "average", "deconv1", "restored"):
        f, c = frc(hv[f"{m}_0"], hv[f"{m}_1"]); below = np.where(c < 1 / 7)[0]
        res[m] = dict(res_cm=float(px / (2 * (f[below[0]] if len(below) else 0.5))), curve=c.tolist())
    np.savez_compressed(f"{OUT}/halves_{j['stop']}.npz", **{k: v.astype(np.float32) for k, v in hv.items()})
    return dict(stop=j["stop"], frames_used=len(used), kept=kept, locks=locks, gate=gate, psf=PSF, mu=MU, frc=res, f=f.tolist())


if __name__ == "__main__":
    jobs = []
    only = set(os.environ["ONLY"].split(",")) if os.environ.get("ONLY") else None
    for td in TRUTHS.split(","):
        for fp in sorted(glob.glob(f"{td}/*.npz")):
            stop = os.path.basename(fp)[:-4]
            if only and stop not in only: continue
            d = np.load(fp); L0, S0 = gt.latlon_to_ortho(float(d["lat"]), float(d["lon"]))
            jobs.append(dict(id=stop, stop=stop, ytop=int(round(L0)) - L // 2, xleft=int(round(S0)) - L // 2))
    jobs = jobs[:int(os.environ.get("LIMIT", len(jobs)))]
    print(f"{len(jobs)} stops, frames: 2 orthos + {FRAMES}, psf {PSF} px, mu {MU}", flush=True)
    runner.run(job, jobs, f"{OUT}/restore_log.jsonl")

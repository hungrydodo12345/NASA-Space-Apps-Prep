"""Run the grounded two-stage model anywhere HiRISE has repeat coverage - no rover imagery needed at the site.

Inputs at the site: only HiRISE RDR passes. Rover imagery is used solely through what was learned at Jezero:
the Stage-1 recogniser weights (scripts/train_recognizer.py) and the Stage-2 rock/texture library (built here from the
Jezero rover truth). Steps:
  1. reference pass = lowest emission angle; all passes locked to it (coarse phase correlation over the whole area);
  2. the area is cut into 32 m tiles (+8 m margins); per tile: dense motion, photometric match, quality gate,
     multi-frame restoration at 6.25 cm with the corrected 1.07 px blur (same settings as scripts/restore_real_v2.py);
  3. Stage 1: the three cross-validation recogniser models are averaged -> rock list and surface classes;
  4. trust: each tile is compared with the HiRISE appearance of the training stops (Mahalanobis distance of texture
     statistics + recogniser confidence); tiles beyond the 95th percentile of the training stops are flagged UNFAMILIAR
     and are not filled;
  5. Stage 2: grounded fill (rover-photo texture and rocks, exactly locked to the HiRISE measurement).
Outputs in <out_dir>: area.npz (images, maps, trust), rocks.csv (lat, lon, diameter class, score, trusted),
summary.json, overview.png.
Usage: python scripts/run_anywhere.py <lat> <lon> <size_m> <rdr_dir> <recognizer_dir> <restored_dir> <truth_dir> <out_dir>
       env: FRAMES (comma list of RDR ids; default the 8 Jezero passes that exist in rdr_dir), PROCS
"""
import sys, os, glob, json, math, pickle
import numpy as np, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from multiprocessing import Pool
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline import multiframe as mf, fuse_flow, hero, recognizer as rz, grounded_fill as gf
from pipeline.hirise_io import Rdr, parse_label
sv = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[7]]
from experiments import exp3_rocks as E
sys.argv = sv

LAT, LON, SIZE = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
RDR, REC, RESTORED, TRUTH, OUT = sys.argv[4:9]
os.makedirs(OUT, exist_ok=True)
PSF, MU, SC, N, MG = 1.07, 0.3, 4, 128, 32
L = N + 2 * MG; PX_CM = 25.0 / SC; R_MARS = hero.R_MARS
FR = [f for f in os.environ.get("FRAMES", "").split(",") if f] or sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{RDR}/*_RED.JP2"))


def emission(oid):
    t = open(f"{RDR}/{oid}_RED.LBL", errors="replace").read()
    import re
    m = re.search(r"EMISSION_ANGLE\s*=\s*([-+0-9.]+)", t); return float(m.group(1)) if m else 99.0


def global_to_latlon(gl, gs, scale=0.25, clat=15.0, clon=180.0):
    return math.degrees(-gl * scale / R_MARS), clon + math.degrees(gs * scale / (R_MARS * math.cos(math.radians(clat))))


FR = sorted(FR, key=emission); REF = FR[0]
cgl, cgs = hero.latlon_to_rdr_global(LAT, LON)
nt = max(1, int(round(SIZE / (N * 0.25)))); area = nt * N
g0l, g0s = cgl - area / 2, cgs - area / 2                 # area origin (global RDR grid, reference frame)


def lock_all():
    """Integer offsets of every pass relative to the reference over the whole area (+ margins)."""
    rdrs = [Rdr(f"{RDR}/{o}_RED") for o in FR]
    nc = int(2 ** np.ceil(np.log2(area + 2 * MG + 600)))
    off, resp = mf.coarse_offsets(rdrs, g0l + area / 2 - (area + 2 * MG) / 2, g0s + area / 2 - (area + 2 * MG) / 2, area + 2 * MG, nc=nc)
    return {o: dict(dx=d[0], dy=d[1], resp=float(r)) for o, d, r in zip(FR, off, resp)}


def tile_job(a):
    ty, tx, offs = a
    gl, gs = g0l + ty * N - MG, g0s + tx * N - MG
    frames = []
    for o in FR:
        w = Rdr(f"{RDR}/{o}_RED").read_if(gl - offs[o]["dy"], gs - offs[o]["dx"], L, L)
        if np.isfinite(w).mean() > 0.999: frames.append((o, np.nan_to_num(w, nan=np.nanmean(w))))
    if not frames or frames[0][0] != REF: return ty, tx, None
    ref = frames[0][1]; yy, xx = np.mgrid[0:L, 0:L].astype(float)
    used, flows, kept = [ref], [(np.zeros((L, L)), np.zeros((L, L)))], [REF]
    for o, f in frames[1:]:
        fl = mf.dense_flow_2pass(ref, f); fm = mf.photometric_match(ref, f, fl, m=MG)
        w = ndi.map_coordinates(fm, [yy - fl[1], xx - fl[0]], order=3, mode="nearest")
        a_, b_ = mf.hp(ref)[MG:-MG, MG:-MG], mf.hp(w)[MG:-MG, MG:-MG]
        if float((a_ * b_).sum() / np.sqrt((a_ * a_).sum() * (b_ * b_).sum() + 1e-12)) >= 0.5: used.append(fm); flows.append(fl); kept.append(o)
    up = lambda z: ndi.zoom(z, SC, order=3, grid_mode=True, mode="nearest")
    core = lambda z: z[MG * SC:(MG + N) * SC, MG * SC:(MG + N) * SC]
    warp = lambda i: ndi.map_coordinates(used[i], [yy - flows[i][1], xx - flows[i][0]], order=3, mode="nearest")
    fuse = lambda idx: core(fuse_flow.fuse_flow([used[i] for i in idx], [flows[i] for i in idx], scale=SC, psf_sigma=PSF * SC, mu=MU, iters=40)[0])
    prods = dict(bicubic=core(up(ref)), average=core(up(np.mean([warp(i) for i in range(len(used))], 0))),
                 deconv_1frame=fuse([0]), restored_multi=fuse(list(range(len(used)))))
    return ty, tx, dict(prods={k: v.astype(np.float32) for k, v in prods.items()}, kept=kept)


def feats(img, cp=None):
    h = img - ndi.gaussian_filter(img, 8); b = ndi.gaussian_filter(img, 8) - ndi.gaussian_filter(img, 32)
    f = [np.log(h.std() + 1e-9) - np.log(img.std() + 1e-9), np.log(b.std() + 1e-9) - np.log(img.std() + 1e-9),
         float(np.mean(np.abs(h) > 2 * h.std())), float(((h - h.mean()) ** 3).mean() / (h.std() ** 3 + 1e-12))]
    if cp is not None: f.append(float(cp.max(0).mean()))
    return np.array(f)


if __name__ == "__main__":
    print(f"site {LAT:.5f}, {LON:.5f}, {nt}x{nt} tiles of 32 m, passes {FR} (reference {REF})", flush=True)
    offs = lock_all(); print("locks:", {o: round(v['resp'], 2) for o, v in offs.items()}, flush=True)
    jobs = [(ty, tx, offs) for ty in range(nt) for tx in range(nt)]
    with Pool(int(os.environ.get("PROCS", 4))) as p: tiles = p.map(tile_job, jobs)
    models = []
    for f in sorted(glob.glob(f"{REC}/recognizer_fold*.pt")):
        m = rz.UNet(); m.load_state_dict(torch.load(f, map_location="cpu")); m.eval(); models.append(m)
    thr = float(np.mean([float(np.load(p)["thr"]) for p in glob.glob(f"{REC}/pred_*.npz")]))
    # training appearance distribution (features of training stops' restorations + recogniser confidence)
    lib_cache = f"{REC}/library_all.pkl"
    if os.path.exists(lib_cache): lib, ref_feats = pickle.load(open(lib_cache, "rb"))
    else:
        T, Rk, ref_feats = [], [], []
        for f in sorted(glob.glob(f"{RESTORED}/*.npz")):
            s = os.path.basename(f)[:-4]
            if s.startswith("halves_") or not os.path.exists(f"{TRUTH}/{s}.npz"): continue
            r = dict(np.load(f)); t = np.load(f"{TRUTH}/{s}.npz")
            X, _, _, M, rocks = rz.stop_arrays(r, t, E.detect, E.PX)
            if M.sum() < 2000: continue
            T.append(t); Rk.append(rocks)
            pc = np.mean([rz.predict(m, X)[1] for m in models], 0)
            ref_feats.append(feats(r["restored_multi"].astype(float), pc))
        lib = gf.build_library(T, Rk, E.PX, PSF * SC); ref_feats = np.array(ref_feats)
        pickle.dump((lib, ref_feats), open(lib_cache, "wb"))
    mu_f, cov = ref_feats.mean(0), np.cov(ref_feats.T) + 1e-6 * np.eye(ref_feats.shape[1]); icov = np.linalg.inv(cov)
    dtrain = np.sqrt(np.einsum("ij,jk,ik->i", ref_feats - mu_f, icov, ref_feats - mu_f)); dlim = float(np.percentile(dtrain, 95))
    n = nt * N * SC
    keys = ["bicubic", "restored_multi", "grounded"]; IM = {k: np.full((n, n), np.nan, np.float32) for k in keys}
    CP = np.zeros((rz.NCLS, n, n), np.float32); TRUST = np.zeros((nt, nt)); rocks_out = []; tinfo = []
    for ty, tx, res in tiles:
        if res is None: tinfo.append(dict(ty=ty, tx=tx, status="no coverage")); continue
        P = res["prods"]; X = np.stack([rz._std(P[k]) for k in rz.INPUTS])
        outs = [rz.predict(m, X) for m in models]; rp = np.mean([o[0] for o in outs], 0); cp = np.mean([o[1] for o in outs], 0)
        d = float(np.sqrt((feats(P["restored_multi"].astype(float), cp) - mu_f) @ icov @ (feats(P["restored_multi"].astype(float), cp) - mu_f)))
        ok = d <= dlim; TRUST[ty, tx] = d
        rocks = rz.decode(rp, thr)
        sl = (slice(ty * N * SC, (ty + 1) * N * SC), slice(tx * N * SC, (tx + 1) * N * SC))
        IM["bicubic"][sl] = P["bicubic"]; IM["restored_multi"][sl] = P["restored_multi"]; CP[:, sl[0], sl[1]] = cp
        if ok:
            g, info = gf.fill(P["restored_multi"].astype(float), cp, rocks, lib, PSF * SC, seed=ty * 97 + tx); IM["grounded"][sl] = g
        else: info = dict(lock_residual_rel=None)
        for y, x, D, s in rocks:
            gl_ = g0l + ty * N + y / SC; gs_ = g0s + tx * N + x / SC; la, lo = global_to_latlon(gl_, gs_)
            rocks_out.append((la, lo, D, s, ok))
        tinfo.append(dict(ty=ty, tx=tx, passes=res["kept"], familiarity_distance=d, trusted=ok, rocks=len(rocks), lock_residual_rel=info["lock_residual_rel"]))
        print(f"tile {ty},{tx}: {len(res['kept'])} passes, distance {d:.2f} (limit {dlim:.2f}) {'TRUSTED' if ok else 'UNFAMILIAR - not filled'}, {len(rocks)} rocks", flush=True)
    np.savez_compressed(f"{OUT}/area.npz", **IM, class_prob=CP.astype(np.float16), familiarity=TRUST, px_cm=PX_CM, origin_global=(g0l, g0s))
    with open(f"{OUT}/rocks.csv", "w") as fh:
        fh.write("lat,lon,diameter_class_cm,score,trusted\n")
        for la, lo, D, s, ok in rocks_out: fh.write(f"{la:.7f},{lo:.7f},{'25-40' if D < 50 else '>=50'},{s:.3f},{int(ok)}\n")
    json.dump(dict(lat=LAT, lon=LON, size_m=nt * 32, passes=FR, reference=REF, locks=offs, familiarity_limit=dlim, tiles=tinfo,
                   rocks=len(rocks_out), rocks_trusted=sum(r[4] for r in rocks_out)), open(f"{OUT}/summary.json", "w"), indent=1)
    # overview figure
    CL = np.array([[0.80, 0.68, 0.50], [0.55, 0.45, 0.40], [0.93, 0.80, 0.45], [0.35, 0.25, 0.25]])
    fig, ax = plt.subplots(1, 5, figsize=(24, 5.4))
    for a, (t_, im) in zip(ax, (("Raw HiRISE (reference pass)", IM["bicubic"]), ("Multi-frame restoration", IM["restored_multi"]),
                                 ("Stage 1: recognised surface + rocks", None), ("Stage 2: grounded fill (trusted tiles)", IM["grounded"]), ("Trust map", "trust"))):
        if im is None:
            rgb = np.tensordot(np.moveaxis(CP.astype(float), 0, -1), CL, axes=([2], [0])); a.imshow(np.clip(rgb, 0, 1))
            for la, lo, D, s, ok in rocks_out:
                gl_, gs_ = hero.latlon_to_rdr_global(la, lo); y, x = (gl_ - g0l) * SC, (gs_ - g0s) * SC
                a.plot(x, y, "o", ms=2.5 if D < 50 else 4.5, mfc="none", mec="#00e5ff" if D < 50 else "#ff1744", mew=0.8)
        elif isinstance(im, str):
            h = a.imshow(TRUST, cmap="RdYlGn_r", vmin=0, vmax=2 * dlim); plt.colorbar(h, ax=a, fraction=0.046, label="unfamiliarity (limit = %.1f)" % dlim)
        else:
            lo_, hi_ = np.nanpercentile(im, [1, 99]); a.imshow(im, cmap="gray", vmin=lo_, vmax=hi_)
        a.set_title(t_, fontsize=10); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"No rover imagery at this site: {LAT:.4f}N {LON:.4f}E, {nt * 32} m square, {len(FR)} HiRISE passes", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/overview.png", dpi=100)
    print(f"done: {len(rocks_out)} rocks ({sum(r[4] for r in rocks_out)} in trusted tiles) -> {OUT}", flush=True)

"""Rover ground truth v3: Navcam stereo onto the true terrain (plane sweep around the HiRISE DTM), full CAHVORE lens model.

Per rover stop (site_drive), every left/right Navcam pair is swept over heights around the HiRISE DTM surface; cells where
left and right agree (NCC >= 0.6) are kept, pairs are merged, and isolated height outliers are removed. Rover hardware is
removed three ways: AI4Mars null/rover pixels where a label exists, a 2 m exclusion radius around the rover-nav origin,
and the stereo test itself (hardware is not on the ground surface, so it does not correlate at ground heights).

Output per stop (same 320 x 320 north-up 5 cm grid and keys as v1, plus height and confidence):
  z (texture, z-scored), cls (surface class, NaN unknown), height (m, up positive, relative to the rover origin),
  conf (mean NCC), n_pairs, lat, lon, sol, n_images
Usage: python scripts/build_rover_truth_v3.py <navcam_dir> <navcam_right_dir> <meta_dir> <meta_right_dir> <ai4mars_labels_root>
       <waypoints.json> <hero_navcam.json> <stereo_pairs.json> <dtm.IMG> <out_dir>   [PROCS env, default 4]
"""
import sys, os, re, json, glob, collections
from multiprocessing import Pool
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline import rover_ortho as ro, rover_stereo as rs, rover_project as rp, detail as dt, groundtruth as gt
from pipeline.dtm import Dtm

NAV, NAVR, META, METAR, LAB, WP, ITEMS, PAIRS, DTM, OUT = sys.argv[1:11]
HALF = float(os.environ.get("HALF", 8.0)); RMAX = float(os.environ.get("RMAX", 8.0)); RES, RMIN = 0.05, 2.0
DZ = np.arange(-1.0, 1.0001, 0.02)          # height search around the DTM prior (m); rough ground needs +-1 m
os.makedirs(OUT, exist_ok=True)


def labels():
    lab = {}
    for f in glob.glob(f"{LAB}/**/*.png", recursive=True):
        kind = "NAV" if "/NAV/" in f else "GEO"
        lab.setdefault(re.sub(r"_merged\d+\.png$", "", os.path.basename(f)), (f, kind))
    return lab


def dtm_prior(lat, lon, NN, EE):
    """Prior ground height on the local grid, local-level z (down positive) relative to the rover origin."""
    d = Dtm(DTM); l0, s0 = gt.latlon_to_ortho(lat, lon)
    px = gt.ORTHO["scale"]; m = int(np.ceil(HALF / px)) + 8
    top, left = int(round(l0)) - m, int(round(s0)) - m
    h, _ = d.heights_for_ortho(top, left, 2 * m, 2 * m, step=4)            # every 4 ortho px = 1 DTM post
    ll = top + np.arange(0, 2 * m, 4) + 1.5; ss = left + np.arange(0, 2 * m, 4) + 1.5
    L = l0 - NN / px; S = s0 + EE / px                                      # north-up ortho: line grows southward
    li = np.interp(L.ravel(), ll, np.arange(len(ll))); si = np.interp(S.ravel(), ss, np.arange(len(ss)))
    hh = ndi.map_coordinates(h, [li, si], order=1, mode="nearest").reshape(NN.shape)
    h0 = ndi.map_coordinates(h, [[np.interp(l0, ll, np.arange(len(ll)))], [np.interp(s0, ss, np.arange(len(ss)))]], order=1)[0]
    return -(hh - h0)


def build(job):
    stop, ms, w, lab, pairs = job
    NN, EE = rs.grid(HALF, RES)
    prior = dtm_prior(w["lat"], w["lon"], NN, EE)
    res, used = [], []
    for m in ms:
        lid = m["imageid"]; p = pairs.get(lid)
        if not p or p["overlap"] < 0.5: continue
        fr = f"{NAVR}/{p['right']}.png"; mr = f"{METAR}/{p['right']}.json"
        if not (os.path.exists(fr) and os.path.exists(mr)): continue
        L = ro.load_gray(f"{NAV}/{lid}.png"); R = ro.load_gray(fr); mR = json.load(open(mr))
        if lid in lab:
            lf, kind = lab[lid]; Lb = np.asarray(Image.open(lf)); Lb = Lb[..., 0] if Lb.ndim == 3 else Lb
            Lb = np.asarray(Image.fromarray(Lb).resize(L.shape[::-1], Image.NEAREST))
            L = np.where(Lb == 255, np.nan, L)
        r = rs.sweep_pair(L, R, m, mR, prior, NN, EE, dz=DZ, rmin=RMIN, rmax=RMAX)
        if r is not None: res.append(r); used.append(m)
    f = rs.fuse_stop(res)
    if f is None: return stop, None
    # isolated height outliers: > 5 cm from the 9x9 median of confident neighbours
    hgt = f["height"]; ok = np.isfinite(hgt)
    med = ndi.generic_filter(np.where(ok, hgt, np.nan), np.nanmedian, size=9, mode="constant", cval=np.nan)
    bad = ok & (np.abs(hgt - med) > 0.05)
    for k in ("z", "height", "conf"): f[k][bad] = np.nan
    # off-ground structures (rover hardware, shadow edges, mismatches): > 0.60 m from the smooth ground surface (1.5 m median)
    hgt = f["height"]; ok = np.isfinite(hgt)
    surf = ndi.generic_filter(np.where(ok, hgt, np.nan), np.nanmedian, size=31, mode="constant", cval=np.nan)
    bad = ok & (np.abs(hgt - surf) > 0.60)   # boulders up to ~0.6 m tall survive
    # thin spokes / slivers (< ~25 cm across): not ground patches; rocks are compact and survive a 5 x 5 opening
    keep = ndi.binary_opening(ok & ~bad, structure=np.ones((5, 5)))
    for k in ("z", "height", "conf"): f[k][~keep] = np.nan
    verified = np.isfinite(f["height"])
    # ---- terrain mesh: stereo heights where verified, DTM prior + smoothed stereo residual elsewhere
    resid = np.where(verified, f["height"] - prior, 0.0); wgt = verified.astype(float)
    sm = ndi.gaussian_filter(resid, 10) / np.maximum(ndi.gaussian_filter(wgt, 10), 1e-6)
    base = np.nanmedian((f["height"] - prior)[verified]) if verified.any() else 0.0
    support = ndi.uniform_filter(wgt, 31)                                  # share of verified cells within ~1.5 m
    mesh = prior + np.where(ndi.gaussian_filter(wgt, 10) > 0.05, sm, base)
    mesh = np.where(verified, f["height"], mesh)
    offground = ndi.binary_dilation(bad, iterations=3)
    allowed = (support >= 0.15) & ~offground
    # ---- texture: every left image projected onto the mesh (the rover photo seen from straight above)
    orths = []
    rng_ = np.hypot(NN, EE)
    for m in used:
        L = ro.load_gray(f"{NAV}/{m['imageid']}.png")
        if m["imageid"] in lab:
            lf, kind = lab[m["imageid"]]; Lb = np.asarray(Image.open(lf)); Lb = Lb[..., 0] if Lb.ndim == 3 else Lb
            L = np.where(np.asarray(Image.fromarray(Lb).resize(L.shape[::-1], Image.NEAREST)) == 255, np.nan, L)
        cam = rs.parse_cahvore(m); Rm = rp.quat_to_R(rp.attitude(m))
        sel = allowed & (rng_ >= RMIN) & (rng_ <= RMAX)
        P = np.stack([NN[sel], EE[sel], mesh[sel]], -1) @ Rm
        o = np.full(NN.shape, np.nan); o[sel] = rs._sample(L, cam, P); orths.append(o)
    proj = rs.blend_photos(orths)
    if proj is not None:
        f["z_stereo_only"] = f["z"]; f["z"] = np.where(allowed, proj, np.nan)
        f["height"] = np.where(np.isfinite(f["z"]), mesh, np.nan)
    f["verified"] = verified
    # surface class: sample each labelled image at the fused height (nearest), majority vote
    clss = []
    for m in used:
        if m["imageid"] not in lab: continue
        lf, kind = lab[m["imageid"]]; Lb = np.asarray(Image.open(lf)); Lb = Lb[..., 0] if Lb.ndim == 3 else Lb
        cls_img = dt.label_to_class(Lb, kind)
        cam = rs.parse_cahvore(m); Rm = rp.quat_to_R(rp.attitude(m))
        dw, dh = [float(t) for t in m["extended"]["dimension"].strip("()").split(",")]
        okc = np.isfinite(f["height"])
        P = np.stack([NN[okc], EE[okc], f["height"][okc]], -1) @ Rm
        x, y, fr_ = rs.project(cam, P)
        u = np.round(x * Lb.shape[1] / dw).astype(int); v = np.round(y * Lb.shape[0] / dh).astype(int)
        inn = fr_ & (u >= 0) & (u < Lb.shape[1]) & (v >= 0) & (v < Lb.shape[0])
        c = np.full(NN.shape, np.nan); cc = np.full(okc.sum(), np.nan); cc[inn] = cls_img[v[inn], u[inn]]; c[okc] = cc
        clss.append(c)
    cls = dt.class_mosaic(clss) if clss else np.full(NN.shape, np.nan)
    cls[~np.isfinite(f["z"])] = np.nan
    np.savez_compressed(f"{OUT}/{stop}.npz", z=f["z"].astype(np.float32), cls=cls.astype(np.float32),
                        height=(-f["height"]).astype(np.float32), conf=f["conf"].astype(np.float32),
                        n_pairs=f["n_pairs"].astype(np.int16), lat=w["lat"], lon=w["lon"], sol=w["sol"], n_images=len(used),
                        verified=f["verified"], z_stereo_only=f.get("z_stereo_only", f["z"]).astype(np.float32))
    return stop, dict(pairs=len(used), coverage=float(np.isfinite(f["z"]).mean()), verified=float(f["verified"].mean()),
                      conf=float(np.nanmedian(f["conf"])),
                      height_iqr_m=float(np.nanpercentile(-f["height"], 75) - np.nanpercentile(-f["height"], 25)))


if __name__ == "__main__":
    lab = labels(); pairs = json.load(open(PAIRS))
    wp = {f["properties"]["RMC"]: f["properties"] for f in json.load(open(WP))["features"]}
    by = collections.defaultdict(list)
    for iid, url, el, sol in json.load(open(ITEMS)):
        if os.path.exists(f"{NAV}/{iid}.png"):
            m = json.load(open(f"{META}/{iid}.json")); by[f'{m["site"]}_{m["drive"]}'].append(m)
    jobs = [(s, ms, wp[s], lab, pairs) for s, ms in sorted(by.items()) if s in wp]
    if os.environ.get("ONLY"): jobs = [j for j in jobs if j[0] in os.environ["ONLY"].split(",")]
    if os.environ.get("SKIP_DONE"): jobs = [j for j in jobs if not os.path.exists(f"{OUT}/{j[0]}.npz")]
    print(len(jobs), "stops", flush=True)
    log = open(f"{OUT}/build_log.jsonl", "a")
    with Pool(int(os.environ.get("PROCS", 4))) as pool:
        for stop, info in pool.imap_unordered(build, jobs):
            print(stop, info, flush=True); log.write(json.dumps(dict(stop=stop, **(info or {"skipped": True}))) + "\n"); log.flush()

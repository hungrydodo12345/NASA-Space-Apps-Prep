"""Collect candidate ground-texture patches from SINGLE rover photos (no mosaic seams) with quality features.

Usage: python scripts/collect_texture_candidates.py <navcam_dir> <meta_dir> <ai4mars_labels_root> <waypoints.json> <hero_navcam.json> <out_npz>
Patches are 40x40 px at 5 cm (2 m x 2 m), centred 2-6 m from the rover (ground the camera sees at a useful angle).
"""
import sys, os, re, json, glob, collections
import numpy as np
from multiprocessing import Pool
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline import rover_ortho as ro, detail as dt, texture_quality as tq

NAV, META, LAB, WP, ITEMS, OUT = sys.argv[1:7]
lab = {}
for f in glob.glob(f"{LAB}/**/*.png", recursive=True):
    lab.setdefault(re.sub(r"_merged\d+\.png$", "", os.path.basename(f)), (f, "NAV" if "/NAV/" in f else "GEO"))
wp = {f["properties"]["RMC"]: f["properties"] for f in json.load(open(WP))["features"]}
jobs = []
for iid, url, el, sol in json.load(open(ITEMS)):
    if os.path.exists(f"{NAV}/{iid}.png") and iid in lab:
        m = json.load(open(f"{META}/{iid}.json")); jobs.append((m, lab[iid]))
P, STRIDE, HALF = 40, 20, 8.0


def work(job):
    m, (lf, kind) = job
    img = ro.load_gray(f"{NAV}/{m['imageid']}.png")
    L = np.asarray(Image.open(lf)); L = L[..., 0] if L.ndim == 3 else L
    L = np.asarray(Image.fromarray(L).resize(img.shape[::-1], Image.NEAREST))
    img = np.where(L == 255, np.nan, img)
    o = ro.project_ground(m, img, half=HALF, res=0.05, rmin=1.5, rmax=6.5)
    c = ro.project_ground(m, dt.label_to_class(L, kind), half=HALF, res=0.05, rmin=1.5, rmax=6.5, order=0)
    n = o.shape[0]; out = []
    for y in range(0, n - P + 1, STRIDE):
        for x in range(0, n - P + 1, STRIDE):
            p = o[y:y + P, x:x + P]; cl = c[y:y + P, x:x + P]
            if np.isfinite(p).mean() < 0.995 or np.isfinite(cl).mean() < 0.95: continue
            r = float(np.hypot(y + P / 2 - n / 2, x + P / 2 - n / 2) * 0.05)
            if not (2.0 <= r <= 6.0): continue
            v = cl[np.isfinite(cl)].astype(int); cnt = np.bincount(v, minlength=4); k = int(cnt.argmax())
            if cnt[k] / v.size < 0.8: continue
            f = tq.features(np.nan_to_num(p, nan=np.nanmean(p)))
            out.append(dict(stop=f"{m['site']}_{m['drive']}", img=m["imageid"], r=r, cls=k, patch=np.nan_to_num(p).astype(np.float32), **f))
    return out


if __name__ == "__main__":
    with Pool(2) as pool: res = pool.map(work, jobs, chunksize=8)
    rows = [r for rr in res for r in rr]
    keys = ["r", "cls", "peak", "kurt", "step", "coh_fine", "coh_coarse", "satur", "mean", "std"]
    np.savez_compressed(OUT, patches=np.stack([r["patch"] for r in rows]), stop=np.array([r["stop"] for r in rows]), img=np.array([r["img"] for r in rows]),
                        **{k: np.array([r[k] for r in rows]) for k in keys})
    print(f"{len(jobs)} photos -> {len(rows)} candidate patches at {len(set(r['stop'] for r in rows))} stops; per class:",
          {dt.CLASSES[k]: int(sum(r['cls'] == k for r in rows)) for k in range(4)})

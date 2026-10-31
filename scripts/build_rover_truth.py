"""Rover ground truth per stop: top-down 5 cm image (z-scored mosaic) + surface-class map.

Usage: python scripts/build_rover_truth.py <navcam_dir> <meta_dir> <ai4mars_labels_root> <waypoints.json> <hero_navcam.json> <out_dir>
"""
import sys, os, re, json, glob, collections
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import rover_ortho as ro, detail as dt

NAV, META, LAB, WP, ITEMS, OUT = sys.argv[1:7]
os.makedirs(OUT, exist_ok=True)
lab = {}
for f in glob.glob(f"{LAB}/**/*.png", recursive=True):
    kind = "NAV" if "/NAV/" in f else "GEO"
    lab.setdefault(re.sub(r"_merged\d+\.png$", "", os.path.basename(f)), (f, kind))
wp = {f["properties"]["RMC"]: f["properties"] for f in json.load(open(WP))["features"]}
by = collections.defaultdict(list)
for iid, url, el, sol in json.load(open(ITEMS)):
    if os.path.exists(f"{NAV}/{iid}.png") and iid in lab:
        m = json.load(open(f"{META}/{iid}.json")); by[f'{m["site"]}_{m["drive"]}'].append(m)
for stop, ms in sorted(by.items()):
    w = wp.get(stop)
    if not w: continue
    orths, clss = [], []
    for m in ms:
        img = ro.load_gray(f"{NAV}/{m['imageid']}.png")
        lf, kind = lab[m["imageid"]]
        L = np.asarray(Image.open(lf)); L = L[..., 0] if L.ndim == 3 else L
        L = np.asarray(Image.fromarray(L).resize(img.shape[::-1], Image.NEAREST))
        img = np.where(L == 255, np.nan, img)
        orths.append(ro.project_ground(m, img))
        clss.append(ro.project_ground(m, dt.label_to_class(L, kind), order=0))
    z = ro.mosaic(orths)
    if z is None or np.isfinite(z).mean() < 0.3: continue
    c = dt.class_mosaic(clss); c[~np.isfinite(z)] = np.nan
    np.savez_compressed(f"{OUT}/{stop}.npz", z=z.astype(np.float32), cls=c.astype(np.float32), lat=w["lat"], lon=w["lon"], sol=w["sol"], n_images=len(ms))
    share = [float(np.nanmean(c == k)) for k in range(4)]
    print(f"{stop:8s} sol {w['sol']:3d}: {len(ms):3d} images, coverage {100*np.isfinite(z).mean():.0f}%, classes " +
          " ".join(f"{n} {100*s:.0f}%" for n, s in zip(dt.CLASSES, share)), flush=True)

"""Project AI4Mars Perseverance labels onto 1 m HiRISE ortho cells. Usage: python scripts/build_surface_groundtruth.py <data_dir> NAV|GEO"""
import os
import sys, json, glob, os, re
import numpy as np
from PIL import Image
S = sys.argv[1]; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline import rover_project as rp, groundtruth as gt
KIND = sys.argv[2]                                   # NAV | GEO
root = f"{S}/ai4mars/ai4mars-dataset-merged-0.6/m2020/labels/" + ("NAV" if KIND == "NAV" else "M2020_GEO/ncam")
groups = gt.NAV_GROUPS if KIND == "NAV" else gt.GEO_GROUPS
names = list(groups)
lut = np.full(256, -1, int)
for gi, (g, vals) in enumerate(groups.items()):
    for v in vals: lut[v] = gi
wp = {f["properties"]["RMC"]: f["properties"] for f in json.load(open(f"{S}/rover/waypoints.json"))["features"]}
tr = gt.Traverse(f"{S}/rover/traverse.json")
P = gt.ORTHO; CELL = 4                                # 4 ortho px = ~1 m cells
acc = {}
used = 0
for png in glob.glob(f"{root}/*.png"):
    iid = re.sub(r"_merged\d+\.png$", "", os.path.basename(png))
    mp = f"{S}/rover/meta/{iid}.json"
    if not os.path.exists(mp): continue
    m = json.load(open(mp))
    lab = np.asarray(Image.open(png)); lab = lab[..., 0] if lab.ndim == 3 else lab
    vv, uu = np.mgrid[0:lab.shape[0]:3, 0:lab.shape[1]:3]
    cls = lut[lab[vv, uu]].ravel(); keep = cls >= 0
    if keep.sum() == 0: continue
    try: n, e = rp.ground_points(m, uu.ravel()[keep].astype(float), vv.ravel()[keep].astype(float), lab.shape)
    except Exception: continue
    w = wp.get(f'{m["site"]}_{m["drive"]}')
    lon, lat = (w["lon"], w["lat"]) if w else tr.position(float(m["extended"]["sclk"]))
    L0, S0 = gt.latlon_to_ortho(lat, lon)
    ok = np.isfinite(n)
    li = np.floor((L0 - n[ok] / P["scale"]) / CELL).astype(int); si = np.floor((S0 + e[ok] / P["scale"]) / CELL).astype(int)
    for a, b, c in zip(li, si, cls[keep][ok]):
        acc.setdefault((a, b), np.zeros(len(names)))[c] += 1
    used += 1
cells = np.array(list(acc.keys())); counts = np.array(list(acc.values()))
np.savez(f"{S}/rover/projected_{KIND}.npz", cells=cells, counts=counts, names=np.array(names))
print(f"{KIND}: images used {used}, labelled 1 m ground cells {len(cells)}, total samples {int(counts.sum())}")
print("class share:", dict(zip(names, np.round(counts.sum(0) / counts.sum(), 3))))

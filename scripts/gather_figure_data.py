"""Gather the small image crops the paper figures need, from local HiRISE RDRs and a chain directory, into benchmark/results/figdata/.
1. boulder_passes.npz: the 32 m boulder tile in all 8 passes, each warped onto the reference grid with its Gotcha flow (from <boulder_chain>/chain.npz).
2. hardware_crops.npz: 40 m crops at the parachute and the backshell in 8 passes (integer locks from <site_summary>), reference-frame coordinates found by
   the sol-11 vs 2021-04-01 change search (two strongest disappearing/ changing bright objects).
Usage: python scripts/gather_figure_data.py <rdr_dir> <boulder_chain_dir> <site_summary.json>"""
import sys, os, json, numpy as np
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline.hirise_io import Rdr
RDR, CH, SUM = sys.argv[1:4]; OUT = os.path.join(ROOT, "benchmark", "results", "figdata"); os.makedirs(OUT, exist_ok=True)
d = np.load(f"{CH}/chain.npz"); meta = json.load(open(f"{CH}/chain.json")); used = meta["passes_used"]; FR = d["frames"].astype(float)
H, W = FR[0].shape; yy, xx = np.mgrid[0:H, 0:W].astype(float)
al = np.array([ndi.map_coordinates(FR[i], [yy - d["flows_dy"][i], xx - d["flows_dx"][i]], order=3, mode="nearest") for i in range(len(used))], np.float32)
dates = {o: next(l.split("=")[1].strip()[:10] for l in open(f"{RDR}/{o}_RED.LBL") if l.strip().startswith("START_TIME")) for o in used}
np.savez_compressed(f"{OUT}/boulder_passes.npz", aligned=al, passes=np.array(used), dates=np.array([dates[o] for o in used]), origin_global=np.array(meta["origin_global"]))
LK = json.load(open(SUM))["locks"]; C = {"parachute": (-4372517, -23480471), "backshell": (-4372385, -23480563)}; N = 160; crops = {}
for name, (gl, gs) in C.items():
    crops[name] = np.array([Rdr(f"{RDR}/{o}_RED").read_if(gl - N // 2 - LK[o]["dy"], gs - N // 2 - LK[o]["dx"], N, N) for o in used], np.float32)
np.savez_compressed(f"{OUT}/hardware_crops.npz", passes=np.array(used), dates=np.array([dates[o] for o in used]), centres=np.array(list(C.values())), names=np.array(list(C)), **crops)
print("wrote", OUT, dates)

"""Build a site directory (summary.json + area.npz) for scripts/run_chain.py at an arbitrary global-grid position: per-pass integer locks by phase correlation of a 400 m window
(sign confirmed by the correlation of the re-read window), relative to the reference pass. The tile is 128 native px (32 m) centred on (gl, gs).
Usage: python scripts/make_site.py <rdr_dir> <gl> <gs> <reference_pass> <pass1,pass2,...> <out_dir>"""
import sys, json, numpy as np, cv2, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline.hirise_io import Rdr
RDR, GL, GS, REF, PASSES, OUT = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5].split(","), sys.argv[6]; N = 1600
hp = lambda a: (a - cv2.GaussianBlur(a, (0, 0), 6)).astype(np.float32); win = cv2.createHanningWindow((N, N), cv2.CV_32F)
def rd(o, dy=0, dx=0):
    w = Rdr(f"{RDR}/{o}_RED").read_if(GL - N // 2 - dy, GS - N // 2 - dx, N, N); return np.nan_to_num(w, nan=float(np.nanmedian(w)))
ref = rd(REF); locks = {REF: dict(dx=0, dy=0, resp=1.0)}
for o in PASSES:
    if o == REF: continue
    p = rd(o); (sx, sy), resp = cv2.phaseCorrelate(hp(ref), hp(p), win); best = None
    for sgn in (1, -1):
        dx, dy = int(round(sgn * sx)), int(round(sgn * sy)); q = rd(o, dy, dx); c = np.corrcoef(hp(ref)[200:-200, 200:-200].ravel(), hp(q)[200:-200, 200:-200].ravel())[0, 1]
        if best is None or c > best[0]: best = (c, dx, dy)
    locks[o] = dict(dx=best[1], dy=best[2], resp=float(best[0])); print(o, "lock", best[1], best[2], "corr", round(float(best[0]), 3), flush=True)
os.makedirs(OUT, exist_ok=True); json.dump(dict(passes=[REF] + [p for p in PASSES if p != REF], reference=REF, locks=locks, tile_origin=[GL - 64, GS - 64]), open(f"{OUT}/summary.json", "w"), indent=1)
np.savez(f"{OUT}/area.npz", bicubic=np.zeros((512, 512), np.float32), origin_global=np.array([GL - 64, GS - 64], float))

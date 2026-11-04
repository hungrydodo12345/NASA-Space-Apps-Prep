"""Refine weak per-pass locks of a site directory by a local brute-force search of the high-passed correlation around hints from another site (their locks are relative to a different
reference, converted through the shared pass). Usage: python scripts/refine_locks.py <rdr_dir> <site_dir> <hint_site_dir> <min_corr> [radius_px]"""
import sys, json, numpy as np, cv2, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline.hirise_io import Rdr
RDR, SITE, HINT, MINC = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]); RAD = int(sys.argv[5]) if len(sys.argv) > 5 else 40; N = 700
sm = json.load(open(f"{SITE}/summary.json")); hm = json.load(open(f"{HINT}/summary.json")); ref = sm["reference"]; GL, GS = np.load(f"{SITE}/area.npz")["origin_global"] + 64
hp = lambda a: (a - cv2.GaussianBlur(a, (0, 0), 6)).astype(np.float32)
def read_big(o, dy, dx, R): w = Rdr(f"{RDR}/{o}_RED").read_if(GL - N // 2 - dy - R, GS - N // 2 - dx - R, N + 2 * R, N + 2 * R); return hp(np.nan_to_num(w, nan=float(np.nanmedian(w))))
def rd(o, dy, dx): w = Rdr(f"{RDR}/{o}_RED").read_if(GL - N // 2 - dy, GS - N // 2 - dx, N, N); return hp(np.nan_to_num(w, nan=float(np.nanmedian(w))))
R0 = rd(ref, 0, 0)[100:-100, 100:-100]
for o, lk in sm["locks"].items():
    if o == ref or lk["resp"] >= MINC: continue
    hint = hm["locks"][o]; hr = hm["locks"][ref]; cand = (hint["dx"] - hr["dx"], hint["dy"] - hr["dy"]) if o in hm["locks"] and ref in hm["locks"] else (lk["dx"], lk["dy"]); best = (-9, 0, 0); R = RAD + 6
    for cdx, cdy in ((cand[0], cand[1]), (lk["dx"], lk["dy"])):
        big = read_big(o, cdy, cdx, R)                                   # one read; every trial shift is an in-memory crop
        def at(ddy, ddx): return big[R - ddy:R - ddy + N, R - ddx:R - ddx + N][100:-100, 100:-100]
        cb = (-9, 0, 0)
        for ddy in range(-RAD, RAD + 1, 3):
            for ddx in range(-RAD, RAD + 1, 3):
                c = np.corrcoef(R0.ravel(), at(ddy, ddx).ravel())[0, 1]
                if c > cb[0]: cb = (c, ddy, ddx)
        for ddy in range(cb[1] - 2, cb[1] + 3):
            for ddx in range(cb[2] - 2, cb[2] + 3):
                c = np.corrcoef(R0.ravel(), at(ddy, ddx).ravel())[0, 1]
                if c > cb[0]: cb = (c, ddy, ddx)
        if cb[0] > best[0]: best = (cb[0], cdx + cb[2], cdy + cb[1])
    print(f"{o}: old lock ({lk['dx']},{lk['dy']}) corr {lk['resp']:.3f} -> new ({best[1]},{best[2]}) corr {best[0]:.3f}  [hint {cand}]", flush=True)
    if best[0] > lk["resp"]: sm["locks"][o] = dict(dx=int(best[1]), dy=int(best[2]), resp=float(best[0]))
json.dump(sm, open(f"{SITE}/summary.json", "w"), indent=1)

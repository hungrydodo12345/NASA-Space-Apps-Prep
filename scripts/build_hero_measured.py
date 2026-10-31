"""Build the MEASURED imagery layer for a hero area by fusing lighting-matched HiRISE observations.

Usage: python scripts/build_hero_measured.py <hirise_jezero_dir> <hirise_rdr_dir> <out_dir> [centre_line centre_sample size]
Reference grid = ESP_045994 orthoimage (DTM-controlled; same frame as the rover's waypoints).
"""
import sys, os, json, time
import numpy as np, cv2, glymur
from scipy import ndimage as ndi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import multiframe as mf, hero, register
from pipeline.hirise_io import Rdr

JZ, RDR, OUT = sys.argv[1:4]
CL, CS, N = (int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (24934, 15331, 2560)
M = 96                                                     # margin for flow / borders
TOP, LEFT = CL - N // 2 - M, CS - N // 2 - M
NN = N + 2 * M
ORTHO_PX, RDR_PX = 0.25241583171779, 0.25
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
ref = np.asarray(glymur.Jp2k(f"{JZ}/ESP_045994_1985_RED_A_01_ORTHO.JP2")[TOP:TOP + NN, LEFT:LEFT + NN]).astype(np.float64)
frames = {"ESP_045994 (ortho, reference)": ref,
          "ESP_046060 (ortho)": np.asarray(glymur.Jp2k(f"{JZ}/ESP_046060_1985_RED_A_01_ORTHO.JP2")[TOP:TOP + NN, LEFT:LEFT + NN]).astype(np.float64)}
lat, lon = hero.ortho_to_latlon(CL, CS)
gl, gs = hero.latlon_to_rdr_global(lat, lon)
DS = 4
def prep(x):
    x = np.nan_to_num(x, nan=np.nanmean(x)); s = ndi.zoom(ndi.gaussian_filter(x, 2), 1 / DS, order=1)
    return (s - ndi.gaussian_filter(s, 6)).astype(np.float32)
for oid in ("ESP_042315_1985", "ESP_068822_1985", "ESP_068294_1985", "ESP_068360_1985", "ESP_068426_1985"):
    r = Rdr(f"{RDR}/{oid}_RED")
    W = int(NN * ORTHO_PX / RDR_PX) + 1600
    big = r.read_if(gl - W / 2, gs - W / 2, W, W)
    big = np.nan_to_num(big, nan=np.nanmean(big))
    big = ndi.zoom(big, RDR_PX / ORTHO_PX, order=3)        # to ortho pitch
    c0 = NN // 2 - 512                                       # lock on the central ~260 m (robust to edges)
    Mt = cv2.matchTemplate(prep(big), prep(ref[c0:c0 + 1024, c0:c0 + 1024]), cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(Mt)
    t, l = loc[1] * DS - c0, loc[0] * DS - c0
    if mx < 0.35 or t < 0 or l < 0:
        print(f"{oid}: lock score {mx:.2f} too weak or off-image - excluded"); continue
    win = big[t:t + NN, l:l + NN]
    if win.shape != ref.shape: print(f"{oid}: window off-image, excluded"); continue
    frames[oid.replace("_1985", "") + " (RDR)"] = win
    print(f"{oid}: locked to reference grid (template score {mx:.2f}) [{time.time()-t0:.0f}s]", flush=True)
names = list(frames); stack = [frames[k] for k in names]
# --- per tile: dense motion to reference, photometric match, warp
T = 512
warped = [np.full_like(ref, np.nan) for _ in stack]; warped[0] = ref.copy()
yy, xx = np.mgrid[0:T + 2 * M, 0:T + 2 * M].astype(np.float64)
gate = []
for ty in range(M, M + N, T):
    for tx in range(M, M + N, T):
        sl = (slice(ty - M, ty + T + M), slice(tx - M, tx + T + M))
        rt = ref[sl]
        for k in range(1, len(stack)):
            ft = stack[k][sl]
            fx, fy = mf.dense_flow_2pass(rt, ft)
            fm = mf.photometric_match(rt, ft, (fx, fy), m=M)
            w = ndi.map_coordinates(fm, [yy - fy, xx - fx], order=3, mode="nearest")
            ha, hb = mf.hp(rt)[M:M + T, M:M + T], mf.hp(w)[M:M + T, M:M + T]
            ncc = float((ha * hb).sum() / np.sqrt((ha * ha).sum() * (hb * hb).sum() + 1e-12))
            if ncc >= 0.5:                                     # quality gate: only frames that truly agree
                warped[k][ty:ty + T, tx:tx + T] = w[M:M + T, M:M + T]
            gate.append((names[k], ty - M, tx - M, round(ncc, 3)))
        print(f"  tile ({ty-M},{tx-M}) done [{time.time()-t0:.0f}s]", flush=True)
W = np.stack([w[M:M + N, M:M + N] for w in warped])
cnt = np.isfinite(W).sum(0)
srt = np.sort(np.where(np.isfinite(W), W, np.inf), axis=0)  # gated-out frames (NaN) sorted to the end
lo = (cnt >= 4).astype(int)                                 # trim brightest & darkest when >= 4 frames agree
idx = np.arange(W.shape[0])[:, None, None]
use = (idx >= lo) & (idx < cnt - lo)
vals = np.where(use, srt, 0.0)
fused = vals.sum(0) / np.maximum(use.sum(0), 1)
spread = np.sqrt(np.maximum((vals ** 2).sum(0) / np.maximum(use.sum(0), 1) - fused ** 2, 0))
np.save(f"{OUT}/measured_count.npy", cnt.astype(np.uint8))
json.dump(gate, open(f"{OUT}/gate.json", "w"))
print("frames per pixel: mean %.1f, min %d" % (cnt.mean(), cnt.min()))
np.save(f"{OUT}/measured_fused.npy", fused.astype(np.float32))
np.save(f"{OUT}/measured_single.npy", W[0].astype(np.float32))
np.save(f"{OUT}/measured_spread.npy", spread.astype(np.float32))
np.save(f"{OUT}/measured_stack.npy", W.astype(np.float32))
json.dump(dict(frames=names, centre=[CL, CS], size=N, top=CL - N // 2, left=CS - N // 2, pixel_m=ORTHO_PX), open(f"{OUT}/measured_meta.json", "w"))
print(f"done: {len(names)} frames fused over {N*ORTHO_PX:.0f} m [{time.time()-t0:.0f}s]")

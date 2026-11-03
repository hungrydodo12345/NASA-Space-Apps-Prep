import sys, json, numpy as np, cv2
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep")
from pipeline.hirise_io import Rdr
S = sys.argv[1]; g0l, g0s = -4371755.17190706, -23476249.22764247; cl, cs = g0l + 64, g0s + 64; k = 4
cy, cx = 1310, 423; gl_c = cl - 6000 + cy * k; gs_c = cs - 6000 + cx * k; gl0, gs0 = gl_c - 160, gs_c - 160
bl, bs = gl0 + 235, gs0 + 87                                    # backshell in the global grid (from the 80 m crop, 4x zoom: x=350/4, y=940/4)
tl, ts = int(round(bl - 64)), int(round(bs - 64))              # 128 px (32 m) tile centred on the backshell
passes = ["ESP_068294_1985", "ESP_068360_1985", "ESP_068426_1985", "ESP_068822_1985", "ESP_073068_1985"]
N = 1600; hp = lambda a: (a - cv2.GaussianBlur(a, (0, 0), 6)).astype(np.float32); win = cv2.createHanningWindow((N, N), cv2.CV_32F)
def rd(o, dy=0, dx=0):
    w = Rdr(f"{S}/hirise_rdr/{o}_RED").read_if(bl - N // 2 - dy, bs - N // 2 - dx, N, N); return np.nan_to_num(w, nan=np.nanmedian(w))
ref = rd(passes[0]); locks = {passes[0]: dict(dx=0, dy=0, resp=1.0)}
for o in passes[1:]:
    p = rd(o); (sx, sy), resp = cv2.phaseCorrelate(hp(ref), hp(p), win)
    best = None
    for sgn in (1, -1):                                        # confirm the sign convention by the correlation of the re-read window
        dx, dy = int(round(sgn * sx)), int(round(sgn * sy)); q = rd(o, dy, dx); c = np.corrcoef(hp(ref)[200:-200, 200:-200].ravel(), hp(q)[200:-200, 200:-200].ravel())[0, 1]
        if best is None or c > best[0]: best = (c, dx, dy)
    locks[o] = dict(dx=best[1], dy=best[2], resp=float(best[0])); print(o, "lock", best[1], best[2], "corr", round(float(best[0]), 3), "phase resp", round(resp, 3), flush=True)
json.dump(dict(lat=18.4, lon=77.4, size_m=32, passes=passes, reference=passes[0], locks=locks, tile_origin=[tl, ts], backshell_global=[bl, bs]), open(f"{S}/debris_site/summary.json", "w"), indent=1)
np.savez(f"{S}/debris_site/area.npz", bicubic=np.zeros((512, 512), np.float32), origin_global=np.array([tl, ts], float))
print("tile origin", tl, ts)

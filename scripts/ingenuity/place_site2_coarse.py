"""Site 2 placement, step 1: coarse 1 m band-passed correlation of the flights 4-5 mosaic (25 cm version) against several HiRISE passes, 4 orientations, +-500 m around the waypoint landing coordinates. Usage: python place_site2_coarse.py <scratch> <mosaic_25cm.npy> [res_m]"""
import sys, json, numpy as np, cv2
from scipy import ndimage as ndi
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep")
from pipeline.hirise_io import Rdr
S = sys.argv[1]; MOS = sys.argv[2]; LK = json.load(open(f"{S}/rover_all/summary.json"))["locks"]; RES = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
k = 3394839.8133 * np.pi / 180 / 0.25; ks = k * np.cos(np.radians(15)); gl0, gs0 = -18.44462715 * k, (77.45088572 - 180) * ks
f = int(RES / 0.25); M = np.load(MOS); ysc = np.cos(np.radians(15)) / np.cos(np.radians(18.4446))
M = cv2.resize(np.nan_to_num(M, nan=0), None, fx=0.25 / RES * ysc, fy=0.25 / RES, interpolation=cv2.INTER_AREA); mm = cv2.resize((np.isfinite(np.load(MOS))).astype(np.float32), M.shape[::-1], interpolation=cv2.INTER_AREA) > 0.9
def bp(a, m, lo=1.5, hi=8):
    a = np.where(m, a, a[m].mean()); return np.where(m, ndi.gaussian_filter(a, lo) - ndi.gaussian_filter(a, hi), 0).astype(np.float32)
W = 4000; res = {}
for o in ["ESP_073068_1985", "ESP_045994_1985", "PSP_002387_1985", "ESP_068294_1985"]:
    L = LK[o]; H = Rdr(f"{S}/hirise_rdr/{o}_RED").read_if(int(gl0) - W // 2 - L["dy"], int(gs0) - W // 2 - L["dx"], W, W); hm = np.isfinite(H); H = np.where(hm, H, np.nanmedian(H))
    Hs = H.reshape(W // f, f, W // f, f).mean((1, 3)); Hb = bp(Hs, np.ones_like(Hs, bool)); res[o] = {}
    for fl in (0, 1):
        for fs in (0, 1):
            T = M[::-1] if fl else M; T = T[:, ::-1] if fs else T; tm = mm[::-1] if fl else mm; tm = tm[:, ::-1] if fs else tm
            Tb = bp(T, tm); c = cv2.matchTemplate(Hb, Tb, cv2.TM_CCORR_NORMED, mask=tm.astype(np.float32)); c = np.nan_to_num(c)
            y, x = np.unravel_index(np.argmax(c), c.shape); cc = c.copy(); cc[max(0, y - 15):y + 15, max(0, x - 15):x + 15] = 0
            res[o][f"{fl}{fs}"] = (float(c[y, x]), float(cc.max()), int(y), int(x))
    print(o, {kk: tuple(round(v, 3) if isinstance(v, float) else v for v in vv) for kk, vv in res[o].items()}, flush=True)
json.dump(dict(res=res, res_m=RES, window=W, gl0=gl0, gs0=gs0), open(f"place_coarse_{RES}.json", "w"), indent=1)

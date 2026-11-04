"""Site 2 placement, step 2: refine the mosaic north-west corner at 25 cm (bands 0.5-2 m and 0.25-1 m) per pass. Usage: python place_site2_refine.py <scratch> <glNW> <gsNW>"""
import sys, json, numpy as np, cv2
from scipy import ndimage as ndi
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep")
from pipeline.hirise_io import Rdr
S = sys.argv[1]; glNW, gsNW = int(sys.argv[2]), int(sys.argv[3]); LK = json.load(open(f"{S}/rover_all/summary.json"))["locks"]
M = np.load("f45_25cm.npy"); ysc = np.cos(np.radians(15)) / np.cos(np.radians(18.4446)); m0 = np.isfinite(M)
M = cv2.resize(np.nan_to_num(M, nan=0), None, fx=ysc, fy=1, interpolation=cv2.INTER_LINEAR); m = cv2.resize(m0.astype(np.float32), M.shape[::-1]) > 0.99
m = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
def bp(a, mm, lo, hi): a = np.where(mm, a, a[mm].mean()); return np.where(mm, ndi.gaussian_filter(a, lo) - ndi.gaussian_filter(a, hi), 0).astype(np.float32)
P = 48; out = {}
for o in ["ESP_073068_1985", "ESP_068822_1985", "ESP_068294_1985", "ESP_071077_1985"]:
    L = LK.get(o)
    if L is None: continue
    H = Rdr(f"{S}/hirise_rdr/{o}_RED").read_if(glNW - P - L["dy"], gsNW - P - L["dx"], M.shape[0] + 2 * P, M.shape[1] + 2 * P); H = np.where(np.isfinite(H), H, np.nanmedian(H))
    for band in ((2, 8), (1, 4)):
        c = cv2.matchTemplate(bp(H, np.ones_like(H, bool), *band), bp(M, m, *band), cv2.TM_CCORR_NORMED, mask=m.astype(np.float32)); c = np.nan_to_num(c)
        y, x = np.unravel_index(np.argmax(c), c.shape)
        # sub-pixel by parabola
        dy = 0.5 * (c[y - 1, x] - c[y + 1, x]) / (c[y - 1, x] - 2 * c[y, x] + c[y + 1, x]) if 0 < y < c.shape[0] - 1 else 0
        dx = 0.5 * (c[y, x - 1] - c[y, x + 1]) / (c[y, x - 1] - 2 * c[y, x] + c[y, x + 1]) if 0 < x < c.shape[1] - 1 else 0
        cc = c.copy(); cc[max(0, y - 8):y + 8, max(0, x - 8):x + 8] = 0
        out[f"{o}|{band}"] = dict(peak=float(c[y, x]), second=float(cc.max()), shift_lines=float(y + dy - P), shift_samples=float(x + dx - P))
        print(o, band, {kk: round(v, 3) for kk, v in out[f"{o}|{band}"].items()}, flush=True)
json.dump(dict(nw_coarse=[glNW, gsNW], refine=out), open("refine_f45.json", "w"), indent=1)

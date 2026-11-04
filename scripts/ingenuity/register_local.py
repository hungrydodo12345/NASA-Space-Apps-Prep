"""Local registration of the backshell neighbourhood: a +-HALF-metre patch of the Ingenuity mosaic around the backshell against HiRISE, with the translation pinned
to the known backshell position (+-1.5 m) and rotation 0-359 deg / scale scanned. Reports the NCC-versus-rotation curve and whether one rotation stands out.
Usage: python register_local.py <scratch_dir> <mosaic_npz> <backshell_x> <backshell_y> <out_json> [half_m]"""
import sys, os, json, numpy as np, cv2
from scipy.signal import fftconvolve
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep"); from pipeline.hirise_io import Rdr
S, MOS, BX, BY, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]; HALF = float(sys.argv[6]) if len(sys.argv) > 6 else 6.0
PXH, PSF, NW = 0.2524, 0.889, 200
site = json.load(open(f"{S}/debris_site/summary.json")); bl, bs = site["backshell_global"]
I = Rdr(f"{S}/hirise_rdr/ESP_068294_1985_RED").read_if(bl - NW // 2, bs - NW // 2, NW, NW); I = np.nan_to_num(I, nan=float(np.nanmedian(I))).astype(np.float32)
B0, B1 = float(os.environ.get("BAND_LO", 1.0)), float(os.environ.get("BAND_HI", 4.0)); band = lambda a: (cv2.GaussianBlur(a, (0, 0), B0) - cv2.GaussianBlur(a, (0, 0), B1)).astype(np.float32)
Ib = band(I); I2 = Ib * Ib; m = np.load(MOS); M0 = m["mosaic"].astype(np.float32); V0 = np.isfinite(M0); M0 = np.nan_to_num(M0, nan=1.0); mpp0 = float(m["m_per_px"])
hp = int(HALF / mpp0); y0, y1, x0, x1 = max(BY - hp, 0), min(BY + hp, M0.shape[0]), max(BX - hp, 0), min(BX + hp, M0.shape[1]); A = M0[y0:y1, x0:x1]; Vv = V0[y0:y1, x0:x1]; cbx, cby = BX - x0, BY - y0
def mncc(T, Mk):
    Tm = T * Mk; n = Mk.sum(); s_tm = Tm.sum(); denT = (Tm * T).sum() - s_tm ** 2 / n; c = lambda P, Q: fftconvolve(P, Q[::-1, ::-1], mode="valid")
    cI = c(Ib, Tm); cM = c(Ib, Mk); cI2 = c(I2, Mk); return (cI - cM * s_tm / n) / np.sqrt(np.maximum(cI2 - cM ** 2 / n, 1e-9) * max(denT, 1e-9))
res = []
for scale in [float(x) for x in os.environ.get("SCALES", "0.90,0.95,1.0,1.05").split(",")]:
    f = scale * mpp0 / PXH; a = cv2.resize(A, None, fx=f, fy=f, interpolation=cv2.INTER_AREA); vm = cv2.resize(Vv.astype(np.float32), a.shape[::-1], interpolation=cv2.INTER_AREA) > 0.95; px, py = cbx * f, cby * f
    for theta in range(0, 360, 2):
        H, W = a.shape; c = (W / 2, H / 2); Rm = cv2.getRotationMatrix2D(c, theta, 1.0); cs, sn = abs(Rm[0, 0]), abs(Rm[0, 1]); W2, H2 = int(H * sn + W * cs), int(H * cs + W * sn); Rm[0, 2] += W2 / 2 - c[0]; Rm[1, 2] += H2 / 2 - c[1]
        ar = cv2.warpAffine(a, Rm, (W2, H2)); vr = cv2.warpAffine(vm.astype(np.float32), Rm, (W2, H2)) > 0.99; xt, yt = Rm @ np.array([px, py, 1.0])
        T = band(cv2.GaussianBlur(np.where(vr, ar, float(np.median(a[vm]))), (0, 0), PSF)); vr = cv2.erode(vr.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        r = mncc(T, vr.astype(np.float32)); 
        # translation pinned: the template's backshell must land within 6 px (1.5 m) of the window centre
        yy, xx = np.mgrid[0:r.shape[0], 0:r.shape[1]]; X = xx + xt; Y = yy + yt; ok = np.hypot(X - NW / 2, Y - NW / 2) <= 6; rr = np.where(ok, r, -1)
        k = np.unravel_index(np.argmax(rr), rr.shape); res.append((float(rr[k]), theta, scale))
by_theta = {}
for v, th, sc in res: by_theta[th] = max(by_theta.get(th, -1), v)
th = np.array(sorted(by_theta)); v = np.array([by_theta[t] for t in th]); o = np.argsort(-v); best = th[o[0]]
far = [t for t in th[o] if abs(((t - best + 180) % 360) - 180) > 20][0]
print("patch half-size %.1f m; best rotation %d deg (NCC %.3f); best rotation >20 deg away: %d deg (NCC %.3f); median over rotations %.3f; 95th pct %.3f" % (HALF, best, v[o[0]], far, by_theta[far], np.median(v), np.percentile(v, 95)))
print("top 6 rotations:", [(int(th[i]), round(float(v[i]), 3)) for i in o[:6]])
json.dump(dict(half_m=HALF, best_theta=int(best), best_ncc=float(v[o[0]]), second_mode_theta=int(far), second_mode_ncc=float(by_theta[far]), median=float(np.median(v)), p95=float(np.percentile(v, 95)), curve=[[int(a_), float(b_)] for a_, b_ in zip(th, v)]), open(OUT, "w"), indent=1)

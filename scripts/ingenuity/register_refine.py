"""Fine registration of the Ingenuity backshell mosaic to a HiRISE pass: rotation, scale and translation (+-2 m around the backshell) by masked NCC in a band-passed view.
Each HiRISE pass is placed through its own lock (summary.json) so the same global-grid backshell position is used; the passes have independent noise,
so agreement of the optimum between passes is a validation. Usage: python register_refine.py <scratch_dir> <mosaic_npz> <pass_id> <half_m> <bx> <by> <out_json>"""
import sys, os, json, numpy as np, cv2
from scipy.signal import fftconvolve
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep"); from pipeline.hirise_io import Rdr
S, MOS, PASS, HALF, BX, BY, OUT = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
PXH, NW = 0.2524, 200; site = json.load(open(f"{S}/debris_site/summary.json")); bl, bs = site["backshell_global"]; lk = site["locks"].get(PASS, dict(dx=0, dy=0))
PSF = json.load(open("/home/user/NASA-Space-Apps-Prep/data_products/pass_blur.json"))[PASS]["sigma_px"]
I = Rdr(f"{S}/hirise_rdr/{PASS}_RED").read_if(bl - NW // 2 - lk["dy"], bs - NW // 2 - lk["dx"], NW, NW); I = np.nan_to_num(I, nan=float(np.nanmedian(I))).astype(np.float32)
band = lambda a: (cv2.GaussianBlur(a, (0, 0), 1.0) - cv2.GaussianBlur(a, (0, 0), 4.0)).astype(np.float32); Ib = band(I); I2 = Ib * Ib
m = np.load(MOS); M0 = m["mosaic"].astype(np.float32); V0 = np.isfinite(M0); M0 = np.nan_to_num(M0, nan=1.0); mpp0 = float(m["m_per_px"]); hp = int(HALF / mpp0)
y0, y1, x0, x1 = max(BY - hp, 0), min(BY + hp, M0.shape[0]), max(BX - hp, 0), min(BX + hp, M0.shape[1]); A = M0[y0:y1, x0:x1]; Vv = V0[y0:y1, x0:x1]; cbx, cby = BX - x0, BY - y0
def mncc(T, Mk):
    Tm = T * Mk; n = Mk.sum(); s_tm = Tm.sum(); denT = (Tm * T).sum() - s_tm ** 2 / n; c = lambda P, Q: fftconvolve(P, Q[::-1, ::-1], mode="valid")
    return (c(Ib, Tm) - c(Ib, Mk) * s_tm / n) / np.sqrt(np.maximum(c(I2, Mk) - c(Ib, Mk) ** 2 / n, 1e-9) * max(denT, 1e-9))
TH = np.arange(float(os.environ.get("TH0", 196)), float(os.environ.get("TH1", 232)) + 1e-9, 1.0); SCL = np.arange(0.0250, 0.0300 + 1e-9, 0.0005); best = (-1,); grid = {}
for sc in SCL:
    f = sc / PXH; a = cv2.resize(A, None, fx=f, fy=f, interpolation=cv2.INTER_AREA); vm = cv2.resize(Vv.astype(np.float32), a.shape[::-1], interpolation=cv2.INTER_AREA) > 0.95; px, py = cbx * f, cby * f
    for th in TH:
        H, W = a.shape; c = (W / 2, H / 2); Rm = cv2.getRotationMatrix2D(c, th, 1.0); cs, sn = abs(Rm[0, 0]), abs(Rm[0, 1]); W2, H2 = int(H * sn + W * cs), int(H * cs + W * sn); Rm[0, 2] += W2 / 2 - c[0]; Rm[1, 2] += H2 / 2 - c[1]
        ar = cv2.warpAffine(a, Rm, (W2, H2)); vr = cv2.warpAffine(vm.astype(np.float32), Rm, (W2, H2)) > 0.99; xt, yt = Rm @ np.array([px, py, 1.0])
        T = band(cv2.GaussianBlur(np.where(vr, ar, float(np.median(a[vm]))), (0, 0), PSF)); vr = cv2.erode(vr.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        r = mncc(T, vr.astype(np.float32)); yy, xx = np.mgrid[0:r.shape[0], 0:r.shape[1]]; X = xx + xt - NW / 2; Y = yy + yt - NW / 2; ok = np.hypot(X, Y) <= 8                       # backshell within 2 m of the known position
        rr = np.where(ok, r, -1); k = np.unravel_index(np.argmax(rr), rr.shape); v = float(rr[k]); grid[(round(float(th), 1), round(float(sc), 4))] = v
        if v > best[0]: best = (v, float(th), float(sc), float(X[k]), float(Y[k]))
ths = sorted({t for t, _ in grid}); scs = sorted({s for _, s in grid}); Z = np.array([[grid[(t, s)] for s in scs] for t in ths])
print(f"{PASS} patch {HALF:.0f} m: best NCC {best[0]:.3f} at theta {best[1]:.0f} deg, scale {best[2]:.4f} m/px, backshell offset ({best[3]*PXH:+.2f}, {best[4]*PXH:+.2f}) m; surface median {np.median(Z):.3f}, 95th {np.percentile(Z,95):.3f}")
th_prof = Z.max(1); print("  NCC vs rotation (max over scale), every 4 deg:", {int(t): round(float(v), 3) for t, v in zip(ths[::4], th_prof[::4])})
json.dump(dict(pass_id=PASS, half_m=HALF, ncc=best[0], theta=best[1], scale=best[2], dx_m=best[3] * PXH, dy_m=best[4] * PXH, median=float(np.median(Z)), p95=float(np.percentile(Z, 95)), theta_profile={str(t): float(v) for t, v in zip(ths, th_prof)}), open(OUT, "w"), indent=1)

"""Measure each HiRISE pass's point-spread function against the Ingenuity truth patch (3 cm truth, placed in HiRISE; truth_patch.py).
Forward model for pass k (the pipeline's, pipeline/fuse_flow.FlowModel): warp the truth to pass k with its Gotcha motion, blur with kernel h, sample at the 25 cm grid,
then gain and offset (the truth was imaged at midday, HiRISE in the afternoon): Y_k = g * (h * warp_k(T))[::4, ::4] + o, fitted on native pixels whose kernel support lies inside the truth
and within RAD metres of the backshell. Kernel families (HR pixels, 6.31 cm): (1) isotropic Gaussian sigma; (2) anisotropic Gaussian (sa, sb, angle); (3) generalised Gaussian
exp(-(r/a)^beta) with anisotropy. Gain/offset are solved linearly inside the nonlinear fit. Also splits the region into two halves to report held-out residuals.
Usage: python fit_psf.py <chain_dir> <truth_npz> <out_json> [radius_m]"""
import sys, json, numpy as np, cv2
from scipy.optimize import least_squares
from scipy import ndimage as ndi
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep"); from pipeline.fuse_flow import _warp_matrix
CH, TR, OUT = sys.argv[1:4]; RAD = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0; HR = 0.2524 / 4; SC = 4
d = np.load(f"{CH}/chain.npz"); meta = json.load(open(f"{CH}/chain.json")); t = np.load(TR); T = t["truth"].astype(np.float64); valid = t["valid"]; N = T.shape[0]; Tf = np.where(valid, T, np.nanmean(T))
FR = d["frames"].astype(np.float64); FL = list(zip(d["flows_dx"].astype(np.float64), d["flows_dy"].astype(np.float64))); passes = meta["passes_used"]
bs = np.array([N / 2 + t["params"][2] / HR, N / 2 + t["params"][3] / HR])                                       # backshell centre (x, y) in HR px
KR = 18                                                                                                          # kernel half-size in HR px
yy, xx = np.mgrid[-KR:KR + 1, -KR:KR + 1].astype(np.float64)
def kernel(p, family):
    p = p[2:]                                            # p[0], p[1] are the translation of the truth (HR px), handled in model()
    if family == "iso": a, b, ph, be = np.exp(p[0]), np.exp(p[0]), 0.0, 2.0
    elif family == "aniso": a, b, ph, be = np.exp(p[0]), np.exp(p[1]), p[2], 2.0
    else: a, b, ph, be = np.exp(p[0]), np.exp(p[1]), p[2], 0.6 + 2.4 / (1 + np.exp(-p[3]))                       # beta in (0.6, 3.0)
    c, s = np.cos(ph), np.sin(ph); u = (c * xx + s * yy) / a; v = (-s * xx + c * yy) / b; h = np.exp(-0.5 * (np.sqrt(u * u + v * v) ** be)) ** 1.0 if family == "gen" else np.exp(-0.5 * (u * u + v * v))
    return h / h.sum()
def second_moments(h): m = h.sum(); mx = (h * xx).sum() / m; my = (h * yy).sum() / m; C = np.array([[(h * (xx - mx) ** 2).sum(), (h * (xx - mx) * (yy - my)).sum()], [(h * (xx - mx) * (yy - my)).sum(), (h * (yy - my) ** 2).sum()]]) / m; return np.sqrt(np.linalg.eigvalsh(C))
res = {}
for k, pid in enumerate(passes):
    Z = (_warp_matrix(*FL[k], SC) @ Tf.ravel()).reshape(N, N); Vz = (_warp_matrix(*FL[k], SC) @ valid.astype(np.float64).ravel()).reshape(N, N) > 0.98          # truth seen from pass k's geometry
    Vz = ndi.binary_erosion(Vz, iterations=KR); Y = FR[k]; ys, xs = np.mgrid[0:Y.shape[0], 0:Y.shape[1]]; px, py = xs * SC + (SC - 1) / 2.0, ys * SC + (SC - 1) / 2.0   # native pixel centres on the HR grid
    ok = Vz[np.clip(py.astype(int), 0, N - 1), np.clip(px.astype(int), 0, N - 1)] & (np.hypot(px - bs[0], py - bs[1]) < RAD / HR) & (px > KR) & (py > KR) & (px < N - KR) & (py < N - KR)
    pys, pxs = py[ok].astype(int), px[ok].astype(int); y = Y[ok]; n = len(y); half = (pxs < np.median(pxs))
    def model(p, family, idx=slice(None)):
        h = kernel(p, family); Zs = cv2.warpAffine(Z, np.array([[1, 0, p[0]], [0, 1, p[1]]], np.float64), (N, N), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        B = cv2.filter2D(Zs, -1, h[::-1, ::-1], borderType=cv2.BORDER_REPLICATE); return B[pys[idx], pxs[idx]]
    def resid(p, family, idx=slice(None)):
        m = model(p, family, idx); A = np.stack([m, np.ones_like(m)], 1); co = np.linalg.lstsq(A, y[idx], rcond=None)[0]; return A @ co - y[idx]
    out = dict(n_pixels=int(n)); starts = {"iso": [0.0, 0.0, np.log(3.6)], "aniso": [0.0, 0.0, np.log(3.6), np.log(3.6), 0.0], "gen": [0.0, 0.0, np.log(3.6), np.log(3.6), 0.0, 0.0]}
    for fam in ("iso", "aniso", "gen"):
        best = None
        for s0 in ([0.0, 0.0, np.log(2.5)], [0.0, 0.0, np.log(3.6)], [4.0, 4.0, np.log(3.6)], [-4.0, -4.0, np.log(3.6)]) if fam == "iso" else [None]:
            p0 = np.array(s0 if s0 is not None else starts[fam], float)
            if fam != "iso":
                for ph0 in (0.0, 0.8, 1.6, 2.4):
                    p1 = p0.copy(); p1[4] = ph0; r = least_squares(resid, p1, args=(fam,)); best = r if best is None or r.cost < best.cost else best
            else: r = least_squares(resid, p0, args=(fam,)); best = r if best is None or r.cost < best.cost else best
        h = kernel(best.x, fam); sa, sb = second_moments(h)[::-1]; rms = float(np.sqrt(2 * best.cost / n)); out[fam] = dict(rms=rms, sigma_major_hr=float(sa), sigma_minor_hr=float(sb), sigma_major_native=float(sa / SC), sigma_minor_native=float(sb / SC), params=[float(v) for v in best.x], shift_native=[float(best.x[0] / SC), float(best.x[1] / SC)])
        if fam == "gen": out[fam]["beta"] = float(0.6 + 2.4 / (1 + np.exp(-best.x[5])))
        # held-out check: fit on one half, score on the other (and the reverse)
        ho = []
        for tr, te in ((half, ~half), (~half, half)):
            if tr.sum() < 40 or te.sum() < 40: continue
            r2 = least_squares(lambda p: resid(p, fam, tr), best.x); ho.append(float(np.sqrt(np.mean(resid(r2.x, fam, te) ** 2))))
        out[fam]["heldout_rms"] = float(np.mean(ho)) if ho else None
    res[pid] = out; base = out["iso"]["rms"]
    print(f"{pid[:10]}: n={n:4d} | iso sigma {out['iso']['sigma_major_native']:.3f} px shift ({out['iso']['shift_native'][0]:+.2f},{out['iso']['shift_native'][1]:+.2f}) native px rms {out['iso']['rms']:.4f} (held-out {out['iso']['heldout_rms']:.4f}) | aniso {out['aniso']['sigma_major_native']:.3f}/{out['aniso']['sigma_minor_native']:.3f} rms {out['aniso']['rms']:.4f} (ho {out['aniso']['heldout_rms']:.4f}) | gen beta {out['gen']['beta']:.2f} rms {out['gen']['rms']:.4f} (ho {out['gen']['heldout_rms']:.4f})", flush=True)
json.dump(res, open(OUT, "w"), indent=1)

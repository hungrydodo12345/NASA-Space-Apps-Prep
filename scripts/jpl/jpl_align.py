import sys, glob, os, json, numpy as np, glymur, collections
from scipy import ndimage as ndi
S, LO, HI, ERODE = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]); sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep")
from pipeline import groundtruth as gt
OA = glymur.Jp2k(f"{S}/hirise_jezero/ESP_045994_1985_RED_A_01_ORTHO.JP2"); P = gt.ORTHO["scale"]; W = 640; R = 240
band = lambda a: ndi.gaussian_filter(a, LO) - ndi.gaussian_filter(a, HI)
def xcorr(A, V, B):
    A = np.where(V, A - A[V].mean(), 0); A /= np.sqrt((A ** 2).sum()) + 1e-12
    F = np.real(np.fft.ifft2(np.fft.fft2(B) * np.conj(np.fft.fft2(A))))
    E = np.sqrt(np.maximum(np.real(np.fft.ifft2(np.fft.fft2(B ** 2) * np.conj(np.fft.fft2(V.astype(float))))), 1e-9))
    C = np.fft.fftshift(F / E); c0 = W // 2; return C[c0 - R:c0 + R + 1, c0 - R:c0 + R + 1]
ops = {"as-is": lambda a: a, "rot90": lambda a: np.rot90(a, 1), "rot180": lambda a: np.rot90(a, 2), "rot270": lambda a: np.rot90(a, 3),
       "flipud": np.flipud, "fliplr": np.fliplr, "transpose": lambda a: a.T, "anti-transpose": lambda a: np.rot90(a, 2).T}
votes = collections.Counter(); out = []
for f in sorted(glob.glob(f"{S}/hero/rover_truth_jpl/*.npz")):
    t = np.load(f); z = t["z"]; v = np.isfinite(z)
    if v.mean() < 0.2: continue
    k = 0.05 / P; zt = ndi.zoom(np.where(v, z, 0), k, order=1); vt = ndi.zoom(v.astype(float), k, order=1) > 0.9
    zt = np.where(vt, zt, zt[vt].mean())
    l0, s0 = gt.latlon_to_ortho(float(t["lat"]), float(t["lon"])); l0, s0 = int(round(l0)), int(round(s0))
    ref = np.asarray(OA[l0 - W // 2:l0 + W // 2, s0 - W // 2:s0 + W // 2]).astype(float); B = band(ref); B = (B - B.mean()) / B.std()
    res = []
    for nm, op in ops.items():
        a = op(band(zt)); vv = op(ndi.binary_erosion(vt, iterations=ERODE)); tt = a.shape[0]; o = (W - tt) // 2
        Ap = np.zeros((W, W)); Vp = np.zeros((W, W), bool); Ap[o:o+tt, o:o+tt] = a; Vp[o:o+tt, o:o+tt] = vv
        C = xcorr(Ap, Vp, B); iy, ix = np.unravel_index(np.argmax(C), C.shape); pk = C[iy, ix]
        C2 = C.copy(); C2[max(0, iy-int(2*HI)):iy+int(2*HI)+1, max(0, ix-int(2*HI)):ix+int(2*HI)+1] = -9
        res.append((pk, pk - C2.max(), nm, (iy - R) * P, (ix - R) * P))
    res.sort(reverse=True); b = res[0]; votes[b[2]] += 1; out.append(dict(stop=os.path.basename(f)[:-4], best=b[2], corr=b[0], margin=b[1], dS=b[3], dE=b[4]))
    print(f"{os.path.basename(f)[:-4]:8s} best {b[2]:14s} corr {b[0]:.2f} margin {b[1]:+.2f} (next orientation {res[1][2]} {res[1][0]:.2f})  offset {b[3]:+6.1f} m S, {b[4]:+6.1f} m E", flush=True)
print(dict(votes)); json.dump(out, open(f"{S}/jpl_align_{LO}_{HI}.json", "w"))

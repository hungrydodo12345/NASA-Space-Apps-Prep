"""Position-matched version of rocks_real.py: for each rock the rover photos show (same detector, clean ground only),
is a HiRISE-product detection found at the same place (within max(2 px, 0.5 D))? Also counts detections with no rover rock
nearby (false per 100 m2). Pooled across stops; 95% bootstrap CIs over stops.
Usage: python scripts/rocks_real_matched.py <restored_dir> <rover_truth_dir> <texture_filter.json> <out_json>
"""
import sys, os, glob, json
import numpy as np
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline import texture_quality as tq
sv = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[2]]
from experiments import exp3_rocks as E
sys.argv = sv
RD, TRUTH, FILT, OUT = sys.argv[1:5]
filt = tq.load_filter(FILT); PX = E.PX
THR = float(os.environ.get("THR", "0.85"))   # clean-ground score threshold (0 = every valid pixel)
BINS = {"15-20 cm": (15, 20), "25-40 cm": (25, 40), ">=50 cm": (50, 100)}
NAMES = ["bicubic", "average", "deconv_1frame", "restored_multi"]
ALIGN = int(os.environ.get("ALIGN", "0"))      # >0: search +-ALIGN px for the HiRISE-to-rover offset per stop


def offset(img, tz, o, valid, R):
    """Integer shift (dy, dx) of the HiRISE product that best matches the rover mosaic (smoothed, normalised)."""
    if R <= 0: return 0, 0
    a = ndi.gaussian_filter(img.astype(float), 2.0); b = ndi.gaussian_filter(tz, 2.0); t = tz.shape[0]
    best, arg = -9, (0, 0)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            y0, x0 = o + dy, o + dx
            if y0 < 0 or x0 < 0 or y0 + t > a.shape[0] or x0 + t > a.shape[1]: continue
            w = a[y0:y0 + t, x0:x0 + t][valid]; v = b[valid]
            c = np.corrcoef(w, v)[0, 1]
            if c > best: best, arg = c, (dy, dx)
    return arg


def clean_mask(z, valid, win=32, stride=16, thr=None):
    thr = THR if thr is None else thr
    m = np.zeros_like(valid)
    for y in range(0, z.shape[0] - win + 1, stride):
        for x in range(0, z.shape[1] - win + 1, stride):
            if valid[y:y + win, x:x + win].mean() >= 0.98 and tq.ground_score(z[y:y + win, x:x + win], filt) >= thr: m[y:y + win, x:x + win] = True
    return m & valid


def near(p, dets, tol=0.5):
    return any(np.hypot(p[0] - q[0], p[1] - q[1]) <= max(2.0, tol * max(p[2], q[2]) / PX) for q in dets)


REG = int(os.environ.get("REG", "0")); PSFH = 1.07 * 4; unreg = []


def register_truth(bic, zt, vt, o, R=48, rmin=0.3):
    """Shift (dy, dx) of the rover truth that best matches raw HiRISE at HiRISE resolution (Amendment 2): high-passed,
    blurred truth vs high-passed bicubic HiRISE, +-R px, coarse 2 px then 1 px. None if the peak is weak or on the edge."""
    n = bic.shape[0]; t = zt.shape[0]
    Z = np.zeros((n, n)); V = np.zeros((n, n)); Z[o:o + t, o:o + t] = np.where(vt, zt, 0.0); V[o:o + t, o:o + t] = vt
    zb = ndi.gaussian_filter(Z, PSFH) / np.maximum(ndi.gaussian_filter(V, PSFH), 1e-6)
    hpf = lambda a: a - ndi.gaussian_filter(a, 4 * PSFH)
    A = hpf(zb); B = hpf(bic); vv = ndi.binary_erosion(V > 0.5, iterations=int(2 * PSFH))   # only the truth is masked; HiRISE covers the window
    if vv.sum() < 2000: return None, None, None
    def corr(dy, dx): return np.corrcoef(A[vv], np.roll(np.roll(B, -dy, 0), -dx, 1)[vv])[0, 1]
    best = max(((corr(dy, dx), dy, dx) for dy in range(-R, R + 1, 2) for dx in range(-R, R + 1, 2)))
    best = max(((corr(dy, dx), dy, dx) for dy in range(best[1] - 1, best[1] + 2) for dx in range(best[2] - 1, best[2] + 2)))
    if best[0] < rmin or abs(best[1]) >= R - 1 or abs(best[2]) >= R - 1: return None, None, None
    return best[1], best[2], best[0]                   # corr compares truth(y) with HiRISE(y + d): move truth by +d


rows = []
for f in sorted(glob.glob(f"{RD}/halves_*.npz")):
    stop = os.path.basename(f)[7:-4]; full = f"{RD}/{stop}.npz"; tf = f"{TRUTH}/{stop}.npz"
    if not (os.path.exists(full) and os.path.exists(tf)): continue
    h = np.load(f); fr = np.load(full); z5 = np.load(tf)["z"].astype(float)
    k = 0.05 / (PX / 100)
    zt = ndi.zoom(np.nan_to_num(z5), k, order=1); vt = ndi.zoom(np.isfinite(z5).astype(float), k, order=1) > 0.99
    cm = clean_mask(np.where(vt, zt, 0.0), ndi.binary_erosion(vt, iterations=4))
    n = fr["restored_multi"].shape[0]; t = zt.shape[0]; o = (n - t) // 2
    mask = np.zeros((n, n), bool); mask[o:o + t, o:o + t] = cm
    a = mask.sum() * (PX / 100) ** 2
    if a < 2.0: continue
    tz = np.where(vt, zt, np.nanmean(np.where(vt, zt, np.nan)))
    if REG:
        dy, dx, rr = register_truth(fr["bicubic"].astype(float), zt, vt, o)
        if dy is None: unreg.append(stop); continue
    else:
        dy, dx = offset(fr["bicubic"], tz, o, ndi.binary_erosion(vt, iterations=8), ALIGN)
    truth = [(y + o + dy, x + o + dx, D) for y, x, D, _ in E.detect(tz) if cm[y, x]]
    mask = np.roll(np.roll(mask, dy, 0), dx, 1)
    row = dict(stop=stop, area_m2=a, offset_px=[dy, dx], truth={b: sum(lo <= D <= hi for _, _, D in truth) for b, (lo, hi) in BINS.items()})
    for nm in NAMES + ["restored_repeatable"]:
        if nm == "restored_repeatable":
            d0 = [(y, x, D) for y, x, D, _ in E.detect(h["restored_0"].astype(float))]
            d1 = [(y, x, D) for y, x, D, _ in E.detect(h["restored_1"].astype(float))]
            d = [p for p in d0 if near(p, d1)]
        else:
            d = [(y, x, D) for y, x, D, _ in E.detect(fr[nm].astype(float))]
        d = [p for p in d if mask[int(p[0]), int(p[1])]]
        found = {b: sum(near(p, d) for p in truth if lo <= p[2] <= hi) for b, (lo, hi) in BINS.items()}
        false = sum(not near(p, truth, tol=1.0) for p in d)
        # chance level: same detections shifted by a fixed 40 px (decorrelated), to show what matching gives by luck
        sh = [((p[0] + 40) % n, (p[1] + 40) % n, p[2]) for p in d]
        chance = {b: sum(near(p, sh) for p in truth if lo <= p[2] <= hi) for b, (lo, hi) in BINS.items()}
        row[nm] = dict(found=found, chance=chance, false=false, detections=len(d))
    rows.append(row)


def boot(num, den, B=2000, rng=np.random.default_rng(0)):
    num, den = np.array(num, float), np.array(den, float)
    if den.sum() == 0: return None
    idx = rng.integers(0, len(num), (B, len(num)))
    s = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-9)
    return [100 * num.sum() / den.sum(), 100 * np.percentile(s, 2.5), 100 * np.percentile(s, 97.5)]


area = sum(r["area_m2"] for r in rows); summ = {}
for nm in NAMES + ["restored_repeatable"]:
    summ[nm] = dict(false_per_100m2=100 * sum(r[nm]["false"] for r in rows) / area)
    for b in BINS:
        den = [r["truth"][b] for r in rows]
        summ[nm][b] = dict(rover_rocks=int(sum(den)), found_pct=boot([r[nm]["found"][b] for r in rows], den),
                           chance_pct=boot([r[nm]["chance"][b] for r in rows], den))
json.dump(dict(stops=len(rows), clean_area_m2=area, registered=bool(REG), unregistered_stops=unreg, summary=summ, per_stop=rows), open(OUT, "w"), indent=1)
if REG: print(f"registration: {len(rows)} stops registered and scored, {len(unreg)} excluded (no well-defined peak)")
print(f"{len(rows)} stops, {area:.0f} m2 clean rover-verified ground")
for nm, s in summ.items():
    print(f"{nm:22s} false/100m2 {s['false_per_100m2']:6.1f} | " + " | ".join(
        f"{b} (n={s[b]['rover_rocks']}): " + (f"{s[b]['found_pct'][0]:.0f}% [{s[b]['found_pct'][1]:.0f}-{s[b]['found_pct'][2]:.0f}] chance {s[b]['chance_pct'][0]:.0f}%" if s[b]['found_pct'] else "n/a") for b in BINS))

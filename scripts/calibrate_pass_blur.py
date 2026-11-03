"""Per-pass blur (PSF) calibration, using the parked Perseverance rover as a test target (absolute), then static ground
for passes without the rover (relative).

1. Absolute (passes where the rover is parked and visible, sols 1-11): brightness profiles across the rover's short axis
   (native 25 cm pixels, cubic interpolation at 0.1 px) are fitted with a blurred bright bar on a linear background,
       f(t) = b0 + b1 t + a [Phi((t - t1)/s) - Phi((t - t2)/s)],
   where s^2 = sigma_psf^2 + 1/12 (the 1/12 px^2 is the pixel's own box integration). Several parallel profiles are fitted
   and the median taken; the spread gives the uncertainty.
2. Relative (every other pass): both passes are aligned with Gotcha on static ground; the extra Gaussian blur d that makes
   the sharper pass best match the other (least squares with gain/offset, over d in 0-1.5 px) gives
       sigma_other^2 = sigma_ref^2 +/- d^2.
Output: data_products/pass_blur.json  {pass: {sigma_px, method, ...}}
Usage: python scripts/calibrate_pass_blur.py <hirise_rdr_dir> <rover_target_dir> <ground_site_dir> <out_json>
"""
import sys, os, json, numpy as np
from scipy import ndimage as ndi, optimize
from scipy.special import ndtr
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline.hirise_io import Rdr
from pipeline import gotcha

RDR, ROVER, SITE, OUT = sys.argv[1:5]


def window(oid, area_dir, size=None):
    d = np.load(f"{area_dir}/area.npz"); s = json.load(open(f"{area_dir}/summary.json"))
    g0l, g0s = d["origin_global"]; n = size or d["bicubic"].shape[0] // 4; o = s["locks"].get(oid, {"dx": 0, "dy": 0})
    w = Rdr(f"{RDR}/{oid}_RED").read_if(g0l - o["dy"], g0s - o["dx"], n, n)
    return np.nan_to_num(w, nan=np.nanmean(w)).astype(float)


def bar_model(t, b0, b1, a, t1, t2, s): return b0 + b1 * t + a * (ndtr((t - t1) / s) - ndtr((t - t2) / s))


def rover_sigma(img):
    sm = ndi.gaussian_filter(img, 1); c = img.shape[0] // 2
    y0, x0 = np.unravel_index(np.argmax(sm[c - 24:c + 24, c - 24:c + 24]), (48, 48)); y0 += c - 24; x0 += c - 24
    bg = np.median(img); pk = sm[y0, x0]; mask = sm > bg + 0.5 * (pk - bg); lab, _ = ndi.label(mask); obj = lab == lab[y0, x0]
    yy, xx = np.nonzero(obj); cy, cx = yy.mean(), xx.mean(); ev, evec = np.linalg.eigh(np.cov(np.stack([yy - cy, xx - cx])))
    u = evec[:, 0]; v = evec[:, 1]                                     # u: short axis, v: long axis
    sig = []
    for off in np.linspace(-1.5, 1.5, 7):                              # parallel profiles across the short axis
        t = np.arange(-8, 8.01, 0.1); py = cy + off * v[0] + t * u[0]; px = cx + off * v[1] + t * u[1]
        p = ndi.map_coordinates(img, [py, px], order=3)
        try:
            pr, _ = optimize.curve_fit(bar_model, t, p, p0=[bg, 0, pk - bg, -2.5, 2.5, 1.2], bounds=([-np.inf, -np.inf, 0, -8, -1, 0.3], [np.inf, np.inf, np.inf, 1, 8, 4]))
            s = pr[5]
            if s * s > 1 / 12 and pr[4] - pr[3] > 2: sig.append(np.sqrt(s * s - 1 / 12))
        except Exception: pass
    return (float(np.median(sig)), float(np.std(sig)), len(sig)) if sig else (np.nan, np.nan, 0)


def relative_sigma(ref, img, s_ref):
    r = gotcha.gotcha(ref, img, step=6)
    if r is None: return np.nan, 0.0
    yy, xx = np.mgrid[0:ref.shape[0], 0:ref.shape[1]].astype(float)
    a = ndi.map_coordinates(img, [yy - r["dy"], xx - r["dx"]], order=3, mode="nearest")    # img brought onto ref's grid
    c = (slice(16, -16), slice(16, -16)); A, B = ref[c], a[c]
    def cost(d, sharper, other):
        bl = ndi.gaussian_filter(sharper, d) if d > 0 else sharper; X = np.stack([bl.ravel(), np.ones(bl.size)], 1)
        co = np.linalg.lstsq(X, other.ravel(), rcond=None)[0]; return float(np.mean((X @ co - other.ravel()) ** 2))
    ds = np.linspace(0, 1.5, 31)
    c1 = [cost(d, A, B) for d in ds]; c2 = [cost(d, B, A) for d in ds]          # blur ref to match img, or img to match ref
    if min(c1) <= min(c2): d = ds[int(np.argmin(c1))]; return float(np.sqrt(s_ref ** 2 + d * d)), float(r["n_matched"])
    d = ds[int(np.argmin(c2))]; return float(np.sqrt(max(s_ref ** 2 - d * d, 0.05))), float(r["n_matched"])


if __name__ == "__main__":
    out = {}
    rover_passes = ["ESP_068294_1985", "ESP_068360_1985", "ESP_068426_1985"]
    for oid in rover_passes:
        s, e, k = rover_sigma(window(oid, ROVER)); out[oid] = dict(sigma_px=s, sigma_spread_px=e, profiles=k, method="rover edge fit (absolute)")
        print(oid, out[oid], flush=True)
    ref_id = min(rover_passes, key=lambda o: out[o]["sigma_px"] if np.isfinite(out[o]["sigma_px"]) else 9)
    others = [f.split("_RED")[0] for f in sorted(os.listdir(RDR)) if f.endswith("_RED.JP2") and f.split("_RED")[0] not in rover_passes]
    ref = window(ref_id, SITE, 192)
    for oid in others:
        s, nm = relative_sigma(ref, window(oid, SITE, 192), out[ref_id]["sigma_px"])
        out[oid] = dict(sigma_px=s, method=f"relative to {ref_id} on static ground (Gotcha-aligned)", matched_points=nm); print(oid, out[oid], flush=True)
    json.dump(out, open(OUT, "w"), indent=1)

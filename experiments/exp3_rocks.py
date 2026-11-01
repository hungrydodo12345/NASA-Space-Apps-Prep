"""Experiment 3: rock detectability by size - the yardstick UCL used for HiRISE super-resolution - with KNOWN rocks.

Rocks (sunlit body + shadow cast at the HiRISE sun geometry) of 15-100 cm are inserted into real rover ground texture,
imaged through the rover-calibrated HiRISE blur (sigma 0.78 px) with noise and registration error, reconstructed by each
algorithm at 6.31 cm, and detected with one fixed blob detector. Completeness = share of inserted rocks of a size found;
false detections are counted against everything the same detector finds in the noiseless truth.
"""
import sys, json, itertools, os
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep")
from experiments import algos, common as C, runner, scenes

TRUTH = sys.argv[1]
HP = json.load(open("/home/user/NASA-Space-Apps-Prep/results/tuned_hp.json"))
SIZES = [15, 20, 25, 30, 40, 50, 70, 100]           # rock diameters, cm
PX = C.OUT_PX_CM
SCN = {}


def insert_rocks(bg, rng, n_per_size=4, sun_az=np.radians(165), inc=np.radians(47)):
    t = bg.copy(); H, W = t.shape; s = bg.std(); placed = []
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dx, dy = np.sin(sun_az), -np.cos(sun_az)           # shadow falls away from the sun
    for D in SIZES:
        for _ in range(n_per_size):
            r = D / 2 / PX
            for _try in range(50):
                cy, cx = rng.uniform(20 + r, H - 20 - r), rng.uniform(20 + r, W - 20 - r)
                if all(np.hypot(cy - p[0], cx - p[1]) > (r + p[2] / 2 / PX) * 2.2 for p in placed): break
            body = 1 / (1 + np.exp((np.hypot(yy - cy, xx - cx) - r) / 0.6))
            L = 0.5 * D * np.tan(inc) / PX                  # shadow length from rock height ~ D/2
            sy, sx = cy + dy * (r + L / 2), cx + dx * (r + L / 2)
            u = (xx - sx) * dx + (yy - sy) * dy; v = -(xx - sx) * dy + (yy - sy) * dx
            shadow = 1 / (1 + np.exp((np.sqrt((u / max(L / 2, 0.5)) ** 2 + (v / r) ** 2) - 1) / 0.15))
            t = t + 1.2 * s * body - 1.6 * s * shadow * (1 - body)
            placed.append((cy, cx, D))
    return t, placed


def detect(img, valid=None, k=3.5):
    """Bright-blob detector: scale-normalised LoG maxima above k x robust noise, one detection per blob."""
    out = []
    resp = []
    for D in SIZES:
        sg = D / 2 / PX / np.sqrt(2)
        r = -sg ** 2 * ndi.gaussian_laplace(img, sg)
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        resp.append(r / (mad + 1e-12))
    R = np.stack(resp); M = R.max(0); arg = R.argmax(0)
    peaks = (M == ndi.maximum_filter(M, size=5)) & (M > k)
    if valid is not None: peaks &= valid
    for y, x in zip(*np.nonzero(peaks)): out.append((y, x, SIZES[arg[y, x]], M[y, x]))
    return out


def match(dets, rocks, tol=0.5):
    """Indices of rocks found (centre within max(2 px, tol*D))."""
    found = set()
    for y, x, _, _ in dets:
        for i, (cy, cx, D) in enumerate(rocks):
            if np.hypot(y - cy, x - cx) <= max(2.0, tol * D / PX): found.add(i)
    return found


def job(j):
    if "s" not in SCN: SCN["s"] = scenes.load_rover_scenes(TRUTH)
    name, bg, valid = SCN["s"][j["idx"] % len(SCN["s"])]
    rng = np.random.default_rng(1000 * j["idx"] + j["seed"])
    truth, rocks = insert_rocks(bg, rng)
    frames, est, _ = C.simulate(truth, j["n"], np.random.default_rng(77 * j["idx"] + j["seed"] + j["n"]), noise_rel=j["noise"], reg_err=0.05)
    hr = algos.reconstruct(j["algo"], frames, est, C.SCALE, C.PSF_HR, **HP.get(j["algo"], {}))
    d = detect(hr)
    found = match(d, rocks)
    bg_dets = detect(truth); bg_rocks = [(y, x, D) for y, x, D, _ in bg_dets if not any(np.hypot(y - r[0], x - r[1]) <= max(2.0, 0.5 * r[2] / PX) for r in rocks)]
    explained = 0
    for y, x, D, _ in d:
        if any(np.hypot(y - r[0], x - r[1]) <= max(2.0, 0.5 * r[2] / PX) for r in rocks + bg_rocks): explained += 1
    area_m2 = (truth.shape[0] * PX / 100) ** 2
    comp = {str(D): float(np.mean([i in found for i, r in enumerate(rocks) if r[2] == D])) for D in SIZES}
    return dict(algo=j["algo"], n=j["n"], idx=j["idx"], seed=j["seed"], noise=j["noise"], completeness=comp,
                false_per_100m2=float((len(d) - explained) / area_m2 * 100), detections=len(d))


if __name__ == "__main__":
    out = "/home/user/NASA-Space-Apps-Prep/results/exp3_rocks.jsonl"
    A = ["bicubic", "average", "cg_tikhonov", "ibp", "btv_l1", "richardson_lucy"]
    ns = [1, 2, 4, 6, 8, 16]
    nsc = len(scenes.load_rover_scenes(TRUTH))
    jobs = []
    for idx, seed, n, a in itertools.product(range(min(nsc, 12)), (0, 1), ns, A):
        if a == "bicubic" and n > 1: continue
        j = dict(idx=idx, seed=seed, n=n, algo=a, noise=0.02)
        j["id"] = "|".join(f"{k}={j[k]}" for k in sorted(j)); jobs.append(j)
    runner.run(job, jobs, out)

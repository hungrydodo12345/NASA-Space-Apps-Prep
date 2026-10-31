"""Export the hero area to compact web assets (web/data/) for the no-build Three.js viewer.

Usage: python scripts/export_web.py <hero_dir> <rover_truth_dir> <dtm_img> <ortho_jp2> <cells_npz> <waypoints.json> <out_dir>
"""
import sys, os, json, glob, math
import numpy as np, cv2, glymur
from scipy import ndimage as ndi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import detail as dt, groundtruth as gt, surface_classifier as scl, hero
from pipeline.dtm import Dtm
from sklearn.ensemble import RandomForestRegressor

HERO, TRUTH, DTM, ORTHO, CELLS, WPJ, OUT = sys.argv[1:8]
os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f"{HERO}/measured_meta.json"))
N, PX, TOP, LEFT = meta["size"], meta["pixel_m"], meta["top"], meta["left"]
CL, CS = meta["centre"]
SIZE_M = N * PX
psf = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data_products", "psf_calibration.json")))["sigma_px_median"]
SC = 4; PSF_HR = psf * SC

# ---- measured base image (fused HiRISE), normalised to 0..1
fused = np.load(f"{HERO}/measured_fused.npy").astype(np.float64)
lo, hi = np.percentile(fused, [0.5, 99.5]); v = np.clip((fused - lo) / (hi - lo), 0, 1)
cv2.imwrite(f"{OUT}/base.jpg", (v * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
# local contrast of the measured coarse band (sets physical detail amplitude); stored at 1 m
blur, broad = ndi.gaussian_filter(v, psf), ndi.gaussian_filter(v, 4 * psf)
contrast = np.sqrt(ndi.gaussian_filter((blur - broad) ** 2, 8 * psf))
con1 = ndi.zoom(contrast, 0.25, order=1); cmax = float(np.percentile(con1, 99.9))
cv2.imwrite(f"{OUT}/contrast.png", (np.clip(con1 / cmax, 0, 1) * 255).astype(np.uint8))

# ---- heights from the HiRISE stereo DTM (12-bit packed, rover-prototype format)
d = Dtm(DTM); hgt, holes = d.heights_for_ortho(TOP, LEFT, N, N, 4)
emin, emax = float(hgt.min()) - 0.5, float(hgt.max()) + 0.5
q = np.clip(np.rint((hgt - emin) / (emax - emin) * 4095), 0, 4095).astype(np.uint16)
rgb = np.zeros(hgt.shape + (3,), np.uint8); rgb[..., 0] = q >> 4; rgb[..., 1] = (q & 15) << 4
cv2.imwrite(f"{OUT}/height.png", rgb[..., ::-1])

# ---- fuzzy surface-class memberships at 4 m (orbital classifier, trained on rover-projected ground truth)
cz = np.load(CELLS); rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(cz["X"], cz["Y"])
J = glymur.Jp2k(ORTHO); H1, H2 = 32, 96
img = np.asarray(J[TOP - H2:TOP + N + H2, LEFT - H2:LEFT + N + H2]).astype(float)
cells = N // 16; P = np.zeros((cells, cells, 4))
feats = []
for i in range(cells):
    for j in range(cells):
        cy, cx = H2 + i * 16 + 8, H2 + j * 16 + 8
        feats.append(np.r_[scl.features(img[cy - H1:cy + H1, cx - H1:cx + H1]), scl.features(img[cy - H2:cy + H2, cx - H2:cx + H2])])
pr = np.clip(rf.predict(np.array(feats)), 0, None); pr /= pr.sum(1, keepdims=True)
P = pr.reshape(cells, cells, 4)
cv2.imwrite(f"{OUT}/classes.png", cv2.cvtColor((P * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA))

# ---- rover truth: texture library (simulated layer) + corridor tiles (real layer)
wp = {f["properties"]["RMC"]: f["properties"] for f in json.load(open(WPJ))["features"]}
zoomf = 0.05 / (PX / SC)
tiles = {k: [] for k in range(4)}; ratios = {k: [] for k in range(4)}; corridor = []
for f in sorted(glob.glob(f"{TRUTH}/*.npz")):
    s = np.load(f); stop = os.path.basename(f)[:-4]
    z = ndi.zoom(np.nan_to_num(s["z"]), zoomf, order=1); vz = ndi.zoom(np.isfinite(s["z"]).astype(float), zoomf, order=1) > 0.99
    c = ndi.zoom(np.nan_to_num(s["cls"], nan=-1), zoomf, order=0)
    z = np.where(vz, z, np.nan); c = np.where(vz & (c >= 0), c, np.nan)
    for k, pts in dt.extract_patches(z, c, PSF_HR, 64).items(): tiles[k] += pts
    fine, coarse = dt.bands(z, PSF_HR)
    for k in range(4):
        m = c == k
        if m.sum() > 2000: ratios[k].append(np.nanstd(fine[m]) / (np.nanstd(coarse[m]) + 1e-12))
    L0, S0 = gt.latlon_to_ortho(float(s["lat"]), float(s["lon"]))
    x, zz = (S0 - CS) * PX, (L0 - CL) * PX
    if abs(x) < SIZE_M / 2 - 10 and abs(zz) < SIZE_M / 2 - 10:
        corridor.append(dict(stop=stop, x=float(x), z=float(zz), sol=int(s["sol"]), fine=fine, ratio=float(np.nanstd(fine) / (np.nanstd(coarse) + 1e-12)),
                             yaw=float(wp[stop]["yaw"]) if stop in wp else 0.0))
amp = [float(np.median(ratios[k])) if ratios[k] else 0.0 for k in range(4)]
# fill classes with too few rover examples from the most similar class (documented in manifest)
fallback = {3: 1, 2: 0, 1: 0, 0: 1}; filled = {}
TPC = 4
atlas = np.full((4 * 64, TPC * 64), 128, np.uint8)
for k in range(4):
    src = tiles[k] if len(tiles[k]) >= TPC else tiles[k] + tiles[fallback[k]]
    filled[dt.CLASSES[k]] = len(tiles[k])
    if amp[k] == 0.0: amp[k] = amp[fallback[k]]
    rng = np.random.default_rng(k); pick = rng.choice(len(src), size=min(TPC, len(src)), replace=False)
    for t, idx in enumerate(pick):
        atlas[k * 64:(k + 1) * 64, t * 64:(t + 1) * 64] = (np.clip(src[idx] / 6 + 0.5, 0, 1) * 255).astype(np.uint8)
cv2.imwrite(f"{OUT}/atlas.png", atlas)
TP = 256; cols = int(math.ceil(math.sqrt(len(corridor))))
cor = np.zeros((cols * TP, cols * TP, 3), np.uint8); cor[..., 0] = 128
stops = []
for i, cdat in enumerate(corridor):
    f = cdat["fine"]; n = f.shape[0]; ok = np.isfinite(f)
    fz = np.where(ok, f / (np.nanstd(f) + 1e-12), 0.0)
    tile = np.zeros((TP, TP)); m = np.zeros((TP, TP), bool)
    o = (TP - n) // 2
    if o >= 0: tile[o:o + n, o:o + n] = fz; m[o:o + n, o:o + n] = ok
    else: tile = fz[-o:-o + TP, -o:-o + TP]; m = ok[-o:-o + TP, -o:-o + TP]
    r, cc = divmod(i, cols)
    cor[r * TP:(r + 1) * TP, cc * TP:(cc + 1) * TP, 0] = (np.clip(tile / 6 + 0.5, 0, 1) * 255).astype(np.uint8)
    cor[r * TP:(r + 1) * TP, cc * TP:(cc + 1) * TP, 1] = m.astype(np.uint8) * 255
    stops.append(dict(stop=cdat["stop"], sol=cdat["sol"], x=cdat["x"], z=cdat["z"], ratio=cdat["ratio"], tile=i))
cv2.imwrite(f"{OUT}/corridor.png", cor[..., ::-1])
half_m = TP / 2 * PX / SC
first = min(corridor, key=lambda c: c["sol"])
lat, lon = hero.ortho_to_latlon(CL, CS)
man = dict(name="Jezero - Octavia E. Butler landing area (sols 13-120)", center=dict(lat=lat, lon=lon), size_m=SIZE_M,
           base=dict(file="base.jpg", px=N, pixel_m=PX, frames=json.load(open(f"{HERO}/measured_meta.json"))["frames"]),
           height=dict(file="height.png", n=int(hgt.shape[0]), emin=emin, emax=emax, datum=float(np.mean(hgt)), holes_pct=100 * holes, source="DTEEC_045994_1985_046060_1985_U01"),
           contrast=dict(file="contrast.png", n=int(con1.shape[0]), scale=cmax),
           classes=dict(file="classes.png", n=cells, cell_m=16 * PX, names=dt.CLASSES),
           atlas=dict(file="atlas.png", tiles_per_class=TPC, tile_px=64, tile_m=64 * PX / SC, rover_examples=filled),
           amp_ratio=amp, psf_sigma_hirise_px=psf,
           corridor=dict(file="corridor.png", tile_px=TP, cols=cols, half_m=half_m, stops=stops),
           start=dict(x=first["x"], z=first["z"] + 3.0, yaw=-math.radians(first["yaw"])))
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
sizes = {f: os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT)}
print("exported:", {k: f"{v/1e6:.2f} MB" for k, v in sizes.items()}, "| total %.1f MB" % (sum(sizes.values()) / 1e6))
print("rover texture examples per class:", filled, "| amp ratios:", [round(a, 2) for a in amp], "| corridor stops:", len(stops))

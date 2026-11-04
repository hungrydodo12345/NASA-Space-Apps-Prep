"""Build the game terrain tileset around the Perseverance landing spot.

Layers (bottom to top):
  1. base    : HiRISE RED orthoimage on the DTM grid (25 cm, co-registered with the DTM by construction)      [measured]
  2. restored: optional 6.25 cm multi-pass restoration of a chain tile (scripts/run_chain.py), registered to the base by correlation [measured]
  3. ingenuity: optional Ingenuity orthomosaic (1 cm, here resampled to 5 cm) placed in the RDR grid (site-2 placement)              [measured]
  4. detail  : optional repeating detail texture, applied only where no layer finer than 0.25 m exists                                [simulated]
Geometry: HiRISE stereo DTM (1 m posts), on the Mars sphere, ENU frame at the scene centre.

Usage: python scripts/build_game_tiles.py <out_dir> --dtm DTM.IMG --ortho ORTHO.JP2 --ortho-lbl ORTHO.LBL --rdr-lbl RDR_REF.LBL
         [--center-global L S] [--size-m 2000] [--base-level 3] [--chain DIR] [--ingenuity MOSAIC_5CM.npy --placement site2_placement.json]
         [--detail-png P] [--focus-m 240 --focus-level 6]
"""
import os, sys, json, argparse, math
import numpy as np
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from tiling.mars import Equirect, R_HIRISE
from tiling.sources import DtmSource, Jp2Source, ArraySource, chain_restoration_source
from tiling.texture import Layer, match_overlay
from tiling.build import TileBuilder

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--dtm", required=True); ap.add_argument("--ortho", required=True); ap.add_argument("--ortho-lbl", required=True)
ap.add_argument("--rdr-lbl", required=True); ap.add_argument("--center-global", nargs=2, type=float, default=None, help="scene centre in the reference RDR grid (global line, sample)")
ap.add_argument("--size-m", type=float, default=2000); ap.add_argument("--base-level", type=int, default=3); ap.add_argument("--grid", type=int, default=65); ap.add_argument("--tex", type=int, default=256)
ap.add_argument("--chain"); ap.add_argument("--site-summary", help="summary.json with pass locks (reference RDR -> the orthoimage's observation)"); ap.add_argument("--chain-key", default="restored"); ap.add_argument("--ingenuity"); ap.add_argument("--placement"); ap.add_argument("--detail-png")
ap.add_argument("--focus-m", type=float, default=240); ap.add_argument("--focus-level", type=int, default=6); ap.add_argument("--overlay-level", type=int, default=7)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rdr = Equirect.from_file(a.rdr_lbl); dtm = DtmSource(a.dtm)
lbl = open(a.ortho_lbl).read(); import re
gain = float(re.search(r"SCALING_FACTOR\s*=\s*([-+0-9.eE]+)", lbl).group(1)); off = float(re.search(r"^\s*OFFSET\s*=\s*([-+0-9.eE]+)", lbl, re.M).group(1))
ortho = Jp2Source(a.ortho, a.ortho_lbl, gain, off)
def register_rdr_to_ortho(lat_c, lon_c, half_m=150.0, res=0.5, search_m=40.0):
    """Shift (RDR lines, samples) to add to RDR-grid sources so they land on the orthoimage/DTM: the reference RDR pass and the orthoimage are sampled
    on the same lat/lon grid (res m) and a central template of the RDR image is matched (normalised cross-correlation) within +-search_m."""
    import cv2
    # HiRISE RDRs of different observations disagree in geolocation by tens of metres. The orthoimage is made from one observation (ortho_obs), so we chain:
    # reference RDR -> ortho_obs RDR (the pipeline's integer pass lock, from the site summary) -> orthoimage (this NCC, small residual).
    obs = os.path.basename(a.ortho).split("_RED")[0]; rlbl = os.path.join(os.path.dirname(a.rdr_lbl), f"{obs}_RED.LBL")
    lock = json.load(open(a.site_summary))["locks"][f"{obs}"] if a.site_summary else dict(dy=0, dx=0)
    rdr_src = Jp2Source(rlbl.replace(".LBL", ".JP2"), rlbl)
    n = int(2 * (half_m + search_m) / res); dl_ = math.degrees(res / R); do_ = math.degrees(res / (R * math.cos(math.radians(lat_c))))
    la = lat_c + (n / 2 - np.arange(n))[:, None] * dl_ * np.ones((1, n)); lo = lon_c + np.ones((n, 1)) * (np.arange(n) - n / 2)[None, :] * do_
    vo, mo = ortho.sample(la, lo, res); vr, mr = rdr_src.sample(la, lo, res)
    hp = lambda x, m: (np.where(m, x, x[m].mean()) - ndi.gaussian_filter(np.where(m, x, x[m].mean()), 6)).astype(np.float32)
    O, Rr = hp(vo, mo), hp(vr, mr); p = int(search_m / res); T = Rr[p:-p, p:-p]
    c = cv2.matchTemplate(O, T, cv2.TM_CCOEFF_NORMED); y, x = np.unravel_index(np.argmax(c), c.shape)
    dy = 0.5 * (c[y - 1, x] - c[y + 1, x]) / (c[y - 1, x] - 2 * c[y, x] + c[y + 1, x]); dx = 0.5 * (c[y, x - 1] - c[y, x + 1]) / (c[y, x - 1] - 2 * c[y, x] + c[y, x + 1])
    cc = c.copy(); cc[max(0, y - 10):y + 10, max(0, x - 10):x + 10] = -1
    d_lines, d_samples = (y + dy - p) * res / rdr.scale, (x + dx - p) * res / rdr.scale * (math.cos(math.radians(rdr.clat)) / math.cos(math.radians(lat_c)))
    return float(d_lines - lock["dy"]), float(d_samples - lock["dx"]), float(c[y, x]), float(cc.max())

R = R_HIRISE; reg = {}
cl, cs = a.center_global if a.center_global else (None, None)
SH = (0.0, 0.0)
if cl is not None:                                                                        # scene centre given in the reference RDR grid: register that grid first
    la_, lo_ = rdr.pix_to_latlon(rdr.loff + cl, rdr.soff + cs); dl, ds, pk, sec = register_rdr_to_ortho(float(la_), float(lo_)); SH = (dl, ds)
    reg["rdr_to_ortho"] = dict(shift_rdr_lines=dl, shift_rdr_samples=ds, ncc_peak=pk, ncc_runner_up=sec, method="pass lock (reference RDR -> orthoimage observation RDR) + 0.5 m NCC of that RDR vs the orthoimage over 300 m")
    print("RDR -> ortho/DTM registration:", reg["rdr_to_ortho"])
lat0, lon0 = rdr.pix_to_latlon(rdr.loff + cl + SH[0], rdr.soff + cs + SH[1]) if cl is not None else ((dtm.proj.pix_to_latlon(dtm.lines / 2, dtm.samples / 2)))
lat0, lon0 = float(lat0), float(lon0)
dlat = math.degrees(a.size_m / 2 / R); dlon = math.degrees(a.size_m / 2 / (R * math.cos(math.radians(lat0))))
root = (lat0 - dlat, lat0 + dlat, lon0 - dlon, lon0 + dlon); print("scene centre", lat0, lon0, "root", root)
def bbox_m(lat, lon, half):
    return (lat - math.degrees(half / R), lat + math.degrees(half / R), lon - math.degrees(half / (R * math.cos(math.radians(lat)))), lon + math.degrees(half / (R * math.cos(math.radians(lat)))))
g = np.linspace(-1, 1, 64); GL, GO = np.meshgrid(lat0 + g * dlat, lon0 + g * dlon, indexing="ij")
base = Layer("hirise_ortho_25cm", ortho, provenance="measured", native_res_m=ortho.proj.scale, extras=dict(product=os.path.basename(a.ortho))).set_range_from(GL, GO, a.size_m / 64)
layers = [base]; refine = [(*bbox_m(lat0, lon0, a.focus_m / 2), a.focus_level)]

if a.chain:
    meta = json.load(open(f"{a.chain}/chain.json")); SC = meta["scale"]; gl0, gs0 = meta["origin_global"]; n = meta["tile_native_px"]
    dl, ds = SH
    la, lo = rdr.pix_to_latlon(rdr.loff + gl0 + dl + np.linspace(4, n - 4, 96)[:, None] * np.ones((1, 96)), rdr.soff + gs0 + ds + np.ones((96, 1)) * np.linspace(4, n - 4, 96)[None, :])
    src = chain_restoration_source(f"{a.chain}/chain.npz", f"{a.chain}/chain.json", a.rdr_lbl, key=a.chain_key, shift_lines=dl, shift_samples=ds)
    fp_lat, fp_lon = rdr.pix_to_latlon(rdr.loff + gl0 + np.array([0, n]) + dl, rdr.soff + gs0 + np.array([0, n]) + ds)
    fp = (float(min(fp_lat)), float(max(fp_lat)), float(min(fp_lon)), float(max(fp_lon)))
    L = Layer("restored_6cm", src, provenance="measured", native_res_m=rdr.scale / SC, footprint=fp, feather_px=10, extras=dict(chain=os.path.basename(a.chain.rstrip("/")), key=a.chain_key))
    layers.append(match_overlay(base, L, la, lo, 0.25)); refine.append((*fp, a.overlay_level))

if a.ingenuity and a.placement:
    p = json.load(open(a.placement)); glNW, gsNW = p["nw_corner_global_ref"]; M = np.load(a.ingenuity).astype(np.float64)       # 5 cm version of the mosaic
    ysc = math.cos(math.radians(15)) / math.cos(math.radians(p["lat_deg"])); st = 0.05 / rdr.scale                                 # RDR lines per mosaic pixel
    dl, ds = SH                                                                                                                     # same RDR->ortho shift (same reference pass)
    # mosaic pixel (i, j) centre -> RDR (line, sample): lines step st; samples step st * ysc. ArraySource needs one step, so resample columns first.
    Mc = ndi.zoom(np.nan_to_num(M, nan=0), (1, ysc), order=1); V = ndi.zoom(np.isfinite(M).astype(np.float32), (1, ysc), order=1) > 0.99
    src = ArraySource(np.where(V, Mc, np.nan), rdr, rdr.loff + glNW + 0.5 * st - 0.5 + dl, rdr.soff + gsNW + 0.5 * st - 0.5 + ds, st, valid=V)
    fl, fs = rdr.pix_to_latlon(rdr.loff + glNW + np.array([0, M.shape[0] * st]) + dl, rdr.soff + gsNW + np.array([0, Mc.shape[1] * st]) + ds)
    fp = (float(min(fl)), float(max(fl)), float(min(fs)), float(max(fs)))
    la, lo = np.meshgrid(np.linspace(fp[0], fp[1], 160), np.linspace(fp[2], fp[3], 60), indexing="ij")
    L = Layer("ingenuity_flights4-5", src, provenance="measured", native_res_m=0.05, footprint=fp, feather_px=8, extras=dict(source=p.get("mosaic", "")))
    layers.append(match_overlay(base, L, la, lo, 0.25)); refine.append((*fp, a.focus_level))

if a.detail_png:
    from PIL import Image
    img = np.asarray(Image.open(a.detail_png).convert("L"), np.float64) / 255.0
    layers.append(Layer("detail_regolith", kind="tiled", provenance="simulated", image=img, repeat_m=1.5, amplitude=0.12, only_below_res_m=0.2, native_res_m=0.01,
                        extras=dict(note="repeating texture, applied only where no measured layer finer than 0.2 m exists; flagged as simulated")))

tb = TileBuilder(dtm, layers, root, a.out, grid=a.grid, tex=a.tex, base_level=a.base_level, refine=refine)
tb.build(); json.dump(dict(registration=reg, refine_regions=refine), open(os.path.join(a.out, "build_info.json"), "w"), indent=1)
print("tiles:", len(tb.records), "->", a.out)

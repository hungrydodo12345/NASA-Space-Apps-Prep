"""Data sources for the game tiles: elevation (HiRISE stereo DTM) and georeferenced texture rasters.

Every source answers the same question: given arrays of lat/lon, return values (and a validity mask). Sources are therefore interchangeable layers.
"""
import json
import numpy as np
from scipy import ndimage as ndi
from .mars import Equirect


class DtmSource:
    """HiRISE stereo DTM (DTEEC_*.IMG, PDS3, float32, embedded label). Heights in metres above the areoid; bilinear sampling, holes filled from neighbours."""
    def __init__(self, path):
        import re
        head = open(path, "rb").read(60000).decode("latin-1")
        g = lambda k: re.search(rf"^\s*{k}\s*=\s*([-+0-9.eE]+)", head, re.M).group(1)
        self.proj = Equirect.from_label(head); self.lines, self.samples = int(g("LINES")), int(g("LINE_SAMPLES"))
        rec, start = int(g("RECORD_BYTES")), int(re.search(r"\^IMAGE\s*=\s*(\d+)", head).group(1))
        self.a = np.memmap(path, dtype="<f4", mode="r", offset=rec * (start - 1), shape=(self.lines, self.samples))
        self.valid_min = float(g("VALID_MINIMUM")); self.post_m = self.proj.scale

    def sample(self, lat, lon):
        l, s = self.proj.latlon_to_pix(lat, lon)
        l0, l1 = int(np.floor(l.min())) - 2, int(np.ceil(l.max())) + 3; s0, s1 = int(np.floor(s.min())) - 2, int(np.ceil(s.max())) + 3
        l0c, s0c = max(l0, 0), max(s0, 0); l1c, s1c = min(l1, self.lines), min(s1, self.samples)
        blk = np.full((l1 - l0, s1 - s0), np.nan, np.float64)
        if l1c > l0c and s1c > s0c: blk[l0c - l0:l1c - l0, s0c - s0:s1c - s0] = self.a[l0c:l1c, s0c:s1c]
        bad = ~np.isfinite(blk) | (blk < self.valid_min); valid_blk = ~bad
        if bad.all(): return np.zeros_like(l), np.zeros(l.shape, bool)
        if bad.any(): blk = blk[tuple(ndi.distance_transform_edt(bad, return_distances=False, return_indices=True))]
        h = ndi.map_coordinates(blk, [l - l0, s - s0], order=1, mode="nearest")
        v = ndi.map_coordinates(valid_blk.astype(np.float32), [l - l0, s - s0], order=1, mode="constant", cval=0) > 0.5
        return h, v


class Jp2Source:
    """Map-projected JPEG2000 raster (HiRISE RED ORTHO or RDR) read through its resolution pyramid. Values are DN; `gain/offset` convert to I/F if given."""
    def __init__(self, jp2_path, label_path, gain=1.0, offset=0.0):
        import glymur
        self.j = glymur.Jp2k(jp2_path); self.proj = Equirect.from_file(label_path); self.gain, self.offset = gain, offset
        self.lines, self.samples = self.j.shape[:2]

    def sample(self, lat, lon, out_res_m=None):
        l, s = self.proj.latlon_to_pix(lat, lon)
        f = 1 if out_res_m is None else max(1, int(2 ** np.floor(np.log2(max(out_res_m / self.proj.scale, 1.0)))))   # pyramid level no finer than needed
        f = min(f, 256)
        l0, l1 = int(np.floor(l.min())) - 2 * f, int(np.ceil(l.max())) + 2 * f; s0, s1 = int(np.floor(s.min())) - 2 * f, int(np.ceil(s.max())) + 2 * f
        l0, s0 = max(0, l0 - l0 % f), max(0, s0 - s0 % f); l1, s1 = min(self.lines, l1), min(self.samples, s1)
        if l1 <= l0 or s1 <= s0: return np.zeros_like(l), np.zeros(l.shape, bool)
        blk = np.asarray(self.j[l0:l1:f, s0:s1:f] if f > 1 else self.j[l0:l1, s0:s1]).astype(np.float64)
        valid = blk > 0; blk = blk * self.gain + self.offset
        if not valid.any(): return np.zeros_like(l), np.zeros(l.shape, bool)
        fill = np.where(valid, blk, blk[valid].mean())
        val = ndi.map_coordinates(fill, [(l - l0) / f, (s - s0) / f], order=1, mode="nearest")
        v = ndi.map_coordinates(valid.astype(np.float32), [(l - l0) / f, (s - s0) / f], order=1, mode="constant", cval=0) > 0.5
        return val, v


class ArraySource:
    """An in-memory georeferenced raster on an equirectangular grid: value at pixel (i, j) has projection coordinates (line0 + i * step, samp0 + j * step)
    in `proj`. Used for restorations (6.25 cm grid inside a HiRISE RDR projection) and for Ingenuity mosaics placed in that grid."""
    def __init__(self, array, proj, line0, samp0, step, valid=None, order=1):
        self.a = np.asarray(array, np.float64); self.proj, self.l0, self.s0, self.step, self.order = proj, line0, samp0, step, order
        self.v = np.isfinite(self.a) if valid is None else (np.asarray(valid, bool) & np.isfinite(self.a))
        self.fill = np.where(self.v, self.a, np.nanmean(self.a[self.v]) if self.v.any() else 0.0)

    def sample(self, lat, lon, out_res_m=None):
        l, s = self.proj.latlon_to_pix(lat, lon); i, j = (l - self.l0) / self.step, (s - self.s0) / self.step
        src = self.fill
        if out_res_m is not None:                                                                         # area-average when the output is coarser
            k = out_res_m / (self.proj.scale * self.step)
            if k > 1.5: src = ndi.gaussian_filter(self.fill, k / 2.5)
        val = ndi.map_coordinates(src, [i, j], order=self.order, mode="nearest")
        v = ndi.map_coordinates(self.v.astype(np.float32), [i, j], order=1, mode="constant", cval=0) > 0.99
        return val, v


def rdr_projection(label_path):
    return Equirect.from_file(label_path)


def chain_restoration_source(chain_npz, chain_json, rdr_label, key="restored", shift_lines=0.0, shift_samples=0.0):
    """A restoration from scripts/run_chain.py as a texture source. The chain grid is the reference pass's RDR grid; `origin_global` is (line, sample)
    minus the label projection offsets ('global' coordinates); the restored image is `scale` x finer. An optional shift registers the RDR grid to the DTM."""
    d = np.load(chain_npz); meta = json.load(open(chain_json)); proj = Equirect.from_file(rdr_label); SC = meta["scale"]
    gl0, gs0 = meta["origin_global"]; X = d[key].astype(np.float64)
    line0 = proj.loff + gl0 + (0.5 / SC - 0.5) + shift_lines; samp0 = proj.soff + gs0 + (0.5 / SC - 0.5) + shift_samples     # HR pixel centres in RDR pixels
    return ArraySource(X, proj, line0, samp0, 1.0 / SC)

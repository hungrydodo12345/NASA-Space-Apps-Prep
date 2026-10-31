"""HiRISE stereo DTM (DTEEC_*.IMG, PDS3, float32) on the same equirectangular grid as the orthoimage.

DTM pixel = 4 ortho pixels (1.0097 m vs 0.2524 m), so ortho(line, sample) = (4*l - 2.5, 4*s - 3.5).
"""
import re
import numpy as np
from scipy import ndimage as ndi


class Dtm:
    def __init__(self, path):
        head = open(path, "rb").read(30000).decode("latin-1")
        g = lambda k: re.search(rf"^\s*{k}\s*=\s*([-+0-9.eE]+)", head, re.M).group(1)
        self.lines, self.samples = int(g("LINES")), int(g("LINE_SAMPLES"))
        rec, start = int(g("RECORD_BYTES")), int(re.search(r"\^IMAGE\s*=\s*(\d+)", head).group(1))
        self.a = np.memmap(path, dtype="<f4", mode="r", offset=rec * (start - 1), shape=(self.lines, self.samples))
        self.valid_min = float(g("VALID_MINIMUM"))

    def heights_for_ortho(self, top, left, h, w, step=4):
        """Heights (m, areoid) on an ortho-pixel window, sampled every `step` ortho px (bilinear)."""
        ol = top + np.arange(0, h, step) + 0.5 * (step - 1)
        os_ = left + np.arange(0, w, step) + 0.5 * (step - 1)
        dl, ds = (ol + 2.5) / 4.0, (os_ + 3.5) / 4.0
        l0, l1 = int(np.floor(dl.min())) - 2, int(np.ceil(dl.max())) + 3
        s0, s1 = int(np.floor(ds.min())) - 2, int(np.ceil(ds.max())) + 3
        blk = np.array(self.a[l0:l1, s0:s1], dtype=np.float64)
        bad = blk < self.valid_min
        if bad.any():   # fill rare holes from neighbours (marked so callers can flag them)
            idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
            blk = blk[tuple(idx)]
        LL, SS = np.meshgrid(dl - l0, ds - s0, indexing="ij")
        return ndi.map_coordinates(blk, [LL, SS], order=1, mode="nearest"), float(bad.mean())

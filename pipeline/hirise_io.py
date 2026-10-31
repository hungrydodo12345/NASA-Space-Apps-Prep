"""HiRISE RDR access: label parsing, global-grid addressing, region reads, I/F calibration.

All RDR products here share one equirectangular grid (same centre + scale); products differ only
by LINE/SAMPLE_PROJECTION_OFFSET, so co-registration to the pixel is an integer translation.
"""
import re
import numpy as np
import glymur


def parse_label(path):
    t = open(path, errors="replace").read()
    def num(key):
        m = re.search(rf"^\s*{key}\s*=\s*([-+0-9.eE]+)", t, re.M)
        return float(m.group(1)) if m else None
    return dict(lines=int(num("LINES")), samples=int(num("LINE_SAMPLES")),
                line_off=num("LINE_PROJECTION_OFFSET"), sample_off=num("SAMPLE_PROJECTION_OFFSET"),
                scale=num("MAP_SCALE"), gain=num("SCALING_FACTOR"), offset=num("OFFSET"))


class Rdr:
    def __init__(self, base):
        self.lbl = parse_label(base + ".LBL")
        self.jp2 = glymur.Jp2k(base + ".JP2")
        self.id = base.split("/")[-1]

    def extent(self):
        """Global (line, sample) range [lo, hi) covered by this product."""
        L = self.lbl
        return (-L["line_off"], -L["line_off"] + L["lines"]), (-L["sample_off"], -L["sample_off"] + L["samples"])

    def read_dn(self, gl, gs, h, w):
        """Raw DN for global-grid window starting at (gl, gs)."""
        L = self.lbl
        r, c = int(round(gl + L["line_off"])), int(round(gs + L["sample_off"]))
        return np.asarray(self.jp2[r:r + h, c:c + w])

    def read_if(self, gl, gs, h, w):
        dn = self.read_dn(gl, gs, h, w).astype(np.float64)
        return np.where(dn > 0, dn * self.lbl["gain"] + self.lbl["offset"], np.nan)

"""Rover ground truth: geolocate AI4Mars Perseverance labels and map them onto the HiRISE ortho grid."""
import json, glob, os, re, math
import numpy as np
from PIL import Image

GEO_GROUPS = {"bedrock": range(0, 7), "float_rock": range(10, 18), "sand_bedform": (20, 21), "sand_plain": (22,), "pebbles": (30,)}
NAV_GROUPS = {"soil": (0,), "bedrock": (1,), "sand": (2,), "big_rock": (3,)}

# ESP_045994_1985_RED_A_01_ORTHO (DTEEC pair) projection, from its PDS label
ORTHO = dict(R=3394839.8133163, clat=15.0, clon=77.44, scale=0.25241583171779, loff=4353979.5, soff=12838.5,
             lines=57064, samples=26691)


def latlon_to_ortho(lat, lon, P=ORTHO):
    y = P["R"] * math.radians(lat)
    x = P["R"] * math.cos(math.radians(P["clat"])) * math.radians(lon - P["clon"])
    return P["loff"] - y / P["scale"], P["soff"] + x / P["scale"]       # (line, sample), 0-based approx (+-1 px)


class Traverse:
    def __init__(self, path):
        f = json.load(open(path))["features"]
        self.seg = sorted(((p["properties"]["SCLK_START"], p["properties"]["SCLK_END"], np.array(p["geometry"]["coordinates"])[:, :2])
                           for p in f), key=lambda s: s[0])

    def position(self, sclk):
        """(lon, lat) of the rover at spacecraft clock `sclk`."""
        prev = self.seg[0][2][0]
        for s0, s1, xy in self.seg:
            if sclk < s0: return tuple(prev)
            if sclk <= s1:
                t = (sclk - s0) / max(s1 - s0, 1)
                d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
                if d[-1] == 0: return tuple(xy[-1])
                k = np.searchsorted(d, t * d[-1]); k = min(max(k, 1), len(xy) - 1)
                a = (t * d[-1] - d[k - 1]) / max(d[k] - d[k - 1], 1e-12)
                return tuple(xy[k - 1] + a * (xy[k] - xy[k - 1]))
            prev = xy[-1]
        return tuple(prev)


def parse_name(fn):
    m = re.match(r"^[A-Z0-9]{3}_(\d{4})_(\d{10})_", os.path.basename(fn))
    return (int(m.group(1)), int(m.group(2))) if m else None


def fractions(png, groups):
    a = np.asarray(Image.open(png))
    if a.ndim == 3: a = a[..., 0]
    valid = a != 255
    nv = int(valid.sum())
    if nv == 0: return None, 0
    return {g: float(np.isin(a[valid], list(v)).sum()) / nv for g, v in groups.items()}, nv


def build(label_root, traverse_path):
    tr = Traverse(traverse_path)
    rows = []
    for kind, groups in (("M2020_GEO", GEO_GROUPS), ("NAV", NAV_GROUPS)):
        for png in glob.glob(f"{label_root}/{kind}/**/*.png", recursive=True):
            ps = parse_name(png)
            if not ps: continue
            sol, sclk = ps
            fr, nv = fractions(png, groups)
            if fr is None: continue
            lon, lat = tr.position(sclk)
            line, samp = latlon_to_ortho(lat, lon)
            rows.append(dict(kind=kind, file=os.path.basename(png), sol=sol, sclk=sclk, lon=lon, lat=lat,
                             line=line, sample=samp, npx=nv, **{f"f_{k}": v for k, v in fr.items()}))
    return rows

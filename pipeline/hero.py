"""Hero-area geometry helpers: ortho grid <-> lat/lon <-> RDR global grid."""
import math
from .groundtruth import ORTHO

R_MARS = ORTHO["R"]


def ortho_to_latlon(line, sample, P=ORTHO):
    y = (P["loff"] - line) * P["scale"]; x = (sample - P["soff"]) * P["scale"]
    return math.degrees(y / P["R"]), P["clon"] + math.degrees(x / (P["R"] * math.cos(math.radians(P["clat"]))))


def latlon_to_rdr_global(lat, lon, scale=0.25, clat=15.0, clon=180.0):
    """Global (line, sample) on the shared RDR grid (see hirise_io: global = pixel - projection offset)."""
    y = R_MARS * math.radians(lat); x = R_MARS * math.cos(math.radians(clat)) * math.radians(lon - clon)
    return -y / scale, x / scale

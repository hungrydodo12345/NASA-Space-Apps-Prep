"""Mars geometry for the game tiles: map projections of HiRISE products, and curved-surface coordinates.

Curvature. Every vertex is placed on a sphere of radius R + h (R = the reference radius of the HiRISE products, h = DTM elevation) in Mars body-fixed
coordinates, then expressed in an east-north-up (ENU) frame tangent at the scene origin. The surface therefore falls away from the tangent plane by
about d^2 / (2R): 0.15 m at 1 km, 3.7 m at 5 km, 14.7 m at 10 km, so horizons and long sight lines behave as on Mars.

glTF axes (right-handed, +Y up): x = east, y = up, z = -north.
"""
import re
import numpy as np

R_HIRISE = 3394839.8133163          # m; A_AXIS_RADIUS of the Jezero HiRISE products (sphere radius at the projection's centre latitude)


class Equirect:
    """PDS equirectangular projection (HiRISE RDR, ORTHO and DTM labels). Pixel coordinates are 0-based, pixel centres at integers."""
    def __init__(self, center_lat, center_lon, scale, line_off, sample_off, radius=R_HIRISE, lines=None, samples=None):
        self.clat, self.clon, self.scale, self.loff, self.soff, self.R = center_lat, center_lon, scale, line_off, sample_off, radius
        self.lines, self.samples = lines, samples

    @classmethod
    def from_label(cls, text):
        g = lambda k: float(re.search(rf"^\s*{k}\s*=\s*([-+0-9.eE]+)", text, re.M).group(1))
        gi = lambda k: int(re.search(rf"^\s*{k}\s*=\s*(\d+)", text, re.M).group(1)) if re.search(rf"^\s*{k}\s*=\s*\d+", text, re.M) else None
        return cls(g("CENTER_LATITUDE"), g("CENTER_LONGITUDE"), g("MAP_SCALE"), g("LINE_PROJECTION_OFFSET"), g("SAMPLE_PROJECTION_OFFSET"),
                   g("A_AXIS_RADIUS") * 1000.0, gi("LINES"), gi("LINE_SAMPLES"))

    @classmethod
    def from_file(cls, path):
        head = open(path, "rb").read(60000).decode("latin-1")
        return cls.from_label(head)

    def latlon_to_pix(self, lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        line = self.loff - np.radians(lat) * self.R / self.scale
        samp = self.soff + np.radians(lon - self.clon) * self.R * np.cos(np.radians(self.clat)) / self.scale
        return line, samp

    def pix_to_latlon(self, line, samp):
        line, samp = np.asarray(line, float), np.asarray(samp, float)
        lat = np.degrees((self.loff - line) * self.scale / self.R)
        lon = self.clon + np.degrees((samp - self.soff) * self.scale / (self.R * np.cos(np.radians(self.clat))))
        return lat, lon


def body_fixed(lat, lon, h, R=R_HIRISE):
    """Mars body-fixed Cartesian coordinates (m) of points at planetocentric lat/lon (deg, east-positive) and height h above the reference sphere."""
    la, lo = np.radians(lat), np.radians(lon); r = R + np.asarray(h, float)
    return np.stack([r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)], -1)


class EnuFrame:
    """East-north-up frame tangent to the sphere at (lat0, lon0, h0). Converts body-fixed points to local metres (curvature included)."""
    def __init__(self, lat0, lon0, h0=0.0, R=R_HIRISE):
        self.lat0, self.lon0, self.h0, self.R = lat0, lon0, h0, R
        la, lo = np.radians(lat0), np.radians(lon0)
        self.M = np.array([[-np.sin(lo), np.cos(lo), 0.0],
                           [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                           [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])       # rows: east, north, up
        self.P0 = body_fixed(lat0, lon0, h0, R)

    def from_latlon(self, lat, lon, h):
        return (body_fixed(lat, lon, h, self.R) - self.P0) @ self.M.T                                  # (..., 3) east, north, up

    def to_gltf(self, enu):
        """ENU -> glTF axes (x east, y up, z = -north)."""
        return np.stack([enu[..., 0], enu[..., 2], -enu[..., 1]], -1)

    def enu_to_body_matrix(self):
        """4x4 matrix (column-major list, as 3D Tiles 'transform') taking z-up ENU coordinates (x east, y north, z up) to body-fixed metres.
        3D Tiles rotates y-up glTF content to z-up before applying it, which maps our glTF axes (x E, y U, z -N) onto this ENU frame."""
        A = np.eye(4); A[:3, :3] = self.M.T; A[:3, 3] = self.P0
        return A.T.reshape(-1).tolist()


def curvature_drop(d, R=R_HIRISE):
    """Height of the sphere below its tangent plane at ground distance d (m): R (1 - cos(d/R)) ~ d^2 / 2R."""
    return R * (1 - np.cos(np.asarray(d, float) / R))

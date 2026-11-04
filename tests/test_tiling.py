"""Tests for the game tiles: curvature, projections, mesh seams and GLB validity (synthetic data, no downloads)."""
import os, json, math, tempfile
import numpy as np
from tiling.mars import Equirect, EnuFrame, curvature_drop, R_HIRISE
from tiling.build import TileBuilder, grid_mesh
from tiling.texture import Layer
from tiling.gltf import read_glb_json


class FlatDtm:                                     # synthetic DTM: a gentle slope plus a 3 m bump, heights in metres
    post_m = 1.0
    def sample(self, lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        h = 50.0 * (lat - 18.44) + 3.0 * np.exp(-((lat - 18.441) ** 2 + (lon - 77.451) ** 2) / 1e-7)
        return h, np.ones(lat.shape, bool)


class Ramp:                                        # synthetic texture source
    def sample(self, lat, lon, res=None):
        return (np.asarray(lon) - 77.44) * 100.0, np.ones(np.shape(lat), bool)


def test_equirect_roundtrip():
    p = Equirect(15.0, 77.44, 0.2524, 4353979.5, 12838.5)
    lat, lon = 18.4413, 77.4519; l, s = p.latlon_to_pix(lat, lon); la, lo = p.pix_to_latlon(l, s)
    assert abs(la - lat) < 1e-10 and abs(lo - lon) < 1e-10


def test_curvature_drop_matches_sphere():
    f = EnuFrame(18.44, 77.45, 0.0)
    for d in (1000.0, 5000.0, 10000.0):
        dlat = math.degrees(d / R_HIRISE); up = f.from_latlon(18.44 + dlat, 77.45, 0.0)[2]
        assert abs(-up - curvature_drop(d)) < 1e-3                                   # true sphere drop in the tangent frame
        assert abs(curvature_drop(d) - d * d / (2 * R_HIRISE)) < 1e-3 * curvature_drop(d) + 1e-6
    assert 14.0 < curvature_drop(10000.0) < 15.5


def test_enu_axes():
    f = EnuFrame(18.44, 77.45, 0.0); dl = math.degrees(10.0 / R_HIRISE)
    n = f.from_latlon(18.44 + dl, 77.45, 0.0); e = f.from_latlon(18.44, 77.45 + dl / math.cos(math.radians(18.44)), 0.0); u = f.from_latlon(18.44, 77.45, 10.0)
    assert n[1] > 9.99 and abs(n[0]) < 1e-6 and e[0] > 9.99 and abs(e[1]) < 1e-3 and abs(u[2] - 10) < 1e-6


def test_mesh_faces_up_and_skirt_below():
    f = EnuFrame(18.44, 77.45, 0.0); n = 9
    lat = np.linspace(18.4410, 18.4400, n)[:, None] * np.ones((1, n)); lon = np.ones((n, 1)) * np.linspace(77.450, 77.451, n)[None, :]
    P, N, UV, T, c, _ = grid_mesh(f, lat, lon, np.zeros_like(lat), skirt_m=2.0)
    assert (N[: n * n, 1] > 0.99).all()                                               # glTF y is up
    v = P[T[:2 * (n - 1) ** 2]]; nrm = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]); assert (nrm[:, 1] > 0).all()   # counter-clockwise from above
    assert P[n * n:, 1].max() < P[: n * n, 1].min() - 1.9                           # skirt hangs 2 m below the (flat) surface


def test_tileset_seams_and_glb():
    with tempfile.TemporaryDirectory() as d:
        root = (18.4400, 18.4420, 77.4500, 77.4520)
        L = Layer("ramp", Ramp(), lo=0.0, hi=2.0)
        tb = TileBuilder(FlatDtm(), [L], root, d, grid=17, tex=32, base_level=1); tb.build(verbose=False)
        assert len(tb.records) == 5 and os.path.exists(os.path.join(d, "tileset.json")) and os.path.exists(os.path.join(d, "scene.json"))
        g = read_glb_json(os.path.join(d, "tiles/1/0/0.glb")); assert g["asset"]["version"] == "2.0" and "images" in g and g["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] >= 0
        # neighbouring tiles share their edge exactly: rebuild the two edge rows and compare in scene coordinates
        f = tb.frame; n = 17
        from tiling.build import tile_bounds
        a = tile_bounds(root, 1, 0, 0); b = tile_bounds(root, 1, 1, 0); assert abs(a[3] - b[2]) < 1e-12
        lat = np.linspace(a[1], a[0], n); ea = f.from_latlon(lat, np.full(n, a[3]), FlatDtm().sample(lat, np.full(n, a[3]))[0]); eb = f.from_latlon(lat, np.full(n, b[2]), FlatDtm().sample(lat, np.full(n, b[2]))[0])
        assert np.abs(ea - eb).max() < 1e-9
        ts = json.load(open(os.path.join(d, "tileset.json"))); assert len(ts["root"]["children"]) == 4 and len(ts["root"]["transform"]) == 16

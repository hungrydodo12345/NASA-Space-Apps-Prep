"""Quadtree of curved, textured terrain tiles (glTF) with a 3D-Tiles-style tileset and a flat manifest for game engines.

Tiles are cut on a latitude/longitude grid, so neighbours share edge vertices exactly. Each tile's vertices are on the Mars sphere (R + DTM height),
in the scene's ENU frame, stored relative to the tile centre (glTF node translation) so float32 keeps millimetre precision. Skirts hide cracks
between levels. Heights never get finer than the DTM: below one DTM post the mesh is just a denser sampling of the same bilinear surface.
"""
import os, io, json, math
import numpy as np
from PIL import Image
from .mars import EnuFrame
from .texture import composite
from .gltf import write_glb


def tile_bounds(root, level, x, y):
    a0, a1, o0, o1 = root; n = 2 ** level; dlat, dlon = (a1 - a0) / n, (o1 - o0) / n
    return (a1 - (y + 1) * dlat, a1 - y * dlat, o0 + x * dlon, o0 + (x + 1) * dlon)        # y counts from the north


def grid_mesh(frame, lat, lon, h, skirt_m):
    """Vertices, normals, uvs and triangle indices for a regular n x n grid (rows north->south, columns west->east), plus a skirt."""
    n = lat.shape[0]; enu = frame.from_latlon(lat, lon, h)
    c = enu.reshape(-1, 3).mean(0); P = enu - c
    dx = np.gradient(P, axis=1); dy = np.gradient(P, axis=0); nr = np.cross(dy, dx); nr /= np.linalg.norm(nr, axis=-1, keepdims=True) + 1e-12
    if nr[..., 2].mean() < 0: nr = -nr                                                       # up-facing in ENU
    up = frame.from_latlon(lat, lon, h + 1.0) - enu                                          # local vertical (curved), for the skirt direction
    u = np.broadcast_to(np.linspace(0, 1, n)[None, :], (n, n)); v = np.broadcast_to(np.linspace(0, 1, n)[:, None], (n, n))
    V = [P.reshape(-1, 3)]; N = [nr.reshape(-1, 3)]; UV = [np.stack([u, v], -1).reshape(-1, 2)]
    idx = np.arange(n * n).reshape(n, n); tris = []
    a, b, cc, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    tris.append(np.stack([a, cc, b], -1).reshape(-1, 3)); tris.append(np.stack([b, cc, d], -1).reshape(-1, 3))
    ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])        # border loop, clockwise from the north-west corner
    base = n * n; sk = (P.reshape(-1, 3)[ring] - skirt_m * up.reshape(-1, 3)[ring])
    V.append(sk); N.append(nr.reshape(-1, 3)[ring]); UV.append(UV[0][ring])
    k = len(ring); r0 = ring; r1 = base + np.arange(k); r0n = np.roll(r0, -1); r1n = np.roll(r1, -1)
    tris.append(np.stack([r0, r0n, r1], -1)); tris.append(np.stack([r0n, r1n, r1], -1))
    Pv = np.concatenate(V); Nv = np.concatenate(N)
    to_g = lambda q: np.stack([q[:, 0], q[:, 2], -q[:, 1]], -1)                              # ENU -> glTF axes
    T = np.concatenate(tris)
    return to_g(Pv), to_g(Nv), np.concatenate(UV), T, c, (enu.reshape(-1, 3).min(0), enu.reshape(-1, 3).max(0))


class TileBuilder:
    def __init__(self, dtm, layers, root_bounds, out_dir, frame=None, grid=65, tex=256, base_level=3, refine=None, skirt_factor=2.0):
        """dtm: DtmSource. layers: list of texture Layers (bottom to top). root_bounds: (lat_min, lat_max, lon_min, lon_max).
        refine: list of (lat_min, lat_max, lon_min, lon_max, max_level) regions refined beyond base_level."""
        self.dtm, self.layers, self.root, self.out, self.grid, self.tex, self.base_level = dtm, layers, root_bounds, out_dir, grid, tex, base_level
        self.refine = refine or []; self.skirt_factor = skirt_factor
        a0, a1, o0, o1 = root_bounds; lat0, lon0 = (a0 + a1) / 2, (o0 + o1) / 2
        h0 = float(dtm.sample(np.array([lat0]), np.array([lon0]))[0][0])
        self.frame = frame or EnuFrame(lat0, lon0, h0)
        os.makedirs(out_dir, exist_ok=True); self.records = {}

    def size_m(self, b):
        a0, a1, o0, o1 = b; R = self.frame.R
        return math.radians(a1 - a0) * R, math.radians(o1 - o0) * R * math.cos(math.radians((a0 + a1) / 2))

    def max_level_for(self, b):
        m = self.base_level
        for (a0, a1, o0, o1, lv) in self.refine:
            if not (b[1] < a0 or b[0] > a1 or b[3] < o0 or b[2] > o1): m = max(m, lv)
        return m

    def build_tile(self, level, x, y):
        b = tile_bounds(self.root, level, x, y); a0, a1, o0, o1 = b; n = self.grid
        lat = np.linspace(a1, a0, n)[:, None] * np.ones((1, n)); lon = np.ones((n, 1)) * np.linspace(o0, o1, n)[None, :]
        h, hv = self.dtm.sample(lat, lon)
        hm, wm = self.size_m(b); spacing = max(hm, wm) / (n - 1)
        P, N, UV, T, c, (mn, mx) = grid_mesh(self.frame, lat, lon, h, self.skirt_factor * spacing)
        t = self.tex; tl = np.linspace(a1, a0, 2 * t + 1)[1::2]; to = np.linspace(o0, o1, 2 * t + 1)[1::2]
        TL, TO = np.meshgrid(tl, to, indexing="ij"); texel = max(hm, wm) / t
        en = self.frame.from_latlon(TL, TO, np.zeros_like(TL))
        rgb, prov, info = composite(self.layers, TL, TO, en[..., 0], en[..., 1], texel)
        d = os.path.join(self.out, "tiles", str(level), str(x)); os.makedirs(d, exist_ok=True)
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
        cg = [float(c[0]), float(c[2]), float(-c[1])]
        write_glb(os.path.join(d, f"{y}.glb"), P, N, UV, T, buf.getvalue(), translation=cg, name=f"L{level}_{x}_{y}",
                  extras=dict(level=level, x=x, y=y, bounds_deg=list(b), texel_m=texel, vertex_spacing_m=spacing, dtm_valid_frac=float(hv.mean()), **info))
        Image.fromarray(prov).save(os.path.join(d, f"{y}_prov.png"), optimize=True)
        half = (mx - mn) / 2 + np.array([0, 0, self.skirt_factor * spacing / 2]); ctr = (mx + mn) / 2
        rec = dict(level=level, x=x, y=y, uri=f"tiles/{level}/{x}/{y}.glb", provenance_uri=f"tiles/{level}/{x}/{y}_prov.png", bounds_deg=list(b),
                   center_enu=ctr.tolist(), half_size_enu=half.tolist(), geometric_error_m=float(spacing), texel_m=float(texel),
                   measured_frac=info["measured_frac"], simulated_frac=info["simulated_frac"], dtm_valid_frac=float(hv.mean()), children=[])
        self.records[(level, x, y)] = rec
        return rec

    def build(self, verbose=True):
        def rec(level, x, y):
            r = self.build_tile(level, x, y)
            if verbose: print(f"tile L{level} {x},{y}: texel {r['texel_m']:.3f} m, mesh {r['geometric_error_m']:.2f} m, measured {r['measured_frac']:.2f}", flush=True)
            b = tile_bounds(self.root, level, x, y)
            if level < self.max_level_for(b):
                for dy in (0, 1):
                    for dx in (0, 1):
                        cx, cy = 2 * x + dx, 2 * y + dy                     # always all four: REPLACE refinement must cover the parent completely
                        r["children"].append(rec(level + 1, cx, cy)["uri"])
            return r
        rec(0, 0, 0); self.write_index()

    def write_index(self):
        R = self.records
        def node(k):
            r = R[k]; c, h = r["center_enu"], r["half_size_enu"]
            kids = [kk for kk in R if kk[0] == k[0] + 1 and kk[1] // 2 == k[1] and kk[2] // 2 == k[2]]
            return dict(boundingVolume=dict(box=[c[0], c[1], c[2], h[0], 0, 0, 0, h[1], 0, 0, 0, h[2]]), geometricError=r["geometric_error_m"] if kids else 0.0,
                        refine="REPLACE", content=dict(uri=r["uri"]), children=[node(kk) for kk in sorted(kids)])
        root = node((0, 0, 0)); root["transform"] = self.frame.enu_to_body_matrix()
        ts = dict(asset=dict(version="1.1", tilesetVersion="1", generator="NASA-Space-Apps-Prep tiling"), geometricError=R[(0, 0, 0)]["geometric_error_m"] * 4,
                  root=root, extras=dict(body="Mars", radius_m=self.frame.R, frame="ENU (x east, y north, z up) at the scene origin; transform maps it to Mars body-fixed metres"))
        json.dump(ts, open(os.path.join(self.out, "tileset.json"), "w"), indent=1)
        scene = dict(body="Mars", reference_radius_m=self.frame.R, origin=dict(lat_deg=self.frame.lat0, lon_deg=self.frame.lon0, height_m=self.frame.h0),
                     axes="glTF: x = east, y = up, z = -north (metres, curved surface: true sphere heights in this tangent frame)",
                     root_bounds_deg=list(self.root), grid_vertices=self.grid, texture_px=self.tex,
                     layers=[dict(name=L.name, kind=L.kind, provenance=L.provenance, native_res_m=L.native_res_m, footprint=L.footprint, **L.extras) for L in self.layers],
                     provenance_texture="R = measured weight, G = simulated weight (0-255)",
                     tiles=sorted(R.values(), key=lambda r: (r["level"], r["x"], r["y"])))
        json.dump(scene, open(os.path.join(self.out, "scene.json"), "w"), indent=1)

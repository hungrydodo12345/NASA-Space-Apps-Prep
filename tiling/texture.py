"""Texture layer stack for the game tiles.

A layer is either
  - 'georef': a georeferenced raster (any object with .sample(lat, lon, out_res_m) -> (values, valid)), e.g. the HiRISE orthoimage, a 6.25 cm
    restoration, or an Ingenuity mosaic; or
  - 'tiled':  a small image repeated every `repeat_m` metres on the ground (detail texture), modulating the layers below.
Every layer carries a provenance tag, 'measured' (from data) or 'simulated' (added detail), which is written to a separate per-tile provenance texture
(R = measured weight, G = simulated weight) so the game can show, hide or label simulated detail. Layers are composited bottom to top.
"""
import numpy as np
from dataclasses import dataclass, field
from scipy import ndimage as ndi


@dataclass
class Layer:
    name: str
    source: object = None                 # georef: has .sample(lat, lon, out_res_m)
    kind: str = "georef"                  # 'georef' | 'tiled'
    provenance: str = "measured"          # 'measured' | 'simulated'
    lo: float = 0.0                       # value range mapped to 0..1
    hi: float = 1.0
    opacity: float = 1.0
    feather_px: float = 6.0               # soft edge where a georef layer ends (in output texels)
    native_res_m: float = 0.25            # finest useful resolution (for LOD refinement)
    footprint: tuple = None               # (lat_min, lat_max, lon_min, lon_max) of the data, for refinement
    image: np.ndarray = None              # tiled: the repeating image (0..1)
    repeat_m: float = 2.0                 # tiled: ground size of one repeat
    amplitude: float = 0.15               # tiled: modulation depth
    only_below_res_m: float = None        # tiled: apply only where the measured layers are coarser than this (no simulated detail over real detail)
    extras: dict = field(default_factory=dict)

    def set_range_from(self, lat, lon, res_m, pct=(0.5, 99.5)):
        v, m = self.source.sample(lat, lon, res_m)
        if m.any(): self.lo, self.hi = [float(x) for x in np.percentile(v[m], pct)]
        return self


def match_overlay(base: Layer, overlay: Layer, lat, lon, res_m):
    """Set overlay.lo/hi so its mean and spread match the base layer over their common area (seamless insertion of a finer layer)."""
    vb, mb = base.source.sample(lat, lon, res_m); vo, mo = overlay.source.sample(lat, lon, res_m); m = mb & mo
    if m.sum() < 50: return overlay
    nb = (vb[m] - base.lo) / (base.hi - base.lo)
    a = np.std(nb) / (np.std(vo[m]) + 1e-12); b = np.mean(nb) - a * np.mean(vo[m])          # normalised = a * value + b
    overlay.lo, overlay.hi = -b / a, (1 - b) / a
    return overlay


def composite(layers, lat, lon, east_m, north_m, texel_m):
    """Composite the layer stack on a texture grid. lat/lon/east_m/north_m: (H, W) arrays of texel centres. Returns (rgb uint8 (H, W, 3),
    provenance uint8 (H, W, 3), info). Grayscale data are tinted with a fixed Mars-like ramp; the provenance texture is independent of it."""
    H, W = lat.shape; val = np.full((H, W), 0.5); meas = np.zeros((H, W)); sim = np.zeros((H, W)); finest = np.full((H, W), np.inf); used = []
    for L in layers:
        if L.kind == "georef":
            if L.footprint is not None:
                a0, a1, o0, o1 = L.footprint
                if lat.max() < a0 or lat.min() > a1 or lon.max() < o0 or lon.min() > o1: continue
            v, m = L.source.sample(lat, lon, texel_m)
            if not m.any(): continue
            n = np.clip((v - L.lo) / (L.hi - L.lo + 1e-12), 0, 1)
            w = ndi.distance_transform_edt(m) if L.feather_px > 0 else m.astype(float)
            w = np.clip(w / max(L.feather_px, 1e-9), 0, 1) * L.opacity if L.feather_px > 0 else w * L.opacity
            val = (1 - w) * val + w * n
            if L.provenance == "measured": meas = (1 - w) * meas + w; sim = (1 - w) * sim; finest = np.where(w > 0.5, np.minimum(finest, max(L.native_res_m, texel_m)), finest)
            else: sim = (1 - w) * sim + w; meas = (1 - w) * meas
            used.append(L.name)
        elif L.kind == "tiled":
            if L.image is None: continue
            h, w_ = L.image.shape[:2]
            u = (east_m / L.repeat_m * w_) % w_; vv = (-north_m / L.repeat_m * h) % h
            d = ndi.map_coordinates(np.pad(L.image.astype(float), 1, mode="wrap"), [vv + 1, u + 1], order=1, mode="wrap") - 0.5
            gate = np.ones((H, W)) if L.only_below_res_m is None else (finest > L.only_below_res_m).astype(float)
            k = L.amplitude * L.opacity * gate
            val = np.clip(val * (1 + 2 * k * d), 0, 1); sim = np.maximum(sim, gate * L.opacity); used.append(L.name)
    ramp = np.array([[0.20, 0.11, 0.07], [0.62, 0.40, 0.27], [0.96, 0.82, 0.66]])          # dark - mid - bright Mars tint
    x = np.clip(val, 0, 1)[..., None]; rgb = np.where(x < 0.5, ramp[0] + (ramp[1] - ramp[0]) * (x / 0.5), ramp[1] + (ramp[2] - ramp[1]) * ((x - 0.5) / 0.5))
    prov = np.stack([meas, np.clip(sim, 0, 1), np.zeros_like(meas)], -1)
    return (rgb * 255 + 0.5).astype(np.uint8), (prov * 255 + 0.5).astype(np.uint8), dict(layers=used, measured_frac=float((meas > 0.5).mean()), simulated_frac=float((sim > 0.5).mean()))

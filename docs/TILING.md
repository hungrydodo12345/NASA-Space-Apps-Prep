# Game terrain tiles (branch `game-tiling`)

Terrain for the rover game: curved, textured terrain tiles cut from the HiRISE stereo elevation model, written as glTF 2.0 tiles in a level-of-detail quadtree. glTF loads directly in three.js, Unity, Unreal and Godot. Code: `tiling/`; build script: `scripts/build_game_tiles.py`; viewer: `web/tiles/`.

## What a tile is

- **Geometry.** A regular grid of vertices (default 65 × 65) on a latitude/longitude grid. Neighbouring tiles share their edge vertices exactly. A skirt around each tile hides cracks between levels of detail.
- **Topography.** Heights come from the HiRISE stereo DTM (`DTEEC_045994_1985_046060_1985`, 1 m posts, metres above the areoid), sampled bilinearly. The mesh never invents relief: below one DTM post, a finer mesh just samples the same smooth surface more densely.
- **Curvature.** Every vertex is placed on the Mars sphere (the HiRISE products' reference radius, 3394.84 km, plus the DTM height) and then expressed in an east-north-up frame tangent at the scene centre. The ground therefore drops below the tangent plane as on Mars: 0.15 m at 1 km, 3.7 m at 5 km, 14.7 m at 10 km. The horizon for a 2 m rover camera is about 3.7 km away.
- **Precision.** Positions are stored as float32 relative to each tile's centre, with the centre in the glTF node translation, so precision stays at the millimetre level anywhere in the scene.
- **Axes (glTF).** x = east, y = up, z = −north, in metres.

## Textures: a layer stack with provenance

Layers are composited bottom to top. Every layer is tagged **measured** (from data) or **simulated** (added detail):

| Layer | Source | Resolution | Provenance |
|---|---|---|---|
| `hirise_ortho_25cm` | HiRISE RED orthoimage on the DTM grid (co-registered with the DTM) | 25 cm | measured |
| `restored_6cm` | multi-pass restoration of a chain tile (`scripts/run_chain.py`) | 6.25 cm grid (real detail to ~30–70 cm) | measured |
| `ingenuity_flights4-5` | JPL Ingenuity orthomosaic, sol 76 (here resampled to 5 cm) | 5 cm | measured |
| `detail_regolith` | repeating detail texture (`data_products/detail_regolith_256.png`) | 1.5 m repeat | **simulated** |

- **Adding a texture.** Any georeferenced raster becomes a layer through `tiling.sources.ArraySource` (an array plus its position in an equirectangular projection) or `Jp2Source` (a PDS JPEG2000 product). A repeating texture is a `Layer(kind="tiled", image=..., repeat_m=...)`. Finer layers are brightness-matched to the base (`texture.match_overlay`) and feathered at their edges, so they blend in without seams.
- **Simulated detail is gated.** The tiled detail texture is applied only where no measured layer finer than 0.2 m exists (`only_below_res_m`). It never covers real centimetre data.
- **Provenance texture.** Each tile also has `<y>_prov.png`: red = measured weight, green = simulated weight. Yellow means measured data with simulated detail on top. The viewer's "Show provenance" switch displays it; a game can use it to label or hide simulated detail.
- **Refinement.** Tiles refine to deeper levels only where a finer layer exists (around the scene centre, the restoration and the Ingenuity mosaic). Elsewhere the quadtree stops at the base level.

## Registration (images on the DTM)

HiRISE image products of different observations disagree in geolocation by tens of metres. The orthoimage and DTM are built from ESP_045994/046060, so restorations and Ingenuity mosaics (placed in the ESP_068294 image grid) are moved onto the DTM in two steps:
1. The pipeline's pass lock, ESP_068294 → ESP_045994 (from `summary.json`).
2. A 0.5 m normalised cross-correlation of the ESP_045994 image product against its orthoimage over 300 m. At the landing area this peaks at 0.66 against a runner-up of 0.03.

The result is written to `build_info.json`. At the landing area the total shift is about 80 m north–south and 52 m east–west.

## Outputs

- `tiles/<level>/<x>/<y>.glb`: mesh, normals, UVs and the composited colour texture (embedded PNG). `extras` hold the bounds, texel size, mesh spacing and the layers used.
- `tiles/<level>/<x>/<y>_prov.png`: the provenance texture.
- `tileset.json`: a 3D Tiles 1.1 style tileset. Boxes are in the ENU frame, the root `transform` maps ENU to Mars body-fixed metres, and refinement is REPLACE.
- `scene.json`: a flat manifest for game engines and the viewer. It lists the origin, axes, layers and every tile with its bounds, centre, size, geometric error and children.

## Build and view

```
python scripts/build_game_tiles.py <out> --dtm DTEEC_045994_1985_046060_1985_U01.IMG \
  --ortho ESP_045994_1985_RED_A_01_ORTHO.JP2 --ortho-lbl ESP_045994_1985_RED_A_01_ORTHO.LBL \
  --rdr-lbl ESP_068294_1985_RED.LBL --site-summary site_wbf/summary.json --center-global -4370690 -23476196 \
  --size-m 600 --base-level 2 --focus-m 100 --focus-level 4 --overlay-level 5 \
  --chain chain_wbf --ingenuity f45_5cm.npy --placement benchmark/results/ingenuity_truth/site2_placement.json \
  --detail-png data_products/detail_regolith_256.png
ln -s <out> web/tiles/data && cd web/tiles && python -m http.server   # open http://localhost:8000
```

The example covers 600 m × 600 m around the landing spot (the site-3 origin) in 69 tiles (20 MB). Levels run from 2.3 m mesh spacing and 0.6 m texels down to 0.29 m spacing and 7 cm texels over the restoration. Screenshots: `results/tiling/`.

## Limits

- Elevation detail is that of the DTM (1 m posts, with about 0.3 m vertical precision for HiRISE stereo). Rocks smaller than a few metres exist in the textures only, not in the geometry.
- The sphere uses one reference radius. Height above the areoid is treated as height above that sphere, which is accurate for a scene a few kilometres across.
- The base texture is one observation (2016, afternoon lighting), so shading is baked in.

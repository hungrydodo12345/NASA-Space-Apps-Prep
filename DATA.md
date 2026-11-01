# Data guide

Everything the pipeline reads, makes and reports. Raw NASA data is not stored in git (about 12 GB): it is listed
with source URLs and SHA-256 checksums in `data_products/MANIFEST.json` and fetched by `scripts/fetch_data.py`.
Small derived datasets are stored here (28 MB in `data_products/derived/`) so the evaluation steps can re-run
without the raw files.

## 1. Raw inputs (public NASA data)

| Input | What it is | Size | Used for |
|---|---|---|---|
| HiRISE orthoimages ESP_045994_1985 and ESP_046060_1985 (`..._RED_A_01_ORTHO.JP2`) | 8-bit, 0.2524 m/px, DTM-controlled; the reference grid | 0.8 GB | measured layer, classifier features |
| HiRISE stereo DTM `DTEEC_045994_1985_046060_1985_U01.IMG` | float32 heights, 1.01 m posts | 0.4 GB | terrain, rover slope limits |
| HiRISE RDR observations ESP_068822, 068294, 068360, 068426 (`..._RED.JP2`) | 16-bit, 0.25 m/px, map-projected, **not** co-registered (offsets 16-100 m) | 3.0 GB final (7.8 GB downloaded while exploring) | fusion |
| Perseverance Navcam images (948, sols 13-120, looking down) | 8-bit PNG browse images | 2.2 GB | rover ground truth |
| Perseverance camera models + attitude (NASA raw-image API), traverse and waypoints (MMGIS) | CAHVORE camera model per image | 76 MB | projecting photos to the ground |
| AI4Mars labels, Perseverance subset ([Zenodo 10.5281/zenodo.15995036](https://doi.org/10.5281/zenodo.15995036), CC-BY-4.0) | crowdsourced terrain labels, 12,336 files | 0.15 GB | surface classes, masking the rover body |
| HiRISE catalogue `RDRCUMINDEX.TAB` | which observations cover the site (29 RED) | 0.16 GB | choosing observations |

Observations were chosen for similar sun azimuth (165-178 deg) and incidence (41-50 deg). ESP_042315 was dropped by
the fusion's registration lock (score 0.17) and is not in any result.

## 2. Derived datasets stored in this repo (`data_products/derived/`)

| File | Content | Made by |
|---|---|---|
| `measured_layer_u16.npz` | fused and single-frame HiRISE layer of the hero area (2560 x 2560 px, 646 m), uint16 with `lo`/`hi` scale. **Units: 8-bit DN of the ESP_045994 orthoimage grid** (not I/F). Quantisation error 0.01% of image std. | `scripts/build_hero_measured.py` |
| `measured_meta.json`, `gate.json` | frames used, window position, per-tile quality gate (all 25 tiles of every frame passed) | same |
| `rover_truth/*.npz` (35 stops) | top-down 5 cm rover mosaic `z` (z-scored, NaN = unseen), class map `cls` (0 soil, 1 bedrock, 2 sand, 3 big rock), position, sol | `scripts/build_rover_truth.py` |
| `texture_candidates.npz` | 339 single-photo ground patches (40 x 40 px at 5 cm) with quality features | `scripts/collect_texture_candidates.py` |
| `cells_NAV_4m.npz` | 1,966 four-metre ground cells: orbital texture features and rover-measured class fractions | `scripts/build_surface_groundtruth.py`, `train_surface_classifier.py` |
| `hero_navcam.json`, `waypoints.json` | the Navcam image list and rover waypoints | `scripts/fetch_data.py` |
| `scores_detail_v1.json` | scores of the first detail-layer version | `scripts/evaluate_hero.py` |

Hand labels: `results/texture_labels.json` holds 82 labels (44 real ground, 38 hardware / shadow / seam) made by eye
by the author from numbered contact sheets; 14 ambiguous tiles were skipped. They train the texture-quality filter
(`data_products/texture_filter.json`).

## 3. Results (`results/`)

`tune.jsonl`, `tuned_hp.json` (hyper-parameter tuning on separate scenes), `detail_v2.json` / `.jsonl` (clean-patch
scoring of the detail layer), plus every experiment's `*.jsonl` as they are produced. One JSON object per run.

## 4. Not stored (regenerate)

Raw data: `python scripts/fetch_data.py data`, then check against the manifest. Large intermediates (the 6-frame warped
stack, 157 MB; rover-truth projections for all 948 photos) are rebuilt by the scripts in about 10 minutes.

## 5. Known data limits

- **Rover ground truth is projected, not surveyed.** Photos are flattened onto a locally flat ground plane using the
  camera model; ground higher than the plane (rover hardware, rocks) lands in the wrong place. Rover position is
  accurate to roughly 1-3 m, so rover pixels cannot be registered to HiRISE pixels. All scores are therefore
  registration-free (statistics, not pixel differences).
- **Hardware and seams.** Even after masking the rover with the AI4Mars labels, hardware and photo seams remain in the
  mosaics. The clean texture library keeps only patches a hand-labelled classifier accepts (97% precision at the
  chosen threshold, validated on held-out stops).
- **Class coverage.** The clean library holds 93 soil and 12 bedrock patches and **no sand or big-rock patches**; those
  classes borrow soil and bedrock texture.
- **One site.** All rover truth is from the first 120 sols near the landing site.
- **Observation geometry.** The four RDR frames have emission angles of 19-24 deg and are aligned by dense
  per-pixel motion, not DTM orthorectification, so some parallax residual remains.

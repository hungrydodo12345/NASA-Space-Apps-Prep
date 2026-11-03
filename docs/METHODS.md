# Methods: rover-anchored, confidence-scored HiRISE restoration

This document is written for the paper and grows with the code. Each section gives:

- what the component does;
- its equations;
- every parameter and why it has that value;
- how it differs from Tao & Muller (2016, *Planet. Space Sci.* 121, 103–114, "GPT");
- the test that validates it, and the result.

All tests run on CPU in this repository.

## Pipeline order

1. **Gotcha alignment.** Sub-pixel motion of every HiRISE pass against a reference pass. The rover, which moves between passes, is masked out.
2. **PDE-TV restoration.** Four-times upsampled MAP estimate from all passes. Passes are weighted by quality, and pixels that changed between passes are masked.
3. **Per-pass blur.** Each pass's blur is measured on the parked rover where it is visible (Section 3).
4. **Uncertainty.** A per-pixel standard deviation of the restoration, from rebuilding it on subsets of the passes (Section 5).
5. **Rover anchoring.** The rover's own top-down imagery is placed on the restored map using the rover's position in HiRISE (Section 7). This is the only step that uses rover data, and it comes after the restoration: the measurement and the prediction stay separate.
6. **Confidence-scored fill.**
   - A surface/rock classifier plus real rover-photo exemplars add detail finer than a HiRISE pixel.
   - Each pixel and each rock gets a confidence. The user picks a threshold, and only detail above it is shown (Section 6).
   - The confidences are calibrated against the anchored rover truth (Section 7).
7. **Products.** Hazard maps with confidence (Section 8) and the benchmark (Section 9).

## 1. Gotcha dense sub-pixel matching (`pipeline/gotcha.py`)

Re-implemented from the paper's text (Sec. 3.2, Eq. 10, MSA-SIFT steps i–vi); no reference code exists.

- **Seeds (MSA-SIFT).**
  - OpenCV SIFT, up to 4000 features per image.
  - Lowe ratio test at 0.75.
  - RANSAC affine model, 1.5 px reprojection threshold, 4000 iterations.
  - Up to 300 seeds.
  - The paper's mutual shape adaptation (elliptical region update) is replaced by direct ALSC refinement of each seed, which serves the same purpose (affine-adapted, sub-pixel seeds).
- **ALSC (Eq. 10).** Gauss–Newton fit of `ref(p) ≈ g · img(A(p − c) + c + t) + o`:
  - Six geometric parameters plus gain `g` and offset `o`.
  - 15 × 15 px window; at most 15 iterations; stops when |Δt| < 1e-3 px.
  - Fits are rejected if the affine diagonal falls outside 0.7–1.3 or the shear exceeds 0.3.
- **Region growing.**
  - Best-first: highest correlation first, on a 4 px grid.
  - Each neighbour starts from the current point's affine transform.
  - A point is accepted if its correlation is at least 0.6 and its translation jumps by no more than 0.5 px from the prediction.
  - The resulting grid is interpolated to a dense field (linear, nearest-neighbour fill) and smoothed with σ = 2 px.
- **Convention.** The output `d` satisfies `pass(x) ≈ ref(x + d(x))`, the convention of `pipeline/fuse_flow.FlowModel`.
- **Validation.**
  - Setup: real HiRISE texture (PSP_002387, 192 × 192 px) warped by a known smooth field (0.3–0.7 px amplitude), with 2% noise.
  - Gotcha: median error **0.038 px** (90th percentile 0.089 px), 1788 of 1849 grid points matched, 13 s per pair.
  - Previous tile-wise ECC: median 0.149 px (90th percentile 0.240 px).

## 2. PDE-TV MAP restoration (`pipeline/pde_tv.py`)

- **Energy** (Eqs. 5–8 of the paper):
  - `E(X) = Σ_k w_k Σ_p m_kp (A_k X − Y_k)_p² + λ Σ_p sqrt(|∇²X|_p² + μ²)`
  - `A_k = D H_k F_k` combines motion from Gotcha, blur `H_k`, and decimation by 4.
  - The regulariser's gradient is the fourth-order PDE term `∇²(∇²X / sqrt(|∇²X|² + μ²))` (Eq. 8).
- **Solver.** Steepest descent (Eqs. 11–12) with Nesterov momentum.
  - The step is set from the measured largest eigenvalue of `AᵀA` (power iteration, 12 steps) plus the regulariser bound 64 λ s₀ / μ.
  - Assuming the textbook bound 1/s² made the solver diverge: the true value is 0.69 for σ = 4.28 high-resolution px at scale 4. Measuring it is required for stability.
- **Parameters.**
  - λ = 0.005–0.02 (relative to the image standard deviation s₀).
  - μ = 0.2 s₀; 300 iterations; step factor 0.9.
- **Weighting (our addition, not in GPT).** The weights are re-estimated every 20 iterations:
  - Per-pass weight `w_k = (σ_min / σ_k)²`, where σ_k is pass k's robust (MAD) residual against the current estimate.
  - Per-pixel Tukey biweight `m_kp` with c = 4.685 σ_k. Pixels where one pass disagrees with the consensus get weight 0: moving sand, new tracks, the rover, dust-devil streaks.
  - `weighted=False` gives the original, unweighted GPT data term.
- **Validation (simulation).**
  - Setup: fractal terrain, 8 passes, measured blur σ = 1.07 px, 3% noise, 0.03 px registration error.
  - One pass carries a planted 2 m bright change that the others lack.

  | Method | Error vs truth (% of scene contrast) | Artefact at the planted change (× scene std) |
  |---|---|---|
  | Bicubic, 1 pass | 47.6 | 1.63 |
  | CG Tikhonov (previous) | 41.5 | 1.77 |
  | PDE-TV unweighted (GPT), λ = 0.005 | 37.5 | 1.78 |
  | PDE-TV weighted (ours), λ = 0.005 | 37.6 | 1.65 |

  - The weighted version gives the changed pixels of that pass weight 0.0, so the artefact returns to the single-pass level; the unweighted version adds it to the restoration.
- **Not yet reproduced:** the paper's tiled segmentation by motion-vector spread (Eqs. 13–14) and its per-segment blur. Their thresholds are not given in the paper. Per-pass blur is handled by measurement instead (Section 3).

## 3. Per-pass blur from the parked rover (`scripts/calibrate_pass_blur.py` → `data_products/pass_blur.json`)

HiRISE sharpness varies between passes: spacecraft jitter, viewing angle and focus all change. A single assumed point-spread function (GPT assumes a Gaussian with σ = 1) is replaced by one measured value per pass.

- **Absolute: on the rover.** In ESP_068294, ESP_068360 and ESP_068426 (sols 1–11), Perseverance is parked at the landing site and visible as a bright, roughly 12-pixel object.
  - Seven parallel brightness profiles are taken across its short axis, on native pixels with cubic interpolation at 0.1 px.
  - Each is fitted with a blurred bright bar on a linear background: `f(t) = b₀ + b₁t + a[Φ((t−t₁)/s) − Φ((t−t₂)/s)]`. The linear background absorbs the shadow.
  - The pixel's box integration is removed: `σ_psf = sqrt(s² − 1/12)`.
- **Relative: on static ground.** Every other pass is aligned to the sharpest rover-calibrated pass with Gotcha, on a 48 m window at the no-rover site (18.4472 N, 77.4127 E).
  - The extra Gaussian blur `d` (0–1.5 px, least squares with gain and offset) that makes the sharper of the two match the other gives `σ² = σ_ref² ± d²`.
  - Gotcha's SIFT seeding finds no matches between passes years apart under different sun angles. A phase-correlation seed grid is used as a fallback (`gotcha.seeds_phase`).
- **Results (σ in HiRISE pixels):**

  | Pass | Date | σ | Method | Note |
  |---|---|---|---|---|
  | ESP_068360 | 2021, sol 6 | 0.59 ± 0.10 | rover edge | sharpest |
  | ESP_068426 | 2021, sol 11 | 0.83 ± 0.11 | rover edge | |
  | ESP_068294 | 2021, sol 1 | 0.89 ± 0.06 | rover edge | |
  | ESP_073068 | 2022 | 1.16 | relative | 32 matched points |
  | PSP_002387 | 2007 | 1.20 | relative | 18 matched points |
  | ESP_045994 | 2016 | 1.34 | relative | 16 matched points |
  | ESP_068822 | 2021, sol 41 | 1.38 | relative | 79 matched points |
  | ESP_046060 | 2016 | 1.43 | relative | 8 matched points |

- **Caveats.**
  - The rover is not an ideal sharp edge (wheels, deck edges), so the absolute values are upper bounds on the true point-spread function.
  - The relative values rest on few matched points, and residual misalignment inflates apparent blur, so they may be biased high.
  - Blur varies by about a factor of 2.4 between passes, which supports using per-pass values rather than one assumed value.

## 4. Multi-illumination shadow evidence (`pipeline/shadow_evidence.py`, test: `experiments/exp5_shadow_evidence.py`)

- **Model.** A rock of diameter D in pass k gives:
  - a bright sunlit disc;
  - a dark shadow displaced away from the sun, with length `L = h · tan(incidence_k)` and height `h = D/2`.

  The sun azimuth is HiRISE `SUB_SOLAR_AZIMUTH`, measured clockwise from image-right with north up (`NORTH_AZIMUTH = 270`). The shadow vector is `(dy, dx) = (−sin a, −cos a)`. The template is rendered at 1/8 px, integrated over 25 cm pixels, and blurred by the pass's measured σ (Section 3).
- **Per-pass evidence.** `z_k(p, D)` is the local normalised correlation of the high-passed, aligned pass with that template.
- **Combination.** Matched-filter weights `w_k ∝ (template energy after blur) / (pass noise)`, and `S = Σ w_k z_k / sqrt(Σ w_k²)`. The per-pass agreement (the share of passes with `z_k > 0`) is kept for the confidence model.
- **Sun geometry available.** The 8 restoration passes all have sun azimuth 165–178° (chosen for consistent shading), with incidence 46–59°. Two passes with different azimuths were added for diversity: ESP_084883 (about 137°) and ESP_071077 (about 196°).
- **Validation (simulation).**
  - Fractal background.
  - 18 real rocks and 18 decoys per size. Decoys are bright discs with no height, so no shadow.
  - Every pass is rendered with its own sun geometry, its measured σ and 6% noise.
  - The scores below are AUCs: real rock against decoy, and real rock against random ground.

  | Setup | 40 cm | 60 cm | 100 cm | 150 cm |
  |---|---|---|---|---|
  | 1 pass (σ = 0.89) | 0.52 / 0.71 | 0.48 / 0.82 | 0.86 / 0.98 | 0.88 / 0.99 |
  | 8 same-sun passes | 0.56 / 0.62 | 0.53 / 0.78 | 0.90 / 0.99 | 0.86 / 0.99 |
  | 10 passes incl. 2 different azimuths | 0.55 / 0.65 | 0.52 / 0.80 | 0.90 / 0.99 | 0.86 / 0.99 |

- **Findings.**
  - **About 1 m and up:** shadows separate rocks from flat bright patches (AUC 0.86–0.90). Combining passes adds a modest gain at 1 m.
  - **Below about 60 cm:** shadows are about 1 px long at 25 cm per pixel, and rock and decoy are indistinguishable (AUC about 0.5).
  - **Why combining passes does not help small rocks.** The limiting factor is background clutter, which is the same ground texture in every pass, not independent noise. Stacking passes cannot average it away, and blurrier passes add little.
- **Use.** In the confidence model (Section 6), shadow evidence raises confidence only for rocks of 100 cm and above, where it is informative. It is not credited below that.

## 5. Uncertainty from a pass-subset bootstrap (`pipeline/uncertainty.py`, test: `experiments/exp6_bootstrap_uncertainty.py`)

- **Method.** The restoration of Section 2 is rebuilt B times. Each rebuild uses K passes drawn with replacement from the K available, so a pass drawn twice counts twice. The spread of the rebuilds at each pixel is `σ_boot(p) = std_b X_b(p)`.
  - It captures what depends on which passes were taken: noise, registration error, per-pass blur, and changes between passes.
  - It cannot capture error that every pass shares, chiefly detail finer than any pass recorded. That part is absorbed by one factor κ, fitted in simulation where the truth is known.
  - The calibrated uncertainty is `σ = κ · σ_boot`.
- **Scale matters.** All rebuilds, the mean and the truth are evaluated after smoothing by a Gaussian of 0, 2 or 4 output pixels, which corresponds to 6.25 cm, 12.5 cm and 25 cm.
- **Simulation.**
  - Fractal terrain, 64 × 64 native pixels, scale 4.
  - 8 passes, each with its own measured blur (Section 3 values), 3% noise, 0.03 px registration error, and a planted 2 m change in one pass.
  - B = 16 rebuilds, 200 iterations each, about 250 s per bootstrap on 4 CPU cores.
  - κ is fitted on scene A (seed 2001) and tested on scene B (seed 2002), which is held out.

  | Scale | κ (fitted on A) | Held-out coverage ±1σ / ±2σ (target 68.3 / 95.4 %) | Spearman(σ, \|error\|) on 4 × 4 blocks, A / B | Error in least- vs most-confident 10 % of blocks (B) | σ at the planted change ÷ median σ (A / B) |
  |---|---|---|---|---|---|
  | 6.25 cm (output pixel) | 12.8 | 81.5 / 96.0 % | 0.14 / 0.14 | 25.7 vs 21.6 % | 12.5 / 11.5 |
  | 12.5 cm | 4.8 | 85.1 / 97.9 % | 0.32 / 0.34 | 9.5 vs 4.5 % | 18.1 / 17.6 |
  | **25 cm (native HiRISE pixel)** | **2.1** | **86.4 / 99.1 %** | **0.45 / 0.39** | **3.8 vs 0.6 %** | **41.3 / 44.5** |

  Errors are given as a percentage of the true scene contrast (standard deviation). Uncalibrated, the bootstrap covers only 34–51% (±1σ) at 25 cm, and 7–11% at 6.25 cm.
- **Findings.**
  - **At the native 25 cm scale the bootstrap is informative and close to honest after one factor.**
    - κ = 2.1 fitted on one scene gives slightly conservative coverage on another (86% where 68% is the target).
    - σ ranks errors (Spearman about 0.4): the least-confident tenth of the map has about 6× the error of the most-confident tenth.
    - A change present in only one pass stands out at about 43× the median σ.
  - **At the 6.25 cm output pixel, per-pixel uncertainty is not meaningful.** The error there is dominated by detail no pass recorded. That detail is missing from every rebuild in the same way, so the spread cannot see it: κ must be about 13, and σ barely ranks errors (Spearman 0.14).
  - **A second, texture-driven term does not fix this.** We tested a model with the bootstrap plus local restored texture, `σ² = (κ σ_boot)² + (β · tex)²`, fitted by maximum likelihood. It gives κ = 7.5, β = 1.6 and held-out coverage of 81 / 98%, but Spearman of only 0.11. Missing fine detail is not predictable from the restoration.
- **Use.** The pipeline reports `σ_native = 2.1 · std_b(G₄ ∗ X_b)` (25 cm scale) as the trust map. The confidence fill (Section 6) adds no detail where σ_native exceeds a limit. The 6.25 cm pixels inherit that 25 cm σ; we do not claim a finer uncertainty.
- **Real data** (`scripts/run_chain.py`).
  - Passes taken years apart differ by up to about 60% in mean brightness, while ground texture is about 1% of the signal. Without radiometric normalisation, the bootstrap spread measured these offsets: σ came out at about 2000% of scene contrast in a smoke test.
  - Each pass is therefore matched in gain and offset to the reference over the Gotcha-aligned overlap before restoration.
  - Results on real tiles are in Section 5b.

## 6. Confidence-scored fill (`pipeline/confidence.py`, `scripts/confidence_fill.py`)

- **Output.**
  - At threshold `t`: `out(t) = X + lock( T · 1[t ≤ c_tex] · trust  +  Σ_{rocks i: P_i ≥ t, trusted} R_i )`.
  - `X` is the measured restoration (Section 2).
  - `T` is rover-photo texture of the class the recogniser assigns.
  - `R_i` is a real rover-photographed rock sprite with a cast shadow along the HiRISE sun direction (`grounded_fill.fill`).
  - `lock` is the exact projection onto the null space of the HiRISE observation. At every threshold the output, seen through HiRISE's blur, equals the measurement; the lock residual is reported for each render.
- **Where confidence comes from.**
  - **Rock.** `P = logistic(w · [z, log(D/25 cm), S·1[D ≥ 100 cm], 1[D ≥ 100 cm]])`.
    - `z` is the blob response (negative LoG at the rock's scale) in the measured restoration, in units of that scale's robust spread over the image.
    - `S` is the multi-pass shadow evidence (Section 4).
    - `w` is fitted on simulations with exact truth (Section 7).
    - The recogniser's own rock score is not used, because on real data it does not rank candidates (Section 7).
  - **Texture.** The recogniser's class map chooses the exemplars. Its accuracy against rover truth is 26.6%, below the 49.9% you get by always guessing soil (Section 7). The calibrator therefore stores `c_tex = 0.266`, so texture is shown only at thresholds of 0.27 or below.
  - **Trust.** No detail is added where the calibrated 25 cm uncertainty (Section 5) exceeds `--sigma-limit` × scene standard deviation (default 0.5).
- **User control.**
  - CLI: `--confidence t` renders one threshold.
  - `--thresholds` renders several and writes `confidence_viewer.html`, a self-contained page with a threshold slider, rock circles coloured by P, and a measured-only toggle.
  - `rocks.csv` lists every candidate with its P, so any threshold can be applied afterwards.

## 7. Calibration and reliability

### 7a. Real data: the rover answer key cannot verify confidences below about 1 m

- **Setup.** 50 traverse stops with rover top-down truth: 6373 recogniser candidates, 2110 truth rocks of 25 cm and up, and 59,534 AI4Mars-labelled pixels. Predictions are out-of-fold (spatial 3-fold cross-validation).
- **Recogniser rock score** (`scripts/recogniser_reliability_traverse.py`).
  - Raw ECE is 10.5%. Cross-fitted isotonic calibration makes it worse (15.8%), because the folds do not agree on the score-to-truth relation.
  - Within each fold, every feature ranks real against false candidates at about chance: score AUC 0.46–0.50; HiRISE blob support 0.46–0.48; size 0.50–0.60.
- **Recogniser surface class.** Top-class accuracy is 26.6%, against 49.9% for always guessing soil. Confidence-versus-correct AUC is 0.41–0.48 in the two large folds. The isotonic map collapses to the base rate (held-out ECE 1.4%, all mass in one bin).
- **Physics P(rock) on the same stops** (`scripts/anchored_reliability.py`, no per-pass stack, so S = 0).

  | Evidence | Candidates | Share matching a truth rock | Share matching truth shifted 40 px | AUC | AUC vs shifted truth |
  |---|---|---|---|---|---|
  | Physics P(rock) | 1084 | 29.2% | 17.9% | 0.43 | 0.44 |
  | Recogniser score | 4148 | 28.8% | 15.5% | 0.46 | 0.48 |

  Candidate positions match the truth 1.6–1.9× more often than chance, so the answer key does carry positional signal. But no score, physical or learned, ranks candidates against it.
- **Why.** The truth's registration to HiRISE is too poor at most stops:
  - Only 4 of 27 stops could be registered (`results/rocks_wide_registered_*.json`), on 4–56 m² of ground, with shifts up to 35–45 px (2–3 m, near the search limit).
  - Rover waypoints carry about 10–20 m of local error (`results/geolocation_finding.md`).
  - At the landing site, the JPL mosaic's fine-scale alignment correlation is only 0.14, and its orientation is unresolved.

  A rock-sized misregistration turns every label into noise.
- **Consequence.** Confidences are calibrated in simulation (7b). The real-data numbers above are reported as the limit of what can currently be verified.

### 7b. Simulation calibration (`experiments/exp7_confidence_calibration.py` → `data_products/confidence_calibrator.json`)

- **Scenes.**
  - 32 × 32 m, 6.25 cm truth grid, fractal ground.
  - Rocks of 25 / 40 / 60 / 100 / 150 cm (counts 40 / 30 / 20 / 10 / 8), brightness 0.4–1.2 × ground standard deviation, each with a shadow for every pass's real sun geometry.
  - 15 decoys per size: flat bright patches with no shadow.
- **Passes and restoration.** The 8 restoration passes with their measured blur, random offsets, 3% noise and 0.03 px registration error, restored with weighted PDE-TV.
- **Fit and test.** 10 scenes fit the model, 6 held-out scenes test it. A candidate counts as real if it lies within max(3 px, D/2) of a real rock's centre. About 88 s per scene on CPU.
- **Fitted model.** `P = σ(0.941 z − 0.248 log(D/25) + 1.396 S·1[D≥100] − 0.722·1[D≥100] − 4.297)`.
- **Held-out results** (1924 candidates, 648 real rocks).
  - AUC 0.72; blob z alone gives 0.71; without the shadow feature, 0.717.
  - ECE 1.5%. The reliability bins (predicted → observed) are 0.08→0.08, 0.14→0.14, 0.24→0.23, 0.34→0.39, 0.44→0.52, 0.55→0.45, 0.64→0.68 and 0.74→0.59. The top bins hold fewer than 20 candidates each.
  - AUC by size: 25 cm 0.69, 40 cm 0.60, 60 cm 0.87, 100 cm 0.79, 150 cm 0.77.

  | Threshold | Rocks shown | Real (precision) | Real rocks found (recall) | Recall 25 / 40 / 60 / 100 / 150 cm |
  |---|---|---|---|---|
  | 0.3 | 265 | 47% | 18.8% | — |
  | 0.4 | 141 | 55% | 11.7% | 0.4 / 1.7 / 34 / 30 / 27% |
  | 0.5 | 91 | 56% | 7.9% | 0.4 / 1.1 / 24 / 15 / 21% |
  | 0.6 | 53 | 64% | 5.2% | 0.4 / 1.1 / 17 / 7 / 15% |

- **Findings.**
  - The probabilities are honest: they match observed frequencies to within 1.5% on scenes the model never saw.
  - Confidence is useful only for rocks of 60 cm and up. Below that, very few rocks are found at any useful threshold, consistent with the about-40% repeatable-detail limit.
  - Shadows add almost nothing over blob strength once the restoration is good. This matches Section 4: clutter, not noise, is the limit.
- **Limits.** The calibration is only as good as the simulation's realism: fractal ground, idealised rock photometry, Gaussian blur. The real-data positional check (7a) is consistent with it but cannot confirm it.

## 8–9

These sections are filled in as each component lands: per-pass blur from the rover, shadow evidence, uncertainty, confidence fill, rover-anchored calibration, hazard maps, and the benchmark.

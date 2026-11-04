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
- **Weighting (our addition, not in GPT).**
  - **Per-pass weight** `w_k = (n_min / n_k)²`, where `n_k` is pass k's white-noise level measured from its own finest-scale pixel differences: the Immerkær (1996) Laplacian-difference kernel with a robust MAD (`pde_tv.pass_noise`, `pass_weights="noise"`).
  - **Rejected first version** (`pass_weights="residual"`): `w_k = (σ_min/σ_k)²` from each pass's robust residual against the current estimate. It works in simulation, but on real HiRISE it reads the illumination and season differences between passes as noise and collapses onto one pass. On the 8-pass site, one pass got weight 1.00 and the rest 0.10–0.23; the noise-based weights are 0.20–1.00, most 0.4–0.7. It is kept as an ablation (benchmark `full-resw`).
  - **Per-pixel Tukey biweight** `m_kp` with c = 4.685 σ_k, where σ_k is the pass's robust residual, re-estimated every 20 iterations. Pixels where one pass disagrees with the consensus get weight 0: moving sand, new tracks, the rover, dust-devil streaks.
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
- **Consequence.** Confidences are calibrated in simulation (7b). The real-data numbers above are the limit of what the traverse truth can verify; Section 7c adds a physical truth at one object near the landing area.

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

### 7c. Ingenuity's downward images as centimetre-scale truth

This took several rounds. The earlier rounds failed for reasons that are listed because they are instructive; the last one succeeded, and the placement was validated by two independent methods.

- **Data.** Ingenuity flight 26 (Perseverance sol 414, 2022-04-20): 190 downward navigation frames (640 × 480) and 10 colour frames. The parachute and backshell are in both HiRISE (post-landing passes) and these images. Code: `scripts/ingenuity/`.
- **HiRISE side.** Found by differencing the 2016 and 2021 passes: a cluster of new bright objects about 1.1 km west and 0.2 km north of the landing spot. The parachute is a crumpled patch about 13 × 15 m; the backshell is two bright halves about 3.4 m apart. The alignment chain on that 32 m tile with the 5 post-landing passes matched 40–90% of grid points at correlations of 0.87–0.91.
- **Why the first rounds failed (and what fixed it).**
  - **Camera model.** The downward camera is an equidistant fisheye (CAHVORE type 3, linearity 0, f = 275 px, 133° field of view), not a pinhole. Treating it as one broke every ground-plane homography at wide angles. Camera-fixed features (the helicopter's shadow, lens marks) also made 61 of 189 pairs look static on featureless ground. The fix (`rectify.py`, `track_flight_h.py`): rectify to a pinhole, subtract the fixed pattern, and solve all poses in one robust pose graph. This chained 97 frames into one group (38 before).
  - **Residual weakness.** A per-step in-plane rotation bias of about −1.5° and real yaw jumps of 25–29° mean the absolute heading of a long strip is unreliable. Only the stretch around the backshell (frames 132–158, 27 frames) is used.
  - **A wrong compass.** My first helicopter-shadow detector locked onto the shadows of rocks, giving an answer exactly 180° off the image evidence and making the placement look contradictory for a whole round. Reading the shadow positions off the frames showed the true shadow (a thin black cross about 150 px, 4.3 m, from the nadir point) and gave a consistent direction.
  - **Photogrammetry of the colour frames failed.** Raw frames: 1 of 45 pairs verified. After orthorectification: 4–14 inliers between consecutive frames. Not used.
- **Placement (validated by two independent methods).**
  - **Sun and shadow.** The computed sun at 11:38 local time on sol 414 (Ls 212°) has elevation 58° and azimuth 170°, so the helicopter's shadow points to azimuth 350°, and its length is consistent (150–160 px measured against 172 px expected). The six hand-read shadow positions (frames 136–154) are consistent in the mosaic frame (circular sd 12°). They give a mosaic-to-north rotation of **217° counter-clockwise ± 12°**.
  - **Image correlation.** A masked correlation of the backshell patch (translation pinned to the backshell) against three independent HiRISE passes (068294, 068360, 068426: different noise and different alignment locks) peaks at **208°**, with neighbouring rotations high, at scale **0.0290 m per pixel**, and with the same translation in all three passes. This scale equals the nominal 8 m / 275 px, which the search was not told.
  - **They agree to within 9° of each other** (217° and 208°), and the combined estimate used is 211°. A small refinement (±0.4 m, ±3% scale) is covered by the robustness test below.
- **Test 1, the rock field (not informative).** The 181 m² truth patch with the backshell excluded: correlation with the truth at 0.5–2 m is 0.00–0.07 for every method, including raw HiRISE (−0.03). The differences between methods are inside the noise (best variant +0.038 over UCL, 95% CI −0.015 to +0.100). The truth was taken at midday and HiRISE in the afternoon, so rock faces and shadows look completely different, and little that is comparable remains (`rock_field_patch_scores.json`).
- **Test 2, the backshell object (the informative one).** Within 4 m of the backshell the truth and HiRISE share strong structure (correlation 0.92–0.97 for all methods at 1–4 m scales). This test was chosen *after* seeing Test 1, which is why it carries the safeguards below. Each method is tuned to its own minimum of error against the truth.
  - **Settings searched.** UCL re-implementation, UCL + measured blur, this work with equal or with noise-based pass weights, and measured blur × 0.8 and × 1.2. Each uses λ from 0.002 to 1 (up to 30 for UCL, whose optimum lies beyond 1).
  - **Error versus λ** (1 − correlation, 50–25 cm band). Both curves are U-shaped.

    | Method | Error at its minimum | at λ | at λ = 0.005 | at λ = 0.04 |
    |---|---|---|---|---|
    | raw HiRISE | 0.249 | none | none | none |
    | UCL re-implementation | 0.229 | 3.0 | 0.368 | 0.333 |
    | UCL + measured blur | 0.227 | 3.0 | 0.296 | 0.286 |
    | this work, equal weights | 0.214 | 0.1 | 0.281 | 0.234 |
    | **this work** | **0.203** | 0.1 | 0.248 | 0.206 |

  - **Tuned, in-sample.** Differences against UCL tuned to its own minimum, with block-bootstrap intervals: 1 m–50 cm **+0.010** (95% CI +0.003 to +0.015); 50–25 cm **+0.023** (95% CI -0.004 to +0.051, positive in 95% of resamples); 2–1 m +0.003, a tie.
  - **At the same λ the gap is large.** At λ = 0.04 and 50–25 cm, the error is 0.206 for this work against 0.333 for UCL (38% lower); at λ = 0.005, 0.248 against 0.368.
  - **Out of sample** (λ chosen on one half of the disc, scored on the other, four splits): this work minus UCL is +0.001 at 1 m–50 cm and +0.009 at 50–25 cm. Most of the in-sample gain comes from tuning on the same data.
  - **Disc radius** (this work minus UCL, each tuned; correlation): 1 m–50 cm: 2.5 m: +0.009 (CI +0.000 to +0.018), 4 m: +0.010 (CI +0.003 to +0.015), 6 m: +0.008 (CI +0.003 to +0.012), 8 m: +0.007 (CI +0.002 to +0.012). 50–25 cm: 2.5 m: +0.012 (CI -0.010 to +0.038), 4 m: +0.023 (CI -0.004 to +0.051), 6 m: +0.013 (CI -0.013 to +0.039), 8 m: -0.006 (CI -0.038 to +0.019). The 1 m–50 cm advantage is stable (+0.007 to +0.010; the interval excludes zero at 4, 6 and 8 m). The 25–50 cm advantage is not: it is +0.012 to +0.023 within 6 m but -0.006 at 8 m, where the patch includes the rock field the truth cannot resolve at that scale, and no interval excludes zero.
  - **Placement robustness.** Re-rendering the truth for 45 placements (rotation ±8°, scale ±3%, translation ±0.4 m): this work is ahead of UCL in 91% of placements at 50–25 cm (mean +0.034) and 84% at 1 m–50 cm (mean +0.010); 2–1 m is a tie.
- **What this does and does not show.**
  - **One object.** A single 1.5 m object, a post-hoc choice after Test 1 was uninformative, with a truth taken at a different time of day. It cannot establish a general result.
  - **Supports.** A small, stable advantage at 50 cm–1 m (about +0.01 in correlation, about 8% lower error) once each method is tuned; at any fixed regularisation, this work reproduces the object better than the UCL re-implementation, by a wide margin. The measured blur and the noise weighting each contribute (UCL → UCL + blur → this work). That agrees with the simulation ablation.
  - **Does not support.** A large advantage once each method is tuned to the truth. At 25–50 cm the advantage depends on the patch radius and no interval excludes zero; out of sample the gap is about 0.01.
- **Location metadata is published separately, and it confirms the placement.** The NASA raw-image feed carries no position (`xyz` and `attitude` are `UNK` on all 200 flight-26 frames), but the PDS Imaging Node bundle `mars2020_helicam` (`data/sol/<sol>/ids/rdr/heli/`) has a label for every frame (RAD/RAF/RAS/RZS products). Each label gives the camera model in the flight frame `HELI_G_FRAME_<flight>`, that frame's offset and rotation relative to the rover site frame (`SITE_FRAME 3` for flight 26), and a per-image pose (`HELI_M_FRAME`). Code: `scripts/ingenuity/telemetry_pose.py`.
  - **Heading.** For frame 140 (the mosaic's reference), with one of the two possible sign conventions for the flight-frame yaw, telemetry puts the image axes at azimuths 231.5° (right) and 321.5° (down); my image-based placement gives 239° and 329°. They agree within 7.5°. The other convention is 22° off.
  - **Direction of flight.** Over frames 134–156 telemetry gives 248°, my pose graph with my placement gives 254° (6°).
  - **Distance along the path disagrees.** Telemetry: 31.6 m over frames 134–156; my pose graph: 20.1 m (36% shorter), so the chained geometry over tens of metres is compressed. The scored patch (within 4–8 m of the backshell) is dominated by single frames and is much less affected. The telemetry also puts the camera 6.7–9.3 m above the takeoff point, consistent with the scale 0.0290 m/px (8 m above local ground).
  - **Absolute position.** The flight frame is tied to the rover's site frame, not directly to HiRISE, so telemetry fixes heading, scale and the path geometry but not the translation; that still needs the site-to-HiRISE offset (about 140 m on the rover waypoints, with 10–20 m of local variation).
  - **Ready-made mosaics exist.** The same bundle holds JPL-made mosaics: vertical mosaics for flight 26 (`V_MMMM_0414_RAS_0030000_VRT_S_F26LS01–04`) and flight 27, and DEM-orthorectified colour mosaics for other flights (`h_mrgb_..._dem_..._orr_...`), 13 sols in the inventory index (61, 76, 152, 163, 174, 193, 254, 268, 282, 292, 362, 403, 418) plus the flight-26 set found by browsing. These were not used here; they are the obvious next source for more truth patches (`results/ingenuity/helicam_bundle_notes.md`).
- **Telemetry check of my own chain, and a cleaner mosaic.** The helicopter's telemetry (position by inertial unit and onboard visual odometry, so only trustworthy relative to the start of the flight) gives each frame's camera height. My pose chain reproduces the relative zoom between frames to within 1–2% for frames 134–146, the stretch around the backshell (for example 1.12 against 1.11, 1.18 against 1.18), but collapses after frame 148 (0.5–0.6 against a true 1.0). That collapse explains the 36% path-length shortfall. The truth mosaic was therefore rebuilt from the accurate frames only (132–147) and re-registered on the same three HiRISE passes (rotation 207–214°, scale 0.0295 m/px, translation (−0.3 to −0.6, +1.0 to +1.1) m).
  - **The object test reproduces on the cleaner mosaic.** This work minus UCL, each tuned: 1 m–50 cm +0.009 (95% CI +0.005 to +0.013); 50–25 cm +0.013 (95% CI -0.014 to +0.041); 2–1 m +0.002. Out of sample: +0.001 (1 m–50 cm) and +0.021 (50–25 cm). Ahead of UCL in 87% (1 m–50 cm) and 93% (50–25 cm) of 45 placements. Error minima: this work 0.208 (λ 0.04–0.1), UCL 0.223 (λ 3). Files: `*_cleanmosaic.json`.
- **Eye check** (`results/ingenuity/eye_check_placement.png`, `scripts/ingenuity/eye_check.py`): the mosaic's backshell sits on the HiRISE backshell with a small residual offset (about 0.3 m), and the small bright piece northwest of it appears in both images at the best placement (white in the overlay) but only in HiRISE in the two rotated controls. The rock field to the lower left shows interleaved, non-overlapping blobs in every placement, as expected from the different lighting. The controls rotate about the backshell, so they test only what surrounds it. The eye supports the placement but cannot prove it on its own.
- **Comparison with JPL's flight-26 mosaics: not possible on the backshell.** JPL published four vertical mosaics (`V_MMMM_0414_RAS_0030000_VRT_S_F26LS01–04`, 4000 × 2500 px at 2 cm, site frame 3, 39% / 30% / 16% / 6% filled). They were built from frames 82–163, the same stretch as my mosaic, but they show the dune stretch (ripples, the helicopter's own shadow, the pinched footprints of single fisheye frames) and not the backshell. A content match against my mosaic found nothing (6 inliers, chance level), and the telemetry position under either sign convention falls outside the tiles' stated extent, so the tiles could not be located relative to mine. The only fair statement is that both use the same input frames. (`results/ingenuity/jpl_flight26_tiles.png`, `scripts/ingenuity/compare_jpl.py`.)
- **Measuring the HiRISE blur shape against the truth: negative.** Fitting an isotropic, an anisotropic and a generalised Gaussian kernel per pass to the truth (with free gain, offset and translation; `scripts/ingenuity/fit_psf.py`) gives 2.0–2.5 native pixels for all five passes, three to four times the rover-edge values (0.6–0.9). The cause is the lighting mismatch (midday truth, afternoon HiRISE): shading differences are absorbed as blur. The kernel fit is therefore not used. The rover-edge calibration (Section 3) stays the blur measurement, because the object and the blur come from the same image.
- **Registration summary:** `results/ingenuity/registration_attempt.json`. Placement files: `benchmark/results/ingenuity_truth/`.

### 7d. Changed ground: the parked rover (`benchmark/changed_ground.py`, negative result)

- **Setup.** Perseverance is in 3 HiRISE passes at the landing spot (sols 1, 7, 11) and absent from 5 (2016, 2021-04 onward). Background = per-pixel median of the aligned rover-free passes; footprint = 31 native pixels (2.0 m2); rover contrast is only 1.5x the pass noise. Ghost fraction = mean over the footprint of (result - background) / (rover-present mean - background); 0 = rejected, 1 = kept. Results: `benchmark/results/changed_ground.json`, figure `results/changed_ground/rover_ghost.png`.
- **Result.** With 3 rover + 5 rover-free passes: GPT re-implementation 0.17, GPT + blur 0.15, this work 0.86 (equal weights 0.82, no robust mask 0.50, median-initialised 0.86). With only 1 rover-present pass: GPT -0.04, this work 0.70. Away from the rover, background RMS is 0.002-0.004 for all methods.
- **Diagnosis.** Pass weights are not the cause (equal weights still 0.82), and neither is the start point. Switching the Tukey masks off drops the ghost from 0.86 to 0.50. Inside the footprint the masks give 0.82-0.92 weight to the three rover passes and 0.25-0.59 to the five rover-free passes (0.88-0.92 elsewhere): the rover-free passes differ from each other in season and lighting, so the mask trusts the group that is mutually consistent, not the majority.
- **Honest status.** This contradicts the claim that change masking helps on real data. The simulated planted-change gain holds only for a change large against the noise in a single pass. Not tried: masks against the all-pass median after lighting matching. Note the GPT figure below the plain-average expectation (3/8 = 0.38) was not fully explained.

## 8. Hazard maps (`pipeline/hazard.py`, written by `scripts/confidence_fill.py`)

- **Per cell** (default 1 m), from the calibrated rocks of Section 6:
  - `p_hazard = 1 − Π_i (1 − P_i)` over rocks with `D_i ≥ d_hazard` (default 40 cm). This is the probability that at least one hazard rock is in the cell, treating rock calls as independent.
  - `expected_rocks = Σ_i P_i`.
  - `status`:
    - `unknown` where less than half of the cell is trusted base (Section 5);
    - `hazard` where `p_hazard ≥ p_alarm` (default 0.5), or where slope ≥ 15° if a slope map is supplied;
    - otherwise `clear`.
- **Outputs.** `hazard.png` (probability map and status map over the restoration) and `hazard.csv` (one row per cell).
- **Options.** `--d-hazard`, `--p-alarm` and `--cell` change the defaults.
- **Caveat.** Probabilities for rocks under about 60 cm rest on the simulation calibration only (Section 7). The status map does not separate a confidently clear cell from one whose rocks are too small to see; the `unknown` class covers only base trust.

## 9. Benchmark and paper (`benchmark/run_benchmark.py`, `scripts/make_paper.py`)

- **Methods.**
  - raw;
  - previous pipeline;
  - our GPT re-implementation;
  - ablations: GPT + noise weighting, GPT + measured blur, full with residual weights;
  - full;
  - oracle motion (simulation only).
- **Tests.** Simulation with exact truth (5 scenes), real split-half consistency (8 passes), and the real rover edge with ringing. Each part is saved on its own as it finishes and merged into `benchmark/results/benchmark_results.json`.
- **Results.** `python scripts/make_paper.py` fills `docs/paper/paper_template.md` from the result files, writing `docs/paper/PAPER.md`. `python benchmark/update_readme.py` refreshes the README table. Any missing result shows as **[pending]**.


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

## 3–9

These sections are filled in as each component lands: per-pass blur from the rover, shadow evidence, uncertainty, confidence fill, rover-anchored calibration, hazard maps, and the benchmark.

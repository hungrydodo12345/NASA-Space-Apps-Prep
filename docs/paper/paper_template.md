# Measuring each pass's blur and noise improves repeat-pass super-resolution of HiRISE: validation against rover and helicopter imagery at Jezero crater


## Abstract

Repeat-pass super-resolution can sharpen orbital images of Mars. But such methods have been judged mostly by visual comparison and resolution claims, rarely against independent ground truth or with honest uncertainty.

We build on the Gotcha-PDE-TV (GPT) pipeline of Tao & Muller (2016). Their code is not public, so we re-implement a simplified version from the paper and add three measured ingredients:

- **Per-pass blur measured on a static target.** A compact, bright boulder at the landing area, visible in every pass, gives each pass's point-spread function. Blur varies {{blur_ratio}}× between the passes.
- **Noise-weighted fusion with robust pixel masks.** Each pass is weighted by its own white-noise level. The pixel masks reduce artefacts at a planted change in simulation; a real changed-ground test remains to be done.
- **A calibrated per-pixel uncertainty** from a pass-subset bootstrap.

On top of the measured restoration we add a confidence-scored detail layer. Rover-photo texture and rock exemplars are shown only above a threshold the user chooses, and are locked so that the result, seen through HiRISE's blur, is unchanged (lock residual ≤ {{lock_resid_max}}%).

In simulation with exact truth ({{n_sim}} scenes), the extended pipeline lowers the 25 cm error from {{gpt_e25}}% to {{full_e25}}% of scene contrast against our GPT re-implementation ({{rel_e25}}% relative). In simulation it also reduces the artefact at a planted change from {{gpt_art}} to {{full_art}}. {{abstract_real}} {{abstract_truth}}

The bootstrap uncertainty is conservative at the native 25 cm scale: held-out ±1σ coverage is {{boot_cov1}}%, against a nominal 68%, so it over-covers rather than under-covers. It is not reliable at the 6.25 cm output pixel.

Limits we measured:

- Detail that repeats between independent pass subsets reaches about {{repeat_limit}} cm on real HiRISE and about {{sim_resolved}} cm in simulation; the 6.3 cm output grid is a sampling choice, not a resolution claim.
- Confidences are calibrated in simulation tuned to real pass-to-pass disagreement (expected calibration error {{cal_ece}}%). The rover traverse truth is too poorly registered to check them below about 1 m (AUC {{anch_auc_phys}}–{{anch_auc_reco}}).

Everything runs on a 4-core CPU.

## 1. Introduction

HiRISE images Mars at 25 cm per pixel (McEwen et al., 2007). Many sites, among them the Jezero crater landing site of Perseverance, have been imaged repeatedly. Multi-image super-resolution restoration (SRR) combines such passes into a finer image. Tao & Muller (2016) introduced GPT, which has three parts:

- **Gotcha** dense sub-pixel matching;
- a **fourth-order PDE-regularised, total-variation-like MAP** reconstruction;
- a 5 cm output grid, which the authors describe as set heuristically from eight 25 cm passes, with a check of rover wheel-track spacing against Navcam imagery (to within 8 cm) and rock counts.

Later work by the same group moved to learned, GAN-based SRR (e.g. MARSGAN; Tao et al., 2021).

Three questions are rarely answered for such methods:

1. How much of the restored detail is in the data rather than in the prior?
2. How uncertain is each pixel?
3. Can a user trust detail added below the sensor's reach?

Jezero gives an unusual chance to test this: many repeat passes, a compact bright boulder visible in all of them that serves as a blur target, rover photographs of the ground, and Ingenuity's centimetre-scale images of landing hardware.

Contributions:

1. **An open, simplified re-implementation of GPT** (one blur width instead of the original's per-tile 0–1 px choice; no tiling) (MSA-SIFT seeding, adaptive least-squares correlation, best-first region growing, PDE-TV with a measured step bound), with no reference code available.
2. **Per-pass blur measured on a static bright boulder** and transferred to the other passes on static ground.
3. **Noise-weighted, change-masked fusion.** We also show that residual-based pass weighting fails on real HiRISE, because lighting differences between passes masquerade as noise.
4. **A pass-subset bootstrap uncertainty**, calibrated in simulation and reported at the scale where it is honest.
5. **A confidence-scored, measurement-locked detail layer with a user threshold.** Its confidence comes from measured evidence: split-half repeatability, multi-illumination shadows and size. It is calibrated in a simulation whose pass-to-pass disagreement is tuned to real data.
6. **A CPU benchmark against the GPT re-implementation**, with ablations, simulation with exact truth, real split-half consistency and the rover-edge test. Negative results are reported alongside.

## 2. Related work

- **Repeat-pass SRR for Mars.**
  - GPT (Tao & Muller, 2016) uses Gotcha matching (Otto & Chau, 1989), seeded by SIFT (Lowe, 2004) and refined by ALSC (Gruen, 1985), with a fourth-order PDE regulariser (You & Kaveh, 2000).
  - Learned SRR (e.g. MARSGAN; Tao et al., 2021) produces sharper images whose added texture comes from training data, so measurement and learned prior cannot be separated by the user.
  - We stay with the measurement and add detail only as a separate, locked, confidence-scored layer.
- **PSF estimation.** HiRISE blur varies with spacecraft jitter and focus. We measure it directly on a known target rather than assume one Gaussian.
- **Uncertainty and calibration.** We use the bootstrap (Efron, 1979) over passes; isotonic (Zadrozny & Elkan, 2002) and logistic calibration; and reliability diagrams with expected calibration error (Guo et al., 2017).
- **Ground truth.** Perseverance Navcam (Maki et al., 2020) images were projected to a top-down truth. AI4Mars labels (Swan et al., 2021) give surface classes.

## 3. Data

- **HiRISE.** {{n_passes}} HiRISE RDR RED observations of the Jezero landing area, 2007–2024 (`benchmark/data_manifest.json`).
  - Sun incidence is {{inc_min}}–{{inc_max}}°.
  - Eight observations have a sun azimuth of 165–178° and are used for restoration.
  - Two (azimuths about 137° and 196°) add illumination diversity for shadow evidence.
- **Test tiles.**
  - A no-rover site at 18.4472 N, 77.4127 E, with 8 passes.
  - A boulder tile at 18.4456 N, 77.4520 E near the landing area, with 3 passes from sols 1–11, containing a compact bright boulder (about 2.5 × 1.8 m) used as the blur and edge target.
- **Rover ground truth.** Perseverance Navcam photos projected to the ground at 50 traverse stops, plus AI4Mars classes.
- **Ingenuity truth.** 190 downward navigation frames (3 cm per pixel from 8 m) of flight 26 (sol 414) over the parachute and backshell, 1.1 km west of the landing spot (Section 6.5).

![Figure 1](../../results/paper/fig1_static_boulder.png)

**Figure 1.** The blur and edge target: a compact bright boulder visible in all 8 passes from 2007 to 2022 (arrow), on the reference grid after Gotcha alignment. 32 m tile, 25 cm pixels, each panel stretched separately.

![Figure 2](../../results/paper/fig2_hardware_alignment.png)

**Figure 2.** Alignment check over 15 years. The parachute (top) and backshell (bottom) are absent in the 2007 and 2016 passes and present from sol 1, at the same position in every pass. 40 m crops; the parachute's change of shape after March 2021 is visible.

## 4. Method

### 4.1 Alignment (Gotcha)

- **Seeds.** SIFT matches, filtered by a ratio test and RANSAC affine; when SIFT fails across years, a phase-correlation seed grid is used instead.
- **Refinement and growth.** Each seed is refined by 8-parameter ALSC (6 affine, gain, offset) and grown best-first on a 4 px grid.
- **Acceptance.** Correlation ≥ 0.6 and a jump ≤ 0.5 px. The rover is masked.
- **Accuracy.** On real texture with a known warp, the median error is {{gotcha_med}} px (90th percentile {{gotcha_p90}} px), against {{ecc_med}} px for tile-wise ECC.

### 4.2 Restoration (PDE-TV)

- **Energy.** We minimise `E(X) = Σ_k w_k Σ_p m_kp (A_k X − Y_k)_p² + λ Σ_p sqrt(|∇²X|² + μ²)`.
  - `A_k` combines warp, the pass's own Gaussian blur σ_k, and decimation by 4.
- **Solver.** Steepest descent with Nesterov momentum. The step is set from the largest eigenvalue of AᵀA, measured by power iteration. The textbook bound underestimates it and diverges.
- **Pass weights.** `w_k` is the inverse white-noise variance of pass k, measured from its finest-scale pixel differences (Immerkær, 1996).
- **Pixel weights.** `m_kp` are Tukey biweights on the residual, re-estimated every 20 iterations. They mask pixels that changed in one pass.
- **Radiometry.** Each pass is gain/offset-normalised to the reference over the aligned overlap.
- **Rejected variant.** Residual-based pass weights, `w_k ∝ 1/σ²(residual)`, are kept as an ablation (Section 6).

### 4.3 Per-pass blur from the rover

- **Absolute.** On three passes, brightness profiles across the boulder's short axis are fitted with a blurred bar on a linear background. A boulder edge is not a perfect step, so the fitted σ is an upper bound on each pass's blur; the ratios between passes are the robust quantity. With the pixel's box integration removed, `σ = sqrt(s² − 1/12)`.
- **Relative.** The other passes are aligned to the sharpest rover pass on static ground. The extra Gaussian blur that best matches them gives their σ.
- **Results.** {{blur_table_inline}}.

### 4.4 Uncertainty

- **Bootstrap.** The restoration is rebuilt B times from passes drawn with replacement, giving `σ_boot`.
- **Calibration factor.** One factor κ, fitted on one simulated scene and tested on another, absorbs error shared by all passes.
- **Scale.** The uncertainty is reported at the native 25 cm scale.

### 4.5 Confidence-scored detail

- **Rock probability.** Rock candidates are blob maxima in the restoration. Each gets `P(rock) = σ(w · [z_rep, log D, S·1[D≥1 m], 1[D≥1 m]])`.
  - `z_rep` is the blob response divided by its measurement noise, estimated from two restorations made from disjoint halves of the passes.
  - `S` is multi-illumination shadow evidence: a template correlation per pass, with the shadow on that pass's anti-sun side and length (D/2)·tan(incidence), combined with matched-filter weights.
- **Calibration.** `w` is fitted on simulations whose pass-to-pass disagreement is tuned to real HiRISE (Section 5).
- **Rendering at threshold t.**
  - Rover-photographed rock sprites are added for rocks with P ≥ t.
  - Recogniser-chosen rover texture is added only if t ≤ the class accuracy verified on rover truth.
  - Nothing is added where the base uncertainty exceeds a limit.
  - Every added layer is projected onto the null space of the HiRISE observation.
- **Hazard maps.** Per 1 m cell, the hazard probability is 1 − Π(1 − P_i) over rocks ≥ 40 cm. A cell is marked unknown where the base is not trusted.

## 5. Experiments

All experiments run on a 4-core CPU (`benchmark/run_benchmark.py`).

### 5.1 Methods compared

| Method | Description |
|---|---|
| raw | One pass, bicubic. |
| previous | Tile ECC + CG Tikhonov. |
| GPT | Our simplified re-implementation: Gotcha + unweighted PDE-TV, one assumed σ = 1 px, no tiling (the original varies σ between 0 and 1 px per tile). |
| GPT + weights | Ablation: noise weighting only. |
| GPT + blur | Ablation: measured per-pass σ only. |
| full (residual weights) | Ablation: the first weighting variant. |
| **full** | This work. |
| oracle | Full with exact motion; simulation only. |

### 5.2 Tests

- **A. Simulation with exact truth.**
  - {{n_sim}} scenes of fractal ground.
  - 8 passes with the measured blurs, a random offset plus 0.3 px sinusoidal jitter, 3% noise, and a 2 m change planted in one pass.
  - Every method estimates its own motion.
- **B. Real split-half consistency.** The 8 real passes are split into disjoint halves, each restored independently. Fourier-ring coherence between the halves measures detail that repeats in independent data.
- **C. Boulder edge.** The edge width of the bright boulder in each restoration, with ringing (RMS residual of the bar fit) and the largest undershoot beside the edge.
- **D. Uncertainty calibration.** As Section 4.4.
- **E. Confidence calibration and the real-data check.**
  - The simulation's per-pass noise and illumination mismatch are tuned so its split-half disagreement matches the real tile at every rock size. Best setting: noise {{tune_noise}}, illumination {{tune_light}}; log-ratio cost {{tune_cost}}.
  - Then 10 training and 6 held-out scenes.
  - Real check: the same scores against the rover answer key, with a shifted-truth control.

## 6. Results

### 6.1 Simulation (test A)

{{table_sim}}

![Figure 3](../../results/paper/fig3_simulation.png)

**Figure 3.** Test A, simulation with exact truth. (a) RMS error at the native 25 cm scale; (b) at the 6.25 cm output pixel; (c) the largest error at a change planted in one pass. Bars are means over scenes and dots are individual scenes; lower is better.

Paired against GPT, by scene:

- **25 cm:** {{pair_e25}}.
- **Artefact at the planted change:** {{pair_art}}.
- **6.25 cm:** {{pair_e6}}.

{{sim_ablation_text}}

### 6.2 Real HiRISE (tests B and C)

{{table_split}}

{{table_rover}}

![Figure 4](../../results/paper/fig4_split_half.png)

**Figure 4.** Test B, real HiRISE. Coherence between restorations made from two disjoint sets of 4 passes, against scale. Detail is treated as repeatable where coherence is at least 0.5 (dashed line).

![Figure 5](../../results/paper/fig5_boulder_edge.png)

**Figure 5.** Test C, real HiRISE, the bright boulder's edge. (a) Fitted edge width; (b) RMS ringing; (c) largest undershoot, both as a share of the edge's brightness step.

![Figure 6](../../results/comparison/site_8pass_comparison.png)

**Figure 6.** Real HiRISE, 8 passes. (a) One raw pass with its 25 cm pixels shown as blocks; (b) GPT (simplified re-implementation); (c) GPT with measured per-pass blur; (d) this work. 6.25 cm output pixels shown 1:1, with one shared contrast stretch.

{{real_text}}

### 6.3 Uncertainty (test D)

{{table_boot}}

![Figure 7](../../results/paper/fig6_uncertainty.png)

**Figure 7.** Test D. Held-out coverage of the calibrated bootstrap uncertainty at three scales; dashed lines are the nominal 68% and 95%.

### 6.4 Confidence (test E)

{{conf_text}}

{{table_conf}}

**Real-data check.**

- Against the traverse rover truth, candidate positions match truth rocks {{anch_match}}% of the time, against {{anch_shift}}% with the truth shifted 40 px. So the positions carry signal.
- But no score ranks candidates: AUC {{anch_auc_phys}} for the physical score and {{anch_auc_reco}} for the recogniser.
- The recogniser's surface class is right {{cls_acc}}% of the time, against {{cls_base}}% for always guessing soil.
- Only 4 of 27 stops could be registered, with shifts of up to 2–3 m. The answer key, not the scores, is the limit.

### 6.5 Real physical truth: Ingenuity's view of the backshell

{{truth_text}}

{{table_truth}}

![Figure 8](../../results/ingenuity/truth_test_error_vs_lambda.png)

**Figure 8.** Backshell truth test: error against Ingenuity's view within 4 m of the backshell, versus regularisation strength, in (a) the 1 m–50 cm band and (b) the 50–25 cm band. Stars mark each method's minimum; the dashed line is raw HiRISE.

![Figure 9](../../results/ingenuity/truth_test_backshell_comparison.png)

**Figure 9.** The backshell neighbourhood: (a) Ingenuity truth beside (b) raw HiRISE and (c–f) restorations at each method's truth-optimal and typical λ. The bottom row is band-passed to 25 cm–1 m, the scales the test scores.

### 6.6 Shadow evidence

{{table_shadow}}

## 7. Discussion

**Scope of the claims.** We claim a measured improvement at the 25 cm scale in simulation and on real HiRISE, a small improvement at one physical object, calibrated uncertainty at 25 cm, and a reproducible protocol for testing restoration against independent truth. We do not claim detail below the scales reported in Section 6, and our baseline is a simplified GPT (Section 5.1), so the comparison is with that baseline and not with the original software.


- **What the gain is.** The improvement over GPT comes from measuring what GPT assumes:
  - the blur of each pass;
  - which passes are noisy;
  - which pixels changed.

  It does not come from a stronger prior.
- **Assumed blur sharpens and rings.** In our baseline the assumed blur (1 px) is larger than the boulder-measured blur of the sharp passes, so on real data it deconvolves harder. The original GPT chooses σ between 0 and 1 px per tile, which may reduce this effect; we did not test it. That sharpens edges and produces ringing (Section 6.2). Edge sharpness alone therefore rewards a wrong blur model, and ringing should be reported with it.
- **Residual weighting collapses on real data.** Weights based on residuals work in simulation but collapse onto a single pass on real data. Real passes differ by illumination and season far more than by noise. Noise must be measured from the pass itself.
- **No 5 cm claim.** On real HiRISE, detail that repeats in independent passes stops at about {{repeat_limit}} cm; even in simulation with exact truth, the finest resolved scale is about {{sim_resolved}} cm. Below that, a restoration's content comes from the prior. Tao & Muller (2016) report 5 cm on eight passes over the Gusev crater traverse; we did not test that site, and our evidence is limited to the sites and passes above. We do not claim 5 cm, and we do not claim to refute it.
- **Where confidence is limited.** Confidence below about 60 cm cannot be verified with current rover truth. Our confidences are therefore simulation-calibrated, with a simulation whose disagreement is tuned to real data. The real-data check confirms positional signal but cannot confirm the probabilities.
- **A physical truth at one object.** Ingenuity's images, placed in HiRISE by two independent methods (a sun-and-shadow compass and image correlation on three passes), give a centimetre-scale truth for the backshell. It supports a clear advantage over the GPT re-implementation at any fixed regularisation and a small advantage once each is tuned (Section 6.5), but it is a single object. A wider validation needs truth with several metre-scale shared structures.
- **What would close the gap.** Better-registered truth would: more shared hardware or tracks as tie points, or stereo from rover-mast cameras.

## 8. Conclusion

Measuring each pass's blur on a static target and weighting passes by their own noise gives a measurable gain over GPT at the native 25 cm scale in simulation and on real HiRISE, and a small gain at one physical object. A confidence-scored, measurement-locked detail layer lets users decide how much unverified detail to see, with probabilities that are calibrated where truth exists and labelled where it does not.

## Reproducibility

- **Code and tests.** All code, tests (`pytest`, run on GitHub CI) and result files are in this repository. Raw data (about 12 GB) is listed with URLs in `benchmark/data_manifest.json`.
- **CPU time (4 cores).**
  - Benchmark part A: {{cpu_A}} s.
  - Part B: {{cpu_B}} s.
  - Part C: {{cpu_C}} s.
  - Bootstrap on a 32 m tile with 8 passes: {{cpu_boot}} s.

## References

*All references were checked against publisher, repository or library-catalogue records (November 2026); Tao & Muller (2016) was read in full.*

- Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Ann. Statist.* 7(1), 1–26. doi:10.1214/aos/1176344552
- Fischler, M. A., Bolles, R. C. (1981). Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. *Commun. ACM* 24(6), 381–395. doi:10.1145/358669.358692
- Gruen, A. (1985). Adaptive least squares correlation: a powerful image matching technique. *S. Afr. J. Photogramm. Remote Sens. Cartogr.* 14(3), 175–187.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proc. 34th ICML*, PMLR 70, 1321–1330.
- Immerkær, J. (1996). Fast noise variance estimation. *Comput. Vis. Image Underst.* 64(2), 300–302. doi:10.1006/cviu.1996.0060
- Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *Int. J. Comput. Vis.* 60(2), 91–110.
- Maki, J. N., et al. (2020). The Mars 2020 Engineering Cameras and Microphone on the Perseverance Rover: A Next-Generation Imaging System for Mars Exploration. *Space Sci. Rev.* 216, 137. doi:10.1007/s11214-020-00765-9
- McEwen, A. S., et al. (2007). Mars Reconnaissance Orbiter's High Resolution Imaging Science Experiment (HiRISE). *J. Geophys. Res. Planets* 112, E05S02. doi:10.1029/2005JE002605
- Otto, G. P., Chau, T. K. W. (1989). 'Region-growing' algorithm for matching of terrain images. *Image Vis. Comput.* 7(2), 83–94. doi:10.1016/0262-8856(89)90001-2
- Swan, R. M., et al. (2021). AI4MARS: A dataset for terrain-aware autonomous driving on Mars. *CVPR Workshops*, 1982–1991. doi:10.1109/CVPRW53098.2021.00226
- Tao, Y., Muller, J.-P. (2016). A novel method for surface exploration: super-resolution restoration of Mars repeat-pass orbital imagery. *Planet. Space Sci.* 121, 103–114. doi:10.1016/j.pss.2015.11.010
- Tao, Y., Conway, S. J., Muller, J.-P., Putri, A. R. D., Thomas, N., Cremonese, G. (2021). Single image super-resolution restoration of TGO CaSSIS colour images: demonstration with Perseverance rover landing site and Mars science targets (MARSGAN). *Remote Sens.* 13(9), 1777. doi:10.3390/rs13091777
- You, Y.-L., Kaveh, M. (2000). Fourth-order partial differential equations for noise removal. *IEEE Trans. Image Process.* 9(10), 1723–1730. doi:10.1109/83.869184
- Zadrozny, B., Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *Proc. 8th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining* (KDD 2002).

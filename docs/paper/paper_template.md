# How much detail can repeat-pass super-resolution recover from HiRISE? Measuring each pass's blur and noise, and testing against Ingenuity's view of the ground

## Abstract

Orbital images of Mars at 25 cm per pixel are routinely sharpened by combining several passes over the same ground, and resolutions as fine as 5 cm have been reported. Such results have mostly been judged by eye, because independent truth at the centimetre scale has rarely been available. We revisit repeat-pass super-resolution of HiRISE at the Perseverance landing area, where two things are now possible. First, each pass's blur can be measured instead of assumed. On a compact bright boulder visible in every pass, the blur differs by a factor of {{blur_ratio}} between passes. Second, the Ingenuity helicopter photographed the descent backshell from 8 m at about 3 cm per pixel, which gives centimetre-scale truth that no orbital method has seen. Starting from the Gotcha–PDE-TV (GPT) method of Tao & Muller (2016), we model each pass with its own measured blur and weight it by its own measured noise. In simulation with exact truth, this lowers the error at the native 25 cm scale from {{e25_gpt}}% to {{e25_full}}% of scene contrast. Against the Ingenuity truth, it is clearly closer to the ground at a fixed regularisation strength, and slightly closer once each method is tuned. On real HiRISE, detail that two independent sets of passes agree on extends to about {{repeat_limit}} cm, against {{sim_resolved}} cm in simulation. These scales, not the output pixel size, describe what multi-pass restoration recovers. A pass-subset bootstrap gives uncertainties that are conservative at 25 cm.

## 1. Introduction

The High Resolution Imaging Science Experiment (HiRISE) on Mars Reconnaissance Orbiter images the surface at about 25 cm per pixel (McEwen et al., 2007). That is enough to see boulders a metre across, but not the smaller rocks, ripples and fractures that matter for rover traverses and for picking targets on the ground. Many sites, among them Jezero crater, have been imaged many times. Each pass samples the ground on a slightly different grid, so a stack of passes contains information that no single image has. Multi-image super-resolution restoration uses this to estimate a finer image.

For Mars, the reference method is Gotcha–PDE-TV (GPT; Tao & Muller, 2016). GPT aligns each pass to a reference with the Gotcha matcher, then solves for a fine image whose blurred and down-sampled copies match every pass, with a fourth-order regulariser that keeps edges. Applied to eight HiRISE passes over the Spirit traverse in Gusev crater, GPT produced images on a 5 cm grid. The authors checked rover wheel-track spacing against the rover's own Navcam images (to within 8 cm) and counted many more rocks than in the original image. They also note that the 5 cm figure was set heuristically.

Two questions remain open. **Which inputs matter?** GPT assumes a single, small Gaussian blur, while real HiRISE passes differ in focus, spacecraft jitter and noise. **How much of the restored detail is real?** Visual comparison and edge sharpness reward a restoration that looks crisp, and they cannot separate recovered detail from detail supplied by the regulariser.

Jezero now makes both questions answerable. A bright, compact boulder visible in every pass from 2007 to 2022 lets us measure each pass's blur. Ingenuity's flight 26 photographed the Perseverance backshell from 8 m, at about 3 cm per pixel, which gives an independent view of the ground at a scale far finer than HiRISE.

This paper makes four contributions:

1. **Measured blur and noise per pass.** We measure each pass's blur on the boulder and its noise from the image itself, and use both in a GPT-style restoration.
2. **Tests against exact and independent truth.** Simulation with known ground, the Ingenuity truth, agreement between independent sets of real passes, and the sharpness and ringing of a real edge.
3. **A measured resolution scale.** We report the finest scale at which detail is recovered, in simulation and on real data, as an alternative to the output pixel size.
4. **Calibrated uncertainty.** A bootstrap over passes gives a per-pixel uncertainty, calibrated in simulation and reported at the scale where it holds.

## 2. Background

**Repeat-pass restoration.** Each low-resolution pass `Y_k` is modelled as the unknown fine image `X`, warped by the pass's motion, blurred by its point-spread function and down-sampled. A maximum a posteriori estimate balances fit to all passes against a smoothness prior. GPT obtains the motion with Gotcha, a region-growing matcher built on adaptive least-squares correlation (Otto & Chau, 1989; Gruen, 1985), seeded by scale-invariant features (Lowe, 2004). Its prior is a fourth-order PDE regulariser (You & Kaveh, 2000), which avoids the staircase artefacts of plain total variation. Later work by the same group moved to learned single-image methods such as MARSGAN (Tao et al., 2021). These produce sharp images, but their added texture comes from training data, so measured and learned detail cannot be told apart.

**Why the blur matters.** The restoration undoes each pass's blur. If the assumed blur is larger than the true blur, the method sharpens too hard and creates ringing next to edges. If it is smaller, real detail is left smeared. Because HiRISE passes differ, a single assumed blur is wrong for most of them.

**How restorations have been judged.** Resolution claims have rested on visual comparison, counts of detected features, and edge sharpness. All three favour a crisper image whether or not the extra detail is real. A fair test needs truth that is independent of the HiRISE passes and finer than them.

## 3. Data

**HiRISE passes.** We use {{n_passes}} HiRISE RED observations of the Jezero landing area, from 2007 to 2024. All share one map projection, so passes are aligned to the pixel by an integer shift and refined to sub-pixel accuracy by Gotcha. Eight passes with similar sun azimuth (165–178°) are used for restoration.

**Test areas.** We use two 32 m tiles. The first is a texture-rich tile at 18.4472 N, 77.4127 E, seen in all 8 passes. The second, near the landing area at 18.4456 N, 77.4520 E, contains a compact bright boulder of about 2.5 × 1.8 m (Figure 1). Because the boulder is present and unchanged in every pass, it serves as a common target for measuring blur.

![Figure 1](../../results/paper/fig1_static_boulder.png)

**Figure 1.** The blur target: a compact bright boulder visible in every pass from 2007 to 2022 (arrow). Each panel is the same 32 m tile after alignment, at 25 cm per pixel.

**Alignment across fifteen years.** As a check on registration, Figure 2 shows the parachute and backshell 1.1 km west of the boulder tile. They are absent in 2007 and 2016 and appear from the first pass after landing, at the same position in every later pass.

![Figure 2](../../results/paper/fig2_hardware_alignment.png)

**Figure 2.** The parachute (top) and backshell (bottom), absent before landing and present at the same place in every pass after it. 40 m crops; the parachute's change of shape after March 2021 is visible.

**Ingenuity truth.** On its 26th flight (sol 414), Ingenuity photographed the parachute and backshell from about 8 m with its downward navigation camera, at about 3 cm per pixel. We rectified the fisheye frames and built a mosaic around the backshell. We placed it in HiRISE by two independent methods: the direction of the helicopter's own shadow against the computed sun, and image correlation against three separate HiRISE passes. The two agree on the rotation within their uncertainty (217° ± 12° from the shadow, 208° from correlation), and the correlation recovers the scale expected from the flight height.

## 4. Method

**Forward model.** We estimate the fine image `X`, on a grid four times finer than HiRISE (6.3 cm), by minimising

`E(X) = Σ_k w_k Σ_p m_kp (A_k X − Y_k)_p² + λ Σ_p sqrt(|∇²X|² + μ²)`,

where `A_k` warps `X` by pass k's motion, blurs it with that pass's own Gaussian of width `σ_k`, and down-samples it. The second term is the fourth-order regulariser of GPT, and λ sets its strength. We solve with accelerated gradient descent. The step size comes from the largest eigenvalue of the forward operator, which we measure by power iteration.

**Measured blur.** On three passes we fit brightness profiles across the boulder's short axis with a blurred bar. The other passes are matched to the sharpest one on static ground, which gives their extra blur. The resulting widths range from {{blur_min}} to {{blur_max}} HiRISE pixels. A boulder edge is not a perfect step, so the absolute widths are upper bounds. The ratios between passes are what the restoration relies on.

**Noise weights.** Each pass's weight `w_k` is the inverse of its white-noise variance, measured from its finest-scale pixel differences (Immerkær, 1996). This is deliberately independent of the fit. Weights based on the fit residual, the obvious alternative, read lighting differences between passes as noise and on real data collapse onto a single pass (Section 6.3). Robust per-pixel weights `m_kp` (Tukey) limit the influence of pixels that disagree strongly in one pass.

**Uncertainty.** We rebuild the restoration from passes drawn with replacement. The spread of these rebuilds, scaled by a single factor fitted on one simulated scene and tested on another, gives a per-pixel uncertainty.

**Baseline.** Our GPT baseline follows the paper's equations with one assumed blur width (1 pixel) for every pass and no weighting. The original also varies the blur width between 0 and 1 pixel by image region, which we did not reproduce, so we refer to it as a simplified re-implementation. Its components (Gotcha alignment, the regulariser, the solver) are the same as ours, so the comparison isolates the effect of measured blur and noise.

## 5. Experiments

We use four tests, each answering a different question.

- **A. Simulation with exact truth.** Fractal ground, eight passes with the measured blurs, random sub-pixel offsets plus 0.3-pixel jitter, 3% noise, and a 2 m change planted in one pass. Every method estimates its own motion. This test asks how close each restoration is to the true ground ({{n_sim}} scenes).
- **B. The Ingenuity truth.** Within 4 m of the backshell, we compare each restoration with Ingenuity's view in two bands, 1 m–50 cm and 50–25 cm. We use band-passed correlation with a ±25 cm placement tolerance applied equally to all methods. This test asks whether the restored detail matches the real ground.
- **C. Real-data consistency.** We split the eight real passes into two disjoint halves and restore each half separately. Detail that both halves agree on comes from the data, not the prior. This test asks down to what scale the detail is real.
- **D. A real edge.** On the boulder, we measure the edge width of each restoration together with the ringing and undershoot beside it. This test asks how much artefact each method pays for its sharpness.

## 6. Results

### 6.1 Simulation: measured blur and noise reduce the error at 25 cm

Table 1 and Figure 3 summarise the simulation. At the native 25 cm scale, the GPT baseline leaves an error of {{e25_gpt}}% of scene contrast. Using the measured blur alone reduces it to {{e25_gptb}}%. Adding noise weighting brings it to {{e25_full}}%, {{rel_abs}}% lower than the baseline, and the improvement holds in every scene. With exact motion the error would be {{e25_oracle}}%, so alignment is not the limiting step. Noise weighting without the measured blur makes the result worse ({{e25_gptw}}%): weights only help once the blur model is right. At the finer 6.25 cm output pixel all methods remain far from the truth (errors near 30%), which is the first sign that detail at that scale is not being recovered.

**Table 1.** Simulation with exact truth, mean over {{n_sim}} scenes. Lower is better.

{{table_sim2}}

![Figure 3](../../results/paper/fig3_simulation.png)

**Figure 3.** Simulation with exact truth. (a) Error at the native 25 cm scale, (b) at the 6.25 cm output pixel, (c) the largest error at a change planted in one pass. Bars are means and dots are individual scenes.

### 6.2 The Ingenuity truth: closer to the real ground

Figure 4 shows the error against Ingenuity's view as the regularisation strength λ varies. At a fixed λ = 0.04, our restoration is clearly closer to the ground than the GPT baseline in the 50–25 cm band ({{t04_full_50}} against {{t04_gpt_50}}, where 0 is perfect agreement). It also does better in the 1 m–50 cm band ({{t04_full_100}} against {{t04_gpt_100}}). When each method is given its own best λ, the advantage is smaller: {{tuned_100}} in correlation at 1 m–50 cm and {{tuned_50}} at 50–25 cm. Choosing λ on one half of the area and scoring on the other leaves {{held_100}} at 1 m–50 cm, essentially no difference, and {{held_50}} at 50–25 cm. Across 45 plausible placements of the truth, our restoration is ahead in {{place_100}}% of placements at 1 m–50 cm and {{place_50}}% at 50–25 cm. The 1 m–50 cm advantage is stable across the radius of the test area (2.5–8 m); the 50–25 cm advantage is not.

**Table 2.** Error against the Ingenuity truth (1 − correlation) within 4 m of the backshell. Lower is better.

{{table_truth2}}

![Figure 4](../../results/ingenuity/truth_test_error_vs_lambda.png)

**Figure 4.** Error against Ingenuity's view versus regularisation strength, in (a) the 1 m–50 cm and (b) the 50–25 cm band. Stars mark each method's minimum; the dashed line is raw HiRISE.

![Figure 5](../../results/ingenuity/truth_test_backshell_comparison.png)

**Figure 5.** The backshell neighbourhood: (a) Ingenuity's view beside (b) raw HiRISE and (c–f) restorations at each method's best and at a typical λ. The bottom row is band-passed to 25 cm–1 m, the scales the test scores.

### 6.3 Real HiRISE: where detail stops being real

Restoring two disjoint halves of the real passes gives a direct measure of repeatable detail (Figure 6). For every method, the two halves agree well at scales above about 1 m and stop agreeing near {{repeat_limit}} cm. Below that, neither the GPT baseline nor our restoration recovers detail that the other half confirms. Over 25–100 cm the GPT baseline's halves agree slightly more than ours ({{coh_gpt}} against {{coh_full}}).

On the boulder edge (Figure 7), the GPT baseline produces the narrowest edge ({{edge_gpt}} cm against {{edge_full}} cm), but at a price. Its ringing is {{ring_gpt}}% of the edge step against {{ring_full}}% for ours, and its largest undershoot is {{under_gpt}}% against {{under_full}}%. The assumed blur is larger than the measured blur of the sharpest passes, so the baseline deconvolves too hard. Edge sharpness alone therefore rewards a wrong blur model. Figure 8 shows the restorations side by side.

**Table 3.** Real HiRISE: agreement between independent halves (higher is better) and the boulder edge (narrower is sharper; lower ringing is cleaner).

{{table_real2}}

![Figure 6](../../results/paper/fig4_split_half.png)

**Figure 6.** Agreement between restorations made from two disjoint sets of four real passes, against scale. Detail counts as repeatable where coherence is at least 0.5 (dashed).

![Figure 7](../../results/paper/fig5_boulder_edge.png)

**Figure 7.** The boulder edge on real HiRISE: (a) edge width, (b) ringing, (c) largest undershoot.

![Figure 8](../../results/comparison/site_8pass_comparison.png)

**Figure 8.** Eight real passes, 32 m tile. (a) One raw pass; (b) GPT baseline; (c) GPT with measured blur; (d) this work. 6.25 cm pixels shown 1:1 with one shared contrast stretch.

### 6.4 Uncertainty

After calibration on one simulated scene, the bootstrap uncertainty covers the true error on a held-out scene in {{cov25_1}}% of pixels at ±1σ and {{cov25_2}}% at ±2σ, at the native 25 cm scale (Figure 9). That is more than the nominal 68% and 95%, so the uncertainty errs on the safe side. At the 6.25 cm output pixel, coverage can be matched by calibration, but the uncertainty no longer ranks which pixels are wrong, so we report it at 25 cm.

![Figure 9](../../results/paper/fig6_uncertainty.png)

**Figure 9.** Held-out coverage of the calibrated bootstrap uncertainty at three scales; dashed lines are the nominal 68% and 95%.

## 7. Discussion

**What the gain is.** The improvement over the GPT baseline comes from measuring two things the baseline assumes: how blurred each pass is, and how noisy. It does not come from a stronger prior. In simulation the measured blur provides most of the gain at 25 cm, and noise weighting adds to it once the blur is right. Against the Ingenuity truth the gain is largest when both methods use the same regularisation, and small once each is tuned. That is what one would expect if the measured blur mainly removes over-sharpening.

**What multi-pass restoration recovers.** On real HiRISE, detail that independent passes agree on reaches about {{repeat_limit}} cm, and in simulation with exact truth about {{sim_resolved}} cm. Both are well above the 6.3 cm output pixel. The output grid is a sampling choice that lets edges be placed precisely. It is not the resolution of the result, and finer structure in a restored image comes mostly from the regulariser. We suggest that resolution claims for multi-pass restoration be stated as a scale validated against independent passes or independent truth, alongside the output pixel size.

**Edge sharpness is not a fair score on its own.** On the real boulder edge, the sharpest result was also the one with the most ringing. A blur model that is too large looks sharper, so edge-based measures should be reported together with ringing.

**Limitations.**
- The truth comparison uses a single object and its surroundings, about 4 m across. The rock field around it gave no discrimination between methods, because Ingenuity imaged it at midday and HiRISE in the afternoon, so rocks are lit from different directions.
- Our baseline is a simplified GPT, without the original's region-dependent blur width.
- The blur target is a boulder, not a manufactured object, so its absolute blur widths are upper bounds.
- Robust per-pixel weights reduced artefacts at a planted change in simulation, but we have not yet tested them on a real change between passes.

**Outlook.** The same measurements support further products: a per-pixel confidence for small rocks, using agreement between pass subsets and shadows under different sun angles, and maps that flag where the restoration should not be trusted. Validating them needs more centimetre-scale truth, for example from further Ingenuity flights or stereo from rover mast cameras.

## 8. Conclusion

Measuring each HiRISE pass's blur on a static target, and weighting each pass by its own noise, makes repeat-pass super-resolution measurably more accurate: {{rel_abs}}% lower error at the native 25 cm scale in simulation, and closer to Ingenuity's centimetre-scale view of the ground. Independent passes agree on detail down to about {{repeat_limit}} cm. That is the scale such restorations recover, and we suggest it be reported alongside the output pixel size.

## Data and code availability

All HiRISE observations are public through the Planetary Data System; the list with download links is in the repository's `benchmark/data_manifest.json`. The Ingenuity navigation images are from the PDS Mars 2020 helicopter camera archive. All code, tests, result files and the scripts that generate every table and figure in this paper are in the project repository. Full method details are in `docs/METHODS.md`. Every number in this paper is filled in from a result file by `scripts/make_paper.py`. All computation ran on a 4-core CPU.

## References

- Gruen, A. (1985). Adaptive least squares correlation: a powerful image matching technique. *S. Afr. J. Photogramm. Remote Sens. Cartogr.* 14(3), 175–187.
- Immerkær, J. (1996). Fast noise variance estimation. *Comput. Vis. Image Underst.* 64(2), 300–302. doi:10.1006/cviu.1996.0060
- Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *Int. J. Comput. Vis.* 60(2), 91–110.
- McEwen, A. S., et al. (2007). Mars Reconnaissance Orbiter's High Resolution Imaging Science Experiment (HiRISE). *J. Geophys. Res. Planets* 112, E05S02. doi:10.1029/2005JE002605
- Otto, G. P., Chau, T. K. W. (1989). 'Region-growing' algorithm for matching of terrain images. *Image Vis. Comput.* 7(2), 83–94. doi:10.1016/0262-8856(89)90001-2
- Tao, Y., Muller, J.-P. (2016). A novel method for surface exploration: super-resolution restoration of Mars repeat-pass orbital imagery. *Planet. Space Sci.* 121, 103–114. doi:10.1016/j.pss.2015.11.010
- Tao, Y., Conway, S. J., Muller, J.-P., Putri, A. R. D., Thomas, N., Cremonese, G. (2021). Single image super-resolution restoration of TGO CaSSIS colour images: demonstration with Perseverance rover landing site and Mars science targets. *Remote Sens.* 13(9), 1777. doi:10.3390/rs13091777
- You, Y.-L., Kaveh, M. (2000). Fourth-order partial differential equations for noise removal. *IEEE Trans. Image Process.* 9(10), 1723–1730. doi:10.1109/83.869184

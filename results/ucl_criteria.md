# Pre-registered criteria: "better than UCL"

Written on 2026-11-01, **before** the traverse-wide (sols 13–600) real-data rock results were computed.

## What UCL claims

Tao & Muller (2016) report the following for GPT super-resolution restoration (SRR) from 6 or more HiRISE passes at Gale crater, using count-based comparison with Navcam:

- (a) 25 cm HiRISE alone resolves individual rocks only when they are larger than 100 cm.
- (b) 6.25 cm SRR resolves *most* individual rocks larger than 50 cm.
- (c) SRR resolves a *limited* number of rocks between 25 and 50 cm.

## Our test

Our test is stricter than UCL's:

- **Position matching, not counts.** Each rock seen in the rover truth must be detected at the same place, within the larger of 2 px or half its diameter.
- **Truth:** terrain-aware Navcam stereo (rover truth v3).
- **Chance level:** detections shifted 40 px.
- **False detections:** counted per 100 m².
- **Confidence intervals:** 95% bootstrap over stops.
- **Products compared:**
  - raw HiRISE upsampled bicubically (one pass);
  - single-frame deconvolution;
  - shift-and-average;
  - our multi-frame restoration (8 passes with sun azimuth 165–178°, blur 1.07 px).
- **Fixed settings:** the detector (`experiments/exp3_rocks.py`) and all thresholds stay as they are; nothing is tuned on these results.

## We call it "better than UCL" only if ALL of the following hold

1. **Rocks ≥ 50 cm:** the multi-frame restoration finds more than 50% of the rover-verified rocks ≥ 50 cm, and the lower bound of the 95% confidence interval is above 50%. This meets claim (b) under a stricter, position-matched test. There must be at least 30 such rocks.
2. **Rocks 25–50 cm:** the multi-frame restoration finds more of them than raw HiRISE, and the paired bootstrap confidence interval of the difference excludes zero. This turns UCL's qualitative "limited" into a measured, position-verified gain.
3. **False detections:** the multi-frame restoration's false-detection rate is no higher than raw HiRISE's.
4. **Above chance:** every completeness number used above is clearly above its chance level.

If only some of these hold, we report exactly which ones and do not claim to beat UCL.

## Amendment 1 (2026-11-01, before any result on the extended ground truth)

**Why:** the traverse rock test showed that the rover-verified ground holds only 4 rocks ≥ 50 cm, so criterion 1 (at least 30 such rocks) cannot be tested.

**The cause** is how the truth was built. It used only steeply downward Navcam tiles (mast elevation ≤ −15°), which see 2–8 m around the rover, and the rover parks away from boulders.

**The change:**

- The truth is extended to ±16 m around each stop, the full 32 m area each restoration already covers.
- Navcam tiles with mast elevation ≤ −5° are now included.
- The terrain-aware stereo method, the detector, the matching rule and all thresholds are unchanged.
- Results on the original ±8 m truth stay reported (`results/rocks_traverse_thr*.json`).

## Amendment 2 (2026-11-01, before any registered result)

**Why:** a check that uses no rock detections showed that the rover truth and HiRISE are misregistered by 0.6–1.4 m at typical stops. The matching tolerance for a 50 cm rock is 25 cm, so at the given positions the test cannot credit any method.

The evidence: at the nominal position, the high-pass correlation between the rover map blurred to HiRISE resolution and raw HiRISE has a median of 0.07; at its peak the median is 0.28.

**The change:**

- **Registration:** each stop's rover truth is shifted to the peak of that image correlation. The search covers ±3 m, in 2 px steps followed by a 1 px refinement, against raw HiRISE (bicubic).
- **Same shift for every product:** the shift is applied to all products alike.
- **Which stops count:** only stops with a well-defined peak are scored, meaning the peak lies inside the search window and the correlation is at least 0.3. Stops that fail are reported as unregistered.
- **Unchanged:** the detector, the matching rule and the thresholds.

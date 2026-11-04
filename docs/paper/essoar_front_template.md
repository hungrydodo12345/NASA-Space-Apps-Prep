# Revisiting Repeat-Pass Super-Resolution of HiRISE: Measured Blur, Measured Noise, and Centimeter-Scale Truth From Ingenuity

**[Author Name 1]<sup>1</sup>**, **[Author Name 2]<sup>2</sup>**

<sup>1</sup> [Department, Institution, City, Country]; ORCID [0000-0000-0000-0000]

<sup>2</sup> [Department, Institution, City, Country]; ORCID [0000-0000-0000-0000]

**Corresponding author:** [Name] ([email address])

**Keywords:** Mars; HiRISE; super-resolution; Ingenuity; image restoration; Jezero crater

## Key Points

- Measuring each HiRISE pass's blur and noise improves multi-pass image restoration, tested against Ingenuity helicopter images at two sites
- Settings tuned at one Ingenuity site transfer to the other for the new method, but not for the Gotcha–PDE-TV baseline
- Independent passes confirm detail to about {{repeat_limit}} cm, and HiRISE optics pass nothing finer than about 20 cm

## Abstract

Multi-pass super-resolution restoration of HiRISE images has been reported to recover detail far below the 25 cm pixel, but such results have rarely been tested against independent truth. Building on the Gotcha–PDE-TV (GPT) method of Tao and Muller (2016), we measure each pass's blur on a static target and weight each pass by its measured noise. We evaluate restorations of the Perseverance landing area in Jezero crater against centimeter-scale truth at two sites imaged by the Ingenuity helicopter. In simulation, the measured inputs lower the error at the native 25 cm scale from {{e25_gpt}}% to {{e25_full}}% of scene contrast, provided each pass's blur is known to within 20%. Against the Ingenuity truth, our restoration is closer to the ground than every GPT variant at a fixed regularization strength. With each method fully tuned, the advantage is small, and statistically clear only in the 50–25 cm band at the second site ({{s2_tuned_50}} in correlation). Settings chosen at one truth site transfer to the other with almost no loss for our method, but not for GPT, which then falls to or below a single raw pass. Detail confirmed by independent pass subsets reaches about {{repeat_limit}} cm, and HiRISE's optics pass nothing finer than about 20 cm in half-period, so the output grid of such restorations should not be read as their resolution. A pass-subset bootstrap gives conservative per-pixel uncertainties at 25 cm.

## Plain Language Summary

The HiRISE camera orbiting Mars takes pictures in which each pixel covers 25 cm of ground. By combining several pictures of the same place taken on different orbits, computer methods can produce sharper images, and some studies have reported detail as fine as 5 cm. It has been hard to check how much of that detail is real, because there were no independent pictures of the ground at that scale. The Ingenuity helicopter changed this: it photographed the ground at Jezero crater from a few meters up. We used two of its sites as an answer key. We found that measuring how blurred and how noisy each orbital picture is, instead of assuming it, makes the combined image closer to what Ingenuity saw. The improvement is modest, but it carries over from one site to the other, which the earlier method's settings do not. We also show that real detail stops at about two thirds of a meter in these images, and that the telescope itself cannot record anything finer than about 20 cm, so a 5 cm grid does not mean 5 cm detail.


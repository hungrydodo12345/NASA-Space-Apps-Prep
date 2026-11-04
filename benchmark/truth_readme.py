"""Write the Ingenuity-truth results into README.md between the TRUTH markers, from benchmark/results/ingenuity_truth/*.json. Usage: python benchmark/truth_readme.py"""
import os, re, json
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); R = os.path.join(ROOT, "benchmark", "results", "ingenuity_truth")
fin = json.load(open(f"{R}/object_final.json")); ho = json.load(open(f"{R}/object_heldout.json")); rb = json.load(open(f"{R}/object_robust_placements.json")); b = fin["best"]; cv = fin["curves"]; bs = fin["bootstrap"]
e = lambda band, f: 1 - b[band][f][1]
L = ["## Real physical truth: Ingenuity's view of the parachute and backshell\n",
     "Ingenuity photographed the debris field from 8 m up (flight 26, sol 414) at about 3 cm per pixel. We placed that mosaic in HiRISE using two independent methods that agree (the helicopter's shadow plus the computed sun: **217° ± 12°**; image correlation on three independent HiRISE passes: **208°**, with the scale 0.0290 m/px equal to the nominal 8 m / 275 px). Details: [`docs/METHODS.md`](docs/METHODS.md) section 7c; figures in [`results/ingenuity/`](results/ingenuity/).\n",
     "![backshell comparison](results/ingenuity/truth_test_backshell_comparison.png)\n",
     "**Test on the backshell** (within 4 m; correlation with the truth in band-passed images, each method tuned to its own minimum of error; error = 1 − correlation, 50–25 cm band):\n",
     "| Method | Error at its minimum | at λ = 0.005 | at λ = 0.04 |", "|---|---|---|---|",
     f"| raw HiRISE (one pass) | {e('50-25 cm','raw'):.3f} | — | — |",
     f"| UCL re-implementation | {e('50-25 cm','gpt'):.3f} (λ {b['50-25 cm']['gpt'][0]}) | {1-cv['50-25 cm']['gpt']['0.005']:.3f} | {1-cv['50-25 cm']['gpt']['0.04']:.3f} |",
     f"| UCL + our measured blur | {e('50-25 cm','gpt+blur'):.3f} (λ {b['50-25 cm']['gpt+blur'][0]}) | {1-cv['50-25 cm']['gpt+blur']['0.005']:.3f} | {1-cv['50-25 cm']['gpt+blur']['0.04']:.3f} |",
     f"| **This work** | **{e('50-25 cm','full'):.3f}** (λ {b['50-25 cm']['full'][0]}) | {1-cv['50-25 cm']['full']['0.005']:.3f} | {1-cv['50-25 cm']['full']['0.04']:.3f} |\n",
     "![error vs lambda](results/ingenuity/truth_test_error_vs_lambda.png)\n",
     f"- **At the same regularisation, this work is much closer to the truth** than the UCL re-implementation (at λ = 0.04: {e('50-25 cm','full')*0+1-cv['50-25 cm']['full']['0.04']:.3f} against {1-cv['50-25 cm']['gpt']['0.04']:.3f}).",
     f"- **Each tuned to its own minimum:** 1 m–50 cm {bs['1 m-50 cm']['full']['mean']:+.3f} (95% CI {bs['1 m-50 cm']['full']['ci95'][0]:+.3f} to {bs['1 m-50 cm']['full']['ci95'][1]:+.3f}); 50–25 cm {bs['50-25 cm']['full']['mean']:+.3f} (95% CI {bs['50-25 cm']['full']['ci95'][0]:+.3f} to {bs['50-25 cm']['full']['ci95'][1]:+.3f}) in correlation; both curves are U-shaped, so the minima are real.",
     f"- **Out of sample** (settings chosen on one half of the disc, scored on the other): {ho['50-25 cm']['full']['heldout']-ho['50-25 cm']['gpt']['heldout']:+.3f} at 50–25 cm. Most of the in-sample gain comes from tuning on the same data.",
     "- **Disc radius:** the 1 m–50 cm advantage is stable at 2.5, 4, 6 and 8 m (+0.007 to +0.010 in correlation, bootstrap interval above zero at 4, 6 and 8 m); the 25–50 cm advantage is not (it disappears at 8 m, where the patch includes the rock field the truth cannot resolve at that scale).",
     f"- **Placement robustness:** ahead of UCL in {100*rb['summary']['50-25 cm']['vs_gpt']['full']['frac_better']:.0f}% of 45 plausible placements at 50–25 cm.",
     "- **Cleaner mosaic and eye check:** telemetry showed my frame chain is accurate near the backshell but mis-scaled late in the flight; rebuilding the truth from the accurate frames only reproduces the result (1 m–50 cm +0.009, 50–25 cm +0.013 in-sample and +0.021 out of sample, ahead in 87–93% of placements). Eye check: [`results/ingenuity/eye_check_placement.png`](results/ingenuity/eye_check_placement.png).",
     "- **Limits:** one 1.5 m object, chosen after a 181 m² rock-field patch proved uninformative (the truth was taken at midday and HiRISE in the afternoon, so rocks look completely different); this cannot establish a general result.\n"]
block = "<!-- TRUTH:START -->\n" + "\n".join(L) + "\n<!-- TRUTH:END -->"; p = os.path.join(ROOT, "README.md"); s = open(p).read()
if "<!-- TRUTH:START -->" in s: s = re.sub(r"<!-- TRUTH:START -->.*?<!-- TRUTH:END -->", lambda _: block, s, flags=re.S)
else: s = s.replace("## How it works", block + "\n\n## How it works", 1)
open(p, "w").write(s); print("README updated")

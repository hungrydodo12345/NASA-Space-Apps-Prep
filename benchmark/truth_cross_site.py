"""Cross-site transfer between the two Ingenuity truth sites: choose each method's setting (lambda; for GPT also its assumed blur 0.5/0.75/1 px) on one site
and score it on the other. This is the situation of any real use, where no truth exists at the target. Reads the error-vs-setting curves of both sites.
Usage: python benchmark/truth_cross_site.py <backshell_object_final.json> <site2_final.json> <out_json>"""
import sys, json
A, B, OUT = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
G = {"gpt-any": ["gpt", "gpt-s0.5", "gpt-s0.75"], "full": ["full"], "gpt+blur": ["gpt+blur"]}
out = {}
for b in ("1 m-50 cm", "50-25 cm"):
    out[b] = {}
    for name, (src, dst) in {"backshell_to_site2": (A, B), "site2_to_backshell": (B, A)}.items():
        r = dict(raw_at_target=dst["curves"][b]["raw"]["0.0"])
        for g, grp in G.items():
            cand = [(f, l) for f in grp if f in src["curves"][b] for l in src["curves"][b][f] if l in dst["curves"][b].get(f, {})]
            f, l = max(cand, key=lambda c: src["curves"][b][c[0]][c[1]])
            r[g] = dict(setting=f"{f}__{l}", corr_at_target=dst["curves"][b][f][l], best_possible_at_target=max(dst["curves"][b][ff][ll] for ff, ll in cand))
        out[b][name] = r
        print(b, name, {g: (v["setting"], round(v["corr_at_target"], 3), round(v["best_possible_at_target"], 3)) if isinstance(v, dict) else round(v, 3) for g, v in r.items()})
json.dump(out, open(OUT, "w"), indent=1)

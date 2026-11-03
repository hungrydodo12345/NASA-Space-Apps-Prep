import sys, math, json, numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep")
from experiments import scenes
from pipeline import shadow_evidence as se
rng = np.random.default_rng(3); F = 4                                  # render on a 6.25 cm grid, image at 25 cm
f = scenes.fractal(7, n=1024); bg = (f[0] if isinstance(f, tuple) else f); bg = (bg - bg.mean()) / bg.std() * 0.6
N = bg.shape[0] // F
geoms = [dict(sun_az=a, incidence=i) for a, i in [(177.4, 47.6), (176.8, 49.9), (165.3, 45.8), (167.0, 46.0), (168.7, 46.2), (177.7, 47.7), (167.9, 58.9), (169.8, 55.6), (137.0, 47.0), (196.0, 49.0)]]
sig = [1.34, 1.43, 0.89, 0.59, 0.83, 1.38, 1.16, 1.20, 1.1, 1.1]
objs = []                                                               # (y, x, D, is_rock) in 25 cm px
def place(D, rock):
    for _ in range(200):
        y, x = rng.uniform(12, N - 12, 2)
        if all(math.hypot(y - o[0], x - o[1]) > 8 for o in objs): objs.append((y, x, D, rock)); return
for D in (40, 60, 100, 150):
    for _ in range(18): place(D, True)
    for _ in range(18): place(D, False)
yy, xx = (np.mgrid[0:bg.shape[0], 0:bg.shape[1]] + 0.5) / F
passes = []
for g, s in zip(geoms, sig):
    sc = bg.copy(); sy, sx = se.shadow_vector(g["sun_az"])
    for y, x, D, rock in objs:
        r = D / 2 / 25; body = np.hypot(yy - y, xx - x) <= r; sc[body] += 1.2
        if rock:
            L = (D / 2) * math.tan(math.radians(g["incidence"])) / 25; cy, cx = y + sy * (r + L / 2), x + sx * (r + L / 2)
            u = (yy - cy) * sy + (xx - cx) * sx; v = -(yy - cy) * sx + (xx - cx) * sy
            sc[((u / max(L / 2, .25)) ** 2 + (v / max(r, .25)) ** 2 <= 1) & ~body] -= 1.6
    lr = ndi.gaussian_filter(sc.reshape(N, F, N, F).mean((1, 3)), s) + rng.normal(0, 0.06, (N, N))
    passes.append(lr)
def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg); return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
setups = {"1 pass": [2], "8 same-sun passes": list(range(8)), "10 passes incl. 2 different sun angles": list(range(10))}
res = {}
for nm, idx in setups.items():
    ev = se.evidence([passes[i] for i in idx], [geoms[i] for i in idx], [sig[i] for i in idx])
    res[nm] = {}
    for D in (40, 60, 100, 150):
        S = ev[D][0]; val = lambda y, x: float(S[int(round(y)) - 1:int(round(y)) + 2, int(round(x)) - 1:int(round(x)) + 2].max())
        rk = [val(y, x) for y, x, d, r in objs if d == D and r]; dc = [val(y, x) for y, x, d, r in objs if d == D and not r]
        bgv = [val(*rng.uniform(12, N - 12, 2)) for _ in range(200)]
        res[nm][D] = dict(rock_vs_decoy=auc(rk, dc), rock_vs_ground=auc(rk, bgv))
    print(f"{nm:40s} " + "  ".join(f"{D}cm: decoy {res[nm][D]['rock_vs_decoy']:.2f} / ground {res[nm][D]['rock_vs_ground']:.2f}" for D in (40, 60, 100, 150)), flush=True)
json.dump(res, open("/home/user/NASA-Space-Apps-Prep/results/shadow_evidence_simulation.json", "w"), indent=1)

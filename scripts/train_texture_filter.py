"""Train + validate a small 'real ground vs hardware/shadow/seam' classifier on hand-labelled rover patches.

Usage: python scripts/train_texture_filter.py <candidates.npz> <labels.json> <out_model.json>
Validation holds out whole rover stops (GroupKFold) so a stop's tiles are never both in train and test.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline import texture_quality as tq
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

C, L, OUT = sys.argv[1:4]
d = np.load(C); lab = json.load(open(L))["labels"]
base = lambda i: dict(peak=d["peak"][i], kurt=d["kurt"][i], step=d["step"][i], coh_fine=d["coh_fine"][i], coh_coarse=d["coh_coarse"][i], satur=d["satur"][i])
Xall = np.stack([tq.feature_vector(d["patches"][i].astype(float), base(i)) for i in range(len(d["r"]))])
ids = np.array([l["index"] for l in lab]); y = np.array([l["label"] == "ground" for l in lab]).astype(int)
X, g = Xall[ids], d["stop"][ids]
print(f"labelled {len(ids)} (ground {y.sum()}, other {(1-y).sum()}) at {len(set(g))} stops; features: {tq.FEATURE_NAMES}")
models = {"logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
          "random_forest": lambda: RandomForestClassifier(300, min_samples_leaf=2, random_state=0)}
res = {}
for name, mk in models.items():
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(6, len(set(g)))).split(X, y, g):
        m = mk().fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    res[name] = p
    for thr in (0.5, 0.7, 0.85):
        a = p >= thr; prec = y[a].mean() if a.any() else float("nan"); rec = (a & (y == 1)).sum() / max(1, (y == 1).sum())
        print(f"  {name:14s} thr {thr:.2f}: accepts {a.sum():3d}/{len(y)} | precision (accepted that are real ground) {100*prec:5.1f}% | recall {100*rec:5.1f}% | accuracy {100*((p>=0.5)==y).mean():.1f}%")
base_acc = max(y.mean(), 1 - y.mean())
print(f"majority-class baseline accuracy {100*base_acc:.1f}%")
hand = np.array([tq.accept(dict(step=Xall[i][2], coh_coarse=Xall[i][4], kurt=np.expm1(Xall[i][1]), peak=Xall[i][0], flat=tq.flat_fraction(d['patches'][i].astype(float)), satur=Xall[i][5]))[0] for i in ids])
print(f"hand-set rules: accepts {hand.sum()}/{len(y)} | precision {100*y[hand].mean():.1f}% | recall {100*(hand & (y==1)).sum()/y.sum():.1f}%")
final = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(X, y)
sc, lr = final.steps[0][1], final.steps[1][1]
json.dump(dict(features=tq.FEATURE_NAMES, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]),
               cv={k: dict(accuracy=float(((v >= 0.5) == y).mean())) for k, v in res.items()}, n_labels=int(len(y))), open(OUT, "w"), indent=1)
pall = final.predict_proba(Xall)[:, 1]
np.save(OUT.replace(".json", "_scores.npy"), pall)
print("coefficients:", {f: round(c, 2) for f, c in zip(tq.FEATURE_NAMES, lr.coef_[0])})
print(f"all {len(pall)} candidates: accepted at 0.85 -> {(pall>=0.85).sum()}, at 0.7 -> {(pall>=0.7).sum()}")

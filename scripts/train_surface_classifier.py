"""Train + spatially cross-validate the orbital surface-type classifier. Usage: python scripts/train_surface_classifier.py <data_dir> NAV|GEO [min_samples] [cell_m]"""
import os
import sys, warnings, time
import numpy as np, glymur
warnings.filterwarnings("ignore")
S = sys.argv[1]; KIND = sys.argv[2]; MINC = int(sys.argv[3]) if len(sys.argv) > 3 else 30
AGG = int(sys.argv[4]) if len(sys.argv) > 4 else 1
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline import surface_classifier as sc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
d = np.load(f"{S}/rover/projected_{KIND}.npz"); cells, counts, names = d["cells"], d["counts"], list(d["names"])
if AGG > 1:
    agg = {}
    for (a, b), c in zip(cells // AGG, counts): agg.setdefault((a, b), np.zeros(counts.shape[1]))[:] += c
    cells = np.array(list(agg)); counts = np.array(list(agg.values()))
keep = counts.sum(1) >= MINC; cells, counts = cells[keep], counts[keep]
Y = counts / counts.sum(1, keepdims=True)
CELL = 4 * AGG; H1 = max(8, 2 * CELL); H2 = 3 * H1                              # cell 1 m; patches 16 px (4 m) and 48 px (12 m)
L = cells[:, 0] * CELL + CELL // 2; Sm = cells[:, 1] * CELL + CELL // 2
l0, l1, s0, s1 = L.min() - H2 - 2, L.max() + H2 + 2, Sm.min() - H2 - 2, Sm.max() + H2 + 2
t = time.time(); J = glymur.Jp2k(f"{S}/hirise_jezero/ESP_045994_1985_RED_A_01_ORTHO.JP2")
img = np.asarray(J[l0:l1, s0:s1]).astype(np.float64); print(f"read ortho block {img.shape} in {time.time()-t:.0f}s", flush=True)
X, ok = [], []
for li, si in zip(L - l0, Sm - s0):
    a = img[li - H1:li + H1, si - H1:si + H1]; b = img[li - H2:li + H2, si - H2:si + H2]
    if (b == 0).mean() > 0.01: ok.append(False); X.append(None); continue
    X.append(np.r_[sc.features(a), sc.features(b)]); ok.append(True)
ok = np.array(ok); X = np.array([x for x, o in zip(X, ok) if o]); Y = Y[ok]; L, Sm = L[ok], Sm[ok]
grp = np.unique((L // 400) * 100000 + (Sm // 400), return_inverse=True)[1]
print(f"{KIND} @ {AGG} m cells: cells {len(X)} (>= {MINC} samples each), spatial 100 m blocks {grp.max()+1}, features {X.shape[1]}")
pred = np.zeros_like(Y); base = np.zeros_like(Y)
for tr, te in GroupKFold(n_splits=min(10, grp.max() + 1)).split(X, Y, grp):
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(X[tr], Y[tr])
    pred[te] = m.predict(X[te]); base[te] = Y[tr].mean(0)
t_dom, p_dom, b_dom = Y.argmax(1), pred.argmax(1), base.argmax(1)
# prior-corrected decision (divide by training prior) for balanced accuracy
p_bal = (pred / np.maximum(base, 1e-6)).argmax(1)
mae_p, mae_b = np.abs(pred - Y).mean(), np.abs(base - Y).mean()
print(f"\nHeld-out 100 m blocks:")
print(f"  dominant type accuracy      : classifier {100*(p_dom==t_dom).mean():.1f}%  | most-common baseline {100*(b_dom==t_dom).mean():.1f}%")
print(f"  balanced accuracy (all types equal weight): classifier {100*balanced_accuracy_score(t_dom, p_bal):.1f}%  | chance {100/len(names):.1f}%")
print(f"  composition error            : {100*(1-mae_p/mae_b):.1f}% lower than baseline")
for i, c in enumerate(names):
    r = np.corrcoef(pred[:, i], Y[:, i])[0, 1]
    print(f"  {c:13s}: share {100*Y[:,i].mean():5.1f}% | error reduction {100*(1-np.abs(pred[:,i]-Y[:,i]).mean()/np.abs(base[:,i]-Y[:,i]).mean()):+5.1f}% | correlation {r:+.2f}")
np.savez(f"{S}/rover/cells_{KIND}_{AGG}m.npz", X=X, Y=Y, grp=grp, L=L, S=Sm, names=np.array(names))

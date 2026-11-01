"""Train and evaluate the Stage-1 recogniser with spatial cross-validation, then score it on UCL's yardstick.

* Stops are grouped into 200 m map cells; cells are split into 3 folds. Each fold's model never sees its test cells.
* Inside each fold, 20% of the training cells are held back for early stopping and to choose the detection threshold.
* Every stop therefore gets an out-of-fold rock list. These are compared with rocks the rover truth shows (position
  match, as scripts/rocks_real_matched.py), next to the fixed blob detector run on raw HiRISE and on the restoration.
Usage: python scripts/train_recognizer.py <restored_dir> <truth_dir> <texture_filter.json> <out_dir>
       env: ITERS (default 2500), THR_CLEAN (0.85), SEED
"""
import sys, os, glob, json, time
import numpy as np, torch
from scipy import ndimage as ndi
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline import recognizer as rz, texture_quality as tq, groundtruth as gt
sv = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[2]]
from experiments import exp3_rocks as E
sys.argv = sv
RD, TRUTH, FILT, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
ITERS = int(os.environ.get("ITERS", 2500)); SEED = int(os.environ.get("SEED", 0)); THR_CLEAN = float(os.environ.get("THR_CLEAN", 0.85))
PX = E.PX; torch.manual_seed(SEED); rng = np.random.default_rng(SEED); torch.set_num_threads(4)
filt = tq.load_filter(FILT)


def clean_mask(z, valid, win=32, stride=16):
    m = np.zeros_like(valid)
    for y in range(0, z.shape[0] - win + 1, stride):
        for x in range(0, z.shape[1] - win + 1, stride):
            if valid[y:y + win, x:x + win].mean() >= 0.98 and (THR_CLEAN <= 0 or tq.ground_score(z[y:y + win, x:x + win], filt) >= THR_CLEAN): m[y:y + win, x:x + win] = True
    return m & valid


# ---------------- data
D = []
for f in sorted(glob.glob(f"{RD}/*.npz")):
    stop = os.path.basename(f)[:-4]
    if stop.startswith("halves_") or not os.path.exists(f"{TRUTH}/{stop}.npz"): continue
    r = dict(np.load(f)); t = np.load(f"{TRUTH}/{stop}.npz")
    X, H, C, M, rocks = rz.stop_arrays(r, t, E.detect, PX)
    if M.sum() < 2000: continue
    n = X.shape[1]; k = 5.0 / PX; z5 = t["z"].astype(float)
    zt = ndi.zoom(np.nan_to_num(z5), k, order=1); vt = ndi.zoom(np.isfinite(z5).astype(float), k, order=1) > 0.99
    o = (n - zt.shape[0]) // 2
    cm = np.zeros((n, n), bool); cm[o:o + zt.shape[0], o:o + zt.shape[0]] = clean_mask(np.where(vt, zt, 0.0), ndi.binary_erosion(vt, iterations=4))
    l0, s0 = gt.latlon_to_ortho(float(t["lat"]), float(t["lon"]))
    cell = (int(l0 * gt.ORTHO["scale"] // 200), int(s0 * gt.ORTHO["scale"] // 200))
    D.append(dict(stop=stop, X=X, H=H, C=C, M=M, rocks=rocks, clean=cm, cell=cell, r=r))
cells = sorted({d["cell"] for d in D}); rng.shuffle(cells)
fold_of = {c: i % 3 for i, c in enumerate(cells)}
print(f"{len(D)} stops with truth, {len(cells)} map cells, rocks >=25 cm in truth: {sum(len(d['rocks']) for d in D)}", flush=True)


def near(p, ds, tol=0.5): return any(np.hypot(p[0] - q[0], p[1] - q[1]) <= max(2.0, tol * max(p[2], q[2]) / PX) for q in ds)


def f1(dsets, thr, probs):
    tp = fp = fn = 0
    for d, pr in zip(dsets, probs):
        det = [p for p in rz.decode(pr, thr) if d["M"][p[0], p[1]]]
        tp += sum(near(p, det) for p in d["rocks"]); fn += sum(not near(p, det) for p in d["rocks"]); fp += sum(not near(p, d["rocks"], 1.0) for p in det)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def batch(ds, bs=16, s=128):
    xb, hb, cb, mb = [], [], [], []
    while len(xb) < bs:
        d = ds[rng.integers(len(ds))]; ys, xs = np.nonzero(d["M"])
        i = rng.integers(len(ys)); y0 = int(np.clip(ys[i] - s // 2, 0, d["X"].shape[1] - s)); x0 = int(np.clip(xs[i] - s // 2, 0, d["X"].shape[2] - s))
        sl = (slice(y0, y0 + s), slice(x0, x0 + s))
        x, h, c, m = d["X"][:, sl[0], sl[1]], d["H"][:, sl[0], sl[1]], d["C"][sl], d["M"][sl]
        if rng.random() < 0.5: x, h, c, m = x[:, :, ::-1], h[:, :, ::-1], c[:, ::-1], m[:, ::-1]   # E-W flip only: sun is from the south
        xb.append(x.copy()); hb.append(h.copy()); cb.append(c.copy()); mb.append(m.copy())
    return [torch.from_numpy(np.stack(a)) for a in (xb, hb, cb, mb)]


oof = {}
for fold in range(3):
    test = [d for d in D if fold_of[d["cell"]] == fold]; train_all = [d for d in D if fold_of[d["cell"]] != fold]
    if not test or not train_all: continue
    tc = sorted({d["cell"] for d in train_all}); vc = set(tc[::5])
    val = [d for d in train_all if d["cell"] in vc]; train = [d for d in train_all if d["cell"] not in vc] or train_all
    model = rz.UNet(); opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=ITERS)
    best, best_state, t0 = 1e9, None, time.time()
    for it in range(ITERS):
        model.train(); x, h, c, m = batch(train); loss = rz.loss_fn(model(x), h, c, m)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (it + 1) % 250 == 0 and val:
            model.eval(); vl = 0.0
            with torch.no_grad():
                for d in val:
                    xx = torch.from_numpy(d["X"][None]); vl += float(rz.loss_fn(model(xx), torch.from_numpy(d["H"][None]), torch.from_numpy(d["C"][None]), torch.from_numpy(d["M"][None])))
            vl /= len(val)
            if vl < best: best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"fold {fold} it {it + 1} train {float(loss):.4f} val {vl:.4f} [{time.time() - t0:.0f}s]", flush=True)
    if best_state: model.load_state_dict(best_state)
    vp = [rz.predict(model, d["X"])[0] for d in val] if val else [rz.predict(model, d["X"])[0] for d in train[:10]]
    grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    thr = max(grid, key=lambda t: f1(val or train[:10], t, vp))
    torch.save(model.state_dict(), f"{OUT}/recognizer_fold{fold}.pt")
    for d in test:
        pr, pc = rz.predict(model, d["X"]); oof[d["stop"]] = (pr, pc, thr)
        np.savez_compressed(f"{OUT}/pred_{d['stop']}.npz", rock_prob=pr.astype(np.float16), class_prob=pc.astype(np.float16), thr=thr, fold=fold)
    print(f"fold {fold}: {len(train)} train / {len(val)} val / {len(test)} test stops, threshold {thr}", flush=True)

# ---------------- UCL yardstick on out-of-fold predictions (clean, rover-verified ground)
BINS = {"25-40 cm": (25, 40), ">=50 cm": (50, 100)}
prods = {"raw HiRISE (blob detector)": lambda d: [(y, x, D_) for y, x, D_, _ in E.detect(d["r"]["bicubic"].astype(float)) if D_ >= 25],
         "multi-frame restoration (blob detector)": lambda d: [(y, x, D_) for y, x, D_, _ in E.detect(d["r"]["restored_multi"].astype(float)) if D_ >= 25],
         "recogniser (U-Net, out-of-fold)": lambda d: [(y, x, D_) for y, x, D_, _ in rz.decode(oof[d["stop"]][0], oof[d["stop"]][2])]}
rows = []
for d in D:
    if d["stop"] not in oof: continue
    m = d["clean"]; a = m.sum() * (PX / 100) ** 2
    if a < 2: continue
    truth = [p for p in d["rocks"] if m[p[0], p[1]]]
    row = dict(stop=d["stop"], area_m2=a, truth={b: sum(lo <= p[2] <= hi for p in truth) for b, (lo, hi) in BINS.items()})
    for name, fn in prods.items():
        det = [p for p in fn(d) if m[p[0], p[1]]]
        sh = [((p[0] + 40) % m.shape[0], (p[1] + 40) % m.shape[1], p[2]) for p in det]
        row[name] = dict(found={b: sum(near(p, det) for p in truth if lo <= p[2] <= hi) for b, (lo, hi) in BINS.items()},
                         chance={b: sum(near(p, sh) for p in truth if lo <= p[2] <= hi) for b, (lo, hi) in BINS.items()},
                         false=sum(not near(p, truth, 1.0) for p in det), n=len(det))
    rows.append(row)


def boot(num, den, B=4000, r=np.random.default_rng(1)):
    num, den = np.array(num, float), np.array(den, float)
    if den.sum() == 0: return None
    idx = r.integers(0, len(num), (B, len(num))); s = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-9)
    return [100 * num.sum() / den.sum(), 100 * np.percentile(s, 2.5), 100 * np.percentile(s, 97.5)]


def boot_diff(a, b, den, B=4000, r=np.random.default_rng(2)):
    a, b, den = (np.array(v, float) for v in (a, b, den))
    if den.sum() == 0: return None
    idx = r.integers(0, len(a), (B, len(a))); s = (a[idx].sum(1) - b[idx].sum(1)) / np.maximum(den[idx].sum(1), 1e-9)
    return [100 * (a.sum() - b.sum()) / den.sum(), 100 * np.percentile(s, 2.5), 100 * np.percentile(s, 97.5)]


area = sum(r["area_m2"] for r in rows); summ = {}
for name in prods:
    summ[name] = dict(false_per_100m2=100 * sum(r[name]["false"] for r in rows) / max(area, 1e-9))
    for b in BINS:
        den = [r["truth"][b] for r in rows]
        summ[name][b] = dict(rover_rocks=int(sum(den)), found_pct=boot([r[name]["found"][b] for r in rows], den), chance_pct=boot([r[name]["chance"][b] for r in rows], den))
cmp = {}
for b in BINS:
    den = [r["truth"][b] for r in rows]
    for name in list(prods)[1:]:
        cmp[f"{name} minus raw HiRISE, {b}"] = boot_diff([r[name]["found"][b] for r in rows], [r["raw HiRISE (blob detector)"]["found"][b] for r in rows], den)
json.dump(dict(stops=len(rows), clean_area_m2=area, threshold_clean=THR_CLEAN, summary=summ, paired_differences=cmp, per_stop=rows),
          open(f"{OUT}/ucl_yardstick_thr{THR_CLEAN}.json", "w"), indent=1)
print(f"\n{len(rows)} held-out stops, {area:.0f} m2 clean rover-verified ground")
for name, s in summ.items():
    print(f"{name:42s} false/100m2 {s['false_per_100m2']:5.1f} | " + " | ".join(
        f"{b} (n={s[b]['rover_rocks']}): " + (f"{s[b]['found_pct'][0]:.0f}% [{s[b]['found_pct'][1]:.0f}-{s[b]['found_pct'][2]:.0f}] chance {s[b]['chance_pct'][0]:.0f}%" if s[b]["found_pct"] else "n/a") for b in BINS))
for k, v in cmp.items(): print(f"  {k}: " + (f"{v[0]:+.0f} pts [{v[1]:+.0f}, {v[2]:+.0f}]" if v else "n/a"))

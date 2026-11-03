"""Fast checks of the paper pipeline components (Sections 1-8 of docs/METHODS.md). Runs in well under a minute on CPU."""
import math
import numpy as np
from scipy import ndimage as ndi
from pipeline import gotcha, pde_tv, uncertainty as U, confidence as cf, hazard as hz, shadow_evidence as se, grounded_fill as gf


def texture(n=96, seed=0):
    r = np.random.default_rng(seed); a = ndi.gaussian_filter(r.standard_normal((n, n)), 1.5)
    return (a - a.mean()) / a.std()


def test_gotcha_recovers_subpixel_shift():
    ref = texture(); dx, dy = 0.4, -0.3
    yy, xx = np.mgrid[0:96, 0:96].astype(float)
    img = ndi.map_coordinates(ref, [yy + dy, xx + dx], order=3, mode="reflect")        # img(x) = ref(x + d)
    r = gotcha.gotcha(ref, img, seeds=gotcha.seeds_phase(ref, img))
    c = (slice(20, -20), slice(20, -20))
    assert r is not None and abs(np.median(r["dx"][c]) - dx) < 0.05 and abs(np.median(r["dy"][c]) - dy) < 0.05


def _sim(n_lr=24, sc=4, seed=1):
    r = np.random.default_rng(seed); truth = ndi.zoom(texture(n_lr * 2, seed), 2, order=3)
    frames, flows = [], []
    for k in range(4):
        dx, dy = (0.0, 0.0) if k == 0 else r.uniform(-1, 1, 2)
        lr = ndi.gaussian_filter(ndi.shift(truth, (-sc * dy, -sc * dx), order=3, mode="reflect"), 1.0 * sc)[::sc, ::sc]
        frames.append(lr + r.normal(0, 0.02, lr.shape)); flows.append((np.full(lr.shape, dx), np.full(lr.shape, dy)))
    return truth, frames, flows


def test_pde_tv_beats_bicubic_and_weights_reject_change():
    truth, frames, flows = _sim(); frames[2] = frames[2].copy(); frames[2][10:13, 10:13] += 5.0
    X, info = pde_tv.pde_tv(frames, flows, scale=4, psf_sigma=4.0, lam=0.005, iters=80)
    bic = ndi.zoom(frames[0], 4, order=3, grid_mode=True, mode="nearest"); c = (slice(16, -16), slice(16, -16))
    g = lambda a: ndi.gaussian_filter(a, 4)
    assert np.sqrt(np.mean((g(X) - g(truth))[c] ** 2)) < np.sqrt(np.mean((g(bic) - g(truth))[c] ** 2))
    assert info["pixel_weights"][2][11, 11] < 0.2                                     # the changed pixel is masked


def test_bootstrap_shapes_and_coverage_helpers():
    truth, frames, flows = _sim()
    b = U.bootstrap(frames, flows, 4.0, B=3, scale=4, lam=0.005, iters=20)
    assert b["members"].shape[0] == 3 and b["std"].shape == truth.shape and np.all(b["std"] >= 0)
    e = np.random.default_rng(0).standard_normal(10000); s = np.ones_like(e)
    assert abs(U.coverage(e, s)[1.0] - 0.683) < 0.02 and abs(U.fit_kappa(e, s) - 1.0) < 0.05


def test_shadow_falls_opposite_the_sun():
    t = se.template(100, 180.0, 50.0, 0.8)                      # sun azimuth 180 deg (clockwise from image-right) = image-left
    yy, xx = np.mgrid[0:t.shape[0], 0:t.shape[1]] - t.shape[0] // 2
    dark = t < -0.5 * abs(t.min())
    assert xx[dark].mean() > 0                                  # shadow on the right, away from the sun


def test_lock_keeps_the_measurement():
    base = ndi.zoom(texture(32), 4, order=3); d = texture(128, 3) * 0.5
    out = base + gf.lock(d, 4.0)
    ob = gf.observe(base, 4.0)
    assert np.std(gf.observe(out, 4.0) - ob) / np.std(ob) < 5e-3                         # regularised projection: ~0.1 %


def test_confidence_and_hazard():
    X = np.zeros((128, 128)); yy, xx = np.mgrid[0:128, 0:128]
    X += 5 * np.exp(-((yy - 64) ** 2 + (xx - 64) ** 2) / (2 * 3.0 ** 2)); X += np.random.default_rng(0).normal(0, 0.1, X.shape)
    c = cf.blob_candidates(X); assert any(abs(y - 64) <= 2 and abs(x - 64) <= 2 for y, x, _, _ in c)
    cal = dict(coef=[1.0, 0.0, 0.0, 0.0], intercept=-3.0, features=cf.FEATURE_NAMES)
    p = cf.predict(cal, cf.features(c, None)); assert np.all((p >= 0) & (p <= 1))
    H = hz.cells(X.shape, [(64, 64, 60, 0.7), (64, 66, 60, 0.5), (10, 10, 25, 0.9)], np.ones(X.shape, bool), d_hazard=40)
    k = int(round(1.0 / 0.0625)); ph = H["p_hazard"][64 // k, 64 // k]
    assert math.isclose(ph, 1 - 0.3 * 0.5, rel_tol=1e-6) and H["p_hazard"][0, 0] == 0          # 25 cm rock is below the hazard size

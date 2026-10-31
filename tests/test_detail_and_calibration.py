import numpy as np
from scipy import ndimage as ndi
from pipeline import blur_calib as bc, detail as dt, scoring as sc, synth


def _ground(n=320, seed=1):
    # rover-like ground at 5 cm: fractal relief + scattered pebbles
    g = synth.fractal_terrain(n, seed=seed, octaves=8, persistence=0.75)
    rng = np.random.default_rng(seed)
    for _ in range(400):
        y, x = rng.integers(0, n, 2); r = rng.integers(1, 4)
        g[max(0, y - r):y + r, max(0, x - r):x + r] += rng.normal(0.3, 0.1)
    return g


def test_spectral_calibration_recovers_known_blur():
    true_sigma = 0.8
    est = []
    for seed in range(4):
        g = _ground(seed=seed)
        hirise = bc.degrade(g, 0.05, true_sigma) + np.random.default_rng(seed).normal(0, 1e-4, (63, 63))[:bc.degrade(g, 0.05, 0).shape[0], :bc.degrade(g, 0.05, 0).shape[1]]
        ref = bc.box_to_hirise(g, 0.05)
        f, ph = bc.radial_psd(hirise); _, pr = bc.radial_psd(ref)
        ratio = ph / pr; ratio /= np.median(ratio[f < 0.08])
        est.append(bc.fit_sigma(f, ratio))
    print("estimated sigmas", np.round(est, 2))
    assert abs(np.median(est) - true_sigma) < 0.15


def test_detail_layer_keeps_the_measurement():
    n = 64; psf_hr = 0.78 * 4
    base = ndi.zoom(synth.fractal_terrain(n, seed=3), 4, order=3, grid_mode=True, mode="nearest")
    tiles = {k: [np.random.default_rng(k * 10 + i).standard_normal((64, 64)).astype(np.float32) for i in range(4)] for k in range(4)}
    probs = np.stack([np.full(base.shape, 0.25)] * 4)
    out, det = dt.synthesize(base, probs, tiles, {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}, psf_hr)
    measured = base[::4, ::4]
    c_out = sc.consistency(out, ndi.uniform_filter(ndi.gaussian_filter(base, psf_hr), 4)[2::4, 2::4], 4, psf_hr)
    print("detail rms", det.std(), "consistency %", c_out)
    assert det.std() > 0
    assert c_out > 90


def test_scores_are_100_for_identical_and_lower_for_blurred():
    t = _ground(256)
    assert abs(sc.spectral_match(t, t, (0.06, 0.45), (0.012, 0.03)) - 100) < 1e-6
    assert abs(sc.texture_match(t, t, 3.0) - 100) < 1e-6
    b = ndi.gaussian_filter(t, 3)
    assert sc.spectral_match(b, t, (0.06, 0.45), (0.012, 0.03)) < 50
    assert sc.texture_match(b, t, 3.0) < 90

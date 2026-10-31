import numpy as np
from pipeline import synth, stereo


def _pair(k=0.8, relief_m=12.0):
    height = synth.fractal_terrain(256, seed=5, octaves=4) * relief_m   # smooth relief, metres
    tex = synth.fractal_terrain(256, seed=9, octaves=6, persistence=0.8) # rich albedo texture
    left, right, d = synth.make_stereo_pair(height, tex, px_per_metre=k)
    return height, left, right, k


def test_stereo_recovers_height_with_honest_mask():
    height, left, right, k = _pair()
    disp, valid = stereo.disparity(left, right, max_disp=16)
    h = stereo.height_from_disparity(disp, valid, k)
    m = np.zeros_like(valid); m[24:-24, 40:-24] = True   # ignore border where matching is undefined
    use = valid & m
    frac = use.sum() / m.sum()
    rmse = float(np.sqrt(np.nanmean((h[use] - height[use]) ** 2)))
    print(f"valid fraction {frac:.2f}  height RMSE {rmse:.3f} m  (relief 12 m)")
    assert frac > 0.6
    assert rmse < 0.6

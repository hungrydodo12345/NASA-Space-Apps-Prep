import numpy as np
from scipy import ndimage as ndi
from pipeline import synth, register, fuse, stereo, export

K = 0.8  # parallax px per metre (in HR pixels)


def _fused(hr_img, n=8, seed=0):
    frames, _ = synth.make_observations(hr_img, scale=2, n_frames=n, noise=0.01, seed=seed)
    est = register.register_stack(frames)
    f, _ = fuse.fuse(frames, est, scale=2, iters=10)
    bic = ndi.zoom(frames[0], 2, order=3, grid_mode=True, mode="nearest")
    return f.astype(np.float32), bic.astype(np.float32)


def test_fused_stereo_beats_single_frame_and_exports(tmp_path):
    height = synth.fractal_terrain(256, seed=5, octaves=4) * 12.0
    tex = synth.fractal_terrain(256, seed=9, octaves=6, persistence=0.8)
    L, R, _ = synth.make_stereo_pair(height, tex, px_per_metre=K, noise=0.0)
    Lf, Lb = _fused(L, seed=1)
    Rf, Rb = _fused(R, seed=2)
    m = np.zeros(L.shape, bool); m[24:-24, 40:-24] = True

    def run(l, r):
        d, v = stereo.disparity(l, r, max_disp=16)
        h = stereo.height_from_disparity(d, v, K)
        use = v & m
        return float(np.sqrt(np.nanmean((h[use] - height[use]) ** 2))), float(use.sum() / m.sum()), h, v

    rm_f, vf_f, h_f, v_f = run(Lf, Rf)
    rm_b, vf_b, _, _ = run(Lb, Rb)
    print(f"single-frame bicubic: RMSE {rm_b:.3f} m valid {vf_b:.2f} | fused: RMSE {rm_f:.3f} m valid {vf_f:.2f}")
    assert rm_f < rm_b

    filled, inferred = export.fill_gaps(h_f, v_f)
    emin, emax = float(np.floor(height.min() - 1)), float(np.ceil(height.max() + 1))
    export.write_png(tmp_path / "dem.png", export.pack_elevation(filled, emin, emax))
    export.write_png(tmp_path / "conf.png", (v_f * 255).astype(np.uint8))
    back = export.unpack_elevation(export.read_png(tmp_path / "dem.png"), emin, emax)
    assert np.abs(back - filled).max() <= (emax - emin) / 4095 / 2 + 1e-9

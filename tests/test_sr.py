import numpy as np
from scipy import ndimage as ndi
from pipeline import synth, register, fuse

C = slice(16, -16)


def psnr(a, b):
    return 10 * np.log10(1.0 / ((a - b) ** 2).mean())


def _scene(n_frames=8):
    hr = synth.fractal_terrain(256, seed=3, persistence=0.8)  # energy near Nyquist
    frames, shifts = synth.make_observations(hr, scale=2, n_frames=n_frames, noise=0.003)
    return hr, frames, shifts


def test_register_does_not_mutate_inputs():
    _, frames, _ = _scene(4)
    before = [f.copy() for f in frames]
    register.register_stack(frames)
    assert all(np.array_equal(a, b) for a, b in zip(before, frames))


def test_registration_subpixel():
    _, frames, shifts = _scene(6)
    est = register.register_stack(frames)
    err = max(max(abs(e[0] - t[0]), abs(e[1] - t[1])) for e, t in zip(est, shifts))
    print("max reg error px:", err)
    assert err < 0.05


def test_fusion_beats_interpolation():
    hr, frames, _ = _scene(8)
    est = register.register_stack(frames)
    fused, _ = fuse.fuse(frames, est, scale=2, iters=10)
    bicubic = ndi.zoom(frames[0], 2, order=3, grid_mode=True, mode="nearest")
    p_f, p_b = psnr(fused[C, C], hr[C, C]), psnr(bicubic[C, C], hr[C, C])
    print(f"PSNR bicubic={p_b:.2f} dB  fused={p_f:.2f} dB")
    assert p_b > 20, "baseline suspiciously low - test setup is broken"
    assert p_f > p_b + 2.0

import warnings
import numpy as np
from scipy import ndimage as ndi
from pipeline import synth, register, fuse_cg, metrics

warnings.filterwarnings("ignore")
PX_CM = 25.0          # native HiRISE sampling
SCALE = 4             # -> 6.25 cm/px
C = slice(24, -24)


def _run(n_frames, noise=0.005):
    hr = synth.fractal_terrain(240, seed=3, octaves=7, persistence=0.85)
    sig = 0.4 * SCALE
    frames, _ = synth.make_observations(hr, scale=SCALE, n_frames=n_frames, psf_sigma=sig, noise=noise, max_shift=4.0)
    est = register.register_stack(frames)
    f, _ = fuse_cg.fuse_cg(frames, est, scale=SCALE, psf_sigma=sig, mu=1e-2, iters=60)
    bic = ndi.zoom(frames[0], SCALE, order=3, grid_mode=True, mode="nearest")
    cf = metrics.effective_cutoff(metrics.radial_coherence(f[C, C], hr[C, C]))
    cb = metrics.effective_cutoff(metrics.radial_coherence(bic[C, C], hr[C, C]))
    return cf, cb


def feature_cm(cutoff):
    """Smallest resolvable feature (half-period) in cm at the output sampling."""
    out_px = PX_CM / SCALE
    return out_px / cutoff


def test_scale4_resolves_detail_well_below_native_pixel():
    cf, cb = _run(25)
    print(f"cutoff fused {cf:.2f} -> {feature_cm(cf):.1f} cm features | bicubic {cb:.2f} -> {feature_cm(cb):.1f} cm")
    assert cf >= 0.4 and cf > 2 * cb
    # Measured: ~14 cm with 25 frames @ noise 0.005 (11 cm at noise 1e-3). The 5-10 cm goal is NOT
    # reached here; the ceiling is set by the camera PSF and noise, not by the output sampling.
    assert feature_cm(cf) < 16.0

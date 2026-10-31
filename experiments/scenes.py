"""Ground-truth scenes for simulation: fractal terrain, and REAL Perseverance ground texture (semi-real)."""
import glob, os
import numpy as np
from scipy import ndimage as ndi
from pipeline import synth
from . import common as C


def fractal(seed, n=160):
    return synth.fractal_terrain(n, seed=seed % 997, octaves=7, persistence=0.8), None


def load_rover_scenes(truth_dir, n=160, min_valid=0.45):
    """Real rover ground mosaics resampled to the output grid (6.31 cm). Returns [(name, truth, valid)].

    Holes (unviewed ground) are filled with a smooth local mean so the imaging model can run; only pixels whose
    whole PSF neighbourhood is real are scored (valid mask, eroded by 12 px).
    """
    k = 0.05 / (C.OUT_PX_CM / 100.0)
    scenes = []
    for f in sorted(glob.glob(os.path.join(truth_dir, "*.npz"))):
        z = np.load(f)["z"].astype(float)
        v = np.isfinite(z); a = np.nan_to_num(z)
        zz = ndi.zoom(a, k, order=1); vv = ndi.zoom(v.astype(float), k, order=1) > 0.99
        m = vv.astype(float)
        fill = ndi.gaussian_filter(zz * m, 12) / np.maximum(ndi.gaussian_filter(m, 12), 1e-6)
        full = np.where(vv, zz, fill)
        H, W = full.shape; c = H // 2
        sl = (slice(c - n // 2, c - n // 2 + n), slice(c - n // 2, c - n // 2 + n))
        t, val = full[sl], ndi.binary_erosion(vv[sl], iterations=12)
        if val.mean() < min_valid: continue
        t = (t - t.mean()) / (t.std() + 1e-9); t = 0.5 + 0.12 * t           # reflectance-like range
        scenes.append((os.path.basename(f)[:-4], t, val))
    return scenes

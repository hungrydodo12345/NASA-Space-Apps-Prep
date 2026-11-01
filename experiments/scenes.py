"""Ground-truth scenes for simulation: fractal terrain, and REAL Perseverance ground texture (semi-real)."""
import glob, os
import numpy as np
from scipy import ndimage as ndi
from pipeline import synth
from . import common as C


def fractal(seed, n=160):
    return synth.fractal_terrain(n, seed=seed % 997, octaves=7, persistence=0.8), None


def load_rover_scenes(truth_dir, n=160, min_valid=0.25):
    """Real rover ground mosaics resampled to the output grid (6.31 cm). Returns [(name, truth, valid)].

    The rover cannot see the ground under itself, so the window is slid to the best-covered position. Holes (unviewed
    ground) are filled with a smooth local mean so the imaging model can run; only pixels whose whole PSF neighbourhood
    is real are scored (valid mask, eroded by 12 px).
    """
    k = 0.05 / (C.OUT_PX_CM / 100.0)
    scenes = []
    for f in sorted(glob.glob(os.path.join(truth_dir, "*.npz"))):
        d = np.load(f); z = d["z"].astype(float); v = np.isfinite(z)
        zz = ndi.zoom(np.nan_to_num(z), k, order=1); vv = ndi.zoom(v.astype(float), k, order=1) > 0.99
        m = vv.astype(float)
        fill = ndi.gaussian_filter(zz * m, 12) / np.maximum(ndi.gaussian_filter(m, 12), 1e-6)
        full = np.where(vv, zz, fill)
        H, W = full.shape
        if H < n or W < n: continue
        best = None
        for y in range(0, H - n + 1, 16):
            for x in range(0, W - n + 1, 16):
                val = ndi.binary_erosion(vv[y:y + n, x:x + n], iterations=12)
                if best is None or val.mean() > best[0]: best = (val.mean(), y, x, val)
        if best[0] < min_valid: continue
        _, y, x, val = best
        t = full[y:y + n, x:x + n]
        t = (t - t[val].mean()) / (t[val].std() + 1e-9); t = 0.5 + 0.12 * t          # reflectance-like range
        scenes.append((os.path.basename(f)[:-4], t, val))
    return scenes

"""Figure: raw HiRISE vs standard AI upscaler vs our pipeline vs the rover's own photo of the same ground.
Usage: python scripts/make_comparison.py <figdump_dir> <out_png> [n_rows]
Each panel is contrast-stretched on its own (1-99 percentile); the rover panel is the real top-down Navcam mosaic.
"""
import sys, glob, os
import numpy as np, cv2
D, OUT = sys.argv[1:3]; NR = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rows = []
for f in sorted(glob.glob(f"{D}/*.npz")):
    d = np.load(f); t = d["truth"]
    rows.append((np.isfinite(t).mean(), f, d))
rows.sort(key=lambda r: -r[0])
pick = rows[:NR]
PX = 6.31                                         # cm per pixel on the output grid
TITLES = ["RAW HiRISE (25 cm pixels, as downloaded)", "Standard AI upscaler (EDSR x4)", "OUR PIPELINE (6 fused passes + rover-texture detail)", "REAL GROUND (rover's own photos, top-down)"]
def stretch(a, ref=None):
    v = np.isfinite(a); lo, hi = np.percentile(a[v], [1, 99]); g = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo + 1e-12), 0, 1)
    out = (g * 255).astype(np.uint8); return out, v
panels = []
for cov, f, d in pick:
    n = d["truth"].shape[0]; tiles = []
    for key in ("raw", "edsr", "ours", "truth"):
        img = d[key].astype(float)
        if key == "raw": img = img                      # native pixels, nearest-neighbour blocks (no smoothing)
        g, v = stretch(img)
        g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if key == "truth": g[~v] = (60, 60, 60)
        tiles.append(cv2.resize(g, (420, 420), interpolation=cv2.INTER_NEAREST if key == "raw" else cv2.INTER_AREA))
    px = 420 / n; bar = int(round(200 / PX * px))        # 2 m scale bar
    for t in tiles: cv2.rectangle(t, (12, 396), (12 + bar, 402), (255, 255, 255), -1); cv2.putText(t, "2 m", (12, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    panels.append(np.hstack([np.pad(t, ((0, 0), (0, 6), (0, 0)), constant_values=18) for t in tiles]))
W = panels[0].shape[1]; head = np.full((74, W, 3), 18, np.uint8)
for i, t in enumerate(TITLES):
    x = i * 426 + 8; words = t.split(" (")
    cv2.putText(head, words[0], (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (232, 210, 184), 1, cv2.LINE_AA)
    if len(words) > 1: cv2.putText(head, "(" + words[1], (x, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 150, 140), 1, cv2.LINE_AA)
img = np.vstack([head] + [np.vstack([p, np.full((6, W, 3), 18, np.uint8)]) for p in panels])
cv2.imwrite(OUT, img); print("wrote", OUT, img.shape, "stops:", [os.path.basename(p[1])[:-4] for p in pick])

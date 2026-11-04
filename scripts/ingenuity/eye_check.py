"""Eye check of the Ingenuity placement: the mosaic laid over HiRISE at the validated parameters and at deliberately wrong ones (rotation +-30 deg, 1.5 m shift),
on two independent HiRISE passes. Each tile: truth blurred to the pass's PSF (red) over the HiRISE pass (cyan); the first row also shows the truth at native resolution.
Usage: python eye_check.py <scratch_dir> <mosaic_npz> <bx> <by> <theta_ccw> <scale> <dx_m> <dy_m> <out_png>"""
import sys, json, numpy as np, cv2
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep"); from pipeline.hirise_io import Rdr
from PIL import Image, ImageDraw, ImageFont
S, MOS, BX, BY, TH, SC, DX, DY, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), sys.argv[9]
NW, Z, PXH = 100, 9, 0.2524; site = json.load(open(f"{S}/debris_site/summary.json")); bl, bs = site["backshell_global"]; B = json.load(open("/home/user/NASA-Space-Apps-Prep/data_products/pass_blur.json"))
m = np.load(MOS); M = np.nan_to_num(m["mosaic"], nan=1.0).astype(np.float32); V = np.isfinite(m["mosaic"]); font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
def hirise(pid):
    lk = site["locks"].get(pid, dict(dx=0, dy=0)); I = Rdr(f"{S}/hirise_rdr/{pid}_RED").read_if(bl - NW // 2 - lk["dy"], bs - NW // 2 - lk["dx"], NW, NW); I = np.nan_to_num(I, nan=float(np.nanmedian(I))).astype(np.float32)
    return cv2.resize(I, None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC), B[pid]["sigma_px"]
def place(th, sc, dx, dy, sigma=None):
    f = Z * sc / PXH; c, s = np.cos(np.radians(th)), np.sin(np.radians(th)); Rm = np.array([[f * c, f * s], [-f * s, f * c]]); t = np.array([NW * Z / 2 + dx / PXH * Z, NW * Z / 2 + dy / PXH * Z]) - Rm @ np.array([BX, BY]); W = np.hstack([Rm, t[:, None]])
    a = cv2.warpAffine(M, W, (NW * Z, NW * Z), flags=cv2.INTER_AREA); v = cv2.warpAffine(V.astype(np.float32), W, (NW * Z, NW * Z)) > 0.9
    if sigma: a = cv2.GaussianBlur(np.where(v, a, a[v].mean()), (0, 0), sigma * Z)
    return a, v
def norm(a, mask=None): x = a[mask] if mask is not None and mask.any() else a; lo, hi = np.percentile(x, [1, 99.5]); return (np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)
variants = [("best placement", TH, SC, DX, DY), (f"rotated {TH + 30:.0f} deg (wrong)", TH + 30, SC, DX, DY), (f"rotated {TH - 30:.0f} deg (wrong)", TH - 30, SC, DX, DY), ("shifted 1.5 m (wrong)", TH, SC, DX + 1.5, DY)]
tiles = []; hdr = []
for pid in ("ESP_068294_1985", "ESP_068426_1985"):
    H, sg = hirise(pid); hn = norm(H - cv2.GaussianBlur(H, (0, 0), 6 * Z)); row = [(f"{pid[:10]} HiRISE (high-passed)", np.stack([hn] * 3, -1))]
    for lab, th, sc, dx, dy in variants:
        a, v = place(th, sc, dx, dy, sg); a = a - cv2.GaussianBlur(a, (0, 0), 6 * Z); an = norm(a, v); ov = np.stack([an, hn, hn], -1); ov[~v] = np.stack([hn] * 3, -1)[~v]; row.append((f"{pid[:10]}: {lab}", ov))
    tiles.append(row)
a, v = place(TH, SC, DX, DY); an = norm(a, v); an[~v] = 25; tiles[0].insert(1, ("Ingenuity truth (3 cm), placed", np.stack([an] * 3, -1)))
cols = max(len(r) for r in tiles); w = NW * Z; can = Image.new("RGB", (cols * (w + 6), 2 * (w + 34)), (20, 20, 20)); d = ImageDraw.Draw(can)
for r, row in enumerate(tiles):
    for c, (lab, im) in enumerate(row): can.paste(Image.fromarray(im), (c * (w + 6), r * (w + 34) + 30)); d.text((c * (w + 6) + 4, r * (w + 34) + 4), lab, font=font, fill=(255, 255, 0))
L = 5 / PXH * Z; d.line([(14, 2 * (w + 34) - 14), (14 + L, 2 * (w + 34) - 14)], fill=(255, 255, 0), width=4); d.text((20 + L, 2 * (w + 34) - 32), "5 m", font=font, fill=(255, 255, 0)); can.save(OUT); print(can.size)

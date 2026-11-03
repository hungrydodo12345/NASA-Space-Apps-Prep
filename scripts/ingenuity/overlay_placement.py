import sys, json, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
S = sys.argv[1]; scale = float(sys.argv[2]); theta = float(sys.argv[3]); flip = int(sys.argv[4]); out = sys.argv[5]
d = np.load(f"{S}/chain_debris/chain.npz"); fr = d["frames"][0].astype(np.float32); H0 = cv2.resize(fr, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
m = np.load(f"{S}/ingenuity/mosaic_a.npz")["mosaic"].astype(np.float32); V0 = np.isfinite(m); M = np.nan_to_num(m, nan=float(np.nanmedian(m)))
anchor_h = (259, 257)                                         # (x, y) HiRISE backshell centre in hr px: mid-point of the two bright halves
anchor_m = (131, 336) if not flip else (M.shape[1] - 1 - 131, 336)
A = M[:, ::-1] if flip else M; V = V0[:, ::-1] if flip else V0
r = scale / 0.0631; c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta)); Rm = np.array([[r * c, -r * s], [r * s, r * c]])
t = np.array(anchor_h) - Rm @ np.array(anchor_m); W = np.hstack([Rm, t[:, None]])
a = cv2.warpAffine(A, W, H0.shape[::-1], flags=cv2.INTER_AREA); v = cv2.warpAffine(V.astype(np.float32), W, H0.shape[::-1]) > 0.9
a_sharp = a.copy(); a_bl = cv2.GaussianBlur(a, (0, 0), 3.6)
def u8(x, mask=None, lo=None, hi=None):
    y = x[mask] if mask is not None and mask.any() else x; lo, hi = np.percentile(y, [1, 99.7]); return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
Hn = u8(H0); An = u8(a_sharp, v); Bn = u8(a_bl, v); An[~v] = 20; Bn[~v] = 20
ov = np.stack([Bn, Hn, Hn], -1); ov[~v] = np.stack([Hn, Hn, Hn], -1)[~v]
P = [("HiRISE ESP_068294 (6.3 cm px, bicubic)", np.stack([Hn] * 3, -1)), ("Ingenuity mosaic placed (full detail)", np.stack([An] * 3, -1)), ("overlay: Ingenuity (red, at HiRISE blur) on HiRISE", ov)]
f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18); w = H0.shape[1]; can = Image.new("RGB", (3 * w + 40, H0.shape[0] + 50), (20, 20, 20)); dr = ImageDraw.Draw(can)
for i, (lab, im) in enumerate(P):
    can.paste(Image.fromarray(im), (10 + i * (w + 10), 40)); dr.text((10 + i * (w + 10), 10), lab, font=f, fill=(255, 255, 0))
    L = 5 / 0.0631; dr.line([(20 + i * (w + 10), H0.shape[0] + 20), (20 + i * (w + 10) + L, H0.shape[0] + 20)], fill=(255, 255, 0), width=4); dr.text((30 + i * (w + 10) + L, H0.shape[0] + 8), "5 m", font=f, fill=(255, 255, 0))
can.save(out); print(can.size, f"scale {scale} theta {theta} flip {flip}")

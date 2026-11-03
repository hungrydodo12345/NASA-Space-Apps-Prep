"""Native-resolution crop of the parachute/backshell field in a HiRISE pass (shown 4x, nearest neighbour). Usage: python crop_debris.py <scratch_dir> <observation_id>"""
import sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/home/user/NASA-Space-Apps-Prep")
from pipeline.hirise_io import Rdr
S = sys.argv[1]; oid = sys.argv[2]; g0l, g0s = -4371755.17190706, -23476249.22764247; cl, cs = g0l + 64, g0s + 64; k = 4
cy, cx = 1310, 423                                                                    # cluster centre in the 4x-downsampled 12 km window of 068294
gl = cl - 6000 + cy * k - 160; gs = cs - 6000 + cx * k - 160
R = Rdr(f"{S}/hirise_rdr/{oid}_RED"); w = R.read_if(gl, gs, 320, 320); w = np.nan_to_num(w, nan=np.nanmedian(w))
lo, hi = np.percentile(w, [0.5, 99.8]); im = (np.clip((w - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
img = Image.fromarray(im).resize((1280, 1280), Image.NEAREST).convert("RGB"); d = ImageDraw.Draw(img); f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
d.text((10, 8), f"{oid}: 80 m window, 25 cm pixels shown 4x (nearest), north up", font=f, fill=(255, 255, 0))
L = 10 / 0.2524 * 4; d.line([(20, 1250), (20 + L, 1250)], fill=(255, 255, 0), width=5); d.text((20, 1215), "10 m", font=f, fill=(255, 255, 0))
img.save(f"{S}/eye/debris_{oid}.png"); print(img.size)

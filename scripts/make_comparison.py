"""Labelled side-by-side comparison images at true resolution (lossless PNG, no downscaling, no smoothing).

For a chain tile (scripts/run_chain.py) it restores the same passes with:
  1. Raw HiRISE, the reference pass at its native 25 cm. Each HiRISE pixel is shown as a 4 x 4 block
     (nearest-neighbour), so its real pixels are visible rather than interpolated.
  2. The UCL pipeline (our re-implementation of Tao & Muller 2016: Gotcha + unweighted PDE-TV, assumed blur 1 px).
  3. UCL + our measured per-pass blur.
  4. This work (Gotcha + noise-weighted, change-masked PDE-TV + measured per-pass blur).
  5. Optional: this work + confidence-scored fill at threshold t (pass --fill-png from scripts/confidence_fill.py).
Every restoration is shown at its own 6.25 cm pixel, 1 image pixel = 1 screen pixel. One contrast stretch (0.5-99.5 %
of the raw pass) is used for every panel, so brightness differences are real. A zoom row repeats a central crop, with
each pixel enlarged by an integer factor (nearest-neighbour).
Outputs: <out_dir>/<name>_comparison.png, <out_dir>/<name>_zoom.png, and one 16-bit PNG per panel (<name>_<panel>.png).
Usage: python scripts/make_comparison.py <chain_dir> <out_dir> <name> [--title T] [--zoom-m 8] [--zoom-factor 4] [--fill-png P]
"""
import sys, os, json, argparse, numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from pipeline import pde_tv

ap = argparse.ArgumentParser(); ap.add_argument("chain"); ap.add_argument("out"); ap.add_argument("name")
ap.add_argument("--title", default=""); ap.add_argument("--zoom-m", type=float, default=8.0); ap.add_argument("--zoom-factor", type=int, default=4)
ap.add_argument("--fill-png"); ap.add_argument("--fill-label", default="This work + confidence fill (P ≥ 0.5)")
ap.add_argument("--iters", type=int, default=300)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
d = np.load(f"{a.chain}/chain.npz"); meta = json.load(open(f"{a.chain}/chain.json")); SC = meta["scale"]
blur = json.load(open(os.path.join(ROOT, "data_products", "pass_blur.json")))
frames = list(d["frames"].astype(float)); flows = list(zip(d["flows_dx"].astype(float), d["flows_dy"].astype(float)))
used = meta["passes_used"]; sig = [blur[o]["sigma_px"] * SC for o in used]; LAM = 0.005

panels = [("raw", f"Raw HiRISE {used[0].replace('_1985', '')}\n1 pass, native 25 cm pixels (shown as 4×4 blocks)", np.kron(frames[0], np.ones((SC, SC))))]
panels.append(("ucl", "UCL pipeline\n(our re-implementation: Gotcha + PDE-TV, assumed blur)",
               pde_tv.pde_tv(frames, flows, scale=SC, psf_sigma=1.0 * SC, lam=LAM, iters=a.iters, weighted=False)[0]))
panels.append(("ucl_blur", "UCL + our measured per-pass blur", pde_tv.pde_tv(frames, flows, scale=SC, psf_sigma=sig, lam=LAM, iters=a.iters, weighted=False)[0]))
panels.append(("ours", "This work\n(noise-weighted, change-masked, measured blur)", pde_tv.pde_tv(frames, flows, scale=SC, psf_sigma=sig, lam=LAM, iters=a.iters)[0]))
if a.fill_png: panels.append(("fill", a.fill_label + "\n(added detail is simulated, locked to HiRISE)", None))

lo, hi = np.percentile(frames[0], [0.5, 99.5])
to8 = lambda x: (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
H, W = panels[1][2].shape
imgs = []
for key, label, X in panels:
    if X is None:
        im = np.array(Image.open(a.fill_png).convert("L"))
        if im.shape != (H, W): raise SystemExit(f"fill image is {im.shape}, expected {(H, W)}")
    else:
        X = X[:H, :W]; im = to8(X)
        Image.fromarray((np.clip((X - lo) / (hi - lo), 0, 1) * 65535).astype(np.uint16)).save(f"{a.out}/{a.name}_{key}_16bit.png")
    imgs.append((label, im))

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
px_m = 0.25 / SC


def sheet(tiles, title, foot, scale_px_per_m):
    gap, lab, top, bot = 12, 48, 40 if title else 8, 34
    th, tw = tiles[0][1].shape
    out = Image.new("RGB", (len(tiles) * tw + (len(tiles) + 1) * gap, top + lab + th + bot + gap), (18, 18, 18)); dr = ImageDraw.Draw(out)
    if title: dr.text((gap, 10), title, font=bold, fill=(240, 240, 240))
    for i, (label, im) in enumerate(tiles):
        x0 = gap + i * (tw + gap)
        dr.multiline_text((x0, top), label, font=font, fill=(230, 230, 230), spacing=3)
        out.paste(Image.fromarray(im).convert("RGB"), (x0, top + lab))
        L = int(round(scale_px_per_m)); yb = top + lab + th - 14                       # 1 m scale bar
        dr.rectangle([x0 + 10, yb, x0 + 10 + L, yb + 5], fill=(255, 255, 255)); dr.text((x0 + 14 + L, yb - 6), "1 m", font=font, fill=(255, 255, 255))
    dr.text((gap, top + lab + th + 8), foot, font=font, fill=(170, 170, 170))
    return out


foot = (f"{used[0][:3]} tile {H * px_m:.0f} m wide, {len(used)} passes ({', '.join(u.replace('_1985', '') for u in used)}). "
        "Every panel: 1 image pixel = 6.25 cm, shown 1:1; one shared contrast stretch; lossless PNG.")
sheet(imgs, a.title, foot, 1 / px_m).save(f"{a.out}/{a.name}_comparison.png", optimize=True)
z = int(round(a.zoom_m / px_m)); cy, cx = H // 2, W // 2; f = a.zoom_factor
zt = [(lb.replace("shown as 4×4 blocks", f"shown as {f * SC}×{f * SC} blocks"), np.kron(im[cy - z // 2:cy + z // 2, cx - z // 2:cx + z // 2], np.ones((f, f), np.uint8))) for lb, im in imgs]
sheet(zt, (a.title + f" — central {a.zoom_m:.0f} m, enlarged {f}× (nearest-neighbour, no smoothing)").strip(" —"),
      f"Each 6.25 cm pixel is shown as a {f}×{f} block (raw HiRISE 25 cm pixels as {f * SC}×{f * SC}); nothing is interpolated.", f / px_m).save(f"{a.out}/{a.name}_zoom.png", optimize=True)
print("wrote", f"{a.out}/{a.name}_comparison.png", f"{a.out}/{a.name}_zoom.png")

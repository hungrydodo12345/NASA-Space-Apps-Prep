"""One large, zoomable poster of the project's results (full-resolution image pairs + every key figure + key facts).

Usage: python scripts/make_poster.py <scratch_dir> <no_rover_site_dir> <rover_target_dir> <out_png>
"""
import sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

S, SITE, ROVER, OUT = sys.argv[1:5]
R = "/home/user/NASA-Space-Apps-Prep"
W = 6000; M = 80; BG = (250, 250, 248); INK = (25, 25, 30); MUT = (85, 85, 95); ACC = (20, 90, 160); BAD = (170, 40, 40)
Image.MAX_IMAGE_PIXELS = None


def font(sz, bold=False):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf", sz)
    except Exception: return ImageFont.load_default()


def gray(a):
    a = np.asarray(a, float); f = np.isfinite(a); lo, hi = np.percentile(a[f], [0.5, 99.7])
    u = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo + 1e-12), 0, 1); u[~f] = 1.0
    return Image.fromarray((u * 255).astype(np.uint8)).convert("RGB")


blocks = []


def text(lines, pad=30):
    """lines: (text, size, bold, colour). Wraps long lines to the poster width."""
    out = []
    for t, s, b, c in lines:
        f = font(s, b); words = t.split(" "); cur = ""
        for w in words:
            trial = (cur + " " + w).strip()
            if f.getlength(trial) > W - 2 * M and cur: out.append((cur, f, s, c)); cur = w
            else: cur = trial
        out.append((cur, f, s, c))
    h = sum(s + 16 for _, _, s, _ in out) + 2 * pad
    im = Image.new("RGB", (W, h), BG); d = ImageDraw.Draw(im); y = pad
    for t, f, s, c in out: d.text((M, y), t, fill=c, font=f); y += s + 16
    blocks.append(im)


def rule():
    im = Image.new("RGB", (W, 40), BG); ImageDraw.Draw(im).line([(M, 20), (W - M, 20)], fill=(200, 200, 205), width=4); blocks.append(im)


def row(images, labels):
    """Images side by side, scaled to the same height so the row fills the poster width (nearest-neighbour: pixels stay visible)."""
    gap = 40; ar = [i.width / i.height for i in images]
    h = int((W - 2 * M - gap * (len(images) - 1)) / sum(ar))
    ims = [i.resize((int(i.width * h / i.height), h), Image.NEAREST) for i in images]
    im = Image.new("RGB", (W, h + 100), BG); d = ImageDraw.Draw(im); x = M
    for i, lab in zip(ims, labels): im.paste(i, (x, 85)); d.text((x, 20), lab, fill=INK, font=font(46, True)); x += i.width + gap
    blocks.append(im)


def figure(path, caption):
    i = Image.open(path).convert("RGB"); i = i.resize((W - 2 * M, int(i.height * (W - 2 * M) / i.width)), Image.LANCZOS)
    im = Image.new("RGB", (W, i.height + 120), BG); im.paste(i, (M, 100)); ImageDraw.Draw(im).text((M, 25), caption, fill=INK, font=font(48, True))
    blocks.append(im)


# ------------------------------------------------------------------ header + key facts for colleagues
text([("Seeing more of Mars from orbit: what repeat HiRISE passes can and cannot reveal", 110, True, INK),
      ("Multi-frame HiRISE restoration, a grounded rover-scale fill, and honest validation against NASA's Perseverance rover (Jezero crater). Work in progress, 1 Nov 2026.", 54, False, MUT),
      ("How to read this poster: section 1-2 images are at full resolution (6.25 cm per pixel) - zoom in. Every input is HiRISE satellite imagery only; rover data is used only to teach and to check.", 46, False, ACC)])
rule()
text([("Key facts", 76, True, INK),
      ("DATA - 8 HiRISE red-channel passes of the same ground, 2007-2022, 25 cm/pixel (PSP_002387, ESP_045994, ESP_046060, ESP_068294, ESP_068360, ESP_068426, ESP_068822, ESP_073068), chosen for matching sun direction; the HiRISE stereo terrain model DTEEC_045994_046060; Perseverance Navcam stereo photos (raw image API) at ~100 rover stops, sols 13-600; JPL's official Navcam orthomosaics (PDS mars2020_navcam_ops_mosaic) for 13 stops.", 46, False, INK),
      ("METHOD - (1) Restoration: every pass is locked to a reference, dense sub-pixel motion is estimated, and all passes are inverted jointly through the measured camera blur onto a 6.25 cm grid. (2) Blur: measured against rover truth and corrected with a controlled test to sigma = 1.07 HiRISE pixels (~27 cm). (3) Grounded fill: real rover-photo texture and rocks are added only below the scale HiRISE can resolve, and an exact projection removes anything HiRISE would see, so the result always reproduces the measurement. (4) A small neural network maps surface type and rocky areas; a trust map refuses to fill terrain unlike anything the rover drove over.", 46, False, INK),
      ("RESULTS", 58, True, INK),
      ("- Cleaner images: the 8-pass restoration removes most single-pass noise and makes 50-70 % fewer false 'rocks' than a single raw HiRISE image (e.g. 5.8 -> 1.8 false rocks per 100 m2).", 46, False, INK),
      ("- Real sharpness gain is small: repeatable detail improves 5.5 % (33.2 vs 35.3 cm, split-half Fourier ring correlation, 35 stops). On a known object - the parked Perseverance rover - edge sharpness is unchanged (145 vs 147 cm). In simulation at the corrected blur, fusing passes adds only 2-3 % over single-image deconvolution.", 46, False, INK),
      ("- Grounded fill: rover-scale texture that matches the HiRISE measurement to within 0.1-0.3 % at HiRISE resolution; rocks are drawn only where HiRISE itself shows a rock-sized bright spot.", 46, False, INK),
      ("- Works where no rover has been: a 128 m site 400 m from the rover route is restored, mapped and (on trusted tiles only) filled from HiRISE alone.", 46, False, INK),
      ("- Lessons for validation: rock-count checks (as used in earlier published work) can agree even when every detection is wrong; home-made rover 'truth' contained seams, smears and rover hardware; a naive Navcam projection that ignores the lens model misplaces pixels by a median 61 px (up to 158 px).", 46, False, INK),
      ("IMPORTANT CAVEATS", 58, True, BAD),
      ("- Rock-by-rock scores against rover photos are NOT yet valid: we found that the rover's waypoint coordinates sit ~140 m (138 m N-S, 19 m E-W) from the rover's true position in our HiRISE map, plus ~10-20 m that varies across the site. Every rock comparison so far compared different patches of ground. Results that compare HiRISE with HiRISE (restoration, split-half sharpness, the rover edge test) are unaffected.", 46, False, BAD),
      ("- The neural network recognises rocky areas, not individual 25-50 cm rocks (those are 1-2 blurred HiRISE pixels). The grounded fill is for display only and is never used for detection or scoring.", 46, False, BAD),
      ("NEXT STEP - place every rover stop on HiRISE to ~25 cm (anchors: the rover and its tracks in later passes), register JPL's official orthomosaics, and redo the rock comparison on the correct ground.", 46, False, ACC),
      ("CODE & DATA - github.com/hungrydodo12345/NASA-Space-Apps-Prep, branch ccr-f108bd81-7y5n2r. Pipeline: pipeline/, scripts/ (restore_real_v2.py, run_anywhere.py, build_rover_truth_v3.py, train_recognizer.py). Results and audits: results/.", 46, False, MUT)])
rule()

# ------------------------------------------------------------------ 1. full-resolution centrepiece: no-rover site
d = np.load(f"{SITE}/area.npz"); b, r, g = d["bicubic"], d["restored_multi"], d["grounded"]
y0, x0, h, w = 1450, 1250, 560, 900
text([("1. A site the rover never visited (400 m from its route): HiRISE in, cleaner map out", 76, True, INK),
      ("Native 6.25 cm/pixel crops, 35 m x 56 m. Left: one raw HiRISE pass, upscaled. Right: our 8-pass restoration of the same ground - grain and false bright specks are gone, crater rims and boulders keep their shape.", 46, False, MUT)])
row([gray(b[y0:y0 + h, x0:x0 + w]), gray(r[y0:y0 + h, x0:x0 + w])], ["Raw HiRISE (1 pass)", "Our restoration (8 passes)"])
m = np.isfinite(g); lab, nl = ndi.label(m)
if nl:
    k = max(range(1, nl + 1), key=lambda q: (lab == q).sum()); yy, xx = np.nonzero(lab == k); cy, cx = int(yy.mean()), int(xx.mean())
    text([("Grounded fill on a trusted tile (display only - never used for detection or scoring)", 60, True, INK),
          ("31 m x 31 m. The fill adds rover-scale texture taken from real Perseverance photos; seen at HiRISE resolution it equals the measurement, so nothing HiRISE saw is changed.", 46, False, MUT)])
    row([gray(r[cy - 250:cy + 250, cx - 250:cx + 250]), gray(g[cy - 250:cy + 250, cx - 250:cx + 250])], ["Restoration", "Grounded fill"])
rule()

# ------------------------------------------------------------------ 2. known object
rt = np.load(f"{ROVER}/area.npz"); rb, rr = rt["bicubic"], rt["restored_multi"]; sm = ndi.gaussian_filter(rr, 3)
py, px = np.unravel_index(np.argmax(sm[100:412, 100:412]), (312, 312)); py += 100; px += 100
text([("2. Known-object test: Perseverance itself, parked at the landing site in 3 HiRISE passes (sols 1, 6, 11)", 76, True, INK),
      ("8 m x 8 m crops around the 3 m rover (bright body, dark shadow). The restoration is visibly cleaner, but the rover's edges are no sharper (10-90 % edge rise 145 cm vs 147 cm): combining passes removes noise, not blur.", 46, False, MUT)])
row([gray(rb[py - 64:py + 64, px - 64:px + 64]), gray(rr[py - 64:py + 64, px - 64:px + 64])], ["Raw HiRISE (1 pass)", "Restored (3 passes)"])
figure(f"{R}/results/paper/fig_rover_target.png", "Rover outlines and brightness profiles across the rover, raw vs restored")
rule()

# ------------------------------------------------------------------ 3+ figure set
for p, c in [(f"{R}/results/paper/fig_grounded_model_wide.png", "3. Held-out rover stops: raw | restoration | network's surface & rock map | grounded fill | rover photos (rover photos are ~140 m off - see 8)"),
             (f"{SITE}/overview.png", "4. Whole no-rover site (128 m): raw | restoration | recognised map | fill on trusted tiles only | trust map"),
             (f"{R}/results/paper/fig_sharpness_tradeoff.png", "5. Sharper is not truer: weaker smoothing looks crisper but invents ringing; repeatable detail is unchanged"),
             (f"{R}/results/paper/fig_scaling_rover.png", "6a. Simulation on real rover ground texture: detail recovered vs number of passes, 8 algorithms"),
             (f"{R}/results/paper/fig_rocks.png", "6b. Simulation with inserted rocks of known size: share found by size"),
             (f"{R}/results/paper/fig_truth_v1_v3.png", "7a. Rover reference: flat-plane projection (left) vs our stereo-on-terrain with the full lens model (middle) and its height map (right)"),
             (f"{R}/results/paper/fig_audit_rocks.png", "7b. Visual audit: most auto-detected 'big rocks' in rover maps were smears, seams and bright patches"),
             (f"{R}/results/paper/fig_audit_height_rocks.png", "7c. Visual audit by height: per-photo height steps, not rocks"),
             (f"{R}/results/paper/fig_rover_geolocation_check.png", "8. Root cause found: locating the rover in HiRISE shows the waypoint coordinates sit ~140 m off in our map frame")]:
    figure(p, c)
rule()
text([("Bottom line", 76, True, INK),
      ("Repeat HiRISE passes make a cleaner, more trustworthy map but, at HiRISE's true blur, very little extra sharpness - shown on a known object. Our grounded fill adds rover-scale realism without ever contradicting the measurement. A fair rock-by-rock comparison with published super-resolution claims needs every rover stop placed on HiRISE to ~25 cm; that is the next step.", 50, False, INK)])

H = sum(x.height for x in blocks); out = Image.new("RGB", (W, H), BG); y = 0
for x in blocks: out.paste(x, (0, y)); y += x.height
out.save(OUT, optimize=True); print(out.size)

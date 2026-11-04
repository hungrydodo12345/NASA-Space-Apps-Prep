"""Heading of the Ingenuity frames from the helicopter's own shadow (independent of the HiRISE image).

The shadow falls on the ground opposite the sun, displaced from the nadir point by h * tan(zenith). In a rectified (pinhole) nadir frame the
vector from the principal point to the shadow therefore points along the anti-sun azimuth. Per frame the shadow is the darkest compact cross-shaped blob in a
window around its median position; its vector, carried into the mosaic frame with the pose-graph rotation, must be the same for every frame
(the sun does not move in 80 s): that is a check of both the tracker and the detection.
Sun position: Allison & McEwen (2000) style low-order Mars ephemeris for Ls, declination and hour angle at Jezero (18.44 N, 77.45 E).
Usage: python shadow_heading.py <scratch_dir> <track_npz> <first_idx> <last_idx> <out_json>
"""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from rectify import rectify, OUT_W, OUT_H
S, TR, A0, A1, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
order = json.load(open(f"{S}/ingenuity/nav_order.json")); d = np.load(TR); P = d["params"]; idx0 = int(d["idx"][0]); ref = int(d["ref"]); grp = d["groups"]; SCL = np.array([1, 1, 1, 1, 1, 1, 1000., 1000.])
def T(i): q = P[i - idx0] / SCL; return np.array([[q[0], q[1], q[2]], [q[3], q[4], q[5]], [q[6], q[7], 1.0]])
cx, cy = OUT_W / 2, OUT_H / 2
fr = {}
for i in range(A0, A1 + 1):
    a = cv2.imread(f"{S}/ingenuity/{order[i]}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32); fr[i] = rectify(a / (cv2.GaussianBlur(a, (0, 0), 45) + 1e-3))
valid = cv2.erode((rectify(np.ones((480, 640), np.float32)) > 0.99).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
# darkness of a small cross-shaped blob: local mean (sigma 14) minus local mean (sigma 3), strongest negative
dark = {i: cv2.GaussianBlur(fr[i], (0, 0), 14) - cv2.GaussianBlur(fr[i], (0, 0), 3.0) for i in fr}
stack = np.stack([np.where(valid, dark[i], 0) for i in fr]); mean_dark = stack.mean(0)
y0, x0 = np.unravel_index(np.argmax(mean_dark), mean_dark.shape); print("mean shadow darkness peaks at rectified px (x=%d, y=%d), i.e. offset from the principal point (%+d, %+d) px" % (x0, y0, x0 - cx, y0 - cy), flush=True)
pos = {}
for i in fr:
    m = np.zeros_like(valid); m[max(y0 - 70, 0):y0 + 70, max(x0 - 90, 0):x0 + 90] = True; m &= valid; r = np.where(m, dark[i], -1e9); y, x = np.unravel_index(np.argmax(r), r.shape); pos[i] = (x, y, float(r[y, x]))
V = {i: (pos[i][0] - cx, pos[i][1] - cy) for i in pos}
use = [i for i in range(A0, A1 + 1) if i - idx0 >= 0 and grp[i - idx0] == grp[ref - idx0]]
ang_img = {i: np.degrees(np.arctan2(V[i][1], V[i][0])) for i in use}                               # image angle, clockwise from +x (y down)
# direction of the shadow vector in the reference (mosaic) frame: rotate with the pose-graph map (use the Jacobian of the homography at the frame centre)
def to_ref(i):
    Ti = T(i); p = Ti @ np.array([cx, cy, 1.0]); q = Ti @ np.array([cx + V[i][0], cy + V[i][1], 1.0]); return (q[:2] / q[2]) - (p[:2] / p[2])
Vr = {i: to_ref(i) for i in use}; ang_ref = {i: np.degrees(np.arctan2(Vr[i][1], Vr[i][0])) for i in use}; lenpx = {i: float(np.hypot(*V[i])) for i in use}
ar = np.array([ang_ref[i] for i in use]); med = np.degrees(np.arctan2(np.median(np.sin(np.radians(ar))), np.median(np.cos(np.radians(ar)))))
dev = (ar - med + 180) % 360 - 180
print("shadow vector in the mosaic frame over %d frames: median angle %.1f deg, spread (MAD) %.1f deg, %d of %d within 15 deg" % (len(use), med, 1.4826 * np.median(np.abs(dev)), (np.abs(dev) < 15).sum(), len(use)))
print("shadow offset length: median %.0f px  -> %.1f m at 0.029 m/px" % (np.median(list(lenpx.values())), np.median(list(lenpx.values())) * 0.029))
# sun: Mars ephemeris (Allison & McEwen 2000), sol 414 at 11:38 LMST
import math
lat, lon_e = 18.4447, 77.4508; jd = 2459689.55                                              # 2022-04-20 ~01:32 UTC (image time)
D = jd - 2451545.0; M = math.radians((19.3871 + 0.52402073 * D) % 360); al = (270.3863 + 0.52403840 * D) % 360
pbs = 0.0071 * math.cos(math.radians(0.985626 * D / 2.2353 + 49.409)) + 0.0057 * math.cos(math.radians(0.985626 * D / 2.7543 + 168.173)) + 0.0039 * math.cos(math.radians(0.985626 * D / 1.1177 + 191.837)) + 0.0037 * math.cos(math.radians(0.985626 * D / 15.7866 + 21.736)) + 0.0021 * math.cos(math.radians(0.985626 * D / 2.1354 + 15.704)) + 0.0020 * math.cos(math.radians(0.985626 * D / 2.4694 + 95.528)) + 0.0018 * math.cos(math.radians(0.985626 * D / 32.8493 + 49.095))
eoc = (10.691 + 3.0e-7 * D) * math.sin(M) + 0.623 * math.sin(2 * M) + 0.050 * math.sin(3 * M) + 0.005 * math.sin(4 * M) + 0.0005 * math.sin(5 * M) + pbs
Ls = (al + eoc) % 360; dec = math.degrees(math.asin(0.42565 * math.sin(math.radians(Ls)) + 0.25 * math.sin(math.radians(Ls)) ** 0 * 0 + 0.0))
dec = math.degrees(math.asin(math.sin(math.radians(25.19)) * math.sin(math.radians(Ls))))
H = math.radians(-(12 - (11 + 38 / 60)) * 15.0)                                                 # hour angle: 22 min before local noon
el = math.degrees(math.asin(math.sin(math.radians(lat)) * math.sin(math.radians(dec)) + math.cos(math.radians(lat)) * math.cos(math.radians(dec)) * math.cos(H)))
az = (math.degrees(math.atan2(math.sin(H), math.cos(H) * math.sin(math.radians(lat)) - math.tan(math.radians(dec)) * math.cos(math.radians(lat)))) + 180) % 360
print("sun at Jezero for the images: Ls %.1f, declination %.1f, elevation %.1f deg, azimuth %.1f deg (from north, clockwise)" % (Ls, dec, el, az))
shadow_az = (az + 180) % 360; zen = 90 - el; print("shadow azimuth %.1f deg; expected offset at 8 m: %.1f m" % (shadow_az, 8 * math.tan(math.radians(zen))))
json.dump(dict(median_angle_ref_deg=float(med), mad_deg=float(1.4826 * np.median(np.abs(dev))), frames=len(use), within15=int((np.abs(dev) < 15).sum()), median_offset_px=float(np.median(list(lenpx.values()))),
               sun_ls=Ls, sun_elevation=el, sun_azimuth=az, shadow_azimuth=shadow_az, shadow_pos_px={int(i): [float(pos[i][0]), float(pos[i][1])] for i in use}, ang_ref={int(i): float(ang_ref[i]) for i in use}), open(OUT, "w"), indent=1)

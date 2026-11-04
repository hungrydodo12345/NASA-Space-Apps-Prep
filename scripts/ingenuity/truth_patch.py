"""Render the Ingenuity backshell mosaic onto the 6.31 cm grid of a chain tile (scripts/run_chain.py) using the validated placement.
Placement: rotation THETA (counter-clockwise, OpenCV), scale m/px, backshell offset (dx, dy) m from the tile centre (= the reference backshell position).
The chain tile is 128 native px (32 m) = 512 HR px at 6.31 cm, centred on the backshell. Output: truth (float, NaN outside the mosaic) and a validity mask."""
import sys, json, numpy as np, cv2
S, MOS, TH, SCALE, DX, DY, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7]; BX, BY = 151, 330
HR = 0.2524 / 4; N = 512
m = np.load(MOS); M = np.nan_to_num(m["mosaic"], nan=1.0).astype(np.float32); V = np.isfinite(m["mosaic"]); f = SCALE / HR; c, s = np.cos(np.radians(TH)), np.sin(np.radians(TH))
Rm = np.array([[f * c, f * s], [-f * s, f * c]]); target = np.array([N / 2 + DX / HR, N / 2 + DY / HR]); t = target - Rm @ np.array([BX, BY]); W = np.hstack([Rm, t[:, None]])
a = cv2.warpAffine(M, W, (N, N), flags=cv2.INTER_AREA); v = cv2.warpAffine(V.astype(np.float32), W, (N, N), flags=cv2.INTER_AREA) > 0.97
a = np.where(v, a, np.nan).astype(np.float32); np.savez_compressed(OUT, truth=a, valid=v, params=np.array([TH, SCALE, DX, DY])); print("truth patch: %.1f m^2 valid" % (v.sum() * HR ** 2))

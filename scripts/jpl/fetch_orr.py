import sys, json, os, urllib.request, numpy as np, time
from concurrent.futures import ThreadPoolExecutor
S = sys.argv[1]; D = f"{S}/jpl_mosaic"; OUT = f"{S}/hero/rover_truth_jpl"; os.makedirs(OUT, exist_ok=True)
wp = {f["properties"]["RMC"]: f["properties"] for f in json.load(open(f"{S}/rover/waypoints.json"))["features"]}
prods = [p for p in json.load(open(f"{D}/products.json")) if p["waypoint"] and p["restored"]]
seen = set(); prods = [p for p in prods if not (p["rmc"] in seen or seen.add(p["rmc"]))]
N, B, L0, L1 = 9000, 3, 2900, 6100           # +-16 m around the centre line/sample 4500
def get(p):
    out = f"{OUT}/{p['rmc']}.npz"
    if os.path.exists(out): return p["rmc"], "exists"
    a, b = L0 * B * N * 2, L1 * B * N * 2 - 1
    for k in range(5):
        try:
            req = urllib.request.Request(p["img"], headers={"Range": f"bytes={a}-{b}"})
            raw = urllib.request.urlopen(req, timeout=600).read(); break
        except Exception as e: time.sleep(2 ** k); raw = None
    if raw is None or len(raw) != b - a + 1: return p["rmc"], "download failed"
    arr = np.frombuffer(raw, ">i2").reshape(L1 - L0, B, N)[:, :, L0:L1].astype(np.float32)   # BIL: line, band, sample
    valid = (arr > 0).all(1); img = np.where(valid, arr.mean(1), np.nan)
    n = (L1 - L0) // 5
    blk = img.reshape(n, 5, n, 5); v = valid.reshape(n, 5, n, 5).mean((1, 3)) > 0.6
    z = np.where(v, np.nanmean(blk, (1, 3)), np.nan)
    z = (z - np.nanmean(z)) / (np.nanstd(z) + 1e-9)
    w = wp[p["rmc"]]
    np.savez_compressed(out, z=z.astype(np.float32), lat=w["lat"], lon=w["lon"], sol=w["sol"], source=p["img"], res_m=0.05, half_m=16.0)
    return p["rmc"], f"ok, coverage {np.isfinite(z).mean():.0%}"
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(get, prods): print(*r, flush=True)

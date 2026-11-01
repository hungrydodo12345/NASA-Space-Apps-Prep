import sys, json, os, re, urllib.request, numpy as np, time
from concurrent.futures import ThreadPoolExecutor
S = sys.argv[1]; D = f"{S}/jpl_mosaic"; OUT = f"{D}/dem"; os.makedirs(OUT, exist_ok=True)
prods = [p for p in json.load(open(f"{D}/products.json")) if p["waypoint"] and p["restored"]]
seen = set(); prods = [p for p in prods if not (p["rmc"] in seen or seen.add(p["rmc"]))][:13]
N, NB, STEP = 9000, 3, 50
def line(url, i):
    a = i * NB * N * 4; b = a + NB * N * 4 - 1
    for k in range(5):
        try: return np.frombuffer(urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"}), timeout=120).read(), ">f4").reshape(NB, N)[:, ::STEP]
        except Exception: time.sleep(2 ** k)
    return np.full((NB, N // STEP), np.nan, np.float32)
def get(p):
    out = f"{OUT}/{p['rmc']}.npy"
    if os.path.exists(out): return p["rmc"], "exists"
    sol = p["sol"]; base = p["img"].rsplit("/", 1)[0] + "/"
    lst = urllib.request.urlopen(base, timeout=60).read().decode()
    sd = p["name"][16:23]; f = sorted(set(re.findall(rf'(N_LRGB_{sol:04d}_DEM_{sd}_ORR_L_45M01CMJ\d+\.IMG)', lst)))
    if not f: return p["rmc"], "no DEM"
    url = base + f[-1]
    with ThreadPoolExecutor(8) as ex: rows = list(ex.map(lambda i: line(url, i), range(0, N, STEP)))
    arr = np.stack(rows, 1)          # band, line, sample at 0.5 m
    np.save(out, arr.astype(np.float32)); return p["rmc"], f"ok {arr.shape}, band ranges " + ", ".join(f"[{np.nanmin(np.where(b == 0, np.nan, b)):.1f},{np.nanmax(np.where(b == 0, np.nan, b)):.1f}]" for b in arr)
for p in prods: print(*get(p), flush=True)

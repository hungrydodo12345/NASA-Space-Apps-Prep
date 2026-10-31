"""Download every public NASA input the pipeline uses into <data_dir> (about 9 GB).

Usage: python scripts/fetch_data.py <data_dir>

  hirise_jezero/  ESP_045994 + ESP_046060 orthoimages (DTM-controlled) and the DTEEC stereo DTM
  hirise_rdr/     lighting-matched RDR observations of the landing area (16-bit)
  rover/          Perseverance traverse + waypoints (MMGIS), per-image camera metadata (raw-image API)
  ai4mars/        Perseverance terrain labels only (fetched by HTTP byte range from the 16 GB Zenodo zip)
  navcam/         labelled, downward-looking Navcam images in the hero area (sols <= 120)
"""
import json, os, re, sys, time, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor

D = sys.argv[1]
HI = "https://hirise-pds.lpl.arizona.edu/PDS"
DTM_DIR = f"{HI}/DTM/ESP/ORB_045900_045999/ESP_045994_1985_ESP_046060_1985"
RDRS = ["ESP_045994_1985", "ESP_046060_1985", "ESP_042315_1985", "ESP_068822_1985", "ESP_068294_1985", "ESP_068360_1985", "ESP_068426_1985"]
ZEN = "https://zenodo.org/api/records/15995036/files/ai4mars-dataset-merged-0.6.zip/content"
API = "https://mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020&feedtype=json&id="


def get(url, out, tries=4):
    if os.path.exists(out) and os.path.getsize(out) > 0: return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(out + ".part", "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b)
            os.replace(out + ".part", out); return
        except Exception as e:
            if a == tries - 1: raise
            time.sleep(2 ** a)


def rdr_path(oid):
    n = int(oid.split("_")[1]); lo = n // 100 * 100
    return f"{HI}/RDR/ESP/ORB_{lo:06d}_{lo + 99:06d}/{oid}/{oid}_RED"


print("HiRISE orthos + DTM ...")
for f in ("ESP_045994_1985_RED_A_01_ORTHO", "ESP_046060_1985_RED_A_01_ORTHO"):
    for ext in (".JP2", ".LBL"): get(f"{DTM_DIR}/{f}{ext}", f"{D}/hirise_jezero/{f}{ext}")
get(f"{DTM_DIR}/DTEEC_045994_1985_046060_1985_U01.IMG", f"{D}/hirise_jezero/DTEEC_045994_1985_046060_1985_U01.IMG")
print("HiRISE RDR observations ...")
for oid in RDRS:
    for ext in (".JP2", ".LBL"): get(rdr_path(oid) + ext, f"{D}/hirise_rdr/{oid}_RED{ext}")
print("Perseverance traverse ...")
for f in ("M20_waypoints.json", "M20_traverse.json"):
    get(f"https://mars.nasa.gov/mmgis-maps/M20/Layers/json/{f}", f"{D}/rover/{f.replace('M20_', '')}")

print("AI4Mars Perseverance labels (byte range of the Zenodo zip) ...")
lab_root = f"{D}/ai4mars/ai4mars-dataset-merged-0.6/m2020/labels"
if not os.path.isdir(lab_root):
    from remotezip import RemoteZip
    P = "ai4mars-dataset-merged-0.6/m2020/labels/"
    with RemoteZip(ZEN) as z:
        inf = z.infolist()
    labs = [i for i in inf if i.filename.startswith(P) and not i.filename.endswith("/")]
    lo = min(i.header_offset for i in labs)
    hi = max(i.header_offset + 30 + len(i.filename.encode()) + len(i.extra) + i.compress_size for i in labs) + 4096
    size = int(urllib.request.urlopen(urllib.request.Request(ZEN, method="HEAD"), timeout=60).headers["Content-Length"])
    sparse = f"{D}/ai4mars/sparse.zip"; os.makedirs(f"{D}/ai4mars", exist_ok=True)
    with open(sparse, "wb") as f:
        f.truncate(size)
        for a, b in ((lo, hi), (size - 30_000_000, size - 1)):
            req = urllib.request.Request(ZEN, headers={"Range": f"bytes={a}-{b}"})
            f.seek(a); f.write(urllib.request.urlopen(req, timeout=600).read())
    with zipfile.ZipFile(sparse) as z:
        for i in labs: z.extract(i, f"{D}/ai4mars")
    os.remove(sparse)

print("camera metadata for labelled Navcam images ...")
ids = sorted({re.sub(r"_merged\d+\.png$", "", f) for k in ("NAV", "M2020_GEO/ncam") for f in os.listdir(f"{lab_root}/{k}") if f.endswith(".png")})
os.makedirs(f"{D}/rover/meta", exist_ok=True)
def meta(i):
    out = f"{D}/rover/meta/{i}.json"
    if os.path.exists(out): return
    try:
        d = json.loads(urllib.request.urlopen(API + i, timeout=60).read())
        if d.get("image") and d["image"][0].get("imageid") == i: json.dump(d["image"][0], open(out, "w"))
    except Exception: pass
with ThreadPoolExecutor(8) as ex: list(ex.map(meta, ids))

print("Navcam images for the hero area (sols <= 120, looking down) ...")
wp = {f["properties"]["RMC"]: f["properties"] for f in json.load(open(f"{D}/rover/waypoints.json"))["features"]}
items = []
for f in os.listdir(f"{D}/rover/meta"):
    m = json.load(open(f"{D}/rover/meta/{f}"))
    try: el = float(m["extended"]["mastEl"])
    except Exception: continue
    if f'{m["site"]}_{m["drive"]}' in wp and m["sol"] <= 120 and el <= -15:
        items.append((m["imageid"], m["image_files"]["full_res"], el, m["sol"]))
json.dump(items, open(f"{D}/rover/hero_navcam.json", "w"))
with ThreadPoolExecutor(8) as ex:
    list(ex.map(lambda it: get(it[1], f"{D}/navcam/{it[0]}.png"), items))
print(f"done: {len(items)} Navcam images, {len(ids)} labelled images with metadata")

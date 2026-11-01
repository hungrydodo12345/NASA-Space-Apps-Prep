"""Write data_products/MANIFEST.json: every raw NASA input with source URL, size, SHA-256, and the role it plays.

Usage: python scripts/make_data_manifest.py <data_dir> <out_json>      (data_dir laid out as scripts/fetch_data.py does)
"""
import sys, os, json, hashlib
from concurrent.futures import ThreadPoolExecutor
D, OUT = sys.argv[1:3]
HI = "https://hirise-pds.lpl.arizona.edu/PDS"
DTM_DIR = f"{HI}/DTM/ESP/ORB_045900_045999/ESP_045994_1985_ESP_046060_1985"
FINAL_RDR = {"ESP_068822_1985", "ESP_068294_1985", "ESP_068360_1985", "ESP_068426_1985"}
def rdr_url(name):
    oid = name.split("_RED")[0]; n = int(oid.split("_")[1]); lo = n // 100 * 100
    return f"{HI}/RDR/ESP/ORB_{lo:06d}_{lo + 99:06d}/{oid}/{name}"
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
items = []
for name in sorted(os.listdir(f"{D}/hirise_jezero")):
    role = ("orthoimage, DTM-controlled (measured layer reference grid)" if "ORTHO" in name else "stereo DTM (terrain heights)") if name.endswith((".JP2", ".IMG")) else "PDS label"
    items.append(dict(path=f"hirise_jezero/{name}", url=f"{DTM_DIR}/{name}", role=role, used_in_final=True))
for name in sorted(os.listdir(f"{D}/hirise_rdr")):
    if not name.endswith((".JP2", ".LBL")): continue
    oid = name.split("_RED")[0]
    items.append(dict(path=f"hirise_rdr/{name}", url=rdr_url(name), role="HiRISE RDR observation (16-bit, map-projected)", used_in_final=oid in FINAL_RDR))
nav = json.load(open(f"{D}/rover/hero_navcam.json"))
for iid, url, el, sol in nav:
    items.append(dict(path=f"navcam/{iid}.png", url=url, role="Perseverance Navcam image (ground projection)", used_in_final=True, sol=sol, mast_elevation_deg=el))
for f, role, url in (("rover/waypoints.json", "rover waypoints", "https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_waypoints.json"),
                     ("rover/traverse.json", "rover traverse", "https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_traverse.json")):
    items.append(dict(path=f, url=url, role=role, used_in_final=True))
def fill(it):
    p = f"{D}/{it['path']}"
    if os.path.exists(p): it["bytes"] = os.path.getsize(p); it["sha256"] = sha(p)
    return it
with ThreadPoolExecutor(4) as ex: items = list(ex.map(fill, items))
man = dict(description="Raw public inputs of the Jezero pipeline. Re-download with scripts/fetch_data.py; verify with sha256.",
           ai4mars=dict(doi="10.5281/zenodo.15995036", license="CC-BY-4.0", use="Perseverance terrain labels (m2020/labels); 12,336 label files",
                        zip_url="https://zenodo.org/api/records/15995036/files/ai4mars-dataset-merged-0.6.zip/content"),
           rover_metadata_api="https://mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020&feedtype=json&id=<imageid>",
           hirise_catalogue="https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.TAB (29 RED observations cover 18.44N 77.44E)",
           files=items, total_gb=round(sum(i.get("bytes", 0) for i in items) / 1e9, 2))
json.dump(man, open(OUT, "w"), indent=1)
print(f"{len(items)} files, {man['total_gb']} GB; missing on disk: {sum('sha256' not in i for i in items)}")

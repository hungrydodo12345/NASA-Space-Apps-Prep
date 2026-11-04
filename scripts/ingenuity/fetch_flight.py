"""Download one Ingenuity flight's frames (HELI_NAV downward and HELI_RTE colour) from NASA's raw-image API. Usage: python fetch_flight.py <scratch_dir> <flight_number> <sol> [rte_only]"""
import sys, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
S, FL, SOL = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); RTE_ONLY = len(sys.argv) > 4
out = f"{S}/ingenuity/f{FL:02d}"; os.makedirs(out, exist_ok=True); items = []; seen = set()
for page in range(0, 6):
    try: d = json.loads(urllib.request.urlopen(f"https://mars.nasa.gov/rss/api/?feed=raw_images&category=ingenuity&feedtype=json&num=100&page={page}&order=sol+desc&sol={SOL}", timeout=60).read())
    except Exception as e: print("error", e); break
    imgs = d.get("images") or []
    if not imgs: break
    for i in imgs:
        m = re.search(r"_N(\d{3})\d{4}HELI", i["imageid"])
        if m and int(m.group(1)) == FL and i["imageid"] not in seen and (not RTE_ONLY or i["camera"]["instrument"] == "HELI_RTE"): seen.add(i["imageid"]); items.append(i)
def get(i):
    f = f"{out}/{i['imageid']}.png"
    if os.path.exists(f) and os.path.getsize(f) > 5000: return 1
    for a in range(4):
        try: open(f, "wb").write(urllib.request.urlopen(i["image_files"]["full_res"], timeout=60).read()); return 1
        except Exception: time.sleep(2 ** a)
    return 0
with ThreadPoolExecutor(8) as ex: r = list(ex.map(get, items))
json.dump(items, open(f"{out}/meta.json", "w")); print(f"flight {FL} sol {SOL}: {len(items)} images, {sum(r)} downloaded ->", out)

"""Download every Ingenuity image of flight 26 (Perseverance sol 414, 2022-04-20): 190 downward navigation frames (HELI_NAV, 640x480, with CAHVORE
camera model) and 10 colour frames (HELI_RTE), from the NASA raw-image API (category=ingenuity). Usage: python fetch_flight26.py <scratch_dir>"""
import json, sys, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
S = sys.argv[1]; items = []; seen = set()
for page in range(0, 4):
    try: d = json.loads(urllib.request.urlopen(f"https://mars.nasa.gov/rss/api/?feed=raw_images&category=ingenuity&feedtype=json&num=100&page={page}&order=sol+desc&sol=414", timeout=60).read())
    except Exception as e: print("page", page, "error", e); break
    imgs = d.get("images") or []
    if not imgs: break
    for i in imgs:
        if i["imageid"] not in seen: seen.add(i["imageid"]); items.append(i)
print("images", len(items)); json.dump(items, open(f"{S}/ingenuity/sol414_meta.json", "w"))
def get(i):
    out = f"{S}/ingenuity/{i['imageid']}.png"
    if os.path.exists(out) and os.path.getsize(out) > 5000: return 1
    for a in range(4):
        try: open(out, "wb").write(urllib.request.urlopen(i["image_files"]["full_res"], timeout=60).read()); return 1
        except Exception: time.sleep(2 ** a)
    return 0
with ThreadPoolExecutor(8) as ex: r = list(ex.map(get, items))
print("ok", sum(r), "of", len(r))
import collections; print(collections.Counter((i["camera"]["instrument"], i["image_files"]["full_res"].split("/")[-1][:3]) for i in items))

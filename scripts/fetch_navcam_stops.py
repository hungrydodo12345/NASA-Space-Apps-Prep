"""Fetch downward-looking Navcam stereo pairs (left + right twin) at Perseverance waypoint stops, with full metadata.

Selection (same rule as the original rover truth): NAVCAM_LEFT full-resolution tiles with mast elevation <= -15 deg, taken
at a waypoint stop (site_drive), paired with the NAVCAM_RIGHT tile of the same image time whose sensor window overlaps
most (kept if overlap >= 50%).

Writes into <out>: items.json ([imageid, url, mastEl, sol] like hero_navcam.json), pairs.json (left -> right, overlap),
meta/ and meta_right/ (API metadata), left/ and right/ (PNG).
Usage: python scripts/fetch_navcam_stops.py <waypoints.json> <out_dir> <sol_min> <sol_max> [STOPS env: comma list of RMC]
"""
import sys, os, json, time, urllib.request, collections
from concurrent.futures import ThreadPoolExecutor

WP, OUT, SMIN, SMAX = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
MAXEL = float(os.environ.get("MAXEL", -15))      # keep tiles with mast elevation <= MAXEL degrees
API = "https://mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020&feedtype=json&search={cam}&sol={sol}"
for d in ("meta", "meta_right", "left", "right", "lists"): os.makedirs(f"{OUT}/{d}", exist_ok=True)


def get_json(url, tries=5):
    for a in range(tries):
        try: return json.loads(urllib.request.urlopen(url, timeout=90).read())
        except Exception: time.sleep(2 ** a)
    return None


def sol_list(cam, sol):
    f = f"{OUT}/lists/{cam}_{sol}.json"
    if os.path.exists(f): return json.load(open(f))
    d = get_json(API.format(cam=cam, sol=sol)); im = (d or {}).get("images", [])
    if d is not None: json.dump(im, open(f, "w"))
    return im


def rect(s): return [int(float(t)) for t in s.strip("()").split(",")]


def download(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 10000: return 1
    for a in range(5):
        try:
            open(out + ".part", "wb").write(urllib.request.urlopen(url, timeout=120).read()); os.replace(out + ".part", out); return 1
        except Exception: time.sleep(2 ** a)
    return 0


if __name__ == "__main__":
    wp = {f["properties"]["RMC"]: f["properties"] for f in json.load(open(WP))["features"]}
    want = set(os.environ["STOPS"].split(",")) if os.environ.get("STOPS") else None
    sols = sorted({p["sol"] for k, p in wp.items() if SMIN <= p["sol"] <= SMAX and (want is None or k in want)})
    # waypoint sol is the end-of-drive sol; imaging at that stop happens on that sol and the following few
    qsols = sorted({s + k for s in sols for k in range(0, 4)})
    with ThreadPoolExecutor(6) as ex:
        L = list(ex.map(lambda s: sol_list("NAVCAM_LEFT", s), qsols)); R = list(ex.map(lambda s: sol_list("NAVCAM_RIGHT", s), qsols))
    right_by_time = collections.defaultdict(list)
    for lst in R:
        for i in lst: right_by_time[i["imageid"].split("_")[2]].append(i)
    items, pairs = [], {}
    for lst in L:
        for m in lst:
            stop = f'{m.get("site")}_{m.get("drive")}'
            if stop not in wp or (want is not None and stop not in want) or not (SMIN <= wp[stop]["sol"] <= SMAX): continue
            ext = m.get("extended", {})
            try: el = float(ext["mastEl"])
            except (KeyError, ValueError): continue
            if el > MAXEL or ext.get("scaleFactor") != "1" or m["camera"].get("camera_model_type") != "CAHVORE": continue
            x, y, w, h = rect(ext["subframeRect"]); best = None
            for r in right_by_time.get(m["imageid"].split("_")[2], []):
                if r["extended"].get("scaleFactor") != "1": continue
                X, Y, W, H = rect(r["extended"]["subframeRect"])
                o = max(0, min(x + w, X + W) - max(x, X)) * max(0, min(y + h, Y + H) - max(y, Y)) / (w * h)
                if best is None or o > best[0]: best = (o, r)
            if not best or best[0] < 0.5: continue
            r = best[1]
            json.dump(m, open(f"{OUT}/meta/{m['imageid']}.json", "w")); json.dump(r, open(f"{OUT}/meta_right/{r['imageid']}.json", "w"))
            items.append([m["imageid"], m["image_files"]["full_res"], el, m["sol"]])
            pairs[m["imageid"]] = dict(right=r["imageid"], url=r["image_files"]["full_res"], overlap=best[0])
    json.dump(items, open(f"{OUT}/items.json", "w")); json.dump(pairs, open(f"{OUT}/pairs.json", "w"))
    by = collections.Counter(f'{json.load(open(f"{OUT}/meta/{i[0]}.json"))["site"]}_{json.load(open(f"{OUT}/meta/{i[0]}.json"))["drive"]}' for i in items)
    print(f"{len(items)} stereo pairs at {len(by)} stops (sols {SMIN}-{SMAX})", flush=True)
    jobs = [(u, f"{OUT}/left/{i}.png") for i, u, _, _ in items] + [(p["url"], f"{OUT}/right/{p['right']}.png") for p in pairs.values()]
    with ThreadPoolExecutor(8) as ex: ok = sum(ex.map(lambda j: download(*j), jobs))
    print(f"downloaded {ok} of {len(jobs)} images", flush=True)

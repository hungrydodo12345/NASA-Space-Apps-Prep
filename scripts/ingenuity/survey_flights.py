"""List every Ingenuity flight in NASA's raw-image archive with its downward-camera (HELI_NAV) and colour-camera (HELI_RTE) frame counts, sol and date.
The flight number is in the image id (N0260001HELI...: flight 026). Usage: python survey_flights.py <out_json>"""
import sys, json, re, time, urllib.request, collections
OUT = sys.argv[1]; flights = collections.defaultdict(lambda: dict(nav=0, rte=0, sols=set(), first=None, ids=[]))
for page in range(0, 200):
    url = f"https://mars.nasa.gov/rss/api/?feed=raw_images&category=ingenuity&feedtype=json&num=100&page={page}&order=sol+asc"
    for a in range(4):
        try: d = json.loads(urllib.request.urlopen(url, timeout=60).read()); break
        except Exception: time.sleep(2 ** a); d = {}
    imgs = d.get("images") or []
    if not imgs: break
    for i in imgs:
        m = re.search(r"_N(\d{3})\d{4}HELI", i["imageid"]); f = int(m.group(1)) if m else -1; F = flights[f]; cam = i["camera"]["instrument"]
        F["nav" if cam == "HELI_NAV" else "rte"] += 1; F["sols"].add(i["sol"]); F["first"] = F["first"] or i["date_taken_utc"][:10]
    if page % 20 == 0: print("page", page, "flights so far", len(flights), flush=True)
res = {str(k): dict(nav=v["nav"], rte=v["rte"], sols=sorted(v["sols"]), date=v["first"]) for k, v in sorted(flights.items())}
json.dump(res, open(OUT, "w"), indent=1)
print("flights:", len(res)); [print(f"  flight {k:>3}: sol {v['sols'][0] if v['sols'] else '?':>5}  {v['date']}  nav {v['nav']:4d}  rte {v['rte']:3d}") for k, v in res.items()]

import re, sys, csv, json, collections, numpy as np
S = sys.argv[1]; D = f"{S}/hirise_index"
lbl = open(f"{D}/RDRCUMINDEX.LBL", errors="replace").read()
names = [re.search(r'NAME\s*=\s*"?([A-Z_0-9]+)', m.group(1)).group(1) for m in re.finditer(r'OBJECT\s*=\s*COLUMN(.*?)END_OBJECT', lbl, re.S)]
ix = {n: i for i, n in enumerate(names)}
obs = []
with open(f"{D}/RDRCUMINDEX.TAB", errors="replace", newline="") as f:
    for row in csv.reader(f):
        if len(row) < len(names): continue
        g = lambda n: row[ix[n]].strip().strip('"')
        try:
            if not g("PRODUCT_ID").endswith("_RED"): continue
            la0, la1, lo0, lo1 = (float(g(k)) for k in ("MINIMUM_LATITUDE", "MAXIMUM_LATITUDE", "MINIMUM_LONGITUDE", "MAXIMUM_LONGITUDE"))
            if la1 < 18.2 or la0 > 18.7 or lo1 < 77.1 or lo0 > 77.6: continue
            obs.append(dict(id=g("OBSERVATION_ID"), la0=la0, la1=la1, lo0=lo0, lo1=lo1, scale=float(g("MAP_SCALE")), inc=float(g("INCIDENCE_ANGLE")),
                            emi=float(g("EMISSION_ANGLE")), az=float(g("SUB_SOLAR_AZIMUTH")), file=g("FILE_NAME_SPECIFICATION")))
        except ValueError: continue
print(len(obs), "RED observations in the Jezero box")
w = [f["properties"] for f in json.load(open(f"{S}/rover/waypoints.json"))["features"]]
rows = []
for p in w:
    cov = [o for o in obs if o["la0"] <= p["lat"] <= o["la1"] and o["lo0"] <= p["lon"] <= o["lo1"] and o["scale"] <= 0.26]
    good = [o for o in cov if 30 <= o["inc"] <= 60 and o["emi"] <= 25]
    rows.append(dict(rmc=p["RMC"], sol=p["sol"], lat=p["lat"], lon=p["lon"], n_all=len(cov), n_good=len(good), good=[o["id"] for o in good]))
json.dump(dict(obs=obs, stops=rows), open(f"{S}/rover/hirise_coverage_all.json", "w"))
# summarise by sol band
for a, b in ((13, 120), (120, 200), (200, 300), (300, 400), (400, 500), (500, 600), (600, 800), (800, 1000), (1000, 1300), (1300, 1700), (1700, 2100)):
    r = [x for x in rows if a <= x["sol"] < b]
    if r: print(f"sols {a:4d}-{b:4d}: {len(r):3d} stops, good passes median {np.median([x['n_good'] for x in r]):.0f}, max {max(x['n_good'] for x in r)}, stops with >=6 good: {sum(x['n_good']>=6 for x in r)}")

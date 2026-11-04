"""List the JPL Ingenuity mosaics in the PDS mars2020_helicam bundle (per sol): resolution, site-frame extent, size, time. Usage: python list_helicam_mosaics.py > out.txt"""
import re, urllib.request, json, sys
B = "https://planetarydata.jpl.nasa.gov/img/data/mars2020/mars2020_helicam/data/sol"
out = {}
for sol in [47, 58, 61, 76, 91, 107, 120, 133, 152, 163, 174, 193, 254, 268, 282, 292, 362, 403, 414, 418]:
    u = f"{B}/{sol:05d}/ids/rdr/mosaic/"
    try: h = urllib.request.urlopen(u, timeout=60).read().decode()
    except Exception as e: out[sol] = str(e)[:60]; continue
    names = sorted(set(re.findall(r'href="([^"/?]+\.xml)"', h)))
    out[sol] = names
    for n in names:
        if not re.search(r"_RAS_|_ras_", n): continue
        x = urllib.request.urlopen(u + n, timeout=60).read().decode()
        g = lambda t: (re.search(rf"<{t}[^>]*>([^<]*)<", x) or [None, None])[1]
        el = re.findall(r"<elements>(\d+)</elements>", x)
        print(sol, n, "res", g("cart:pixel_resolution_x"), "x", g("cart:x_axis_minimum"), g("cart:x_axis_maximum"), "y", g("cart:y_axis_minimum"), g("cart:y_axis_maximum"),
              "site", g("geom:index_value_number"), "frame", g("geom:coordinate_space_frame_type"), "size", el[:2], g("start_date_time"), flush=True)
json.dump(out, open("mosaic_dirs.json", "w"), indent=1)

"""Story chapters and open-world points of interest for the game, in the tileset's scene coordinates.

Every location comes from data: the rover's position on a given sol is its last localized waypoint on or before that sol (rover waypoints, shifted onto
the DTM by the landing-spot offset, see traverse.json), and the landing hardware and Ingenuity truth site are our own HiRISE placements (shifted by the
RDR->DTM registration in build_info.json). Chapter texts state only well-documented mission facts.
Usage: python scripts/build_story.py <tiles_dir> <rdr_ref.LBL>"""
import os, sys, json, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from tiling.mars import Equirect, EnuFrame

D, LBL = sys.argv[1:3]
sc = json.load(open(f"{D}/scene.json")); tr = json.load(open(f"{D}/traverse.json"))["points"]; info = json.load(open(f"{D}/build_info.json"))
o = sc["origin"]; f = EnuFrame(o["lat_deg"], o["lon_deg"], o["height_m"]); rdr = Equirect.from_file(LBL); sh = info["registration"]["rdr_to_ortho"]
def at_sol(sol):
    p = [t for t in tr if t["sol"] <= sol]; return (p[-1] if p else tr[0])
def rdr_point(gl, gs):                                     # a point in the reference RDR grid (global line, sample) -> scene ENU (height filled by the game)
    la, lo = rdr.pix_to_latlon(rdr.loff + gl + sh["shift_rdr_lines"], rdr.soff + gs + sh["shift_rdr_samples"]); e = f.from_latlon(np.array([la]), np.array([lo]), np.array([0.0]))[0]
    return [round(float(e[0]), 2), round(float(e[1]), 2)]
P = lambda t: [t["enu"][0], t["enu"][1]]
backshell, parachute = rdr_point(-4372385, -23480563), rdr_point(-4372517, -23480471)
site2 = rdr_point(-4370522 + 64, -23476162 + 64); boulder = rdr_point(-4371691, -23476185)

chapters = [
    dict(id="touchdown", title="Touchdown", target=P(at_sol(13)), radius=25, sol=0,
         text="Perseverance landed in Jezero crater on 18 February 2021. The landing spot was later named Octavia E. Butler Landing. The rover stayed parked here for its first two weeks, testing its instruments.",
         task="Look around, then drive to the end of the rover's first drive."),
    dict(id="first_drive", title="First drive", target=P(at_sol(14)), radius=12, sol=14,
         text="On sol 14 (4 March 2021) Perseverance made its first drive, a few metres forward, a turn, and back. Its tracks were the first rover tracks in Jezero.",
         task="Follow the traverse to where the rover released the Ingenuity helicopter."),
    dict(id="ingenuity_dropoff", title="Helicopter drop-off", target=P(at_sol(44)), radius=15, sol=43,
         text="Here the rover lowered the Ingenuity helicopter onto the ground and drove away from it. On 19 April 2021 (sol 58) Ingenuity made the first powered, controlled flight on another planet.",
         task="Drive south to the ground Ingenuity photographed on its fourth and fifth flights."),
    dict(id="helicopter_view", title="The helicopter's view", target=site2, radius=20, sol=76,
         text="Ingenuity's downward camera photographed this ground at about 1 cm per pixel. Compare it with HiRISE, which sees the same ground from orbit at 25 cm per pixel. In the provenance view, this is one of the few places where the textures are measured at centimetre scale.",
         task="Head south along the traverse toward the crater floor."),
    dict(id="south", title="Onto the crater floor", target=P(at_sol(110)), radius=20, sol=110,
         text="In the following months Perseverance drove south across the fractured crater floor to begin its first science campaign.",
         task="Find the site of the first sampling attempt."),
    dict(id="roubion", title="An empty tube", target=P(at_sol(164)), radius=20, sol=164,
         text="On sol 164 (6 August 2021) Perseverance tried to core its first sample, from a rock nicknamed Roubion. The rock crumbled and the tube came back empty of rock.",
         task="Drive on to where the first rock cores were collected."),
    dict(id="montdenier", title="First rock samples", target=P(at_sol(196)), radius=25, sol=190,
         text="On 1 September 2021 (sol 190) Perseverance collected its first rock core, Montdenier, and a few days later a second one from the same rock. These are among the samples intended for return to Earth.",
         task="Travel west to the parachute and backshell, 1 km from the landing spot."),
    dict(id="debris", title="The debris field", target=backshell, radius=25, sol=414,
         text="The backshell and parachute protected the rover and slowed it on the way down. Ingenuity flew over them on its 26th flight in April 2022, and those images are one of the two centimetre-scale answer keys used to test the HiRISE restorations in this project.",
         task="Story complete. Keep exploring the open world."),
]
pois = [dict(id=c["id"], title=c["title"], pos=c["target"], kind="story") for c in chapters]
pois += [dict(id="parachute", title="Parachute", pos=parachute, kind="hardware"), dict(id="boulder", title="Bright boulder (HiRISE blur target)", pos=boulder, kind="science")]
pois += [dict(id=f"sol{t['sol']}", title=f"Rover on sol {t['sol']}", pos=P(t), kind="traverse") for t in tr[::12]]
story = dict(frame="scene ENU (m): [east, north]; heights come from the terrain", start=dict(pos=P(at_sol(13)), heading_deg=0),
             chapters=chapters, pois=pois, sources=dict(rover_positions="rover waypoints (traverse.json), last localization on or before each sol",
                                                        hardware="HiRISE placements of this project (parachute/backshell crops; site-2 truth tile)"))
json.dump(story, open(f"{D}/story.json", "w"), indent=1); print("wrote", f"{D}/story.json", len(chapters), "chapters,", len(pois), "points of interest")
for c in chapters: print(f"  {c['id']:18s} at E {c['target'][0]:8.1f} N {c['target'][1]:8.1f}")

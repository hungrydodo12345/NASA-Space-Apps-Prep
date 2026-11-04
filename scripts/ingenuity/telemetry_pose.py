"""Per-frame camera pose of an Ingenuity downward frame from the PDS helicam RDR label (HELI_NAV RAS product).
The label gives (a) the CAHVORE camera model expressed in HELI_G_FRAME_<flight> (the flight frame) with H, V, A vectors, and (b) the flight frame's origin offset and rotation relative to SITE_FRAME <n>.
Image-right and image-down unit vectors in the flight frame follow from H, V, A: right = (H - (A.H) A) / |A x H|, down = (V - (A.V) A) / |A x V|.
Usage as a module: pose(image_id) -> dict(centre_g, right_g, down_g, axis_g, g_offset, g_quat)."""
import re, urllib.request, numpy as np
BASE = "https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_helicam/data/sol/{sol}/ids/rdr/heli/"
def label(image_id, cache=None):
    sol = image_id.split("_")[1].zfill(5); name = image_id.replace("ECM", "RAS") + "04.xml"
    return urllib.request.urlopen(BASE.format(sol=sol) + name, timeout=90).read().decode("utf-8", "replace")
def vec(block, tags): return np.array([float(re.search(rf"<geom:{t}[^>]*>([^<]+)<", block).group(1)) for t in tags])
def pose(image_id):
    t = label(image_id); cam = re.search(r"<geom:CAHVORE_Model>(.*?)</geom:CAHVORE_Model>", t, re.S).group(1)
    C = vec(re.search(r"<geom:Vector_Center>(.*?)</geom:Vector_Center>", cam, re.S).group(1), ["x_position", "y_position", "z_position"])
    A = vec(re.search(r"<geom:Vector_Axis>(.*?)</geom:Vector_Axis>", cam, re.S).group(1), ["x_unit", "y_unit", "z_unit"])
    H = vec(re.search(r"<geom:Vector_Horizontal>(.*?)</geom:Vector_Horizontal>", cam, re.S).group(1), ["x_pixel", "y_pixel", "z_pixel"])
    V = vec(re.search(r"<geom:Vector_Vertical>(.*?)</geom:Vector_Vertical>", cam, re.S).group(1), ["x_pixel", "y_pixel", "z_pixel"])
    hs = np.linalg.norm(np.cross(A, H)); vs = np.linalg.norm(np.cross(A, V)); right = (H - (A @ H) * A) / hs; down = (V - (A @ V) * A) / vs
    g = re.search(r"<local_identifier>HELI_G_FRAME_\d+</local_identifier>(.*?)</geom:Coordinate_Space_Definition>", t, re.S).group(1)
    off = vec(re.search(r"<geom:Vector_Origin_Offset>(.*?)</geom:Vector_Origin_Offset>", g, re.S).group(1), ["x_position", "y_position", "z_position"])
    q = vec(re.search(r"<geom:Quaternion_Plus_Direction>(.*?)</geom:Quaternion_Plus_Direction>", g, re.S).group(1), ["qcos", "qsin1", "qsin2", "qsin3"])
    site = int(re.search(r"SITE_FRAME</geom:coordinate_space_frame_type>.*?<geom:index_value_number>(\d+)<", g, re.S).group(1))
    return dict(centre_g=C, right_g=right, down_g=down, axis_g=A, g_offset=off, g_quat=q, site=site, fx=hs, fy=vs)

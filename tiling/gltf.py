"""Minimal, dependency-free glTF 2.0 binary (.glb) writer for textured terrain meshes."""
import json, struct
import numpy as np


def write_glb(path, positions, normals, uvs, indices, png_bytes=None, translation=(0, 0, 0), name="tile", extras=None, mime="image/png"):
    pos = np.ascontiguousarray(positions, np.float32); nrm = np.ascontiguousarray(normals, np.float32); uv = np.ascontiguousarray(uvs, np.float32)
    big = len(positions) > 65535; idx = np.ascontiguousarray(indices, np.uint32 if big else np.uint16).reshape(-1)
    chunks, views, acc = [], [], []
    def add(data, target=None):
        off = sum(len(c) for c in chunks); b = data.tobytes() if isinstance(data, np.ndarray) else data
        pad = (-len(b)) % 4; chunks.append(b + b"\x00" * pad); v = dict(buffer=0, byteOffset=off, byteLength=len(b))
        if target: v["target"] = target
        views.append(v); return len(views) - 1
    def accessor(view, ctype, count, typ, mn=None, mx=None):
        a = dict(bufferView=view, componentType=ctype, count=int(count), type=typ)
        if mn is not None: a["min"], a["max"] = [float(x) for x in mn], [float(x) for x in mx]
        acc.append(a); return len(acc) - 1
    a_pos = accessor(add(pos, 34962), 5126, len(pos), "VEC3", pos.min(0), pos.max(0))
    a_nrm = accessor(add(nrm, 34962), 5126, len(nrm), "VEC3"); a_uv = accessor(add(uv, 34962), 5126, len(uv), "VEC2")
    a_idx = accessor(add(idx, 34963), 5125 if big else 5123, len(idx), "SCALAR")
    g = dict(asset=dict(version="2.0", generator="NASA-Space-Apps-Prep tiling"), scene=0, scenes=[dict(nodes=[0])],
             nodes=[dict(name=name, mesh=0, translation=[float(t) for t in translation])],
             meshes=[dict(name=name, primitives=[dict(attributes=dict(POSITION=a_pos, NORMAL=a_nrm, TEXCOORD_0=a_uv), indices=a_idx, material=0)])],
             materials=[dict(name=name + "_terrain", pbrMetallicRoughness=dict(metallicFactor=0.0, roughnessFactor=1.0, baseColorFactor=[1, 1, 1, 1]), doubleSided=False)],
             accessors=acc, bufferViews=views)
    if png_bytes is not None:
        iv = add(png_bytes); g["images"] = [dict(bufferView=iv, mimeType=mime)]
        g["samplers"] = [dict(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)]; g["textures"] = [dict(source=0, sampler=0)]
        g["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = dict(index=0)
    if extras: g["nodes"][0]["extras"] = extras
    blob = b"".join(chunks); g["buffers"] = [dict(byteLength=len(blob))]
    js = json.dumps(g, separators=(",", ":")).encode(); js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total)); f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(blob), 0x004E4942)); f.write(blob)
    return total


def read_glb_json(path):
    """Header check and JSON chunk of a .glb (used by tests)."""
    b = open(path, "rb").read(); magic, ver, total = struct.unpack("<III", b[:12]); n, typ = struct.unpack("<II", b[12:20])
    assert magic == 0x46546C67 and ver == 2 and total == len(b) and typ == 0x4E4F534A
    return json.loads(b[20:20 + n])

"""Per-world object size (nablaengine.params field object_size) in the forest engine's text: the half-extents of one box
geom (PhysicsParams.size_geom, or com_body's single box collision geom) are each world's, read from its com row, in
every place the engine reads that geom's size. A build without the field is unchanged."""
import copy
import re

import numpy as np

FIELD = "object_size"


def active(params):
    return bool(params) and FIELD in params.fields


def size_geom(m, params, com_body=None):
    """The object geom's model id: PhysicsParams.size_geom (a geom name or id), else com_body's single box collision
    geom. ValueError where it is not a box collision geom on a moving body."""
    import mujoco
    from nablaengine.forest_engine import step
    box = int(mujoco.mjtGeom.mjGEOM_BOX)
    cg = step.collision_geoms(m)
    g = params.size_geom
    if g is None:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, com_body) if com_body else -1
        if b < 1:
            raise ValueError("PhysicsParams object_size: name the box geom (PhysicsParams(size_geom=...)) or the body "
                             f"that has it (init(com_body=...)); got com_body={com_body!r}")
        boxes = [x for x in cg if int(m.geom_bodyid[x]) == b and int(m.geom_type[x]) == box]
        if len(boxes) != 1:
            raise ValueError(f"PhysicsParams object_size: body {com_body!r} has {len(boxes)} box collision geoms "
                             "(one needed; else name it with PhysicsParams(size_geom=...))")
        return boxes[0]
    gid = int(g) if not isinstance(g, str) else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    if not 0 <= gid < m.ngeom:
        raise ValueError(f"PhysicsParams size_geom={g!r}: no such geom")
    if int(m.geom_type[gid]) != box:
        raise ValueError(f"PhysicsParams size_geom={g!r}: not a box geom")
    if gid not in cg:
        raise ValueError(f"PhysicsParams size_geom={g!r}: in no contact pair")
    if int(m.geom_bodyid[gid]) == 0:
        raise ValueError(f"PhysicsParams size_geom={g!r}: a world geom (the object must be on a moving body)")
    return gid


def check(m, params, com_body, terrain=None):
    """ValueError where object_size does not apply to forest model m."""
    if not active(params):
        return
    g = size_geom(m, params, com_body)
    if terrain is not None:
        from nablaengine import terrain as _terrain
        if any(int(x) == g for x, _, _ in _terrain.robot_terrain_geoms(m, [], terrain.geoms)):
            raise ValueError("PhysicsParams object_size: the object geom against the terrain is not supported")


def resolve(m, params, com_body):
    """params with size_geom as the object geom's model id (size_geom()); params itself without object_size."""
    if not active(params):
        return params
    import dataclasses
    return dataclasses.replace(params, size_geom=size_geom(m, params, com_body))


def bound_model(m, params):
    """m, or (object_size) a copy whose object geom has object_size_max's size, bounding radius and box: the model the
    build's text and constants come from (module docstring). params: resolve()'s."""
    if not active(params):
        return m
    g = size_geom(m, params)
    mb = copy.deepcopy(m)
    mx = np.asarray(params.object_size_max, np.float64)
    mb.geom_size[g] = mx
    mb.geom_rbound[g] = float(np.linalg.norm(mx))
    mb.geom_aabb[g] = np.concatenate([np.zeros(3), mx])
    return mb


def defines(m, params):
    """#defines of the object geom: OSZ_G (its index in the engine's geom table), GSZ, and (in world_params' header)
    PW_OSIZE."""
    from nablaengine.forest_engine import step
    g = step.collision_geoms(m).index(size_geom(m, params))
    return (f"#define OSZ_G {g}  // the object geom (forest_engine.object_size: its size per world, osz)\n"
            "#define GSZ(g) ((g) == OSZ_G ? osz : c_gsize + 3 * (g))  // a geom's size (object_size)\n"
            "#define GIX(p) ((p) == osz ? OSZ_G : (int)((p) - c_gsize) / 3)  // the geom of a size pointer (object_size)\n")


_FN = re.compile(r"(?:__device__|__global__)[^;{}()#]*?\b([A-Za-z_]\w*)\s*\(")
_GSIZE = re.compile(r"c_gsize \+ 3 \* (\w+(?:\(\w+\))?)")
_GINDEX = re.compile(r"\(int\)\((\w+) - c_gsize\) / 3")
_OSZ_ARG = "const float* osz, "


def _close(src, i, o, c):
    d = 0
    for j in range(i, len(src)):
        if src[j] == o:
            d += 1
        elif src[j] == c:
            d -= 1
            if d == 0:
                return j + 1
    raise RuntimeError("object_size: unbalanced text")


def _mask(src):
    out, stack = [], []
    for line in src.split("\n"):
        t = line.strip()
        if t.startswith("#"):
            d = t[1:].split()[0] if len(t) > 1 and t[1:].split() else ""
            if d in ("if", "ifdef", "ifndef"):
                stack.append(False)
            elif d in ("elif", "else") and stack:
                stack[-1] = True
            elif d == "endif" and stack:
                stack.pop()
            out.append(" " * len(line))
        elif any(stack):
            out.append(" " * len(line))
        else:
            i = line.find("//")
            out.append(line if i < 0 else line[:i] + " " * (len(line) - i))
    return "\n".join(out)


def functions(src):
    """[(name, start, params (a, b), body (a, b) or None for a prototype)] of the __device__ / __global__ functions
    (positions in src; found in _mask(src): the #if branches only)."""
    src = _mask(src)
    out, pos = [], 0
    while True:
        mt = _FN.search(src, pos)
        if not mt:
            return out
        p0 = mt.end() - 1
        p1 = _close(src, p0, "(", ")")
        k = p1
        while k < len(src) and src[k] not in "{;=":
            k += 1
        if k >= len(src) or src[k] == "=":
            pos = p1
            continue
        if src[k] == ";":
            out.append((mt.group(1), mt.start(), (p0 + 1, p1 - 1), None))
            pos = k + 1
            continue
        b1 = _close(src, k, "{", "}")
        out.append((mt.group(1), mt.start(), (p0 + 1, p1 - 1), (k + 1, b1 - 1)))
        pos = b1


def _calls(body, names):
    return any(re.search(r"\b" + re.escape(n) + r"\s*\(", body) for n in names)


def _has_state(params):
    return re.search(r"&\s*s\s*(,|$)", params.strip()) is not None


def _aliases(src, names):
    out = set()
    for a, f in re.findall(r"^#define (\w+) (\w+)\b", src, flags=re.M):
        if f in names:
            out.add(a)
    return out


def engine(src, m, params):
    """The final engine text with the object's per-world size (module docstring); unchanged without object_size."""
    if not active(params):
        return src
    n0 = len(_GSIZE.findall(src))
    if not n0:
        raise RuntimeError("object_size: no size read in the engine text")
    src = _GSIZE.sub(r"GSZ(\1)", src)
    src = _GINDEX.sub(r"GIX(\1)", src)
    code = re.sub(r"//[^\n]*", "", src)
    rest = [x for x in re.findall(r"c_gsize[^\n]*", code) if not x.startswith("c_gsize[")]
    reads = re.findall(r"c_gsize\[[^\]]*\]", code)
    decl = [x for x in reads if re.fullmatch(r"c_gsize\[(\d+|NGEOMC \* 3)\]", x)]
    if rest or len(reads) != len(decl):
        raise RuntimeError(f"object_size: c_gsize reads left: {rest + [x for x in reads if x not in decl]}")
    hand = "  float gx[NGEOMC][3], gq[NGEOMC][4];  // k_pre_a's geom poses, for k_ccd\n"
    if hand in src:
        if src.count(hand) != 1:
            raise RuntimeError("object_size: struct Hand's geom poses found more than once")
        src = src.replace(hand, hand + "  float osz[3];  // the object's size (object_size)\n")
    fns = functions(src)
    takes, local = set(), set()
    while True:
        names = takes | _aliases(src, takes)
        changed = False
        for name, start, (pa, pb), body in fns:
            if body is None or name in takes or name in local:
                continue
            b = src[body[0]:body[1]]
            if "GSZ(" in b or "GIX(" in b or _calls(b, names):
                if "__global__" in src[start:pa]:
                    raise RuntimeError(f"object_size: kernel {name} needs the object's size")
                if _has_state(src[pa:pb]):
                    local.add(name)
                else:
                    takes.add(name)
                    changed = True
        if not changed:
            break
    names = takes | _aliases(src, takes)
    edits = []
    for name, start, (pa, pb), body in fns:
        if name in takes:
            edits.append((pa, 0, _OSZ_ARG))
        if body is not None and name in local:
            edits.append((body[0], 0, "\n  const float* osz = s.pw + PW_OSIZE;  // (object_size)"))
    call = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")(\s*\()")
    heads = {pa for name, _, (pa, _b), _ in fns if name in takes}
    for mt in call.finditer(src):
        if mt.end() in heads:
            continue
        edits.append((mt.end(), 0, "osz, "))
    for pos, n, ins in sorted(edits, key=lambda e: e[0], reverse=True):
        src = src[:pos] + ins + src[pos + n:]
    return defines(m, params) + src


def host(src, engine_src):
    """The host kernels' text (k_pre_a, k_ccd) with the object's size: k_pre_a hands the world's size to k_ccd through
    Hand, k_ccd passes it to the functions that take it (those of engine_src, engine()'s text)."""
    takes = {name for name, _, (pa, pb), _ in functions(engine_src) if engine_src.startswith(_OSZ_ARG, pa)}
    if not takes:
        raise RuntimeError("object_size: no function takes the object's size in the engine text")
    names = takes | _aliases(engine_src, takes) | _aliases(src, takes)
    old = "  Hand& g = work[w];\n  const int g1 = PG1(pair), g2 = PG2(pair), t1 = c_gtype[g1], t2 = c_gtype[g2];\n"
    copy_ = ("  for (int i = lane; i < NGEOMC; i += 32) {\n    for (int t = 0; t < 3; t++) g.gx[i][t] = s.gx[i][t];\n"
             "    for (int t = 0; t < 4; t++) g.gq[i][t] = s.gq[i][t];\n  }\n")
    if "k_ccd" in src:
        if src.count(old) != 1 or src.count(copy_) != 1:
            raise RuntimeError("object_size: k_ccd / k_pre_a anchors not found once")
        src = src.replace(old, old.replace("  Hand& g = work[w];\n", "  Hand& g = work[w];\n  const float* osz = g.osz;"
                                           "  // (object_size: k_pre_a's)\n"))
        src = src.replace(copy_, copy_ + "  if (lane < 3) g.osz[lane] = s.pw[PW_OSIZE + lane];  // (object_size)\n")
    for name, start, (pa, pb), body in reversed(functions(src)):
        if body is None:
            continue
        b = src[body[0]:body[1]]
        if "GSZ(" in b or "c_gsize" in b:
            raise RuntimeError(f"object_size: host function {name} reads a geom size")
        nb = re.sub(r"\b(" + "|".join(map(re.escape, names)) + r")(\s*\()", r"\1\2osz, ", b)
        if nb != b:
            if name != "k_ccd":
                raise RuntimeError(f"object_size: host function {name} calls the narrowphase")
            src = src[:body[0]] + nb + src[body[1]:]
    return src


QUERY_SIZE = "  for (int k = 0; k < 3; k++) { G.c[k] = Xb[k] + t[k]; G.s[k] = c_hq_size[3 * g + k]; }\n"
QUERY_STATE = "  int sb[@NS@ > 0 ? @NS@ : 1][2];\n};\n"
QUERY_LOAD = "  load_fixtures(s, com, w, lane);\n  if (lane == 0) {\n    s.Q[0][0] = 1.f;"


def queries(src, m, params, geoms, ns="DST_NS"):
    """A distance-query library's text (forest_engine.dist_device.apply's, or a library built on it) with the
    object's per-world size in its query geoms (geoms: the query geom ids, c_hq_size's order): the geom setup reads
    the world's size for that geom (from the state's pw: the query kernels' state gets pw, loaded with the fixtures;
    ns: the slot-count macro that ends that state's last row); unchanged without object_size or when the object is
    not a query geom."""
    if not active(params):
        return src
    g = size_geom(m, params)
    if g not in list(geoms):
        return src
    hq = list(geoms).index(g)
    state = QUERY_STATE.replace("@NS@", ns)
    for a in (QUERY_SIZE, state, QUERY_LOAD):
        if src.count(a) != 1:
            raise RuntimeError(f"object_size (queries): anchor found {src.count(a)} times: {a[:60]!r}")
    src = src.replace(QUERY_SIZE, f"  const float* hsz = g == {hq} ? s.pw + PW_OSIZE : c_hq_size + 3 * g;  "
                      "// (object_size)\n" + QUERY_SIZE.replace("c_hq_size[3 * g + k]", "hsz[k]"))
    src = src.replace(state, state[:-3] + "  float pw[NPARAM];  // the world's parameters (object_size)\n};\n")
    src = src.replace(QUERY_LOAD, "  for (int i = lane; i < NPARAM; i += 32) s.pw[i] = com[(size_t)w * COM_STRIDE + PW_OFF"
                      " + i];  // (object_size)\n" + QUERY_LOAD)
    return src

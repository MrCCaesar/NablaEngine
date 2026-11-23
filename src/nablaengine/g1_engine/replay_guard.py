"""The contact adjoint's guard against a contact its replay does not reproduce (the tree engine; the text edits are
shared with forest_engine.replay_guard)."""
import re

import mujoco

NO_REPLAY = ("#define ADJ_NO_REPLAY __int_as_float(0x7f800000)  // adj_contact_one: the replay found no contact `slot` "
             "(forest_engine.replay_guard)\n")
NO_REPLAY_TREE = NO_REPLAY.replace("(forest_engine.replay_guard)", "(g1_engine.replay_guard)")
ONE_HEAD = "__device__ __forceinline__ float adj_contact_one("
REPLAY_RE = re.compile(r"\n( *)const int br = adj_pair_geom(_aux)?\(pair, slot, (s\.taux\[c\], )?s\.gx, s\.gq, &d, p, fr, "
                       r"&nu, nullptr, nullptr[^\n]*\);\n")
GUARD = "{ind}if (br < 0) return ADJ_NO_REPLAY;  // (replay_guard: no contact `slot` in the replay; cs stays 0)\n"
CALL_RE = re.compile(r"      bad = !\(adj_contact_one\((s, c, pair, slot, zone, cw, dbg[^)]*)\) < 0\.f\);\n"
                     r"      if \(zone\) cb = c_gbody\[PG1\(pair\)\] \| c_gbody\[PG2\(pair\)\] << 8;\n")
CALL_NEW = ("      const float dr = adj_contact_one({args});\n"
            "      bad = !(dr < 0.f);\n"
            "      if (zone && dr != ADJ_NO_REPLAY) cb = c_gbody[PG1(pair)] | c_gbody[PG2(pair)] << 8;  // (replay_guard)\n")


def primitive_only(m):
    """Every contact pair is between planes, spheres and capsules."""
    G = mujoco.mjtGeom
    simple = {int(G.mjGEOM_PLANE), int(G.mjGEOM_SPHERE), int(G.mjGEOM_CAPSULE)}
    return all(int(m.geom_type[m.pair_geom1[p]]) in simple and int(m.geom_type[m.pair_geom2[p]]) in simple
               for p in range(m.npair))


def needed(m, terrain=None):
    """A contact pair with a geom other than plane, sphere and capsule, or a terrain build."""
    return terrain is not None or not primitive_only(m)


def guard_one(src, no_replay=NO_REPLAY):
    """src with ADJ_NO_REPLAY and the early return after adj_contact_one's replay call (RuntimeError when not found
    once). The callers' edits are the engine's (apply here, forest_engine.replay_guard.apply)."""
    if src.count(ONE_HEAD) != 1:
        raise RuntimeError("replay_guard: adj_contact_one not found once")
    src = src.replace(ONE_HEAD, no_replay + ONE_HEAD, 1)
    i = src.index(ONE_HEAD)
    mt = REPLAY_RE.search(src, i)
    if mt is None or mt.start() > src.index("\n}\n", i):
        raise RuntimeError("replay_guard: adj_contact_one's replay call not found")
    return src[:mt.end()] + GUARD.format(ind=mt.group(1)) + src[mt.end():]


def apply(src, m, terrain=None):
    """src (g1_engine.adjoint.cuda_source's text) with the guard where needed(m, terrain); else unchanged. A text
    without the adjoint (no adj_contact_one) is returned unchanged."""
    if not needed(m, terrain) or ONE_HEAD not in src:
        return src
    src = guard_one(src, NO_REPLAY_TREE)
    if len(CALL_RE.findall(src)) != 1:
        raise RuntimeError(f"replay_guard: adj_contacts' call found {len(CALL_RE.findall(src))} times")
    return CALL_RE.sub(lambda mt: CALL_NEW.format(args=mt.group(1)), src)

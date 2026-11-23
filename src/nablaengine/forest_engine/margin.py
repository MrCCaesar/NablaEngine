"""Contact margins on the forest engine (MuJoCo's pair margin / gap)."""
import re

import mujoco
import numpy as np

G = mujoco.mjtGeom
SUPPORTED = {tuple(sorted((int(a), int(b)))) for a, b in (
    (G.mjGEOM_PLANE, G.mjGEOM_SPHERE), (G.mjGEOM_PLANE, G.mjGEOM_CAPSULE), (G.mjGEOM_PLANE, G.mjGEOM_ELLIPSOID),
    (G.mjGEOM_PLANE, G.mjGEOM_CYLINDER), (G.mjGEOM_PLANE, G.mjGEOM_BOX), (G.mjGEOM_SPHERE, G.mjGEOM_SPHERE),
    (G.mjGEOM_SPHERE, G.mjGEOM_CAPSULE), (G.mjGEOM_CAPSULE, G.mjGEOM_CAPSULE), (G.mjGEOM_SPHERE, G.mjGEOM_CYLINDER))}
CONVEX = {tuple(sorted((int(a), int(b)))) for a, b in (
    (G.mjGEOM_SPHERE, G.mjGEOM_BOX), (G.mjGEOM_CAPSULE, G.mjGEOM_BOX), (G.mjGEOM_BOX, G.mjGEOM_BOX),
    (G.mjGEOM_PLANE, G.mjGEOM_MESH), (G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER), (G.mjGEOM_CYLINDER, G.mjGEOM_CYLINDER),
    (G.mjGEOM_CYLINDER, G.mjGEOM_BOX), (G.mjGEOM_SPHERE, G.mjGEOM_ELLIPSOID), (G.mjGEOM_CAPSULE, G.mjGEOM_ELLIPSOID),
    (G.mjGEOM_ELLIPSOID, G.mjGEOM_ELLIPSOID), (G.mjGEOM_ELLIPSOID, G.mjGEOM_CYLINDER),
    (G.mjGEOM_ELLIPSOID, G.mjGEOM_BOX))} | {tuple(sorted((int(t), int(G.mjGEOM_MESH)))) for t in (
        G.mjGEOM_SPHERE, G.mjGEOM_CAPSULE, G.mjGEOM_ELLIPSOID, G.mjGEOM_CYLINDER, G.mjGEOM_BOX, G.mjGEOM_MESH)}


def _tname(t):
    return G(int(t)).name[7:].lower()


def pair_types(m, p):
    """Pair p's MuJoCo geom types, sorted."""
    return tuple(sorted((int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]]))))


def supported_pair(m, p):
    """Whether pair p's geom types take a margin on the forest engine (SUPPORTED)."""
    return pair_types(m, p) in SUPPORTED or pair_types(m, p) in CONVEX


def convex_active(m):
    """Whether model m has a convex pair (CONVEX) with a nonzero margin (forest_engine.margin_convex's edits)."""
    return any(float(m.pair_margin[p]) != 0 and pair_types(m, p) in CONVEX for p in range(m.npair))


def supported_types(t1, t2):
    """Whether a pair of MuJoCo geom types takes a margin on the forest engine (the converter's rule)."""
    t = tuple(sorted((int(t1), int(t2))))
    return t in SUPPORTED or t in CONVEX


def active(m):
    """Whether model m has a pair with a nonzero margin (the margin build)."""
    return bool(m.npair) and bool(np.any(np.asarray(m.pair_margin) != 0))


def problems(m):
    """[(pair name, reason)] of the margins the forest engine does not take (empty: none, or all supported). The
    build options that refuse margins are checked by check()."""
    out = []
    for p in range(m.npair):
        mg, gap = float(m.pair_margin[p]), float(m.pair_gap[p])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_PAIR, p) or str(p)
        if hasattr(m, "pair_adhesion") and float(m.pair_adhesion[p]) != 0:
            out.append((name, f"adhesion {float(m.pair_adhesion[p])} (the engine has no contact adhesion)"))
        elif mg < 0 or gap < 0:
            out.append((name, f"negative margin / gap ({mg}, {gap})"))
        elif (mg != 0 or gap != 0) and not supported_pair(m, p):
            out.append((name, f"margin {mg} / gap {gap} on a {'-'.join(map(_tname, pair_types(m, p)))} pair (no "
                              "margin on this pair type)"))
    from nablaengine.forest_engine import margin_convex
    return out + margin_convex.problems(m)


def check(m, terrain=None, surrogate=False, noslip=False, midphase=False):
    """ValueError where the margins of model m do not fit the forest build (no margins: nothing)."""
    if not active(m) and not problems(m):
        return
    bad = problems(m)
    if bad:
        raise ValueError(f"contact margins the forest engine does not take: {bad[:6]}{' ...' if len(bad) > 6 else ''}")
    from nablaengine.forest_engine import step
    opts = [k for k, v in (("a terrain build", terrain is not None), ("contact_surrogate", surrogate),
                           ("noslip (opt.noslip_iterations > 0)", noslip), ("the collision mid-phase", midphase),
                           (f"more than {step.GROUP_PAIRS} pairs (the two-level broad phase)",
                            m.npair > step.GROUP_PAIRS)) if v]
    if opts:
        raise ValueError(f"contact margins (pair_margin != 0): not with {', '.join(opts)} (forest_engine.margin)")


def constants(m):
    """([c_pmarg as float32, then forest_engine.margin_convex's arrays], their layout entries) for
    adjoint.engine_constants; None without margins."""
    if not active(m):
        return None
    from nablaengine.forest_engine import margin_convex
    arrs, lays = [np.asarray(m.pair_margin, np.float64).astype(np.float32)], [("c_pmarg", "f", int(m.npair), 1)]
    cv = margin_convex.constants(m)
    if cv is not None:
        arrs, lays = arrs + cv[0], lays + cv[1]
    return np.concatenate(arrs), lays


DECL = ("DKC float c_pmarg[NPAIR];  // the pairs' contact margins (forest_engine.margin)\n"
        "#define PMARGIN(pr) (c_pmarg[pr])\n")
_DECL_AT = "DKC int c_gbody["
_ADD_HEAD = ("__device__ __forceinline__ bool add_contact(S& s, int pair, int slot, float dist, const float* pos,\n"
             "                                            const float* frame) {\n")
_ADD_OLD = "  s.con_dist[c] = dist;\n"
_ADD_NEW = "  s.con_dist[c] = RSUB(dist, PMARGIN(pair));  // (forest_engine.margin: d - includemargin)\n"
_BOUND = (("    return dot3(dif, n) <= RFMA(c_grbound[g2], 1.f + 1e-5f, 1e-6f);\n",
           "    return dot3(dif, n) <= RFMA(RADD(c_grbound[g2], PMARGIN(pr)), 1.f + 1e-5f, 1e-6f);  // (margin)\n"),
          ("  float rb = RADD(c_grbound[g1], c_grbound[g2]);\n  return dot3(dif, dif) <=",
           "  float rb = RADD(RADD(c_grbound[g1], c_grbound[g2]), PMARGIN(pr));  // (margin)\n"
           "  return dot3(dif, dif) <="))
_CP_HEAD = "__device__ DK_INL_COLLIDE void collide_pair(S& s, int pair) {\n"
_CP_EDITS = (("    if (dot3_rn(dif, n) > c_grbound[g2]) return;",
              "    if (dot3_rn(dif, n) > RADD(c_grbound[g2], PMARGIN(pair))) return;  // (margin)"),
             ("rb = RADD(c_grbound[g1], c_grbound[g2]);\n    if (dot3_rn(dif, dif) > RMUL(rb, rb)) return;",
              "rb = RADD(RADD(c_grbound[g1], c_grbound[g2]), PMARGIN(pair));  // (margin)\n"
              "    if (dot3_rn(dif, dif) > RMUL(rb, rb)) return;"))
_THRESH = re.compile(r"if \((dist|d4\[i\]) < 0\.f\) (add_contact|add_sphere_contact)\(s, pair, ")
_PARALLEL = "        if (dist <= 0.f) {\n"
_REPLAY = re.compile(r"\n( *)const int br = adj_pair_geom(_aux)?\(pair, slot, (s\.taux\[c\], )?s\.gx, s\.gq, &d, p, "
                     r"fr, &nu, nullptr, nullptr[^\n]*\);\n")


def _sub1(src, a, b, what):
    k = src.count(a)
    if k != 1:
        raise RuntimeError(f"forest_engine.margin: {what}: anchor found {k} times: {a[:80]!r}")
    return src.replace(a, b)


def engine(src, m):
    """The final engine text (adjoint.cuda_source, last) with the margins; unchanged without them."""
    if not active(m):
        return src
    src = _sub1(src, _DECL_AT, DECL + _DECL_AT, "declaration")
    i = src.index(_ADD_HEAD) if src.count(_ADD_HEAD) == 1 else -1
    if i < 0:
        raise RuntimeError("forest_engine.margin: add_contact's head")
    j = src.index("\n}\n", i)
    body = _sub1(src[i:j], _ADD_OLD, _ADD_NEW, "add_contact's distance")
    src = src[:i] + body + src[j:]
    for a, b in _BOUND:
        src = _sub1(src, a, b, "pair_bound")
    i = src.index(_CP_HEAD) if src.count(_CP_HEAD) == 1 else -1
    if i < 0:
        raise RuntimeError("forest_engine.margin: collide_pair's head")
    j = src.index("\n}\n", i)
    fn = src[i:j]
    for a, b in _CP_EDITS:
        fn = _sub1(fn, a, b, "collide_pair's culls")
    n = len(_THRESH.findall(fn))
    if n < 1:
        raise RuntimeError("forest_engine.margin: collide_pair's contact tests")
    fn = _THRESH.sub(lambda mt: f"if ({mt.group(1)} < PMARGIN(pair)) {mt.group(2)}(s, pair, ", fn)
    if fn.count(_PARALLEL) > 1:
        raise RuntimeError("forest_engine.margin: capsule-capsule's parallel candidates")
    fn = fn.replace(_PARALLEL, "        if (dist <= PMARGIN(pair)) {  // (margin: MuJoCo Warp's capsule_capsule)\n")
    left = re.findall(r"[^\n]*(?:< 0\.f|<= 0\.f)\)[^\n]*add_(?:sphere_)?contact\(s, pair[^\n]*", fn)
    if left:
        raise RuntimeError(f"forest_engine.margin: collide_pair's contact tests without the margin: {left[:3]}")
    src = src[:i] + fn + src[j:]
    hits = _REPLAY.findall(src)
    if "adj_contact_one(" in src:
        if len(hits) != 1:
            raise RuntimeError(f"forest_engine.margin: adj_contact_one's replay found {len(hits)} times")
        src = _REPLAY.sub(lambda mt: mt.group(0) + f"{mt.group(1)}d = RSUB(d, PMARGIN(pair));  // (forest_engine."
                          "margin: the forward's d - includemargin)\n", src)
    from nablaengine.forest_engine import margin_convex
    return margin_convex.engine(src, m)

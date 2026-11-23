"""Contact margins on the forest engine's convex pairs (forest_engine.margin's second part, MuJoCo Warp 3.17's margin
path; mujoco_warp/_src/collision_gjk.py, collision_convex.py, collision_primitive.py, Copyright 2025 The Newton
Developers, Apache License 2.0)."""
import os
import re

import numpy as np

from nablaengine.forest_engine import margin as _margin


def _pairs(m):
    from nablaengine.forest_engine import step
    geoms = step.collision_geoms(m)
    gid = {g: i for i, g in enumerate(geoms)}
    out = []
    for p in range(m.npair):
        mg = float(m.pair_margin[p])
        if mg == 0 or _margin.pair_types(m, p) not in _margin.CONVEX:
            continue
        g1, g2 = step.pair_order(m, int(m.pair_geom1[p]), int(m.pair_geom2[p]))
        out.append((p, gid[g1], gid[g2], mg))
    return out


def active(m):
    """A convex pair with a nonzero margin (this module's edits)."""
    return _margin.convex_active(m)


def problems(m):
    """[(pair name, reason)] of the convex-pair margins this module does not take."""
    import mujoco
    if not active(m):
        return []
    out = []
    if os.environ.get("DK_CCD_FLOAT", "0") not in ("", "0"):
        out.append(("*", "DK_CCD_FLOAT (the float forward experiment) has no margins"))
    G = mujoco.mjtGeom
    from nablaengine.forest_engine import model_hooks
    bb = (int(G.mjGEOM_BOX), int(G.mjGEOM_BOX))
    sbb = None
    seen = {}
    for p in range(m.npair):
        mg = float(m.pair_margin[p])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_PAIR, p) or str(p)
        key = tuple(sorted((int(m.pair_geom1[p]), int(m.pair_geom2[p]))))
        if key in seen and seen[key] != np.float32(mg) and _margin.pair_types(m, p) in _margin.CONVEX:
            out.append((name, "two pairs of the same geoms with different margins"))
        seen[key] = np.float32(mg)
        if mg != 0 and _margin.pair_types(m, p) == bb and not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD):
            if sbb is None:
                sbb = model_hooks.sbb_boxbox(m)
            if sbb:
                out.append((name, "box-box margin on a G1-class model (nablaengine.ccd's box-box path: no margin)"))
    return out


def constants(m):
    """([c_cvmk, c_cvmv] as float32 arrays, their layout entries) or None without convex-pair margins."""
    if not active(m):
        return None
    from nablaengine.forest_engine import step
    ng = len(step.collision_geoms(m))
    kv = {}
    for _, g1, g2, mg in _pairs(m):
        for a, b in ((g1, g2), (g2, g1)):
            kv[a * ng + b] = np.float32(mg)
    keys = sorted(kv)
    assert keys[-1] < (1 << 24), "forest_engine.margin_convex: geom keys beyond float32's integers"
    k = np.asarray(keys, np.float64).astype(np.float32)
    v = np.asarray([kv[x] for x in keys], np.float32)
    return [k, v], [("c_cvmk", "i", len(keys), 1), ("c_cvmv", "f", len(keys), 1)]


def _pmesh_max(m):
    import mujoco
    G = mujoco.mjtGeom
    t = (int(G.mjGEOM_PLANE), int(G.mjGEOM_MESH))
    return max([float(np.float32(m.pair_margin[p])) for p in range(m.npair) if _margin.pair_types(m, p) == t]
               + [0.0])


def decl(m):
    """The declarations and cvm_margin (inserted after forest_engine.margin's)."""
    from nablaengine.forest_engine import step
    ng = len(step.collision_geoms(m))
    n = 2 * len(_pairs(m))
    pm = f"{np.float32(_pmesh_max(m)):.9g}"
    pm = pm if ("." in pm or "e" in pm) else pm + ".0"
    return (f"#define CVM_N {n}\n#define CVM_NG {ng}\n#define CVM_PMESH_MAX {pm}f\n"
            "DKC int c_cvmk[CVM_N];    // the convex pairs' geom keys g1 * CVM_NG + g2, both orders, sorted\n"
            "DKC float c_cvmv[CVM_N];  // their margins (forest_engine.margin_convex)\n"
            "// the margin of the pair of collision geoms g1, g2 (0: none)\n"
            "__device__ __forceinline__ float cvm_margin(int g1, int g2) {\n"
            "  const int key = g1 * CVM_NG + g2;\n"
            "  int lo = 0, hi = CVM_N - 1;\n"
            "  while (lo < hi) {\n"
            "    const int mid = (lo + hi) >> 1;\n"
            "    if (c_cvmk[mid] < key) lo = mid + 1; else hi = mid;\n"
            "  }\n"
            "  return c_cvmk[lo] == key ? c_cvmv[lo] : 0.f;\n"
            "}\n"
            "#define CVM_SZ(sz) ((int)((sz) - c_gsize) / 3)  // a collision geom's row from its size pointer\n")


def _sub(src, a, b, what, n=1):
    k = src.count(a)
    if k != n:
        raise RuntimeError(f"forest_engine.margin_convex: {what}: anchor found {k} times (expected {n}): {a[:90]!r}")
    return src.replace(a, b)


_SHRINK_OLD = ("  real fm1 = 0.f, fm2 = 0.f;\n"
               "  if (g1.type == 1 || g1.type == 2) { fm1 = g1.size.x; g1.size.x = 0.f; }\n"
               "  if (g2.type == 1 || g2.type == 2) { fm2 = g2.size.x; g2.size.x = 0.f; }\n")
_SHRINK_NEW = """  const real om1 = g1.margin, om2 = g2.margin;  // (forest_engine.margin_convex: 3.17's gjk_phase with margins)
  real fm1 = 0.f, fm2 = 0.f;
  const bool fc1 = g1.type == 3 || g1.type == 5 || g1.type == 6, fc2 = g2.type == 3 || g2.type == 5 || g2.type == 6;
  const bool hnm = ((fc1 && fc2) || (g1.type == 2 && fc2) || (fc1 && g2.type == 2)) && (om1 > 0.f || om2 > 0.f);
  if (g1.type == 1 || g1.type == 2) { fm1 = g1.size.x + 0.5f * om1; g1.margin = 0.f; g1.size.x = 0.f; }
  else if (hnm) { fm1 = 0.5f * om1; g1.margin = 0.f; }
  if (g2.type == 1 || g2.type == 2) { fm2 = g2.size.x + 0.5f * om2; g2.margin = 0.f; g2.size.x = 0.f; }
  else if (hnm) { fm2 = 0.5f * om2; g2.margin = 0.f; }
"""
_SUPPORT_HEAD = re.compile(r"(__device__ +)SupportPoint cv_support\(const CGeom& g, f3 dir\) \{\n")
_SUPPORT_WRAP = """// (forest_engine.margin_convex) MuJoCo Warp 3.17's support with the geom's margin: a sphere's radius + M / 2, the
// other supports moved by M / 2 along dir
__device__ SupportPoint cv_support(const CGeom& g, f3 dir) {
  if (g.type == 1) {
    SupportPoint sp;
    sp.cached = -1;
    sp.vidx = -1;
    sp.point = g.pos + (g.size.x + 0.5f * g.margin) * dir;
    return sp;
  }
  SupportPoint sp = cv_support0(g, dir);
  if (g.margin > 0.f) sp.point = sp.point + dir * (0.5f * g.margin);
  return sp;
}
"""
_RIM_OLD = "    real sgn = v1i != 0 ? 1.f : -1.f;\n"
_RIM_NEW = """    real sgn = v1i != 0 ? 1.f : -1.f;
    if (g.margin > 0.f) {  // (forest_engine.margin_convex) the margin-inflated vertex projected onto the rim
      f3 diff = v - g.pos;
      real l0 = dt3(cy_col(g.R, 0), diff), l1 = dt3(cy_col(g.R, 1), diff);
      real n2 = l0 * l0 + l1 * l1;
      real scl = n2 >= CV_MINVAL * CV_MINVAL ? g.size.x / t_sqrt(n2) : real(0.f);
      v = rmul(g.R, mk3(scl * l0, scl * l1, -sgn * g.size.y)) + g.pos;
      *vout = v;
    }
"""
_MC_OLD = "  int nc;\n  if (edge1) {  // face1 is an edge"
_MC_NEW = """  // (forest_engine.margin_convex) the margins not in the faces (a capsule's edge has it through x1 / x2)
  const real cm1 = g1.type != 2 ? real(0.5f * g1.margin) : real(0.f), cm2 = g2.type != 2 ? real(0.5f * g2.margin) : real(0.f);
  mb.marg = cm1 + cm2;
  int nc;
  if (edge1) {  // face1 is an edge"""
_MC_END_OLD = "  if (nc < 1) { mb.cset = mb.cset == 2 ? 2 : 0; return 1; }\n  return nc;\n}"
_MC_END_NEW = """  if (nc < 1) { mb.cset = mb.cset == 2 ? 2 : 0; return 1; }
  if (cm1 != 0.f || cm2 != 0.f) {  // (forest_engine.margin_convex) witnesses onto the inflated surfaces
    const f3 nrm = edge2 ? mb.n1[j] : -mb.n2[j];
    for (int k = 0; k < nc; k++) {
      if (cm1 != 0.f) w1[k] = w1[k] + cm1 * nrm;
      if (cm2 != 0.f) w2[k] = w2[k] - cm2 * nrm;
      dists[k] = dists[k] - mb.marg;
    }
  }
  return nc;
}"""


def _cylccd_instance(seg):
    seg = _sub(seg, "  int mesh;   // type 6: the packed tables' mesh (g1_engine.cylccd)\n};",
               "  int mesh;   // type 6: the packed tables' mesh (g1_engine.cylccd)\n"
               "  real margin;  // the pair's margin (forest_engine.margin_convex)\n"
               "  int gid;      // the collision geom\n};", "CGeom")
    mt = _SUPPORT_HEAD.search(seg)
    if mt is None or len(_SUPPORT_HEAD.findall(seg)) != 1:
        raise RuntimeError("forest_engine.margin_convex: cv_support's head")
    i = mt.start()
    j = seg.index("\n}\n", i) + 3
    seg = seg[:i] + seg[i:j].replace("SupportPoint cv_support(", "SupportPoint cv_support0(", 1) + _SUPPORT_WRAP + seg[j:]
    disc = "{ return (t1 == 3 || t1 == 6) && (t2 == 3 || t2 == 6); }  // (box, the engine's mesh)\n"
    seg = _sub(seg, disc, disc + "__device__ __forceinline__ bool cy_disc(const CGeom& a, const CGeom& b) {  // (margins 0: "
               "forest_engine.margin_convex)\n  return is_discrete(a.type, b.type) && a.margin == 0.f && b.margin == 0.f;\n"
               "}\n", "is_discrete")
    seg = _sub(seg, "is_discrete(g1.type, g2.type)", "cy_disc(g1, g2)", "is_discrete calls", 4)
    seg = _sub(seg, _SHRINK_OLD, _SHRINK_NEW, "gjk_phase's shrink")
    seg = _sub(seg, "if (rec && !use && r.dist != CV_FLOAT_MAX) cy_record_gjk(",
               "if (rec && !use && r.dist != CV_FLOAT_MAX && g1.margin == 0.f && g2.margin == 0.f) cy_record_gjk(",
               "kind 2's record")
    seg = _sub(seg, "g1.size = s1; g2.size = s2;", "g1.size = s1; g2.size = s2; g1.margin = om1; g2.margin = om2;",
               "gjk_phase's restore", 2)
    seg = _sub(seg, "  o.n = 0; o.unsupported = 0;\n  if (rec && !use) rec->kind = 0;\n",
               "  o.n = 0; o.unsupported = 0;\n  if (rec && !use) rec->kind = 0;\n"
               "  g1.margin = g2.margin = real(cvm_margin(g1.gid, g2.gid));  // (forest_engine.margin_convex)\n",
               "cy_pair's head")
    seg = _sub(seg, "  if (!(dist < 0.f)) return;\n  const bool f1 = ",
               "  if (!(dist < 0.f)) return;\n  dist = dist + g1.margin;  // (forest_engine.margin_convex: the true "
               "distance)\n  const bool f1 = ", "cy_pair's distance")
    seg = _sub(seg, "else if (!needs_epa && r.dim > 0) {", "else if (!needs_epa && dist < g1.margin && r.dim > 0) {",
               "cy_pair's simplex face")
    seg = _sub(seg, "    const real di = nc > 1 ? dists[i] : dist;\n    if (di < 0.f) sink_put(",
               "    const real di = nc > 1 ? dists[i] + g1.margin : dist;\n    if (di < g1.margin) sink_put(",
               "cy_pair's contacts")
    seg = _sub(seg, "  int cset;\n};", "  int cset;\n  real marg;  // the clip's margin (forest_engine.margin_convex)\n};",
               "CyBuf")
    seg = _sub(seg, "    if (dt3(poly[i] - face1[0], n) <= 0.f) {", "    if (dt3(poly[i] - face1[0], n) <= mb.marg) {",
               "polygon_clip's pruning")
    seg = _sub(seg, _RIM_OLD, _RIM_NEW, "cylinder_edge_normals")
    seg = _sub(seg, _MC_OLD, _MC_NEW, "multicontact's margins")
    seg = _sub(seg, _MC_END_OLD, _MC_END_NEW, "multicontact's witnesses")
    return seg


_NS = re.compile(r"namespace (cyc[a-z]) \{  // g1_engine\.cylccd over \w+\n")


def _cylccd(src):
    out, pos, n = [], 0, 0
    for mt in _NS.finditer(src):
        if mt.start() < pos:
            continue
        end = src.index(f"}}  // namespace {mt.group(1)}\n", mt.end())
        out += [src[pos:mt.start()], _cylccd_instance(src[mt.start():end])]
        pos = end
        n += 1
    out.append(src[pos:])
    return "".join(out), n


_CGEOM = "  c.type = t;\n  c.index = -1;\n  return c;\n}"
_OBB = "  const float h1[3] = {a1[3], a1[4], a1[5]}, h2[3] = {a2[3], a2[4], a2[5]};\n"
_OBB_NEW = ("  const float om = 0.5f * cvm_margin(g1, g2);  // (forest_engine.margin_convex: the boxes widened by M / 2)\n"
            "  const float h1[3] = {a1[3] + om, a1[4] + om, a1[5] + om}, h2[3] = {a2[3] + om, a2[4] + om, a2[5] + om};\n")
_CB = "    if (dd.v < 0.f) {\n      float pos[3] = {pp[0].v, pp[1].v, pp[2].v}, n[3] = {nn[0].v, nn[1].v, nn[2].v}, fr[9];"
_CB_NEW = ("    if (dd.v < cvm_margin(CVM_SZ(sz1), CVM_SZ(sz2))) {  // (forest_engine.margin_convex)\n"
           "      float pos[3] = {pp[0].v, pp[1].v, pp[2].v}, n[3] = {nn[0].v, nn[1].v, nn[2].v}, fr[9];")
_BB_FWD = (("    const int nb = bb_box_box_t<RF>(P1, Q1, s1, P2, Q2, s2, 0.f, cd, cp, cn);",
            "    const float mgb = cvm_margin(CVM_SZ(s1), CVM_SZ(s2));  // (forest_engine.margin_convex)\n"
            "    const int nb = bb_box_box_t<RF>(P1, Q1, s1, P2, Q2, s2, mgb, cd, cp, cn);"),
           ("        if (cd[i].v < 0.f) {", "        if (cd[i].v < mgb) {"))
_BB_SLOT = (("  const int n = bb_box_box_t(p1, R1, s1, p2, R2, s2, 0.f, cd, cp, nrm);\n"
             "  if (slot < 0 || slot >= n || !(cd[slot] < 0.f)) return 0;",
             "  const float mgb = cvm_margin(CVM_SZ(s1), CVM_SZ(s2));  // (forest_engine.margin_convex)\n"
             "  const int n = bb_box_box_t(p1, R1, s1, p2, R2, s2, mgb, cd, cp, nrm);\n"
             "  if (slot < 0 || slot >= n || !(cd[slot] < mgb)) return 0;"),)
_PM_EARLY = "  if (c_mpnum[mi] <= 0 || dist0 > 0.f) return 1;  // (margin 0)"
_PM_EARLY_NEW = "  if (c_mpnum[mi] <= 0 || dist0 > CVM_PMESH_MAX) return 1;  // (forest_engine.margin_convex)"
_PM_HEAD = ("__device__ __noinline__ void plane_mesh_mj(const float* pp, const float* n, int g2i, const float* p2, "
            "const float* q2,\n                                           CSink& o) {")
_PM_HEAD_NEW = ("__device__ __noinline__ void plane_mesh_mj(const float* pp, const float* n, int g2i, const float* p2, "
                "const float* q2,\n                                           CSink& o, int g1i) {  // (g1i: "
                "forest_engine.margin_convex)")
_PM_FWD = "    if (d4[i] < 0.f) add_contact(o, 0, i, d4[i], p4 + 3 * i, fr);"
_PM_FWD_NEW = "    if (d4[i] < cvm_margin(g1i, g2i)) add_contact(o, 0, i, d4[i], p4 + 3 * i, fr);"
_PM_ADJ = "          if (d4[i] < 0.f) cvd::add_contact(o, 0, i, d4[i], p4 + 3 * i, f9);"
_PM_ADJ_NEW = "          if (d4[i] < cvm_margin(g1i, g2i)) cvd::add_contact(o, 0, i, d4[i], p4 + 3 * i, f9);"
_PM_CALL = "      CV_PLANE(p1, n, g2, p2, q2, o);\n"
_PM_CALL_HOST = "      CV_PLANE(g.gx[g1], n, g2, g.gx[g2], g.gq[g2], o);\n"
_HOST_DIST = "    g.con_dist[c] = o.dist[j];\n"


def _has(m, *pairs):
    import mujoco
    G = mujoco.mjtGeom
    want = {tuple(sorted((int(getattr(G, "mjGEOM_" + a)), int(getattr(G, "mjGEOM_" + b))))) for a, b in pairs}
    return any(float(m.pair_margin[p]) != 0 and _margin.pair_types(m, p) in want for p in range(m.npair))


def _nativeccd_off(m):
    import mujoco
    return bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD)


def engine(src, m):
    """The final engine text with the convex pairs' margins (forest_engine.margin.engine, last); unchanged without
    them."""
    if not active(m):
        return src
    src = _sub(src, _margin.DECL, _margin.DECL + decl(m), "declarations")
    ccd = _has(m, ("CAPSULE", "CYLINDER"), ("CYLINDER", "CYLINDER"), ("CYLINDER", "BOX"), ("SPHERE", "ELLIPSOID"),
               ("CAPSULE", "ELLIPSOID"), ("ELLIPSOID", "ELLIPSOID"), ("ELLIPSOID", "CYLINDER"), ("ELLIPSOID", "BOX"),
               *[(t, "MESH") for t in ("SPHERE", "CAPSULE", "ELLIPSOID", "CYLINDER", "BOX", "MESH")]) \
        or (_has(m, ("BOX", "BOX")) and not _nativeccd_off(m))
    if "namespace cycr {  // g1_engine.cylccd over" in src:
        src, n = _cylccd(src)
        if n < 2:
            raise RuntimeError(f"forest_engine.margin_convex: {n} instances of g1_engine.cylccd's text")
        src = _sub(src, _CGEOM, "  c.type = t;\n  c.index = -1;\n  c.gid = gi;\n  c.margin = R(0.f);  // (cy_pair sets the pair's: "
                   "forest_engine.margin_convex)\n  return c;\n}", "cyc_cgeom")
    elif ccd:
        raise RuntimeError("forest_engine.margin_convex: native CCD pairs with margins but no g1_engine.cylccd text")
    if "__device__ bool obb_separated(" in src:
        src = _sub(src, _OBB, _OBB_NEW, "obb_separated", src.count(_OBB) or 1)
    if src.count(_CB):
        src = _sub(src, _CB, _CB_NEW, "cb_contacts", 1)
    elif _has(m, ("SPHERE", "BOX"), ("CAPSULE", "BOX")):
        raise RuntimeError("forest_engine.margin_convex: sphere-box / capsule-box margins without cb_contacts")
    if "bb_box_box_t<RF>(P1, Q1, s1, P2, Q2, s2, 0.f, cd, cp, cn);" in src:
        for a, b in _BB_FWD:
            src = _sub(src, a, b, "box_box's primitive")
        for a, b in _BB_SLOT:
            src = _sub(src, a, b, "bb_slot_t", src.count(a) or 1)
    elif _has(m, ("BOX", "BOX")) and _nativeccd_off(m):
        raise RuntimeError("forest_engine.margin_convex: box-box primitive margins without bb_box_box_t")
    if _PM_HEAD in src:
        src = _sub(src, _PM_EARLY, _PM_EARLY_NEW, "plane_convex's early out", src.count(_PM_EARLY) or 1)
        src = _sub(src, _PM_HEAD, _PM_HEAD_NEW, "plane_mesh_mj")
        src = _sub(src, _PM_FWD, _PM_FWD_NEW, "plane_mesh_mj's contacts")
        src = _sub(src, _PM_ADJ, _PM_ADJ_NEW, "adj_mesh_geom's plane-mesh contacts")
        src = _sub(src, _PM_CALL, _PM_CALL.replace("o);", "o, g1);"), "collide_pair's plane-mesh")
    elif _has(m, ("PLANE", "MESH")):
        raise RuntimeError("forest_engine.margin_convex: plane-mesh margins without plane_mesh_mj")
    return src


def host(text, m):
    """The host kernels' text (host.KERNEL) with the split forward's k_ccd storing d - M (add_contact's) and its
    plane-mesh call with the plane's geom; unchanged without convex-pair margins."""
    if not active(m):
        return text
    text = _sub(text, _HOST_DIST, "    g.con_dist[c] = RSUB(o.dist[j], PMARGIN(pair));  // (forest_engine.margin_convex: "
                "d - includemargin)\n", "k_ccd's distance")
    return _sub(text, _PM_CALL_HOST, _PM_CALL_HOST.replace("o);", "o, g1);"), "k_ccd's plane-mesh")

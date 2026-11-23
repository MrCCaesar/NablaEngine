"""Ellipsoid contacts of the forest engine (model_hooks): every pair MuJoCo 3.17 collides with an ellipsoid among the
forest engine's geom types.

- plane-ellipsoid (1 contact): MuJoCo Warp's primitive (mujoco_warp/_src/collision_primitive_core.py plane_ellipsoid;
  Copyright 2025 The Newton Developers, Apache License 2.0) in its order of operations, written once as a template
  over the scalar type: the forward and the adjoint's replay call the same __noinline__ float instance (ell_plane, the
  rotation matrix from ell_mat: the replay computes the forward's bits), the VJP replays the contact over duals
  (nablaengine.dual D1) along the ellipsoid's 6 pose directions (position; rotation about the world axes about its
  centre), as g1_engine.cylinder's plane-cylinder.
- sphere-, capsule-, ellipsoid-, cylinder-, box- and mesh-ellipsoid (1 contact): MuJoCo Warp 3.17's convex collision
  (GJK, EPA; an ellipsoid has neither faces nor edges, so no multicontact) as g1_engine.cylccd ports it, with MuJoCo
  Warp's ellipsoid support (collision_gjk.py support: R normalize(size * R'd) * size + pos; not discrete) added to its
  text: the forward over RF past the oriented-bounding-box test, the adjoint's replay over D1 (cyc_adj, the full
  replay: the support is smooth). mesh-ellipsoid is one of model_hooks.convex_mj_hooks' mesh pairs (CV_PAIR, ADJ_CCD);
  the others get the forward branch here and widen ADJ_CCD's condition."""
import mujoco

GEOM = 7

CCD_COND = "((t2 == 7 && (t1 == 1 || t1 == 2 || t1 == 7)) || (t1 == 7 && (t2 == 5 || t2 == 3)))"


def _types(m):
    return [tuple(sorted((int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]]))))
            for p in range(m.npair)]


def has_ellipsoid(m):
    """An ellipsoid in a contact pair."""
    ell = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    return any(ell in t for t in _types(m))


def has_plane(m):
    """A plane-ellipsoid pair."""
    G = mujoco.mjtGeom
    return (int(G.mjGEOM_PLANE), int(G.mjGEOM_ELLIPSOID)) in _types(m)


def ccd_pairs(m):
    """A sphere-, capsule-, ellipsoid-, cylinder- or box-ellipsoid pair (MuJoCo's convex collision, CCD_COND)."""
    G = mujoco.mjtGeom
    ell = int(G.mjGEOM_ELLIPSOID)
    want = {tuple(sorted((ell, int(t)))) for t in (G.mjGEOM_SPHERE, G.mjGEOM_CAPSULE, G.mjGEOM_ELLIPSOID,
                                                    G.mjGEOM_CYLINDER, G.mjGEOM_BOX)}
    return any(t in want for t in _types(m))


def check(m):
    """The ellipsoid pairs of model m the engine has none of (empty: all are supported): an ellipsoid against a geom
    type other than a plane, sphere, capsule, ellipsoid, cylinder, box or mesh."""
    G = mujoco.mjtGeom
    ell = int(G.mjGEOM_ELLIPSOID)
    ok = {tuple(sorted((int(t), ell))) for t in (G.mjGEOM_PLANE, G.mjGEOM_SPHERE, G.mjGEOM_CAPSULE, G.mjGEOM_ELLIPSOID,
                                                  G.mjGEOM_CYLINDER, G.mjGEOM_BOX, G.mjGEOM_MESH)}
    return sorted({t for t in _types(m) if ell in t and t not in ok})


DEVICE = r"""
// ellipsoid contacts (forest_engine.ellipsoid): MuJoCo Warp's plane_ellipsoid over a scalar type
// plane (point pp, unit normal n) and ellipsoid (centre c, rotation R row-major, radii sz): dist, pos
template <class T>
__device__ void ell_plane_t(const T* pp, const T* n, const T* c, const T* R, const T* sz, T* dist, T* pos) {
  T s[3];
  for (int k = 0; k < 3; k++) s[k] = (R[k] * n[0] + R[3 + k] * n[1] + R[6 + k] * n[2]) * sz[k];  // (R' n) * size
  T l = t_sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
  for (int k = 0; k < 3; k++) s[k] = (l > 0.f ? -(s[k] / l) : T(0.f)) * sz[k];  // -normalize(.) * size
  for (int k = 0; k < 3; k++) pos[k] = c[k] + (R[3 * k] * s[0] + R[3 * k + 1] * s[1] + R[3 * k + 2] * s[2]);
  T d[3] = {pos[0] - pp[0], pos[1] - pp[1], pos[2] - pp[2]};
  *dist = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
  for (int k = 0; k < 3; k++) pos[k] = pos[k] - n[k] * *dist * 0.5f;
}
// the forward's and the replay's single float copies: the geom's rotation matrix (columns: q rotating the axes), the
// contact
__device__ __noinline__ void ell_mat(const float* q, float* R) {
  for (int k = 0; k < 3; k++) {
    float e[3] = {0.f, 0.f, 0.f}, col[3];
    e[k] = 1.f;
    qrot(q, e, col);
    for (int r = 0; r < 3; r++) R[3 * r + k] = col[r];
  }
}
__device__ __noinline__ void ell_plane(const float* pp, const float* n, const float* c, const float* R, const float* sz,
                                       float* dist, float* pos) {
  ell_plane_t<float>(pp, n, c, R, sz, dist, pos);
}

"""

FWD_PLANE = """    else if (t2 == 7) {  // plane-ellipsoid (forest_engine.ellipsoid)
      float R[9], fr[9], dist, pos[3];
      ell_mat(q2, R);
      ell_plane(p1, n, p2, R, sz2, &dist, pos);
      make_frame(n, fr);
      if (dist < 0.f) add_contact(s, pair, 0, dist, pos, fr);
    }
"""

ADJ_PLANE = """    if (t2 == 7) {  // plane-ellipsoid (forest_engine.ellipsoid): the forward's contact, then its VJP over duals (the
      // ellipsoid's 6 pose directions; the frame is the plane's)
      float R[9];
      ell_mat(q2, R);
      ell_plane(p1, n, p2, R, sz2, dist, pos);
      make_frame(n, fr);
      if (!(-0.5f < n[1] && n[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
      if (cot) {
        D1 P[3], N[3], S[3];
        for (int t = 0; t < 3; t++) { P[t] = D1(p1[t]); N[t] = D1(n[t]); S[t] = D1(sz2[t]); }
        for (int j = 0; j < 6; j++) {
          float ea[3] = {0.f, 0.f, 0.f};
          ea[j % 3] = 1.f;
          D1 C[3], RR[9], dd, pp[3];
          for (int t = 0; t < 3; t++) C[t] = D1(p2[t], j < 3 && t == j ? 1.f : 0.f);
          for (int k = 0; k < 3; k++) {  // column k rotates: ea x col_k
            float col[3] = {R[k], R[3 + k], R[6 + k]}, tc[3];
            cross(ea, col, tc);
            for (int r = 0; r < 3; r++) RR[3 * r + k] = D1(R[3 * r + k], j >= 3 ? tc[r] : 0.f);
          }
          ell_plane_t<D1>(P, N, C, RR, S, &dd, pp);
          float gj = cot[0] * dd.d;
          for (int t = 0; t < 3; t++) gj += cot[1 + t] * pp[t].d;
          gb[6 + j] = gj;
        }
      }
      return br;
    }
"""

SUPPORT = """  if (g.type == 7) {  // ellipsoid (forest_engine.ellipsoid: MuJoCo Warp 3.17's support)
    f3 res = nz3(mk3(ld.x * g.size.x, ld.y * g.size.y, ld.z * g.size.z));
    sp.point = rmul(g.R, mk3(res.x * g.size.x, res.y * g.size.y, res.z * g.size.z)) + g.pos;
    return sp;
  }
"""
SUPPORT_ANCHOR = "  if (g.type == 5) {  // cylinder (g1_engine.cylccd: MuJoCo Warp 3.17's support)\n"


def support_text(dev):
    """cylccd's device text dev with the ellipsoid's support in each instance of cv_support."""
    n = dev.count(SUPPORT_ANCHOR)
    if n < 1:
        raise RuntimeError("ellipsoid: no cylinder support in cylccd's text")
    return dev.replace(SUPPORT_ANCHOR, SUPPORT + SUPPORT_ANCHOR)


def fwd_ccd(fn="cyc_pair_r"):
    """The forward's branch for CCD_COND (collide_pair): past the OBB test (forest_engine.convex), the contact of
    cylccd's forward (fn: its RF wrapper, or model_hooks' DK_CCD_FLOAT experiment's)."""
    return ("  if " + CCD_COND + " {  // sphere-, capsule-, ellipsoid-, cylinder-, box-ellipsoid (forest_engine.ellipsoid:\n"
            "    // g1_engine.cylccd's MuJoCo Warp 3.17 CCD), past the OBB test\n"
            "    if (obb_separated(g1, p1, q1, g2, p2, q2)) return;\n"
            "    CSink o;\n"
            f"    {fn}(t1, g1, p1, q1, sz1, t2, g2, p2, q2, sz2, o);\n"
            "    sink_flush(s, pair, o);\n"
            "    return;\n  }\n")


def hooks(m, part, fn="cyc_pair_r"):
    """The (anchor, replacement, count) edits of the forest text of model m (part "engine" or "forward") for its
    ellipsoid pairs (none without an ellipsoid in a pair): plane-ellipsoid's forward and adjoint, the CCD pairs'
    forward branch (fwd_ccd; their device code, the support and the adjoint: model_hooks.convex_mj_hooks)."""
    if part not in ("engine", "forward") or not has_ellipsoid(m):
        return ()
    bad = check(m)
    if bad:
        raise ValueError(f"ellipsoid pairs the forest engine does not have (MuJoCo geom types): {bad}")
    out = ()
    if has_plane(m):
        from nablaengine.dual import DUAL_DEVICE
        plane_anchor = "#if HAS_BOX\n    else if (t2 == 3) {\n"
        dev_anchor = ("// box-box (MuJoCo Warp's box_box, margin 0): separating-axis test, then face-vertex or edge-edge "
                      "clipping\n")
        out += ((plane_anchor, FWD_PLANE + plane_anchor, 1), (dev_anchor, DUAL_DEVICE + DEVICE + dev_anchor, 1))
        if part == "engine":
            out += (("    if (t2 != 2) return -1;\n", ADJ_PLANE + "    if (t2 != 2) return -1;\n", 1),)
    if ccd_pairs(m):
        anchor = "  atomicOr(&s.unsupported, 1);  // other type combinations do not occur in the G1 model\n"
        out += ((anchor, fwd_ccd(fn) + anchor, 1),)
    return out

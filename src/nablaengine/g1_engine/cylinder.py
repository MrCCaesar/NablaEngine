"""Cylinder contacts of the G1 engine (model_hooks): plane-cylinder and sphere-cylinder, MuJoCo Warp's primitives
(mujoco_warp/_src/collision_primitive_core.py plane_cylinder, sphere_cylinder; Copyright 2025 The Newton Developers,
Apache License 2.0) in their order of operations, so that the contacts agree with MuJoCo Warp's up to float rounding."""
import mujoco

GEOM = 5

DEVICE = r"""
// cylinder contacts (g1_engine.cylinder): MuJoCo Warp's plane_cylinder and sphere_cylinder over a scalar type
template <class T>
__device__ __forceinline__ T cyl_dot(const T* a, const T* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
template <class T>
__device__ __forceinline__ T cyl_safe_div(T x, T y) { return x / (y != 0.f ? y : T(1e-15f)); }

// plane (point pp, unit normal n) and cylinder (centre c, unit axis, radius r, half height hh): dist[4], pos[4][3]
template <class T>
__device__ void cyl_plane_t(const T* pp, const T* n, const T* c, const T* axis_in, T r, T hh, T* dist, T* pos) {
  T ax[3] = {axis_in[0], axis_in[1], axis_in[2]};
  T prjaxis = cyl_dot(n, ax);
  if (prjaxis > 0.f) {  // the axis towards the plane
    for (int k = 0; k < 3; k++) ax[k] = -ax[k];
    prjaxis = -prjaxis;
  }
  T dc[3] = {c[0] - pp[0], c[1] - pp[1], c[2] - pp[2]};
  T dist0 = cyl_dot(dc, n);
  T vec[3];
  for (int k = 0; k < 3; k++) vec[k] = ax[k] * prjaxis - n[k];  // -n without its axis component
  T len_sqr = cyl_dot(vec, vec);
  if (len_sqr >= 1e-12f) {
    T s = cyl_safe_div(r, t_sqrt(len_sqr));
    for (int k = 0; k < 3; k++) vec[k] = vec[k] * s;
  } else {  // the axis along the normal: the cylinder's world x
    vec[0] = T(1.f) * r; vec[1] = T(0.f) * r; vec[2] = T(0.f) * r;
  }
  T prjvec = cyl_dot(vec, n);
  for (int k = 0; k < 3; k++) ax[k] = ax[k] * hh;
  prjaxis = prjaxis * hh;
  dist[0] = dist0 + prjaxis + prjvec;  // the nearer cap's rim
  for (int k = 0; k < 3; k++) pos[k] = c[k] + vec[k] + ax[k] - n[k] * (dist[0] * 0.5f);
  dist[1] = dist0 - prjaxis + prjvec;  // the farther cap's rim
  for (int k = 0; k < 3; k++) pos[3 + k] = c[k] + vec[k] - ax[k] - n[k] * (dist[1] * 0.5f);
  T prjvec1 = -prjvec * 0.5f;
  T dist3 = dist0 + prjaxis + prjvec1;
  T vec1[3] = {vec[1] * ax[2] - vec[2] * ax[1], vec[2] * ax[0] - vec[0] * ax[2], vec[0] * ax[1] - vec[1] * ax[0]};
  T l1 = t_sqrt(cyl_dot(vec1, vec1));  // wp.normalize, then scaled by r sqrt(3) / 2
  T s1 = r * t_sqrt(3.f) * 0.5f;
  for (int k = 0; k < 3; k++) vec1[k] = (l1 > 0.f ? vec1[k] / l1 : T(0.f)) * s1;
  dist[2] = dist3;
  for (int k = 0; k < 3; k++) pos[6 + k] = c[k] + vec1[k] + ax[k] - vec[k] * 0.5f - n[k] * (dist3 * 0.5f);
  dist[3] = dist3;
  for (int k = 0; k < 3; k++) pos[9 + k] = c[k] - vec1[k] + ax[k] - vec[k] * 0.5f - n[k] * (dist3 * 0.5f);
}

// sphere (centre sp, radius sr) and cylinder (centre c, unit axis, radius cr, half height hh): dist, pos, normal n
// (sphere to cylinder); returns the centre distance of the sphere core (the tape's min_nu), infinity on a cap
template <class T>
__device__ T cyl_sphere_t(const T* sp, T sr, const T* c, const T* axis, T cr, T hh, T* dist, T* pos, T* n) {
  T vec[3] = {sp[0] - c[0], sp[1] - c[1], sp[2] - c[2]};
  T x = cyl_dot(vec, axis);
  T a_proj[3], p_proj[3];
  for (int k = 0; k < 3; k++) { a_proj[k] = axis[k] * x; p_proj[k] = vec[k] - a_proj[k]; }
  T p_proj_sqr = cyl_dot(p_proj, p_proj);
  bool side = t_fabs(x) < hh, cap = p_proj_sqr < cr * cr;
  if (side && cap) {
    T dist_cap = hh - t_fabs(x), dist_radius = cr - t_sqrt(p_proj_sqr);
    if (dist_cap < dist_radius) side = false; else cap = false;
  }
  T target[3], r2;
  if (side) {  // the axis point level with the sphere, radius cr
    for (int k = 0; k < 3; k++) target[k] = c[k] + a_proj[k];
    r2 = cr;
  } else if (cap) {  // plane_sphere on the cap's plane, the normal flipped
    T sg = x > 0.f ? T(1.f) : T(-1.f);
    T pc[3], pn[3];
    for (int k = 0; k < 3; k++) { pn[k] = axis[k] * sg; pc[k] = c[k] + axis[k] * (sg * hh); }
    T d[3] = {sp[0] - pc[0], sp[1] - pc[1], sp[2] - pc[2]};
    *dist = cyl_dot(d, pn) - sr;
    for (int k = 0; k < 3; k++) { pos[k] = sp[k] - pn[k] * (sr + 0.5f * *dist); n[k] = -pn[k]; }
    return T(__int_as_float(0x7f800000));
  } else {  // the rim's corner
    T inv_len = cyl_safe_div(T(1.f), t_sqrt(p_proj_sqr));
    T sx = x < 0.f ? T(-1.f) : T(1.f);  // wp.sign
    for (int k = 0; k < 3; k++) {
      p_proj[k] = p_proj[k] * (cr * inv_len);
      target[k] = c[k] + axis[k] * (sx * hh) + p_proj[k];
    }
    r2 = T(0.f);
  }
  T dir[3] = {target[0] - sp[0], target[1] - sp[1], target[2] - sp[2]};  // sphere_sphere(sp, sr, target, r2)
  T d = t_sqrt(cyl_dot(dir, dir));
  if (d == 0.f) { n[0] = T(1.f); n[1] = T(0.f); n[2] = T(0.f); }
  else for (int k = 0; k < 3; k++) n[k] = dir[k] / d;
  *dist = d - (sr + r2);
  for (int k = 0; k < 3; k++) pos[k] = sp[k] + n[k] * (sr + 0.5f * *dist);
  return d;
}

// the forward's and the replay's single float copies
__device__ __noinline__ void cyl_plane(const float* pp, const float* n, const float* c, const float* axis, float r,
                                       float hh, float* dist, float* pos) {
  cyl_plane_t<float>(pp, n, c, axis, r, hh, dist, pos);
}
__device__ __noinline__ float cyl_sphere(const float* sp, float sr, const float* c, const float* axis, float cr,
                                         float hh, float* dist, float* pos, float* n) {
  return cyl_sphere_t<float>(sp, sr, c, axis, cr, hh, dist, pos, n);
}

"""

FWD_PLANE = """    else if (t2 == 5) {  // plane-cylinder (g1_engine.cylinder)
      float axis[3], fr[9], d4[4], p4[12];
      zaxis_rn(q2, axis);
      cyl_plane(p1, n, p2, axis, sz2[0], sz2[1], d4, p4);
      make_frame(n, fr);
      for (int i = 0; i < 4; i++)
        if (d4[i] < 0.f) add_contact(s, pair, i, d4[i], p4 + 3 * i, fr);
    }
"""

FWD_SPHERE = """  if (t1 == 1 && t2 == 5) {  // sphere-cylinder (g1_engine.cylinder)
    float axis[3], dist, pos[3], n[3];
    zaxis_rn(q2, axis);
    float nu = cyl_sphere(p1, sz1[0], p2, axis, sz2[0], sz2[1], &dist, pos, n);
    if (dist < 0.f) add_sphere_contact(s, pair, 0, nu, dist, pos, n);
    return;
  }
"""

ADJ_PLANE = """    if (t2 == 5) {  // plane-cylinder (g1_engine.cylinder): the forward's contact `slot`, then its VJP over duals (the
      // cylinder's 6 pose directions; the frame is the plane's)
      float axis[3], d4[4], p4[12];
      zaxis_rn(q2, axis);
      cyl_plane(p1, n, p2, axis, sz2[0], sz2[1], d4, p4);
      *dist = d4[slot];
      for (int k = 0; k < 3; k++) pos[k] = p4[3 * slot + k];
      make_frame(n, fr);
      if (!(-0.5f < n[1] && n[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
      if (cot) {
        D1 P[3], N[3];
        for (int t = 0; t < 3; t++) { P[t] = D1(p1[t]); N[t] = D1(n[t]); }
        for (int j = 0; j < 6; j++) {
          float ea[3] = {0.f, 0.f, 0.f}, ta[3];
          ea[j % 3] = 1.f;
          cross(ea, axis, ta);
          D1 C[3], A[3], dd[4], pp[12];
          for (int t = 0; t < 3; t++) {
            C[t] = D1(p2[t], j < 3 && t == j ? 1.f : 0.f);
            A[t] = D1(axis[t], j >= 3 ? ta[t] : 0.f);
          }
          cyl_plane_t<D1>(P, N, C, A, D1(sz2[0]), D1(sz2[1]), dd, pp);
          float gj = cot[0] * dd[slot].d;
          for (int t = 0; t < 3; t++) gj += cot[1 + t] * pp[3 * slot + t].d;
          gb[6 + j] = gj;
        }
      }
      return br;
    }
"""

ADJ_SPHERE = """  if (t1 == 1 && t2 == 5) {  // sphere-cylinder (g1_engine.cylinder): the forward's contact, then its VJP over duals
    // (the sphere's 3 and the cylinder's 6 pose directions)
    float axis[3], cn[3];
    zaxis_rn(q2, axis);
    *nu = cyl_sphere(p1, sz1[0], p2, axis, sz2[0], sz2[1], dist, pos, cn);
    make_frame(cn, fr);
    if (!(-0.5f < cn[1] && cn[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
    if (cot) {
      float nbar[3];
      make_frame_vjp(fr, cot + 4, nbar);
      for (int j = 0; j < 12; j++) {
        const int g = j / 6, jj = j % 6;  // geom g (0: the sphere, 1: the cylinder), direction jj
        if (g == 0 && jj >= 3) continue;  // (a sphere's rotation moves nothing)
        float ea[3] = {0.f, 0.f, 0.f}, ta[3];
        ea[jj % 3] = 1.f;
        cross(ea, axis, ta);
        D1 S[3], C[3], A[3], dd, pp[3], nn[3];
        for (int t = 0; t < 3; t++) {
          S[t] = D1(p1[t], g == 0 && t == jj ? 1.f : 0.f);
          C[t] = D1(p2[t], g == 1 && jj < 3 && t == jj ? 1.f : 0.f);
          A[t] = D1(axis[t], g == 1 && jj >= 3 ? ta[t] : 0.f);
        }
        cyl_sphere_t<D1>(S, D1(sz1[0]), C, A, D1(sz2[0]), D1(sz2[1]), &dd, pp, nn);
        float gj = cot[0] * dd.d;
        for (int t = 0; t < 3; t++) gj += cot[1 + t] * pp[t].d + nbar[t] * nn[t].d;
        gb[j] = gj;
      }
    }
    return br;
  }
"""


def pairs(m):
    """(model m has a pair with a cylinder, every such pair is one the engine has: with a plane or a sphere (this
    module), a capsule, a cylinder or a box (g1_engine.cylccd))."""
    G = mujoco.mjtGeom
    cyl = int(G.mjGEOM_CYLINDER)
    ok = {tuple(sorted((int(t), cyl))) for t in (G.mjGEOM_PLANE, G.mjGEOM_SPHERE, G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER,
                                                  G.mjGEOM_BOX, G.mjGEOM_MESH)}
    with_cyl = [tuple(sorted((int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]]))))
                for p in range(m.npair)]
    with_cyl = [t for t in with_cyl if cyl in t]
    return bool(with_cyl), all(t in ok for t in with_cyl)


def hooks():
    """The (anchor, replacement, count) edits of the engine text for cylinder pairs."""
    from nablaengine.dual import DUAL_DEVICE
    plane_anchor = "#if HAS_BOX\n    else if (t2 == 3) {\n"
    sphere_anchor = "  atomicOr(&s.unsupported, 1);  // other type combinations do not occur in the G1 model\n"
    dev_anchor = "// box-box (MuJoCo Warp's box_box, margin 0): separating-axis test, then face-vertex or edge-edge clipping\n"
    return ((plane_anchor, FWD_PLANE + plane_anchor, 1),
            (sphere_anchor, FWD_SPHERE + sphere_anchor, 1),
            (dev_anchor, DUAL_DEVICE + DEVICE + dev_anchor, 1),
            ("    if (t2 != 2) return -1;\n", ADJ_PLANE + "    if (t2 != 2) return -1;\n", 1),
            ("  if (t1 != 2 || t2 != 2) return -1;\n", ADJ_SPHERE + "  if (t1 != 2 || t2 != 2) return -1;\n", 1))

"""Mesh collision geoms of the G1 engine (model_hooks): plane-mesh contacts, MuJoCo Warp 3.17's plane_convex
(mujoco_warp/_src/collision_primitive.py plane_convex, _hull4, _area4; Copyright 2025 The Newton Developers, Apache
License 2.0) in its order of operations: the mesh's deepest vertex along the plane's normal (exhaustive search below
10 vertices, else hill climbing on the convex hull's graph), then the incident polygon most anti-aligned with the
normal, pruned to at most four vertices keeping the deepest (MuJoCo's hull4), each a contact unless it lies above the
mesh's centre; up to 4 contacts, slots 0 .. 3."""
import mujoco
import numpy as np

GEOM = 6


def _geoms(m):
    from nablaengine.g1_engine import step
    return step.collision_geoms(m)


def tables(m):
    """g1_engine.mesh_tables.mesh_tables of the collision geoms (None without a mesh among them)."""
    from nablaengine.g1_engine import mesh_tables as convex
    return convex.mesh_tables(m, _geoms(m))


def layout(m):
    """[(symbol, kind, length)] of the mesh arrays (empty without mesh collision geoms)."""
    from nablaengine.g1_engine import mesh_tables as convex
    return convex.layout(m, _geoms(m))


def constants(m):
    """The mesh arrays in layout() order (float32; ints as values)."""
    from nablaengine.g1_engine import mesh_tables as convex
    return [a.astype(np.float32) for a in convex.constants(m, _geoms(m))]


def header(m):
    """The mesh arrays' declarations."""
    from nablaengine.g1_engine import mesh_tables as convex
    mt = tables(m)
    s = mt["sizes"]
    return "".join(f"DKC {'int' if k == 'i' else 'float'} {sym}[{s[n]}];  // (g1_engine.mesh)\n"
                   for sym, k, n in convex.MESH_LAYOUT)


def pairs(m):
    """(model m has a pair with a mesh, every such pair is one the engine has: plane-mesh here, a mesh with a sphere,
    capsule, cylinder, box or mesh in g1_engine.cylccd)."""
    G = mujoco.mjtGeom
    mesh = int(G.mjGEOM_MESH)
    ok = {tuple(sorted((int(t), mesh))) for t in (G.mjGEOM_PLANE, G.mjGEOM_SPHERE, G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER,
                                                  G.mjGEOM_BOX, G.mjGEOM_MESH)}
    with_mesh = [tuple(sorted((int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]]))))
                 for p in range(m.npair)]
    with_mesh = [t for t in with_mesh if mesh in t]
    return bool(with_mesh), all(t in ok for t in with_mesh)


DEVICE = r"""
// plane-mesh contacts (g1_engine.mesh): MuJoCo Warp 3.17's plane_convex over a scalar type (mesh data: c_m*)
template <class T>
__device__ __forceinline__ void msh_vert(int v, T* o) {  // vertex v of the packed table (mesh vertex units)
  o[0] = T(c_mvert[3 * v]); o[1] = T(c_mvert[3 * v + 1]); o[2] = T(c_mvert[3 * v + 2]);
}
template <class T>
__device__ __forceinline__ T msh_dot(const T* a, const T* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
template <class T>
__device__ T msh_area4(int va, int fva, int a, int b, int c, int d) {  // _area4 on the polygon's vertices
  T A[3], B[3], C[3], D[3];
  msh_vert<T>(va + c_mpvert[fva + a], A); msh_vert<T>(va + c_mpvert[fva + b], B);
  msh_vert<T>(va + c_mpvert[fva + c], C); msh_vert<T>(va + c_mpvert[fva + d], D);
  T u[3] = {A[0] - C[0], A[1] - C[1], A[2] - C[2]}, w[3] = {B[0] - D[0], B[1] - D[1], B[2] - D[2]};
  T x[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
  return 0.5f * t_sqrt(msh_dot(x, x));
}
template <class T>
__device__ int msh_hull4(int va, int fva, int nhull, int a, int* res) {  // _hull4: keeps anchor a
  int b = (a + 1) % nhull, c = (a + 2) % nhull, d = (a + 3) % nhull;
  res[0] = a; res[1] = b; res[2] = c; res[3] = d;
  if (nhull <= 4) return nhull;
  T m = msh_area4<T>(va, fva, a, b, c, d);
  while (true) {
    int d_next = (d + 1) % nhull;
    T m_next = msh_area4<T>(va, fva, a, b, c, d_next);
    if (m_next <= m) break;
    d = d_next; m = m_next; res[0] = a; res[1] = b; res[2] = c; res[3] = d;
    while (true) {
      int c_next = (c + 1) % nhull;
      m_next = msh_area4<T>(va, fva, a, b, c_next, d);
      if (m_next <= m) break;
      c = c_next; m = m_next; res[0] = a; res[1] = b; res[2] = c; res[3] = d;
    }
    while (true) {
      int b_next = (b + 1) % nhull;
      m_next = msh_area4<T>(va, fva, a, b_next, c, d);
      if (m_next <= m) break;
      b = b_next; m = m_next; res[0] = a; res[1] = b; res[2] = c; res[3] = d;
    }
  }
  return 4;
}
// plane (point pp, unit normal n) and mesh mi at (p2, R2 row major): dist[4], pos[4][3] (unused: 1e10); returns the
// count of filled slots
template <class T>
__device__ int msh_plane_t(const T* pp, const T* n, int mi, const T* p2, const T* R2, T* dist, T* pos) {
  for (int k = 0; k < 4; k++) dist[k] = T(1e10f);
  T ln[3], ld[3];
  for (int k = 0; k < 3; k++) { ln[k] = R2[k] * n[0] + R2[3 + k] * n[1] + R2[6 + k] * n[2]; ld[k] = -ln[k]; }
  const int va = c_mvadr[mi], vn = c_mvnum[mi], ga = c_mgadr[mi];
  int vertindex = -1;
  T maxs = T(-1e10f), v[3];
  if (ga == -1 || vn < 10) {
    for (int i = 0; i < vn; i++) {
      msh_vert<T>(va + i, v);
      T s = msh_dot(ld, v);
      if (s > maxs) { maxs = s; vertindex = i; }
    }
  } else {
    const int* gr = c_mgraph + ga;
    const int numvert = gr[0], vert_edgeadr = 2, vert_globalid = 2 + numvert, edge_localid = 2 + 2 * numvert;
    int prev = -1, imax = 0;
    msh_vert<T>(va + gr[vert_globalid], v);
    maxs = msh_dot(ld, v);
    while (imax != prev) {
      prev = imax;
      int i = gr[vert_edgeadr + imax];
      while (gr[edge_localid + i] >= 0) {
        const int subidx = gr[edge_localid + i], idx = gr[vert_globalid + subidx];
        msh_vert<T>(va + idx, v);
        T s = msh_dot(ld, v);
        if (s > maxs) { maxs = s; imax = subidx; }
        i++;
      }
    }
    vertindex = gr[vert_globalid + imax];
  }
  if (vertindex < 0) return 0;
  T v0[3], d0[3];
  msh_vert<T>(va + vertindex, v);
  for (int k = 0; k < 3; k++) v0[k] = p2[k] + (R2[3 * k] * v[0] + R2[3 * k + 1] * v[1] + R2[3 * k + 2] * v[2]);
  for (int k = 0; k < 3; k++) d0[k] = v0[k] - pp[k];
  T dist0 = msh_dot(n, d0);
  dist[0] = dist0;
  for (int k = 0; k < 3; k++) pos[k] = v0[k] - 0.5f * dist0 * n[k];
  if (c_mpnum[mi] <= 0 || dist0 > 0.f) return 1;  // (margin 0)
  const int pmadr = c_mpmadr[va + vertindex], pmnum = c_mpmnum[va + vertindex], pa = c_mpadr[mi];
  int best_poly = -1;
  T best_dot = T(1.f);
  for (int i = 0; i < pmnum; i++) {
    const int idx = c_mpmap[pmadr + i];
    T pn[3] = {T(c_mpnorm[3 * (pa + idx)]), T(c_mpnorm[3 * (pa + idx) + 1]), T(c_mpnorm[3 * (pa + idx) + 2])};
    T nd = msh_dot(pn, ln);
    if (nd < best_dot) { best_dot = nd; best_poly = idx; }
  }
  if (best_poly < 0) return 1;
  const int fva = c_mpvadr[pa + best_poly], fvn = c_mpvnum[pa + best_poly];
  int a = 0;
  for (int i = 0; i < fvn; i++)
    if (c_mpvert[fva + i] == vertindex) { a = i; break; }
  int ind[4];
  const int nh = msh_hull4<T>(va, fva, fvn, a, ind);
  int ncon = 1;
  for (int i = 1; i < nh; i++) {
    const int vi = c_mpvert[fva + ind[i]];
    T pt[3], dc[3], dp[3];
    msh_vert<T>(va + vi, v);
    for (int k = 0; k < 3; k++) pt[k] = p2[k] + (R2[3 * k] * v[0] + R2[3 * k + 1] * v[1] + R2[3 * k + 2] * v[2]);
    for (int k = 0; k < 3; k++) { dc[k] = pt[k] - p2[k]; dp[k] = pt[k] - pp[k]; }
    if (msh_dot(n, dc) > 0.f) continue;  // (MuJoCo's mesh-centre filter)
    T vd = msh_dot(n, dp);
    dist[ncon] = vd;
    for (int k = 0; k < 3; k++) pos[3 * ncon + k] = pt[k] - 0.5f * vd * n[k];
    ncon++;
  }
  return ncon;
}
// the forward's and the replay's single copy (over RF): dist[4], pos[12] as floats
__device__ __noinline__ void msh_plane(const float* pp, const float* n, int mi, const float* p2, const float* q2,
                                       float* dist, float* pos) {
  float R2[9];
  quat2mat(q2, R2);
  RF P[3], N[3], X[3], R[9], D[4], Q[12];
  for (int k = 0; k < 3; k++) { P[k] = RF(pp[k]); N[k] = RF(n[k]); X[k] = RF(p2[k]); }
  for (int k = 0; k < 9; k++) R[k] = RF(R2[k]);
  msh_plane_t<RF>(P, N, mi, X, R, D, Q);
  for (int k = 0; k < 4; k++) dist[k] = D[k].v;
  for (int k = 0; k < 12; k++) pos[k] = Q[k].v;
}

"""

FWD = """    else if (t2 == 6) {  // plane-mesh (g1_engine.mesh)
      float fr[9], d4[4], p4[12];
      msh_plane(p1, n, c_gmesh[g2], p2, q2, d4, p4);
      make_frame(n, fr);
      for (int i = 0; i < 4; i++)
        if (d4[i] < 0.f) add_contact(s, pair, i, d4[i], p4 + 3 * i, fr);
    }
"""

ADJ = """    if (t2 == 6) {  // plane-mesh (g1_engine.mesh): the forward's contact `slot`, then its VJP over duals (the mesh's 6
      // pose directions; the frame is the plane's)
      const int mi = c_gmesh[(int)(sz2 - c_gsize) / 3];  // (the mesh geom's row of the geom table)
      float d4[4], p4[12], R2[9];
      msh_plane(p1, n, mi, p2, q2, d4, p4);
      *dist = d4[slot];
      for (int k = 0; k < 3; k++) pos[k] = p4[3 * slot + k];
      make_frame(n, fr);
      if (!(-0.5f < n[1] && n[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
      if (cot) {
        quat2mat(q2, R2);
        D1 P[3], N[3];
        for (int t = 0; t < 3; t++) { P[t] = D1(p1[t]); N[t] = D1(n[t]); }
        for (int j = 0; j < 6; j++) {
          float ea[3] = {0.f, 0.f, 0.f};
          ea[j % 3] = 1.f;
          D1 X[3], R[9], dd[4], pp[12];
          for (int t = 0; t < 3; t++) X[t] = D1(p2[t], j < 3 && t == j ? 1.f : 0.f);
          for (int c = 0; c < 3; c++) {  // d col = e x col
            float col[3] = {R2[c], R2[3 + c], R2[6 + c]}, dc[3];
            cross(ea, col, dc);
            for (int r = 0; r < 3; r++) R[3 * r + c] = D1(R2[3 * r + c], j >= 3 ? dc[r] : 0.f);
          }
          msh_plane_t<D1>(P, N, mi, X, R, dd, pp);
          float gj = cot[0] * dd[slot].d;
          for (int t = 0; t < 3; t++) gj += cot[1 + t] * pp[3 * slot + t].d;
          gb[6 + j] = gj;
        }
      }
      return br;
    }
"""


SEARCH_ONCE = (
    ("__device__ int msh_plane_t(const T* pp, const T* n, int mi, const T* p2, const T* R2, T* dist, T* pos) {\n",
     "__device__ int msh_plane_t(const T* pp, const T* n, int mi, const T* p2, const T* R2, T* dist, T* pos,\n"
     "                           int* vid) {  // vid[slot]: the slot's vertex (g1_engine.mesh)\n"),
    ("  T dist0 = msh_dot(n, d0);\n  dist[0] = dist0;\n",
     "  T dist0 = msh_dot(n, d0);\n  dist[0] = dist0;\n  vid[0] = vertindex;\n"),
    ("    T vd = msh_dot(n, dp);\n    dist[ncon] = vd;\n",
     "    T vd = msh_dot(n, dp);\n    dist[ncon] = vd;\n    vid[ncon] = vi;\n"),
    ("__device__ __noinline__ void msh_plane(const float* pp, const float* n, int mi, const float* p2, const float* q2,\n"
     "                                       float* dist, float* pos) {\n",
     "__device__ __noinline__ void msh_plane(const float* pp, const float* n, int mi, const float* p2, const float* q2,\n"
     "                                       float* dist, float* pos, int* vid) {\n"),
    ("  msh_plane_t<RF>(P, N, mi, X, R, D, Q);\n", "  msh_plane_t<RF>(P, N, mi, X, R, D, Q, vid);\n"),
    ("      float fr[9], d4[4], p4[12];\n      msh_plane(p1, n, c_gmesh[g2], p2, q2, d4, p4);\n",
     "      float fr[9], d4[4], p4[12];\n      int vid_[4];\n      msh_plane(p1, n, c_gmesh[g2], p2, q2, d4, p4, vid_);\n"),
    ("// the forward's and the replay's single copy (over RF): dist[4], pos[12] as floats\n",
     "// one contact's distance and position from its vertex vi (g1_engine.mesh: msh_plane_t's expressions for slot 0 and the\n"
     "// polygon's slots, the same operations in the same order)\n"
     "template <class T>\n"
     "__device__ __forceinline__ void msh_slot_t(const T* pp, const T* n, int va, int vi, const T* p2, const T* R2, T* dist,\n"
     "                                           T* pos) {\n"
     "  T v[3], pt[3], dp[3];\n"
     "  msh_vert<T>(va + vi, v);\n"
     "  for (int k = 0; k < 3; k++) pt[k] = p2[k] + (R2[3 * k] * v[0] + R2[3 * k + 1] * v[1] + R2[3 * k + 2] * v[2]);\n"
     "  for (int k = 0; k < 3; k++) dp[k] = pt[k] - pp[k];\n"
     "  T vd = msh_dot(n, dp);\n"
     "  *dist = vd;\n"
     "  for (int k = 0; k < 3; k++) pos[k] = pt[k] - 0.5f * vd * n[k];\n"
     "}\n"
     "// the forward's and the replay's single copy (over RF): dist[4], pos[12] as floats\n"),
    ("      float d4[4], p4[12], R2[9];\n      msh_plane(p1, n, mi, p2, q2, d4, p4);\n",
     "      float d4[4], p4[12], R2[9];\n      int vid[4] = {0, 0, 0, 0};\n      msh_plane(p1, n, mi, p2, q2, d4, p4, vid);\n"),
    ("          D1 X[3], R[9], dd[4], pp[12];\n", "          D1 X[3], R[9], dd[4], pp[12];  // (g1_engine.mesh: slot `slot` only)\n"),
    ("          msh_plane_t<D1>(P, N, mi, X, R, dd, pp);\n",
     "          msh_slot_t<D1>(P, N, c_mvadr[mi], vid[slot], X, R, dd + slot, pp + 3 * slot);\n"),
    ("                                             float* dist, float* pos, float* fr, float* nu, const float* cot,\n"
     "                                             float* gb) {\n  int br = 0;\n",
     "                                             float* dist, float* pos, float* fr, float* nu, const float* cot,\n"
     "                                             float* gb, int* mvid) {  // mvid: a mesh slot's vertex (g1_engine.mesh)\n  int br = 0;\n"),
    ("      int vid[4] = {0, 0, 0, 0};\n      msh_plane(p1, n, mi, p2, q2, d4, p4, vid);\n"
     "      *dist = d4[slot];\n      for (int k = 0; k < 3; k++) pos[k] = p4[3 * slot + k];\n",
     "      int vid[4] = {0, 0, 0, 0};\n"
     "      if (cot && *mvid >= 0) {  // the contact's replay ran on this lane: its vertex (dist, pos are not read here)\n"
     "        vid[slot] = *mvid;\n        *dist = 0.f;\n        for (int k = 0; k < 3; k++) pos[k] = 0.f;\n"
     "      } else {\n        msh_plane(p1, n, mi, p2, q2, d4, p4, vid);\n        *mvid = vid[slot];\n"
     "        *dist = d4[slot];\n        for (int k = 0; k < 3; k++) pos[k] = p4[3 * slot + k];\n      }\n"),
    ("__device__ __forceinline__ int adj_pair_geom(int pair, int slot, const float (*gx)[3], const float (*gq)[4],\n"
     "                                             float* dist, float* pos, float* fr, float* nu, const float* cot,\n"
     "                                             float* gb) {\n",
     "__device__ __forceinline__ int adj_pair_geom(int pair, int slot, const float (*gx)[3], const float (*gq)[4],\n"
     "                                             float* dist, float* pos, float* fr, float* nu, const float* cot,\n"
     "                                             float* gb, int* mvid) {\n"),
    ("                          c_gsize + 3 * g2, dist, pos, fr, nu, cot, gb);\n",
     "                          c_gsize + 3 * g2, dist, pos, fr, nu, cot, gb, mvid);\n"),
    ("  const int br = adj_pair_geom(pair, slot, s.gx, s.gq, &d, p, fr, &nu, nullptr, nullptr);\n",
     "  int mvid = -1;  // (g1_engine.mesh: the replay's vertex for the VJP's call)\n"
     "  const int br = adj_pair_geom(pair, slot, s.gx, s.gq, &d, p, fr, &nu, nullptr, nullptr, &mvid);\n"),
    ("    adj_pair_geom(pair, slot, s.gx, s.gq, &d2, p2, fr2, &nu2, cot, gb);\n",
     "    adj_pair_geom(pair, slot, s.gx, s.gq, &d2, p2, fr2, &nu2, cot, gb, &mvid);\n"),
)


def hooks(m):
    """The (anchor, replacement, count) edits of the engine text for models with mesh collision geoms: the mesh arrays'
    declarations (with the device code: after DKC's definition, before g1_engine.cylccd's) and the plane-mesh pairs."""
    from nablaengine.dual import DUAL_DEVICE, RF_DEVICE
    plane_anchor = "#if HAS_BOX\n    else if (t2 == 3) {\n"
    dev_anchor = "// box-box (MuJoCo Warp's box_box, margin 0): separating-axis test, then face-vertex or edge-edge clipping\n"
    dev = header(m) + DUAL_DEVICE + "#ifndef DK_RF_DEVICE\n#define DK_RF_DEVICE\n" + RF_DEVICE + "#endif\n" + DEVICE
    return ((plane_anchor, FWD + plane_anchor, 1),
            (dev_anchor, dev + dev_anchor, 1),
            ("    if (t2 != 2) return -1;\n", ADJ + "    if (t2 != 2) return -1;\n", 1)) + tuple((a, b, 1) for a, b in SEARCH_ONCE)

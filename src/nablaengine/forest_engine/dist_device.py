"""CUDA source of the forest engine's distance queries (nablaengine.dist_query is the interface)."""
import mujoco
import numpy as np

QTYPE = {int(mujoco.mjtGeom.mjGEOM_PLANE): 0, int(mujoco.mjtGeom.mjGEOM_SPHERE): 1,
         int(mujoco.mjtGeom.mjGEOM_CAPSULE): 2, int(mujoco.mjtGeom.mjGEOM_BOX): 3, int(mujoco.mjtGeom.mjGEOM_MESH): 4,
         int(mujoco.mjtGeom.mjGEOM_CYLINDER): 5, int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): 6}
TERRAIN_GEOM = -1


def _arr(name, kind, vals):
    vals = list(vals) or [0]
    if kind == "f":
        body = ", ".join(repr(float(np.float32(v))) + "f" for v in vals)
        return f"__device__ const float {name}[{len(vals)}] = {{{body}}};\n"
    return f"__device__ const int {name}[{len(vals)}] = {{{', '.join(str(int(v)) for v in vals)}}};\n"


def _fixture_of(m):
    from nablaengine.forest_engine import step
    table = step.collision_geoms(m)
    out = {}
    for i, f in enumerate(step.fixture_groups(m)):
        for g in f["geoms"]:
            out[table[g]] = i
    return out


def tables(m, slots, terrain=False, extra_pairs=()):
    """#defines and __device__ const arrays of the query geoms and the slots (lists of model geom id pairs). terrain:
    a terrain build (TERRAIN_GEOM in a pair: the terrain query geom DST_TG, the last query geom). extra_pairs: more
    model geom id pairs whose geoms join the query geom table (not queried by the slots; a library built on this one
    may use them). Returns (the header text, the query geoms' model ids in table order)."""
    allg = {g for s in slots for pr in s for g in pr} | {g for pr in extra_pairs for g in pr}
    ter = TERRAIN_GEOM in allg
    if ter:
        if not terrain:
            raise ValueError("a terrain query pair needs a terrain build (init(terrain=...))")
        for pr in [pr for s in slots for pr in s] + list(extra_pairs):
            if pr[0] == TERRAIN_GEOM or (pr[1] == TERRAIN_GEOM and (
                    int(m.geom_bodyid[pr[0]]) == 0 or int(m.geom_type[pr[0]]) == mujoco.mjtGeom.mjGEOM_PLANE)):
                raise ValueError(f"terrain pair {pr}: (a moving non-plane geom, the terrain), the terrain second")
    geoms = sorted(allg - {TERRAIN_GEOM})
    gid = {g: i for i, g in enumerate(geoms)}
    if ter:
        gid[TERRAIN_GEOM] = len(geoms)
    for g in geoms:
        if int(m.geom_type[g]) not in QTYPE:
            raise ValueError(f"geom {g} ({m.geom(g).name}): type {int(m.geom_type[g])} is not supported by the queries")
    for s in slots:
        for a, b in s:
            if int(m.geom_type[a]) == mujoco.mjtGeom.mjGEOM_PLANE and int(m.geom_type[b]) == mujoco.mjtGeom.mjGEOM_PLANE:
                raise ValueError("a query pair of two planes")
    fix = _fixture_of(m)
    meshes = sorted({int(m.geom_dataid[g]) for g in geoms if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH})
    mid = {k: i for i, k in enumerate(meshes)}
    mv, vadr, vnum, eadr, ea, nbadr, nb = [], [], [], [], [], [], []
    for k in meshes:
        va, vn, ga = int(m.mesh_vertadr[k]), int(m.mesh_vertnum[k]), int(m.mesh_graphadr[k])
        vadr.append(len(mv) // 3)
        if ga >= 0:
            g = m.mesh_graph
            nh, nf = int(g[ga]), int(g[ga + 1])
            edgeadr = g[ga + 2:ga + 2 + nh]
            globalid = g[ga + 2 + nh:ga + 2 + 2 * nh]
            local = g[ga + 2 + 2 * nh:ga + 2 + 2 * nh + nh + 3 * nf]
            mv += list(m.mesh_vert[va + globalid].ravel())
            vnum.append(nh)
            eadr.append(len(ea))
            ea += [int(x) for x in edgeadr]
            nbadr.append(len(nb))
            nb += [int(x) for x in local]
        else:
            mv += list(m.mesh_vert[va:va + vn].ravel())
            vnum.append(vn)
            eadr.append(-1)
            nbadr.append(0)
    pairs = [pr for s in slots for pr in s]
    start = np.cumsum([0] + [len(s) for s in slots])
    h = (f"#define DST_NG {len(geoms) + ter}  // query geoms (dist_device.tables)\n"
         f"#define DST_NS {len(slots)}  // slots\n"
         f"#define DST_NP {len(pairs)}  // the slots' geom pairs\n#define DST_NMESH {len(meshes)}\n")
    tg = [0] if ter else []
    if terrain:
        h += (f"#define DST_TG {len(geoms) if ter else -2}  // the terrain query geom\n"
              "#define DST_FW 4  // query face words\n")
    h += _arr("c_hq_body", "i", [m.geom_bodyid[g] for g in geoms] + tg)
    h += _arr("c_hq_fix", "i", [fix.get(g, -1) for g in geoms] + [-1] * len(tg))
    h += _arr("c_hq_type", "i", [QTYPE[int(m.geom_type[g])] for g in geoms] + [7] * len(tg))
    h += _arr("c_hq_mesh", "i", [mid[int(m.geom_dataid[g])] if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH else -1
                                 for g in geoms] + [-1] * len(tg))
    h += _arr("c_hq_size", "f", list(np.asarray(m.geom_size[geoms]).ravel()) + [0.0] * 3 * len(tg))
    h += _arr("c_hq_pos", "f", list(np.asarray(m.geom_pos[geoms]).ravel()) + [0.0] * 3 * len(tg))
    h += _arr("c_hq_quat", "f", list(np.asarray(m.geom_quat[geoms]).ravel()) + [1.0, 0.0, 0.0, 0.0] * len(tg))
    h += _arr("c_hq_rb", "f", [1e30 if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE else m.geom_rbound[g]
                               for g in geoms] + [1e30] * len(tg))
    h += _arr("c_hp_g", "i", [gid[g] for pr in pairs for g in pr])
    h += _arr("c_hs_start", "i", start)
    h += _arr("c_hm_v", "f", mv)
    h += _arr("c_hm_vadr", "i", vadr)
    h += _arr("c_hm_vnum", "i", vnum)
    h += _arr("c_hm_eadr", "i", eadr)
    h += _arr("c_hm_ea", "i", ea)
    h += _arr("c_hm_nbadr", "i", nbadr)
    h += _arr("c_hm_nb", "i", nb)
    return h, geoms


DEVICE = r"""
// ============================== distance queries (nablaengine.forest_engine.dist_device) =============================
#ifndef DST_GJK_ITERATIONS
#define DST_GJK_ITERATIONS 64
#endif
#ifndef DST_EPA_ITERATIONS
#define DST_EPA_ITERATIONS 24
#endif
#define DST_EPA_V (4 + DST_EPA_ITERATIONS)
#define DST_EPA_F (4 + 2 * DST_EPA_ITERATIONS + 8)
#define DST_GJK_EPS 1e-6f  // GJK's relative stop: |v|^2 - v.w <= eps |v|^2

struct HGeom { float c[3], R[9], s[3]; int type, mesh, cache; };

__device__ __forceinline__ void hq_mat(const float* q, float* R) {  // unit quaternion -> row-major rotation
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  R[0] = 1.f - 2.f * (y * y + z * z); R[1] = 2.f * (x * y - w * z); R[2] = 2.f * (x * z + w * y);
  R[3] = 2.f * (x * y + w * z); R[4] = 1.f - 2.f * (x * x + z * z); R[5] = 2.f * (y * z - w * x);
  R[6] = 2.f * (x * z - w * y); R[7] = 2.f * (y * z + w * x); R[8] = 1.f - 2.f * (x * x + y * y);
}
__device__ __forceinline__ float hq_dot(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// body pose of query geom g (fixture placement, world, or its body's pose) in the state s (struct Adj or DstS)
template <class S>
__device__ __forceinline__ void dst_body_pose(const S& s, int g, const float** Q, const float** X) {
  const int b = c_hq_body[g];
#if NFIX
  if (c_hq_fix[g] >= 0) { *Q = s.fixq[c_hq_fix[g]]; *X = s.fixx[c_hq_fix[g]]; return; }
#endif
  *Q = s.Q[b]; *X = s.X[b];
}
template <class S>
__device__ __forceinline__ void dst_geom(const S& s, int g, HGeom& G) {
  const float *Qb, *Xb;
  dst_body_pose(s, g, &Qb, &Xb);
  float t[3], q[4];
  qrot(Qb, c_hq_pos + 3 * g, t);
  for (int k = 0; k < 3; k++) { G.c[k] = Xb[k] + t[k]; G.s[k] = c_hq_size[3 * g + k]; }
  qmul(Qb, c_hq_quat + 4 * g, q);
  const float n = rsqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (int k = 0; k < 4; k++) q[k] *= n;
  hq_mat(q, G.R);
  G.type = c_hq_type[g]; G.mesh = c_hq_mesh[g]; G.cache = 0;
}
__device__ __forceinline__ float dst_radius(const HGeom& G) { return (G.type == 1 || G.type == 2) ? G.s[0] : 0.f; }
// the box face at a closest point (dist_query.FACE_*): o is the box's outward unit normal there (world; the
// direction to the other geom, the contact normal), its box-frame component of largest magnitude names the face:
// 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z (edges and corners: the face the other geom lies most in front of; an exact tie
// goes to the lower axis, a zero component to the + face). -1 for geoms other than boxes.
__device__ __forceinline__ int dst_face_id(const HGeom& G, const float* o) {
  if (G.type != 3) return -1;
  int f = -1;
  float best = -1.f;
  for (int k = 0; k < 3; k++) {
    const float l = G.R[k] * o[0] + G.R[3 + k] * o[1] + G.R[6 + k] * o[2], a = fabsf(l);
    if (a > best) { best = a; f = 2 * k + (l < 0.f ? 1 : 0); }
  }
  return f;
}

// support point of the geom's core in world direction d
__device__ void dst_support(HGeom& G, const float* d, float* p) {
  if (G.type == 1) { for (int k = 0; k < 3; k++) p[k] = G.c[k]; return; }
  if (G.type == 2) {
    const float a[3] = {G.R[2], G.R[5], G.R[8]};
    const float h = hq_dot(d, a) >= 0.f ? G.s[1] : -G.s[1];
    for (int k = 0; k < 3; k++) p[k] = G.c[k] + h * a[k];
    return;
  }
  const float ld[3] = {G.R[0] * d[0] + G.R[3] * d[1] + G.R[6] * d[2], G.R[1] * d[0] + G.R[4] * d[1] + G.R[7] * d[2],
                       G.R[2] * d[0] + G.R[5] * d[1] + G.R[8] * d[2]};
  float v[3] = {0.f, 0.f, 0.f};
  if (G.type == 3) {
    for (int k = 0; k < 3; k++) v[k] = ld[k] >= 0.f ? G.s[k] : -G.s[k];
  } else if (G.type == 5) {
    const float n = sqrtf(ld[0] * ld[0] + ld[1] * ld[1]);
    if (n > 0.f) { v[0] = G.s[0] * ld[0] / n; v[1] = G.s[0] * ld[1] / n; }
    v[2] = ld[2] >= 0.f ? G.s[1] : -G.s[1];
  } else if (G.type == 6) {
    const float t[3] = {G.s[0] * ld[0], G.s[1] * ld[1], G.s[2] * ld[2]};
    const float n = sqrtf(hq_dot(t, t));
    if (n > 0.f)
      for (int k = 0; k < 3; k++) v[k] = G.s[k] * t[k] / n;
  } else if (G.type == 4) {
    const int k = G.mesh, nv = c_hm_vnum[k];
    const float* V = c_hm_v + 3 * c_hm_vadr[k];
    int i = G.cache;
    float best = hq_dot(ld, V + 3 * i);
    if (c_hm_eadr[k] < 0) {
      for (int j = 0; j < nv; j++) {
        const float t = hq_dot(ld, V + 3 * j);
        if (t > best) { best = t; i = j; }
      }
    } else {  // hill climbing on the hull graph (MuJoCo's mesh support)
      const int* EA = c_hm_ea + c_hm_eadr[k];
      const int* NBR = c_hm_nb + c_hm_nbadr[k];
      for (int guard = 0; guard < nv; guard++) {
        int imax = i;
        for (int e = EA[i]; NBR[e] >= 0; e++) {
          const int j = NBR[e];
          const float t = hq_dot(ld, V + 3 * j);
          if (t > best) { best = t; imax = j; }
        }
        if (imax == i) break;
        i = imax;
      }
    }
    G.cache = i;
    for (int c = 0; c < 3; c++) v[c] = V[3 * i + c];
  }
  for (int k = 0; k < 3; k++) p[k] = G.c[k] + G.R[3 * k] * v[0] + G.R[3 * k + 1] * v[1] + G.R[3 * k + 2] * v[2];
}

// the Minkowski difference A - B: support in direction u (w = a - b)
__device__ __forceinline__ void dst_mink(HGeom& A, HGeom& B, const float* u, float* w, float* a, float* b) {
  const float nu[3] = {-u[0], -u[1], -u[2]};
  dst_support(A, u, a);
  dst_support(B, nu, b);
  for (int k = 0; k < 3; k++) w[k] = a[k] - b[k];
}

struct HSx { float w[4][3], a[4][3], b[4][3], l[4]; int n; };

// closest point of triangle (a, b, c) to the origin (Ericson, Real-Time Collision Detection 5.1.5): barycentrics
__device__ void dst_tri(const float* a, const float* b, const float* c, float* l) {
  float ab[3], ac[3], ap[3], bp[3], cp[3];
  for (int k = 0; k < 3; k++) { ab[k] = b[k] - a[k]; ac[k] = c[k] - a[k]; ap[k] = -a[k]; bp[k] = -b[k]; cp[k] = -c[k]; }
  const float d1 = hq_dot(ab, ap), d2 = hq_dot(ac, ap);
  if (d1 <= 0.f && d2 <= 0.f) { l[0] = 1.f; l[1] = l[2] = 0.f; return; }
  const float d3 = hq_dot(ab, bp), d4 = hq_dot(ac, bp);
  if (d3 >= 0.f && d4 <= d3) { l[1] = 1.f; l[0] = l[2] = 0.f; return; }
  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) { const float v = d1 / (d1 - d3); l[0] = 1.f - v; l[1] = v; l[2] = 0.f; return; }
  const float d5 = hq_dot(ab, cp), d6 = hq_dot(ac, cp);
  if (d6 >= 0.f && d5 <= d6) { l[2] = 1.f; l[0] = l[1] = 0.f; return; }
  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) { const float t = d2 / (d2 - d6); l[0] = 1.f - t; l[2] = t; l[1] = 0.f; return; }
  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
    const float t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    l[1] = 1.f - t; l[2] = t; l[0] = 0.f;
    return;
  }
  const float den = 1.f / (va + vb + vc), v = vb * den, t = vc * den;
  l[0] = 1.f - v - t; l[1] = v; l[2] = t;
}

// reduce the simplex to the feature closest to the origin; v = that point; returns true if the origin is inside a
// tetrahedron (or on a degenerate simplex)
__device__ bool dst_simplex(HSx& S, float* v) {
  float l[4] = {0.f, 0.f, 0.f, 0.f};
  if (S.n == 1) {
    l[0] = 1.f;
  } else if (S.n == 2) {
    float ab[3];
    for (int k = 0; k < 3; k++) ab[k] = S.w[1][k] - S.w[0][k];
    const float den = hq_dot(ab, ab);
    float t = den > 0.f ? -hq_dot(S.w[0], ab) / den : 0.f;
    t = fminf(fmaxf(t, 0.f), 1.f);
    l[0] = 1.f - t; l[1] = t;
  } else if (S.n == 3) {
    dst_tri(S.w[0], S.w[1], S.w[2], l);
  } else {  // tetrahedron: the origin inside, or the closest of the faces it lies outside of
    const int F[4][4] = {{0, 1, 2, 3}, {0, 3, 1, 2}, {0, 2, 3, 1}, {1, 3, 2, 0}};
    float best = 3.4e38f;
    bool inside = true;
    for (int f = 0; f < 4; f++) {
      const float *a = S.w[F[f][0]], *b = S.w[F[f][1]], *c = S.w[F[f][2]], *d = S.w[F[f][3]];
      float ab[3], ac[3], nn[3], ad[3];
      for (int k = 0; k < 3; k++) { ab[k] = b[k] - a[k]; ac[k] = c[k] - a[k]; ad[k] = d[k] - a[k]; }
      cross(ab, ac, nn);
      const float so = -hq_dot(nn, a), sd = hq_dot(nn, ad);
      // the origin on d's side of this face (strictly: a flat tetrahedron, d in the face's plane, decides nothing and
      // its faces are searched as triangles)
      if (so * sd > 0.f && fabsf(sd) > 1e-5f * sqrtf(hq_dot(nn, nn) * hq_dot(ad, ad))) continue;
      inside = false;
      float lt[3], p[3];
      dst_tri(a, b, c, lt);
      for (int k = 0; k < 3; k++) p[k] = lt[0] * a[k] + lt[1] * b[k] + lt[2] * c[k];
      const float dd = hq_dot(p, p);
      if (dd < best) {
        best = dd;
        l[0] = l[1] = l[2] = l[3] = 0.f;
        l[F[f][0]] = lt[0]; l[F[f][1]] = lt[1]; l[F[f][2]] = lt[2];
      }
    }
    if (inside) {
      for (int k = 0; k < 4; k++) S.l[k] = 0.25f;
      v[0] = v[1] = v[2] = 0.f;
      return true;
    }
  }
  int n = 0;
  for (int i = 0; i < S.n; i++) {
    if (l[i] <= 0.f) continue;
    for (int k = 0; k < 3; k++) { S.w[n][k] = S.w[i][k]; S.a[n][k] = S.a[i][k]; S.b[n][k] = S.b[i][k]; }
    S.l[n] = l[i];
    n++;
  }
  if (n == 0) { n = 1; S.l[0] = 1.f; }
  S.n = n;
  if (n == 3) {
    // a face: the origin's projection on its plane, v = n (n . w0) / |n|^2. The barycentric sum loses the distance
    // to cancellation in float32 when the face is large against the distance (parallel flat faces: a 0.4 m face at
    // 0.5 mm read 0.15 mm too far, and GJK stopped there on a duplicate support point)
    float e1[3], e2[3], nn[3];
    for (int k = 0; k < 3; k++) { e1[k] = S.w[1][k] - S.w[0][k]; e2[k] = S.w[2][k] - S.w[0][k]; }
    cross(e1, e2, nn);
    const float n2 = hq_dot(nn, nn);
    if (n2 > 0.f) {
      const float t = hq_dot(nn, S.w[0]) / n2;
      for (int k = 0; k < 3; k++) v[k] = t * nn[k];
      return false;
    }
  }
  for (int k = 0; k < 3; k++) {
    v[k] = 0.f;
    for (int i = 0; i < n; i++) v[k] += S.l[i] * S.w[i][k];
  }
  return false;
}

// GJK on the cores of A and B: 0 separated (v = a - b the closest point of A - B, S its feature), 1 overlapping (S a
// tetrahedron around the origin when S.n == 4)
__device__ int dst_gjk(HGeom& A, HGeom& B, HSx& S, float* v) {
  for (int k = 0; k < 3; k++) v[k] = A.c[k] - B.c[k];
  if (hq_dot(v, v) < 1e-20f) { v[0] = 1e-6f; v[1] = v[2] = 0.f; }
  S.n = 0;
  float vv_prev = 3.4e38f;
  for (int it = 0; it < DST_GJK_ITERATIONS; it++) {
    const float u[3] = {-v[0], -v[1], -v[2]};
    float w[3], a[3], b[3];
    dst_mink(A, B, u, w, a, b);
    const float vv = hq_dot(v, v);
    if (S.n > 0 && vv - hq_dot(v, w) <= DST_GJK_EPS * vv) break;
    bool dup = false;
    for (int i = 0; i < S.n; i++) {
      const float e[3] = {w[0] - S.w[i][0], w[1] - S.w[i][1], w[2] - S.w[i][2]};
      dup |= hq_dot(e, e) <= 1e-12f * fmaxf(vv, 1e-12f);
    }
    if (dup) break;
    const HSx keep = S;  // (restored when the step makes no progress: a degenerate simplex's solve can be off)
    for (int k = 0; k < 3; k++) { S.w[S.n][k] = w[k]; S.a[S.n][k] = a[k]; S.b[S.n][k] = b[k]; }
    S.n++;
    const float vk[3] = {v[0], v[1], v[2]};
    if (dst_simplex(S, v)) return 1;
    const float vn = hq_dot(v, v);
    if (vn < 1e-14f) return 1;
    if (it > 0 && vn >= vv_prev) {
      S = keep;
      for (int k = 0; k < 3; k++) v[k] = vk[k];
      break;
    }
    vv_prev = vn;
  }
  return 0;
}

struct HEpa {
  float w[DST_EPA_V][3], a[DST_EPA_V][3], b[DST_EPA_V][3];
  int f[DST_EPA_F][3];
  float fn[DST_EPA_F][4];  // unit outward normal, distance to the origin
  int nv, nf;
};
__device__ bool dst_face(HEpa& P, int i, int j, int k, int slot) {
  float ab[3], ac[3], n[3];
  for (int c = 0; c < 3; c++) { ab[c] = P.w[j][c] - P.w[i][c]; ac[c] = P.w[k][c] - P.w[i][c]; }
  cross(ab, ac, n);
  const float nn = sqrtf(hq_dot(n, n));
  P.f[slot][0] = i; P.f[slot][1] = j; P.f[slot][2] = k;
  if (!(nn > 1e-20f)) {  // degenerate: never chosen, never visible
    P.fn[slot][0] = P.fn[slot][1] = P.fn[slot][2] = 0.f; P.fn[slot][3] = -1.f;
    return false;
  }
  for (int c = 0; c < 3; c++) P.fn[slot][c] = n[c] / nn;
  P.fn[slot][3] = hq_dot(P.fn[slot], P.w[i]);
  return true;
}

// EPA from GJK's tetrahedron S (origin inside): depth (>= 0), the unit normal n of A - B's closest face (the direction
// that separates B from A), the points a (on A's core) and b (on B's core). Returns false if it failed (degenerate)
__device__ bool dst_epa(HGeom& A, HGeom& B, const HSx& S, float* depth, float* n, float* pa, float* pb) {
  HEpa P;
  P.nv = 4;
  for (int i = 0; i < 4; i++)
    for (int k = 0; k < 3; k++) { P.w[i][k] = S.w[i][k]; P.a[i][k] = S.a[i][k]; P.b[i][k] = S.b[i][k]; }
  const int T[4][4] = {{0, 1, 2, 3}, {0, 3, 1, 2}, {0, 2, 3, 1}, {1, 3, 2, 0}};
  P.nf = 0;
  for (int f = 0; f < 4; f++) {
    int i = T[f][0], j = T[f][1], k = T[f][2];
    float ab[3], ac[3], nn[3], ad[3];
    for (int c = 0; c < 3; c++) {
      ab[c] = P.w[j][c] - P.w[i][c]; ac[c] = P.w[k][c] - P.w[i][c]; ad[c] = P.w[T[f][3]][c] - P.w[i][c];
    }
    cross(ab, ac, nn);
    if (hq_dot(nn, ad) > 0.f) { const int t = j; j = k; k = t; }  // outward: away from the fourth vertex
    dst_face(P, i, j, k, P.nf++);
  }
  int best = -1;
  for (int it = 0; it < DST_EPA_ITERATIONS; it++) {
    best = -1;
    float bd = 3.4e38f;
    for (int f = 0; f < P.nf; f++)
      if (P.fn[f][3] >= 0.f && P.fn[f][3] < bd) { bd = P.fn[f][3]; best = f; }
    if (best < 0) return false;
    float w[3], a[3], b[3];
    dst_mink(A, B, P.fn[best], w, a, b);
    const float dw = hq_dot(P.fn[best], w);
    if (dw - bd <= 1e-6f + 1e-5f * bd || P.nv >= DST_EPA_V) break;
    const int nw = P.nv++;
    for (int k = 0; k < 3; k++) { P.w[nw][k] = w[k]; P.a[nw][k] = a[k]; P.b[nw][k] = b[k]; }
    // faces visible from w: removed (fn[3] = -2); their edges not shared with another removed face form the horizon
    bool vis[DST_EPA_F];
    for (int f = 0; f < P.nf; f++) {
      float e[3];
      for (int k = 0; k < 3; k++) e[k] = w[k] - P.w[P.f[f][0]][k];
      vis[f] = P.fn[f][3] > -1.5f && hq_dot(P.fn[f], e) > 0.f;
    }
    int hor[DST_EPA_F][2], nh = 0;
    for (int f = 0; f < P.nf; f++) {
      if (!vis[f]) continue;
      for (int e = 0; e < 3; e++) {
        const int i = P.f[f][e], j = P.f[f][(e + 1) % 3];
        bool shared = false;
        for (int g = 0; g < P.nf && !shared; g++) {
          if (g == f || !vis[g]) continue;
          for (int q = 0; q < 3; q++)
            shared |= P.f[g][q] == j && P.f[g][(q + 1) % 3] == i;
        }
        if (!shared && nh < DST_EPA_F) { hor[nh][0] = i; hor[nh][1] = j; nh++; }
      }
    }
    // compact the kept faces, then add the horizon's
    int nf = 0;
    for (int f = 0; f < P.nf; f++) {
      if (vis[f] || P.fn[f][3] < -1.5f) continue;
      if (nf != f) {
        for (int k = 0; k < 3; k++) P.f[nf][k] = P.f[f][k];
        for (int k = 0; k < 4; k++) P.fn[nf][k] = P.fn[f][k];
      }
      nf++;
    }
    if (nf + nh > DST_EPA_F) { P.nf = nf; break; }
    P.nf = nf;
    for (int e = 0; e < nh; e++) dst_face(P, hor[e][0], hor[e][1], nw, P.nf++);
  }
  best = -1;
  float bd = 3.4e38f;
  for (int f = 0; f < P.nf; f++)
    if (P.fn[f][3] >= 0.f && P.fn[f][3] < bd) { bd = P.fn[f][3]; best = f; }
  if (best < 0) return false;
  *depth = bd;
  for (int k = 0; k < 3; k++) n[k] = P.fn[best][k];
  // barycentrics of the origin's projection bd n on the face
  const int i = P.f[best][0], j = P.f[best][1], k3 = P.f[best][2];
  float v0[3], v1[3], v2[3];
  for (int c = 0; c < 3; c++) {
    v0[c] = P.w[j][c] - P.w[i][c]; v1[c] = P.w[k3][c] - P.w[i][c]; v2[c] = bd * n[c] - P.w[i][c];
  }
  const float d00 = hq_dot(v0, v0), d01 = hq_dot(v0, v1), d11 = hq_dot(v1, v1), d20 = hq_dot(v2, v0),
              d21 = hq_dot(v2, v1);
  const float den = d00 * d11 - d01 * d01;
  float lv = den != 0.f ? (d11 * d20 - d01 * d21) / den : 0.f, lw = den != 0.f ? (d00 * d21 - d01 * d20) / den : 0.f;
  lv = fminf(fmaxf(lv, 0.f), 1.f); lw = fminf(fmaxf(lw, 0.f), 1.f - lv);
  const float lu = 1.f - lv - lw;
  for (int c = 0; c < 3; c++) {
    pa[c] = lu * P.a[i][c] + lv * P.a[j][c] + lw * P.a[k3][c];
    pb[c] = lu * P.b[i][c] + lv * P.b[j][c] + lw * P.b[k3][c];
  }
  return true;
}

// signed distance of query geoms g1, g2 in state s: p1 on geom 1, p2 on geom 2, n the unit normal from 1 to 2
// (p2 - p1 = d n)
template <class S>
__device__ float dst_distance(const S& s, int g1, int g2, float* p1, float* p2, float* n) {
  HGeom A, B;
  dst_geom(s, g1, A);
  dst_geom(s, g2, B);
  const float r1 = dst_radius(A), r2 = dst_radius(B);
  if (A.type == 0 || B.type == 0) {  // a plane (its z axis the normal) against the other geom
    const bool fl = B.type == 0;
    HGeom& Pl = fl ? B : A;
    HGeom& G = fl ? A : B;
    const float r = fl ? r1 : r2;
    float pn[3] = {Pl.R[2], Pl.R[5], Pl.R[8]}, mn[3] = {-Pl.R[2], -Pl.R[5], -Pl.R[8]}, b[3], e[3];
    dst_support(G, mn, b);
    for (int k = 0; k < 3; k++) e[k] = b[k] - Pl.c[k];
    const float dc = hq_dot(pn, e), d = dc - r;
    float* po = fl ? p1 : p2;  // the other geom's point
    float* pp = fl ? p2 : p1;  // the plane's
    for (int k = 0; k < 3; k++) {
      po[k] = b[k] - r * pn[k];
      pp[k] = b[k] - dc * pn[k];
      n[k] = fl ? -pn[k] : pn[k];
    }
    return d;
  }
  HSx X;
  float v[3], a[3] = {0.f, 0.f, 0.f}, b[3] = {0.f, 0.f, 0.f};
  const int ov = dst_gjk(A, B, X, v);
  for (int i = 0; i < X.n; i++)
    for (int k = 0; k < 3; k++) { a[k] += X.l[i] * X.a[i][k]; b[k] += X.l[i] * X.b[i][k]; }
  float dc;
  if (!ov) {
    dc = sqrtf(hq_dot(v, v));
    for (int k = 0; k < 3; k++) {  // n from v; a, b shifted by half the barycentric sum's error each, so a - b = v
      const float e = 0.5f * ((a[k] - b[k]) - v[k]);
      a[k] -= e; b[k] += e;
      n[k] = -v[k] / dc;
    }
  } else {
    float depth = 0.f;
    const bool solid = (A.type != 1 && A.type != 2) || (B.type != 1 && B.type != 2);
    if (!(solid && X.n == 4 && dst_epa(A, B, X, &depth, n, a, b))) {  // touching cores, or two point / segment cores
      float e[3];
      for (int k = 0; k < 3; k++) e[k] = B.c[k] - A.c[k];
      const float en = sqrtf(hq_dot(e, e));
      for (int k = 0; k < 3; k++) n[k] = en > 0.f ? e[k] / en : (k == 2 ? 1.f : 0.f);
      depth = 0.f;
    }
    dc = -depth;
  }
  for (int k = 0; k < 3; k++) { p1[k] = a[k] + r1 * n[k]; p2[k] = b[k] - r2 * n[k]; }
  return dc - r1 - r2;
}

// the Jacobian body of query geom g (0 for world and fixture geoms)
__device__ __forceinline__ int dst_jbody(int g) { return c_hq_fix[g] >= 0 ? 0 : c_hq_body[g]; }

// dof columns of body lane rb at point p: col[e] = d p / d qdot_e (free root: 3 translations, then rotations about the
// world axes zl[e] through X; hinge: z x (p - X); slide: z), dotted with F: out[e] += F . col[e]
__device__ __forceinline__ void dst_cols_dot(int jt, const float (&zl)[3][3], const float* X, const float* p,
                                             const float* F, float* out) {
  if (jt == JFREE) {
    float r[3], c[3];
    for (int k = 0; k < 3; k++) { out[k] += F[k]; r[k] = p[k] - X[k]; }
    for (int e = 0; e < 3; e++) { cross(zl[e], r, c); out[3 + e] += hq_dot(c, F); }
  } else if (jt == JSLIDE) {
    out[0] += hq_dot(zl[0], F);
  } else {
    float r[3], c[3];
    for (int k = 0; k < 3; k++) r[k] = p[k] - X[k];
    cross(zl[0], r, c);
    out[0] += hq_dot(c, F);
  }
}
// the lane's rotation axes (free body: R e_k; hinge / slide: z = R axis)
template <class S>
__device__ __forceinline__ void dst_axes(const S& s, int rb, int jt, float (&zl)[3][3]) {
  for (int e = 0; e < 3; e++)
    for (int k = 0; k < 3; k++) zl[e][k] = 0.f;
  if (jt == JFREE) {
    for (int e = 0; e < 3; e++) {
      const float ek[3] = {e == 0 ? 1.f : 0.f, e == 1 ? 1.f : 0.f, e == 2 ? 1.f : 0.f};
      qrot(s.Q[rb], ek, zl[e]);
    }
  } else {
    qrot(s.Q[rb], c_axis + 3 * rb, zl[0]);
  }
}
"""

QUERY_KERNELS = r"""
// ================================ query kernels (nablaengine.forest_engine.dist_device) ===============================
#define DST_WPB 4  // worlds (warps) per block
struct DstS {
  float q[NQ], zv[NV];
  float Q[NB][4], X[NB][3], W[NB][3], V[NB][3], AL[NB][3], A[NB][3];
  float fixq[NFIX > 0 ? NFIX : 1][4], fixx[NFIX > 0 ? NFIX : 1][3];
  float sp[DST_NS > 0 ? DST_NS : 1][12];  // the VJP's per slot: p1, p2, F1, F2
  int sb[DST_NS > 0 ? DST_NS : 1][2];
};
__device__ void dst_fk(DstS& s, const float* qpos, const float* com, int w, int lane) {
  for (int i = lane; i < NQ; i += 32) s.q[i] = qpos[(size_t)w * NQ + i];
  for (int i = lane; i < NV; i += 32) s.zv[i] = 0.f;
  load_fixtures(s, com, w, lane);
  if (lane == 0) {
    s.Q[0][0] = 1.f; s.Q[0][1] = s.Q[0][2] = s.Q[0][3] = 0.f;
    for (int k = 0; k < 3; k++) { s.X[0][k] = s.W[0][k] = s.V[0][k] = s.AL[0][k] = 0.f; s.A[0][k] = -c_grav[k]; }
  }
  __syncwarp();
  const int b = lane, dpt = b < NB ? c_depth[b] : -1, p = b < NB ? c_parent[b] : 0;
  for (int L = 1; L <= MAXD; L++) {  // as adjoint.py adj_poses (zero velocity and acceleration)
    if (dpt == L) {
      const float *Qp = s.Q[p], *Xp = s.X[p];
#if NFIX
      if (c_fixb[b] >= 0) { Qp = s.fixq[c_fixb[b]]; Xp = s.fixx[c_fixb[b]]; }
#endif
      fwd_body(b, Qp, Xp, s.W[p], s.V[p], s.AL[p], s.A[p], s.q, s.zv, s.zv, s.Q[b], s.X[b], s.W[b], s.V[b],
               s.AL[b], s.A[b]);
    }
    __syncwarp();
  }
}
// slot k's smallest distance over its geom pairs (the first minimum): d, p1, p2, n, the bodies of the pair (and
// with gp, its query geoms)
__device__ float dst_slot(const DstS& s, int k, float* p1, float* p2, float* n, int* b1, int* b2, int* gp = nullptr) {
  float d = 3.4e38f;
  for (int i = c_hs_start[k]; i < c_hs_start[k + 1]; i++) {
    float a1[3], a2[3], an[3];
    const int g1 = c_hp_g[2 * i], g2 = c_hp_g[2 * i + 1];
    const float di = dst_distance(s, g1, g2, a1, a2, an);
    if (di < d) {
      d = di;
      for (int c = 0; c < 3; c++) { p1[c] = a1[c]; p2[c] = a2[c]; n[c] = an[c]; }
      *b1 = dst_jbody(g1); *b2 = dst_jbody(g2);
      if (gp) { gp[0] = g1; gp[1] = g2; }
    }
  }
  return d;
}
// dist (nw, DST_NS): the signed distance, max_distance where larger (far = 1); pts (nw, DST_NS, 2, 3): p1, p2;
// nrm (nw, DST_NS, 3); far (nw, DST_NS); face (nw, DST_NS, 2) or nullptr: the box face of geom 1 and of geom 2 at
// their closest points (dst_face_id; -1 for other geoms)
__global__ void __launch_bounds__(32 * DST_WPB) k_dst_query(int nw, const float* qpos, const float* com, float dmax,
                                                            float* dist, float* pts, float* nrm, int* far,
                                                            int* face) {
  __shared__ DstS sh[DST_WPB];
  const int lane = threadIdx.x & 31, wl = threadIdx.x >> 5, w = blockIdx.x * DST_WPB + wl;
  if (w >= nw) return;
  DstS& s = sh[wl];
  dst_fk(s, qpos, com, w, lane);
  for (int k = lane; k < DST_NS; k += 32) {
    float p1[3], p2[3], n[3];
    int b1 = 0, b2 = 0, gp[2] = {0, 0};
    const float d = dst_slot(s, k, p1, p2, n, &b1, &b2, gp);
    const size_t o = (size_t)w * DST_NS + k;
    if (face) {
      HGeom G;
      const float mn[3] = {-n[0], -n[1], -n[2]};
      dst_geom(s, gp[0], G);
      face[2 * o] = dst_face_id(G, n);
      dst_geom(s, gp[1], G);
      face[2 * o + 1] = dst_face_id(G, mn);
    }
    const bool f = !(d <= dmax);
    dist[o] = f ? dmax : d;
    far[o] = f ? 1 : 0;
    for (int c = 0; c < 3; c++) { pts[6 * o + c] = p1[c]; pts[6 * o + 3 + c] = p2[c]; nrm[3 * o + c] = n[c]; }
  }
}
// the VJP: cotangents gd (nw, DST_NS) of dist and gp (nw, DST_NS, 2, 3) of pts -> gq (nw, NQ) of qpos (a far slot's
// distance gets no gradient: it is the constant max_distance)
__global__ void __launch_bounds__(32 * DST_WPB) k_dst_query_vjp(int nw, const float* qpos, const float* com,
                                                                float dmax, const float* gd, const float* gp,
                                                                float* gq) {
  __shared__ DstS sh[DST_WPB];
  const int lane = threadIdx.x & 31, wl = threadIdx.x >> 5, w = blockIdx.x * DST_WPB + wl;
  if (w >= nw) return;
  DstS& s = sh[wl];
  dst_fk(s, qpos, com, w, lane);
  for (int k = lane; k < DST_NS; k += 32) {
    float p1[3], p2[3], n[3];
    int b1 = 0, b2 = 0;
    const float d = dst_slot(s, k, p1, p2, n, &b1, &b2);
    const size_t o = (size_t)w * DST_NS + k;
    const float g = d <= dmax ? gd[o] : 0.f;
    for (int c = 0; c < 3; c++) {
      s.sp[k][c] = p1[c]; s.sp[k][3 + c] = p2[c];
      s.sp[k][6 + c] = -g * n[c] + gp[6 * o + c];
      s.sp[k][9 + c] = g * n[c] + gp[6 * o + 3 + c];
    }
    s.sb[k][0] = b1; s.sb[k][1] = b2;
  }
  __syncwarp();
  const bool body = lane >= 1 && lane < NB;
  const int rb = body ? lane : 1, jt = JT_OF(rb);
  float zl[3][3], G[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  dst_axes(s, rb, jt, zl);
  if (body)
    for (int k = 0; k < DST_NS; k++) {
      if ((c_ancmask[s.sb[k][0]] >> rb) & 1u) dst_cols_dot(jt, zl, s.X[rb], s.sp[k], s.sp[k] + 6, G);
      if ((c_ancmask[s.sb[k][1]] >> rb) & 1u) dst_cols_dot(jt, zl, s.X[rb], s.sp[k] + 3, s.sp[k] + 9, G);
    }
  float* o = gq + (size_t)w * NQ;
  if (body) {
    const int qa = c_qadr[rb];
    if (jt != JFREE) {
      o[qa] = G[0];
    } else {  // dof tangent to qpos form (adjoint.py adj_compose: 2 q (0, g_rot), q as given)
      for (int k = 0; k < 3; k++) o[qa + k] = G[k];
      const float gg[4] = {0.f, G[3], G[4], G[5]};
      float t[4];
      qmul(s.q + qa + 3, gg, t);
      for (int k = 0; k < 4; k++) o[qa + 3 + k] = 2.f * t[k];
    }
  }
}
"""


TER_DEVICE = r"""
// ---- the terrain as a query geom (dist_device: terrain builds) ----
struct HTer { int e, tri; float tn[3]; HGeom B; };
// heightfield j's cell (r, c) triangle t as a prism HGeom (world vertices), its top's upward unit normal tn (world)
__device__ void dst_prism(int j, int r, int c, int t, HGeom& G, float* tn) {
  const float* hh = g_ter.hfh + 16 * j;
  const int* hi = g_ter.hfi + 4 * j;
  const float *hs = hh + 12, *hd = g_ter.hfd + hi[2], *R = hh + 3;
  const int ncol = hi[1], nrow = hi[0];
  const float dx = 2.f * hs[0] / float(ncol - 1), dy = 2.f * hs[1] / float(nrow - 1);
  const int cc[3] = {c, c + 1, t ? c + 1 : c}, rr[3] = {t ? r + 1 : r, r, r + 1};
  float l[6][3];
  for (int i = 0; i < 3; i++) {
    l[i][0] = l[3 + i][0] = float(cc[i]) * dx - hs[0];
    l[i][1] = l[3 + i][1] = float(rr[i]) * dy - hs[1];
    l[i][2] = hd[rr[i] * ncol + cc[i]] * hs[2];
    l[3 + i][2] = -hs[3];
  }
  float e1[3], e2[3], nl[3];
  for (int k = 0; k < 3; k++) { e1[k] = l[1][k] - l[0][k]; e2[k] = l[2][k] - l[0][k]; }
  nl[0] = e1[1] * e2[2] - e1[2] * e2[1]; nl[1] = e1[2] * e2[0] - e1[0] * e2[2]; nl[2] = e1[0] * e2[1] - e1[1] * e2[0];
  const float il = rsqrtf(hq_dot(nl, nl));
  for (int k = 0; k < 3; k++) tn[k] = (R[3 * k] * nl[0] + R[3 * k + 1] * nl[1] + R[3 * k + 2] * nl[2]) * il;
  for (int k = 0; k < 3; k++) G.c[k] = 0.f;
  for (int i = 0; i < 6; i++)
    for (int k = 0; k < 3; k++) {
      G.v[i][k] = hh[k] + R[3 * k] * l[i][0] + R[3 * k + 1] * l[i][1] + R[3 * k + 2] * l[i][2];
      G.c[k] += G.v[i][k] * (1.f / 6.f);
    }
  for (int k = 0; k < 9; k++) G.R[k] = (k % 4) ? 0.f : 1.f;
  G.s[0] = G.s[1] = G.s[2] = 0.f;
  G.type = 7; G.mesh = -1; G.cache = 0;
}
// signed distance of query geom g1 to the terrain (the nearest element within reach; 3.4e38 if none, the points then
// the geom's centre, n +z): p1 on the geom, p2 on the terrain, n from the geom to the terrain; the element in T
// want_depth false: stop at the first element with d <= 0 (the pair is in contact), without its penetration
// depth
template <class S>
__device__ float dst_terrain(const S& s, int g1, float reach, float* p1, float* p2, float* n, HTer* T,
                             bool want_depth = true) {
  HGeom A;
  dst_geom(s, g1, A);
  const float ra = dst_radius(A), rb = c_hq_rb[g1];
  float best = 3.4e38f;
  for (int k = 0; k < 3; k++) { p1[k] = p2[k] = A.c[k]; n[k] = k == 2 ? 1.f : 0.f; }
  if (T) { T->e = -1; T->tri = -1; }
  if (g_ter.grid == nullptr || !(reach >= 0.f)) return best;
  const float x0 = g_ter.grid[0], y0 = g_ter.grid[1], ic = g_ter.grid[2];
  const int nx = int(g_ter.grid[3]), ny = int(g_ter.grid[4]), nbox = int(g_ter.grid[5]);
  const float L = rb + reach;
  int c0 = int(floorf((A.c[0] - L - x0) * ic)), c1 = int(floorf((A.c[0] + L - x0) * ic));
  int r0 = int(floorf((A.c[1] - L - y0) * ic)), r1 = int(floorf((A.c[1] + L - y0) * ic));
  if (c1 < 0 || r1 < 0 || c0 >= nx || r0 >= ny) return best;
  c0 = max(c0, 0); r0 = max(r0, 0); c1 = min(c1, nx - 1); r1 = min(r1, ny - 1);
  float a1[3], a2[3], an[3];
  for (int r = r0; r <= r1; r++)
    for (int c = c0; c <= c1; c++)
      for (int q = g_ter.cstart[r * nx + c]; q < g_ter.cstart[r * nx + c + 1]; q++) {
        const int e = g_ter.citems[q];
        if (c != max(c0, g_ter.ecell[2 * e]) || r != max(r0, g_ter.ecell[2 * e + 1])) continue;  // once per element
        const float lim = fminf(reach, best);
        if (e < nbox) {
          const float* b = g_ter.box + 15 * e;
          float q2 = 0.f;
          for (int i = 0; i < 3; i++) {
            const float l = b[3 + i] * (A.c[0] - b[0]) + b[6 + i] * (A.c[1] - b[1]) + b[9 + i] * (A.c[2] - b[2]);
            const float o = fmaxf(fabsf(l) - b[12 + i], 0.f);
            q2 += o * o;
          }
          if (q2 > (rb + lim) * (rb + lim) * 1.0002f + 1e-6f) continue;  // (|centre - box| - rb: a lower bound)
          HGeom B;
          for (int k = 0; k < 3; k++) { B.c[k] = b[k]; B.s[k] = b[12 + k]; }
          for (int k = 0; k < 9; k++) B.R[k] = b[3 + k];
          B.type = 3; B.mesh = -1; B.cache = 0;
          float d;
          if (A.type == 1) {  // a sphere: its centre against the box in closed form (the box's closest point)
            float l[3], qb[3], e2 = 0.f;
            for (int i = 0; i < 3; i++) {
              l[i] = b[3 + i] * (A.c[0] - b[0]) + b[6 + i] * (A.c[1] - b[1]) + b[9 + i] * (A.c[2] - b[2]);
              qb[i] = fminf(fmaxf(l[i], -b[12 + i]), b[12 + i]);
              e2 += (l[i] - qb[i]) * (l[i] - qb[i]);
            }
            if (e2 > 0.f) {
              const float dc = sqrtf(e2);
              for (int k = 0; k < 3; k++) {
                a2[k] = b[k] + b[3 + 3 * k] * qb[0] + b[4 + 3 * k] * qb[1] + b[5 + 3 * k] * qb[2];
                an[k] = (a2[k] - A.c[k]) / dc;
                a1[k] = A.c[k] + ra * an[k];
              }
              d = dc - ra;
            } else {  // the centre inside: the depth by GJK / EPA (none without want_depth)
              d = dst_gdist(A, B, a1, a2, an, want_depth);
            }
          } else {
            d = dst_gdist(A, B, a1, a2, an, want_depth);
          }
          if (d < best) {
            best = d;
            for (int k = 0; k < 3; k++) { p1[k] = a1[k]; p2[k] = a2[k]; n[k] = an[k]; }
            if (T) { T->e = e; T->tri = -1; T->B = B; }
            if (!want_depth && d <= 0.f) return best;
          }
          continue;
        }
        const int j = e - nbox;
        const float* hh = g_ter.hfh + 16 * j;
        const int* hi = g_ter.hfi + 4 * j;
        const float *hs = hh + 12, *R = hh + 3;
        const int ncol = hi[1], nrow = hi[0];
        float pl[3];
        for (int i = 0; i < 3; i++)
          pl[i] = R[i] * (A.c[0] - hh[0]) + R[3 + i] * (A.c[1] - hh[1]) + R[6 + i] * (A.c[2] - hh[2]);
        if (pl[2] - rb > hs[2] + lim) continue;  // above the heightfield's top by more than the cutoff
        const float dx = 2.f * hs[0] / float(ncol - 1), dy = 2.f * hs[1] / float(nrow - 1), Lh = rb + lim;
        const int k0 = max(0, int(floorf((pl[0] - Lh + hs[0]) / dx)));
        const int k1 = min(ncol - 2, int(floorf((pl[0] + Lh + hs[0]) / dx)));
        const int q0 = max(0, int(floorf((pl[1] - Lh + hs[1]) / dy)));
        const int q1 = min(nrow - 2, int(floorf((pl[1] + Lh + hs[1]) / dy)));
        for (int rr = q0; rr <= q1; rr++)
          for (int cc = k0; cc <= k1; cc++)
#pragma unroll 1
            for (int t = 0; t < 2; t++) {
              HGeom B;
              float tn[3], sp[3], w[3];
              dst_prism(j, rr, cc, t, B, tn);
              const float mt[3] = {-tn[0], -tn[1], -tn[2]};
              dst_support(A, mt, sp);  // the geom core's lowest point against the top plane
              for (int k = 0; k < 3; k++) w[k] = sp[k] - B.v[0][k];
              const float lb = hq_dot(tn, w) - ra;  // the prism lies below its top plane: d >= lb
              if (lb >= best || lb > reach) continue;
              const float d = dst_gdist(A, B, a1, a2, an, want_depth);
              if (d < best) {
                best = d;
                for (int k = 0; k < 3; k++) { p1[k] = a1[k]; p2[k] = a2[k]; n[k] = an[k]; }
                if (T) {
                  T->e = e; T->tri = 2 * (rr * (ncol - 1) + cc) + t;
                  for (int k = 0; k < 3; k++) T->tn[k] = tn[k];
                }
                if (!want_depth && d <= 0.f) return best;
              }
            }
      }
  return best;
}
// dst_distance with the terrain (g2 == DST_TG: dst_terrain within reach) and its element (T; e = -1 for other pairs)
template <class S>
__device__ float dst_distance_r(const S& s, int g1, int g2, float reach, float* p1, float* p2, float* n, HTer* T,
                               bool want_depth = true) {
  if (g2 == DST_TG) return dst_terrain(s, g1, reach, p1, p2, n, T, want_depth);
  if (T) { T->e = -1; T->tri = -1; }
  return dst_distance(s, g1, g2, p1, p2, n);
}
"""

TER_EDITS = (
    ("struct HGeom { float c[3], R[9], s[3]; int type, mesh, cache; };",
     "struct HGeom { float c[3], R[9], s[3]; int type, mesh, cache; float v[6][3]; };  // v: a t"
     "errain prism's"),
    ("__device__ void dst_support(HGeom& G, const float* d, float* p) {\n",
     "__device__ void dst_support(HGeom& G, const float* d, float* p) {\n"
     "  if (G.type == 7) {  // a terrain prism: the vertex farthest along d (world)\n    int i = 0;\n"
     "    float best = hq_dot(d, G.v[0]);\n    for (int j = 1; j < 6; j++) {\n"
     "      const float t = hq_dot(d, G.v[j]);\n      if (t > best) { best = t; i = j; }\n    }\n"
     "    for (int k = 0; k < 3; k++) p[k] = G.v[i][k];\n    return;\n  }\n"),
    ("template <class S>\n"
     "__device__ float dst_distance(const S& s, int g1, int g2, float* p1, float* p2, float* n) {\n"
     "  HGeom A, B;\n  dst_geom(s, g1, A);\n  dst_geom(s, g2, B);\n",
     "__device__ __noinline__ float dst_gdist(HGeom& A, HGeom& B, float* p1, float* p2, float* n,\n"
     "                                        bool want_depth = true) {\n"),
    ("  } else {\n    float depth = 0.f;\n",
     "  } else {\n"
     "    if (!want_depth) {  // overlapping cores: d <= 0, its depth not needed (want_depth false)\n"
     "      for (int k = 0; k < 3; k++) { p1[k] = A.c[k]; p2[k] = B.c[k]; n[k] = k == 2 ? 1.f : 0.f; }\n"
     "      return -1.f;\n    }\n    float depth = 0.f;\n"),
    ("  return dc - r1 - r2;\n}\n",
     "  return dc - r1 - r2;\n}\ntemplate <class S>\n"
     "__device__ float dst_distance(const S& s, int g1, int g2, float* p1, float* p2, float* n) {\n"
     "  HGeom A, B;\n  dst_geom(s, g1, A);\n  dst_geom(s, g2, B);\n  return dst_gdist(A, B, p1, p2, n);\n}\n"
     "@TER_DEVICE@"),
    ("int* b1, int* b2, int* gp = nullptr) {\n",
     "int* b1, int* b2, int* gp = nullptr,\n                     float reach = 0.f, HTer* T = nullptr) {\n"),
    ("    const float di = dst_distance(s, g1, g2, a1, a2, an);\n    if (di < d) {\n",
     "    HTer t;\n    const float di = dst_distance_r(s, g1, g2, reach, a1, a2, an, T ? &t : nullptr);\n"
     "    if (di < d || i == c_hs_start[k]) {  // (the first pair always: a terrain pair may fin"
     "d no element)\n      if (T) *T = t;\n"),
    ("    const float d = dst_slot(s, k, p1, p2, n, &b1, &b2, gp);\n",
     "    HTer T;\n    T.e = T.tri = -1;\n"
     "    const float d = dst_slot(s, k, p1, p2, n, &b1, &b2, gp, dmax, &T);\n"),
    ("      face[2 * o] = dst_face_id(G, n);\n      dst_geom(s, gp[1], G);\n"
     "      face[2 * o + 1] = dst_face_id(G, mn);\n",
     "      face[DST_FW * o] = dst_face_id(G, n);\n"
     "      if (gp[1] == DST_TG) {  // the terrain: a box's face, the element\n"
     "        face[DST_FW * o + 1] = T.e >= 0 && T.tri < 0 ? dst_face_id(T.B, mn) : -1;\n"
     "        face[DST_FW * o + 2] = T.e;\n        face[DST_FW * o + 3] = T.tri;\n      } else {\n"
     "        dst_geom(s, gp[1], G);\n        face[DST_FW * o + 1] = dst_face_id(G, mn);\n"
     "        face[DST_FW * o + 2] = face[DST_FW * o + 3] = -1;\n      }\n"),
    ("    const float d = dst_slot(s, k, p1, p2, n, &b1, &b2);\n",
     "    const float d = dst_slot(s, k, p1, p2, n, &b1, &b2, nullptr, dmax);\n"),
)


def terrain_text(device, kernels):
    """DEVICE and QUERY_KERNELS of a terrain build (TER_EDITS, TER_DEVICE)."""
    t = device + "@@KER@@" + kernels
    for a, b in TER_EDITS:
        n = t.count(a)
        if n != 1:
            raise RuntimeError(f"dist_device: terrain anchor found {n} times: {a[:60]!r}")
        t = t.replace(a, b)
    t = t.replace("@TER_DEVICE@", TER_DEVICE)
    device, kernels = t.split("@@KER@@")
    return device, kernels


def apply(engine, m, slots, terrain=False):
    """The engine text (adjoint.cuda_source's, include removed) followed by the query tables, the distance code and
    the query kernels. terrain: a terrain build's engine text (the terrain query geom, TER_DEVICE; the pairs may hold
    TERRAIN_GEOM as geom 2). Returns (text, query geom ids)."""
    head, geoms = tables(m, slots, terrain)
    dev, ker = DEVICE, QUERY_KERNELS
    if terrain:
        dev, ker = terrain_text(dev, ker)
    return engine + head + dev + ker, geoms

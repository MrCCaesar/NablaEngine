"""Convex mesh collisions for the forest engine: MuJoCo Warp's general convex collision
(mujoco_warp/_src/collision_gjk.py: support, GJK, EPA, multicontact by polygon clipping; collision_convex.py: the
contact from the witness points; collision_primitive.py: plane_convex), written in CUDA for one lane per pair, in its
order of operations, so that the contacts agree with MuJoCo Warp's up to float rounding (pairs ordered as MuJoCo's
collision table: plane, sphere, capsule, box before mesh; margin and gap 0)."""
import mujoco
import numpy as np

MESH = 4
from nablaengine.g1_engine.mesh_tables import MESH_LAYOUT, constants, layout, mesh_tables


def header(m, geoms):
    """#defines and declarations of the convex collision code (HAS_MESH 0 without mesh collision geoms)."""
    mt = mesh_tables(m, geoms)
    if mt is None:
        return "#define HAS_MESH 0\n"
    s = mt["sizes"]
    epa_it = int(m.opt.ccd_iterations)
    decl = "".join(f"DKC {'int' if k == 'i' else 'float'} {sym}[{s[n]}];\n" for sym, k, n in MESH_LAYOUT)
    multiccd = int(not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD))
    return (f"#define HAS_MESH 1\n#define CCD_ITERATIONS {epa_it}\n#define CCD_TOLERANCE {float(m.opt.ccd_tolerance):.9g}f\n"
            f"#define MULTICCD {multiccd}\n#define NPOLYMAX {max(mt['poly_max'], 4)}\n"
            f"#define NMESHDEG {max(mt['deg_max'], 3)}\n" + decl)


COMMON = r"""
// vector helpers (forest_engine.convex) and the OBB test of the pairs the split forward computes in k_ccd
struct f3 { float x, y, z; };
__device__ __forceinline__ f3 mk3(float x, float y, float z) { f3 r; r.x = x; r.y = y; r.z = z; return r; }
__device__ __forceinline__ f3 ld3(const float* p) { return mk3(p[0], p[1], p[2]); }
__device__ __forceinline__ f3 operator+(f3 a, f3 b) { return mk3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ f3 operator-(f3 a, f3 b) { return mk3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ f3 operator-(f3 a) { return mk3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ f3 operator*(float s, f3 a) { return mk3(s * a.x, s * a.y, s * a.z); }
__device__ __forceinline__ f3 operator*(f3 a, float s) { return mk3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ f3 operator/(f3 a, float s) { return mk3(a.x / s, a.y / s, a.z / s); }
__device__ __forceinline__ float dt3(f3 a, f3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ f3 cr3(f3 a, f3 b) { return mk3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__device__ __forceinline__ float nm3(f3 a) { return sqrtf(dt3(a, a)); }
__device__ __forceinline__ f3 nz3(f3 a) { float n = nm3(a); return n > 0.f ? a / n : mk3(0.f, 0.f, 0.f); }  // wp.normalize
__device__ __forceinline__ float cvsign(float x) { return x < 0.f ? -1.f : 1.f; }  // wp.sign
__device__ __forceinline__ float cget(f3 a, int i) { return i == 0 ? a.x : (i == 1 ? a.y : a.z); }
__device__ __forceinline__ void cset(f3& a, int i, float v) { if (i == 0) a.x = v; else if (i == 1) a.y = v; else a.z = v; }
// R (row-major 3x3) times v, and its transpose
__device__ __forceinline__ f3 rmul(const float* R, f3 v) {
  return mk3(R[0] * v.x + R[1] * v.y + R[2] * v.z, R[3] * v.x + R[4] * v.y + R[5] * v.z, R[6] * v.x + R[7] * v.y + R[8] * v.z);
}
__device__ __forceinline__ f3 rtmul(const float* R, f3 v) {
  return mk3(R[0] * v.x + R[3] * v.y + R[6] * v.z, R[1] * v.x + R[4] * v.y + R[7] * v.z, R[2] * v.x + R[5] * v.y + R[8] * v.z);
}

// the geoms' oriented bounding boxes (MuJoCo's geom_aabb in the geom frames) are separated: a separating-axis test on
// the 15 axes with a slack of 1e-6 (+1e-6 on |R|: parallel edges), conservative, so a pair it drops has no contact
__device__ bool obb_separated(int g1, const float* p1, const float* q1, int g2, const float* p2, const float* q2) {
  float R1[9], R2[9];
  quat2mat(q1, R1);
  quat2mat(q2, R2);
  const float* a1 = c_gaabb + 6 * g1;
  const float* a2 = c_gaabb + 6 * g2;
  f3 c1 = ld3(p1) + rmul(R1, ld3(a1)), c2 = ld3(p2) + rmul(R2, ld3(a2));
  const float h1[3] = {a1[3], a1[4], a1[5]}, h2[3] = {a2[3], a2[4], a2[5]};
  f3 tw = c2 - c1;
  float t[3] = {R1[0] * tw.x + R1[3] * tw.y + R1[6] * tw.z, R1[1] * tw.x + R1[4] * tw.y + R1[7] * tw.z,
                R1[2] * tw.x + R1[5] * tw.y + R1[8] * tw.z};  // in frame 1
  float R[3][3], A[3][3];  // R = R1' R2: R[i][j] = axis i of 1 . axis j of 2
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      R[i][j] = R1[i] * R2[j] + R1[3 + i] * R2[3 + j] + R1[6 + i] * R2[6 + j];
      A[i][j] = fabsf(R[i][j]) + 1e-6f;
    }
  const float slack = 1e-6f;
  for (int i = 0; i < 3; i++)
    if (fabsf(t[i]) > h1[i] + h2[0] * A[i][0] + h2[1] * A[i][1] + h2[2] * A[i][2] + slack) return true;
  for (int j = 0; j < 3; j++)
    if (fabsf(t[0] * R[0][j] + t[1] * R[1][j] + t[2] * R[2][j]) > h1[0] * A[0][j] + h1[1] * A[1][j] + h1[2] * A[2][j] + h2[j] + slack)
      return true;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      const int i1 = (i + 1) % 3, i2 = (i + 2) % 3, j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      float ra = h1[i1] * A[i2][j] + h1[i2] * A[i1][j], rb = h2[j1] * A[i][j2] + h2[j2] * A[i][j1];
      if (fabsf(t[i2] * R[i1][j] - t[i1] * R[i2][j]) > ra + rb + slack) return true;
    }
  return false;
}

"""

DEVICE = r"""
#if HAS_MESH
// ---------------------------------------------------------------- convex collisions (MuJoCo Warp's collision_gjk.py)
#define CV_FLOAT_MAX 1e30f
#define CV_FLOAT_MIN (-1e30f)
#define CV_MINVAL 1e-15f
#define CV_MINVAL2 1e-30f
#define CV_MIN_DIST2 1e-10f
#define CV_MIN_DIST3 1e-10f
#define CV_MIN_DIST4 1e-17f
#define CV_MIN_EPATOL 1e-7f
#define CV_FACE_TOL 0.99605140f  // cos(0.0889)
#define CV_EDGE_TOL 0.08878295f  // sin(0.0889)
#define CV_INTERSECT_TOL 0.0000003f
#define EPA_NV (10 + 2 * CCD_ITERATIONS)  // polytope vertices (pairs: geom 1, geom 2)
#define EPA_NF (6 + 5 * CCD_ITERATIONS)   // faces (MJ_MAX_EPAFACES = 5)
#define EPA_NH 24                         // horizon edges (MJ_MAX_EPAHORIZON)
#define FACE_DELETED 0x80000000u
#define FACE_INVALID 0x40000000u

// a geom as MuJoCo Warp's collision_core.Geom (the fields the convex code reads); margins are 0
struct CGeom {
  f3 pos;
  float R[9];
  f3 size;
  int type;   // the engine's: 1 sphere, 2 capsule, 3 box, 4 mesh
  int mesh;   // compact mesh index (c_gmesh), -1
  int index;  // the support's cached vertex (MuJoCo Warp's geom.index)
};
struct SupportPoint { f3 point; int cached; int vidx; };

__device__ __forceinline__ f3 mvert(int mesh, int i) {
  const float* v = c_mvert + 3 * (c_mvadr[mesh] + i);
  return mk3(__ldg(v), __ldg(v + 1), __ldg(v + 2));
}
#ifndef CV_EXHAUSTIVE
#define CV_EXHAUSTIVE 1  // 1: the support by exhaustive search over the vertices (MuJoCo Warp: below 10 vertices)
#endif

// support point of a geom in direction dir (MuJoCo Warp's support)
__device__ SupportPoint cv_support(const CGeom& g, f3 dir) {
  SupportPoint sp;
  sp.cached = -1;
  sp.vidx = -1;
  if (g.type == 1) { sp.point = g.pos + g.size.x * dir; return sp; }
  f3 ld = rtmul(g.R, dir);
  if (g.type == 3) {
    f3 tmp = mk3(cvsign(ld.x), cvsign(ld.y), cvsign(ld.z));
    f3 res = mk3(tmp.x * g.size.x, tmp.y * g.size.y, tmp.z * g.size.z);
    sp.point = rmul(g.R, res) + g.pos;
    sp.vidx = (tmp.x > 0.f ? 1 : 0) + (tmp.y > 0.f ? 2 : 0) + (tmp.z > 0.f ? 4 : 0);
    return sp;
  }
  if (g.type == 2) {
    f3 res = g.size.x * ld;
    res.z += cvsign(ld.z) * g.size.y;
    sp.point = rmul(g.R, res) + g.pos;
    return sp;
  }
  // mesh
  const int vn = c_mvnum[g.mesh], ga = c_mgadr[g.mesh];
  float max_dist = CV_FLOAT_MIN;
  f3 pt = mk3(0.f, 0.f, 0.f);
  if (CV_EXHAUSTIVE || ga == -1 || vn < 10) {
    if (g.index > -1) { sp.cached = g.index; max_dist = dt3(mvert(g.mesh, g.index), ld); pt = mvert(g.mesh, g.index); }
    for (int i = 0; i < vn; i++) {
      f3 v = mvert(g.mesh, i);
      float d = dt3(v, ld);
      if (d > max_dist) { max_dist = d; pt = v; sp.cached = i; }
    }
    sp.vidx = sp.cached;
  } else {
    const int* gr = c_mgraph + ga;
    const int numvert = gr[0], vert_edgeadr = 2, vert_globalid = 2 + numvert, edge_localid = 2 + 2 * numvert;
    int prev = -1, imax = g.index > -1 ? g.index : 0;
    max_dist = dt3(ld, mvert(g.mesh, gr[vert_globalid + imax]));
    while (imax != prev) {  // hill-climb until no change
      prev = imax;
      int i = gr[vert_edgeadr + imax];
      int subidx = gr[edge_localid + i];
      while (subidx >= 0) {
        int idx = gr[vert_globalid + subidx];
        float d = dt3(ld, mvert(g.mesh, idx));
        if (d > max_dist) { imax = subidx; max_dist = d; }
        i++;
        subidx = gr[edge_localid + i];
      }
    }
    sp.cached = imax;
    sp.vidx = gr[vert_globalid + imax];
    pt = mvert(g.mesh, sp.vidx);
  }
  sp.point = rmul(g.R, pt) + g.pos;
  return sp;
}

struct Simplex { f3 v[4]; };
struct GJKResult {
  bool separated;
  float dist;
  f3 x1, x2;
  int dim;
  f3 simplex[4], simplex1[4], simplex2[4];
  int sidx1[4], sidx2[4];
  int index1, index2;
};

__device__ __forceinline__ f3 lin_combine(int n, const float* l, const f3* m) {
  if (n == 1) return l[0] * m[0];
  if (n == 2) return l[0] * m[0] + l[1] * m[1];
  if (n == 3) return l[0] * m[0] + l[1] * m[1] + l[2] * m[2];
  return l[0] * m[0] + l[1] * m[1] + l[2] * m[2] + l[3] * m[3];
}
__device__ __forceinline__ float det3f(f3 a, f3 b, f3 c) { return dt3(a, cr3(b, c)); }
__device__ __forceinline__ int same_sign(float a, float b) {
  if (a > 0.f && b > 0.f) return 1;
  if (a < 0.f && b < 0.f) return -1;
  return 0;
}
__device__ __forceinline__ f3 proj_origin_line(f3 v1, f3 v2) {
  f3 diff = v2 - v1;
  float scl = -(dt3(v2, diff) / dt3(diff, diff));
  return v2 + scl * diff;
}
// returns ret (1: degenerate)
__device__ __forceinline__ int proj_origin_plane(f3 v1, f3 v2, f3 v3, f3* r) {
  f3 d21 = v2 - v1, d31 = v3 - v1, d32 = v3 - v2;
  f3 n = cr3(d32, d21);
  float nv = dt3(n, v2), nn = dt3(n, n);
  if (nn == 0.f) { *r = mk3(0.f, 0.f, 0.f); return 1; }
  if (nv != 0.f && nn > CV_MINVAL) { *r = (nv / nn) * n; return 0; }
  n = cr3(d21, d31);
  nv = dt3(n, v1); nn = dt3(n, n);
  if (nn == 0.f) { *r = mk3(0.f, 0.f, 0.f); return 1; }
  if (nv != 0.f && nn > CV_MINVAL) { *r = (nv / nn) * n; return 0; }
  n = cr3(d31, d32);
  nv = dt3(n, v3); nn = dt3(n, n);
  *r = (nv / nn) * n;
  return 0;
}
__device__ void S1D(f3 s1, f3 s2, float* l) {
  f3 p_o = proj_origin_line(s1, s2);
  float mu_max = s1.x - s2.x;
  int index = 0;
  float mu = s1.y - s2.y;
  if (fabsf(mu) >= fabsf(mu_max)) { mu_max = mu; index = 1; }
  mu = s1.z - s2.z;
  if (fabsf(mu) >= fabsf(mu_max)) { mu_max = mu; index = 2; }
  float C1 = cget(p_o, index) - cget(s2, index), C2 = cget(s1, index) - cget(p_o, index);
  if (same_sign(mu_max, C1) && same_sign(mu_max, C2)) { l[0] = C1 / mu_max; l[1] = C2 / mu_max; return; }
  l[0] = 0.f; l[1] = 1.f;
}
__device__ void S2D(f3 s1, f3 s2, f3 s3, float* l) {
  f3 p_o;
  if (proj_origin_plane(s1, s2, s3, &p_o)) { S1D(s1, s2, l); l[2] = 0.f; return; }
  float M_14 = s2.y * s3.z - s2.z * s3.y - s1.y * s3.z + s1.z * s3.y + s1.y * s2.z - s1.z * s2.y;
  float M_24 = s2.x * s3.z - s2.z * s3.x - s1.x * s3.z + s1.z * s3.x + s1.x * s2.z - s1.z * s2.x;
  float M_34 = s2.x * s3.y - s2.y * s3.x - s1.x * s3.y + s1.y * s3.x + s1.x * s2.y - s1.y * s2.x;
  float M_max, a1, a2, b1, b2, c1, c2, p1, p2;
  float mu1 = fabsf(M_14), mu2 = fabsf(M_24), mu3 = fabsf(M_34);
  if (mu1 >= mu2 && mu1 >= mu3) { M_max = M_14; a1 = s1.y; a2 = s1.z; b1 = s2.y; b2 = s2.z; c1 = s3.y; c2 = s3.z; p1 = p_o.y; p2 = p_o.z; }
  else if (mu2 >= mu3) { M_max = M_24; a1 = s1.x; a2 = s1.z; b1 = s2.x; b2 = s2.z; c1 = s3.x; c2 = s3.z; p1 = p_o.x; p2 = p_o.z; }
  else { M_max = M_34; a1 = s1.x; a2 = s1.y; b1 = s2.x; b2 = s2.y; c1 = s3.x; c2 = s3.y; p1 = p_o.x; p2 = p_o.y; }
  float C31 = p1 * b2 + p2 * c1 + b1 * c2 - p1 * c2 - p2 * b1 - c1 * b2;
  float C32 = p1 * c2 + p2 * a1 + c1 * a2 - p1 * a2 - p2 * c1 - a1 * c2;
  float C33 = p1 * a2 + p2 * b1 + a1 * b2 - p1 * b2 - p2 * a1 - b1 * a2;
  int comp1 = same_sign(M_max, C31), comp2 = same_sign(M_max, C32), comp3 = same_sign(M_max, C33);
  if (comp1 && comp2 && comp3) { l[0] = C31 / M_max; l[1] = C32 / M_max; l[2] = C33 / M_max; return; }
  float dmin = CV_FLOAT_MAX, sub[2];
  l[0] = l[1] = l[2] = 0.f;
  if (!comp1) {
    S1D(s2, s3, sub);
    f3 x = sub[0] * s2 + sub[1] * s3;
    l[0] = 0.f; l[1] = sub[0]; l[2] = sub[1];
    dmin = dt3(x, x);
  }
  if (!comp2) {
    S1D(s1, s3, sub);
    f3 x = sub[0] * s1 + sub[1] * s3;
    float d = dt3(x, x);
    if (d < dmin) { l[0] = sub[0]; l[1] = 0.f; l[2] = sub[1]; dmin = d; }
  }
  if (!comp3) {
    S1D(s1, s2, sub);
    f3 x = sub[0] * s1 + sub[1] * s2;
    float d = dt3(x, x);
    if (d < dmin) { l[0] = sub[0]; l[1] = sub[1]; l[2] = 0.f; }
  }
}
__device__ void S3D(f3 s1, f3 s2, f3 s3, f3 s4, float* l) {
  float C41 = -det3f(s2, s3, s4), C42 = det3f(s1, s3, s4), C43 = -det3f(s1, s2, s4), C44 = det3f(s1, s2, s3);
  float m_det = C41 + C42 + C43 + C44;
  int comp1 = same_sign(m_det, C41), comp2 = same_sign(m_det, C42), comp3 = same_sign(m_det, C43),
      comp4 = same_sign(m_det, C44);
  if (comp1 && comp2 && comp3 && comp4) { l[0] = C41 / m_det; l[1] = C42 / m_det; l[2] = C43 / m_det; l[3] = C44 / m_det; return; }
  l[0] = l[1] = l[2] = l[3] = 0.f;
  float dmin = CV_FLOAT_MAX, sub[3];
  if (!comp1) {
    S2D(s2, s3, s4, sub);
    f3 x = sub[0] * s2 + sub[1] * s3 + sub[2] * s4;
    l[0] = 0.f; l[1] = sub[0]; l[2] = sub[1]; l[3] = sub[2];
    dmin = dt3(x, x);
  }
  if (!comp2) {
    S2D(s1, s3, s4, sub);
    f3 x = sub[0] * s1 + sub[1] * s3 + sub[2] * s4;
    float d = dt3(x, x);
    if (d < dmin) { l[0] = sub[0]; l[1] = 0.f; l[2] = sub[1]; l[3] = sub[2]; dmin = d; }
  }
  if (!comp3) {
    S2D(s1, s2, s4, sub);
    f3 x = sub[0] * s1 + sub[1] * s2 + sub[2] * s4;
    float d = dt3(x, x);
    if (d < dmin) { l[0] = sub[0]; l[1] = sub[1]; l[2] = 0.f; l[3] = sub[2]; dmin = d; }
  }
  if (!comp4) {
    S2D(s1, s2, s3, sub);
    f3 x = sub[0] * s1 + sub[1] * s2 + sub[2] * s3;
    float d = dt3(x, x);
    if (d < dmin) { l[0] = sub[0]; l[1] = sub[1]; l[2] = sub[2]; l[3] = 0.f; }
  }
}
__device__ __forceinline__ void subdistance(int n, const f3* s, float* l) {
  if (n == 4) { S3D(s[0], s[1], s[2], s[3], l); return; }
  if (n == 3) { S2D(s[0], s[1], s[2], l); l[3] = 0.f; return; }
  if (n == 2) { S1D(s[0], s[1], l); l[2] = l[3] = 0.f; return; }
  l[0] = 1.f; l[1] = l[2] = l[3] = 0.f;
}

__device__ __forceinline__ bool is_discrete(int t1, int t2) { return (t1 == 3 || t1 == 4) && (t2 == 3 || t2 == 4); }

// GJK (MuJoCo Warp's gjk), cutoff 0 or the capsule / sphere radii
__device__ void cv_gjk(float tol, int iters, CGeom& g1, CGeom& g2, f3 x1_0, f3 x2_0, float cutoff, bool disc,
                       GJKResult& r) {
  f3 simplex[4], simplex1[4], simplex2[4];
  int si1[4] = {0, 0, 0, 0}, si2[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; i++) simplex[i] = simplex1[i] = simplex2[i] = mk3(0.f, 0.f, 0.f);
  int n = 0;
  float l[4] = {1.f, 0.f, 0.f, 0.f};
  float epsilon = disc ? 0.f : 0.5f * tol * tol, min_norm = disc ? CV_MINVAL : tol;
  f3 x_k = x1_0 - x2_0;
  float xnorm = sqrtf(dt3(x_k, x_k)), xnorm_prev = 0.f;
  for (int it = 0; it < iters; it++) {
    if (xnorm < min_norm || fabsf(xnorm_prev - xnorm) < CV_MINVAL) break;
    f3 dir_neg = x_k / xnorm;  // (_gjk_support: direction tuning for discrete geoms)
    if (disc && xnorm < 1e-4f) {
      if (n == 2) {
        f3 edge = simplex[1] - simplex[0];
        float e2 = dt3(edge, edge);
        if (e2 > CV_MINVAL2) {
          float proj = dt3(dir_neg, edge) / e2;
          dir_neg = dir_neg - proj * edge;
          float dn = nm3(dir_neg);
          if (dn > CV_MINVAL) dir_neg = dir_neg / dn;
        }
      } else if (n == 3) {
        f3 e1 = simplex[1] - simplex[0], e2 = simplex[2] - simplex[0], nrm = cr3(e1, e2);
        float nn = nm3(nrm);
        if (nn > CV_MINVAL) dir_neg = cvsign(dt3(dir_neg, nrm)) * nrm / nn;
      }
    }
    SupportPoint sp1 = cv_support(g1, -dir_neg), sp2 = cv_support(g2, dir_neg);
    simplex1[n] = sp1.point; g1.index = sp1.cached; si1[n] = sp1.vidx;
    simplex2[n] = sp2.point; g2.index = sp2.cached; si2[n] = sp2.vidx;
    simplex[n] = simplex1[n] - simplex2[n];
    if (dt3(x_k, x_k - simplex[n]) < epsilon) break;
    float lower = dt3(x_k, simplex[n]);
    if (cutoff == 0.f) {
      if (lower > 0.f) { r.separated = true; r.dim = 0; r.dist = CV_FLOAT_MAX; r.index1 = g1.index; r.index2 = g2.index; return; }
    } else if (cutoff < CV_FLOAT_MAX) {
      if (lower > 0.f && lower >= cutoff * xnorm) {
        r.separated = true; r.dim = 0; r.dist = CV_FLOAT_MAX; r.index1 = g1.index; r.index2 = g2.index; return;
      }
    }
    subdistance(n + 1, simplex, l);
    n = 0;
    for (int i = 0; i < 4; i++) {
      if (l[i] == 0.f) continue;
      simplex[n] = simplex[i]; simplex1[n] = simplex1[i]; simplex2[n] = simplex2[i];
      si1[n] = si1[i]; si2[n] = si2[i]; l[n] = l[i];
      n++;
    }
    if (n < 1) break;
    x_k = lin_combine(n, l, simplex);
    xnorm_prev = xnorm;
    xnorm = sqrtf(dt3(x_k, x_k));
    if (n == 4) break;
  }
  r.separated = false;
  r.x1 = n == 0 ? x1_0 : lin_combine(n, l, simplex1);
  r.x2 = n == 0 ? x2_0 : lin_combine(n, l, simplex2);
  if (xnorm > 0.f) {
    f3 dir = x_k / xnorm;
    SupportPoint sp1 = cv_support(g1, -dir), sp2 = cv_support(g2, dir);
    r.separated = dt3(x_k, sp1.point - sp2.point) > 0.f;
  }
  r.dist = (n == 4 && !r.separated) ? 0.f : xnorm;
  r.dim = n;
  for (int i = 0; i < 4; i++) {
    r.simplex[i] = simplex[i]; r.simplex1[i] = simplex1[i]; r.simplex2[i] = simplex2[i];
    r.sidx1[i] = si1[i]; r.sidx2[i] = si2[i];
  }
  r.index1 = g1.index;
  r.index2 = g2.index;
}

// ------------------------------------------------------------------------------------------------ EPA polytope
struct Polytope {
  f3 vert[EPA_NV];
  int vidx[EPA_NV];
  int nvert;
  f3 center;
  unsigned face[EPA_NF];
  f3 face_pr[EPA_NF];
  float face_norm2[EPA_NF];
  int nface;
  int horizon[EPA_NH];
  int nhorizon;
  int status;
};
__device__ __forceinline__ void face_verts(unsigned f, int* v) { v[0] = f & 0x3FF; v[1] = (f >> 10) & 0x3FF; v[2] = (f >> 20) & 0x3FF; }
__device__ __forceinline__ bool face_deleted(unsigned f) { return f & FACE_DELETED; }
__device__ __forceinline__ bool face_invalid(unsigned f) { return f & (FACE_DELETED | FACE_INVALID); }

__device__ float attach_face(Polytope& pt, int idx, int v1, int v2, int v3) {
  if (pt.nface == EPA_NF) return 0.f;
  f3 p1 = pt.vert[2 * v1] - pt.vert[2 * v1 + 1], p2 = pt.vert[2 * v2] - pt.vert[2 * v2 + 1],
     p3 = pt.vert[2 * v3] - pt.vert[2 * v3 + 1], r;
  if (proj_origin_plane(p3, p2, p1, &r)) return 0.f;
  if (dt3(r, p1 - pt.center) < 0.f) r = -r;
  pt.face[idx] = (unsigned)(v1 + (v2 << 10) + (v3 << 20));
  pt.face_pr[idx] = r;
  pt.face_norm2[idx] = dt3(r, r);
  return pt.face_norm2[idx];
}
__device__ void epa_support(Polytope& pt, int idx, CGeom& g1, CGeom& g2, f3 dir, int* i1, int* i2) {
  SupportPoint sp = cv_support(g1, dir);
  pt.vert[2 * idx] = sp.point; pt.vidx[2 * idx] = sp.vidx; *i1 = sp.cached;
  sp = cv_support(g2, -dir);
  pt.vert[2 * idx + 1] = sp.point; pt.vidx[2 * idx + 1] = sp.vidx; *i2 = sp.cached;
}
__device__ void replace_simplex3(const Polytope& pt, int v1, int v2, int v3, GJKResult& r) {
  const int v[3] = {v1, v2, v3};
  for (int i = 0; i < 3; i++) {
    r.simplex1[i] = pt.vert[2 * v[i]];
    r.simplex2[i] = pt.vert[2 * v[i] + 1];
    r.simplex[i] = r.simplex1[i] - r.simplex2[i];
    r.sidx1[i] = pt.vidx[2 * v[i]];
    r.sidx2[i] = pt.vidx[2 * v[i] + 1];
  }
}
__device__ void rotmat120(f3 axis, float* R) {
  float n = nm3(axis), u1 = axis.x / n, u2 = axis.y / n, u3 = axis.z / n;
  const float s = 0.86602540378f, c = -0.5f;
  R[0] = c + u1 * u1 * (1.f - c); R[1] = u1 * u2 * (1.f - c) - u3 * s; R[2] = u1 * u3 * (1.f - c) + u2 * s;
  R[3] = u2 * u1 * (1.f - c) + u3 * s; R[4] = c + u2 * u2 * (1.f - c); R[5] = u2 * u3 * (1.f - c) - u1 * s;
  R[6] = u1 * u3 * (1.f - c) - u2 * s; R[7] = u2 * u3 * (1.f - c) + u1 * s; R[8] = c + u3 * u3 * (1.f - c);
}
__device__ int ray_triangle(f3 v1, f3 v2, f3 v3, f3 v4, f3 v5) {
  float vol1 = det3f(v3 - v1, v4 - v1, v2 - v1), vol2 = det3f(v4 - v1, v5 - v1, v2 - v1), vol3 = det3f(v5 - v1, v3 - v1, v2 - v1);
  if (vol1 >= 0.f && vol2 >= 0.f && vol3 >= 0.f) return 1;
  if (vol1 <= 0.f && vol2 <= 0.f && vol3 <= 0.f) return -1;
  return 0;
}
__device__ f3 tri_affine(f3 v1, f3 v2, f3 v3, f3 p) {
  float M_14 = v2.y * v3.z - v2.z * v3.y - v1.y * v3.z + v1.z * v3.y + v1.y * v2.z - v1.z * v2.y;
  float M_24 = v2.x * v3.z - v2.z * v3.x - v1.x * v3.z + v1.z * v3.x + v1.x * v2.z - v1.z * v2.x;
  float M_34 = v2.x * v3.y - v2.y * v3.x - v1.x * v3.y + v1.y * v3.x + v1.x * v2.y - v1.y * v2.x;
  float M_max;
  int x, y;
  float mu1 = fabsf(M_14), mu2 = fabsf(M_24), mu3 = fabsf(M_34);
  if (mu1 >= mu2 && mu1 >= mu3) { M_max = M_14; x = 1; y = 2; }
  else if (mu2 >= mu3) { M_max = M_24; x = 0; y = 2; }
  else { M_max = M_34; x = 0; y = 1; }
  float px = cget(p, x), py = cget(p, y), ax = cget(v1, x), ay = cget(v1, y), bx = cget(v2, x), by = cget(v2, y),
        cx = cget(v3, x), cy = cget(v3, y);
  float C31 = px * by + py * cx + bx * cy - px * cy - py * bx - cx * by;
  float C32 = px * cy + py * ax + cx * ay - px * ay - py * cx - ax * cy;
  float C33 = px * ay + py * bx + ax * by - px * by - py * ax - bx * ay;
  return mk3(C31 / M_max, C32 / M_max, C33 / M_max);
}
__device__ bool tri_point_intersect(f3 v1, f3 v2, f3 v3, f3 p) {
  f3 c = tri_affine(v1, v2, v3, p);
  if (c.x < 0.f || c.y < 0.f || c.z < 0.f) return false;
  f3 pr = mk3(v1.x * c.x + v2.x * c.y + v3.x * c.z, v1.y * c.x + v2.y * c.y + v3.y * c.z, v1.z * c.x + v2.z * c.y + v3.z * c.z);
  return nm3(pr - p) < CV_MINVAL;
}
__device__ bool same_side(f3 p0, f3 p1, f3 p2, f3 p3) {
  f3 n = cr3(p1 - p0, p2 - p0);
  float d1 = dt3(n, p3 - p0), d2 = dt3(n, -p0);
  return (d1 > 0.f && d2 > 0.f) || (d1 < 0.f && d2 < 0.f);
}
__device__ bool test_tetra(f3 p0, f3 p1, f3 p2, f3 p3) {
  return same_side(p0, p1, p2, p3) && same_side(p1, p2, p3, p0) && same_side(p2, p3, p0, p1) && same_side(p3, p0, p1, p2);
}
__device__ int add_edge(Polytope& pt, int e1, int e2) {
  int n = pt.nhorizon;
  if (n < 0) return -1;
  int edge = (min(e1, e2) << 10) | max(e1, e2);
  for (int i = 0; i < n; i++)
    if (edge == pt.horizon[i]) { pt.horizon[i] = pt.horizon[n - 1]; return n - 1; }
  if (n == EPA_NH) return -1;
  pt.horizon[n] = edge;
  return n + 1;
}

__device__ void polytope2(Polytope& pt, GJKResult& r, CGeom& g1, CGeom& g2) {
  f3 diff = r.simplex[1] - r.simplex[0];
  pt.center = 0.5f * (r.simplex[0] + r.simplex[1]);
  float value = CV_FLOAT_MAX;
  int index = 0;
  for (int i = 0; i < 3; i++)
    if (fabsf(cget(diff, i)) < value) { value = fabsf(cget(diff, i)); index = i; }
  f3 e = mk3(0.f, 0.f, 0.f);
  cset(e, index, 1.f);
  f3 d1 = cr3(e, diff);
  float R[9];
  rotmat120(diff, R);
  f3 d2 = rmul(R, d1), d3 = rmul(R, d2);
  pt.vert[0] = r.simplex1[0]; pt.vert[1] = r.simplex2[0]; pt.vert[2] = r.simplex1[1]; pt.vert[3] = r.simplex2[1];
  pt.vidx[0] = r.sidx1[0]; pt.vidx[1] = r.sidx2[0]; pt.vidx[2] = r.sidx1[1]; pt.vidx[3] = r.sidx2[1];
  int i1, i2;
  epa_support(pt, 2, g1, g2, d1 / nm3(d1), &i1, &i2);
  epa_support(pt, 3, g1, g2, d2 / nm3(d2), &i1, &i2);
  epa_support(pt, 4, g1, g2, d3 / nm3(d3), &i1, &i2);
  const int F[6][3] = {{0, 2, 3}, {0, 4, 2}, {0, 3, 4}, {1, 3, 2}, {1, 2, 4}, {1, 4, 3}};
  for (int f = 0; f < 6; f++)
    if (attach_face(pt, f, F[f][0], F[f][1], F[f][2]) < CV_MIN_DIST2) {
      pt.status = -1;
      replace_simplex3(pt, F[f][0], F[f][1], F[f][2], r);
      return;
    }
  f3 v2 = pt.vert[4] - pt.vert[5], v3 = pt.vert[6] - pt.vert[7], v4 = pt.vert[8] - pt.vert[9];
  if (!ray_triangle(r.simplex[0], r.simplex[1], v2, v3, v4)) { pt.status = 1; return; }
  pt.nvert = 5; pt.nface = 6; pt.status = 0;
}

__device__ void polytope3(Polytope& pt, const GJKResult& r, CGeom& g1, CGeom& g2) {
  pt.center = (r.simplex[0] + r.simplex[1] + r.simplex[2]) * (1.f / 3.f) ;
  f3 n = cr3(r.simplex[1] - r.simplex[0], r.simplex[2] - r.simplex[0]);
  float norm = nm3(n);
  if (norm < CV_MINVAL) { pt.status = 2; return; }
  n = n / norm;
  for (int i = 0; i < 3; i++) {
    pt.vert[2 * i] = r.simplex1[i]; pt.vert[2 * i + 1] = r.simplex2[i];
    pt.vidx[2 * i] = r.sidx1[i]; pt.vidx[2 * i + 1] = r.sidx2[i];
  }
  int i1, i2;
  epa_support(pt, 3, g1, g2, -n, &i1, &i2);
  epa_support(pt, 4, g1, g2, n, &i1, &i2);
  f3 v1 = r.simplex[0], v2 = r.simplex[1], v3 = r.simplex[2], v4 = pt.vert[6] - pt.vert[7], v5 = pt.vert[8] - pt.vert[9];
  if (tri_point_intersect(v1, v2, v3, v4)) { pt.status = 3; return; }
  if (tri_point_intersect(v1, v2, v3, v5)) { pt.status = 4; return; }
  if (r.dist > 1e-5f && !test_tetra(v1, v2, v3, v4) && !test_tetra(v1, v2, v3, v5)) { pt.status = 5; return; }
  const int F[6][3] = {{4, 0, 1}, {4, 2, 0}, {4, 1, 2}, {3, 1, 0}, {3, 0, 2}, {3, 2, 1}};
  for (int f = 0; f < 6; f++)
    if (attach_face(pt, f, F[f][0], F[f][1], F[f][2]) < CV_MIN_DIST3) { pt.status = 6 + f; return; }
  pt.nvert = 5; pt.nface = 6; pt.status = 0;
}
"""

DEVICE += r"""

__device__ void polytope4(Polytope& pt, GJKResult& r) {
  pt.center = 0.25f * (r.simplex[0] + r.simplex[1] + r.simplex[2] + r.simplex[3]);
  for (int i = 0; i < 4; i++) {
    pt.vert[2 * i] = r.simplex1[i]; pt.vert[2 * i + 1] = r.simplex2[i];
    pt.vidx[2 * i] = r.sidx1[i]; pt.vidx[2 * i + 1] = r.sidx2[i];
  }
  const int F[4][3] = {{0, 1, 2}, {0, 3, 1}, {0, 2, 3}, {3, 2, 1}};
  float dist[4];
  int idx = 0;
  for (int f = 0; f < 4; f++) {
    dist[f] = attach_face(pt, f, F[f][0], F[f][1], F[f][2]);
    if (dist[f] < CV_MIN_DIST4) { pt.status = -1; replace_simplex3(pt, F[f][0], F[f][1], F[f][2], r); return; }
    if (f == 1) idx = dist[0] < dist[1] ? 0 : 1;
    else if (f > 1) idx = dist[f] < dist[idx] ? f : idx;
  }
  if (!test_tetra(r.simplex[0], r.simplex[1], r.simplex[2], r.simplex[3])) {
    if (dist[idx] > CV_MINVAL) { pt.status = 12; return; }
    pt.status = -1;
    replace_simplex3(pt, F[idx][0], F[idx][1], F[idx][2], r);
    return;
  }
  pt.nvert = 4; pt.nface = 4; pt.status = 0;
}

// EPA (MuJoCo Warp's _epa): the closest face idx (-1: none) and the witness points
__device__ int cv_epa(float tol, int iters, Polytope& pt, CGeom& g1, CGeom& g2, bool disc) {
  float upper = CV_FLOAT_MAX, upper2 = CV_FLOAT_MAX;
  int idx = -1, pidx = -1;
  float epsilon = disc ? CV_MIN_EPATOL : tol;
  int nvalid = pt.nface;
  iters = min(iters, 1000);
  for (int it = 0; it < iters; it++) {
    pidx = idx;
    idx = -1;
    float lower2 = CV_FLOAT_MAX;
    for (int i = 0; i < pt.nface; i++)
      if (!face_invalid(pt.face[i]) && pt.face_norm2[i] < lower2) { idx = i; lower2 = pt.face_norm2[i]; }
    if (lower2 > upper2 || idx < 0) { idx = pidx; break; }
    if (lower2 <= 0.f) break;
    float lower = sqrtf(lower2);
    int wi = pt.nvert;
    if (2 * wi + 1 >= EPA_NV) break;  // (MuJoCo Warp's buffer holds iters + 5 vertices: not reached)
    f3 face_pr = pt.face_pr[idx];
    int i1, i2;
    epa_support(pt, wi, g1, g2, face_pr / lower, &i1, &i2);
    f3 w = pt.vert[2 * wi] - pt.vert[2 * wi + 1];
    g1.index = i1; g2.index = i2;
    pt.nvert++;
    float upper_k = dt3(face_pr, w) / lower;
    if (upper_k < upper) { upper = upper_k; upper2 = upper * upper; }
    if (upper - lower < epsilon) break;
    if (disc) {
      bool rep = false;
      for (int i = 0; i < pt.nvert - 1; i++)
        if (pt.vidx[2 * i] == pt.vidx[2 * wi] && pt.vidx[2 * i + 1] == pt.vidx[2 * wi + 1]) { rep = true; break; }
      if (rep) break;
    }
    nvalid--;
    pt.face[idx] |= FACE_DELETED;
    int fv[3];
    face_verts(pt.face[idx], fv);
    pt.nhorizon = add_edge(pt, fv[0], fv[1]);
    pt.nhorizon = add_edge(pt, fv[1], fv[2]);
    pt.nhorizon = add_edge(pt, fv[2], fv[0]);
    if (pt.nhorizon == -1) { idx = -1; break; }
    for (int i = 0; i < pt.nface; i++) {
      if (face_deleted(pt.face[i])) continue;
      if (dt3(pt.face_pr[i], w) - pt.face_norm2[i] > 1e-10f) {
        if (!face_invalid(pt.face[i])) nvalid--;
        pt.face[i] |= FACE_DELETED;
        face_verts(pt.face[i], fv);
        pt.nhorizon = add_edge(pt, fv[0], fv[1]);
        pt.nhorizon = add_edge(pt, fv[1], fv[2]);
        pt.nhorizon = add_edge(pt, fv[2], fv[0]);
        if (pt.nhorizon == -1) { idx = -1; break; }
      }
    }
    for (int i = 0; i < pt.nhorizon && idx != -1; i++) {
      int e0 = pt.horizon[i] & 0x3FF, e1 = (pt.horizon[i] >> 10) & 0x3FF;
      float d2 = attach_face(pt, pt.nface, wi, e0, e1);
      if (d2 == 0.f) { idx = -1; break; }
      pt.nface++;
      if (d2 >= lower2 && d2 <= upper2) nvalid++;
      else pt.face[pt.nface - 1] |= FACE_INVALID;
    }
    if (nvalid == 0 || idx == -1) break;
    pt.nhorizon = 0;
  }
  return idx;
}
// the witness points of face idx (MuJoCo Warp's _epa_witness without the height field case); returns the distance
__device__ float epa_witness(const Polytope& pt, int idx, f3* x1, f3* x2) {
  int fv[3];
  face_verts(pt.face[idx], fv);
  f3 v1 = pt.vert[2 * fv[0]] - pt.vert[2 * fv[0] + 1], v2 = pt.vert[2 * fv[1]] - pt.vert[2 * fv[1] + 1],
     v3 = pt.vert[2 * fv[2]] - pt.vert[2 * fv[2] + 1];
  f3 c = tri_affine(v1, v2, v3, pt.face_pr[idx]);
  v1 = pt.vert[2 * fv[0] + 1]; v2 = pt.vert[2 * fv[1] + 1]; v3 = pt.vert[2 * fv[2] + 1];
  *x2 = mk3(v1.x * c.x + v2.x * c.y + v3.x * c.z, v1.y * c.x + v2.y * c.y + v3.y * c.z, v1.z * c.x + v2.z * c.y + v3.z * c.z);
  v1 = pt.vert[2 * fv[0]]; v2 = pt.vert[2 * fv[1]]; v3 = pt.vert[2 * fv[2]];
  *x1 = mk3(v1.x * c.x + v2.x * c.y + v3.x * c.z, v1.y * c.x + v2.y * c.y + v3.y * c.z, v1.z * c.x + v2.z * c.y + v3.z * c.z);
  return -sqrtf(pt.face_norm2[idx]);
}

// ------------------------------------------------------------------------------------- multicontact (polygon clip)
struct MultiBuf {  // MuJoCo Warp's per-contact multiccd buffers
  f3 polygon[2 * NPOLYMAX], clipped[2 * NPOLYMAX], pnormal[NPOLYMAX];
  float pdist[NPOLYMAX];
  int idx1[NMESHDEG], idx2[NMESHDEG];
  f3 n1[NMESHDEG], n2[NMESHDEG], endvert[NMESHDEG], face1[NPOLYMAX], face2[NPOLYMAX];
};
__device__ int feature_dim(const int* fv, const Polytope& pt, int off, int* fi, f3* fx) {
  int v1i = pt.vidx[2 * fv[0] + off], v2i = pt.vidx[2 * fv[1] + off], v3i = pt.vidx[2 * fv[2] + off];
  fi[0] = v1i; fi[1] = v2i; fi[2] = v3i;
  fx[0] = pt.vert[2 * fv[0] + off]; fx[1] = pt.vert[2 * fv[1] + off]; fx[2] = pt.vert[2 * fv[2] + off];
  if (v1i != v2i) return (v3i == v1i || v3i == v2i) ? 2 : 3;
  fi[1] = v3i;
  fx[1] = pt.vert[2 * fv[2] + off];
  return v1i != v3i ? 2 : 1;
}
__device__ int aligned_faces(const f3* a, int la, const f3* b, int lb, int* res) {
  for (int i = 0; i < la; i++)
    for (int j = 0; j < lb; j++)
      if (dt3(a[i], b[j]) < -CV_FACE_TOL) { res[0] = i; res[1] = j; return 1; }
  return 0;
}
__device__ int aligned_face_edge(const f3* edge, int nedge, const f3* face, int nface, int* res) {
  for (int i = 0; i < nface; i++)
    for (int j = 0; j < nedge; j++)
      if (fabsf(dt3(edge[j], face[i])) < CV_EDGE_TOL) { res[0] = j; res[1] = i; return 1; }
  return 0;
}
// polygons of a mesh vertex: c_mpmap[c_mpmadr[mesh vertadr + v] ..]
__device__ int intersect1(int a1, int n1, int a2, int n2, int* res) {
  int count = 0;
  for (int i = a1; i < a1 + n1; i++)
    for (int j = a2; j < a2 + n2; j++)
      if (c_mpmap[i] == c_mpmap[j]) { res[count++] = c_mpmap[i]; if (count == 2) return 2; }
  return count;
}
__device__ int intersect2(const int* a1, int n1, int a2, int n2, int* res) {
  int count = 0, out[2] = {0, 0};
  for (int i = 0; i < n1; i++)
    for (int j = a2; j < a2 + n2; j++)
      if (a1[i] == c_mpmap[j]) { out[count++] = a1[i]; if (count == 2) { res[0] = out[0]; res[1] = out[1]; return 2; } }
  res[0] = out[0]; res[1] = out[1];
  return count;
}
__device__ __forceinline__ f3 mpnorm(int mesh, int poly) { return ld3(c_mpnorm + 3 * (c_mpadr[mesh] + poly)); }
__device__ int mesh_normals(int dim, const int* fi, const CGeom& g, f3* nout, int* iout) {
  const int va = c_mvadr[g.mesh];
  if (dim == 3) {
    int e[2] = {0, 0}, f[2] = {0, 0};
    int n = intersect1(c_mpmadr[va + fi[0]], c_mpmnum[va + fi[0]], c_mpmadr[va + fi[1]], c_mpmnum[va + fi[1]], e);
    if (n == 0) return 0;
    n = intersect2(e, n, c_mpmadr[va + fi[2]], c_mpmnum[va + fi[2]], f);
    if (n == 0) return 0;
    nout[0] = rmul(g.R, mpnorm(g.mesh, f[0]));
    iout[0] = f[0];
    return 1;
  }
  if (dim == 2) {
    int e[2] = {0, 0};
    int n = intersect1(c_mpmadr[va + fi[0]], c_mpmnum[va + fi[0]], c_mpmadr[va + fi[1]], c_mpmnum[va + fi[1]], e);
    if (n == 0) return 0;
    for (int i = 0; i < n; i++) { nout[i] = rmul(g.R, mpnorm(g.mesh, e[i])); iout[i] = e[i]; }
    return n;
  }
  if (dim == 1) {
    int a = c_mpmadr[va + fi[0]], n = c_mpmnum[va + fi[0]];
    for (int i = 0; i < n; i++) { int p = c_mpmap[a + i]; nout[i] = rmul(g.R, mpnorm(g.mesh, p)); iout[i] = p; }
    return n;
  }
  return 0;
}
__device__ int mesh_edge_normals(int dim, const CGeom& g, f3 v1, f3 v2, int v1i, f3* nout, f3* eout) {
  if (dim == 2) { eout[0] = v2; nout[0] = nz3(v2 - v1); return 1; }
  if (dim == 1) {
    const int va = c_mvadr[g.mesh], pa = c_mpadr[g.mesh];
    int a = c_mpmadr[va + v1i], n = c_mpmnum[va + v1i];
    for (int i = 0; i < n; i++) {
      int p = c_mpmap[a + i], adr = c_mpvadr[pa + p], nv = c_mpvnum[pa + p];
      for (int j = 0; j < nv; j++)
        if (c_mpvert[adr + j] == v1i) {
          int k = j == 0 ? nv - 1 : j - 1;
          eout[i] = rmul(g.R, mvert(g.mesh, c_mpvert[adr + k])) + g.pos;
          nout[i] = nz3(eout[i] - v1);
        }
    }
    return n;
  }
  return 0;
}
__device__ __forceinline__ f3 box_fn(int i) {  // mat63 face normals: +x, -x, +y, -y, +z, -z
  return mk3(i == 0 ? 1.f : (i == 1 ? -1.f : 0.f), i == 2 ? 1.f : (i == 3 ? -1.f : 0.f), i == 4 ? 1.f : (i == 5 ? -1.f : 0.f));
}
__device__ int box_normals2(const float* R, f3 n, f3* nout, int* iout) {
  f3 ln = nz3(rtmul(R, n));
  for (int i = 0; i < 6; i++)
    if (dt3(ln, box_fn(i)) > CV_FACE_TOL) { nout[0] = rmul(R, box_fn(i)); iout[0] = i; return 1; }
  return 0;
}
__device__ int box_normals(int dim, const int* fi, const float* R, f3 dir, f3* nout, int* iout) {
  int v1 = fi[0], v2 = fi[1], v3 = fi[2];
  if (dim == 3) {
    int c = 0;
    float x = float((v1 & 1) && (v2 & 1) && (v3 & 1)) - float(!(v1 & 1) && !(v2 & 1) && !(v3 & 1));
    float y = float((v1 & 2) && (v2 & 2) && (v3 & 2)) - float(!(v1 & 2) && !(v2 & 2) && !(v3 & 2));
    float z = float((v1 & 4) && (v2 & 4) && (v3 & 4)) - float(!(v1 & 4) && !(v2 & 4) && !(v3 & 4));
    nout[0] = rmul(R, mk3(x, y, z));
    float sgn = x + y + z;
    if (x != 0.f) iout[c++] = 0;
    if (y != 0.f) iout[c++] = 2;
    if (z != 0.f) iout[c++] = 4;
    if (sgn == -1.f) iout[0] = iout[0] + 1;
    if (c == 1) return 1;
    return box_normals2(R, dir, nout, iout);
  }
  if (dim == 2) {
    int c = 0;
    float x = float((v1 & 1) && (v2 & 1)) - float(!(v1 & 1) && !(v2 & 1));
    float y = float((v1 & 2) && (v2 & 2)) - float(!(v1 & 2) && !(v2 & 2));
    float z = float((v1 & 4) && (v2 & 4)) - float(!(v1 & 4) && !(v2 & 4));
    if (x != 0.f) { nout[c] = rmul(R, mk3(x, 0.f, 0.f)); iout[c] = x > 0.f ? 0 : 1; c++; }
    if (y != 0.f) { nout[c] = rmul(R, mk3(0.f, y, 0.f)); iout[c] = y > 0.f ? 2 : 3; c++; }
    if (z != 0.f) { nout[c] = rmul(R, mk3(0.f, 0.f, z)); iout[c] = z > 0.f ? 4 : 5; c++; }
    if (c == 1 || c == 2) return c;
    return box_normals2(R, dir, nout, iout);
  }
  if (dim == 1) {
    float x = (v1 & 1) ? 1.f : -1.f, y = (v1 & 2) ? 1.f : -1.f, z = (v1 & 4) ? 1.f : -1.f;
    nout[0] = rmul(R, mk3(x, 0.f, 0.f)); nout[1] = rmul(R, mk3(0.f, y, 0.f)); nout[2] = rmul(R, mk3(0.f, 0.f, z));
    iout[0] = x > 0.f ? 0 : 1; iout[1] = y > 0.f ? 2 : 3; iout[2] = z > 0.f ? 4 : 5;
    return 3;
  }
  return 0;
}
__device__ int box_edge_normals(int dim, const CGeom& g, f3 v1, f3 v2, int v1i, f3* nout, f3* eout) {
  if (dim == 2) { eout[0] = v2; nout[0] = nz3(v2 - v1); return 1; }
  if (dim == 1) {
    float x = (v1i & 1) ? g.size.x : -g.size.x, y = (v1i & 2) ? g.size.y : -g.size.y, z = (v1i & 4) ? g.size.z : -g.size.z;
    eout[0] = rmul(g.R, mk3(-x, y, z)) + g.pos; nout[0] = nz3(eout[0] - v1);
    eout[1] = rmul(g.R, mk3(x, -y, z)) + g.pos; nout[1] = nz3(eout[1] - v1);
    eout[2] = rmul(g.R, mk3(x, y, -z)) + g.pos; nout[2] = nz3(eout[2] - v1);
    return 3;
  }
  return 0;
}
__device__ int box_face(const CGeom& g, int idx, f3* f) {
  const float sx = g.size.x, sy = g.size.y, sz = g.size.z;
  f3 c[4];
  if (idx == 0) { c[0] = mk3(sx, sy, sz); c[1] = mk3(sx, sy, -sz); c[2] = mk3(sx, -sy, -sz); c[3] = mk3(sx, -sy, sz); }
  else if (idx == 1) { c[0] = mk3(-sx, sy, -sz); c[1] = mk3(-sx, sy, sz); c[2] = mk3(-sx, -sy, sz); c[3] = mk3(-sx, -sy, -sz); }
  else if (idx == 2) { c[0] = mk3(-sx, sy, -sz); c[1] = mk3(sx, sy, -sz); c[2] = mk3(sx, sy, sz); c[3] = mk3(-sx, sy, sz); }
  else if (idx == 3) { c[0] = mk3(-sx, -sy, sz); c[1] = mk3(sx, -sy, sz); c[2] = mk3(sx, -sy, -sz); c[3] = mk3(-sx, -sy, -sz); }
  else if (idx == 4) { c[0] = mk3(-sx, sy, sz); c[1] = mk3(sx, sy, sz); c[2] = mk3(sx, -sy, sz); c[3] = mk3(-sx, -sy, sz); }
  else if (idx == 5) { c[0] = mk3(sx, sy, -sz); c[1] = mk3(-sx, sy, -sz); c[2] = mk3(-sx, -sy, -sz); c[3] = mk3(sx, -sy, -sz); }
  else return 0;
  for (int i = 0; i < 4; i++) f[i] = rmul(g.R, c[i]) + g.pos;
  return 4;
}
__device__ int mesh_face(const CGeom& g, int idx, f3* f) {
  const int pa = c_mpadr[g.mesh], adr = c_mpvadr[pa + idx], nv = c_mpvnum[pa + idx];
  int j = 0;
  for (int i = nv - 1; i >= 0; i--) f[j++] = rmul(g.R, mvert(g.mesh, c_mpvert[adr + i])) + g.pos;
  return nv;
}
__device__ __forceinline__ float area4(f3 a, f3 b, f3 c, f3 d) { return 0.5f * nm3(cr3(a - c, b - d)); }
__device__ void polygon_quad(const f3* p, int np, int* q) {  // MuJoCo Warp's _polygon_quad
  int b = 1, c = 2, d = 3;
  q[0] = 0; q[1] = b; q[2] = c; q[3] = d;
  float m = area4(p[0], p[b], p[c], p[d]);
  for (int a = 0; a < np; a++) {
    while (true) {
      float m_next = area4(p[a], p[b], p[c], p[(d + 1) % np]);
      if (m_next <= m) break;
      m = m_next; d = (d + 1) % np; q[0] = a; q[1] = b; q[2] = c; q[3] = d;
      while (true) {
        m_next = area4(p[a], p[b], p[(c + 1) % np], p[d]);
        if (m_next <= m) break;
        m = m_next; c = (c + 1) % np; q[0] = a; q[1] = b; q[2] = c; q[3] = d;
      }
      while (true) {
        m_next = area4(p[a], p[(b + 1) % np], p[c], p[d]);
        if (m_next <= m) break;
        m = m_next; b = (b + 1) % np; q[0] = a; q[1] = b; q[2] = c; q[3] = d;
      }
    }
    if (b == a) {
      b = (b + 1) % np;
      if (c == b) {
        c = (c + 1) % np;
        if (d == c) d = (d + 1) % np;
      }
    }
  }
}
__device__ __forceinline__ void witness_on_face(f3 v, f3 p, f3 n, f3 dir, f3* w1, f3* w2, float* d) {
  float dist = dt3(v - p, n);
  *w1 = v - fabsf(dist) * dir;
  *w2 = v;
  *d = dist;
}
// clip polygon face2 against face1 (MuJoCo Warp's _polygon_clip): the contacts (witnesses, distances), count
__device__ int polygon_clip(MultiBuf& mb, const f3* face1, int nface1, const f3* face2, int nface2, f3 n, f3 dir,
                            f3* w1o, f3* w2o, float* dists) {
  if (nface1 < 3) return 0;
  for (int i = 0; i < nface1; i++) {
    f3 a = face1[i], b = face1[(i + 1) % nface1];
    f3 v3 = a + n, res = cr3(b - a, v3 - a);
    mb.pdist[i] = dt3(res, a);
    mb.pnormal[i] = res;
  }
  f3* poly = mb.polygon;
  f3* clip = mb.clipped;
  int npoly = nface2, nclip = 0;
  for (int i = 0; i < nface2; i++) poly[i] = face2[i];
  for (int e = 0; e < nface1; e++) {
    for (int i = 0; i < npoly; i++) {
      f3 P = poly[i], Q = poly[(i + 1) % npoly];
      bool in1 = dt3(P - face1[e], mb.pnormal[e]) > -1e-10f, in2 = dt3(Q - face1[e], mb.pnormal[e]) > -1e-10f;
      if (!in1 && !in2) continue;
      if (in1 && in2) { if (nclip < 2 * NPOLYMAX) clip[nclip++] = Q; continue; }
      float dot = dt3(mb.pnormal[e], Q - P), t = fabsf(dot) < 1e-10f ? CV_FLOAT_MAX : (mb.pdist[e] - dt3(mb.pnormal[e], P)) / dot;
      if (t > -CV_INTERSECT_TOL && t < 1.f + CV_INTERSECT_TOL) {
        t = fminf(fmaxf(t, 0.f), 1.f);
        if (nclip < 2 * NPOLYMAX) clip[nclip++] = P + t * (Q - P);
      }
      if (in2 && nclip < 2 * NPOLYMAX) clip[nclip++] = Q;
    }
    f3* tmp = poly; poly = clip; clip = tmp;
    npoly = nclip;
    nclip = 0;
  }
  int m = npoly;
  npoly = 0;
  for (int i = 0; i < m; i++)
    if (dt3(poly[i] - face1[0], n) <= 0.f) { if (npoly != i) poly[npoly] = poly[i]; npoly++; }
  if (npoly < 1) return 0;
  if (nface2 == 2 && npoly > 2) {
    int b1 = 0, b2 = 1;
    float max_d = 0.f;
    for (int i = 0; i < npoly; i++)
      for (int j = i + 1; j < npoly; j++) {
        f3 df = poly[j] - poly[i];
        float d2 = dt3(df, df);
        if (d2 > max_d) { max_d = d2; b1 = i; b2 = j; }
      }
    witness_on_face(poly[b1], face1[0], n, dir, &w1o[0], &w2o[0], &dists[0]);
    witness_on_face(poly[b2], face1[0], n, dir, &w1o[1], &w2o[1], &dists[1]);
    return 2;
  }
  if (npoly > 4) {
    int q[4];
    polygon_quad(poly, npoly, q);
    for (int i = 0; i < 4; i++) witness_on_face(poly[q[i]], face1[0], n, dir, &w1o[i], &w2o[i], &dists[i]);
    return 4;
  }
  for (int i = 0; i < npoly; i++) witness_on_face(poly[i], face1[0], n, dir, &w1o[i], &w2o[i], &dists[i]);
  return npoly;
}

// MuJoCo Warp's multicontact: up to 4 contacts (witnesses w1, w2, distances) from EPA face fidx; returns the count
// (1: the EPA witnesses x1, x2 alone)
__device__ int multicontact(MultiBuf& mb, const Polytope& pt, int fidx, f3 x1, f3 x2, const CGeom& g1,
                            const CGeom& g2, f3* w1, f3* w2, float* dists) {
  w1[0] = x1; w2[0] = x2;
  int fv[3], fi1[3], fi2[3];
  f3 fx1[3], fx2[3];
  face_verts(pt.face[fidx], fv);
  int nface1 = feature_dim(fv, pt, 0, fi1, fx1), nface2 = feature_dim(fv, pt, 1, fi2, fx2);
  f3 dir = x2 - x1, dir_neg = -dir;
  int nn1 = 0, nn2 = 0;
  if (g1.type == 3) nn1 = box_normals(nface1, fi1, g1.R, dir_neg, mb.n1, mb.idx1);
  else nn1 = mesh_normals(nface1, fi1, g1, mb.n1, mb.idx1);
  if (g2.type == 3) nn2 = box_normals(nface2, fi2, g2.R, dir, mb.n2, mb.idx2);
  else nn2 = mesh_normals(nface2, fi2, g2, mb.n2, mb.idx2);
  int edge1 = 0, edge2 = 0, res[2] = {0, 0};
  int nres = aligned_faces(mb.n1, nn1, mb.n2, nn2, res);
  if (!nres) {
    if (nface1 < 3 && nface1 <= nface2) {
      nn1 = g1.type == 3 ? box_edge_normals(nface1, g1, fx1[0], fx1[1], fi1[0], mb.n1, mb.endvert)
                         : mesh_edge_normals(nface1, g1, fx1[0], fx1[1], fi1[0], mb.n1, mb.endvert);
      nres = aligned_face_edge(mb.n1, nn1, mb.n2, nn2, res);
      if (!nres) return 1;
      edge1 = 1;
    } else if (nface2 < 3) {
      nn2 = g2.type == 3 ? box_edge_normals(nface2, g2, fx2[0], fx2[1], fi2[0], mb.n2, mb.endvert)
                         : mesh_edge_normals(nface2, g2, fx2[0], fx2[1], fi2[0], mb.n2, mb.endvert);
      nres = aligned_face_edge(mb.n2, nn2, mb.n1, nn1, res);
      if (!nres) return 1;
      edge2 = 1;
    } else {
      return 1;
    }
  }
  int i = res[0], j = res[1], nf1, nf2;
  if (edge1) { mb.face1[0] = pt.vert[2 * fv[0]]; mb.face1[1] = mb.endvert[i]; nf1 = 2; }
  else {
    int ind = edge2 ? mb.idx1[j] : mb.idx1[i];
    nf1 = g1.type == 3 ? box_face(g1, ind, mb.face1) : mesh_face(g1, ind, mb.face1);
  }
  if (edge2) { mb.face2[0] = pt.vert[2 * fv[0] + 1]; mb.face2[1] = mb.endvert[i]; nf2 = 2; }
  else nf2 = g2.type == 3 ? box_face(g2, mb.idx2[j], mb.face2) : mesh_face(g2, mb.idx2[j], mb.face2);
  if (edge1) {
    f3 c1[4], c2[4];
    int n = polygon_clip(mb, mb.face2, nf2, mb.face1, nf1, mb.n2[j], -mb.n2[j], c1, c2, dists);
    for (int k = 0; k < n; k++) { w1[k] = c2[k]; w2[k] = c1[k]; }  // (flipped back)
    return n;
  }
  if (edge2) return polygon_clip(mb, mb.face1, nf1, mb.face2, nf2, mb.n1[j], -mb.n1[j], w1, w2, dists);
  return polygon_clip(mb, mb.face1, nf1, mb.face2, nf2, mb.n1[i], mb.n2[j], w1, w2, dists);
}

// ------------------------------------------------------------------------------- the pair (collision_convex.ccd)
__device__ __forceinline__ CGeom cgeom(int g, const float* p, const float* q, int t) {
  CGeom c;
  c.pos = ld3(p);
  quat2mat(q, c.R);
  c.size = ld3(c_gsize + 3 * g);
  c.type = t;
  c.mesh = c_gmesh[g];
  c.index = -1;
  return c;
}

// a convex pair (t1 <= t2, t2 = mesh; t1 sphere, capsule, box or mesh) that passed the OBB test (the caller's:
// before the split forward queues it): GJK, EPA, multicontact; the contacts with dist < 0. One non-template function:
// the fused and the split forward run the same code
__device__ __noinline__ void convex_pair(int g1i, int t1, const float* p1, const float* q1, int g2i, int t2,
                                         const float* p2, const float* q2, CSink& o) {
  o.n = 0; o.unsupported = 0;
  CGeom g1 = cgeom(g1i, p1, q1, t1), g2 = cgeom(g2i, p2, q2, t2);
  const float tol = CCD_TOLERANCE;
  const int iters = CCD_ITERATIONS;
  f3 origin = g1.pos;  // (translation invariance: geom 1 at the origin)
  g1.pos = mk3(0.f, 0.f, 0.f);
  g2.pos = g2.pos - origin;
  // gjk_phase: a sphere / capsule shrunk to its centre / segment first
  const bool disc = is_discrete(t1, t2);
  float size1 = 0.f, size2 = 0.f, fm1 = 0.f, fm2 = 0.f;
  CGeom s1 = g1, s2 = g2;
  if (t1 == 1 || t1 == 2) { size1 = g1.size.x; fm1 = size1; s1.size.x = 0.f; }
  GJKResult r;
  float dist;
  f3 x1, x2;
  bool needs_epa = false;
  if (size1 + size2 > 0.f) {
    float cutoff = fm1 + fm2;
    cv_gjk(tol, iters, s1, s2, s1.pos, s2.pos, cutoff, disc, r);
    g1.index = r.index1; g2.index = r.index2;
    if (r.dist > tol) {
      if (r.dist == CV_FLOAT_MAX) return;
      f3 n = nz3(r.x2 - r.x1);  // _inflate
      x1 = r.x1; x2 = r.x2;
      if (fm1 > 0.f) x1 = x1 + fm1 * n;
      if (fm2 > 0.f) x2 = x2 - fm2 * n;
      dist = r.dist - (fm1 + fm2);
      goto write1;
    }
  }
  cv_gjk(tol, iters, g1, g2, g1.pos, g2.pos, 0.f, disc, r);
  g1.index = r.index1; g2.index = r.index2;
  if (r.dist > tol || r.dim < 2 || r.separated) { dist = r.dist; x1 = r.x1; x2 = r.x2; goto write1; }
  needs_epa = true;
  {
    Polytope pt;
    pt.nface = 0; pt.nvert = 0; pt.nhorizon = 0; pt.status = 0;
    if (r.dim == 2) { polytope2(pt, r, g1, g2); if (pt.status == -1) r.dim = 3; }
    else if (r.dim == 4) { polytope4(pt, r); if (pt.status == -1) r.dim = 3; }
    if (r.dim == 3) { pt.nface = 0; pt.nvert = 0; pt.nhorizon = 0; polytope3(pt, r, g1, g2); }
    if (pt.status) { dist = r.dist; x1 = r.x1; x2 = r.x2; goto write1; }
    int fidx = cv_epa(tol, iters, pt, g1, g2, disc);
    if (fidx == -1) return;
    dist = epa_witness(pt, fidx, &x1, &x2);
    if (!(dist < 0.f)) return;
    f3 w1[4], w2[4];
    float dists[4] = {0.f, 0.f, 0.f, 0.f};
    int nc = 1;
    w1[0] = x1; w2[0] = x2;
#if MULTICCD
    if ((t1 == 3 || t1 == 4) && (t2 == 3 || t2 == 4)) {
      MultiBuf mb;
      nc = multicontact(mb, pt, fidx, x1, x2, g1, g2, w1, w2, dists);
      if (nc < 1) { nc = 1; w1[0] = x1; w2[0] = x2; }
    }
#endif
    float nrm[3], fr[9];
    f3 nv = nz3(w1[0] - w2[0]);
    nrm[0] = nv.x; nrm[1] = nv.y; nrm[2] = nv.z;
    make_frame(nrm, fr);
    for (int i = 0; i < nc; i++) {
      float di = nc > 1 ? dists[i] : dist;
      f3 pc = 0.5f * (w1[i] + w2[i]) + origin;
      float pos[3] = {pc.x, pc.y, pc.z};
      if (di < 0.f) add_contact(o, 0, i, di, pos, fr);
    }
    return;
  }
write1:
  (void)needs_epa;
  if (!(dist < 0.f)) return;
  {
    float nrm[3], fr[9];
    f3 nv = nz3(x1 - x2);
    nrm[0] = nv.x; nrm[1] = nv.y; nrm[2] = nv.z;
    make_frame(nrm, fr);
    f3 pc = 0.5f * (x1 + x2) + origin;
    float pos[3] = {pc.x, pc.y, pc.z};
    add_contact(o, 0, 0, dist, pos, fr);
  }
}

// plane-convex (MuJoCo Warp's plane_convex): up to 4 vertices of the mesh below the plane; n: the plane's normal
__device__ __noinline__ void plane_mesh(const float* pp, const float* n, int g2i, const float* p2, const float* q2,
                                        CSink& o) {
  o.n = 0; o.unsupported = 0;
  CGeom cv = cgeom(g2i, p2, q2, 4);
  const float HUGE = 1e6f;
  f3 pl = rtmul(cv.R, ld3(pp) - cv.pos), nl = rtmul(cv.R, ld3(n));
  int ind[4] = {-1, -1, -1, -1};
  const int vn = c_mvnum[cv.mesh], ga = c_mgadr[cv.mesh];
  if (ga == -1 || vn < 10) {
    float ms = -HUGE;
    f3 a = mk3(0.f, 0.f, 0.f), b = a, c = a;
    for (int i = 0; i < vn; i++) { f3 v = mvert(cv.mesh, i); float sp = dt3(pl - v, nl); if (sp > ms) { ms = sp; ind[0] = i; a = v; } }
    if (ms < 0.f) return;
    float th = ms - 1e-3f, bd = -HUGE;
    for (int i = 0; i < vn; i++) {
      f3 v = mvert(cv.mesh, i);
      float sp = dt3(pl - v, nl), d = dt3(a - v, a - v) + (sp > th ? 0.f : -HUGE);
      if (d > bd) { ind[1] = i; bd = d; b = v; }
    }
    f3 ab = cr3(nl, a - b);
    float cd = -HUGE;
    for (int i = 0; i < vn; i++) {
      f3 v = mvert(cv.mesh, i);
      float sp = dt3(pl - v, nl), d = fabsf(dt3(a - v, ab)) + (sp > th ? 0.f : -HUGE);
      if (d > cd) { ind[2] = i; cd = d; c = v; }
    }
    f3 ac = cr3(nl, a - c), bc = cr3(nl, b - c);
    float dd = -HUGE;
    for (int i = 0; i < vn; i++) {
      f3 v = mvert(cv.mesh, i);
      float sp = dt3(pl - v, nl), mask = sp > th ? 0.f : -HUGE;
      float d = fabsf(dt3(a - v, ac)) + mask + fabsf(dt3(b - v, bc)) + mask;
      if (d > dd) { ind[3] = i; dd = d; }
    }
  } else {
    const int* gr = c_mgraph + ga;
    const int numvert = gr[0], vea = 2, vgi = 2 + numvert, eli = 2 + 2 * numvert;
    float ms = -HUGE;
    int prev = -1, imax = 0;
    while (true) {
      prev = imax;
      for (int i = gr[vea + imax]; gr[eli + i] >= 0; i++) {
        int sub = gr[eli + i];
        float sp = dt3(pl - mvert(cv.mesh, gr[vgi + sub]), nl);
        if (sp > ms) { ms = sp; imax = sub; }
      }
      if (imax == prev) break;
    }
    float th = fmaxf(0.f, ms - 1e-3f), ad = -HUGE;
    while (true) {
      prev = imax;
      for (int i = gr[vea + imax]; gr[eli + i] >= 0; i++) {
        int sub = gr[eli + i];
        float sp = dt3(pl - mvert(cv.mesh, gr[vgi + sub]), nl), d = sp > th ? sp : -HUGE;
        if (d > ad) { ad = d; imax = sub; }
      }
      if (imax == prev) break;
    }
    ind[0] = gr[vgi + imax];
    f3 a = mvert(cv.mesh, ind[0]);
    float bd = -HUGE;
    while (true) {
      prev = imax;
      for (int i = gr[vea + imax]; gr[eli + i] >= 0; i++) {
        int sub = gr[eli + i];
        f3 v = mvert(cv.mesh, gr[vgi + sub]);
        float sp = dt3(pl - v, nl), d = dt3(a - v, a - v) + (sp > th ? 0.f : -HUGE);
        if (d > bd) { bd = d; imax = sub; }
      }
      if (imax == prev) break;
    }
    ind[1] = gr[vgi + imax];
    f3 b = mvert(cv.mesh, ind[1]), ab = cr3(nl, a - b);
    float cd = -HUGE;
    while (true) {
      prev = imax;
      for (int i = gr[vea + imax]; gr[eli + i] >= 0; i++) {
        int sub = gr[eli + i];
        f3 v = mvert(cv.mesh, gr[vgi + sub]);
        float sp = dt3(pl - v, nl), d = fabsf(dt3(a - v, ab)) + (sp > th ? 0.f : -HUGE);
        if (d > cd) { cd = d; imax = sub; }
      }
      if (imax == prev) break;
    }
    ind[2] = gr[vgi + imax];
    f3 c = mvert(cv.mesh, ind[2]), ac = cr3(nl, a - c), bc = cr3(nl, b - c);
    float dd = -HUGE;
    while (true) {
      prev = imax;
      for (int i = gr[vea + imax]; gr[eli + i] >= 0; i++) {
        int sub = gr[eli + i];
        f3 v = mvert(cv.mesh, gr[vgi + sub]);
        float sp = dt3(pl - v, nl), mask = sp > th ? 0.f : -HUGE;
        float d = fabsf(dt3(a - v, ac)) + mask + fabsf(dt3(b - v, bc)) + mask;
        if (d > dd) { dd = d; imax = sub; }
      }
      if (imax == prev) break;
    }
    ind[3] = gr[vgi + imax];
  }
  float fr[9];
  make_frame(n, fr);
  int cnt = 0;
  for (int i = 3; i >= 0; i--) {
    int idx = ind[i], count = 0;
    for (int j = 0; j <= i; j++) if (ind[j] == idx) count++;
    if (count == 1) {
      f3 v = mvert(cv.mesh, idx), pw = cv.pos + rmul(cv.R, v);
      float dist = -dt3(pl - v, nl);
      f3 pc = pw - 0.5f * dist * ld3(n);
      float pos[3] = {pc.x, pc.y, pc.z};
      if (dist < 0.f) add_contact(o, 0, cnt, dist, pos, fr);
      cnt++;
    }
  }
}

#endif
"""


def instantiate(ns="cvd", real="D1"):
    """This module's vector helpers and convex collision code (GJK, EPA, multicontact, plane_mesh) over a scalar type,
    generated from their text into namespace ns (real: the adjoint's dual D1, or float / double for checks), cgeom takes the rotation matrix (with its
    tangent) in place of the quaternion, the contacts go to a DSink (dual distance, position, frame) through make_frame
    without the forward's explicit roundings (same values up to rounding, exact tangents). The adjoint replays a mesh
    pair's contact slot with it once per pose direction (ADJ_MESH)."""
    import re
    common = COMMON[:COMMON.index("// the geoms' oriented bounding boxes")]
    body = common + DEVICE
    for a, b in (("  quat2mat(q, c.R);\n", "  for (int k = 0; k < 9; k++) c.R[k] = q[k];  // (cvd: the rotation matrix)\n"),
                 ("CSink& o", "DSink& o")):
        if a not in body:
            raise RuntimeError(f"convex_dual_source: no {a!r}")
        body = body.replace(a, b)
    if real == "D1":
        for a, b in CLEAN_NORMAL:
            if body.count(a) != 1:
                raise RuntimeError(f"instantiate: {body.count(a)} x {a[:60]!r}")
            body = body.replace(a, b)
    body = re.sub(r"\bfloat\b", "real", body)
    body = re.sub(r"\b(fabs|sqrt|fmin|fmax)f\(", r"t_\1(", body)
    ld = "__device__ __forceinline__ f3 ld3(const real* p) { return mk3(p[0], p[1], p[2]); }"
    for a, b in (("const real* v = c_mvert", "const float* v = c_mvert"),
                 (ld, ld + ("" if real == "float" else
                            "\n__device__ __forceinline__ f3 ld3(const float* p) { return mk3(p[0], p[1], p[2]); }"))):
        if a not in body:
            raise RuntimeError(f"convex_dual_source: no {a!r}")
        body = body.replace(a, b)
    head = r"""
namespace @NS@ {  // forest_engine.convex over @REAL@ (model_hooks.convex_dual_source)
using real = @REAL@;
struct DSink {
  int n, unsupported;
  int slot[SINK_MAX];
  real dist[SINK_MAX], pos[SINK_MAX][3], fr[SINK_MAX][9];
};
__device__ __forceinline__ bool add_contact(DSink& o, int, int slot, real dist, const real* pos, const real* frame) {
  if (o.n >= SINK_MAX) { o.unsupported |= 4; return false; }
  o.slot[o.n] = slot;
  o.dist[o.n] = dist;
  for (int k = 0; k < 3; k++) o.pos[o.n][k] = pos[k];
  for (int k = 0; k < 9; k++) o.fr[o.n][k] = frame[k];
  o.n++;
  return true;
}
__device__ __forceinline__ void flag_unsupported(DSink& o, int bits) { o.unsupported |= bits; }
__device__ __forceinline__ void make_frame(const real* n, real* fr) {  // (as step.py's, without explicit roundings)
  real b[3] = {0.f, 0.f, 0.f};
  if (-0.5f < n[1] && n[1] < 0.5f) b[1] = 1.f; else b[2] = 1.f;
  real d = n[0] * b[0] + n[1] * b[1] + n[2] * b[2];
  for (int k = 0; k < 3; k++) b[k] = b[k] - n[k] * d;
  real bn = t_sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
  for (int k = 0; k < 3; k++) b[k] = b[k] / bn;
  real c[3] = {n[1] * b[2] - n[2] * b[1], n[2] * b[0] - n[0] * b[2], n[0] * b[1] - n[1] * b[0]};
  for (int k = 0; k < 3; k++) { fr[k] = n[k]; fr[3 + k] = b[k]; fr[6 + k] = c[k]; }
}
"""
    return ("#if HAS_MESH\n" + head + body + "\n}  // namespace @NS@\n#endif\n").replace("@NS@", ns).replace(
        "@REAL@", real)


CLEAN_NORMAL = (
    ("  f3 n1[NMESHDEG], n2[NMESHDEG], endvert[NMESHDEG], face1[NPOLYMAX], face2[NPOLYMAX];\n};",
     "  f3 n1[NMESHDEG], n2[NMESHDEG], endvert[NMESHDEG], face1[NPOLYMAX], face2[NPOLYMAX];\n"
     "  f3 cdir;  // (CLEAN_NORMAL) polygon_clip's dir\n  int cset;\n};"),
    ("                            f3* w1o, f3* w2o, float* dists) {\n  if (nface1 < 3) return 0;\n",
     "                            f3* w1o, f3* w2o, float* dists) {\n  if (nface1 < 3) return 0;\n"
     "  mb.cdir = dir; mb.cset = 1;  // (CLEAN_NORMAL)\n"),
    ("const CGeom& g2, f3* w1, f3* w2, float* dists) {\n  w1[0] = x1; w2[0] = x2;\n",
     "const CGeom& g2, f3* w1, f3* w2, float* dists) {\n  w1[0] = x1; w2[0] = x2;\n  mb.cset = 0;  // (CLEAN_NORMAL)\n"),
    ("#if MULTICCD\n    if ((t1 == 3 || t1 == 4) && (t2 == 3 || t2 == 4)) {\n      MultiBuf mb;\n"
     "      nc = multicontact(mb, pt, fidx, x1, x2, g1, g2, w1, w2, dists);\n",
     "    f3 cdir = mk3(0.f, 0.f, 0.f);  // (CLEAN_NORMAL)\n    int cset = 0;\n"
     "#if MULTICCD\n    if ((t1 == 3 || t1 == 4) && (t2 == 3 || t2 == 4)) {\n      MultiBuf mb;\n"
     "      nc = multicontact(mb, pt, fidx, x1, x2, g1, g2, w1, w2, dists);\n"
     "      if (nc >= 1 && mb.cset) { cdir = mb.cdir; cset = 1; }  // (CLEAN_NORMAL)\n"),
    ("    f3 nv = nz3(w1[0] - w2[0]);\n",
     "    f3 nv = nz3(w1[0] - w2[0]);\n"
     "    if (cset) {  // (CLEAN_NORMAL) the value as computed, the tangent the face normal's\n"
     "      const int sg = (nv.x * cdir.x + nv.y * cdir.y + nv.z * cdir.z).v < 0.f ? -1 : 1;\n"
     "      nv.x.d = sg * cdir.x.d; nv.y.d = sg * cdir.y.d; nv.z.d = sg * cdir.z.d;\n"
     "    } else {  // (CLEAN_NORMAL) the EPA witnesses: x1 - x2 is the EPA face's point nearest the origin, so the\n"
     "      // normal is that face's (its vertices, Minkowski differences, centimetres apart: well conditioned)\n"
     "      int fq[3];\n      face_verts(pt.face[fidx], fq);\n"
     "      f3 m0 = pt.vert[2 * fq[0]] - pt.vert[2 * fq[0] + 1], m1 = pt.vert[2 * fq[1]] - pt.vert[2 * fq[1] + 1],\n"
     "         m2 = pt.vert[2 * fq[2]] - pt.vert[2 * fq[2] + 1];\n      f3 fn = nz3(cr3(m1 - m0, m2 - m0));\n"
     "      const int sg = (nv.x * fn.x + nv.y * fn.y + nv.z * fn.z).v < 0.f ? -1 : 1;\n"
     "      nv.x.d = sg * fn.x.d; nv.y.d = sg * fn.y.d; nv.z.d = sg * fn.z.d;\n    }\n"),
)


def rounded_source():
    """The forward's mesh collision entries CV_PAIR / CV_PLANE: by default (CV_ROUNDED 1) the collision code generated
    over RF (nablaengine.dual: D1's values alone, every operation explicitly
    rounded), whose values are bitwise those of the adjoint's dual replay (instantiate(real="D1"), ADJ_MESH): a
    contact at a depth of ~0 is then found by the replay too; -DCV_ROUNDED=0 ($DK_CV_ROUNDED=0): convex_pair /
    plane_mesh in float (FMA-contracted: their near-zero contacts are missed by the replay in ~6 % of the place_empty_cup
    worlds; closer to MuJoCo Warp's rounding)."""
    from nablaengine.dual import DUAL_DEVICE, RF_DEVICE
    wrap = r"""
#if CV_ROUNDED && HAS_MESH
// the forward's mesh collision over RF (convex.rounded_source): the poses' rotations as quat2mat's, the contacts' values
__device__ __forceinline__ void cvr_out(const cvr::DSink& d, CSink& o) {
  o.n = d.n; o.unsupported = d.unsupported;
  for (int i = 0; i < d.n; i++) {
    o.slot[i] = d.slot[i]; o.dist[i] = d.dist[i].v;
    for (int k = 0; k < 3; k++) o.pos[i][k] = d.pos[i][k].v;
    for (int k = 0; k < 9; k++) o.fr[i][k] = d.fr[i][k].v;
  }
}
__device__ __noinline__ void convex_pair_r(int g1i, int t1, const float* p1, const float* q1, int g2i, int t2,
                                           const float* p2, const float* q2, CSink& o) {
  float R1[9], R2[9];
  quat2mat(q1, R1);
  quat2mat(q2, R2);
  cvr::real P1[3], Q1[9], P2[3], Q2[9];
  for (int k = 0; k < 3; k++) { P1[k] = RF(p1[k]); P2[k] = RF(p2[k]); }
  for (int k = 0; k < 9; k++) { Q1[k] = RF(R1[k]); Q2[k] = RF(R2[k]); }
  cvr::DSink d;
  cvr::convex_pair(g1i, t1, P1, Q1, g2i, t2, P2, Q2, d);
  cvr_out(d, o);
}
__device__ __noinline__ void plane_mesh_r(const float* pp, const float* n, int g2i, const float* p2, const float* q2,
                                          CSink& o) {
  float R2[9];
  quat2mat(q2, R2);
  cvr::real P1[3], N[3], P2[3], Q2[9];
  for (int k = 0; k < 3; k++) { P1[k] = RF(pp[k]); N[k] = RF(n[k]); P2[k] = RF(p2[k]); }
  for (int k = 0; k < 9; k++) Q2[k] = RF(R2[k]);
  cvr::DSink d;
  cvr::plane_mesh(P1, N, g2i, P2, Q2, d);
  cvr_out(d, o);
}
#define CV_PAIR convex_pair_r
#define CV_PLANE plane_mesh_r
#else
#define CV_PAIR convex_pair
#define CV_PLANE plane_mesh
#endif
"""
    return ("#ifndef CV_ROUNDED\n#define CV_ROUNDED 1  // (convex.rounded_source; $DK_CV_ROUNDED=0: the float forward)\n#endif\n"
            "#if CV_ROUNDED && HAS_MESH\n"
            + DUAL_DEVICE + "#ifndef DK_RF_DEVICE\n#define DK_RF_DEVICE\n" + RF_DEVICE + "#endif\n"
            + instantiate("cvr", "RF") + "#endif\n" + wrap)


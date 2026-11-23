"""Convex contacts with cylinders for the G1 engine (model_hooks): capsule-cylinder, cylinder-cylinder and cylinder-box
pairs as MuJoCo Warp 3.17's native CCD (mujoco_warp/_src/collision_convex.py ccd_kernel, collision_gjk.py gjk_phase,
epa_phase, multicontact with the cylinder's caps as 16-gons and the capsule's segment as an edge; Copyright 2025 The
Newton Developers, Apache License 2.0), margin and gap 0, multiccd on (MuJoCo's default), in their order of operations."""
import math
import re

import mujoco
import numpy as np

TYPES = ((2, 5), (5, 5), (5, 3), (1, 6), (2, 6), (5, 6), (3, 6), (6, 6))
COND = "((t1 == 2 && t2 == 5) || (t1 == 5 && (t2 == 5 || t2 == 3)) || (t2 == 6 && t1 != 0))"


def _lit(x):
    t = f"{np.float32(x):.9g}"
    return (t if ("." in t or "e" in t) else t + ".0") + "f"


def _cs16():
    return (", ".join(_lit(math.cos(i * math.pi / 8.0)) for i in range(16)),
            ", ".join(_lit(math.sin(i * math.pi / 8.0)) for i in range(16)))


def header(m):
    """#defines of the convex CCD (the model's ccd_iterations and ccd_tolerance: MuJoCo Warp's EPA uses ccd_iterations
    when a convex pair other than box-box exists, which these are); with mesh collision geoms the multicontact's
    buffers (MuJoCo Warp's npolygonmax / nmeshdegmax: the largest polygon and vertex degree of the meshes)."""
    from nablaengine.g1_engine import mesh
    mt = mesh.tables(m)
    sizes = (f"#define CY_NPOLY {max(16, mt['poly_max'])}\n#define CY_NDEG {max(3, mt['deg_max'])}\n#define CY_HAS_MESH 1\n"
             if mt is not None else "")
    return (f"#define CY_GJK_ITERATIONS {int(m.opt.ccd_iterations)}\n"
            f"#define CY_EPA_ITERATIONS {int(m.opt.ccd_iterations)}\n"
            f"#define CY_CCD_TOLERANCE {float(m.opt.ccd_tolerance):.9g}f\n" + sizes)


def pairs(m):
    """Whether model m has a capsule-cylinder, cylinder-cylinder, cylinder-box pair or a mesh with a sphere, capsule,
    cylinder, box or mesh."""
    G = mujoco.mjtGeom
    mesh = int(G.mjGEOM_MESH)
    want = {tuple(sorted(x)) for x in ((int(G.mjGEOM_CAPSULE), int(G.mjGEOM_CYLINDER)),
                                       (int(G.mjGEOM_CYLINDER), int(G.mjGEOM_CYLINDER)),
                                       (int(G.mjGEOM_CYLINDER), int(G.mjGEOM_BOX)))}
    want |= {tuple(sorted((int(t), mesh))) for t in (G.mjGEOM_SPHERE, G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER, G.mjGEOM_BOX,
                                                     G.mjGEOM_MESH)}
    return any(tuple(sorted((int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]])))) in want
               for p in range(m.npair))


def core_text():
    """ccd.DEVICE's helpers, support, GJK and EPA (to epa_witness), the cylinder's support added, its macros renamed
    (CY_*: this module's iteration counts)."""
    from nablaengine import ccd
    src = ccd.DEVICE
    end = src.index("// ------------------------------------------------------------------------------------- multicontact")
    body = src[:end]
    old = "  return sp;  // (no meshes in this module)\n}"
    new = """  if (g.type == 5) {  // cylinder (g1_engine.cylccd: MuJoCo Warp 3.17's support)
    f3 res = mk3(0.f, 0.f, 0.f);
    float d = sqrtf(ld.x * ld.x + ld.y * ld.y);
    if (d > CV_MINVAL) {
      float scl = g.size.x / d;
      res.x = ld.x * scl;
      res.y = ld.y * scl;
    }
    res.z = ld.z >= 0.f ? g.size.y : -g.size.y;
    sp.vidx = ld.z >= 0.f ? 0 : 1;
    sp.point = rmul(g.R, res) + g.pos;
    return sp;
  }
  return sp;
}"""
    if body.count(old) != 1:
        raise RuntimeError("cylccd: no cv_support tail in ccd.DEVICE")
    body = body.replace(old, new)
    old = "  if (g.type == 5) {  // cylinder (g1_engine.cylccd: MuJoCo Warp 3.17's support)"
    body = body.replace(old, MESH_SUPPORT + old)
    cg = "  int index;  // the support's cached vertex (MuJoCo Warp's geom.index)\n};"
    if body.count(cg) != 1:
        raise RuntimeError("cylccd: no CGeom in ccd.DEVICE")
    body = body.replace(cg, "  int index;  // the support's cached vertex (MuJoCo Warp's geom.index)\n"
                            "  int mesh;   // type 6: the packed tables' mesh (g1_engine.cylccd)\n};")
    disc = "{ return (t1 == 3 || t1 == 4) && (t2 == 3 || t2 == 4); }"
    if body.count(disc) != 1:
        raise RuntimeError("cylccd: no is_discrete in ccd.DEVICE")
    body = body.replace(disc, "{ return (t1 == 3 || t1 == 6) && (t2 == 3 || t2 == 6); }  // (box, the engine's mesh)")
    for old, new in (("  int index1, index2;\n};",
                      "  int index1, index2;\n  int ln, lv1[4], lv2[4];  // (g1_engine.cylccd: the last subdistance "
                      "call's vertices)\n};"),
                     ("si1[n] = sp1.vidx;",
                      "si1[n] = g1.type == 2 ? (rtmul(g1.R, -dir_neg).z >= 0.f ? -3 : -2) : sp1.vidx;"),
                     ("    subdistance(n + 1, simplex, l);\n",
                      "    r.ln = n + 1;  // (g1_engine.cylccd)\n"
                      "    for (int i = 0; i <= n; i++) { r.lv1[i] = si1[i]; r.lv2[i] = si2[i]; }\n"
                      "    subdistance(n + 1, simplex, l);\n"),
                     ("r.sidx1[i] = si1[i]; r.sidx2[i] = si2[i];",
                      "r.sidx1[i] = si1[i] < -1 ? -1 : si1[i]; r.sidx2[i] = si2[i];")):
        if body.count(old) != 1:
            raise RuntimeError(f"cylccd: {old!r} not once in ccd.DEVICE's GJK")
        body = body.replace(old, new)
    body = re.sub(r"\bEPA_(ITERATIONS|NV|NF|NH)\b", r"CY_EPA_\1", body)
    return body


MESH_SUPPORT = r"""#if CY_HAS_MESH
  if (g.type == 6) {  // mesh (g1_engine.cylccd: MuJoCo Warp 3.17's support; g.mesh: the packed tables' mesh)
    const int va = c_mvadr[g.mesh], vn = c_mvnum[g.mesh], ga = c_mgadr[g.mesh];
    float max_dist = CV_FLOAT_MIN;
    f3 pt = mk3(0.f, 0.f, 0.f);
    if (ga == -1 || vn < 10) {
      if (g.index > -1) {  // (the cached index is global here, as MuJoCo Warp's)
        sp.cached = g.index;
        pt = mk3(c_mvert[3 * g.index], c_mvert[3 * g.index + 1], c_mvert[3 * g.index + 2]);
        max_dist = dt3(pt, ld);
      }
      for (int i = 0; i < vn; i++) {
        f3 v = mk3(c_mvert[3 * (va + i)], c_mvert[3 * (va + i) + 1], c_mvert[3 * (va + i) + 2]);
        float d = dt3(v, ld);
        if (d > max_dist) { max_dist = d; pt = v; sp.cached = va + i; }
      }
      sp.vidx = sp.cached - va;
    } else {
      const int* gr = c_mgraph + ga;
      const int numvert = gr[0], vert_edgeadr = 2, vert_globalid = 2 + numvert, edge_localid = 2 + 2 * numvert;
      int prev = -1, imax = g.index > -1 ? g.index : 0;
      int q = va + gr[vert_globalid + imax];
      max_dist = dt3(ld, mk3(c_mvert[3 * q], c_mvert[3 * q + 1], c_mvert[3 * q + 2]));
      while (imax != prev) {  // hill-climb until no change
        prev = imax;
        int i = gr[vert_edgeadr + imax];
        int subidx = gr[edge_localid + i];
        while (subidx >= 0) {
          q = va + gr[vert_globalid + subidx];
          float d = dt3(ld, mk3(c_mvert[3 * q], c_mvert[3 * q + 1], c_mvert[3 * q + 2]));
          if (d > max_dist) { imax = subidx; max_dist = d; }
          i++;
          subidx = gr[edge_localid + i];
        }
      }
      sp.cached = imax;
      sp.vidx = gr[vert_globalid + imax];
      q = va + sp.vidx;
      pt = mk3(c_mvert[3 * q], c_mvert[3 * q + 1], c_mvert[3 * q + 2]);
    }
    sp.point = rmul(g.R, pt) + g.pos;
    return sp;
  }
#endif
"""


MULTI = r"""
// ------------------------------------------------- multicontact (MuJoCo Warp 3.17, collision_gjk.py; margins 0)
#ifndef CY_NPOLY
#define CY_NPOLY 16  // (the cylinder's 16-gon; meshes: their largest polygon, header)
#endif
#ifndef CY_NDEG
#define CY_NDEG 3  // (a box vertex's faces; meshes: their largest vertex degree, header)
#endif
struct CyBuf {
  f3 polygon[2 * CY_NPOLY], clipped[2 * CY_NPOLY], pnormal[CY_NPOLY];
  float pdist[CY_NPOLY];
  int idx1[CY_NDEG], idx2[CY_NDEG];
  f3 n1[CY_NDEG], n2[CY_NDEG], endvert[CY_NDEG], face1[CY_NPOLY], face2[CY_NPOLY];
  f3 cdir;   // the clip's dir (CLEAN_NORMAL)
  int cset;
};
__device__ int cy_feature_dim(const int* fv, const Polytope& pt, int off, int* fi, f3* fx) {
  int v1i = pt.vidx[2 * fv[0] + off], v2i = pt.vidx[2 * fv[1] + off], v3i = pt.vidx[2 * fv[2] + off];
  fi[0] = v1i; fi[1] = v2i; fi[2] = v3i;
  fx[0] = pt.vert[2 * fv[0] + off]; fx[1] = pt.vert[2 * fv[1] + off]; fx[2] = pt.vert[2 * fv[2] + off];
  if (v1i != v2i) return (v3i == v1i || v3i == v2i) ? 2 : 3;
  fi[1] = v3i;
  fx[1] = pt.vert[2 * fv[2] + off];
  return v1i != v3i ? 2 : 1;
}
__device__ int cy_aligned_faces(const f3* a, int la, const f3* b, int lb, f3 dir, int* res) {
  float dir_tol = CV_EDGE_TOL * nm3(dir);
  float best = -CV_FACE_TOL;
  int found = 0;
  for (int i = 0; i < la; i++) {
    if (dt3(a[i], dir) >= -dir_tol) continue;
    for (int j = 0; j < lb; j++) {
      if (dt3(b[j], dir) <= dir_tol) continue;
      float d = dt3(a[i], b[j]);
      if (d < best) { best = d; res[0] = i; res[1] = j; found = 1; }
    }
  }
  return found;
}
__device__ int cy_aligned_face_edge(const f3* edge, int nedge, const f3* face, int nface, f3 dir, int* res) {
  for (int i = 0; i < nface; i++) {
    if (dt3(face[i], dir) <= CV_MINVAL) continue;
    for (int j = 0; j < nedge; j++)
      if (fabsf(dt3(edge[j], face[i])) < CV_EDGE_TOL) { res[0] = j; res[1] = i; return 1; }
  }
  return 0;
}
__device__ __forceinline__ f3 cy_fn(int i) {  // mat63 face normals: +x, -x, +y, -y, +z, -z
  return mk3(i == 0 ? 1.f : (i == 1 ? -1.f : 0.f), i == 2 ? 1.f : (i == 3 ? -1.f : 0.f), i == 4 ? 1.f : (i == 5 ? -1.f : 0.f));
}
__device__ int cy_box_normals2(const float* R, f3 n, f3* nout, int* iout) {
  f3 ln = nz3(rtmul(R, n));
  for (int i = 0; i < 6; i++)
    if (dt3(ln, cy_fn(i)) > CV_FACE_TOL) { nout[0] = rmul(R, cy_fn(i)); iout[0] = i; return 1; }
  return 0;
}
__device__ int cy_box_normals(int dim, const int* fi, const float* R, f3 dir, f3* nout, int* iout) {
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
    return cy_box_normals2(R, dir, nout, iout);
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
    return cy_box_normals2(R, dir, nout, iout);
  }
  if (dim == 1) {
    float x = (v1 & 1) ? 1.f : -1.f, y = (v1 & 2) ? 1.f : -1.f, z = (v1 & 4) ? 1.f : -1.f;
    nout[0] = rmul(R, mk3(x, 0.f, 0.f)); nout[1] = rmul(R, mk3(0.f, y, 0.f)); nout[2] = rmul(R, mk3(0.f, 0.f, z));
    iout[0] = x > 0.f ? 0 : 1; iout[1] = y > 0.f ? 2 : 3; iout[2] = z > 0.f ? 4 : 5;
    return 3;
  }
  return 0;
}
// 3.17's _box_edge_normals: the vertices recomputed from their indices; v: the edge's first vertex
__device__ int cy_box_edge_normals(int dim, const CGeom& g, int v1i, int v2i, f3* nout, f3* eout, f3* v) {
  float x = (v1i & 1) ? g.size.x : -g.size.x, y = (v1i & 2) ? g.size.y : -g.size.y, z = (v1i & 4) ? g.size.z : -g.size.z;
  f3 v1 = rmul(g.R, mk3(x, y, z)) + g.pos;
  *v = v1;
  if (dim == 2) {
    float x2 = (v2i & 1) ? g.size.x : -g.size.x, y2 = (v2i & 2) ? g.size.y : -g.size.y, z2 = (v2i & 4) ? g.size.z : -g.size.z;
    f3 v2 = rmul(g.R, mk3(x2, y2, z2)) + g.pos;
    eout[0] = v2;
    nout[0] = nz3(v2 - v1);
    return 1;
  }
  if (dim == 1) {
    eout[0] = rmul(g.R, mk3(-x, y, z)) + g.pos; nout[0] = nz3(eout[0] - v1);
    eout[1] = rmul(g.R, mk3(x, -y, z)) + g.pos; nout[1] = nz3(eout[1] - v1);
    eout[2] = rmul(g.R, mk3(x, y, -z)) + g.pos; nout[2] = nz3(eout[2] - v1);
    return 3;
  }
  return 0;
}
__device__ int cy_box_face(const CGeom& g, int idx, f3* f) {
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
__device__ __forceinline__ f3 cy_col(const float* R, int c) { return mk3(R[c], R[3 + c], R[6 + c]); }  // mat[:, c]
__device__ int cy_cylinder_normals(int dim, const int* fi, const float* R, f3* nout, int* iout) {
  if (dim == 1) {
    float sgn = fi[0] != 0 ? -1.f : 1.f;
    nout[0] = cy_col(R, 2) * sgn;
    iout[0] = fi[0];
    return 1;
  }
  return 0;
}
__device__ int cy_cylinder_edge_normals(int dim, const CGeom& g, f3 v, int v1i, f3* nout, f3* eout, f3* vout) {
  *vout = v;
  if (dim == 1 || dim == 2) {
    float sgn = v1i != 0 ? 1.f : -1.f;
    f3 res = cy_col(g.R, 2) * sgn;
    nout[0] = res;
    eout[0] = v + res * (2.f * g.size.y);
    return 1;
  }
  return 0;
}
__device__ __forceinline__ void cy_cs16(int i, flt* c, flt* s) {
  const flt C[16] = {@COS16@};
  const flt S[16] = {@SIN16@};
  *c = C[i]; *s = S[i];
}
__device__ int cy_cylinder_face(const CGeom& g, int idx, f3* f) {
  float sgn = idx != 0 ? -1.f : 1.f;
  f3 center = cy_col(g.R, 2) * (sgn * g.size.y) + g.pos;
  f3 col0 = cy_col(g.R, 0) * g.size.x;
  f3 col1 = cy_col(g.R, 1) * (g.size.x * sgn);
  for (int i = 0; i < 16; i++) {
    flt c, s;
    cy_cs16(i, &c, &s);
    f[i] = col0 * c - col1 * s + center;
  }
  return 16;
}
#if CY_HAS_MESH
__device__ __forceinline__ f3 cy_mvert(int q) { return mk3(c_mvert[3 * q], c_mvert[3 * q + 1], c_mvert[3 * q + 2]); }
__device__ __forceinline__ f3 cy_mpnorm(int q) { return mk3(c_mpnorm[3 * q], c_mpnorm[3 * q + 1], c_mpnorm[3 * q + 2]); }
// _intersect1 / _intersect2: up to 2 common entries of polymap ranges
__device__ int cy_intersect1(int s1, int s2, int l1, int l2, int* res) {
  int count = 0;
  for (int i = s1; i < s1 + l1; i++)
    for (int j = s2; j < s2 + l2; j++)
      if (c_mpmap[i] == c_mpmap[j]) { res[count++] = c_mpmap[i]; if (count == 2) return 2; }
  return count;
}
__device__ int cy_intersect2(const int* a1, int s2, int l1, int l2, int* res) {
  int count = 0;
  for (int i = 0; i < l1; i++)
    for (int j = s2; j < s2 + l2; j++)
      if (a1[i] == c_mpmap[j]) { res[count++] = a1[i]; if (count == 2) return 2; }
  return count;
}
__device__ int cy_mesh_normals(int dim, const int* fi, const CGeom& g, f3* nout, int* iout) {  // _mesh_normals
  const int va = c_mvadr[g.mesh], pa = c_mpadr[g.mesh];
  if (dim == 3) {
    int edgeset[2], faceset[2];
    int n = cy_intersect1(c_mpmadr[va + fi[0]], c_mpmadr[va + fi[1]], c_mpmnum[va + fi[0]], c_mpmnum[va + fi[1]], edgeset);
    if (n == 0) return 0;
    n = cy_intersect2(edgeset, c_mpmadr[va + fi[2]], n, c_mpmnum[va + fi[2]], faceset);
    if (n == 0) return 0;
    nout[0] = rmul(g.R, cy_mpnorm(pa + faceset[0]));
    iout[0] = faceset[0];
    return 1;
  }
  if (dim == 2) {
    int edgeset[2];
    int n = cy_intersect1(c_mpmadr[va + fi[0]], c_mpmadr[va + fi[1]], c_mpmnum[va + fi[0]], c_mpmnum[va + fi[1]], edgeset);
    if (n == 0) return 0;
    for (int i = 0; i < n; i++) { nout[i] = rmul(g.R, cy_mpnorm(pa + edgeset[i])); iout[i] = edgeset[i]; }
    return n;
  }
  if (dim == 1) {
    const int adr = c_mpmadr[va + fi[0]], num = c_mpmnum[va + fi[0]];
    for (int i = 0; i < num && i < CY_NDEG; i++) {
      const int index = c_mpmap[adr + i];
      nout[i] = rmul(g.R, cy_mpnorm(pa + index));
      iout[i] = index;
    }
    return num < CY_NDEG ? num : CY_NDEG;
  }
  return 0;
}
__device__ int cy_mesh_edge_normals(int dim, const CGeom& g, int v1i, int v2i, f3* nout, f3* eout, f3* v) {
  const int va = c_mvadr[g.mesh], pa = c_mpadr[g.mesh];
  f3 v1 = rmul(g.R, cy_mvert(va + v1i)) + g.pos;
  *v = v1;
  if (dim == 2) {
    f3 v2 = rmul(g.R, cy_mvert(va + v2i)) + g.pos;
    eout[0] = v2;
    nout[0] = nz3(v2 - v1);
    return 1;
  }
  if (dim == 1) {
    const int adr1 = c_mpmadr[va + v1i], num1 = c_mpmnum[va + v1i];
    for (int i = 0; i < num1 && i < CY_NDEG; i++) {
      const int idx = c_mpmap[adr1 + i], adr = c_mpvadr[pa + idx], nvert = c_mpvnum[pa + idx];
      for (int j = 0; j < nvert; j++)
        if (c_mpvert[adr + j] == v1i) {
          const int k = j == 0 ? nvert - 1 : j - 1;
          eout[i] = rmul(g.R, cy_mvert(va + c_mpvert[adr + k])) + g.pos;
          nout[i] = nz3(eout[i] - v1);
        }
    }
    return num1 < CY_NDEG ? num1 : CY_NDEG;
  }
  return 0;
}
__device__ int cy_mesh_face(const CGeom& g, int idx, f3* f) {  // _mesh_face: the polygon's vertices reversed
  const int va = c_mvadr[g.mesh], pa = c_mpadr[g.mesh];
  const int adr = c_mpvadr[pa + idx], nvert = c_mpvnum[pa + idx];
  int j = 0;
  for (int i = nvert - 1; i >= 0; i--) f[j++] = rmul(g.R, cy_mvert(va + c_mpvert[adr + i])) + g.pos;
  return nvert;
}
#else  // (no meshes: never called)
__device__ __forceinline__ int cy_mesh_normals(int, const int*, const CGeom&, f3*, int*) { return 0; }
__device__ __forceinline__ int cy_mesh_edge_normals(int, const CGeom&, int, int, f3*, f3*, f3*) { return 0; }
__device__ __forceinline__ int cy_mesh_face(const CGeom&, int, f3*) { return 0; }
#endif
__device__ int cy_capsule_edge_normals(const CGeom& g, f3 w, f3* nout, f3* eout, f3* vout) {
  f3 res = cy_col(g.R, 2);
  float pr = dt3(w - g.pos, res);
  float len = g.size.y;
  nout[0] = res;
  *vout = w + res * (-pr - len);
  eout[0] = w + res * (-pr + len);
  return 1;
}
__device__ __forceinline__ void cy_witness_on_face(f3 v, f3 p, f3 n, f3 dir, f3* w1, f3* w2, float* d) {
  f3 dd = v - p;
  f3 a = v - dt3(dd, n) * n;
  float dist = dt3(a - v, dir);
  *w1 = a;
  *w2 = a - dist * dir;
  *d = dist;
}
__device__ __forceinline__ float cy_area4(f3 a, f3 b, f3 c, f3 d) { return 0.5f * nm3(cr3(a - c, b - d)); }
__device__ void cy_polygon_quad(const f3* p, int np, int* q) {  // MuJoCo Warp's _polygon_quad
  int b = 1, c = 2, d = 3;
  q[0] = 0; q[1] = b; q[2] = c; q[3] = d;
  float m = cy_area4(p[0], p[b], p[c], p[d]);
  for (int a = 0; a < np; a++) {
    while (true) {
      float m_next = cy_area4(p[a], p[b], p[c], p[(d + 1) % np]);
      if (m_next <= m) break;
      m = m_next; d = (d + 1) % np; q[0] = a; q[1] = b; q[2] = c; q[3] = d;
      while (true) {
        m_next = cy_area4(p[a], p[b], p[(c + 1) % np], p[d]);
        if (m_next <= m) break;
        m = m_next; c = (c + 1) % np; q[0] = a; q[1] = b; q[2] = c; q[3] = d;
      }
      while (true) {
        m_next = cy_area4(p[a], p[(b + 1) % np], p[c], p[d]);
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
// clip polygon face2 against face1 (3.17's _polygon_clip, margin 0): the contacts (witnesses, distances), count
__device__ int cy_polygon_clip(CyBuf& mb, const f3* face1, int nface1, const f3* face2, int nface2, f3 n, f3 dir,
                               f3* w1o, f3* w2o, float* dists) {
  if (nface1 < 3) return 0;
  mb.cdir = dir; mb.cset = 1;
  for (int i = 0; i < nface1; i++) {
    f3 a = face1[i], b = face1[(i + 1) % nface1];
    f3 v3 = a + n, res = cr3(b - a, v3 - a);
    mb.pdist[i] = dt3(res, a);
    mb.pnormal[i] = res;
  }
  f3* poly = mb.polygon;
  f3* clip = mb.clipped;
  int npoly = nface2, nclip = 0, over = 0;
  for (int i = 0; i < nface2; i++) poly[i] = face2[i];
  for (int e = 0; e < nface1; e++) {
    for (int i = 0; i < npoly; i++) {
      f3 P = poly[i], Q = poly[(i + 1) % npoly];
      bool in1 = dt3(P - face1[e], mb.pnormal[e]) > -1e-10f, in2 = dt3(Q - face1[e], mb.pnormal[e]) > -1e-10f;
      if (!in1 && !in2) continue;
      if (in1 && in2) {
        if (nclip < 2 * CY_NPOLY) clip[nclip++] = Q; else over = 1;
        continue;
      }
      float dot = dt3(mb.pnormal[e], Q - P), t = fabsf(dot) < 1e-10f ? CV_FLOAT_MAX : (mb.pdist[e] - dt3(mb.pnormal[e], P)) / dot;
      if (t > -CV_INTERSECT_TOL && t < 1.f + CV_INTERSECT_TOL) {
        t = fminf(fmaxf(t, 0.f), 1.f);
        if (nclip < 2 * CY_NPOLY) clip[nclip++] = P + t * (Q - P); else over = 1;
      }
      if (in2) { if (nclip < 2 * CY_NPOLY) clip[nclip++] = Q; else over = 1; }
    }
    f3* tmp = poly; poly = clip; clip = tmp;
    npoly = nclip;
    nclip = 0;
  }
  if (over) mb.cset = 2;  // (a clipped polygon beyond MuJoCo Warp's buffer: flagged by the caller)
  int m = npoly;
  npoly = 0;
  for (int i = 0; i < m; i++)
    if (dt3(poly[i] - face1[0], n) <= 0.f) {
      if (npoly != i) poly[npoly] = poly[i];
      npoly++;
    }
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
    cy_witness_on_face(poly[b1], face1[0], n, dir, &w1o[0], &w2o[0], &dists[0]);
    cy_witness_on_face(poly[b2], face1[0], n, dir, &w1o[1], &w2o[1], &dists[1]);
    return 2;
  }
  if (npoly > 4) {
    int q[4];
    cy_polygon_quad(poly, npoly, q);
    for (int i = 0; i < 4; i++) cy_witness_on_face(poly[q[i]], face1[0], n, dir, &w1o[i], &w2o[i], &dists[i]);
    return 4;
  }
  for (int i = 0; i < npoly; i++) cy_witness_on_face(poly[i], face1[0], n, dir, &w1o[i], &w2o[i], &dists[i]);
  return npoly;
}
// 3.17's multicontact: up to 4 contacts (witnesses w1, w2, distances) from polytope face fidx; returns the count
// (1: the witnesses x1, x2 alone)
__device__ int cy_multicontact(CyBuf& mb, const Polytope& pt, int fidx, f3 x1, f3 x2, const CGeom& g1,
                               const CGeom& g2, f3* w1, f3* w2, float* dists) {
  w1[0] = x1; w2[0] = x2;
  mb.cset = 0;
  int fv[3], fi1[3], fi2[3];
  f3 fx1[3], fx2[3];
  face_verts(pt.face[fidx], fv);
  int nface1 = cy_feature_dim(fv, pt, 0, fi1, fx1), nface2 = cy_feature_dim(fv, pt, 1, fi2, fx2);
  f3 v1 = fx1[0], v2 = fx2[0];
  f3 dir = x2 - x1, dir_neg = -dir;
  int nn1 = 0, nn2 = 0;
  if (g1.type == 3) nn1 = cy_box_normals(nface1, fi1, g1.R, dir_neg, mb.n1, mb.idx1);
  else if (g1.type == 6) nn1 = cy_mesh_normals(nface1, fi1, g1, mb.n1, mb.idx1);
  else if (g1.type == 5) nn1 = cy_cylinder_normals(nface1, fi1, g1.R, mb.n1, mb.idx1);
  if (g2.type == 3) nn2 = cy_box_normals(nface2, fi2, g2.R, dir, mb.n2, mb.idx2);
  else if (g2.type == 6) nn2 = cy_mesh_normals(nface2, fi2, g2, mb.n2, mb.idx2);
  else if (g2.type == 5) nn2 = cy_cylinder_normals(nface2, fi2, g2.R, mb.n2, mb.idx2);
  int edge1 = 0, edge2 = 0, res[2] = {0, 0};
  int nres = cy_aligned_faces(mb.n1, nn1, mb.n2, nn2, dir, res);
  if (!nres) {
    if (nface1 < 3 && nn2 && (nface1 <= nface2 || !nn1)) {
      nn1 = 0;
      if (g1.type == 3) nn1 = cy_box_edge_normals(nface1, g1, fi1[0], fi1[1], mb.n1, mb.endvert, &v1);
      else if (g1.type == 6) nn1 = cy_mesh_edge_normals(nface1, g1, fi1[0], fi1[1], mb.n1, mb.endvert, &v1);
      else if (g1.type == 5) nn1 = cy_cylinder_edge_normals(nface1, g1, fx1[0], fi1[0], mb.n1, mb.endvert, &v1);
      else if (g1.type == 2) nn1 = cy_capsule_edge_normals(g1, x1, mb.n1, mb.endvert, &v1);
      nres = cy_aligned_face_edge(mb.n1, nn1, mb.n2, nn2, dir, res);
      if (!nres) return 1;
      edge1 = 1;
    } else if (nface2 < 3) {
      nn2 = 0;
      if (g2.type == 3) nn2 = cy_box_edge_normals(nface2, g2, fi2[0], fi2[1], mb.n2, mb.endvert, &v2);
      else if (g2.type == 6) nn2 = cy_mesh_edge_normals(nface2, g2, fi2[0], fi2[1], mb.n2, mb.endvert, &v2);
      else if (g2.type == 5) nn2 = cy_cylinder_edge_normals(nface2, g2, fx2[0], fi2[0], mb.n2, mb.endvert, &v2);
      else if (g2.type == 2) nn2 = cy_capsule_edge_normals(g2, x2, mb.n2, mb.endvert, &v2);
      nres = cy_aligned_face_edge(mb.n2, nn2, mb.n1, nn1, dir_neg, res);
      if (!nres) return 1;
      edge2 = 1;
    } else {
      return 1;
    }
  }
  const int i = res[0], j = res[1];
  int nf1 = 0, nf2 = 0;
  if (edge1) { mb.face1[0] = v1; mb.face1[1] = mb.endvert[i]; nf1 = 2; }
  else {
    const int ind = edge2 ? mb.idx1[j] : mb.idx1[i];
    if (g1.type == 3) nf1 = cy_box_face(g1, ind, mb.face1);
    else if (g1.type == 6) nf1 = cy_mesh_face(g1, ind, mb.face1);
    else if (g1.type == 5) nf1 = cy_cylinder_face(g1, ind, mb.face1);
  }
  if (edge2) { mb.face2[0] = v2; mb.face2[1] = mb.endvert[i]; nf2 = 2; }
  else {
    if (g2.type == 3) nf2 = cy_box_face(g2, mb.idx2[j], mb.face2);
    else if (g2.type == 6) nf2 = cy_mesh_face(g2, mb.idx2[j], mb.face2);
    else if (g2.type == 5) nf2 = cy_cylinder_face(g2, mb.idx2[j], mb.face2);
  }
  int nc;
  if (edge1) {  // face1 is an edge: clipped against face2, then flipped back
    f3 c1[4], c2[4];
    nc = cy_polygon_clip(mb, mb.face2, nf2, mb.face1, nf1, mb.n2[j], -mb.n2[j], c1, c2, dists);
    for (int k = 0; k < nc; k++) { w1[k] = c2[k]; w2[k] = c1[k]; }
  } else if (edge2) {
    nc = cy_polygon_clip(mb, mb.face1, nf1, mb.face2, nf2, mb.n1[j], -mb.n1[j], w1, w2, dists);
  } else {
    nc = cy_polygon_clip(mb, mb.face1, nf1, mb.face2, nf2, mb.n1[i], mb.n2[j], w1, w2, dists);
  }
  if (nc < 1) { mb.cset = mb.cset == 2 ? 2 : 0; return 1; }
  return nc;
}
"""

ENTRY = r"""
// ------------------------------------------------------------- the pair (3.17's ccd_kernel; margin, gap 0)
// The adjoint's record (CyRec, g1_engine.cylccd): every branch of the pair reads only values, which the D1 replays
// share bit for bit with the RF replay, so a D1 replay's tangents are those of the straight-line program the RF replay
// ran. With supports at discrete features (box and mesh vertices, the shrunk sphere's centre and capsule's segment
// ends) the outputs depend on the poses only through the final EPA face's three vertex pairs (kind 1: the deep path)
// or the shrunk shapes' final GJK simplex (kind 2: the inflated path); a D1 replay with the record rebuilds those from
// their recorded features and runs the forward's own code from there (attach_face and epa_witness, or cv_gjk's last
// subdistance; then the multicontact and the assembly), without the searches. Smooth supports (the cylinder's rim, the
// full sphere and capsule) record kind 0: the full replay.
// a support point from its recorded feature: cv_support's expressions for a box vertex (v: the sign bits) and a mesh
// vertex (v: the mesh's local index); the shrunk sphere and capsule (size.x 0; geom 1, at the origin) through
// cv_support itself, at a direction of the recorded end (capsule: v -3 the +z end, -2 the -z end; the other terms
// zeros)
__device__ f3 cy_vertex(const CGeom& g, int v) {
  if (g.type == 3) {
    f3 tmp = mk3((v & 1) ? 1.f : -1.f, (v & 2) ? 1.f : -1.f, (v & 4) ? 1.f : -1.f);
    f3 res = mk3(tmp.x * g.size.x, tmp.y * g.size.y, tmp.z * g.size.z);
    return rmul(g.R, res) + g.pos;
  }
#if CY_HAS_MESH
  if (g.type == 6) {
    const int q = c_mvadr[g.mesh] + v;
    f3 pt = mk3(c_mvert[3 * q], c_mvert[3 * q + 1], c_mvert[3 * q + 2]);
    return rmul(g.R, pt) + g.pos;
  }
#endif
  f3 dir = mk3(0.f, 0.f, 1.f);
  if (g.type == 2) dir = mk3(cy_val(g.R[2]), cy_val(g.R[5]), cy_val(g.R[8])) * (v == -3 ? 1.f : -1.f);
  return cv_support(g, dir).point;
}
// kind 2: the shrunk shapes' GJK result from cv_gjk's last subdistance call (its vertices rebuilt from their recorded
// features: the dropped ones too, so that the call is the same), then cv_gjk's compaction and combinations
__device__ void cy_gjk_rec(const CyRec& rc, CGeom& g1, CGeom& g2, GJKResult& r) {
  f3 simplex[4], simplex1[4], simplex2[4];
  int si1[4] = {0, 0, 0, 0}, si2[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; i++) simplex[i] = simplex1[i] = simplex2[i] = mk3(0.f, 0.f, 0.f);
  float l[4] = {1.f, 0.f, 0.f, 0.f};
  int n = rc.n;
  for (int i = 0; i < n; i++) {
    simplex1[i] = cy_vertex(g1, rc.v1[i]); si1[i] = rc.v1[i];
    simplex2[i] = cy_vertex(g2, rc.v2[i]); si2[i] = rc.v2[i];
    simplex[i] = simplex1[i] - simplex2[i];
  }
  subdistance(n, simplex, l);
  n = 0;
  for (int i = 0; i < 4; i++) {
    if (l[i] == 0.f) continue;
    simplex[n] = simplex[i]; simplex1[n] = simplex1[i]; simplex2[n] = simplex2[i];
    si1[n] = si1[i]; si2[n] = si2[i]; l[n] = l[i];
    n++;
  }
  f3 x_k = lin_combine(n, l, simplex);
  float xnorm = sqrtf(dt3(x_k, x_k));
  r.separated = false;
  r.x1 = lin_combine(n, l, simplex1);
  r.x2 = lin_combine(n, l, simplex2);
  r.dist = xnorm;
  r.dim = n;
  for (int i = 0; i < 4; i++) {
    r.simplex[i] = simplex[i]; r.simplex1[i] = simplex1[i]; r.simplex2[i] = simplex2[i];
    r.sidx1[i] = si1[i] < -1 ? -1 : si1[i]; r.sidx2[i] = si2[i];
  }
  r.index1 = g1.index;
  r.index2 = g2.index;
}
// kind 2's record: the inflated path of a shrunk sphere or capsule (geom 1) against a box or mesh, cv_gjk's last
// subdistance call's vertices (r.ln, r.lv1, r.lv2; the final simplex at most a triangle)
__device__ void cy_record_gjk(CyRec& rc, const GJKResult& r, const CGeom& g1, const CGeom& g2) {
  if (!((g1.type == 1 || g1.type == 2) && (g2.type == 3 || g2.type == 6)) || r.dim < 1 || r.dim > 3) return;
  rc.kind = 2;
  rc.n = r.ln;
  for (int i = 0; i < r.ln; i++) { rc.v1[i] = r.lv1[i]; rc.v2[i] = r.lv2[i]; }
}
// kind 1: the final EPA face rebuilt in the polytope from its three vertex pairs (at their polytope indices fv, the
// face at its index n: the replay's layout), the polytope's centre the recorded values (attach_face's sign test),
// attached by attach_face; returns the face's index
__device__ int cy_face_rec(const CyRec& rc, CGeom& g1, CGeom& g2, Polytope& pt) {
  for (int k = 0; k < 3; k++) {
    pt.vert[2 * rc.fv[k]] = cy_vertex(g1, rc.v1[k]); pt.vidx[2 * rc.fv[k]] = rc.v1[k];
    pt.vert[2 * rc.fv[k] + 1] = cy_vertex(g2, rc.v2[k]); pt.vidx[2 * rc.fv[k] + 1] = rc.v2[k];
  }
  pt.center = mk3(rc.c[0], rc.c[1], rc.c[2]);
  attach_face(pt, rc.n, rc.fv[0], rc.fv[1], rc.fv[2]);
  return rc.n;
}
// gjk_phase: true when EPA is needed; dist, x1, x2 otherwise the result (r.dim 0: no simplex for the multicontact);
// gn: the shrunk shapes' direction of an inflated contact (CLEAN_NORMAL)
// (the adjoint's record, CyRec: rec null in the forward; use false: the search runs and fills rec, true: rec's kind 1
// or 2 replaces the search, which the other kinds run)
__device__ bool cy_gjk_phase(float tol, int iters, CGeom& g1, CGeom& g2, float* dist, f3* x1, f3* x2, GJKResult& r,
                             f3* gn, CyRec* rec, bool use) {
  const f3 s1 = g1.size, s2 = g2.size;
  float fm1 = 0.f, fm2 = 0.f;
  if (g1.type == 1 || g1.type == 2) { fm1 = g1.size.x; g1.size.x = 0.f; }
  if (g2.type == 1 || g2.type == 2) { fm2 = g2.size.x; g2.size.x = 0.f; }
  if (fm1 + fm2 > 0.f) {
    if (use && rec->kind == 2) cy_gjk_rec(*rec, g1, g2, r);
    else cv_gjk(tol, iters, g1, g2, g1.pos, g2.pos, fm1 + fm2, is_discrete(g1.type, g2.type), r);
    g1.index = r.index1; g2.index = r.index2;
    if (r.dist > tol) {
      if (rec && !use && r.dist != CV_FLOAT_MAX) cy_record_gjk(*rec, r, g1, g2);
      g1.size = s1; g2.size = s2;
      if (r.dist == CV_FLOAT_MAX) { *dist = r.dist; *x1 = r.x1; *x2 = r.x2; r.dim = 0; return false; }
      f3 n = nz3(r.x2 - r.x1);  // _inflate
      f3 a = r.x1, b = r.x2;
      if (fm1 > 0.f) a = a + fm1 * n;
      if (fm2 > 0.f) b = b - fm2 * n;
      *dist = r.dist - (fm1 + fm2);
      *x1 = a; *x2 = b;
      *gn = -n;
      return false;
    }
    g1.size = s1; g2.size = s2;
  }
  cv_gjk(tol, iters, g1, g2, g1.pos, g2.pos, 0.f, is_discrete(g1.type, g2.type), r);
  g1.index = r.index1; g2.index = r.index2;
  *dist = r.dist; *x1 = r.x1; *x2 = r.x2;
  if (r.dist > tol || r.dim < 2 || r.separated) { r.dim = 0; return false; }
  return true;
}
__device__ void cy_simplex_to_epa(const GJKResult& r, Polytope& pt) {  // _gjk_simplex_to_epa
  const int i2 = r.dim > 1 ? 1 : 0, i3 = r.dim > 2 ? 2 : 0;
  pt.vert[0] = r.simplex1[0]; pt.vert[1] = r.simplex2[0];
  pt.vert[2] = r.simplex1[i2]; pt.vert[3] = r.simplex2[i2];
  pt.vert[4] = r.simplex1[i3]; pt.vert[5] = r.simplex2[i3];
  pt.vidx[0] = r.sidx1[0]; pt.vidx[1] = r.sidx2[0];
  pt.vidx[2] = r.sidx1[i2]; pt.vidx[3] = r.sidx2[i2];
  pt.vidx[4] = r.sidx1[i3]; pt.vidx[5] = r.sidx2[i3];
  pt.face[0] = 0u | (1u << 10) | (2u << 20);
}
// the contacts of convex pair (g1, g2) (types 2 capsule, 3 box, 5 cylinder, MuJoCo's order) with dist < 0 into o:
// slots the multicontact's points 0 .. 3, the frame from the first one's witnesses
// (rec, use: the adjoint's record, see cy_gjk_phase; rec null in the forward)
__device__ void cy_pair(CGeom g1, CGeom g2, Sink& o, CyRec* rec, bool use) {
  o.n = 0; o.unsupported = 0;
  if (rec && !use) rec->kind = 0;
  const bool face_rec = use && rec->kind == 1;  // (kind 1: the final EPA face from the record)
  const float tol = CY_CCD_TOLERANCE;
  const f3 origin = g1.pos;  // (translation invariance: geom 1 at the origin)
  g1.pos = mk3(0.f, 0.f, 0.f);
  g2.pos = g2.pos - origin;
  GJKResult r;
  float dist;
  f3 x1, x2, gn = mk3(0.f, 0.f, 0.f);
  const bool needs_epa = face_rec || cy_gjk_phase(tol, CY_GJK_ITERATIONS, g1, g2, &dist, &x1, &x2, r, &gn, rec, use);
  Polytope pt;
  pt.nface = 0; pt.nvert = 0; pt.nhorizon = 0; pt.status = 0;
  int midx = -1, efid = -1, tsrc = needs_epa ? 0 : 2;  // the normal's tangent: 0 the EPA face (efid), 1 the clip,
  // 2 GJK's direction
  if (needs_epa) {
    if (face_rec) {}  // (the polytope from the record, below)
    else if (r.dim == 2) { polytope2(pt, r, g1, g2); if (pt.status == -1) r.dim = 3; }
    else if (r.dim == 4) { polytope4(pt, r); if (pt.status == -1) r.dim = 3; }
    if (!face_rec && r.dim == 3) { pt.nface = 0; pt.nvert = 0; pt.nhorizon = 0; polytope3(pt, r, g1, g2); }
    if (pt.status) {
      dist = r.dist; x1 = r.x1; x2 = r.x2;  // (the origin on the boundary: not penetrating)
    } else {
      midx = face_rec ? cy_face_rec(*rec, g1, g2, pt) : cv_epa(tol, CY_EPA_ITERATIONS, pt, g1, g2, is_discrete(g1.type, g2.type));
      if (midx == -1) dist = CV_FLOAT_MAX;
      else dist = epa_witness(pt, midx, &x1, &x2);
      efid = midx;
      if (rec && !use && midx > -1 && is_discrete(g1.type, g2.type)) {  // kind 1's record (box, mesh)
        int fv[3];
        face_verts(pt.face[midx], fv);
        rec->kind = 1;
        rec->n = midx;
        for (int k = 0; k < 3; k++) {
          rec->v1[k] = pt.vidx[2 * fv[k]]; rec->v2[k] = pt.vidx[2 * fv[k] + 1]; rec->fv[k] = fv[k];
        }
        rec->c[0] = cy_val(pt.center.x); rec->c[1] = cy_val(pt.center.y); rec->c[2] = cy_val(pt.center.z);
      }
    }
  }
  if (!(dist < 0.f)) return;
  const bool f1 = g1.type == 3 || g1.type == 5 || g1.type == 6, f2 = g2.type == 3 || g2.type == 5 || g2.type == 6;
  if (!((f1 && f2) || (g1.type == 2 && f2) || (f1 && g2.type == 2))) midx = -1;  // (_supports_multiccd)
  else if (!needs_epa && r.dim > 0) {  // the GJK simplex as the multicontact's polytope face
    cy_simplex_to_epa(r, pt);
    midx = 0;
  }
  f3 w1[4], w2[4];
  float dists[4] = {0.f, 0.f, 0.f, 0.f};
  int nc = 1;
  w1[0] = x1; w2[0] = x2;
  CyBuf mb;
  mb.cset = 0;
  if (midx > -1) {
    nc = cy_multicontact(mb, pt, midx, x1, x2, g1, g2, w1, w2, dists);
    if (nc < 1) { nc = 1; w1[0] = x1; w2[0] = x2; }
    if (mb.cset == 2) o.unsupported |= 8;
    if (nc > 1 || mb.cset) tsrc = mb.cset ? 1 : tsrc;
  }
  f3 nv = nz3(w1[0] - w2[0]);
  f3 tn = nv;  // (CLEAN_NORMAL) the direction whose tangent the normal takes
  if (tsrc == 1) tn = mb.cdir;
  else if (tsrc == 2) tn = gn;
  else {
    int fq[3];
    face_verts(pt.face[efid], fq);
    f3 m0 = pt.vert[2 * fq[0]] - pt.vert[2 * fq[0] + 1], m1 = pt.vert[2 * fq[1]] - pt.vert[2 * fq[1] + 1],
       m2 = pt.vert[2 * fq[2]] - pt.vert[2 * fq[2] + 1];
    tn = nz3(cr3(m1 - m0, m2 - m0));
  }
  nv = cy_clean(nv, tn);
  float nrm[3] = {nv.x, nv.y, nv.z}, fr[9];
  make_frame(nrm, fr);
  for (int i = 0; i < nc; i++) {
    const float di = nc > 1 ? dists[i] : dist;
    if (di < 0.f) sink_put(o, i, di, 0.5f * (w1[i] + w2[i]) + origin, nrm, fr);
  }
}
"""


def instantiate(ns, real):
    """core_text() + MULTI + ENTRY over scalar type real in namespace ns, with its sink (dist, pos, normal, frame) and
    make_frame (step.py's arithmetic without the explicit roundings: the same function); cy_clean(nv, tn): nv, with
    tn's tangent (sign matched) when real is D1 or DN (dual.dn_device: CY_NT tangents)."""
    c16, s16 = _cs16()
    body = core_text() + MULTI.replace("@COS16@", c16).replace("@SIN16@", s16) + ENTRY
    body = re.sub(r"\bfloat\b", "real", body)
    body = re.sub(r"\b(fabs|sqrt|fmin|fmax)f\(", r"t_\1(", body)
    clean = ("__device__ __forceinline__ f3 cy_clean(f3 nv, f3 tn) {  // (CLEAN_NORMAL)\n"
             "  const int sg = (nv.x * tn.x + nv.y * tn.y + nv.z * tn.z).v < 0.f ? -1 : 1;\n"
             "  nv.x.d = sg * tn.x.d; nv.y.d = sg * tn.y.d; nv.z.d = sg * tn.z.d;\n  return nv;\n}\n"
             if real == "D1" else
             "__device__ __forceinline__ f3 cy_clean(f3 nv, f3 tn) {  // (CLEAN_NORMAL; each tangent as D1's)\n"
             "  const int sg = (nv.x * tn.x + nv.y * tn.y + nv.z * tn.z).v < 0.f ? -1 : 1;\n"
             "  DN_SIGNED_TANGENTS(nv.x, sg, tn.x); DN_SIGNED_TANGENTS(nv.y, sg, tn.y); DN_SIGNED_TANGENTS(nv.z, sg, tn.z);\n"
             "  return nv;\n}\n"
             if real == "DN" else "__device__ __forceinline__ f3 cy_clean(f3 nv, f3) { return nv; }\n")
    head = r"""
namespace @NS@ {  // g1_engine.cylccd over @REAL@
using real = @REAL@;
typedef float flt;
__device__ __forceinline__ flt cy_val(real x) { return @VAL@; }  // (the value)
#define CY_SINK_MAX 4
struct Sink {
  int n, unsupported;
  int slot[CY_SINK_MAX];
  real dist[CY_SINK_MAX], pos[CY_SINK_MAX][3], nrm[CY_SINK_MAX][3], fr[CY_SINK_MAX][9];
};
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
    tail = r"""
}  // namespace @NS@
"""
    sink = r"""
__device__ __forceinline__ void sink_put(Sink& o, int slot, real dist, f3 pc, const real* nrm, const real* frame) {
  if (o.n >= CY_SINK_MAX) { o.unsupported |= 4; return; }
  o.slot[o.n] = slot; o.dist[o.n] = dist;
  o.pos[o.n][0] = pc.x; o.pos[o.n][1] = pc.y; o.pos[o.n][2] = pc.z;
  for (int k = 0; k < 3; k++) o.nrm[o.n][k] = nrm[k];
  for (int k = 0; k < 9; k++) o.fr[o.n][k] = frame[k];
  o.n++;
}
"""
    i = body.index("// ------------------------------------------------------------- the pair")
    body = body[:i] + clean + sink + body[i:]
    if real == "DN":
        outl = ("cv_gjk", "cy_gjk_rec", "polytope2", "cy_pair")
        n = [0]

        def mark(mt):
            n[0] += mt.group(2) in outl
            return f"__device__ {'__noinline__' if mt.group(2) in outl else ''} {mt.group(1)}{mt.group(2)}("
        body = re.sub(r"^__device__ (?!__)([\w:]+[ *&]+)(\w+)\(", mark, body, flags=re.M)
        if n[0] != len(outl):
            raise RuntimeError("cylccd.instantiate: not every out-of-line function of the D1 instance found")
    return ((head + body + tail).replace("@NS@", ns).replace("@REAL@", real)
            .replace("@VAL@", "x" if real == "float" else "x.v"))


GLOBAL = r"""
// convex pairs with a cylinder (g1_engine.cylccd): the forward over RF (cycr), the adjoint's replay over D1 (cycd)
@CSINK@
template <class R>
__device__ __forceinline__ void cyc_geom(int t, const float* p, const float* M, const float* s, R* P, R* Q, R* S) {
  for (int k = 0; k < 3; k++) { P[k] = R(p[k]); S[k] = R(s[k]); }
  for (int k = 0; k < 9; k++) Q[k] = R(M[k]);
}
#if CY_HAS_MESH
#define CY_MESH(g) c_gmesh[g]
#else
#define CY_MESH(g) -1
#endif
template <class NS_GEOM, class R>
__device__ __forceinline__ NS_GEOM cyc_cgeom(int t, int gi, const R* P, const R* Q, const R* S) {
  NS_GEOM c;
  c.mesh = t == 6 ? CY_MESH(gi) : -1;
  c.pos.x = P[0]; c.pos.y = P[1]; c.pos.z = P[2];
  for (int k = 0; k < 9; k++) c.R[k] = Q[k];
  c.size.x = S[0]; c.size.y = S[1]; c.size.z = S[2];
  c.type = t;
  c.index = -1;
  return c;
}
// the forward's contacts of the pair (types t1, t2; centres, quaternions, sizes): the bits of the replay's values
// (rec, use: the adjoint's record, cycr::cy_pair's; null in the forward)
__device__ __noinline__ void cyc_pair_x(int t1, int g1, const float* p1, const float* q1, const float* s1, int t2,
                                        int g2, const float* p2, const float* q2, const float* s2, CSink& o,
                                        CyRec* rec, bool use) {
  float R1[9], R2[9];
  quat2mat(q1, R1);
  quat2mat(q2, R2);
  RF P1[3], Q1[9], S1[3], P2[3], Q2[9], S2[3];
  cyc_geom<RF>(t1, p1, R1, s1, P1, Q1, S1);
  cyc_geom<RF>(t2, p2, R2, s2, P2, Q2, S2);
  cycr::Sink d;
  cycr::cy_pair(cyc_cgeom<cycr::CGeom, RF>(t1, g1, P1, Q1, S1), cyc_cgeom<cycr::CGeom, RF>(t2, g2, P2, Q2, S2), d, rec,
                use);
  o.n = d.n; o.unsupported = d.unsupported;
  for (int i = 0; i < d.n; i++) {
    o.slot[i] = d.slot[i]; o.dist[i] = d.dist[i].v;
    for (int k = 0; k < 3; k++) o.pos[i][k] = d.pos[i][k].v;
    for (int k = 0; k < 9; k++) o.fr[i][k] = d.fr[i][k].v;
  }
}
__device__ __forceinline__ void cyc_pair_r(int t1, int g1, const float* p1, const float* q1, const float* s1, int t2,
                                           int g2, const float* p2, const float* q2, const float* s2, CSink& o) {
  cyc_pair_x(t1, g1, p1, q1, s1, t2, g2, p2, q2, s2, o, nullptr, false);
}
"""

FWD = """  if """ + COND + """ {  // convex pairs with a cylinder (g1_engine.cylccd: MuJoCo Warp 3.17's CCD)
    CSink o;
    cyc_pair_r(t1, g1, p1, q1, sz1, t2, g2, p2, q2, sz2, o);
    for (int i = 0; i < o.n; i++) add_contact(s, pair, o.slot[i], o.dist[i], o.pos[i], o.fr[i]);
    if (o.unsupported) atomicOr(&s.unsupported, o.unsupported);
    return;
  }
"""

ADJ_FN = r"""
// the forward's contact `slot` of a convex pair with a cylinder (dist, pos, fr: its bits) and, with cot, its VJP onto
// the geoms' poses gb (g1_engine.cylccd): the pair replayed over D1 once per pose direction of each geom (position;
// rotation about the world axes about the geom's centre: the rotation's columns turn with it). Returns -1 when the
// slot is not a contact. rec (may be null): the contact's record on this lane (CyRec; kind -1 before the replay):
// the replay fills it (the search's discrete choices, checked to reproduce the forward's contacts bit for bit over
// RF, else kind 0, and the slot's dist, pos, frame); a call with cot after the replay on the lane does not replay
// again, and with kind 1 or 2 the D1 replays rebuild the final EPA face or GJK simplex from it instead of searching
__device__ __noinline__ int cyc_adj(int t1, const float* p1, const float* q1, const float* s1, int t2, const float* p2,
                                    const float* q2, const float* s2, int slot, const float* cot, float* gb,
                                    float* dist, float* pos, float* fr, CyRec* rec) {
  CyRec lrec;
  if (!rec) { rec = &lrec; rec->kind = -1; }
  const int g1 = (int)(s1 - c_gsize) / 3, g2 = (int)(s2 - c_gsize) / 3;  // (the geoms' rows: their sizes' offsets)
  if (cot && rec->kind >= 0) {  // (the replay ran on this lane)
    *dist = rec->dist;
    for (int k = 0; k < 3; k++) pos[k] = rec->pos[k];
    for (int k = 0; k < 9; k++) fr[k] = rec->fr[k];
  } else {
    CSink o;
#if CY_REC
    cyc_pair_x(t1, g1, p1, q1, s1, t2, g2, p2, q2, s2, o, rec, false);
#else
    cyc_pair_r(t1, g1, p1, q1, s1, t2, g2, p2, q2, s2, o);
    rec->kind = 0;
#endif
    int c = -1;
    for (int i = 0; i < o.n; i++) if (o.slot[i] == slot) c = i;
    if (c < 0) { rec->kind = -1; return -1; }
    *dist = o.dist[c];
    for (int k = 0; k < 3; k++) pos[k] = o.pos[c][k];
    for (int k = 0; k < 9; k++) fr[k] = o.fr[c][k];
#if CY_REC
    if (rec->kind > 0) {  // the record's contact `slot` over RF: the forward's bits, else the full replay
      cyc_pair_x(t1, g1, p1, q1, s1, t2, g2, p2, q2, s2, o, rec, true);
      c = -1;
      for (int i = 0; i < o.n; i++) if (o.slot[i] == slot) c = i;
      bool same = c >= 0 && __float_as_int(o.dist[c]) == __float_as_int(*dist);
      for (int k = 0; k < 3; k++) same = same && __float_as_int(o.pos[c][k]) == __float_as_int(pos[k]);
      for (int k = 0; k < 9; k++) same = same && __float_as_int(o.fr[c][k]) == __float_as_int(fr[k]);
      if (!same) rec->kind = 0;
    }
#endif
    rec->dist = *dist;
    for (int k = 0; k < 3; k++) rec->pos[k] = pos[k];
    for (int k = 0; k < 9; k++) rec->fr[k] = fr[k];
  }
  if (!cot) return 0;
  const bool use = rec->kind > 0;
  float nbar[3], R1[9], R2[9];
  make_frame_vjp(fr, cot + 4, nbar);
  quat2mat(q1, R1);
  quat2mat(q2, R2);
#if CY_NT > 1
  // CY_NT directions per pass, one tangent each (DN: the values computed once, each tangent by D1's expressions;
  // not D1's bits, NT_HEAD)
  for (int j0 = 0; j0 < 12; j0 += CY_NT) {
    DN P1[3], Q1[9], S1[3], P2[3], Q2[9], S2[3];
    cyc_geom<DN>(t1, p1, R1, s1, P1, Q1, S1);
    cyc_geom<DN>(t2, p2, R2, s2, P2, Q2, S2);
    for (int k = 0; k < CY_NT; k++) {
      const int j = j0 + k, g = j / 6, jj = j % 6;  // geom g, direction jj
      const float* R = g == 0 ? R1 : R2;
      float ea[3] = {0.f, 0.f, 0.f};
      ea[jj % 3] = 1.f;
      DN* P = g == 0 ? P1 : P2;
      DN* Q = g == 0 ? Q1 : Q2;
      if (jj < 3) DN_D(P[jj], k) = 1.f;
      else
        for (int cc = 0; cc < 3; cc++) {  // d col = e x col
          float col[3] = {R[cc], R[3 + cc], R[6 + cc]}, dc[3];
          cross(ea, col, dc);
          for (int r = 0; r < 3; r++) DN_D(Q[3 * r + cc], k) = dc[r];
        }
    }
    cycd::Sink d;
    cycd::cy_pair(cyc_cgeom<cycd::CGeom, DN>(t1, g1, P1, Q1, S1), cyc_cgeom<cycd::CGeom, DN>(t2, g2, P2, Q2, S2), d,
                  use ? rec : nullptr, use);
    int cd = -1;
    for (int i = 0; i < d.n; i++) if (d.slot[i] == slot) cd = i;
    for (int k = 0; k < CY_NT; k++) {
      if (cd < 0) { gb[j0 + k] = 0.f; continue; }  // (not reached: the replay's values are the forward's bits)
      float gj = cot[0] * DN_D(d.dist[cd], k);
      for (int t = 0; t < 3; t++) gj += cot[1 + t] * DN_D(d.pos[cd][t], k) + nbar[t] * DN_D(d.nrm[cd][t], k);
      gb[j0 + k] = gj;
    }
  }
#else
  for (int j = 0; j < 12; j++) {
    const int g = j / 6, jj = j % 6;  // geom g, direction jj
    const float* R = g == 0 ? R1 : R2;
    float ea[3] = {0.f, 0.f, 0.f};
    ea[jj % 3] = 1.f;
    D1 P1[3], Q1[9], S1[3], P2[3], Q2[9], S2[3];
    cyc_geom<D1>(t1, p1, R1, s1, P1, Q1, S1);
    cyc_geom<D1>(t2, p2, R2, s2, P2, Q2, S2);
    D1* P = g == 0 ? P1 : P2;
    D1* Q = g == 0 ? Q1 : Q2;
    if (jj < 3) P[jj].d = 1.f;
    else
      for (int cc = 0; cc < 3; cc++) {  // d col = e x col
        float col[3] = {R[cc], R[3 + cc], R[6 + cc]}, dc[3];
        cross(ea, col, dc);
        for (int r = 0; r < 3; r++) Q[3 * r + cc].d = dc[r];
      }
    cycd::Sink d;
    cycd::cy_pair(cyc_cgeom<cycd::CGeom, D1>(t1, g1, P1, Q1, S1), cyc_cgeom<cycd::CGeom, D1>(t2, g2, P2, Q2, S2), d,
                  use ? rec : nullptr, use);
    int cd = -1;
    for (int i = 0; i < d.n; i++) if (d.slot[i] == slot) cd = i;
    if (cd < 0) { gb[j] = 0.f; continue; }  // (not reached: the replay's values are the forward's bits)
    float gj = cot[0] * d.dist[cd].d;
    for (int t = 0; t < 3; t++) gj += cot[1 + t] * d.pos[cd][t].d + nbar[t] * d.nrm[cd][t].d;
    gb[j] = gj;
  }
#endif
  return 0;
}
"""

ADJ = """  if """ + COND + """ {  // convex pairs with a cylinder (g1_engine.cylccd): the forward's contact `slot`, VJP by D1
    if (cyc_adj(t1, p1, q1, sz1, t2, p2, q2, sz2, slot, cot, gb, dist, pos, fr, crec) < 0) return -1;
    if (!(-0.5f < fr[1] && fr[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
    return br;
  }
"""


def device():
    """RF and D1 or DN (CY_NT > 1: the adjoint's replay with CY_NT tangents per pass) instances (with the duals' and
    RF's device code, once), and the forward's wrapper."""
    from nablaengine.dual import DUAL_DEVICE, RF_DEVICE, dn_device
    csink = ("#ifndef CSINK_DEFINED\n#define CSINK_DEFINED\n#define SINK_MAX 8\nstruct CSink {\n  int n, unsupported;\n"
             "  int slot[SINK_MAX];\n  float dist[SINK_MAX], pos[SINK_MAX][3], fr[SINK_MAX][9];\n};\n#endif")
    return (DUAL_DEVICE + "#ifndef DK_RF_DEVICE\n#define DK_RF_DEVICE\n" + RF_DEVICE + "#endif\n" + CYREC
            + NT_HEAD.replace("@NT@", str(ccd_nt())) + "#if CY_NT > 1\n#define DN_K CY_NT\n" + dn_device() + "#endif\n"
            + instantiate("cycr", "RF") + "#if CY_NT > 1\n" + instantiate("cycd", "DN") + "#else\n"
            + instantiate("cycd", "D1") + "#endif\n" + GLOBAL.replace("@CSINK@", csink))


def ccd_nt():
    """$DK_CCD_NT (default 4): the CCD replay's tangents per pass (CY_NT, NT_HEAD)."""
    import os
    v = (os.environ.get("DK_CCD_NT") or "4").strip()
    if v not in ("1", "2", "3", "4", "6", "12"):
        raise ValueError(f"DK_CCD_NT={v!r}: one of 1, 2, 3, 4, 6, 12")
    return int(v)


NT_HEAD = """#ifndef CY_NT
#define CY_NT @NT@
#endif
#if 12 % CY_NT != 0
#error "CY_NT: one of 1, 2, 3, 4, 6, 12"
#endif
"""


CYREC = r"""
#ifndef CY_REC
#define CY_REC 1  // (0: cyc_adj without its record, model_hooks' DK_CCD_FLOAT experiment)
#endif
// a convex contact's record in the adjoint (g1_engine.cylccd, cy_pair, cyc_adj): kind -1 none, 0 the full replay,
// 1 the final EPA face of discrete supports (v1, v2: its vertices' features on geom 1 and 2, fv: their polytope
// indices, n: the face's, c: the polytope's centre), 2 the shrunk shapes' final GJK simplex (n vertices, v1, v2);
// the slot's dist, pos, frame (the replay's)
struct CyRec {
  int kind, n;
  int v1[4], v2[4], fv[3];
  float c[3];
  float dist, pos[3], fr[9];
};
"""


def apply_rec(src):
    """src (an adjoint text with cyc_adj) with the contact's record handed from adj_contact_one's replay call of
    adj_pair_geom to its VJP call (cyc_adj does not replay again): a parameter CyRec* crec after the last one of
    adj_contact_geom, adj_pair_geom (and adj_pair_geom_aux, terrain builds), passed through by them, the record a lane
    variable of adj_contact_one (its two calls), null for any other caller. Applied after every other edit."""
    heads = ("__device__ __noinline__ int adj_contact_geom(", "__device__ __forceinline__ int adj_pair_geom(",
             "__device__ __forceinline__ int adj_pair_geom_aux(")
    for h in heads:
        n = src.count(h)
        if n == 0 and h != heads[2]:
            raise RuntimeError(f"cylccd.apply_rec: no {h!r}")
        if n > 1:
            raise RuntimeError(f"cylccd.apply_rec: {h!r} {n} times")
        if n:
            j = src.index(") {", src.index(h))
            src = src[:j] + ", CyRec* crec" + src[j:]
    out, i, nrec = [], 0, 0
    for mt in re.finditer(r"\b(adj_contact_geom|adj_pair_geom|adj_pair_geom_aux)\(", src):
        if src[max(0, mt.start() - 4):mt.start()] == "int ":
            continue
        k, depth = mt.end(), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(src[k], 0)
            k += 1
        args = src[mt.end():k - 1]
        if "&nu, nullptr, nullptr" in args or "&nu2, cot, gb" in args:
            arg, nrec = "&crec", nrec + 1
        elif re.search(r"\bcot, gb(, mvid)?$", args):
            arg = "crec"
        else:
            arg = "nullptr"
        out.append(src[i:k - 1] + ", " + arg)
        i = k - 1
    src = "".join(out) + src[i:]
    if nrec != 2:
        raise RuntimeError(f"cylccd.apply_rec: {nrec} adj_contact_one calls (expected 2)")
    first = re.search(r"\n( *)(const int br = adj_pair_geom(_aux)?\(pair, slot, [^\n]*&crec\);\n)", src)
    if first is None:
        raise RuntimeError("cylccd.apply_rec: no replay call in adj_contact_one")
    decl = f"\n{first.group(1)}CyRec crec;  // (g1_engine.cylccd: the replay's record for the VJP's call)\n" \
           f"{first.group(1)}crec.kind = -1;\n{first.group(1)}"
    return src[:first.start()] + decl + first.group(2) + src[first.end():]


def hooks():
    """The (anchor, replacement, count) edits of the engine text for convex pairs with a cylinder."""
    sphere_anchor = "  atomicOr(&s.unsupported, 1);  // other type combinations do not occur in the G1 model\n"
    dev_anchor = "template <class S>\n__device__ DK_INL_COLLIDE void collide_pair(S& s, int pair) {"
    adj_head = "__device__ __noinline__ int adj_contact_geom("
    return ((sphere_anchor, FWD + sphere_anchor, 1),
            (dev_anchor, device() + dev_anchor, 1),
            (adj_head, ADJ_FN + adj_head, 1),
            ("  if (t1 != 2 || t2 != 2) return -1;\n", ADJ + "  if (t1 != 2 || t2 != 2) return -1;\n", 1))

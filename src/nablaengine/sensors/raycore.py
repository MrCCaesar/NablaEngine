"""The ray-casting library: CUDA source, build (nvcc, cached), and the scene data layout.

One thread per (world, ray). A ray is given in a per-world sensor frame (pos (B, 3), rot (B, 3, 3) row major): origin
pos + rot off, direction rot dir (dir unit, off / dir (N, 3)); it is cast against
  terrain    the static boxes and heightfields of a nablaengine.terrain.Terrain (pack(cell): its 2-D broad-phase grid),
             walked cell by cell along the ray (2-D DDA), stopping at the first cell whose exit lies beyond the nearest
             hit; heightfields by a DDA over their cells (two triangles per cell, MuJoCo Warp's ray_hfield split along
             the (c, r)-(c+1, r+1) diagonal) plus their base box and side walls, as ray_hfield;
  floor      optionally the plane z = floor_z;
  objects    per-world boxes (B, K, 15: pos, rot row major, half sizes; half size 0 = absent);
  robot      triangle meshes, one per robot geom, each with a BVH (bvh.py), posed per world by its body's pose (xpos
             (B, nbody, 3), xquat (B, nbody, 4)) and the geom's local pose; meshes of exclude_body are skipped (the
             sensor's own body, as mjlab's RayCastSensor exclude_parent_body).
Output per ray: dist (the nearest hit's distance along the ray, -1 for none or beyond max_distance), id (int32: ID_*
category | index; -1 none) and the world normal. Ray-box and ray-triangle tests follow MuJoCo Warp's ray.py
(ray_box, _ray_triangle; github.com/google-deepmind/mujoco_warp, Apache License 2.0) for the terrain and objects; the
robot's triangles use the two-sided Moller-Trumbore test (MuJoCo's mju_rayMesh also tests every face from both sides)."""
import hashlib
import os
import subprocess
from pathlib import Path

ID_BOX, ID_HFIELD, ID_FLOOR, ID_OBJECT, ID_ROBOT = 0 << 24, 1 << 24, 2 << 24, 3 << 24, 4 << 24
ID_MASK = (1 << 24) - 1

SOURCE = r"""
#include <cuda_runtime.h>
#include <math.h>
#define TR_MINVAL 1e-15f
#define TR_INF 1e30f
#define ID_BOX 0
#define ID_HFIELD (1 << 24)
#define ID_FLOOR (2 << 24)
#define ID_OBJECT (3 << 24)
#define ID_ROBOT (4 << 24)
#define MP 12  // floats per (world, mesh geom) in mpose
#ifdef RAY_STATS  // counters (debug): grid cells, box tests, heightfield calls, heightfield cells, rays
__device__ unsigned long long g_rstat[8];
#define RSTAT(k, n) atomicAdd(&g_rstat[k], (unsigned long long)(n))
#else
#define RSTAT(k, n) do {} while (0)
#endif
extern "C" int dk_rays_stats(unsigned long long* o, int reset) {
#ifdef RAY_STATS
  cudaMemcpyFromSymbol(o, g_rstat, sizeof(g_rstat));
  if (reset) { unsigned long long z[8] = {0}; cudaMemcpyToSymbol(g_rstat, z, sizeof(z)); }
  return 0;
#else
  (void)o; (void)reset;
  return -1;
#endif
}
#ifndef RAY_BLOCK
#define RAY_BLOCK 128  // threads per block of the ray kernels
#endif
#ifndef STATIC_MINB
#define STATIC_MINB 1  // k_rays_static's __launch_bounds__ minimum blocks per SM (a register cap when > 1)
#endif
#ifndef ROBOT_MINB
#define ROBOT_MINB 1  // k_rays_robot's __launch_bounds__ minimum blocks per SM (a register cap when > 1)
#endif
#ifndef HF_PAD
#define HF_PAD 0  // heightfield cells tested around the DDA's: 0 the cell alone, 1 its 3 x 3 block
#endif

struct Scene {
  // terrain (nablaengine.terrain.Terrain.pack): boxes (nbox, 15), heightfield headers (nhf, 16) / ints (nhf, 4) / data,
  // grid [x0, y0, 1 / cell, nx, ny, nbox], cell_start (nx ny + 1), cell_items; grid null: no terrain
  const float *box, *hfh, *hfd, *grid, *cellz;  // cellz: the top of each grid cell's elements
  const int *hfi, *cstart, *citems;
  int has_floor;
  float floor_z;
  float zmax;  // the terrain's highest point (cellz's maximum): the rays' parts above it skip the grid
  // per-world objects (B, nobj, 15)
  int nobj;
  const float* obj;
  // robot meshes: per mesh geom (body, local pos 3, local quat 4, bounding-sphere centre 3, radius) = 12 floats, its
  // root node; BVH nodes: lo 3, hi 3 (float), ints (left, right) or (-(first tri) - 1, count); triangles 9 floats
  int nmesh, nbody, exclude_body;
  const float* mgeom;
  const int* mroot;
  const float* nodeb;
  const int* nodei;
  const float* tri;
  const float *xpos, *xquat;  // (B, nbody, 3 / 4)
  float* mpose;  // work: (B, nmesh, MP) each mesh geom's world position, quaternion, its bounding sphere's centre
                 // from the sensor origin and the cosine of its angular radius from there (k_mesh_cull)
  unsigned* mmask;  // work: (B, 2) the mesh geoms that can meet the sensor's ray cone (bit g)
};

__device__ __forceinline__ float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
__device__ __forceinline__ void cross(const float* a, const float* b, float* o) {
  o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
}
__device__ __forceinline__ void mv3(const float* A, const float* v, float* o) {
  for (int i = 0; i < 3; i++) o[i] = A[3 * i] * v[0] + A[3 * i + 1] * v[1] + A[3 * i + 2] * v[2];
}
__device__ __forceinline__ void mtv3(const float* A, const float* v, float* o) {
  for (int i = 0; i < 3; i++) o[i] = A[i] * v[0] + A[3 + i] * v[1] + A[6 + i] * v[2];
}
__device__ __forceinline__ void qrot(const float* q, const float* v, float* o) {  // o = q v q^-1
  float u[3] = {q[1], q[2], q[3]}, t[3], c[3];
  cross(u, v, t);
  for (int k = 0; k < 3; k++) t[k] *= 2.f;
  cross(u, t, c);
  for (int k = 0; k < 3; k++) o[k] = v[k] + q[0] * t[k] + c[k];
}
__device__ __forceinline__ void qrot_inv(const float* q, const float* v, float* o) {
  const float qc[4] = {q[0], -q[1], -q[2], -q[3]};
  qrot(qc, v, o);
}
__device__ __forceinline__ void qmul(const float* a, const float* b, float* o) {
  o[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  o[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  o[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  o[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}
__device__ __forceinline__ float tr_div(float x, float y) { return x / (y != 0.f ? y : TR_MINVAL); }

__device__ __forceinline__ float tr_quad(float a, float b, float c) {
  float det = b * b - a * c;
  if (det < TR_MINVAL) return -1.f;
  det = sqrtf(det);
  const float den = tr_div(1.f, a), x0 = (-b - det) * den, x1 = (-b + det) * den;
  return x0 >= 0.f ? x0 : (x1 >= 0.f ? x1 : -1.f);
}

// ray_box (MuJoCo Warp): all[6] per side solutions (or -1), nrm world
__device__ float tr_box(const float* pos, const float* R, const float* sz, const float* pnt, const float* vec, float* all,
                        float* nrm) {
  for (int i = 0; i < 6; i++) all[i] = -1.f;
  for (int k = 0; k < 3; k++) nrm[k] = 0.f;
  float dif[3] = {pnt[0] - pos[0], pnt[1] - pos[1], pnt[2] - pos[2]};
  if (tr_quad(dot3(vec, vec), dot3(vec, dif), dot3(dif, dif) - dot3(sz, sz)) < 0.f) return -1.f;
  float lp[3], lv[3];
  mtv3(R, dif, lp);
  mtv3(R, vec, lv);
  float x = -1.f;
  int fa = -1, fs = -1;
  for (int i = 0; i < 3; i++) {
    if (!(fabsf(lv[i]) > TR_MINVAL)) continue;
    for (int side = -1; side < 2; side += 2) {
      const float sol = (float(side) * sz[i] - lp[i]) / lv[i];
      if (sol >= 0.f) {
        const int i0 = i == 0 ? 1 : 0, i1 = i == 2 ? 1 : 2;
        const float p0 = lp[i0] + sol * lv[i0], p1 = lp[i1] + sol * lv[i1];
        if (fabsf(p0) <= sz[i0] && fabsf(p1) <= sz[i1]) {
          if (x < 0.f || sol < x) { x = sol; fa = i; fs = side; }
          all[2 * i + (side + 1) / 2] = sol;
        }
      }
    }
  }
  if (x >= 0.f) {
    float nl[3] = {0.f, 0.f, 0.f};
    nl[fa] = float(fs);
    mv3(R, nl, nrm);
  }
  return x;
}

// _ray_triangle (MuJoCo Warp; local frame; b0, b1: the orthogonal basis of the ray direction)
__device__ float tr_tri(const float* v0, const float* v1, const float* v2, const float* pnt, const float* vec,
                        const float* b0, const float* b1, float* nrm) {
  float d0[3], d1[3], d2[3];
  for (int k = 0; k < 3; k++) { d0[k] = v0[k] - pnt[k]; d1[k] = v1[k] - pnt[k]; d2[k] = v2[k] - pnt[k]; }
  const float p00 = dot3(d0, b0), p01 = dot3(d0, b1), p10 = dot3(d1, b0), p11 = dot3(d1, b1), p20 = dot3(d2, b0),
              p21 = dot3(d2, b1);
  if ((p00 > 0.f && p10 > 0.f && p20 > 0.f) || (p00 < 0.f && p10 < 0.f && p20 < 0.f) ||
      (p01 > 0.f && p11 > 0.f && p21 > 0.f) || (p01 < 0.f && p11 < 0.f && p21 < 0.f))
    return -1.f;
  const float A00 = p00 - p20, A10 = p10 - p20, A01 = p01 - p21, A11 = p11 - p21, bb0 = -p20, bb1 = -p21;
  const float det = A00 * A11 - A10 * A01;
  if (fabsf(det) < TR_MINVAL) return -1.f;
  const float t0 = (A11 * bb0 - A10 * bb1) / det, t1 = (-A01 * bb0 + A00 * bb1) / det;
  if (t0 < 0.f || t1 < 0.f || t0 + t1 > 1.f) return -1.f;
  float e0[3], e1[3], e2[3], n[3];
  for (int k = 0; k < 3; k++) { e0[k] = v0[k] - v2[k]; e1[k] = v1[k] - v2[k]; e2[k] = pnt[k] - v2[k]; }
  cross(e0, e1, n);
  const float den = dot3(vec, n);
  if (fabsf(den) < TR_MINVAL) return -1.f;
  const float dist = -dot3(e2, n) / den, il = rsqrtf(dot3(n, n));
  for (int k = 0; k < 3; k++) nrm[k] = n[k] * il;
  return dist >= 0.f ? dist : -1.f;
}

// ray_hfield with a DDA over the heightfield's cells: the base box, the top box, the triangles of the cells the ray
// crosses inside the top box (in ray order, stopping after the first cell with a hit within it), the side walls
#ifndef HF_NOINLINE
#define HF_NOINLINE 0
#endif
#if HF_NOINLINE
__device__ __noinline__
#else
__device__
#endif
float tr_hfield(const Scene& S, int j, const float* pnt, const float* vec, float* nrm) {
  const float* hh = S.hfh + 16 * j;
  const int* hi = S.hfi + 4 * j;
  const float *pos = hh, *R = hh + 3, *size = hh + 12, *hd = S.hfd + hi[2];
  const int nrow = hi[0], ncol = hi[1];
  const float col[3] = {R[2], R[5], R[8]};
  const float bs = size[3] * 0.5f, ts = size[2] * 0.5f;
  float bsz[3] = {size[0], size[1], bs}, tsz[3] = {size[0], size[1], ts}, bpos[3], tpos[3], allb[6], all[6], nb[3],
        nt[3];
  for (int k = 0; k < 3; k++) { bpos[k] = pos[k] - col[k] * bs; tpos[k] = pos[k] + col[k] * ts; }
  float x = tr_box(bpos, R, bsz, pnt, vec, allb, nb);
  const float top = tr_box(tpos, R, tsz, pnt, vec, all, nt);
  if (top < 0.f) { for (int k = 0; k < 3; k++) nrm[k] = nb[k]; return x; }
  float dif[3] = {pnt[0] - pos[0], pnt[1] - pos[1], pnt[2] - pos[2]}, lp[3], lv[3];
  mtv3(R, dif, lp);
  mtv3(R, vec, lv);
  const float sg = lv[2] >= 0.f ? 1.f : -1.f, a = -1.f / (sg + lv[2]), b = lv[0] * lv[1] * a;
  const float b0[3] = {1.f + sg * lv[0] * lv[0] * a, sg * b, -sg * lv[0]}, b1[3] = {b, sg + lv[1] * lv[1] * a, -lv[1]};
  float s0 = 0.f, s1 = top;
  for (int i = 0; i < 6; i++)
    if (all[i] > s1) { s0 = top; s1 = all[i]; }
  const float dx = tr_div(2.f * size[0], float(ncol - 1)), dy = tr_div(2.f * size[1], float(nrow - 1));
  float nl[3] = {0.f, 0.f, 0.f};
  if (x >= 0.f) mtv3(R, nb, nl);
  {  // DDA from s0 to s1 in cell coordinates u = (lp + s lv + size) / d
    const float ux = tr_div(lp[0] + s0 * lv[0] + size[0], dx), uy = tr_div(lp[1] + s0 * lv[1] + size[1], dy);
    int c = min(max(int(floorf(ux)), 0), ncol - 2), r = min(max(int(floorf(uy)), 0), nrow - 2);
    const float gx = tr_div(lv[0], dx), gy = tr_div(lv[1], dy);  // cell units per unit s
    const int stx = gx > 0.f ? 1 : -1, sty = gy > 0.f ? 1 : -1;
    const float tdx = fabsf(gx) > TR_MINVAL ? 1.f / fabsf(gx) : TR_INF, tdy = fabsf(gy) > TR_MINVAL ? 1.f / fabsf(gy) : TR_INF;
    float tnx = fabsf(gx) > TR_MINVAL ? s0 + ((gx > 0.f ? float(c + 1) : float(c)) - ux) / gx : TR_INF;
    float tny = fabsf(gy) > TR_MINVAL ? s0 + ((gy > 0.f ? float(r + 1) : float(r)) - uy) / gy : TR_INF;
    for (int it = 0; it < ncol + nrow + 4; it++) {
      // this cell, and (HF_PAD 1) its 8 neighbours as MuJoCo Warp's 1-cell pad against rounding at the cell borders
      float best = -1.f, bn[3];
      for (int rr = max(r - HF_PAD, 0); rr <= min(r + HF_PAD, nrow - 2); rr++)
        for (int cc = max(c - HF_PAD, 0); cc <= min(c + HF_PAD, ncol - 2); cc++) {
          const float x0 = dx * float(cc) - size[0], x1 = dx * float(cc + 1) - size[0];
          const float y0 = dy * float(rr) - size[1], y1 = dy * float(rr + 1) - size[1];
          const float h00 = hd[rr * ncol + cc] * size[2], h10 = hd[rr * ncol + cc + 1] * size[2];
          const float h11 = hd[(rr + 1) * ncol + cc + 1] * size[2], h01 = hd[(rr + 1) * ncol + cc] * size[2];
          const float va[3] = {x0, y0, h00}, vb[3] = {x1, y0, h10}, vc[3] = {x1, y1, h11}, vd[3] = {x0, y1, h01};
          float nt2[3];
          float sol = tr_tri(va, vb, vc, lp, lv, b0, b1, nt2);
          if (sol >= 0.f && (best < 0.f || sol < best)) { best = sol; for (int k = 0; k < 3; k++) bn[k] = nt2[k]; }
          sol = tr_tri(va, vc, vd, lp, lv, b0, b1, nt2);
          if (sol >= 0.f && (best < 0.f || sol < best)) { best = sol; for (int k = 0; k < 3; k++) bn[k] = nt2[k]; }
        }
      RSTAT(3, 1);
      if (best >= 0.f && (x < 0.f || best < x)) { x = best; for (int k = 0; k < 3; k++) nl[k] = bn[k]; }
      const float texit = fminf(tnx, tny);
      if (x >= 0.f && x <= texit) break;  // the hits of later cells lie beyond this cell's exit
      if (texit > s1) break;
      if (tnx < tny) { c += stx; tnx += tdx; } else { r += sty; tny += tdy; }
      if (c < 0 || c > ncol - 2 || r < 0 || r > nrow - 2) break;
    }
  }
  for (int i = 0; i < 4; i++) {
    if (!(all[i] >= 0.f && (all[i] < x || x < 0.f))) continue;
    const float z = tr_div(lp[2] + all[i] * lv[2], size[2]);
    float y, y0, z0, z1;
    if (i < 2) {
      y = tr_div(lp[1] + all[i] * lv[1] + size[1], dy);
      y0 = fmaxf(0.f, fminf(float(nrow - 2), floorf(y)));
      const int k0 = int(rintf(y0)), k1 = int(rintf(y0 + 1.f));
      z0 = i == 1 ? hd[k0 * ncol + ncol - 1] : hd[k0 * ncol];
      z1 = i == 1 ? hd[k1 * ncol + ncol - 1] : hd[k1 * ncol];
    } else {
      y = tr_div(lp[0] + all[i] * lv[0] + size[0], dx);
      y0 = fmaxf(0.f, fminf(float(ncol - 2), floorf(y)));
      const int k0 = int(rintf(y0)), k1 = int(rintf(y0 + 1.f));
      z0 = i == 3 ? hd[k0 + (nrow - 1) * ncol] : hd[k0];
      z1 = i == 3 ? hd[k1 + (nrow - 1) * ncol] : hd[k1];
    }
    if (z < z0 * (y0 + 1.f - y) + z1 * (y - y0)) {
      x = all[i];
      nl[0] = float(i == 1) - float(i == 0); nl[1] = float(i == 3) - float(i == 2); nl[2] = 0.f;
    }
  }
  if (x >= 0.f) mv3(R, nl, nrm);
  else for (int k = 0; k < 3; k++) nrm[k] = 0.f;
  return x;
}

__device__ __forceinline__ void take(float d, int id, const float* n, float* best, int* bid, float* bn) {
  if (d >= 0.f && (*best < 0.f || d < *best)) { *best = d; *bid = id; for (int k = 0; k < 3; k++) bn[k] = n[k]; }
}

// the terrain along the ray (2-D DDA over the broad-phase grid; up to maxd)
__device__ void cast_terrain(const Scene& S, const float* p, const float* v, float maxd, float* best, int* bid,
                             float* bn) {
  const float x0 = S.grid[0], y0 = S.grid[1], ic = S.grid[2];
  const int nx = int(S.grid[3]), ny = int(S.grid[4]), nbox = int(S.grid[5]);
  // the ray's part over the grid rectangle
  float t0 = 0.f, t1 = maxd;
  const float lo[2] = {x0, y0}, hi[2] = {x0 + nx / ic, y0 + ny / ic};
  for (int k = 0; k < 2; k++) {
    if (fabsf(v[k]) < TR_MINVAL) { if (p[k] < lo[k] || p[k] > hi[k]) return; continue; }
    float a = (lo[k] - p[k]) / v[k], b = (hi[k] - p[k]) / v[k];
    if (a > b) { const float t = a; a = b; b = t; }
    t0 = fmaxf(t0, a); t1 = fminf(t1, b);
  }
  // and below the terrain's top
  if (v[2] < -TR_MINVAL) t0 = fmaxf(t0, (p[2] - S.zmax) / -v[2]);
  else if (p[2] > S.zmax) return;
  else if (v[2] > TR_MINVAL) t1 = fminf(t1, (S.zmax - p[2]) / v[2]);
  if (t0 > t1) return;
  const float ux = (p[0] + t0 * v[0] - x0) * ic, uy = (p[1] + t0 * v[1] - y0) * ic;
  int c = min(max(int(floorf(ux)), 0), nx - 1), r = min(max(int(floorf(uy)), 0), ny - 1);
  const float gx = v[0] * ic, gy = v[1] * ic;
  const int stx = gx > 0.f ? 1 : -1, sty = gy > 0.f ? 1 : -1;
  const float tdx = fabsf(gx) > TR_MINVAL ? 1.f / fabsf(gx) : TR_INF, tdy = fabsf(gy) > TR_MINVAL ? 1.f / fabsf(gy) : TR_INF;
  float tnx = fabsf(gx) > TR_MINVAL ? t0 + ((gx > 0.f ? float(c + 1) : float(c)) - ux) / gx : TR_INF;
  float tny = fabsf(gy) > TR_MINVAL ? t0 + ((gy > 0.f ? float(r + 1) : float(r)) - uy) / gy : TR_INF;
  float tin = t0;
  int seen[4] = {-1, -1, -1, -1}, nseen = 0;  // heightfields already cast (each in many cells: once per ray)
  for (int it = 0; it < nx + ny + 2; it++) {
    const float texit0 = fminf(fminf(tnx, tny), t1);
    // the cell's elements unless the ray stays above their top over the cell
    const bool above = fminf(p[2] + tin * v[2], p[2] + texit0 * v[2]) > S.cellz[r * nx + c];
    RSTAT(0, 1);
    if (!above)
    for (int k = S.cstart[r * nx + c]; k < S.cstart[r * nx + c + 1]; k++) {
      const int e = S.citems[k];
      float all[6], nn[3], d;
      if (e < nbox) { RSTAT(1, 1); const float* b = S.box + 15 * e; d = tr_box(b, b + 3, b + 12, p, v, all, nn); take(d, ID_BOX + e, nn, best, bid, bn); }
      else {
        const int j = e - nbox;
        if (j == seen[0] || j == seen[1] || j == seen[2] || j == seen[3]) continue;
        seen[nseen & 3] = j;
        nseen++;
        RSTAT(2, 1);
        d = tr_hfield(S, j, p, v, nn);
        take(d, ID_HFIELD + j, nn, best, bid, bn);
      }
    }
    const float texit = fminf(tnx, tny);
    if (*best >= 0.f && *best <= texit) return;
    if (texit > t1) return;
    tin = texit;
    if (tnx < tny) { c += stx; tnx += tdx; } else { r += sty; tny += tdy; }
    if (c < 0 || c >= nx || r < 0 || r >= ny) return;
  }
}

// per (world, mesh geom): its world pose into mpose, and whether its bounding sphere meets the sensor's ray cone
// (apex pos, axis R e_x, half angle with cos ch / sin sh, length maxd) into mmask
// offmax: the largest ray offset from the sensor origin (the angular test's sphere is inflated by it)
__global__ void k_mesh_cull(Scene S, int B, const float* pos, const float* rot, float ch, float sh, float maxd,
                            float offmax) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= B * S.nmesh) return;
  const int w = tid / S.nmesh, g = tid % S.nmesh;
  const float* mg = S.mgeom + 12 * g;
  const int body = int(mg[0]);
  const float* bx = S.xpos + ((size_t)w * S.nbody + body) * 3;
  const float* bq = S.xquat + ((size_t)w * S.nbody + body) * 4;
  float gq[4], gp[3], t[3], c[3];
  qmul(bq, mg + 4, gq);
  qrot(bq, mg + 1, t);
  for (int k = 0; k < 3; k++) gp[k] = bx[k] + t[k];
  float* o = S.mpose + (size_t)tid * MP;
  for (int k = 0; k < 3; k++) o[k] = gp[k];
  for (int k = 0; k < 4; k++) o[3 + k] = gq[k];
  qrot(gq, mg + 8, t);
  for (int k = 0; k < 3; k++) c[k] = gp[k] + t[k] - pos[3 * w + k];  // the sphere centre from the apex
  {  // the angular test of the rays: a unit direction v can meet the (inflated) sphere only if v . c >= cos |c|
    const float ri = mg[11] + offmax, dc = sqrtf(dot3(c, c));
    for (int k = 0; k < 3; k++) o[8 + k] = c[k];
    o[11] = dc > ri ? sqrtf(fmaxf(dc * dc - ri * ri, 0.f)) / dc : -2.f;  // cos of the angular radius; inside: always
  }
  const float ax[3] = {rot[9 * w], rot[9 * w + 3], rot[9 * w + 6]}, rad = mg[11];
  const float along = dot3(c, ax), d2 = dot3(c, c), perp = sqrtf(fmaxf(d2 - along * along, 0.f));
  // sphere vs cone: the distance from the centre to the cone surface (Eberly) below the radius, or the centre inside
  bool keep = d2 <= rad * rad || (along * ch + perp * sh >= -rad && perp * ch - along * sh <= rad);
  keep = keep && along <= maxd + rad && sqrtf(d2) <= maxd + rad;
  if (dot3(c, c) > (maxd + rad) * (maxd + rad)) keep = false;
  if (body == S.exclude_body) keep = false;
  if (keep) atomicOr(S.mmask + 2 * w + (g >> 5), 1u << (g & 31));
}

// the robot's mesh geoms that can meet the ray cone (mmask), each in its frame (mpose), their BVH (two-sided
// Moller-Trumbore), nearest child first
__device__ void cast_robot(const Scene& S, int w, const float* p, const float* v, float* best, int* bid, float* bn) {
  for (int h = 0; h < 2; h++) {
    unsigned bits = S.mmask[2 * w + h];
    while (bits) {
      const int g = 32 * h + __ffs(bits) - 1;
      bits &= bits - 1u;
      const float* mg = S.mgeom + 12 * g;
      const float* mp = S.mpose + ((size_t)w * S.nmesh + g) * MP;
      {  // the angular test (k_mesh_cull): the direction against the sphere's cone from the sensor origin
        const float vc = v[0] * mp[8] + v[1] * mp[9] + v[2] * mp[10];
        const float cc = sqrtf(mp[8] * mp[8] + mp[9] * mp[9] + mp[10] * mp[10]);
        if (vc < mp[11] * cc) continue;
      }
      const float gq[4] = {mp[3], mp[4], mp[5], mp[6]};
      float dif[3] = {p[0] - mp[0], p[1] - mp[1], p[2] - mp[2]}, lp[3], lv[3];
      qrot_inv(gq, dif, lp);
      qrot_inv(gq, v, lv);
      {  // bounding sphere (centre mg[8..10], radius mg[11], mesh frame)
        float dc[3] = {lp[0] - mg[8], lp[1] - mg[9], lp[2] - mg[10]};
        const float bb = dot3(dc, lv), cc = dot3(dc, dc) - mg[11] * mg[11];
        if (cc > 0.f && (bb > 0.f || bb * bb - cc < 0.f)) continue;
        const float tnear = -bb - sqrtf(fmaxf(bb * bb - cc, 0.f));
        if (*best >= 0.f && tnear > *best) continue;
      }
      const float iv[3] = {1.f / lv[0], 1.f / lv[1], 1.f / lv[2]};
      int stack[32], sp = 0;  // (median-split BVH depth: about log2(triangles / LEAF) + 1 < 20)
      stack[sp++] = S.mroot[g];
      float hit = -1.f, hn[3] = {0.f, 0.f, 0.f};
      while (sp) {
        const int nd = stack[--sp];
        const float* nb = S.nodeb + 6 * nd;
        float tmin = 0.f, tmax = *best >= 0.f ? *best : TR_INF;
        if (hit >= 0.f) tmax = fminf(tmax, hit);
        for (int k = 0; k < 3; k++) {
          float a = (nb[k] - lp[k]) * iv[k], b = (nb[3 + k] - lp[k]) * iv[k];
          if (a > b) { const float t = a; a = b; b = t; }
          tmin = fmaxf(tmin, a); tmax = fminf(tmax, b);
        }
        if (tmin > tmax) continue;
        const int l = S.nodei[2 * nd], rr = S.nodei[2 * nd + 1];
        if (l >= 0) {  // push the farther child first (by the children's box centres along the ray)
          const float *bl = S.nodeb + 6 * l, *br = S.nodeb + 6 * rr;
          float dl = 0.f, dr = 0.f;
          for (int k = 0; k < 3; k++) { dl += (bl[k] + bl[3 + k]) * lv[k]; dr += (br[k] + br[3 + k]) * lv[k]; }
          if (sp + 2 <= 32) {
            if (dl <= dr) { stack[sp++] = rr; stack[sp++] = l; } else { stack[sp++] = l; stack[sp++] = rr; }
          }
          continue;
        }
        for (int i = -l - 1, e = -l - 1 + rr; i < e; i++) {
          const float* T = S.tri + 9 * i;
          float e1[3], e2[3], pv[3], tv[3], qv[3];
          for (int k = 0; k < 3; k++) { e1[k] = T[3 + k] - T[k]; e2[k] = T[6 + k] - T[k]; }
          cross(lv, e2, pv);
          const float det = dot3(e1, pv);
          if (fabsf(det) < 1e-12f) continue;
          const float id = 1.f / det;
          for (int k = 0; k < 3; k++) tv[k] = lp[k] - T[k];
          const float u = dot3(tv, pv) * id;
          if (u < 0.f || u > 1.f) continue;
          cross(tv, e1, qv);
          const float vv = dot3(lv, qv) * id;
          if (vv < 0.f || u + vv > 1.f) continue;
          const float tt = dot3(e2, qv) * id;
          if (tt >= 0.f && (hit < 0.f || tt < hit) && (*best < 0.f || tt < *best)) { hit = tt; cross(e1, e2, hn); }
        }
      }
      if (hit >= 0.f) {
        float nw[3];
        qrot(gq, hn, nw);
        const float il = rsqrtf(fmaxf(dot3(nw, nw), 1e-30f)), sgn = dot3(nw, v) > 0.f ? -1.f : 1.f;
        for (int k = 0; k < 3; k++) nw[k] *= il * sgn;
        take(hit, ID_ROBOT + g, nw, best, bid, bn);
      }
    }
  }
}

__device__ __forceinline__ void world_ray(int w, int n, const float* pos, const float* rot, const float* off,
                                          const float* dir, float* p, float* v) {
  const float *P = pos + 3 * w, *R = rot + 9 * w;
  float t[3];
  mv3(R, off + 3 * n, t);
  for (int k = 0; k < 3; k++) p[k] = P[k] + t[k];
  mv3(R, dir + 3 * n, v);
}

// pass 1: the terrain, the floor and the objects (dist -1 for none or beyond maxd)
__global__ void __launch_bounds__(RAY_BLOCK, STATIC_MINB) k_rays_static(Scene S, int B, int N, const float* pos, const float* rot, const float* off,
                              const float* dir, float maxd, float* dist, int* oid, float* onrm, int dpw) {
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (long long)B * N) return;
  const int w = int(tid / N), n = int(tid % N);
  if (dpw) { off += (size_t)w * N * 3; dir += (size_t)w * N * 3; }  // per-world rays (B, N, 3)
  float p[3], v[3];
  world_ray(w, n, pos, rot, off, dir, p, v);
  float best = -1.f, bn[3] = {0.f, 0.f, 0.f};
  int bid = -1;
  RSTAT(4, 1);
  if (S.grid) cast_terrain(S, p, v, maxd, &best, &bid, bn);
  if (S.has_floor && fabsf(v[2]) > TR_MINVAL) {
    const float d = (S.floor_z - p[2]) / v[2], nf[3] = {0.f, 0.f, 1.f};
    take(d, ID_FLOOR, nf, &best, &bid, bn);
  }
  for (int k = 0; k < S.nobj; k++) {
    const float* b = S.obj + ((size_t)w * S.nobj + k) * 15;
    if (!(b[12] > 0.f)) continue;
    float all[6], nn[3];
    take(tr_box(b, b + 3, b + 12, p, v, all, nn), ID_OBJECT + k, nn, &best, &bid, bn);
  }
  if (best > maxd) { best = -1.f; bid = -1; }
  dist[tid] = best;
  oid[tid] = bid;
  if (onrm) for (int k = 0; k < 3; k++) onrm[3 * tid + k] = best >= 0.f ? bn[k] : 0.f;
}

// pass 2: the robot's meshes nearer than pass 1's hit (or maxd)
__global__ void __launch_bounds__(RAY_BLOCK, ROBOT_MINB) k_rays_robot(Scene S, int B, int N, const float* pos, const float* rot, const float* off,
                             const float* dir, float maxd, float* dist, int* oid, float* onrm, int dpw) {
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (long long)B * N) return;
  const int w = int(tid / N), n = int(tid % N);
  if (!S.mmask[2 * w] && !S.mmask[2 * w + 1]) return;
  if (dpw) { off += (size_t)w * N * 3; dir += (size_t)w * N * 3; }
  float p[3], v[3];
  world_ray(w, n, pos, rot, off, dir, p, v);
  const float d0 = dist[tid];
  float best = d0 >= 0.f ? d0 : maxd, bn[3] = {0.f, 0.f, 0.f};
  int bid = d0 >= 0.f ? oid[tid] : -1;
  cast_robot(S, w, p, v, &best, &bid, bn);
  if (bid >= ID_ROBOT && bid != (d0 >= 0.f ? oid[tid] : -1)) {
    dist[tid] = best;
    oid[tid] = bid;
    if (onrm) for (int k = 0; k < 3; k++) onrm[3 * tid + k] = bn[k];
  }
}

// cone: cos and sin of the half angle of a cone from the sensor origin along its x axis that holds every ray (the mesh
// culling; offsets must be small against the meshes' distances, as a camera's zeros); dpw: off / dir are per world
// (B, N, 3) instead of shared (N, 3)
extern "C" int dk_scene_rays_pw(cudaStream_t st, const Scene* S, int B, int N, const float* pos, const float* rot,
                                const float* off, const float* dir, float maxd, float ch, float sh, float offmax,
                                float* dist, int* id, float* nrm, int dpw) {
  const long long n = (long long)B * N;
  if (n == 0) return 0;
  if (S->nmesh) {
    if (S->nmesh > 64) return -1;
    cudaMemsetAsync(S->mmask, 0, sizeof(unsigned) * 2 * B, st);
    k_mesh_cull<<<(B * S->nmesh + 127) / 128, 128, 0, st>>>(*S, B, pos, rot, ch, sh, maxd, offmax);
  }
  const unsigned nb = (unsigned)((n + RAY_BLOCK - 1) / RAY_BLOCK);
  k_rays_static<<<nb, RAY_BLOCK, 0, st>>>(*S, B, N, pos, rot, off, dir, maxd, dist, id, nrm, dpw);
  if (S->nmesh) k_rays_robot<<<nb, RAY_BLOCK, 0, st>>>(*S, B, N, pos, rot, off, dir, maxd, dist, id, nrm, dpw);
  return (int)cudaGetLastError();
}
extern "C" int dk_scene_rays(cudaStream_t st, const Scene* S, int B, int N, const float* pos, const float* rot,
                             const float* off, const float* dir, float maxd, float ch, float sh, float offmax,
                             float* dist, int* id, float* nrm) {
  return dk_scene_rays_pw(st, S, B, N, pos, rot, off, dir, maxd, ch, sh, offmax, dist, id, nrm, 0);
}
extern "C" int dk_scene_size() { return (int)sizeof(Scene); }
"""


FFI_SOURCE = r"""
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;
#define DK_MAX_DEV 16
static Scene g_scene[DK_MAX_DEV];
static bool g_have[DK_MAX_DEV];
extern "C" int dk_rays_set_scene(int dev, const Scene* s) {
  if (dev < 0 || dev >= DK_MAX_DEV) return -1;
  g_scene[dev] = *s;
  g_have[dev] = true;
  return 0;
}
static ffi::Error bad(const char* m) { return ffi::Error(ffi::ErrorCode::kInvalidArgument, m); }
using BufF = ffi::Buffer<ffi::F32>;
using RetF = ffi::Result<ffi::Buffer<ffi::F32>>;
using RetI = ffi::Result<ffi::Buffer<ffi::S32>>;
static ffi::Error scene_rays_impl(cudaStream_t st, BufF pos, BufF rot, BufF off, BufF dir, BufF xpos, BufF xquat,
                                  BufF obj, float maxd, float ch, float sh, float offmax, int32_t robot,
                                  int32_t exclude_body,
                                  RetF dist, RetI id, RetF nrm, RetF mpose, RetI mmask) {
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= DK_MAX_DEV || !g_have[dev]) return bad("no scene on this device (jax_rays.setup)");
  Scene S = g_scene[dev];
  if (!robot) S.nmesh = 0;  // (the height scan: the terrain alone)
  S.exclude_body = exclude_body;  // (a LiDAR: its own body's meshes)
  const auto dp = pos.dimensions(), dd = off.dimensions(), dx = xpos.dimensions(), dob = obj.dimensions();
  const auto dr = dir.dimensions();
  const int dpw = dd.size() == 3;  // per-world rays: off / dir (B, N, 3)
  if (dp.size() != 2 || dp[1] != 3 || dd.size() < 2 || dd.size() > 3 || dd[dd.size() - 1] != 3 ||
      dr.size() != dd.size() || (dpw && dd[0] != dp[0]))
    return bad("pos (B, 3), off / dir (N, 3) or (B, N, 3)");
  for (size_t k = 0; k < dd.size(); k++)
    if (dr[k] != dd[k]) return bad("off and dir: the same shape");
  const int64_t B = dp[0], N = dd[dd.size() - 2];
  if (B * N > INT_MAX) return bad("too many rays");
  if (dx.size() != 3 || dx[0] != B || dx[2] != 3 || dob.size() != 3 || dob[0] != B || dob[2] != 15)
    return bad("xpos (B, nbody, 3), objects (B, K, 15)");
  if (S.nmesh && dx[1] != S.nbody) return bad("xpos: the scene's body count");
  S.xpos = xpos.typed_data();
  S.xquat = xquat.typed_data();
  S.nobj = (int)dob[1];
  S.obj = dob[1] ? obj.typed_data() : nullptr;
  S.mpose = mpose->typed_data();
  S.mmask = reinterpret_cast<unsigned*>(mmask->typed_data());
  int rc = dk_scene_rays_pw(st, &S, (int)B, (int)N, pos.typed_data(), rot.typed_data(), off.typed_data(),
                            dir.typed_data(), maxd, ch, sh, offmax, dist->typed_data(), id->typed_data(),
                            nrm->typed_data(), dpw);
  return rc ? ffi::Error(ffi::ErrorCode::kInternal, "dk_scene_rays") : ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkSceneRays, scene_rays_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<BufF>().Arg<BufF>().Arg<BufF>().Arg<BufF>().Arg<BufF>().Arg<BufF>().Arg<BufF>()
                                  .Attr<float>("max_distance").Attr<float>("cone_cos").Attr<float>("cone_sin")
                                  .Attr<float>("offset_max")
                                  .Attr<int32_t>("robot").Attr<int32_t>("exclude_body")
                                  .Ret<BufF>().Ret<ffi::Buffer<ffi::S32>>().Ret<BufF>().Ret<BufF>()
                                  .Ret<ffi::Buffer<ffi::S32>>());
"""


def cache_dir():
    d = Path(os.environ.get("DK_CACHE_DIR") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
             / "nablaengine")
    d.mkdir(parents=True, exist_ok=True)
    return d


def build(arch, include=None):
    """The shared library (nvcc -shared, the CUDA runtime static) for architecture arch (sm_XY), cached by source;
    include: XLA's FFI headers (jax.ffi.include_dir()) for the JAX variant with the handler DkSceneRays."""
    from nablaengine import cuda
    nvcc = cuda.nvcc()
    flags = ["-O3", f"-arch={arch}", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-cudart", "static"]
    flags += os.environ.get("DK_RAYS_FLAGS", "").split()
    text = SOURCE if include is None else SOURCE.replace("#include <math.h>\n", "#include <math.h>\n#include <climits>\n") + FFI_SOURCE
    if include is not None:
        flags.append(f"-I{include}")
    key = hashlib.sha256((text + " ".join(flags) + nvcc).encode()).hexdigest()[:20]
    out = cache_dir() / f"libdk_rays{'_jax' if include else ''}_{arch}_{key}.so"
    if not out.exists():
        src = out.with_suffix(".cu")
        src.write_text(text)
        tmp = out.with_suffix(f".tmp{os.getpid()}.so")
        subprocess.run([nvcc, *flags, "-o", str(tmp), str(src)], check=True)
        os.replace(tmp, out)
    return out

"""The ray library's OptiX backend (RT cores): the same casts as raycore's software kernels,
with the hardware finding each ray's nearest primitive and the software's exact tests (raycore: MuJoCo Warp's ray_box /
_ray_triangle, the heightfield cells, the robot's Moller-Trumbore) recomputing its distance, id and normal in the
original frames.

OptiX's headers are NVIDIA's (proprietary / BSD-3 per file, the NVIDIA SDK license): they are not part of this
repository. $OPTIX_ROOT names an OptiX SDK (or NVIDIA/optix-dev) checkout whose include/ has optix.h (8.0 or later; the
driver must support its ABI). An optional backend: chosen by backend="optix" (or "auto" with $OPTIX_ROOT set);
the default is the software kernels, and without $OPTIX_ROOT nothing here is built."""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

from nablaengine.sensors import raycore

TERRAIN_WALL = -1

DEVICE_PROGRAMS = r"""
#include <optix.h>
struct Params {
  OptixTraversableHandle ias;
  int B, N;
  const float *pos, *rot, *off, *dir;
  const float* shift;  // (B, 3) each world's cell offset
  float maxd;
  unsigned mask;
  float* t;  // (B N) the nearest hit's distance (-1 none)
  int *inst, *prim;
};
extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__rays() {
  const uint3 ix = optixGetLaunchIndex();
  const int n = ix.x, w = ix.y;
  const float *P = params.pos + 3 * w, *R = params.rot + 9 * w, *o = params.off + 3 * n, *d = params.dir + 3 * n;
  const float* s = params.shift + 3 * w;
  float3 org, vec;
  org.x = P[0] + R[0] * o[0] + R[1] * o[1] + R[2] * o[2] + s[0];
  org.y = P[1] + R[3] * o[0] + R[4] * o[1] + R[5] * o[2] + s[1];
  org.z = P[2] + R[6] * o[0] + R[7] * o[1] + R[8] * o[2] + s[2];
  vec.x = R[0] * d[0] + R[1] * d[1] + R[2] * d[2];
  vec.y = R[3] * d[0] + R[4] * d[1] + R[5] * d[2];
  vec.z = R[6] * d[0] + R[7] * d[1] + R[8] * d[2];
  unsigned p0 = __float_as_uint(-1.f), p1 = 0xFFFFFFFFu, p2 = 0xFFFFFFFFu;
  optixTrace(params.ias, org, vec, 0.f, params.maxd, 0.f, (OptixVisibilityMask)params.mask,
             OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0, p1, p2);
  const size_t i = (size_t)w * params.N + n;
  params.t[i] = __uint_as_float(p0);
  params.inst[i] = (int)p1;
  params.prim[i] = (int)p2;
}
extern "C" __global__ void __closesthit__rays() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetInstanceId());
  optixSetPayload_2(optixGetPrimitiveIndex());
}
extern "C" __global__ void __miss__rays() {}
"""

HOST = r"""
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cstdio>
#include <cstring>
#include <vector>

struct XParams {
  OptixTraversableHandle ias;
  int B, N;
  const float *pos, *rot, *off, *dir;
  const float* shift;
  float maxd;
  unsigned mask;
  float* t;
  int *inst, *prim;
};

// the instances of one cast: world w's terrain (instance id 0 | w << 8 ... encoded below), its mesh geoms, its objects
// instance id = w * NI + k (k = 0 terrain, 1 .. nmesh meshes, then objects); mask: 1 terrain, 2 objects, 4 robot, 8 the
// excluded body's meshes
__global__ void k_instances(OptixInstance* out, int B, int nmesh, int nobj, const float* shift,
                            OptixTraversableHandle terrain, const OptixTraversableHandle* meshes,
                            OptixTraversableHandle box, const float* mgeom, const float* xpos, const float* xquat,
                            int nbody, const float* obj, int exclude_body) {
  const int NI = 1 + nmesh + nobj;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= B * NI) return;
  const int w = tid / NI, k = tid % NI;
  const float* s = shift + 3 * w;
  float M[12] = {1, 0, 0, s[0], 0, 1, 0, s[1], 0, 0, 1, s[2]};
  OptixInstance& I = out[tid];
  I.instanceId = (unsigned)tid;
  I.sbtOffset = 0;
  I.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
  I.visibilityMask = 1;
  if (k == 0) {
    I.traversableHandle = terrain;
    I.visibilityMask = terrain ? 1 : 0;
  } else if (k <= nmesh) {
    const int g = k - 1;
    const float* mg = mgeom + 12 * g;
    const int body = (int)mg[0];
    const float* bx = xpos + ((size_t)w * nbody + body) * 3;
    const float* bq = xquat + ((size_t)w * nbody + body) * 4;
    float gq[4], gp[3], t[3];
    qmul(bq, mg + 4, gq);
    qrot(bq, mg + 1, t);
    for (int c = 0; c < 3; c++) gp[c] = bx[c] + t[c];
    const float w0 = gq[0], x = gq[1], y = gq[2], z = gq[3];
    const float R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w0), 2 * (x * z + y * w0),
                        2 * (x * y + z * w0), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w0),
                        2 * (x * z - y * w0), 2 * (y * z + x * w0), 1 - 2 * (x * x + y * y)};
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) M[4 * r + c] = R[3 * r + c];
      M[4 * r + 3] = gp[r] + s[r];
    }
    I.traversableHandle = meshes[g];
    I.visibilityMask = body == exclude_body ? 8 : 4;
  } else {
    const float* b = obj + ((size_t)w * nobj + (k - 1 - nmesh)) * 15;
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) M[4 * r + c] = b[3 + 3 * r + c] * b[12 + c];
      M[4 * r + 3] = b[r] + s[r];
    }
    I.traversableHandle = box;
    I.visibilityMask = b[12] > 0.f ? 2 : 0;
  }
  for (int c = 0; c < 12; c++) I.transform[c] = M[c];
}

// exact recomputation of each ray's hit (the original frame, raycore's tests) -> dist, id, normal
__global__ void k_refine(Scene S, int B, int N, const float* pos, const float* rot, const float* off, const float* dir,
                         float maxd, int nmesh, int nobj, const float* t, const int* inst, const int* prim,
                         const int* tri_elem, const int* tri_cell, const int* mesh_first, float* dist, int* oid,
                         float* onrm) {
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (long long)B * N) return;
  const int w = int(tid / N), n = int(tid % N);
  float p[3], v[3];
  world_ray(w, n, pos, rot, off, dir, p, v);
  float best = -1.f, bn[3] = {0.f, 0.f, 0.f};
  int bid = -1;
  const int NI = 1 + nmesh + nobj, ii = inst[tid], pr = prim[tid];
  if (t[tid] >= 0.f && ii >= 0) {
    const int k = ii % NI;
    float nn[3], all[6];
    if (k == 0) {  // terrain
      const int e = tri_elem[pr], nbox = int(S.grid[5]);
      if (e < nbox) {
        const float* b = S.box + 15 * e;
        float d = tr_box(b, b + 3, b + 12, p, v, all, nn);
        if (d < 0.f) d = t[tid];  // (grazing an edge: the hardware's distance)
        take(d, ID_BOX + e, nn, &best, &bid, bn);
      } else {
        const int j = e - nbox, cell = tri_cell[pr];
        float d = -1.f;
        if (cell >= 0) d = hf_tri(S, j, cell >> 1, cell & 1, p, v, nn);
        if (d < 0.f) d = tr_hfield(S, j, p, v, nn);  // (walls, bottom; or the triangle's test missed by rounding)
        take(d, ID_HFIELD + j, nn, &best, &bid, bn);
      }
    } else if (k <= nmesh) {
      const int g = k - 1;
      const float* mg = S.mgeom + 12 * g;
      const int body = (int)mg[0];
      const float* bx = S.xpos + ((size_t)w * S.nbody + body) * 3;
      const float* bq = S.xquat + ((size_t)w * S.nbody + body) * 4;
      float gq[4], gp[3], tt[3], lp[3], lv[3], dif[3];
      qmul(bq, mg + 4, gq);
      qrot(bq, mg + 1, tt);
      for (int c = 0; c < 3; c++) { gp[c] = bx[c] + tt[c]; dif[c] = p[c] - gp[c]; }
      qrot_inv(gq, dif, lp);
      qrot_inv(gq, v, lv);
      const float* T = S.tri + 9 * ((size_t)mesh_first[g] + pr);
      float e1[3], e2[3], pv[3], tv[3], qv[3], hn[3];
      for (int c = 0; c < 3; c++) { e1[c] = T[3 + c] - T[c]; e2[c] = T[6 + c] - T[c]; }
      cross(lv, e2, pv);
      const float det = dot3(e1, pv), id = 1.f / det;
      for (int c = 0; c < 3; c++) tv[c] = lp[c] - T[c];
      cross(tv, e1, qv);
      float d = dot3(e2, qv) * id;
      if (!(d >= 0.f)) d = t[tid];  // (degenerate: the hardware's distance)
      cross(e1, e2, hn);
      float nw[3];
      qrot(gq, hn, nw);
      const float il = rsqrtf(fmaxf(dot3(nw, nw), 1e-30f)), sgn = dot3(nw, v) > 0.f ? -1.f : 1.f;
      for (int c = 0; c < 3; c++) nw[c] *= il * sgn;
      take(d, ID_ROBOT + g, nw, &best, &bid, bn);
    } else {
      const int kk = k - 1 - nmesh;
      const float* b = S.obj + ((size_t)w * nobj + kk) * 15;
      float d = tr_box(b, b + 3, b + 12, p, v, all, nn);
      if (d < 0.f) d = t[tid];
      take(d, ID_OBJECT + kk, nn, &best, &bid, bn);
    }
  }
  if (S.has_floor && fabsf(v[2]) > TR_MINVAL) {
    const float d = (S.floor_z - p[2]) / v[2], nf[3] = {0.f, 0.f, 1.f};
    take(d, ID_FLOOR, nf, &best, &bid, bn);
  }
  if (best > maxd) { best = -1.f; bid = -1; }
  dist[tid] = best;
  oid[tid] = best >= 0.f ? bid : -1;
  if (onrm) for (int c = 0; c < 3; c++) onrm[3 * tid + c] = best >= 0.f ? bn[c] : 0.f;
}

struct XState {
  OptixDeviceContext ctx = nullptr;
  OptixModule mod = nullptr;
  OptixPipeline pipe = nullptr;
  OptixProgramGroup pg[3] = {nullptr, nullptr, nullptr};
  OptixShaderBindingTable sbt = {};
  CUdeviceptr sbt_mem = 0, params = 0;
  OptixTraversableHandle terrain = 0, box = 0;
  std::vector<OptixTraversableHandle> meshes;
  OptixTraversableHandle* d_meshes = nullptr;
  std::vector<void*> keep;  // device buffers of the GASes
  void *inst_buf = nullptr, *ias_buf = nullptr, *tmp_buf = nullptr, *res_buf = nullptr;
  size_t inst_cap = 0, ias_cap = 0, tmp_cap = 0, res_cap = 0;
};
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRec { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

static void* dmalloc(size_t n) { void* p = nullptr; cudaMalloc(&p, n ? n : 1); return p; }
static void dgrow(void** p, size_t* cap, size_t need) {
  if (need <= *cap) return;
  if (*p) cudaFree(*p);
  *p = dmalloc(need);
  *cap = need;
}
#define OK(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) return (int)r_; } while (0)

// a triangle GAS over ntri triangles (9 floats each, device memory)
static int build_gas(XState* X, const float* tri, unsigned ntri, OptixTraversableHandle* out, cudaStream_t st) {
  if (!ntri) { *out = 0; return 0; }
  OptixBuildInput in = {};
  in.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  CUdeviceptr vb = (CUdeviceptr)tri;
  in.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  in.triangleArray.vertexStrideInBytes = 12;
  in.triangleArray.numVertices = 3 * ntri;
  in.triangleArray.vertexBuffers = &vb;
  unsigned flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
  in.triangleArray.flags = &flags;
  in.triangleArray.numSbtRecords = 1;
  OptixAccelBuildOptions opt = {};
  opt.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  opt.operation = OPTIX_BUILD_OPERATION_BUILD;
  OptixAccelBufferSizes sz;
  OK(optixAccelComputeMemoryUsage(X->ctx, &opt, &in, 1, &sz));
  void* tmp = dmalloc(sz.tempSizeInBytes);
  void* buf = dmalloc(sz.outputSizeInBytes);
  OK(optixAccelBuild(X->ctx, st, &opt, &in, 1, (CUdeviceptr)tmp, sz.tempSizeInBytes, (CUdeviceptr)buf,
                     sz.outputSizeInBytes, out, nullptr, 0));
  cudaStreamSynchronize(st);
  cudaFree(tmp);
  X->keep.push_back(buf);
  return 0;
}

extern "C" int dkx_create(const char* ptx, size_t ptx_len, void** handle);
extern "C" int dkx_create_on(int dev, const char* ptx, size_t ptx_len, void** handle) {
  cudaSetDevice(dev);
  return dkx_create(ptx, ptx_len, handle);
}
extern "C" int dkx_create(const char* ptx, size_t ptx_len, void** handle) {
  cudaFree(0);  // (the current device's primary context)
  if (optixInit() != OPTIX_SUCCESS) return -100;
  XState* X = new XState();
  OK(optixDeviceContextCreate(0, nullptr, &X->ctx));
  OptixModuleCompileOptions mco = {};
  OptixPipelineCompileOptions pco = {};
  pco.usesMotionBlur = 0;
  pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  pco.numPayloadValues = 3;
  pco.numAttributeValues = 2;
  pco.pipelineLaunchParamsVariableName = "params";
  pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
  char log[4096];
  size_t ll = sizeof log;
  OK(optixModuleCreate(X->ctx, &mco, &pco, ptx, ptx_len, log, &ll, &X->mod));
  OptixProgramGroupDesc d[3] = {};
  d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = X->mod; d[0].raygen.entryFunctionName = "__raygen__rays";
  d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = X->mod; d[1].miss.entryFunctionName = "__miss__rays";
  d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = X->mod;
  d[2].hitgroup.entryFunctionNameCH = "__closesthit__rays";
  OptixProgramGroupOptions po = {};
  ll = sizeof log;
  OK(optixProgramGroupCreate(X->ctx, d, 3, &po, log, &ll, X->pg));
  OptixPipelineLinkOptions plo = {};
  plo.maxTraceDepth = 1;
  ll = sizeof log;
  OK(optixPipelineCreate(X->ctx, &pco, &plo, X->pg, 3, log, &ll, &X->pipe));
  OK(optixPipelineSetStackSize(X->pipe, 0, 0, 1024, 2));
  SbtRec rec[3];
  for (int i = 0; i < 3; i++) OK(optixSbtRecordPackHeader(X->pg[i], &rec[i]));
  X->sbt_mem = (CUdeviceptr)dmalloc(sizeof rec);
  cudaMemcpy((void*)X->sbt_mem, rec, sizeof rec, cudaMemcpyHostToDevice);
  X->sbt.raygenRecord = X->sbt_mem;
  X->sbt.missRecordBase = X->sbt_mem + sizeof(SbtRec);
  X->sbt.missRecordStrideInBytes = sizeof(SbtRec);
  X->sbt.missRecordCount = 1;
  X->sbt.hitgroupRecordBase = X->sbt_mem + 2 * sizeof(SbtRec);
  X->sbt.hitgroupRecordStrideInBytes = sizeof(SbtRec);
  X->sbt.hitgroupRecordCount = 1;
  X->params = (CUdeviceptr)dmalloc(sizeof(XParams));
  *handle = X;
  return 0;
}

// the static geometry: terrain triangles, the robot's mesh geoms (their triangles at mesh offsets, device), a unit box
extern "C" int dkx_set_geometry(void* h, const float* terrain_tri, unsigned nterrain, const float* mesh_tri,
                                const int* mesh_first, const int* mesh_count, int nmesh, const float* box_tri);
extern "C" int dkx_set_geometry_on(int dev, void* h, const float* terrain_tri, unsigned nterrain, const float* mesh_tri,
                                   const int* mesh_first, const int* mesh_count, int nmesh, const float* box_tri) {
  cudaSetDevice(dev);
  return dkx_set_geometry(h, terrain_tri, nterrain, mesh_tri, mesh_first, mesh_count, nmesh, box_tri);
}
extern "C" int dkx_set_geometry(void* h, const float* terrain_tri, unsigned nterrain, const float* mesh_tri,
                                const int* mesh_first, const int* mesh_count, int nmesh, const float* box_tri) {
  XState* X = (XState*)h;
  cudaStream_t st = 0;
  int rc = build_gas(X, terrain_tri, nterrain, &X->terrain, st);
  if (rc) return rc;
  X->meshes.assign(nmesh, 0);
  for (int g = 0; g < nmesh; g++)
    if ((rc = build_gas(X, mesh_tri + 9 * (size_t)mesh_first[g], (unsigned)mesh_count[g], &X->meshes[g], st))) return rc;
  if ((rc = build_gas(X, box_tri, 12, &X->box, st))) return rc;
  if (X->d_meshes) cudaFree(X->d_meshes);
  X->d_meshes = (OptixTraversableHandle*)dmalloc(sizeof(OptixTraversableHandle) * (nmesh ? nmesh : 1));
  if (nmesh) cudaMemcpy(X->d_meshes, X->meshes.data(), sizeof(OptixTraversableHandle) * nmesh, cudaMemcpyHostToDevice);
  return 0;
}

// the work memory of a cast of B worlds with NI instances each: the instance array, the IAS build's temporary memory
// and the IAS (bytes; each rounded up to 256 for alignment)
static size_t up256(size_t x) { return (x + 255) & ~(size_t)255; }
extern "C" int dkx_work_sizes(void* h, int B, int NI, size_t* inst_bytes, size_t* tmp_bytes, size_t* ias_bytes) {
  XState* X = (XState*)h;
  OptixBuildInput in = {};
  in.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  in.instanceArray.numInstances = (unsigned)(B * NI);
  OptixAccelBuildOptions opt = {};
  opt.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
  opt.operation = OPTIX_BUILD_OPERATION_BUILD;
  OptixAccelBufferSizes sz;
  OK(optixAccelComputeMemoryUsage(X->ctx, &opt, &in, 1, &sz));
  *inst_bytes = up256(sizeof(OptixInstance) * (size_t)B * NI);
  *tmp_bytes = up256(sz.tempSizeInBytes);
  *ias_bytes = up256(sz.outputSizeInBytes);
  return 0;
}

// one cast: instances, the IAS, the trace, the refinement. work: t (B N float), inst, prim (B N int); ext: null (the
// library's own buffers, cudaMalloc) or the three work buffers of dkx_work_sizes in one allocation (the caller's, e.g.
// XLA's scratch memory)
extern "C" int dkx_cast_ext(void* h, cudaStream_t st, const Scene* S, int B, int N, const float* pos, const float* rot,
                            const float* off, const float* dir, float maxd, const float* shift, unsigned mask,
                            const int* tri_elem, const int* tri_cell, const int* mesh_first, float* t, int* inst,
                            int* prim, float* dist, int* id, float* nrm, void* ext) {
  XState* X = (XState*)h;
  const long long n = (long long)B * N;
  if (n == 0) return 0;
  const int nmesh = S->nmesh, nobj = S->nobj, NI = 1 + nmesh + nobj;
  size_t ib, tb, ob;
  int rc = dkx_work_sizes(h, B, NI, &ib, &tb, &ob);
  if (rc) return rc;
  void *ibuf, *tbuf, *obuf;
  if (ext) {
    ibuf = ext; tbuf = (char*)ext + ib; obuf = (char*)ext + ib + tb;
  } else {
    dgrow(&X->inst_buf, &X->inst_cap, ib);
    dgrow(&X->tmp_buf, &X->tmp_cap, tb);
    dgrow(&X->ias_buf, &X->ias_cap, ob);
    ibuf = X->inst_buf; tbuf = X->tmp_buf; obuf = X->ias_buf;
  }
  k_instances<<<(B * NI + 127) / 128, 128, 0, st>>>((OptixInstance*)ibuf, B, nmesh, nobj, shift, X->terrain,
                                                    X->d_meshes, X->box, S->mgeom, S->xpos, S->xquat, S->nbody, S->obj,
                                                    S->exclude_body);
  OptixBuildInput in = {};
  in.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  in.instanceArray.instances = (CUdeviceptr)ibuf;
  in.instanceArray.numInstances = (unsigned)(B * NI);
  OptixAccelBuildOptions opt = {};
  opt.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
  opt.operation = OPTIX_BUILD_OPERATION_BUILD;
  XParams P = {};
  { OptixResult r_ = optixAccelBuild(X->ctx, st, &opt, &in, 1, (CUdeviceptr)tbuf, tb, (CUdeviceptr)obuf, ob, &P.ias,
                                     nullptr, 0);
    if (r_ != OPTIX_SUCCESS) {
      size_t fr = 0, tot = 0;
      cudaMemGetInfo(&fr, &tot);
      fprintf(stderr, "nablaengine optix: optixAccelBuild failed (%d), %zu of %zu bytes of device memory free (OptiX "
              "needs some outside the framework's preallocation, e.g. XLA_PYTHON_CLIENT_MEM_FRACTION)\n", (int)r_, fr, tot);
      return 100000 + (int)r_;
    } }
  P.B = B; P.N = N; P.pos = pos; P.rot = rot; P.off = off; P.dir = dir; P.shift = shift; P.maxd = maxd; P.mask = mask;
  P.t = t; P.inst = inst; P.prim = prim;
  cudaMemcpyAsync((void*)X->params, &P, sizeof P, cudaMemcpyHostToDevice, st);
  { OptixResult r_ = optixLaunch(X->pipe, st, X->params, sizeof(XParams), &X->sbt, (unsigned)N, (unsigned)B, 1);
    if (r_ != OPTIX_SUCCESS) return 200000 + (int)r_; }
  k_refine<<<(unsigned)((n + 127) / 128), 128, 0, st>>>(*S, B, N, pos, rot, off, dir, maxd, nmesh, nobj, t, inst, prim,
                                                        tri_elem, tri_cell, mesh_first, dist, id, nrm);
  return (int)cudaGetLastError();
}
extern "C" int dkx_cast(void* h, cudaStream_t st, const Scene* S, int B, int N, const float* pos, const float* rot,
                        const float* off, const float* dir, float maxd, const float* shift, unsigned mask,
                        const int* tri_elem, const int* tri_cell, const int* mesh_first, float* t, int* inst,
                        int* prim, float* dist, int* id, float* nrm) {
  return dkx_cast_ext(h, st, S, B, N, pos, rot, off, dir, maxd, shift, mask, tri_elem, tri_cell, mesh_first, t, inst,
                      prim, dist, id, nrm, nullptr);
}
"""


def optix_include():
    """The OptiX include directory ($OPTIX_ROOT/include), or None."""
    root = os.environ.get("OPTIX_ROOT")
    if root and (Path(root) / "include" / "optix.h").exists():
        return str(Path(root) / "include")
    return None


def available():
    return optix_include() is not None


def build(arch, xla_include=None):
    """(the host library .so, the device programs' PTX) for arch, cached; xla_include: XLA's FFI headers for the JAX
    variant (the handler DkOptixRays)."""
    from nablaengine import cuda
    inc = optix_include()
    if inc is None:
        raise RuntimeError("OptiX not available: set $OPTIX_ROOT to an OptiX SDK (include/optix.h)")
    nvcc = cuda.nvcc()
    core = raycore.SOURCE.replace("extern \"C\" int dk_scene_rays(", "extern \"C\" int dk_scene_rays_sw(")
    host = core + HF_TRI + HOST
    if xla_include is not None:
        host = host.replace("#include <math.h>\n", "#include <math.h>\n#include <climits>\n", 1) + FFI_SOURCE
    key = hashlib.sha256((host + DEVICE_PROGRAMS + arch + nvcc + inc + str(xla_include)).encode()).hexdigest()[:20]
    d = raycore.cache_dir()
    lib, ptx = d / f"libdk_optix{'_jax' if xla_include else ''}_{arch}_{key}.so", d / f"dk_optix_{arch}_{key}.ptx"
    if not lib.exists():
        src = lib.with_suffix(".cu")
        src.write_text(host)
        tmp = lib.with_suffix(f".tmp{os.getpid()}.so")
        subprocess.run([nvcc, "-O3", f"-arch={arch}", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-cudart",
                        "static", f"-I{inc}"] + ([f"-I{xla_include}"] if xla_include else [])
                       + ["-o", str(tmp), str(src), "-ldl"], check=True)
        os.replace(tmp, lib)
    if not ptx.exists():
        src = ptx.with_suffix(".prog.cu")
        src.write_text(DEVICE_PROGRAMS)
        virt = "compute_" + arch.split("_")[1]
        subprocess.run([nvcc, "-O3", f"-arch={virt}", "-std=c++17", "-ptx", f"-I{inc}", "--use_fast_math",
                        "-o", str(ptx), str(src)], check=True)
    return lib, ptx


HF_TRI = r"""
__device__ float hf_tri(const Scene& S, int j, int cell, int k, const float* pnt, const float* vec, float* nrm) {
  const float* hh = S.hfh + 16 * j;
  const int* hi = S.hfi + 4 * j;
  const float *pos = hh, *R = hh + 3, *size = hh + 12, *hd = S.hfd + hi[2];
  const int nrow = hi[0], ncol = hi[1], r = cell / ncol, c = cell % ncol;
  float dif[3] = {pnt[0] - pos[0], pnt[1] - pos[1], pnt[2] - pos[2]}, lp[3], lv[3];
  mtv3(R, dif, lp);
  mtv3(R, vec, lv);
  const float sg = lv[2] >= 0.f ? 1.f : -1.f, a = -1.f / (sg + lv[2]), b = lv[0] * lv[1] * a;
  const float b0[3] = {1.f + sg * lv[0] * lv[0] * a, sg * b, -sg * lv[0]}, b1[3] = {b, sg + lv[1] * lv[1] * a, -lv[1]};
  const float dx = tr_div(2.f * size[0], float(ncol - 1)), dy = tr_div(2.f * size[1], float(nrow - 1));
  const float x0 = dx * float(c) - size[0], x1 = dx * float(c + 1) - size[0];
  const float y0 = dy * float(r) - size[1], y1 = dy * float(r + 1) - size[1];
  const float h00 = hd[r * ncol + c] * size[2], h10 = hd[r * ncol + c + 1] * size[2];
  const float h11 = hd[(r + 1) * ncol + c + 1] * size[2], h01 = hd[(r + 1) * ncol + c] * size[2];
  const float va[3] = {x0, y0, h00}, vb[3] = {x1, y0, h10}, vc[3] = {x1, y1, h11}, vd[3] = {x0, y1, h01};
  float nl[3];
  const float d = k == 0 ? tr_tri(va, vb, vc, lp, lv, b0, b1, nl) : tr_tri(va, vc, vd, lp, lv, b0, b1, nl);
  if (d >= 0.f) mv3(R, nl, nrm);
  return d;
}
"""


def terrain_triangles(terrain):
    """(triangles (T, 9) world float32, element (T,) int32: box e or nbox + heightfield j, cell (T,) int32: 2 (r ncol + c)
    + k for a heightfield cell's triangle k, TERRAIN_WALL for its side walls and bottom, -1 for a box)."""
    tris, elem, cell = [], [], []
    nbox = len(terrain.boxes)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64)
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6),
             (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    for e, b in enumerate(terrain.boxes):
        R, h = np.asarray(b[3:12]).reshape(3, 3), np.asarray(b[12:15])
        v = (corners * h) @ R.T + b[:3]
        for f in faces:
            tris.append(v[list(f)].reshape(-1))
            elem.append(e)
            cell.append(-1)
    for j, (hh, hi) in enumerate(zip(terrain.hf_head, terrain.hf_int)):
        pos, R, size = np.asarray(hh[:3]), np.asarray(hh[3:12]).reshape(3, 3), np.asarray(hh[12:16])
        nrow, ncol, adr = int(hi[0]), int(hi[1]), int(hi[2])
        h = np.asarray(terrain.hf_data[adr:adr + nrow * ncol]).reshape(nrow, ncol) * size[2]
        xs = np.linspace(-size[0], size[0], ncol)
        ys = np.linspace(-size[1], size[1], nrow)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        V = np.stack([gx, gy, h], -1)
        va, vb, vc, vd = V[:-1, :-1], V[:-1, 1:], V[1:, 1:], V[1:, :-1]
        r, c = np.meshgrid(np.arange(nrow - 1), np.arange(ncol - 1), indexing="ij")
        code = (r * ncol + c) * 2
        t0 = np.concatenate([va, vb, vc], -1).reshape(-1, 9)
        t1 = np.concatenate([va, vc, vd], -1).reshape(-1, 9)
        loc = np.concatenate([t0, t1])
        codes = np.concatenate([code.reshape(-1), code.reshape(-1) + 1])
        walls = []
        bot = -size[3]
        for line in (V[0], V[-1], V[:, 0], V[:, -1]):
            for a, b2 in zip(line[:-1], line[1:]):
                a0, b0 = a.copy(), b2.copy()
                a0[2] = b0[2] = bot
                walls += [np.concatenate([a0, b0, b2]), np.concatenate([a0, b2, a])]
        x0, x1, y0, y1 = -size[0], size[0], -size[1], size[1]
        walls += [np.array([x0, y0, bot, x1, y0, bot, x1, y1, bot]), np.array([x0, y0, bot, x1, y1, bot, x0, y1, bot])]
        loc = np.concatenate([loc, np.asarray(walls)])
        codes = np.concatenate([codes, np.full(len(walls), TERRAIN_WALL)])
        wv = loc.reshape(-1, 3) @ R.T + pos
        tris.append(wv.reshape(-1, 9))
        elem.append(np.full(len(loc), nbox + j))
        cell.append(codes)
    tri = np.concatenate([np.asarray(x, np.float64).reshape(-1, 9) for x in tris]) if tris else np.zeros((0, 9))
    el = np.concatenate([np.atleast_1d(np.asarray(x)) for x in elem]) if elem else np.zeros(0)
    ce = np.concatenate([np.atleast_1d(np.asarray(x)) for x in cell]) if cell else np.zeros(0)
    return tri.astype(np.float32), el.astype(np.int32), ce.astype(np.int32)


UNIT_BOX = None


def unit_box():
    global UNIT_BOX
    if UNIT_BOX is None:
        from nablaengine.terrain import Terrain
        t, _, _ = terrain_triangles(Terrain(boxes=np.concatenate([np.zeros(3), np.eye(3).ravel(), np.ones(3)])[None]))
        UNIT_BOX = t
    return UNIT_BOX


class OptixSceneRays:
    """SceneRays (torch_rays) on the OptiX backend: the same constructor arguments and cast()."""

    def __init__(self, device="cuda", terrain=None, cell=1.0, floor_z=None, robot=None, nbody=0, exclude_body=-1):
        import torch
        from nablaengine.sensors.torch_rays import SceneRays
        self.torch, self.device = torch, torch.device(device)
        self.sw = SceneRays(device, terrain=terrain, cell=cell, floor_z=floor_z, robot=robot, nbody=nbody,
                            exclude_body=exclude_body)
        cc = torch.cuda.get_device_capability(self.device)
        libp, ptx = build(f"sm_{cc[0]}{cc[1]}")
        lib = ctypes.cdll.LoadLibrary(str(libp))
        vp, ci, cf, cu = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_uint
        lib.dkx_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(vp)]
        lib.dkx_set_geometry.argtypes = [vp, vp, cu, vp, vp, vp, ci, vp]
        lib.dkx_cast.argtypes = [vp, vp, vp, ci, ci, vp, vp, vp, vp, cf, vp, cu, vp, vp, vp, vp, vp, vp, vp, vp, vp]
        self.lib = lib
        h = vp()
        code = Path(ptx).read_bytes()
        with torch.cuda.device(self.device):
            rc = lib.dkx_create(code, len(code), ctypes.byref(h))
        if rc:
            raise RuntimeError(f"dkx_create failed ({rc})")
        self.h = h
        t = lambda x, dt: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=self.device)
        if terrain is not None:
            tri, el, ce = terrain_triangles(terrain)
            lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
        else:
            tri, el, ce = np.zeros((0, 9), np.float32), np.zeros(1, np.int32), np.zeros(1, np.int32)
            lo, hi = np.zeros(3), np.zeros(3)
        self.extent = (lo, hi)
        self.ttri, self.tel, self.tce = t(tri if len(tri) else np.zeros((1, 9)), torch.float32), t(el, torch.int32), t(ce, torch.int32)
        if robot is not None:
            first, count = self._mesh_ranges(robot)
        else:
            first, count = np.zeros(1, np.int32), np.zeros(1, np.int32)
        self.mfirst, self.mcount = t(first, torch.int32), t(count, torch.int32)
        self.box = t(unit_box(), torch.float32)
        mt = self.sw.scene.tri if robot is not None else None
        rc = lib.dkx_set_geometry(h, self.ttri.data_ptr(), len(tri), mt, first.ctypes.data, count.ctypes.data,
                                  len(first) if robot is not None else 0, self.box.data_ptr())
        if rc:
            raise RuntimeError(f"dkx_set_geometry failed ({rc})")
        self._shift = {}

    @staticmethod
    def _mesh_ranges(robot):
        roots = list(robot["mroot"]) + [len(robot["nodei"])]
        first, count = [], []
        for g in range(len(robot["mroot"])):
            ni = robot["nodei"][roots[g]:roots[g + 1]]
            leaf = ni[:, 0] < 0
            starts = -ni[leaf, 0] - 1
            first.append(int(starts.min()))
            count.append(int(ni[leaf, 1].sum()))
        return np.asarray(first, np.int32), np.asarray(count, np.int32)

    def shift(self, B, maxd):
        """(B, 3) world cell offsets: a 3-D grid of cells larger than the terrain's extent plus twice the ray length
        (and 4 m of robot), centred at the origin."""
        key = (B, round(float(maxd), 3))
        if key not in self._shift:
            lo, hi = self.extent
            size = (hi - lo) + 2 * float(maxd) + 4.0
            n = int(np.ceil(B ** (1 / 3)))
            i = np.arange(B)
            g = np.stack([i % n, (i // n) % n, i // (n * n)], 1) - (n - 1) / 2
            self._shift[key] = self.torch.as_tensor((g * size).astype(np.float32), device=self.device)
        return self._shift[key]

    def cast(self, pos, rot, off, dirs, max_distance=10.0, xpos=None, xquat=None, objects=None, normals=False,
             robot=True, exclude_body=None):
        torch = self.torch
        from nablaengine.sensors.torch_rays import _Scene
        B, N = pos.shape[0], off.shape[0]
        f = lambda x: x.to(self.device, torch.float32).contiguous()
        pos, rot, off, dirs = f(pos), f(rot).reshape(B, 9), f(off), f(dirs)
        s = _Scene.from_buffer_copy(self.sw.scene)
        keep = []
        if s.nmesh and robot:
            xpos, xquat = f(xpos), f(xquat)
            keep += [xpos, xquat]
            s.xpos, s.xquat = xpos.data_ptr(), xquat.data_ptr()
        else:
            s.nmesh = 0
        if exclude_body is not None:
            s.exclude_body = int(exclude_body)
        if objects is not None:
            objects = f(objects)
            keep.append(objects)
            s.nobj, s.obj = objects.shape[1], objects.data_ptr()
        else:
            s.nobj = 0
        t = torch.empty(B, N, device=self.device)
        inst = torch.empty(B, N, dtype=torch.int32, device=self.device)
        prim = torch.empty(B, N, dtype=torch.int32, device=self.device)
        dist = torch.empty(B, N, device=self.device)
        gid = torch.empty(B, N, dtype=torch.int32, device=self.device)
        nrm = torch.empty(B, N, 3, device=self.device) if normals else None
        st = torch.cuda.current_stream(self.device).cuda_stream
        mask = 1 | 2 | (4 if s.nmesh else 0)
        sh = self.shift(B, max_distance)
        rc = self.lib.dkx_cast(self.h, st, ctypes.byref(s), B, N, pos.data_ptr(), rot.data_ptr(), off.data_ptr(),
                               dirs.data_ptr(), float(max_distance), sh.data_ptr(), mask, self.tel.data_ptr(),
                               self.tce.data_ptr(), self.mfirst.data_ptr(), t.data_ptr(), inst.data_ptr(),
                               prim.data_ptr(), dist.data_ptr(), gid.data_ptr(), nrm.data_ptr() if normals else None)
        if rc:
            raise RuntimeError(f"dkx_cast failed ({rc})")
        return (dist, gid, nrm) if normals else (dist, gid)


FFI_SOURCE = r"""
#include "xla/ffi/api/ffi.h"
#include <cstdio>
#include <cstdlib>
namespace ffi = xla::ffi;
#define DKX_MAX_DEV 16
struct XJax { void* h; Scene S; const int *tri_elem, *tri_cell, *mesh_first; bool have; };
static XJax g_x[DKX_MAX_DEV];
extern "C" int dkx_jax_setup(int dev, void* h, const Scene* S, const int* tri_elem, const int* tri_cell,
                             const int* mesh_first) {
  if (dev < 0 || dev >= DKX_MAX_DEV) return -1;
  g_x[dev] = XJax{h, *S, tri_elem, tri_cell, mesh_first, true};
  return 0;
}
using XBufF = ffi::Buffer<ffi::F32>;
using XRetF = ffi::Result<ffi::Buffer<ffi::F32>>;
using XRetI = ffi::Result<ffi::Buffer<ffi::S32>>;
static ffi::Error xbad(const char* m) { return ffi::Error(ffi::ErrorCode::kInvalidArgument, m); }
static ffi::Error optix_rays_impl(cudaStream_t st, ffi::ScratchAllocator scratch, XBufF pos, XBufF rot, XBufF off, XBufF dir, XBufF xpos, XBufF xquat,
                                  XBufF obj, XBufF shift, float maxd, int32_t robot, int32_t exclude_body, XRetF dist,
                                  XRetI id, XRetF nrm, XRetF t, XRetI inst, XRetI prim) {
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= DKX_MAX_DEV || !g_x[dev].have) return xbad("no OptiX scene on this device (jax_rays.setup)");
  XJax& X = g_x[dev];
  Scene S = X.S;
  const auto dp = pos.dimensions(), dd = off.dimensions(), dob = obj.dimensions(), dx = xpos.dimensions();
  if (dp.size() != 2 || dp[1] != 3 || dd.size() != 2 || dd[1] != 3) return xbad("pos (B, 3), off (N, 3)");
  const int64_t B = dp[0], N = dd[0];
  if (B * N > INT_MAX) return xbad("too many rays");
  if (dob.size() != 3 || dob[0] != B || dob[2] != 15 || dx.size() != 3 || dx[0] != B) return xbad("objects, xpos");
  if (!robot) S.nmesh = 0;
  S.exclude_body = exclude_body;
  S.xpos = xpos.typed_data();
  S.xquat = xquat.typed_data();
  S.nobj = (int)dob[1];
  S.obj = dob[1] ? obj.typed_data() : nullptr;
  const unsigned mask = 1u | 2u | (S.nmesh ? 4u : 0u);
  size_t ib, tb, ob;  // the IAS work memory from XLA's allocator (no cudaMalloc beside XLA's preallocation)
  int rc = dkx_work_sizes(X.h, (int)B, 1 + S.nmesh + S.nobj, &ib, &tb, &ob);
  if (rc) return xbad("dkx_work_sizes");
  auto mem = scratch.Allocate(ib + tb + ob + 256, 16);  // (XLA's scratch allocator aligns to at most 16 bytes; OptiX's
  if (!mem.has_value()) return ffi::Error(ffi::ErrorCode::kResourceExhausted, "OptiX work memory");  // need 128)
  void* aligned = (void*)(((uintptr_t)*mem + 255) & ~(uintptr_t)255);
  rc = dkx_cast_ext(X.h, st, &S, (int)B, (int)N, pos.typed_data(), rot.typed_data(), off.typed_data(), dir.typed_data(),
                    maxd, shift.typed_data(), mask, X.tri_elem, X.tri_cell, X.mesh_first, t->typed_data(),
                    inst->typed_data(), prim->typed_data(), dist->typed_data(), id->typed_data(), nrm->typed_data(),
                    aligned);
  if (rc) {
    static char msg[128];
    snprintf(msg, sizeof msg, "dkx_cast failed (%d: OptiX result or cudaError)", rc);
    return ffi::Error(ffi::ErrorCode::kInternal, msg);
  }
  return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkOptixRays, optix_rays_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
                                  .Arg<XBufF>().Arg<XBufF>().Arg<XBufF>().Arg<XBufF>().Arg<XBufF>().Arg<XBufF>()
                                  .Arg<XBufF>().Arg<XBufF>().Attr<float>("max_distance").Attr<int32_t>("robot")
                                  .Attr<int32_t>("exclude_body").Ret<XBufF>().Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<XBufF>().Ret<XBufF>().Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>());
"""

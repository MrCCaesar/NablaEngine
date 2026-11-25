"""The G1 engine as a differentiable JAX op: one control step with an analytic VJP, and the kinematics outputs of MJX's
mjx.kinematics + com_pos + com_vel with their VJP, for differentiable rollouts (BPTT through lax.scan)."""

import ctypes
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

import nablaengine
from nablaengine import cuda, fixtures, provenance
from nablaengine.model import capability
from nablaengine import engines
from nablaengine.g1_engine import adjoint, kinematics as g1_kin

ITERATIONS, SUBSTEPS, CLAMP_KV, NORMAL = 10, 4, 0, 3
CLAMP_SOFT, GRAD_DECAY = 0.2, 1.0
VARIANTS = tuple((kv, n) for kv in (0, 1, 2) for n in (0, 1, 2, 3))
FWD_VARIANT = (0, 3)
FWD_UNIT = "dk_fwd"
NONFINITE_QPOS, NONFINITE_QVEL, NONFINITE_CTRL, NONFINITE_COT = 1, 2, 4, 8
NONFINITE_SHIFT = 16
DIAG = ("fwd_calls", "fwd_worlds", "fwd_unsupported_worlds", "fwd_iteration_limit_worlds", "tape_records",
        "refact_records", "nonconv_records", "singular_records", "bwd_calls", "bwd_rows", "nonfinite_rows",
        "nonfinite_qpos_rows", "nonfinite_qvel_rows", "nonfinite_ctrl_rows", "nonfinite_cotangent_rows",
        "replay_mismatch_worlds", "fwd_contact_overflow_worlds", "fwd_row_overflow_worlds")
OVERFLOW_CONTACTS, OVERFLOW_ROWS = 4, 16
_OVERFLOW_WARN = [os.environ.get("DK_OVERFLOW_WARN", "0") not in ("", "0")]
LAST = ("worlds", "contacts", "contacts_max", "rows_max")
NSLOT = 16
MAX_DEVICES = 64
UPLOAD_ERR_SMEM = -1
FOREST_FWD_ATTR = ("  if (cudaFuncSetAttribute(k_substep, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Sim)) != "
                   "cudaSuccess)\n    (void)cudaGetLastError();  // (the fused forward unavailable: the split one only)")
TAPE_NAME, OUT_NAME = "dk_tape", "dk_out"
KIN_NAME = "dk_kin"
REMAT_NAMES = (TAPE_NAME, OUT_NAME, KIN_NAME)

VARIANT_UNIT = r"""
#include <cuda_runtime.h>
namespace {  // internal linkage: every variant's kernels, device functions and DKC constants are its own
@ENGINE@
// per device: k_adjoint's dynamic shared memory, adj_launch_smem()'s production choice for that GPU (on V100 more
// than sizeof(Adj), so that at most 4 worlds share an SM); 0 until the device's upload
size_t g_adj_smem[@MAX_DEVICES@];
}  // namespace

// the packed model constants -> this unit's DKC arrays, and k_adjoint's launch configuration on the current device;
// returns a cudaError_t, or UPLOAD_ERR_SMEM if adj_launch_smem() failed (returned 0)
extern "C" int @PFX@_upload(const float* p, cudaStream_t st) {
@LOAD_CONSTS@
  int dev = -1;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return (int)e;
  if (dev < 0 || dev >= @MAX_DEVICES@) return (int)cudaErrorInvalidDevice;
  const size_t smem = adj_launch_smem();  // also sets k_adjoint's shared-memory opt-in to it
  if (smem == 0) return @UPLOAD_ERR_SMEM@;
  g_adj_smem[dev] = smem;
@FWD_ATTR@
  return (int)cudaGetLastError();
}
// ctrl, clamp_soft: the soft clamp's (DK_ADJ_CLAMP_KV=2: the step's ctrl, required, and delta); the other units ignore
// them (their k_adjoint gets nullptr as before and has no delta argument)
extern "C" int @PFX@_adjoint(cudaStream_t st, int nw, int nsub, const float* tape, const float* fric, const float* com,
                           const float* gq_out, const float* gv_out, float* gq_in, float* gv_in, float* gctrl,
                           int* info, const float* ctrl, float clamp_soft) {
  int dev = -1;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return (int)e;
  if (dev < 0 || dev >= @MAX_DEVICES@ || g_adj_smem[dev] == 0)  // ensure_ready() uploads before the first launch
    return (int)cudaErrorInvalidConfiguration;
  if (DK_ADJ_CLAMP_KV == 2 && !ctrl) return (int)cudaErrorInvalidValue;
  (void)clamp_soft;
  k_adjoint<<<nw, 32, g_adj_smem[dev], st>>>(nw, nsub, tape, fric, com, gq_out, gv_out, gq_in, gv_in, gctrl, info,
                                             nullptr, DK_ADJ_CLAMP_KV == 2 ? ctrl : nullptr ADJ_SOFT_ARG(clamp_soft));
  return (int)cudaGetLastError();
}
// the launch on device dev (after its upload): o[0] the dynamic shared memory, o[1] the resident worlds per SM
extern "C" int @PFX@_launch(int dev, long long* o) {
  o[0] = dev >= 0 && dev < @MAX_DEVICES@ ? (long long)g_adj_smem[dev] : 0;
  o[1] = 0;
  if (!o[0]) return 0;
  int old = -1, n = 0;
  cudaGetDevice(&old);
  cudaSetDevice(dev);
  cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k_adjoint, 32, (size_t)o[0]);
  cudaSetDevice(old);
  o[1] = n;
  return (int)e;
}
extern "C" void @PFX@_sizes(long long* o) {  // checked by the Python side against tape.py and the model
  o[0] = (long long)sizeof(Sim); o[1] = (long long)sizeof(Adj); o[2] = REC_WORDS; o[3] = NQ; o[4] = NV; o[5] = MAXCON;
  o[6] = DK_ADJ_CLAMP_KV; o[7] = DK_ADJ_NORMAL; o[8] = OFF_FLAGS;
}
@FWD_SHIM@
"""

FWD_SHIM = r"""
// the forward (host.KERNEL, DK_SPLIT): the split kernels with work memory of @PFX@_work_bytes(nw) bytes (the handler's
// scratch memory), or k_substep (none)
extern "C" long long @PFX@_work_bytes(int nw) { return DK_SPLIT ? (long long)dk_work_bytes(nw) : 0; }
extern "C" int @PFX@_forward(cudaStream_t st, int nw, const float* qpos, const float* qvel, const float* warm,
                           const float* ctrl, const float* fric, const float* com, int iterations, int nsub, float* oq,
                           float* ov, float* ow, int* info, float* ocon, float* tape, void* work) {
#if DK_SPLIT
  return (int)dk_forward_split(st, nw, qpos, qvel, warm, ctrl, fric, com, iterations, nsub, oq, ov, ow, info, ocon,
                               tape, work);
#else
  (void)work;
  k_substep<<<nw, 32, sizeof(Sim), st>>>(nw, qpos, qvel, warm, ctrl, fric, com, iterations, nsub, oq, ov, ow, info, ocon,
                                         tape);
  return (int)cudaGetLastError();
#endif
}
// the kinematics outputs of a state (g1_engine.kinematics.KIN_DEVICE), and their VJP (gq_in / gv_in: null, or
// added to the result last)
extern "C" int @PFX@_kinematics(cudaStream_t st, int nw, const float* qpos, const float* qvel, const float* com,
                              float* qn, float* xpos, float* xquat, float* cvel, float* sc) {
  k_kin<<<kin_grid(nw), 32 * KIN_WPB, 0, st>>>(nw, qpos, qvel, com, qn, xpos, xquat, cvel, sc);
  return (int)cudaGetLastError();
}
extern "C" int @PFX@_kinematics_vjp(cudaStream_t st, int nw, const float* qpos, const float* qvel, const float* com,
                                  const float* gqn, const float* gxpos, const float* gxquat, const float* gcvel,
                                  const float* gsc, const float* gq_in, const float* gv_in, float* gq, float* gv) {
  k_kin_vjp<<<kin_grid(nw), 32 * KIN_WPB, 0, st>>>(nw, qpos, qvel, com, gqn, gxpos, gxquat, gcvel, gsc, gq_in, gv_in,
                                                   gq, gv);
  return (int)cudaGetLastError();
}
"""

FFI_UNIT = r"""
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

@DEFINES@
typedef int (*UploadFn)(const float*, cudaStream_t);
typedef int (*AdjointFn)(cudaStream_t, int, int, const float*, const float*, const float*, const float*, const float*,
                         float*, float*, float*, int*, const float*, float);
typedef int (*ForwardFn)(cudaStream_t, int, const float*, const float*, const float*, const float*, const float*,
                         const float*, int, int, float*, float*, float*, int*, float*, float*, void*);
typedef int (*LaunchFn)(int, long long*);
typedef int (*KinFn)(cudaStream_t, int, const float*, const float*, const float*, float*, float*, float*, float*,
                     float*);
typedef int (*KinVjpFn)(cudaStream_t, int, const float*, const float*, const float*, const float*, const float*,
                        const float*, const float*, const float*, const float*, const float*, float*, float*);
@DECLS@

__device__ unsigned long long g_diag[NDIAG];  // the diagnostic counters (jax_ffi.DIAG), zero at module load
// the last forward call's high-water marks (jax_ffi.LAST; EnginePhysics.stats): slot 0 of any batch size, slot s > 0 of
// the batch size g_slot_B[s]; overwritten by every forward call, read without reset
__device__ unsigned g_last[NSLOT][NLAST];

// one counter += the block's sum of v (warp shuffles, then one integer atomic per warp: exact in any order); every
// thread of the block calls it (no early exit before it), so the full-mask shuffle is in converged control flow
__device__ __forceinline__ void diag_add(int k, unsigned v) {
  __syncwarp();
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0 && v) atomicAdd(&g_diag[k], (unsigned long long)v);
}

// after the forward: info (nw, 4) of k_substep, the tape (null for dk_step) for its records' flags
__global__ void k_diag_fwd(int nw, int nsub, int iterations, const int* info, const float* tape) {
  unsigned unsup = 0, lim = 0, refact = 0, nonconv = 0, sing = 0, ocon = 0, orow = 0;
  for (int w = blockIdx.x * blockDim.x + threadIdx.x; w < nw; w += gridDim.x * blockDim.x) {
    unsup += info[4 * w + 3] != 0;
    ocon += (info[4 * w + 3] & OVERFLOW_CONTACTS) != 0; orow += (info[4 * w + 3] & OVERFLOW_ROWS) != 0;
    lim += info[4 * w] >= iterations;
    if (tape)
      for (int k = 0; k < nsub; k++) {
        unsigned f = __float_as_uint(tape[((size_t)w * nsub + k) * REC_WORDS + OFF_FLAGS]);
        refact += (f & TAPE_REFACT) != 0; nonconv += (f & TAPE_NONCONV) != 0; sing += (f & TAPE_SINGULAR) != 0;
      }
  }
  diag_add(D_FWD_UNSUPPORTED_WORLDS, unsup); diag_add(D_FWD_ITERATION_LIMIT_WORLDS, lim);
  diag_add(D_REFACT_RECORDS, refact); diag_add(D_NONCONV_RECORDS, nonconv); diag_add(D_SINGULAR_RECORDS, sing);
  diag_add(D_FWD_CONTACT_OVERFLOW_WORLDS, ocon); diag_add(D_FWD_ROW_OVERFLOW_WORLDS, orow);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(&g_diag[D_FWD_CALLS], 1ull); atomicAdd(&g_diag[D_FWD_WORLDS], (unsigned long long)nw);
    if (tape) atomicAdd(&g_diag[D_TAPE_RECORDS], (unsigned long long)nw * nsub);
  }
}

// after the forward, one block: the call's record (worlds, contacts summed over the worlds, contacts and constraint rows
// of the most loaded world; info's counts are the last substep's, as MuJoCo Warp's nacon / nefc after its last step)
// into slot 0 and slot (0 < slot < NSLOT, else only slot 0)
__global__ void k_last_fwd(int nw, const int* info, int slot) {
  __shared__ unsigned sm[3][32];
  unsigned sum = 0, cmax = 0, rmax = 0;
  for (int w = threadIdx.x; w < nw; w += blockDim.x) {
    unsigned c = (unsigned)max(info[4 * w + 1], 0), r = (unsigned)max(info[4 * w + 2], 0);
    sum += c; cmax = max(cmax, c); rmax = max(rmax, r);
  }
  for (int o = 16; o > 0; o >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, o);
    cmax = max(cmax, __shfl_down_sync(0xffffffffu, cmax, o));
    rmax = max(rmax, __shfl_down_sync(0xffffffffu, rmax, o));
  }
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
  if (lane == 0) { sm[0][wid] = sum; sm[1][wid] = cmax; sm[2][wid] = rmax; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < nwarp; i++) { sum += sm[0][i]; cmax = max(cmax, sm[1][i]); rmax = max(rmax, sm[2][i]); }
    const unsigned rec[NLAST] = {(unsigned)nw, sum, cmax, rmax};
    for (int k = 0; k < NLAST; k++) {
      g_last[0][k] = rec[k];
      if (slot > 0 && slot < NSLOT) g_last[slot][k] = rec[k];
    }
  }
}
__global__ void k_last_read(unsigned* out) {  // (NSLOT, NLAST)
  for (int k = threadIdx.x; k < NSLOT * NLAST; k += blockDim.x) out[k] = g_last[k / NLAST][k % NLAST];
}

__device__ __forceinline__ bool all_finite(const float* x, int n) {
  bool ok = true;
  for (int i = 0; i < n; i++) ok &= isfinite(x[i]);
  return ok;
}
// after the adjoint: per world which gradients (and incoming cotangents) are non-finite, into info's bits 16-19
__global__ void k_diag_bwd(int nw, const float* gq, const float* gv, const float* gu, const float* cq, const float* cv,
                           int* info) {
  unsigned rows = 0, rq = 0, rv = 0, ru = 0, rc = 0, rm = 0;
  for (int w = blockIdx.x * blockDim.x + threadIdx.x; w < nw; w += gridDim.x * blockDim.x) {
    unsigned b = (all_finite(gq + (size_t)w * NQ, NQ) ? 0u : (unsigned)NONFINITE_QPOS)
                 | (all_finite(gv + (size_t)w * NV, NV) ? 0u : (unsigned)NONFINITE_QVEL)
                 | (all_finite(gu + (size_t)w * NU, NU) ? 0u : (unsigned)NONFINITE_CTRL)
                 | (all_finite(cq + (size_t)w * NQ, NQ) && all_finite(cv + (size_t)w * NV, NV) ? 0u
                                                                                                : (unsigned)NONFINITE_COT);
    int f = info[w];
    info[w] = f | (int)(b << NONFINITE_SHIFT);
    rows += (b & 7u) != 0; rq += (b & NONFINITE_QPOS) != 0; rv += (b & NONFINITE_QVEL) != 0;
    ru += (b & NONFINITE_CTRL) != 0; rc += (b & NONFINITE_COT) != 0; rm += (f & ADJ_REPLAY_MISMATCH) != 0;
  }
  diag_add(D_NONFINITE_ROWS, rows); diag_add(D_NONFINITE_QPOS_ROWS, rq); diag_add(D_NONFINITE_QVEL_ROWS, rv);
  diag_add(D_NONFINITE_CTRL_ROWS, ru); diag_add(D_NONFINITE_COTANGENT_ROWS, rc);
  diag_add(D_REPLAY_MISMATCH_WORLDS, rm);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(&g_diag[D_BWD_CALLS], 1ull); atomicAdd(&g_diag[D_BWD_ROWS], (unsigned long long)nw);
  }
}
__global__ void k_diag_read(unsigned* out, int reset) {  // (NDIAG, 2) low / high words
  for (int k = threadIdx.x; k < NDIAG; k += blockDim.x) {
    unsigned long long v = g_diag[k];
    out[2 * k] = (unsigned)v; out[2 * k + 1] = (unsigned)(v >> 32);
    if (reset) g_diag[k] = 0ull;
  }
}
static unsigned diag_grid(long long nw) { long long g = (nw + 255) / 256; return (unsigned)(g < 1 ? 1 : g > 160 ? 160 : g); }

static std::vector<float> g_consts;  // the packed model constants (dk_set_constants, once per process)
static std::mutex g_mu;
static bool g_ready[MAX_DEVICES];  // per device: constants uploaded and shared-memory limits set in every unit
static long long g_slot_B[NSLOT];  // the batch size of each last-forward slot > 0 (0: free); under g_mu

extern "C" int dk_set_constants(const float* p, long long n) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (n != NCONST) return 1;
  std::vector<float> c(p, p + n);
  if (!g_consts.empty() && c != g_consts) return 2;  // another model: the units were uploaded with this one
  g_consts.swap(c);
  return 0;
}
extern "C" void dk_sizes(long long* o) {  // variant order of the units (VARIANTS), then their sizes
  for (int v = 0; v < NVARIANT; v++) kSizes[v](o + 9 * v);
}
// per variant (VARIANTS order) k_adjoint's launch on device dev: (dynamic shared memory, resident worlds per SM); zeros
// before the device's first call
extern "C" int dk_launch_info(int dev, long long* o) {
  int rc = 0;
  for (int v = 0; v < NVARIANT; v++) rc |= kLaunch[v](dev, o + 2 * v);
  return rc;
}
extern "C" void dk_last_slots(long long* o) {  // the batch size of each last-forward slot (slot 0: any)
  std::lock_guard<std::mutex> lock(g_mu);
  for (int s = 0; s < NSLOT; s++) o[s] = g_slot_B[s];
}
static int last_slot(long long B) {  // the slot of batch size B (a new one on its first call; 0 when all are taken)
  std::lock_guard<std::mutex> lock(g_mu);
  for (int s = 1; s < NSLOT; s++)
    if (g_slot_B[s] == B) return s;
  for (int s = 1; s < NSLOT; s++)
    if (g_slot_B[s] == 0) { g_slot_B[s] = B; return s; }
  return 0;
}

static ffi::Error cuda_error(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return ffi::Error::Success();
  return ffi::Error::Internal(std::string("nablaengine.jax_ffi: ") + what + ": " + cudaGetErrorString(e));
}
// before the first call on a device: every unit's constants, its kernels' dynamic shared-memory limit and k_adjoint's
// launch configuration (adj_launch_smem), on XLA's stream (then synchronized once)
static ffi::Error ensure_ready(cudaStream_t st) {
  int dev = -1;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return cuda_error(e, "cudaGetDevice");
  std::lock_guard<std::mutex> lock(g_mu);
  if (dev < 0 || dev >= MAX_DEVICES) return ffi::Error::Internal("nablaengine.jax_ffi: device ordinal out of range");
  if (g_ready[dev]) return ffi::Error::Success();
  if ((long long)g_consts.size() != NCONST)
    return ffi::Error::Internal("nablaengine.jax_ffi: no model constants (dk_set_constants; use jax_ffi.init())");
  for (int v = 0; v < NVARIANT; v++) {
    int rc = kUpload[v](g_consts.data(), st);
    if (rc == UPLOAD_ERR_SMEM)
      return ffi::Error::Internal("nablaengine.jax_ffi: k_adjoint's launch configuration failed (adj_launch_smem "
                                  "returned 0)");
    if (rc) return cuda_error((cudaError_t)rc, "constant upload");
  }
  e = cudaStreamSynchronize(st);
  if (e != cudaSuccess) return cuda_error(e, "constant upload (synchronize)");
  g_ready[dev] = true;
  return ffi::Error::Success();
}

template <class B> static bool dims_are(const B& b, std::initializer_list<int64_t> want) {  // rank and every dimension
  auto d = b.dimensions();
  if (d.size() != want.size()) return false;
  size_t i = 0;
  for (int64_t n : want)
    if (d[i++] != n) return false;
  return true;
}
static ffi::Error bad(const std::string& m) { return ffi::Error::InvalidArgument("nablaengine.jax_ffi: " + m); }

typedef ffi::Buffer<ffi::F32> BufF;
typedef ffi::Result<ffi::Buffer<ffi::F32>> RetF;
// the kinematics outputs (or their cotangents) of B worlds: qpos_n (B, nq), xpos (B, nbody, 3), xquat (B, nbody,
// 4), cvel (B, nbody, 6), subtree_com (B, nbody, 3)
static bool kin_shapes(int64_t B, const BufF& qn, const BufF& xp, const BufF& xq, const BufF& cv, const BufF& sc) {
  return dims_are(qn, {B, NQ}) && dims_are(xp, {B, NB, 3}) && dims_are(xq, {B, NB, 4}) && dims_are(cv, {B, NB, 6})
         && dims_are(sc, {B, NB, 3});
}
#define KIN_SHAPES "qpos_n (B, nq), xpos (B, nbody, 3), xquat (B, nbody, 4), cvel (B, nbody, 6), subtree_com (B, nbody, 3)"

// outputs con: k_substep's contacts of the last substep (B, MAXCON * 8), not used by the interface (XLA's buffer); tape:
// null for dk_step; scratch: the split forward's work memory (B records of struct Hand, freed when the handler returns,
// in stream order); kin: null, or the five kinematics outputs of the output state (k_kin after the last k_post).
// Every buffer's rank and dimensions are checked before any is read or written
static ffi::Error forward_impl(cudaStream_t st, ffi::ScratchAllocator& scratch, ffi::Buffer<ffi::F32> qpos,
                               ffi::Buffer<ffi::F32> qvel,
                               ffi::Buffer<ffi::F32> warm, ffi::Buffer<ffi::F32> ctrl, ffi::Buffer<ffi::F32> fric,
                               ffi::Buffer<ffi::F32> com, int32_t iterations, int32_t substeps,
                               ffi::Result<ffi::Buffer<ffi::F32>> qo, ffi::Result<ffi::Buffer<ffi::F32>> vo,
                               ffi::Result<ffi::Buffer<ffi::F32>> wo, ffi::Result<ffi::Buffer<ffi::S32>> info,
                               ffi::Result<ffi::Buffer<ffi::F32>> con, ffi::Result<ffi::Buffer<ffi::F32>>* tape,
                               RetF* kin = nullptr) {
  auto d = qpos.dimensions();
  if (d.size() != 2 || d[1] != NQ) return bad("qpos must be (B, nq) float32");
  const int64_t B = d[0];
  if (!dims_are(qvel, {B, NV}) || !dims_are(warm, {B, NV}) || !dims_are(ctrl, {B, NU}) || !dims_are(fric, {B})
      || !dims_are(com, {B, 3}))
    return bad("shapes: qpos (B, nq), qvel (B, nv), warm (B, nv), ctrl (B, nu), friction (B,), com (B, 3)");
  if (substeps < 1 || iterations < 0) return bad("substeps >= 1 and iterations >= 0");
  if (!dims_are(*qo, {B, NQ}) || !dims_are(*vo, {B, NV}) || !dims_are(*wo, {B, NV}) || !dims_are(*info, {B, 4}))
    return bad("outputs: qpos' (B, nq), qvel' (B, nv), warm' (B, nv) float32, info (B, 4) int32");
  if (!dims_are(*con, {B, MAXCON * 8})) return bad("contact output must be (B, MAXCON * 8) float32");
  if (tape && !dims_are(**tape, {B, (int64_t)substeps * REC_WORDS}))
    return bad("tape must be (B, substeps * REC_WORDS) float32");
  if (kin && !kin_shapes(B, *kin[0], *kin[1], *kin[2], *kin[3], *kin[4]))
    return bad("kinematics outputs: " KIN_SHAPES " float32");
  if (B > INT_MAX) return bad("B too large");
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  float* tp = tape ? (*tape)->typed_data() : nullptr;
  void* work = nullptr;
  if (const long long wb = kWorkBytes((int)B)) {  // (XLA's scratch allocator takes alignments up to 16)
    std::optional<void*> a = scratch.Allocate((size_t)wb, 16);
    if (!a) return ffi::Error::Internal("nablaengine.jax_ffi: no scratch memory for the forward's work buffer");
    work = *a;
  }
  int rc = kForward(st, (int)B, qpos.typed_data(), qvel.typed_data(), warm.typed_data(), ctrl.typed_data(),
                    fric.typed_data(), com.typed_data(), iterations, substeps, qo->typed_data(), vo->typed_data(),
                    wo->typed_data(), info->typed_data(), con->typed_data(), tp, work);
  if (rc) return cuda_error((cudaError_t)rc, "k_substep launch");
  if (kin) {  // the output state's kinematics (the outputs are complete after the last k_post, in stream order)
    rc = kKin(st, (int)B, qo->typed_data(), vo->typed_data(), com.typed_data(), kin[0]->typed_data(),
              kin[1]->typed_data(), kin[2]->typed_data(), kin[3]->typed_data(), kin[4]->typed_data());
    if (rc) return cuda_error((cudaError_t)rc, "k_kin launch");
  }
  k_diag_fwd<<<diag_grid(B), 256, 0, st>>>((int)B, substeps, iterations, info->typed_data(), tp);
  ffi::Error rd = cuda_error(cudaGetLastError(), "k_diag_fwd launch");
  if (rd.failure()) return rd;
  k_last_fwd<<<1, 1024, 0, st>>>((int)B, info->typed_data(), last_slot(B));
  return cuda_error(cudaGetLastError(), "k_last_fwd launch");
}

static ffi::Error step_impl(cudaStream_t st, ffi::ScratchAllocator scratch, ffi::Buffer<ffi::F32> qpos,
                            ffi::Buffer<ffi::F32> qvel, ffi::Buffer<ffi::F32> warm, ffi::Buffer<ffi::F32> ctrl,
                            ffi::Buffer<ffi::F32> fric, ffi::Buffer<ffi::F32> com, int32_t iterations, int32_t substeps,
                            ffi::Result<ffi::Buffer<ffi::F32>> qo, ffi::Result<ffi::Buffer<ffi::F32>> vo,
                            ffi::Result<ffi::Buffer<ffi::F32>> wo, ffi::Result<ffi::Buffer<ffi::S32>> info,
                            ffi::Result<ffi::Buffer<ffi::F32>> con) {
  return forward_impl(st, scratch, qpos, qvel, warm, ctrl, fric, com, iterations, substeps, qo, vo, wo, info, con,
                      nullptr);
}

static ffi::Error step_fwd_impl(cudaStream_t st, ffi::ScratchAllocator scratch, ffi::Buffer<ffi::F32> qpos,
                                ffi::Buffer<ffi::F32> qvel, ffi::Buffer<ffi::F32> warm, ffi::Buffer<ffi::F32> ctrl,
                                ffi::Buffer<ffi::F32> fric, ffi::Buffer<ffi::F32> com, int32_t iterations,
                                int32_t substeps,
                                ffi::Result<ffi::Buffer<ffi::F32>> qo, ffi::Result<ffi::Buffer<ffi::F32>> vo,
                                ffi::Result<ffi::Buffer<ffi::F32>> wo, ffi::Result<ffi::Buffer<ffi::S32>> info,
                                ffi::Result<ffi::Buffer<ffi::F32>> con, ffi::Result<ffi::Buffer<ffi::F32>> tape) {
  return forward_impl(st, scratch, qpos, qvel, warm, ctrl, fric, com, iterations, substeps, qo, vo, wo, info, con,
                      &tape);
}

// dk_step / dk_step_fwd with the output state's kinematics (qpos_n, xpos, xquat, cvel, subtree_com) as five more
// outputs
static ffi::Error step_kin_impl(cudaStream_t st, ffi::ScratchAllocator scratch, BufF qpos, BufF qvel, BufF warm,
                                BufF ctrl, BufF fric, BufF com, int32_t iterations, int32_t substeps, RetF qo, RetF vo,
                                RetF wo, ffi::Result<ffi::Buffer<ffi::S32>> info, RetF con, RetF qn, RetF xp, RetF xq,
                                RetF cv, RetF sc) {
  RetF kin[5] = {qn, xp, xq, cv, sc};
  return forward_impl(st, scratch, qpos, qvel, warm, ctrl, fric, com, iterations, substeps, qo, vo, wo, info, con,
                      nullptr, kin);
}
static ffi::Error step_fwd_kin_impl(cudaStream_t st, ffi::ScratchAllocator scratch, BufF qpos, BufF qvel, BufF warm,
                                    BufF ctrl, BufF fric, BufF com, int32_t iterations, int32_t substeps, RetF qo,
                                    RetF vo, RetF wo, ffi::Result<ffi::Buffer<ffi::S32>> info, RetF con, RetF tape,
                                    RetF qn, RetF xp, RetF xq, RetF cv, RetF sc) {
  RetF kin[5] = {qn, xp, xq, cv, sc};
  return forward_impl(st, scratch, qpos, qvel, warm, ctrl, fric, com, iterations, substeps, qo, vo, wo, info, con,
                      &tape, kin);
}

// the step's backward with its kinematics outputs: the output state (qpos', qvel') and the cotangents of qpos_n,
// xpos, xquat, cvel, subtree_com, pulled back onto (qpos', qvel') by k_kin_vjp and added to their cotangents before
// k_adjoint
struct KinBwd {
  const BufF *q1, *v1, *gqn, *gxp, *gxq, *gcv, *gsc;
};

// ctrl: the control step's (B, nu) for clamp_kv 2 (dk_step_bwd_soft), null for 0 / 1 (dk_step_bwd); clamp_soft: 2's
// delta; kin (with scratch): null, or the kinematics outputs' VJP inputs (the combined cotangents in scratch memory)
static ffi::Error bwd_impl(cudaStream_t st, ffi::Buffer<ffi::F32> tape, ffi::Buffer<ffi::F32> fric,
                           ffi::Buffer<ffi::F32> com, const ffi::Buffer<ffi::F32>* ctrl, ffi::Buffer<ffi::F32> gq_out,
                           ffi::Buffer<ffi::F32> gv_out, int32_t substeps, int32_t clamp_kv, int32_t normal,
                           float clamp_soft, ffi::Result<ffi::Buffer<ffi::F32>> gq,
                           ffi::Result<ffi::Buffer<ffi::F32>> gv, ffi::Result<ffi::Buffer<ffi::F32>> gu,
                           ffi::Result<ffi::Buffer<ffi::S32>> info, const KinBwd* kin = nullptr,
                           ffi::ScratchAllocator* scratch = nullptr) {
  auto d = tape.dimensions();
  if (substeps < 1 || d.size() != 2 || d[1] != (int64_t)substeps * REC_WORDS)
    return bad("tape must be (B, substeps * REC_WORDS) float32");
  const int64_t B = d[0];
  if (!dims_are(fric, {B}) || !dims_are(com, {B, 3}) || !dims_are(gq_out, {B, NQ}) || !dims_are(gv_out, {B, NV}))
    return bad("shapes: friction (B,), com (B, 3), g_qpos (B, nq), g_qvel (B, nv)");
  if (ctrl && !dims_are(*ctrl, {B, NU})) return bad("ctrl must be (B, nu) float32");
  if (kin && (!dims_are(*kin->q1, {B, NQ}) || !dims_are(*kin->v1, {B, NV})
              || !kin_shapes(B, *kin->gqn, *kin->gxp, *kin->gxq, *kin->gcv, *kin->gsc)))
    return bad("kinematics: qpos' (B, nq), qvel' (B, nv) and the cotangents of " KIN_SHAPES " float32");
  if (!dims_are(*gq, {B, NQ}) || !dims_are(*gv, {B, NV}) || !dims_are(*gu, {B, NU}) || !dims_are(*info, {B}))
    return bad("outputs: g_qpos (B, nq), g_qvel (B, nv), g_ctrl (B, nu) float32, info (B,) int32");
  if (clamp_kv < 0 || clamp_kv > 2 || normal < 0 || normal > 3) return bad("clamp_kv in {0, 1, 2}, normal in 0..3");
  if ((clamp_kv == 2) != (ctrl != nullptr))
    return bad("clamp_kv 2 (the soft clamp) through dk_step_bwd_soft with ctrl, 0 and 1 through dk_step_bwd");
  if (ctrl && !(clamp_soft >= 0.f && clamp_soft < INFINITY)) return bad("clamp_soft must be finite and >= 0");
  AdjointFn adj = kAdjoint[clamp_kv][normal];
  if (!adj) return bad("this adjoint variant was not built");
  if (B > INT_MAX) return bad("B too large");
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  const float *cq = gq_out.typed_data(), *cv = gv_out.typed_data();  // the cotangents of (qpos', qvel')
  if (kin) {  // + the kinematics outputs' pulled back onto (qpos', qvel'), in scratch memory (freed in stream order)
    std::optional<void*> a = scratch->Allocate((size_t)B * (NQ + NV) * sizeof(float), 16);
    if (!a) return ffi::Error::Internal("nablaengine.jax_ffi: no scratch memory for the combined cotangents");
    float* sq = static_cast<float*>(*a);
    float* sv = sq + (size_t)B * NQ;
    int rk = kKinVjp(st, (int)B, kin->q1->typed_data(), kin->v1->typed_data(), com.typed_data(),
                     kin->gqn->typed_data(), kin->gxp->typed_data(), kin->gxq->typed_data(), kin->gcv->typed_data(),
                     kin->gsc->typed_data(), cq, cv, sq, sv);
    if (rk) return cuda_error((cudaError_t)rk, "k_kin_vjp launch");
    cq = sq;
    cv = sv;
  }
  int rc = adj(st, (int)B, substeps, tape.typed_data(), fric.typed_data(), com.typed_data(), cq, cv, gq->typed_data(),
               gv->typed_data(), gu->typed_data(), info->typed_data(), ctrl ? ctrl->typed_data() : nullptr,
               clamp_soft);
  if (rc) return cuda_error((cudaError_t)rc, "k_adjoint launch");
  k_diag_bwd<<<diag_grid(B), 256, 0, st>>>((int)B, gq->typed_data(), gv->typed_data(), gu->typed_data(), cq, cv,
                                            info->typed_data());
  return cuda_error(cudaGetLastError(), "k_diag_bwd launch");
}

static ffi::Error step_bwd_impl(cudaStream_t st, ffi::Buffer<ffi::F32> tape, ffi::Buffer<ffi::F32> fric,
                                ffi::Buffer<ffi::F32> com, ffi::Buffer<ffi::F32> gq_out, ffi::Buffer<ffi::F32> gv_out,
                                int32_t substeps, int32_t clamp_kv, int32_t normal,
                                ffi::Result<ffi::Buffer<ffi::F32>> gq, ffi::Result<ffi::Buffer<ffi::F32>> gv,
                                ffi::Result<ffi::Buffer<ffi::F32>> gu, ffi::Result<ffi::Buffer<ffi::S32>> info) {
  return bwd_impl(st, tape, fric, com, nullptr, gq_out, gv_out, substeps, clamp_kv, normal, 0.f, gq, gv, gu, info);
}

// clamp_kv 2: the soft clamp's backward, with the step's ctrl (B, nu) and delta (attribute clamp_soft, float32)
static ffi::Error step_bwd_soft_impl(cudaStream_t st, ffi::Buffer<ffi::F32> tape, ffi::Buffer<ffi::F32> fric,
                                     ffi::Buffer<ffi::F32> com, ffi::Buffer<ffi::F32> ctrl,
                                     ffi::Buffer<ffi::F32> gq_out, ffi::Buffer<ffi::F32> gv_out, int32_t substeps,
                                     int32_t normal, float clamp_soft, ffi::Result<ffi::Buffer<ffi::F32>> gq,
                                     ffi::Result<ffi::Buffer<ffi::F32>> gv, ffi::Result<ffi::Buffer<ffi::F32>> gu,
                                     ffi::Result<ffi::Buffer<ffi::S32>> info) {
  return bwd_impl(st, tape, fric, com, &ctrl, gq_out, gv_out, substeps, 2, normal, clamp_soft, gq, gv, gu, info);
}

// dk_step_bwd / dk_step_bwd_soft of a step with kinematics outputs: + the output state q1, v1 and the cotangents
// of qpos_n, xpos, xquat, cvel, subtree_com
static ffi::Error step_bwd_kin_impl(cudaStream_t st, ffi::ScratchAllocator scratch, BufF tape, BufF fric, BufF com,
                                    BufF q1, BufF v1, BufF gq_out, BufF gv_out, BufF gqn, BufF gxp, BufF gxq,
                                    BufF gcv, BufF gsc, int32_t substeps, int32_t clamp_kv, int32_t normal, RetF gq,
                                    RetF gv, RetF gu, ffi::Result<ffi::Buffer<ffi::S32>> info) {
  const KinBwd k{&q1, &v1, &gqn, &gxp, &gxq, &gcv, &gsc};
  return bwd_impl(st, tape, fric, com, nullptr, gq_out, gv_out, substeps, clamp_kv, normal, 0.f, gq, gv, gu, info, &k,
                  &scratch);
}
static ffi::Error step_bwd_soft_kin_impl(cudaStream_t st, ffi::ScratchAllocator scratch, BufF tape, BufF fric,
                                         BufF com, BufF ctrl, BufF q1, BufF v1, BufF gq_out, BufF gv_out, BufF gqn,
                                         BufF gxp, BufF gxq, BufF gcv, BufF gsc, int32_t substeps, int32_t normal,
                                         float clamp_soft, RetF gq, RetF gv, RetF gu,
                                         ffi::Result<ffi::Buffer<ffi::S32>> info) {
  const KinBwd k{&q1, &v1, &gqn, &gxp, &gxq, &gcv, &gsc};
  return bwd_impl(st, tape, fric, com, &ctrl, gq_out, gv_out, substeps, 2, normal, clamp_soft, gq, gv, gu, info, &k,
                  &scratch);
}

// jax_ffi.kinematics: the kinematics outputs of a state (qpos, qvel, com), and their VJP (the cotangents of the
// five outputs -> of qpos and qvel)
static ffi::Error kin_state_dims(const BufF& qpos, const BufF& qvel, const BufF& com, int64_t* B) {
  auto d = qpos.dimensions();
  if (d.size() != 2 || d[1] != NQ) return bad("qpos must be (B, nq) float32");
  *B = d[0];
  if (!dims_are(qvel, {*B, NV}) || !dims_are(com, {*B, 3})) return bad("shapes: qpos (B, nq), qvel (B, nv), com (B, 3)");
  if (*B > INT_MAX) return bad("B too large");
  return ffi::Error::Success();
}
static ffi::Error kin_impl(cudaStream_t st, BufF qpos, BufF qvel, BufF com, RetF qn, RetF xp, RetF xq, RetF cv,
                           RetF sc) {
  int64_t B = 0;
  ffi::Error e = kin_state_dims(qpos, qvel, com, &B);
  if (e.failure()) return e;
  if (!kin_shapes(B, *qn, *xp, *xq, *cv, *sc)) return bad("outputs: " KIN_SHAPES " float32");
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  int rc = kKin(st, (int)B, qpos.typed_data(), qvel.typed_data(), com.typed_data(), qn->typed_data(), xp->typed_data(),
                xq->typed_data(), cv->typed_data(), sc->typed_data());
  return rc ? cuda_error((cudaError_t)rc, "k_kin launch") : ffi::Error::Success();
}
static ffi::Error kin_bwd_impl(cudaStream_t st, BufF qpos, BufF qvel, BufF com, BufF gqn, BufF gxp, BufF gxq, BufF gcv,
                               BufF gsc, RetF gq, RetF gv) {
  int64_t B = 0;
  ffi::Error e = kin_state_dims(qpos, qvel, com, &B);
  if (e.failure()) return e;
  if (!kin_shapes(B, gqn, gxp, gxq, gcv, gsc)) return bad("cotangents: " KIN_SHAPES " float32");
  if (!dims_are(*gq, {B, NQ}) || !dims_are(*gv, {B, NV})) return bad("outputs: g_qpos (B, nq), g_qvel (B, nv) float32");
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  int rc = kKinVjp(st, (int)B, qpos.typed_data(), qvel.typed_data(), com.typed_data(), gqn.typed_data(),
                   gxp.typed_data(), gxq.typed_data(), gcv.typed_data(), gsc.typed_data(), nullptr, nullptr,
                   gq->typed_data(), gv->typed_data());
  return rc ? cuda_error((cudaError_t)rc, "k_kin_vjp launch") : ffi::Error::Success();
}

static ffi::Error diag_impl(cudaStream_t st, ffi::RemainingArgs, int32_t reset, ffi::Result<ffi::Buffer<ffi::U32>> out) {
  if (!dims_are(*out, {NDIAG, 2})) return bad("diagnostics output must be (NDIAG, 2) uint32");
  k_diag_read<<<1, 32, 0, st>>>(out->typed_data(), reset);
  return cuda_error(cudaGetLastError(), "k_diag_read launch");
}

static ffi::Error last_impl(cudaStream_t st, ffi::RemainingArgs, ffi::Result<ffi::Buffer<ffi::U32>> out) {
  if (!dims_are(*out, {NSLOT, NLAST})) return bad("last-forward output must be (NSLOT, NLAST) uint32");
  k_last_read<<<1, 64, 0, st>>>(out->typed_data());
  return cuda_error(cudaGetLastError(), "k_last_read launch");
}

#define FWD_BIND(B) B.Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()                        \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()                          \
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()                          \
    .Attr<int32_t>("iterations").Attr<int32_t>("substeps")                                                           \
    .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::S32>>() \
    .Ret<ffi::Buffer<ffi::F32>>()
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStep, step_impl, FWD_BIND(ffi::Ffi::Bind()));
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepFwd, step_fwd_impl, FWD_BIND(ffi::Ffi::Bind()).Ret<ffi::Buffer<ffi::F32>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepBwd, step_bwd_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Attr<int32_t>("substeps").Attr<int32_t>("clamp_kv").Attr<int32_t>("normal")
                                  .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepBwdSoft, step_bwd_soft_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Attr<int32_t>("substeps").Attr<int32_t>("normal").Attr<float>("clamp_soft")
                                  .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>());
#define F32ARG .Arg<ffi::Buffer<ffi::F32>>()
#define F32RET .Ret<ffi::Buffer<ffi::F32>>()
#define KIN5(X) X X X X X  // the five kinematics outputs (or their cotangents)
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepKin, step_kin_impl, FWD_BIND(ffi::Ffi::Bind()) KIN5(F32RET));
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepFwdKin, step_fwd_kin_impl, FWD_BIND(ffi::Ffi::Bind()) F32RET KIN5(F32RET));
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepBwdKin, step_bwd_kin_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
                                  F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG KIN5(F32ARG)
                                  .Attr<int32_t>("substeps").Attr<int32_t>("clamp_kv").Attr<int32_t>("normal")
                                  F32RET F32RET F32RET .Ret<ffi::Buffer<ffi::S32>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepBwdSoftKin, step_bwd_soft_kin_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
                                  F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG KIN5(F32ARG)
                                  .Attr<int32_t>("substeps").Attr<int32_t>("normal").Attr<float>("clamp_soft")
                                  F32RET F32RET F32RET .Ret<ffi::Buffer<ffi::S32>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkKin, kin_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() F32ARG F32ARG F32ARG
                                  KIN5(F32RET));
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkKinBwd, kin_bwd_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() F32ARG F32ARG F32ARG
                                  KIN5(F32ARG) F32RET F32RET);
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkDiag, diag_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs()
                                  .Attr<int32_t>("reset").Ret<ffi::Buffer<ffi::U32>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkLast, last_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs()
                                  .Ret<ffi::Buffer<ffi::U32>>());
"""

FFI_SURROGATE = r"""
static ffi::Error sur_check(float cone_mu, float impulse_clip) {
  if (!(cone_mu >= 0.f && cone_mu < INFINITY)) return bad("cone_mu must be finite and >= 0");
  if (!(impulse_clip > 0.f)) return bad("impulse_clip must be > 0 (+inf: off)");
  return ffi::Error::Success();
}
static ffi::Error step_bwd_sur_impl(cudaStream_t st, BufF tape, BufF fric, BufF com, BufF ctrl, BufF gq_out,
                                    BufF gv_out, int32_t substeps, int32_t clamp_kv, int32_t normal, float clamp_soft,
                                    float cone_mu, float impulse_clip, RetF gq, RetF gv, RetF gu,
                                    ffi::Result<ffi::Buffer<ffi::S32>> info) {
  ffi::Error e = sur_check(cone_mu, impulse_clip);
  if (e.failure()) return e;
  return bwd_impl(st, tape, fric, com, clamp_kv == 2 ? &ctrl : nullptr, gq_out, gv_out, substeps, clamp_kv, normal,
                  clamp_soft, gq, gv, gu, info, nullptr, nullptr, cone_mu, impulse_clip);
}
static ffi::Error step_bwd_sur_kin_impl(cudaStream_t st, ffi::ScratchAllocator scratch, BufF tape, BufF fric, BufF com,
                                        BufF ctrl, BufF q1, BufF v1, BufF gq_out, BufF gv_out, BufF gqn, BufF gxp,
                                        BufF gxq, BufF gcv, BufF gsc, int32_t substeps, int32_t clamp_kv,
                                        int32_t normal, float clamp_soft, float cone_mu, float impulse_clip, RetF gq,
                                        RetF gv, RetF gu, ffi::Result<ffi::Buffer<ffi::S32>> info) {
  ffi::Error e = sur_check(cone_mu, impulse_clip);
  if (e.failure()) return e;
  const KinBwd k{&q1, &v1, &gqn, &gxp, &gxq, &gcv, &gsc};
  return bwd_impl(st, tape, fric, com, clamp_kv == 2 ? &ctrl : nullptr, gq_out, gv_out, substeps, clamp_kv, normal,
                  clamp_soft, gq, gv, gu, info, &k, &scratch, cone_mu, impulse_clip);
}
#define SUR_ATTRS .Attr<int32_t>("substeps").Attr<int32_t>("clamp_kv").Attr<int32_t>("normal") \
                  .Attr<float>("clamp_soft").Attr<float>("cone_mu").Attr<float>("impulse_clip")
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepBwdSur, step_bwd_sur_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG SUR_ATTRS
                                  F32RET F32RET F32RET .Ret<ffi::Buffer<ffi::S32>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepBwdSurKin, step_bwd_sur_kin_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
                                  F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG KIN5(F32ARG) SUR_ATTRS
                                  F32RET F32RET F32RET .Ret<ffi::Buffer<ffi::S32>>());
"""


def ensure_jax_ffi():
    """jax.ffi on every supported JAX: jax older than 0.5.0 (e.g. 0.4.38 of JetPack 5's jetson-containers) has the same
    FFI API (include_dir, pycapsule, register_ffi_target, ffi_call) only as jax.extend.ffi; it is then also exposed as
    jax.ffi. A no-op on jax 0.5.0 or newer."""
    import sys
    import jax
    if not hasattr(jax, "ffi"):
        from jax.extend import ffi
        jax.ffi = sys.modules["jax.ffi"] = ffi


def surrogate_units(units, ffi_unit="dk_ffi.cu"):
    """A contact_surrogate build's edits of the library's units (in place): every variant unit's <unit>_adjoint takes
    cone_mu, impulse_clip after clamp_soft and passes them to k_adjoint; the FFI unit's declarations, bwd_impl (the
    existing handlers pass the surrogates off) and the handlers FFI_SURROGATE (absent: the PyTorch library)."""
    sig = "int* info, const float* ctrl, float clamp_soft) {"
    for u in list(units):
        t = units[u]
        if sig in t:
            assert t.count(sig) == 1 and t.count("ADJ_SOFT_ARG(clamp_soft));") == 1
            units[u] = (t.replace(sig, "int* info, const float* ctrl, float clamp_soft, float cone_mu, float impulse_clip) {")
                        .replace("ADJ_SOFT_ARG(clamp_soft));", "ADJ_SOFT_ARG(clamp_soft), cone_mu, impulse_clip);"))
    if ffi_unit in units:
        t = units[ffi_unit]
        n = t.count("int*, const float*, float);")
        assert n == 1 + len(VARIANTS), n
        t = t.replace("int*, const float*, float);", "int*, const float*, float, float, float);")
        a = "ffi::ScratchAllocator* scratch = nullptr) {"
        b = "               clamp_soft);\n  if (rc) return cuda_error((cudaError_t)rc, \"k_adjoint launch\");"
        assert t.count(a) == 1 and t.count(b) == 1
        t = t.replace(a, "ffi::ScratchAllocator* scratch = nullptr, float cone_mu = 0.f,\n"
                         "                           float impulse_clip = INFINITY) {  // (the surrogates: off)")
        t = t.replace(b, b.replace("clamp_soft);", "clamp_soft, cone_mu, impulse_clip);"))
        units[ffi_unit] = t + FFI_SURROGATE


FFI_TERRAIN = r"""
extern "C" int @FWD@_terrain_rays(cudaStream_t, int, const float*, const float*, float, float*, float*);
static ffi::Error rays_impl(cudaStream_t st, BufF pnt, BufF vec, float maxd, RetF dist, RetF nrm) {
  auto d = pnt.dimensions();
  if (d.size() != 3 || d[2] != 3) return bad("ray origins must be (B, N, 3) float32");
  const int64_t B = d[0], N = d[1];
  if (!dims_are(vec, {B, N, 3}) || !dims_are(*dist, {B, N}) || !dims_are(*nrm, {B, N, 3}))
    return bad("shapes: origins, directions (B, N, 3), distances (B, N), normals (B, N, 3)");
  if (B * N > INT_MAX) return bad("too many rays");
  if (B * N == 0) return ffi::Error::Success();
  int rc = @FWD@_terrain_rays(st, (int)(B * N), pnt.typed_data(), vec.typed_data(), maxd, dist->typed_data(),
                              nrm->typed_data());
  return rc ? cuda_error((cudaError_t)rc, "k_terrain_rays launch") : ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkTerrainRays, rays_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() F32ARG F32ARG
                                  .Attr<float>("max_distance") F32RET F32RET);
"""

_LOCK = threading.Lock()
_STATE = {}
STATS = dict(builds=0, build_seconds=0.0, loads=0)


def _com_words():
    return int(_STATE["info"].get("com_words", 3)) if _STATE else 3


def _info_words():
    return int(_STATE["info"].get("info_words", 4)) if _STATE else 4


def _tag(v):
    return f"dk_kv{v[0]}_n{v[1]}"


def adj_stage_default(E):
    """The engine E's (engines.modules) default DK_ADJ_STAGE: its adjoint text's #define."""
    import inspect
    hit = re.search(r"#ifndef DK_ADJ_STAGE\n#define DK_ADJ_STAGE (\d)", inspect.getsource(E.adjoint))
    return int(hit.group(1))


def adj_stage(E):
    """The adjoint stage a build of engine E gets (info["adj_stage"]): $DK_ADJ_STAGE, else the engine's default
    (adj_stage_default; docs/configuration.md)."""
    v = os.environ.get("DK_ADJ_STAGE")
    return int(v) if v else adj_stage_default(E)


def com_layout(m, terrain=None, params=None):
    """{name: (offset, width)} of the com input's words per world (info["com_layout"]; their sum is
    info["com_words"]): com_offset (0, 3), the COM offset of the COM-offset body (info["com_body"]; none on the forest
    engine by default, the words are then read and ignored); then the engine's extras: fixtures (the forest engine:
    7 words per fixture group, translation and quaternion relative to the exported placement) or objects (a terrain
    build's object slots, nablaengine.terrain); then each per-world parameter field (params.<field>,
    nablaengine.params)."""
    out, o = {"com_offset": (0, 3)}, 3
    if engines.is_forest(m):
        from nablaengine._optional import FOREST, load
        fstep = load("nablaengine.forest_engine", FOREST, "step")
        nfix = fstep.com_words(m) - 3
        if nfix:
            out["fixtures"], o = (3, nfix), 3 + nfix
    if terrain is not None and terrain.com_words() > 3:
        from nablaengine import terrain as _terrain
        nobj = _terrain.OBJ_WORDS * int(terrain.objects)
        out["objects"], o = (o, nobj), o + nobj
        nvel = terrain.com_words() - 3 - nobj - terrain.material_words()
        if nvel:
            out["object_velocity"], o = (o, nvel), o + nvel
        if terrain.material_words():
            out["object_material"], o = (o, terrain.material_words()), o + terrain.material_words()
    if params:
        for f, (off, w) in params.layout(m.nv).items():
            out[f"params.{f}"] = (o + off, w)
    return out


def _model(model_dir=None, model=None, foot_pairs=None, com_body=None):
    if model is not None:
        assert foot_pairs is not None, "init(model=...) needs foot_pairs (the foot-floor pair indices)"
        if engines.is_forest(model) and not engines.tree_defaults(model):
            return model, [int(p) for p in foot_pairs], com_body
        return model, [int(p) for p in foot_pairs], com_body or capability.COM_BODY
    d = model_dir or fixtures.fixtures_dir()
    m, mt = fixtures.load_model(d, check=None)
    return m, mt["foot_pairs"], com_body or mt.get("torso_body", capability.COM_BODY)


def con_words(m, contact_forces=False):
    """Words per contact of the contact table (info["con_words"]): 12 with the forces (the forest engine's table, or
    the G1 engine's with contact_forces=True: g1_engine.contact_forces), else 8."""
    from nablaengine.g1_engine import contact_forces as _cf
    if isinstance(contact_forces, _cf.Spec):
        return int(contact_forces.con_words)
    if engines.is_forest(m):
        from nablaengine._optional import FOREST, load
        fstep = load("nablaengine.forest_engine", FOREST, "step")
        return int(fstep.CON_WORDS)
    if contact_forces:
        return int(_cf.CON_WORDS)
    return 8


def contact_spec(m, contact_forces=False, foot_wrench=None, contact_frame=False, terrain=None, foot_normal=False):
    """The G1 engine's contact-force options (g1_engine.contact_forces.Spec; None: the build without them) of init()'s
    contact_forces ("last" / True, "substeps"), foot_wrench (True: the G1's ankle-roll links, or body names / ids) and
    contact_frame; terrain: the build's TerrainConfig; foot_normal: the foot contacts' foot and toe-weight words
    (Spec.foot_split; the forest engine with foot_offset). The forest engine's table always has the forces: there the
    plain opt-in (True / "last" without foot_wrench / contact_frame) gives None, the default build byte for byte, and
    the other options a Spec for forest_engine.contact_forces."""
    from nablaengine.g1_engine import contact_forces as _cf
    if isinstance(contact_forces, _cf.Spec):
        return contact_forces
    if engines.is_forest(m):
        if (contact_forces in (False, None, True, "last") and foot_wrench in (None, False) and not contact_frame
                and not foot_normal and not (terrain is not None and contact_forces in (True, "last"))):
            return None
        spec = _cf.as_spec(contact_forces, foot_wrench, contact_frame, terrain is not None, m, bool(foot_normal),
                           bool(foot_normal) and bool(getattr(terrain, "terrain_sand", False)))
        from nablaengine._optional import FOREST, load
        load("nablaengine.forest_engine", FOREST, "contact_forces").check(m, spec)
        return spec
    return _cf.as_spec(contact_forces, foot_wrench, contact_frame, terrain is not None, m, bool(foot_normal))


def contact_info(m, spec):
    """info["contact_forces"]: None, or dict(mode, feet (body ids), foot_bodies (names), frame, terrain, con_layout;
    foot_split: True with foot_normal)."""
    if spec is None:
        return None
    out = dict(mode=spec.mode, feet=list(spec.feet),
               foot_bodies=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in spec.feet],
               frame=spec.frame, terrain=spec.terrain, con_layout={k: list(v) for k, v in spec.layout().items()})
    if spec.foot_split:
        out["foot_split"] = True
    if spec.foot_sand:
        out["foot_sand"] = True
    return out


def con_width(info, substeps):
    """Words per world of forward / step's contact output (contacts=True) for `substeps` substeps."""
    cf, mc, cw = info.get("contact_forces"), int(info["maxcon"]), int(info.get("con_words", 8))
    if not cf:
        return mc * cw
    nf = len(cf["feet"])
    return (int(substeps) if cf["mode"] == "substeps" else 1) * mc * cw + (int(substeps) * nf * 6 + nf if nf else 0)


LATENCY_VAR = "DK_FOREST_LATENCY"


def latency_mode():
    """$DK_FOREST_LATENCY: "1" builds the forest engine's latency mode (small batches: k_pre with two warps per world,
    warp 0 the smooth dynamics, warp 1 the collision; the same bits; forest_engine.host DK_PRE2W), "0" or unset not.
    Part of the build (the forward unit's -D options), so it keys the library cache."""
    v = (os.environ.get(LATENCY_VAR) or "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"${LATENCY_VAR}={v!r}: 0 or 1")
    return v == "1"


def capacity(m, terrain=None, max_contacts=None, max_rows=None):
    """A build's capacities per world: (max_contacts, max_rows) as the engine generators take them (None: their
    defaults; a terrain build's TerrainConfig.maxcon when max_contacts is None) and the resulting (contacts, rows).
    Raises ValueError where they do not fit the model's engine (step.check_capacity: G1 class 64 contacts, forest
    engine 255; both 1023 rows) or contradict TerrainConfig.maxcon."""
    E = engines.modules(m)
    tmc = terrain.maxcon if terrain is not None and terrain.maxcon else None
    if tmc is not None and max_contacts is not None and int(max_contacts) != int(tmc):
        raise ValueError(f"max_contacts={max_contacts} but the terrain configuration's maxcon={tmc}")
    mc_in = int(max_contacts) if max_contacts is not None else tmc
    mr_in = int(max_rows) if max_rows is not None else None
    if E.forest:
        E.step.check_capacity(m, mc_in, mr_in)
        mc = mc_in or E.step.maxcon(m)
        out = mc_in, mr_in, mc, E.step.nefc(mc, m, mr_in)
    else:
        E.step.check_capacity(m, mc_in, mr_in, None)
        out = mc_in, mr_in, mc_in or E.step.maxcon(m), E.step.nefc(mr_in)
    _warn_rows(m, *out)
    return out


def _warn_rows(m, mc_in, mr_in, mc, nefc):
    from nablaengine.forest_engine import step as fstep
    if mc_in is None and mr_in is None and engines.is_g1_class(m):
        return
    need = fstep.rows_needed(mc, m)
    if nefc < need:
        import warnings
        reserve = fstep.rows_needed(0, m)
        per = (need - reserve) // mc
        warnings.warn(f"constraint rows {nefc} < {need} that {mc} contacts can take ({per} per contact, {reserve} for "
                      f"the dofs): with the dof rows all in use the rows fill at {max(0, nefc - reserve) // per} "
                      "contacts and the rows beyond are dropped (their forces lost; diagnostics: row overflow). "
                      + (f"Raise max_rows to {need} or lower max_contacts" if need <= fstep.MAX_ROWS else
                         f"The rows stop at {fstep.MAX_ROWS}: max_contacts <= {(fstep.MAX_ROWS - reserve) // per} "
                         "keeps every contact's rows"), RuntimeWarning, stacklevel=3)


def library_sources(m, foot_pairs, forward_unit=False, target=None, terrain=None, com_body=None, params=None,
                    max_contacts=None, max_rows=None, motor=None, contact_forces=False, bam=None, surrogate=False,
                    contact_grad=False, noslip_pairs=None, capacity_diag=False):
    """({file name: CUDA source} of the library's translation units, the packed constants, the per-unit -D options).
    forward_unit: the forward and kinematics kernels in a unit of their own, FWD_UNIT.cu (the text of FWD_VARIANT's unit,
    built with other architecture flags: cuda.arch_flags('forward')), every adjoint variant's unit without them. target:
    the build target (default cuda.arch()), whose forward-unit -D options (cuda.unit_defines: the register caps on
    sm_90) the unit with the forward kernels gets. terrain: a nablaengine.terrain.TerrainConfig (a terrain build: every
    unit gets <unit>_set_terrain; the PyTorch interface's library only). com_body: the COM-offset body. params: a
    nablaengine.params.PhysicsParams (per-world parameters in the com rows; none: the text unchanged). max_contacts,
    max_rows: the capacities per world (capacity(); None: the engine's defaults, the text unchanged). motor:
    actuator torque-speed limits (nablaengine.motor.as_motor's argument; None: the text unchanged). contact_forces:
    the contact table with the forces, 12 words per contact (con_words(); the G1 engine's opt-in,
    g1_engine.contact_forces; False: the text unchanged; the forest engine's table always has them). bam: a
    nablaengine.actuators.Bam, the BAM servo model (the tree engine: g1_engine.bam_hooks; the forward entry and the
    FFI handlers dk_step_act / dk_step_fwd_act take the state rows; None: the text unchanged). surrogate: the
    backward-only contact-gradient surrogates (the forest engine: forest_engine.surrogate, surrogate_units; False: the
    text unchanged). contact_grad: the contact output's VJP (the forest engine: forest_engine.contact_vjp, a Spec in
    contact_forces; False: the text unchanged). capacity_diag: the forward's per-world capacity counts in info's words
    4, 5 (the forest engine: forest_engine.capacity_diag, info (B, 6); False: the text unchanged)."""
    from nablaengine.params import as_params
    from nablaengine.g1_engine import world_params
    from nablaengine.motor import as_motor
    params, motor = as_params(params), as_motor(motor)
    E = engines.modules(m)
    if E.forest:
        from nablaengine._optional import FOREST, load
        world_params = load("nablaengine.forest_engine", FOREST, "world_params")
        world_params.check(m, params, foot_pairs, com_body)
        body_params = load("nablaengine.forest_engine", FOREST, "body_params")
        body_params.check(m, params, foot_pairs, com_body, terrain, surrogate)
        params = body_params.resolve(m, params, foot_pairs)
        foot_offset = load("nablaengine.forest_engine", FOREST, "foot_offset")
        if foot_offset.active(params):
            from nablaengine import terrain as _tr
            foot_offset.check(m, params, foot_pairs, (terrain, _tr.robot_terrain_geoms(m, foot_pairs, terrain.geoms))
                              if terrain is not None else None, surrogate)
        params = foot_offset.resolve(m, params)
        if "object_size" in params.fields:
            object_size = load("nablaengine.forest_engine", FOREST, "object_size")
            object_size.check(m, params, com_body, terrain)
            params = object_size.resolve(m, params, com_body)
            m = object_size.bound_model(m, params)
    elif ({"object_size", "gravity", "xfrc", "body_inertial", "sole_thickness", "foot_offset", "dof_frictionloss",
           "frictionloss_solref", "frictionloss_solimp"} & set(params.fields)
          or getattr(params, "invweight0", "model") != "model"):
        bad = [f for f in params.fields if f in ("object_size", "gravity", "xfrc", "body_inertial", "sole_thickness",
                                                 "foot_offset", "dof_frictionloss", "frictionloss_solref",
                                                 "frictionloss_solimp")]
        bad += [f"invweight0={params.invweight0!r}"] if params.invweight0 != "model" else []
        raise ValueError(f"PhysicsParams {bad}: the forest engine only (this model goes to the G1 engine; route it "
                         "with nablaengine.engines.prefer_forest)")
    adjoint, host, model_hooks, g1_kin, g1_step, g1_tape = E.adjoint, E.host, E.model_hooks, E.kinematics, E.step, E.tape
    ter = None
    cspec = contact_spec(m, contact_forces, terrain=terrain)
    if cspec is not None and cspec.terrain != (terrain is not None):
        raise ValueError("contact_forces: the Spec's terrain flag must match the build")
    if terrain is not None:
        from nablaengine import terrain as _terrain
        if getattr(terrain, "object_velocity", False) and bam is not None:
            raise ValueError("TerrainConfig.object_velocity: not with the BAM servo model")
        if getattr(terrain, "terrain_material", False) and not E.forest:
            raise ValueError("TerrainConfig.terrain_material: the forest engine only (this model goes to the G1 engine; "
                             "engines.prefer_forest or DK_ENGINE=forest)")
        if getattr(terrain, "terrain_material", False) and bam is not None:
            raise ValueError("TerrainConfig.terrain_material: not with the BAM servo model")
        if getattr(terrain, "terrain_sand", False) and not (E.forest and params and "foot_offset" in params.fields):
            raise ValueError("TerrainConfig.terrain_sand: the forest engine with the foot_offset parameter "
                             "(PhysicsParams((\"foot_offset\", ...)): the sand elements scale its offsets)")
        if getattr(terrain, "object_material", False) and not E.forest:
            raise ValueError("TerrainConfig.object_material: the forest engine only (this model goes to the G1 engine; "
                             "engines.prefer_forest or DK_ENGINE=forest)")
        if getattr(terrain, "object_material", False) and bam is not None:
            raise ValueError("TerrainConfig.object_material: not with the BAM servo model")
        if getattr(terrain, "object_velocity", False) and not E.forest:
            raise ValueError("TerrainConfig.object_velocity: the forest engine only (this model goes to the G1 engine; "
                             "engines.prefer_forest or DK_ENGINE=forest)")
        ter = (terrain, _terrain.robot_terrain_geoms(m, foot_pairs, terrain.geoms))
        if terrain.mode == "native" and any(int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_BOX for g, _, _ in ter[1]):
            raise ValueError("box robot geoms against the terrain: the analytic heightfield mode only (TerrainConfig("
                             "mode='analytic'); the native mode's prisms collide spheres and capsules)")
    consts, lays, sizes = (adjoint.engine_constants(m, foot_pairs, com_body, **({"terrain": ter} if ter else {}))
                           if E.forest else adjoint.engine_constants(m, foot_pairs, ter, com_body))
    load = host.loader(lays, sizes)
    load = re.sub(r"cudaMemcpyToSymbol\(([^;]*)\);", r"cudaMemcpyToSymbolAsync(\1, 0, cudaMemcpyHostToDevice, st);", load)
    assert "cudaMemcpyToSymbol(" not in load
    mc_in, mr_in, mc, _ = capacity(m, terrain, max_contacts, max_rows)
    kw = {"params": params} if params else {}
    if mr_in is not None:
        kw["max_rows"] = mr_in
    if motor is not None:
        kw["motor"] = motor
    if bam is not None:
        kw["bam"] = bam
    if surrogate:
        if not E.forest:
            raise ValueError("contact_surrogate: the forest engine only (this model goes to the G1 engine)")
        if terrain is not None and getattr(terrain, "object_material", False):
            raise ValueError("contact_surrogate: not with TerrainConfig.object_material")
        if terrain is not None and terrain.object_velocity:
            raise ValueError("contact_surrogate: not with TerrainConfig.object_velocity (the surrogate's contact VJP "
                             "has no slot velocity)")
        kw["surrogate"] = True
    if noslip_pairs is not None:
        if not E.forest:
            raise ValueError("noslip_pairs: the forest engine only (this model goes to the G1 engine)")
        kw["noslip_pairs"] = noslip_pairs
    if E.forest and params and int(m.opt.noslip_iterations) > 0:
        from nablaengine._optional import FOREST, load as _ld
        _ld("nablaengine.forest_engine", FOREST, "noslip").check_foot(m, params, foot_pairs, noslip_pairs)
    engine = adjoint.cuda_source(m, mc_in, ter, **kw)
    pw_off = (g1_step.com_words(m) if E.forest else 3) + (terrain.com_words() - 3 if terrain is not None else 0)
    assert engine.count("#include <cuda_runtime.h>\n") == 1 and engine.count("#include") == 1, \
        "the engine source may only include cuda_runtime.h (it is wrapped in a namespace)"
    engine = engine.replace("#include <cuda_runtime.h>\n", "")
    host_kernel = host.KERNEL
    if E.forest:
        from nablaengine._optional import FOREST, load as _load
        host_kernel = _load("nablaengine.forest_engine", FOREST, "nv128").host(host_kernel, m)
        host_kernel = _load("nablaengine.forest_engine", FOREST, "margin_convex").host(host_kernel, m)
    if cspec is not None:
        from nablaengine.g1_engine import contact_forces as _cf
        if E.forest:
            from nablaengine._optional import FOREST, load as _load
            _fcf = _load("nablaengine.forest_engine", FOREST, "contact_forces")
            engine, host_kernel = _fcf.engine(engine, cspec), _fcf.host(host_kernel, cspec)
            if cspec.foot_split:
                if "foot_offset" not in params.fields:
                    raise ValueError("foot_normal: needs the foot_offset parameter in the build (PhysicsParams; its "
                                     "feet, heel / toe line and axis define the split)")
                host_kernel = foot_offset.host_split(host_kernel, **({"sand": True} if cspec.foot_sand else {}))
        else:
            engine, host_kernel = _cf.engine(engine, cspec), _cf.host(host_kernel, cspec)
    _cvjp = None
    if contact_grad:
        if not E.forest:
            raise ValueError("contact_grad: the forest engine only (this model goes to the G1 engine; "
                             "engines.prefer_forest or DK_ENGINE=forest)")
        from nablaengine._optional import FOREST, load as _load
        _cvjp = _load("nablaengine.forest_engine", FOREST, "contact_vjp")
        _cvjp.check(m, cspec, ter, surrogate, bam)
        engine = _cvjp.engine(engine, cspec, m)
        if cspec.foot_split:
            engine = foot_offset.engine_split(engine)
    act_hooks, act_n = None, None
    if bam is not None:
        from nablaengine._optional import BAM, load as _load_opt
        act_hooks = _load_opt("nablaengine.forest_engine" if E.forest else "nablaengine.g1_engine", BAM, "bam_hooks")
        act_n = m.nv
        host_kernel = act_hooks.host(host_kernel)
    elif E.forest and _muscle_build(m):
        from nablaengine._optional import FOREST, load as _load
        act_hooks, act_n = _load("nablaengine.forest_engine", FOREST, "muscle"), int(m.na)
        host_kernel = act_hooks.host(host_kernel)
    _capd = None
    if capacity_diag:
        if not E.forest:
            raise ValueError("capacity_diag: the forest engine only (this model goes to the G1 engine; "
                             "engines.prefer_forest or DK_ENGINE=forest)")
        from nablaengine._optional import FOREST, load as _load
        _capd = _load("nablaengine.forest_engine", FOREST, "capacity_diag")
        engine = _capd.engine(engine)
    units, defines = {}, {}

    def osz_host(text):
        text = object_size.host(text, engine) if "object_size" in params.fields else text
        return _capd.host(text) if _capd is not None else text

    def unit(v, fwd, pfx):
        src = (VARIANT_UNIT.replace("@FWD_SHIM@", FWD_SHIM if fwd else "")
               .replace("@FWD_ATTR@", (FOREST_FWD_ATTR if E.forest else
                                       "  cudaFuncSetAttribute(k_substep, cudaFuncAttributeMaxDynamicSharedMemorySize, "
                                       "(int)sizeof(Sim));") if fwd else "")
               .replace("@PFX@", pfx).replace("@MAX_DEVICES@", str(MAX_DEVICES))
               .replace("@UPLOAD_ERR_SMEM@", str(UPLOAD_ERR_SMEM)).replace("@LOAD_CONSTS@", load)
               .replace("@ENGINE@", engine + (osz_host(world_params.apply(g1_step.object_hooks(
                                                  host_kernel, g1_step.OBJECT_HOST_HOOKS, ter,
                                                  **({"m": m} if E.forest else {})), m, params, "host", pw_off))
                                              + world_params.apply(model_hooks.apply(g1_step.object_hooks(
                                                  g1_kin.KIN_DEVICE, g1_step.OBJECT_KIN_HOOKS, ter), m, "kinematics"),
                                                  m, params, "kinematics", pw_off)
                                              if fwd else "")))
        if ter is not None:
            src += (f'extern "C" int {pfx}_set_terrain(int dev, const void* p) {{\n'
                    f'  int old = 0;\n  cudaGetDevice(&old);\n  cudaError_t e = cudaSetDevice(dev);\n'
                    f'  if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_ter, p, sizeof(TerrainPtrs));\n'
                    + (f'  if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_tmat, (const void* const*)p + 8, '
                       f'sizeof(const float*));  // (TerrainConfig.terrain_material: the elements\' materials)\n'
                       if getattr(terrain, "terrain_material", False) or getattr(terrain, "terrain_sand", False)
                       else '')
                    + f'  cudaSetDevice(old);\n  return (int)e;\n}}\n'
                    f'extern "C" int {pfx}_terrain_rays(cudaStream_t st, int n, const float* pnt, const float* vec, '
                    f'float maxd, float* dist, float* nrm) {{\n'
                    f'  if (n > 0) k_terrain_rays<<<(n + 127) / 128, 128, 0, st>>>(n, pnt, vec, maxd, dist, nrm);\n'
                    f'  return (int)cudaGetLastError();\n}}\n')
        units[f"{pfx}.cu"] = src
        defines[f"{pfx}.cu"] = [f"-DDK_ADJ_CLAMP_KV={v[0]}", f"-DDK_ADJ_NORMAL={v[1]}"]
        if os.environ.get("DK_ADJ_STAGE") and (E.forest or adj_stage(E) != adj_stage_default(E)):
            defines[f"{pfx}.cu"].append(f"-DDK_ADJ_STAGE={int(os.environ['DK_ADJ_STAGE'])}")

    for v in VARIANTS:
        unit(v, v == FWD_VARIANT and not forward_unit, _tag(v))
    if forward_unit:
        unit(FWD_VARIANT, True, FWD_UNIT)
    fp = FWD_UNIT if forward_unit else _tag(FWD_VARIANT)
    if act_hooks is not None:
        units[f"{fp}.cu"] = act_hooks.fwd_shim(units[f"{fp}.cu"])
        for u in list(units):
            units[u] = act_hooks.adj_unit(units[u])
    defines[f"{fp}.cu"] += cuda.unit_defines("forward", target, defines[f"{fp}.cu"],
                                             forest=E.forest)
    if E.forest and latency_mode():
        if E.step.has_defer(m):
            raise ValueError(f"${LATENCY_VAR}=1: not for a model with deferred pairs (mesh pairs: forest_engine.step."
                             "has_defer); unset it")
        defines[f"{fp}.cu"] += ["-DDK_PRE2W=1"]
        for u in units:
            units[u] = units[u].replace("k_adjoint<<<nw, 32,", "k_adjoint<<<nw, 64,").replace(
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k_adjoint, 32,",
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k_adjoint, 64,")
            defines[u] = defines.get(u, []) + ["-DDK_ADJ_WARPS=2"]
    from nablaengine.g1_engine.adjoint import with_adj_bounds
    light = E.forest and cuda.forest_adj_light(m)
    for v in VARIANTS:
        u = f"{_tag(v)}.cu"
        caps = cuda.adjoint_defines(target, defines[u], forest=E.forest, light=light)
        caps += cuda.tree_adj_cap_defines(target, defines[u] + caps, forest=E.forest, terrain=ter is not None)
        if caps:
            units[u] = with_adj_bounds(units[u])
            defines[u] += caps
        defines[u] += cuda.capbox_coop_defines(target, units[u], defines[u], forest=E.forest)
    if E.forest:
        for u in units:
            defines.setdefault(u, [])
            defines[u] += cuda.forest_hold_defines(target, units[u], defines[u])
    defs = [f"#define {k} {val}" for k, val in (
        ("NQ", m.nq), ("NV", m.nv), ("NU", m.nu), ("NB", m.nbody), ("MAXCON", mc),
        ("REC_WORDS", g1_tape.rec_words(m, mc, g1_tape.kind(ter), mr_in, **({"motor": True} if motor else {}))
         + (_bam_tape_words(m) if bam is not None else 0)),
        ("OFF_FLAGS", g1_tape.layout(m)["flags"][0]), ("TAPE_REFACT", f"{g1_tape.FLAG_REFACT}u"),
        ("TAPE_NONCONV", f"{g1_tape.FLAG_NONCONV}u"), ("TAPE_SINGULAR", f"{g1_tape.FLAG_SINGULAR}u"),
        ("ADJ_REPLAY_MISMATCH", adjoint.ADJ_REPLAY_MISMATCH), ("NONFINITE_QPOS", NONFINITE_QPOS),
        ("NONFINITE_QVEL", NONFINITE_QVEL), ("NONFINITE_CTRL", NONFINITE_CTRL), ("NONFINITE_COT", NONFINITE_COT),
        ("NONFINITE_SHIFT", NONFINITE_SHIFT), ("NCONST", len(consts)), ("NVARIANT", len(VARIANTS)),
        ("NDIAG", len(DIAG)), ("MAX_DEVICES", MAX_DEVICES), ("NSLOT", NSLOT), ("NLAST", len(LAST)),
        ("UPLOAD_ERR_SMEM", UPLOAD_ERR_SMEM), ("OVERFLOW_CONTACTS", OVERFLOW_CONTACTS),
        ("OVERFLOW_ROWS", OVERFLOW_ROWS))]
    defs += [f"#define D_{name.upper()} {i}" for i, name in enumerate(DIAG)]
    decls = []
    for v in VARIANTS:
        t = _tag(v)
        decls.append(f'extern "C" int {t}_upload(const float*, cudaStream_t);')
        decls.append(f'extern "C" int {t}_adjoint(cudaStream_t, int, int, const float*, const float*, const float*, '
                     f'const float*, const float*, float*, float*, float*, int*, const float*, float);')
        decls.append(f'extern "C" void {t}_sizes(long long*);')
        decls.append(f'extern "C" int {t}_launch(int, long long*);')
    decls.append(f'extern "C" int {fp}_forward(cudaStream_t, int, const float*, const float*, '
                 f'const float*, const float*, const float*, const float*, int, int, float*, float*, float*, int*, '
                 f'float*, float*, void*);')
    decls.append(f'extern "C" long long {fp}_work_bytes(int);')
    decls.append(f'extern "C" int {fp}_kinematics(cudaStream_t, int, const float*, const float*, '
                 f'const float*, float*, float*, float*, float*, float*);')
    decls.append(f'extern "C" int {fp}_kinematics_vjp(cudaStream_t, int, const float*, const float*, '
                 f'const float*, const float*, const float*, const float*, const float*, const float*, const float*, '
                 f'const float*, float*, float*);')
    decls.append(f"static const KinFn kKin = {fp}_kinematics;")
    decls.append(f"static const KinVjpFn kKinVjp = {fp}_kinematics_vjp;")
    upload = [f"{_tag(v)}_upload" for v in VARIANTS]
    if forward_unit:
        t = _tag(FWD_VARIANT)
        decls.append(f'extern "C" int {FWD_UNIT}_upload(const float*, cudaStream_t);')
        decls.append(f"static int {t}_upload_fwd(const float* p, cudaStream_t st) {{ const int rc = {t}_upload(p, st); "
                     f"return rc ? rc : {FWD_UNIT}_upload(p, st); }}")
        upload[VARIANTS.index(FWD_VARIANT)] = f"{t}_upload_fwd"
    decls.append("static const UploadFn kUpload[NVARIANT] = {" + ", ".join(upload) + "};")
    decls.append("typedef void (*SizesFn)(long long*);")
    decls.append("static const SizesFn kSizes[NVARIANT] = {" + ", ".join(f"{_tag(v)}_sizes" for v in VARIANTS) + "};")
    decls.append("static const LaunchFn kLaunch[NVARIANT] = {" + ", ".join(f"{_tag(v)}_launch" for v in VARIANTS)
                 + "};")
    adj = [[f"{_tag((kv, n))}_adjoint" if (kv, n) in VARIANTS else "nullptr" for n in range(4)] for kv in range(3)]
    decls.append("static const AdjointFn kAdjoint[3][4] = {" + ", ".join("{" + ", ".join(r) + "}" for r in adj) + "};")
    decls.append(f"static const ForwardFn kForward = {fp}_forward;")
    decls.append(f"static long long (*const kWorkBytes)(int) = {fp}_work_bytes;")
    units["dk_ffi.cu"] = FFI_UNIT.replace("@DEFINES@", "\n".join(defs)).replace("@DECLS@", "\n".join(decls))
    if act_hooks is not None:
        units["dk_ffi.cu"] = act_hooks.ffi(units["dk_ffi.cu"], act_n)
    if ter is not None:
        units["dk_ffi.cu"] += FFI_TERRAIN.replace("@FWD@", fp)
    if cspec is not None:
        units["dk_ffi.cu"] = _cf.ffi(units["dk_ffi.cu"], cspec)
    elif con_words(m, contact_forces) != 8:
        u = units["dk_ffi.cu"]
        assert u.count("MAXCON * 8") >= 2
        units["dk_ffi.cu"] = u.replace("MAXCON * 8", f"MAXCON * {con_words(m, contact_forces)}")
    nov = terrain.velocity_words() + terrain.material_words() if terrain is not None else 0
    if params or nov:
        u, gw = units["dk_ffi.cu"], m.nu + (params.width(m.nv) + 1 if params else 0) + nov
        assert u.count("dims_are(*gu, {B, NU})") == 1 and u.count("all_finite(gu + (size_t)w * NU, NU)") == 1
        units["dk_ffi.cu"] = (u.replace("dims_are(*gu, {B, NU})", f"dims_are(*gu, {{B, {gw}}})")
                              .replace("all_finite(gu + (size_t)w * NU, NU)", f"all_finite(gu + (size_t)w * {gw}, {gw})"))
    if (ter is not None and terrain.objects) or params or (E.forest and g1_step.com_words(m) != 3):
        cw = pw_off + params.width(m.nv)
        u = units["dk_ffi.cu"]
        assert u.count("dims_are(com, {B, 3})") == 2 and u.count("dims_are(com, {*B, 3})") == 1
        u = u.replace("dims_are(com, {B, 3})", f"dims_are(com, {{B, {cw}}})").replace("dims_are(com, {*B, 3})", f"dims_are(com, {{*B, {cw}}})")
        units["dk_ffi.cu"] = u
    if surrogate:
        surrogate_units(units)
    if _cvjp is not None:
        _cvjp.units(units, cspec, len(VARIANTS))
    if _capd is not None:
        units["dk_ffi.cu"] = _capd.ffi(units["dk_ffi.cu"])
    defines["dk_ffi.cu"] = []
    return units, consts, defines


def _muscle_build(m):
    try:
        from nablaengine.forest_engine import muscle
    except ImportError:
        return False
    return muscle.applies(m)


def _act_info(m, bam):
    if bam is None:
        if engines.is_forest(m) and _muscle_build(m):
            return dict(act_words=int(m.na), act_layout="mujoco_act", bam=None)
        return dict(act_words=0, act_layout=None, bam=None)
    from nablaengine._optional import BAM, load
    _bam = load("nablaengine.actuators", BAM, "bam")
    return dict(act_words=_bam.act_words(m.nv), act_layout=_bam.layout(m.nv),
                bam=dict(key=bam.key(), **{k: v for k, v in _bam.servo(bam).items()}))


def _bam_tape_words(m):
    from nablaengine._optional import BAM, load
    _bam = load("nablaengine.actuators", BAM, "bam")
    return _bam.tape_words(m.nv)


def _noslip_pairs(m, foot_pairs, spec):
    if spec is None:
        return None
    from nablaengine._optional import FOREST, load
    return load("nablaengine.forest_engine", FOREST, "noslip").resolve_pairs(m, spec)


def build_plan(m, foot_pairs, arch, terrain=None, com_body=None, params=None, max_contacts=None, max_rows=None,
               motor=None, contact_forces=False, bam=None, surrogate=False, contact_grad=False, noslip_pairs=None,
               capacity_diag=False):
    """How init() builds the library for build target arch: dict(units, consts, defines (library_sources), unit_arch
    ({unit: architecture flags} where they are not -arch=<arch>: every unit on sm_120 by default, the forward unit only
    with $DK_SM120_FRONTEND=split; cuda.arch_flags), arch_flags (forward, adjoint), split (the forward / kinematics
    kernels in a unit of their own, FWD_UNIT), forward_unit (the prefix of the forward / kinematics entries),
    forward_defines (the -D options the forward unit gets for the target: cuda.unit_defines)). Also the build plan of
    scripts/identity_gate.py."""
    fwd_flags = cuda.arch_flags("forward", arch)
    adj_flags = cuda.arch_flags(None, arch)
    split = fwd_flags != adj_flags
    cap = {k: v for k, v in (("max_contacts", max_contacts), ("max_rows", max_rows), ("motor", motor))
           if v is not None}
    if contact_forces:
        cap["contact_forces"] = contact_forces
    if bam is not None:
        cap["bam"] = bam
    if surrogate:
        cap["surrogate"] = True
    if contact_grad:
        cap["contact_grad"] = True
    if noslip_pairs is not None:
        cap["noslip_pairs"] = noslip_pairs
    if capacity_diag:
        cap["capacity_diag"] = True
    units, consts, defines = library_sources(m, foot_pairs, forward_unit=split, target=arch, terrain=terrain,
                                              com_body=com_body, **({"params": params} if params else {}), **cap)
    fu = FWD_UNIT if split else _tag(FWD_VARIANT)
    fwd_defines = [d for d in defines[f"{fu}.cu"]
                   if not d.startswith(("-DDK_ADJ_CLAMP_KV=", "-DDK_ADJ_NORMAL="))]
    unit_arch = {n: f for n, f in ((n, fwd_flags if split and n == f"{FWD_UNIT}.cu" else adj_flags) for n in units)
                 if f != [f"-arch={arch}"]}
    return dict(units=units, consts=consts, defines=defines, unit_arch=unit_arch,
                arch_flags=dict(forward=fwd_flags, adjoint=adj_flags), split=split, forward_unit=fu,
                forward_defines=fwd_defines)


def common_flags(arch):
    """The nvcc flags of every library unit that has no architecture flags of its own (build_plan's unit_arch)."""
    return ["-O3", f"-arch={arch}", "-std=c++17", "-Xcompiler", "-fPIC"]


def unit_command(name, arch, defines, unit_arch=None, include=None):
    """nvcc's arguments (after the nvcc path, before "-c -o <object> <source>") for the library's translation unit
    name: common_flags(arch), or -O3 with the unit's own architecture flags (unit_arch) and the rest of them, then its
    -D options (defines[name]), and for the FFI unit dk_ffi.cu -I<include> (XLA's FFI headers)."""
    common = common_flags(arch)
    unit_arch = unit_arch or {}
    flags = ["-O3", *unit_arch[name], *common[2:]] if name in unit_arch else common
    inc = [f"-I{include}"] if name == "dk_ffi.cu" else []
    return [*flags, *defines[name], *inc]


def _device_arch():
    try:
        import jax
        return "sm_" + str(jax.devices("gpu")[0].compute_capability).replace(".", "")
    except Exception:
        return None


def _arch():
    if os.environ.get("DK_ARCH"):
        return cuda.arch()
    return _device_arch() or cuda.arch()


def cache_dir():
    d = Path(os.environ.get("DK_CACHE_DIR") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
             / "nablaengine")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build(units, defines, arch, include, consts, unit_arch=None, prefix="libdk_g1"):
    nvcc = cuda.nvcc()
    ver = subprocess.run([nvcc, "--version"], check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout
    if include is None:
        assert "dk_ffi.cu" not in units, "the FFI unit dk_ffi.cu needs XLA's FFI headers (include)"
        hdr = None
    else:
        hdr = hashlib.sha256(b"".join((Path(include) / "xla" / "ffi" / "api" / f).read_bytes()
                                      for f in ("api.h", "ffi.h", "c_api.h"))).hexdigest()
    common = common_flags(arch)
    unit_arch = dict(unit_arch or {})
    kd = dict(units=units, defines=defines, common=common, nvcc=ver, ffi_headers=hdr,
              consts=hashlib.sha256(consts.tobytes()).hexdigest())
    if unit_arch:
        kd["unit_arch"] = unit_arch
    if arch == "sm_120":
        kd["sm120_frontend"] = cuda.frontend()
    elif arch in cuda.N7_VIRTUAL:
        kd[f"{arch}_frontend"] = cuda.frontend_of(arch)
    if arch in cuda.REG_CAPS:
        kd["reg_caps"] = cuda.reg_caps(arch)
    if arch in cuda.FOREST_ADJ_CAPS and os.environ.get(cuda.FOREST_ADJ_VAR):
        kd["forest_adj_cap"] = cuda.forest_adj_cap()
    key = hashlib.sha256(json.dumps(kd, sort_keys=True).encode()).hexdigest()[:20]
    out = cache_dir() / f"{prefix}_{arch}_{key}.so"
    if out.exists():
        return out, False, key
    t0 = time.time()
    work = Path(tempfile.mkdtemp(prefix=f"build_{key}_", dir=cache_dir()))
    try:
        def compile_unit(name):
            (work / name).write_text(units[name])
            r = subprocess.run([nvcc, *unit_command(name, arch, defines, unit_arch, include), "-c", "-o",
                                str(work / (name + ".o")), str(work / name)], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
            if r.returncode:
                raise RuntimeError(f"nvcc failed on {name}:\n{r.stdout[-4000:]}")
        with ThreadPoolExecutor(max_workers=max(1, min(len(units), int(os.environ.get("DK_BUILD_JOBS") or 0) or os.cpu_count() or 1))) as ex:
            list(ex.map(compile_unit, sorted(units)))
        r = subprocess.run([nvcc, "-shared", f"-arch={arch}", "-o", str(work / "lib.so"),
                            *(str(work / (n + ".o")) for n in sorted(units))], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True)
        if r.returncode:
            raise RuntimeError(f"nvcc link failed:\n{r.stdout[-4000:]}")
        os.replace(work / "lib.so", out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    STATS["builds"] += 1
    STATS["build_seconds"] += time.time() - t0
    return out, True, key


def init(model_dir=None, model=None, foot_pairs=None, terrain=None, com_body=None, params=None, max_contacts=None,
         max_rows=None, motor_limits=None, contact_forces=False, actuator=None, contact_surrogate=False,
         foot_wrench=None, contact_frame=False, contact_grad=False, noslip_pairs=None, foot_normal=False,
         capacity_diag=False):
    """Load the library for the model (build it on first use), hand it the model constants and register the FFI
    targets; idempotent (later calls return the loaded library; a different model raises). model_dir: a fixtures
    directory (default $DK_FIXTURES, else fixtures/mjlab); model, foot_pairs: an MjModel instead (e.g. the same mjlab
    model when the fixtures are not installed) and its foot-floor pair indices. A model the engine does not support
    raises nablaengine.model.capability.UnsupportedModel (a ValueError and an AssertionError) before anything is
    built, with the failing checks' codes and values (the input validation capability.check: the G1 engine's engine
    scope, forward and adjoint, or the forest engine's scope "forest"; nablaengine.engines picks the engine);
    $DK_ALLOW_UNSUPPORTED lets named codes through with a warning (experiments only; docs/configuration.md). Returns a
    dict (engine: "g1" or "forest", the engine the model went to, docs/engines.md; adj_stage: the adjoint's
    DK_ADJ_STAGE; com_body: the COM-offset body, None for none; com_layout: {name: (offset, width)} of com's words,
    com_layout(); path, key, arch, rec_words, sizes per variant, built, arch_flags {forward, adjoint}, forward_defines: the -D options
    the forward unit gets for the target, e.g. sm_90's register caps, [] elsewhere; forward_unit: the prefix of the
    library's forward / kinematics entries, <forward_unit>_forward / _kinematics / _kinematics_vjp: dk_fwd where that
    unit has flags of its own, else FWD_VARIANT's unit dk_kv0_n3). terrain: a nablaengine.terrain.TerrainConfig for a
    terrain build (set_terrain() uploads the terrain, raycast() measures it). com_body: the body that
    takes the per-world COM offset (default torso_link, or a fixtures directory's torso_body, for a G1-class model on
    either engine unless routed with engines.prefer_forest; none for the forest engine's other models). params: per-world
    physics parameters (nablaengine.params.PhysicsParams or field names): com is then (B, com_words) with each
    world's values after the COM offset (info["com_words"], info["params"]; PhysicsParams.pack). max_contacts,
    max_rows: the contact and constraint-row capacity per world (None: the engine's defaults, info["maxcon"] /
    info["nefc"]: 64 / 192 for the mjlab G1; at most 64 / 1023 for the G1 class, 255 / 1023 for the forest engine;
    contacts or rows beyond the capacity are dropped and counted, see read_diagnostics). actuator: an actuator
    model on top of the model's actuators (nablaengine.actuators: torque-speed limits "unitree_g1", a MotorLimits, a
    {joint regex: MotorCurve} dict or an Actuators, both engines; "bam" / a Bam, the BAM servo model, the tree engine
    only: the model is prepared for it (actuators.bam.prepare), and forward / step / make_step take and return its
    per-world state act (B, info["act_words"]), docs/actuators.md; None: off, the text unchanged, info["actuator"]
    None, info["act_words"] 0). motor_limits: a permanent alias of actuator (giving both raises;
    info["motor_limits"] the torque-speed limits' key, None without them). contact_forces: the contact table (forward / step contacts=True,
    contact_table) with the contact forces, info["con_words"] 12, on the G1 engine too (g1_engine.contact_forces: an
    opt-in, the state and the tape bitwise the default build's; the forest engine's table always has them; False: the
    G1 engine's 8 words; "last" = True; "substeps": a table per substep; either engine (forest_engine.contact_forces):
    foot_wrench (True: the G1's ankle-roll links, or body names / ids): per substep and foot the summed contact force and moment about the body
    origin, and the largest total normal force per control step, after the table(s); contact_frame: the contact
    frame's tangent t1 in 3 more words; a terrain build adds the terrain element's word; info["contact_forces"]
    (layout), con_width(), contact_table(); docs/api.md "Contact forces"). contact_surrogate: a forest-engine build whose backward also takes the contact-gradient
    surrogates (backward / make_step / step cone_mu=, impulse_clip=; forest_engine.surrogate, docs/api.md); the
    forward, the tape and the exact gradients are those of the build without it; info["contact_surrogate"].
    contact_grad: a forest-engine build whose contact output is differentiable (forest_engine.contact_vjp): step /
    make_step with contacts=True take the contact output's cotangent in their VJP, backward(g_contacts=); without
    contact_forces it is the last substep's table; info["contact_grad"]; docs/api.md "Contact forces".
    noslip_pairs: a model with noslip (opt.noslip_iterations > 0) on the forest engine: the noslip pass on these
    contact pairs' rows only (forest_engine.noslip.resolve_pairs: pair indices or names, (body, body) or (geom, geom)
    name pairs, or "objects"; the frictionless rows stay as MuJoCo's noslip has them; None: MuJoCo's noslip on every
    contact); info["noslip_pairs"] (the pair indices, None for all). foot_normal: the forest engine with the
    foot_offset parameter: two more contact-table words (the contact's foot, its toe weight along the foot; contact
    output on, "last" unless contact_forces says otherwise), and contact_table()'s foot_normal (B, [substeps,] 4): the
    normal force per foot split linearly into heel and toe as foot_offset's s(x) (left heel, left toe, right heel,
    right toe); differentiable on a contact_grad build (docs/api.md "Foot offsets"). capacity_diag: the forest engine
    with two more forward info words per world, info (B, 6) (info["info_words"]): the contacts found and the
    constraint rows needed before the capacities truncate them, each the largest over the call's substeps
    (capacity_counts(); forest_engine.capacity_diag; docs/api.md "Capacity diagnostics"); the state, tape, contacts
    and gradients bitwise those of the build without it."""
    import jax
    from nablaengine.params import as_params
    from nablaengine import actuators
    ensure_jax_ffi()
    params, act = as_params(params), actuators.resolve(actuator, motor_limits, "jax_ffi.init")
    motor = act.motor if act is not None else None
    with _LOCK:
        if (_STATE and model_dir is None and model is None and terrain is None and not params and max_contacts is None
                and max_rows is None and act is None and not contact_forces and not contact_surrogate
                and foot_wrench is None and not contact_frame and not contact_grad
                and noslip_pairs is None and not foot_normal and not capacity_diag):
            return _STATE["info"]
        m, fp, com = _model(model_dir, model, foot_pairs, com_body)
        E = engines.modules(m)
        actuators.check_engine(act, E.forest, "jax_ffi.init")
        actuators.check_bam_build(act, "jax_ffi.init", terrain, params,
                                  contact_forces or foot_wrench not in (None, False) or contact_frame)
        m = actuators.prepare_model(act, m)
        bam = act.bam if act is not None else None
        g1_kin, g1_step, g1_tape = E.kinematics, E.step, E.tape
        engines.require(m, fp, com, "jax_ffi.init")
        arch = _arch()
        mc_in, mr_in, mc, nefc = capacity(m, terrain, max_contacts, max_rows)
        mkey = motor.key(m) if motor is not None else None
        akey = act.key(m) if act is not None else None
        cspec = contact_spec(m, contact_forces, foot_wrench, contact_frame, terrain, foot_normal)
        if cspec is None and contact_grad:
            from nablaengine.g1_engine import contact_forces as _cf
            cspec = _cf.as_spec("last", None, False, terrain is not None, m)
        if cspec is not None:
            contact_forces = cspec
        nsp = _noslip_pairs(m, fp, noslip_pairs)
        plan = build_plan(m, fp, arch, terrain, com, params, mc_in, mr_in, motor, contact_forces,
                          **({"bam": bam} if bam is not None else {}),
                          **({"surrogate": True} if contact_surrogate else {}),
                          **({"contact_grad": True} if contact_grad else {}),
                          **({"noslip_pairs": nsp} if nsp is not None else {}),
                          **({"capacity_diag": True} if capacity_diag else {}))
        units, consts, defines, unit_arch = plan["units"], plan["consts"], plan["defines"], plan["unit_arch"]
        fwd_flags, adj_flags, split = plan["arch_flags"]["forward"], plan["arch_flags"]["adjoint"], plan["split"]
        fwd_defines = plan["forward_defines"]
        chash = hashlib.sha256(consts.tobytes()).hexdigest()
        if _STATE:
            if (_STATE["consts_hash"] != chash or _STATE["info"].get("terrain") != (terrain.key() if terrain else None)
                    or _STATE["info"].get("params") != (params.key() if params else None)
                    or _STATE["info"].get("motor_limits") != mkey or _STATE["info"].get("actuator") != akey
                    or _STATE["info"]["con_words"] != con_words(m, contact_forces)
                    or _STATE["info"].get("contact_forces") != contact_info(m, cspec)
                    or bool(_STATE["info"].get("contact_surrogate")) != bool(contact_surrogate)
                    or bool(_STATE["info"].get("contact_grad")) != bool(contact_grad)
                    or bool(_STATE["info"].get("capacity_diag")) != bool(capacity_diag)
                    or _STATE["info"].get("noslip_pairs") != (list(nsp) if nsp is not None else None)
                    or (_STATE["info"]["maxcon"], _STATE["info"]["nefc"]) != (mc, nefc)
                    or _STATE["info"].get("clamp_implicit") != (list(g1_step.clamp_implicit())
                                                                 if hasattr(g1_step, "clamp_implicit") else None)
                    or _STATE["info"].get("tree_chol") != (None if E.forest else g1_step.tree_chol_stages(m))):
                raise RuntimeError("nablaengine.jax_ffi: already initialised with another model, terrain build, "
                                   "capacity, actuator model, contact_forces, DK_CLAMP_IMPLICIT or DK_TREE_CHOL (one "
                                   "per process)")
            return _STATE["info"]
        path, built, key = _build(units, defines, arch, jax.ffi.include_dir(), consts, unit_arch or None)
        lib = ctypes.cdll.LoadLibrary(str(path))
        STATS["loads"] += 1
        lib.dk_set_constants.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        c = np.ascontiguousarray(consts, np.float32)
        rc = lib.dk_set_constants(c.ctypes.data, len(c))
        if rc:
            raise RuntimeError(f"nablaengine.jax_ffi: dk_set_constants failed ({rc})")
        sz = np.zeros(9 * len(VARIANTS), np.int64)
        lib.dk_sizes(ctypes.c_void_p(sz.ctypes.data))
        sz = sz.reshape(len(VARIANTS), 9)
        rec = g1_tape.rec_words(m, mc, g1_tape.kind(terrain), mr_in, **({"motor": True} if motor else {}))
        if bam is not None:
            rec += _bam_tape_words(m)
        for v, row in zip(VARIANTS, sz):
            assert tuple(row[2:9]) == (rec, m.nq, m.nv, mc, v[0], v[1], g1_tape.layout(m)["flags"][0]), (v, row)
        lib.dk_launch_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
        lib.dk_last_slots.argtypes = [ctypes.c_void_p]
        targets = {}
        for kind, sym in (("step", "DkStep"), ("fwd", "DkStepFwd"), ("bwd", "DkStepBwd"), ("bwd_soft", "DkStepBwdSoft"),
                          ("diag", "DkDiag"), ("last", "DkLast"), ("step_kin", "DkStepKin"), ("fwd_kin", "DkStepFwdKin"),
                          ("bwd_kin", "DkStepBwdKin"), ("bwd_soft_kin", "DkStepBwdSoftKin"), ("kin", "DkKin"),
                          ("kin_bwd", "DkKinBwd")):
            targets[kind] = f"dk_{kind}_{key[:8]}"
            jax.ffi.register_ffi_target(targets[kind], jax.ffi.pycapsule(getattr(lib, sym)), platform="CUDA")
        if bam is not None or (E.forest and _muscle_build(m)):
            for kind, sym in (("step_act", "DkStepAct"), ("fwd_act", "DkStepFwdAct"), ("bwd_act", "DkStepBwdAct")):
                targets[kind] = f"dk_{kind}_{key[:8]}"
                jax.ffi.register_ffi_target(targets[kind], jax.ffi.pycapsule(getattr(lib, sym)), platform="CUDA")
        if contact_surrogate:
            for kind, sym in (("bwd_sur", "DkStepBwdSur"), ("bwd_sur_kin", "DkStepBwdSurKin")):
                targets[kind] = f"dk_{kind}_{key[:8]}"
                jax.ffi.register_ffi_target(targets[kind], jax.ffi.pycapsule(getattr(lib, sym)), platform="CUDA")
        if contact_grad:
            for kind, sym in (("bwd_con", "DkStepBwdCon"), ("bwd_con_kin", "DkStepBwdConKin")):
                targets[kind] = f"dk_{kind}_{key[:8]}"
                jax.ffi.register_ffi_target(targets[kind], jax.ffi.pycapsule(getattr(lib, sym)), platform="CUDA")
        if terrain is not None:
            targets["rays"] = f"dk_rays_{key[:8]}"
            jax.ffi.register_ffi_target(targets["rays"], jax.ffi.pycapsule(lib.DkTerrainRays), platform="CUDA")
            lib_units = [_tag(v) for v in VARIANTS] + ([FWD_UNIT] if split else [])
            for u in lib_units:
                getattr(lib, f"{u}_set_terrain").argtypes = [ctypes.c_int, ctypes.c_void_p]
        torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, com) if com else -1
        try:
            g1_kin.check_model(m)
            kin_error = None
        except AssertionError as e:
            kin_error = f"the model does not meet the kinematics outputs' assumptions ({e!r})"
        commit, dirty = provenance.source_commit()
        info = dict(path=str(path), key=key, arch=arch, built=built, version=nablaengine.__version__, commit=commit,
                    dirty=dirty, engine=E.name, adj_stage=adj_stage(E), com_body=com,
                    com_layout=com_layout(m, terrain, params), rec_words=rec, nq=m.nq, nv=m.nv, nu=m.nu,
                    nbody=m.nbody, kinematics_error=kin_error, maxcon=mc, nefc=nefc,
                    terrain=terrain.key() if terrain is not None else None,
                    elliptic=bool(int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)),
                    com_words=(((g1_step.com_words(m) if E.forest else 3) + (terrain.com_words() - 3 if terrain is not None else 0))
                               + params.width(m.nv)),
                    params=params.key() if params else None, param_words=params.width(m.nv) if params else 0,
                    obj_vel_words=terrain.velocity_words() if terrain is not None else 0,
                    obj_mat_words=terrain.material_words() if terrain is not None else 0,
                    motor_limits=mkey, actuator=akey, **_act_info(m, bam),
                    clamp_implicit=(list(E.step.clamp_implicit()) if hasattr(E.step, "clamp_implicit") else None),
                    tree_chol=None if E.forest else g1_step.tree_chol_stages(m),
                    con_words=con_words(m, contact_forces), contact_forces=contact_info(m, cspec),
                    contact_surrogate=bool(contact_surrogate), contact_grad=bool(contact_grad),
                    noslip_pairs=list(nsp) if nsp is not None else None,
                    capacity_diag=bool(capacity_diag), info_words=6 if capacity_diag else 4,
                    terrain_units=lib_units if terrain is not None else None,
                    sizeof_sim=int(sz[0, 0]), sizeof_adj={_tag(v): int(r[1]) for v, r in zip(VARIANTS, sz)},
                    variants=[list(v) for v in VARIANTS], timestep=float(m.opt.timestep), torso_id=int(torso),
                    foot_pairs=[int(p) for p in fp], consts_sha256=chash,
                    arch_flags=dict(forward=fwd_flags, adjoint=adj_flags), forward_defines=fwd_defines,
                    forward_unit=FWD_UNIT if split else _tag(FWD_VARIANT))
        if unit_arch or arch in cuda.N7_VIRTUAL:
            info["front_end"] = {k: cuda.nvvm_version(cuda.virtual_arch(u, arch) or "compute_" + arch[3:])
                                 for k, u in (("forward", "forward"), ("adjoint", None))}
        _STATE.update(lib=lib, targets=targets, info=info, consts_hash=chash, consts=consts, m=m, bam=bam,
                      params=params)
        import atexit
        atexit.register(_exit_report)
        return info



_TERRAIN = {}


def set_terrain(terrain, devices=None, cell=1.0):
    """Upload a nablaengine.terrain.Terrain for a terrain build (init(terrain=...)) to the given JAX GPU devices
    (default all); replaces the previous one. cell: the broad-phase grid's cell size in metres."""
    import jax
    import jax.numpy as jnp
    init()
    if not _STATE["info"].get("terrain"):
        raise RuntimeError("nablaengine.jax_ffi: set_terrain needs a terrain build: init(terrain=TerrainConfig(...))")
    packed = terrain.pack(cell)
    _terrain_material_check(packed)
    packed["material"] = terrain.material_upload(_STATE["info"].get("terrain"))
    _STATE["terrain_sand"] = packed["sand"]
    order = ("boxes", "hf_head", "hf_int", "hf_data", "grid", "cell_start", "cell_items", "elem_cell", "material")
    for d in devices or [d for d in jax.devices() if d.platform == "gpu"]:
        arrs = []
        for k in order:
            x = np.ascontiguousarray(packed[k]).reshape(-1)
            if x.size == 0:
                x = np.zeros(1, x.dtype)
            arrs.append(jax.device_put(jnp.asarray(x), d))
        for x in arrs:
            x.block_until_ready()
        ptrs = (ctypes.c_void_p * 9)(*[x.unsafe_buffer_pointer() for x in arrs])
        for u in _STATE["info"]["terrain_units"]:
            rc = getattr(_STATE["lib"], f"{u}_set_terrain")(int(d.local_hardware_id), ctypes.cast(ptrs, ctypes.c_void_p))
            if rc:
                raise RuntimeError(f"nablaengine.jax_ffi: the terrain upload failed ({rc})")
        for f in _STATE.get("terrain_hooks", ()):
            rc = f(int(d.local_hardware_id), ctypes.cast(ptrs, ctypes.c_void_p))
            if rc:
                raise RuntimeError(f"nablaengine.jax_ffi: the terrain upload failed ({rc})")
        _TERRAIN[d.id] = arrs


def _terrain_material_check(packed, info=None):
    info = info if info is not None else _STATE["info"]
    mat = packed["material"]
    if mat.size and (mat[:, 7] != 0).any() and not (info.get("terrain") or {}).get("terrain_material"):
        raise ValueError("Terrain.material sets element materials: the build needs TerrainConfig(terrain_material=True) "
                         "(the forest engine)")
    sd = packed.get("sand")
    if sd is not None and sd.size and (sd[:, 0] != 0).any() and not (info.get("terrain") or {}).get("terrain_sand"):
        raise ValueError("Terrain.material sets sand fields (k > 0): the build needs TerrainConfig(terrain_sand=True) "
                         "(the forest engine, with the foot_offset parameter)")


def _yaw_rotation(rot):
    import jax.numpy as jnp
    x = rot[:, :, 0].at[:, 2].set(0.0)
    xn = jnp.linalg.norm(x, axis=1)
    y = rot[:, :, 1].at[:, 2].set(0.0)
    y = y / jnp.maximum(jnp.linalg.norm(y, axis=1), 1e-6)[:, None]
    sing = xn < 0.1
    xy = jnp.stack([y[:, 1], -y[:, 0], jnp.zeros_like(y[:, 0])], 1)
    x = jnp.where(sing[:, None], xy, x)
    xn = jnp.where(sing, 1.0, xn)
    x = x / jnp.maximum(xn, 1e-6)[:, None]
    z = jnp.zeros_like(x[:, 0])
    return jnp.stack([jnp.stack([x[:, 0], -x[:, 1], z], 1), jnp.stack([x[:, 1], x[:, 0], z], 1),
                      jnp.stack([z, z, jnp.ones_like(z)], 1)], 1)


def grid_pattern(size=(1.0, 1.0), resolution=0.1, direction=(0.0, 0.0, -1.0)):
    """(offsets (N, 3), directions (N, 3)) as mjlab's GridPatternCfg (NumPy float32; torch_ffi.grid_pattern)."""
    x = np.arange(-size[0] / 2, size[0] / 2 + resolution * 0.5, resolution, dtype=np.float32)
    y = np.arange(-size[1] / 2, size[1] / 2 + resolution * 0.5, resolution, dtype=np.float32)
    gx, gy = np.meshgrid(x, y, indexing="xy")
    off = np.zeros((gx.size, 3), np.float32)
    off[:, 0], off[:, 1] = gx.ravel(), gy.ravel()
    d = np.asarray(direction, np.float32)
    return off, np.broadcast_to(d / np.linalg.norm(d), off.shape).copy()


def raycast(pos, rot, offsets, directions, alignment="base", max_distance=10.0):
    """Height scan / depth rays against the terrain, as torch_ffi.raycast (mjlab's RayCastSensor): pos (B, 3), rot
    (B, 3, 3), offsets / directions (N, 3); returns dict(distances (B, N), normals_w (B, N, 3), hit_pos_w (B, N, 3)).
    Traceable (jax.jit); not differentiable."""
    import jax
    import jax.numpy as jnp
    init()
    if "rays" not in _STATE["targets"]:
        raise RuntimeError("nablaengine.jax_ffi: raycast needs a terrain build: init(terrain=TerrainConfig(...))")
    pos, rot = jnp.asarray(pos, jnp.float32), jnp.asarray(rot, jnp.float32)
    offsets, directions = jnp.asarray(offsets, jnp.float32), jnp.asarray(directions, jnp.float32)
    if alignment == "base":
        R = rot
    elif alignment == "yaw":
        R = _yaw_rotation(rot)
    elif alignment == "world":
        R = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), rot.shape)
    else:
        raise ValueError(f"alignment {alignment!r}: 'base', 'yaw' or 'world'")
    origins = pos[:, None] + jnp.einsum("bij,nj->bni", R, offsets)
    rays = jnp.einsum("bij,nj->bni", R, directions)
    B, N = origins.shape[:2]
    dist, nrm = jax.ffi.ffi_call(_STATE["targets"]["rays"],
                                 (jax.ShapeDtypeStruct((B, N), jnp.float32), jax.ShapeDtypeStruct((B, N, 3), jnp.float32)))(
        origins, rays, max_distance=np.float32(max_distance))
    dist, nrm = jax.lax.stop_gradient(dist), jax.lax.stop_gradient(nrm)
    hit = dist >= 0
    hit_pos = jnp.where(hit[..., None], origins + rays * jnp.maximum(dist, 0.0)[..., None], origins)
    return dict(distances=dist, normals_w=nrm, hit_pos_w=hit_pos)

def rec_words():
    """Words (float32) per tape record (one world, one substep): 1472 for the mjlab model."""
    return init()["rec_words"]


def tape_bytes(nworld, substeps=SUBSTEPS, steps=1):
    """Bytes of the tape residuals of `steps` control steps (a differentiated rollout of H steps keeps H of them)."""
    return 4 * rec_words() * substeps * nworld * steps


def _check(name, x, shape):
    if x.dtype != np.float32 or tuple(x.shape) != tuple(shape):
        raise TypeError(f"nablaengine.jax_ffi: {name} must be float32 {tuple(shape)}, got {x.dtype} {tuple(x.shape)}")


def _forest_kin():
    lib = init()
    if lib["engine"] != "forest" or not lib["kinematics_error"]:
        return None
    from nablaengine import dyn_query
    lay = lib["com_layout"]
    if "fixtures" in lay:
        raise ValueError("nablaengine.jax_ffi: the kinematics outputs of a model with per-world fixture placements "
                         f"are not available ({lib['kinematics_error']})")
    dyn_query.init(model=_STATE["m"], foot_pairs=lib["foot_pairs"], com_body=lib["com_body"],
                   params=_STATE.get("params"), com_layout=lay)
    return dyn_query


def _kin_lib():
    lib = init()
    if lib["kinematics_error"]:
        raise ValueError(f"nablaengine.jax_ffi: {lib['kinematics_error']}")
    return lib


def _kin_structs(lib, B):
    import jax
    import jax.numpy as jnp
    return [jax.ShapeDtypeStruct(s, jnp.float32) for s in g1_kin.shapes(_STATE["m"], B).values()]


def _check_kin(lib, B, arrays, what):
    for name, x, shape in zip(g1_kin.OUTPUTS, arrays, g1_kin.shapes(_STATE["m"], B).values()):
        _check(f"{what}{name}", x, shape)


def _act_check(lib, act, B):
    aw = int(lib.get("act_words", 0))
    if aw and act is None:
        if lib.get("act_layout") == "mujoco_act":
            raise ValueError("nablaengine.jax_ffi: this build has actuators with activation: pass their state act= (B, "
                             f"{aw}) float32 (MuJoCo's act; docs/api.md)")
        raise ValueError("nablaengine.jax_ffi: this build has the BAM servo model: pass its state act= (B, "
                         f"{aw}) float32 (nablaengine.actuators.bam.initial_state; docs/actuators.md)")
    if not aw and act is not None:
        raise ValueError("nablaengine.jax_ffi: act= is the state of a stateful actuator model (init(actuator='bam'), "
                         "or a model with activation dynamics); this build has none")
    if act is not None:
        _check("act", act, (B, aw))


def forward(qpos, qvel, warm, ctrl, friction, com, *, iterations=ITERATIONS, substeps=SUBSTEPS, tape=False,
            kinematics=False, contacts=False, act=None):
    """One control step without custom_vjp: (qpos', qvel', warm', info (B, info["info_words"]) int32[, contacts][, tape (B, substeps *
    REC_WORDS) float32][, qpos_n, xpos, xquat, cvel, subtree_com]); dk_step, or dk_step_fwd with tape=True;
    kinematics=True: the output state's kinematics (kinematics_forward(qpos', qvel', com), bitwise) from the same call
    (dk_step_kin / dk_fwd_kin); contacts=True: the last substep's contact table (B, maxcon * con_words) float32
    (contact_table() decodes it). act: the actuator state (B, info["act_words"]) of a BAM build (required there,
    refused elsewhere): the result then has act' after warm', (qpos', qvel', warm', act', info, ...); not with
    kinematics=True yet."""
    import jax
    import jax.numpy as jnp
    if kinematics and act is None and _forest_kin() is not None:
        r = forward(qpos, qvel, warm, ctrl, friction, com, iterations=iterations, substeps=substeps, tape=tape,
                    contacts=contacts)
        return tuple(r) + tuple(kinematics_forward(r[0], r[1], com))
    lib = _kin_lib() if kinematics else init()
    B = qpos.shape[0]
    nq, nv, nu = lib["nq"], lib["nv"], lib["nu"]
    for name, x, shape in (("qpos", qpos, (B, nq)), ("qvel", qvel, (B, nv)), ("warm", warm, (B, nv)),
                           ("ctrl", ctrl, (B, nu)), ("friction", friction, (B,)), ("com", com, (B, _com_words()))):
        _check(name, x, shape)
    _act_check(lib, act, B)
    f32 = jnp.float32
    if act is not None:
        if kinematics:
            raise ValueError("nablaengine.jax_ffi: kinematics=True is not available with the BAM servo model yet")
        outs = [jax.ShapeDtypeStruct((B, nq), f32), jax.ShapeDtypeStruct((B, nv), f32),
                jax.ShapeDtypeStruct((B, nv), f32), jax.ShapeDtypeStruct((B, _info_words()), jnp.int32),
                jax.ShapeDtypeStruct((B, con_width(lib, substeps)), f32)]
        if tape:
            outs.append(jax.ShapeDtypeStruct((B, substeps * lib["rec_words"]), f32))
        outs.append(jax.ShapeDtypeStruct(act.shape, f32))
        call = jax.ffi.ffi_call(_STATE["targets"]["fwd_act" if tape else "step_act"], outs)
        res = call(qpos, qvel, warm, ctrl, friction, com, act, iterations=np.int32(iterations),
                   substeps=np.int32(substeps))
        return (tuple(res[:3]) + (res[-1], res[3]) + ((res[4],) if contacts else ())
                + ((res[5],) if tape else ()))
    outs = [jax.ShapeDtypeStruct((B, nq), f32), jax.ShapeDtypeStruct((B, nv), f32), jax.ShapeDtypeStruct((B, nv), f32),
            jax.ShapeDtypeStruct((B, _info_words()), jnp.int32),
            jax.ShapeDtypeStruct((B, con_width(lib, substeps)), f32)]
    if tape:
        outs.append(jax.ShapeDtypeStruct((B, substeps * lib["rec_words"]), f32))
    if kinematics:
        outs += _kin_structs(lib, B)
    call = jax.ffi.ffi_call(_STATE["targets"][("fwd" if tape else "step") + ("_kin" if kinematics else "")], outs)
    res = call(qpos, qvel, warm, ctrl, friction, com, iterations=np.int32(iterations), substeps=np.int32(substeps))
    return tuple(res[:4]) + ((res[4],) if contacts else ()) + tuple(res[5:])


def contact_table(con, substeps=None):
    """The contact output of forward / step (contacts=True), (B, con_width(info, substeps)), as a dict of (B, maxcon)
    arrays ((B, substeps, maxcon) with contact_forces="substeps"): valid (pair >= 0), pair (the contact pair index,
    -1 past the world's contacts; a terrain build's virtual pairs from info["npair"] on), dist, pos (world), normal
    (world, the contact frame's first axis); with the forces also force (world; on the pair's second geom in the
    engine's geom order, the normal pointing from the first geom to the second) and force_normal (>= 0); with
    contact_frame tangent1, tangent2 (= normal x tangent1) and force_t1, force_t2 (the force's components along them);
    in a terrain build element (-1: a model pair; else the terrain element, see docs/api.md); with foot_wrench
    foot_force, foot_moment (B, substeps, nfoot, 3) and foot_fn_max (B, nfoot); with foot_normal foot_side,
    toe_weight, foot_normal, and in a TerrainConfig(terrain_sand=True) build also sand_k, foot_normal_split,
    foot_sand_presence and foot_sand_params (after set_terrain; docs/terrain.md "Sand elements"). substeps: None infers
    it from the width. As torch_ffi.contact_table (g1_engine.contact_forces.decode)."""
    import jax.numpy as jnp
    from nablaengine.g1_engine import contact_forces as _cf
    out = _cf.decode(con, init(), substeps, _STATE.get("terrain_sand"))
    out["pair"] = out["pair"].astype(jnp.int32)
    out["valid"] = out["pair"] >= 0
    return out


def kinematics_forward(qpos, qvel, com):
    """The kinematics outputs of the states (qpos (B, nq), qvel (B, nv), com (B, 3)) without custom_vjp (dk_kin):
    (qpos_n (B, nq), xpos (B, nbody, 3), xquat (B, nbody, 4), cvel (B, nbody, 6), subtree_com (B, nbody, 3)), MJX's
    mjx.kinematics + com_pos + com_vel (g1_engine.kinematics)."""
    import jax
    dq = _forest_kin()
    lib = init() if dq is not None else _kin_lib()
    B = qpos.shape[0]
    for name, x, shape in (("qpos", qpos, (B, lib["nq"])), ("qvel", qvel, (B, lib["nv"])), ("com", com, (B, _com_words()))):
        _check(name, x, shape)
    if dq is not None:
        return dq.kinematics_forward(qpos, qvel, com)
    return tuple(jax.ffi.ffi_call(_STATE["targets"]["kin"], _kin_structs(lib, B))(qpos, qvel, com))


def kinematics_backward(qpos, qvel, com, g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com):
    """The VJP of kinematics_forward at (qpos, qvel, com) for the cotangents of its five outputs: (g_qpos (B, nq),
    g_qvel (B, nv)) (dk_kin_bwd; com has none)."""
    import jax
    import jax.numpy as jnp
    dq = _forest_kin()
    lib = init() if dq is not None else _kin_lib()
    B = qpos.shape[0]
    nq, nv = lib["nq"], lib["nv"]
    for name, x, shape in (("qpos", qpos, (B, nq)), ("qvel", qvel, (B, nv)), ("com", com, (B, _com_words()))):
        _check(name, x, shape)
    gk = (g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com)
    _check_kin(lib, B, gk, "g_")
    if dq is not None:
        return dq.kinematics_backward(qpos, qvel, com, *gk)
    outs = [jax.ShapeDtypeStruct((B, nq), jnp.float32), jax.ShapeDtypeStruct((B, nv), jnp.float32)]
    return tuple(jax.ffi.ffi_call(_STATE["targets"]["kin_bwd"], outs)(qpos, qvel, com, *gk))


@functools.lru_cache(maxsize=1)
def _kinematics_fn():
    import jax
    import jax.numpy as jnp
    from jax.ad_checkpoint import checkpoint_name

    @jax.custom_vjp
    def kin(qpos, qvel, com):
        return kinematics_forward(qpos, qvel, com)

    def kin_fwd(qpos, qvel, com):
        out = tuple(checkpoint_name(x, KIN_NAME) for x in kinematics_forward(qpos, qvel, com))
        return out, (checkpoint_name(qpos, KIN_NAME), checkpoint_name(qvel, KIN_NAME), com)

    def kin_bwd(res, cot):
        qpos, qvel, com = res
        g_q, g_v = kinematics_backward(qpos, qvel, com, *cot)
        return g_q, g_v, jnp.zeros_like(com)

    kin.defvjp(kin_fwd, kin_bwd)
    return kin


def kinematics(qpos, qvel, com):
    """Differentiable kinematics outputs of states not produced by the engine (e.g. initial or externally supplied
    states): (qpos_n, xpos, xquat, cvel, subtree_com), as MJX's mjx.kinematics + smooth.com_pos + smooth.com_vel on
    the model with the torso COM offset com added to the torso's body_ipos (as step's); a jax.custom_vjp whose backward
    (dk_kin_bwd) returns the cotangents of qpos (through the quaternion's normalization: its radial part removed) and
    qvel, zeros for com. The same kernels as step(..., kinematics=True): kinematics(q', v', com) of a step's output is
    bitwise its kinematics outputs. Any B, float32; see the module docstring."""
    return _kinematics_fn()(qpos, qvel, com)


def _soft(clamp_kv, clamp_soft):
    if int(clamp_kv) != 2:
        if clamp_soft is not None:
            raise ValueError(f"nablaengine.jax_ffi: clamp_soft is the soft clamp's (clamp_kv=2), not clamp_kv={clamp_kv}")
        return None
    d = float(np.float32(CLAMP_SOFT if clamp_soft is None else clamp_soft))
    if not (np.isfinite(d) and d >= 0.0):
        raise ValueError(f"nablaengine.jax_ffi: clamp_soft must be finite and >= 0, got {clamp_soft}")
    return d


def _sur(cone_mu, impulse_clip):
    mu = float(np.float32(cone_mu or 0.0))
    if not (np.isfinite(mu) and mu >= 0.0):
        raise ValueError(f"nablaengine.jax_ffi: cone_mu must be finite and >= 0, got {cone_mu}")
    clip = np.inf if impulse_clip is None else float(np.float32(impulse_clip))
    if not clip > 0.0:
        raise ValueError(f"nablaengine.jax_ffi: impulse_clip must be > 0 (None: off), got {impulse_clip}")
    if mu == 0.0 and clip == np.inf:
        return None
    return mu, clip


def _sur_lib(lib, sur):
    if sur is not None and not lib.get("contact_surrogate"):
        raise ValueError("nablaengine.jax_ffi: cone_mu / impulse_clip need a library built with "
                         "init(contact_surrogate=True) (the forest engine)")


def backward(tape, friction, com, g_qpos, g_qvel, *, substeps=SUBSTEPS, clamp_kv=CLAMP_KV, normal=NORMAL, ctrl=None,
             clamp_soft=None, kinematics=None, act=None, g_act=None, cone_mu=0.0, impulse_clip=None,
             g_contacts=None):
    """The adjoint of one control step from its tape and the cotangents of (qpos', qvel'): (g_qpos (B, nq), g_qvel (B,
    nv), g_ctrl (B, nu), info (B,) int32: the adjoint's flags | NONFINITE_* << NONFINITE_SHIFT). clamp_kv 2 (the soft
    clamp, delta clamp_soft: None for CLAMP_SOFT) needs the step's ctrl (B, nu); 0 and 1 ignore ctrl. kinematics:
    None, or (qpos', qvel', g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com) of a step with kinematics outputs: their
    cotangents are pulled back onto (qpos', qvel') (kinematics_backward) and added to g_qpos, g_qvel in the same call,
    before the adjoint (dk_bwd_kin / dk_bwd_soft_kin); the non-finite cotangent bit then refers to those sums. A build
    with per-world parameters (init(params=)) also returns g_params (B, param_words) and g_friction (B,): the
    parameters' cotangents and the foot friction input's. A terrain build with TerrainConfig.object_velocity returns
    g_object_velocity (B, 18 K) last: the cotangents of the slots' motion words (com's object_velocity words). A BAM
    build (init(actuator="bam")) takes the step's input
    state act (B, act_words) and the cotangent g_act of its output state act' (None: zeros) and returns (g_qpos,
    g_qvel, g_ctrl, g_act, info), g_act the gradient of the input state (the torques' and the supply's); clamp_kv 0
    and 1 are the same there (the servo force has no kv), 2 and kinematics are refused. cone_mu, impulse_clip: the
    backward-only contact-gradient surrogates of a contact_surrogate build (init; forest_engine.surrogate): the
    central-path relaxation of the pyramidal cone rows' slope with barrier cone_mu ((m/s^2)^2; 0: off) and the clip of
    each contact's partials to the impulse impulse_clip (N s per substep; None: off); both off: the exact adjoint.
    g_contacts: a contact_grad build's (init(contact_grad=True)) cotangent of the step's contact output (B,
    con_width(info, substeps)), added to the gradients (forest_engine.contact_vjp; None: none)."""
    import jax
    import jax.numpy as jnp
    if kinematics is not None and act is None and _forest_kin() is not None:
        kin = tuple(kinematics)
        if len(kin) != 7:
            raise ValueError("nablaengine.jax_ffi: kinematics = (qpos', qvel', g_qpos_n, g_xpos, g_xquat, g_cvel, "
                             "g_subtree_com)")
        gq, gv = kinematics_backward(kin[0], kin[1], com, *kin[2:])
        return backward(tape, friction, com, g_qpos + gq, g_qvel + gv, substeps=substeps, clamp_kv=clamp_kv,
                        normal=normal, ctrl=ctrl, clamp_soft=clamp_soft, cone_mu=cone_mu, impulse_clip=impulse_clip)
    lib = _kin_lib() if kinematics is not None else init()
    if lib.get("act_words"):
        return _backward_act(lib, tape, friction, com, g_qpos, g_qvel, substeps, clamp_kv, normal, kinematics, act,
                             g_act)
    if act is not None or g_act is not None:
        raise ValueError("nablaengine.jax_ffi: act= / g_act= belong to a build with the BAM servo model")
    B = tape.shape[0]
    nq, nv, nu = lib["nq"], lib["nv"], lib["nu"]
    for name, x, shape in (("tape", tape, (B, substeps * lib["rec_words"])), ("friction", friction, (B,)),
                           ("com", com, (B, _com_words())), ("g_qpos", g_qpos, (B, nq)), ("g_qvel", g_qvel, (B, nv))):
        _check(name, x, shape)
    if (int(clamp_kv), int(normal)) not in VARIANTS:
        raise ValueError(f"nablaengine.jax_ffi: no adjoint variant clamp_kv={clamp_kv} normal={normal}")
    soft = _soft(clamp_kv, clamp_soft)
    sur = _sur(cone_mu, impulse_clip)
    _sur_lib(lib, sur)
    kin, sfx = (), ""
    if kinematics is not None:
        kin, sfx = tuple(kinematics), "_kin"
        if len(kin) != 7:
            raise ValueError("nablaengine.jax_ffi: kinematics = (qpos', qvel', g_qpos_n, g_xpos, g_xquat, g_cvel, "
                             "g_subtree_com)")
        _check("qpos'", kin[0], (B, nq))
        _check("qvel'", kin[1], (B, nv))
        _check_kin(lib, B, kin[2:], "g_")
        kin = (kin[0], kin[1], g_qpos, g_qvel) + kin[2:]
    else:
        kin = (g_qpos, g_qvel)
    f32 = jnp.float32
    npw = int(lib.get("param_words", 0))
    nov = int(lib.get("obj_vel_words", 0))
    nom = int(lib.get("obj_mat_words", 0))
    gw = nu + (npw + 1 if npw else 0) + nov + nom
    outs = [jax.ShapeDtypeStruct((B, nq), f32), jax.ShapeDtypeStruct((B, nv), f32),
            jax.ShapeDtypeStruct((B, gw), f32), jax.ShapeDtypeStruct((B,), jnp.int32)]
    if g_contacts is not None:
        if not lib.get("contact_grad"):
            raise ValueError("nablaengine.jax_ffi: g_contacts needs a library built with init(contact_grad=True)")
        if sur is not None:
            raise ValueError("nablaengine.jax_ffi: g_contacts with the contact-gradient surrogates")
        _check("g_contacts", g_contacts, (B, con_width(lib, substeps)))
        if soft is not None and ctrl is None:
            raise ValueError("nablaengine.jax_ffi: clamp_kv=2 (the soft clamp) needs the step's ctrl")
        if ctrl is None:
            ctrl = jnp.zeros((B, nu), f32)
        _check("ctrl", ctrl, (B, nu))
        call = jax.ffi.ffi_call(_STATE["targets"]["bwd_con" + sfx], outs)
        r = tuple(call(tape, friction, com, ctrl, *kin, g_contacts, substeps=np.int32(substeps),
                       clamp_kv=np.int32(clamp_kv), normal=np.int32(normal), clamp_soft=np.float32(soft or 0.0)))
    elif sur is not None:
        if soft is not None and ctrl is None:
            raise ValueError("nablaengine.jax_ffi: clamp_kv=2 (the soft clamp) needs the step's ctrl")
        if ctrl is None:
            ctrl = jnp.zeros((B, nu), f32)
        _check("ctrl", ctrl, (B, nu))
        call = jax.ffi.ffi_call(_STATE["targets"]["bwd_sur" + sfx], outs)
        r = tuple(call(tape, friction, com, ctrl, *kin, substeps=np.int32(substeps), clamp_kv=np.int32(clamp_kv),
                       normal=np.int32(normal), clamp_soft=np.float32(soft or 0.0), cone_mu=np.float32(sur[0]),
                       impulse_clip=np.float32(sur[1])))
    elif soft is not None:
        if ctrl is None:
            raise ValueError("nablaengine.jax_ffi: clamp_kv=2 (the soft clamp) needs the step's ctrl")
        _check("ctrl", ctrl, (B, nu))
        call = jax.ffi.ffi_call(_STATE["targets"]["bwd_soft" + sfx], outs)
        r = tuple(call(tape, friction, com, ctrl, *kin, substeps=np.int32(substeps), normal=np.int32(normal),
                       clamp_soft=np.float32(soft)))
    else:
        call = jax.ffi.ffi_call(_STATE["targets"]["bwd" + sfx], outs)
        r = tuple(call(tape, friction, com, *kin, substeps=np.int32(substeps), clamp_kv=np.int32(clamp_kv),
                       normal=np.int32(normal)))
    ov = (r[2][:, gw - nom - nov:gw - nom],) if nov else ()
    ov += (r[2][:, gw - nom:],) if nom else ()
    if npw:
        return (r[0], r[1], r[2][:, :nu], r[3], r[2][:, nu:nu + npw], r[2][:, nu + npw]) + ov
    if nov or nom:
        return (r[0], r[1], r[2][:, :nu], r[3]) + ov
    return r


def _backward_act(lib, tape, friction, com, g_qpos, g_qvel, substeps, clamp_kv, normal, kinematics, act, g_act):
    import jax
    import jax.numpy as jnp
    if kinematics is not None:
        raise ValueError("nablaengine.jax_ffi: kinematics outputs are not available with the BAM servo model yet")
    if int(clamp_kv) == 2:
        raise ValueError("nablaengine.jax_ffi: the BAM servo model's adjoint takes clamp_kv 0 or 1 (the same there: "
                         "its servo force has no kv), not the soft clamp 2")
    if act is None:
        raise ValueError("nablaengine.jax_ffi: this build has the BAM servo model: backward needs the step's input "
                         "state act= (B, act_words)")
    B = tape.shape[0]
    nq, nv, nu, aw = lib["nq"], lib["nv"], lib["nu"], int(lib["act_words"])
    if g_act is None:
        g_act = jnp.zeros((B, aw), jnp.float32)
    for name, x, shape in (("tape", tape, (B, substeps * lib["rec_words"])), ("friction", friction, (B,)),
                           ("com", com, (B, _com_words())), ("g_qpos", g_qpos, (B, nq)), ("g_qvel", g_qvel, (B, nv)),
                           ("act", act, (B, aw)), ("g_act", g_act, (B, aw))):
        _check(name, x, shape)
    if (int(clamp_kv), int(normal)) not in VARIANTS:
        raise ValueError(f"nablaengine.jax_ffi: no adjoint variant clamp_kv={clamp_kv} normal={normal}")
    f32 = jnp.float32
    outs = [jax.ShapeDtypeStruct((B, nq), f32), jax.ShapeDtypeStruct((B, nv), f32), jax.ShapeDtypeStruct((B, nu), f32),
            jax.ShapeDtypeStruct((B,), jnp.int32), jax.ShapeDtypeStruct((B, aw), f32)]
    call = jax.ffi.ffi_call(_STATE["targets"]["bwd_act"], outs)
    r = tuple(call(tape, friction, com, act, g_qpos, g_qvel, g_act, substeps=np.int32(substeps),
                   clamp_kv=np.int32(clamp_kv), normal=np.int32(normal)))
    return r[0], r[1], r[2], r[4], r[3]


def make_step(iterations=ITERATIONS, substeps=SUBSTEPS, clamp_kv=CLAMP_KV, normal=NORMAL, clamp_soft=None,
              grad_decay=GRAD_DECAY, kinematics=False, contacts=False, cone_mu=0.0, impulse_clip=None):
    """step(qpos, qvel, warm, ctrl, friction, com) -> (qpos', qvel', warm') with these static options, a
    jax.custom_vjp function (one per option set, so jit caches stay valid: the options are normalized first, so every
    spelling of a set gives the same function). kinematics=True: -> (qpos', qvel', warm', qpos_n, xpos, xquat,
    cvel, subtree_com), the output state's kinematics in the same call and the same VJP. cone_mu, impulse_clip: the
    contact-gradient surrogates of the backward (backward(); a contact_surrogate build)."""
    if (int(clamp_kv), int(normal)) not in VARIANTS:
        raise ValueError(f"nablaengine.jax_ffi: no adjoint variant clamp_kv={clamp_kv} normal={normal}")
    gd = float(np.float32(grad_decay))
    if not np.isfinite(gd):
        raise ValueError(f"nablaengine.jax_ffi: grad_decay must be finite, got {grad_decay}")
    if int(init().get("act_words", 0)):
        if kinematics:
            raise ValueError("nablaengine.jax_ffi: kinematics=True is not available with the BAM servo model yet")
        if int(clamp_kv) == 2:
            raise ValueError("nablaengine.jax_ffi: the BAM servo model's adjoint takes clamp_kv 0 or 1 (the same "
                             "there: its servo force has no kv), not the soft clamp 2")
        return _make_step_act(int(iterations), int(substeps), int(clamp_kv), int(normal), gd, bool(contacts))
    sur = _sur(cone_mu, impulse_clip)
    _sur_lib(init(), sur)
    return _make_step(int(iterations), int(substeps), int(clamp_kv), int(normal), _soft(clamp_kv, clamp_soft), gd,
                      bool(kinematics), bool(contacts), *(sur if sur is not None else ()))


@functools.lru_cache(maxsize=None)
def _make_step_act(iterations, substeps, clamp_kv, normal, grad_decay, con=False):
    import jax
    import jax.numpy as jnp
    gd = np.float32(grad_decay)

    @jax.custom_vjp
    def step_fn(qpos, qvel, warm, ctrl, friction, com, act):
        r = forward(qpos, qvel, warm, ctrl, friction, com, iterations=iterations, substeps=substeps, contacts=con,
                    act=act)
        return tuple(r[:4]) + ((r[5],) if con else ())

    def step_fwd(qpos, qvel, warm, ctrl, friction, com, act):
        from jax.ad_checkpoint import checkpoint_name
        r = forward(qpos, qvel, warm, ctrl, friction, com, iterations=iterations, substeps=substeps, tape=True,
                    contacts=con, act=act)
        outs = tuple(checkpoint_name(x, OUT_NAME) for x in r[:4]) + ((r[5],) if con else ())
        return outs, (checkpoint_name(r[-1], TAPE_NAME), friction, com, act)

    def step_bwd(res, cot):
        tp, friction, com, act = res
        g_q, g_v, g_a = cot[0], cot[1], cot[3]
        dq, dv, du, da, _ = backward(tp, friction, com, g_q, g_v, substeps=substeps, clamp_kv=clamp_kv, normal=normal,
                                     act=act, g_act=g_a)
        if grad_decay != 1.0:
            dq, dv, da = dq * gd, dv * gd, da * gd
        return dq, dv, jnp.zeros_like(dv), du, jnp.zeros_like(friction), jnp.zeros_like(com), da

    step_fn.defvjp(step_fwd, step_bwd)
    return step_fn


@functools.lru_cache(maxsize=None)
def _make_step(iterations, substeps, clamp_kv, normal, clamp_soft, grad_decay, kin=False, con=False, cone_mu=0.0,
               impulse_clip=None):
    import jax
    import jax.numpy as jnp
    soft = clamp_soft is not None
    gd = np.float32(grad_decay)

    @jax.custom_vjp
    def step_fn(qpos, qvel, warm, ctrl, friction, com):
        r = forward(qpos, qvel, warm, ctrl, friction, com, iterations=iterations, substeps=substeps, kinematics=kin,
                    contacts=con)
        return tuple(r[:3]) + tuple(r[4:])

    def step_fwd(qpos, qvel, warm, ctrl, friction, com):
        from jax.ad_checkpoint import checkpoint_name
        r = forward(qpos, qvel, warm, ctrl, friction, com, iterations=iterations, substeps=substeps, tape=True,
                    kinematics=kin, contacts=con)
        if con:
            ctab = r[4]
            r = tuple(r[:4]) + tuple(r[5:])
        outs = tuple(checkpoint_name(x, OUT_NAME) for x in tuple(r[:3]) + tuple(r[5:]))
        if con:
            outs = outs[:3] + (ctab,) + outs[3:]
        res = (checkpoint_name(r[4], TAPE_NAME), friction, com) + (outs[:2] if kin else ()) + ((ctrl,) if soft else ())
        return outs, res

    cgrad = con and bool(_STATE["info"].get("contact_grad"))

    def step_bwd(res, cot):
        tp, friction, com = res[:3]
        g_q, g_v = cot[:2]
        kc = tuple(res[3:5]) + tuple(cot[3 + con:]) if kin else None
        r = backward(tp, friction, com, g_q, g_v, substeps=substeps, clamp_kv=clamp_kv, normal=normal,
                     ctrl=res[-1] if soft else None, clamp_soft=clamp_soft, kinematics=kc, cone_mu=cone_mu,
                     impulse_clip=impulse_clip, **({"g_contacts": cot[3]} if cgrad else {}))
        dq, dv, du = r[:3]
        if grad_decay != 1.0:
            dq, dv = dq * gd, dv * gd
        gcom, gfr = jnp.zeros_like(com), jnp.zeros_like(friction)
        npw, nov = int(_STATE["info"].get("param_words", 0)), int(_STATE["info"].get("obj_vel_words", 0))
        nom = int(_STATE["info"].get("obj_mat_words", 0))
        if npw:
            gcom = gcom.at[:, com.shape[1] - npw:].set(r[4])
            gfr = r[5]
        if nov:
            gcom = gcom.at[:, com.shape[1] - npw - nom - nov:com.shape[1] - npw - nom].set(r[-1 - (nom > 0)])
        if nom:
            gcom = gcom.at[:, com.shape[1] - npw - nom:com.shape[1] - npw].set(r[-1])
        return dq, dv, jnp.zeros_like(dv), du, gfr, gcom

    step_fn.defvjp(step_fwd, step_bwd)
    return step_fn


def step(qpos, qvel, warm, ctrl, friction, com, *, iterations=ITERATIONS, substeps=SUBSTEPS, clamp_kv=CLAMP_KV,
         normal=NORMAL, clamp_soft=None, grad_decay=GRAD_DECAY, kinematics=False, contacts=False, act=None, cone_mu=0.0,
         impulse_clip=None):
    """One control step, differentiable (see the module docstring): (qpos', qvel', warm'); kinematics=True:
    (qpos', qvel', warm', qpos_n, xpos, xquat, cvel, subtree_com), the kinematics of the output state (qpos', qvel')
    computed in the same FFI call after the last substep, their cotangents in the same VJP; contacts=True: the last
    substep's contact table after warm' (forward's; not differentiated: its cotangent is ignored). act: the BAM
    servo model's state (B, info["act_words"]) in a BAM build (required there, refused elsewhere): (qpos', qvel',
    warm', act'[, contacts]), differentiable in act too (docs/actuators.md). cone_mu, impulse_clip: the backward's
    contact-gradient surrogates (backward(); a contact_surrogate build)."""
    fn = make_step(iterations, substeps, clamp_kv, normal, clamp_soft, grad_decay, kinematics, contacts, cone_mu,
                   impulse_clip)
    aw = int(init().get("act_words", 0))
    if aw or act is not None:
        import jax.numpy as jnp
        _act_check(init(), None if act is None else jnp.asarray(act), qpos.shape[0])
        return fn(qpos, qvel, warm, ctrl, friction, com, act)
    return fn(qpos, qvel, warm, ctrl, friction, com)


def scale_grad(tree, gamma):
    """The identity on a pytree whose cotangent the backward multiplies by gamma (a Python float, static; the multiply
    in each leaf's dtype): the gradient decay of the module docstring, for any differentiable function. Only the leaves
    with an inexact dtype (float, complex) go through the custom_vjp; the others (ints, bools, PRNG keys, Python
    scalars and other non-array leaves) are returned untouched, so it can wrap a whole carried state (the full-state
    form of the module docstring: at the top of the scan body, every derivative across n step boundaries gets
    gamma^n). On the state arguments of a physics call
    alone (qpos, qvel: the physics-only form) it gives the gradients of make_step(grad_decay=gamma) bitwise
    (scripts/jax_ffi_check.py, section options). gamma = 1.0 returns tree itself."""
    g = float(gamma)
    if not np.isfinite(g):
        raise ValueError(f"nablaengine.jax_ffi: scale_grad's gamma must be finite, got {gamma}")
    if g == 1.0:
        return tree
    import jax
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    idx = [i for i, x in enumerate(leaves)
           if hasattr(x, "dtype") and jax.dtypes.issubdtype(x.dtype, np.inexact)]
    if not idx:
        return tree
    scaled = _scale_grad_fn(g)([leaves[i] for i in idx])
    for i, x in zip(idx, scaled):
        leaves[i] = x
    return jax.tree_util.tree_unflatten(treedef, leaves)


@functools.lru_cache(maxsize=None)
def _scale_grad_fn(gamma):
    import jax

    @jax.custom_vjp
    def ident(tree):
        return tree

    def fwd(tree):
        return tree, None

    def bwd(_, cot):
        return (jax.tree_util.tree_map(lambda c: c * np.asarray(gamma, c.dtype), cot),)

    ident.defvjp(fwd, bwd)
    return ident


def remat_policy(*extra_names):
    """The jax.checkpoint policy for a BPTT body that calls step: save the tape and the step's outputs (REMAT_NAMES,
    plus extra_names of the caller's own checkpoint_name values), recompute the rest. A plain jax.checkpoint(body)
    saves nothing, so the backward reruns dk_step_fwd for the tape (2 forward calls per control step); with this policy
    each step runs dk_step_fwd once; the gradients equal those without remat up to XLA's rounding of the recomputed body
    (~1e-7 relative; bitwise for jax_ffi_check's rollout body). step(..., kinematics=True)'s kinematics outputs are
    named OUT_NAME with the step's outputs, jax_ffi.kinematics's outputs and residual state KIN_NAME ("dk_kin"): saved
    too, so neither kernel reruns in the backward (step: B x 532 float32 per call, 8.7 MB at B = 4096; kinematics
    called inside the body: B x 603, its residual qpos, qvel included, 9.9 MB)."""
    import jax
    return jax.checkpoint_policies.save_only_these_names(*REMAT_NAMES, *extra_names)


def launch_info(device=0):
    """{variant tag: {smem, worlds_per_sm}}: k_adjoint's launch on the CUDA device ordinal (adjoint.adj_launch_smem,
    set at the device's first call; zeros before)."""
    init()
    o = np.zeros(2 * len(VARIANTS), np.int64)
    rc = _STATE["lib"].dk_launch_info(int(device), ctypes.c_void_p(o.ctypes.data))
    if rc:
        raise RuntimeError(f"nablaengine.jax_ffi: dk_launch_info failed ({rc})")
    return {_tag(v): dict(smem=int(o[2 * i]), worlds_per_sm=int(o[2 * i + 1])) for i, v in enumerate(VARIANTS)}


def diagnostics(*deps, reset=True):
    """The counters (NDIAG, 2) uint32 (low, high words; decode_diagnostics), read on the GPU after the arrays deps are
    computed (inside jit: pass what must run first, e.g. the gradients); reset=True zeroes them."""
    import jax
    import jax.numpy as jnp
    init()
    call = jax.ffi.ffi_call(_STATE["targets"]["diag"], jax.ShapeDtypeStruct((len(DIAG), 2), jnp.uint32),
                            has_side_effect=True)
    return call(*deps, reset=np.int32(bool(reset)))


def decode_diagnostics(arr):
    """(NDIAG, 2) uint32 -> {name: int}; with set_overflow_warnings(True) it warns (warn_overflow) when worlds
    exceeded the contact or row capacity."""
    a = np.asarray(arr).astype(np.uint64)
    d = {name: int(a[i, 0] | (a[i, 1] << np.uint64(32))) for i, name in enumerate(DIAG)}
    if _OVERFLOW_WARN[0]:
        warn_overflow(d)
    return d


def warn_overflow(d, where="nablaengine", maxcon=None, nefc=None):
    """A RuntimeWarning when the counters d count worlds whose contacts or constraint rows exceeded the per-world
    capacity (maxcon / nefc, default the loaded library's info["maxcon"] / info["nefc"]): the ones beyond were dropped,
    so those worlds miss support and friction forces. The worlds of a call: info[:, 3] & OVERFLOW_CONTACTS /
    OVERFLOW_ROWS of forward(). init(max_contacts=, max_rows=) raises the capacities."""
    import warnings
    nc, nr = d.get("fwd_contact_overflow_worlds", 0), d.get("fwd_row_overflow_worlds", 0)
    info = _STATE["info"] if _STATE else {}
    maxcon, nefc = maxcon or info.get("maxcon", 64), nefc or info.get("nefc", 192)
    if nc or nr:
        warnings.warn(f"{where}: {nc} world-steps exceeded the contact capacity ({maxcon} per world) and {nr} the "
                      f"constraint-row capacity ({nefc}); their extra contacts / rows were dropped, so those worlds miss "
                      f"support and friction forces (e.g. deep interpenetration; init(max_contacts=, max_rows=) raises "
                      f"the capacities). See docs/api.md, Diagnostics.", RuntimeWarning, stacklevel=3)
    return nc + nr


def capacity_counts(info):
    """(contacts_found, rows_needed), two (B,) int32 arrays, from forward()'s info of a build with
    init(capacity_diag=True) (info (B, 6)): per world the largest over the call's substeps of the contacts the
    collision found (those beyond info["maxcon"] included) and of the constraint rows the assembly needed (dof,
    equality and contact rows, the dropped contacts' rows included), to size max_contacts / max_rows. Where they
    exceed info["maxcon"] / info["nefc"] the extra contacts / rows were dropped (OVERFLOW_CONTACTS / OVERFLOW_ROWS).
    Traceable; ValueError for the (B, 4) info of a build without the option (forest_engine.capacity_diag)."""
    if len(info.shape) != 2 or info.shape[1] != 6:
        raise ValueError(f"nablaengine.jax_ffi.capacity_counts: info of shape {tuple(info.shape)}; the counts need a "
                         "build with init(capacity_diag=True), whose forward info is (B, 6)")
    return info[:, 4], info[:, 5]


def set_overflow_warnings(enabled):
    """Switch the capacity-overflow warnings on or off (default off, or $DK_OVERFLOW_WARN=1): when on,
    read_diagnostics / decode_diagnostics warn (warn_overflow) if worlds exceeded the per-world contact or
    constraint-row capacity, and the counts nobody read are reported once more at exit. The counters and info's bits
    are kept either way. Returns the previous setting."""
    prev = _OVERFLOW_WARN[0]
    _OVERFLOW_WARN[0] = bool(enabled)
    return prev


def _exit_report():
    try:
        if _OVERFLOW_WARN[0] and _STATE and STATS.get("loads"):
            read_diagnostics(reset=False)
    except Exception:
        pass


@functools.lru_cache(maxsize=2)
def _diag_fn(reset):
    import jax
    return jax.jit(lambda: diagnostics(reset=reset))


def read_diagnostics(reset=True):
    """{name: int} of the counters since the last reset (host sync; call outside jit, after the work to count)."""
    init()
    return decode_diagnostics(_diag_fn(bool(reset))())


@functools.lru_cache(maxsize=1)
def _last_fn():
    import jax
    import jax.numpy as jnp
    return jax.jit(lambda: jax.ffi.ffi_call(_STATE["targets"]["last"], jax.ShapeDtypeStruct((NSLOT, len(LAST)),
                                                                                             jnp.uint32),
                                            has_side_effect=True)())


def last_forward(nworld=None):
    """{LAST name: int} of the last forward call (dk_step or dk_step_fwd) on the device with batch size nworld (None:
    of any batch size); zeros before the first such call. A read on the GPU's stream after the work already
    dispatched (host sync; call outside jit), without reset."""
    init()
    slots = np.zeros(NSLOT, np.int64)
    _STATE["lib"].dk_last_slots(ctypes.c_void_p(slots.ctypes.data))
    rec = np.asarray(_last_fn()())
    if nworld is None:
        s = 0
    else:
        hit = np.flatnonzero(slots[1:] == int(nworld))
        if not len(hit):
            return {name: 0 for name in LAST}
        s = int(hit[0]) + 1
    return {name: int(rec[s, i]) for i, name in enumerate(LAST)}


def check_model(mjm, foot_pairs=None, torso_id=None):
    """Raise ValueError unless the MjModel mjm (with its foot-floor pair indices and torso body id) is a model its
    engine supports (engines.require: nablaengine.model.capability's G1 engine scope or forest scope, UnsupportedModel,
    a ValueError, lists the failing checks; the packed constants below cannot see physics the engine does not model,
    such as joint damping or the integrator) and the model whose constants the library holds (init()): the engine,
    nq / nv / nu, the timestep, the foot pairs (in order), the torso id (the COM-offset body's; -1 for none) and the
    packed physics constants (the engine's adjoint.engine_constants with the library's COM-offset body, compared
    bitwise; the solver iterations are an option of the call, not a constant). Returns the library info."""
    info = init()
    fp = info["foot_pairs"] if foot_pairs is None else [int(p) for p in np.asarray(foot_pairs).ravel()]
    if _STATE.get("bam") is not None:
        from nablaengine._optional import BAM, load
        _bam = load("nablaengine.actuators", BAM, "bam")
        mjm = _bam.prepare(mjm, _STATE["bam"])
    E = engines.modules(mjm)
    com = info.get("com_body", capability.COM_BODY)
    if E.name != info.get("engine", engines.G1):
        raise ValueError(f"nablaengine.jax_ffi: the model is one for the {E.name} engine, the library's is for the "
                         f"{info.get('engine')} engine (nablaengine.engines; init(model=, foot_pairs=) builds it for "
                         f"this model)")
    engines.require(mjm, fp, com, "jax_ffi.check_model")
    msg = []
    if (mjm.nq, mjm.nv, mjm.nu) != (info["nq"], info["nv"], info["nu"]):
        msg.append(f"nq / nv / nu {(mjm.nq, mjm.nv, mjm.nu)} != {(info['nq'], info['nv'], info['nu'])}")
    if float(mjm.opt.timestep) != info["timestep"]:
        msg.append(f"timestep {mjm.opt.timestep} != {info['timestep']}")
    if fp != info["foot_pairs"]:
        msg.append(f"foot pairs {fp} != {info['foot_pairs']}")
    if torso_id is not None and int(torso_id) != info["torso_id"]:
        msg.append(f"torso id {int(torso_id)} != {info['torso_id']}")
    if not msg:
        try:
            consts, lays, sizes = (E.adjoint.engine_constants(mjm, fp, com) if E.forest
                                   else E.adjoint.engine_constants(mjm, fp, None, com))
        except AssertionError as e:
            raise ValueError(f"nablaengine.jax_ffi: the {E.name} engine's constant packing rejects the model "
                             f"({e!r})") from None
        mine = _STATE["consts"]
        if len(consts) != len(mine):
            msg.append(f"{len(consts)} packed constants != {len(mine)}")
        elif not np.array_equal(consts.view(np.uint32), np.asarray(mine, np.float32).view(np.uint32)):
            names, at = [], 0
            for sym, kind, count, width in lays:
                n = (sizes[count] if isinstance(count, str) else int(count)) * width * (2 if kind == "u" else 1)
                if not np.array_equal(consts[at:at + n].view(np.uint32), np.asarray(mine[at:at + n]).view(np.uint32)):
                    names.append(sym)
                at += n
            msg.append(f"packed constants differ ({', '.join(names) or 'layout'})")
    if msg:
        raise ValueError("nablaengine.jax_ffi: the model differs from the library's (the fixtures' unless "
                         "init(model=...)): " + "; ".join(msg) + ". Use the same mjlab model, or EnginePhysics(..., "
                         "use_model=True) / init(model=, foot_pairs=) to build the library for this one")
    return info


class EnginePhysics:
    """A batched physics object for a model: EnginePhysics(mjm, nworld, foot_pairs, torso_id, ...) with .step(qpos,
    qvel, warm, ctrl, friction, com), .stats() (the last forward call's high-water marks: nacon, naconmax, nefc_max,
    njmax), .nconmax / .njmax (the engine's per-world contact and constraint-row capacities) and .diagnostics() (the
    cumulative counters). nworld is the batch size stats() reports (the library serves any B); iterations default to
    mjm.opt.iterations; clamp_kv, normal, clamp_soft, grad_decay and kinematics as make_step's (grad_decay: the
    physics-only form of the gradient decay, see the module docstring; kinematics=True: .step also returns the output
    state's qpos_n, xpos, xquat, cvel, subtree_com). The model constants come from the fixtures (init()), and mjm (with
    foot_pairs and torso_id) must then be the same model (check_model, else ValueError), unless use_model=True: then
    from mjm, for either engine (nablaengine.engines: the G1 engine or the forest engine, info["engine"]; torso_id is
    the COM-offset body's id, -1 for none, the forest engine's default). Either way a model its engine does not
    support raises nablaengine.model.capability.UnsupportedModel (a ValueError) before anything runs. Other keyword options (those of a MuJoCo Warp based physics object, such as
    graph modes, capacities or a warm-started recompute) are accepted and ignored: the engine has none of them, and
    its tape replaces the recompute. params: per-world physics parameters (nablaengine.params.PhysicsParams or field
    names, e.g. ("foot_solref", "foot_solimp", "dof_damping")): .step then takes them as rand=, a (B, width) array
    (.pack_rand(**values); .rand_layout(): {field: (offset, width)}), and com stays (B, com_words - param_words): (B, 3)
    without engine extras (info["com_layout"]). max_contacts, max_rows:
    the engine's contact and constraint-row capacity per world (init(); MuJoCo Warp's nconmax / njmax options are
    still ignored: their meaning differs between versions). actuator: an actuator model (init(); nablaengine.actuators,
    e.g. "unitree_g1", or "bam": then .step(qpos, qvel, warm, ctrl, friction, com, act) -> (qpos', qvel', warm',
    act'), .initial_act(); None: off); motor_limits: its permanent alias. contact_forces, foot_wrench, contact_frame: the
    contact output's options (init()); contacts=True: .step also returns the contact output after warm' (make_step;
    contact_table() decodes it).
    cone_mu, impulse_clip: the backward's contact-gradient surrogates (backward(); either set builds the library with
    init(contact_surrogate=True), the forest engine only)."""

    def __init__(self, mjm=None, nworld=None, foot_pairs=None, torso_id=None, *, iterations=None, substeps=SUBSTEPS,
                 clamp_kv=CLAMP_KV, normal=NORMAL, clamp_soft=None, grad_decay=GRAD_DECAY, kinematics=False,
                 use_model=False, params=None, max_contacts=None, max_rows=None, motor_limits=None,
                 contact_forces=False, actuator=None, cone_mu=0.0, impulse_clip=None, foot_wrench=None,
                 contact_frame=False, contacts=False, **_ignored):
        from nablaengine.params import as_params
        self.params = as_params(params)
        kw = {"params": self.params} if self.params else {}
        kw.update({k: v for k, v in (("max_contacts", max_contacts), ("max_rows", max_rows),
                                     ("motor_limits", motor_limits), ("actuator", actuator)) if v is not None})
        if contact_forces:
            kw["contact_forces"] = contact_forces
        if foot_wrench not in (None, False):
            kw["foot_wrench"] = foot_wrench
        if contact_frame:
            kw["contact_frame"] = True
        if _sur(cone_mu, impulse_clip) is not None:
            kw["contact_surrogate"] = True
        if use_model:
            self.info = init(model=mjm, foot_pairs=foot_pairs, **kw)
            if torso_id is not None:
                check_model(mjm, foot_pairs, torso_id)
        else:
            self.info = init(**kw)
            if mjm is not None:
                check_model(mjm, foot_pairs, torso_id)
        self.nworld = None if nworld is None else int(nworld)
        self.nconmax, self.njmax = int(self.info["maxcon"]), int(self.info["nefc"])
        self.iterations = int(iterations or (mjm.opt.iterations if mjm is not None else ITERATIONS))
        self.options = dict(iterations=self.iterations, substeps=int(substeps), clamp_kv=int(clamp_kv),
                            normal=int(normal), clamp_soft=_soft(clamp_kv, clamp_soft),
                            grad_decay=float(np.float32(grad_decay)), kinematics=bool(kinematics),
                            cone_mu=float(np.float32(cone_mu or 0.0)),
                            impulse_clip=None if impulse_clip is None else float(np.float32(impulse_clip)))
        self.step = make_step(self.iterations, int(substeps), int(clamp_kv), int(normal), clamp_soft, grad_decay,
                              kinematics, cone_mu=cone_mu, impulse_clip=impulse_clip,
                              **({"contacts": True} if contacts else {}))
        if self.params:
            import jax.numpy as jnp
            inner = self.step

            def step(qpos, qvel, warm, ctrl, friction, com, rand):
                return inner(qpos, qvel, warm, ctrl, friction, jnp.concatenate([com, rand], axis=-1))
            self.step = step

    def initial_act(self, batch=None, vin0=None, drop=0.0):
        """A BAM build's initial actuator state rows (B, info["act_words"]) float32 (actuators.bam.initial_state):
        zero torques, supply vin0 (default the servo model's nominal voltage), drop gain drop; .step then takes it as
        its last argument and returns act' after warm'."""
        if not self.info.get("act_words"):
            raise ValueError("nablaengine.jax_ffi.EnginePhysics: this build has no actuator state (actuator='bam')")
        from nablaengine._optional import BAM, load
        _bam = load("nablaengine.actuators", BAM, "bam")
        B = int(batch if batch is not None else self.nworld)
        return _bam.initial_state(self.info["nv"], B, self.info["bam"]["vin_nom"] if vin0 is None else vin0, drop)

    def rand_layout(self):
        """{field: (offset, width)} of .step's rand rows (per-world parameters)."""
        return self.params.layout(int(self.info["nv"]))

    def pack_rand(self, **values):
        """.step's rand (B, width) float32 from per-field arrays (nablaengine.params.PhysicsParams.pack)."""
        return self.params.pack(int(self.info["nv"]), **values)

    def stats(self):
        """High-water marks of the last forward call with this batch size (host sync; call outside jit), with MuJoCo
        Warp's names: nacon (contacts of the last substep, summed over the worlds), naconmax (nconmax x worlds),
        nefc_max (constraint rows of the most loaded world), njmax. A read: nothing is reset."""
        r = last_forward(self.nworld)
        nw = self.nworld if self.nworld is not None else r["worlds"]
        return {"nacon": r["contacts"], "naconmax": self.nconmax * nw, "nefc_max": r["rows_max"], "njmax": self.njmax}

    def diagnostics(self, reset=True):
        """The cumulative diagnostic counters since the last reset (read_diagnostics; host sync, call outside jit)."""
        return read_diagnostics(reset)

"""The fused CUDA backward of the ray gradients (sensors/raygrad): one block per world sums every ray's contribution to
the cotangents of the sensor frame (pos 3, rot 9), of each body (position 3, rotation matrix 9) and of each object
slot (position 3, rotation 9) in shared memory; no per-ray tangent is materialised."""
import ctypes
import hashlib
import os
import subprocess

from nablaengine.sensors import raycore

SOURCE = r"""
#include <cuda_runtime.h>
#include <math.h>
struct VjpArgs {
  int B, N, nbody, K, nmesh, ncoef, ptype;
  float p0, p1, min_cos;
  float acam[3], awld[3], light[3];
};
__device__ __forceinline__ float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
__device__ __forceinline__ void mv3(const float* A, const float* v, float* o) {
  for (int i = 0; i < 3; i++) o[i] = A[3 * i] * v[0] + A[3 * i + 1] * v[1] + A[3 * i + 2] * v[2];
}
__device__ __forceinline__ void mtv3(const float* A, const float* v, float* o) {
  for (int i = 0; i < 3; i++) o[i] = A[i] * v[0] + A[3 + i] * v[1] + A[6 + i] * v[2];
}
__device__ __forceinline__ void qmat(const float* q, float* R) {  // as G1Sensors._qmat (not normalised)
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - z * w); R[2] = 2 * (x * z + y * w);
  R[3] = 2 * (x * y + z * w); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - x * w);
  R[6] = 2 * (x * z - y * w); R[7] = 2 * (y * z + x * w); R[8] = 1 - 2 * (x * x + y * y);
}
#define VJP_BLOCK 256
__global__ void __launch_bounds__(VJP_BLOCK) k_ray_vjp(VjpArgs A, const float* pos, const float* rot,
    const float* off, const float* dir, const float* xpos, const float* xquat, const float* obj, const int* mbody,
    const float* dist, const int* gid, const float* nrm, const float* coef, float* out) {
  extern __shared__ float sh[];
  const int w = blockIdx.x, S = 12 + 12 * (A.nbody + A.K);
  for (int i = threadIdx.x; i < S; i += blockDim.x) sh[i] = 0.f;
  __syncthreads();
  const float* P = pos + 3 * w;
  const float* R = rot + 9 * w;
  float acc[12];
  for (int k = 0; k < 12; k++) acc[k] = 0.f;
  for (int n = threadIdx.x; n < A.N; n += blockDim.x) {
    const long long id = (long long)w * A.N + n;
    const float t = dist[id];
    if (!(t >= 0.f)) continue;
    const float* cf = coef + id * A.ncoef;
    const float at = cf[0], av = A.ncoef > 1 ? cf[1] : 0.f, bu = A.ncoef > 1 ? cf[2] : 0.f,
                bv = A.ncoef > 1 ? cf[3] : 0.f, an = A.ncoef > 4 ? cf[4] : 0.f;
    if (at == 0.f && av == 0.f && bu == 0.f && bv == 0.f && an == 0.f) continue;
    const float nn[3] = {nrm[3 * id], nrm[3 * id + 1], nrm[3 * id + 2]};
    float xcam[3], xc[3], r[3];
    for (int k = 0; k < 3; k++) xcam[k] = off[3 * n + k] + t * dir[3 * n + k];
    mv3(R, xcam, xc);
    mv3(R, dir + 3 * n, r);
    const float den = dot3(nn, r);
    const float ft = fabsf(den) >= A.min_cos ? at / den : 0.f;
    float s[3] = {av * A.acam[0], av * A.acam[1], av * A.acam[2]};
    if (bu != 0.f || bv != 0.f) {
      if (A.ptype == 0) {
        const float iX = 1.f / xcam[0];
        s[0] += (bu * A.p0 * xcam[1] + bv * A.p1 * xcam[2]) * iX * iX;
        s[1] -= bu * A.p0 * iX;
        s[2] -= bv * A.p1 * iX;
      } else {
        s[0] += bu * A.p0;
        s[1] += bv * A.p0;
      }
    }
    float Rs[3], gx[3];
    mv3(R, s, Rs);
    for (int k = 0; k < 3; k++) gx[k] = ft * nn[k] + Rs[k] + av * A.awld[k];
    for (int i = 0; i < 3; i++) {
      acc[i] -= gx[i];
      for (int j = 0; j < 3; j++) acc[3 + 3 * i + j] += xc[i] * s[j] - ft * nn[i] * xcam[j];
    }
    const int g = gid[id], cat = g >> 24, idx = g & 0xFFFFFF;
    int slot = -1;
    const float* sp = nullptr;
    float RS[9];
    if (cat == 3 && idx < A.K) {
      slot = A.nbody + idx;
      sp = obj + ((size_t)w * A.K + idx) * 15;
      for (int k = 0; k < 9; k++) RS[k] = sp[3 + k];
    } else if (cat == 4 && idx < A.nmesh) {
      const int b = mbody[idx];
      slot = b;
      sp = xpos + ((size_t)w * A.nbody + b) * 3;
      qmat(xquat + ((size_t)w * A.nbody + b) * 4, RS);
    }
    if (slot >= 0) {
      float d[3], loc[3], nl[3];
      for (int k = 0; k < 3; k++) d[k] = P[k] + xc[k] - sp[k];
      mtv3(RS, d, loc);
      mtv3(RS, nn, nl);
      float* o = sh + 12 + 12 * slot;
      for (int i = 0; i < 3; i++) {
        atomicAdd(o + i, gx[i]);
        for (int j = 0; j < 3; j++) atomicAdd(o + 3 + 3 * i + j, gx[i] * loc[j] - an * A.light[i] * nl[j]);
      }
    }
  }
  for (int k = 0; k < 12; k++) {
    float v = acc[k];
    for (int m = 16; m > 0; m >>= 1) v += __shfl_down_sync(0xffffffffu, v, m);
    if ((threadIdx.x & 31) == 0) atomicAdd(sh + k, v);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < S; i += blockDim.x) out[(size_t)w * S + i] = sh[i];
}
extern "C" int dk_ray_vjp(cudaStream_t st, const VjpArgs* A, const float* pos, const float* rot, const float* off,
                          const float* dir, const float* xpos, const float* xquat, const float* obj, const int* mbody,
                          const float* dist, const int* gid, const float* nrm, const float* coef, float* out) {
  if (A->B <= 0) return 0;
  const size_t shm = sizeof(float) * (12 + 12 * (size_t)(A->nbody + A->K));
  if (shm > 48 * 1024) return 2;
  k_ray_vjp<<<A->B, VJP_BLOCK, shm, st>>>(*A, pos, rot, off, dir, xpos, xquat, obj, mbody, dist, gid, nrm, coef, out);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
extern "C" int dk_ray_vjp_args_size() { return (int)sizeof(VjpArgs); }
"""

FFI_SOURCE = r"""
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;
using BufF = ffi::Buffer<ffi::F32>;
using BufI = ffi::Buffer<ffi::S32>;
static ffi::Error vjp_impl(cudaStream_t st, BufF pos, BufF rot, BufF off, BufF dir, BufF xpos, BufF xquat, BufF obj,
                           BufI mbody, BufF dist, BufI gid, BufF nrm, BufF coef, int32_t ptype, int32_t nmesh,
                           float p0, float p1, float min_cos, float ac0, float ac1, float ac2, float aw0, float aw1,
                           float aw2, float l0, float l1, float l2, ffi::Result<BufF> out) {
  const auto dd = dist.dimensions(), dx = xpos.dimensions(), dob = obj.dimensions(), dc = coef.dimensions();
  if (dd.size() != 2 || dx.size() != 3 || dob.size() != 3 || dc.size() != 3)
    return ffi::Error::InvalidArgument("shapes");
  VjpArgs A;
  A.B = (int)dd[0]; A.N = (int)dd[1]; A.nbody = (int)dx[1]; A.K = (int)dob[1]; A.nmesh = nmesh;
  A.ncoef = (int)dc[2]; A.ptype = ptype; A.p0 = p0; A.p1 = p1; A.min_cos = min_cos;
  A.acam[0] = ac0; A.acam[1] = ac1; A.acam[2] = ac2; A.awld[0] = aw0; A.awld[1] = aw1; A.awld[2] = aw2;
  A.light[0] = l0; A.light[1] = l1; A.light[2] = l2;
  int rc = dk_ray_vjp(st, &A, pos.typed_data(), rot.typed_data(), off.typed_data(), dir.typed_data(),
                      xpos.typed_data(), xquat.typed_data(), A.K ? obj.typed_data() : nullptr, mbody.typed_data(),
                      dist.typed_data(), gid.typed_data(), nrm.typed_data(), coef.typed_data(), out->typed_data());
  return rc ? ffi::Error(ffi::ErrorCode::kInternal, "dk_ray_vjp") : ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkRayVjp, vjp_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<BufF>().Arg<BufF>().Arg<BufF>().Arg<BufF>().Arg<BufF>().Arg<BufF>().Arg<BufF>()
                                  .Arg<BufI>().Arg<BufF>().Arg<BufI>().Arg<BufF>().Arg<BufF>()
                                  .Attr<int32_t>("ptype").Attr<int32_t>("nmesh")
                                  .Attr<float>("p0").Attr<float>("p1").Attr<float>("min_cos")
                                  .Attr<float>("ac0").Attr<float>("ac1").Attr<float>("ac2")
                                  .Attr<float>("aw0").Attr<float>("aw1").Attr<float>("aw2")
                                  .Attr<float>("l0").Attr<float>("l1").Attr<float>("l2")
                                  .Ret<BufF>());
"""


class VjpArgs(ctypes.Structure):
    _fields_ = [(k, ctypes.c_int) for k in ("B", "N", "nbody", "K", "nmesh", "ncoef", "ptype")] + \
        [(k, ctypes.c_float) for k in ("p0", "p1", "min_cos")] + [(k, ctypes.c_float * 3) for k in ("acam", "awld",
                                                                                                   "light")]


def build(arch, include=None):
    """The backward's shared library for arch (sm_XY), cached by source (raycore.cache_dir); include: XLA's FFI
    headers for the JAX variant (handler DkRayVjp)."""
    from nablaengine import cuda
    nvcc = cuda.nvcc()
    flags = ["-O3", f"-arch={arch}", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-cudart", "static"]
    text = SOURCE + (FFI_SOURCE if include is not None else "")
    if include is not None:
        flags.append(f"-I{include}")
    key = hashlib.sha256((text + " ".join(flags) + nvcc).encode()).hexdigest()[:20]
    out = raycore.cache_dir() / f"libdk_rayvjp{'_jax' if include else ''}_{arch}_{key}.so"
    if not out.exists():
        src = out.with_suffix(".cu")
        src.write_text(text)
        tmp = out.with_suffix(f".tmp{os.getpid()}.so")
        subprocess.run([nvcc, *flags, "-o", str(tmp), str(src)], check=True)
        os.replace(tmp, out)
    return out


def params(spec=None, min_cos=1e-2, light=(0.0, 0.0, 0.0)):
    """(ptype, the 12 floats p0, p1, min_cos, a_cam, a_world, light) of an image spec (raygrad.ImageSpec; None: the
    hit distance alone)."""
    import numpy as np
    if spec is None:
        return 0, np.array([0, 0, min_cos] + [0] * 9, np.float32)
    return spec.ptype, np.array([spec.p0, spec.p1, min_cos, *spec.a_cam, *spec.a_world, *light], np.float32)

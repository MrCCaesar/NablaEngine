"""Dynamics queries of the forest engine: the joint-space mass matrix M(q), the bias force qfrc_bias(q, qdot) and the
Jacobians of world points on bodies, batched over worlds, on the GPU (no host round trip), from JAX and PyTorch."""
import ctypes
import threading

import numpy as np

UNIT = "dq.cu"

FFI = r"""
#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;
@DEFINES@
extern "C" int dq_upload(const float*, cudaStream_t);
extern "C" int dq_dynamics(cudaStream_t, int, const float*, const float*, const float*, float*, float*);
extern "C" int dq_jacobian(cudaStream_t, int, int, const float*, const int*, const float*, float*, float*);
extern "C" int dq_jacobian_vjp(cudaStream_t, int, int, const float*, const int*, const float*, const float*,
                               const float*, float*, float*);
extern "C" int dq_kinematics(cudaStream_t, int, const float*, const float*, const float*, float*, float*, float*,
                             float*, float*);
extern "C" int dq_kinematics_vjp(cudaStream_t, int, const float*, const float*, const float*, const float*,
                                 const float*, const float*, const float*, const float*, const float*, const float*,
                                 float*, float*);
static std::vector<float> g_consts;
static std::mutex g_mu;
static bool g_ready[64];
extern "C" int dq_set_constants(const float* p, long long n) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (n != NCONST) return 1;
  g_consts.assign(p, p + n);
  for (int d = 0; d < 64; d++) g_ready[d] = false;
  return 0;
}
static ffi::Error cuda_error(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return ffi::Error::Success();
  return ffi::Error::Internal(std::string("nablaengine.dyn_query: ") + what + ": " + cudaGetErrorString(e));
}
static ffi::Error bad(const std::string& m) { return ffi::Error::InvalidArgument("nablaengine.dyn_query: " + m); }
static ffi::Error ensure_ready(cudaStream_t st) {
  int dev = -1;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return cuda_error(e, "cudaGetDevice");
  std::lock_guard<std::mutex> lock(g_mu);
  if (dev < 0 || dev >= 64) return ffi::Error::Internal("nablaengine.dyn_query: device ordinal out of range");
  if (g_ready[dev]) return ffi::Error::Success();
  if ((long long)g_consts.size() != NCONST) return ffi::Error::Internal("nablaengine.dyn_query: no constants (init)");
  int rc = dq_upload(g_consts.data(), st);
  if (rc) return cuda_error((cudaError_t)rc, "constant upload");
  e = cudaStreamSynchronize(st);
  if (e != cudaSuccess) return cuda_error(e, "constant upload (synchronize)");
  g_ready[dev] = true;
  return ffi::Error::Success();
}
template <class B> static bool dims_are(const B& b, std::initializer_list<int64_t> want) {
  auto d = b.dimensions();
  if (d.size() != want.size()) return false;
  size_t i = 0;
  for (int64_t n : want)
    if (d[i++] != n) return false;
  return true;
}
typedef ffi::Buffer<ffi::F32> BufF;
typedef ffi::Buffer<ffi::S32> BufI;
typedef ffi::Result<ffi::Buffer<ffi::F32>> RetF;
static ffi::Error batch(const BufF& qpos, int64_t* B) {
  auto d = qpos.dimensions();
  if (d.size() != 2 || d[1] != NQ) return bad("qpos must be (B, nq) float32");
  *B = d[0];
  if (*B > INT_MAX) return bad("B too large");
  return ffi::Error::Success();
}
static ffi::Error dyn_impl(cudaStream_t st, BufF qpos, BufF qvel, BufF com, RetF M, RetF bias) {
  int64_t B = 0;
  ffi::Error e = batch(qpos, &B);
  if (e.failure()) return e;
  if (!dims_are(qvel, {B, NV}) || !dims_are(com, {B, COMW})) return bad("shapes: qvel (B, nv), com (B, com_words)");
  if (!dims_are(*M, {B, NV, NV}) || !dims_are(*bias, {B, NV})) return bad("outputs: M (B, nv, nv), bias (B, nv)");
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  int rc = dq_dynamics(st, (int)B, qpos.typed_data(), qvel.typed_data(), com.typed_data(), M->typed_data(),
                       bias->typed_data());
  return rc ? cuda_error((cudaError_t)rc, "k_dq_dyn launch") : ffi::Error::Success();
}
static ffi::Error jac_dims(const BufF& qpos, const BufI& bodies, const BufF& pts, int64_t* B, int64_t* P) {
  ffi::Error e = batch(qpos, B);
  if (e.failure()) return e;
  auto d = bodies.dimensions();
  if (d.size() != 2 || d[0] != *B) return bad("bodies must be (B, P) int32");
  *P = d[1];
  if (*P > INT_MAX / 4) return bad("P too large");
  if (!dims_are(pts, {*B, *P, 3})) return bad("points must be (B, P, 3) float32");
  return ffi::Error::Success();
}
static ffi::Error jac_impl(cudaStream_t st, BufF qpos, BufI bodies, BufF pts, RetF jacp, RetF jacr) {
  int64_t B = 0, P = 0;
  ffi::Error e = jac_dims(qpos, bodies, pts, &B, &P);
  if (e.failure()) return e;
  if (!dims_are(*jacp, {B, P, 3, NV}) || !dims_are(*jacr, {B, P, 3, NV}))
    return bad("outputs: jacp, jacr (B, P, 3, nv)");
  if (B == 0 || P == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  int rc = dq_jacobian(st, (int)B, (int)P, qpos.typed_data(), bodies.typed_data(), pts.typed_data(),
                       jacp->typed_data(), jacr->typed_data());
  return rc ? cuda_error((cudaError_t)rc, "k_dq_jac launch") : ffi::Error::Success();
}
static ffi::Error jac_vjp_impl(cudaStream_t st, BufF qpos, BufI bodies, BufF pts, BufF gjacp, BufF gjacr, RetF gq,
                               RetF gpts) {
  int64_t B = 0, P = 0;
  ffi::Error e = jac_dims(qpos, bodies, pts, &B, &P);
  if (e.failure()) return e;
  if (!dims_are(gjacp, {B, P, 3, NV}) || !dims_are(gjacr, {B, P, 3, NV}))
    return bad("cotangents: jacp, jacr (B, P, 3, nv)");
  if (!dims_are(*gq, {B, NQ}) || !dims_are(*gpts, {B, P, 3})) return bad("outputs: g_qpos (B, nq), g_points (B, P, 3)");
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  int rc = dq_jacobian_vjp(st, (int)B, (int)P, qpos.typed_data(), bodies.typed_data(), pts.typed_data(),
                           gjacp.typed_data(), gjacr.typed_data(), gq->typed_data(), gpts->typed_data());
  return rc ? cuda_error((cudaError_t)rc, "k_dq_jac_vjp launch") : ffi::Error::Success();
}
static bool kin_shapes(int64_t B, const BufF& qn, const BufF& xp, const BufF& xq, const BufF& cv, const BufF& sc) {
  return dims_are(qn, {B, NQ}) && dims_are(xp, {B, NB, 3}) && dims_are(xq, {B, NB, 4}) && dims_are(cv, {B, NB, 6})
         && dims_are(sc, {B, NB, 3});
}
#define KIN_SHAPES "qpos_n (B, nq), xpos (B, nbody, 3), xquat (B, nbody, 4), cvel (B, nbody, 6), subtree_com (B, nbody, 3)"
static ffi::Error kin_impl(cudaStream_t st, BufF qpos, BufF qvel, BufF com, RetF qn, RetF xp, RetF xq, RetF cv,
                           RetF sc) {
  int64_t B = 0;
  ffi::Error e = batch(qpos, &B);
  if (e.failure()) return e;
  if (!dims_are(qvel, {B, NV}) || !dims_are(com, {B, COMW})) return bad("shapes: qvel (B, nv), com (B, com_words)");
  if (!kin_shapes(B, *qn, *xp, *xq, *cv, *sc)) return bad("outputs: " KIN_SHAPES);
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  int rc = dq_kinematics(st, (int)B, qpos.typed_data(), qvel.typed_data(), com.typed_data(), qn->typed_data(),
                         xp->typed_data(), xq->typed_data(), cv->typed_data(), sc->typed_data());
  return rc ? cuda_error((cudaError_t)rc, "k_dq_kin launch") : ffi::Error::Success();
}
static ffi::Error kin_vjp_impl(cudaStream_t st, BufF qpos, BufF qvel, BufF com, BufF gqn, BufF gxp, BufF gxq,
                               BufF gcv, BufF gsc, RetF gq, RetF gv) {
  int64_t B = 0;
  ffi::Error e = batch(qpos, &B);
  if (e.failure()) return e;
  if (!dims_are(qvel, {B, NV}) || !dims_are(com, {B, COMW})) return bad("shapes: qvel (B, nv), com (B, com_words)");
  if (!kin_shapes(B, gqn, gxp, gxq, gcv, gsc)) return bad("cotangents: " KIN_SHAPES);
  if (!dims_are(*gq, {B, NQ}) || !dims_are(*gv, {B, NV})) return bad("outputs: g_qpos (B, nq), g_qvel (B, nv)");
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  int rc = dq_kinematics_vjp(st, (int)B, qpos.typed_data(), qvel.typed_data(), com.typed_data(), gqn.typed_data(),
                             gxp.typed_data(), gxq.typed_data(), gcv.typed_data(), gsc.typed_data(), nullptr, nullptr,
                             gq->typed_data(), gv->typed_data());
  return rc ? cuda_error((cudaError_t)rc, "k_dq_kin_vjp launch") : ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(DqKin, kin_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<BufF>().Arg<BufF>()
                                  .Arg<BufF>().Ret<BufF>().Ret<BufF>().Ret<BufF>().Ret<BufF>().Ret<BufF>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DqKinVjp, kin_vjp_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<BufF>().Arg<BufF>()
                                  .Arg<BufF>().Arg<BufF>().Arg<BufF>().Arg<BufF>().Arg<BufF>().Arg<BufF>()
                                  .Ret<BufF>().Ret<BufF>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DqDyn, dyn_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<BufF>().Arg<BufF>()
                                  .Arg<BufF>().Ret<BufF>().Ret<BufF>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DqJac, jac_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<BufF>().Arg<BufI>()
                                  .Arg<BufF>().Ret<BufF>().Ret<BufF>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DqJacVjp, jac_vjp_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<BufF>().Arg<BufI>()
                                  .Arg<BufF>().Arg<BufF>().Arg<BufF>().Ret<BufF>().Ret<BufF>());
"""


def _resolve(model, foot_pairs, com_body, params):
    from nablaengine import engines, jax_ffi
    from nablaengine.forest_engine import step as fstep, world_params
    from nablaengine.params import as_params
    if model is not None and foot_pairs is None:
        foot_pairs = []
    m, fp, com = jax_ffi._model(None, model, foot_pairs, com_body)
    params = as_params(params)
    engines.require_forest(m, com, "dyn_query.init", fp)
    if fstep.fixture_groups(m):
        raise ValueError("nablaengine.dyn_query: models with per-world fixture placements are not supported")
    if params:
        world_params.check(m, params, fp, com)
        if "body_inertial" in params.fields:
            raise ValueError("nablaengine.dyn_query: PhysicsParams body_inertial is not supported by the queries yet "
                             "(nor the kinematics outputs of models the step library's kernels do not cover)")
    return m, fp, com, params


def library_sources(m, com_body=None, params=None, ffi=True, pw_off=3):
    """({unit: source}, packed constants, {unit: -D options}) of the query library (ffi: with the XLA FFI unit; pw_off:
    the parameters' offset in a com row, 3 or the step's layout's: jax_ffi.com_layout)."""
    from nablaengine.forest_engine import dyn_device
    src, consts = dyn_device.cuda_source(m, com_body, params, pw_off)
    units, defines = {UNIT: src}, {UNIT: []}
    if ffi:
        defs = [f"#define {k} {v}" for k, v in (("NQ", m.nq), ("NV", m.nv), ("NB", m.nbody), ("NCONST", len(consts)),
                                                  ("COMW", pw_off + (params.width(m.nv) if params else 0)))]
        units["dk_ffi.cu"] = FFI.replace("@DEFINES@", "\n".join(defs))
        defines["dk_ffi.cu"] = []
    return units, consts, defines


def _build(m, com_body, params, ffi, arch, include, prefix, pw_off=3):
    from nablaengine import cuda, jax_ffi
    units, consts, defines = library_sources(m, com_body, params, ffi, pw_off)
    flags = cuda.arch_flags(None, arch)
    unit_arch = {n: flags for n in units} if flags != [f"-arch={arch}"] else None
    path, built, key = jax_ffi._build(units, defines, arch, include, consts, unit_arch, prefix=prefix)
    return dict(path=str(path), built=built, key=key, consts=consts)


def _check_sizes(lib, m, comw):
    sz = np.zeros(6, np.int64)
    lib.dq_sizes(ctypes.c_void_p(sz.ctypes.data))
    assert tuple(sz[:4]) == (m.nq, m.nv, m.nbody, comw), sz
    return int(sz[4])


def _default_com(m, com_body, params, pw_off=3):
    row = np.zeros(pw_off + (params.width(m.nv) if params else 0), np.float32)
    if params:
        import mujoco
        lay = params.layout(m.nv)
        if "gravity" in lay:
            o, n = lay["gravity"]
            row[pw_off + o:pw_off + o + n] = m.opt.gravity
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, com_body) if com_body else -1
        if "dof_armature" in lay:
            o, n = lay["dof_armature"]
            row[pw_off + o:pw_off + o + n] = m.dof_armature
        if "torso_mass" in lay:
            row[pw_off + lay["torso_mass"][0]] = m.body_mass[b]
        if "torso_inertia" in lay:
            o, n = lay["torso_inertia"]
            row[pw_off + o:pw_off + o + n] = m.body_inertia[b]
    return row


def _info(m, fp, com, params, b, sizeof, pw_off=3):
    return dict(path=b["path"], key=b["key"], built=b["built"], nq=m.nq, nv=m.nv, nbody=m.nbody, com_body=com,
                com_words=pw_off + (params.width(m.nv) if params else 0), params=params.key() if params else None,
                foot_pairs=[int(p) for p in fp], sizeof_state=sizeof)


def pw_offset(m, params, com_layout=None, pw_off=3):
    """The parameters' offset in a com row: com_layout (a jax_ffi build's info["com_layout"]: COM offset, object slots,
    their motion words, then the parameters) gives it, checked against params; else pw_off. A layout with per-world
    fixture placements is refused (not supported by the queries)."""
    if com_layout is None:
        return int(pw_off)
    lay = {k: tuple(int(x) for x in v) for k, v in dict(com_layout).items()}
    if "fixtures" in lay:
        raise ValueError("nablaengine.dyn_query: com_layout with per-world fixture placements is not supported")
    names = [k[len("params."):] for k in lay if k.startswith("params.")]
    want = list(params.layout(m.nv)) if params else []
    if names != want:
        raise ValueError(f"nablaengine.dyn_query: com_layout's parameters {names} are not params' fields {want}")
    if not names:
        return max(o + w for o, w in lay.values())
    off = lay["params." + names[0]][0]
    for f, (o, w) in params.layout(m.nv).items():
        if lay["params." + f] != (off + o, w):
            raise ValueError(f"nablaengine.dyn_query: com_layout's params.{f} {lay['params.' + f]} is not at the "
                             f"parameters' layout ({off + o}, {w})")
    return off


_LOCK = threading.Lock()
_J = {}


def init(model=None, foot_pairs=None, com_body=None, params=None, pw_off=3, com_layout=None):
    """Build / load the query library for the model (default: the fixtures' G1, as jax_ffi.init) and register its FFI
    targets. com_body: the body of com's COM offset and of the torso fields (default as jax_ffi.init: the G1's
    torso_link, none for the forest engine's other models). params: per-world physics parameters (com's layout).
    pw_off: the parameters' offset in a com row (3); com_layout: a jax_ffi build's info["com_layout"] in its place (a
    terrain build: the object slots' and their motion words come before the parameters; the query then takes that
    build's com rows as they are; pw_offset). Returns a dict (path, key, com_words, ...). One query library per process
    (a second init with another model or layout raises)."""
    from nablaengine import jax_ffi
    jax_ffi.ensure_jax_ffi()
    import jax
    with _LOCK:
        m, fp, com, params = _resolve(model, foot_pairs, com_body, params)
        pw_off = pw_offset(m, params, com_layout, pw_off)
        if _J:
            if _J["consts_hash"] != _hash(m, com, params) or _J["info"]["pw_off"] != pw_off:
                raise RuntimeError("nablaengine.dyn_query: already initialised with another model (one per process)")
            return _J["info"]
        arch = jax_ffi._arch()
        b = _build(m, com, params, True, arch, jax.ffi.include_dir(), "libdk_dynq", pw_off)
        lib = ctypes.cdll.LoadLibrary(b["path"])
        lib.dq_set_constants.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        c = np.ascontiguousarray(b["consts"], np.float32)
        if lib.dq_set_constants(c.ctypes.data, len(c)):
            raise RuntimeError("nablaengine.dyn_query: dq_set_constants failed")
        sizeof = _check_sizes(lib, m, pw_off + (params.width(m.nv) if params else 0))
        targets = {}
        for kind, sym in (("dyn", "DqDyn"), ("jac", "DqJac"), ("jac_vjp", "DqJacVjp"), ("kin", "DqKin"),
                          ("kin_vjp", "DqKinVjp")):
            targets[kind] = f"dq_{kind}_{b['key'][:8]}"
            jax.ffi.register_ffi_target(targets[kind], jax.ffi.pycapsule(getattr(lib, sym)), platform="CUDA")
        info = dict(_info(m, fp, com, params, b, sizeof, pw_off), pw_off=pw_off)
        _J.update(lib=lib, targets=targets, info=info, m=m, consts_hash=_hash(m, com, params),
                  com_row=_default_com(m, com, params, pw_off))
        return info


def _hash(m, com, params):
    import hashlib
    from nablaengine.forest_engine import dyn_device
    c = dyn_device.constants(m, com)[0]
    return hashlib.sha256(c.tobytes() + repr(params.key() if params else None).encode()).hexdigest()


def _need():
    if not _J:
        raise RuntimeError("nablaengine.dyn_query: call dyn_query.init() first")
    return _J


def dynamics(qpos, qvel, com=None):
    """(M (B, nv, nv), bias (B, nv)) of the states (qpos (B, nq), qvel (B, nv)); com (B, com_words) or None (the
    model's values). Not differentiable."""
    import jax
    import jax.numpy as jnp
    J = _need()
    nv = J["info"]["nv"]
    qpos = jnp.asarray(qpos, jnp.float32)
    B = qpos.shape[0]
    if com is None:
        com = jnp.broadcast_to(jnp.asarray(J["com_row"]), (B, J["com_row"].shape[0]))
    outs = [jax.ShapeDtypeStruct((B, nv, nv), jnp.float32), jax.ShapeDtypeStruct((B, nv), jnp.float32)]
    M, bias = jax.ffi.ffi_call(J["targets"]["dyn"], outs)(qpos, jnp.asarray(qvel, jnp.float32),
                                                         jnp.asarray(com, jnp.float32))
    return jax.lax.stop_gradient(M), jax.lax.stop_gradient(bias)


def _jac_args(qpos, bodies, points):
    import jax.numpy as jnp
    qpos = jnp.asarray(qpos, jnp.float32)
    B = qpos.shape[0]
    points = jnp.asarray(points, jnp.float32)
    if points.ndim == 2:
        points = jnp.broadcast_to(points, (B,) + points.shape)
    bodies = jnp.asarray(bodies, jnp.int32)
    if bodies.ndim == 1:
        bodies = jnp.broadcast_to(bodies, (B, bodies.shape[0]))
    if points.shape[:2] != bodies.shape or points.shape[2] != 3 or bodies.shape[0] != B:
        raise ValueError(f"dyn_query.jacobian: bodies {bodies.shape} and points {points.shape}: (B, P) and (B, P, 3)")
    return qpos, bodies, points


def jacobian_forward(qpos, bodies, points):
    """(jacp, jacr) (B, P, 3, nv) without custom_vjp; bodies (B, P) int32, points (B, P, 3)."""
    import jax
    import jax.numpy as jnp
    J = _need()
    B, P, nv = qpos.shape[0], bodies.shape[1], J["info"]["nv"]
    outs = [jax.ShapeDtypeStruct((B, P, 3, nv), jnp.float32)] * 2
    return tuple(jax.ffi.ffi_call(J["targets"]["jac"], outs)(qpos, bodies, points))


def jacobian_backward(qpos, bodies, points, g_jacp, g_jacr):
    """(g_qpos (B, nq), g_points (B, P, 3)) for the cotangents of jacp and jacr."""
    import jax
    import jax.numpy as jnp
    J = _need()
    outs = [jax.ShapeDtypeStruct(qpos.shape, jnp.float32), jax.ShapeDtypeStruct(points.shape, jnp.float32)]
    return tuple(jax.ffi.ffi_call(J["targets"]["jac_vjp"], outs)(
        qpos, bodies, points, jnp.asarray(g_jacp, jnp.float32), jnp.asarray(g_jacr, jnp.float32)))


_JFN = []


def _jac_fn():
    if not _JFN:
        import jax
        import numpy as _np

        @jax.custom_vjp
        def f(q, b, p):
            return jacobian_forward(q, b, p)

        def f_fwd(q, b, p):
            return jacobian_forward(q, b, p), (q, b, p)

        def f_bwd(res, ct):
            q, b, p = res
            gq, gp = jacobian_backward(q, b, p, ct[0], ct[1])
            return gq, _np.zeros(b.shape, jax.dtypes.float0), gp

        f.defvjp(f_fwd, f_bwd)
        _JFN.append(f)
    return _JFN[0]


def jacobian(qpos, bodies, points):
    """MuJoCo's mj_jac for P points per world: (jacp, jacr) (B, P, 3, nv). bodies (P,) or (B, P) int (the body each
    point moves with), points (P, 3) or (B, P, 3) world coordinates. Differentiable in qpos and points."""
    return _jac_fn()(*_jac_args(qpos, bodies, points))


KIN_OUTPUTS = ("qpos_n", "xpos", "xquat", "cvel", "subtree_com")


def _kin_structs(B):
    import jax
    import jax.numpy as jnp
    i = _need()["info"]
    nb = i["nbody"]
    return [jax.ShapeDtypeStruct(s, jnp.float32) for s in ((B, i["nq"]), (B, nb, 3), (B, nb, 4), (B, nb, 6), (B, nb, 3))]


def _com_arg(com, B, row):
    import jax.numpy as jnp
    if com is None:
        return jnp.broadcast_to(jnp.asarray(row), (B, row.shape[0]))
    com = jnp.asarray(com, jnp.float32)
    if com.shape[1] < row.shape[0]:
        com = jnp.concatenate([com, jnp.broadcast_to(jnp.asarray(row[com.shape[1]:]), (B, row.shape[0] - com.shape[1]))],
                              axis=1)
    return com


def kinematics_forward(qpos, qvel, com=None):
    """(qpos_n (B, nq), xpos (B, nbody, 3), xquat (B, nbody, 4), cvel (B, nbody, 6), subtree_com (B, nbody, 3))
    without custom_vjp: jax_ffi.kinematics's outputs and conventions on any forest model (several trees, free bodies,
    slides): every free joint's quaternion normalized into qpos_n, subtree_com of every body (body 0: all trees),
    cvel at the subtree COM of each body's tree root (MuJoCo's cvel). com: (B, com_words), (B, 3) (the parameters at
    the model's values) or None."""
    import jax
    import jax.numpy as jnp
    J = _need()
    qpos = jnp.asarray(qpos, jnp.float32)
    B = qpos.shape[0]
    return tuple(jax.ffi.ffi_call(J["targets"]["kin"], _kin_structs(B))(
        qpos, jnp.asarray(qvel, jnp.float32), _com_arg(com, B, J["com_row"])))


def kinematics_backward(qpos, qvel, com, g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com):
    """(g_qpos (B, nq), g_qvel (B, nv)) of kinematics_forward at (qpos, qvel, com) (com gets none)."""
    import jax
    import jax.numpy as jnp
    J = _need()
    qpos = jnp.asarray(qpos, jnp.float32)
    B = qpos.shape[0]
    outs = [jax.ShapeDtypeStruct((B, J["info"]["nq"]), jnp.float32), jax.ShapeDtypeStruct((B, J["info"]["nv"]),
                                                                                           jnp.float32)]
    gk = [jnp.asarray(g, jnp.float32) for g in (g_qpos_n, g_xpos, g_xquat, g_cvel, g_subtree_com)]
    return tuple(jax.ffi.ffi_call(J["targets"]["kin_vjp"], outs)(
        qpos, jnp.asarray(qvel, jnp.float32), _com_arg(com, B, J["com_row"]), *gk))


_KFN = []


def kinematics(qpos, qvel, com):
    """Differentiable kinematics outputs (kinematics_forward), a jax.custom_vjp: cotangents of qpos (the free
    quaternions through their normalization) and qvel, zeros for com. com: (B, com_words) or (B, 3)."""
    if not _KFN:
        import jax
        import jax.numpy as jnp
        from jax.ad_checkpoint import checkpoint_name
        from nablaengine.jax_ffi import KIN_NAME

        @jax.custom_vjp
        def f(q, v, c):
            return kinematics_forward(q, v, c)

        def f_fwd(q, v, c):
            out = tuple(checkpoint_name(x, KIN_NAME) for x in kinematics_forward(q, v, c))
            return out, (checkpoint_name(q, KIN_NAME), checkpoint_name(v, KIN_NAME), c)

        def f_bwd(res, ct):
            q, v, c = res
            gq, gv = kinematics_backward(q, v, c, *ct)
            return gq, gv, jnp.zeros_like(c)

        f.defvjp(f_fwd, f_bwd)
        _KFN.append(f)
    import jax.numpy as jnp
    qpos = jnp.asarray(qpos, jnp.float32)
    return _KFN[0](qpos, jnp.asarray(qvel, jnp.float32), _com_arg(com, qpos.shape[0], _need()["com_row"]))


_T = {}


def torch_init(device=None, model=None, foot_pairs=None, com_body=None, params=None, pw_off=3, com_layout=None):
    """The query library for PyTorch (ctypes entries on torch's current stream; no FFI unit, no jax needed): arguments
    as init(); device: the CUDA device (default the current one)."""
    import torch
    from nablaengine import cuda
    with _LOCK:
        m, fp, com, params = _resolve(model, foot_pairs, com_body, params)
        pw_off = pw_offset(m, params, com_layout, pw_off)
        if _T:
            if _T["consts_hash"] != _hash(m, com, params) or _T["info"]["pw_off"] != pw_off:
                raise RuntimeError("nablaengine.dyn_query: torch_init already called with another model")
            return _T["info"]
        dev = torch.device(device if device is not None else "cuda").index
        dev = torch.cuda.current_device() if dev is None else dev
        cc = torch.cuda.get_device_capability(dev)
        arch = f"sm_{cc[0]}{cc[1]}"
        cuda.arch()
        b = _build(m, com, params, False, arch, None, "libdk_dynq_torch", pw_off)
        lib = ctypes.cdll.LoadLibrary(b["path"])
        P = ctypes.c_void_p
        lib.dq_upload.argtypes = [P, P]
        lib.dq_dynamics.argtypes = [P, ctypes.c_int, P, P, P, P, P]
        lib.dq_jacobian.argtypes = [P, ctypes.c_int, ctypes.c_int, P, P, P, P, P]
        lib.dq_jacobian_vjp.argtypes = [P, ctypes.c_int, ctypes.c_int, P, P, P, P, P, P, P]
        sizeof = _check_sizes(lib, m, pw_off + (params.width(m.nv) if params else 0))
        lib.dq_kinematics.argtypes = [P, ctypes.c_int] + [P] * 8
        lib.dq_kinematics_vjp.argtypes = [P, ctypes.c_int] + [P] * 12
        info = dict(_info(m, fp, com, params, b, sizeof, pw_off), pw_off=pw_off)
        _T.update(lib=lib, info=info, consts=np.ascontiguousarray(b["consts"], np.float32), ready=set(), m=m,
                  consts_hash=_hash(m, com, params), com_row=_default_com(m, com, params, pw_off))
        return info


def _tlib(device):
    import torch
    if not _T:
        raise RuntimeError("nablaengine.dyn_query: call dyn_query.torch_init() first")
    dev = device.index if device.index is not None else torch.cuda.current_device()
    st = torch.cuda.current_stream(dev)
    if dev not in _T["ready"]:
        with torch.cuda.device(dev):
            rc = _T["lib"].dq_upload(_T["consts"].ctypes.data, st.cuda_stream)
            st.synchronize()
        if rc:
            raise RuntimeError(f"nablaengine.dyn_query: constant upload failed ({rc})")
        _T["ready"].add(dev)
    return _T["lib"], dev, st.cuda_stream


def _tcheck(rc, what):
    if rc:
        raise RuntimeError(f"nablaengine.dyn_query: {what} failed (cudaError {rc})")


def torch_dynamics(qpos, qvel, com=None):
    """(M (B, nv, nv), bias (B, nv)) as dynamics(), torch tensors on qpos's device (no gradient)."""
    import torch
    lib, dev, st = _tlib(qpos.device)
    nv = _T["info"]["nv"]
    qpos = qpos.detach().to(torch.float32).contiguous()
    qvel = qvel.detach().to(torch.float32).contiguous()
    B = qpos.shape[0]
    if com is None:
        com = torch.as_tensor(_T["com_row"], device=qpos.device).expand(B, -1)
    com = com.detach().to(torch.float32).contiguous()
    assert qpos.shape == (B, _T["info"]["nq"]) and qvel.shape == (B, nv) and com.shape == (B, _T["info"]["com_words"])
    M = torch.empty((B, nv, nv), dtype=torch.float32, device=qpos.device)
    bias = torch.empty((B, nv), dtype=torch.float32, device=qpos.device)
    with torch.cuda.device(dev):
        _tcheck(lib.dq_dynamics(st, B, qpos.data_ptr(), qvel.data_ptr(), com.data_ptr(), M.data_ptr(),
                                bias.data_ptr()), "k_dq_dyn")
    return M, bias


def _tjac_args(qpos, bodies, points):
    import torch
    B = qpos.shape[0]
    points = torch.as_tensor(points, dtype=torch.float32, device=qpos.device)
    if points.dim() == 2:
        points = points.expand(B, -1, -1)
    bodies = torch.as_tensor(bodies, dtype=torch.int32, device=qpos.device)
    if bodies.dim() == 1:
        bodies = bodies.expand(B, -1)
    if tuple(points.shape[:2]) != tuple(bodies.shape) or points.shape[2] != 3:
        raise ValueError(f"dyn_query.torch_jacobian: bodies {tuple(bodies.shape)} and points {tuple(points.shape)}")
    return bodies.contiguous(), points


def _torch_jac_fn():
    import torch

    class Jac(torch.autograd.Function):
        @staticmethod
        def forward(ctx, qpos, bodies, points):
            lib, dev, st = _tlib(qpos.device)
            q = qpos.detach().to(torch.float32).contiguous()
            p = points.detach().to(torch.float32).contiguous()
            B, P, nv = q.shape[0], bodies.shape[1], _T["info"]["nv"]
            jp = torch.empty((B, P, 3, nv), dtype=torch.float32, device=q.device)
            jr = torch.empty_like(jp)
            with torch.cuda.device(dev):
                _tcheck(lib.dq_jacobian(st, B, P, q.data_ptr(), bodies.data_ptr(), p.data_ptr(), jp.data_ptr(),
                                        jr.data_ptr()), "k_dq_jac")
            ctx.save_for_backward(q, bodies, p)
            return jp, jr

        @staticmethod
        def backward(ctx, gjp, gjr):
            q, bodies, p = ctx.saved_tensors
            lib, dev, st = _tlib(q.device)
            gjp = (torch.zeros((*bodies.shape, 3, _T["info"]["nv"]), device=q.device) if gjp is None
                   else gjp.to(torch.float32).contiguous())
            gjr = torch.zeros_like(gjp) if gjr is None else gjr.to(torch.float32).contiguous()
            gq, gp = torch.empty_like(q), torch.empty_like(p)
            with torch.cuda.device(dev):
                _tcheck(lib.dq_jacobian_vjp(st, q.shape[0], bodies.shape[1], q.data_ptr(), bodies.data_ptr(),
                                            p.data_ptr(), gjp.data_ptr(), gjr.data_ptr(), gq.data_ptr(),
                                            gp.data_ptr()), "k_dq_jac_vjp")
            return gq, None, gp

    return Jac


def torch_jacobian(qpos, bodies, points):
    """(jacp, jacr) (B, P, 3, nv) as jacobian(), differentiable (torch.autograd) in qpos and points."""
    if "jac_fn" not in _T:
        _T["jac_fn"] = _torch_jac_fn()
    bodies, points = _tjac_args(qpos, bodies, points)
    return _T["jac_fn"].apply(qpos, bodies, points)


def _torch_kin_fn():
    import torch

    class Kin(torch.autograd.Function):
        @staticmethod
        def forward(ctx, qpos, qvel, com):
            lib, dev, st = _tlib(qpos.device)
            q, v = qpos.detach().to(torch.float32).contiguous(), qvel.detach().to(torch.float32).contiguous()
            c = com.detach().to(torch.float32).contiguous()
            B, nb, nq = q.shape[0], _T["info"]["nbody"], _T["info"]["nq"]
            kw = dict(dtype=torch.float32, device=q.device)
            outs = [torch.empty((B, nq), **kw), torch.empty((B, nb, 3), **kw), torch.empty((B, nb, 4), **kw),
                    torch.empty((B, nb, 6), **kw), torch.empty((B, nb, 3), **kw)]
            with torch.cuda.device(dev):
                _tcheck(lib.dq_kinematics(st, B, q.data_ptr(), v.data_ptr(), c.data_ptr(),
                                          *[o.data_ptr() for o in outs]), "k_dq_kin")
            ctx.save_for_backward(q, v, c)
            return tuple(outs)

        @staticmethod
        def backward(ctx, *g):
            q, v, c = ctx.saved_tensors
            lib, dev, st = _tlib(q.device)
            B, nb, nq = q.shape[0], _T["info"]["nbody"], _T["info"]["nq"]
            shapes = [(B, nq), (B, nb, 3), (B, nb, 4), (B, nb, 6), (B, nb, 3)]
            g = [torch.zeros(s, dtype=torch.float32, device=q.device) if x is None else x.to(torch.float32).contiguous()
                 for x, s in zip(g, shapes)]
            gq, gv = torch.empty_like(q), torch.empty_like(v)
            with torch.cuda.device(dev):
                _tcheck(lib.dq_kinematics_vjp(st, B, q.data_ptr(), v.data_ptr(), c.data_ptr(),
                                              *[x.data_ptr() for x in g], None, None, gq.data_ptr(), gv.data_ptr()),
                        "k_dq_kin_vjp")
            return gq, gv, None

    return Kin


def torch_kinematics(qpos, qvel, com=None):
    """(qpos_n, xpos, xquat, cvel, subtree_com) as kinematics(), differentiable (torch.autograd) in qpos and qvel;
    com (B, com_words), (B, 3) or None."""
    import torch
    if "kin_fn" not in _T:
        _T["kin_fn"] = _torch_kin_fn()
    _tlib(qpos.device)
    B = qpos.shape[0]
    row = torch.as_tensor(_T["com_row"], device=qpos.device)
    if com is None:
        com = row.expand(B, -1)
    elif com.shape[1] < row.shape[0]:
        com = torch.cat([com.to(torch.float32), row[com.shape[1]:].expand(B, -1)], dim=1)
    return _T["kin_fn"].apply(qpos, qvel, com)

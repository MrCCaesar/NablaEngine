"""Distance queries of the forest engine: signed distances, closest points and normals between chosen geom or body
pairs, batched over worlds on the GPU, with the VJP with respect to qpos, from JAX and PyTorch."""
import ctypes
import os
import threading

import numpy as np

QUERY_TYPES = ("plane", "sphere", "capsule", "box", "mesh", "cylinder", "ellipsoid")
MAX_SLOTS = 128
TERRAIN = "@terrain"
TERRAIN_GEOM = -1
FACE_PX, FACE_NX, FACE_PY, FACE_NY, FACE_PZ, FACE_NZ = range(6)
UNIT_NAME = "dist_query.cu"
PREFIX = "dist"

UNIT = r"""
#include <cuda_runtime.h>
namespace {
@ENGINE@
}  // namespace
extern "C" int @PFX@_upload(const float* p, cudaStream_t st) {
@LOAD_CONSTS@
  return (int)cudaGetLastError();
}
extern "C" int @PFX@_query(cudaStream_t st, int nw, const float* qpos, const float* com, float dmax, float* dist,
                         float* pts, float* nrm, int* far, int* face) {
  if (nw > 0) k_dst_query<<<(nw + DST_WPB - 1) / DST_WPB, 32 * DST_WPB, 0, st>>>(nw, qpos, com, dmax, dist, pts, nrm, far,
                                                                                     face);
  return (int)cudaGetLastError();
}
extern "C" int @PFX@_query_vjp(cudaStream_t st, int nw, const float* qpos, const float* com, float dmax,
                             const float* gd, const float* gp, float* gq) {
  if (nw > 0) k_dst_query_vjp<<<(nw + DST_WPB - 1) / DST_WPB, 32 * DST_WPB, 0, st>>>(nw, qpos, com, dmax, gd, gp, gq);
  return (int)cudaGetLastError();
}
extern "C" void @PFX@_sizes(long long* o) {
  o[0] = (long long)sizeof(DstS); o[1] = NQ; o[2] = NV; o[3] = DST_NS; o[4] = COM_WORDS;
}
"""

FFI = r"""
#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;
@DEFINES@
typedef int (*UploadFn)(const float*, cudaStream_t);
typedef int (*QueryFn)(cudaStream_t, int, const float*, const float*, float, float*, float*, float*, int*, int*);
typedef int (*QueryVjpFn)(cudaStream_t, int, const float*, const float*, float, const float*, const float*, float*);
@DECLS@
static std::vector<float> g_consts;
static std::mutex g_mu;
static bool g_ready[64];
extern "C" int dst_set_constants(const float* p, long long n) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (n != NCONST) return 1;
  g_consts.assign(p, p + n);
  for (int d = 0; d < 64; d++) g_ready[d] = false;
  return 0;
}
static ffi::Error cuda_error(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return ffi::Error::Success();
  return ffi::Error::Internal(std::string("nablaengine.dist_query: ") + what + ": " + cudaGetErrorString(e));
}
static ffi::Error bad(const std::string& m) { return ffi::Error::InvalidArgument("nablaengine.dist_query: " + m); }
static ffi::Error ensure_ready(cudaStream_t st) {
  int dev = -1;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return cuda_error(e, "cudaGetDevice");
  std::lock_guard<std::mutex> lock(g_mu);
  if (dev < 0 || dev >= 64) return ffi::Error::Internal("nablaengine.dist_query: device ordinal out of range");
  if (g_ready[dev]) return ffi::Error::Success();
  if ((long long)g_consts.size() != NCONST) return ffi::Error::Internal("nablaengine.dist_query: no constants (dist_query.init)");
  for (int v = 0; v < NVAR; v++) {
    int rc = kUpload[v](g_consts.data(), st);
    if (rc) return cuda_error((cudaError_t)rc, "constant upload");
  }
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
typedef ffi::Result<ffi::Buffer<ffi::F32>> RetF;
typedef ffi::Result<ffi::Buffer<ffi::S32>> RetI;
static ffi::Error query_impl(cudaStream_t st, BufF qpos, BufF com, float dmax, RetF dist, RetF pts, RetF nrm, RetI far,
                             RetI face) {
  auto d = qpos.dimensions();
  if (d.size() != 2 || d[1] != NQ) return bad("qpos must be (B, nq) float32");
  const int64_t B = d[0];
  if (B > INT_MAX) return bad("B too large");
  if (!dims_are(com, {B, COMW})) return bad("com must be (B, com_words) float32");
  if (!dims_are(*dist, {B, NS}) || !dims_are(*pts, {B, NS, 2, 3}) || !dims_are(*nrm, {B, NS, 3}) || !dims_are(*far, {B, NS})
      || !dims_are(*face, {B, NS, 2}))
    return bad("outputs: dist (B, S), points (B, S, 2, 3), normal (B, S, 3), far (B, S), face (B, S, 2)");
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  int rc = kQuery(st, (int)B, qpos.typed_data(), com.typed_data(), dmax, dist->typed_data(), pts->typed_data(),
                  nrm->typed_data(), far->typed_data(), face->typed_data());
  return rc ? cuda_error((cudaError_t)rc, "k_dst_query launch") : ffi::Error::Success();
}
static ffi::Error query_vjp_impl(cudaStream_t st, BufF qpos, BufF com, BufF gd, BufF gp, float dmax, RetF gq) {
  auto d = qpos.dimensions();
  if (d.size() != 2 || d[1] != NQ) return bad("qpos must be (B, nq) float32");
  const int64_t B = d[0];
  if (B > INT_MAX) return bad("B too large");
  if (!dims_are(com, {B, COMW}) || !dims_are(gd, {B, NS}) || !dims_are(gp, {B, NS, 2, 3}) || !dims_are(*gq, {B, NQ}))
    return bad("shapes: com (B, com_words), g_dist (B, S), g_points (B, S, 2, 3), out (B, nq)");
  if (B == 0) return ffi::Error::Success();
  ffi::Error r = ensure_ready(st);
  if (r.failure()) return r;
  int rc = kQueryVjp(st, (int)B, qpos.typed_data(), com.typed_data(), dmax, gd.typed_data(), gp.typed_data(),
                     gq->typed_data());
  return rc ? cuda_error((cudaError_t)rc, "k_dst_query_vjp launch") : ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(DstQuery, query_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<BufF>().Arg<BufF>()
                                  .Attr<float>("dmax").Ret<BufF>().Ret<BufF>().Ret<BufF>().Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DstQueryVjp, query_vjp_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<BufF>().Arg<BufF>()
                                  .Arg<BufF>().Arg<BufF>().Attr<float>("dmax").Ret<BufF>());
"""

TERRAIN_UNIT = r"""extern "C" int @PFX@_set_terrain(int dev, const void* p) {
  int old = 0;
  cudaGetDevice(&old);
  cudaError_t e = cudaSetDevice(dev);
  if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_ter, p, sizeof(TerrainPtrs));
  cudaSetDevice(old);
  return (int)e;
}
"""

FFI_TERRAIN = (("!dims_are(*face, {B, NS, 2})", "!dims_are(*face, {B, NS, 4})"),
               ("face (B, S, 2)\");", "face (B, S, 4)\");"))


def _collision_geoms_of(m, b, who="nablaengine.dist_query"):
    import mujoco
    from nablaengine.forest_engine import step
    cg = set(step.collision_geoms(m))
    gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b and g in cg]
    if not gs:
        gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b
              and int(m.geom_type[g]) in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                                          int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_MESH),
                                          int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID))]
    if not gs:
        raise ValueError(f"{who}: body {m.body(b).name!r} has no geom to query")
    return gs


def resolve(m, pairs, who="nablaengine.dist_query"):
    """(slots: per pair a list of model geom id pairs, labels) of the query pairs (TERRAIN as the second name, or
    TERRAIN_GEOM as the second geom id: the terrain of a terrain build). A body's geoms are its collision geoms (the
    engine's), or, if it has none, its geoms of the supported solid types."""
    import mujoco
    slots, labels = [], []
    for p in pairs:
        if len(p) == 2:
            g1, g2 = int(p[0]), int(p[1])
            slots.append([(g1, g2)])
            labels.append(f"geom:{m.geom(g1).name}|{TERRAIN if g2 == TERRAIN_GEOM else m.geom(g2).name}")
            continue
        kind, a, b = p
        if a == TERRAIN:
            raise ValueError(f"{who}: the terrain is the second geom of a pair: {p}")
        if kind == "geom":
            ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, x) if x != TERRAIN else TERRAIN_GEOM for x in (a, b)]
            if ids[0] < 0 or (ids[1] < 0 and b != TERRAIN):
                raise ValueError(f"{who}: unknown geom in {p}")
            slots.append([tuple(ids)])
        elif kind == "body":
            ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, x) if x != TERRAIN else TERRAIN_GEOM for x in (a, b)]
            if ids[0] < 0 or (ids[1] < 0 and b != TERRAIN):
                raise ValueError(f"{who}: unknown body in {p}")
            g2s = [TERRAIN_GEOM] if b == TERRAIN else _collision_geoms_of(m, ids[1], who)
            slots.append([(g1, g2) for g1 in _collision_geoms_of(m, ids[0], who) for g2 in g2s])
        else:
            raise ValueError(f"{who}: a pair is ('geom' | 'body', name1, name2) or (geom1, geom2): {p}")
        labels.append(f"{kind}:{a}|{b}")
    if not slots or len(slots) > MAX_SLOTS:
        raise ValueError(f"{who}: 1 .. {MAX_SLOTS} query pairs")
    return slots, labels


def check_terrain_params(terrain, params, who="nablaengine.dist_query"):
    """ValueError for a terrain build with per-world parameters (not supported by the query libraries, as the
    engine's object_size against the terrain)."""
    from nablaengine.params import as_params
    if terrain is not None and as_params(params):
        raise ValueError(f"{who}: per-world parameters (params=) with a terrain build (terrain=) are not supported")


def engine_text(m, foot_pairs, com_body, max_contacts=None, max_rows=None, terrain=None, params=None,
                who="nablaengine.dist_query"):
    """The forest engine's text a query library is built on and its inputs: a dict with m (the build's model:
    object_size's bound model with that field), params, ter ((TerrainConfig, robot terrain geoms) or None), consts
    (packed constants), load (the constant loader, asynchronous on stream st), engine (adjoint.cuda_source's text,
    include removed), mc (the contact capacity), mr (the row capacity or None). terrain: a TerrainConfig; params:
    per-world physics parameters (nablaengine.params). Not both (ValueError)."""
    import re
    from nablaengine.forest_engine import adjoint, host, object_size, world_params
    from nablaengine import engines, jax_ffi
    from nablaengine.params import as_params
    if not engines.is_forest(m):
        raise ValueError(f"{who}: the forest engine's models only (nablaengine.engines)")
    params = as_params(params)
    check_terrain_params(terrain, params, who)
    if params:
        world_params.check(m, params, foot_pairs, com_body)
        if "sole_thickness" in params.fields or params.invweight0 != "model":
            raise ValueError(f"{who}: PhysicsParams sole_thickness / invweight0='per_world' are not supported in this "
                             "build yet")
        params = object_size.resolve(m, params, com_body)
        m = object_size.bound_model(m, params)
    ter = None
    if terrain is not None:
        from nablaengine import terrain as _terrain
        if getattr(terrain, "terrain_sand", False):
            raise ValueError(f"{who}: TerrainConfig.terrain_sand is not supported in this build")
        ter = (terrain, _terrain.robot_terrain_geoms(m, foot_pairs, terrain.geoms))
    consts, lays, sizes = adjoint.engine_constants(m, foot_pairs, com_body, **({"terrain": ter} if ter else {}))
    load = host.loader(lays, sizes)
    load = re.sub(r"cudaMemcpyToSymbol\(([^;]*)\);", r"cudaMemcpyToSymbolAsync(\1, 0, cudaMemcpyHostToDevice, st);", load)
    mc_in, mr_in, mc, _ = jax_ffi.capacity(m, terrain, max_contacts, max_rows)
    engine = adjoint.cuda_source(m, mc_in, ter, **({"max_rows": mr_in} if mr_in is not None else {}),
                                 **({"params": params} if params else {}))
    engine = engine.replace("#include <cuda_runtime.h>\n", "")
    return dict(m=m, params=params, ter=ter, consts=consts, load=load, engine=engine, mc=mc, mr=mr_in)


def terrain_unit(prefix, ter):
    """The unit's terrain upload entry <prefix>_set_terrain (jax_ffi.set_terrain / torch_ffi.set_terrain's pointers;
    with TerrainConfig.terrain_material also the elements' materials, the 9th pointer)."""
    tu = TERRAIN_UNIT.replace("@PFX@", prefix)
    if getattr(ter[0], "terrain_material", False):
        tu = tu.replace("  cudaSetDevice(old);\n", "  if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_tmat, "
                        "(const void* const*)p + 8, sizeof(const float*));\n  cudaSetDevice(old);\n")
    return tu


def com_words(m, ter, params):
    """Floats per world of the com input of a build (the step's com row)."""
    from nablaengine.forest_engine import step as fstep
    comw = fstep.com_words(m) + (ter[0].com_words() - 3 if ter else 0)
    return comw + (params.width(m.nv) if params else 0)


def library_sources(m, foot_pairs, com_body, slots, max_contacts=None, max_rows=None, ffi=True, terrain=None,
                    params=None):
    """({unit: source}, packed constants, {unit: -D options}, query geom ids) of the query library. slots: resolve's.
    terrain: the build's nablaengine.terrain.TerrainConfig (a terrain build: the engine's terrain text, the terrain
    query geom, the unit's dist_set_terrain). params: per-world physics parameters (nablaengine.params; the com rows
    as the engine library's; object_size also in the queries' geom, forest_engine.object_size.queries). Not both
    (ValueError)."""
    from nablaengine.forest_engine import dist_device, object_size
    e = engine_text(m, foot_pairs, com_body, max_contacts, max_rows, terrain, params)
    m, params, ter = e["m"], e["params"], e["ter"]
    engine, geoms = dist_device.apply(e["engine"], m, slots, **({"terrain": True} if ter else {}))
    engine = object_size.queries(engine, m, params, geoms)
    unit = UNIT.replace("@PFX@", PREFIX).replace("@LOAD_CONSTS@", e["load"]).replace("@ENGINE@", engine)
    if params:
        assert unit.count("o[4] = COM_WORDS;") == 1
        unit = unit.replace("o[4] = COM_WORDS;", "o[4] = COM_STRIDE;")
    if ter:
        unit += terrain_unit(PREFIX, ter)
    units, defines = {UNIT_NAME: unit}, {UNIT_NAME: []}
    if ffi:
        defs = [f"#define {k} {val}" for k, val in (("NQ", m.nq), ("NV", m.nv), ("NS", len(slots)),
                                                    ("NCONST", len(e["consts"])),
                                                    ("COMW", com_words(m, ter, params)), ("NVAR", 1))]
        decls = [f'extern "C" int {PREFIX}_upload(const float*, cudaStream_t);',
                 f'extern "C" int {PREFIX}_query(cudaStream_t, int, const float*, const float*, float, float*, float*, '
                 'float*, int*, int*);',
                 f'extern "C" int {PREFIX}_query_vjp(cudaStream_t, int, const float*, const float*, float, '
                 'const float*, const float*, float*);',
                 f"static const QueryFn kQuery = {PREFIX}_query;",
                 f"static const QueryVjpFn kQueryVjp = {PREFIX}_query_vjp;",
                 f"static const UploadFn kUpload[NVAR] = {{{PREFIX}_upload}};"]
        ffi_text = FFI
        if ter:
            for a, b in FFI_TERRAIN:
                assert ffi_text.count(a) == 1
                ffi_text = ffi_text.replace(a, b)
        units["dk_ffi.cu"] = ffi_text.replace("@DEFINES@", "\n".join(defs)).replace("@DECLS@", "\n".join(decls))
        defines["dk_ffi.cu"] = []
    return units, e["consts"], defines, geoms


def _build(m, foot_pairs, com_body, pairs, max_contacts, max_rows, ffi, arch, include, prefix, terrain=None,
           params=None):
    from nablaengine import cuda, jax_ffi
    slots, labels = resolve(m, pairs)
    units, consts, defines, geoms = library_sources(m, foot_pairs, com_body, slots, max_contacts, max_rows, ffi,
                                                     **({"terrain": terrain} if terrain else {}),
                                                     **({"params": params} if params else {}))
    for n in units:
        defines[n] = list(defines[n]) + cuda.forest_hold_defines(arch, units[n], defines[n])
    flags = cuda.arch_flags(None, arch)
    unit_arch = {n: flags for n in units} if flags != [f"-arch={arch}"] else None
    path, built, key = jax_ffi._build(units, defines, arch, include, consts, unit_arch, prefix=prefix)
    return dict(path=str(path), built=built, key=key, slots=slots, labels=labels, geoms=geoms, consts=consts,
                nslot=len(slots))


def _check_sizes(lib, nq, nv, nslot, comw, who="nablaengine.dist_query"):
    sz = np.zeros(5, np.int64)
    fn = getattr(lib, f"{PREFIX}_sizes")
    fn(ctypes.c_void_p(sz.ctypes.data))
    if tuple(int(x) for x in sz[1:5]) != (nq, nv, nslot, comw):
        raise RuntimeError(f"{who}: the library's sizes {sz.tolist()} do not match (nq, nv, S, com_words) = "
                           f"{(nq, nv, nslot, comw)}")


def _info(b, max_distance, einfo, terrain):
    return dict(path=b["path"], key=b["key"], built=b["built"], labels=b["labels"],
                slots=[[list(map(int, p)) for p in s] for s in b["slots"]], max_distance=float(max_distance),
                engine=einfo, nslot=b["nslot"], terrain=terrain is not None, geoms=[int(g) for g in b["geoms"]])


def jax_terrain_hooks(lib, prefixes, who="nablaengine.dist_query"):
    """Register the units' terrain upload (<prefix>_set_terrain) with jax_ffi.set_terrain and apply a terrain already
    uploaded."""
    import jax
    from nablaengine import jax_ffi
    fns = []
    for t in prefixes:
        f = getattr(lib, f"{t}_set_terrain")
        f.argtypes = [ctypes.c_int, ctypes.c_void_p]
        fns.append(f)
    jax_ffi._STATE.setdefault("terrain_hooks", []).extend(fns)
    devs = {d.id: d for d in jax.devices()}
    for did, arrs in jax_ffi._TERRAIN.items():
        ptrs = (ctypes.c_void_p * 8)(*[x.unsafe_buffer_pointer() for x in arrs])
        for f in fns:
            rc = f(int(devs[did].local_hardware_id), ctypes.cast(ptrs, ctypes.c_void_p))
            if rc:
                raise RuntimeError(f"{who}: the terrain upload failed ({rc})")


def torch_terrain_hooks(lib, prefixes, who="nablaengine.dist_query"):
    """Register the units' terrain upload with torch_ffi.set_terrain and apply a terrain already uploaded."""
    from nablaengine import torch_ffi
    tf = []
    for t in prefixes:
        f = getattr(lib, f"{t}_set_terrain")
        f.argtypes = [ctypes.c_int, ctypes.c_void_p]
        tf.append(f)
    torch_ffi._STATE["fns"]["set_terrain"] = list(torch_ffi._STATE["fns"]["set_terrain"]) + tf
    order = ("boxes", "hf_head", "hf_int", "hf_data", "grid", "cell_start", "cell_items", "elem_cell")
    for tdev, tens in torch_ffi._TERRAIN.items():
        ptrs = (ctypes.c_void_p * 8)(*[tens[k].data_ptr() for k in order])
        for f in tf:
            if f(int(tdev), ctypes.cast(ptrs, ctypes.c_void_p)):
                raise RuntimeError(f"{who}: the terrain upload failed")


_LOCK = threading.Lock()
_J = {}


def init(model, pairs, max_distance=1.0, *, foot_pairs=(), com_body=None, max_contacts=None, max_rows=None,
         terrain=None, params=None):
    """Initialise jax_ffi with the model (jax_ffi.init(model=model, foot_pairs=..., com_body=..., max_contacts=...,
    max_rows=..., terrain=..., params=...)) and build / load the query library for the pairs (module docstring).
    max_distance: the queries' cutoff (m; query takes another per call). terrain: a nablaengine.terrain.TerrainConfig
    (a terrain build: pairs with TERRAIN; the terrain jax_ffi.set_terrain uploads, before or after init, reaches the
    query library too). params: per-world physics parameters (nablaengine.params, as jax_ffi.init's: com is then
    (B, com_words) with them; object_size applies to the queries). terrain and params: not both (ValueError). Returns
    a dict (path, key, labels, slots: per pair its model geom id pairs, geoms, max_distance, nslot, engine: jax_ffi's
    info). One query library per process."""
    import jax
    from nablaengine import jax_ffi
    check_terrain_params(terrain, params)
    with _LOCK:
        if _J:
            raise RuntimeError("nablaengine.dist_query: already initialised (one query library per process)")
        einfo = jax_ffi.init(model=model, foot_pairs=list(foot_pairs), com_body=com_body, max_contacts=max_contacts,
                             max_rows=max_rows, **({"terrain": terrain} if terrain is not None else {}),
                             **({"params": params} if params else {}))
        if einfo["engine"] != "forest":
            raise ValueError("nablaengine.dist_query: the forest engine's models only")
        if einfo.get("actuator"):
            raise ValueError("nablaengine.dist_query: not with an actuator model yet")
        m = jax_ffi._STATE["m"]
        b = _build(m, einfo["foot_pairs"], einfo["com_body"], pairs, max_contacts, max_rows, True, einfo["arch"],
                   jax.ffi.include_dir(), "libdk_dist", terrain, params)
        lib = ctypes.cdll.LoadLibrary(b["path"])
        if terrain is not None:
            jax_terrain_hooks(lib, [PREFIX])
        lib.dst_set_constants.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        c = np.ascontiguousarray(b["consts"], np.float32)
        if lib.dst_set_constants(c.ctypes.data, len(c)):
            raise RuntimeError("nablaengine.dist_query: dst_set_constants failed")
        _check_sizes(lib, m.nq, m.nv, b["nslot"], int(einfo["com_words"]))
        targets = {}
        for kind, sym in (("query", "DstQuery"), ("query_vjp", "DstQueryVjp")):
            targets[kind] = f"dist_{kind}_{b['key'][:8]}"
            jax.ffi.register_ffi_target(targets[kind], jax.ffi.pycapsule(getattr(lib, sym)), platform="CUDA")
        info = _info(b, max_distance, einfo, terrain)
        _J.update(lib=lib, targets=targets, info=info, dmax=float(np.float32(max_distance)), m=m, qfn={})
        return info


def _need():
    if not _J:
        raise RuntimeError("nablaengine.dist_query: call dist_query.init(model, pairs) first")
    return _J


def jax_query_forward(J, qpos, com, max_distance=None):
    """query_forward on the interface state J (a dict with targets["query"], info, dmax)."""
    import jax
    import jax.numpy as jnp
    B, S = qpos.shape[0], J["info"]["nslot"]
    dmax = np.float32(J["dmax"] if max_distance is None else max_distance)
    f32 = jnp.float32
    fw = 4 if J["info"].get("terrain") else 2
    outs = [jax.ShapeDtypeStruct((B, S), f32), jax.ShapeDtypeStruct((B, S, 2, 3), f32),
            jax.ShapeDtypeStruct((B, S, 3), f32), jax.ShapeDtypeStruct((B, S), jnp.int32),
            jax.ShapeDtypeStruct((B, S, fw), jnp.int32)]
    return tuple(jax.ffi.ffi_call(J["targets"]["query"], outs)(qpos, com, dmax=dmax))


def jax_query_backward(J, qpos, com, g_dist, g_points, max_distance=None):
    """query_backward on the interface state J."""
    import jax
    import jax.numpy as jnp
    dmax = np.float32(J["dmax"] if max_distance is None else max_distance)
    out = jax.ShapeDtypeStruct(qpos.shape, jnp.float32)
    return jax.ffi.ffi_call(J["targets"]["query_vjp"], out)(qpos, com, g_dist, g_points, dmax=dmax)


def query_outputs(r, faces, elements, who="nablaengine.dist_query"):
    """(dist, points, normal, far) of a query's raw outputs, then face (faces=True) and element (elements=True)."""
    if not (faces or elements):
        return r[:4]
    if elements and r[4].shape[-1] != 4:
        raise ValueError(f"{who}: elements=True needs a terrain build (init(terrain=...))")
    return r[:4] + ((r[4][..., :2],) if faces else ()) + ((r[4][..., 2:],) if elements else ())


def jax_query(J, cache, qpos, com, max_distance=None, faces=False, elements=False, who="nablaengine.dist_query"):
    """query on the interface state J (cache: {max_distance: custom_vjp function})."""
    import jax
    import jax.numpy as jnp
    dm = float(np.float32(J["dmax"] if max_distance is None else max_distance))
    if dm not in cache:
        @jax.custom_vjp
        def f(q, c):
            return jax_query_forward(J, q, c, dm)

        def f_fwd(q, c):
            return jax_query_forward(J, q, c, dm), (q, c)

        def f_bwd(res, ct):
            q, c = res
            return jax_query_backward(J, q, c, ct[0], ct[1], dm), jnp.zeros_like(c)

        f.defvjp(f_fwd, f_bwd)
        cache[dm] = f
    return query_outputs(cache[dm](qpos, com), faces, elements, who)


def query_forward(qpos, com, max_distance=None):
    """(dist (B, S), points (B, S, 2, 3), normal (B, S, 3), far (B, S) int32, face (B, S, 2) int32; terrain builds
    (B, S, 4): face, face, element, triangle) without custom_vjp. face: the box face (FACE_PX ... FACE_NZ) of geom 1
    and of geom 2 at the closest points, -1 for other geoms."""
    return jax_query_forward(_need(), qpos, com, max_distance)


def query_backward(qpos, com, g_dist, g_points, max_distance=None):
    """The cotangent of qpos (B, nq, qpos form) for cotangents of dist and points."""
    return jax_query_backward(_need(), qpos, com, g_dist, g_points, max_distance)


def query(qpos, com, max_distance=None, faces=False, elements=False):
    """Differentiable queries (dist, points, normal, far), with faces=True also face (B, S, 2) int32, with
    elements=True (terrain builds) then element (B, S, 2) int32 (module docstring): a jax.custom_vjp whose VJP gives
    qpos's cotangent from dist's and points' (normal, far, face and element get none; com zeros). qpos (B, nq)
    float32, com (B, com_words) float32 (info["engine"]["com_words"])."""
    J = _need()
    return jax_query(J, J["qfn"], qpos, com, max_distance, faces, elements)


_T = {}


def torch_init(model, pairs, max_distance=1.0, *, device=None, foot_pairs=(), com_body=None, max_contacts=None,
               max_rows=None, terrain=None, params=None):
    """torch_ffi.init(device, model=model, ...) and the query library for PyTorch (built where nvcc and mujoco are
    available; ctypes entries on torch's current stream; no jax needed). Arguments as init(); returns its dict."""
    import torch
    from nablaengine import torch_ffi
    check_terrain_params(terrain, params)
    with _LOCK:
        if _T:
            raise RuntimeError("nablaengine.dist_query: torch_init already called (one query library per process)")
        einfo = torch_ffi.init(device, model=model, foot_pairs=list(foot_pairs), com_body=com_body,
                               max_contacts=max_contacts, max_rows=max_rows,
                               **({"terrain": terrain} if terrain is not None else {}),
                               **({"params": params} if params else {}))
        if einfo["engine"] != "forest":
            raise ValueError("nablaengine.dist_query: the forest engine's models only")
        if einfo.get("actuator"):
            raise ValueError("nablaengine.dist_query: not with an actuator model yet")
        dev = torch_ffi._device_index(device)
        cc = torch.cuda.get_device_capability(dev)
        arch = os.environ.get("DK_ARCH") or f"sm_{cc[0]}{cc[1]}"
        b = _build(model, einfo["foot_pairs"], einfo["com_body"], pairs, max_contacts, max_rows, False, arch, None,
                   "libdk_dist_torch", terrain, params)
        lib = ctypes.cdll.LoadLibrary(b["path"])
        if terrain is not None:
            torch_terrain_hooks(lib, [PREFIX])
        vp, i32, f32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        up = getattr(lib, f"{PREFIX}_upload")
        up.argtypes = [vp, vp]
        q = getattr(lib, f"{PREFIX}_query")
        q.argtypes = [vp, i32, vp, vp, f32, vp, vp, vp, vp, vp]
        qv = getattr(lib, f"{PREFIX}_query_vjp")
        qv.argtypes = [vp, i32, vp, vp, f32, vp, vp, vp]
        _check_sizes(lib, int(einfo["nq"]), int(einfo["nv"]), b["nslot"], int(einfo["com_words"]))
        info = _info(b, max_distance, einfo, terrain)
        _T.update(lib=lib, query=q, query_vjp=qv, upload=[up], consts=np.ascontiguousarray(b["consts"], np.float32),
                  ready=set(), info=info, dmax=float(np.float32(max_distance)))
        return info


def torch_ready(T, t, who="nablaengine.dist_query"):
    """The device index of tensor t with the interface state T's constants uploaded there (once per device)."""
    import torch
    if not T:
        raise RuntimeError(f"{who}: call torch_init(model, pairs) first")
    dev = t.device.index
    if dev not in T["ready"]:
        with torch.cuda.device(dev):
            st = torch.cuda.current_stream(dev)
            for f in T["upload"]:
                rc = f(ctypes.c_void_p(T["consts"].ctypes.data), ctypes.c_void_p(st.cuda_stream))
                if rc:
                    raise RuntimeError(f"{who}: constant upload failed ({rc})")
            st.synchronize()
        T["ready"].add(dev)
    return dev


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


def torch_query_forward(T, qpos, com, dmax, who="nablaengine.dist_query"):
    """(dist, points, normal, far, face) tensors of the interface state T's query entry (no autograd)."""
    import torch
    dev = torch_ready(T, qpos, who)
    B, S = qpos.shape[0], T["info"]["nslot"]
    kw = dict(dtype=torch.float32, device=qpos.device)
    dist, pts, nrm = torch.empty((B, S), **kw), torch.empty((B, S, 2, 3), **kw), torch.empty((B, S, 3), **kw)
    far = torch.empty((B, S), dtype=torch.int32, device=qpos.device)
    face = torch.empty((B, S, 4 if T["info"].get("terrain") else 2), dtype=torch.int32, device=qpos.device)
    with torch.cuda.device(dev):
        st = torch.cuda.current_stream(dev).cuda_stream
        rc = T["query"](ctypes.c_void_p(st), B, _p(qpos), _p(com), dmax, _p(dist), _p(pts), _p(nrm), _p(far),
                        _p(face))
    if rc:
        raise RuntimeError(f"{who}: query launch failed ({rc})")
    return dist, pts, nrm, far, face


def torch_query_backward(T, qpos, com, gd, gp, dmax, who="nablaengine.dist_query"):
    """The qpos cotangent of the interface state T's query VJP entry."""
    import torch
    dev = torch_ready(T, qpos, who)
    gq = torch.empty_like(qpos)
    with torch.cuda.device(dev):
        st = torch.cuda.current_stream(dev).cuda_stream
        rc = T["query_vjp"](ctypes.c_void_p(st), qpos.shape[0], _p(qpos), _p(com), dmax, _p(gd.contiguous()),
                            _p(gp.contiguous()), _p(gq))
    if rc:
        raise RuntimeError(f"{who}: query VJP launch failed ({rc})")
    return gq


def torch_query_on(T, qpos, com, max_distance=None, faces=False, elements=False, who="nablaengine.dist_query"):
    """torch_query on the interface state T."""
    import torch
    if not T:
        raise RuntimeError(f"{who}: call torch_init(model, pairs) first")
    dmax = float(np.float32(T["dmax"] if max_distance is None else max_distance))

    class Q(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, c):
            ctx.save_for_backward(q, c)
            d, p, n, f, fc = torch_query_forward(T, q, c, dmax, who)
            ctx.mark_non_differentiable(n, f, fc)
            return d, p, n, f, fc

        @staticmethod
        def backward(ctx, gd, gp, gn, gf, gfc):
            q, c = ctx.saved_tensors
            gd = torch.zeros((q.shape[0], T["info"]["nslot"]), device=q.device) if gd is None else gd
            gp = torch.zeros((q.shape[0], T["info"]["nslot"], 2, 3), device=q.device) if gp is None else gp
            return torch_query_backward(T, q, c, gd, gp, dmax, who), None

    return query_outputs(Q.apply(qpos, com), faces, elements, who)


def torch_query(qpos, com, max_distance=None, faces=False, elements=False):
    """query() for torch tensors (qpos (B, nq), com (B, com_words) float32 CUDA tensors, contiguous; autograd: qpos's
    gradient from dist's and points'; faces=True: also face; elements=True: also element, terrain builds)."""
    return torch_query_on(_T, qpos, com, max_distance, faces, elements)

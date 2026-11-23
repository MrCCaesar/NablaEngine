"""JAX interface of the ray library (raycore): setup() uploads the static scene (terrain, robot meshes) to the GPU
devices once; cast() casts batched rays inside jit (an XLA FFI call). cast(..., grad="first_order") makes the hit
distance differentiable (raygrad: w.r.t. the sensor pose, the robot's body poses and the object slots' poses)."""
import ctypes
from pathlib import Path

import numpy as np

from nablaengine.sensors import raycore, raygrad
from nablaengine.sensors.torch_rays import _Scene, cell_tops

_STATE = {}


def _lib():
    if "lib" not in _STATE:
        import jax
        from nablaengine.jax_ffi import ensure_jax_ffi
        ensure_jax_ffi()
        dev = jax.devices("gpu")[0]
        arch = "sm_" + str(dev.compute_capability).replace(".", "")
        lib = ctypes.cdll.LoadLibrary(str(raycore.build(arch, jax.ffi.include_dir())))
        lib.dk_rays_set_scene.argtypes = [ctypes.c_int, ctypes.c_void_p]
        assert lib.dk_scene_size() == ctypes.sizeof(_Scene), "Scene layout mismatch"
        jax.ffi.register_ffi_target("dk_scene_rays", jax.ffi.pycapsule(lib.DkSceneRays), platform="CUDA")
        _STATE["lib"] = lib
    return _STATE["lib"]


def setup(terrain=None, cell=1.0, floor_z=None, robot=None, nbody=0, exclude_body=-1, devices=None, backend="software"):
    """Upload the static scene to the JAX GPU devices (default all): terrain (nablaengine.terrain.Terrain), an optional
    floor plane z = floor_z, robot meshes (robot_mesh.load's dict) on nbody bodies, exclude_body (-1: none). backend:
    "software" (raycore, the default), "optix" (an optional backend: RT cores, needs an OptiX SDK at $OPTIX_ROOT,
    sensors/optix_rays), "auto" (OptiX where $OPTIX_ROOT is set, else software)."""
    import jax
    import jax.numpy as jnp
    from nablaengine.sensors import optix_rays
    if backend not in ("auto", "optix", "software"):
        raise ValueError(f"backend {backend!r}")
    use_optix = backend == "optix" or (backend == "auto" and optix_rays.available())
    lib = _lib()
    keep = []
    xlib = None
    if use_optix:
        dev0 = jax.devices("gpu")[0]
        arch = "sm_" + str(dev0.compute_capability).replace(".", "")
        libp, ptx = optix_rays.build(arch, jax.ffi.include_dir())
        xlib = ctypes.cdll.LoadLibrary(str(libp))
        vp, ci, cu = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint
        xlib.dkx_create_on.argtypes = [ci, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(vp)]
        xlib.dkx_set_geometry_on.argtypes = [ci, vp, vp, cu, vp, vp, vp, ci, vp]
        xlib.dkx_jax_setup.argtypes = [ci, vp, vp, vp, vp, vp]
        if "optix_target" not in _STATE:
            jax.ffi.register_ffi_target("dk_optix_rays", jax.ffi.pycapsule(xlib.DkOptixRays), platform="CUDA")
            _STATE["optix_target"] = True
        ptx_code = Path(ptx).read_bytes()
    for d in devices or [d for d in jax.devices() if d.platform == "gpu"]:
        put = lambda x, dt: jax.device_put(jnp.asarray(np.ascontiguousarray(x).reshape(-1) if np.size(x) else
                                                       np.zeros(1), dt), d)
        s = _Scene()
        arrs = []
        if terrain is not None:
            pk = terrain.pack(cell)
            ent = dict(box=(pk["boxes"], jnp.float32), hfh=(pk["hf_head"], jnp.float32),
                       hfd=(pk["hf_data"], jnp.float32), grid=(pk["grid"], jnp.float32),
                       cellz=(cell_tops(terrain, pk), jnp.float32), hfi=(pk["hf_int"], jnp.int32),
                       cstart=(pk["cell_start"], jnp.int32), citems=(pk["cell_items"], jnp.int32))
            s.zmax = float(np.max(ent["cellz"][0]))
            for k, (x, dt) in ent.items():
                a = put(x, dt)
                arrs.append(a)
                setattr(s, k, a.unsafe_buffer_pointer())
        s.has_floor, s.floor_z = int(floor_z is not None), float(floor_z or 0.0)
        if robot is not None:
            for k in ("mgeom", "mroot", "nodeb", "nodei", "tri"):
                a = put(robot[k], jnp.int32 if k in ("mroot", "nodei") else jnp.float32)
                arrs.append(a)
                setattr(s, k, a.unsafe_buffer_pointer())
            s.nmesh = len(robot["mgeom"])
        s.nbody, s.exclude_body = int(nbody), int(exclude_body)
        for a in arrs:
            a.block_until_ready()
        rc = lib.dk_rays_set_scene(int(d.local_hardware_id), ctypes.byref(s))
        if rc:
            raise RuntimeError(f"dk_rays_set_scene failed ({rc})")
        if use_optix:
            dev = int(d.local_hardware_id)
            h = ctypes.c_void_p()
            rc = xlib.dkx_create_on(dev, ptx_code, len(ptx_code), ctypes.byref(h))
            if rc:
                raise RuntimeError(f"dkx_create failed ({rc})")
            if terrain is not None:
                tri, el, ce = optix_rays.terrain_triangles(terrain)
                ext = (tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0))
            else:
                tri, el, ce = np.zeros((1, 9), np.float32), np.zeros(1, np.int32), np.zeros(1, np.int32)
                ext = (np.zeros(3), np.zeros(3))
            first, count = (optix_rays.OptixSceneRays._mesh_ranges(robot) if robot is not None
                            else (np.zeros(1, np.int32), np.zeros(1, np.int32)))
            xa = dict(tri=put(tri, jnp.float32), el=put(el, jnp.int32), ce=put(ce, jnp.int32),
                      first=put(first, jnp.int32), box=put(optix_rays.unit_box(), jnp.float32))
            for a in xa.values():
                a.block_until_ready()
            rc = xlib.dkx_set_geometry_on(dev, h, xa["tri"].unsafe_buffer_pointer(), len(tri) if terrain is not None
                                          else 0, s.tri if robot is not None else None, first.ctypes.data,
                                          count.ctypes.data, len(first) if robot is not None else 0,
                                          xa["box"].unsafe_buffer_pointer())
            if rc:
                raise RuntimeError(f"dkx_set_geometry failed ({rc})")
            rc = xlib.dkx_jax_setup(dev, h, ctypes.byref(s), xa["el"].unsafe_buffer_pointer(),
                                    xa["ce"].unsafe_buffer_pointer(), xa["first"].unsafe_buffer_pointer())
            if rc:
                raise RuntimeError(f"dkx_jax_setup failed ({rc})")
            arrs += list(xa.values())
            _STATE["extent"] = ext
        keep.append(arrs)
    _STATE.update(keep=keep, nmesh=int(robot is not None and len(robot["mgeom"])), nbody=int(nbody),
                  mbody=(np.asarray(robot["mgeom"])[:, 0].astype(np.int32) if robot is not None
                         else np.zeros(1, np.int32)),
                  backend="optix" if use_optix else "software")


def _shift(B, maxd):
    lo, hi = _STATE["extent"]
    size = (hi - lo) + 2 * float(maxd) + 4.0
    n = int(np.ceil(B ** (1 / 3)))
    i = np.arange(B)
    g = np.stack([i % n, (i // n) % n, i // (n * n)], 1) - (n - 1) / 2
    return (g * size).astype(np.float32)


def cone(dirs, off=None):
    """(cos, sin, offset_max): the half angle about the sensor x axis that holds every direction of dirs (N, 3), and
    the largest offset norm of off (N, 3) (NumPy; the mesh culling's constants of a ray pattern)."""
    cx = float(np.clip(np.min(np.asarray(dirs)[:, 0]), -1, 1))
    om = float(np.max(np.linalg.norm(np.asarray(off), axis=1))) if off is not None and len(off) else 0.0
    return cx, float(np.sqrt(max(0.0, 1 - cx * cx))), om


def cast(pos, rot, off, dirs, xpos=None, xquat=None, objects=None, max_distance=10.0, cone_cs=None, robot=True,
         exclude_body=-1, backend=None, grad="none", min_cos=raygrad.MIN_COS, impl="fused"):
    """dist (B, N) (-1: none), id (B, N) int32 (raycore.ID_*), normal (B, N, 3) of rays from sensor frames pos (B, 3),
    rot (B, 3, 3) with offsets / unit directions (N, 3) in the sensor frame; xpos (B, nbody, 3) / xquat (B, nbody, 4):
    the body poses (needed with robot meshes); objects (B, K, 15) or None. Traceable (jit); no gradient. cone_cs:
    cone(dirs, off) when they are traced (default: computed from them, which must then be concrete). robot: False skips the
    robot's meshes (e.g. a terrain height scan); exclude_body: a body whose meshes are skipped (-1: none). backend:
    None (setup's), or "software" (the CUDA kernels, set up next to OptiX: better for a few rays per world, as a height
    scan, where OptiX's per-cast IAS build dominates). grad: "none" (dist without gradient, the default) or
    "first_order" (dist differentiable w.r.t. pos, rot, off, dirs, xpos, xquat, objects by the hit plane's first-order
    rule, raygrad; zero for misses and for hits with |n . r| < min_cos; id and normal carry no gradient); the forward
    outputs are the same either way. impl (first_order): "fused" (reverse mode by the fused CUDA backward,
    raygrad_cuda; no forward mode) or "xla" (a custom JVP in XLA: forward and reverse mode; off / dirs get cotangents
    too). Per-world rays: off / dirs (B, N, 3) (e.g. per-world camera intrinsics) with the software kernels, grad
    "none" and cone_cs given (a cone that holds every world's rays); shared (N, 3) rays are cast as before."""
    import jax
    import jax.numpy as jnp
    if np.ndim(dirs) == 3 or np.ndim(off) == 3:
        if grad != "none":
            raise ValueError("cast: per-world rays (B, N, 3) have no gradient (grad='none')")
        if cone_cs is None:
            raise ValueError("cast: per-world rays (B, N, 3) need cone_cs (a cone that holds every world's rays)")
        if _STATE.get("backend") == "optix" and backend != "software":
            raise ValueError("cast: per-world rays (B, N, 3) use the software kernels (backend='software')")
    if grad == "first_order":
        return _cast_first_order(pos, rot, off, dirs, xpos, xquat, objects, max_distance, cone_cs, robot,
                                 exclude_body, backend, min_cos, impl)
    if grad != "none":
        raise ValueError(f"grad {grad!r}: 'none' or 'first_order'")
    if "lib" not in _STATE:
        raise RuntimeError("nablaengine.sensors.jax_rays: setup() first")
    f = lambda x: jnp.asarray(x, jnp.float32)
    pos, rot, off, dirs = f(pos), f(rot), f(off), f(dirs)
    B, N = pos.shape[0], off.shape[-2]
    if off.ndim != dirs.ndim:
        off, dirs = jnp.broadcast_to(off, (B, N, 3)), jnp.broadcast_to(dirs, (B, N, 3))
    nb = max(_STATE["nbody"], 1)
    xpos = f(xpos) if xpos is not None else jnp.zeros((B, nb, 3), jnp.float32)
    xquat = f(xquat) if xquat is not None else jnp.zeros((B, nb, 4), jnp.float32)
    objects = f(objects) if objects is not None else jnp.zeros((B, 0, 15), jnp.float32)
    ch, sh, offmax = cone_cs if cone_cs is not None else cone(np.asarray(dirs), np.asarray(off))
    if _STATE.get("backend") == "optix" and backend != "software":
        out = (jax.ShapeDtypeStruct((B, N), jnp.float32), jax.ShapeDtypeStruct((B, N), jnp.int32),
               jax.ShapeDtypeStruct((B, N, 3), jnp.float32), jax.ShapeDtypeStruct((B, N), jnp.float32),
               jax.ShapeDtypeStruct((B, N), jnp.int32), jax.ShapeDtypeStruct((B, N), jnp.int32))
        dist, gid, nrm, _, _, _ = jax.ffi.ffi_call("dk_optix_rays", out)(
            pos, rot.reshape(B, 3, 3), off, dirs, xpos, xquat, objects, jnp.asarray(_shift(B, max_distance)),
            max_distance=np.float32(max_distance), robot=np.int32(bool(robot)), exclude_body=np.int32(exclude_body))
        return jax.lax.stop_gradient(dist), gid, jax.lax.stop_gradient(nrm)
    nm = max(_STATE["nmesh"], 1)
    out = (jax.ShapeDtypeStruct((B, N), jnp.float32), jax.ShapeDtypeStruct((B, N), jnp.int32),
           jax.ShapeDtypeStruct((B, N, 3), jnp.float32), jax.ShapeDtypeStruct((B, nm, 12), jnp.float32),
           jax.ShapeDtypeStruct((B, 2), jnp.int32))
    dist, gid, nrm, _, _ = jax.ffi.ffi_call("dk_scene_rays", out)(
        pos, rot.reshape(B, 3, 3), off, dirs, xpos, xquat, objects, max_distance=np.float32(max_distance),
        cone_cos=np.float32(ch), cone_sin=np.float32(sh), offset_max=np.float32(offmax), robot=np.int32(bool(robot)),
        exclude_body=np.int32(exclude_body))
    return jax.lax.stop_gradient(dist), gid, jax.lax.stop_gradient(nrm)


def _prep(pos, rot, off, dirs, xpos, xquat, objects):
    import jax.numpy as jnp
    f = lambda x: jnp.asarray(x, jnp.float32)
    pos, rot, off, dirs = f(pos), f(rot), f(off), f(dirs)
    B = pos.shape[0]
    nb = max(_STATE["nbody"], 1)
    xpos = f(xpos) if xpos is not None else jnp.zeros((B, nb, 3), jnp.float32)
    xquat = f(xquat) if xquat is not None else jnp.zeros((B, nb, 4), jnp.float32)
    objects = f(objects) if objects is not None else jnp.zeros((B, 0, 15), jnp.float32)
    return pos, rot.reshape(B, 3, 3), off, dirs, xpos, xquat, objects


def _vjp_lib():
    if "vjp" not in _STATE:
        import jax
        from nablaengine.sensors import raygrad_cuda
        dev = jax.devices("gpu")[0]
        arch = "sm_" + str(dev.compute_capability).replace(".", "")
        lib = ctypes.cdll.LoadLibrary(str(raygrad_cuda.build(arch, jax.ffi.include_dir())))
        assert lib.dk_ray_vjp_args_size() == ctypes.sizeof(raygrad_cuda.VjpArgs), "VjpArgs layout mismatch"
        jax.ffi.register_ffi_target("dk_ray_vjp", jax.ffi.pycapsule(lib.DkRayVjp), platform="CUDA")
        _STATE["vjp"] = lib


def ray_vjp(coef, pos, rot, off, dirs, xpos, xquat, objects, dist, gid, nrm, spec=None, min_cos=raygrad.MIN_COS,
            light=(0.0, 0.0, 0.0)):
    """The fused backward (raygrad_cuda): cotangents (pos, rot, xpos, xquat, objects) of per-ray coefficients coef
    (B, N, 1 | 4 | 5) on (dt, value, col, row, (-light) . n) at the cast's outputs dist, gid, nrm; spec: the image
    sensor (raygrad.ImageSpec) for the value / image coordinates (None: dt alone)."""
    import jax
    import jax.numpy as jnp
    from nablaengine.sensors import raygrad_cuda
    _vjp_lib()
    B, nb, K = pos.shape[0], xpos.shape[1], objects.shape[1]
    ptype, par = raygrad_cuda.params(spec, min_cos, light)
    S = 12 + 12 * (nb + K)
    names = ("p0", "p1", "min_cos", "ac0", "ac1", "ac2", "aw0", "aw1", "aw2", "l0", "l1", "l2")
    out = jax.ffi.ffi_call("dk_ray_vjp", jax.ShapeDtypeStruct((B, S), jnp.float32))(
        pos, rot.reshape(B, 3, 3), off, dirs, xpos, xquat, objects, jnp.asarray(_STATE["mbody"], jnp.int32),
        dist, gid, nrm, jnp.asarray(coef, jnp.float32), ptype=np.int32(ptype),
        nmesh=np.int32(_STATE["nmesh"]), **{k: np.float32(v) for k, v in zip(names, par)})
    bod = out[:, 12:12 + 12 * nb].reshape(B, nb, 12)
    cq = jax.vjp(lambda q: raygrad.qmat(jnp, q), xquat)[1](bod[..., 3:].reshape(B, nb, 3, 3))[0]
    co = jnp.concatenate([out[:, 12 + 12 * nb:].reshape(B, K, 12), jnp.zeros((B, K, 3), jnp.float32)], -1)
    return out[:, :3], out[:, 3:12].reshape(B, 3, 3), bod[..., :3], cq, co


def _cast_first_order(pos, rot, off, dirs, xpos, xquat, objects, max_distance, cone_cs, robot, exclude_body, backend,
                      min_cos, impl="fused"):
    import jax
    import jax.numpy as jnp
    pos, rot, off, dirs, xpos, xquat, objects = _prep(pos, rot, off, dirs, xpos, xquat, objects)
    if cone_cs is None:
        cone_cs = cone(np.asarray(dirs), np.asarray(off))
    kw = dict(max_distance=max_distance, cone_cs=cone_cs, robot=robot, exclude_body=exclude_body, backend=backend)
    mbody = _STATE["mbody"]
    if impl == "fused":
        @jax.custom_vjp
        def runv(pos, rot, off, dirs, xpos, xquat, objects):
            return cast(pos, rot, off, dirs, xpos, xquat, objects, **kw)

        def fwd(*a):
            out = runv(*a)
            return out, (a, out)

        def bwd(r, cts):
            a, (dist, gid, nrm) = r
            cp, cr, cx, cq, co = ray_vjp(cts[0][..., None], *a, dist, gid, nrm, None, min_cos)
            return cp, cr, jnp.zeros_like(a[2]), jnp.zeros_like(a[3]), cx, cq, co

        runv.defvjp(fwd, bwd)
        return runv(pos, rot, off, dirs, xpos, xquat, objects)
    if impl != "xla":
        raise ValueError(f"impl {impl!r}: 'fused' or 'xla'")

    @jax.custom_jvp
    def run(pos, rot, off, dirs, xpos, xquat, objects):
        return cast(pos, rot, off, dirs, xpos, xquat, objects, **kw)

    @run.defjvp
    def run_jvp(primals, tangents):
        dist, gid, nrm = run(*primals)
        res = raygrad.residuals(jnp, *primals, dist, gid, nrm, jnp.asarray(mbody), min_cos)
        _, ddist = jax.jvp(lambda *a: raygrad.tangent_dist(jnp, *a, res), primals, tangents)
        return (dist, gid, nrm), (ddist, np.zeros(gid.shape, jax.dtypes.float0), jnp.zeros_like(nrm))

    return run(pos, rot, off, dirs, xpos, xquat, objects)


def cast_image(spec, value_fn, sigma, pos, rot, off, dirs, xpos=None, xquat=None, objects=None, impl="fused",
               min_cos=raygrad.MIN_COS, light=None, **cast_kw):
    """An image sensor's value with the smoothed gradient (raygrad: sigma in pixels / grid cells; sigma None: the
    per-pixel value tangent alone, the RGB's "first_order"). value_fn(dist, gid, nrm, pos, rot) -> (image (B, H, W[,
    C]), aux (not differentiable, e.g. the seg ids), m (B, H, W) the moving-value mask, kappa (B, N, C) or None: d
    image / d ((-light) . n) per channel (the RGB), the normal the tangents use (B, N, 3)); the forward value is
    value_fn's. impl "fused" (custom VJP, the fused backward) or "xla" (custom JVP). cast_kw: cast()'s max_distance,
    cone_cs, robot, exclude_body, backend. Returns (image, aux)."""
    import jax
    import jax.numpy as jnp
    pos, rot, off, dirs, xpos, xquat, objects = _prep(pos, rot, off, dirs, xpos, xquat, objects)
    if cast_kw.get("cone_cs") is None:
        cast_kw["cone_cs"] = cone(np.asarray(dirs), np.asarray(off))
    mbody = _STATE["mbody"]
    B, N = pos.shape[0], off.shape[0]
    lt = tuple(float(x) for x in light) if light is not None else (0.0, 0.0, 0.0)

    def forward(pos, rot, off, dirs, xpos, xquat, objects):
        dist, gid, nrm = cast(pos, rot, off, dirs, xpos, xquat, objects, **cast_kw)
        img, aux, m, kappa, nf = value_fn(dist, gid, nrm, pos, rot)
        return (img, aux), (dist, gid, nf, img, m.astype(jnp.float32), kappa)

    def to_img(x):
        return x.reshape((B, spec.H, spec.W) + x.shape[2:])

    if impl == "fused":
        @jax.custom_vjp
        def runv(*a):
            return forward(*a)[0]

        def fwd(*a):
            out, r = forward(*a)
            return out, (a, r)

        def bwd(res, cts):
            a, (dist, gid, nf, img, m, kappa) = res
            c = cts[0]
            if sigma is None:
                cV = (m[..., None] if img.ndim == 4 else m) * c
                cc = cr = jnp.zeros_like(m)
            else:
                sm = raygrad.Smoother(jnp, spec.H, spec.W, sigma)
                cV, cc, cr = raygrad.image_transpose(sm, img, m, to_img(dist), c)
            cV = cV.reshape((B, N) + cV.shape[3:])
            z = jnp.zeros((B, N), jnp.float32)
            if kappa is not None:
                coef = jnp.stack([z, z, cc.reshape(B, N), cr.reshape(B, N), (cV * kappa).sum(-1)], -1)
            else:
                coef = jnp.stack([z, cV, cc.reshape(B, N), cr.reshape(B, N)], -1)
            cp, crot, cx, cq, co = ray_vjp(coef, *a, dist, gid, nf, spec, min_cos, lt)
            return cp, crot, jnp.zeros_like(a[2]), jnp.zeros_like(a[3]), cx, cq, co

        runv.defvjp(fwd, bwd)
        return runv(pos, rot, off, dirs, xpos, xquat, objects)
    if impl != "xla":
        raise ValueError(f"impl {impl!r}: 'fused' or 'xla'")

    @jax.custom_jvp
    def run(*a):
        return forward(*a)[0]

    @run.defjvp
    def run_jvp(primals, tangents):
        (img, aux), (dist, gid, nf, _, m, kappa) = forward(*primals)
        res = raygrad.residuals(jnp, *primals, dist, gid, nf, jnp.asarray(mbody), min_cos)
        _, d = jax.jvp(lambda *a: raygrad.functionals(jnp, *a, res, spec, light), primals, tangents)
        dV = d["ndl"][..., None] * kappa if kappa is not None else d["V"]
        dV, dc, dr = to_img(dV), to_img(d["col"]), to_img(d["row"])
        if sigma is None:
            dimg = (m[..., None] if img.ndim == 4 else m) * dV
        else:
            sm = raygrad.Smoother(jnp, spec.H, spec.W, sigma)
            dimg = raygrad.image_tangent(sm, img, m, to_img(dist), dV, dc, dr)
        return (img, aux), (dimg, np.zeros(aux.shape, jax.dtypes.float0))

    return run(pos, rot, off, dirs, xpos, xquat, objects)

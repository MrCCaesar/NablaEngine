"""PyTorch interface of the ray library (raycore): SceneRays holds the scene on one CUDA device (terrain, per-world
object capacity, robot meshes) and casts batched rays; cast(..., grad="first_order") makes the hit distance
differentiable (raygrad: w.r.t. the sensor pose, the robot's body poses and the object slots' poses)."""
import ctypes

import numpy as np

from nablaengine.sensors import raycore, raygrad


class _Scene(ctypes.Structure):
    _fields_ = [("box", ctypes.c_void_p), ("hfh", ctypes.c_void_p), ("hfd", ctypes.c_void_p), ("grid", ctypes.c_void_p),
                ("cellz", ctypes.c_void_p),
                ("hfi", ctypes.c_void_p), ("cstart", ctypes.c_void_p), ("citems", ctypes.c_void_p),
                ("has_floor", ctypes.c_int), ("floor_z", ctypes.c_float), ("zmax", ctypes.c_float),
                ("nobj", ctypes.c_int),
                ("obj", ctypes.c_void_p), ("nmesh", ctypes.c_int), ("nbody", ctypes.c_int),
                ("exclude_body", ctypes.c_int), ("mgeom", ctypes.c_void_p), ("mroot", ctypes.c_void_p),
                ("nodeb", ctypes.c_void_p), ("nodei", ctypes.c_void_p), ("tri", ctypes.c_void_p),
                ("xpos", ctypes.c_void_p), ("xquat", ctypes.c_void_p), ("mpose", ctypes.c_void_p),
                ("mmask", ctypes.c_void_p)]


_LIB = {}


def cell_tops(terrain, pk):
    """The highest point of each broad-phase cell's elements (nx ny,): a box's top corner, a heightfield's top
    (pos z + size z); -inf for an empty cell."""
    tops = [b[2] + float(np.abs(np.asarray(b[3:12]).reshape(3, 3)[2]) @ np.asarray(b[12:15])) for b in terrain.boxes]
    tops += [h[2] + abs(h[11]) * h[14] + abs(h[9]) * h[12] + abs(h[10]) * h[13] for h in terrain.hf_head]
    tops = np.asarray(tops + [0.0], np.float64)
    nx, ny = int(pk["grid"][3]), int(pk["grid"][4])
    start, items = pk["cell_start"], pk["cell_items"]
    out = np.full(nx * ny, -1e30, np.float32)
    for c in range(nx * ny):
        if start[c + 1] > start[c]:
            out[c] = tops[items[start[c]:start[c + 1]]].max() + 1e-4
    return out


def _lib(arch):
    if arch not in _LIB:
        lib = ctypes.cdll.LoadLibrary(str(raycore.build(arch)))
        vp = ctypes.c_void_p
        f = ctypes.c_float
        lib.dk_scene_rays.argtypes = [vp, vp, ctypes.c_int, ctypes.c_int, vp, vp, vp, vp, f, f, f, f, vp, vp, vp]
        assert lib.dk_scene_size() == ctypes.sizeof(_Scene), "Scene layout mismatch"
        lib.dk_rays_stats.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _LIB[arch] = lib
    return _LIB[arch]


class SceneRays:
    """scene = SceneRays(device, terrain=None, cell=1.0, floor_z=None, robot=None (robot_mesh.load's dict), nbody=0,
    exclude_body=-1); scene.cast(pos, rot, off, dir, max_distance, xpos, xquat, objects) -> dist (B, N), id (B, N) int32,
    normal (B, N, 3). objects: None or (B, K, 15) per-world boxes. grad="first_order": dist differentiable (autograd)
    w.r.t. pos, rot, off, dir, xpos, xquat, objects (raygrad; the forward outputs are the same as grad="none")."""

    def __init__(self, device="cuda", terrain=None, cell=1.0, floor_z=None, robot=None, nbody=0, exclude_body=-1):
        import torch
        self.torch, self.device = torch, torch.device(device)
        cc = torch.cuda.get_device_capability(self.device)
        self.lib = _lib(f"sm_{cc[0]}{cc[1]}")
        t = lambda x, dt: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=self.device)
        self._keep = []
        s = _Scene()
        if terrain is not None:
            pk = terrain.pack(cell)
            arrs = {k: t(pk[k].reshape(-1) if pk[k].size else np.zeros(1), torch.int32 if pk[k].dtype == np.int32
                         else torch.float32) for k in ("boxes", "hf_head", "hf_data", "grid", "hf_int", "cell_start",
                                                       "cell_items")}
            ct = cell_tops(terrain, pk)
            arrs["cellz"] = t(ct, torch.float32)
            s.zmax = float(ct.max())
            self._keep += list(arrs.values())
            s.box, s.hfh, s.hfd, s.grid, s.cellz = (arrs[k].data_ptr() for k in ("boxes", "hf_head", "hf_data", "grid",
                                                                                  "cellz"))
            s.hfi, s.cstart, s.citems = (arrs[k].data_ptr() for k in ("hf_int", "cell_start", "cell_items"))
        s.has_floor, s.floor_z = int(floor_z is not None), float(floor_z or 0.0)
        if robot is not None:
            r = {k: t(robot[k], torch.int32 if k in ("mroot", "nodei") else torch.float32)
                 for k in ("mgeom", "mroot", "nodeb", "nodei", "tri")}
            self._keep += list(r.values())
            s.nmesh = len(robot["mgeom"])
            s.mgeom, s.mroot, s.nodeb, s.nodei, s.tri = (r[k].data_ptr() for k in ("mgeom", "mroot", "nodeb", "nodei", "tri"))
        s.nbody, s.exclude_body = int(nbody), int(exclude_body)
        self.scene = s
        self.mbody = t(np.asarray(robot["mgeom"])[:, 0].astype(np.int32) if robot is not None
                       else np.zeros(1, np.int32), torch.int64)
        self.mbody32 = self.mbody.to(torch.int32)
        self.arch = f"sm_{cc[0]}{cc[1]}"

    def ray_vjp(self, coef, pos, rot, off, dirs, xpos, xquat, objects, dist, gid, nrm, spec=None,
                min_cos=raygrad.MIN_COS, light=(0.0, 0.0, 0.0)):
        """The fused backward (raygrad_cuda, as jax_rays.ray_vjp): cotangents (pos, rot, xpos, xquat, objects) of the
        per-ray coefficients coef (B, N, 1 | 4 | 5)."""
        torch = self.torch
        from nablaengine.sensors import raygrad_cuda
        lib = _vjp_lib(self.arch)
        B, N = dist.shape
        nb, K = xpos.shape[1], objects.shape[1]
        ptype, par = raygrad_cuda.params(spec, min_cos, light)
        A = raygrad_cuda.VjpArgs(B, N, nb, K, self.scene.nmesh, coef.shape[2], ptype, *[float(x) for x in par[:3]])
        A.acam[:], A.awld[:], A.light[:] = ([float(x) for x in par[i:i + 3]] for i in (3, 6, 9))
        c = lambda x: x.detach().contiguous()
        ins = [c(pos), c(rot), c(off), c(dirs), c(xpos), c(xquat), c(objects), self.mbody32, c(dist), c(gid), c(nrm),
               c(coef)]
        out = torch.empty(B, 12 + 12 * (nb + K), device=self.device)
        st = torch.cuda.current_stream(self.device).cuda_stream
        rc = lib.dk_ray_vjp(st, ctypes.byref(A), *[x.data_ptr() if x.numel() else None for x in ins], out.data_ptr())
        if rc:
            raise RuntimeError(f"dk_ray_vjp failed ({rc})")
        bod = out[:, 12:12 + 12 * nb].reshape(B, nb, 12)
        with torch.enable_grad():
            q = xquat.detach().requires_grad_(True)
            cq, = torch.autograd.grad(raygrad.qmat(torch, q), q, bod[..., 3:].reshape(B, nb, 3, 3))
        co = torch.cat([out[:, 12 + 12 * nb:].reshape(B, K, 12), torch.zeros(B, K, 3, device=self.device)], -1)
        return out[:, :3], out[:, 3:12].reshape(B, 3, 3), bod[..., :3].contiguous(), cq, co

    def cast_image(self, spec, value_fn, sigma, pos, rot, off, dirs, max_distance=10.0, xpos=None, xquat=None,
                   objects=None, min_cos=raygrad.MIN_COS):
        """An image sensor's value with the smoothed gradient (as jax_rays.cast_image; the fused backward; no RGB):
        value_fn(dist, gid, nrm, pos, rot) -> (image (B, H, W), aux, m (B, H, W), None, normal); sigma None: the
        per-pixel value tangent alone. Returns (image, aux)."""
        torch = self.torch
        B = pos.shape[0]
        f = lambda x: x.to(self.device, torch.float32)
        nb = max(self.scene.nbody, 1)
        xpos = f(xpos) if xpos is not None else torch.zeros(B, nb, 3, device=self.device)
        xquat = f(xquat) if xquat is not None else torch.zeros(B, nb, 4, device=self.device)
        objects = f(objects) if objects is not None else torch.zeros(B, 0, 15, device=self.device)
        img, aux = _image_function().apply(self, spec, value_fn, sigma, float(max_distance), float(min_cos), f(pos),
                                           f(rot).reshape(B, 3, 3), f(off), f(dirs), xpos, xquat, objects)
        return img, aux

    def _cone(self, dirs):
        key = (dirs.data_ptr(), tuple(dirs.shape))
        if getattr(self, "_cone_key", None) != key:
            cx = float(dirs[:, 0].min().clamp(-1, 1))
            self._cone_key, self._cone_val = key, (cx, float(np.sqrt(max(0.0, 1 - cx * cx))))
        return self._cone_val

    def cast(self, pos, rot, off, dirs, max_distance=10.0, xpos=None, xquat=None, objects=None, normals=False,
             grad="none", min_cos=raygrad.MIN_COS, impl="fused"):
        if grad == "first_order":
            torch = self.torch
            B = pos.shape[0]
            f = lambda x: x.to(self.device, torch.float32)
            nb = max(self.scene.nbody, 1)
            xpos = f(xpos) if xpos is not None else torch.zeros(B, nb, 3, device=self.device)
            xquat = f(xquat) if xquat is not None else torch.zeros(B, nb, 4, device=self.device)
            objects = f(objects) if objects is not None else torch.zeros(B, 0, 15, device=self.device)
            dist, gid, nrm = _autograd_function().apply(self, impl, float(max_distance), float(min_cos), f(pos),
                                                   f(rot).reshape(B, 3, 3), f(off), f(dirs), xpos, xquat, objects)
            return (dist, gid, nrm) if normals else (dist, gid)
        if grad != "none":
            raise ValueError(f"grad {grad!r}: 'none' or 'first_order'")
        torch = self.torch
        B, N = pos.shape[0], off.shape[0]
        f = lambda x: x.to(self.device, torch.float32).contiguous()
        pos, rot, off, dirs = f(pos), f(rot).reshape(B, 9), f(off), f(dirs)
        s = _Scene.from_buffer_copy(self.scene)
        keep = []
        if self.scene.nmesh:
            assert xpos is not None and xquat is not None, "robot meshes need the body poses xpos, xquat"
            xpos, xquat = f(xpos), f(xquat)
            mpose = torch.empty(B, self.scene.nmesh, 12, device=self.device)
            mmask = torch.empty(B, 2, dtype=torch.int32, device=self.device)
            keep += [xpos, xquat, mpose, mmask]
            s.xpos, s.xquat, s.mpose, s.mmask = xpos.data_ptr(), xquat.data_ptr(), mpose.data_ptr(), mmask.data_ptr()
        if objects is not None:
            objects = f(objects)
            keep.append(objects)
            s.nobj, s.obj = objects.shape[1], objects.data_ptr()
        dist = torch.empty(B, N, device=self.device)
        gid = torch.empty(B, N, dtype=torch.int32, device=self.device)
        nrm = torch.empty(B, N, 3, device=self.device) if normals else None
        st = torch.cuda.current_stream(self.device).cuda_stream
        cone = self._cone(dirs)
        rc = self.lib.dk_scene_rays(st, ctypes.byref(s), B, N, pos.data_ptr(), rot.data_ptr(), off.data_ptr(),
                                    dirs.data_ptr(), float(max_distance), cone[0], cone[1],
                                    float(off.norm(dim=1).max()) if N else 0.0, dist.data_ptr(),
                                    gid.data_ptr(), nrm.data_ptr() if normals else None)
        if rc:
            raise RuntimeError(f"dk_scene_rays failed ({rc})")
        return (dist, gid, nrm) if normals else (dist, gid)


def _autograd_function():
    if "fn" in _LIB:
        return _LIB["fn"]
    import torch

    class CastFirstOrder(torch.autograd.Function):
        """SceneRays.cast with dist differentiable: forward = the kernel unchanged; backward = the transpose of
        raygrad.tangent_dist's derivative at the saved hits in closed form (raygrad.vjp_torch)."""

        @staticmethod
        def forward(ctx, scene, impl, maxd, min_cos, pos, rot, off, dirs, xpos, xquat, objects):
            nobj = objects if objects.shape[1] else None
            dist, gid, nrm = scene.cast(pos, rot, off, dirs, maxd, xpos, xquat, nobj, normals=True)
            ctx.scene, ctx.min_cos, ctx.impl = scene, min_cos, impl
            ctx.save_for_backward(pos, rot, off, dirs, xpos, xquat, objects, dist, gid, nrm)
            ctx.mark_non_differentiable(gid, nrm)
            return dist, gid, nrm

        @staticmethod
        def backward(ctx, gdist, _gid, _nrm):
            pos, rot, off, dirs, xpos, xquat, objects, dist, gid, nrm = ctx.saved_tensors
            ins = (pos, rot, off, dirs, xpos, xquat, objects)
            need = ctx.needs_input_grad[4:]
            if ctx.impl == "fused":
                cp, cr, cx, cq, co = ctx.scene.ray_vjp(gdist.contiguous()[..., None], *ins, dist, gid, nrm, None,
                                                       ctx.min_cos)
                out = [cp, cr, None, None, cx, cq, co]
                out = [o if n else None for o, n in zip(out, need)]
            else:
                out = raygrad.vjp_torch(gdist, *ins, dist, gid, nrm, ctx.scene.mbody, ctx.min_cos, need)
            return (None, None, None, None, *out)

    _LIB["fn"] = CastFirstOrder
    return CastFirstOrder



def _vjp_lib(arch):
    key = ("vjp", arch)
    if key not in _LIB:
        from nablaengine.sensors import raygrad_cuda
        lib = ctypes.cdll.LoadLibrary(str(raygrad_cuda.build(arch)))
        lib.dk_ray_vjp.argtypes = [ctypes.c_void_p, ctypes.POINTER(raygrad_cuda.VjpArgs)] + [ctypes.c_void_p] * 13
        assert lib.dk_ray_vjp_args_size() == ctypes.sizeof(raygrad_cuda.VjpArgs), "VjpArgs layout mismatch"
        _LIB[key] = lib
    return _LIB[key]


def _image_function():
    if "img" in _LIB:
        return _LIB["img"]
    import torch

    class CastImage(torch.autograd.Function):
        """SceneRays.cast_image: forward = the kernel and value_fn; backward = raygrad.image_transpose and the fused
        backward."""

        @staticmethod
        def forward(ctx, scene, spec, value_fn, sigma, maxd, min_cos, pos, rot, off, dirs, xpos, xquat, objects):
            nobj = objects if objects.shape[1] else None
            dist, gid, nrm = scene.cast(pos, rot, off, dirs, maxd, xpos, xquat, nobj, normals=True)
            img, aux, m, _, nf = value_fn(dist, gid, nrm, pos, rot)
            ctx.scene, ctx.spec, ctx.sigma, ctx.min_cos = scene, spec, sigma, min_cos
            ctx.save_for_backward(pos, rot, off, dirs, xpos, xquat, objects, dist, gid, nf, img, m.to(torch.float32))
            ctx.mark_non_differentiable(aux)
            return img, aux

        @staticmethod
        def backward(ctx, c, _aux):
            pos, rot, off, dirs, xpos, xquat, objects, dist, gid, nf, img, m = ctx.saved_tensors
            B, N = dist.shape
            sp = ctx.spec
            if ctx.sigma is None:
                cV, cc, cr = m * c, torch.zeros_like(m), torch.zeros_like(m)
            else:
                cV, cc, cr = raygrad.image_transpose(raygrad.Smoother(torch, sp.H, sp.W, ctx.sigma, like=c), img, m,
                                                     dist.reshape(m.shape), c)
            coef = torch.stack([torch.zeros_like(dist), cV.reshape(B, N), cc.reshape(B, N), cr.reshape(B, N)], -1)
            cp, crot, cx, cq, co = ctx.scene.ray_vjp(coef, pos, rot, off, dirs, xpos, xquat, objects, dist, gid, nf,
                                                     sp, ctx.min_cos)
            need = ctx.needs_input_grad[6:]
            out = [cp, crot, None, None, cx, cq, co]
            return (None,) * 6 + tuple(o if n else None for o, n in zip(out, need))

    _LIB["img"] = CastImage
    return CastImage

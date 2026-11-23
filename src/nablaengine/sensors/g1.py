"""The G1's sensors in JAX (work in progress): G1Sensors casts the D435i depth camera, the MID-360 LiDAR and the
height scan on a batch of states (body poses from nablaengine.jax_ffi.kinematics) against the scene (terrain, per-world
objects, the robot's own visual meshes), with optional noise. Under jit; no gradient by default, first-order
gradients w.r.t. the body poses and object poses with grad="first_order" (raygrad)."""
from dataclasses import dataclass

import mujoco
import numpy as np

from nablaengine.sensors import cameras, jax_rays, raycore, raygrad, robot_mesh


def _einsum(*args):
    import jax
    import jax.numpy as jnp
    return jnp.einsum(*args, precision=jax.lax.Precision.HIGHEST)


@dataclass
class DepthNoise:
    """The depth noise model (RealSense-like; parameters to be fitted to the real camera): z-proportional-squared
    Gaussian sigma = sigma_z2 z^2, random pixel dropout, dropout at depth discontinuities (|dz| to a 4-neighbour above
    edge_jump, kept with probability 1 - edge_drop); invalid pixels are 0."""
    sigma_z2: float = 0.001
    dropout: float = 0.01
    edge_jump: float = 0.1
    edge_drop: float = 0.5


@dataclass
class Shading:
    """The RGB camera's shading (G1Sensors(rgb=...)): each hit's colour by what it hit (the robot's visual meshes:
    their MJCF rgba; terrain boxes, heightfields, the floor plane, object slots: these colours, or per-element arrays
    box_rgb (nbox, 3), hf_rgb (nhf, 3), object_rgb (K, 3)), lit by a directional light (light_dir: the light's
    travel direction, world) with Lambert's cosine plus an ambient term; sky for no hit. No shadows, no textures."""
    light_dir: tuple = (-0.3, -0.2, -0.93)
    ambient: float = 0.35
    diffuse: float = 0.65
    sky: tuple = (0.62, 0.74, 0.88)
    box: tuple = (0.62, 0.58, 0.52)
    hfield: tuple = (0.46, 0.50, 0.40)
    floor: tuple = (0.55, 0.55, 0.55)
    obj: tuple = (0.80, 0.35, 0.20)
    box_rgb: object = None
    hf_rgb: object = None
    object_rgb: object = None


class G1Sensors:
    def __init__(self, model, terrain=None, robot_mjcf=None, depth=cameras.D435I_DEPTH, lidar=None,
                 height_scan=cameras.HeightScan(), housing=0.03, cell=1.0, floor_z=None, mesh_cache=None,
                 backend="software", rgb=None, shading=None, grad="none", grad_impl="fused", robot_geometry="visual",
                 mesh_simplify=0.0, rgb_seg=False):
        """model: the engine's MjModel (the bodies the poses refer to); robot_mjcf: an MJCF with the robot's visual
        meshes (geom group 2) on bodies of the same names (None: the robot is not seen); housing: the clearance in
        front of each sensor (hits nearer are ignored: its own housing). depth / lidar / height_scan: None to skip.
        backend: the ray backend (jax_rays.setup): "software" (the CUDA kernels, the default), "optix" (the optional
        OptiX backend: RT cores, an OptiX SDK at $OPTIX_ROOT) or "auto" (OptiX where $OPTIX_ROOT is set). rgb: a
        pinhole camera (e.g. cameras.D435I_COLOR.with_resolution(w, h)) rendered as shaded geometry (shading: a
        Shading, default Shading()). grad: "none" (no gradient, the default), "first_order" (the hit plane's
        gradient w.r.t. xpos, xquat and objects: sensors/raygrad; rgb: the shading of turning surfaces) or
        ("smoothed", sigma) (depth, height scan, rgb: the gradient of the image smoothed by a Gaussian of sigma pixels /
        grid cells, edges included; raygrad), for all sensors or per sensor as a dict {"depth": ..., "lidar": ...,
        "height_scan": ..., "rgb": ...} (missing: "none"; the LiDAR: "none" or "first_order", a global
        ("smoothed", sigma) gives it "first_order"). The forward outputs do not depend on it. grad_impl: "fused"
        (reverse mode by the fused CUDA backward, the default) or "xla" (custom JVPs in XLA: forward and reverse
        mode). robot_geometry: what the rays see of the robot: "visual" (robot_mjcf's visual meshes, the default) or
        "collision" (the model's collision geoms tessellated, robot_mesh.from_collision: cheaper, coarser; robot_mjcf
        not needed); mesh_simplify > 0: the visual meshes simplified by vertex clustering at that cell size in metres
        (robot_mesh.cluster; default 0: as loaded). rgb_seg: also return the RGB cast's hit ids and z-depth
        (rgb_seg (B, H, W), rgb_depth (B, H, W) within the rgb camera's [min_depth, max_depth], 0 invalid) with
        grad["rgb"] "none" (default False: not returned). Changing these changes the outputs; their defaults are
        bitwise as before."""
        import jax
        self.m, self.depth, self.lidar, self.hs, self.housing = model, depth, lidar, height_scan, housing
        self.rgb = rgb
        names = ("depth", "lidar", "height_scan", "rgb")
        if isinstance(grad, dict) and set(grad) - set(names):
            raise ValueError(f"grad: unknown sensors {sorted(set(grad) - set(names))}")
        self.grad = {k: grad.get(k, "none") for k in names} if isinstance(grad, dict) else dict.fromkeys(names, grad)
        if not isinstance(grad, dict) and self.grad["lidar"] != "none":
            self.grad["lidar"] = "first_order"
        for k, v in self.grad.items():
            sm = isinstance(v, tuple) and len(v) == 2 and v[0] == "smoothed" and float(v[1]) >= 0
            if not (v in ("none", "first_order") or (sm and k != "lidar")):
                raise ValueError(f"grad[{k!r}] = {v!r}: 'none', 'first_order' or ('smoothed', sigma) (not the LiDAR)")
        self.grad_impl = grad_impl
        if robot_geometry == "collision":
            first = next((s for s in (depth, rgb, lidar) if s is not None), None)
            rob = robot_mesh.from_collision(model, body=first.body if first is not None else None)
        elif robot_geometry != "visual":
            raise ValueError(f"robot_geometry {robot_geometry!r}: 'visual' or 'collision'")
        elif robot_mjcf and mesh_simplify:
            rob = robot_mesh.load(robot_mjcf, model, cache=mesh_cache, simplify=float(mesh_simplify))
        else:
            rob = robot_mesh.load(robot_mjcf, model, cache=mesh_cache) if robot_mjcf else None
        self.rgb_seg = bool(rgb_seg)
        if self.rgb_seg and (rgb is None or self.grad["rgb"] != "none"):
            raise ValueError("rgb_seg: needs an rgb camera with grad['rgb'] 'none'")
        if rgb is not None:
            self.palette = self._palette(shading or Shading(), terrain, rob)
            l = np.asarray((shading or Shading()).light_dir, np.float64)
            self.light = tuple(float(x) for x in l / np.linalg.norm(l))
        jax_rays.setup(terrain=terrain, cell=cell, floor_z=floor_z, robot=rob, nbody=model.nbody, backend=backend)
        self.body = {}
        self.pat = {}
        for name, s in (("depth", depth), ("lidar", lidar), ("height_scan", height_scan), ("rgb", rgb)):
            if s is None:
                continue
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s.body)
            assert b >= 0, f"no body {s.body!r}"
            off, dirs = s.rays()
            if name == "lidar":
                dirs = s.frames()
                off = housing * dirs
                cs = jax_rays.cone(dirs.reshape(-1, 3), off.reshape(-1, 3))
            else:
                if name != "height_scan":
                    off = off + housing * dirs
                cs = jax_rays.cone(dirs, off)
            mp, mR = s.mount() if hasattr(s, "mount") else (np.zeros(3), np.eye(3))
            self.body[name] = b
            self.pat[name] = (jax.numpy.asarray(off), jax.numpy.asarray(dirs), np.asarray(mp, np.float32),
                              np.asarray(mR, np.float32), cs)
        self._dr_cone = {}

    @staticmethod
    def _palette(sh, terrain, rob):
        import jax.numpy as jnp
        nbox = len(terrain.boxes) if terrain is not None else 0
        nhf = len(terrain.hf_head) if terrain is not None else 0
        tab = lambda arr, n, c: (np.asarray(arr, np.float32).reshape(-1, 3) if arr is not None
                                 else np.tile(np.float32(c), (max(n, 1), 1)))
        robot = rob["rgba"][:, :3] if rob is not None and "rgba" in rob else np.zeros((1, 3), np.float32)
        l = np.asarray(sh.light_dir, np.float64)
        return dict(box=jnp.asarray(tab(sh.box_rgb, nbox, sh.box)), hf=jnp.asarray(tab(sh.hf_rgb, nhf, sh.hfield)),
                    floor=jnp.asarray(np.float32(sh.floor)), obj=jnp.asarray(tab(sh.object_rgb, 1, sh.obj)),
                    robot=jnp.asarray(np.asarray(robot, np.float32)), sky=jnp.asarray(np.float32(sh.sky)),
                    light=jnp.asarray((l / np.linalg.norm(l)).astype(np.float32)), ambient=float(sh.ambient),
                    diffuse=float(sh.diffuse))

    def _shade(self, gid, nrm, vec):
        import jax.numpy as jnp
        P = self.palette
        cat, idx = gid >> 24, gid & 0xFFFFFF
        pick = lambda t: t[jnp.clip(idx, 0, t.shape[0] - 1)]
        base = jnp.where((cat == raycore.ID_ROBOT >> 24)[..., None], pick(P["robot"]),
                         jnp.where((cat == raycore.ID_OBJECT >> 24)[..., None], pick(P["obj"]),
                                   jnp.where((cat == raycore.ID_HFIELD >> 24)[..., None], pick(P["hf"]),
                                             jnp.where((cat == raycore.ID_BOX >> 24)[..., None], pick(P["box"]),
                                                       P["floor"]))))
        n = jnp.where((jnp.sum(nrm * vec, -1) > 0)[..., None], -nrm, nrm)
        lam = jnp.clip(-jnp.sum(n * P["light"], -1), 0.0, 1.0)
        c = base * (P["ambient"] + P["diffuse"] * lam)[..., None]
        return jnp.where((gid >= 0)[..., None], jnp.clip(c, 0.0, 1.0), P["sky"])

    APPEARANCE_KEYS = ("object_rgb", "floor_rgb", "robot_gain", "sky_rgb", "light_dir", "ambient", "diffuse")

    def _shade_dr(self, gid, nrm, vec, app):
        import jax.numpy as jnp
        bad = set(app) - set(self.APPEARANCE_KEYS)
        if bad:
            raise ValueError(f"appearance: unknown keys {sorted(bad)} ({', '.join(self.APPEARANCE_KEYS)})")
        P = self.palette
        B = gid.shape[0]
        f = lambda x: jnp.asarray(x, jnp.float32)
        cat, idx = gid >> 24, gid & 0xFFFFFF
        pick = lambda t: t[jnp.clip(idx, 0, t.shape[0] - 1)]
        robot = pick(P["robot"])
        if app.get("robot_gain") is not None:
            robot = robot * f(app["robot_gain"]).reshape(B, 1, -1)
        if app.get("object_rgb") is not None:
            tab = f(app["object_rgb"]).reshape(B, -1, 3)
            obj = jnp.take_along_axis(tab, jnp.clip(idx, 0, tab.shape[1] - 1)[..., None], axis=1)
        else:
            obj = pick(P["obj"])
        floor = f(app["floor_rgb"]).reshape(B, 1, 3) if app.get("floor_rgb") is not None else P["floor"]
        base = jnp.where((cat == raycore.ID_ROBOT >> 24)[..., None], robot,
                         jnp.where((cat == raycore.ID_OBJECT >> 24)[..., None], obj,
                                   jnp.where((cat == raycore.ID_HFIELD >> 24)[..., None], pick(P["hf"]),
                                             jnp.where((cat == raycore.ID_BOX >> 24)[..., None], pick(P["box"]),
                                                       floor))))
        if app.get("light_dir") is not None:
            ld = f(app["light_dir"]).reshape(B, 3)
            light = (ld / jnp.linalg.norm(ld, axis=-1, keepdims=True))[:, None, :]
        else:
            light = P["light"]
        amb = f(app["ambient"]).reshape(B, 1) if app.get("ambient") is not None else P["ambient"]
        dif = f(app["diffuse"]).reshape(B, 1) if app.get("diffuse") is not None else P["diffuse"]
        sky = f(app["sky_rgb"]).reshape(B, 1, 3) if app.get("sky_rgb") is not None else P["sky"]
        n = jnp.where((jnp.sum(nrm * vec, -1) > 0)[..., None], -nrm, nrm)
        lam = jnp.clip(-jnp.sum(n * light, -1), 0.0, 1.0)
        c = base * (amb + dif * lam)[..., None]
        return jnp.where((gid >= 0)[..., None], jnp.clip(c, 0.0, 1.0), sky)

    def _shade_parts(self, gid, nrm, vec):
        import jax.numpy as jnp
        P = self.palette
        cat, idx = gid >> 24, gid & 0xFFFFFF
        pick = lambda t: t[jnp.clip(idx, 0, t.shape[0] - 1)]
        base = jnp.where((cat == raycore.ID_ROBOT >> 24)[..., None], pick(P["robot"]),
                         jnp.where((cat == raycore.ID_OBJECT >> 24)[..., None], pick(P["obj"]),
                                   jnp.where((cat == raycore.ID_HFIELD >> 24)[..., None], pick(P["hf"]),
                                             jnp.where((cat == raycore.ID_BOX >> 24)[..., None], pick(P["box"]),
                                                       P["floor"]))))
        n = jnp.where((jnp.sum(nrm * vec, -1) > 0)[..., None], -nrm, nrm)
        lam = jnp.clip(-jnp.sum(n * P["light"], -1), 0.0, 1.0)
        c = base * (P["ambient"] + P["diffuse"] * lam)[..., None]
        rgb = jnp.where((gid >= 0)[..., None], jnp.clip(c, 0.0, 1.0), P["sky"])
        lraw = -jnp.sum(n * P["light"], -1)
        live = (gid >= 0) & (lraw > 0.0) & (lraw < 1.0)
        kappa = jnp.where(live[..., None] & (c > 0.0) & (c < 1.0), base * P["diffuse"], 0.0)
        return rgb, kappa, n

    @staticmethod
    def _qmat(q):
        import jax.numpy as jnp
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return jnp.stack([jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                          jnp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                          jnp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)

    @staticmethod
    def _expmap(r):
        import jax.numpy as jnp
        t2 = jnp.sum(r * r, -1)
        t = jnp.sqrt(jnp.maximum(t2, 1e-30))
        small = t2 < 1e-8
        a = jnp.where(small, 1 - t2 / 6, jnp.sin(t) / t)
        b = jnp.where(small, 0.5 - t2 / 24, (1 - jnp.cos(t)) / jnp.maximum(t2, 1e-30))
        z = jnp.zeros_like(r[:, 0])
        K = jnp.stack([jnp.stack([z, -r[:, 2], r[:, 1]], -1), jnp.stack([r[:, 2], z, -r[:, 0]], -1),
                       jnp.stack([-r[:, 1], r[:, 0], z], -1)], -2)
        return jnp.eye(3) + a[:, None, None] * K + b[:, None, None] * _einsum("bij,bjk->bik", K, K)

    def _frame_dr(self, name, xpos, xquat, dr):
        import jax.numpy as jnp
        bad = set(dr) - {"pos", "rot", "intrinsics"}
        if bad:
            raise ValueError(f"camera_dr[{name!r}]: unknown keys {sorted(bad)} (pos, rot, intrinsics)")
        b = self.body[name]
        off, dirs, mp, mR, cs = self.pat[name]
        R = self._qmat(xquat[:, b])
        mpw = jnp.asarray(mp) + (jnp.asarray(dr["pos"], jnp.float32) if dr.get("pos") is not None else 0.0)
        p = xpos[:, b] + _einsum("bij,bj->bi", R, jnp.broadcast_to(mpw, (R.shape[0], 3)))
        Rs = _einsum("bij,jk->bik", R, mR)
        if dr.get("rot") is not None:
            Rs = _einsum("bij,bjk->bik", Rs, self._expmap(jnp.asarray(dr["rot"], jnp.float32)))
        if dr.get("intrinsics") is not None:
            cam = self.depth if name == "depth" else self.rgb
            if name not in self._dr_cone:
                c = cam.noisy_cone_cos()
                self._dr_cone[name] = (c, float(np.sqrt(max(0.0, 1 - c * c))), float(cs[2]) * (1 + 1e-6))
            dirs = cameras.pixel_dirs(jnp, cam.width, cam.height, *cam.noisy_intrinsics(dr["intrinsics"]))
            dirs = dirs.astype(jnp.float32)
            off = (self.housing * dirs).astype(jnp.float32)
            cs = self._dr_cone[name]
        return p, Rs, off, dirs, cs

    def _frame(self, name, xpos, xquat, dr=None):
        import jax.numpy as jnp
        if dr is not None and name in ("depth", "rgb"):
            return self._frame_dr(name, xpos, xquat, dr)
        b = self.body[name]
        off, dirs, mp, mR, cs = self.pat[name]
        R = self._qmat(xquat[:, b])
        if name == "height_scan":
            yaw = jnp.arctan2(R[:, 1, 0], R[:, 0, 0])
            c, s = jnp.cos(yaw), jnp.sin(yaw)
            z, o = jnp.zeros_like(c), jnp.ones_like(c)
            Ry = jnp.stack([jnp.stack([c, -s, z], -1), jnp.stack([s, c, z], -1), jnp.stack([z, z, o], -1)], -2)
            return xpos[:, b], Ry, off, dirs, cs
        return xpos[:, b] + _einsum("bij,j->bi", R, mp), _einsum("bij,jk->bik", R, mR), off, dirs, cs

    def __call__(self, xpos, xquat, objects=None, lidar_frame=0, camera_dr=None, appearance=None):
        """dict of depth (B, H, W) z-depth in metres (0 invalid: no hit, nearer than min_depth, beyond max_depth), seg
        (B, H, W) int32 ids (raycore.ID_*, -1 none), lidar (B, n, 3) points in the LiDAR frame (zeros for no return),
        lidar_valid (B, n), height_scan (B, ny, nx) (body z - hit z; max_distance - offset_z for no hit), rgb (B, H, W,
        3) in [0, 1] (with an rgb camera). lidar_frame:
        the LiDAR pattern's frame (an int, traceable; modulo the pattern's frames: cameras.Lidar.frames), e.g. the step
        count at the LiDAR's rate; every world sees the same frame (a per-world yaw of the mount varies the rays).
        camera_dr (optional, per-world camera randomization of the depth and rgb cameras; None, the default: bitwise
        as without it): a dict with any of pos (B, 3) metres added to the mount's position in its body's frame, rot
        (B, 3) an axis-angle rotation (radians) of the camera in its own frame (x forward, y left, z up) after the
        mount's rotation, intrinsics (B, 4) (relative change of fx, of fy, shift of cx as a fraction of the width, of
        cy as a fraction of the height; each clipped to +-cameras.INTRINSICS_NOISE_MAX); the same for both cameras, or
        {"depth": {...}, "rgb": {...}} per camera. Intrinsics noise casts per-world rays (software backend, grad
        "none" for that camera). appearance (optional, per-world RGB shading; None, the default: bitwise as without
        it): a dict with any of object_rgb (B, K, 3) the object slots' colours, floor_rgb (B, 3), robot_gain (B,) or
        (B, 3) a multiplier of the robot meshes' colours, sky_rgb (B, 3), light_dir (B, 3) the light's travel
        direction (normalised here), ambient (B,), diffuse (B,) (the directional light's intensity; Shading's
        ambient 0.35 and diffuse 0.65 by default); missing keys keep Shading's values; rgb grad "none" only."""
        import jax.numpy as jnp
        out = {}
        if camera_dr is not None:
            per = set(camera_dr) <= {"depth", "rgb"} and len(camera_dr) > 0
            cdr = {k: (camera_dr.get(k) if per else camera_dr) for k in ("depth", "rgb")}
        else:
            cdr = {"depth": None, "rgb": None}
        for k in ("depth", "rgb"):
            if cdr[k] is not None and cdr[k].get("intrinsics") is not None and self.grad[k] != "none":
                raise ValueError(f"camera_dr intrinsics: per-world rays have no gradient (grad[{k!r}] must be 'none')")
        if appearance is not None and self.grad["rgb"] != "none":
            raise ValueError("appearance: per-world shading has no gradient (grad['rgb'] must be 'none')")
        if self.depth is not None and cdr["depth"] is not None:
            p, R, off, dirs, cs = self._frame("depth", xpos, xquat, cdr["depth"])
            maxd = self.depth.max_depth / max(float(cs[0]) if dirs.ndim == 3 else
                                              float(np.min(np.asarray(dirs)[:, 0])), 1e-3)
            H, W = self.depth.height, self.depth.width
            g = self.grad["depth"]
            if isinstance(g, tuple):
                def vdep(d, gid, nrm, pos, rot, dirs=dirs):
                    d = jnp.where(d >= 0, d + self.housing, -1.0)
                    z = d * dirs[:, 0]
                    ok = (d >= 0) & (z >= self.depth.min_depth) & (z <= self.depth.max_depth)
                    return (jnp.where(ok, z, 0.0).reshape(-1, H, W), jnp.where(d >= 0, gid, -1).reshape(-1, H, W),
                            ok.reshape(-1, H, W), None, nrm)
                out["depth"], out["seg"] = jax_rays.cast_image(
                    raygrad.image_spec(self.depth, "depth"), vdep, float(g[1]), p, R, off, dirs, xpos, xquat, objects,
                    impl=self.grad_impl, max_distance=maxd, cone_cs=cs)
            else:
                d, gid, _ = jax_rays.cast(p, R, off, dirs, xpos, xquat, objects, maxd, cs, grad=g,
                                          impl=self.grad_impl,
                                          backend="software" if dirs.ndim == 3 else None)
                d = jnp.where(d >= 0, d + self.housing, -1.0)
                z = d * dirs[..., 0]
                ok = (d >= 0) & (z >= self.depth.min_depth) & (z <= self.depth.max_depth)
                out["depth"] = jnp.where(ok, z, 0.0).reshape(-1, H, W)
                out["seg"] = jnp.where(d >= 0, gid, -1).reshape(-1, H, W)
        elif self.depth is not None:
            p, R, off, dirs, cs = self._frame("depth", xpos, xquat)
            maxd = self.depth.max_depth / max(float(np.min(np.asarray(dirs)[:, 0])), 1e-3)
            H, W = self.depth.height, self.depth.width
            g = self.grad["depth"]
            if isinstance(g, tuple):
                def vdep(d, gid, nrm, pos, rot, dirs=dirs):
                    d = jnp.where(d >= 0, d + self.housing, -1.0)
                    z = d * dirs[:, 0]
                    ok = (d >= 0) & (z >= self.depth.min_depth) & (z <= self.depth.max_depth)
                    return (jnp.where(ok, z, 0.0).reshape(-1, H, W), jnp.where(d >= 0, gid, -1).reshape(-1, H, W),
                            ok.reshape(-1, H, W), None, nrm)
                out["depth"], out["seg"] = jax_rays.cast_image(
                    raygrad.image_spec(self.depth, "depth"), vdep, float(g[1]), p, R, off, dirs, xpos, xquat, objects,
                    impl=self.grad_impl, max_distance=maxd, cone_cs=cs)
            else:
                d, gid, _ = jax_rays.cast(p, R, off, dirs, xpos, xquat, objects, maxd, cs, grad=g,
                                          impl=self.grad_impl)
                d = jnp.where(d >= 0, d + self.housing, -1.0)
                z = d * dirs[:, 0]
                ok = (d >= 0) & (z >= self.depth.min_depth) & (z <= self.depth.max_depth)
                out["depth"] = jnp.where(ok, z, 0.0).reshape(-1, H, W)
                out["seg"] = jnp.where(d >= 0, gid, -1).reshape(-1, H, W)
        if self.lidar is not None:
            p, R, off, dirs, cs = self._frame("lidar", xpos, xquat)
            k = jnp.asarray(lidar_frame, jnp.int32) % dirs.shape[0]
            off, dirs = off[k], dirs[k]
            d, gid, _ = jax_rays.cast(p, R, off, dirs, xpos, xquat, objects, self.lidar.max_range, cs,
                                      exclude_body=self.body["lidar"],
                                      grad=self.grad["lidar"], impl=self.grad_impl)
            d = jnp.where(d >= 0, d + self.housing, -1.0)
            ok = (d >= self.lidar.min_range) & (d <= self.lidar.max_range)
            out["lidar"] = jnp.where(ok[..., None], d[..., None] * dirs[None], 0.0)
            out["lidar_valid"] = ok
            out["lidar_seg"] = jnp.where(ok, gid, -1)
        if self.rgb is not None and (cdr["rgb"] is not None or appearance is not None):
            p, R, off, dirs, cs = self._frame("rgb", xpos, xquat, cdr["rgb"])
            H, W = self.rgb.height, self.rgb.width
            g = self.grad["rgb"]
            if g != "none":
                def vrgb(d, gid, nrm, pos, rot, dirs=dirs):
                    vec = _einsum("bij,nj->bni", rot, dirs)
                    rgb, kappa, nf = self._shade_parts(jnp.where(d >= 0, gid, -1), nrm, vec)
                    return rgb.reshape(-1, H, W, 3), gid, (d >= 0).reshape(-1, H, W), kappa, nf
                out["rgb"], _ = jax_rays.cast_image(
                    raygrad.image_spec(self.rgb, "rgb"), vrgb, float(g[1]) if isinstance(g, tuple) else None, p, R,
                    off, dirs, xpos, xquat, objects, impl=self.grad_impl, light=self.light,
                    max_distance=self.rgb.max_depth, cone_cs=cs)
            else:
                d, gid, nrm = jax_rays.cast(p, R, off, dirs, xpos, xquat, objects, self.rgb.max_depth, cs,
                                            backend="software" if dirs.ndim == 3 else None)
                vec = (_einsum("bij,bnj->bni", R, dirs) if dirs.ndim == 3 else
                       _einsum("bij,nj->bni", R, dirs))
                ids = jnp.where(d >= 0, gid, -1)
                rgb = self._shade(ids, nrm, vec) if appearance is None else self._shade_dr(ids, nrm, vec, appearance)
                out["rgb"] = rgb.reshape(-1, H, W, 3)
                if self.rgb_seg:
                    self._rgb_extra(out, d, ids, dirs)
        elif self.rgb is not None:
            p, R, off, dirs, cs = self._frame("rgb", xpos, xquat)
            H, W = self.rgb.height, self.rgb.width
            g = self.grad["rgb"]
            if g != "none":
                def vrgb(d, gid, nrm, pos, rot, dirs=dirs):
                    vec = _einsum("bij,nj->bni", rot, dirs)
                    rgb, kappa, nf = self._shade_parts(jnp.where(d >= 0, gid, -1), nrm, vec)
                    return rgb.reshape(-1, H, W, 3), gid, (d >= 0).reshape(-1, H, W), kappa, nf
                out["rgb"], _ = jax_rays.cast_image(
                    raygrad.image_spec(self.rgb, "rgb"), vrgb, float(g[1]) if isinstance(g, tuple) else None, p, R,
                    off, dirs, xpos, xquat, objects, impl=self.grad_impl, light=self.light,
                    max_distance=self.rgb.max_depth, cone_cs=cs)
            else:
                d, gid, nrm = jax_rays.cast(p, R, off, dirs, xpos, xquat, objects, self.rgb.max_depth, cs)
                vec = _einsum("bij,nj->bni", R, dirs)
                out["rgb"] = self._shade(jnp.where(d >= 0, gid, -1), nrm, vec).reshape(-1, H, W, 3)
                if self.rgb_seg:
                    self._rgb_extra(out, d, jnp.where(d >= 0, gid, -1), dirs)
        if self.hs is not None:
            p, R, off, dirs, cs = self._frame("height_scan", xpos, xquat)
            ny = len(np.arange(-self.hs.size[1] / 2, self.hs.size[1] / 2 + self.hs.resolution * 0.5, self.hs.resolution))
            g = self.grad["height_scan"]
            if isinstance(g, tuple):
                def vhs(d, gid, nrm, pos, rot):
                    hit_z = pos[:, None, 2] + self.hs.offset_z - jnp.where(d >= 0, d, self.hs.max_distance)
                    img = (pos[:, None, 2] - hit_z).reshape(pos.shape[0], ny, -1)
                    return img, gid.reshape(img.shape), (d >= 0).reshape(img.shape), None, nrm
                out["height_scan"], _ = jax_rays.cast_image(
                    raygrad.image_spec(self.hs, "height"), vhs, float(g[1]), p, R, off, dirs, xpos, xquat, objects,
                    impl=self.grad_impl, max_distance=self.hs.max_distance, cone_cs=cs, robot=False,
                    backend="software")
            else:
                d, _, _ = jax_rays.cast(p, R, off, dirs, xpos, xquat, objects, self.hs.max_distance, cs, robot=False,
                                        backend="software",
                                        grad=g, impl=self.grad_impl)
                hit_z = p[:, None, 2] + self.hs.offset_z - jnp.where(d >= 0, d, self.hs.max_distance)
                out["height_scan"] = (p[:, None, 2] - hit_z).reshape(p.shape[0], ny, -1)
        return out

    def _rgb_extra(self, out, d, ids, dirs):
        import jax.numpy as jnp
        H, W = self.rgb.height, self.rgb.width
        d = jnp.where(d >= 0, d + self.housing, -1.0)
        z = d * dirs[..., 0]
        ok = (d >= 0) & (z >= self.rgb.min_depth) & (z <= self.rgb.max_depth)
        out["rgb_seg"] = ids.reshape(-1, H, W)
        out["rgb_depth"] = jnp.where(ok, z, 0.0).reshape(-1, H, W)

    @staticmethod
    def depth_noise(key, depth, noise=DepthNoise()):
        """depth (B, H, W) with DepthNoise applied (invalid stays 0)."""
        import jax
        import jax.numpy as jnp
        k1, k2, k3 = jax.random.split(key, 3)
        valid = depth > 0
        z = depth + noise.sigma_z2 * depth ** 2 * jax.random.normal(k1, depth.shape)
        drop = jax.random.uniform(k2, depth.shape) < noise.dropout
        pad = jnp.pad(depth, ((0, 0), (1, 1), (1, 1)), mode="edge")
        jump = jnp.maximum(jnp.maximum(jnp.abs(pad[:, 1:-1, 2:] - depth), jnp.abs(pad[:, 1:-1, :-2] - depth)),
                           jnp.maximum(jnp.abs(pad[:, 2:, 1:-1] - depth), jnp.abs(pad[:, :-2, 1:-1] - depth)))
        edge = (jump > noise.edge_jump) & (jax.random.uniform(k3, depth.shape) < noise.edge_drop)
        return jnp.where(valid & ~drop & ~edge, z, 0.0)


class Delay:
    """A latency buffer of L frames for a sensor output (functional: init(x) -> state; push(state, x) ->
    (state, the output L frames ago))."""

    def __init__(self, frames):
        self.L = int(frames)

    def init(self, x):
        import jax.numpy as jnp
        return jnp.repeat(x[None], self.L + 1, 0)

    def push(self, state, x):
        import jax.numpy as jnp
        state = jnp.concatenate([state[1:], x[None]], 0)
        return state, state[0]


def label_map(ids, table):
    """Semantic labels from ray ids: table maps (category, index) to a label: a dict {raycore.ID_*: array of labels per
    index, or one label}; unmapped -> -1."""
    import jax.numpy as jnp
    cat, idx = ids >> 24, ids & raycore.ID_MASK
    out = jnp.full(ids.shape, -1, jnp.int32)
    for c, lab in table.items():
        lab = jnp.asarray(lab, jnp.int32)
        v = lab if lab.ndim == 0 else lab[jnp.clip(idx, 0, lab.shape[0] - 1)]
        out = jnp.where((ids >= 0) & (cat == (c >> 24)), v, out)
    return out


class Imu:
    """An IMU at a site of the engine's model (the G1's imu_in_torso / imu_in_pelvis: Unitree's URDF mounts): gyro =
    the body's angular velocity in the site frame, accelerometer = the site's linear acceleration minus gravity in the
    site frame (the specific force; read(gravity=) takes per-world gravity), the acceleration a finite difference of the site velocity over the control step (MuJoCo's accelerometer
    reads the instantaneous acceleration; over a 20 ms step they agree to the step's change of it). Inputs are the
    kinematics outputs of nablaengine.jax_ffi.kinematics (xpos, xquat, cvel at the root subtree's COM, subtree_com)."""

    def __init__(self, model, site="imu_in_torso", dt=None):
        s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        assert s >= 0, f"no site {site!r}"
        self.body = int(model.site_bodyid[s])
        self.pos = np.asarray(model.site_pos[s], np.float32)
        self.R = cameras.quat_matrix(model.site_quat[s]).astype(np.float32)
        self.dt = float(dt) if dt is not None else None
        self.g = np.asarray(model.opt.gravity, np.float32)
        self.root = 0 if int(model.ntree) <= 1 else int(model.body_rootid[self.body])

    def frame(self, xpos, xquat, cvel, subtree_com):
        """(site position (B, 3), site rotation (B, 3, 3), angular velocity (B, 3) world, site linear velocity (B, 3))."""
        import jax.numpy as jnp
        Rb = G1Sensors._qmat(xquat[:, self.body])
        p = xpos[:, self.body] + _einsum("bij,j->bi", Rb, self.pos)
        R = _einsum("bij,jk->bik", Rb, self.R)
        w, v0 = cvel[:, self.body, :3], cvel[:, self.body, 3:]
        v = v0 + jnp.cross(w, p - subtree_com[:, self.root])
        return p, R, w, v

    def read(self, kin_now, kin_prev, dt=None, gravity=None):
        """dict(gyro (B, 3), accel (B, 3)) in the site frame from the kinematics outputs (xpos, xquat, cvel,
        subtree_com) of this step and the previous one. gravity: each world's gravity (B, 3) or (3,) (a build with
        PhysicsParams gravity: the worlds' values, so the specific force matches each world's dynamics); None: the
        model's opt.gravity (unchanged)."""
        import jax.numpy as jnp
        dt = dt or self.dt
        _, R, w, v = self.frame(*kin_now)
        _, _, _, vp = self.frame(*kin_prev)
        g = self.g if gravity is None else jnp.asarray(gravity, v.dtype)
        acc = (v - vp) / dt - g
        return dict(gyro=_einsum("bji,bj->bi", R, w), accel=_einsum("bji,bj->bi", R, acc))

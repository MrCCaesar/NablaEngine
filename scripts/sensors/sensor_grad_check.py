"""First-order sensor gradients (G1Sensors(grad="first_order"), sensors/raygrad) against float64 central differences
of MuJoCo C's mj_ray on the same scene."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from nablaengine import fixtures
from nablaengine.sensors import cameras, raycore as rc, raygrad
from nablaengine.terrain import Terrain

CATS = {0: "terrain_box", 1: "heightfield", 2: "floor", 3: "object", 4: "robot_mesh"}


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_exp(w):
    a = np.linalg.norm(w)
    if a < 1e-300:
        return np.array([1.0, 0, 0, 0])
    return np.concatenate([[np.cos(a / 2)], np.sin(a / 2) * w / a])


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def terrain_scene(rng):
    """A terrain-only MjSpec: two heightfield patches, stairs and tilted boxes around the origin."""
    spec = mujoco.MjSpec()
    n = 33
    xs = np.linspace(0, 1, n)
    for j, (cx, cy) in enumerate(((2.4, 0.0), (-2.4, 0.3))):
        h = np.zeros((n, n))
        for _ in range(6):
            a, b, s = rng.uniform(0, 1, 2).tolist() + [rng.uniform(0.08, 0.25)]
            h += rng.uniform(0.3, 1.0) * np.exp(-((xs[None] - a) ** 2 + (xs[:, None] - b) ** 2) / (2 * s * s))
        h = (h - h.min()) / (h.max() - h.min())
        spec.add_hfield(name=f"hf{j}", size=[1.5, 1.5, 0.3, 0.05], nrow=n, ncol=n, userdata=h.ravel().tolist())
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=f"hf{j}", pos=[cx, cy, 0.0],
                                quat=[np.cos(0.2 * j), 0, 0, np.sin(0.2 * j)])
    for k in range(4):
        for sgn in (1, -1):
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[1.0, 0.15, 0.06 * (k + 1)],
                                    pos=[0.2, sgn * (1.6 + 0.3 * k), 0.06 * (k + 1)])
    for k in range(5):
        a = np.pi / 4 + k * np.pi / 2.5
        q = quat_mul(quat_exp(np.array([0, 0, a])), quat_exp(rng.normal(0, 0.3, 3)))
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=rng.uniform(0.1, 0.35, 3).tolist(),
                                pos=[2.2 * np.cos(a), 2.2 * np.sin(a), 0.2], quat=q.tolist())
    return Terrain.from_model(spec.compile())


class Reference:
    """MuJoCo C reference: mjlab's g1.xml + the terrain + a floor plane + K mocap boxes; geom poses set from body
    poses (no kinematics), rays by mj_ray in float64."""

    def __init__(self, g1_mjcf, ter, half, m):
        spec = mujoco.MjSpec.from_file(g1_mjcf)
        ter.add_to_spec(spec)
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1])
        for k, hk in enumerate(half):
            b = spec.worldbody.add_body(name=f"obj{k}", mocap=True)
            b.add_geom(name=f"obj{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=hk.tolist())
        self.ref = r = spec.compile()
        self.d = mujoco.MjData(r)
        mujoco.mj_kinematics(r, self.d)
        self.rmesh = [g for g in range(r.ngeom)
                      if r.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and r.geom_group[g] == 2]
        self.rbody = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, r.body(r.geom_bodyid[g]).name) for g in self.rmesh]
        self.gmat = [cameras.quat_matrix(r.geom_quat[g]) for g in self.rmesh]
        self.ogeom = [mujoco.mj_name2id(r, mujoco.mjtObj.mjOBJ_GEOM, f"obj{k}") for k in range(len(half))]
        ids = {}
        for i in range(len(ter.boxes)):
            ids[mujoco.mj_name2id(r, mujoco.mjtObj.mjOBJ_GEOM, f"ter_box{i}")] = rc.ID_BOX + i
        for j in range(len(ter.hf_head)):
            ids[mujoco.mj_name2id(r, mujoco.mjtObj.mjOBJ_GEOM, f"ter_hf{j}")] = rc.ID_HFIELD + j
        ids[mujoco.mj_name2id(r, mujoco.mjtObj.mjOBJ_GEOM, "floor")] = rc.ID_FLOOR
        for k, g in enumerate(self.ogeom):
            ids[g] = rc.ID_OBJECT + k
        for i, g in enumerate(self.rmesh):
            ids[g] = rc.ID_ROBOT + i
        self.ids = ids

    def pose(self, xpos, xquat, obj):
        d = self.d
        for g, b, gm in zip(self.rmesh, self.rbody, self.gmat):
            Rb = cameras.quat_matrix(xquat[b])
            d.geom_xpos[g] = xpos[b] + Rb @ self.ref.geom_pos[g]
            d.geom_xmat[g] = (Rb @ gm).ravel()
        for k, g in enumerate(self.ogeom):
            d.geom_xpos[g] = obj[k, :3]
            d.geom_xmat[g] = obj[k, 3:12]

    def cast(self, orig, vec, group, exclude=-1):
        N = len(orig)
        dist, gid, nrm = np.zeros(N), np.full(N, -1, np.int64), np.zeros((N, 3))
        g1, n1 = np.zeros(1, np.int32), np.zeros(3)
        for i in range(N):
            dist[i] = mujoco.mj_ray(self.ref, self.d, orig[i], vec[i], group, 1, exclude, g1, n1)
            if dist[i] >= 0:
                gid[i] = self.ids.get(int(g1[0]), -2)
                nrm[i] = n1
        return dist, gid, nrm


def sensor_frames(sens, xpos, xquat):
    """float64 mirror of G1Sensors._frame for one world: {name: (origins (N, 3), directions (N, 3))}."""
    out = {}
    for name in ("depth", "height_scan", "lidar"):
        b = sens.body[name]
        off, dirs, mp, mR, _ = sens.pat[name]
        off, dirs = np.asarray(off, np.float64), np.asarray(dirs, np.float64)
        if name == "lidar":
            off, dirs = off[0], dirs[0]
        R = cameras.quat_matrix(xquat[b])
        if name == "height_scan":
            Rf, p = rotz(np.arctan2(R[1, 0], R[0, 0])), xpos[b]
        else:
            Rf, p = R @ mR.astype(np.float64), xpos[b] + R @ mp.astype(np.float64)
        out[name] = (p[None] + off @ Rf.T, dirs @ Rf.T)
    return out


def ref_outputs(ref, sens, xpos, xquat, obj):
    """Reference depth (N,), height scan (M,), LiDAR range (frame 0) and per-ray (id, normal, n . r, valid) as
    G1Sensors defines them."""
    ref.pose(xpos, xquat, obj)
    fr = sensor_frames(sens, xpos, xquat)
    cam, hs = sens.depth, sens.hs
    o, v = fr["depth"]
    t, gid, n = ref.cast(o, v, np.array([1, 1, 1, 0, 0, 0], np.uint8))
    maxd = cam.max_depth / max(float(np.min(np.asarray(sens.pat["depth"][1])[:, 0])), 1e-3)
    t[t > maxd] = -1
    gid[t < 0] = -1
    dsum = np.where(t >= 0, t + sens.housing, -1.0)
    z = dsum * np.asarray(sens.pat["depth"][1], np.float64)[:, 0]
    ok = (dsum >= 0) & (z >= cam.min_depth) & (z <= cam.max_depth)
    depth = np.where(ok, z, 0.0)
    dep = dict(out=depth, id=gid, nrm=n, ndr=np.sum(n * v, 1), valid=ok)
    o, v = fr["height_scan"]
    t, gid, n = ref.cast(o, v, np.array([1, 0, 0, 0, 0, 0], np.uint8))
    t[t > hs.max_distance] = -1
    gid[t < 0] = -1
    out = np.where(t >= 0, t, hs.max_distance) - hs.offset_z
    hsd = dict(out=out, id=gid, nrm=n, ndr=np.sum(n * v, 1), valid=t >= 0)
    li = sens.lidar
    o, v = fr["lidar"]
    excl = ref.ref.body(sens.m.body(sens.body["lidar"]).name).id
    t, gid, n = ref.cast(o, v, np.array([1, 1, 1, 0, 0, 0], np.uint8), excl)
    t[t > li.max_range] = -1
    gid[t < 0] = -1
    dsum = np.where(t >= 0, t + sens.housing, -1.0)
    ok = (dsum >= li.min_range) & (dsum <= li.max_range)
    gid[~ok] = -1
    lid = dict(out=np.where(ok, dsum, 0.0), id=gid, nrm=n, ndr=np.sum(n * v, 1), valid=ok)
    return dep, hsd, lid


def perturb(xpos, xquat, obj, tan, h):
    dx, dw, dop, dow = tan
    xq = np.stack([quat_mul(quat_exp(h * dw[b]), xquat[b]) for b in range(len(xquat))])
    ob = obj.copy()
    ob[:, :3] += h * dop
    for k in range(len(ob)):
        ob[k, 3:12] = (cameras.quat_matrix(quat_exp(h * dow[k])) @ obj[k, 3:12].reshape(3, 3)).ravel()
    return xpos + h * dx, xq, ob


def stats(a, f, cat, keep, floor):
    out = {}
    for c, name in CATS.items():
        s = keep & (cat == c)
        if not s.any():
            out[name] = dict(n=0)
            continue
        err = np.abs(a[s] - f[s])
        rel = err / (np.abs(f[s]) + floor)
        out[name] = dict(n=int(s.sum()), rel_median=float(np.median(rel)), rel_p99=float(np.quantile(rel, 0.99)),
                         rel_max=float(rel.max()), abs_median=float(np.median(err)),
                         abs_p99=float(np.quantile(err, 0.99)), deriv_abs_median=float(np.median(np.abs(f[s]))))
    return out


def provenance():
    def run(*c):
        try:
            return subprocess.run(c, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    import jax
    import torch
    return dict(commit=run("git", "rev-parse", "HEAD"), date=time.strftime("%Y-%m-%d"),
                gpu=torch.cuda.get_device_name(), driver=run("nvidia-smi", "--query-gpu=driver_version",
                                                             "--format=csv,noheader", "-i", "0"),
                jax=jax.__version__, torch=torch.__version__, mujoco=mujoco.__version__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1-mjcf", required=True, help="mjlab's g1.xml (the robot's visual meshes)")
    ap.add_argument("--nworld", type=int, default=16)
    ap.add_argument("--objects", type=int, default=4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--h", type=float, default=1e-5, help="central difference step (float64 reference)")
    ap.add_argument("--lidar-rays", type=int, default=1024)
    ap.add_argument("--time-worlds", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="runs/sensors_cache", help="robot_mesh cache directory (created)")
    ap.add_argument("--dump", default=None, help="npz of the inputs and grad='none' outputs (bitwise comparisons)")
    ap.add_argument("--out", default="docs/results/sensor_grad_check.json")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.force:
        sys.exit(f"{a.out} exists (--force to overwrite)")
    import jax
    import jax.numpy as jnp
    import torch
    from nablaengine.sensors import g1, jax_rays, robot_mesh
    from nablaengine.sensors.torch_rays import SceneRays

    rng = np.random.default_rng(a.seed)
    if a.cache:
        Path(a.cache).mkdir(parents=True, exist_ok=True)
    m, _ = fixtures.load_model()
    ter = terrain_scene(rng)
    B, K = a.nworld, a.objects
    half = rng.uniform(0.08, 0.3, (K, 3))
    ref = Reference(a.g1_mjcf, ter, half, m)
    cam = cameras.D435I_DEPTH.with_resolution(a.width, a.height)
    hs = cameras.HeightScan()
    lidar = cameras.MID360.with_rays(a.lidar_rays)
    sens = g1.G1Sensors(m, terrain=ter, robot_mjcf=a.g1_mjcf, depth=cam, height_scan=hs, floor_z=0.0,
                        lidar=lidar, mesh_cache=a.cache, grad="first_order",
                        grad_impl="xla")
    sens0 = g1.G1Sensors(m, terrain=ter, robot_mjcf=a.g1_mjcf, depth=cam, height_scan=hs, floor_z=0.0,
                         lidar=lidar, mesh_cache=a.cache)

    z = fixtures.load_corpus("mf_s0")
    d = mujoco.MjData(m)
    xpos, xquat, obj = np.zeros((B, m.nbody, 3)), np.zeros((B, m.nbody, 4)), np.zeros((B, K, 15))
    for w in range(B):
        d.qpos[:] = z["qpos"][rng.integers(len(z["qpos"]))]
        yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[3:7] = quat_mul(quat_exp(np.array([0, 0, yaw])), d.qpos[3:7])
        d.qpos[:2] = rng.uniform(-0.4, 0.4, 2)
        d.qpos[2] += 0.05
        mujoco.mj_kinematics(m, d)
        xpos[w], xquat[w] = d.xpos, d.xquat
        for k in range(K):
            ang = yaw + rng.normal(0, 0.7)
            r = rng.uniform(0.7, 2.5)
            R = rotz(ang + rng.uniform(-1, 1)) @ cameras.quat_matrix(quat_exp(rng.normal(0, 0.25, 3)))
            obj[w, k] = np.concatenate([[d.qpos[0] + r * np.cos(ang), d.qpos[1] + r * np.sin(ang),
                                         half[k, 2] + rng.uniform(0, 0.3)], R.ravel(), half[k]])
    tans = [(rng.normal(0, 1, (m.nbody, 3)), rng.normal(0, 1, (m.nbody, 3)), rng.normal(0, 1, (K, 3)),
             rng.normal(0, 1, (K, 3))) for _ in range(B)]

    f32 = lambda x: jnp.asarray(x, jnp.float32)
    keys = ("depth", "seg", "height_scan", "lidar", "lidar_seg")
    lid_dirs = jnp.asarray(sens.pat["lidar"][1][0])

    def pick(o):
        r = {k: v for k, v in o.items() if k in keys}
        r["lidar"] = jnp.sum(r["lidar"] * lid_dirs[None], -1)
        return r
    run = jax.jit(lambda xp, xq, ob: pick(sens(xp, xq, ob)))
    run0 = jax.jit(lambda xp, xq, ob: pick(sens0(xp, xq, ob)))
    dxq = np.zeros_like(xquat)
    dob = np.zeros_like(obj)
    for w in range(B):
        dx, dw, dop, dow = tans[w]
        for b in range(m.nbody):
            dxq[w, b] = 0.5 * quat_mul(np.concatenate([[0], dw[b]]), xquat[w, b])
        dob[w, :, :3] = dop
        for k in range(K):
            dob[w, k, 3:12] = (skew(dow[k]) @ obj[w, k, 3:12].reshape(3, 3)).ravel()
    dxp = np.stack([t[0] for t in tans])
    primals = (f32(xpos), f32(xquat), f32(obj))
    out1, jv = jax.jit(lambda p, t: jax.jvp(lambda *x: run(*x), p, t))(primals, (f32(dxp), f32(dxq), f32(dob)))
    out0 = run0(*primals)
    bitwise_fwd = {k: bool(np.array_equal(np.asarray(out0[k]), np.asarray(out1[k]))) for k in out0}
    if a.dump:
        np.savez(a.dump, xpos=xpos, xquat=xquat, obj=obj, **{k: np.asarray(v) for k, v in out0.items()})
        ter.save(str(a.dump).removesuffix(".npz") + ".terrain.npz")

    res = {k: dict(a=[], f=[], cat=[], keep=[], excl={}) for k in ("depth", "height_scan", "lidar")}
    t0 = time.perf_counter()
    for w in range(B):
        r0 = ref_outputs(ref, sens, xpos[w], xquat[w], obj[w])
        rp = ref_outputs(ref, sens, *perturb(xpos[w], xquat[w], obj[w], tans[w], a.h))
        rm = ref_outputs(ref, sens, *perturb(xpos[w], xquat[w], obj[w], tans[w], -a.h))
        for i, name in enumerate(("depth", "height_scan", "lidar")):
            c, p, q = r0[i], rp[i], rm[i]
            fd = (p["out"] - q["out"]) / (2 * a.h)
            ours = np.asarray(jv[name][w], np.float64).ravel()
            ours0 = np.asarray(out1[name][w], np.float64).ravel()
            hit = c["id"] >= 0
            same = (p["id"] == c["id"]) & (q["id"] == c["id"]) & (p["valid"] == c["valid"]) & (q["valid"] == c["valid"])
            face = (np.abs(p["nrm"] - c["nrm"]).max(1) < 1e-3) & (np.abs(q["nrm"] - c["nrm"]).max(1) < 1e-3)
            graze = hit & (np.abs(c["ndr"]) < raygrad.MIN_COS)
            if name in ("depth", "lidar"):
                match = np.asarray(out1["seg" if name == "depth" else "lidar_seg"][w]).ravel() == c["id"]
            else:
                match = np.abs(ours0 - c["out"]) < 1e-4
            keep = hit & c["valid"] & same & face & ~graze & match
            e = res[name]["excl"]
            for k_, v_ in dict(rays=hit.size, hits=hit, id_or_validity_changes=hit & ~same,
                               face_changes=hit & same & ~face,
                               grazing=hit & same & face & graze,
                               mismatch_vs_reference=hit & same & face & ~graze & ~match,
                               invalid=hit & same & face & ~graze & match & ~c["valid"]).items():
                e[k_] = e.get(k_, 0) + int(np.sum(v_))
            res[name]["a"].append(ours)
            res[name]["f"].append(fd)
            res[name]["cat"].append(np.where(hit, c["id"] >> 24, -1))
            res[name]["keep"].append(keep)
    t_ref = time.perf_counter() - t0
    out = dict(provenance=provenance(), nworld=B, objects=K, depth=f"{a.width}x{a.height}",
               height_scan=list(np.asarray(out1["height_scan"]).shape[1:]), h=a.h, min_cos=raygrad.MIN_COS,
               ref_seconds=round(t_ref, 1), bitwise_forward_none_vs_first_order=bitwise_fwd, fd={})
    ok = all(bitwise_fwd.values())
    for name, r in res.items():
        A, F, C, Kp = (np.concatenate(r[k]) for k in ("a", "f", "cat", "keep"))
        st = stats(A, F, C, Kp, floor=1e-3)
        nh = r["excl"]["hits"]
        out["fd"][name] = dict(per_category=st, counts=r["excl"], kept=int(Kp.sum()),
                               excluded_share_of_hits={k: round(v / max(nh, 1), 5) for k, v in r["excl"].items()
                                                       if k not in ("rays", "hits")},
                               ray_hit_share=round(nh / r["excl"]["rays"], 4),
                               misses_nonzero_grad=int(np.sum((C < 0) & (np.abs(A) > 0))))
        for c in st.values():
            if c.get("n", 0) >= 20 and c["rel_median"] > 1e-3:
                ok = False

    p, R, off, dirs, cs = sens._frame("depth", primals[0], primals[1])
    maxd = 10.0
    dj, gj, nj = jax_rays.cast(p, R, off, dirs, primals[0], primals[1], primals[2], maxd, cs, grad="first_order")
    dj0, gj0, _ = jax_rays.cast(p, R, off, dirs, primals[0], primals[1], primals[2], maxd, cs)
    u = rng.normal(0, 1, dj.shape).astype(np.float32)
    gjax = jax.grad(lambda *x: jnp.sum(jax_rays.cast(*x[:4], *x[4:], maxd, cs, grad="first_order")[0] * u),
                    argnums=(0, 1, 4, 5, 6))(p, R, off, dirs, *primals)
    rob = robot_mesh.load(a.g1_mjcf, m, cache=a.cache)
    dev = torch.device("cuda")
    scene = SceneRays(dev, terrain=ter, floor_z=0.0, robot=rob, nbody=m.nbody)
    T = lambda x: torch.as_tensor(np.asarray(x), device=dev).requires_grad_(True)
    tin = [T(p), T(R), torch.as_tensor(np.asarray(off), device=dev), torch.as_tensor(np.asarray(dirs), device=dev),
           T(primals[0]), T(primals[1]), T(primals[2])]
    dt, gt, nt = scene.cast(tin[0], tin[1], tin[2], tin[3], maxd, tin[4], tin[5], tin[6], normals=True,
                            grad="first_order")
    dt0, gt0 = scene.cast(*[x.detach() for x in tin[:4]], maxd, *[x.detach() for x in tin[4:]])
    (dt * torch.as_tensor(u, device=dev)).sum().backward()
    gtor = [tin[i].grad.cpu().numpy() for i in (0, 1, 4, 5, 6)]
    par = dict(dist_bitwise_jax_vs_torch=bool(np.array_equal(np.asarray(dj), dt.detach().cpu().numpy())),
               id_equal=bool(np.array_equal(np.asarray(gj), gt.cpu().numpy())),
               normal_bitwise=bool(np.array_equal(np.asarray(nj), nt.cpu().numpy())),
               jax_none_vs_first_order_bitwise=bool(np.array_equal(np.asarray(dj0), np.asarray(dj))),
               torch_none_vs_first_order_bitwise=bool(np.array_equal(dt0.cpu().numpy(), dt.detach().cpu().numpy())),
               grad_rel_maxdiff={})
    for nm_, x, y in zip(("pos", "rot", "xpos", "xquat", "objects"), gjax, gtor):
        x = np.asarray(x, np.float64)
        par["grad_rel_maxdiff"][nm_] = float(np.abs(x - y).max() / max(np.abs(x).max(), 1e-30))
    out["parity"] = par
    ok = ok and par["dist_bitwise_jax_vs_torch"] and par["jax_none_vs_first_order_bitwise"] and \
        par["torch_none_vs_first_order_bitwise"] and max(par["grad_rel_maxdiff"].values()) < 1e-5

    Bt = a.time_worlds
    idx = np.arange(Bt) % B
    P = tuple(x[idx] for x in primals)

    def bench(f, *args, n=20):
        jax.block_until_ready(f(*args))
        jax.block_until_ready(f(*args))
        t = time.perf_counter()
        for _ in range(n):
            r = f(*args)
        jax.block_until_ready(r)
        return round((time.perf_counter() - t) / n * 1e3, 3)

    tim = dict(nworld=Bt)
    for name in ("depth", "height_scan"):
        g = dict(sens.grad)
        sj = g1.G1Sensors.__new__(g1.G1Sensors)
        sj.__dict__.update(sens.__dict__)
        sj.depth, sj.hs, sj.lidar, sj.rgb = (sens.depth if name == "depth" else None,
                                             sens.hs if name == "height_scan" else None, None, None)
        fwd = jax.jit(lambda xp, xq, ob, s=sj: s(xp, xq, ob)[name])
        s0 = g1.G1Sensors.__new__(g1.G1Sensors)
        s0.__dict__.update(sj.__dict__)
        s0.grad = dict.fromkeys(g, "none")
        fwd0 = jax.jit(lambda xp, xq, ob, s=s0: s(xp, xq, ob)[name])
        U = jnp.asarray(rng.normal(0, 1, np.asarray(fwd(*P)).shape), jnp.float32)
        both = jax.jit(lambda xp, xq, ob, u, s=sj: jax.grad(lambda *x: jnp.sum(s(*x)[name] * u), argnums=(0, 1, 2))(
            xp, xq, ob))
        _, vjp = jax.vjp(lambda *x: fwd(*x), *P)
        bwd = jax.jit(lambda f, c: f(c))
        tim[name] = dict(forward_none_ms=bench(fwd0, *P), forward_first_order_ms=bench(fwd, *P),
                         forward_backward_ms=bench(both, *P, U), backward_only_ms=bench(bwd, vjp, U))
    out["timing_jax"] = tim

    tt = [torch.as_tensor(np.asarray(x)[idx], device=dev) for x in (p, R)] + tin[2:4] + \
        [torch.as_tensor(np.asarray(x)[idx], device=dev) for x in primals]
    Ut = torch.as_tensor(rng.normal(0, 1, (Bt, tt[2].shape[0])).astype(np.float32), device=dev)

    def tbench(fn, n=20):
        fn()
        fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return round((time.perf_counter() - t) / n * 1e3, 3)

    def tfb():
        x = [v.detach().requires_grad_(i not in (2, 3)) for i, v in enumerate(tt)]
        d_ = scene.cast(x[0], x[1], x[2], x[3], maxd, x[4], x[5], x[6], grad="first_order")[0]
        torch.autograd.grad((d_ * Ut).sum(), [x[i] for i in (0, 1, 4, 5, 6)])
    out["timing_torch_cast_depth"] = dict(
        nworld=Bt, forward_none_ms=tbench(lambda: scene.cast(tt[0], tt[1], tt[2], tt[3], maxd, *tt[4:])),
        forward_backward_ms=tbench(tfb))
    out["ok"] = bool(ok)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

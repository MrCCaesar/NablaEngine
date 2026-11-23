"""Smoothed sensor gradients (G1Sensors(grad=("smoothed", sigma)), sensors/raygrad) against central differences of the
smoothed images, the fused CUDA backward against the XLA path, PyTorch parity and timing."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sensor_grad_check as base

from nablaengine import fixtures
from nablaengine.sensors import cameras, raygrad


def worlds(m, rng, B, K):
    """Corpus poses at random positions and headings; K objects 0.7 - 2.0 m ahead (as sensor_grad_check)."""
    half = rng.uniform(0.08, 0.3, (K, 3))
    z = fixtures.load_corpus("mf_s0")
    d = mujoco.MjData(m)
    xpos, xquat, obj = np.zeros((B, m.nbody, 3)), np.zeros((B, m.nbody, 4)), np.zeros((B, K, 15))
    for w in range(B):
        d.qpos[:] = z["qpos"][rng.integers(len(z["qpos"]))]
        yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[3:7] = base.quat_mul(base.quat_exp(np.array([0, 0, yaw])), d.qpos[3:7])
        d.qpos[:2] = rng.uniform(-0.4, 0.4, 2)
        d.qpos[2] += 0.05
        mujoco.mj_kinematics(m, d)
        xpos[w], xquat[w] = d.xpos, d.xquat
        for k in range(K):
            ang = yaw + rng.normal(0, 0.6)
            r = rng.uniform(0.7, 2.0)
            R = base.rotz(ang + rng.uniform(-1, 1)) @ cameras.quat_matrix(base.quat_exp(rng.normal(0, 0.25, 3)))
            obj[w, k] = np.concatenate([[d.qpos[0] + r * np.cos(ang), d.qpos[1] + r * np.sin(ang),
                                         half[k, 2] + rng.uniform(0, 0.3)], R.ravel(), half[k]])
    return xpos, xquat, obj


def fine_scan(hs, k):
    """A HeightScan whose grid points are the k x k sub-cells of hs's cells."""
    from dataclasses import replace
    r = hs.resolution
    return replace(hs, size=(hs.size[0] + r - r / k, hs.size[1] + r - r / k), resolution=r / k)


def box(img, k):
    """(B, H k, W k[, C]) -> the k x k block means (B, H, W[, C])."""
    B, Hk, Wk = img.shape[:3]
    sh = (B, Hk // k, k, Wk // k, k) + img.shape[3:]
    return img.reshape(sh).mean((2, 4))


KINDS = ("validity", "occlusion", "id_change", "crease", "value_jump")


def classify(ids, valid, t, jump, r, val=None, vjump=None):
    """Pixel classes by the discontinuities within r (Chebyshev, no wrap) between 4-neighbours: "validity" (a valid
    value next to a miss / out-of-range pixel: silhouettes, range entry / exit), "occlusion" (another hit id with a
    ray-distance jump > jump), "id_change" (another hit id without a jump: contact lines, adjacent links), "crease"
    (same id, a second difference of the ray distance > jump: box edges, heightfield ridges), "value_jump" (same
    id, a value difference > vjump: the RGB's shading edges between faces); a pixel near one kind
    only is in that class, near several in "mixed", near none "smooth"."""
    B, H, W = ids.shape
    near = {}
    for kind in KINDS:
        ch = np.zeros(ids.shape, bool)
        for ax in (1, 2):
            sl = lambda a_, b_: tuple(slice(a_, b_) if i == ax else slice(None) for i in range(3))
            A, Bs = sl(None, -1), sl(1, None)
            vch = valid[A] != valid[Bs]
            both = valid[A] & valid[Bs]
            idc = (ids[A] != ids[Bs]) & both
            jmp = np.abs(t[A] - t[Bs]) > jump
            if kind == "crease":
                d2 = np.abs(t[sl(2, None)] - 2 * t[sl(1, -1)] + t[sl(None, -2)]) > jump
                d2 &= (ids[sl(2, None)] == ids[sl(1, -1)]) & (ids[sl(1, -1)] == ids[sl(None, -2)])
                d2 &= valid[sl(2, None)] & valid[sl(1, -1)] & valid[sl(None, -2)]
                ch[sl(1, -1)] |= d2
                continue
            if kind == "value_jump":
                if val is None:
                    continue
                dv = np.abs(val[A] - val[Bs])
                e = ((dv.max(-1) if dv.ndim == 4 else dv) > vjump) & both & (ids[A] == ids[Bs])
            else:
                e = {"validity": vch, "occlusion": idc & jmp, "id_change": idc & ~jmp}[kind]
            ch[A] |= e
            ch[Bs] |= e
        out = np.zeros_like(ch)
        pad = np.pad(ch, ((0, 0), (r, r), (r, r)))
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                out |= pad[:, dy:dy + H, dx:dx + W]
        near[kind] = out
    cnt = sum(near[k].astype(int) for k in KINDS)
    border = np.zeros(ids.shape, bool)
    border[:, :r], border[:, -r:], border[:, :, :r], border[:, :, -r:] = True, True, True, True
    cls = {k: near[k] & (cnt == 1) & ~border for k in KINDS}
    cls["mixed"], cls["smooth"] = (cnt > 1) & ~border, (cnt == 0) & ~border
    cls["border"] = border
    cls["interior"] = ~border
    return cls


def compare(a, f, mask):
    a, f = a[mask].astype(np.float64), f[mask].astype(np.float64)
    if a.size == 0:
        return dict(n=0)
    rms = np.sqrt(np.mean(f ** 2))
    if rms < 1e-6:
        return dict(n=int(a.size), ref_rms=float(rms), abs_rms=float(np.sqrt(np.mean((a - f) ** 2))))
    rel = np.abs(a - f) / (np.abs(f) + 0.01 * rms)
    return dict(n=int(a.size), rel_l2=float(np.linalg.norm(a - f) / np.linalg.norm(f)),
                cosine=float(np.sum(a * f) / max(np.linalg.norm(a) * np.linalg.norm(f), 1e-30)),
                rel_median=float(np.median(rel)), rel_p99=float(np.quantile(rel, 0.99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1-mjcf", required=True, help="mjlab's g1.xml (the robot's visual meshes)")
    ap.add_argument("--nworld", type=int, default=16)
    ap.add_argument("--objects", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--small-sigma", type=float, nargs="*", default=[0.0, 0.5])
    ap.add_argument("--k", type=int, default=8, help="supersampling of the reference (k x k per pixel)")
    ap.add_argument("--h", type=float, default=2e-3, help="central difference step")
    ap.add_argument("--time-worlds", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="runs/sensors_cache", help="robot_mesh cache directory (created)")
    ap.add_argument("--out", default="docs/results/sensor_grad_smooth_check.json")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    ap.add_argument("--dump", default=None, help="npz of the per-pixel tangents, references and classes")
    ap.add_argument("--accuracy-only", action="store_true", help="skip parity and timing")
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.force:
        sys.exit(f"{a.out} exists (--force to overwrite)")
    import jax
    import jax.numpy as jnp
    import torch
    from nablaengine.sensors import g1, jax_rays
    from nablaengine.sensors.torch_rays import SceneRays
    from nablaengine.sensors import robot_mesh

    Path(a.cache).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    m, _ = fixtures.load_model()
    ter = base.terrain_scene(rng)
    B, K, k = a.nworld, a.objects, a.k
    cam = cameras.D435I_DEPTH.with_resolution(64, 48)
    rgbc = cameras.D435I_COLOR.with_resolution(32, 24)
    hs = cameras.HeightScan()
    common = dict(terrain=ter, robot_mjcf=a.g1_mjcf, floor_z=0.0, mesh_cache=a.cache)
    mk = lambda **kw: g1.G1Sensors(m, depth=cam, height_scan=hs, rgb=rgbc, **common, **kw)
    sm_x = mk(grad=("smoothed", a.sigma), grad_impl="xla")
    fo_x = mk(grad="first_order", grad_impl="xla")
    small = {s: mk(grad=("smoothed", s), grad_impl="xla") for s in a.small_sigma}
    fine = g1.G1Sensors(m, depth=cam.with_resolution(64 * k, 48 * k), height_scan=fine_scan(hs, k),
                        rgb=rgbc.with_resolution(32 * k, 24 * k), **common)
    keys = ("depth", "height_scan", "rgb")
    xpos, xquat, obj = worlds(m, rng, B, K)
    tans = [(rng.normal(0, 1, (m.nbody, 3)), rng.normal(0, 1, (m.nbody, 3)), rng.normal(0, 1, (K, 3)),
             rng.normal(0, 1, (K, 3))) for _ in range(B)]
    dxq, dob = np.zeros_like(xquat), np.zeros_like(obj)
    for w in range(B):
        dx, dw, dop, dow = tans[w]
        for b in range(m.nbody):
            dxq[w, b] = 0.5 * base.quat_mul(np.concatenate([[0], dw[b]]), xquat[w, b])
        dob[w, :, :3] = dop
        for j in range(K):
            dob[w, j, 3:12] = (base.skew(dow[j]) @ obj[w, j, 3:12].reshape(3, 3)).ravel()
    f32 = lambda x: jnp.asarray(x, jnp.float32)
    P = (f32(xpos), f32(xquat), f32(obj))
    T = (f32(np.stack([t[0] for t in tans])), f32(dxq), f32(dob))
    jvp = lambda s: jax.jit(lambda p, t: jax.jvp(lambda *x: {q: s(*x)[q] for q in keys}, p, t))(P, T)
    out0, j_sm = jvp(sm_x)
    _, j_fo = jvp(fo_x)
    j_small = {s: jvp(v)[1] for s, v in small.items()}

    run_fine = jax.jit(lambda *x: {q: fine(*x)[q] for q in keys})

    def ref_smoothed(xp, xq, ob):
        o = run_fine(f32(xp), f32(xq), f32(ob))
        r = {}
        for q in keys:
            img = box(np.asarray(o[q], np.float64), k)
            sm = raygrad.Smoother(jnp, *img.shape[1:3], a.sigma)
            r[q] = np.asarray(sm.g(jnp.asarray(img, jnp.float32)), np.float64)
        return r
    pert = [base.perturb(xpos[w], xquat[w], obj[w], tans[w], a.h) for w in range(B)]
    perm = [base.perturb(xpos[w], xquat[w], obj[w], tans[w], -a.h) for w in range(B)]
    stack = lambda L: tuple(np.stack([x[i] for x in L]) for i in range(3))
    rp, rm = ref_smoothed(*stack(pert)), ref_smoothed(*stack(perm))
    fd = {q: (rp[q] - rm[q]) / (2 * a.h) for q in keys}

    rr = int(np.ceil(3 * a.sigma)) + 1
    ids, tt_ = {}, {}
    for q, name, robot in (("depth", "depth", True), ("rgb", "rgb", True), ("height_scan", "height_scan", False)):
        p, R, off, dirs, cs = sm_x._frame(name, P[0], P[1])
        maxd = (cam.max_depth / float(np.min(np.asarray(dirs)[:, 0])) if q == "depth" else
                rgbc.max_depth if q == "rgb" else hs.max_distance)
        dist, gid, _ = jax_rays.cast(p, R, off, dirs, P[0], P[1], P[2], maxd, cs, robot=robot)
        ids[q] = np.asarray(gid).reshape(np.asarray(out0[q]).shape[:3])
        tt_[q] = np.asarray(dist).reshape(ids[q].shape)
    res = dict(provenance=base.provenance(), nworld=B, objects=K, sigma=a.sigma, k=k, h=a.h, support_radius=rr,
               sensors={})
    ok = True
    dump = {}
    for q, jump in (("depth", 0.05), ("height_scan", 0.03), ("rgb", 0.05)):
        img = np.asarray(out0[q])
        valid = img > 0 if q == "depth" else (ids[q] >= 0)
        cls = classify(ids[q], valid, tt_[q], jump, rr, img, {"depth": 0.05, "height_scan": 0.05, "rgb": 0.02}[q])
        if q == "rgb":
            cls = {c: np.broadcast_to(v[..., None], img.shape) for c, v in cls.items()}
        sm = raygrad.Smoother(jnp, *img.shape[1:3], a.sigma)
        fo_sm = np.asarray(sm.g(j_fo[q]))
        A = np.asarray(j_sm[q])
        allm = np.ones(img.shape, bool)
        r = dict(class_share={c: round(float(v.mean()), 4) for c, v in cls.items()},
                 smoothed={c: compare(A, fd[q], v) for c, v in cls.items()} | dict(all=compare(A, fd[q], allm)),
                 first_order_smoothed_after={c: compare(fo_sm, fd[q], v) for c, v in cls.items()}
                 | dict(all=compare(fo_sm, fd[q], allm)),
                 small_sigma_vs_first_order_smooth_pixels={str(s): compare(np.asarray(j[q]), np.asarray(j_fo[q]),
                                                                           cls["smooth"]) for s, j in j_small.items()}
                 if q != "rgb" else "n/a (faceted shading: no smooth RGB field)")
        res["sensors"][q] = r
        dump[q] = dict(ours=A, fd=fd[q], fo_smoothed=fo_sm, fo=np.asarray(j_fo[q]), img=img, ids=ids[q], t=tt_[q],
                       **{"cls_" + c: v for c, v in cls.items()})
        if r["smoothed"]["smooth"].get("rel_l2", 0) > 0.1 or r["smoothed"]["interior"].get("cosine", 1) < 0.8:
            ok = False

    if a.dump:
        np.savez(a.dump, **{f"{q}/{n}": v for q, d_ in dump.items() for n, v in d_.items()})
    if a.accuracy_only:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
        sys.exit(0 if ok else 1)

    par = {}
    for mode in (("smoothed", a.sigma), "first_order"):
        sx, sf = mk(grad=mode, grad_impl="xla"), mk(grad=mode, grad_impl="fused")
        for q in keys:
            U = jnp.asarray(rng.normal(0, 1, np.asarray(out0[q]).shape), jnp.float32)
            gx, gf = (jax.jit(jax.grad(lambda *x, s=s: jnp.sum(s(*x)[q] * U), argnums=(0, 1, 2)))(*P) for s in (sx, sf))
            par[f"{mode if isinstance(mode, str) else 'smoothed'}/{q}"] = [
                float(np.abs(np.asarray(x) - np.asarray(y)).max() / max(np.abs(np.asarray(x)).max(), 1e-30))
                for x, y in zip(gx, gf)]
    res["fused_vs_xla_rel_maxdiff_xpos_xquat_objects"] = par
    ok = ok and max(max(v) for v in par.values()) < 1e-5

    rob = robot_mesh.load(a.g1_mjcf, m, cache=a.cache)
    dev = torch.device("cuda")
    scene = SceneRays(dev, terrain=ter, floor_z=0.0, robot=rob, nbody=m.nbody)
    sf = mk(grad=("smoothed", a.sigma), grad_impl="fused")
    tp = {}
    for q, name in (("depth", "depth"), ("height_scan", "height_scan")):
        p, R, off, dirs, cs = sf._frame(name, P[0], P[1])
        spec = raygrad.image_spec(cam if q == "depth" else hs, "depth" if q == "depth" else "height")
        U = rng.normal(0, 1, np.asarray(out0[q]).shape).astype(np.float32)
        Uj = jnp.asarray(U)
        gj = jax.grad(lambda p_, R_, *x: jnp.sum(jax_rays.cast_image(
            spec, (lambda d, gid, nrm, pos, rot: _jvalue(q, sf, d, gid, nrm, pos, rot, dirs)), a.sigma, p_, R_, off,
            dirs, *x, max_distance=_maxd(q, sf, dirs), cone_cs=cs, robot=q == "depth")[0] * Uj),
            argnums=(0, 1, 2, 3, 4))(p, R, *P)
        tt = lambda x: torch.as_tensor(np.asarray(x), device=dev)
        tin = [tt(p).requires_grad_(), tt(R).requires_grad_(), tt(off), tt(dirs), tt(P[0]).requires_grad_(),
               tt(P[1]).requires_grad_(), tt(P[2]).requires_grad_()]
        vfn = raygrad.depth_value(torch, cam, sf.housing, tin[3]) if q == "depth" else raygrad.height_value(torch, hs)
        if q == "depth":
            img, _ = scene.cast_image(spec, vfn, a.sigma, *tin[:4], _maxd(q, sf, dirs), *tin[4:])
        else:
            sc2 = SceneRays(dev, terrain=ter, floor_z=0.0, nbody=m.nbody)
            img, _ = sc2.cast_image(spec, vfn, a.sigma, *tin[:4], hs.max_distance, *tin[4:])
        (img * torch.as_tensor(U, device=dev)).sum().backward()
        gt = [tin[i].grad.cpu().numpy() for i in (0, 1, 4, 5, 6)]
        tp[q] = dict(image_bitwise_vs_jax=bool(np.array_equal(img.detach().cpu().numpy(), np.asarray(out0[q]))),
                     grad_rel_maxdiff={nm: float(np.abs(np.asarray(x) - y).max() / max(np.abs(np.asarray(x)).max(),
                                                                                       1e-30))
                                       for nm, x, y in zip(("pos", "rot", "xpos", "xquat", "objects"), gj, gt)})
    res["torch_vs_jax_smoothed"] = tp
    ok = ok and all(v["image_bitwise_vs_jax"] and max(v["grad_rel_maxdiff"].values()) < 1e-5 for v in tp.values())

    Path("runs").mkdir(exist_ok=True)
    Path("runs/sensor_grad_smooth_check.partial.json").write_text(json.dumps(res, indent=1) + "\n")

    Bt = a.time_worlds
    idx = np.arange(Bt) % B
    Pt = tuple(x[idx] for x in P)

    def bench(f, *args, n=20):
        jax.block_until_ready(f(*args))
        jax.block_until_ready(f(*args))
        t = time.perf_counter()
        for _ in range(n):
            r = f(*args)
        jax.block_until_ready(r)
        return round((time.perf_counter() - t) / n * 1e3, 3)
    tim = dict(nworld=Bt, gpu=res["provenance"]["gpu"])
    for q, kw in (("depth", dict(depth=cam, height_scan=None)), ("height_scan", dict(depth=None, height_scan=hs))):
        row = {}
        for label, mode, impl in (("none", "none", "fused"), ("first_order_xla", "first_order", "xla"),
                                  ("first_order_fused", "first_order", "fused"),
                                  ("smoothed_xla", ("smoothed", a.sigma), "xla"),
                                  ("smoothed_fused", ("smoothed", a.sigma), "fused")):
            s = g1.G1Sensors(m, rgb=None, **common, **kw, grad=mode, grad_impl=impl)
            fwd = jax.jit(lambda *x, s=s: s(*x)[q])
            if label == "none":
                row["forward_ms"] = bench(fwd, *Pt)
                continue
            U = jnp.asarray(rng.normal(0, 1, np.asarray(fwd(*Pt)).shape), jnp.float32)
            _, vjp = jax.vjp(lambda *x: fwd(*x), *Pt)
            row[label] = dict(backward_ms=bench(jax.jit(lambda f, c: f(c)), vjp, U),
                              forward_backward_ms=bench(jax.jit(lambda xp, xq, ob, u, s=s: jax.grad(
                                  lambda *y: jnp.sum(s(*y)[q] * u), argnums=(0, 1, 2))(xp, xq, ob)), *Pt, U))
        tim[q] = row
    res["timing_jax"] = tim

    p, R, off, dirs, cs = sf._frame("depth", Pt[0], Pt[1])
    tt = lambda x: torch.as_tensor(np.asarray(x), device=dev)
    base_in = [tt(p), tt(R), tt(off), tt(dirs), tt(Pt[0]), tt(Pt[1]), tt(Pt[2])]
    Ut = torch.as_tensor(rng.normal(0, 1, (Bt, off.shape[0])).astype(np.float32), device=dev)

    def tbench(fn, n=20):
        fn()
        fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return round((time.perf_counter() - t) / n * 1e3, 3)

    def tfb(impl):
        x = [v.detach().requires_grad_(i not in (2, 3)) for i, v in enumerate(base_in)]
        d_ = scene.cast(x[0], x[1], x[2], x[3], 20.0, x[4], x[5], x[6], grad="first_order", impl=impl)[0]
        torch.autograd.grad((d_ * Ut).sum(), [x[i] for i in (0, 1, 4, 5, 6)])
    res["timing_torch_cast_depth"] = dict(
        nworld=Bt, forward_ms=tbench(lambda: scene.cast(*base_in[:4], 20.0, *base_in[4:])),
        forward_backward_fused_ms=tbench(lambda: tfb("fused")), forward_backward_xla_ms=tbench(lambda: tfb("xla")))
    res["ok"] = bool(ok)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    sys.exit(0 if ok else 1)


def _maxd(q, sens, dirs):
    return sens.depth.max_depth / max(float(np.min(np.asarray(dirs)[:, 0])), 1e-3) if q == "depth" else \
        sens.hs.max_distance


def _jvalue(q, sens, d, gid, nrm, pos, rot, dirs):
    import jax.numpy as jnp
    if q == "depth":
        return raygrad.depth_value(jnp, sens.depth, sens.housing, dirs)(d, gid, nrm, pos, rot)
    return raygrad.height_value(jnp, sens.hs)(d, gid, nrm, pos, rot)


if __name__ == "__main__":
    main()

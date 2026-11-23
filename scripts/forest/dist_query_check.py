"""Checks of nablaengine.dist_query (the forest engine's distance queries) on public models.

Models (--model):
  scene    an inline MJCF (SCENE below): a floor plane, free bodies (box, sphere, capsule, cylinder, ellipsoid, a convex
           mesh) and a hinged two-link arm; geom pairs of every type and one body pair
  g1       the G1 fixture (fixtures/mjlab, forest engine): body pairs between the feet, the hands and the pelvis
  terrain  the scene on a forest terrain build (TerrainConfig(geoms="body")) with a terrain of one box and one
           heightfield (TERRAIN below): the scene's sphere, capsule and box against the terrain (dist_query.TERRAIN)

Checks:
  forward  each slot's distance against MuJoCo's mj_geomDistance (the smallest over the slot's geom pairs; the terrain:
           against the terrain box geom of a reference model holding the terrain geoms, slots whose element is the box)
           where both are below max_distance (separated geoms: the value; overlapping geoms: the sign, as the
           engine's EPA depth and MuJoCo's pair functions' depths may differ); p2 - p1 = dist * normal for separated
           geoms; far where the reference is beyond max_distance
  vjp      along a random tangent per world (free joints: the quaternion rotated about body-frame axes, the engine's
           qpos-form convention): dist's VJP against a central difference of dist; the points' VJP against a central
           difference of the closest points moved as material points of their bodies (MuJoCo's body poses), the
           derivative the points' VJP is defined as (the closest features held fixed)
  torch    (--torch) torch_query's outputs and gradient equal the JAX interface's bitwise (one process, both builds)
The gates are loose sanity gates (float32 GJK / EPA against MuJoCo's double-precision distance); exit status 1 if one
fails.

  python scripts/forest/dist_query_check.py --model scene [--torch] [--worlds 32] [--out result.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

SCENE = """
<mujoco model="dist_query_scene">
  <option timestep="0.002"/>
  <default><geom contype="0" conaffinity="0"/></default>
  <asset>
    <mesh name="oct" vertex="0.06 0 0  -0.06 0 0  0 0.05 0  0 -0.05 0  0 0 0.08  0 0 -0.08"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 1"/>
    <body name="box" pos="0 0 0.1"><freejoint/><geom name="box_g" type="box" size="0.1 0.08 0.06" mass="1"/></body>
    <body name="ball" pos="0.3 0 0.1"><freejoint/><geom name="ball_g" type="sphere" size="0.05" mass="0.3"/></body>
    <body name="rod" pos="-0.3 0 0.1"><freejoint/><geom name="rod_g" type="capsule" size="0.03 0.1" mass="0.3"/></body>
    <body name="can" pos="0 0.3 0.1"><freejoint/><geom name="can_g" type="cylinder" size="0.04 0.06" mass="0.3"/></body>
    <body name="egg" pos="0 -0.3 0.1"><freejoint/><geom name="egg_g" type="ellipsoid" size="0.06 0.04 0.03" mass="0.2"/>
    </body>
    <body name="gem" pos="0.3 0.3 0.1"><freejoint/><geom name="gem_g" type="mesh" mesh="oct" mass="0.2"/></body>
    <body name="arm" pos="-0.4 0 0.4">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom name="arm_g" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.3"/>
      <body name="finger" pos="0.2 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <geom name="finger_g" type="capsule" fromto="0 0 0 0.15 0 -0.12" size="0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <contact>
    <pair geom1="floor" geom2="box_g"/>
    <pair geom1="floor" geom2="ball_g"/>
    <pair geom1="floor" geom2="rod_g"/>
    <pair geom1="ball_g" geom2="box_g"/>
  </contact>
  <actuator>
    <motor name="shoulder" joint="shoulder" gear="1" ctrlrange="-5 5" ctrllimited="true"/>
    <motor name="elbow" joint="elbow" gear="1" ctrlrange="-5 5" ctrllimited="true"/>
  </actuator>
</mujoco>
"""
SCENE_PAIRS = [("geom", "floor", "box_g"), ("geom", "ball_g", "box_g"), ("geom", "rod_g", "can_g"),
               ("geom", "egg_g", "box_g"), ("geom", "gem_g", "box_g"), ("geom", "gem_g", "ball_g"),
               ("geom", "can_g", "egg_g"), ("geom", "floor", "egg_g"), ("body", "finger", "box"),
               ("geom", "arm_g", "rod_g")]
TERRAIN_GEOMS = """
    <geom name="ter_box" type="box" pos="0.6 0 0.05" size="0.2 0.2 0.05" euler="0 0 20"/>
    <geom name="ter_hf" type="hfield" hfield="hf" pos="-0.6 0 0"/>
"""
TERRAIN_ASSET = """
    <hfield name="hf" nrow="6" ncol="6" size="0.3 0.3 0.1 0.05"
            elevation="0 0.1 0.2 0.3 0.2 0.1  0.1 0.3 0.5 0.6 0.4 0.2  0.2 0.5 0.8 1 0.6 0.3
                       0.2 0.4 0.7 0.9 0.5 0.2  0.1 0.2 0.4 0.5 0.3 0.1  0 0.1 0.1 0.2 0.1 0"/>
"""
G1_PAIRS = [("body", "left_ankle_roll_link", "right_ankle_roll_link"), ("body", "pelvis", "left_wrist_yaw_link"),
            ("body", "pelvis", "right_wrist_yaw_link"), ("body", "left_knee_link", "right_knee_link"),
            ("body", "left_wrist_yaw_link", "right_wrist_yaw_link")]


def scene_xml(terrain=False):
    x = SCENE
    if terrain:
        x = x.replace("  </asset>", TERRAIN_ASSET + "  </asset>").replace(
            '    <geom name="floor" type="plane" size="0 0 1"/>\n',
            '    <geom name="floor" type="plane" size="0 0 1"/>\n' + TERRAIN_GEOMS)
    return x


def terrain_only_xml():
    return ("<mujoco><asset>" + TERRAIN_ASSET + "</asset><worldbody>" + TERRAIN_GEOMS + "</worldbody></mujoco>")


def free_joints(m):
    import mujoco
    return [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def random_quat(rng):
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def scene_states(m, B, rng, terrain=False):
    """B states: the free bodies scattered around the box (some apart, some touching, some overlapping, some beyond
    max_distance), random orientations, the arm's joints random. terrain: the bodies over the terrain box and
    heightfield."""
    import mujoco
    out = np.tile(m.qpos0, (B, 1))
    fj = free_joints(m)
    for i in range(B):
        for k, j in enumerate(fj):
            a = int(m.jnt_qposadr[j])
            if terrain:
                cx = 0.6 if (i + k) % 2 == 0 else -0.6
                out[i, a:a + 3] = [cx + rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), rng.uniform(0.0, 0.35)]
            elif m.body(int(m.jnt_bodyid[j])).name == "box":
                out[i, a:a + 3] = [rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), rng.uniform(0.0, 0.12)]
            else:
                r = rng.uniform(0.05, 0.45)
                t = rng.uniform(0, 2 * np.pi)
                out[i, a:a + 3] = [r * np.cos(t), r * np.sin(t), rng.uniform(-0.02, 0.25)]
            out[i, a + 3:a + 7] = random_quat(rng)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                out[i, m.jnt_qposadr[j]] = rng.uniform(-1.2, 1.2)
    return out


def g1_states(m, B, rng):
    import mujoco
    out = np.tile(m.qpos0, (B, 1))
    for i in range(B):
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                a = int(m.jnt_qposadr[j])
                lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-1.0, 1.0)
                out[i, a] = np.clip(m.qpos0[a] + rng.normal(0, 0.5), lo, hi)
        out[i, 3:7] = quat_mul(out[i, 3:7], np.r_[np.cos(0.1), np.sin(0.1) * random_quat(rng)[1:]])
        out[i, 3:7] /= np.linalg.norm(out[i, 3:7])
    return out


def reference(m_ref, qpos, slots, dmax, geom_map=None, slot_ok=None):
    """((B, S) MuJoCo distances (mj_geomDistance, the smallest over each slot's pairs; nan where not compared), (B, S, 2)
    the model geom ids of each slot's nearest pair). geom_map: model geom id -> reference model geom id (the terrain
    geom id -1 -> the terrain box's)."""
    import mujoco
    d = mujoco.MjData(m_ref)
    B, S = qpos.shape[0], len(slots)
    ref = np.full((B, S), np.nan)
    arg = np.array([[sl[0] for sl in slots]] * B, np.int64)
    for i in range(B):
        d.qpos[:qpos.shape[1]] = qpos[i]
        mujoco.mj_kinematics(m_ref, d)
        for k, sl in enumerate(slots):
            if slot_ok is not None and not slot_ok[i, k]:
                continue
            best = np.inf
            for g1, g2 in sl:
                a, b = (geom_map or {}).get(g1, g1), (geom_map or {}).get(g2, g2)
                ft = np.zeros(6)
                v = mujoco.mj_geomDistance(m_ref, d, a, b, 2.0 * dmax, ft)
                if v < best:
                    best, arg[i, k] = v, (g1, g2)
            ref[i, k] = best
    return ref, arg


def material_fd(m, m_ref, q0, qp, qm, pts, arg, eps):
    """(B, S, 2, 3) central difference of the closest points moved as material points of their bodies (the
    derivative the points' VJP takes: the closest features held fixed), body poses from MuJoCo's mj_kinematics."""
    import mujoco
    d = mujoco.MjData(m_ref)
    out = np.zeros(pts.shape)
    for i in range(q0.shape[0]):
        poses = []
        for q in (q0[i], qp[i], qm[i]):
            d.qpos[:q.shape[0]] = q
            mujoco.mj_kinematics(m_ref, d)
            poses.append((d.xpos.copy(), d.xmat.reshape(-1, 3, 3).copy()))
        (X0, R0), (Xp, Rp), (Xm, Rm) = poses
        for k in range(pts.shape[1]):
            for e in range(2):
                g = int(arg[i, k, e])
                b = 0 if g < 0 else int(m.geom_bodyid[g])
                pb = R0[b].T @ (pts[i, k, e] - X0[b])
                out[i, k, e] = ((Xp[b] + Rp[b] @ pb) - (Xm[b] + Rm[b] @ pb)) / (2 * eps)
    return out


def perturb(m, qpos, u, eps):
    """qpos moved by eps along the dof tangent u (free joints: q (x) exp(eps w / 2), body-frame w) and the qpos-form
    direction dq of that motion."""
    import mujoco
    q = qpos.copy()
    dq = np.zeros_like(qpos)
    for j in range(m.njnt):
        a, v = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            q[:, a:a + 3] += eps * u[:, v:v + 3]
            dq[:, a:a + 3] = u[:, v:v + 3]
            for i in range(q.shape[0]):
                w = u[i, v + 3:v + 6]
                th = eps * np.linalg.norm(w)
                e = np.r_[np.cos(th / 2), np.sin(th / 2) * w / max(np.linalg.norm(w), 1e-30)]
                q0 = qpos[i, a + 3:a + 7]
                q[i, a + 3:a + 7] = quat_mul(q0, e)
                dq[i, a + 3:a + 7] = quat_mul(q0, np.r_[0.0, 0.5 * w])
        elif int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            q[:, a] += eps * u[:, v]
            dq[:, a] = u[:, v]
        else:
            raise ValueError("ball joints are not handled by this check")
    return q, dq


def build(a):
    """(engine model, reference model, pairs, init kwargs, terrain or None, geom_map)."""
    import mujoco
    from nablaengine import dist_query, engines
    if a.model == "g1":
        from nablaengine import fixtures
        m, meta = fixtures.load_model(check=None)
        engines.prefer_forest(m)
        return m, m, G1_PAIRS, dict(foot_pairs=meta["foot_pairs"], com_body=meta.get("torso_body")), None, None
    m = mujoco.MjModel.from_xml_string(scene_xml())
    if a.model == "scene":
        return m, m, SCENE_PAIRS, {}, None, None
    from nablaengine import terrain as T
    m_ref = mujoco.MjModel.from_xml_string(scene_xml(terrain=True))
    ter = T.Terrain.from_model(mujoco.MjModel.from_xml_string(terrain_only_xml()))
    pairs = [("geom", "ball_g", dist_query.TERRAIN), ("geom", "rod_g", dist_query.TERRAIN),
             ("body", "box", dist_query.TERRAIN), ("geom", "ball_g", "box_g")]
    gmap = {g: mujoco.mj_name2id(m_ref, mujoco.mjtObj.mjOBJ_GEOM, m.geom(g).name) for g in range(m.ngeom)}
    gmap[dist_query.TERRAIN_GEOM] = mujoco.mj_name2id(m_ref, mujoco.mjtObj.mjOBJ_GEOM, "ter_box")
    return m, m_ref, pairs, dict(terrain=T.TerrainConfig(geoms="body")), ter, gmap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="scene", help="scene, g1 or terrain")
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--dmax", type=float, default=0.3)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.model not in ("scene", "g1", "terrain"):
        ap.error(f"--model {a.model!r}: scene, g1 or terrain")
    import jax
    import jax.numpy as jnp
    from nablaengine import dist_query, jax_ffi
    rng = np.random.default_rng(a.seed)
    m, m_ref, pairs, kw, ter, gmap = build(a)
    info = dist_query.init(m, pairs, max_distance=a.dmax, **kw)
    if ter is not None:
        jax_ffi.set_terrain(ter)
    B, S = a.worlds, info["nslot"]
    qpos = g1_states(m, B, rng) if a.model == "g1" else scene_states(m, B, rng, terrain=ter is not None)
    com = np.zeros((B, int(info["engine"]["com_words"])), np.float32)
    j = lambda x: jnp.asarray(np.ascontiguousarray(x, np.float32))
    q32 = qpos.astype(np.float32).astype(np.float64)
    out = dist_query.query(j(qpos), j(com), faces=True, **({"elements": True} if ter is not None else {}))
    dist, pts, nrm, far, face = (np.asarray(x) for x in out[:5])
    res = dict(model=a.model, worlds=B, slots=S, labels=info["labels"], built=info["built"])
    ok = True
    slot_ok = None
    if ter is not None:
        el = np.asarray(out[5])
        is_ter = np.array([any(g2 == dist_query.TERRAIN_GEOM for _, g2 in sl) for sl in info["slots"]])
        slot_ok = ~is_ter[None, :] | (el[:, :, 0] == 0)
        res["terrain_elements_seen"] = sorted({int(e) for e in el[:, :, 0].ravel()})
    ref, arg = reference(m_ref, q32, info["slots"], a.dmax, gmap, slot_ok)
    cmp = ~np.isnan(ref) & (ref < a.dmax - 1e-3) & (far == 0)
    both = cmp & (ref >= 0)
    over = cmp & (ref < 0)
    err = np.abs(dist - ref)[both]
    tol = 1e-3 + 1e-3 * np.abs(ref[both])
    oerr = np.abs(dist - ref)[over]
    sign_ok = bool(np.all(dist[over] < 1e-4))
    far_ok = bool(np.all(far[~np.isnan(ref) & (ref > a.dmax + 1e-3)] == 1))
    gap = pts[..., 1, :] - pts[..., 0, :]
    dev = np.abs(gap - dist[..., None] * nrm).max(-1)
    cons = dev[(far == 0) & (dist >= 0)]
    pen = dev[(far == 0) & (dist < 0)]
    res["forward"] = dict(compared=int(both.sum()), max_abs_err=float(err.max()) if err.size else None,
                          median_abs_err=float(np.median(err)) if err.size else None,
                          within_tol_share=float(np.mean(err <= tol)) if err.size else None,
                          overlapping=int(over.sum()), overlapping_sign_ok=sign_ok,
                          overlapping_median_abs_err=float(np.median(oerr)) if oerr.size else None,
                          overlapping_max_abs_err=float(oerr.max()) if oerr.size else None,
                          far_flags_ok=far_ok, points_normal_max_dev=float(cons.max()) if cons.size else 0.0,
                          points_normal_max_dev_overlapping=float(pen.max()) if pen.size else 0.0,
                          finite=bool(np.isfinite(dist).all() and np.isfinite(pts).all()))
    f_ok = (err.size > 0 and float(np.mean(err <= tol)) >= 0.95 and float(err.max()) < 1e-2 and far_ok and sign_ok
            and res["forward"]["finite"] and (cons.size == 0 or float(cons.max()) < 1e-4))
    res["forward"]["pass"] = bool(f_ok)
    ok &= f_ok
    gd = rng.standard_normal((B, S)).astype(np.float32)
    gp = rng.standard_normal((B, S, 2, 3)).astype(np.float32)
    zd, zp = np.zeros_like(gd), np.zeros_like(gp)
    u = rng.standard_normal((B, m.nv))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    qp, dq = perturb(m, q32, u, a.eps)
    qm, _ = perturb(m, q32, u, -a.eps)
    dist_p = np.asarray(dist_query.query_forward(j(qp), j(com))[0], np.float64)
    dist_m = np.asarray(dist_query.query_forward(j(qm), j(com))[0], np.float64)
    fd_d = (((dist_p - dist_m) / (2 * a.eps)) * gd).sum(1)
    g_d = np.asarray(dist_query.query_backward(j(qpos), j(com), j(gd), j(zp)), np.float64)
    an_d = (g_d * dq).sum(1)
    fd_p = (material_fd(m, m_ref, q32, qp, qm, pts.astype(np.float64), arg, a.eps) * gp).reshape(B, -1).sum(1)
    g_p = np.asarray(dist_query.query_backward(j(qpos), j(com), j(zd), j(gp)), np.float64)
    an_p = (g_p * dq).sum(1)
    res["vjp"] = dict(eps=a.eps)
    v_ok = True
    for name, fd, an in (("dist", fd_d, an_d), ("points", fd_p, an_p)):
        rel = np.abs(fd - an) / np.maximum(np.maximum(np.abs(fd), np.abs(an)), 1e-6)
        res["vjp"][name] = dict(rel_median=float(np.median(rel)), rel_p90=float(np.quantile(rel, 0.9)),
                                rel_max=float(rel.max()))
        v_ok &= float(np.median(rel)) < 1e-2 and float(np.quantile(rel, 0.9)) < 5e-2
    res["vjp"]["pass"] = bool(v_ok)
    ok &= v_ok
    if a.torch:
        import torch
        dist_query.torch_init(m, pairs, max_distance=a.dmax, device=0, **kw)
        if ter is not None:
            from nablaengine import torch_ffi
            torch_ffi.set_terrain(ter, 0)
        tq = torch.tensor(qpos.astype(np.float32), device="cuda:0", requires_grad=True)
        tc = torch.tensor(com, device="cuda:0")
        to = dist_query.torch_query(tq, tc, faces=True, **({"elements": True} if ter is not None else {}))
        ((to[0] * torch.tensor(gd, device="cuda:0")).sum() + (to[1] * torch.tensor(gp, device="cuda:0")).sum()).backward()
        same = all(np.array_equal(np.asarray(x), y.detach().cpu().numpy()) for x, y in zip(out, to))
        _, vjp = jax.vjp(lambda q: dist_query.query(q, j(com))[:2], j(qpos))
        gj = np.asarray(vjp((j(gd), j(gp)))[0])
        same_g = bool(np.array_equal(gj, tq.grad.cpu().numpy()))
        res["torch"] = dict(outputs_bitwise=bool(same), grad_bitwise=same_g)
        ok &= same and same_g
    res["pass"] = bool(ok)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

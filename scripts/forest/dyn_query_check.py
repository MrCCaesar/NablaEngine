"""Check of the forest engine's dynamics queries (nablaengine.dyn_query) against MuJoCo C in float64."""
import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TOL, TOL_VJP = 1e-6, 1e-3
TOL_KIN_PARAMS = 2e-6
CASES = ("g1", "g1_params", "g1_hands", "multitree", "slide_tendon", "g1_gravity", "multitree_gravity",
         "slide_tendon_gravity", "torch", "kin_tree", "kin_g1_forest", "kin_g1_box", "kin_multitree", "kin_slide_tendon",
         "terrain_g1", "terrain_g1_box")
SEEDS = {"g1": 0, "g1_params": 1, "g1_hands": 2, "multitree": 3, "slide_tendon": 4, "g1_gravity": 5,
         "multitree_gravity": 6, "slide_tendon_gravity": 7}
XFRC_BODIES = ("torso_link", "left_ankle_roll_link")
G1BOX_XML = REPO / "scripts" / "forest" / "models" / "g1box.xml"
try:
    sys.path.insert(0, str(REPO / "scripts" / "internal"))
    import dyn_query_scenes as INTERNAL
    INTERNAL_CASES = tuple(INTERNAL.CASES)
except ImportError:
    INTERNAL, INTERNAL_CASES = None, ()

SLIDE_TENDON_XML = """
<mujoco model="dq_slide_tendon">
  <option timestep="0.002" integrator="implicitfast"/>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 1"/>
    <body name="cart" pos="0 0 0.5">
      <joint name="slider" type="slide" axis="1 0 0" range="-1 1" armature="0.1"/>
      <geom name="cart_g" type="box" size="0.2 0.1 0.05" mass="1"/>
      <body name="pole" pos="0.05 0 0.02" euler="0 10 0" gravcomp="0.5">
        <joint name="hinge" type="hinge" axis="0 1 0" armature="0.02"/>
        <geom name="pole_g" type="capsule" fromto="0 0 0 0.1 0 0.6" size="0.03" mass="0.3"/>
        <body name="tip" pos="0.1 0 0.6" euler="20 0 30">
          <joint name="slide2" type="slide" axis="0 0.6 0.8" range="-0.2 0.2"/>
          <geom name="tip_g" type="sphere" size="0.05" mass="0.2" pos="0.05 0.02 0"/>
          <body name="tip2" pos="0 0 0.1">
            <joint name="hinge2" type="hinge" axis="1 0 0"/>
            <geom name="tip2_g" type="box" size="0.03 0.04 0.05" mass="0.15" pos="0 0.03 0.02" euler="10 20 30"/>
          </body>
        </body>
      </body>
    </body>
    <body name="obj" pos="1 0 0.3">
      <freejoint/>
      <geom name="obj_g" type="box" size="0.05 0.06 0.07" mass="0.5" pos="0.01 0.02 0.03"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="floor" geom2="obj_g"/>
  </contact>
  <tendon>
    <fixed name="t" armature="0.05">
      <joint joint="slider" coef="1"/>
      <joint joint="hinge" coef="0.5"/>
    </fixed>
  </tendon>
</mujoco>
"""


def load_case(name):
    """(MjModel, foot_pairs, com_body, PhysicsParams or None)."""
    import mujoco
    from nablaengine.params import PhysicsParams
    if name in ("g1", "g1_params", "torch", "g1_gravity"):
        m = mujoco.MjModel.from_binary_path(str(REPO / "fixtures" / "mjlab" / "g1_train.mjb"))
        meta = json.loads((REPO / "fixtures" / "mjlab" / "g1_meta.json").read_text())
        params = None
        if name == "g1_params":
            params = PhysicsParams(("dof_armature", "torso_mass", "torso_inertia"))
        if name == "g1_gravity":
            params = PhysicsParams(("gravity", "xfrc", "torso_mass"), xfrc_bodies=XFRC_BODIES)
        return m, meta["foot_pairs"], "torso_link", params
    if name == "multitree_gravity":
        m = mujoco.MjModel.from_xml_path(str(REPO / "scripts" / "contact_forces" / "multitree.xml"))
        return m, [], None, PhysicsParams(("gravity",))
    if name == "slide_tendon_gravity":
        return mujoco.MjModel.from_xml_string(SLIDE_TENDON_XML), [], None, PhysicsParams(("gravity", "dof_armature"))
    if name == "g1_hands":
        m = mujoco.MjModel.from_xml_path(str(REPO / "fixtures" / "robots" / "g1_with_hands" / "g1_with_hands_flat.xml"))
        return m, [], None, None
    if name == "multitree":
        m = mujoco.MjModel.from_xml_path(str(REPO / "scripts" / "contact_forces" / "multitree.xml"))
        return m, [], None, None
    if name == "slide_tendon":
        return mujoco.MjModel.from_xml_string(SLIDE_TENDON_XML), [], None, None
    raise ValueError(name)


def random_states(m, B, rng):
    import mujoco
    q = np.tile(m.qpos0, (B, 1)).astype(np.float64)
    for j in range(m.njnt):
        a, t = m.jnt_qposadr[j], int(m.jnt_type[j])
        if t == int(mujoco.mjtJoint.mjJNT_FREE):
            q[:, a:a + 3] = m.qpos0[a:a + 3] + rng.normal(0, 0.3, (B, 3))
            u = rng.normal(size=(B, 4))
            q[:, a + 3:a + 7] = u / np.linalg.norm(u, axis=1, keepdims=True) * rng.uniform(0.8, 1.2, (B, 1))
        else:
            lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi))
            q[:, a] = rng.uniform(lo, hi, B)
    v = rng.normal(0, 1.5, (B, m.nv))
    return q.astype(np.float32), v.astype(np.float32)


def mj_reference(m, q, v, com=None, com_body=None, params=None, pw=None):
    """(M (B, nv, nv), bias (B, nv)) of MuJoCo float64 at the float32 inputs, the per-world values applied to a copy."""
    import mujoco
    B = q.shape[0]
    mm = copy.copy(m)
    d = mujoco.MjData(mm)
    tb = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, com_body) if com_body else -1
    ipos0 = mm.body_ipos[tb].copy() if tb >= 0 else None
    lay = params.layout(m.nv) if params else {}
    M, bias = np.zeros((B, m.nv, m.nv)), np.zeros((B, m.nv))
    for w in range(B):
        if tb >= 0 and com is not None:
            mm.body_ipos[tb] = ipos0 + com[w, :3].astype(np.float64)
        if "dof_armature" in lay:
            o, n = lay["dof_armature"]
            mm.dof_armature[:] = pw[w, o:o + n]
        if "torso_mass" in lay:
            mm.body_mass[tb] = pw[w, lay["torso_mass"][0]]
        if "torso_inertia" in lay:
            o, n = lay["torso_inertia"]
            mm.body_inertia[tb] = pw[w, o:o + n]
        if "gravity" in lay:
            o, n = lay["gravity"]
            mm.opt.gravity[:] = pw[w, o:o + n]
        d.qpos[:] = q[w]
        d.qvel[:] = v[w]
        mujoco.mj_forward(mm, d)
        mujoco.mj_fullM(mm, d, M[w])
        bias[w] = d.qfrc_bias
    return M, bias


def mj_jac_ref(m, q, bodies, pts):
    import mujoco
    d = mujoco.MjData(m)
    B, P = bodies.shape
    jp, jr = np.zeros((B, P, 3, m.nv)), np.zeros((B, P, 3, m.nv))
    for w in range(B):
        d.qpos[:] = q[w]
        mujoco.mj_fwdPosition(m, d)
        for j in range(P):
            mujoco.mj_jac(m, d, jp[w, j], jr[w, j], pts[w, j].astype(np.float64), int(bodies[w, j]))
    return jp, jr


def err(x, ref):
    """(abs, rel) of the worst world: max |x - ref| per world, and that over max |ref| per world."""
    B = x.shape[0]
    dx = np.abs(np.asarray(x, np.float64) - ref).reshape(B, -1).max(1)
    sc = np.abs(ref).reshape(B, -1).max(1)
    rel = dx / np.maximum(sc, 1e-30)
    return dict(abs=float(dx.max()), rel=float(rel.max()), ref_max=float(sc.max()))


def query_points(m, q, P, rng):
    """P random bodies per world (body 0 included in column 0) and world points within 0.15 m of their origins."""
    import mujoco
    B = q.shape[0]
    bodies = rng.integers(1, m.nbody, (B, P)).astype(np.int32)
    bodies[:, 0] = 0
    d = mujoco.MjData(m)
    pts = np.zeros((B, P, 3), np.float32)
    for w in range(B):
        d.qpos[:] = q[w]
        mujoco.mj_kinematics(m, d)
        pts[w] = d.xpos[bodies[w]] + rng.normal(0, 0.15, (P, 3))
    return bodies, pts


def fd_vjp(m, q, bodies, pts, Gp, Gr, h=1e-6):
    """Central differences of L = <Gp, jacp> + <Gr, jacr> of MuJoCo (float64) w.r.t. qpos (raw coordinates) and the
    points, one world."""
    import mujoco
    d = mujoco.MjData(m)
    P = bodies.shape[0]
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def L(qq, pp):
        d.qpos[:] = qq
        mujoco.mj_fwdPosition(m, d)
        s = 0.0
        for j in range(P):
            mujoco.mj_jac(m, d, jp, jr, pp[j], int(bodies[j]))
            s += float((Gp[j] * jp).sum() + (Gr[j] * jr).sum())
        return s
    q0, p0 = q.astype(np.float64), pts.astype(np.float64)
    gq = np.zeros(m.nq)
    for k in range(m.nq):
        e = np.zeros(m.nq)
        e[k] = h
        gq[k] = (L(q0 + e, p0) - L(q0 - e, p0)) / (2 * h)
    gp = np.zeros((P, 3))
    for j in range(P):
        for k in range(3):
            e = np.zeros((P, 3))
            e[j, k] = h
            gp[j, k] = (L(q0, p0 + e) - L(q0, p0 - e)) / (2 * h)
    return gq, gp


def timeit(fn, *args, n=50):
    import jax
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def run_case(name, B, P, nvjp, timing_B, save):
    import jax
    import jax.numpy as jnp
    import mujoco
    from nablaengine import dyn_query as dq
    m, fp, com_body, params = load_case(name)
    rng = np.random.default_rng(SEEDS[name])
    t0 = time.time()
    info = dq.init(model=m, foot_pairs=fp, com_body=com_body, params=params)
    out = dict(case=name, nbody=int(m.nbody), nv=int(m.nv), nq=int(m.nq), ntree=int(m.ntree), B=B, P=P,
               com_body=com_body, params=params.key() if params else None, init_seconds=round(time.time() - t0, 1),
               library_built=info["built"])
    q, v = random_states(m, B, rng)
    com = np.zeros((B, info["com_words"]), np.float32)
    pw = None
    if com_body:
        com[:, :3] = rng.normal(0, 0.02, (B, 3))
    if params:
        pw = param_rows(m, com_body, params, B, rng)
        com[:, 3:] = pw
    M, bias = dq.dynamics(q, v, com)
    M, bias = np.asarray(M), np.asarray(bias)
    Mr, br = mj_reference(m, q.astype(np.float64), v.astype(np.float64), com, com_body, params, pw)
    out["M"], out["bias"] = err(M, Mr), err(bias, br)
    out["M_symmetric"] = bool(np.array_equal(M, np.swapaxes(M, 1, 2)))
    xfrc_ok = True
    if params and "xfrc" in params.fields:
        o, n = params.layout(m.nv)["xfrc"]
        c0 = com.copy()
        c0[:, 3 + o:3 + o + n] = 0.0
        Mx, bx = (np.asarray(x) for x in dq.dynamics(q, v, c0))
        xfrc_ok = bool(np.array_equal(Mx, M) and np.array_equal(bx, bias))
        out["xfrc_words_change_nothing"] = xfrc_ok
    if params and "gravity" in params.fields:
        o, n = params.layout(m.nv)["gravity"]
        out["gravity_range"] = [float(np.abs(pw[:, o:o + n] - m.opt.gravity).max())]
    if params:
        Mm, bm = dq.dynamics(q, v, None)
        Mr0, br0 = mj_reference(m, q.astype(np.float64), v.astype(np.float64))
        out["M_model_values"], out["bias_model_values"] = err(np.asarray(Mm), Mr0), err(np.asarray(bm), br0)
    bodies, pts = query_points(m, q, P, rng)
    jp, jr = dq.jacobian(q, bodies, pts)
    jpr, jrr = mj_jac_ref(m, q.astype(np.float64), bodies, pts)
    out["jacp"], out["jacr"] = err(np.asarray(jp), jpr), err(np.asarray(jr), jrr)
    nw = min(nvjp, B)
    Gp = rng.normal(size=(nw, P, 3, m.nv)).astype(np.float32)
    Gr = rng.normal(size=(nw, P, 3, m.nv)).astype(np.float32)
    f = lambda qq, pp: jnp.sum(jnp.stack([jnp.vdot(a, b) for a, b in zip(dq.jacobian(qq, bodies[:nw], pp), (Gp, Gr))]))
    gq, gp = jax.grad(f, argnums=(0, 1))(jnp.asarray(q[:nw]), jnp.asarray(pts[:nw]))
    gq, gp = np.asarray(gq, np.float64), np.asarray(gp, np.float64)
    rq, rp, aq = [], [], []
    for w in range(nw):
        fq, fpt = fd_vjp(m, q[w], bodies[w], pts[w], Gp[w].astype(np.float64), Gr[w].astype(np.float64))
        rq.append(np.linalg.norm(gq[w] - fq) / max(np.linalg.norm(fq), 1e-30))
        rp.append(np.linalg.norm(gp[w] - fpt) / max(np.linalg.norm(fpt), 1e-30))
        aq.append(np.abs(gq[w] - fq).max())
    out["vjp"] = dict(worlds=nw, qpos_rel=float(max(rq)), qpos_abs=float(max(aq)), points_rel=float(max(rp)),
                      fd_h=1e-6)
    out["mj_dyn"] = "M, bias: no VJP (forward-only)"
    if timing_B:
        qt, vt = random_states(m, timing_B, rng)
        ct = np.tile(com[:1], (timing_B, 1))
        bt, ptt = query_points(m, qt[:1], P, rng)
        bt, ptt = np.tile(bt, (timing_B, 1)), np.tile(ptt, (timing_B, 1, 1))
        qd, vd, cd, bd, pd = map(jnp.asarray, (qt, vt, ct, bt, ptt))
        fdyn = jax.jit(dq.dynamics)
        fjac = jax.jit(dq.jacobian)
        gt = jnp.ones((timing_B, P, 3, m.nv), jnp.float32)
        fvjp = jax.jit(lambda a, b, c: dq.jacobian_backward(a, b, c, gt, gt))
        fkin = jax.jit(dq.kinematics_forward)
        gk = [jnp.ones(x.shape, jnp.float32) for x in fkin(qd, vd, cd)]
        fkv = jax.jit(lambda a, b, c: dq.kinematics_backward(a, b, c, *gk))
        out["timing_ms"] = dict(B=timing_B, P=P, device=jax.devices()[0].device_kind, dynamics=timeit(fdyn, qd, vd, cd),
                                jacobian=timeit(fjac, qd, bd, pd), jacobian_vjp=timeit(fvjp, qd, bd, pd),
                                kinematics=timeit(fkin, qd, vd, cd), kinematics_vjp=timeit(fkv, qd, vd, cd))
    if save:
        ko = [np.asarray(x) for x in dq.kinematics(q, v, com)]
        Gk = [rng.normal(size=x[:nw].shape).astype(np.float32) for x in ko]
        kg = jax.grad(lambda qq, vv: sum(jnp.vdot(g, x) for g, x in zip(Gk, dq.kinematics(qq, vv, com[:nw]))),
                      argnums=(0, 1))(jnp.asarray(q[:nw]), jnp.asarray(v[:nw]))
        np.savez(save, q=q, v=v, com=com, bodies=bodies, pts=pts, M=M, bias=bias, jp=np.asarray(jp),
                 jr=np.asarray(jr), Gp=Gp, Gr=Gr, gq=gq, gp=gp, nw=nw, **{f"k{i}": x for i, x in enumerate(ko)},
                 **{f"G{i}": x for i, x in enumerate(Gk)}, kgq=np.asarray(kg[0]), kgv=np.asarray(kg[1]))
    out["pass"] = bool(max(out["M"]["rel"], out["bias"]["rel"], out["jacp"]["rel"], out["jacr"]["rel"]) <= TOL
                       and max(out["vjp"]["qpos_rel"], out["vjp"]["points_rel"]) <= TOL_VJP and xfrc_ok)
    if params and "M_model_values" in out:
        out["pass"] = bool(out["pass"] and max(out["M_model_values"]["rel"], out["bias_model_values"]["rel"]) <= TOL)
    return out


def param_rows(m, com_body, params, B, rng):
    """Per-world parameter rows (B, width): armature x U(0.5, 2) + U(0, 0.01), torso mass / moments x U(0.7, 1.5),
    gravity opt.gravity + U(-3, 3) per component (world 0: zero gravity), wrenches random (force N(0, 40), torque
    N(0, 4), point the COM + U(-0.1, 0.1))."""
    import mujoco
    lay = params.layout(m.nv)
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, com_body) if com_body else -1
    pw = np.zeros((B, params.width(m.nv)), np.float32)
    if "dof_armature" in lay:
        o, n = lay["dof_armature"]
        pw[:, o:o + n] = m.dof_armature * rng.uniform(0.5, 2.0, (B, m.nv)) + rng.uniform(0, 0.01, (B, m.nv))
    if "torso_mass" in lay:
        pw[:, lay["torso_mass"][0]] = m.body_mass[tb] * rng.uniform(0.7, 1.5, B)
    if "torso_inertia" in lay:
        o, n = lay["torso_inertia"]
        pw[:, o:o + n] = m.body_inertia[tb] * rng.uniform(0.7, 1.5, (B, 3))
    if "gravity" in lay:
        o, n = lay["gravity"]
        pw[:, o:o + n] = m.opt.gravity + rng.uniform(-3, 3, (B, 3))
        pw[0, o:o + n] = 0.0
    if "xfrc" in lay:
        o, n = lay["xfrc"]
        x = np.zeros((B, len(params.xfrc_bodies), 9), np.float32)
        for k, b in enumerate(params.xfrc_bodies):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
            x[:, k, :3] = rng.normal(0, 40, (B, 3))
            x[:, k, 3:6] = rng.normal(0, 4, (B, 3))
            x[:, k, 6:] = m.body_ipos[bid] + rng.uniform(-0.1, 0.1, (B, 3))
        pw[:, o:o + n] = x.reshape(B, -1)
    return pw


KIN_CASES = ("kin_tree", "kin_g1_forest", "kin_g1_box", "kin_multitree", "kin_slide_tendon")
KIN_OUT = ("qpos_n", "xpos", "xquat", "cvel", "subtree_com")


def kin_model(name):
    """(MjModel, foot_pairs, com_body for MuJoCo's reference) of a kinematics case (jax_ffi.init's defaults)."""
    import mujoco
    if name in ("kin_tree", "kin_g1_forest"):
        m = mujoco.MjModel.from_binary_path(str(REPO / "fixtures" / "mjlab" / "g1_train.mjb"))
        return m, json.loads((REPO / "fixtures" / "mjlab" / "g1_meta.json").read_text())["foot_pairs"], "torso_link"
    if name == "kin_g1_box":
        from nablaengine.envs import convert
        return convert.to_engine(G1BOX_XML.read_text()).model, [], None
    if name == "kin_multitree":
        return mujoco.MjModel.from_xml_path(str(REPO / "scripts" / "contact_forces" / "multitree.xml")), [], None
    if name == "kin_slide_tendon":
        return mujoco.MjModel.from_xml_string(SLIDE_TENDON_XML), [], None
    raise ValueError(name)


def mj_kin(m, q, v, com=None, com_body=None):
    """MuJoCo float64 (mj_kinematics + mj_comPos + mj_comVel, by mj_forward) at the inputs: the five outputs."""
    import mujoco
    mm = copy.copy(m)
    d = mujoco.MjData(mm)
    tb = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, com_body) if com_body else -1
    ip0 = mm.body_ipos[tb].copy() if tb >= 0 else None
    B = q.shape[0]
    out = [np.zeros((B, m.nq)), np.zeros((B, m.nbody, 3)), np.zeros((B, m.nbody, 4)), np.zeros((B, m.nbody, 6)),
           np.zeros((B, m.nbody, 3))]
    for w in range(B):
        if tb >= 0 and com is not None:
            mm.body_ipos[tb] = ip0 + np.asarray(com[w, :3], np.float64)
        d.qpos[:] = q[w]
        d.qvel[:] = v[w]
        mujoco.mj_forward(mm, d)
        qn = d.qpos.copy()
        for j in range(m.njnt):
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                a = m.jnt_qposadr[j] + 3
                qn[a:a + 4] /= np.linalg.norm(qn[a:a + 4])
        for k, x in enumerate((qn, d.xpos, d.xquat, d.cvel, d.subtree_com)):
            out[k][w] = x
    return out


def fd_kin(m, q, v, G, com=None, com_body=None, h=1e-6):
    """Central differences of L = sum_k <G_k, out_k> of MuJoCo float64 w.r.t. qpos (raw) and qvel, one world."""
    def L(qq, vv):
        o = mj_kin(m, qq[None], vv[None], None if com is None else com[None], com_body)
        return sum(float((g * x[0]).sum()) for g, x in zip(G, o))
    q0, v0 = q.astype(np.float64), v.astype(np.float64)
    gq, gv = np.zeros(m.nq), np.zeros(m.nv)
    for k in range(m.nq):
        e = np.zeros(m.nq)
        e[k] = h
        gq[k] = (L(q0 + e, v0) - L(q0 - e, v0)) / (2 * h)
    for k in range(m.nv):
        e = np.zeros(m.nv)
        e[k] = h
        gv[k] = (L(q0, v0 + e) - L(q0, v0 - e)) / (2 * h)
    return gq, gv


def run_kin(name, B, nvjp, save):
    """jax_ffi.kinematics (and jax_ffi.step(kinematics=True)) of a case against MuJoCo, its VJP against FD."""
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    m, fp, com_body = kin_model(name)
    rng = np.random.default_rng({"kin_tree": 10, "kin_g1_forest": 10, "kin_g1_box": 11, "kin_multitree": 12,
                                 "kin_slide_tendon": 13}[name])
    info = jax_ffi.init(model=m, foot_pairs=fp)
    out = dict(case=name, engine=info["engine"], nbody=int(m.nbody), nq=int(m.nq), ntree=int(m.ntree), B=B,
               com_body=info["com_body"], library_kinematics=info["kinematics_error"] is None)
    q, v = random_states(m, B, rng)
    com = np.zeros((B, info["com_words"]), np.float32)
    if info["com_body"]:
        com[:, :3] = rng.normal(0, 0.02, (B, 3))
    res = [np.asarray(x) for x in jax_ffi.kinematics(jnp.asarray(q), jnp.asarray(v), jnp.asarray(com))]
    ref = mj_kin(m, q.astype(np.float64), v.astype(np.float64), com, info["com_body"])
    out["vs_mujoco"] = {k: err(x, r) for k, x, r in zip(KIN_OUT, res, ref)}
    ok = max(e["rel"] for e in out["vs_mujoco"].values()) <= TOL
    if name == "kin_tree":
        np.savez(save, q=q, v=v, com=com, **dict(zip(KIN_OUT, res)))
        out["pass"] = bool(ok)
        return out
    if name == "kin_g1_forest":
        z = np.load(save)
        assert np.array_equal(z["q"], q) and np.array_equal(z["com"], com)
        from nablaengine import dyn_query
        dyn_query.init(model=m, foot_pairs=fp, com_body=info["com_body"])
        gen = [np.asarray(x) for x in dyn_query.kinematics(q, v, com)]
        out["jax_ffi_vs_tree"] = {k: err(x, z[k].astype(np.float64)) for k, x in zip(KIN_OUT, res)}
        out["dyn_query_vs_tree"] = {k: err(x, z[k].astype(np.float64)) for k, x in zip(KIN_OUT, gen)}
        out["dyn_query_vs_mujoco"] = {k: err(x, r) for k, x, r in zip(KIN_OUT, gen, ref)}
        ok &= max(e["rel"] for d in ("jax_ffi_vs_tree", "dyn_query_vs_tree", "dyn_query_vs_mujoco")
                  for e in out[d].values()) <= TOL
    nw = min(nvjp, B)
    G = [rng.normal(size=x[:nw].shape).astype(np.float32) for x in res]
    kin = jax_ffi.kinematics
    if name == "kin_g1_forest":
        from nablaengine import dyn_query
        kin = dyn_query.kinematics

    def loss(qq, vv):
        return sum(jnp.vdot(g, x) for g, x in zip(G, kin(qq, vv, jnp.asarray(com[:nw]))))
    gq, gv = (np.asarray(x, np.float64) for x in jax.grad(loss, argnums=(0, 1))(jnp.asarray(q[:nw]), jnp.asarray(v[:nw])))
    rq, rv = [], []
    for w in range(nw):
        fq, fv = fd_kin(m, q[w], v[w], [g[w].astype(np.float64) for g in G], com[w], info["com_body"])
        rq.append(np.linalg.norm(gq[w] - fq) / max(np.linalg.norm(fq), 1e-30))
        rv.append(np.linalg.norm(gv[w] - fv) / max(np.linalg.norm(fv), 1e-30))
    out["vjp"] = dict(worlds=nw, qpos_rel=float(max(rq)), qvel_rel=float(max(rv)), fd_h=1e-6,
                      kernel="dyn_query" if name == "kin_g1_forest" else "jax_ffi.kinematics")
    ok &= max(rq + rv) <= TOL_VJP
    if name != "kin_g1_forest":
        n = min(16, B)
        st = [jnp.asarray(x[:n]) for x in (q, v)]
        warm = jnp.zeros((n, m.nv), jnp.float32)
        ctrl = jnp.zeros((n, m.nu), jnp.float32)
        fr = jnp.ones((n,), jnp.float32)
        cm = jnp.asarray(com[:n])
        sk = jax_ffi.make_step(iterations=5, substeps=2, kinematics=True)
        s0 = jax_ffi.make_step(iterations=5, substeps=2)
        Gk = [jnp.asarray(rng.normal(size=x[:n].shape).astype(np.float32)) for x in res]
        a = sk(*st, warm, ctrl, fr, cm)
        b0 = s0(*st, warm, ctrl, fr, cm)
        bk = jax_ffi.kinematics(b0[0], b0[1], cm)
        same_fwd = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, tuple(b0) + tuple(bk)))

        def l1(qq, vv):
            o = sk(qq, vv, warm, ctrl, fr, cm)
            return jnp.sum(o[0]) + sum(jnp.vdot(g, x) for g, x in zip(Gk, o[3:]))

        def l2(qq, vv):
            o = s0(qq, vv, warm, ctrl, fr, cm)
            return jnp.sum(o[0]) + sum(jnp.vdot(g, x) for g, x in zip(Gk, jax_ffi.kinematics(o[0], o[1], cm)))
        g1 = jax.grad(l1, argnums=(0, 1))(*st)
        g2 = jax.grad(l2, argnums=(0, 1))(*st)
        out["step_kinematics"] = dict(worlds=n, forward_bitwise=bool(same_fwd),
                                      grad_max_abs_diff=float(max(np.abs(np.asarray(x) - np.asarray(y)).max()
                                                                  for x, y in zip(g1, g2))),
                                      grad_finite=bool(all(np.isfinite(np.asarray(x)).all() for x in g1)))
        ok &= same_fwd and out["step_kinematics"]["grad_finite"] and out["step_kinematics"]["grad_max_abs_diff"] < 1e-4
    out["pass"] = bool(ok)
    return out


TERRAIN_CASES = ("terrain_g1", "terrain_g1_box")


def run_terrain(name, B):
    """A forest terrain build with object slots + TerrainConfig.object_velocity + per-world parameters: com rows
    (COM offset, 2 / 1 slots with random words, their motion words, then the parameters; info["com_layout"]).
    terrain_g1: the G1 (library kinematics kernels): jax_ffi.kinematics against MuJoCo (torso_mass per world moves
    subtree_com), dyn_query.init(com_layout=) dynamics against MuJoCo (gravity, torso_mass, armature per world).
    terrain_g1_box: the G1 + free box (two trees): jax_ffi.kinematics through the dispatch to dyn_query (its init with the
    build's com_layout) against MuJoCo, then dyn_query.dynamics on the same com rows (per-world gravity)."""
    import jax.numpy as jnp
    import mujoco
    from nablaengine import dyn_query, jax_ffi, terrain as T
    from nablaengine.params import PhysicsParams
    rng = np.random.default_rng(20 if name == "terrain_g1" else 21)
    if name == "terrain_g1":
        m, fp, com_body = kin_model("kin_g1_forest")
        P = PhysicsParams(("gravity", "torso_mass", "dof_armature"))
        cfg = T.TerrainConfig(geoms="feet", objects=2, object_velocity=True)
    else:
        m, fp, com_body = kin_model("kin_g1_box")
        com_body = None
        P = PhysicsParams(("gravity",))
        cfg = T.TerrainConfig(geoms=("left_foot1_collision", "right_foot1_collision"), objects=1, object_velocity=True)
    info = jax_ffi.init(model=m, foot_pairs=fp, terrain=cfg, params=P, com_body=com_body)
    lay = info["com_layout"]
    out = dict(case=name, engine=info["engine"], B=B, com_layout={k: [int(x) for x in v] for k, v in lay.items()},
               com_words=int(info["com_words"]), library_kinematics=info["kinematics_error"] is None,
               params=P.key())
    q, v = random_states(m, B, rng)
    K = int(cfg.objects)
    objs = rng.normal(size=(B, K, T.OBJ_WORDS)).astype(np.float32)
    vel = rng.normal(size=(B, K, T.OBJ_VEL_WORDS)).astype(np.float32)
    c3 = np.zeros((B, 3), np.float32)
    if com_body:
        c3[:] = rng.normal(0, 0.02, (B, 3))
    pw = param_rows(m, com_body, P, B, rng)
    com = np.ascontiguousarray(np.concatenate([T.pack_com(c3, objs, vel), pw], 1), np.float32)
    assert com.shape[1] == int(info["com_words"])
    for f, (o, w) in P.layout(m.nv).items():
        assert tuple(lay[f"params.{f}"]) == (3 + (T.OBJ_WORDS + T.OBJ_VEL_WORDS) * K + o, w), (f, lay)
    res = [np.asarray(x) for x in jax_ffi.kinematics(jnp.asarray(q), jnp.asarray(v), jnp.asarray(com))]
    if "torso_mass" in P.fields:
        ref = [np.zeros_like(x, np.float64) for x in res]
        tb = m.body(com_body).id
        o = P.layout(m.nv)["torso_mass"][0]
        for w in range(B):
            mm = copy.copy(m)
            mm.body_mass[tb] = pw[w, o]
            mujoco.mj_setConst(mm, mujoco.MjData(mm))
            for k, x in enumerate(mj_kin(mm, q[w:w + 1].astype(np.float64), v[w:w + 1].astype(np.float64),
                                         com[w:w + 1], com_body)):
                ref[k][w] = x[0]
    else:
        ref = mj_kin(m, q.astype(np.float64), v.astype(np.float64), com, com_body)
    out["kinematics_vs_mujoco"] = {k: err(x, r) for k, x, r in zip(KIN_OUT, res, ref)}
    ok = max(e["rel"] for e in out["kinematics_vs_mujoco"].values()) <= TOL_KIN_PARAMS
    if name == "terrain_g1":
        dyn_query.init(model=m, foot_pairs=fp, com_body=com_body, params=P, com_layout=lay)
        gen = [np.asarray(x) for x in dyn_query.kinematics(q, v, com)]
        out["dyn_query_kinematics_vs_mujoco"] = {k: err(x, r) for k, x, r in zip(KIN_OUT, gen, ref)}
        ok &= max(e["rel"] for e in out["dyn_query_kinematics_vs_mujoco"].values()) <= TOL_KIN_PARAMS
    else:
        out["kinematics_dispatch"] = dict(dyn_query_pw_off=int(dyn_query._J["info"]["pw_off"]))
        ok &= dyn_query._J["info"]["pw_off"] == lay["params.gravity"][0]
    M, bias = (np.asarray(x) for x in dyn_query.dynamics(q, v, com))
    Mr, br = mj_reference(m, q.astype(np.float64), v.astype(np.float64), com, com_body, P, pw)
    out["M"], out["bias"] = err(M, Mr), err(bias, br)
    c2 = com.copy()
    c2[:, 3:lay["params." + P.fields[0]][0]] = 0.0
    Mz, bz = (np.asarray(x) for x in dyn_query.dynamics(q, v, c2))
    out["slot_words_change_nothing"] = bool(np.array_equal(Mz, M) and np.array_equal(bz, bias))
    ok &= max(out["M"]["rel"], out["bias"]["rel"]) <= TOL and out["slot_words_change_nothing"]
    try:
        dyn_query.pw_offset(m, PhysicsParams(("dof_armature",)), lay)
        out["mismatch_refused"] = False
    except ValueError:
        out["mismatch_refused"] = True
    ok &= out["mismatch_refused"]
    out["pass"] = bool(ok)
    return out


def run_torch(save):
    import torch
    from nablaengine import dyn_query as dq
    m, fp, com_body, _ = load_case("torch")
    z = np.load(save)
    info = dq.torch_init(0, model=m, foot_pairs=fp, com_body=com_body)
    dev = torch.device("cuda", 0)
    t = lambda x: torch.as_tensor(x, device=dev)
    M, bias = dq.torch_dynamics(t(z["q"]), t(z["v"]), t(z["com"]))
    jp, jr = dq.torch_jacobian(t(z["q"]), t(z["bodies"]), t(z["pts"]))
    nw = int(z["nw"])
    q = t(z["q"][:nw]).requires_grad_(True)
    p = t(z["pts"][:nw]).requires_grad_(True)
    a, b = dq.torch_jacobian(q, t(z["bodies"][:nw]), p)
    ((a * t(z["Gp"])).sum() + (b * t(z["Gr"])).sum()).backward()
    ko = dq.torch_kinematics(t(z["q"]), t(z["v"]), t(z["com"]))
    kq = t(z["q"][:nw]).requires_grad_(True)
    kv = t(z["v"][:nw]).requires_grad_(True)
    sum((x * t(z[f"G{i}"])).sum() for i, x in enumerate(dq.torch_kinematics(kq, kv, t(z["com"][:nw])))).backward()
    d = lambda x, y: float(np.abs(x.detach().cpu().numpy().astype(np.float64) - y).max())
    out = dict(case="torch", built=info["built"], torch=torch.__version__,
               max_abs_vs_jax=dict(M=d(M, z["M"]), bias=d(bias, z["bias"]), jacp=d(jp, z["jp"]), jacr=d(jr, z["jr"]),
                                   g_qpos=d(q.grad, z["gq"]), g_points=d(p.grad, z["gp"]),
                                   **{f"kin_{k}": d(x, z[f"k{i}"]) for i, (k, x) in enumerate(zip(KIN_OUT, ko))},
                                   kin_g_qpos=d(kq.grad, z["kgq"]), kin_g_qvel=d(kv.grad, z["kgv"])))
    out["pass"] = bool(max(out["max_abs_vs_jax"].values()) == 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", default=",".join(CASES + INTERNAL_CASES))
    ap.add_argument("--B", type=int, default=512)
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--vjp-worlds", type=int, default=16)
    ap.add_argument("--timing-B", type=int, default=4096)
    ap.add_argument("--torch-python", default=os.environ.get("DK_TORCH_PY")
                    or (str(Path(os.environ["DK_ENV"]) / "venv_torch" / "bin" / "python") if os.environ.get("DK_ENV")
                        else sys.executable))
    ap.add_argument("--worker", default=None)
    ap.add_argument("--worker-out", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--ref-src", default=None, help="another source tree (e.g. origin/main's src): kin_tree also runs "
                    "there and its outputs must be bitwise this tree's")
    a = ap.parse_args()
    if a.worker:
        res = (run_torch(a.save) if a.worker == "torch" else run_kin(a.worker, a.B, a.vjp_worlds, a.save)
               if a.worker.startswith("kin_") else run_terrain(a.worker, a.B) if a.worker in TERRAIN_CASES
               else INTERNAL.run(a.worker, a.B) if a.worker in INTERNAL_CASES
               else run_case(a.worker, a.B, a.P, a.vjp_worlds, a.timing_B if a.worker == "g1" else 0, a.save))
        Path(a.worker_out).write_text(json.dumps(res))
        return 0
    tmp = Path(tempfile.mkdtemp(prefix="dynq_"))
    save = tmp / "g1.npz"
    results, ok = [], True
    cases = a.cases.split(",")
    if a.ref_src and "kin_tree" in cases:
        cases.insert(cases.index("kin_tree"), "kin_tree_ref")
    for c in cases:
        if c not in CASES + INTERNAL_CASES + ("kin_tree_ref",):
            res = dict(case=c, error="unknown case (or a case on internal scenes, not in this release)",
                       **{"pass": False})
            print(json.dumps(res, indent=1))
            results.append(res)
            ok = False
            continue
        py = a.torch_python if c == "torch" else sys.executable
        if c == "torch" and not save.exists():
            results.append(dict(case=c, skipped="needs the g1 case's outputs (run g1 first)"))
            continue
        o = tmp / f"{c}.json"
        env = dict(os.environ)
        if c == "kin_tree_ref":
            env["PYTHONPATH"] = a.ref_src
        if c.startswith("kin_tree"):
            env.pop("DK_ENGINE", None)
        if c in ("kin_g1_forest", "terrain_g1"):
            env["DK_ENGINE"] = "forest"
        cmd = [py, __file__, "--worker", "kin_tree" if c == "kin_tree_ref" else c, "--worker-out", str(o), "--B",
               str(a.B), "--P", str(a.P), "--vjp-worlds", str(a.vjp_worlds), "--timing-B", str(a.timing_B)]
        if c in ("g1", "torch"):
            cmd += ["--save", str(save)]
        if c.startswith("kin_tree") or c == "kin_g1_forest":
            cmd += ["--save", str(tmp / ("kin_tree_ref.npz" if c == "kin_tree_ref" else "kin_tree.npz"))]
        r = subprocess.run(cmd, env=env)
        res = json.loads(o.read_text()) if o.exists() else dict(case=c, error=f"exit {r.returncode}", **{"pass": False})
        if c == "kin_tree" and a.ref_src and res.get("pass") is not None:
            z1, z2 = np.load(tmp / "kin_tree.npz"), np.load(tmp / "kin_tree_ref.npz")
            res["bitwise_vs_ref_src"] = bool(all(np.array_equal(z1[k], z2[k]) for k in KIN_OUT))
            res["pass"] = bool(res["pass"] and res["bitwise_vs_ref_src"])
        if c == "kin_tree_ref":
            res["case"] = "kin_tree_ref"
        print(json.dumps(res, indent=1))
        results.append(res)
        ok &= bool(res.get("pass"))
    import mujoco
    from nablaengine import provenance
    commit, dirty = provenance.source_commit()
    doc = dict(check="scripts/forest/dyn_query_check.py", date=time.strftime("%Y-%m-%d"), mujoco=mujoco.__version__,
               provenance=dict(commit=commit, dirty=dirty), tolerance=dict(rel=TOL, vjp_rel=TOL_VJP),
               normalisation="per world: abs = max |engine - MuJoCo|, rel = abs / max |MuJoCo| of that quantity "
                             "(float32 engine vs MuJoCo float64 at the same float32-rounded inputs); vjp rel = "
                             "|g - g_fd|_2 / |g_fd|_2 against float64 central differences of mj_jac",
               cases=results, all_pass=bool(ok))
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print("ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

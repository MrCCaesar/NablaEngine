"""Actuator torque-speed limits (nablaengine.motor) on the forest engine against an exact emulation in MuJoCo Warp /
MuJoCo C (scripts/motor_check.py's method for the G1 engine, generalized to affine actuators with gear, ctrlrange and
joint springs).

The emulation: before each substep, ctrl' per actuator such that MuJoCo's actuator force on ctrl' is the limited force
f_lim: f = gear (gain u + b0 + b1 gear q + b2 gear qdot) with u = ctrl clamped to ctrlrange, clamped at the
actuator forcerange and the joint actfrcrange, then at the torque-speed curve (motor.limit, float64, at the substep's
input state); ctrl' = (f_lim / gear - b0 - b1 gear q - b2 gear qdot) / gain, stepped with ctrl clamping disabled. The
implicit kv (implicitfast) does not depend on ctrl, so the emulated step is the engine's motor-limited step wherever
the actuator force is not at its forcerange (the engine drops the implicit kv there, decided before the torque-speed
limit, as MuJoCo Warp's forcerange rule; the emulation cannot): the test models have no actuator forcerange (load_model
clears actuator_forcelimited).

Models (--model): xml:PATH (any MJCF the forest engine takes after nablaengine.envs.convert, e.g. a MuJoCo Menagerie
arm's scene.xml with a tendon actuator: actuators on fixed tendons are not torque-speed limited, their ctrl passes
through), cheetah (dm_control cheetah-run converted by nablaengine.envs: 6 torque motors, Euler, joint springs, floor
contacts; needs dm_control), arm (scripts/collision/models/midphase_arm_box.xml: a 7-joint arm with a slide gripper and
a free box, 9 position servos, implicitfast, no forcerange). The torque-speed curves are set per actuated joint from the states (MotorCurve.unitree
with the peak at --peak times the 90th percentile of the unlimited force, the no-load speed at --speed times the 90th
percentile of |qdot|, the knee at 30 % of it), so a share of the dofs sits on the flat and sloped pieces.

  ref    (MuJoCo Warp env, PYTHONPATH=src): states from MuJoCo C random-action rollouts (qvel times --vscale, ctrl
         uniform over 1.2 x ctrlrange), the model (OUT.mjb) and curves, one call of --substeps substeps in MuJoCo Warp
         with the emulated ctrl and without limits; writes OUT.npz
  check  (JAX): the engine with the curves (and with --motor none: the default build against the unlimited run)
  grad64 (JAX and MuJoCo): the VJP against float64 central differences of the emulation in MuJoCo C (tolerance 1e-12)
         over the worlds whose forward agrees with MuJoCo C

  python scripts/motor_forest_check.py ref --model cheetah --out motor/cheetah.npz
  python scripts/motor_forest_check.py check --ref motor/cheetah.npz [--motor none]
  python scripts/motor_forest_check.py grad64 --ref motor/cheetah.npz [--clamp-kv 1]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_model(name):
    import mujoco
    if name == "cheetah":
        from nablaengine import envs
        from nablaengine.envs import convert
        sp = envs.spec("cheetah-run")
        xml, assets = sp.source()
        m = convert.to_engine(xml, assets, **sp.convert_options()).model
        sp.adjust_model(m)
        return m
    if name == "arm":
        m = mujoco.MjModel.from_xml_path(str(REPO / "scripts" / "collision" / "models" / "midphase_arm_box.xml"))
        m.actuator_forcelimited[:] = 0
        return m
    if name.startswith("xml:"):
        sys.path.insert(0, str(REPO / "scripts" / "eq"))
        import connect_check
        m = connect_check.load(name[4:], uniform_limits=True).model
        m.actuator_forcelimited[:] = 0
        return m
    raise ValueError(name)


def act_arrays(m):
    """Per actuator: dof, gear, gain, b0, b1, b2, ctrllimited, ctrlrange, forcelimited, forcerange, joint actfrc;
    jnt: whether it drives a joint (a tendon actuator is not torque-speed limited: its ctrl passes through, and its
    dof / qadr / force entries are placeholders)."""
    import mujoco
    jm = m.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT)
    j = np.where(jm, m.actuator_trnid[:, 0], 0)
    fr = np.where(jm & (m.jnt_actfrclimited[j] != 0), m.jnt_actfrcrange[j, 1], np.inf)
    return dict(jnt=jm, dof=m.jnt_dofadr[j].astype(int), qadr=m.jnt_qposadr[j].astype(int),
                gear=m.actuator_gear[:, 0].copy(),
                gain=m.actuator_gainprm[:, 0].copy(), b=m.actuator_biasprm[:, :3].copy(),
                cl=m.actuator_ctrllimited.astype(bool), cr=m.actuator_ctrlrange.copy(),
                fl=m.actuator_forcelimited.astype(bool), frg=m.actuator_forcerange.copy(), afr=fr)


def joint_force(A, q, v, u):
    """The unlimited actuator force on each actuated dof (..., nu) at (q, v) under ctrl u, MuJoCo's order (float64)."""
    q, v, u = (np.asarray(x, np.float64) for x in (q, v, u))
    qa, va = q[..., A["qadr"]], v[..., A["dof"]]
    u = np.where(A["cl"], np.clip(u, A["cr"][:, 0], A["cr"][:, 1]), u)
    F = A["gain"] * u + A["b"][:, 0] + A["b"][:, 1] * A["gear"] * qa + A["b"][:, 2] * A["gear"] * va
    F = np.where(A["fl"], np.clip(F, A["frg"][:, 0], A["frg"][:, 1]), F)
    return np.clip(A["gear"] * F, -A["afr"], A["afr"]), qa, va


def limited(A, table, f, va):
    """motor.limit on the joint actuators' forces (the tendon actuators' unchanged, slope 0)."""
    from nablaengine import motor
    fl, sl = motor.limit(table[A["dof"]], f, va)
    return np.where(A["jnt"], fl, f), np.where(A["jnt"], sl, 0.0)


def emulated_ctrl(A, table, q, v, u):
    f, qa, va = joint_force(A, q, v, u)
    fl, _ = limited(A, table, f, va)
    ue = (fl / A["gear"] - A["b"][:, 0] - A["b"][:, 1] * A["gear"] * qa - A["b"][:, 2] * A["gear"] * va) / A["gain"]
    return np.where(A["jnt"], ue, np.where(A["cl"], np.clip(u, A["cr"][:, 0], A["cr"][:, 1]), u))


def limits_of(m, names, rows):
    from nablaengine import motor
    import re
    return motor.MotorLimits(curves=tuple((re.escape(n), motor.MotorCurve(*map(float, r))) for n, r in zip(names, rows)),
                             name="forest_check")


def states(m, B, steps, vscale, seed):
    """B states from MuJoCo C rollouts with uniform random ctrl (one state per rollout at a random time)."""
    import mujoco
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    out = {k: np.zeros((B, n)) for k, n in (("qpos", m.nq), ("qvel", m.nv), ("qacc_warmstart", m.nv), ("ctrl", m.nu))}
    for i in range(B):
        mujoco.mj_resetData(m, d)
        for j in range(m.njnt):
            if m.jnt_type[j] >= 2:
                d.qpos[m.jnt_qposadr[j]] += rng.normal(0, 0.05)
        T = int(rng.integers(steps // 2, steps))
        u = rng.uniform(lo, hi)
        for t in range(T):
            if t % 5 == 0:
                u = rng.uniform(lo, hi)
            d.ctrl[:] = u
            mujoco.mj_step(m, d)
        out["qpos"][i], out["qvel"][i], out["qacc_warmstart"][i] = d.qpos, d.qvel, d.qacc_warmstart
    A = act_arrays(m)
    out["qvel"][:, A["dof"]] *= vscale
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    out["ctrl"] = rng.uniform(mid - 1.2 * half, mid + 1.2 * half, (B, m.nu))
    return out


def cmd_ref(a):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    m = load_model(a.model)
    m.opt.iterations = a.iterations
    z = states(m, a.nworld, a.rollout, a.vscale, a.seed)
    A = act_arrays(m)
    f, _, va = joint_force(A, z["qpos"], z["qvel"], z["ctrl"])
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in m.actuator_trnid[A["jnt"], 0]]
    rows = []
    for k in np.flatnonzero(A["jnt"]):
        y1 = a.peak * np.quantile(np.abs(f[:, k]), 0.9)
        x2 = a.speed * max(np.quantile(np.abs(va[:, k]), 0.9), 1e-3)
        from nablaengine import motor
        rows.append(motor.MotorCurve.unitree(y1, 1.3 * y1, 0.3 * x2, x2).row())
    rows = np.asarray(rows)
    table = limits_of(m, names, rows).table(m)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mjb = out.with_suffix(".mjb")
    mujoco.mj_saveModel(m, str(mjb), None)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    mw = mjw.put_model(m)
    B = a.nworld
    res = {}
    for name, lim in (("lim", True), ("free", False)):
        dw = mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=a.nconmax, njmax=a.njmax)
        for k in ("qpos", "qvel", "qacc_warmstart"):
            wp.copy(getattr(dw, k), wp.array(np.ascontiguousarray(z[k], np.float32)))
        for _ in range(a.substeps):
            u = (emulated_ctrl(A, table, dw.qpos.numpy(), dw.qvel.numpy(), z["ctrl"]) if lim
                 else np.where(A["cl"], np.clip(z["ctrl"], A["cr"][:, 0], A["cr"][:, 1]), z["ctrl"]))
            wp.copy(dw.ctrl, wp.array(np.ascontiguousarray(u, np.float32)))
            mjw.step(mw, dw)
        res[f"out_qpos_{name}"], res[f"out_qvel_{name}"] = dw.qpos.numpy(), dw.qvel.numpy()
    np.savez(out, **{k: v.astype(np.float32) for k, v in z.items()}, substeps=a.substeps, iterations=a.iterations,
             model=a.model, mjb=mjb.name, joint_names=np.asarray(names), curves=rows, **res)
    dv = np.linalg.norm(res["out_qvel_lim"] - res["out_qvel_free"], axis=1)
    fl, sl = limited(A, table, f, va)
    print(json.dumps(dict(model=a.model, nworld=B, out=str(out), worlds_changed_by_limits=float((dv > 1e-6).mean()),
                          input_dofs_at_curve=float((np.abs(fl) < np.abs(f)).mean()),
                          input_dofs_on_slope=float((sl != 0).mean()))))


def _load(a):
    import mujoco
    r = np.load(a.ref)
    m = mujoco.MjModel.from_binary_path(str(Path(a.ref).parent / str(r["mjb"])))
    names, rows = [str(x) for x in r["joint_names"]], r["curves"]
    return r, m, limits_of(m, names, rows)


def _init(m, lim):
    from nablaengine import jax_ffi
    return jax_ffi.init(model=m, foot_pairs=[], motor_limits=lim)


def _com(info, B):
    return np.zeros((B, info["com_words"]), np.float32)


def cmd_check(a):
    import jax.numpy as jnp
    from nablaengine import jax_ffi, motor
    r, m, ml = _load(a)
    use = a.motor != "none"
    info = _init(m, ml if use else None)
    B = len(r["qpos"])
    f = lambda k: jnp.asarray(np.ascontiguousarray(r[k], np.float32))
    q1, v1, _, inf = jax_ffi.forward(f("qpos"), f("qvel"), f("qacc_warmstart"), f("ctrl"), jnp.ones((B,), jnp.float32),
                                     jnp.asarray(_com(info, B)), iterations=int(r["iterations"]),
                                     substeps=int(r["substeps"]))
    q1, v1 = np.asarray(q1), np.asarray(v1)
    ref = "lim" if use else "free"
    rv = r[f"out_qvel_{ref}"]
    ev = np.linalg.norm(v1 - rv, axis=1) / np.maximum(np.linalg.norm(rv, axis=1), 1e-6)
    A = act_arrays(m)
    fpd, _, va = joint_force(A, r["qpos"], r["qvel"], r["ctrl"])
    fl, sl = limited(A, ml.table(m), fpd, va)
    at, slope = (np.abs(fl) < np.abs(fpd)).any(1), (sl != 0).any(1)
    q = lambda x: dict(n=int(x.sum()), median=float(np.median(ev[x])), p99=float(np.quantile(ev[x], 0.99)),
                       max=float(ev[x].max())) if x.any() else None
    res = dict(model=str(r["model"]), motor=a.motor, reference=ref, nworld=B, engine_motor_key=info["motor_limits"],
               rec_words=info["rec_words"], qvel_rel_all=q(np.ones(B, bool)), qvel_rel_with_dof_at_curve=q(at),
               qvel_rel_with_dof_on_slope=q(slope), share_rel_below_1e3=float((ev < 1e-3).mean()),
               share_rel_below_1e4=float((ev < 1e-4).mean()),
               qpos_abs_max=float(np.abs(q1 - r[f"out_qpos_{ref}"]).max()),
               flags=int((np.asarray(inf)[:, 3] != 0).sum()))
    print(json.dumps(res, indent=1))


def cmd_grad64(a):
    import jax
    import jax.numpy as jnp
    import mujoco
    from nablaengine import jax_ffi, motor
    r, m, ml = _load(a)
    use = a.motor != "none"
    info = _init(m, ml if use else None)
    table = ml.table(m)
    A = act_arrays(m)
    B, S = min(a.nworld, len(r["qpos"])), int(r["substeps"])
    f64 = lambda k: np.asarray(r[k][:B], np.float64)
    t = lambda x: jnp.asarray(np.asarray(x, np.float32))
    rng = np.random.default_rng(1)
    yq, yv = rng.standard_normal((B, m.nq)), rng.standard_normal((B, m.nv))
    fr, com = jnp.ones((B,), jnp.float32), jnp.asarray(_com(info, B))

    def loss(q, v, u):
        q1, v1, _ = jax_ffi.step(q, v, t(f64("qacc_warmstart")), u, fr, com, iterations=a.iterations, substeps=S,
                                 clamp_kv=a.clamp_kv)
        return jnp.sum(t(yq) * q1) + jnp.sum(t(yv) * v1), v1

    (_, v1), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(
        t(f64("qpos")), t(f64("qvel")), t(f64("ctrl")))
    v1 = np.asarray(v1, np.float64)
    m.opt.iterations, m.opt.tolerance, m.opt.ls_tolerance, m.opt.ls_iterations = a.iterations, 1e-12, 1e-12, 100
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    d = mujoco.MjData(m)

    def run(i, q, v, u):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.qacc_warmstart[:] = q, v, f64("qacc_warmstart")[i]
        for _ in range(S):
            d.ctrl[:] = (emulated_ctrl(A, table, d.qpos, d.qvel, u) if use
                         else np.where(A["cl"], np.clip(u, A["cr"][:, 0], A["cr"][:, 1]), u))
            mujoco.mj_step(m, d)
        return d.qpos.copy(), d.qvel.copy()

    base = [(f64("qpos")[i], f64("qvel")[i], f64("ctrl")[i]) for i in range(B)]
    vc = np.array([run(i, *base[i])[1] for i in range(B)])
    agree = np.linalg.norm(v1 - vc, axis=1) / np.maximum(np.linalg.norm(vc, axis=1), 1e-6) < a.agree
    fpd, _, va = joint_force(A, f64("qpos"), f64("qvel"), f64("ctrl"))
    _, sl0 = limited(A, table, fpd, va)
    slope = (sl0 != 0).any(1)
    free = np.zeros(m.nq, bool)
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            free[m.jnt_qposadr[j] + 3:m.jnt_qposadr[j] + 7] = True
        elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
            free[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 4] = True
    res = dict(model=str(r["model"]), motor=a.motor, clamp_kv=a.clamp_kv, iterations=a.iterations, nworld=B,
               forward_agree_share=float(agree.mean()), worlds_with_dof_on_slope=int(slope.sum()))
    for k, name in enumerate(("qpos", "qvel", "ctrl")):
        an = np.asarray(grads[k], np.float64)
        rel = []
        for i in range(B):
            u = rng.standard_normal(an.shape[1])
            if name == "qpos":
                u[free] = 0.0
            u /= np.linalg.norm(u)
            lp, lm = [x.copy() for x in base[i]], [x.copy() for x in base[i]]
            lp[k] += a.eps * u
            lm[k] -= a.eps * u
            (qp, vp), (qm, vm) = run(i, *lp), run(i, *lm)
            fd = (yq[i] @ (qp - qm) + yv[i] @ (vp - vm)) / (2 * a.eps)
            g = an[i] @ u
            rel.append(abs(fd - g) / max(abs(fd), abs(g), 1e-12))
        rel = np.asarray(rel)
        res[name] = dict(agree_rel_median=float(np.median(rel[agree])) if agree.any() else None,
                         agree_rel_p90=float(np.quantile(rel[agree], 0.9)) if agree.any() else None)
        sa = agree & slope
        if sa.any():
            res[name].update(slope_agree_n=int(sa.sum()), slope_agree_rel_median=float(np.median(rel[sa])),
                             slope_agree_rel_p90=float(np.quantile(rel[sa], 0.9)))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ref")
    p.add_argument("--model", required=True, help='cheetah, arm or xml:PATH (an MJCF converted by '
                   'nablaengine.envs.convert with uniform joint limits, forceranges removed; tendon actuators unlimited)')
    p.add_argument("--out", required=True)
    p.add_argument("--nworld", type=int, default=1024)
    p.add_argument("--rollout", type=int, default=200, help="MuJoCo C rollout length (steps) per state")
    p.add_argument("--vscale", type=float, default=1.5)
    p.add_argument("--peak", type=float, default=0.6)
    p.add_argument("--speed", type=float, default=1.5)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nconmax", type=int, default=64)
    p.add_argument("--njmax", type=int, default=256)
    for name in ("check", "grad64"):
        p = sub.add_parser(name)
        p.add_argument("--ref", required=True)
        p.add_argument("--motor", default="curves", help='"none": the default build')
    p = sub.choices["grad64"]
    p.add_argument("--clamp-kv", type=int, default=1)
    p.add_argument("--nworld", type=int, default=128)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--agree", type=float, default=1e-4, help="forward agreement with MuJoCo C (relative qvel)")
    a = ap.parse_args()
    {"ref": cmd_ref, "check": cmd_check, "grad64": cmd_grad64}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Actuator torque-speed limits (nablaengine.motor) against an exact emulation in MuJoCo / MuJoCo Warp.

The emulation: before each substep, ctrl' = q + (f_lim + kv qdot) / kp per actuated dof, f_lim the PD force
kp (ctrl - q) - kv qdot clamped at jnt_actfrcrange and then at the torque-speed curve (motor.limit, float64, at the
substep's input state): MuJoCo's own actuator force on ctrl' is f_lim (ctrl clamping disabled, as the engine does not
clamp ctrl: capability H11c), and its implicitfast damping (kv on every actuated dof in MuJoCo 3.17) is the engine's
default. So the emulated step is the motor-limited step.

  ref    (MuJoCo Warp env): one control step of --nworld states, MuJoCo Warp stepped substep by substep with the
         emulated ctrl, and the same without limits; writes OUT.npz. States: the G1 corpus mf_s0, with --stress the
         hinge qvel times --vscale and ctrl = hinge qpos + N(0, --offset) (forces at the curves, joints past the knees)
  check  (JAX): the engine built with --motor (default unitree_g1; "none": the default build, against the reference's
         unlimited run) on the same inputs: qvel / qpos against the reference, and the share of dofs at the curve and
         on a sloped piece (motor.limit at the inputs)
  selffd (JAX): the VJP against the engine's own float32 central differences
  grad64 (JAX and MuJoCo): the VJP against float64 central differences of the emulation in MuJoCo C (the same
         iterations, tolerance 1e-12) over the worlds whose forward agrees with MuJoCo C
  dump / compare: the forward outputs and VJPs (clamp_kv 0 / 1 / 2) through JAX or PyTorch, compared bitwise

  python scripts/motor_check.py ref --out runs/motor/g1_stress.npz --stress     (MuJoCo Warp env)
  python scripts/motor_check.py check --ref runs/motor/g1_stress.npz [--motor none]
  python scripts/motor_check.py grad64 --ref runs/motor/g1_stress.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def act_params(m):
    """(kp, kv, frclim) per actuator (the engine's assumptions: source.constants)."""
    kp = np.asarray(m.actuator_gainprm[:, 0], np.float64)
    kv = -np.asarray(m.actuator_biasprm[:, 2], np.float64)
    j = m.actuator_trnid[:, 0]
    fr = np.where(m.jnt_actfrclimited[j] != 0, m.jnt_actfrcrange[j, 1], np.inf)
    return kp, kv, fr


def emulated_ctrl(m, table, q, v, u):
    """ctrl' (..., nu) whose MuJoCo actuator force is the motor-limited force at (q, v) (float64)."""
    from nablaengine import motor
    kp, kv, fr = act_params(m)
    qh, vh = np.asarray(q, np.float64)[..., 7:], np.asarray(v, np.float64)[..., 6:]
    f = np.clip(kp * (np.asarray(u, np.float64) - qh) - kv * vh, -fr, fr)
    fl, _ = motor.limit(table[6:], f, vh)
    return qh + (fl + kv * vh) / kp


def stress_states(m, z, B, vscale, offset, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(B) % len(z["qpos"])
    out = {k: np.array(z[k])[idx] for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction")}
    out["qvel"][:, 6:] *= vscale
    out["ctrl"] = out["qpos"][:, 7:] + rng.normal(0.0, offset, out["ctrl"].shape)
    return out


def cmd_ref(a):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    from nablaengine import fixtures, motor
    m, mt = fixtures.load_model()
    fp, com_body, sub = list(mt["foot_pairs"]), mt.get("torso_body", "torso_link"), 4
    z = fixtures.load_corpus("mf_s0")
    B = min(a.nworld, len(z["qpos"])) if not a.stress else a.nworld
    z = (stress_states(m, z, B, a.vscale, a.offset) if a.stress
         else {k: np.array(z[k])[:B] for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction")})
    table = motor.as_motor(a.motor).table(m)
    fr = np.asarray(z["friction"][:B], np.float64)
    m.opt.iterations = a.iterations
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    if a.ls_iterations:
        m.opt.ls_iterations = a.ls_iterations
    mw = mjw.put_model(m)
    pfr = np.tile(m.pair_friction, (B, 1, 1)).astype(np.float32)
    for p in fp:
        pfr[:, p, :2] = fr[:, None]
    mw.pair_friction = wp.array(pfr, dtype=mw.pair_friction.dtype)
    res = {}
    for name, lim in (("lim", True), ("free", False)):
        dw = mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=a.nconmax, njmax=a.njmax)
        for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
            wp.copy(getattr(dw, k), wp.array(np.ascontiguousarray(z[k][:B], np.float32)))
        for _ in range(sub):
            if lim:
                u = emulated_ctrl(m, table, dw.qpos.numpy(), dw.qvel.numpy(), z["ctrl"][:B])
                wp.copy(dw.ctrl, wp.array(np.ascontiguousarray(u, np.float32)))
            mjw.step(mw, dw)
        res[f"out_qpos_{name}"], res[f"out_qvel_{name}"] = dw.qpos.numpy(), dw.qvel.numpy()
    np.savez(a.out, qpos=z["qpos"][:B], qvel=z["qvel"][:B], qacc_warmstart=z["qacc_warmstart"][:B],
             ctrl=z["ctrl"][:B], friction=fr, com=np.zeros((B, 3)), substeps=sub, iterations=a.iterations,
             motor=a.motor, foot_pairs=np.asarray(fp), com_body=str(com_body), stress=a.stress, **res)
    dv = np.linalg.norm(res["out_qvel_lim"] - res["out_qvel_free"], axis=1)
    print(json.dumps(dict(nworld=B, out=a.out, stress=a.stress, worlds_changed_by_limits=float((dv > 1e-6).mean()),
                          qvel_change_median=float(np.median(dv)))))


def _engine(a, motor_limits):
    from nablaengine import fixtures, jax_ffi
    r = np.load(a.ref)
    m, _ = fixtures.load_model()
    jax_ffi.init(model=m, foot_pairs=[int(p) for p in r["foot_pairs"]], com_body=str(r["com_body"]),
                 motor_limits=motor_limits)
    return r, m


def cmd_check(a):
    import jax.numpy as jnp
    from nablaengine import jax_ffi, motor
    lim = a.motor != "none"
    r, m = _engine(a, a.motor if lim else None)
    f = lambda k: jnp.asarray(np.ascontiguousarray(r[k], np.float32))
    q1, v1, _, info = jax_ffi.forward(f("qpos"), f("qvel"), f("qacc_warmstart"), f("ctrl"), f("friction"), f("com"),
                                      iterations=int(r["iterations"]), substeps=int(r["substeps"]))
    q1, v1 = np.asarray(q1), np.asarray(v1)
    ref = "lim" if lim else "free"
    ev = np.linalg.norm(v1 - r[f"out_qvel_{ref}"], axis=1) / np.maximum(np.linalg.norm(r[f"out_qvel_{ref}"], axis=1), 1e-6)
    table = motor.as_motor(str(r["motor"])).table(m)
    kp, kv, frc = act_params(m)
    vh = np.asarray(r["qvel"], np.float64)[:, 6:]
    fpd = np.clip(kp * (r["ctrl"] - r["qpos"][:, 7:]) - kv * vh, -frc, frc)
    fl, sl = motor.limit(table[6:], fpd, vh)
    res = dict(nworld=len(q1), motor=a.motor, reference=ref, info_motor_limits=jax_ffi.init()["motor_limits"],
               qvel_rel_median=float(np.median(ev)), qvel_rel_p99=float(np.quantile(ev, 0.99)),
               qvel_rel_max=float(ev.max()), qpos_abs_max=float(np.abs(q1 - r[f"out_qpos_{ref}"]).max()),
               flags=int((np.asarray(info)[:, 3] != 0).sum()),
               input_dofs_at_curve=float((np.abs(fl) < np.abs(fpd)).mean()), input_dofs_on_slope=float((sl != 0).mean()))
    if a.dump:
        np.savez(a.dump, qvel_rel=ev, on_slope=(sl != 0).any(1), at_curve=(np.abs(fl) < np.abs(fpd)).any(1))
    if a.base:
        ok = np.load(a.base)["qvel_rel"] < 1e-4
        def sel(x):
            return dict(n=int(x.sum()), qvel_rel_median=float(np.median(ev[x])) if x.any() else None,
                        qvel_rel_p99=float(np.quantile(ev[x], 0.99)) if x.any() else None,
                        qvel_rel_max=float(ev[x].max()) if x.any() else None)
        slope = (sl != 0).any(1)
        res["base_agree"] = dict(all=sel(ok), with_dof_on_slope=sel(ok & slope),
                                 with_dof_at_curve=sel(ok & (np.abs(fl) < np.abs(fpd)).any(1)))
    print(json.dumps(res, indent=1))


def cmd_selffd(a):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    r, m = _engine(a, a.motor if a.motor != "none" else None)
    B, S, it = len(r["qpos"]), int(r["substeps"]), a.iterations or int(r["iterations"])
    t = lambda k: jnp.asarray(np.ascontiguousarray(r[k], np.float32))
    x0 = [t("qpos"), t("qvel"), t("ctrl")]
    rng = np.random.default_rng(0)
    yq = jnp.asarray(rng.standard_normal((B, m.nq)), jnp.float32)
    yv = jnp.asarray(rng.standard_normal((B, m.nv)), jnp.float32)

    def out(q, v, u):
        q1, v1, _, info = jax_ffi.forward(q, v, t("qacc_warmstart"), u, t("friction"), t("com"), iterations=it,
                                          substeps=S)
        return jnp.sum(yq * q1, 1) + jnp.sum(yv * v1, 1), info[:, 1]

    def loss(q, v, u):
        q1, v1, _ = jax_ffi.step(q, v, t("qacc_warmstart"), u, t("friction"), t("com"), iterations=it, substeps=S,
                                 clamp_kv=a.clamp_kv)
        return jnp.sum(yq * q1) + jnp.sum(yv * v1)

    grads = jax.grad(loss, argnums=(0, 1, 2))(*x0)
    res = dict(nworld=B, iterations=it, motor=a.motor, clamp_kv=a.clamp_kv)
    for k, (name, eps) in enumerate((("qpos", a.eps_q), ("qvel", a.eps_v), ("ctrl", a.eps_u))):
        d = rng.standard_normal(np.asarray(x0[k]).shape)
        if name == "qpos":
            d[:, 3:7] = 0.0
        d = jnp.asarray(d / np.linalg.norm(d, axis=1, keepdims=True), jnp.float32)
        xp, xm = list(x0), list(x0)
        xp[k], xm[k] = x0[k] + eps * d, x0[k] - eps * d
        (lp, np_), (lm, nm) = out(*xp), out(*xm)
        fd = (np.asarray(lp, np.float64) - np.asarray(lm, np.float64)) / (2 * eps)
        an = np.asarray(jnp.sum(grads[k] * d, 1), np.float64)
        rel = np.abs(fd - an) / np.maximum(np.maximum(np.abs(fd), np.abs(an)), 1e-12)
        same = np.asarray(np_) == np.asarray(nm)
        res[name] = dict(rel_median=float(np.median(rel[same])), rel_p90=float(np.quantile(rel[same], 0.9)),
                         same_contacts=float(same.mean()))
    print(json.dumps(res, indent=1))


def cmd_grad64(a):
    import jax
    import jax.numpy as jnp
    import mujoco
    from nablaengine import jax_ffi, motor
    lim = a.motor != "none"
    r, m = _engine(a, a.motor if lim else None)
    table = motor.as_motor(str(r["motor"])).table(m)
    fp = [int(p) for p in r["foot_pairs"]]
    B, S = min(a.nworld, len(r["qpos"])), int(r["substeps"])
    f64 = lambda k: np.asarray(r[k][:B], np.float64)
    t = lambda x: jnp.asarray(np.asarray(x, np.float32))
    rng = np.random.default_rng(1)
    yq, yv = rng.standard_normal((B, m.nq)), rng.standard_normal((B, m.nv))

    def loss(q, v, u):
        q1, v1, _ = jax_ffi.step(q, v, t(f64("qacc_warmstart")), u, t(f64("friction")), t(f64("com")),
                                 iterations=a.iterations, substeps=S, clamp_kv=a.clamp_kv)
        return jnp.sum(t(yq) * q1) + jnp.sum(t(yv) * v1), v1

    (_, v1), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(
        t(f64("qpos")), t(f64("qvel")), t(f64("ctrl")))
    v1 = np.asarray(v1, np.float64)
    m.opt.iterations, m.opt.tolerance, m.opt.ls_tolerance, m.opt.ls_iterations = a.iterations, 1e-12, 1e-12, 100
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    d = mujoco.MjData(m)
    fr = f64("friction")

    def run(i, q, v, u):
        mujoco.mj_resetData(m, d)
        for p in fp:
            m.pair_friction[p, :2] = fr[i]
        d.qpos[:], d.qvel[:], d.qacc_warmstart[:] = q, v, f64("qacc_warmstart")[i]
        for _ in range(S):
            d.ctrl[:] = emulated_ctrl(m, table, d.qpos, d.qvel, u) if lim else u
            mujoco.mj_step(m, d)
        return d.qpos.copy(), d.qvel.copy()

    base = [(f64("qpos")[i], f64("qvel")[i], f64("ctrl")[i]) for i in range(B)]
    vc = np.array([run(i, *base[i])[1] for i in range(B)])
    agree = np.linalg.norm(v1 - vc, axis=1) / np.maximum(np.linalg.norm(vc, axis=1), 1e-6) < 1e-4
    kp, kv, frc = act_params(m)
    vh = f64("qvel")[:, 6:]
    _, sl0 = motor.limit(table[6:], np.clip(kp * (f64("ctrl") - f64("qpos")[:, 7:]) - kv * vh, -frc, frc), vh)
    slope = (sl0 != 0).any(1)
    res = dict(nworld=B, motor=a.motor, clamp_kv=a.clamp_kv, iterations=a.iterations,
               forward_agree_share=float(agree.mean()), worlds_with_dof_on_slope=int(slope.sum()))
    for k, name in enumerate(("qpos", "qvel", "ctrl")):
        an = np.asarray(grads[k], np.float64)
        rel = []
        for i in range(B):
            u = rng.standard_normal(an.shape[1])
            if name == "qpos":
                u[3:7] = 0.0
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


def cmd_dump(a):
    """The forward outputs and the VJPs (clamp_kv 0, 1 and 2, a fixed random output cotangent) of the build with
    --motor through --backend (jax or torch), for compare."""
    from nablaengine import fixtures
    r = np.load(a.ref)
    m, _ = fixtures.load_model()
    B, S, it = min(a.nworld, len(r["qpos"])), int(r["substeps"]), int(r["iterations"])
    x = {k: np.ascontiguousarray(r[k][:B], np.float32) for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction",
                                                                  "com")}
    rng = np.random.default_rng(2)
    gq, gv = rng.standard_normal((B, m.nq)).astype(np.float32), rng.standard_normal((B, m.nv)).astype(np.float32)
    kw = dict(model=m, foot_pairs=[int(p) for p in r["foot_pairs"]], com_body=str(r["com_body"]),
              motor_limits=None if a.motor == "none" else a.motor)
    out = {}
    if a.backend == "jax":
        import jax.numpy as jnp
        from nablaengine import jax_ffi as F
        F.init(**kw)
        t = {k: jnp.asarray(v) for k, v in x.items()}
        args = (t["qpos"], t["qvel"], t["qacc_warmstart"], t["ctrl"], t["friction"], t["com"])
        q1, v1, w1, info, tp = F.forward(*args, iterations=it, substeps=S, tape=True)
        out.update(q1=np.asarray(q1), v1=np.asarray(v1), w1=np.asarray(w1), info=np.asarray(info))
        for kv in (0, 1, 2):
            g = F.backward(tp, t["friction"], t["com"], jnp.asarray(gq), jnp.asarray(gv), substeps=S, clamp_kv=kv,
                           **({"ctrl": t["ctrl"]} if kv == 2 else {}))
            for name, v in zip(("gq", "gv", "gu"), g[:3]):
                out[f"{name}_kv{kv}"] = np.asarray(v)
    else:
        import torch
        from nablaengine import torch_ffi as F
        F.init(**kw)
        t = {k: torch.from_numpy(v).cuda() for k, v in x.items()}
        args = (t["qpos"], t["qvel"], t["qacc_warmstart"], t["ctrl"], t["friction"], t["com"])
        q1, v1, w1, info, tp = F.forward(*args, iterations=it, substeps=S, tape=True)
        out.update(q1=q1.cpu().numpy(), v1=v1.cpu().numpy(), w1=w1.cpu().numpy(), info=info.cpu().numpy())
        for kv in (0, 1, 2):
            g = F.backward(tp, t["friction"], t["com"], torch.from_numpy(gq).cuda(), torch.from_numpy(gv).cuda(),
                           substeps=S, clamp_kv=kv, **({"ctrl": t["ctrl"]} if kv == 2 else {}))
            for name, v in zip(("gq", "gv", "gu"), g[:3]):
                out[f"{name}_kv{kv}"] = v.cpu().numpy()
    np.savez(a.out, **out)
    print(json.dumps(dict(out=a.out, backend=a.backend, motor=a.motor, nworld=B)))


def cmd_compare(a):
    x, y = np.load(a.a), np.load(a.b)
    res = {k: dict(bitwise=bool(np.array_equal(x[k], y[k])), max_abs=float(np.abs(x[k] - y[k]).max()))
           for k in x.files if k in y.files}
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ref")
    p.add_argument("--out", required=True)
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--motor", default="unitree_g1")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--vscale", type=float, default=3.0)
    p.add_argument("--offset", type=float, default=1.0)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--ls-iterations", type=int, default=None, help="MuJoCo Warp's line-search limit (default the model's)")
    p.add_argument("--nconmax", type=int, default=64)
    p.add_argument("--njmax", type=int, default=256)
    for name in ("check", "selffd", "grad64"):
        p = sub.add_parser(name)
        p.add_argument("--ref", required=True)
        p.add_argument("--motor", default="unitree_g1", help='a preset, or "none" (the default build)')
        if name != "check":
            p.add_argument("--clamp-kv", type=int, default=0)
        else:
            p.add_argument("--dump", default=None, help="per-world errors and flags (.npz)")
            p.add_argument("--base", default=None, help="another check's --dump: report on its worlds with error < 1e-4")
    p = sub.choices["selffd"]
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--eps-q", type=float, default=1e-4)
    p.add_argument("--eps-v", type=float, default=1e-3)
    p.add_argument("--eps-u", type=float, default=1e-3)
    p = sub.choices["grad64"]
    p.add_argument("--nworld", type=int, default=128)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-6)
    p = sub.add_parser("dump")
    p.add_argument("--ref", required=True)
    p.add_argument("--motor", default="unitree_g1")
    p.add_argument("--backend", choices=("jax", "torch"), default="jax")
    p.add_argument("--nworld", type=int, default=1024)
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    a = ap.parse_args()
    {"ref": cmd_ref, "check": cmd_check, "selffd": cmd_selffd, "grad64": cmd_grad64, "dump": cmd_dump,
     "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()

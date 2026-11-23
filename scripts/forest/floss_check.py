"""Joint dry friction (MuJoCo's dof frictionloss rows) on the forest engine: forward against MuJoCo Warp and MuJoCo C,
gradients against float64 central differences of MuJoCo C, a hanging hold, per-world frictionloss, cost.

Model: the mjlab G1 (fixtures/mjlab/g1_train.mjb) with frictionloss on its 12 leg dofs (FRIC: a hang-identification
fit, hip pitch / roll / yaw, knee, ankle pitch / roll, N m) and, with --imp stiff (default), the stiff frictionloss
impedance on those dofs (dof_solimp 0.99 / 0.999, dof_solref time constant 0.01 s; MuJoCo's solimpfriction /
solreffriction), so the leg dofs carry their own set and the other dofs MuJoCo's default (step.floss_sets: the per-dof
build). --imp default keeps the model's (0.9 / 0.95, 0.02 s). --pinned: the hanging variant, pelvis at height HANG_Z
(feet clear of the floor) with free-joint armature PIN_ARM on its 6 dofs (an almost fixed base; the hold command also
resets the base pose and velocity after every control step in every simulator).

Commands (one engine library per process):
  fwd      (JAX + MuJoCo Warp) one control step (4 substeps, --iterations) from the corpus mf_s0's G1 states
           (benchmarks/data, --nworld): qvel relative error |dv| / |v| per world against MuJoCo Warp with the same model;
           also the engine against MuJoCo C (float64) and MuJoCo Warp against C. --fric 0: the reference floor
  fd       (JAX + MuJoCo C float64) the VJP of one control step (--substeps) on the pinned G1 against central differences
           (h = --eps) of MuJoCo C (--ref-iters Newton iterations, tolerance 1e-12) for fixed random cotangents of qpos'
           (hinges) and qvel', with respect to qpos (hinges), qvel and ctrl; states: a settled hold (MuJoCo C, 1 s at
           the hold command) with, per world, some leg dofs kicked (velocity, setpoint) so that they slide. Each leg
           dof is classified from MuJoCo C's frictionloss row force over the step's substeps (row_zones): stick
           (quadratic in every substep), slip (linear in every substep), mixed (a transition inside the step). A
           column whose +-h runs change any constraint row's state is invalid (the difference spans a kink) and
           left out. --clamp-kv / --normal: the gradient mode (default the exact mode 1 / 0); --corpus: the free G1 on
           the floor instead, mf_s0's first states (contacts)
  hold     (JAX, MuJoCo C, MuJoCo Warp) the pinned G1 hanging, the leg servos (kp scaled by KP_SCALE) following a
           setpoint sequence of holds, steps and ramps (--steps control steps of 4 x 5 ms) from the same start in every
           simulator; per segment the static joint error q - setpoint of the 12 leg joints at its end, and the
           simulators' differences of it (mrad). MuJoCo C three ways: the model's solver options (5 Newton, 8
           line-search iterations), 50 line-search iterations, converged (100 / 1e-12); MuJoCo Warp and the engine
           with the model's 5 iterations
  params   (JAX) the per-world frictionloss (PhysicsParams fields --fields: dof_frictionloss, frictionloss_solref,
           frictionloss_solimp; floss_values): --mode params, one parameter build with every world's values, or
           --mode plain --world K, a per-model build of world K's values; forward + backward dumps (--dump)
  pcompare a params dump against plain dumps (--plain K=NPZ): world K's outputs and VJPs, bitwise (--all-rows: every
           world and g_params, e.g. a --api torch dump of the same build: --plain 0=TORCH.npz)
  pgrad    a params dump's g_params against central differences of MuJoCo C (float64)
  time     (JAX) forward and forward + backward per control step at --nworld worlds, interleaved rounds: --fric
           (0: the plain G1 build; >0 the frictionloss build), --terrain (the G1 terrain fixture)

  PYTHONPATH=src python scripts/forest/floss_check.py fwd --out runs/floss/fwd.json
  PYTHONPATH=src python scripts/forest/floss_check.py fd --substeps 4 --out runs/floss/fd4.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("DK_ENGINE", "forest")
G1 = REPO / "fixtures" / "mjlab" / "g1_train.mjb"
META = REPO / "fixtures" / "mjlab" / "g1_meta.json"
FRIC = np.array([1.299, 0.933, 0.905, 1.121, 1.495, 0.146] * 2)
LEG = slice(6, 18)
STIFF_SOLIMP, STIFF_TC = (0.99, 0.999), 0.01
KP_SCALE = np.array([0.821, 0.9, 0.9, 0.817, 0.9, 0.9] * 2)
HANG_Z, PIN_ARM = 1.5, 1e4
ITER = 50


def foot_pairs():
    return json.loads(META.read_text())["foot_pairs"]


def model(imp="stiff", fric=1.0, pinned=False):
    import mujoco
    m = mujoco.MjModel.from_binary_path(str(G1))
    m.dof_frictionloss[LEG] = fric * FRIC
    if imp == "stiff":
        m.dof_solimp[LEG, :2] = STIFF_SOLIMP
        m.dof_solref[LEG, 0] = STIFF_TC
    if pinned:
        m.dof_armature[:6] = PIN_ARM
    return m


def hold_pose(m):
    """A hanging pose: pelvis at HANG_Z upright, legs slightly bent (hip pitch -0.3, knee 0.6, ankle pitch -0.3)."""
    q = m.qpos0.copy()
    q[2] = HANG_Z
    for s in (7, 13):
        q[s + 0], q[s + 3], q[s + 4] = -0.3, 0.6, -0.3
    return q


def ref_model(m, iters=100):
    """The MuJoCo C / MuJoCo Warp reference: m with converged Newton (iters, tolerance 1e-12)."""
    import copy
    r = copy.deepcopy(m)
    r.opt.iterations, r.opt.tolerance = iters, 1e-12
    r.opt.ls_iterations = 50
    return r


def row_zones(m, d):
    """Per constraint row of MuJoCo C's last solve: a frictionloss row 1 on its quadratic (stick) zone, 2 / 3 on the
    linear pieces (force +F / -F: |force| = F up to 1e-9 relative), other rows 1 when their force is nonzero (on the
    quadratic zone) else 0. From the forces: efc_state of MuJoCo 3.17's Newton exit is not always the final zone."""
    import mujoco
    n = d.nefc
    t, f = d.efc_type[:n], d.efc_force[:n]
    z = (f != 0).astype(int)
    fl = np.nonzero(t == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)[0]
    F = m.dof_frictionloss[d.efc_id[fl]]
    z[fl] = np.where(np.abs(f[fl]) >= F * (1 - 1e-9), np.where(f[fl] > 0, 2, 3), 1)
    return z


def c_step(m, d, q, v, u, nsub, zones=False):
    """One control step of MuJoCo C from (q, v, u): (q', v', [per substep (efc_type, efc_id, row_zones) of all rows]);
    the rows are those of each substep's mj_step (its constraint pass)."""
    import mujoco
    d.qpos[:], d.qvel[:], d.ctrl[:] = q, v, u
    d.qacc_warmstart[:] = 0.0
    Z = []
    for _ in range(nsub):
        mujoco.mj_step(m, d)
        if zones:
            n = d.nefc
            Z.append((d.efc_type[:n].copy(), d.efc_id[:n].copy(), row_zones(m, d)))
    return d.qpos.copy(), d.qvel.copy(), Z


def _rel_rows(u, v):
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    return np.linalg.norm(u - v, axis=1) / np.maximum(np.linalg.norm(v, axis=1), 1e-6)


def _stats(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return None
    return dict(n=int(x.size), median=float(np.median(x)), p90=float(np.quantile(x, 0.9)),
                p99=float(np.quantile(x, 0.99)), max=float(x.max()))


def engine_init(m, params=None):
    from nablaengine import jax_ffi
    kw = {"params": params} if params else {}
    return jax_ffi.init(model=m, foot_pairs=foot_pairs(), **kw)


def j32(x):
    import jax.numpy as jnp
    return jnp.asarray(np.ascontiguousarray(x, np.float32))


def mjw_step(m, qpos, qvel, ctrl, nsub, warm=None, chunk=512):
    """One control step of MuJoCo Warp (float32) per world, in chunks of `chunk` worlds."""
    if len(qpos) > chunk:
        outs = [mjw_step(m, qpos[i:i + chunk], qvel[i:i + chunk], ctrl[i:i + chunk], nsub,
                         None if warm is None else warm[i:i + chunk], chunk) for i in range(0, len(qpos), chunk)]
        return tuple(np.concatenate([o[k] for o in outs]) for k in range(3))
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    B = len(qpos)
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=64, njmax=256)
    wp.copy(dw.qpos, wp.array(np.ascontiguousarray(qpos, np.float32)))
    wp.copy(dw.qvel, wp.array(np.ascontiguousarray(qvel, np.float32)))
    wp.copy(dw.ctrl, wp.array(np.ascontiguousarray(ctrl, np.float32)))
    if warm is not None:
        wp.copy(dw.qacc_warmstart, wp.array(np.ascontiguousarray(warm, np.float32)))
    for _ in range(nsub):
        mjw.step(mw, dw)
    return dw.qpos.numpy(), dw.qvel.numpy(), dw.qacc_warmstart.numpy()


def cmd_fwd(a):
    from nablaengine import jax_ffi
    import mujoco
    z = np.load(REPO / "benchmarks" / "data" / "mf_s0.npz")
    B = min(a.nworld, len(z["qpos"]))
    m = model(a.imp, a.fric)
    m.opt.iterations = a.iterations
    q0, v0, w0, u0 = (z[k][:B] for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"))
    fr = np.full(B, float(m.pair_friction[foot_pairs()[0], 0]), np.float32)
    info = engine_init(m)
    com = np.zeros((B, info["com_words"]), np.float32)
    r = jax_ffi.forward(j32(q0), j32(v0), j32(w0), j32(u0), j32(fr), j32(com), iterations=a.iterations, substeps=4)
    eq, ev = np.asarray(r[0]), np.asarray(r[1])
    wq, wv, _ = mjw_step(m, q0, v0, u0, 4, w0)
    rm = ref_model(m, 200)
    d = mujoco.MjData(rm)
    nc = min(B, a.c_worlds)
    cq, cv = np.zeros((nc, m.nq)), np.zeros((nc, m.nv))
    stick = slip = 0
    for b in range(nc):
        cq[b], cv[b], Z = c_step(rm, d, q0[b], v0[b], u0[b], 4, zones=True)
        for t, i, s in Z:
            fs = s[t == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF]
            stick += int((fs == 1).sum())
            slip += int((fs != 1).sum())
    rel = _rel_rows(ev, wv)
    worst = [int(b) for b in np.argsort(rel)[-5:][::-1]]
    wc = []
    for b in worst:
        qc, vc, _ = c_step(rm, d, q0[b], v0[b], u0[b], 4)
        wc.append(dict(world=b, engine_vs_mjw=float(rel[b]), engine_vs_c=float(_rel_rows(ev[b:b + 1], vc[None])[0]),
                       mjw_vs_c=float(_rel_rows(wv[b:b + 1], vc[None])[0])))
    res = dict(cmd="fwd", imp=a.imp, fric=a.fric, nworld=B, iterations=a.iterations, engine=info["engine"],
               worst=wc,
               engine_vs_mjw=_stats(_rel_rows(ev, wv)), engine_vs_c=_stats(_rel_rows(ev[:nc], cv)),
               mjw_vs_c=_stats(_rel_rows(wv[:nc], cv)), c_worlds=nc,
               c_floss_rows_quadratic=stick, c_floss_rows_linear=slip,
               qpos_abs_max_vs_mjw=float(np.abs(eq - wq).max()))
    _out(a, res)


def hold_states(m, B, seed, kick=True, settle=1.0):
    """B states of the pinned G1: the hold pose's command held for `settle` s in MuJoCo C (the legs come to rest,
    friction holding them near the edge of their cone), the leg setpoints then moved by the settled friction force
    / kp (the servos take over the friction's share: the joints sit inside the stick zone), then per world a random
    subset of leg dofs kicked: qvel U(+-1.5) rad/s and setpoint +-U(0.05, 0.3) rad (they slide), the others kept
    (they stick) with a setpoint jitter of up to 0.3 frictionloss / kp; the arms' setpoints U(+-0.05) rad."""
    import mujoco
    rng = np.random.default_rng(seed)
    q = hold_pose(m)
    u = q[7:].copy()
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.ctrl[:] = u
    for _ in range(int(settle / m.opt.timestep)):
        mujoco.mj_step(m, d)
        d.qpos[:7], d.qvel[:6] = q[:7], 0.0
    qs, vs = d.qpos.copy(), d.qvel.copy()
    mujoco.mj_forward(m, d)
    kp = m.actuator_gainprm[:12, 0]
    u[:12] -= d.qfrc_constraint[6:18] / kp
    jit = 0.3 * m.dof_frictionloss[LEG] / kp
    Q, V, U, K = np.tile(qs, (B, 1)), np.tile(vs, (B, 1)), np.tile(u, (B, 1)), np.zeros((B, 12), bool)
    for b in range(B):
        k = rng.random(12) < (0.4 if kick else 0.0)
        K[b] = k
        dv = rng.uniform(-1.5, 1.5, 12)
        du = rng.uniform(0.05, 0.3, 12) * rng.choice([-1, 1], 12)
        V[b, 6:18] = np.where(k, dv, V[b, 6:18])
        U[b, :12] += np.where(k, du, rng.uniform(-1.0, 1.0, 12) * jit)
        U[b, 12:] += rng.uniform(-0.05, 0.05, m.nu - 12)
    return Q, V, U, K


def classify(Z, nleg=12):
    """Per leg dof: 0 stick (quadratic every substep), 1 slip (linear every substep), 2 mixed; from c_step's zones."""
    import mujoco
    Q = 1
    st = np.zeros((len(Z), nleg), int)
    for k, (t, i, s) in enumerate(Z):
        sel = t == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
        for dof, ss in zip(i[sel], s[sel]):
            if 6 <= dof < 6 + nleg:
                st[k, dof - 6] = 0 if ss == Q else 1
    out = np.where(st.min(0) == st.max(0), st[0], 2)
    return out


def _zsig(Z):
    return tuple((tuple(t), tuple(i), tuple(s)) for t, i, s in Z)


def cmd_fd(a):
    import jax.numpy as jnp
    import mujoco
    from nablaengine import jax_ffi
    m = model(a.imp, a.fric, pinned=not a.corpus)
    rm = ref_model(m, a.ref_iters)
    B = a.nworld
    if a.corpus:
        z = np.load(REPO / "benchmarks" / "data" / "mf_s0.npz")
        Q, V, U = (z[k][:B].astype(np.float64) for k in ("qpos", "qvel", "ctrl"))
    else:
        Q, V, U, K = hold_states(rm, B, a.seed)
    rng = np.random.default_rng(a.seed + 1)
    hq = [int(m.jnt_qposadr[j]) for j in range(1, m.njnt)]
    yq = np.zeros((B, m.nq))
    yq[:, hq] = rng.standard_normal((B, len(hq)))
    yv = rng.standard_normal((B, m.nv))
    d = mujoco.MjData(rm)
    fd = dict(qpos=np.zeros((B, len(hq))), qvel=np.zeros((B, m.nv)), ctrl=np.zeros((B, m.nu)))
    valid = dict(qpos=np.ones((B, len(hq)), bool), qvel=np.ones((B, m.nv), bool), ctrl=np.ones((B, m.nu), bool))
    cls = np.zeros((B, 12), int)
    h = a.eps
    t0 = time.time()
    for b in range(B):
        _, _, Z0 = c_step(rm, d, Q[b], V[b], U[b], a.substeps, zones=True)
        s0 = _zsig(Z0)
        cls[b] = classify(Z0)
        cols = [("qpos", i, ("q", hq[i])) for i in range(len(hq))] + [("qvel", i, ("v", i)) for i in range(m.nv)] + \
               [("ctrl", i, ("u", i)) for i in range(m.nu)]
        for blk, i, (kind, idx) in cols:
            L = []
            for sg in (1.0, -1.0):
                q, v, u = Q[b].copy(), V[b].copy(), U[b].copy()
                {"q": q, "v": v, "u": u}[kind][idx] += sg * h
                q1, v1, Z = c_step(rm, d, q, v, u, a.substeps, zones=True)
                L.append(float(yq[b] @ q1 + yv[b] @ v1))
                if _zsig(Z) != s0:
                    valid[blk][b, i] = False
            fd[blk][b, i] = (L[0] - L[1]) / (2 * h)
    t_fd = time.time() - t0
    info = engine_init(m)
    com = np.zeros((B, info["com_words"]), np.float32)
    fr = np.full(B, float(m.pair_friction[foot_pairs()[0], 0]), np.float32)
    r = jax_ffi.forward(j32(Q), j32(V), j32(np.zeros((B, m.nv))), j32(U), j32(fr), j32(com), iterations=a.iterations,
                        substeps=a.substeps, tape=True)
    g = jax_ffi.backward(r[5] if len(r) > 5 else r[4], j32(fr), j32(com), j32(yq), j32(yv), substeps=a.substeps,
                         clamp_kv=a.clamp_kv, normal=a.normal)
    eng = dict(qpos=np.asarray(g[0])[:, hq], qvel=np.asarray(g[1]), ctrl=np.asarray(g[2]))
    res = dict(cmd="fd", corpus=a.corpus, imp=a.imp, fric=a.fric, substeps=a.substeps, nworld=B, eps=h,
               iterations=a.iterations,
               ref_iters=a.ref_iters, clamp_kv=a.clamp_kv, normal=a.normal, engine=info["engine"],
               adj_flags=int((np.asarray(g[3]) != 0).sum()), fd_seconds=round(t_fd, 1),
               leg_dofs=dict(stick=int((cls == 0).sum()), slip=int((cls == 1).sum()), mixed=int((cls == 2).sum())),
               worlds=dict(with_stick=int((cls == 0).any(1).sum()), with_slip=int((cls == 1).any(1).sum()),
                           with_mixed=int((cls == 2).any(1).sum())))
    worlds_ok = ~(cls == 2).any(1)
    for blk in ("qpos", "qvel", "ctrl"):
        e = np.abs(eng[blk] - fd[blk]) / (np.abs(fd[blk]) + 1e-3)
        ok = valid[blk]
        res[blk] = dict(invalid_columns=int((~ok).sum()), elem_valid=_stats(e[ok]),
                        elem_valid_no_mixed=_stats(e[ok & worlds_ok[:, None]]),
                        elem_mixed_worlds=_stats(e[ok & ~worlds_ok[:, None]]),
                        world_rel=_stats([np.linalg.norm((eng[blk][b] - fd[blk][b])[ok[b]])
                                          / max(np.linalg.norm(fd[blk][b][ok[b]]), 1e-9) for b in range(B)]))
    for name, c in (("stick", 0), ("slip", 1), ("mixed", 2)):
        for blk, off in (("qvel", 6), ("ctrl", 0)):
            sel = (cls == c) & valid[blk][:, off:off + 12]
            e = np.abs(eng[blk][:, off:off + 12] - fd[blk][:, off:off + 12]) / (np.abs(fd[blk][:, off:off + 12]) + 1e-3)
            res[f"leg_{name}_{blk}"] = _stats(e[sel])
    if a.dump:
        np.savez(a.dump, Q=Q, V=V, U=U, cls=cls, yq=yq, yv=yv, **{f"fd_{k}": v for k, v in fd.items()},
                 **{f"eng_{k}": v for k, v in eng.items()}, **{f"valid_{k}": v for k, v in valid.items()})
    _out(a, res)


def setpoints(m, steps, seed=0):
    """(steps, nu) leg setpoint sequence from the hold pose: per leg joint a piecewise sequence of holds (0.4 s) and
    steps / ramps of +-U(0.05, 0.25) rad, the other joints at the pose; and the hold windows' last step indices."""
    rng = np.random.default_rng(seed)
    base = hold_pose(m)[7:]
    U = np.tile(base, (steps, 1))
    seg = 20
    cur = base[:12].copy()
    ends = []
    for s0 in range(0, steps, seg):
        kind = (s0 // seg) % 3
        tgt = base[:12] + rng.uniform(0.05, 0.25, 12) * rng.choice([-1, 1], 12) if kind else cur
        for k in range(s0, min(s0 + seg, steps)):
            if kind == 2:
                U[k, :12] = cur + (tgt - cur) * min(1.0, (k - s0 + 1) / (seg // 2))
            else:
                U[k, :12] = tgt
        cur = U[min(s0 + seg, steps) - 1, :12].copy()
        ends.append(min(s0 + seg, steps) - 1)
    return U, ends


def hold_model(imp, fric):
    m = model(imp, fric, pinned=True)
    for i in range(12):
        m.actuator_gainprm[i, 0] *= KP_SCALE[i]
        m.actuator_biasprm[i, 1] *= KP_SCALE[i]
    return m


def cmd_hold(a):
    import mujoco
    m = hold_model(a.imp, a.fric)
    U, ends = setpoints(m, a.steps, a.seed)
    q0 = hold_pose(m)
    nsub = 4
    res = dict(cmd="hold", imp=a.imp, fric=a.fric, steps=a.steps, windows=len(ends))
    traj = {}
    import copy
    m_ls = copy.deepcopy(m)
    m_ls.opt.ls_iterations = 50
    for name, mm in (("c", m), ("c_ls50", m_ls), ("c_conv", ref_model(m, 100))):
        d = mujoco.MjData(mm)
        d.qpos[:] = q0
        X = np.zeros((a.steps, 12))
        for k in range(a.steps):
            d.ctrl[:] = U[k]
            for _ in range(nsub):
                mujoco.mj_step(mm, d)
            d.qpos[:7], d.qvel[:6] = q0[:7], 0.0
            X[k] = d.qpos[7:19]
        traj[name] = X
    import warp as wp
    import mujoco_warp as mjw
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=1, nconmax=64, njmax=256)
    qq = q0.copy()[None].astype(np.float32)
    wp.copy(dw.qpos, wp.array(qq))
    X = np.zeros((a.steps, 12))
    for k in range(a.steps):
        wp.copy(dw.ctrl, wp.array(U[k][None].astype(np.float32)))
        for _ in range(nsub):
            mjw.step(mw, dw)
        qn, vn = dw.qpos.numpy(), dw.qvel.numpy()
        qn[0, :7], vn[0, :6] = q0[:7], 0.0
        wp.copy(dw.qpos, wp.array(qn))
        wp.copy(dw.qvel, wp.array(vn))
        X[k] = qn[0, 7:19]
    traj["mjw"] = X
    from nablaengine import jax_ffi
    info = engine_init(m)
    com = j32(np.zeros((1, info["com_words"])))
    fr = j32(np.full(1, float(m.pair_friction[foot_pairs()[0], 0])))
    q, v, w = j32(q0[None]), j32(np.zeros((1, m.nv))), j32(np.zeros((1, m.nv)))
    X = np.zeros((a.steps, 12))
    for k in range(a.steps):
        r = jax_ffi.forward(q, v, w, j32(U[k][None]), fr, com, iterations=int(m.opt.iterations), substeps=nsub)
        qn, vn = np.asarray(r[0]).copy(), np.asarray(r[1]).copy()
        qn[0, :7], vn[0, :6] = q0[:7], 0.0
        q, v, w = j32(qn), j32(vn), r[2]
        X[k] = qn[0, 7:19]
    traj["engine"] = X
    E = {k: (traj[k][ends] - U[ends, :12]) * 1e3 for k in traj}
    res["static_error_rms_mrad"] = {k: float(np.sqrt((E[k] ** 2).mean())) for k in E}
    for x, y in (("engine", "c"), ("engine", "c_ls50"), ("engine", "c_conv"), ("engine", "mjw"), ("mjw", "c_conv"),
                 ("c", "c_conv"), ("c_ls50", "c_conv")):
        dd = np.abs(E[x] - E[y])
        res[f"static_error_diff_{x}_vs_{y}_mrad"] = dict(median=float(np.median(dd)), max=float(dd.max()),
                                                         rms=float(np.sqrt((dd ** 2).mean())))
        tt = np.abs(traj[x] - traj[y]) * 1e3
        res[f"traj_diff_{x}_vs_{y}_mrad"] = dict(median=float(np.median(tt)), max=float(tt.max()))
    if a.dump:
        np.savez(a.dump, U=U, ends=np.array(ends), **traj)
    _out(a, res)


def floss_values(B, seed):
    """Per-world leg frictionloss: the DR ranges hips 0.8-1.5, knee 0.7-1.2, ankle pitch 0.4-1.0, ankle roll
    0.15-0.5 N m; world 0 takes 0 on every leg dof (no force; the rows stay). And per world a frictionloss impedance:
    time constant U(0.008, 0.03) s (damping ratio 1), dmin U(0.9, 0.995), dmax dmin + U(0.0005, 0.004), width and
    midpoint the model's (0.001, 0.5)."""
    lo = np.array([0.8, 0.8, 0.8, 0.7, 0.4, 0.15] * 2)
    hi = np.array([1.5, 1.5, 1.5, 1.2, 1.0, 0.5] * 2)
    rng = np.random.default_rng(seed)
    F = rng.uniform(lo, hi, (B, 12))
    F[0] = 0.0
    sr = np.stack([rng.uniform(0.008, 0.03, B), np.ones(B)], 1)
    dmin = rng.uniform(0.9, 0.995, B)
    si = np.stack([dmin, np.minimum(dmin + rng.uniform(0.0005, 0.004, B), 0.9995), np.full(B, 0.001),
                   np.full(B, 0.5)], 1)
    return F, sr, si


def _world_model(m, fields, F, sr, si, k):
    if "dof_frictionloss" in fields:
        m.dof_frictionloss[LEG] = F[k]
    if "frictionloss_solref" in fields:
        m.dof_solref[LEG] = sr[k]
    if "frictionloss_solimp" in fields:
        m.dof_solimp[LEG, :4] = si[k]
    return m


def _torch_info(m, P):
    from nablaengine import torch_ffi
    return torch_ffi.init(model=m, foot_pairs=foot_pairs(), **({"params": P} if P else {}))


def cmd_params(a):
    if a.api == "jax":
        from nablaengine import jax_ffi
    from nablaengine.params import PhysicsParams
    fields = tuple(a.fields)
    m = model(a.imp, 1.0, pinned=a.pinned)
    B = a.nworld
    F, sr, si = floss_values(B, a.seed)
    if a.pinned:
        Q, V, U, _ = hold_states(ref_model(m), B, a.seed)
    else:
        z = np.load(REPO / "benchmarks" / "data" / "mf_s0.npz")
        Q, V, U = z["qpos"][:B], z["qvel"][:B], z["ctrl"][:B]
    rng = np.random.default_rng(a.seed + 2)
    yq, yv = rng.standard_normal((B, m.nq)), rng.standard_normal((B, m.nv))
    if a.mode == "params":
        P = PhysicsParams(fields)
        info = engine_init(m, P) if a.api == "jax" else _torch_info(m, P)
        vals = P.defaults(m, foot_pairs())
        if "dof_frictionloss" in fields:
            fl = np.tile(vals["dof_frictionloss"], (B, 1))
            fl[:, LEG] = F
            vals["dof_frictionloss"] = fl
        if "frictionloss_solref" in fields:
            vals["frictionloss_solref"] = sr
        if "frictionloss_solimp" in fields:
            vals["frictionloss_solimp"] = si
        pw = P.pack(m.nv, **vals)
        cw0 = int(info["com_words"]) - pw.shape[1]
        com = np.concatenate([np.zeros((B, cw0), np.float32), pw], 1)
        lay = {f: list(v) for f, v in P.layout(m.nv).items()}
    else:
        m = _world_model(m, fields, F, sr, si, a.world)
        info = engine_init(m) if a.api == "jax" else _torch_info(m, None)
        com = np.zeros((B, info["com_words"]), np.float32)
        lay = {}
    if a.api == "torch":
        import torch
        from nablaengine import torch_ffi
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x, np.float32)).cuda()
        tf = t(np.full(B, float(m.pair_friction[foot_pairs()[0], 0])))
        rt = torch_ffi.forward(t(Q), t(V), t(np.zeros((B, m.nv))), t(U), tf, t(com), iterations=ITER,
                               substeps=a.substeps, tape=True)
        gt = torch_ffi.backward(rt[4], tf, t(com), t(yq), t(yv), substeps=a.substeps, clamp_kv=1, normal=0)
        r = [x.cpu().numpy() for x in rt[:4]]
        g = [x.cpu().numpy() for x in gt]
    else:
        fr = j32(np.full(B, float(m.pair_friction[foot_pairs()[0], 0])))
        r = jax_ffi.forward(j32(Q), j32(V), j32(np.zeros((B, m.nv))), j32(U), fr, j32(com), iterations=ITER,
                            substeps=a.substeps, tape=True)
        g = jax_ffi.backward(r[4], fr, j32(com), j32(yq), j32(yv), substeps=a.substeps, clamp_kv=1, normal=0)
    out = dict(F=F, sr=sr, si=si, q1=np.asarray(r[0]), v1=np.asarray(r[1]), info=np.asarray(r[3]),
               g_qpos=np.asarray(g[0]), g_qvel=np.asarray(g[1]), g_ctrl=np.asarray(g[2]), g_info=np.asarray(g[3]),
               fields=np.array(json.dumps(fields)), layout=np.array(json.dumps(lay)))
    if len(g) > 4:
        out["g_params"] = np.asarray(g[4])
    np.savez(a.dump, **out)
    print(json.dumps(dict(mode=a.mode, fields=fields, nworld=B, engine=info["engine"], dump=a.dump)))


def cmd_pcompare(a):
    """--dump: the params dump; --plain K=NPZ: per-model dumps (world K's row compared)."""
    x = np.load(a.dump)
    res = {}
    for kv in a.plain:
        k, p = kv.split("=")
        k = int(k)
        y = np.load(p)
        r = {}
        keys = ("q1", "v1", "g_qpos", "g_qvel", "g_ctrl") + (("g_params",) if a.all_rows else ())
        for key in keys:
            u, v = (x[key], y[key]) if a.all_rows else (x[key][k], y[key][k])
            r[key] = dict(bitwise=bool(np.array_equal(u, v)), max_abs=float(np.abs(u - v).max()))
        res[k] = r
    _out(a, res)


def cmd_pgrad(a):
    """g_params of a params dump (its fields: the leg dofs' dof_frictionloss, the time constant of
    frictionloss_solref, dmin and dmax of frictionloss_solimp) against central differences of MuJoCo C float64
    (--nworld worlds of the dump, h = --eps; a column whose +-h runs change any row's zone is left out, and the
    frictionloss columns of a world whose value is below h)."""
    import mujoco
    x = np.load(a.dump)
    fields, lay = json.loads(str(x["fields"])), json.loads(str(x["layout"]))
    m = model(a.imp, 1.0, pinned=a.pinned)
    rm = ref_model(m, a.ref_iters)
    F, sr, si = x["F"], x["sr"], x["si"]
    B = min(a.nworld, len(F))
    if a.pinned:
        Q, V, U, _ = hold_states(ref_model(m), len(F), a.seed)
    else:
        z = np.load(REPO / "benchmarks" / "data" / "mf_s0.npz")
        Q, V, U = z["qpos"], z["qvel"], z["ctrl"]
    rng = np.random.default_rng(a.seed + 2)
    yq, yv = rng.standard_normal((len(F), m.nq)), rng.standard_normal((len(F), m.nv))
    d = mujoco.MjData(rm)
    gp = x["g_params"]
    cols = []
    if "dof_frictionloss" in fields:
        o = lay["dof_frictionloss"][0]
        for j in range(12):
            cols.append((f"floss{j}", o + 6 + j, lambda r, b, dl, j=j: r.dof_frictionloss.__setitem__(
                6 + j, max(F[b, j] + dl, 0.0))))
    if "frictionloss_solref" in fields:
        o = lay["frictionloss_solref"][0]
        cols.append(("tc", o, lambda r, b, dl: r.dof_solref.__setitem__((LEG, 0), sr[b, 0] + dl)))
    if "frictionloss_solimp" in fields:
        o = lay["frictionloss_solimp"][0]
        cols.append(("dmin", o, lambda r, b, dl: r.dof_solimp.__setitem__((LEG, 0), si[b, 0] + dl)))
        cols.append(("dmax", o + 1, lambda r, b, dl: r.dof_solimp.__setitem__((LEG, 1), si[b, 1] + dl)))
    fdv, val = np.zeros((B, len(cols))), np.ones((B, len(cols)), bool)
    eng = np.stack([gp[:B, c] for _, c, _ in cols], 1)
    for b in range(B):
        for ci, (name, _, edit) in enumerate(cols):
            L, sig = [], []
            for sg in (1.0, -1.0):
                _world_model(rm, fields, F, sr, si, b)
                edit(rm, b, sg * a.eps)
                q1, v1, Z = c_step(rm, d, Q[b], V[b], U[b], a.substeps, zones=True)
                L.append(float(yq[b] @ q1 + yv[b] @ v1))
                sig.append(tuple(tuple(s) for _, _, s in Z))
            val[b, ci] = sig[0] == sig[1] and not (name.startswith("floss") and F[b, int(name[5:])] <= a.eps)
            fdv[b, ci] = (L[0] - L[1]) / (2 * a.eps)
    e = np.abs(eng - fdv) / (np.abs(fdv) + 1e-3)
    res = dict(cmd="pgrad", fields=fields, nworld=B, eps=a.eps, invalid=int((~val).sum()))
    groups = {"dof_frictionloss": [i for i, c in enumerate(cols) if c[0].startswith("floss")],
              "tc": [i for i, c in enumerate(cols) if c[0] == "tc"], "dmin": [i for i, c in enumerate(cols) if c[0] == "dmin"],
              "dmax": [i for i, c in enumerate(cols) if c[0] == "dmax"]}
    for gname, idx in groups.items():
        if idx:
            res[gname] = _stats(e[:, idx][val[:, idx]])
            res[gname + "_fd_abs_median"] = float(np.median(np.abs(fdv[:, idx])))
    if "dof_frictionloss" in fields:
        res["world0_zero_floss_grad_absmax"] = float(np.abs(gp[0, lay["dof_frictionloss"][0] + 6:
                                                             lay["dof_frictionloss"][0] + 18]).max())
    _out(a, res)


def cmd_time(a):
    """Forward and forward + backward per control step (10 iterations, 4 substeps, ms; the median of --rounds rounds
    of --reps calls each, the two alternating within a round). Flat: the mf_s0 states; --terrain: the G1 terrain
    reference (fixtures/mjlab/mjw317_terrain_feet.npz, its stairs scene, the feet's analytic terrain pairs)."""
    import jax
    import mujoco
    from nablaengine import jax_ffi, terrain as T
    m = model(a.imp, a.fric) if a.fric > 0 else mujoco.MjModel.from_binary_path(str(G1))
    B = a.nworld
    kw = {}
    if a.terrain:
        r = np.load(REPO / "fixtures" / "mjlab" / "mjw317_terrain_feet.npz")
        z = dict(qpos=r["qpos"], qvel=r["qvel"], qacc_warmstart=r["warm"], ctrl=r["ctrl"])
        kw["terrain"] = T.TerrainConfig(geoms=str(r["geoms"]), mode="analytic")
    else:
        z = np.load(REPO / "benchmarks" / "data" / "mf_s0.npz")
    idx = np.arange(B) % len(z["qpos"])
    from nablaengine import jax_ffi as J
    info = J.init(model=m, foot_pairs=foot_pairs(), com_body="torso_link", **kw)
    if a.terrain:
        J.set_terrain(T.Terrain.from_model(mujoco.MjModel.from_binary_path(
            str(REPO / "fixtures" / "mjlab" / "g1_terrain_feet.mjb"))))
    q, v, w, u = (j32(z[k][idx]) for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"))
    fr = j32(np.ones(B))
    com = j32(np.zeros((B, info["com_words"])))
    step = jax_ffi.make_step(iterations=10, substeps=4)
    fwd = jax.jit(lambda q, v, w, u: step(q, v, w, u, fr, com))

    def loss(q, v, u):
        a_, b_, _ = step(q, v, w, u, fr, com)
        return a_.sum() + b_.sum()
    bwd = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    for _ in range(3):
        jax.block_until_ready(fwd(q, v, w, u))
        jax.block_until_ready(bwd(q, v, u))
    T_ = dict(fwd=[], fwdbwd=[])
    for _ in range(a.rounds):
        t0 = time.perf_counter()
        for _ in range(a.reps):
            r_ = fwd(q, v, w, u)
        jax.block_until_ready(r_)
        T_["fwd"].append((time.perf_counter() - t0) / a.reps * 1e3)
        t0 = time.perf_counter()
        for _ in range(a.reps):
            r_ = bwd(q, v, u)
        jax.block_until_ready(r_)
        T_["fwdbwd"].append((time.perf_counter() - t0) / a.reps * 1e3)
    res = dict(cmd="time", fric=a.fric, imp=a.imp, terrain=a.terrain, nworld=B, engine=info["engine"],
               ms={k: float(np.median(x)) for k, x in T_.items()},
               rounds={k: [round(t, 3) for t in x] for k, x in T_.items()})
    _out(a, res)


def _out(a, res):
    s = json.dumps(res, indent=1)
    print(s, flush=True)
    if getattr(a, "out", None):
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("fwd", "fd", "hold", "params", "pcompare", "pgrad", "time"))
    ap.add_argument("--imp", default="stiff", choices=("stiff", "default"))
    ap.add_argument("--fric", type=float, default=1.0, help="scale of FRIC on the leg dofs (0: none)")
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--c-worlds", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=ITER)
    ap.add_argument("--ref-iters", type=int, default=100)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--clamp-kv", type=int, default=1)
    ap.add_argument("--normal", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--mode", default="params", choices=("params", "plain"))
    ap.add_argument("--world", type=int, default=0)
    ap.add_argument("--pinned", action="store_true")
    ap.add_argument("--corpus", action="store_true", help="fd: the free G1's mf_s0 states (floor contacts)")
    ap.add_argument("--fields", nargs="+", default=["dof_frictionloss"])
    ap.add_argument("--api", default="jax", choices=("jax", "torch"))
    ap.add_argument("--all-rows", action="store_true", help="pcompare: every world and g_params (e.g. JAX vs torch)")
    ap.add_argument("--plain", nargs="*", default=[])
    ap.add_argument("--terrain", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--dump")
    ap.add_argument("--out")
    a = ap.parse_args()
    {"fwd": cmd_fwd, "fd": cmd_fd, "hold": cmd_hold, "params": cmd_params, "pcompare": cmd_pcompare,
     "pgrad": cmd_pgrad, "time": cmd_time}[a.cmd](a)


if __name__ == "__main__":
    main()

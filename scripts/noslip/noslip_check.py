"""Checks of the forest engine's noslip pass (nablaengine.forest_engine.noslip; a model with opt.noslip_iterations > 0)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GRIP_XML = HERE / "models" / "noslip_grip.xml"
ARM_XML = HERE.parent / "collision" / "models" / "midphase_arm_box.xml"
ARM_LIFT = (0.6304, 1.7424, 0.7688)
try:
    sys.path.insert(0, str(HERE.parent / "internal"))
    import noslip_scenes as INTERNAL
except ImportError:
    INTERNAL = None


def provenance():
    import subprocess
    import mujoco
    from nablaengine import provenance as P
    out = dict(script="scripts/noslip/noslip_check.py", mujoco=mujoco.__version__,
               reference="MuJoCo C float64 (the noslip exception: MuJoCo Warp has no noslip solver)")
    try:
        out["commit"], out["dirty"] = P.source_commit()
    except Exception:
        pass
    try:
        out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i",
                                     "0"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return out


def pairs_spec(a):
    v = getattr(a, "pairs", None)
    if v is None or v == "objects":
        return v
    return [int(x) if x.strip().isdigit() else x.strip() for x in v.split(",")]


def scene(name, noslip, a):
    """(MjModel, states dict(qpos, qvel, ctrl, warm) float64) of a scene; --pairs set up for it (set_pairs)."""
    m, st = _scene(name, noslip, a)
    set_pairs(m, pairs_spec(a))
    return m, st


def _scene(name, noslip, a):
    import mujoco
    from nablaengine import engines, fixtures
    if name == "grip":
        m = grip_model(noslip)
        sn = getattr(a, "state_noslip", None)
        return m, grip_states(m if sn is None else grip_model(sn), a.n, seed=a.seed)
    if name == "g1":
        m, _ = fixtures.load_model(check=None)
        m.opt.noslip_iterations = noslip
        engines.prefer_forest(m)
        return m, g1_states(m, a.n, a.seed)
    if name == "arm":
        m, st = arm_hold(noslip)
        return m, hold_states(m, st, a.n, a.seed)
    r = INTERNAL.scene(name, noslip, a) if INTERNAL is not None else None
    if r is None:
        raise SystemExit(f"--scene {name}: internal scene, not in this release")
    return r


def grip_model(noslip):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(GRIP_XML))
    m.opt.noslip_iterations = int(noslip)
    return m


def grip_states(m, n, seed=0, settle=150):
    """n start states of the grip scene (float64): MuJoCo C settles `settle` steps from the fingers at their origin
    with a squeeze target +-U(0.02, 0.05) m (contacts established)."""
    import mujoco
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    out = {k: [] for k in ("qpos", "qvel", "ctrl", "warm")}
    for _ in range(n):
        mujoco.mj_resetData(m, d)
        sq = rng.uniform(0.02, 0.05)
        d.ctrl[:] = [sq, -sq]
        for _ in range(settle):
            mujoco.mj_step(m, d)
        for k, v in (("qpos", d.qpos), ("qvel", d.qvel), ("ctrl", d.ctrl), ("warm", d.qacc_warmstart)):
            out[k].append(v.copy())
    return {k: np.array(v) for k, v in out.items()}


def arm_hold(noslip):
    """(MjModel with noslip_iterations = noslip, reference state dict(qpos, qvel, ctrl) float64) of the arm scene
    holding its box: MuJoCo C without noslip from keyframe home closes the fingers (0.5 s), lifts the box (the arm's
    targets moved linearly to ARM_LIFT over 1 s) and settles (0.5 s)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(ARM_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    dt = float(m.opt.timestep)
    d.ctrl[7:9] = 0.0
    for _ in range(int(round(0.5 / dt))):
        mujoco.mj_step(m, d)
    c0, n = d.ctrl[[1, 3, 5]].copy(), int(round(1.0 / dt))
    for k in range(n):
        d.ctrl[[1, 3, 5]] = c0 + (np.asarray(ARM_LIFT) - c0) * (k + 1) / n
        mujoco.mj_step(m, d)
    for _ in range(int(round(0.5 / dt))):
        mujoco.mj_step(m, d)
    assert d.qpos[11] > 0.1, "arm_hold: the box was not lifted"
    m.opt.noslip_iterations = int(noslip)
    return m, dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy())


def g1_states(m, n, seed, settle=100):
    """Standing states: MuJoCo C settles from keyframe 0 with PD targets at the keyframe's joints, a random push and
    target offsets (feet on the floor, sticking)."""
    import mujoco
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    out = {k: [] for k in ("qpos", "qvel", "ctrl", "warm")}
    for _ in range(n):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.ctrl[:] = d.qpos[7:7 + m.nu] + rng.normal(0, 0.02, m.nu) if m.nu == m.nq - 7 else d.ctrl
        d.qvel[:6] = rng.normal(0, 0.05, 6)
        for _ in range(settle):
            mujoco.mj_step(m, d)
        for k, v in (("qpos", d.qpos), ("qvel", d.qvel), ("ctrl", d.ctrl), ("warm", d.qacc_warmstart)):
            out[k].append(v.copy())
    return {k: np.array(v) for k, v in out.items()}


def hold_states(m, st, n, seed):
    """States along MuJoCo C's own rollout of a hold from its reference state (one every 25 steps), the first the
    reference state itself."""
    import mujoco
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:] = st["qpos"], st["qvel"], st["ctrl"]
    out = {k: [] for k in ("qpos", "qvel", "ctrl", "warm")}
    for i in range(n):
        for k, v in (("qpos", d.qpos), ("qvel", d.qvel), ("ctrl", d.ctrl), ("warm", d.qacc_warmstart)):
            out[k].append(v.copy())
        for _ in range(25):
            mujoco.mj_step(m, d)
    return {k: np.array(v) for k, v in out.items()}


def f32(x):
    import jax.numpy as jnp
    return jnp.asarray(np.ascontiguousarray(x, np.float32))


PAIRS = dict(spec=None, idx=None, ref_iters=None)


def set_pairs(m, spec):
    """--pairs (None: MuJoCo's noslip, MuJoCo C the reference; else noslip_pairs, the float64 noslip_ref the
    reference: MuJoCo has no restricted noslip)."""
    from nablaengine.forest_engine import noslip
    PAIRS["spec"] = spec
    if spec is not None:
        idx = noslip.resolve_pairs(m, spec)
        PAIRS["idx"] = set(range(m.npair)) if idx is None else set(idx)


def ref_step(m, d):
    """One step of the reference: mj_step, or with --pairs noslip_ref.step on the selected pairs."""
    import mujoco
    if PAIRS["ref_iters"]:
        import noslip_ref
        noslip_ref.step(m, d, PAIRS["ref_iters"], 0.0, PAIRS["idx"] if PAIRS["spec"] is not None else None)
    elif PAIRS["spec"] is None:
        mujoco.mj_step(m, d)
    else:
        import noslip_ref
        noslip_ref.step(m, d, int(m.opt.noslip_iterations), float(m.opt.noslip_tolerance), PAIRS["idx"])


def ref_forward(m, d, iterations=None):
    """mj_forward with the reference's noslip (iterations: noslip_iterations' override)."""
    import mujoco
    n = int(m.opt.noslip_iterations) if iterations is None else iterations
    if PAIRS["ref_iters"]:
        n = PAIRS["ref_iters"] if iterations is None else iterations
    if PAIRS["spec"] is None and not PAIRS["ref_iters"]:
        m0 = m.opt.noslip_iterations
        m.opt.noslip_iterations = n
        mujoco.mj_forward(m, d)
        m.opt.noslip_iterations = m0
        return
    import noslip_ref
    m0 = m.opt.noslip_iterations
    m.opt.noslip_iterations = 0
    mujoco.mj_forward(m, d)
    m.opt.noslip_iterations = m0
    qa, f, _ = noslip_ref.noslip(m, d, n, 0.0 if PAIRS["ref_iters"] else float(m.opt.noslip_tolerance),
                                 PAIRS["idx"] if PAIRS["spec"] is not None else None)
    d.qacc[:] = qa
    d.efc_force[:d.nefc] = f


def engine(m):
    from nablaengine import jax_ffi
    return jax_ffi, jax_ffi.init(model=m, foot_pairs=[], **({"noslip_pairs": PAIRS["spec"]} if PAIRS["spec"] is not None
                                                              else {}))


def c_step(m, st, i, nsub=1):
    import mujoco
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = st["qpos"][i], st["qvel"][i], st["ctrl"][i], st["warm"][i]
    for _ in range(nsub):
        ref_step(m, d)
    return d


def q(x, p):
    x = np.asarray(x, float)
    return float(np.quantile(x, p)) if len(x) else None


def cmd_fwd(a):
    """One substep: the engine (float32, the model's noslip setting) against float64 MuJoCo C (the same model)."""
    import mujoco
    m, st = scene(a.scene, a.noslip, a)
    if a.ns_tol is not None:
        m.opt.noslip_tolerance = a.ns_tol
    J, info = engine(m)
    B = len(st["qpos"])
    r = J.forward(f32(st["qpos"]), f32(st["qvel"]), f32(st["warm"]), f32(st["ctrl"]), f32(np.ones(B)),
                  f32(np.zeros((B, info["com_words"]))), iterations=int(m.opt.iterations), substeps=1, tape=True)
    v = np.asarray(r[1], np.float64)
    ds = [c_step(m, st, i) for i in range(B)]
    ref = np.array([d.qvel for d in ds])
    same = np.asarray(r[3])[:, 1] == np.array([d.ncon for d in ds])
    vfloor = a.vfloor
    rel = np.linalg.norm(v - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), vfloor)
    rs = rel[same]
    res = dict(section="fwd", scene=a.scene, noslip_iterations=int(m.opt.noslip_iterations),
               noslip_tolerance=float(m.opt.noslip_tolerance), iterations=int(m.opt.iterations), nworld=B,
               vfloor=vfloor, rel_median=q(rel, 0.5), rel_p90=q(rel, 0.9), rel_max=float(rel.max()),
               same_ncon=int(same.sum()), same_rel_median=q(rs, 0.5), same_rel_p90=q(rs, 0.9),
               same_rel_max=float(rs.max()) if len(rs) else None,
               info_unsupported=int(np.asarray(r[3])[:, 3].max()), provenance=provenance())
    if a.noslip:
        from nablaengine.forest_engine import tape as T
        dec = T.decode(np.asarray(r[-1]), m)
        res["ns_sweeps"] = np.bincount(dec["ns_iter"].ravel()).tolist()
        res["ns_flags"] = sorted(set(int(x) for x in dec["ns_flags"].ravel()))
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def slip_metrics(m, qs, vs, ctrl, dt, t0=1.0, hand_geoms=None, obj="box"):
    """Hold metrics on a trajectory (qpos, qvel after each step), the contacts and forces from MuJoCo C's
    mj_forward at each state (the model's options): slip (force-weighted tangential speed of the object's material
    point relative to the hand's at the hand-object contacts, median and p95 over t >= t0), the object's sink relative
    to the hands' bodies (linear fit), drop (5 cm), grip lost (a hand without contact for >= 50 ms)."""
    import mujoco
    d = mujoco.MjData(m)
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)
    og = [g for g in range(m.ngeom) if m.geom_bodyid[g] == ob]
    side = {gid(nm): s for s, names in enumerate(hand_geoms) for nm in names}
    hb = [int(m.geom_bodyid[gid(names[0])]) for names in hand_geoms]
    jb, jh, f6 = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros(6)
    T, BZ, RZ, SL, NC = [], [], [], [], []
    z0 = None
    for k, (qq, vv) in enumerate(zip(qs, vs)):
        d.qpos[:], d.qvel[:], d.ctrl[:] = qq, vv, ctrl
        mujoco.mj_forward(m, d)
        if z0 is None:
            z0 = d.xpos[ob, 2]
        fn, sl, nc = np.zeros(2), np.zeros(2), np.zeros(2, int)
        for i in range(d.ncon):
            c = d.contact[i]
            g = {int(c.geom1), int(c.geom2)}
            if not g & set(og):
                continue
            h = (g - set(og)).pop()
            if h not in side:
                continue
            s = side[h]
            mujoco.mj_contactForce(m, d, i, f6)
            mujoco.mj_jac(m, d, jb, None, c.pos, ob)
            mujoco.mj_jac(m, d, jh, None, c.pos, int(m.geom_bodyid[h]))
            rel = (jb - jh) @ d.qvel
            nrm = c.frame[:3]
            rt = rel - (rel @ nrm) * nrm
            fn[s] += abs(f6[0]); sl[s] += abs(f6[0]) * np.linalg.norm(rt); nc[s] += 1
        T.append((k + 1) * dt)
        BZ.append(d.xpos[ob, 2] - z0)
        RZ.append(d.xpos[ob, 2] - 0.5 * (d.xpos[hb[0], 2] + d.xpos[hb[1], 2]))
        SL.append(sl.sum() / max(fn.sum(), 1e-9))
        NC.append(nc)
    T, BZ, RZ, SL, NC = map(np.asarray, (T, BZ, RZ, SL, NC))
    none = (NC == 0).any(1)
    w = max(1, int(round(0.05 / dt)))
    lost = next((float(T[i]) for i in range(len(T) - w + 1) if none[i:i + w].all()), None)
    drop5 = next((float(t) for t, b in zip(T, BZ) if b < -0.05), None)
    late = T >= t0
    on = late & ~none
    fit = lambda y, msk: float(np.polyfit(T[msk], y[msk], 1)[0]) if msk.sum() > 2 else float("nan")
    res = dict(finite=bool(np.isfinite(qs).all()),
               slip_mm_s=1e3 * float(np.median(SL[on])) if on.any() else None,
               slip_p95_mm_s=1e3 * float(np.percentile(SL[on], 95)) if on.any() else None,
               sink_rel_hands_mm_s=1e3 * fit(RZ, on), sink_world_mm_s=1e3 * fit(BZ, late),
               dz_mm={"1s": 1e3 * float(BZ[np.argmin(np.abs(T - 1.0))]), "end": 1e3 * float(BZ[-1]),
                      "window": 1e3 * float(BZ[-1] - BZ[late][0])},
               grip_lost_s=lost, drop_5cm_s=drop5)
    res["held"] = bool(res["finite"] and lost is None and drop5 is None)
    res["accept"] = bool(res["held"] and res["slip_mm_s"] is not None and res["slip_mm_s"] < 1.0
                         and abs(res["sink_rel_hands_mm_s"]) < 1.0)
    return res


def cmd_hold(a):
    import mujoco
    if a.scene == "arm":
        m, st = arm_hold(a.noslip)
        set_pairs(m, pairs_spec(a))
        hands, obj = (("left_finger_pad",), ("right_finger_pad",)), "box"
        st = dict(st, warm=np.zeros(m.nv))
    elif a.scene == "grip":
        m, _ = scene("grip", a.noslip, a)
        d = mujoco.MjData(m)
        d.ctrl[:] = [0.035, -0.035]
        ns0 = m.opt.noslip_iterations
        m.opt.noslip_iterations = 0
        for _ in range(150):
            mujoco.mj_step(m, d)
        m.opt.noslip_iterations = ns0
        st = dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(), warm=d.qacc_warmstart.copy())
        hands, obj = (("f1",), ("f2",)), "box"
    else:
        r = INTERNAL.hold_scene(a.scene, a.noslip, a) if INTERNAL is not None else None
        if r is None:
            raise SystemExit(f"--scene {a.scene}: internal scene, not in this release")
        m, st, hands, obj = r
        set_pairs(m, pairs_spec(a))
    dt = float(m.opt.timestep)
    n = int(round(a.seconds / dt))
    J, info = engine(m)
    qp, qv, w = f32(st["qpos"][None]), f32(st["qvel"][None]), f32(st["warm"][None])
    u, fr, com = f32(st["ctrl"][None]), f32(np.ones(1)), f32(np.zeros((1, info["com_words"])))
    QS, VS = [], []
    t = time.time()
    for _ in range(n):
        qp, qv, w, inf = J.forward(qp, qv, w, u, fr, com, iterations=int(m.opt.iterations), substeps=1)[:4]
        QS.append(np.asarray(qp[0], np.float64)); VS.append(np.asarray(qv[0], np.float64))
    t_eng = time.time() - t
    ours = slip_metrics(m, np.array(QS), np.array(VS), st["ctrl"], dt, hand_geoms=hands, obj=obj)
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:] = st["qpos"], st["qvel"], st["ctrl"]
    CQ, CV = [], []
    for _ in range(n):
        ref_step(m, d)
        CQ.append(d.qpos.copy()); CV.append(d.qvel.copy())
    cref = slip_metrics(m, np.array(CQ), np.array(CV), st["ctrl"], dt, hand_geoms=hands, obj=obj)
    dq = np.abs(np.array(QS) - np.array(CQ)).max(1)
    res = dict(section="hold", scene=a.scene, noslip_iterations=int(m.opt.noslip_iterations), seconds=a.seconds,
               engine=ours, mujoco_c=cref, max_qpos_diff_vs_c={"1s": float(dq[:int(1 / dt)].max()),
                                                              "end": float(dq.max())},
               engine_seconds=t_eng, provenance=provenance())
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def params_model(a):
    """(m0, box geom, states, hands) with noslip --noslip: --scene arm (the arm's hold, its box; the reference state)
    or grip (its box; --n settled states)."""
    import mujoco
    if a.scene == "arm":
        m, st = arm_hold(a.noslip)
        st = dict(st, warm=np.zeros(m.nv))
        st = {k: np.asarray(v, np.float64)[None] for k, v in st.items()}
        hands = (("left_finger_pad",), ("right_finger_pad",))
    elif a.scene == "grip":
        m = grip_model(a.noslip)
        st = grip_states(m, max(1, min(int(getattr(a, "n", 8)), 8)), seed=getattr(a, "seed", 0))
        hands = (("f1",), ("f2",))
    else:
        r = INTERNAL.params_model(a) if INTERNAL is not None else None
        if r is None:
            raise SystemExit(f"--scene {a.scene}: internal scene, not in this release")
        return r
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
    g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == b)
    return m, g, st, hands


def world_model(m0, g, f, mass_f):
    """MuJoCo C's model of a world: the box geom at f times its half-extents, the box body's mass and moments at
    mass_f times the model's (body_subtreemass / dof_M0 as mj_setConst; invweight0 unchanged, as the engine)."""
    import copy
    import mujoco
    m = copy.deepcopy(m0)
    m.geom_size[g] = m0.geom_size[g] * f
    m.geom_rbound[g] = float(np.linalg.norm(m.geom_size[g]))
    m.geom_aabb[g] = np.concatenate([np.zeros(3), m.geom_size[g]])
    b = int(m0.geom_bodyid[g])
    m.body_mass[b] = m0.body_mass[b] * mass_f
    m.body_inertia[b] = m0.body_inertia[b] * mass_f
    m.body_subtreemass[b] = m.body_mass[b]
    j = m.body_jntadr[b]
    if m.body_simple[b] and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        d = m.jnt_dofadr[j]
        m.dof_M0[d:d + 6] = np.r_[np.full(3, m.body_mass[b]), m.body_inertia[b]] + m.dof_armature[d:d + 6]
    return m


def params_diag(a, m0, models, W, st, com, it, r, ds, rel, same):
    """--diag K: the K worlds of largest error (same contact count) taken apart. Per world: the engine's and MuJoCo
    C's Newton iterations (the engine's info[:, 0]; C's solver_niter, either at the model's opt.iterations: not
    converged), whether the contact sets (geom pairs) agree, and the error again (a) with the engine's Newton at
    --diag-iters iterations, (b) against MuJoCo C converged (Newton tolerance 1e-12, --diag-iters iterations; the
    noslip setting unchanged), (c) both. A world whose error falls in (c) was Newton not converging (the model's
    iterations); one that stays is another cause."""
    import copy
    import mujoco
    from nablaengine import jax_ffi
    B = len(W)
    ni = int(a.diag_iters)
    r2 = jax_ffi.forward(f32(st["qpos"]), f32(st["qvel"]), f32(st["warm"]), f32(st["ctrl"]), f32(np.ones(B)),
                         f32(com), iterations=ni, substeps=1)
    v1, v2 = np.asarray(r[1], np.float64), np.asarray(r2[1], np.float64)
    inf1, inf2 = np.asarray(r[3]), np.asarray(r2[3])
    tight = {}
    for w, mm in models.items():
        mt = copy.deepcopy(mm)
        mt.opt.tolerance, mt.opt.iterations = 1e-12, ni
        tight[w] = mt
    dt_ = [c_step(tight[w], st, i) for i, w in enumerate(W)]
    ref, reft = np.array([d.qvel for d in ds]), np.array([d.qvel for d in dt_])
    den = np.maximum(np.linalg.norm(ref, axis=1), a.vfloor)
    rel = lambda v, x: np.linalg.norm(v - x, axis=1) / den
    e_it, e_ct, e_both = rel(v2, ref), rel(v1, reft), rel(v2, reft)
    order = [int(i) for i in np.argsort(-np.where(same, rel(v1, ref), -1.0))[:a.diag] if same[i]]
    out = []
    for i in order:
        w = W[i]
        d = ds[i]
        dc = mujoco.MjData(models[w])
        dc.qpos[:], dc.qvel[:] = st["qpos"][i], st["qvel"][i]
        mujoco.mj_forward(models[w], dc)
        pairs_c = sorted((int(c.geom1), int(c.geom2)) for c in dc.contact[:dc.ncon])
        out.append(dict(world=i, size=w[0], mass=w[1], rel=float(rel(v1, ref)[i]),
                        ncon_engine=int(inf1[i, 1]), ncon_c=int(d.ncon), nefc_engine=int(inf1[i, 2]), nefc_c=int(d.nefc),
                        newton_engine=int(inf1[i, 0]), newton_c=int(d.solver_niter[0]), iterations=it,
                        newton_engine_more=int(inf2[i, 0]), newton_c_tight=int(dt_[i].solver_niter[0]),
                        rel_engine_more_iters=float(e_it[i]), rel_vs_c_converged=float(e_ct[i]),
                        rel_both_converged=float(e_both[i]), c_contact_pairs=pairs_c))
    conv = same
    return dict(diag_iters=ni, worlds=out,
                all_both_converged=dict(median=q(e_both[conv], 0.5), max=float(e_both[conv].max())),
                newton_limit_engine=int((inf1[:, 0] >= it).sum()), newton_limit_c=int(sum(d.solver_niter[0] >= it for d in ds)))


def cmd_params(a):
    """Per-world object_size, torso_mass, torso_inertia (com_body: the box) with noslip: per world (size factor f,
    mass factor) MuJoCo C's model of that world is the reference (world_model).
      --what fwd   one substep from MuJoCo C states of each world's model (the scene's start states settled in each
                   world's model, the box size and mass varied over the worlds), qvel against MuJoCo C;
      --what hold  the scene held --seconds in each world (one batch), the metrics against each world's MuJoCo C;
      --what grad  the VJP of one substep (state, ctrl, and g_params' mass / moments columns) against MuJoCo C
                   central differences of each world's model, on states with a stable active set, cotangents on the
                   box's island (--island box)."""
    import mujoco
    from nablaengine import jax_ffi
    from nablaengine.params import PhysicsParams
    m0, g, st0, hands = params_model(a)
    if a.ns_tol is not None:
        m0.opt.noslip_tolerance = a.ns_tol
    set_pairs(m0, pairs_spec(a))
    if a.what == "grad" and a.tight:
        m0.opt.tolerance, m0.opt.noslip_tolerance = 1e-12, 1e-14
        m0.opt.iterations = max(int(m0.opt.iterations), 50)
    facs = [float(x) for x in a.sizes.split(",")]
    mfacs = [float(x) for x in a.masses.split(",")]
    P = PhysicsParams(("object_size", "torso_mass", "torso_inertia"),
                      object_size_max=tuple(m0.geom_size[g] * max(facs) * 1.001))
    kw = {"noslip_pairs": PAIRS["spec"]} if PAIRS["spec"] is not None else {}
    info = jax_ffi.init(model=m0, foot_pairs=[], com_body="box", params=P, **kw)
    b = int(m0.geom_bodyid[g])
    if a.what == "hold":
        W = [(f, mf) for f in facs for mf in mfacs]
    else:
        W = [(facs[i % len(facs)], mfacs[(i // len(facs)) % len(mfacs)]) for i in range(a.n)]
    B = len(W)
    models = {w: world_model(m0, g, *w) for w in set(W)}
    for mm in models.values():
        pass
    cw0 = int(info["com_words"]) - int(info.get("param_words", 0))

    def com_of(ws):
        sz = np.array([m0.geom_size[g] * f for f, _ in ws])
        ms = np.array([[m0.body_mass[b] * mf] for _, mf in ws])
        it = np.array([m0.body_inertia[b] * mf for _, mf in ws])
        return np.concatenate([np.zeros((len(ws), cw0), np.float32),
                               P.pack(m0.nv, object_size=sz, torso_mass=ms, torso_inertia=it)], 1)
    com = com_of(W)
    it = int(m0.opt.iterations)
    res = dict(section="params", what=a.what, scene=a.scene, noslip_iterations=int(m0.opt.noslip_iterations),
               pairs=PAIRS["spec"], sizes=facs, masses=mfacs, nworld=B, provenance=provenance())
    if a.what in ("fwd", "grad"):
        st = {k: [] for k in ("qpos", "qvel", "ctrl", "warm")}
        for i, w in enumerate(W):
            mm = models[w]
            d = mujoco.MjData(mm)
            j = i % len(st0["qpos"])
            d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = st0["qpos"][j], st0["qvel"][j], st0["ctrl"][j], st0["warm"][j]
            ns0 = mm.opt.noslip_iterations
            if a.state_noslip is not None:
                mm.opt.noslip_iterations = a.state_noslip
            for _ in range(a.settle + 5 * (i // len(set(W)))):
                ref_step(mm, d)
            mm.opt.noslip_iterations = ns0
            for k, v in (("qpos", d.qpos), ("qvel", d.qvel), ("ctrl", d.ctrl), ("warm", d.qacc_warmstart)):
                st[k].append(v.copy())
        st = {k: np.array(v) for k, v in st.items()}
    if a.what == "fwd":
        r = jax_ffi.forward(f32(st["qpos"]), f32(st["qvel"]), f32(st["warm"]), f32(st["ctrl"]), f32(np.ones(B)),
                            f32(com), iterations=it, substeps=1)
        v = np.asarray(r[1], np.float64)
        ds = [c_step(models[w], st, i) for i, w in enumerate(W)]
        ref = np.array([d.qvel for d in ds])
        same = np.asarray(r[3])[:, 1] == np.array([d.ncon for d in ds])
        rel = np.linalg.norm(v - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), a.vfloor)
        per = {}
        for i, w in enumerate(W):
            if same[i]:
                per.setdefault(f"size {w[0]:g} mass {w[1]:g}", []).append(float(rel[i]))
        res.update(same_ncon=int(same.sum()), rel_median=q(rel[same], 0.5), rel_max=float(rel[same].max()),
                   per_world={k: dict(n=len(x), median=q(x, 0.5), max=max(x)) for k, x in per.items()})
        if a.diag:
            res["diag"] = params_diag(a, m0, models, W, st, com, it, r, ds, rel, same)
    elif a.what == "hold":
        n = int(round(a.seconds / float(m0.opt.timestep)))
        st = {k: np.repeat(v[:1], B, 0) for k, v in st0.items()}
        qp, qv, wm = f32(st["qpos"]), f32(st["qvel"]), f32(st["warm"])
        u, fr, cm = f32(st["ctrl"]), f32(np.ones(B)), f32(com)
        QS, VS = [], []
        for _ in range(n):
            qp, qv, wm, inf = jax_ffi.forward(qp, qv, wm, u, fr, cm, iterations=it, substeps=1)[:4]
            QS.append(np.asarray(qp, np.float64)); VS.append(np.asarray(qv, np.float64))
        QS, VS = np.array(QS), np.array(VS)
        dt = float(m0.opt.timestep)
        out = {}
        for i, w in enumerate(W):
            mm = models[w]
            eng = slip_metrics(mm, QS[:, i], VS[:, i], st["ctrl"][i], dt, hand_geoms=hands, obj="box")
            d = mujoco.MjData(mm)
            d.qpos[:], d.qvel[:], d.ctrl[:] = st["qpos"][i], st["qvel"][i], st["ctrl"][i]
            CQ, CV = [], []
            for _ in range(n):
                ref_step(mm, d)
                CQ.append(d.qpos.copy()); CV.append(d.qvel.copy())
            cref = slip_metrics(mm, np.array(CQ), np.array(CV), st["ctrl"][i], dt, hand_geoms=hands, obj="box")
            out[f"size {w[0]:g} mass {w[1]:g}"] = dict(engine=eng, reference=cref,
                                                      max_qpos_diff=float(np.abs(QS[:, i] - np.array(CQ)).max()))
        res["worlds"] = out
    else:
        rng = np.random.default_rng(a.seed + 3)
        yq, yv = rng.standard_normal((B, m0.nq)), rng.standard_normal((B, m0.nv))
        masks = [island_masks(models[w], st, i, "box") if a.island else None for i, w in enumerate(W)]
        for i in range(B):
            if masks[i] is not None:
                yq[i] *= masks[i][0]
                yv[i] *= masks[i][1]
        r = jax_ffi.forward(f32(st["qpos"]), f32(st["qvel"]), f32(st["warm"]), f32(st["ctrl"]), f32(np.ones(B)),
                            f32(com), iterations=it, substeps=1, tape=True)
        gg = jax_ffi.backward(r[-1], f32(np.ones(B)), f32(com), f32(yq), f32(yv), substeps=1, clamp_kv=1, normal=0)
        gq, gv, gu = (np.asarray(x, np.float64) for x in gg[:3])
        gp = np.asarray(gg[4], np.float64)
        lay = P.layout(m0.nv)
        om, oi = cw0 + lay["torso_mass"][0], cw0 + lay["torso_inertia"][0]
        rows = []
        h = a.h
        for i, w in enumerate(W):
            mm = models[w]
            for blk in ("qpos", "qvel", "ctrl", "mass", "inertia"):
                if blk in ("qpos", "qvel", "ctrl"):
                    x = st[blk][i]
                    dd = rng.standard_normal(len(x))
                    if blk == "qpos":
                        for j in range(m0.njnt):
                            if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                                dd[m0.jnt_qposadr[j] + 3:m0.jnt_qposadr[j] + 7] = 0.0
                    if masks[i] is not None:
                        dd *= masks[i][("qpos", "qvel", "ctrl").index(blk)]
                    dd /= max(np.linalg.norm(dd), 1e-30)
                    eng = float((gq, gv, gu)[("qpos", "qvel", "ctrl").index(blk)][i] @ dd)
                else:
                    dd = rng.standard_normal(1 if blk == "mass" else 3)
                    dd /= np.linalg.norm(dd)
                    eng = float(gp[i, (om - cw0) if blk == "mass" else (oi - cw0):][:len(dd)] @ dd)
                vals, sets = [], []
                for sg in (1.0, -1.0, 0.0):
                    s2 = {kk: vv[i:i + 1].copy() for kk, vv in st.items()}
                    m2 = mm
                    if blk in ("qpos", "qvel", "ctrl"):
                        s2[blk][0] = st[blk][i] + sg * h * dd
                    else:
                        import copy
                        m2 = copy.deepcopy(mm)
                        if blk == "mass":
                            m2.body_mass[b] += sg * h * dd[0]
                            m2.body_subtreemass[b] = m2.body_mass[b]
                        else:
                            m2.body_inertia[b] += sg * h * dd
                        jn = m2.body_jntadr[b]
                        if m2.body_simple[b] and m2.jnt_type[jn] == mujoco.mjtJoint.mjJNT_FREE:
                            dof = m2.jnt_dofadr[jn]
                            m2.dof_M0[dof:dof + 6] = np.r_[np.full(3, m2.body_mass[b]), m2.body_inertia[b]] + \
                                m2.dof_armature[dof:dof + 6]
                    d = c_step(m2, s2, 0)
                    vals.append(float(yq[i] @ d.qpos + yv[i] @ d.qvel))
                    d0 = mujoco.MjData(m2)
                    d0.qpos[:], d0.qvel[:], d0.ctrl[:], d0.qacc_warmstart[:] = (s2["qpos"][0], s2["qvel"][0],
                                                                                  s2["ctrl"][0], s2["warm"][0])
                    sets.append(active_set(m2, d0))
                fd = (vals[0] - vals[1]) / (2 * h)
                conv = noslip_gap(mm, {kk: vv[i:i + 1] for kk, vv in st.items()},
                                  masks[i][1] if masks[i] is not None else None)
                rows.append(dict(world=i, size=w[0], mass=w[1], block=blk, fd=fd, eng=eng,
                                 stable=bool(sets[0] == sets[1] == sets[2] and conv < a.conv_tol), conv_gap=conv))
        for blk in ("qpos", "qvel", "ctrl", "mass", "inertia", "all"):
            sel = [r_ for r_ in rows if blk == "all" or r_["block"] == blk]
            ok = [r_ for r_ in sel if r_["stable"]]
            scale = max(np.median([abs(r_["fd"]) for r_ in sel]), 1e-12) * 1e-3
            e = [abs(r_["eng"] - r_["fd"]) / max(abs(r_["fd"]), scale) for r_ in ok]
            res[blk] = dict(n=len(sel), stable=len(ok), rel_median=q(e, 0.5), rel_p90=q(e, 0.9),
                            rel_max=max(e) if e else None)
        res["rows"] = rows
    print(json.dumps({k: v for k, v in res.items() if k not in ("provenance", "rows")}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_parity(a):
    """torch_ffi against jax_ffi on the same GPU (the venv with both): forward with tape and backward, bitwise."""
    import torch
    from nablaengine import torch_ffi
    m, st = scene(a.scene, a.noslip, a)
    J, info = engine(m)
    torch_ffi.init(model=m, foot_pairs=[], **({"noslip_pairs": PAIRS["spec"]} if PAIRS["spec"] is not None else {}))
    B = len(st["qpos"])
    rng = np.random.default_rng(a.seed + 7)
    yq, yv = rng.standard_normal((B, m.nq)), rng.standard_normal((B, m.nv))
    ins = [np.ascontiguousarray(st[k], np.float32) for k in ("qpos", "qvel", "warm", "ctrl")]
    fr, com = np.ones(B, np.float32), np.zeros((B, info["com_words"]), np.float32)
    it = int(m.opt.iterations)
    rj = J.forward(*[f32(x) for x in ins], f32(fr), f32(com), iterations=it, substeps=a.substeps, tape=True)
    gj = J.backward(rj[-1], f32(fr), f32(com), f32(yq), f32(yv), substeps=a.substeps)
    dev = torch.device("cuda")
    tt = lambda x: torch.as_tensor(np.ascontiguousarray(x, np.float32), device=dev)
    rt = torch_ffi.forward(*[tt(x) for x in ins], tt(fr), tt(com), iterations=it, substeps=a.substeps, tape=True)
    gt = torch_ffi.backward(rt[-1], tt(fr), tt(com), tt(yq), tt(yv), substeps=a.substeps)
    same = lambda x, y: bool(np.array_equal(np.asarray(x).view(np.uint8), y.cpu().numpy().view(np.uint8)))
    res = dict(section="parity", scene=a.scene, noslip_iterations=int(m.opt.noslip_iterations), nworld=B,
               substeps=a.substeps, forward=[same(x, y) for x, y in zip(rj, rt)],
               backward=[same(x, y) for x, y in zip(gj, gt)], provenance=provenance())
    res["pass"] = all(res["forward"]) and all(res["backward"])
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    sys.exit(0 if res["pass"] else 1)


def cmd_combine(a):
    """The parts' JSON files (--parts: files or globs) into one results file (--out), without the per-row data."""
    import glob
    parts = []
    for pat in a.parts:
        for f in sorted(glob.glob(pat)):
            r = json.loads(Path(f).read_text())
            r.pop("rows", None)
            r["part"] = Path(f).name
            parts.append(r)
    res = dict(script="scripts/noslip/noslip_check.py", date=time.strftime("%Y-%m-%d"),
               reference="MuJoCo C float64 (the noslip mode's exception: MuJoCo Warp has no noslip solver)",
               parts=parts)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{len(parts)} parts -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section", choices=["fwd", "hold", "grad", "time", "parity", "combine", "params"])
    ap.add_argument("--scene", default="grip", choices=["grip", "g1", "arm"] + list(getattr(INTERNAL, "SCENES", ())))
    ap.add_argument("--what", default="fwd", choices=["fwd", "hold", "grad"], help="params: the check")
    ap.add_argument("--sizes", default="0.95,1.0,1.05", help="params: box size factors")
    ap.add_argument("--masses", default="1.0", help="params: box mass / moment factors")
    ap.add_argument("--state-noslip", type=int, default=None, help="grip / params: settle the states with this noslip "
                    "setting")
    ap.add_argument("--settle", type=int, default=25, help="params: MuJoCo C steps before a fwd / grad state")
    ap.add_argument("--diag", type=int, default=0, help="params --what fwd: take apart the K worlds of largest error")
    ap.add_argument("--diag-iters", type=int, default=100, help="params --diag: Newton iterations of the converged runs")
    ap.add_argument("--noslip", type=int, default=10)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vfloor", type=float, default=1e-2, help="m/s or rad/s: the relative error's denominator floor")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--h", type=float, default=1e-6)
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--island", default="", help="grad: restrict cotangents and directions to the island of this body")
    ap.add_argument("--conv-tol", type=float, default=1e-7, help="grad: states whose noslip solve (MuJoCo C) moves more "
                    "than this (relative qacc) from n to 10 n sweeps are excluded (not converged)")
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--tight", type=int, default=1, help="grad: Newton / noslip tolerances off (converged solves)")
    ap.add_argument("--ns-tol", type=float, default=None, help="fwd / params: override opt.noslip_tolerance")
    ap.add_argument("--parts", nargs="*", default=[], help="combine: the parts' JSON files")
    ap.add_argument("--out", default="")
    ap.add_argument("--pairs", default=None, help="noslip_pairs: 'objects' or comma-separated pair indices / names; "
                    "the reference is then noslip_ref (float64, MuJoCo's noslip on those pairs)")
    ap.add_argument("--ref-iters", type=int, default=None, help="grad: the reference converged (noslip_ref, this many "
                    "sweeps, no tolerance)")
    if INTERNAL is not None:
        INTERNAL.add_arguments(ap)
    a = ap.parse_args()
    PAIRS["ref_iters"] = a.ref_iters
    if a.out and Path(a.out).exists() and "docs/results" in str(Path(a.out).resolve()):
        sys.exit(f"--out {a.out}: exists (committed results are not overwritten)")
    dict(fwd=cmd_fwd, hold=cmd_hold, grad=cmd_grad, time=cmd_time, parity=cmd_parity, combine=cmd_combine, params=cmd_params)[a.section](a)


def active_set(m, d0):
    """The discrete state of MuJoCo C at d0's input state: contacts (geom pairs), the Newton rows' states (noslip off)
    and the noslip pairs' bound pattern (a pyramid row whose noslip force is exactly 0)."""
    import mujoco
    m0 = m.opt.noslip_iterations
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = d0.qpos, d0.qvel, d0.ctrl, d0.qacc_warmstart
    m.opt.noslip_iterations = 0
    mujoco.mj_forward(m, d)
    con = tuple((int(c.geom1), int(c.geom2)) for c in d.contact[:d.ncon])
    import noslip_ref
    x = noslip_ref.dense_J(m, d) @ d.qacc - d.efc_aref[:d.nefc]
    newton = tuple(bool(v < 0) for v in x)
    m.opt.noslip_iterations = m0
    ref_forward(m, d)
    pyr = d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL
    ns = tuple(bool(x) for x in (np.abs(d.efc_force[:d.nefc]) == 0.0) & pyr)
    return con, newton, ns


def island_masks(m, st, i, body):
    """(qpos, qvel, ctrl) 0/1 masks of the coordinates of MuJoCo C's constraint island that holds body's dofs at state
    i (islands on)."""
    import mujoco
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:] = st["qpos"][i], st["qvel"][i], st["ctrl"][i]
    mujoco.mj_forward(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    isl = int(d.dof_island[m.body_dofadr[b]])
    mv = (np.asarray(d.dof_island[:m.nv]) == isl).astype(float)
    mq = np.zeros(m.nq)
    for j in range(m.njnt):
        dof, qa = int(m.jnt_dofadr[j]), int(m.jnt_qposadr[j])
        n = {0: 7, 1: 4, 2: 1, 3: 1}[int(m.jnt_type[j])]
        mq[qa:qa + n] = mv[dof]
    mu = np.array([mv[m.jnt_dofadr[m.actuator_trnid[k, 0]]] for k in range(m.nu)], float)
    return mq, mv, mu


def noslip_gap(m, st, mask=None):
    """|qacc(n sweeps) - qacc(10 n sweeps)| / max(|qacc|, 1) of MuJoCo C at the state (the noslip solve's convergence)."""
    import mujoco
    n = int(m.opt.noslip_iterations)
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = st["qpos"][0], st["qvel"][0], st["ctrl"][0], st["warm"][0]
    ref_forward(m, d)
    a1 = d.qacc.copy()
    ref_forward(m, d, 10 * (PAIRS["ref_iters"] or n))
    w = np.ones(m.nv) if mask is None else mask
    return float(np.linalg.norm(w * (a1 - d.qacc)) / max(np.linalg.norm(w * d.qacc), 1.0))


def cmd_grad(a):
    """VJPs of one control step (1 substep) against float64 MuJoCo C central differences of L = yq . qpos' + yv .
    qvel' along random unit directions of (qpos without quaternion entries, qvel, ctrl), h = --h; states whose
    active set (active_set) differs at x +- h along the direction are excluded."""
    import mujoco
    m, st = scene(a.scene, a.noslip, a)
    if a.tight:
        m.opt.tolerance = 1e-12
        m.opt.noslip_tolerance = 1e-14
        m.opt.iterations = max(int(m.opt.iterations), 50)
    J, info = engine(m)
    B = len(st["qpos"])
    rng = np.random.default_rng(a.seed + 1)
    yq, yv = rng.standard_normal((B, m.nq)), rng.standard_normal((B, m.nv))
    if a.island:
        for i in range(B):
            mq, mv, _ = island_masks(m, st, i, a.island)
            yq[i] *= mq
            yv[i] *= mv
    r = J.forward(f32(st["qpos"]), f32(st["qvel"]), f32(st["warm"]), f32(st["ctrl"]), f32(np.ones(B)),
                  f32(np.zeros((B, info["com_words"]))), iterations=int(m.opt.iterations), substeps=1, tape=True)
    g = J.backward(r[-1], f32(np.ones(B)), f32(np.zeros((B, info["com_words"]))), f32(yq), f32(yv), substeps=1,
                   clamp_kv=1, normal=0)
    gq, gv, gu = (np.asarray(x, np.float64) for x in g[:3])
    ginfo = np.asarray(g[3])
    h = a.h
    rows = []
    for i in range(B):
        masks = island_masks(m, st, i, a.island) if a.island else None
        for k, blk in enumerate(("qpos", "qvel", "ctrl")):
            x = st[blk][i]
            dd = rng.standard_normal(len(x))
            if masks is not None:
                dd *= masks[k]
            if k == 0:
                for j in range(m.njnt):
                    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                        dd[m.jnt_qposadr[j] + 3:m.jnt_qposadr[j] + 7] = 0.0
            dd /= max(np.linalg.norm(dd), 1e-30)
            vals, sets = [], []
            for sg in (1.0, -1.0, 0.0):
                s2 = {kk: vv[i:i + 1].copy() for kk, vv in st.items()}
                s2[blk][0] = x + sg * h * dd
                d = c_step(m, s2, 0)
                vals.append(float(yq[i] @ d.qpos + yv[i] @ d.qvel))
                d0 = mujoco.MjData(m)
                d0.qpos[:], d0.qvel[:], d0.ctrl[:], d0.qacc_warmstart[:] = (s2["qpos"][0], s2["qvel"][0],
                                                                              s2["ctrl"][0], s2["warm"][0])
                sets.append(active_set(m, d0))
            fd = (vals[0] - vals[1]) / (2 * h)
            eng = float((gq, gv, gu)[k][i] @ dd)
            stable = sets[0] == sets[1] == sets[2]
            if k == 0:
                conv = noslip_gap(m, {kk: vv[i:i + 1] for kk, vv in st.items()},
                                  masks[1] if masks is not None else None) if m.opt.noslip_iterations else 0.0
            stable = stable and conv < a.conv_tol
            rows.append(dict(state=i, block=blk, fd=fd, eng=eng, stable=stable, conv_gap=conv,
                             nfree=int(sum(1 for t in sets[2][2] if not t))))
    res = dict(section="grad", scene=a.scene, noslip_iterations=int(m.opt.noslip_iterations), tight=a.tight, h=h,
               nworld=B, adjoint_info=sorted(set(int(x) for x in ginfo.ravel())), provenance=provenance())
    for blk in ("qpos", "qvel", "ctrl", "all"):
        sel = [r_ for r_ in rows if (blk == "all" or r_["block"] == blk)]
        ok = [r_ for r_ in sel if r_["stable"]]
        scale = max(np.median([abs(r_["fd"]) for r_ in sel]), 1e-12) * 1e-3
        e = [abs(r_["eng"] - r_["fd"]) / max(abs(r_["fd"]), scale) for r_ in ok]
        res[blk] = dict(n=len(sel), stable=len(ok), excluded=len(sel) - len(ok), rel_median=q(e, 0.5),
                        rel_p90=q(e, 0.9), rel_max=max(e) if e else None, abs_floor=scale)
    res["rows"] = rows
    print(json.dumps({k: v for k, v in res.items() if k not in ("provenance", "rows")}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_time(a):
    """ms per control step (--substeps substeps) at --nworld worlds: forward without tape, forward with tape, backward
    (median of --reps after a warm-up; the states tiled from --n scene states)."""
    import jax
    m, st = scene(a.scene, a.noslip, a)
    J, info = engine(m)
    B = a.nworld
    idx = np.arange(B) % len(st["qpos"])
    args = [f32(st[k][idx]) for k in ("qpos", "qvel", "warm", "ctrl")] + [f32(np.ones(B)),
                                                                          f32(np.zeros((B, info["com_words"])))]
    it, ns = int(m.opt.iterations), a.substeps

    def timed(fn):
        jax.block_until_ready(fn())
        ts = []
        for _ in range(a.reps):
            t = time.perf_counter()
            jax.block_until_ready(fn())
            ts.append(1e3 * (time.perf_counter() - t))
        return float(np.median(ts))
    fwd = timed(lambda: J.forward(*args, iterations=it, substeps=ns))
    r = J.forward(*args, iterations=it, substeps=ns, tape=True)
    fwt = timed(lambda: J.forward(*args, iterations=it, substeps=ns, tape=True))
    yq, yv = f32(np.ones((B, m.nq)) * 1e-2), f32(np.ones((B, m.nv)) * 1e-2)
    bwd = timed(lambda: J.backward(r[-1], args[4], args[5], yq, yv, substeps=ns))
    res = dict(section="time", scene=a.scene, noslip_iterations=int(m.opt.noslip_iterations), nworld=B, substeps=ns,
               iterations=it, forward_ms=fwd, forward_tape_ms=fwt, backward_ms=bwd, reps=a.reps,
               provenance=provenance())
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

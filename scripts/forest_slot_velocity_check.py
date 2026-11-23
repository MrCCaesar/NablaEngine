"""Object slot motion (TerrainConfig(objects=K, object_velocity=True), the forest engine): moving and accelerating
floors, pushers and conveyor belts. Each slot's motion words in com, (v, w, s, a, al, sd): the slot moves rigidly with
(v, w, a, al) within the control step, its surface at a contact point p moves with u = v + w x (p - pos) + the
tangential part of R s and accelerates with a + al x (p - pos) + the tangential part of R sd; the contact rows act on
the robot's motion relative to the slot's."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

HEAVY = 1e8
NVW = 18


GEOM = {"box": 'type="box" size=".1 .1 .1"', "capsule": 'type="capsule" size=".05" fromto="-.15 0 0 .15 0 0"'}


def box_xml(belt, shape="box"):
    """The box (0.1 m half size, 1 kg; shape "capsule": a 0.05 m capsule, 0.3 m long, lying along x) with a floor
    pair (mu 1); belt: the MuJoCo reference, the floor replaced by a heavy free belt body (half sizes 1, 1, 0.05) with
    the floor pair's parameters."""
    xml = _box_xml(belt).replace('type="box" size=".1 .1 .1"', GEOM[shape])
    return xml


def _box_xml(belt):
    if not belt:
        return """<mujoco><option timestep="0.005"/>
<worldbody><geom name="floor" type="plane" size="5 5 .1"/>
<body name="box" pos="0 0 0.3"><freejoint/><geom name="bx" type="box" size=".1 .1 .1" mass="1"/></body></worldbody>
<contact><pair geom1="floor" geom2="bx" condim="3" friction="1 1 0.005 0.0001 0.0001"/></contact></mujoco>"""
    return f"""<mujoco><option timestep="0.005"/>
<worldbody>
<body name="box" pos="0 0 0.3"><freejoint/><geom name="bx" type="box" size=".1 .1 .1" mass="1"
  contype="0" conaffinity="0"/></body>
<body name="belt" pos="0 0 -0.05" gravcomp="1"><freejoint/>
  <inertial pos="0 0 0" mass="{HEAVY}" diaginertia="{HEAVY} {HEAVY} {HEAVY}"/>
  <geom name="belt" type="box" size="1 1 .05" contype="0" conaffinity="0"/></body></worldbody>
<contact><pair geom1="belt" geom2="bx" condim="3" friction="1 1 0.005 0.0001 0.0001"/></contact></mujoco>"""


def g1_reference():
    raise SystemExit("the G1 scene of this check is not part of this release (--scene box)")


def rot(axis, ang):
    import mujoco
    q, R = np.zeros(4), np.zeros(9)
    axis = np.asarray(axis, np.float64)
    mujoco.mju_axisAngle2Quat(q, axis / max(np.linalg.norm(axis), 1e-12), float(ang))
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3), q


def mat2quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, np.float64).reshape(9))
    return q


def quat2mat(q):
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, np.float64))
    return R.reshape(3, 3)


def engine_init(scene, velocity=True):
    """(jax_ffi, info, MjModel, friction input) of the engine build of scene ("box" / "g1") with one object slot."""
    import mujoco
    from nablaengine import engines, jax_ffi, terrain as T
    if scene in GEOM:
        m = mujoco.MjModel.from_xml_string(box_xml(False, scene))
        engines.prefer_forest(m)
        cfg = T.TerrainConfig(geoms="body", objects=1, object_velocity=velocity)
        info = jax_ffi.init(model=m, foot_pairs=[0], terrain=cfg)
        mu = 1.0
    else:
        os.environ["DK_ENGINE"] = "forest"
        from nablaengine import fixtures
        m, meta = fixtures.load_model(check=None)
        cfg = T.TerrainConfig(geoms="feet", objects=1, object_velocity=velocity)
        info = jax_ffi.init(terrain=cfg)
        mu = float(m.pair_friction[int(meta["foot_pairs"][0]), 0])
    assert info["engine"] == "forest", info["engine"]
    jax_ffi.set_terrain(T.Terrain())
    return jax_ffi, info, m, mu


def pack(objs, vel, B, velocity=True):
    from nablaengine import terrain as T
    return T.pack_com(np.zeros((B, 3), np.float32), np.asarray(objs, np.float32).reshape(B, 1, 15),
                      np.asarray(vel, np.float32).reshape(B, 1, NVW) if velocity else None)


def cmd_conveyor(a):
    import jax.numpy as jnp
    ffi, info, m, _ = engine_init("box")
    from nablaengine import terrain as T
    I3 = np.eye(3)
    Resc, _ = rot([0, 1, 0], -np.deg2rad(15.0))
    Ryaw, _ = rot([0, 0, 1], np.deg2rad(30.0))
    belt, plate = [5.0, 1.0, 0.05], [1.0, 1.0, 0.05]
    scenes = [
        ("belt +x 1 m/s", I3, belt, T.slot_velocity(s=[1.0, 0, 0]), False, [-4, 0, 0], True),
        ("belt -y 0.5 m/s", I3, [1.0, 5.0, 0.05], T.slot_velocity(s=[0, -0.5, 0]), False, [0, 4, 0], True),
        ("belt yawed 30 deg 1.5 m/s", Ryaw, belt, T.slot_velocity(s=[1.5, 0, 0]), False, [-4, 0, 0], True),
        ("escalator 15 deg 0.5 m/s up", Resc, belt, T.slot_velocity(s=[0.5, 0, 0]), False, [-4, 0, 0], True),
        ("moving platform v (0.8, 0.3, 0)", I3, plate, T.slot_velocity(v=[0.8, 0.3, 0]), True, [0, 0, 0], False),
        ("turntable w_z 1 rad/s, r 0.5 m", I3, plate, T.slot_velocity(w=[0, 0, 1.0]), True, [0.5, 0, 0], False),
        ("elevator v_z 0.3 m/s", I3, plate, T.slot_velocity(v=[0, 0, 0.3]), True, [0, 0, 0], False),
    ]
    mus = [1.0] * len(scenes)
    for sc in scenes[:3]:
        scenes.append((sc[0] + ", mu 0.3", *sc[1:6], False))
        mus.append(0.3)
    B, h = len(scenes), 4 * m.opt.timestep
    R = np.stack([np.asarray(sc[1], np.float64) for sc in scenes])
    c = np.tile([0.0, 0.0, -0.05], (B, 1))
    half = np.asarray([sc[2] for sc in scenes], np.float64)
    vel = np.stack([sc[3] for sc in scenes]).astype(np.float64)
    q = np.zeros((B, 7))
    for i, sc in enumerate(scenes):
        q[i, :3] = c[i] + R[i] @ (np.asarray(sc[5], np.float64) + [0, 0, 0.15])
        q[i, 3:7] = mat2quat(R[i])
    warm = jnp.zeros((B, 6))
    ctrl, fr = jnp.zeros((B, 0)), jnp.asarray(mus, jnp.float32)
    nstep = int(round(a.seconds / h))
    hist_v, hist_u, hist_n = [], [], []
    qj, vj = jnp.asarray(q, jnp.float32), jnp.zeros((B, 6), jnp.float32)

    def surface(i, p):
        return vel[i, :3] + np.cross(vel[i, 3:6], p - c[i]) + R[i] @ vel[i, 6:9]

    for k in range(nstep):
        objs = np.stack([T.box_object(c[i], R[i], half[i]) for i in range(B)])
        com = jnp.asarray(pack(objs, vel, B))
        qj, vj, warm, info_ = ffi.forward(qj, vj, warm, ctrl, fr, com, iterations=a.iterations)[:4]
        qn, vn = np.asarray(qj, np.float64), np.asarray(vj, np.float64)
        for i, sc in enumerate(scenes):
            if sc[4]:
                c[i] = c[i] + h * vel[i, :3]
                wn = np.linalg.norm(vel[i, 3:6])
                if wn > 0:
                    R[i] = rot(vel[i, 3:6], wn * h)[0] @ R[i]
        hist_v.append(vn[:, :3]); hist_u.append(np.stack([surface(i, qn[i, :3]) for i in range(B)]))
        hist_n.append(np.asarray(info_)[:, 1])
    hv, hu, hn = np.stack(hist_v), np.stack(hist_u), np.stack(hist_n)
    out = []
    for i, sc in enumerate(scenes):
        rel = np.linalg.norm(hv[:, i] - hu[:, i], axis=1) / max(np.linalg.norm(hu[0, i]), 1e-9)

        def t_within(tol):
            bad = np.nonzero(rel > tol)[0]
            if len(bad) and bad[-1] == nstep - 1:
                return None
            return float((bad[-1] + 2) * h) if len(bad) else float(h)

        nrm = R[i][:, 2]
        cos = []
        for k in range(1, nstep):
            vr = hu[k - 1, i] - hv[k - 1, i]
            vr = vr - nrm * (nrm @ vr)
            if np.linalg.norm(vr) < 0.05 * np.linalg.norm(hu[k - 1, i]) or hn[k, i] == 0:
                continue
            acc = hv[k, i] - hv[k - 1, i]
            acc = acc - nrm * (nrm @ acc)
            cos.append(float(acc @ vr / max(np.linalg.norm(acc) * np.linalg.norm(vr), 1e-30)))
        t1, t2 = t_within(0.01), t_within(0.02)
        out.append(dict(scene=sc[0], mu=mus[i], conveyor=sc[6], t_within_1pct_s=t1, t_within_2pct_s=t2,
                        accept=(t1 is not None and t1 <= 0.5) if sc[6] else None,
                        final_rel_error=float(rel[-1]), final_box_velocity=hv[-1, i].round(5).tolist(),
                        final_surface_velocity=hu[-1, i].round(5).tolist(),
                        slip_steps=len(cos), friction_cos_min=min(cos) if cos else None,
                        friction_cos_median=float(np.median(cos)) if cos else None, contacts_last=int(hn[-1, i])))
        print(f"{sc[0]:34s} t(1%) {t1}  t(2%) {t2}  final rel {rel[-1]:.2e}  friction cos min "
              f"{(min(cos) if cos else float('nan')):.4f} over {len(cos)} slipping steps")
    res = dict(check="conveyor", engine=info["engine"], dt_control=h, substeps=4, iterations=a.iterations,
               box="0.1 m half size, 1 kg, condim 3", ideal_belt_1ms_s=1.0 / 9.81, scenes=out,
               accepted=all(o["accept"] for o in out if o["conveyor"]))
    dump(res, a.out)


def random_states(scene, B, seed, case):
    """(m_ref, (robot nq, nv, nu), robot qpos (B, nq), qvel, ctrl, warm, slot rows (B, 15), the engine's velocity words
    (B, 18), the reference body's (v, w, a, al) (B, 12), friction input) of scene; case "rigid": random (v, w, a, al) for both, "conveyor":
    the engine's s in the top face's plane, the reference's v = R s, "rest": zeros."""
    import mujoco
    rng = np.random.default_rng(seed)
    if scene in GEOM:
        mr = mujoco.MjModel.from_xml_string(box_xml(True, scene))
        nq, nv, nu = 7, 6, 0
        q, v = np.zeros((B, 7)), np.zeros((B, 6))
        objs = np.zeros((B, 15))
        for i in range(B):
            Rb = rot(rng.normal(size=3), rng.uniform(0, np.deg2rad(10)))[0] @ rot([0, 0, 1], rng.uniform(-np.pi, np.pi))[0]
            cb = np.r_[rng.uniform(-0.1, 0.1, 2), -0.05]
            objs[i] = np.r_[cb, Rb.reshape(9), 1.0, 1.0, 0.05]
            tilt = rot([0, 0, 1], rng.uniform(-3, 3))[0]
            off = np.r_[rng.uniform(-0.5, 0.5, 2), (0.15 if scene == "box" else 0.1) - rng.uniform(0.0, 0.003)]
            q[i, :3], q[i, 3:7] = cb + Rb @ off, mat2quat(Rb @ tilt)
            v[i] = np.r_[Rb @ np.r_[rng.normal(0, 0.5, 2), rng.normal(0, 0.05)], Rb @ np.r_[0, 0, rng.normal(0, 1.0)]]
        ctrl, warm, mu = np.zeros((B, 0)), np.zeros((B, 6)), 1.0
        sv, sw = np.r_[0.8, 0.8, 0.25], 0.3
    else:
        from nablaengine import fixtures
        mr, mu = g1_reference()
        nq, nv, nu = mr.nq - 7, mr.nv - 6, mr.nu
        z = fixtures.load_corpus("mf_s0")
        idx = rng.choice(len(z["qpos"]), B, replace=len(z["qpos"]) < B)
        q, v = np.asarray(z["qpos"][idx], np.float64), np.asarray(z["qvel"][idx], np.float64)
        ctrl, warm = np.asarray(z["ctrl"][idx], np.float64), np.asarray(z["qacc_warmstart"][idx], np.float64)
        objs = np.zeros((B, 15))
        for i in range(B):
            Rb = rot([0, 0, 1], rng.uniform(-np.pi, np.pi))[0]
            objs[i] = np.r_[q[i, 0] + rng.uniform(-0.2, 0.2), q[i, 1] + rng.uniform(-0.2, 0.2), -0.05, Rb.reshape(9),
                            1.0, 1.0, 0.05]
        sv, sw = np.r_[0.6, 0.6, 0.15], 0.3
    mr.stat.meaninertia = (mujoco.MjModel.from_xml_string(box_xml(False, scene)) if scene in GEOM
                           else __import__("nablaengine.fixtures", fromlist=["x"]).load_model(check=None)[0]).stat.meaninertia
    eng, ref = np.zeros((B, NVW)), np.zeros((B, 12))
    for i in range(B):
        R = objs[i, 3:12].reshape(3, 3)
        if case == "rigid":
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            ref[i] = np.r_[rng.normal(0, 1, 3) * sv, ax * rng.normal(0, sw), rng.normal(0, 2, 3) * sv,
                           ax * rng.normal(0, 2 * sw)]
            eng[i, 0:6], eng[i, 9:15] = ref[i, 0:6], ref[i, 6:12]
        elif case == "conveyor":
            s = np.r_[rng.normal(0, sv[0], 2), 0.0]
            sd = np.r_[rng.normal(0, 2 * sv[0], 2), 0.0]
            eng[i, 6:9], eng[i, 15:18], ref[i, :3], ref[i, 6:9] = s, sd, R @ s, R @ sd
    return mr, (nq, nv, nu), q, v, ctrl, warm, objs, eng, ref, mu


def ref_state(mr, dims, q, v, objs, ref):
    """MuJoCo's (qpos, qvel, qfrc_applied) of the reference model: the robot's, then the slot body's free joint (pose;
    linear velocity of its origin in world, angular in its own frame) and the force that gives it the acceleration
    (a, al): HEAVY a and HEAVY R^T al (isotropic inertia; gravity compensated)."""
    nq, nv, _ = dims
    B = len(q)
    qq, vv, qf = np.zeros((B, mr.nq)), np.zeros((B, mr.nv)), np.zeros((B, mr.nv))
    qq[:, :nq], vv[:, :nv] = q, v
    for i in range(B):
        R = objs[i, 3:12].reshape(3, 3)
        qq[i, nq:nq + 3], qq[i, nq + 3:nq + 7] = objs[i, :3], mat2quat(R)
        vv[i, nv:nv + 3], vv[i, nv + 3:nv + 6] = ref[i, :3], R.T @ ref[i, 3:6]
        qf[i, nv:nv + 3], qf[i, nv + 3:nv + 6] = HEAVY * ref[i, 6:9], HEAVY * (R.T @ ref[i, 9:12])
    return qq, vv, qf


def cmd_mjw(a):
    import jax.numpy as jnp
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    ffi, info, m, mu_in = engine_init(a.scene)
    res = dict(check="mjw", scene=a.scene, engine=info["engine"], nworld=a.nworld, iterations=a.iterations,
               substeps=a.substeps, heavy_mass=HEAVY, cases={})
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    for case in ("rigid", "conveyor"):
        mr, dims, q, v, ctrl, warm, objs, eng, ref, mu = random_states(a.scene, a.nworld, a.seed, case)
        nq, nv, nu = dims
        mr.opt.iterations = a.iterations
        B = len(q)
        out = {}
        for label, ref_v6, eng_w in (("moving", ref, eng), ("at rest", 0 * ref, 0 * eng),
                                     ("engine velocity zeroed", ref, 0 * eng)):
            qq, vv, qf = ref_state(mr, dims, q, v, objs, ref_v6)
            mw = mjw.put_model(mr)
            dw = mjw.put_data(mr, mujoco.MjData(mr), nworld=B, nconmax=64, njmax=300)
            wq = np.zeros((B, mr.nv)); wq[:, :nv] = warm
            for x, y in ((dw.qpos, qq), (dw.qvel, vv), (dw.qacc_warmstart, wq), (dw.qfrc_applied, qf)):
                wp.copy(x, wp.array(y.astype(np.float32)))
            if nu:
                wp.copy(dw.ctrl, wp.array(ctrl.astype(np.float32)))
            for _ in range(a.substeps):
                mjw.step(mw, dw)
            ref_out = dw.qvel.numpy()[:, :nv].astype(np.float64)
            nacon = int(dw.nacon.numpy()[0])
            ncon = np.bincount(dw.contact.worldid.numpy()[:nacon], minlength=B)
            r = ffi.forward(f32(q), f32(v), f32(warm), f32(ctrl), jnp.full(B, mu_in, jnp.float32),
                            f32(pack(objs, eng_w, B)), iterations=a.iterations, substeps=a.substeps)
            ev = np.asarray(r[1], np.float64)
            en = np.asarray(r[3])[:, 1]
            e = np.linalg.norm(ev - ref_out, axis=1) / np.maximum(np.linalg.norm(ref_out, axis=1), 1e-6)
            w = ncon > 0
            out[label] = dict(qvel_rel_median=float(np.median(e[w])), qvel_rel_p90=float(np.quantile(e[w], 0.9)),
                              qvel_rel_p99=float(np.quantile(e[w], 0.99)), worlds_with_contacts=int(w.sum()),
                              ncon_equal=float((en[w] == ncon[w]).mean()), ncon_mean=float(ncon[w].mean()))
            print(f"{a.scene} {case:9s} {label:24s} qvel rel median {out[label]['qvel_rel_median']:.2e} p99 "
                  f"{out[label]['qvel_rel_p99']:.2e} ncon equal {out[label]['ncon_equal']:.3f} ({w.sum()} worlds)")
        res["cases"][case] = out
    dump(res, a.out)


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def tangent(g, q, nq):
    """qpos-form cotangent g (nq,) of a robot whose first joint is free -> the dof tangent (nv,): MuJoCo's
    mj_integratePos (the root rotation in the body frame, q exp(delta / 2))."""
    gq = g[3:7]
    rot_ = [0.5 * gq @ qmul(q[3:7], np.r_[0, np.eye(3)[k]]) for k in range(3)]
    return np.r_[g[:3], rot_, g[7:nq]]


def cmd_grad(a):
    import jax
    import jax.numpy as jnp
    import mujoco
    ffi, info, m, mu_in = engine_init(a.scene)
    res = dict(check="grad", scene=a.scene, engine=info["engine"], nworld=a.nworld, eps=a.eps,
               iterations=a.iterations, mode="clamp_kv 1, normal 0 (exact)", cases={})
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    o = info["com_layout"]["object_velocity"][0]
    for case, sub, kind in (("rigid, 1 substep", 1, "rigid"), ("conveyor, 4 substeps", 4, "conveyor"),
                            ("rigid, 4 substeps", 4, "rigid"), ("at rest, 4 substeps", 4, "rest")):
        mr, dims, q, v, ctrl, warm, objs, eng, ref, mu = random_states(a.scene, a.nworld, a.seed + 7, kind)
        nq, nv, nu = dims
        B = len(q)
        rng = np.random.default_rng(a.seed + 11)
        yq, yv = rng.normal(size=(B, nq)), rng.normal(size=(B, nv))
        mr.opt.iterations, mr.opt.tolerance, mr.opt.ls_iterations = 100, 1e-12, 50
        d = mujoco.MjData(mr)

        def loss(qq, vv, uu, i, qf=None):
            d.qpos[:], d.qvel[:] = qq, vv
            d.qfrc_applied[:] = qf0[i] if qf is None else qf
            if nu:
                d.ctrl[:] = uu
            d.qacc_warmstart[:] = 0
            n = []
            for _ in range(sub):
                mujoco.mj_forward(mr, d)
                n.append((int(d.ncon), tuple(d.efc_state[:d.nefc].tolist())))
                mujoco.mj_step(mr, d)
            return float(yq[i] @ d.qpos[:nq] + yv[i] @ d.qvel[:nv]), tuple(n)

        def L(qa, va, ua, ca):
            q1, v1, _ = ffi.step(qa, va, f32(warm), ua, jnp.full(B, mu_in, jnp.float32), ca, iterations=a.iterations,
                                 substeps=sub, clamp_kv=1, normal=0)
            return jnp.sum(f32(yq) * q1) + jnp.sum(f32(yv) * v1)

        g = jax.grad(L, (0, 1, 2, 3))(f32(q), f32(v), f32(ctrl), f32(pack(objs, eng, B)))
        fw = ffi.forward(f32(q), f32(v), f32(warm), f32(ctrl), jnp.full(B, mu_in, jnp.float32), f32(pack(objs, eng, B)),
                         iterations=a.iterations, substeps=sub)
        e_v1 = np.asarray(fw[1], np.float64)
        gq, gv, gu, gc = (np.asarray(x, np.float64) for x in g)
        gvel = gc[:, o:o + NVW]
        qq0, vv0, qf0 = ref_state(mr, dims, q, v, objs, ref)
        names = ["qpos", "qvel"] + (["ctrl"] if nu else []) + {
            "rigid": ["slot_v", "slot_w", "slot_a", "slot_al"], "conveyor": ["slot_s", "slot_sd"],
            "rest": ["slot_v", "slot_w", "slot_a", "slot_al"]}[kind]
        blocks = {k: ([], []) for k in names}
        stable, model_ok = np.ones(B, bool), np.zeros(B, bool)
        for i in range(B):
            _, n0 = loss(qq0[i], vv0[i], ctrl[i], i)
            cv = d.qvel[:nv].copy()
            model_ok[i] = np.linalg.norm(cv - e_v1[i]) <= 1e-4 * max(np.linalg.norm(cv), 1e-6)
            R = objs[i, 3:12].reshape(3, 3)

            def column(fn):
                vals = []
                for sgn in (1, -1):
                    qq_, vv_, uu_, qf_ = fn(sgn * a.eps)
                    lv, n = loss(qq_, vv_, uu_, i, qf_)
                    vals.append(lv)
                    stable[i] &= n == n0
                return (vals[0] - vals[1]) / (2 * a.eps)

            def dq(k):
                def f(e):
                    qp = qq0[i].copy(); dl = np.zeros(mr.nv); dl[k] = e
                    mujoco.mj_integratePos(mr, qp, dl, 1.0)
                    return qp, vv0[i], ctrl[i], None
                return f

            def dv(k):
                def f(e):
                    vp = vv0[i].copy(); vp[k] += e
                    return qq0[i], vp, ctrl[i], None
                return f

            def du(k):
                def f(e):
                    up = ctrl[i].copy(); up[k] += e
                    return qq0[i], vv0[i], up, None
                return f

            def dslot(k):
                def f(e):
                    vp, qf = vv0[i].copy(), qf0[i].copy()
                    vw = ref[i].copy()
                    if kind == "conveyor":
                        if k < 6:
                            vw[:3] += R @ (np.eye(3)[k] * e)
                        else:
                            vw[6:9] += R @ (np.eye(3)[k - 6] * e)
                    else:
                        vw[k] += e
                    vp[nv:nv + 3], vp[nv + 3:nv + 6] = vw[:3], R.T @ vw[3:6]
                    qf[nv:nv + 3], qf[nv + 3:nv + 6] = HEAVY * vw[6:9], HEAVY * (R.T @ vw[9:12])
                    return qq0[i], vp, ctrl[i], qf
                return f

            fd = dict(qpos=[column(dq(k)) for k in range(nv)], qvel=[column(dv(k)) for k in range(nv)])
            eng_g = dict(qpos=tangent(gq[i], q[i], nq), qvel=gv[i])
            if nu:
                fd["ctrl"], eng_g["ctrl"] = [column(du(k)) for k in range(nu)], gu[i]
            if kind == "conveyor":
                fd["slot_s"], eng_g["slot_s"] = [column(dslot(k)) for k in range(2)], gvel[i, 6:8]
                fd["slot_sd"], eng_g["slot_sd"] = [column(dslot(6 + k)) for k in range(2)], gvel[i, 15:17]
            else:
                fd["slot_v"], eng_g["slot_v"] = [column(dslot(k)) for k in range(3)], gvel[i, 0:3]
                fd["slot_w"], eng_g["slot_w"] = [column(dslot(3 + k)) for k in range(3)], gvel[i, 3:6]
                fd["slot_a"], eng_g["slot_a"] = [column(dslot(6 + k)) for k in range(3)], gvel[i, 9:12]
                fd["slot_al"], eng_g["slot_al"] = [column(dslot(9 + k)) for k in range(3)], gvel[i, 12:15]
            for k in blocks:
                blocks[k][0].append(eng_g[k]); blocks[k][1].append(np.asarray(fd[k]))
        use = stable & model_ok
        out = dict(substeps=sub, stable_worlds=int(stable.sum()), model_ok_worlds=int(model_ok.sum()),
                   worlds=int(use.sum()))
        for k, (gg, ff) in blocks.items():
            gg, ff = np.asarray(gg)[use], np.asarray(ff)[use]
            e = np.linalg.norm(gg - ff, axis=1) / np.maximum(np.linalg.norm(ff, axis=1), 1e-9)
            cs = np.sum(gg * ff, 1) / np.maximum(np.linalg.norm(gg, axis=1) * np.linalg.norm(ff, axis=1), 1e-30)
            cs = cs[np.linalg.norm(ff, axis=1) > 1e-9]
            if not len(cs):
                cs = np.ones(1)
            out[k] = dict(rel_median=float(np.median(e)), rel_p90=float(np.quantile(e, 0.9)), rel_max=float(e.max()),
                          cos_min=float(cs.min()), fd_norm_median=float(np.median(np.linalg.norm(ff, axis=1))))
            print(f"{a.scene} {case:22s} {k:7s} rel median {out[k]['rel_median']:.2e} p90 {out[k]['rel_p90']:.2e} "
                  f"max {out[k]['rel_max']:.2e} cos min {out[k]['cos_min']:.6f} ({use.sum()} worlds)")
        res["cases"][case] = out
    dump(res, a.out)



def cmd_time(a):
    import jax
    import jax.numpy as jnp
    ffi, info, m, mu_in = engine_init("g1", velocity=not a.off)
    _, dims, q, v, ctrl, warm, objs, eng, ref, mu = random_states("g1", a.nworld, a.seed, "rigid")
    B = len(q)
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    com = f32(pack(objs, eng, B, velocity=not a.off))
    args = [f32(q), f32(v), f32(warm), f32(ctrl), jnp.full(B, mu_in, jnp.float32), com]
    fwd = jax.jit(lambda *x: ffi.step(*x)[:2])

    def L(qa, va, ua, ca):
        q1, v1, _ = ffi.step(qa, va, args[2], ua, args[4], ca)
        return jnp.sum(q1) + jnp.sum(v1)

    bwd = jax.jit(jax.grad(L, (0, 1, 2)))

    def tm(f, *x):
        jax.block_until_ready(f(*x))
        ts = []
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            jax.block_until_ready(f(*x))
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)) * 1e3

    import subprocess
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i",
                          os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]], capture_output=True,
                         text=True).stdout.strip()
    res = dict(check="time", object_velocity=not a.off, nworld=B, substeps=4, iterations=10, gpu=gpu,
               forward_ms=tm(fwd, *args), forward_backward_ms=tm(bwd, args[0], args[1], args[3], args[5]))
    print(json.dumps(res))
    dump(res, a.out)


def dump(res, out):
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1) + "\n")


def cmd_merge(a):
    res = dict(date=time.strftime("%Y-%m-%d"), parts=[json.loads(Path(p).read_text()) for p in a.inputs])
    dump(res, a.out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("conveyor")
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--out")
    p = sp.add_parser("mjw")
    p.add_argument("--scene", choices=("box", "capsule", "g1"), default="box")
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--nworld", type=int, default=256)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p = sp.add_parser("grad")
    p.add_argument("--scene", choices=("box", "capsule", "g1"), default="box")
    p.add_argument("--nworld", type=int, default=32)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p = sp.add_parser("time")
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--off", action="store_true")
    p.add_argument("--out")
    p = sp.add_parser("merge")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(conveyor=cmd_conveyor, mjw=cmd_mjw, grad=cmd_grad, time=cmd_time, merge=cmd_merge)[a.cmd](a)


if __name__ == "__main__":
    main()

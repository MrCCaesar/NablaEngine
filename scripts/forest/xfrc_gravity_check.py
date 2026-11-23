"""Per-world gravity and external wrenches (PhysicsParams fields gravity and xfrc, forest_engine.world_forces) on the
forest engine.

Scene of the one-step checks: scripts/forest/models/g1box.xml (the G1 of fixtures/mjlab and a free box paired with
every robot geom and the floor; the forest engine), wrenches on the bodies WRENCH_BODIES (torso_link,
left_wrist_yaw_link, box). Per world of the corpus: the G1's state from a G1 corpus (--g1-corpus, default
nablaengine.fixtures' default corpus, mf_s0: MuJoCo C rollouts) with the box at rest on the floor (1 mm penetration)
1 m in front of the pelvis (x), a gravity vector g = (0, 0, -9.81) + U(-3, 3)^3, and per wrench body a force
N(0, 40 N) (box: N(0, 10 N)), a torque N(0, 4 N m) and a point (body frame) body_ipos + U(-0.1, 0.1)^3. --lift H
raises the robot and the box by H m (no contacts: the smooth dynamics alone).

The references apply the wrench as MuJoCo applies a point force: before each substep the point's world position from
the substep's kinematics (xpos + xmat p), then xfrc_applied = (f, t + (point - xipos) x f) (mj_applyFT's force at the
COM), and opt.gravity the world's.

Commands (one engine library per process):
  corpus   the corpus's states and per-world values (--out NPZ)
  ref      (MuJoCo Warp) one control step (4 substeps) per world with its gravity (batched opt.gravity) and wrenches:
           qpos', qvel', contact counts; --fields: only these fields applied (the others at the model's values)
  run      (JAX) the engine: --mode plain (no parameters) | defaults (the fields at the model's values, zero wrenches,
           points at the COM) | random (the corpus's); --fields gravity xfrc; forward (qpos', qvel', info, contacts) and
           the backward of fixed random cotangents (g_qpos, g_qvel, g_ctrl, g_params); --dump NPZ
  compare  bitwise equality of two dumps
  fwd      a dump against REF: qvel relative error (|dv| / |v|, robot_check style), qpos error, contact counts
  grad64   (MuJoCo C float64) a dump's VJPs against central differences along random directions: the wrench (all
           9 k values), gravity, qpos (no quaternions), qvel, ctrl; --nworld worlds
  support  the support-acceleration equivalence over --seconds (default 1 s): a free box on a floor box that
           accelerates at a (the floor on three slide joints, masses 1e8 kg each, gravcomp 1, driven by an external
           force) against the box on a static floor with gravity g - a. In MuJoCo C (float64) and in the engine (the
           floor driven through xfrc, the static case through gravity); compared in the floor's frame: the box's
           position, orientation and velocity relative to the floor; also the engine's changed-gravity run against
           MuJoCo C's and MuJoCo Warp's (the forward reference)
  time     (JAX) the step (forward; forward + backward) per call at --nworld worlds: --mode plain or random

  python scripts/forest/xfrc_gravity_check.py corpus --out W/corpus.npz
  python scripts/forest/xfrc_gravity_check.py ref --corpus W/corpus.npz --out W/ref.npz      (MuJoCo Warp's Python)
  python scripts/forest/xfrc_gravity_check.py run --mode random --corpus W/corpus.npz --dump W/D.npz
  python scripts/forest/xfrc_gravity_check.py fwd --corpus W/corpus.npz --ref W/ref.npz --dump W/D.npz
  python scripts/forest/xfrc_gravity_check.py grad64 --corpus W/corpus.npz --dump W/D.npz --nworld 8
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
SCENE = REPO / "scripts" / "forest" / "models" / "g1box.xml"
WRENCH_BODIES = ("torso_link", "left_wrist_yaw_link", "box")
ITER, SUB = 50, 4
G0 = np.array([0.0, 0.0, -9.81])


INTEGRATOR = None


GRAVCOMP = False
GRAVCOMP_BODIES = ("box", "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
                   "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link")


def base_model():
    import mujoco
    from nablaengine.envs import convert
    xml = SCENE.read_text()
    if GRAVCOMP:
        sp = mujoco.MjSpec.from_string(xml)
        for b in GRAVCOMP_BODIES:
            sp.body(b).gravcomp = 0.5
        xml = sp.to_xml()
    m = convert.to_engine(xml).model
    if GRAVCOMP:
        assert np.count_nonzero(m.body_gravcomp) == len(GRAVCOMP_BODIES), "gravcomp lost in the conversion"
    if INTEGRATOR == "euler":
        m.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
    return m


def body_ids(m, names=WRENCH_BODIES):
    import mujoco
    return [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]


def free_quat(m):
    import mujoco
    return [int(m.jnt_qposadr[j]) + 3 + k for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            for k in range(4)]


def free_pos(m):
    import mujoco
    return [int(m.jnt_qposadr[j]) + 2 for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]


def xfrc_of(m, d, ids, x):
    """MuJoCo's xfrc_applied (nbody, 6) of the point-form wrenches x (k, 9) at d's kinematics (mj_applyFT's
    conversion: the force at the COM, the torque plus (point - xipos) x force)."""
    out = np.zeros((m.nbody, 6))
    for k, b in enumerate(ids):
        f, t, p = x[k, :3], x[k, 3:6], x[k, 6:]
        pw = d.xpos[b] + d.xmat[b].reshape(3, 3) @ p
        out[b, :3] = f
        out[b, 3:] = t + np.cross(pw - d.xipos[b], f)
    return out


def g1_box_states(m, B, corpus=None):
    """(qpos, qvel, qacc_warmstart, ctrl) of B worlds of the G1 + box scene: the G1's states from a G1 corpus
    (nablaengine.fixtures.load_corpus; the scene's robot is the corpus model's, its coordinates first), the box upright
    on the floor (1 mm penetration) 1 m in front of the pelvis (world x), at rest."""
    import mujoco
    from nablaengine import fixtures
    z = fixtures.load_corpus(corpus)
    idx = np.arange(B) % len(z["qpos"])
    nq0, nv0 = z["qpos"].shape[1], z["qvel"].shape[1]
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
    qa, da = int(m.jnt_qposadr[m.body_jntadr[b]]), int(m.jnt_dofadr[m.body_jntadr[b]])
    assert (qa, da) == (nq0, nv0) and m.nq == nq0 + 7 and m.nu == z["ctrl"].shape[1], "scene / corpus mismatch"
    q = np.zeros((B, m.nq))
    q[:, :nq0] = z["qpos"][idx]
    q[:, qa:qa + 3] = q[:, :3] + np.array([1.0, 0.0, 0.0])
    q[:, qa + 2] = m.geom_size[m.body_geomadr[b], 2] - 1e-3
    q[:, qa + 3] = 1.0
    v, w = np.zeros((B, m.nv), np.float32), np.zeros((B, m.nv), np.float32)
    v[:, :nv0], w[:, :nv0] = z["qvel"][idx], z["qacc_warmstart"][idx]
    return q, v, w, np.asarray(z["ctrl"][idx], np.float32)


def cmd_corpus(a):
    m = base_model()
    rng = np.random.default_rng(a.seed)
    B = a.nworld
    q, qv, qw, qc = g1_box_states(m, B, a.g1_corpus)
    if a.lift:
        q[:, free_pos(m)] += a.lift
    ids = body_ids(m)
    grav = G0 + rng.uniform(-3, 3, (B, 3))
    x = np.zeros((B, len(ids), 9))
    for k, b in enumerate(ids):
        sf = 10.0 if WRENCH_BODIES[k] == "box" else 40.0
        x[:, k, :3] = rng.normal(0, sf * a.scale, (B, 3))
        x[:, k, 3:6] = rng.normal(0, 4.0 * a.scale, (B, 3))
        x[:, k, 6:] = m.body_ipos[b] + rng.uniform(-0.1, 0.1, (B, 3))
    out = dict(qpos=q.astype(np.float32), qvel=qv, qacc_warmstart=qw, ctrl=qc,
               gravity=grav.astype(np.float32), xfrc=x.astype(np.float32), lift=np.float64(a.lift))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(json.dumps(dict(nworld=B, lift=a.lift, out=a.out)))


def values(z, fields, m):
    """(gravity (B, 3), wrenches (B, k, 9)) as the corpus's for the fields, the model's / none otherwise."""
    B = len(z["qpos"])
    g = np.asarray(z["gravity"], np.float64) if "gravity" in fields else np.tile(np.asarray(m.opt.gravity), (B, 1))
    x = np.asarray(z["xfrc"], np.float64) if "xfrc" in fields else np.zeros_like(z["xfrc"], np.float64)
    if "xfrc" not in fields:
        x[:, :, 6:] = m.body_ipos[body_ids(m)]
    return g, x


def cmd_ref(a):
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    z = np.load(a.corpus)
    m = base_model()
    m.opt.iterations = ITER
    B = len(z["qpos"])
    g, x = values(z, a.fields, m)
    ids = body_ids(m)
    mw = mjw.put_model(m)
    mw.opt.gravity = wp.array(g.astype(np.float32), dtype=wp.vec3)
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=64 * B, njmax=256 * B)
    for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
        wp.copy(getattr(dw, k), wp.array(np.ascontiguousarray(z[k], np.float32)))
    xf = np.zeros((B, m.nbody, 6), np.float32)
    for _ in range(SUB):
        mjw.kinematics(mw, dw)
        xp, xm, xi = dw.xpos.numpy(), dw.xmat.numpy(), dw.xipos.numpy()
        for k, b in enumerate(ids):
            f, t, p = x[:, k, :3], x[:, k, 3:6], x[:, k, 6:]
            pw = xp[:, b] + np.einsum("wij,wj->wi", xm[:, b].reshape(B, 3, 3).astype(np.float64), p)
            xf[:, b, :3] = f
            xf[:, b, 3:] = t + np.cross(pw - xi[:, b], f)
        wp.copy(dw.xfrc_applied, wp.array(xf, dtype=wp.spatial_vector))
        mjw.step(mw, dw)
    nacon = int(dw.nacon.numpy()[0])
    nc = np.bincount(dw.contact.worldid.numpy()[:nacon], minlength=B)
    np.savez(a.out, out_qpos=dw.qpos.numpy(), out_qvel=dw.qvel.numpy(), ncon=nc, fields=np.array(a.fields))
    print(json.dumps(dict(nworld=B, fields=a.fields, contacts_mean=float(nc.mean()), out=a.out)))


def params_for(fields):
    from nablaengine.params import PhysicsParams
    fields = tuple(fields)
    return PhysicsParams(fields, **({"xfrc_bodies": WRENCH_BODIES} if "xfrc" in fields else {}))


def com_rows(P, m, z, mode, info):
    B = len(z["qpos"])
    cw0 = int(info["com_words"]) - int(info.get("param_words", 0))
    vals = P.defaults(m, [])
    if mode == "random":
        g, x = values(z, P.fields, m)
        if "gravity" in vals:
            vals["gravity"] = g
        if "xfrc" in vals:
            vals["xfrc"] = x.reshape(B, -1)
    pw = P.pack(m.nv, **vals)
    return np.concatenate([np.zeros((B, cw0), np.float32), np.broadcast_to(pw, (B, pw.shape[1]))], 1)


def engine(a, m, z):
    if getattr(a, "api", "jax") == "torch":
        from nablaengine import torch_ffi as ffi
    else:
        from nablaengine import jax_ffi as ffi
    B = len(z["qpos"])
    if a.mode == "plain":
        info = ffi.init(model=m, foot_pairs=[])
        return info, np.zeros((B, int(info["com_words"])), np.float32), None
    P = params_for(a.fields)
    info = ffi.init(model=m, foot_pairs=[], params=P)
    return info, com_rows(P, m, z, a.mode, info), P


def torch_step(z, com, yq, yv):
    """(forward outputs, backward outputs) as numpy through torch_ffi (the library init() loaded)."""
    import torch
    from nablaengine import torch_ffi
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x, np.float32)).cuda()
    B = len(z["qpos"])
    fr = t(np.ones(B))
    r = torch_ffi.forward(t(z["qpos"]), t(z["qvel"]), t(z["qacc_warmstart"]), t(z["ctrl"]), fr, t(com),
                          iterations=ITER, substeps=SUB, tape=True, contacts=True)
    g = torch_ffi.backward(r[5], fr, t(com), t(yq), t(yv), substeps=SUB, clamp_kv=1, normal=0)
    return [x.cpu().numpy() for x in r[:6]], [x.cpu().numpy() for x in g]


def cmd_run(a):
    z = np.load(a.corpus)
    m = base_model()
    info, com, P = engine(a, m, z)
    B = len(z["qpos"])
    rng = np.random.default_rng(7)
    yq, yv = rng.standard_normal((B, m.nq)).astype(np.float32), rng.standard_normal((B, m.nv)).astype(np.float32)
    if a.cot == "qpos":
        yv[:] = 0.0
    elif a.cot == "qvel":
        yq[:] = 0.0
    if a.api == "torch":
        (q1, v1, w1, inf, con, tape), g = torch_step(z, com, yq, yv)
    else:
        import jax.numpy as jnp
        from nablaengine import jax_ffi
        j = lambda x: jnp.asarray(np.ascontiguousarray(x, np.float32))
        fr = j(np.ones(B))
        r = jax_ffi.forward(j(z["qpos"]), j(z["qvel"]), j(z["qacc_warmstart"]), j(z["ctrl"]), fr, j(com),
                            iterations=ITER, substeps=SUB, tape=True, contacts=True)
        q1, v1, w1, inf, con, tape = (np.asarray(x) for x in r[:6])
        g = jax_ffi.backward(j(tape), fr, j(com), j(yq), j(yv), substeps=SUB, clamp_kv=1, normal=0)
    out = dict(q1=q1, v1=v1, w1=w1, info=inf, con=con, yq=yq, yv=yv, g_qpos=np.asarray(g[0]),
               g_qvel=np.asarray(g[1]), g_ctrl=np.asarray(g[2]), g_info=np.asarray(g[3]), com=com)
    if len(g) > 4:
        out.update(g_params=np.asarray(g[4]), g_friction=np.asarray(g[5]))
        out["layout"] = np.array(json.dumps({f: list(v) for f, v in P.layout(m.nv).items()}))
    np.savez(a.dump, **out)
    print(json.dumps(dict(mode=a.mode, fields=a.fields, nworld=B, engine=info["engine"],
                          flags=int((inf[:, 3] != 0).sum()), adj_flags=int((out["g_info"] != 0).sum()),
                          contacts_mean=float(inf[:, 1].mean()), dump=a.dump)))


def cmd_compare(a):
    x, y = np.load(a.a), np.load(a.b)
    res = {}
    keys = ["q1", "v1", "w1", "info", "con", "g_qpos", "g_qvel", "g_ctrl", "g_info"]
    keys += [k for k in ("g_params", "g_friction") if k in x.files and k in y.files]
    for k in keys:
        u, v = x[k], y[k]
        eq = np.array_equal(u.view(np.uint32) if u.dtype == np.float32 else u,
                            v.view(np.uint32) if v.dtype == np.float32 else v)
        res[k] = "bitwise" if eq else f"differ (max abs {float(np.abs(u.astype(np.float64) - v).max()):.3g}, worlds " \
            f"{int((~np.all(u == v, axis=tuple(range(1, u.ndim)))).sum())})"
    ok = all(v == "bitwise" for v in res.values())
    print(json.dumps(dict(a=a.a, b=a.b, bitwise=ok, fields=res), indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(bitwise=ok, fields=res), indent=1))
    sys.exit(0 if ok else 1)


def cmd_fwd(a):
    r, d = np.load(a.ref), np.load(a.dump)
    rv, ev = r["out_qvel"].astype(np.float64), d["v1"].astype(np.float64)
    rel = np.linalg.norm(ev - rv, axis=1) / np.maximum(np.linalg.norm(rv, axis=1), 1e-6)
    res = dict(nworld=len(rel), qvel_rel_median=float(np.median(rel)), qvel_rel_p90=float(np.quantile(rel, 0.9)),
               qvel_rel_p99=float(np.quantile(rel, 0.99)), qvel_rel_max=float(rel.max()),
               qpos_abs_max=float(np.abs(d["q1"] - r["out_qpos"]).max()),
               ncon_equal=float((d["info"][:, 1] == r["ncon"]).mean()), contacts_mean=float(r["ncon"].mean()),
               flags=int((d["info"][:, 3] != 0).sum()))
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def c_step(m, d, ids, x):
    import mujoco
    for _ in range(SUB):
        mujoco.mj_kinematics(m, d)
        d.xfrc_applied[:] = xfrc_of(m, d, ids, x)
        mujoco.mj_step(m, d)


def cmd_grad64(a):
    import mujoco
    z, d = np.load(a.corpus), np.load(a.dump)
    m = base_model()
    m.opt.iterations, m.opt.tolerance, m.opt.ls_tolerance = 100, 1e-12, 1e-12
    lay = json.loads(str(d["layout"])) if "layout" in d.files else {}
    ids = body_ids(m)
    g_all, x_all = values(z, list(lay), m)
    dd = mujoco.MjData(m)
    rng = np.random.default_rng(3)
    fq = free_quat(m)
    nk = len(ids)

    def run(i, q, v, u, g, x):
        mujoco.mj_resetData(m, dd)
        dd.qpos[:], dd.qvel[:], dd.ctrl[:] = q, v, u
        dd.qacc_warmstart[:] = z["qacc_warmstart"][i]
        m.opt.gravity[:] = g
        c_step(m, dd, ids, x.reshape(nk, 9))
        return dd.qpos.copy(), dd.qvel.copy()

    names = ["qpos", "qvel", "ctrl"] + [f for f in ("gravity", "xfrc") if f in lay]
    if "xfrc" in lay:
        names += ["xfrc_force", "xfrc_torque", "xfrc_point"]
    part = {"xfrc_force": slice(0, 3), "xfrc_torque": slice(3, 6), "xfrc_point": slice(6, 9)}
    rel = {k: [] for k in names}
    agree = []
    worlds = np.arange(len(z["qpos"]))
    if a.contact_free:
        worlds = worlds[(d["info"][:, 1] == 0) & (d["info"][:, 2] == 0)]
    for i in worlds[:a.nworld]:
        base = {"qpos": np.asarray(z["qpos"][i], np.float64), "qvel": np.asarray(z["qvel"][i], np.float64),
                "ctrl": np.asarray(z["ctrl"][i], np.float64), "gravity": g_all[i].copy(),
                "xfrc": x_all[i].reshape(-1).copy()}
        q1, v1 = run(i, base["qpos"], base["qvel"], base["ctrl"], base["gravity"], base["xfrc"])
        agree.append(np.linalg.norm(d["v1"][i] - v1) / max(np.linalg.norm(v1), 1e-6))
        for name in names:
            key = "xfrc" if name in part else name
            n = len(base[key])
            u = rng.standard_normal(n)
            if name == "qpos":
                u[fq] = 0.0
            if name in part:
                keep = np.zeros((nk, 9), bool)
                keep[:, part[name]] = True
                u[~keep.reshape(-1)] = 0.0
            u /= np.linalg.norm(u)
            p_, m_ = dict(base), dict(base)
            p_[key] = base[key] + a.eps * u
            m_[key] = base[key] - a.eps * u
            qp, vp = run(i, p_["qpos"], p_["qvel"], p_["ctrl"], p_["gravity"], p_["xfrc"])
            qm, vm = run(i, m_["qpos"], m_["qvel"], m_["ctrl"], m_["gravity"], m_["xfrc"])
            fd = (d["yq"][i] @ (qp - qm) + d["yv"][i] @ (vp - vm)) / (2 * a.eps)
            if key in ("gravity", "xfrc"):
                o, w = lay[key]
                gr = float(np.asarray(d["g_params"][i, o:o + w], np.float64) @ u)
            else:
                gr = float(np.asarray(d[f"g_{name}"][i], np.float64) @ u)
            rel[name].append(abs(fd - gr) / max(abs(fd), abs(gr), 1e-12))
    m.opt.gravity[:] = G0
    agree = np.asarray(agree) < a.agree
    res = dict(nworld=len(agree), forward_agree_share=float(agree.mean()), eps=a.eps)
    for name, x in rel.items():
        x = np.asarray(x)
        res[name] = dict(rel_median=float(np.median(x)), rel_p90=float(np.quantile(x, 0.9)),
                         rel_max=float(x.max()))
        if agree.any():
            res[name].update(agree_rel_median=float(np.median(x[agree])),
                             agree_rel_p90=float(np.quantile(x[agree], 0.9)))
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


SUPPORT = """<mujoco model="support_accel">
  <option timestep="0.002" integrator="implicitfast" cone="pyramidal"/>
  <worldbody>
    {floor}
    <body name="obj" pos="0.1 -0.05 0.0595">
      <freejoint name="obj"/>
      <geom name="obj" type="box" size="0.1 0.08 0.06" mass="1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator><motor name="m0" joint="{motor}" gear="1" ctrlrange="-1 1"/></actuator>
  <contact><pair geom1="floor" geom2="obj" friction="0.6 0.6 0.005 0.0001 0.0001"/></contact>
</mujoco>"""
FLOOR_STATIC = '<geom name="floor" type="box" size="2 2 0.1" pos="0 0 -0.1" contype="0" conaffinity="0"/>'
FLOOR_MOVING = """<body name="sx" gravcomp="1"><joint name="sx" type="slide" axis="1 0 0"/>
      <inertial pos="0 0 0" mass="{M}" diaginertia="{M} {M} {M}"/>
      <body name="sy" gravcomp="1"><joint name="sy" type="slide" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="{M}" diaginertia="{M} {M} {M}"/>
        <body name="floor" gravcomp="1"><joint name="sz" type="slide" axis="0 0 1"/>
          <inertial pos="0 0 -0.1" mass="{M}" diaginertia="{M} {M} {M}"/>
          <geom name="floor" type="box" size="2 2 0.1" pos="0 0 -0.1" contype="0" conaffinity="0" mass="0"/>
        </body>
      </body>
    </body>"""
SUPPORT_MASS = 1e8
ACCELS = {"stick": (1.5, -0.8, 2.0), "slide": (8.0, -3.0, 1.5), "glide": (7.5, -1.0, 0.0)}
OBJ_V0S = {"glide": (0.3, 0.0, 0.0, 0.0, 0.0, 0.0)}
TILT = (1.0, -2.0, 0.5)
OBJ_V0 = (0.3, 0.0, 0.0, 0.0, 0.0, 1.0)


def v0_of(a_vec):
    for k, v in ACCELS.items():
        if tuple(v) == tuple(a_vec):
            return OBJ_V0S.get(k, OBJ_V0)
    return OBJ_V0


def support_model(moving):
    from nablaengine.envs import convert
    xml = SUPPORT.format(floor=FLOOR_MOVING.format(M=SUPPORT_MASS) if moving else FLOOR_STATIC,
                         motor="sx" if moving else "obj")
    if not moving:
        xml = xml.replace('  <actuator><motor name="m0" joint="obj" gear="1" ctrlrange="-1 1"/></actuator>\n', "")
    m = convert.to_engine(xml).model
    m.opt.tolerance, m.opt.ls_tolerance, m.opt.iterations = 0.0, 1e-12, 100
    from nablaengine import engines
    engines.prefer_forest(m)
    return m


def floor_force(a):
    """The floor's external force for acceleration a: the slide chain's masses 3M, 2M, M along x, y, z (gravcomp
    cancels its gravity)."""
    return np.array([3 * SUPPORT_MASS * a[0], 2 * SUPPORT_MASS * a[1], SUPPORT_MASS * a[2]])


def rel_state(m, qpos, qvel, moving):
    """The box's state in the floor's frame: position, quaternion, linear velocity, angular velocity (local)."""
    o = 3 if moving else 0
    p, qt, v, w = qpos[..., o:o + 3], qpos[..., o + 3:o + 7], qvel[..., o:o + 3], qvel[..., o + 3:o + 6]
    if moving:
        p, v = p - qpos[..., 0:3], v - qvel[..., 0:3]
    return np.concatenate([p, qt * np.sign(qt[..., :1]), v, w], -1)


def support_c(a_vec, moving, steps, g=G0):
    """MuJoCo C float64 trajectory of the box state in the floor frame (steps + 1, 13); g: the world's gravity (the
    moving floor's gravcomp cancels it; the static floor's case takes g - a)."""
    import mujoco
    m = support_model(moving)
    d = mujoco.MjData(m)
    o = 3 if moving else 0
    d.qvel[o:o + 6] = v0_of(a_vec)
    if moving:
        bf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "floor")
        m.opt.gravity[:] = g
    else:
        m.opt.gravity[:] = np.asarray(g) - np.asarray(a_vec)
    traj = [rel_state(m, d.qpos.copy(), d.qvel.copy(), moving)]
    for _ in range(steps):
        if moving:
            d.xfrc_applied[bf, :3] = floor_force(a_vec)
        mujoco.mj_step(m, d)
        traj.append(rel_state(m, d.qpos.copy(), d.qvel.copy(), moving))
    return np.asarray(traj)


def support_mjw(a_vec, steps):
    """MuJoCo Warp trajectory of the static-floor model with gravity g - a (steps + 1, 13)."""
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    m = support_model(False)
    m.opt.gravity[:] = G0 - np.asarray(a_vec)
    d = mujoco.MjData(m)
    d.qvel[:6] = v0_of(a_vec)
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, d, nworld=1, nconmax=64, njmax=256)
    traj = [rel_state(m, d.qpos.copy(), d.qvel.copy(), False)]
    for _ in range(steps):
        mjw.step(mw, dw)
        traj.append(rel_state(m, dw.qpos.numpy()[0].astype(np.float64), dw.qvel.numpy()[0].astype(np.float64), False))
    return np.asarray(traj)


def support_engine(a_vec, moving, steps, sub, g=None):
    """The engine's trajectory (steps + 1, 13): the moving floor driven through xfrc (the floor body; g: also the
    gravity field at g, which the floor's gravcomp cancels; None: the model's gravity), or the static floor with the
    gravity field at g - a (g None: the model's); --sub substeps per call."""
    import jax.numpy as jnp
    import mujoco
    from nablaengine import jax_ffi
    from nablaengine.params import PhysicsParams
    m = support_model(moving)
    if moving:
        P = PhysicsParams(("xfrc",) + (("gravity",) if g is not None else ()), xfrc_bodies=("floor",))
    else:
        P = PhysicsParams(("gravity",))
    info = jax_ffi.init(model=m, foot_pairs=[], params=P)
    vals = P.defaults(m, [])
    if moving:
        x = vals["xfrc"].reshape(1, 9).astype(np.float64)
        x[0, :3] = floor_force(a_vec)
        x[0, 6:] = m.body_ipos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "floor")]
        vals["xfrc"] = x.reshape(-1)
        if g is not None:
            vals["gravity"] = np.asarray(g)
    else:
        vals["gravity"] = (G0 if g is None else np.asarray(g)) - np.asarray(a_vec)
    cw0 = int(info["com_words"]) - int(info["param_words"])
    com = np.concatenate([np.zeros((1, cw0), np.float32), P.pack(m.nv, **vals)], 1)
    d = mujoco.MjData(m)
    o = 3 if moving else 0
    d.qvel[o:o + 6] = v0_of(a_vec)
    j = lambda x: jnp.asarray(np.ascontiguousarray(np.asarray(x, np.float32)[None]))
    q, v, w = j(d.qpos), j(d.qvel), j(np.zeros(m.nv))
    u, fr, c = j(np.zeros(m.nu)), jnp.ones(1, jnp.float32), jnp.asarray(com)
    traj = [rel_state(m, d.qpos.copy(), d.qvel.copy(), moving)]
    for _ in range(steps // sub):
        q, v, w = jax_ffi.forward(q, v, w, u, fr, c, iterations=ITER, substeps=sub)[:3]
        traj.append(rel_state(m, np.asarray(q[0], np.float64), np.asarray(v[0], np.float64), moving))
    return np.asarray(traj)


VARIANTS = {"moving": (True, False), "static": (False, False), "moving_tilt": (True, True), "static_tilt": (False, True)}


def cmd_support_run(a):
    """One engine trajectory (one library per process): --accel NAME --variant V -> --out NPZ (traj)."""
    moving, tilt = VARIANTS[a.variant]
    g = G0 + np.asarray(TILT) if tilt else None
    np.savez(a.out, traj=support_engine(ACCELS[a.accel], moving, int(round(a.seconds / 0.002)), a.sub, g))


def engine_traj(a, name, variant, work):
    import subprocess
    out = Path(work) / f"{name}_{variant}.npz"
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "support_run", "--accel", name, "--variant", variant,
                    "--seconds", str(a.seconds), "--sub", str(a.sub), "--out", str(out)], check=True)
    return np.load(out)["traj"]


def traj_err(x, y):
    """Max over time of the position (m), orientation (quaternion components), velocity (m/s, rad/s) differences."""
    dx = np.abs(x - y)
    return dict(pos_abs_max=float(dx[:, :3].max()), quat_abs_max=float(dx[:, 3:7].max()),
                linvel_abs_max=float(dx[:, 7:10].max()), angvel_abs_max=float(dx[:, 10:13].max()),
                final_rel_pos=[float(t) for t in y[-1, :3]], max_err=float(dx.max()))


def cmd_support(a):
    import shutil
    import tempfile
    work = tempfile.mkdtemp(prefix="xfrc_support_")
    try:
        _support(a, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _support(a, work):
    steps = int(round(a.seconds / 0.002))
    res = dict(seconds=a.seconds, floor_mass_each=SUPPORT_MASS, accels={k: list(v) for k, v in ACCELS.items()},
               compared="the box's state in the floor's frame (position, quaternion, velocity, local angular velocity)")
    for name, acc in [(k, v) for k, v in ACCELS.items() if not a.cases or k in a.cases]:
        r = {}
        c_mov, c_grav = support_c(acc, True, steps), support_c(acc, False, steps)
        r["mujoco_c_moving_floor_vs_gravity"] = traj_err(c_mov, c_grav)
        if a.engine:
            e_mov = engine_traj(a, name, "moving", work)
            e_grav = engine_traj(a, name, "static", work)
            sel = np.arange(0, steps + 1, a.sub)
            r["engine_moving_floor_vs_gravity"] = traj_err(e_mov, e_grav)
            r["engine_gravity_vs_mujoco_c_gravity"] = traj_err(e_grav, c_grav[sel])
            r["engine_moving_vs_mujoco_c_moving"] = traj_err(e_mov, c_mov[sel])
            gt = G0 + np.asarray(TILT)
            e_mt, e_gt = engine_traj(a, name, "moving_tilt", work), engine_traj(a, name, "static_tilt", work)
            r["tilted_gravity"] = list(gt)
            r["engine_moving_floor_vs_gravity_tilted"] = traj_err(e_mt, e_gt)
            r["mujoco_c_moving_floor_vs_gravity_tilted"] = traj_err(support_c(acc, True, steps, gt),
                                                                    support_c(acc, False, steps, gt))
            r["engine_gravity_tilted_vs_mujoco_c"] = traj_err(e_gt, support_c(acc, False, steps, gt)[sel])
            if a.mjw:
                w_grav = support_mjw(acc, steps)
                r["engine_gravity_vs_mjw_gravity"] = traj_err(e_grav, w_grav[sel])
                r["mjw_gravity_vs_mujoco_c_gravity"] = traj_err(w_grav, c_grav)
        res[name] = r
        print(json.dumps({name: r}, indent=1), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_time(a):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    z0 = dict(np.load(a.corpus))
    reps = (a.nworld + len(z0["qpos"]) - 1) // len(z0["qpos"])
    z = {k: np.concatenate([z0[k]] * reps)[:a.nworld] for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "gravity",
                                                                 "xfrc")}
    m = base_model()
    info, com, _ = engine(a, m, z)
    B = a.nworld
    j = lambda x: jnp.asarray(np.ascontiguousarray(x, np.float32))
    args = [j(z["qpos"]), j(z["qvel"]), j(z["qacc_warmstart"]), j(z["ctrl"]), j(np.ones(B)), j(com)]
    step = jax_ffi.make_step(iterations=a.iterations, substeps=SUB)
    fwd = jax.jit(lambda *x: step(*x)[:2])
    loss = jax.jit(jax.grad(lambda q, v, w, u, fr, c: jnp.sum(step(q, v, w, u, fr, c)[1] ** 2),
                            argnums=(0, 1, 3, 5)))
    res = dict(mode=a.mode, fields=a.fields, nworld=B, iterations=a.iterations, gpu=jax.devices()[0].device_kind)
    for name, fn in (("forward", fwd), ("forward_backward", loss)):
        jax.block_until_ready(fn(*args))
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            ts.append(time.perf_counter() - t0)
        res[f"{name}_ms_median"] = 1e3 * float(np.median(ts))
        res[f"{name}_ms_min"] = 1e3 * float(np.min(ts))
    print(json.dumps(res))
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")


def main():
    global SUB, INTEGRATOR, GRAVCOMP
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("corpus", "ref", "run", "compare", "fwd", "grad64", "support", "support_run", "time"))
    p.add_argument("a", nargs="?")
    p.add_argument("b", nargs="?")
    p.add_argument("--corpus", default="xfrc_gravity_corpus.npz")
    p.add_argument("--g1-corpus", default=None, help="corpus: the G1 corpus of the robot's states (default: "
                   "nablaengine.fixtures' default, mf_s0)")
    p.add_argument("--ref")
    p.add_argument("--dump")
    p.add_argument("--out")
    p.add_argument("--nworld", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lift", type=float, default=0.0)
    p.add_argument("--scale", type=float, default=1.0, help="corpus: the wrenches' scale")
    p.add_argument("--mode", choices=("plain", "defaults", "random"), default="random")
    p.add_argument("--fields", nargs="+", default=["gravity", "xfrc"], choices=("gravity", "xfrc", "none"))
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--agree", type=float, default=1e-3)
    p.add_argument("--integrator", choices=("euler",), help="ref / run / grad64: the scene with the Euler "
                   "integrator (MuJoCo C's implicitfast adds a velocity derivative of free bodies' gyroscopic force "
                   "that MuJoCo Warp's and the engine's do not have: its finite differences then differ for free bodies "
                   "under large applied torques)")
    p.add_argument("--gravcomp", action="store_true", help="ref / run / grad64: gravcomp 0.5 on the box and the "
                   "left arm (gravity's compensation of the world's gravity)")
    p.add_argument("--api", choices=("jax", "torch"), default="jax", help="run: the interface")
    p.add_argument("--cot", choices=("both", "qpos", "qvel"), default="both", help="run: the output cotangents")
    p.add_argument("--contact-free", action="store_true", help="grad64: only the worlds without constraint rows")
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--sub", type=int, default=4)
    p.add_argument("--engine", action="store_true", help="support: also the engine's runs (one process each)")
    p.add_argument("--cases", nargs="+", choices=("stick", "slide", "glide"), help="support: these cases (default all)")
    p.add_argument("--accel", choices=tuple(ACCELS), default="stick", help="support_run: the floor's acceleration")
    p.add_argument("--variant", choices=tuple(VARIANTS), default="static", help="support_run: the engine's run")
    p.add_argument("--mjw", action="store_true", help="support: also MuJoCo Warp's changed-gravity run")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--substeps", type=int, default=SUB, help="ref / run / grad64: substeps per control step")
    a = p.parse_args()
    SUB, INTEGRATOR, GRAVCOMP = a.substeps, a.integrator, a.gravcomp
    {"corpus": cmd_corpus, "ref": cmd_ref, "run": cmd_run, "compare": cmd_compare, "fwd": cmd_fwd,
     "grad64": cmd_grad64, "support": cmd_support, "support_run": cmd_support_run, "time": cmd_time}[a.cmd](a)


if __name__ == "__main__":
    main()

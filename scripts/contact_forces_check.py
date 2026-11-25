"""The contact-force outputs (init(contact_forces=, foot_wrench=, contact_frame=); g1_engine.contact_forces, on the
forest engine forest_engine.contact_forces) against float64 MuJoCo C, and their bitwise properties.

--engine forest runs the mjlab G1 on the forest engine (engines.prefer_forest, com_body torso_link as the tree engine)
and adds the scene multitree (scripts/contact_forces/multitree.xml: a box, a sphere, a capsule and a hinged arm, four
trees; foot_wrench on all four bodies; states from MuJoCo C trajectories). On the forest engine the default build
already has the last substep's 12-word table: compare checks it bitwise against the option builds' last substep.

Subcommands (one engine library per process: `run` once per build, `compare` reads the dumps):
  run      one build (--build, --api jax|torch, --scene mjlab|terrain, --params) on corpus states: one control step
           with the tape, the backward on the corpus cotangents, the contact output, and (--chain) the same control step
           as `substeps` one-substep calls (each substep's input state and its contact output); writes --out .npz
  compare  the dumps of a directory: bitwise (state, tape, gradients of every build the same as the default build's;
           the plain table the same as the last substep of the per-substep table and as the chained calls'; JAX = torch),
           then against MuJoCo C (mj_forward + mj_contactForce, float64, the engine's solver iterations) at the engine's
           own substep input states (from the chained calls: no trajectory divergence), and stepped substep by substep
           from the control step's input (mj_step; the worlds whose control step agrees): per matched contact the
           relative error of the normal force and of the tangential force (|df| / max(|f_ref|, floor)), the contact
           frame's tangent against MuJoCo's frame, per foot the wrench (force, moment about the body origin)
  time     forward (with the contact output) time at --nworld worlds, per build (medians)
  all      run + compare + time for the standard set (BUILDS x scenes x params) on the visible GPU
  xengine  the forest engine's full-build dumps against the tree engine's (--tree DIR --forest DIR): per substep the
           matched contacts' forces and the foot wrenches (forward-difference level agreement, not bitwise)

  PYTHONPATH=src python scripts/contact_forces_check.py all --dir runs/contact_forces/tree --out docs/results/contact_forces_v100.json
  PYTHONPATH=src python scripts/contact_forces_check.py all --engine forest --dir runs/contact_forces/forest --out ...
  PYTHONPATH=src python scripts/contact_forces_check.py xengine --tree runs/contact_forces/tree --forest runs/contact_forces/forest --out ...
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BUILDS = {
    "default": {},
    "last": dict(contact_forces=True),
    "feet": dict(foot_wrench=True),
    "sub": dict(contact_forces="substeps"),
    "full": dict(contact_forces="substeps", foot_wrench=True, contact_frame=True),
}
PARAMS = ("foot_solref", "foot_solimp", "dof_damping")
TERRAIN_SCENE = REPO / "fixtures" / "mjlab" / "g1_terrain_feet.mjb"
TERRAIN_REF = REPO / "fixtures" / "mjlab" / "mjw317_terrain_feet.npz"
MULTITREE_SCENE = REPO / "scripts" / "contact_forces" / "multitree.xml"
MULTITREE_BODIES = ("box", "ball", "rod", "finger")
FLOOR_FRAC = 1e-3


def engine_model(scene, engine):
    """(model, foot_pairs, init() keyword arguments for the model) of a scene on an engine."""
    import mujoco
    from nablaengine import engines, fixtures
    if scene == "multitree":
        m = mujoco.MjModel.from_xml_path(str(MULTITREE_SCENE))
        engines.prefer_forest(m)
        return m, [], dict(model=m, foot_pairs=[])
    m, mt = fixtures.load_model(fixtures.fixtures_dir(), check=None)
    if engine == "forest":
        engines.prefer_forest(m)
        return m, mt["foot_pairs"], dict(model=m, foot_pairs=mt["foot_pairs"],
                                         com_body=mt.get("torso_body", "torso_link"))
    return m, mt["foot_pairs"], {}


def build_kw(build, scene):
    kw = dict(BUILDS[build])
    if scene == "multitree" and kw.get("foot_wrench") is True:
        kw["foot_wrench"] = MULTITREE_BODIES
    return kw


def multitree_inputs(m, B, seed=0):
    """B states of the multitree scene from MuJoCo C trajectories (random ctrl, perturbed starts), float32."""
    import mujoco
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    q, v, w, u = [], [], [], []
    while len(q) < B:
        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.qpos0 + rng.normal(0, 0.01, m.nq) * (np.arange(m.nq) % 7 < 3)
        mujoco.mj_forward(m, d)
        nst = int(rng.integers(25, 750))
        c = rng.uniform(-1.5, 1.5, m.nu)
        for _ in range(nst):
            d.ctrl[:] = c
            mujoco.mj_step(m, d)
        q.append(d.qpos.copy()); v.append(d.qvel.copy()); w.append(d.qacc_warmstart.copy()); u.append(c.copy())
    f = lambda x: np.ascontiguousarray(np.asarray(x), np.float32)
    return f(q), f(v), f(w), f(u), np.ones(B, np.float32), np.zeros((B, 3), np.float32)



def inputs(scene, B, params, m, fp):
    """(qpos, qvel, warm, ctrl, friction, com (with the per-world parameters), param values or None) float32."""
    from nablaengine import fixtures
    if scene == "terrain":
        r = np.load(TERRAIN_REF)
        B = min(B, len(r["qpos"]))
        f = lambda k: np.ascontiguousarray(r[k][:B], np.float32)
        q, v, w, u = f("qpos"), f("qvel"), f("warm"), f("ctrl")
        fric = np.ones(B, np.float32)
        com = np.zeros((B, 3), np.float32)
        extra = dict(region=r["region"][:B])
    else:
        z = np.load(REPO / "benchmarks" / "data" / f"{scene}.npz")
        B = min(B, len(z["qpos"]))
        f = lambda k: np.ascontiguousarray(z[k][:B], np.float32)
        q, v, w, u, fric, com = f("qpos"), f("qvel"), f("qacc_warmstart"), f("ctrl"), f("friction"), f("com")
        extra = dict(cot_qpos=f("cot_qpos")[:, 0], cot_qvel=f("cot_qvel")[:, 0])
    pv = None
    if params:
        from nablaengine.params import PhysicsParams
        P = PhysicsParams(PARAMS)
        d = P.defaults(m, fp)
        rng = np.random.default_rng(3)
        pv = dict(foot_solref=np.asarray(d["foot_solref"], np.float32) * rng.uniform(0.8, 1.2, (B, 2)).astype(np.float32),
                  foot_solimp=np.broadcast_to(np.asarray(d["foot_solimp"], np.float32), (B, 4)).copy(),
                  dof_damping=(np.asarray(d["dof_damping"], np.float32)
                               + rng.uniform(0.0, 0.5, (B, m.nv)).astype(np.float32)))
        pv["dof_damping"][:, :6] = np.asarray(d["dof_damping"], np.float32)[:6]
        com = np.concatenate([com, P.pack(m.nv, **pv)], 1).astype(np.float32)
    return q, v, w, u, fric, com, pv, extra


def cmd_run(a):
    import mujoco
    from nablaengine import terrain as T
    m, fp, mkw = engine_model(a.scene, a.engine)
    kw = dict(build_kw(a.build, a.scene), **mkw)
    if a.params:
        kw["params"] = PARAMS
    cfg = None
    if a.scene == "terrain":
        cfg = T.TerrainConfig(geoms="feet", mode=a.mode)
        kw["terrain"] = cfg
    if a.scene == "multitree":
        q, v, w, u, fric, com = multitree_inputs(m, a.nworld)
        pv, extra = None, {}
    else:
        q, v, w, u, fric, com, pv, extra = inputs(a.scene, a.nworld, a.params, m, fp)
    B = len(q)
    if a.api == "jax":
        import jax
        import jax.numpy as jnp
        from nablaengine import jax_ffi as api
        info = api.init(**kw)
        if cfg is not None:
            api.set_terrain(T.Terrain.from_model(mujoco.MjModel.from_binary_path(str(TERRAIN_SCENE))))
        conv = jnp.asarray
        back = np.asarray
        sync = lambda x: jax.block_until_ready(x)
    else:
        import torch
        from nablaengine import torch_ffi as api
        info = api.init(**kw)
        if cfg is not None:
            api.set_terrain(T.Terrain.from_model(mujoco.MjModel.from_binary_path(str(TERRAIN_SCENE))))
        conv = lambda x: torch.from_numpy(np.ascontiguousarray(x)).cuda()
        back = lambda x: x.cpu().numpy()
        sync = lambda x: torch.cuda.synchronize()
    has_con = bool(info.get("contact_forces"))
    S = a.substeps
    X = [conv(x) for x in (q, v, w, u, fric, com)]
    r = api.forward(*X, iterations=a.iterations, substeps=S, tape=True, contacts=True)
    q1, v1, w1, inf, con, tape = (back(x) for x in r[:6])
    out = dict(q1=q1, v1=v1, w1=w1, info=inf, con=con,
               tape_md5=np.stack([np.frombuffer(hashlib.md5(row.tobytes()).digest(), np.uint8) for row in tape]),
               q=q, v=v, w=w, u=u, fric=fric, com=com, **extra)
    if pv:
        out.update({f"param_{k}": x for k, x in pv.items()})
    gq = extra.get("cot_qpos", np.ones_like(q1))
    gv = extra.get("cot_qvel", np.ones_like(v1))
    g = api.backward(r[5], X[4], X[5], conv(gq), conv(gv), substeps=S)
    for i, x in enumerate(g):
        out[f"grad{i}"] = back(x)
    if a.chain and has_con:
        cq, cv, cw, cc = [q], [v], [w], []
        x = X
        for k in range(S):
            rk = api.forward(*x[:4], X[4], X[5], iterations=a.iterations, substeps=1, contacts=True)
            cc.append(back(rk[4]))
            x = [rk[0], rk[1], rk[2], X[3]]
            if k < S - 1:
                cq.append(back(rk[0]))
                cv.append(back(rk[1]))
                cw.append(back(rk[2]))
        out.update(chain_q=np.stack(cq, 1), chain_v=np.stack(cv, 1), chain_w=np.stack(cw, 1),
                   chain_con=np.stack(cc, 1), chain_q1=back(x[0]), chain_v1=back(x[1]))
    sync(r[0])
    meta = {k: info.get(k) for k in ("maxcon", "con_words", "contact_forces", "nq", "nv", "npair", "terrain", "arch",
                                      "commit", "dirty", "key")}
    meta.update(build=a.build, api=a.api, scene=a.scene, params=a.params, substeps=S, iterations=a.iterations,
                mode=a.mode, npair=int(m.npair), engine=a.engine)
    out["meta"] = np.asarray(json.dumps(meta))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"run {a.engine} {a.build} {a.api} {a.scene} params={a.params}: {B} worlds -> {a.out}")



def cmd_time(a):
    import jax
    from nablaengine import jax_ffi as api
    m, fp, mkw = engine_model("mf_s0", a.engine)
    kw = dict(build_kw(a.build, "mf_s0"), **mkw)
    api.init(**kw)
    q, v, w, u, fric, com, _, _ = inputs("mf_s0", a.nworld, False, m, fp)
    X = [jax.device_put(x) for x in (q, v, w, u, fric, com)]
    res = {}
    for name, tape in (("forward", False), ("forward_tape", True)):
        f = jax.jit(lambda *x, tape=tape: api.forward(*x, iterations=a.iterations, substeps=a.substeps, tape=tape,
                                                       contacts=True))
        jax.block_until_ready(f(*X))
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            jax.block_until_ready(f(*X))
            ts.append(time.perf_counter() - t0)
        res[name + "_ms"] = float(np.median(ts) * 1e3)
    res.update(build=a.build, engine=a.engine, nworld=len(q), substeps=a.substeps, iterations=a.iterations, reps=a.reps,
               con_width=int(api.con_width(api.init(), a.substeps)))
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res) + "\n")



def dec(con, info, substeps):
    """g1_engine.contact_forces.decode with the pair ids as integers."""
    from nablaengine.g1_engine import contact_forces as cf
    out = cf.decode(con, info, substeps)
    out["pair"] = out["pair"].astype(np.int64)
    return out


def load(p):
    z = dict(np.load(p, allow_pickle=False))
    z["meta"] = json.loads(str(z["meta"]))
    return z


def bitwise(ref, x, keys):
    return {k: bool(np.array_equal(ref[k], x[k])) for k in keys if k in ref and k in x}


def mj_setup(scene, params):
    import mujoco
    from nablaengine import fixtures
    if scene == "multitree":
        mc = mujoco.MjModel.from_xml_path(str(MULTITREE_SCENE))
        m = mujoco.MjModel.from_xml_path(str(MULTITREE_SCENE))
        mc.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        return mc, m, dict(foot_pairs=[]), [], None
    if scene == "terrain":
        mc = mujoco.MjModel.from_binary_path(str(TERRAIN_SCENE))
    else:
        mc, _ = fixtures.load_model(fixtures.fixtures_dir(), check=None)
    m, mt = fixtures.load_model(fixtures.fixtures_dir(), check=None)
    mc.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    gname = lambda mm, g: mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM, g)
    foot_geoms = {gname(m, int(m.pair_geom2[p])) for p in mt["foot_pairs"]} | {gname(m, int(m.pair_geom1[p]))
                                                                                for p in mt["foot_pairs"]}
    foot_geoms.discard(gname(m, int(m.pair_geom1[mt["foot_pairs"][0]])))
    foot = [p for p in range(mc.npair) if gname(mc, int(mc.pair_geom1[p])) in foot_geoms
            or gname(mc, int(mc.pair_geom2[p])) in foot_geoms]
    foot = [p for p in foot if mc.geom_bodyid[mc.pair_geom1[p]] == 0 or mc.geom_bodyid[mc.pair_geom2[p]] == 0]
    torso = mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_BODY, mt.get("torso_body", "torso_link"))
    return mc, m, mt, foot, torso


def mj_contacts(mc, d, terrain):
    """MuJoCo C's contacts of the data's last forward: [(key, pos, force on key's second geom (world), normal, t1)]
    with key (geom name A, geom name B): B the robot geom of a terrain contact (A "terrain"), else (geom1, geom2)."""
    import mujoco
    out = []
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(mc, d, i, f6)
        fr = np.array(c.frame, np.float64).reshape(3, 3)
        fw = fr.T @ f6[:3]
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        n1, n2 = mc.geom(g1).name, mc.geom(g2).name
        if terrain and (mc.geom_bodyid[g1] == 0) != (mc.geom_bodyid[g2] == 0):
            if mc.geom_bodyid[g2] == 0:
                out.append((("terrain", n1), c.pos.copy(), -fw, -fr[0], fr[1], f6[0]))
            else:
                out.append((("terrain", n2), c.pos.copy(), fw, fr[0], fr[1], f6[0]))
        else:
            out.append(((n1, n2), c.pos.copy(), fw, fr[0], fr[1], f6[0]))
    return out


def engine_contacts(tab, k, w, names):
    """The engine's contacts of world w (substep k or the last table): [(key, pos, force, normal, t1|None, fn)]."""
    out = []
    sub = tab["pair"].ndim == 3
    sel = (lambda a: a[w, k]) if sub else (lambda a: a[w])
    pair = sel(tab["pair"])
    for c in np.nonzero(pair >= 0)[0]:
        out.append((names[int(pair[c])], sel(tab["pos"])[c], sel(tab["force"])[c], sel(tab["normal"])[c],
                    sel(tab["tangent1"])[c] if "tangent1" in tab else None, sel(tab["force_normal"])[c]))
    return out


def match(eng, ref, tol=2e-3):
    """Pairs (engine contact, MuJoCo contact) with the same key, nearest positions first (within tol m); unmatched
    counts."""
    used, pairs = set(), []
    for e in eng:
        best, bi = tol, None
        for i, r in enumerate(ref):
            if i in used:
                continue
            k = r[0]
            same = k == e[0]
            flip = (not same) and k == (e[0][1], e[0][0])
            if not (same or flip):
                continue
            dd = np.linalg.norm(r[1] - e[1])
            if dd < best:
                best, bi = dd, (i, flip)
        if bi is not None:
            used.add(bi[0])
            r = ref[bi[0]]
            if bi[1]:
                r = (e[0], r[1], -r[2], -r[3], r[4], r[5])
            pairs.append((e, r))
    return pairs, len(eng) - len(pairs), len(ref) - len(used)


def foot_wrench_ref(mc, d, foot_bodies, terrain):
    """Per foot body (scene ids) MuJoCo C's summed force and moment about the body origin, and the normal-force sum."""
    import mujoco
    out = np.zeros((len(foot_bodies), 7))
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(mc, d, i, f6)
        fr = np.array(c.frame, np.float64).reshape(3, 3)
        fw = fr.T @ f6[:3]
        b1, b2 = mc.geom_bodyid[c.geom[0]], mc.geom_bodyid[c.geom[1]]
        for j, b in enumerate(foot_bodies):
            s = 1.0 if b2 == b else (-1.0 if b1 == b else 0.0)
            if s:
                f = s * fw
                out[j, :3] += f
                out[j, 3:6] += np.cross(c.pos - d.xpos[b], f)
                out[j, 6] += f6[0]
    return out


def errs(eng, ref, floor):
    return np.linalg.norm(np.atleast_2d(eng - ref), axis=-1) / np.maximum(np.linalg.norm(np.atleast_2d(ref), axis=-1),
                                                                          floor)


def stats(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return None
    return dict(n=int(x.size), median=float(np.median(x)), p90=float(np.quantile(x, 0.9)),
                p99=float(np.quantile(x, 0.99)), max=float(x.max()))


def ref_compare(z, scene, params, nmax, stepped):
    """The engine's per-substep contact forces and foot wrenches (the "full" build's dump z) against MuJoCo C."""
    import mujoco
    from nablaengine.g1_engine import contact_forces as cf
    from nablaengine import terrain as T
    mc, m, mt, foot_pairs_scene, torso = mj_setup(scene, params)
    meta = z["meta"]
    info = dict(maxcon=meta["maxcon"], con_words=meta["con_words"], contact_forces=meta["contact_forces"])
    S = meta["substeps"]
    mc.opt.iterations = meta["iterations"]
    terrain = scene == "terrain"
    pg = cf.pair_geoms(m, mt["foot_pairs"], T.TerrainConfig(geoms="feet") if terrain else None)
    gname = lambda g: "terrain" if g < 0 else m.geom(int(g)).name
    names = [(gname(a), gname(b)) for a, b in pg]
    feet_names = meta["contact_forces"]["foot_bodies"]
    feet_scene = [mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_BODY, n) for n in feet_names]
    weight = float(sum(m.body_mass)) * 9.81
    floor_f, floor_m = FLOOR_FRAC * weight, FLOOR_FRAC * weight * 0.1
    full = dec(z["con"], info, S)
    chain = [dec(z["chain_con"][:, k], info, 1) for k in range(S)]
    d = mujoco.MjData(mc)
    ipos0 = mc.body_ipos[torso].copy() if torso is not None else None
    fric0, solref0, solimp0, damp0 = (mc.pair_friction.copy(), mc.pair_solref.copy(), mc.pair_solimp.copy(),
                                      mc.dof_damping.copy())
    foot_all = sorted(foot_pairs_scene)
    B = min(nmax, len(z["q"]))
    acc = {k: [] for k in ("fn", "ft", "f", "t1_angle", "wf", "wm")}
    accs = {k: [] for k in ("fn", "ft", "f", "wf", "wm")}
    per_region = {}
    nmatch = nun_e = nun_r = 0
    worlds_all_matched = 0
    agree = 0
    for w in range(B):
        if torso is not None:
            mc.body_ipos[torso] = ipos0 + z["com"][w, :3]
        mc.pair_friction[:] = fric0
        mc.pair_friction[foot_all, 0] = mc.pair_friction[foot_all, 1] = z["fric"][w]
        if params:
            mc.pair_solref[:], mc.pair_solimp[:], mc.dof_damping[:] = solref0, solimp0, damp0
            mc.pair_solref[foot_all] = z["param_foot_solref"][w]
            mc.pair_solimp[foot_all, :4] = z["param_foot_solimp"][w]
            mc.dof_damping[:] = z["param_dof_damping"][w]
        all_ok = True
        for k in range(S):
            d.qpos[:], d.qvel[:], d.qacc_warmstart[:], d.ctrl[:] = (z["chain_q"][w, k], z["chain_v"][w, k],
                                                                    z["chain_w"][w, k], z["u"][w])
            mujoco.mj_forward(mc, d)
            ref = mj_contacts(mc, d, terrain)
            eng = engine_contacts(chain[k], 0, w, names)
            pairs, ue, ur = match(eng, ref)
            nmatch += len(pairs)
            nun_e += ue
            nun_r += ur
            all_ok &= ue == 0 and ur == 0
            reg = int(z["region"][w]) if "region" in z else -1
            for e, r in pairs:
                fn_e, fn_r = e[5], r[5]
                acc["fn"].append(abs(fn_e - fn_r) / max(abs(fn_r), floor_f))
                n = r[3]
                te, tr = e[2] - np.dot(e[2], n) * n, r[2] - np.dot(r[2], n) * n
                acc["ft"].append(np.linalg.norm(te - tr) / max(np.linalg.norm(r[2]), floor_f))
                acc["f"].append(np.linalg.norm(e[2] - r[2]) / max(np.linalg.norm(r[2]), floor_f))
                per_region.setdefault(reg, []).append(acc["f"][-1])
                if e[4] is not None:
                    acc["t1_angle"].append(float(np.arccos(np.clip(abs(np.dot(e[4], r[4])), -1, 1))))
            wr = foot_wrench_ref(mc, d, feet_scene, terrain)
            fe, me = full["foot_force"][w, k], full["foot_moment"][w, k]
            acc["wf"] += list(errs(fe, wr[:, :3], floor_f))
            acc["wm"] += list(errs(me, wr[:, 3:6], floor_m))
        worlds_all_matched += all_ok
        if stepped:
            d.qpos[:], d.qvel[:], d.qacc_warmstart[:], d.ctrl[:] = z["q"][w], z["v"][w], z["w"][w], z["u"][w]
            wrs, refs = [], []
            for k in range(S):
                mujoco.mj_step(mc, d)
                refs.append(mj_contacts(mc, d, terrain))
                wrs.append(foot_wrench_ref(mc, d, feet_scene, terrain))
            ev = np.linalg.norm(d.qvel - z["v1"][w]) / max(np.linalg.norm(d.qvel), 1e-6)
            if ev < 1e-3:
                agree += 1
                for k in range(S):
                    pairs, _, _ = match(engine_contacts(full, k, w, names), refs[k])
                    for e, r in pairs:
                        accs["fn"].append(abs(e[5] - r[5]) / max(abs(r[5]), floor_f))
                        n = r[3]
                        te, tr = e[2] - np.dot(e[2], n) * n, r[2] - np.dot(r[2], n) * n
                        accs["ft"].append(np.linalg.norm(te - tr) / max(np.linalg.norm(r[2]), floor_f))
                        accs["f"].append(np.linalg.norm(e[2] - r[2]) / max(np.linalg.norm(r[2]), floor_f))
                    accs["wf"] += list(errs(full["foot_force"][w, k], wrs[k][:, :3], floor_f))
                    accs["wm"] += list(errs(full["foot_moment"][w, k], wrs[k][:, 3:6], floor_m))
    res = dict(worlds=B, substeps=S, floor_force_N=floor_f, floor_moment_Nm=floor_m, matched=nmatch,
               unmatched_engine=nun_e, unmatched_mujoco=nun_r, worlds_all_matched=worlds_all_matched,
               same_state={k: stats(v) for k, v in acc.items()})
    if per_region and len(per_region) > 1:
        res["same_state_force_by_region"] = {str(k): stats(v) for k, v in sorted(per_region.items())}
    if stepped:
        res["stepped"] = dict(worlds_agree=agree, **{k: stats(v) for k, v in accs.items()})
    return res


def consistency(z):
    """Within the full build's dump: the per-substep tables and wrenches against the chained one-substep calls, the
    chained state against the control step's, each foot's normal-force maximum against the table, the force's
    decomposition into normal and the two tangents."""
    from nablaengine import fixtures, terrain as T
    from nablaengine.g1_engine import contact_forces as cf
    meta = z["meta"]
    info = dict(maxcon=meta["maxcon"], con_words=meta["con_words"], contact_forces=meta["contact_forces"])
    S = meta["substeps"]
    full = dec(z["con"], info, S)
    res = {}
    if "chain_con" in z:
        ch = [dec(z["chain_con"][:, k], info, 1) for k in range(S)]
        res["chain_tables_bitwise"] = bool(all(np.array_equal(full[key][:, k], ch[k][key][:, 0]) for k in range(S)
                                               for key in ("pair", "dist", "pos", "normal", "force", "force_normal")))
        res["chain_wrench_bitwise"] = bool(all(np.array_equal(full["foot_force"][:, k], ch[k]["foot_force"][:, 0])
                                               and np.array_equal(full["foot_moment"][:, k], ch[k]["foot_moment"][:, 0])
                                               for k in range(S)))
        res["chain_state_bitwise"] = bool(np.array_equal(z["chain_q1"], z["q1"])
                                          and np.array_equal(z["chain_v1"], z["v1"]))
    _, m, mt, _, _ = mj_setup(meta["scene"], False)
    pg = cf.pair_geoms(m, mt["foot_pairs"], T.TerrainConfig(geoms="feet") if meta["scene"] == "terrain" else None)
    body = np.where(pg >= 0, m.geom_bodyid[np.maximum(pg, 0)], 0)
    pair = np.maximum(full["pair"], 0)
    valid = full["pair"] >= 0
    fn_tab = np.stack([np.where(valid & ((body[pair, 0] == b) | (body[pair, 1] == b)), full["force_normal"], 0).sum(-1)
                       for b in meta["contact_forces"]["feet"]], -1)
    res["fn_max_rel_err_max"] = float(np.max(np.abs(fn_tab.max(1) - full["foot_fn_max"])
                                             / np.maximum(fn_tab.max(1), 1e-3)))
    if "tangent1" in full:
        res["force_decomposition_max_abs_N"] = float(np.max(np.abs(
            full["force"] - full["force_normal"][..., None] * full["normal"]
            - full["force_t1"][..., None] * full["tangent1"] - full["force_t2"][..., None] * full["tangent2"])))
        t1, n = full["tangent1"], full["normal"]
        res["frame_orthonormal_max_err"] = float(np.max(np.where(valid, np.abs(np.sum(t1 * n, -1))
                                                                 + np.abs(np.linalg.norm(t1, axis=-1) - 1), 0)))
    if "element" in full:
        el = full["element"]
        res["element_model_pairs_minus1"] = bool(np.all(el[valid & (full["pair"] < meta["npair"])] == -1))
        res["element_terrain_pairs_ge0"] = bool(np.all(el[valid & (full["pair"] >= meta["npair"])] >= 0))
        res["terrain_contacts"] = int(np.sum(valid & (full["pair"] >= meta["npair"])))
    return res


def cmd_compare(a):
    d = Path(a.dir)
    dumps = {p.stem: load(p) for p in sorted(d.glob("*.npz"))}
    res = dict(bitwise={}, consistency={}, mujoco={})
    keys = ("q1", "v1", "w1", "info", "tape_md5") + tuple(f"grad{i}" for i in range(6))
    for name, z in dumps.items():
        meta = z["meta"]
        base = f"default_{meta['api']}_{meta['scene']}" + ("_params" if meta["params"] else "")
        if name != base and base in dumps:
            res["bitwise"][name] = bitwise(dumps[base], z, keys)
            res["bitwise"][name]["all"] = all(res["bitwise"][name].values())
        if meta["api"] == "torch":
            jn = name.replace("_torch_", "_jax_")
            if jn in dumps:
                bt = bitwise(dumps[jn], z, keys + ("con",))
                res["bitwise"][f"{name}_vs_jax"] = dict(bt, all=all(bt.values()))
        if meta["build"] in ("last", "feet", "sub") and f"full_{name.split('_', 1)[1]}" in dumps:
            fz = dumps[f"full_{name.split('_', 1)[1]}"]
            from nablaengine.g1_engine import contact_forces as cf
            S = meta["substeps"]
            mine = dec(z["con"], dict(maxcon=meta["maxcon"], con_words=meta["con_words"],
                                            contact_forces=meta["contact_forces"]), S)
            fm = fz["meta"]
            full = dec(fz["con"], dict(maxcon=fm["maxcon"], con_words=fm["con_words"],
                                             contact_forces=fm["contact_forces"]), S)
            ok = all(np.array_equal(full[k] if meta["build"] == "sub" else full[k][:, -1], mine[k])
                     for k in ("pair", "dist", "pos", "normal", "force", "force_normal"))
            if "foot_force" in mine:
                ok &= all(np.array_equal(full[k], mine[k]) for k in ("foot_force", "foot_moment", "foot_fn_max"))
            res["bitwise"][f"{name}_table_vs_full"] = bool(ok)
        if meta.get("engine") == "forest" and meta["build"] == "full":
            base_name = f"default_{name.split('_', 1)[1]}"
            if base_name in dumps:
                dz, S = dumps[base_name], meta["substeps"]
                dm = dz["meta"]
                dflt = dec(dz["con"], dict(maxcon=dm["maxcon"], con_words=dm["con_words"],
                                           contact_forces=dm["contact_forces"]), S)
                full = dec(z["con"], dict(maxcon=meta["maxcon"], con_words=meta["con_words"],
                                          contact_forces=meta["contact_forces"]), S)
                res["bitwise"][f"{name}_last_vs_default_table"] = bool(all(
                    np.array_equal(full[k][:, -1], dflt[k]) for k in ("pair", "dist", "pos", "normal", "force",
                                                                     "force_normal")))
    for name, z in dumps.items():
        meta = z["meta"]
        if meta["build"] == "full":
            res["consistency"][name] = consistency(z)
            if meta["api"] == "jax":
                res["mujoco"][name] = ref_compare(z, meta["scene"], meta["params"], a.nref, a.stepped)
                print(name, json.dumps(res["mujoco"][name]["same_state"]), flush=True)
    if a.old_last:
        o = load(a.old_last)
        n = dumps.get("last_jax_mf_s0")
        if n is not None:
            res["bitwise"]["last_vs_base_commit"] = dict(table=bool(np.array_equal(o["con"], n["con"])),
                                                         **bitwise(o, n, keys))
    res["time"] = [json.loads(p.read_text()) for p in sorted(d.glob("time_*.json"))]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["bitwise"], indent=0))
    ok = all(v["all"] if isinstance(v, dict) and "all" in v else (v if isinstance(v, bool) else True)
             for v in res["bitwise"].values())
    print("bitwise:", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


def cmd_xengine(a):
    """The forest engine's full-build dumps against the tree engine's on the same inputs: per substep the contacts
    matched by geom pair and position (substep 0: the same input state; later substeps: each engine's own state, which
    differ at the forward-difference level) and the foot wrenches."""
    from nablaengine import fixtures
    from nablaengine.g1_engine import contact_forces as cf
    m, mt = fixtures.load_model(fixtures.fixtures_dir(), check=None)
    pg = cf.pair_geoms(m, mt["foot_pairs"])
    names = [(m.geom(int(x)).name, m.geom(int(y)).name) for x, y in pg]
    weight = float(sum(m.body_mass)) * 9.81
    floor_f, floor_m = FLOOR_FRAC * weight, FLOOR_FRAC * weight * 0.1
    res = {}
    for scene in a.scenes.split(","):
        pt, pf = Path(a.tree) / f"full_jax_{scene}.npz", Path(a.forest) / f"full_jax_{scene}.npz"
        if not (pt.exists() and pf.exists()):
            continue
        zt, zf = load(pt), load(pf)
        assert np.array_equal(zt["q"], zf["q"]) and np.array_equal(zt["u"], zf["u"]), "not the same inputs"
        S = zt["meta"]["substeps"]
        tt, tf = (dec(z["con"], dict(maxcon=z["meta"]["maxcon"], con_words=z["meta"]["con_words"],
                                     contact_forces=z["meta"]["contact_forces"]), S) for z in (zt, zf))
        B = min(a.nref, len(zt["q"]))
        r = dict(worlds=B, substeps=S,
                 state_rel_err_qvel=stats(np.linalg.norm(zf["v1"][:B] - zt["v1"][:B], axis=1)
                                          / np.maximum(np.linalg.norm(zt["v1"][:B], axis=1), 1e-6)))
        for k in range(S):
            acc = dict(f=[], fn=[], t1_angle=[])
            nm = ue = ur = 0
            for w in range(B):
                pairs, a1, a2 = match(engine_contacts(tf, k, w, names), engine_contacts(tt, k, w, names))
                nm += len(pairs)
                ue += a1
                ur += a2
                for e, t in pairs:
                    acc["f"].append(np.linalg.norm(e[2] - t[2]) / max(np.linalg.norm(t[2]), floor_f))
                    acc["fn"].append(abs(e[5] - t[5]) / max(abs(t[5]), floor_f))
                    if e[4] is not None and t[4] is not None:
                        acc["t1_angle"].append(float(np.arccos(np.clip(abs(np.dot(e[4], t[4])), -1, 1))))
            wf = errs(tf["foot_force"][:B, k].reshape(-1, 3), tt["foot_force"][:B, k].reshape(-1, 3), floor_f)
            wm = errs(tf["foot_moment"][:B, k].reshape(-1, 3), tt["foot_moment"][:B, k].reshape(-1, 3), floor_m)
            r[f"substep{k}"] = dict(matched=nm, unmatched_forest=ue, unmatched_tree=ur,
                                    **{kk: stats(v) for kk, v in acc.items()}, foot_force=stats(wf),
                                    foot_moment=stats(wm))
        for key in ("foot_fn_max", "foot_fnet_max", "foot_fup_max"):
            r[key] = stats(np.abs(tf[key][:B] - tt[key][:B]) / np.maximum(np.abs(tt[key][:B]), floor_f))
        res[scene] = r
        print(scene, json.dumps(r), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


def cmd_all(a):
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    me = str(Path(__file__).resolve())
    runs = []
    eng = ["--engine", a.engine]
    scenes = ("mf_s0", "mf_s1", "multitree") if a.engine == "forest" else ("mf_s0", "mf_s1", "terrain")
    for scene in scenes:
        for params in ((False, True) if scene == "mf_s0" and a.engine == "tree" else (False,)):
            for b in BUILDS:
                for api in ("jax", "torch"):
                    if api == "torch" and (b not in ("default", "full") or scene == "mf_s1"):
                        continue
                    name = f"{b}_{api}_{scene}" + ("_params" if params else "")
                    runs.append((name, ["run", *eng, "--build", b, "--api", api, "--scene", scene,
                                        "--nworld", str(a.nworld), "--out", str(d / f"{name}.npz")]
                                 + (["--params"] if params else []) + (["--chain"] if b == "full" else [])))
    torch_py = os.environ.get("DK_TORCH_PYTHON", py)
    for name, args in runs:
        if (d / f"{name}.npz").exists():
            continue
        exe = torch_py if "_torch_" in name else py
        subprocess.run([exe, me, *args], check=True)
    if not a.no_time:
        for b in BUILDS:
            p = d / f"time_{b}.json"
            if not p.exists():
                subprocess.run([py, me, "time", *eng, "--build", b, "--nworld", str(a.time_nworld), "--out", str(p)],
                               check=True)
    subprocess.run([py, me, "compare", "--dir", str(d), "--out", a.out, "--nref", str(a.nref)]
                   + (["--stepped"] if a.stepped else []), check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--build", choices=BUILDS, required=True)
    r.add_argument("--engine", choices=("tree", "forest"), default="tree")
    r.add_argument("--api", choices=("jax", "torch"), default="jax")
    r.add_argument("--scene", default="mf_s0")
    r.add_argument("--mode", default="analytic")
    r.add_argument("--params", action="store_true")
    r.add_argument("--chain", action="store_true")
    r.add_argument("--nworld", type=int, default=512)
    r.add_argument("--substeps", type=int, default=4)
    r.add_argument("--iterations", type=int, default=10)
    r.add_argument("--out", required=True)
    t = sp.add_parser("time")
    t.add_argument("--build", choices=BUILDS, required=True)
    t.add_argument("--engine", choices=("tree", "forest"), default="tree")
    t.add_argument("--nworld", type=int, default=4096)
    t.add_argument("--substeps", type=int, default=4)
    t.add_argument("--iterations", type=int, default=10)
    t.add_argument("--reps", type=int, default=30)
    t.add_argument("--out")
    c = sp.add_parser("compare")
    c.add_argument("--dir", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--nref", type=int, default=256)
    c.add_argument("--stepped", action="store_true")
    c.add_argument("--old-last")
    al = sp.add_parser("all")
    al.add_argument("--dir", default="runs/contact_forces/tree")
    al.add_argument("--engine", choices=("tree", "forest"), default="tree")
    al.add_argument("--out", required=True)
    al.add_argument("--nworld", type=int, default=512)
    al.add_argument("--nref", type=int, default=256)
    al.add_argument("--time-nworld", type=int, default=4096)
    al.add_argument("--stepped", action="store_true")
    al.add_argument("--no-time", action="store_true")
    x = sp.add_parser("xengine")
    x.add_argument("--tree", required=True)
    x.add_argument("--forest", required=True)
    x.add_argument("--scenes", default="mf_s0,mf_s1")
    x.add_argument("--nref", type=int, default=256)
    x.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "time": cmd_time, "compare": cmd_compare, "all": cmd_all, "xengine": cmd_xengine}[a.cmd](a)


if __name__ == "__main__":
    main()

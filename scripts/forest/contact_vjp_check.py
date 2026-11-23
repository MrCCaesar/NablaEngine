"""The contact output's VJP (init(contact_grad=True), forest_engine.contact_vjp) and the object slots' per-world
contact material (TerrainConfig(objects=K, object_material=True)) on the forest engine."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
SUB = 4


def dump(res, out):
    s = json.dumps(res, indent=1)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(s)
    print(s[:4000])


def stats(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return None
    return dict(n=int(x.size), median=float(np.median(x)), p90=float(np.quantile(x, 0.9)),
                p99=float(np.quantile(x, 0.99)), max=float(x.max()))


def g1_init(grad=True, mode="substeps"):
    from nablaengine import engines, fixtures, jax_ffi
    m, mt = fixtures.load_model(fixtures.fixtures_dir(), check=None)
    engines.prefer_forest(m)
    kw = dict(contact_forces=mode, foot_wrench=True, contact_frame=True)
    if grad:
        kw["contact_grad"] = True
    info = jax_ffi.init(model=m, foot_pairs=mt["foot_pairs"], com_body=mt.get("torso_body", "torso_link"), **kw)
    assert info["engine"] == "forest"
    return jax_ffi, info, m, mt


def model_init(xml, grad=True, mode="substeps"):
    """A G1-class MJCF (--model; e.g. a G1 with hands and joint equalities) on the forest engine: foot pairs = the
    pairs between a world geom and an ankle-roll link's geom, com body torso_link."""
    import mujoco
    from nablaengine import engines, jax_ffi
    m = mujoco.MjModel.from_xml_path(xml)
    engines.prefer_forest(m)
    body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
    fp = [i for i in range(m.npair) if {int(m.geom_bodyid[m.pair_geom1[i]]) == 0, int(m.geom_bodyid[m.pair_geom2[i]]) == 0}
          == {True, False} and any("ankle_roll" in body(g) for g in (m.pair_geom1[i], m.pair_geom2[i]))]
    mt = dict(foot_pairs=fp, torso_body="torso_link")
    kw = dict(contact_forces=mode, foot_wrench=True, contact_frame=True)
    if grad:
        kw["contact_grad"] = True
    info = jax_ffi.init(model=m, foot_pairs=fp, com_body="torso_link", **kw)
    assert info["engine"] == "forest"
    return jax_ffi, info, m, mt


def model_states(m, npz, key, B, seed, mt, vstd=0.2):
    """B states from the frames npz[key] (qpos): the frame, qvel ~ N(0, vstd), each joint actuator's setpoint the
    joint's position + N(0, 0.05) (kept 1e-3 inside its ctrlrange), no warm start, the model's foot friction."""
    z = np.load(npz)
    fr = np.asarray(z[key], np.float64)
    rng = np.random.default_rng(seed)
    idx = np.arange(B) % len(fr)
    q = fr[idx].copy()
    v = rng.normal(0.0, vstd, (B, m.nv))
    u = np.zeros((B, m.nu))
    for a in range(m.nu):
        j = int(m.actuator_trnid[a, 0])
        u[:, a] = q[:, m.jnt_qposadr[j]] + rng.normal(0.0, 0.05, B)
        if m.actuator_ctrllimited[a]:
            lo, hi = m.actuator_ctrlrange[a]
            u[:, a] = np.clip(u[:, a], lo + 1e-3, hi - 1e-3)
    fric = np.full(B, float(m.pair_friction[mt["foot_pairs"][0], 0]) if mt["foot_pairs"] else 1.0)
    return q, v, u, np.zeros((B, m.nv)), fric


def corpus(B, seed):
    from nablaengine import fixtures
    z = fixtures.load_corpus("mf_s0")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z["qpos"]), B, replace=False)
    f = lambda k: np.asarray(z[k][idx], np.float64)
    return f("qpos"), f("qvel"), f("ctrl"), f("qacc_warmstart"), f("friction")


def cotangent(info, con, rng, loss):
    """A random cotangent of the contact output (B, width) for loss (forces_last / forces_all / wrench / geometry):
    nonzero on the valid contacts' chosen words only."""
    from nablaengine.g1_engine import contact_forces as cf
    lay = info["contact_forces"]["con_layout"]
    mc, cw = int(info["maxcon"]), int(info["con_words"])
    nf = len(info["contact_forces"]["feet"])
    B = con.shape[0]
    y = np.zeros_like(con)
    tab = con[:, :SUB * mc * cw].reshape(B, SUB, mc, cw)
    yt = y[:, :SUB * mc * cw].reshape(B, SUB, mc, cw)
    valid = tab[..., 0] >= 0
    if loss in ("forces_last", "forces_all"):
        words = list(range(lay["force"][0], lay["force"][0] + 3)) + [lay["force_normal"][0]]
        ks = [SUB - 1] if loss == "forces_last" else list(range(SUB))
    elif loss == "geometry":
        words = [lay["dist"][0]] + list(range(2, 8)) + list(range(lay["tangent1"][0], lay["tangent1"][0] + 3))
        ks = list(range(SUB))
    else:
        words, ks = [], []
    for k in ks:
        for wd in words:
            yt[:, k, :, wd] = np.where(valid[:, k], rng.normal(size=(B, mc)), 0.0)
    y[:, :SUB * mc * cw] = yt.reshape(B, -1)
    if loss == "wrench":
        o = SUB * mc * cw
        y[:, o:o + SUB * nf * 6] = rng.normal(size=(B, SUB * nf * 6)) * np.tile([1.0] * 3 + [10.0] * 3, SUB * nf)
    del cf
    return y


def cmd_fd(a):
    import jax
    import jax.numpy as jnp
    import mujoco
    from nablaengine.g1_engine import contact_forces as cf
    from contact_forces_check import mj_contacts, match, foot_wrench_ref
    if a.model:
        ffi, info, m, mt = model_init(a.model)
        q, v, u, warm, fric = model_states(m, a.states, a.states_key, a.nworld, a.seed, mt, a.qvel_std)
    else:
        ffi, info, m, mt = g1_init()
        q, v, u, warm, fric = corpus(a.nworld, a.seed)
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    B, nq, nv, nu = len(q), m.nq, m.nv, m.nu
    com = np.zeros((B, int(info["com_words"])))

    def F(qa, va, ua):
        r = ffi.step(qa, va, f32(warm), ua, f32(fric), f32(com), iterations=a.iterations, substeps=SUB, clamp_kv=1,
                     normal=0, contacts=True)
        return r[1], r[3]

    (v1, con), vjp = jax.vjp(F, f32(q), f32(v), f32(u))
    con = np.asarray(con, np.float64)
    v1 = np.asarray(v1, np.float64)
    dec = cf.decode(con, info, SUB)
    pg = cf.pair_geoms(m, mt["foot_pairs"])
    names = [(m.geom(int(g1)).name, m.geom(int(g2)).name) for g1, g2 in pg]
    feet = list(info["contact_forces"]["feet"])
    mc_, cw = int(info["maxcon"]), int(info["con_words"])
    lay = info["contact_forces"]["con_layout"]
    mr = mujoco.MjModel.from_xml_path(a.model) if a.model else \
        __import__("nablaengine.fixtures", fromlist=["x"]).load_model(check=None)[0]
    mr.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    mr.opt.iterations, mr.opt.tolerance, mr.opt.ls_iterations = 100, 1e-12, 50
    foot_all = sorted(int(p) for p in mt["foot_pairs"])
    fr0 = mr.pair_friction.copy()
    d = mujoco.MjData(mr)
    losses = a.losses.split(",")
    if "state" not in losses:
        losses = ["state"] + losses
    rng = np.random.default_rng(a.seed + 5)
    ys = {L: cotangent(info, con, rng, L) for L in losses if L != "state"}
    grads = {}
    yv1 = rng.normal(size=(B, nv))
    for L in losses:
        if L == "state":
            gq, gv, gu = vjp((f32(yv1), jnp.zeros_like(jnp.asarray(con, jnp.float32))))
            grads[L] = [np.asarray(x, np.float64) for x in (gq, gv, gu)]
            continue
        gq, gv, gu = vjp((jnp.zeros((B, nv), jnp.float32), f32(ys[L])))
        grads[L] = [np.asarray(x, np.float64) for x in (gq, gv, gu)]
    from forest_slot_velocity_check import tangent
    res = dict(check="contact_vjp fd", nworld=B, iterations=a.iterations, eps=a.eps, losses={})
    blocks = {L: {k: ([], []) for k in ("qpos", "qvel", "ctrl")} for L in losses}
    excl = dict(unmatched=0, model=0, active_set=0)
    used, wids = 0, []
    only = [int(x) for x in a.only.split(",")] if a.only else None
    for w in range(B):
        if only is not None and w not in only:
            continue
        mr.pair_friction[:] = fr0
        mr.pair_friction[foot_all, 0] = mr.pair_friction[foot_all, 1] = fric[w]

        def run(qq, vv, uu):
            d.qpos[:], d.qvel[:], d.qacc_warmstart[:], d.ctrl[:] = qq, vv, warm[w], uu
            subs, sets = [], []
            for k in range(SUB):
                mujoco.mj_step(mr, d)
                subs.append((mj_contacts(mr, d, False), foot_wrench_ref(mr, d, [int(b) for b in feet], False),
                             [float(d.contact[i].dist) for i in range(d.ncon)]))
                sets.append((int(d.ncon), tuple(d.efc_state[:d.nefc].tolist()),
                             tuple((int(d.contact[i].geom[0]), int(d.contact[i].geom[1])) for i in range(d.ncon))))
            return subs, tuple(sets), d.qvel.copy()

        subs0, set0, cv = run(q[w], v[w], u[w])
        if np.linalg.norm(cv - v1[w]) > 1e-4 * max(np.linalg.norm(cv), 1e-6):
            excl["model"] += 1
            continue
        maps, ok = [], True
        for k in range(SUB):
            pair = dec["pair"][w, k]
            eng = []
            for c in np.nonzero(pair >= 0)[0]:
                eng.append((names[int(pair[c])], dec["pos"][w, k, c], dec["force"][w, k, c], dec["normal"][w, k, c],
                            None, dec["force_normal"][w, k, c], int(c)))
            ref = [r + (i,) for i, r in enumerate(subs0[k][0])]
            mp, ue, _ = _match_idx(eng, ref, match)
            ok &= ue == 0
            maps.append(mp)
        if not ok:
            excl["unmatched"] += 1
            continue

        def loss_vals(subs, vel):
            out = {}
            for L in losses:
                if L == "state":
                    out[L] = float(yv1[w] @ vel)
                    continue
                y = ys[L][w]
                tot = 0.0
                for k in range(SUB):
                    con_k, wr, dist = subs[k]
                    yk = y[k * mc_ * cw:(k + 1) * mc_ * cw].reshape(mc_, cw)
                    for c, i, flip in maps[k]:
                        key, pos, fw, n, t1, fn = con_k[i]
                        sg = -1.0 if flip else 1.0
                        tot += yk[c, 8:11] @ (sg * fw) + yk[c, 11] * fn + yk[c, 1] * dist[i] + yk[c, 2:5] @ pos
                        if not flip:
                            tot += yk[c, 5:8] @ n + yk[c, lay["tangent1"][0]:lay["tangent1"][0] + 3] @ t1
                    o = SUB * mc_ * cw + k * len(feet) * 6
                    tot += y[o:o + len(feet) * 6] @ wr[:, :6].reshape(-1)
                out[L] = tot
            return out

        stable = True

        def column(fn):
            nonlocal stable
            if not stable:
                return {L: np.nan for L in losses}
            vals = []
            for sgn in (1, -1):
                qq, vv, uu = fn(sgn * a.eps)
                subs, st, vel = run(qq, vv, uu)
                if st != set0:
                    stable = False
                    if excl["active_set"] == 0:
                        k = next(i for i in range(SUB) if st[i] != set0[i])
                        why = ("ncon" if st[k][0] != set0[k][0] else "pairs" if st[k][2] != set0[k][2] else
                               "rows " + str([(r, x, y) for r, (x, y) in enumerate(zip(set0[k][1], st[k][1]))
                                              if x != y][:6]))
                        print(f"world {w}: substep {k} changed under +-eps: {why}", flush=True)
                    return {L: np.nan for L in losses}
                vals.append(loss_vals(subs, vel))
            return {L: (vals[0][L] - vals[1][L]) / (2 * a.eps) for L in losses}

        def dq(k):
            def f(e):
                qp = q[w].copy(); dl = np.zeros(nv); dl[k] = e
                mujoco.mj_integratePos(mr, qp, dl, 1.0)
                return qp, v[w], u[w]
            return f

        def dv(k):
            def f(e):
                vp = v[w].copy(); vp[k] += e
                return q[w], vp, u[w]
            return f

        def du(k):
            def f(e):
                up = u[w].copy(); up[k] += e
                return q[w], v[w], up
            return f

        mk = dict(qpos=(dq, nv), qvel=(dv, nv), ctrl=(du, nu))
        cols = {b: [column(mk[b][0](k)) for k in range(mk[b][1])] for b in a.blocks.split(",")}
        if not stable:
            excl["active_set"] += 1
            continue
        used += 1
        wids.append(w)
        for L in losses:
            gq, gv, gu = grads[L]
            eng = dict(qpos=tangent(gq[w], q[w], nq), qvel=gv[w], ctrl=gu[w])
            for blk in cols:
                blocks[L][blk][0].append(eng[blk])
                blocks[L][blk][1].append(np.asarray([c[L] for c in cols[blk]]))
    res["worlds_used"], res["excluded"] = used, excl
    if a.npz:
        np.savez(a.npz, worlds=np.asarray(wids), **{f"{L}_{b}_{t}": np.asarray(blocks[L][b][j]) for L in losses
                                                     for b in blocks[L] for j, t in ((0, "eng"), (1, "fd"))})
    def rel(L, blk):
        gg, ff = np.asarray(blocks[L][blk][0]), np.asarray(blocks[L][blk][1])
        nf = np.linalg.norm(ff, axis=1)
        return np.linalg.norm(gg - ff, axis=1) / np.maximum(nf, 1e-9), nf

    sok = np.ones(used, bool)
    for blk in ("qpos", "qvel", "ctrl"):
        if blocks["state"][blk][0]:
            sok &= rel("state", blk)[0] <= 1e-3
    res["worlds_state_grad_ok"] = int(sok.sum())
    for L in losses:
        out = {}
        for blk in ("qpos", "qvel", "ctrl"):
            if not blocks[L][blk][0]:
                continue
            e, nf = rel(L, blk)
            has = nf > 1e-9
            out[blk] = dict(rel=stats(e[has]), rel_state_ok=stats(e[has & sok]),
                            fd_norm_median=float(np.median(nf[has])) if has.any() else 0.0,
                            accept_1e3=bool(np.median(e[has]) <= 1e-3) if has.any() else None,
                            worlds=int(has.sum()), worlds_state_ok=int((has & sok).sum()))
            print(f"{L:12s} {blk:5s} rel median {np.median(e[has]):.2e} p90 {np.quantile(e[has], 0.9):.2e} max "
                  f"{e[has].max():.2e} ({has.sum()} worlds); state-ok worlds: max "
                  f"{(e[has & sok].max() if (has & sok).any() else float('nan')):.2e} ({(has & sok).sum()})")
        res["losses"][L] = out
    print("excluded", excl, "used", used)
    dump(res, a.out)


def _match_idx(eng, ref, match):
    pairs, ue, ur = match([e[:6] for e in eng], [r[:6] for r in ref])
    out = []
    used = set()
    for e6, r6 in pairs:
        ce = next(e[6] for e in eng if e[1] is e6[1])
        best, bi = None, None
        for r in ref:
            if r[-1] in used:
                continue
            same = r[0] == e6[0]
            flip = (not same) and r[0] == (e6[0][1], e6[0][0])
            if not (same or flip):
                continue
            dd = np.linalg.norm(r[1] - e6[1])
            if best is None or dd < best:
                best, bi = dd, (r[-1], flip)
        used.add(bi[0])
        out.append((ce, bi[0], bi[1]))
    return out, ue, ur


MAT_W = 7
BOX_XML = """<mujoco><option timestep="0.005"/>
<worldbody><geom name="floor" type="plane" size="5 5 .1"/>
<body name="box" pos="0 0 0.3"><freejoint/><geom name="bx" @GEOM@ mass="1"/></body></worldbody>
<contact><pair geom1="floor" geom2="bx" condim="3" friction="1 1 0.005 0.0001 0.0001"/></contact></mujoco>"""
BLOCK = (0.1, 0.1, 0.02)
CAPSULE = 'type="capsule" size=".05" fromto="-.15 0 0 .15 0 0"'


def geom_attr(half):
    return CAPSULE if half == "capsule" else 'type="box" size="' + " ".join(repr(float(x)) for x in half) + '"'


def box_model_xml(half):
    return BOX_XML.replace("@GEOM@", geom_attr(half))


def ref_xml(mat, half=(1.0, 1.0, 0.05), body="capsule"):
    """MuJoCo's reference of one material: the slot as a mocap box with a pair carrying the material."""
    mu, sr, si = float(mat[0]), [float(x) for x in mat[1:3]], [float(x) for x in mat[3:7]]
    return f"""<mujoco><option timestep="0.005"/>
<worldbody>
<body name="box" pos="0 0 0.3"><freejoint/><geom name="bx" {geom_attr(body)} mass="1"
  contype="0" conaffinity="0"/></body>
<body name="slot" mocap="true"><geom name="slot" type="box" size="{half[0]} {half[1]} {half[2]}" contype="0"
  conaffinity="0"/></body></worldbody>
<contact><pair geom1="slot" geom2="bx" condim="3" friction="{mu!r} {mu!r} 0.005 0.0001 0.0001"
  solref="{sr[0]!r} {sr[1]!r}" solimp="{si[0]!r} {si[1]!r} {si[2]!r} {si[3]!r} 2"/></contact></mujoco>"""


def mat_init(velocity=False, grad=False, half="capsule"):
    import mujoco
    from nablaengine import engines, jax_ffi, terrain as T
    m = mujoco.MjModel.from_xml_string(box_model_xml(half))
    engines.prefer_forest(m)
    cfg = T.TerrainConfig(geoms="body", objects=1, object_velocity=velocity, object_material=True)
    kw = dict(contact_grad=True) if grad else {}
    info = jax_ffi.init(model=m, foot_pairs=[0], terrain=cfg, **kw)
    assert info["engine"] == "forest"
    jax_ffi.set_terrain(T.Terrain())
    return jax_ffi, info, m


def mat_pack(objs, mats, B, vel=None):
    from nablaengine import terrain as T
    return T.pack_com(np.zeros((B, 3), np.float32), np.asarray(objs, np.float32).reshape(B, 1, 15),
                      None if vel is None else np.asarray(vel, np.float32).reshape(B, 1, 18),
                      np.asarray(mats, np.float32).reshape(B, 1, MAT_W))


def rot(axis, ang):
    from forest_slot_velocity_check import rot as r
    return r(axis, ang)


def cmd_slide(a):
    """A block (0.2 x 0.2 x 0.04 m, 1 kg) resting on the slot's top face, the slot tilted by theta about y, material mu (the friction input 1.0
    and the floor pair's mu 1 differ): the smallest theta of the sweep at which the box slides, against atan(mu).
    Sliding: the box's velocity along the slope accelerates, a least-squares acceleration over the run's second half
    above --acc (m/s^2; the soft contact's static friction leaves a constant creep velocity, no acceleration); above
    the onset the measured acceleration against g (sin theta - mu cos theta)."""
    import jax.numpy as jnp
    from nablaengine import terrain as T
    from forest_slot_velocity_check import mat2quat
    ffi, info, m = mat_init(half=BLOCK)
    mus = [float(x) for x in a.mus.split(",")]
    res = dict(check="slide onset", seconds=a.seconds, step_deg=a.step, threshold_acc=a.acc, iterations=a.iterations,
               cases=[])
    h = 4 * m.opt.timestep
    nstep = int(round(a.seconds / h))
    for mu in mus:
        th0 = np.rad2deg(np.arctan(mu))
        ths = np.arange(th0 - a.span, th0 + a.span + 1e-9, a.step)
        B = len(ths)
        R = np.stack([rot([0, 1, 0], np.deg2rad(t))[0] for t in ths])
        objs = np.stack([T.box_object([0.0, 0.0, -0.05], R[i], [8.0, 8.0, 0.05]) for i in range(B)])
        mats = np.stack([T.slot_material(mu, (0.02, 1.0), (0.9, 0.95, 0.001, 0.5)) for _ in range(B)])
        q = np.zeros((B, 7))
        for i in range(B):
            q[i, :3] = np.array([0.0, 0.0, -0.05]) + R[i] @ np.array([0.0, 0.0, 0.05 + BLOCK[2]])
            q[i, 3:7] = mat2quat(R[i])
        com = jnp.asarray(mat_pack(objs, mats, B))
        qj, vj, warm = jnp.asarray(q, jnp.float32), jnp.zeros((B, 6), jnp.float32), jnp.zeros((B, 6), jnp.float32)
        u, fr = jnp.zeros((B, 0)), jnp.ones(B, jnp.float32)
        slope = np.stack([R[i][:, 0] for i in range(B)])
        vs = []
        for _ in range(nstep):
            qj, vj, warm = ffi.forward(qj, vj, warm, u, fr, com, iterations=a.iterations)[:3]
            vs.append(np.einsum("bi,bi->b", np.asarray(vj, np.float64)[:, :3], slope))
        vs = np.stack(vs)
        qn = np.asarray(qj, np.float64)
        disp = np.abs(np.einsum("bi,bi->b", qn[:, :3] - q[:, :3], slope))
        t = h * np.arange(1, nstep + 1)
        hw = slice(nstep // 2, nstep)
        acc = np.polyfit(t[hw], vs[hw], 1)[0]
        slides = acc > a.acc
        onset = float(ths[np.argmax(slides)]) if slides.any() else None
        err = None if onset is None else onset - th0
        th = np.deg2rad(ths)
        a_th = 9.81 * (np.sin(th) - mu * np.cos(th))
        far = ths > th0 + 0.5
        creep = vs[-1][ths < th0 - 0.5]
        res["cases"].append(dict(mu=mu, atan_mu_deg=float(th0), onset_deg=onset, error_deg=err,
                                 accept=bool(err is not None and abs(err) <= 1.0),
                                 acc_rel_err_above_onset=stats(np.abs(acc[far] - a_th[far]) / a_th[far]),
                                 creep_speed_below_onset_max=float(np.abs(creep).max()) if creep.size else None,
                                 sweep_deg=ths.round(3).tolist(), acceleration=acc.round(5).tolist(),
                                 coulomb_acceleration=np.maximum(a_th, 0).round(5).tolist(),
                                 displacement_m=disp.round(5).tolist()))
        print(f"mu {mu:.2f}: atan {th0:.3f} deg, onset {onset} deg, error {err}; acceleration vs Coulomb (0.5 deg "
              f"above): median rel {np.median(np.abs(acc[far] - a_th[far]) / a_th[far]):.3f}; creep below "
              f"{(np.abs(creep).max() if creep.size else 0):.4f} m/s")
    res["accepted"] = bool(all(c["accept"] for c in res["cases"]))
    dump(res, a.out)


def mat_states(B, seed, nmat):
    """Random capsule states (lying on the top face, 0 .. 3 mm deep) on random tilted slots and nmat random materials (B worlds, world i has material i % nmat)."""
    from forest_slot_velocity_check import mat2quat
    rng = np.random.default_rng(seed)
    q, v, objs = np.zeros((B, 7)), np.zeros((B, 6)), np.zeros((B, 15))
    for i in range(B):
        Rb = rot(rng.normal(size=3), rng.uniform(0, np.deg2rad(10)))[0] @ rot([0, 0, 1], rng.uniform(-np.pi, np.pi))[0]
        cb = np.r_[rng.uniform(-0.1, 0.1, 2), -0.05]
        objs[i] = np.r_[cb, Rb.reshape(9), 1.0, 1.0, 0.05]
        tilt = rot([0, 0, 1], rng.uniform(-3, 3))[0]
        off = np.r_[rng.uniform(-0.5, 0.5, 2), 0.1 - rng.uniform(0.0, 0.003)]
        q[i, :3], q[i, 3:7] = cb + Rb @ off, mat2quat(Rb @ tilt)
        v[i] = np.r_[Rb @ np.r_[rng.normal(0, 0.5, 2), rng.normal(0, 0.05)], Rb @ np.r_[0, 0, rng.normal(0, 1.0)]]
    mats = np.zeros((nmat, MAT_W))
    for j in range(nmat):
        mats[j] = np.r_[rng.uniform(0.2, 1.2), rng.uniform(0.01, 0.05), rng.uniform(0.7, 1.5),
                        rng.uniform(0.8, 0.9), rng.uniform(0.92, 0.99), rng.uniform(0.0005, 0.005), rng.uniform(0.3, 0.7)]
    mats = mats.astype(np.float32).astype(np.float64)
    return q, v, objs, mats


def cmd_mjw(a):
    """One control step of a capsule on random tilted slots with random per-world materials against MuJoCo Warp (the
    slot a mocap box, the material as the pair's friction / solref / solimp; one Warp model per material)."""
    import jax.numpy as jnp
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    ffi, info, m = mat_init()
    B = a.nworld
    q, v, objs, mats = mat_states(B, a.seed, a.nmat)
    if a.default_material:
        mats[:] = np.r_[1.0, 0.02, 1.0, 0.9, 0.95, 0.001, 0.5]
    wm = mats[np.arange(B) % a.nmat]
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    r = ffi.forward(f32(q), f32(v), jnp.zeros((B, 6), jnp.float32), jnp.zeros((B, 0)), jnp.full(B, a.fric, jnp.float32),
                    f32(mat_pack(objs, wm, B)), iterations=a.iterations, substeps=4)
    ev, en = np.asarray(r[1], np.float64), np.asarray(r[3])[:, 1]
    rv = np.zeros((B, 6))
    ncon = np.zeros(B, int)
    for j in range(a.nmat):
        idx = np.nonzero(np.arange(B) % a.nmat == j)[0]
        mr = mujoco.MjModel.from_xml_string(ref_xml(mats[j]))
        mr.opt.iterations = a.iterations
        nb = len(idx)
        mw = mjw.put_model(mr)
        dw = mjw.put_data(mr, mujoco.MjData(mr), nworld=nb, nconmax=32, njmax=128)
        mq = np.stack([np.r_[o[:3]] for o in objs[idx]])
        from forest_slot_velocity_check import mat2quat
        mqq = np.stack([mat2quat(o[3:12].reshape(3, 3)) for o in objs[idx]])
        for x, y in ((dw.qpos, q[idx]), (dw.qvel, v[idx]), (dw.mocap_pos, mq[:, None]), (dw.mocap_quat, mqq[:, None])):
            wp.copy(x, wp.array(y.astype(np.float32), dtype=x.dtype))
        for _ in range(4):
            mjw.step(mw, dw)
        rv[idx] = dw.qvel.numpy().astype(np.float64)
        nacon = int(dw.nacon.numpy()[0])
        ncon[idx] = np.bincount(dw.contact.worldid.numpy()[:nacon], minlength=nb)
    e = np.linalg.norm(ev - rv, axis=1) / np.maximum(np.linalg.norm(rv, axis=1), 1e-6)
    w = ncon > 0
    res = dict(check="material vs MuJoCo Warp", nworld=B, nmat=a.nmat, iterations=a.iterations, substeps=4,
               worlds_with_contacts=int(w.sum()), qvel_rel=stats(e[w]), ncon_equal=float((en[w] == ncon[w]).mean()))
    r0 = ffi.forward(f32(q), f32(v), jnp.zeros((B, 6), jnp.float32), jnp.zeros((B, 0)), jnp.full(B, 0.7, jnp.float32),
                     f32(mat_pack(objs, np.tile(mats[0], (B, 1)), B)), iterations=a.iterations, substeps=4)
    e0 = np.linalg.norm(np.asarray(r0[1], np.float64) - rv, axis=1) / np.maximum(np.linalg.norm(rv, axis=1), 1e-6)
    res["qvel_rel_with_one_material"] = stats(e0[w & (np.arange(B) % a.nmat != 0)])
    print(f"qvel rel median {np.median(e[w]):.2e} p99 {np.quantile(e[w], 0.99):.2e} max {e[w].max():.2e}; "
          f"ncon equal {res['ncon_equal']:.3f}; one material for all: median {np.median(e0[w]):.2e}")
    dump(res, a.out)


def cmd_matgrad(a):
    """The engine's VJP of y . (qpos', qvel') (one control step, 4 substeps; exact mode) by the slot's material words
    (and qpos, qvel) against MuJoCo C float64 central differences of the pair's friction / solref / solimp (h 1e-6,
    Newton to 1e-12), on the worlds whose active set is stable at +-h in every column and whose MuJoCo C step agrees;
    --forces: also a random cotangent on the last substep's contact forces (contact_grad)."""
    import jax
    import jax.numpy as jnp
    import mujoco
    from forest_slot_velocity_check import tangent, mat2quat
    ffi, info, m = mat_init(grad=a.forces)
    B = a.nworld
    q, v, objs, mats = mat_states(B, a.seed + 3, B)
    o = info["com_layout"]["object_material"][0]
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    rng = np.random.default_rng(a.seed + 11)
    yq, yv = rng.normal(size=(B, 7)), rng.normal(size=(B, 6))
    com = mat_pack(objs, mats, B)
    sub = 4

    def L(qa, va, ca):
        r = ffi.step(qa, va, jnp.zeros((B, 6), jnp.float32), jnp.zeros((B, 0)), jnp.full(B, 0.7, jnp.float32), ca,
                     iterations=a.iterations, substeps=sub, clamp_kv=1, normal=0, contacts=a.forces)
        out = jnp.sum(f32(yq) * r[0]) + jnp.sum(f32(yv) * r[1])
        if a.forces:
            out = out + jnp.sum(f32(yc) * r[3])
        return out

    yc = None
    if a.forces:
        con = np.asarray(ffi.forward(f32(q), f32(v), jnp.zeros((B, 6), jnp.float32), jnp.zeros((B, 0)),
                                     jnp.full(B, 0.7, jnp.float32), f32(com), iterations=a.iterations, substeps=sub,
                                     contacts=True)[4])
        mc, cw = int(info["maxcon"]), int(info["con_words"])
        yc = np.zeros_like(con)
        t = con.reshape(B, mc, cw)
        y3 = yc.reshape(B, mc, cw)
        for wd in (8, 9, 10, 11):
            y3[:, :, wd] = np.where(t[:, :, 0] >= 0, rng.normal(size=(B, mc)) * 0.1, 0.0)
        yc = y3.reshape(B, -1)
    g = jax.grad(L, (0, 1, 2))(f32(q), f32(v), f32(com))
    gq, gv, gc = (np.asarray(x, np.float64) for x in g)
    gm = gc[:, o:o + MAT_W]
    e_v1 = np.asarray(ffi.forward(f32(q), f32(v), jnp.zeros((B, 6), jnp.float32), jnp.zeros((B, 0)),
                                  jnp.full(B, 0.7, jnp.float32), f32(com), iterations=a.iterations, substeps=sub)[1],
                      np.float64)
    blocks = {k: ([], []) for k in ("qpos", "qvel", "mu", "solref", "solimp")}
    stable_n = model_n = 0
    for i in range(B):
        mr = mujoco.MjModel.from_xml_string(ref_xml(mats[i]))
        mr.opt.iterations, mr.opt.tolerance, mr.opt.ls_iterations = 100, 1e-12, 50
        d = mujoco.MjData(mr)
        d.mocap_pos[0], d.mocap_quat[0] = objs[i, :3], mat2quat(objs[i, 3:12].reshape(3, 3))
        p0 = (mr.pair_friction[0].copy(), mr.pair_solref[0].copy(), mr.pair_solimp[0].copy())

        def run(qq, vv, prm):
            mr.pair_friction[0], mr.pair_solref[0], mr.pair_solimp[0] = prm
            d.qpos[:], d.qvel[:] = qq, vv
            d.qacc_warmstart[:] = 0
            st = []
            fsum = 0.0
            for _ in range(sub):
                mujoco.mj_forward(mr, d)
                st.append((int(d.ncon), tuple(d.efc_state[:d.nefc].tolist())))
                mujoco.mj_step(mr, d)
            if a.forces:
                f6 = np.zeros(6)
                tab = yc[i].reshape(-1, int(info["con_words"]))
                for c in range(d.ncon):
                    mujoco.mj_contactForce(mr, d, c, f6)
                    fr = np.array(d.contact[c].frame).reshape(3, 3)
                    fw = fr.T @ f6[:3]
                    if mr.geom_bodyid[d.contact[c].geom[1]] != 1:
                        fw = -fw
                    j = cmap[c] if c < len(cmap) else -1
                    if j >= 0:
                        fsum += tab[j, 8:11] @ fw + tab[j, 11] * f6[0]
            return float(yq[i] @ d.qpos + yv[i] @ d.qvel) + fsum, tuple(st), d.qvel.copy()

        cmap = []
        if a.forces:
            run(q[i], v[i], p0)
            ecn = con.reshape(B, -1, int(info["con_words"]))[i]
            for c in range(d.ncon):
                dd = np.linalg.norm(ecn[:, 2:5] - d.contact[c].pos, axis=1) + 1e9 * (ecn[:, 0] < 0)
                cmap.append(int(np.argmin(dd)) if dd.min() < 2e-3 else -1)
        _, s0, cv = run(q[i], v[i], p0)
        ok = np.linalg.norm(cv - e_v1[i]) <= 1e-4 * max(np.linalg.norm(cv), 1e-6)
        model_n += ok
        stable = True

        def col(fn):
            nonlocal stable
            vals = []
            for sg in (1, -1):
                lv, st, _ = run(*fn(sg * a.eps))
                stable &= st == s0
                vals.append(lv)
            return (vals[0] - vals[1]) / (2 * a.eps)

        def dq(k):
            def f(e):
                qp = q[i].copy(); dl = np.zeros(6); dl[k] = e
                mujoco.mj_integratePos(mr, qp, dl, 1.0)
                return qp, v[i], p0
            return f

        def dv(k):
            def f(e):
                vp = v[i].copy(); vp[k] += e
                return q[i], vp, p0
            return f

        def dm(k):
            def f(e):
                fr_, sr, si = (x.copy() for x in p0)
                if k == 0:
                    fr_[:2] += e
                elif k < 3:
                    sr[k - 1] += e
                else:
                    si[k - 3] += e
                return q[i], v[i], (fr_, sr, si)
            return f

        fd = dict(qpos=[col(dq(k)) for k in range(6)], qvel=[col(dv(k)) for k in range(6)])
        fm = [col(dm(k)) for k in range(MAT_W)]
        run(q[i], v[i], p0)
        if not (stable and ok):
            continue
        stable_n += 1
        eng = dict(qpos=tangent(gq[i], q[i], 7), qvel=gv[i], mu=gm[i, :1], solref=gm[i, 1:3], solimp=gm[i, 3:7])
        fd.update(mu=fm[:1], solref=fm[1:3], solimp=fm[3:7])
        for k in blocks:
            blocks[k][0].append(eng[k]); blocks[k][1].append(np.asarray(fd[k]))
    res = dict(check="material gradients vs MuJoCo C", nworld=B, worlds_used=stable_n, model_ok=int(model_n),
               forces=bool(a.forces), eps=a.eps, iterations=a.iterations, blocks={})
    for k, (gg, ff) in blocks.items():
        gg, ff = np.asarray(gg), np.asarray(ff)
        e = np.linalg.norm(gg - ff, axis=1) / np.maximum(np.linalg.norm(ff, axis=1), 1e-9)
        res["blocks"][k] = dict(rel=stats(e), fd_norm_median=float(np.median(np.linalg.norm(ff, axis=1))))
        print(f"{k:7s} rel median {np.median(e):.2e} p90 {np.quantile(e, 0.9):.2e} max {e.max():.2e} ({len(e)} worlds)")
    dump(res, a.out)


def _api(name):
    if name == "torch":
        from nablaengine import torch_ffi
        return torch_ffi
    from nablaengine import jax_ffi
    return jax_ffi


def cmd_dump(a):
    """One build's (--scene g1 / box, --build default / grad, --api jax / torch) outputs on fixed inputs: the forward
    (state, contact output), the backward without the contact cotangent, and with a grad build also with a zero and a
    random one (and the box scene's material cotangents) -> --out .npz."""
    import mujoco
    from nablaengine import engines, fixtures, terrain as T
    api = _api(a.api)
    B = a.nworld
    rng = np.random.default_rng(7)
    if a.scene == "g1":
        m, mt = fixtures.load_model(fixtures.fixtures_dir(), check=None)
        engines.prefer_forest(m)
        kw = dict(model=m, foot_pairs=mt["foot_pairs"], com_body=mt.get("torso_body", "torso_link"),
                  contact_forces="substeps", foot_wrench=True, contact_frame=True)
        q, v, u, warm, fric = corpus(B, 3)
    elif a.scene == "course":
        m = mujoco.MjModel.from_xml_string(box_model_xml("capsule"))
        engines.prefer_forest(m)
        kw = dict(model=m, foot_pairs=[0], terrain=T.TerrainConfig(geoms="body", terrain_material=True))
        ter, Rc, cc = course(9, 0)
        q, v, fric = course_states(ter, Rc, cc, B, 4)
        u, warm = np.zeros((B, 0)), np.zeros((B, 6))
    else:
        m = mujoco.MjModel.from_xml_string(box_model_xml("capsule"))
        engines.prefer_forest(m)
        kw = dict(model=m, foot_pairs=[0], terrain=T.TerrainConfig(geoms="body", objects=1, object_material=True))
        q, v, objs, mats = mat_states(B, 3, B)
        u, warm, fric = np.zeros((B, 0)), np.zeros((B, 6)), np.full(B, 0.7)
    if a.build == "grad":
        kw["contact_grad"] = True
    info = api.init(**kw)
    if a.scene != "g1":
        api.set_terrain(ter if a.scene == "course" else T.Terrain())
    com = np.zeros((B, int(info["com_words"])))
    if a.scene == "box":
        com = np.asarray(mat_pack(objs, mats, B), np.float64)
    if a.api == "torch":
        import torch
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device="cuda").contiguous()
        n = lambda x: x.detach().cpu().numpy()
    else:
        import jax.numpy as jnp
        t = lambda x: jnp.asarray(np.asarray(x, np.float32))
        n = np.asarray
    ins = [t(x) for x in (q, v, warm, u, fric, com)]
    r = api.forward(*ins, iterations=10, substeps=SUB, tape=True, contacts=True)
    q1, v1, con, tp = r[0], r[1], r[4], r[5]
    out = dict(q1=n(q1), v1=n(v1), con=n(con), tape=n(tp))
    gq, gv = t(rng.normal(size=(B, m.nq))), t(rng.normal(size=(B, m.nv)))
    bw = lambda **k: api.backward(tp, ins[4], ins[5], gq, gv, substeps=SUB, clamp_kv=1, normal=0, **k)
    r0 = bw()
    out.update(g0_q=n(r0[0]), g0_v=n(r0[1]), g0_u=n(r0[2]), g0_rest=np.concatenate([n(x).reshape(B, -1) for x in
                                                                                       r0[4:]], 1) if len(r0) > 4
               else np.zeros((B, 0)))
    if a.build == "grad":
        yc = rng.normal(size=out["con"].shape) * (out["con"] != 0)
        for tag, y in (("gz", np.zeros_like(yc)), ("gc", yc)):
            rr = bw(g_contacts=t(y))
            out.update({f"{tag}_q": n(rr[0]), f"{tag}_v": n(rr[1]), f"{tag}_u": n(rr[2]),
                        f"{tag}_rest": np.concatenate([n(x).reshape(B, -1) for x in rr[4:]], 1) if len(rr) > 4
                        else np.zeros((B, 0))})
        if a.api == "torch":
            import torch
            xs = [x.clone().requires_grad_(True) for x in (ins[0], ins[1], ins[3])]
            rr = api.step(xs[0], xs[1], ins[2], xs[2], ins[4], ins[5], iterations=10, substeps=SUB, clamp_kv=1,
                          normal=0, contacts=True)
            L = (rr[0] * gq).sum() + (rr[1] * gv).sum() + (rr[3] * t(yc)).sum()
            ga = torch.autograd.grad(L, xs)
        else:
            import jax
            f = lambda q_, v_, u_: api.step(q_, v_, ins[2], u_, ins[4], ins[5], iterations=10, substeps=SUB,
                                            clamp_kv=1, normal=0, contacts=True)
            prim, vjp = jax.vjp(f, ins[0], ins[1], ins[3])
            ga = vjp((gq, gv, jax.numpy.zeros_like(prim[2]), t(yc)))
        out.update(ga_q=n(ga[0]), ga_v=n(ga[1]), ga_u=n(ga[2]))
    np.savez(a.out, **out)
    print("dumped", a.out, sorted(out))


def cmd_compare(a):
    """Bitwise comparisons of dumps: --a and --b, the keys both have (or --keys)."""
    za, zb = np.load(a.a), np.load(a.b)
    keys = a.keys.split(",") if a.keys else sorted(set(za.files) & set(zb.files))
    res = {}
    for k in keys:
        x, y = za[k], zb[k]
        same = x.shape == y.shape and np.array_equal(x.view(np.uint32) if x.dtype == np.float32 else x,
                                                     y.view(np.uint32) if y.dtype == np.float32 else y)
        d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.shape == y.shape and x.size else 0.0
        res[k] = dict(bitwise=bool(same), max_abs_diff=d)
        print(f"{k:10s} {'bitwise' if same else 'DIFFERS'} max |d| {d:.3e}")
    dump(dict(a=str(a.a), b=str(a.b), keys=res, all_bitwise=bool(all(v["bitwise"] for v in res.values()))), a.out)


def ter_init(half="capsule"):
    import mujoco
    from nablaengine import engines, jax_ffi, terrain as T
    m = mujoco.MjModel.from_xml_string(box_model_xml(half))
    engines.prefer_forest(m)
    info = jax_ffi.init(model=m, foot_pairs=[0], terrain=T.TerrainConfig(geoms="body", terrain_material=True))
    assert info["engine"] == "forest"
    return jax_ffi, info, m


def tilted_boxes(angles, spacing=7.0, half=(2.5, 2.5, 0.05)):
    """One box per angle (tilted about y), on a square grid around the origin (float32 positions stay fine):
    (boxes (n, 15), rotations (n, 3, 3), centres)."""
    from nablaengine import terrain as T
    R = np.stack([rot([0, 1, 0], np.deg2rad(t))[0] for t in angles])
    k = int(np.ceil(np.sqrt(len(angles))))
    c = np.stack([[spacing * (i % k - k // 2), spacing * (i // k - k // 2), -0.05] for i in range(len(angles))])
    return np.stack([T.box_object(c[i], R[i], half) for i in range(len(angles))]).astype(np.float64), R, c


def cmd_terslide(a):
    """A block on tilted terrain boxes, the per-world friction input (the block's floor pair is a foot pair) drawn in
    --fric-range; the boxes: an "override" patch of mu --patch, a "max" patch of mu --patch, and plain boxes (no
    material). Per world friction and patch kind the smallest tilt at which the block accelerates down the slope
    (as cmd_slide) against atan(the expected mu): override: --patch; max: max(friction, --patch); plain: friction."""
    import jax.numpy as jnp
    from nablaengine import terrain as T
    from forest_slot_velocity_check import mat2quat
    ffi, info, m = ter_init(half=BLOCK)
    rng = np.random.default_rng(a.seed)
    fr = np.sort(rng.uniform(*[float(x) for x in a.fric_range.split(",")], a.nfric))
    cases = []
    for f in fr:
        cases += [("override", f, a.patch), ("max", f, max(f, a.patch)), ("plain", f, f)]
    step, span = a.step, a.span
    offs = np.arange(-span, span + 1e-9, step)
    angles, mats, rows = [], [], []
    for ci, (kind, f, mu) in enumerate(cases):
        for o in offs:
            angles.append(np.rad2deg(np.arctan(mu)) + o)
            mats.append(T.terrain_material(a.patch, "override") if kind == "override" else
                        T.terrain_material(a.patch, "max") if kind == "max" else
                        T.terrain_material(1.0, "none"))
            rows.append((ci, o))
    boxes, R, c = tilted_boxes(angles)
    ter = T.Terrain(boxes=boxes, material=np.stack(mats))
    ffi.set_terrain(ter)
    B = len(angles)
    q = np.zeros((B, 7))
    for i in range(B):
        q[i, :3] = c[i] + R[i] @ np.array([0.0, 0.0, 0.05 + BLOCK[2]])
        q[i, 3:7] = mat2quat(R[i])
    fric = np.asarray([cases[ci][1] for ci, _ in rows], np.float32)
    com = jnp.zeros((B, int(info["com_words"])), jnp.float32)
    qj, vj, warm = jnp.asarray(q, jnp.float32), jnp.zeros((B, 6), jnp.float32), jnp.zeros((B, 6), jnp.float32)
    h = 4 * m.opt.timestep
    nstep = int(round(a.seconds / h))
    vs = []
    for _ in range(nstep):
        qj, vj, warm = ffi.forward(qj, vj, warm, jnp.zeros((B, 0)), jnp.asarray(fric), com, iterations=a.iterations)[:3]
        vs.append(np.einsum("bi,bi->b", np.asarray(vj, np.float64)[:, :3], R[:, :, 0]))
    vs = np.stack(vs)
    t = h * np.arange(1, nstep + 1)
    hw = slice(nstep // 2, nstep)
    acc = np.polyfit(t[hw], vs[hw], 1)[0]
    out = []
    for ci, (kind, f, mu) in enumerate(cases):
        idx = [i for i, (cj, _) in enumerate(rows) if cj == ci]
        sl = acc[idx] > a.acc
        th0 = np.rad2deg(np.arctan(mu))
        onset = float(th0 + offs[np.argmax(sl)]) if sl.any() else None
        err = None if onset is None else onset - th0
        out.append(dict(kind=kind, friction_input=float(f), expected_mu=float(mu), atan_deg=float(th0),
                        onset_deg=onset, error_deg=err, accept=bool(err is not None and abs(err) <= 1.0)))
        print(f"{kind:8s} friction input {f:.3f} expected mu {mu:.3f}: onset - atan = {err}")
    res = dict(check="terrain material slide onset", patch_mu=a.patch, step_deg=step, seconds=a.seconds,
               threshold_acc=a.acc, cases=out, accepted=bool(all(o["accept"] for o in out)))
    dump(res, a.out)


def course(n, seed):
    """A mixed course: n boxes (random tilt up to 10 deg, side by side), materials cycling none / max / override with
    random mu (0.1 .. 1.5), override also random solref / solimp."""
    from nablaengine import terrain as T
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-10, 10, n)
    boxes, R, c = tilted_boxes(angles, spacing=3.0, half=(1.0, 1.0, 0.05))
    for i in range(n):
        Ry = rot([0, 0, 1], rng.uniform(-np.pi, np.pi))[0]
        R[i] = Ry @ R[i]
        boxes[i, 3:12] = R[i].reshape(9)
    mats = []
    for i in range(n):
        kind = ("none", "max", "override")[i % 3]
        sr = (rng.uniform(0.01, 0.05), rng.uniform(0.7, 1.5)) if kind == "override" else (0.02, 1.0)
        si = ((rng.uniform(0.8, 0.9), rng.uniform(0.92, 0.99), rng.uniform(0.0005, 0.005), rng.uniform(0.3, 0.7))
              if kind == "override" else (0.9, 0.95, 0.001, 0.5))
        mats.append(T.terrain_material(rng.uniform(0.1, 1.5), kind, sr, si))
    return T.Terrain(boxes=boxes, material=np.stack(mats)), R, c


def course_states(ter, R, c, B, seed):
    """The capsule lying on box i % n of the course (0 .. 3 mm deep), random in-plane velocities; per-world friction
    input U(0.3, 1.6)."""
    from forest_slot_velocity_check import mat2quat
    rng = np.random.default_rng(seed)
    n = len(R)
    q, v = np.zeros((B, 7)), np.zeros((B, 6))
    for i in range(B):
        j = i % n
        tilt = rot([0, 0, 1], rng.uniform(-3, 3))[0]
        off = np.r_[rng.uniform(-0.6, 0.6, 2), 0.1 - rng.uniform(0.0, 0.003)]
        q[i, :3], q[i, 3:7] = c[j] + R[j] @ off, mat2quat(R[j] @ tilt)
        v[i] = np.r_[R[j] @ np.r_[rng.normal(0, 0.5, 2), rng.normal(0, 0.05)], R[j] @ np.r_[0, 0, rng.normal(0, 1.0)]]
    return q, v, rng.uniform(0.3, 1.6, B)


def course_ref_xml(ter):
    """The reference: the capsule (friction set per world) and the course's boxes as geoms without explicit pairs, the
    materials as MuJoCo's geom parameters: none = friction 0 (max leaves the capsule's), max = friction mu, override =
    priority 1 with friction / solref / solimp; the capsule and the plain boxes the default solref / solimp."""
    from forest_slot_velocity_check import mat2quat
    geoms = []
    for i, (b, mt) in enumerate(zip(ter.boxes, ter.material)):
        q = [float(x) for x in mat2quat(np.asarray(b[3:12]).reshape(3, 3))]
        b = [float(x) for x in b]
        mode = int(mt[7])
        f = 0.0 if mode == 0 else float(mt[0])
        extra = ""
        if mode == 2:
            extra = (f' priority="1" solref="{float(mt[1])!r} {float(mt[2])!r}" solimp="{float(mt[3])!r} '
                     f'{float(mt[4])!r} {float(mt[5])!r} {float(mt[6])!r} 2"')
        geoms.append(f'<geom name="b{i}" type="box" size="{b[12]!r} {b[13]!r} {b[14]!r}" pos="{b[0]!r} {b[1]!r} '
                     f'{b[2]!r}" quat="{q[0]!r} {q[1]!r} {q[2]!r} {q[3]!r}" friction="{f!r} 0.005 0.0001"{extra}/>')
    return f"""<mujoco><option timestep="0.005"/>
<worldbody>{''.join(geoms)}
<body name="box" pos="0 0 0.3"><freejoint/><geom name="bx" {CAPSULE} mass="1" friction="1 0.005 0.0001"/></body>
</worldbody></mujoco>"""


def cmd_termjw(a):
    """One control step (4 substeps) of the capsule on the mixed course with per-world friction inputs against MuJoCo
    Warp (course_ref_xml; the capsule's geom friction per world: Warp's batched geom_friction)."""
    import jax.numpy as jnp
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    ffi, info, m = ter_init()
    ter, R, c = course(a.nbox, a.seed)
    ffi.set_terrain(ter)
    B = a.nworld
    q, v, fric = course_states(ter, R, c, B, a.seed + 1)
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    fric = fric.astype(np.float32).astype(np.float64)
    r = ffi.forward(f32(q), f32(v), jnp.zeros((B, 6)), jnp.zeros((B, 0)), f32(fric),
                    jnp.zeros((B, int(info["com_words"])), jnp.float32), iterations=a.iterations, substeps=4)
    ev, en = np.asarray(r[1], np.float64), np.asarray(r[3])[:, 1]
    mr = mujoco.MjModel.from_xml_string(course_ref_xml(ter))
    mr.opt.iterations = a.iterations
    gb = mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_GEOM, "bx")
    mw = mjw.put_model(mr)
    gf = np.tile(mr.geom_friction[None].astype(np.float32), (B, 1, 1))
    gf[:, gb, 0] = fric
    mw.geom_friction = wp.array(gf, dtype=wp.vec3)
    dw = mjw.put_data(mr, mujoco.MjData(mr), nworld=B, nconmax=32 * B // 8 + 64, njmax=128)
    for x, y in ((dw.qpos, q), (dw.qvel, v)):
        wp.copy(x, wp.array(y.astype(np.float32)))
    for _ in range(4):
        mjw.step(mw, dw)
    rv = dw.qvel.numpy().astype(np.float64)
    nacon = int(dw.nacon.numpy()[0])
    ncon = np.bincount(dw.contact.worldid.numpy()[:nacon], minlength=B)
    e = np.linalg.norm(ev - rv, axis=1) / np.maximum(np.linalg.norm(rv, axis=1), 1e-6)
    w = ncon > 0
    kinds = np.asarray([int(ter.material[i % a.nbox][7]) for i in range(B)])
    res = dict(check="terrain material vs MuJoCo Warp", nworld=B, nbox=a.nbox, iterations=a.iterations,
               worlds_with_contacts=int(w.sum()), ncon_equal=float((en[w] == ncon[w]).mean()), qvel_rel=stats(e[w]),
               by_mode={nm: stats(e[w & (kinds == k)]) for k, nm in ((0, "none"), (1, "max"), (2, "override"))})
    ter0 = __import__("copy").copy(ter)
    ter0.material = None
    ffi.set_terrain(ter0)
    r0 = ffi.forward(f32(q), f32(v), jnp.zeros((B, 6)), jnp.zeros((B, 0)), f32(fric),
                     jnp.zeros((B, int(info["com_words"])), jnp.float32), iterations=a.iterations, substeps=4)
    e0 = np.linalg.norm(np.asarray(r0[1], np.float64) - rv, axis=1) / np.maximum(np.linalg.norm(rv, axis=1), 1e-6)
    res["qvel_rel_without_material"] = {nm: stats(e0[w & (kinds == k)]) for k, nm in ((1, "max"), (2, "override"))}
    for k, o in res["by_mode"].items():
        print(f"{k:9s} qvel rel median {o['median']:.2e} p99 {o['p99']:.2e} max {o['max']:.2e} ({o['n']} worlds)")
    print("ncon equal", res["ncon_equal"], "without material:", res["qvel_rel_without_material"])
    dump(res, a.out)


def cmd_tergrad(a):
    """The engine's VJP of y . (qpos', qvel') (exact mode, 50 iterations, 4 substeps) on the mixed course against
    MuJoCo C float64 central differences (course_ref_xml; h 1e-6, Newton to 1e-12) of qpos and qvel, on the worlds with
    a stable active set and an agreeing C step. (The terrain's materials are static: no gradient.)"""
    import jax
    import jax.numpy as jnp
    import mujoco
    from forest_slot_velocity_check import tangent
    ffi, info, m = ter_init()
    ter, R, c = course(a.nbox, a.seed)
    ffi.set_terrain(ter)
    B = a.nworld
    q, v, fric = course_states(ter, R, c, B, a.seed + 2)
    fric = fric.astype(np.float32).astype(np.float64)
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    rng = np.random.default_rng(a.seed + 3)
    yq, yv = rng.normal(size=(B, 7)), rng.normal(size=(B, 6))
    com = jnp.zeros((B, int(info["com_words"])), jnp.float32)

    def L(qa, va):
        r = ffi.step(qa, va, jnp.zeros((B, 6), jnp.float32), jnp.zeros((B, 0)), f32(fric), com,
                     iterations=a.iterations, substeps=4, clamp_kv=1, normal=0)
        return jnp.sum(f32(yq) * r[0]) + jnp.sum(f32(yv) * r[1])

    gq, gv = (np.asarray(x, np.float64) for x in jax.grad(L, (0, 1))(f32(q), f32(v)))
    e_v1 = np.asarray(ffi.forward(f32(q), f32(v), jnp.zeros((B, 6)), jnp.zeros((B, 0)), f32(fric), com,
                                  iterations=a.iterations, substeps=4)[1], np.float64)
    mr = mujoco.MjModel.from_xml_string(course_ref_xml(ter))
    mr.opt.iterations, mr.opt.tolerance, mr.opt.ls_iterations = 100, 1e-12, 50
    gb = mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_GEOM, "bx")
    d = mujoco.MjData(mr)
    blocks = {k: ([], []) for k in ("qpos", "qvel")}
    kinds_used = []
    for i in range(B):
        mr.geom_friction[gb, 0] = fric[i]

        def run(qq, vv):
            d.qpos[:], d.qvel[:] = qq, vv
            d.qacc_warmstart[:] = 0
            st = []
            for _ in range(4):
                mujoco.mj_forward(mr, d)
                st.append((int(d.ncon), tuple(d.efc_state[:d.nefc].tolist())))
                mujoco.mj_step(mr, d)
            return float(yq[i] @ d.qpos + yv[i] @ d.qvel), tuple(st), d.qvel.copy()

        _, s0, cv = run(q[i], v[i])
        if np.linalg.norm(cv - e_v1[i]) > 1e-4 * max(np.linalg.norm(cv), 1e-6) or s0[0][0] == 0:
            continue
        stable = True

        def col(fn):
            nonlocal stable
            vals = []
            for sg in (1, -1):
                lv, st, _ = run(*fn(sg * a.eps))
                stable &= st == s0
                vals.append(lv)
            return (vals[0] - vals[1]) / (2 * a.eps)

        def dq(k):
            def f(e):
                qp = q[i].copy(); dl = np.zeros(6); dl[k] = e
                mujoco.mj_integratePos(mr, qp, dl, 1.0)
                return qp, v[i]
            return f

        def dv(k):
            def f(e):
                vp = v[i].copy(); vp[k] += e
                return q[i], vp
            return f

        fd = dict(qpos=[col(dq(k)) for k in range(6)], qvel=[col(dv(k)) for k in range(6)])
        if not stable:
            continue
        kinds_used.append(int(ter.material[i % a.nbox][7]))
        for k, g in (("qpos", tangent(gq[i], q[i], 7)), ("qvel", gv[i])):
            blocks[k][0].append(g); blocks[k][1].append(np.asarray(fd[k]))
    kinds_used = np.asarray(kinds_used)
    res = dict(check="terrain material gradients vs MuJoCo C", nworld=B, worlds_used=len(kinds_used),
               note="the terrain elements' materials are static (no gradient)", blocks={})
    for k, (gg, ff) in blocks.items():
        gg, ff = np.asarray(gg), np.asarray(ff)
        e = np.linalg.norm(gg - ff, axis=1) / np.maximum(np.linalg.norm(ff, axis=1), 1e-9)
        res["blocks"][k] = {nm: stats(e[kinds_used == kk]) for kk, nm in ((0, "none"), (1, "max"), (2, "override"))}
        res["blocks"][k]["all"] = stats(e)
        print(f"{k}: rel median {np.median(e):.2e} max {e.max():.2e} ({len(e)} worlds)")
    dump(res, a.out)


def cmd_time(a):
    """The backward's cost (JAX, the mjlab G1 on the forest engine, corpus states, 4 substeps, 10 iterations,
    contact_forces "substeps" + foot_wrench + contact_frame): --build default: backward(); --build grad: backward()
    without and with g_contacts (a random cotangent on the valid contacts' forces and the wrenches). Medians of --reps
    calls (block_until_ready), fastest of --rounds rounds."""
    import time
    import jax
    import jax.numpy as jnp
    ffi, info, m, mt = g1_init(grad=a.build == "grad")
    B = a.nworld
    q, v, u, warm, fric = corpus(B, 5)
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    com = jnp.zeros((B, int(info["com_words"])), jnp.float32)
    ins = [f32(x) for x in (q, v, warm, u, fric)] + [com]
    r = ffi.forward(*ins, iterations=10, substeps=SUB, tape=True, contacts=True)
    tp, con = r[5], np.asarray(r[4])
    rng = np.random.default_rng(1)
    gq, gv = f32(rng.normal(size=(B, m.nq))), f32(rng.normal(size=(B, m.nv)))
    yc = f32(cotangent(info, con, rng, "forces_all") + cotangent(info, con, rng, "wrench"))
    fns = {"backward": jax.jit(lambda t: ffi.backward(t, ins[4], com, gq, gv, substeps=SUB)[0])}
    if a.build == "grad":
        fns["backward_g_contacts"] = jax.jit(lambda t: ffi.backward(t, ins[4], com, gq, gv, substeps=SUB,
                                                                     g_contacts=yc)[0])
    fwd = jax.jit(lambda *x: ffi.forward(*x, iterations=10, substeps=SUB, tape=True, contacts=True)[0])
    fns["forward_tape"] = lambda t: fwd(*ins)
    res = dict(check="time", build=a.build, nworld=B, gpu=__import__("jax").devices()[0].device_kind, ms={})
    for name, f in fns.items():
        best = None
        for _ in range(a.rounds):
            f(tp).block_until_ready()
            ts = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                f(tp).block_until_ready()
                ts.append(time.perf_counter() - t0)
            md = float(np.median(ts)) * 1e3
            best = md if best is None else min(best, md)
        res["ms"][name] = best
        print(f"{a.build} {name}: {best:.3f} ms")
    sz = info.get("sizeof_adj")
    res["sizeof_adj"] = sz
    dump(res, a.out)


def cmd_terfoot(a):
    """The VJP of a random cotangent on foot_normal (B, 4: heel and toe per foot, the toe weight w(x) of each foot
    contact; forest_engine.foot_offset foot_split) on the G1's capsule feet against terrain boxes (the foot_offset
    terrain build, TerrainConfig(geoms="feet"); sand_elem_check's 4 x 2 grid of cells, tops at the corpus states' floor
    height, no sand fields: every foot contact takes the offset) against float64 MuJoCo C central differences of the
    same split (the last substep's mj_contactForce normal forces, weights from the contact points; each foot contact's
    distance d + s(x), sand_elem_check.c_split), on corpus states whose forward split agrees with C's (--agree). The
    terrain capsule-box contacts take their VJP in the cooperative pass (cb_coop): 2026-11-21 the toe weight's
    foot-pose term was missing there (median qpos error 0.06 to 0.4); this is its regression check: exit 1 when a
    block's median relative error exceeds --accept. One random direction per world and block (qpos with the root
    rotation fixed, qvel, ctrl, offsets)."""
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(REPO / "scripts" / "forest"))
    import sand_elem_check as S
    m, fp, P, info, ffi = S.engine("foff", True, True, max_rows=a.max_rows)
    s, ter, emap = S.scene(a.scene, a.seed)
    ter.material = None
    ffi.set_terrain(ter)
    B = a.nworld
    z = S.corpus(B, a.seed + 7)
    rng = np.random.default_rng(a.seed + 1)
    offs = rng.uniform(0.0, a.smax, (B, 4))
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    step = ffi.make_step(a.iterations, SUB, 1, 0, contacts=True)
    warm, u, fr = f32(z["qacc_warmstart"]), f32(z["ctrl"]), jnp.ones((B,), jnp.float32)
    com = f32(S.com_of(m, P, offs, B))
    ffi.read_diagnostics()
    out = step(f32(z["qpos"]), f32(z["qvel"]), warm, u, fr, com)
    dg = ffi.read_diagnostics()
    FN = np.asarray(ffi.contact_table(out[3], SUB)["foot_normal"], np.float64)
    yf = rng.standard_normal((B, 4))

    def loss(q, v, uu, c):
        return (f32(yf) * ffi.contact_table(step(q, v, warm, uu, fr, c)[3], SUB)["foot_normal"]).sum()
    g = jax.grad(loss, argnums=(0, 1, 2, 3))(f32(z["qpos"]), f32(z["qvel"]), u, com)
    gs = [np.asarray(x, np.float64) for x in g]
    gs[3] = gs[3][:, 3:]
    s.opt.iterations, s.opt.tolerance, s.opt.ls_tolerance = a.c_iterations, 1e-12, 1e-12
    fg = S.scene_feet(s, m, fp)
    kfn = lambda e, p_: (1.0, np.zeros(3))
    csplit = lambda *x: S.c_split(s, fg, emap, kfn, *x).sum(0)
    rows = {"qpos": [], "qvel": [], "ctrl": [], "offsets": []}
    fwd = []
    for i in range(B):
        base = [np.float64(np.float32(np.asarray(z[k][i], np.float64))) for k in ("qpos", "qvel", "ctrl",
                                                                                   "qacc_warmstart")]
        o = np.float64(np.float32(offs[i]))
        Fc = csplit(*base, o)
        fwd.append(np.abs(Fc - FN[i]).max() / max(np.abs(Fc).max(), 1.0))
        for k, name in enumerate(rows):
            an = gs[k][i]
            dv = rng.standard_normal(an.shape)
            if name == "qpos":
                dv[3:7] = 0.0
            dv /= np.linalg.norm(dv)
            x = [b.copy() for b in base] + [np.maximum(o, a.eps) if name == "offsets" else o.copy()]
            xp, xm = [t.copy() for t in x], [t.copy() for t in x]
            xp[k if k < 3 else 4] += a.eps * dv
            xm[k if k < 3 else 4] -= a.eps * dv
            fd = ((yf[i] * csplit(*xp)).sum() - (yf[i] * csplit(*xm)).sum()) / (2 * a.eps)
            gg = an @ dv
            rows[name].append(abs(fd - gg) / max(abs(fd), abs(gg), 1e-9))
    fwd = np.asarray(fwd)
    agree = fwd < a.agree
    res = dict(check="foot_normal VJP on terrain boxes vs MuJoCo C", nworld=B, scene=a.scene, smax=a.smax,
               iterations=a.iterations, eps=a.eps, nefc=int(info["nefc"]),
               row_overflow_worlds=dg["fwd_row_overflow_worlds"],
               contact_overflow_worlds=dg["fwd_contact_overflow_worlds"], forward_rel_median=float(np.median(fwd)),
               worlds_used=int(agree.sum()), accept=a.accept, blocks={})
    for k, v in rows.items():
        res["blocks"][k] = stats(np.asarray(v)[agree])
    res["accepted"] = bool(agree.sum()) and all(r is None or r["median"] <= a.accept for r in res["blocks"].values())
    dump(res, a.out)
    sys.exit(0 if res["accepted"] else 1)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fd")
    f.add_argument("--nworld", type=int, default=32)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--iterations", type=int, default=50)
    f.add_argument("--eps", type=float, default=1e-6)
    f.add_argument("--losses", default="forces_last,forces_all,wrench,geometry")
    f.add_argument("--npz")
    f.add_argument("--only", help="world indices (debugging)")
    f.add_argument("--model", help="an MJCF instead of the mjlab G1 (model_init: e.g. a G1 with joint equalities)")
    f.add_argument("--states", help="with --model: an npz of qpos frames (model_states)")
    f.add_argument("--states-key", default="qpos")
    f.add_argument("--qvel-std", type=float, default=0.2, help="with --model: the states' qvel spread")
    f.add_argument("--blocks", default="qpos,qvel,ctrl",
                   help="the inputs differentiated (qvel,ctrl: the first substep's collision fixed, e.g. face-face "
                        "box contacts whose count changes with the pose)")
    f.add_argument("--out")
    s = sp.add_parser("slide")
    s.add_argument("--mus", default="0.2,0.4,0.6,0.8,1.0")
    s.add_argument("--span", type=float, default=3.0)
    s.add_argument("--step", type=float, default=0.1)
    s.add_argument("--seconds", type=float, default=2.0)
    s.add_argument("--acc", type=float, default=0.05)
    s.add_argument("--iterations", type=int, default=50)
    s.add_argument("--out")
    w = sp.add_parser("mjw")
    w.add_argument("--nworld", type=int, default=512)
    w.add_argument("--nmat", type=int, default=8)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--iterations", type=int, default=10)
    w.add_argument("--default-material", action="store_true")
    w.add_argument("--fric", type=float, default=0.7, help="the friction input (overridden by the slot's material)")
    w.add_argument("--out")
    g = sp.add_parser("matgrad")
    g.add_argument("--nworld", type=int, default=32)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--iterations", type=int, default=50)
    g.add_argument("--eps", type=float, default=1e-6)
    g.add_argument("--forces", action="store_true")
    g.add_argument("--out")
    d = sp.add_parser("dump")
    d.add_argument("--scene", default="g1", choices=("g1", "box", "course"))
    d.add_argument("--build", default="grad", choices=("default", "grad"))
    d.add_argument("--api", default="jax", choices=("jax", "torch"))
    d.add_argument("--nworld", type=int, default=256)
    d.add_argument("--out", required=True)
    c = sp.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--keys")
    c.add_argument("--out")
    tm = sp.add_parser("time")
    tm.add_argument("--build", default="grad", choices=("default", "grad"))
    tm.add_argument("--nworld", type=int, default=4096)
    tm.add_argument("--reps", type=int, default=20)
    tm.add_argument("--rounds", type=int, default=3)
    tm.add_argument("--out")
    ts = sp.add_parser("terslide")
    ts.add_argument("--patch", type=float, default=0.2)
    ts.add_argument("--fric-range", default="0.3,1.6")
    ts.add_argument("--nfric", type=int, default=4)
    ts.add_argument("--seed", type=int, default=0)
    ts.add_argument("--span", type=float, default=3.0)
    ts.add_argument("--step", type=float, default=0.2)
    ts.add_argument("--seconds", type=float, default=2.0)
    ts.add_argument("--acc", type=float, default=0.05)
    ts.add_argument("--iterations", type=int, default=50)
    ts.add_argument("--out")
    tw = sp.add_parser("termjw")
    tw.add_argument("--nworld", type=int, default=512)
    tw.add_argument("--nbox", type=int, default=9)
    tw.add_argument("--seed", type=int, default=0)
    tw.add_argument("--iterations", type=int, default=10)
    tw.add_argument("--out")
    tf = sp.add_parser("terfoot")
    tf.add_argument("--nworld", type=int, default=32)
    tf.add_argument("--scene", default="k1", help="sand_elem_check's grid scene (k1, mixed, flat: the cells' layout)")
    tf.add_argument("--seed", type=int, default=1)
    tf.add_argument("--smax", type=float, default=0.02)
    tf.add_argument("--iterations", type=int, default=50)
    tf.add_argument("--c-iterations", type=int, default=200)
    tf.add_argument("--eps", type=float, default=1e-6)
    tf.add_argument("--agree", type=float, default=1e-3)
    tf.add_argument("--accept", type=float, default=1e-3)
    tf.add_argument("--max-rows", type=int)
    tf.add_argument("--out")
    tg = sp.add_parser("tergrad")
    tg.add_argument("--nworld", type=int, default=45)
    tg.add_argument("--nbox", type=int, default=9)
    tg.add_argument("--seed", type=int, default=0)
    tg.add_argument("--iterations", type=int, default=50)
    tg.add_argument("--eps", type=float, default=1e-6)
    tg.add_argument("--out")
    a = p.parse_args()
    {"terslide": cmd_terslide, "termjw": cmd_termjw, "tergrad": cmd_tergrad, "time": cmd_time, "fd": cmd_fd, "slide": cmd_slide, "mjw": cmd_mjw, "matgrad": cmd_matgrad, "dump": cmd_dump,
     "compare": cmd_compare, "terfoot": cmd_terfoot}[a.cmd](a)


if __name__ == "__main__":
    main()

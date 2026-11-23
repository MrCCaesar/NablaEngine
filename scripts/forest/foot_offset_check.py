"""Per-world foot support offsets (PhysicsParams field foot_offset) and the foot normal force output
(jax_ffi.init(foot_normal=True)) on the forest engine's G1 (forest_engine.foot_offset; docs/api.md "Foot offsets").

The offset of a foot contact is s(x) = s_h + (s_t - s_h) clip((x - x_h) / (x_t - x_h), 0, 1), x the contact point
along the foot (the ankle-roll body's x axis), per foot (left heel, left toe, right heel, right toe): MuJoCo's
contact with includemargin -s(x) (pair margin -s for a uniform offset). The references apply exactly that:

  ref      (MuJoCo Warp) one control step of --nworld corpus states, each world with random offsets U(0, --smax)
           (--mode patch: the four drawn independently, each foot contact's includemargin set to -s(x) between
           MuJoCo Warp's collision and make_constraint; --mode margin: one offset per world on all four, as the foot
           pairs' pair_margin -s, a batched model field); writes inputs, offsets and outputs to OUT.npz
  run      (JAX) the engine on REF's inputs: --mode field (the offsets of REF), zero (the field, all offsets 0),
           plain (the build without the field); qvel against REF, and with --dump the outputs and the VJP of fixed
           random cotangents (qpos, qvel, ctrl, and the parameters' cotangent) for `compare`; --foot-normal adds the
           foot_normal output (with --contact-grad its VJP: a random cotangent on it too)
  parity   (torch) run's dump through torch_ffi (the same options, inputs and cotangents), for `compare`
  compare  bitwise equality of two dumps (the keys both hold; --skip KEYS)
  fd       (JAX + float64 MuJoCo C) the engine's exact mode (clamp_kv 1, normal 0) VJP of a --steps-step loss (random
           linear functional of the final qpos, qvel) w.r.t. the offsets, qpos and qvel against central differences
           of MuJoCo C with the includemargin patch, on corpus states (worlds whose forward agrees with C); --terrain:
           the terrain fixture's course and states (the engine's terrain build; C on the fixture's scene)
  stand    (JAX) the G1 standing from its keyframe for --seconds with s_h = --sh, s_t = 0 on both feet (and 0 for a
           baseline), servos stiffened by --gain-scale (the mjlab gains need a policy to stand): the nominal heel / toe
           sole points' height and the foot's heel-down pitch
  force    (JAX + float64 MuJoCo C) foot_normal: equals the per-foot sums of the table's force_normal; its VJP (a
           random cotangent, contact_grad, exact mode) against central differences of the same split in MuJoCo C
           (the last substep's contact forces, mj_contactForce, weights from the contact points)
  time     (JAX) forward and forward + backward per call at --nworld worlds (the corpus tiled; --terrain: the
           terrain fixture's course and states): --mode field (offsets U(0, --smax)) / plain (the build without it)

  python scripts/forest/foot_offset_check.py ref --out runs/foot_offset/ref_patch.npz
  python scripts/forest/foot_offset_check.py run --ref runs/foot_offset/ref_patch.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
LINE = (-0.064, 0.142)
SUB = 4


def load():
    from nablaengine import fixtures
    m, mt = fixtures.load_model()
    return m, [int(p) for p in mt["foot_pairs"]], fixtures.load_corpus("mf_s0")


def foot_bodies(m):
    import mujoco
    return [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in FEET]


def offset_of(x, sh, st):
    w = np.clip((x - LINE[0]) / (LINE[1] - LINE[0]), 0.0, 1.0)
    return sh + (st - sh) * w, w


def foot_geoms(m, fp, names=None):
    """{geom id: foot (0 left, 1 right)} of the foot pairs' geoms on the feet; names: the G1 model's geom names to
    look up in m instead (a scene with the G1 and a terrain)."""
    fb = foot_bodies(m)
    out = {}
    if names is not None:
        import mujoco
        for n in names:
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
            out[g] = fb.index(int(m.geom_bodyid[g]))
        return out
    for p in fp:
        for g in (int(m.pair_geom1[p]), int(m.pair_geom2[p])):
            if int(m.geom_bodyid[g]) in fb:
                out[g] = fb.index(int(m.geom_bodyid[g]))
    return out


def contact_offsets(m, fg, geom, pos, xpos, xmat, offs):
    """Per contact (geom (n, 2), pos (n, 3)) of one world: (s, side, w); side -1 except for a foot geom (fg: foot_geoms)
    against a static geom (the floor, the terrain). xmat (nbody, 9). offs (4,)."""
    fb = foot_bodies(m)
    n = len(pos)
    s, side, w = np.zeros(n), -np.ones(n, np.int64), np.zeros(n)
    for i in range(n):
        g0, g1 = int(geom[i][0]), int(geom[i][1])
        for a, o in ((g0, g1), (g1, g0)):
            if a in fg and int(m.body_weldid[m.geom_bodyid[o]]) == 0:
                k = fg[a]
                b = fb[k]
                ax = np.asarray(xmat[b]).reshape(3, 3)[:, 0]
                x = float(ax @ (np.asarray(pos[i]) - np.asarray(xpos[b])))
                s[i], w[i] = offset_of(x, offs[2 * k], offs[2 * k + 1])
                side[i] = k
    return s, side, w


def c_substep(m, d, fg, offs, rec=None):
    """One mj_step of MuJoCo C with each foot contact's distance d + s(x) (contacts with d + s >= 0 excluded): mj_step's
    stages with the patch between mj_collision and mj_makeConstraint (offs None: mj_step bitwise). rec: a list that
    gets (contact forces, side, w) of the substep's contacts after the solve."""
    import mujoco
    mujoco.mj_checkPos(m, d)
    mujoco.mj_checkVel(m, d)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_camlight(m, d)
    mujoco.mj_flex(m, d)
    mujoco.mj_tendon(m, d)
    mujoco.mj_makeM(m, d)
    mujoco.mj_factorM(m, d)
    mujoco.mj_collision(m, d)
    side = w = None
    if d.ncon:
        geom = np.asarray([d.contact.geom1[:d.ncon], d.contact.geom2[:d.ncon]]).T
        s, side, w = contact_offsets(m, fg, geom, d.contact.pos[:d.ncon], d.xpos, d.xmat,
                                     np.zeros(4) if offs is None else offs)
        if offs is not None:
            for i in range(d.ncon):
                if side[i] >= 0:
                    d.contact.dist[i] += s[i]
                    if not d.contact.dist[i] < 0:
                        d.contact.exclude[i] = 1
    mujoco.mj_makeConstraint(m, d)
    mujoco.mj_island(m, d)
    mujoco.mj_transmission(m, d)
    mujoco.mj_projectConstraint(m, d)
    mujoco.mj_forwardSkip(m, d, mujoco.mjtStage.mjSTAGE_POS, 0)
    mujoco.mj_checkAcc(m, d)
    if rec is not None:
        f = np.zeros(6)
        fn = []
        for i in range(d.ncon):
            if d.contact.efc_address[i] < 0:
                fn.append(0.0)
                continue
            mujoco.mj_contactForce(m, d, i, f)
            fn.append(f[0])
        rec.append((np.asarray(fn), side, w))
    mujoco.mj_implicit(m, d)


def set_friction(m, fp, mu):
    """The foot pairs' sliding friction in MuJoCo C (the engine's friction input)."""
    for p in fp:
        m.pair_friction[p, :2] = mu


def c_run(m, fp, q, v, u, warm, offs, steps, iterations, rec=None, fg=None):
    import mujoco
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q, v, u, warm
    fg = foot_geoms(m, fp) if fg is None else fg
    for _ in range(steps * SUB):
        c_substep(m, d, fg, offs, rec)
    return d.qpos.copy(), d.qvel.copy()


def cmd_ref(a):
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import constraint
    m, fp, z = load()
    m.opt.iterations = a.iterations
    B = a.nworld
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(z["qpos"]))[:B]
    if a.mode == "margin":
        offs = np.repeat(rng.uniform(0.0, a.smax, (B, 1)), 4, 1)
    else:
        offs = rng.uniform(0.0, a.smax, (B, 4))
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=64, njmax=256)
    pfr = np.tile(m.pair_friction, (B, 1, 1)).astype(np.float32)
    for p in fp:
        pfr[:, p, :2] = np.asarray(z["friction"][idx], np.float32)[:, None]
    mw.pair_friction = wp.array(pfr, dtype=mw.pair_friction.dtype)
    if a.mode == "margin":
        pm = np.tile(m.pair_margin, (B, 1)).astype(np.float32)
        for p in fp:
            pm[:, p] = -offs[:, 0]
        mw.pair_margin = wp.array(pm, dtype=float)
    else:
        orig = constraint.make_constraint
        fg = foot_geoms(m, fp)

        def patched(mm, dd):
            n = int(dd.nacon.numpy()[0])
            if n:
                geom, pos = dd.contact.geom.numpy()[:n], dd.contact.pos.numpy()[:n]
                wid, im = dd.contact.worldid.numpy()[:n], dd.contact.includemargin.numpy()
                xpos, xmat = dd.xpos.numpy(), dd.xmat.numpy().reshape(B, -1, 9)
                for wi in np.unique(wid):
                    sel = np.nonzero(wid == wi)[0]
                    s, _, _ = contact_offsets(m, fg, geom[sel], pos[sel], xpos[wi], xmat[wi], offs[wi])
                    im[sel] = im[sel] - s
                wp.copy(dd.contact.includemargin, wp.array(im.astype(np.float32)))
            return orig(mm, dd)
        constraint.make_constraint = patched
    for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
        wp.copy(getattr(dw, k), wp.array(np.ascontiguousarray(z[k][idx], np.float32)))
    for _ in range(SUB):
        mjw.step(mw, dw)
    nacon = int(dw.nacon.numpy()[0])
    ncon = np.bincount(dw.contact.worldid.numpy()[:nacon], minlength=B)
    np.savez(a.out, idx=idx, qpos=z["qpos"][idx], qvel=z["qvel"][idx], qacc_warmstart=z["qacc_warmstart"][idx],
             ctrl=z["ctrl"][idx], friction=z["friction"][idx], offsets=offs.astype(np.float32),
             out_qpos=dw.qpos.numpy(), out_qvel=dw.qvel.numpy(), ncon=ncon, iterations=a.iterations, mode=a.mode)
    print(f"ref ({a.mode}): {B} worlds, mean contacts {ncon.mean():.2f}; {a.out}")


def engine(m, fp, plain=False, foot_normal=False, contact_grad=False, terrain=None):
    from nablaengine import jax_ffi
    from nablaengine.params import PhysicsParams
    P = None if plain else PhysicsParams(("foot_offset",))
    kw = dict(model=m, foot_pairs=fp, com_body="torso_link")
    if P is not None:
        kw["params"] = P
    if foot_normal:
        kw["foot_normal"] = True
    if contact_grad:
        kw["contact_grad"] = True
    if terrain is not None:
        kw["terrain"] = terrain
    info = jax_ffi.init(**kw)
    return P, info


def com_of(m, P, offs, B):
    com = np.zeros((B, 3), np.float32)
    if P is None:
        return com
    return np.concatenate([com, P.pack(m.nv, foot_offset=np.broadcast_to(offs, (B, 4)))], 1)


def cmd_run(a):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    r = dict(np.load(a.ref))
    m, fp, _ = load()
    B = len(r["qpos"])
    P, info = engine(m, fp, a.mode == "plain", a.foot_normal, a.contact_grad)
    offs = r["offsets"] if a.mode == "field" else np.zeros((B, 4), np.float32)
    com = jnp.asarray(com_of(m, P, offs, B))
    f = lambda k: jnp.asarray(np.asarray(r[k], np.float32))
    it = int(r["iterations"])
    con = a.foot_normal
    step = jax_ffi.make_step(it, SUB, a.clamp_kv, a.normal, contacts=con)
    out = step(f("qpos"), f("qvel"), f("qacc_warmstart"), f("ctrl"), f("friction"), com)
    q1, v1 = np.asarray(out[0]), np.asarray(out[1])
    res = dict(mode=a.mode, nworld=B, iterations=it, engine=info["engine"])
    if a.mode == "field":
        ev = np.linalg.norm(v1 - r["out_qvel"], axis=1) / np.maximum(np.linalg.norm(r["out_qvel"], axis=1), 1e-6)
        res.update(qvel_rel_median=float(np.median(ev)), qvel_rel_p90=float(np.quantile(ev, 0.9)),
                   qvel_rel_p99=float(np.quantile(ev, 0.99)), qvel_rel_max=float(ev.max()),
                   qpos_abs_max=float(np.abs(q1 - r["out_qpos"]).max()))
        fo = jax_ffi.forward(f("qpos"), f("qvel"), f("qacc_warmstart"), f("ctrl"), f("friction"), com, iterations=it,
                             substeps=SUB)
        nc = np.asarray(fo[3])[:, 1]
        res.update(ncon_equal=float((nc == r["ncon"]).mean()), ncon_mean=float(nc.mean()),
                   flagged=int((np.asarray(fo[3])[:, 3] != 0).sum()))
        same = nc == r["ncon"]
        res["qvel_rel_median_same_ncon"] = float(np.median(ev[same])) if same.any() else None
    if a.dump:
        rng = np.random.default_rng(2)
        yq, yv = jnp.asarray(rng.standard_normal(q1.shape), jnp.float32), jnp.asarray(rng.standard_normal(v1.shape),
                                                                                       jnp.float32)
        yf = None
        if con:
            tab = jax_ffi.contact_table(out[3], SUB)
            yf = jnp.asarray(rng.standard_normal(tab["foot_normal"].shape), jnp.float32)

        def loss(q, v, u, c):
            o = step(q, v, f("qacc_warmstart"), u, f("friction"), c)
            L = (yq * o[0]).sum() + (yv * o[1]).sum()
            if con and a.contact_grad:
                L = L + (yf * jax_ffi.contact_table(o[3], SUB)["foot_normal"]).sum()
            return L
        g = jax.grad(loss, argnums=(0, 1, 2, 3))(f("qpos"), f("qvel"), f("ctrl"), com)
        dump = dict(q1=q1, v1=v1, g_qpos=np.asarray(g[0]), g_qvel=np.asarray(g[1]), g_ctrl=np.asarray(g[2]),
                    g_com_param=np.asarray(g[3])[:, 3:])
        if con:
            tab = jax_ffi.contact_table(out[3], SUB)
            dump.update(foot_normal=np.asarray(tab["foot_normal"]), con=np.asarray(out[3]))
        np.savez(a.dump, **dump)
        res["dump"] = a.dump
    print(json.dumps(res, indent=1))


def cmd_parity(a):
    """(torch) run's dump through torch_ffi with the same build options, the same inputs and cotangents."""
    import torch
    from nablaengine import torch_ffi
    from nablaengine.params import PhysicsParams
    r = dict(np.load(a.ref))
    m, fp, _ = load()
    B = len(r["qpos"])
    P = None if a.mode == "plain" else PhysicsParams(("foot_offset",))
    kw = dict(model=m, foot_pairs=fp, com_body="torso_link")
    if P is not None:
        kw["params"] = P
    if a.foot_normal:
        kw["foot_normal"] = True
    if a.contact_grad:
        kw["contact_grad"] = True
    torch_ffi.init(**kw)
    offs = r["offsets"] if a.mode == "field" else np.zeros((B, 4), np.float32)
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32)).cuda()
    com = t(com_of(m, P, offs, B)).requires_grad_()
    xs = [t(r["qpos"]).requires_grad_(), t(r["qvel"]).requires_grad_(), t(r["ctrl"]).requires_grad_()]
    it = int(r["iterations"])
    o = torch_ffi.step(xs[0], xs[1], t(r["qacc_warmstart"]), xs[2], t(r["friction"]), com, iterations=it,
                       substeps=SUB, clamp_kv=a.clamp_kv, normal=a.normal, contacts=a.foot_normal)
    rng = np.random.default_rng(2)
    yq, yv = t(rng.standard_normal(tuple(o[0].shape))), t(rng.standard_normal(tuple(o[1].shape)))
    L = (yq * o[0]).sum() + (yv * o[1]).sum()
    if a.foot_normal:
        fnr = torch_ffi.contact_table(o[3], SUB)["foot_normal"]
        yf = t(rng.standard_normal(tuple(fnr.shape)))
        if a.contact_grad:
            L = L + (yf * fnr).sum()
    L.backward()
    dump = dict(q1=o[0].detach().cpu().numpy(), v1=o[1].detach().cpu().numpy(),
                **{f"g_{n}": x.grad.cpu().numpy() for n, x in zip(("qpos", "qvel", "ctrl"), xs)},
                g_com_param=com.grad.cpu().numpy()[:, 3:])
    if a.foot_normal:
        dump["foot_normal"] = fnr.detach().cpu().numpy()
        dump["con"] = o[3].detach().cpu().numpy()
    np.savez(a.dump, **dump)
    print(json.dumps(dict(dump=a.dump, nworld=B)))


def cmd_compare(a):
    x, y = np.load(a.a), np.load(a.b)
    keys = [k for k in x.files if k in y.files and k not in a.skip]
    same = {k: bool(np.array_equal(x[k], y[k])) for k in keys}
    print(json.dumps(dict(bitwise_equal=same, all=all(same.values()))))
    sys.exit(0 if all(same.values()) else 1)


def cmd_fd(a):
    import jax
    import jax.numpy as jnp
    import mujoco
    from nablaengine import jax_ffi, terrain as T
    m, fp, z = load()
    fgn = None
    if a.terrain:
        names = [m.geom(g).name for g in foot_geoms(m, fp)]
        r = dict(np.load(REPO / TERRAIN_STATES))
        z = dict(qpos=r["qpos"], qvel=r["qvel"], ctrl=r["ctrl"], qacc_warmstart=r["warm"],
                 friction=np.ones(len(r["qpos"]), np.float32))
        P, _ = engine(m, fp, terrain=T.TerrainConfig(geoms=str(r["geoms"]), mode="analytic"))
        scene = mujoco.MjModel.from_binary_path(str(REPO / TERRAIN_SCENE))
        jax_ffi.set_terrain(T.Terrain.from_model(scene))
        fgn = foot_geoms(scene, None, names)
        for p in range(scene.npair):
            if int(scene.pair_geom1[p]) in fgn or int(scene.pair_geom2[p]) in fgn:
                scene.pair_friction[p, :2] = 1.0
        m, fp = scene, []
    else:
        P, _ = engine(m, fp)
    rng = np.random.default_rng(a.seed)
    cand = rng.permutation(len(z["qpos"]))[:a.nworld]
    offs = rng.uniform(0.0, a.smax, (len(cand), 4))
    B = len(cand)
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    step = jax_ffi.make_step(a.iterations, SUB, 1, 0)
    yq, yv = rng.standard_normal((B, m.nq)), rng.standard_normal((B, m.nv))
    warm, u, fr = f32(z["qacc_warmstart"][cand]), f32(z["ctrl"][cand]), f32(z["friction"][cand])

    def loss(q, v, c):
        for _ in range(a.steps):
            q, v, w = step(q, v, warm, u, fr, c)[:3]
        return (f32(yq) * q).sum() + (f32(yv) * v).sum(), (q, v)
    (L, (qe, ve)), g = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(
        f32(z["qpos"][cand]), f32(z["qvel"][cand]), f32(com_of(m, P, offs, B)))
    gq, gv, go = np.asarray(g[0], np.float64), np.asarray(g[1], np.float64), np.asarray(g[2], np.float64)[:, 3:]
    qe, ve = np.asarray(qe, np.float64), np.asarray(ve, np.float64)
    m.opt.iterations, m.opt.tolerance, m.opt.ls_tolerance = a.c_iterations, 1e-12, 1e-12
    res = dict(nworld=B, steps=a.steps, eps=a.eps, smax=a.smax, iterations=a.iterations)
    rows = {"offsets": [], "qpos": [], "qvel": []}
    agree = []
    for i, w in enumerate(cand):
        base = [np.asarray(z[k][w], np.float64) for k in ("qpos", "qvel", "ctrl", "qacc_warmstart")]
        o = offs[i]
        base = [np.float64(np.float32(x)) for x in base]
        set_friction(m, fp, float(np.float32(z["friction"][w])))
        qc, vc = c_run(m, fp, *base, np.float64(np.float32(o)), a.steps, a.c_iterations, fg=fgn)
        e = np.linalg.norm(ve[i] - vc) / max(np.linalg.norm(vc), 1e-6)
        agree.append(e < a.agree)

        def Lc(q, v, oo):
            qq, vv = c_run(m, fp, q, v, base[2], base[3], oo, a.steps, a.c_iterations, fg=fgn)
            return yq[i] @ qq + yv[i] @ vv
        for name, an in (("offsets", go[i]), ("qpos", gq[i]), ("qvel", gv[i])):
            dvec = rng.standard_normal(an.shape)
            if name == "qpos":
                dvec[3:7] = 0.0
            dvec /= np.linalg.norm(dvec)
            if name == "offsets":
                o0 = np.float64(np.float32(o))
                o0 = np.maximum(o0, a.eps)
                fd = (Lc(base[0], base[1], o0 + a.eps * dvec) - Lc(base[0], base[1], o0 - a.eps * dvec)) / (2 * a.eps)
            elif name == "qpos":
                fd = (Lc(base[0] + a.eps * dvec, base[1], np.float64(np.float32(o)))
                      - Lc(base[0] - a.eps * dvec, base[1], np.float64(np.float32(o)))) / (2 * a.eps)
            else:
                fd = (Lc(base[0], base[1] + a.eps * dvec, np.float64(np.float32(o)))
                      - Lc(base[0], base[1] - a.eps * dvec, np.float64(np.float32(o)))) / (2 * a.eps)
            gg = an @ dvec
            rows[name].append(abs(fd - gg) / max(abs(fd), abs(gg), 1e-9))
    agree = np.asarray(agree)
    res["forward_agree_share"] = float(agree.mean())
    for k, v in rows.items():
        x = np.asarray(v)[agree]
        res[k] = dict(rel_median=float(np.median(x)), rel_p90=float(np.quantile(x, 0.9)), rel_max=float(x.max()),
                      n=int(len(x)))
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_stand(a):
    import jax.numpy as jnp
    import mujoco
    from nablaengine import jax_ffi
    m, fp, _ = load()
    m.actuator_gainprm[:, 0] *= a.gain_scale
    m.actuator_biasprm[:, 1:3] *= a.gain_scale
    P, _ = engine(m, fp)
    fb = foot_bodies(m)
    zs = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) == fb[0] and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            ax = R.reshape(3, 3)[:, 2]
            for sg in (-1, 1):
                zs.append((m.geom_pos[g] + sg * m.geom_size[g, 1] * ax)[2] - m.geom_size[g, 0])
    z_sole = float(min(zs))
    pts = np.asarray([[LINE[0], 0.0, z_sole], [LINE[1], 0.0, z_sole]])
    cases = {"baseline": np.zeros(4), "heel": np.asarray([a.sh, 0.0, a.sh, 0.0])}
    B = len(cases)
    q0 = np.tile(m.key_qpos[0], (B, 1)).astype(np.float32)
    u0 = np.tile(m.key_ctrl[0], (B, 1)).astype(np.float32)
    com = jnp.asarray(com_of(m, P, np.stack(list(cases.values())), B))
    step = jax_ffi.make_step(m.opt.iterations, SUB, 0, 3)
    q, v, w = jnp.asarray(q0), jnp.zeros((B, m.nv), jnp.float32), jnp.zeros((B, m.nv), jnp.float32)
    fr = jnp.full((B,), float(m.pair_friction[fp[0], 0]), jnp.float32)
    n = int(round(a.seconds / (SUB * m.opt.timestep)))
    for _ in range(n):
        q, v, w = step(q, v, w, jnp.asarray(u0), fr, com)
    q = np.asarray(q, np.float64)
    d = mujoco.MjData(m)
    res = dict(seconds=a.seconds, sh=a.sh, z_sole=z_sole, line=list(LINE), gain_scale=a.gain_scale)
    for i, k in enumerate(cases):
        d.qpos[:] = q[i]
        mujoco.mj_kinematics(m, d)
        out = {}
        for side, b in zip(("left", "right"), fb):
            R = d.xmat[b].reshape(3, 3)
            hz = [float((d.xpos[b] + R @ p)[2]) for p in pts]
            out[side] = dict(heel_z_mm=1e3 * hz[0], toe_z_mm=1e3 * hz[1],
                             heel_down_deg=float(np.degrees(np.arcsin((hz[1] - hz[0]) / (LINE[1] - LINE[0])))))
        out["max_abs_qvel"] = float(np.abs(np.asarray(v)[i]).max())
        res[k] = out
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def c_foot_normal(m, fp, q, v, u, warm, offs, iterations):
    rec = []
    c_run(m, fp, q, v, u, warm, offs, 1, iterations, rec)
    fn, side, w = rec[-1]
    out = np.zeros(4)
    if side is not None:
        for i in range(len(fn)):
            if side[i] >= 0:
                out[2 * side[i]] += fn[i] * (1 - w[i])
                out[2 * side[i] + 1] += fn[i] * w[i]
    return out


def cmd_force(a):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    m, fp, z = load()
    P, info = engine(m, fp, foot_normal=True, contact_grad=True)
    rng = np.random.default_rng(a.seed)
    cand = rng.permutation(len(z["qpos"]))[:a.nworld]
    B = len(cand)
    offs = rng.uniform(0.0, a.smax, (B, 4))
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    step = jax_ffi.make_step(a.iterations, SUB, 1, 0, contacts=True)
    warm, u, fr = f32(z["qacc_warmstart"][cand]), f32(z["ctrl"][cand]), f32(z["friction"][cand])
    com = f32(com_of(m, P, offs, B))
    out = step(f32(z["qpos"][cand]), f32(z["qvel"][cand]), warm, u, fr, com)
    tab = jax_ffi.contact_table(out[3], SUB)
    fnv, pair = np.asarray(tab["force_normal"]), np.asarray(tab["pair"])
    side, wt, F = np.asarray(tab["foot_side"]), np.asarray(tab["toe_weight"]), np.asarray(tab["foot_normal"])
    fb = foot_bodies(m)
    pg = np.asarray([[int(m.pair_geom1[p]), int(m.pair_geom2[p])] for p in range(m.npair)])
    tot = np.zeros((B, 2))
    for k, b in enumerate(fb):
        on = np.zeros_like(fnv, bool)
        for p in fp:
            if b in (int(m.geom_bodyid[pg[p, 0]]), int(m.geom_bodyid[pg[p, 1]])):
                on |= pair == p
        tot[:, k] = (fnv * on).sum(1)
    sums = np.stack([F[:, 0] + F[:, 1], F[:, 2] + F[:, 3]], 1)
    res = dict(nworld=B, sum_abs_err_max=float(np.abs(sums - tot).max()), sum_max=float(np.abs(tot).max()),
               side_matches_pairs=bool(all(((side == k) == np.isin(pair, [p for p in fp if fb[k] in
                                       (int(m.geom_bodyid[pg[p, 0]]), int(m.geom_bodyid[pg[p, 1]]))])).all()
                                       for k in (0, 1))))
    yf = rng.standard_normal((B, 4))

    def loss(q, v, uu, c):
        o = step(q, v, warm, uu, fr, c)
        return (f32(yf) * jax_ffi.contact_table(o[3], SUB)["foot_normal"]).sum()
    g = jax.grad(loss, argnums=(0, 1, 2, 3))(f32(z["qpos"][cand]), f32(z["qvel"][cand]), u, com)
    gs = [np.asarray(x, np.float64) for x in g]
    gs[3] = gs[3][:, 3:]
    m.opt.iterations, m.opt.tolerance, m.opt.ls_tolerance = a.c_iterations, 1e-12, 1e-12
    rows = {"qpos": [], "qvel": [], "ctrl": [], "offsets": []}
    fwd = []
    for i, w in enumerate(cand):
        base = [np.float64(np.float32(np.asarray(z[k][w], np.float64))) for k in ("qpos", "qvel", "ctrl",
                                                                                   "qacc_warmstart")]
        o = np.float64(np.float32(offs[i]))
        set_friction(m, fp, float(np.float32(z["friction"][w])))
        Fc = c_foot_normal(m, fp, *base, o, a.c_iterations)
        fwd.append(np.abs(Fc - F[i]).max() / max(np.abs(Fc).max(), 1.0))
        for k, name in enumerate(("qpos", "qvel", "ctrl", "offsets")):
            an = gs[k][i]
            dv = rng.standard_normal(an.shape)
            if name == "qpos":
                dv[3:7] = 0.0
            dv /= np.linalg.norm(dv)
            x = [b.copy() for b in base] + [np.maximum(o, a.eps) if name == "offsets" else o.copy()]
            xp, xm = [t.copy() for t in x], [t.copy() for t in x]
            xp[k if k < 3 else 4] += a.eps * dv
            xm[k if k < 3 else 4] -= a.eps * dv
            fd = (yf[i] @ c_foot_normal(m, fp, *xp, a.c_iterations)
                  - yf[i] @ c_foot_normal(m, fp, *xm, a.c_iterations)) / (2 * a.eps)
            gg = an @ dv
            rows[name].append(abs(fd - gg) / max(abs(fd), abs(gg), 1e-9))
    fwd = np.asarray(fwd)
    agree = fwd < a.agree
    res.update(forward_rel_median=float(np.median(fwd)), forward_agree_share=float(agree.mean()))
    for k, v in rows.items():
        x = np.asarray(v)[agree]
        res[k] = dict(rel_median=float(np.median(x)), rel_p90=float(np.quantile(x, 0.9)), rel_max=float(x.max()),
                      n=int(len(x)))
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


TERRAIN_STATES = "fixtures/mjlab/mjw317_terrain_feet.npz"
TERRAIN_SCENE = "fixtures/mjlab/g1_terrain_feet.mjb"


def cmd_time(a):
    import time
    import jax
    import jax.numpy as jnp
    import mujoco
    from nablaengine import jax_ffi, terrain as T
    m, fp, z = load()
    ter = None
    if a.terrain:
        r = dict(np.load(REPO / TERRAIN_STATES))
        r["qacc_warmstart"] = r["warm"]
        r["friction"] = np.ones(len(r["qpos"]), np.float32)
        ter = T.TerrainConfig(geoms=str(r["geoms"]), mode="analytic")
    else:
        r = z
    P, info = engine(m, fp, a.mode == "plain", terrain=ter)
    if ter is not None:
        jax_ffi.set_terrain(T.Terrain.from_model(mujoco.MjModel.from_binary_path(str(REPO / TERRAIN_SCENE))))
    idx = np.arange(a.nworld) % len(r["qpos"])
    f = lambda k: jnp.asarray(np.asarray(r[k], np.float32)[idx])
    offs = np.random.default_rng(0).uniform(0.0, a.smax, (a.nworld, 4)) if a.mode == "field" else None
    com = (jnp.asarray(com_of(m, P, offs, a.nworld)) if P is not None
           else jnp.zeros((a.nworld, info["com_words"]), jnp.float32))
    it = jax_ffi.ITERATIONS if a.iterations is None else a.iterations
    step = jax_ffi.make_step(it, SUB)
    q, v, w, u, fr = f("qpos"), f("qvel"), f("qacc_warmstart"), f("ctrl"), f("friction")
    fwd = jax.jit(lambda q, v, w, u, fr, c: step(q, v, w, u, fr, c)[1])
    both = jax.jit(jax.grad(lambda q, v, w, u, fr, c: step(q, v, w, u, fr, c)[1].sum(), argnums=(0, 1, 3)))
    res = dict(mode=a.mode, terrain=bool(a.terrain), nworld=a.nworld, iterations=it, smax=a.smax,
               gpu=jax.devices()[0].device_kind)
    for key, fn in (("forward_ms", fwd), ("forward_backward_ms", both)):
        for _ in range(3):
            jax.block_until_ready(fn(q, v, w, u, fr, com))
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(q, v, w, u, fr, com))
            ts.append(1e3 * (time.perf_counter() - t0))
        res[key] = round(float(np.median(ts)), 3)
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("ref")
    p.add_argument("--mode", choices=("patch", "margin"), default="patch")
    p.add_argument("--nworld", type=int, default=512)
    p.add_argument("--smax", type=float, default=0.01)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sp.add_parser("run")
    p.add_argument("--ref", required=True)
    p.add_argument("--mode", choices=("field", "zero", "plain"), default="field")
    p.add_argument("--foot-normal", action="store_true")
    p.add_argument("--contact-grad", action="store_true")
    p.add_argument("--clamp-kv", type=int, default=0)
    p.add_argument("--normal", type=int, default=3)
    p.add_argument("--dump")
    p = sp.add_parser("parity")
    p.add_argument("--ref", required=True)
    p.add_argument("--mode", choices=("field", "zero", "plain"), default="field")
    p.add_argument("--foot-normal", action="store_true")
    p.add_argument("--contact-grad", action="store_true")
    p.add_argument("--clamp-kv", type=int, default=0)
    p.add_argument("--normal", type=int, default=3)
    p.add_argument("--dump", required=True)
    p = sp.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--skip", nargs="*", default=())
    for name in ("fd", "force"):
        p = sp.add_parser(name)
        p.add_argument("--nworld", type=int, default=48)
        p.add_argument("--steps", type=int, default=5)
        p.add_argument("--smax", type=float, default=0.01)
        p.add_argument("--eps", type=float, default=1e-6)
        p.add_argument("--iterations", type=int, default=50)
        p.add_argument("--c-iterations", type=int, default=200)
        p.add_argument("--agree", type=float, default=1e-3)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--terrain", action="store_true")
        p.add_argument("--out")
    p = sp.add_parser("stand")
    p.add_argument("--sh", type=float, default=0.01)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--gain-scale", type=float, default=20.0)
    p.add_argument("--out")
    p = sp.add_parser("time")
    p.add_argument("--mode", choices=("field", "plain"), default="field")
    p.add_argument("--smax", type=float, default=0.01)
    p.add_argument("--terrain", action="store_true")
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--iterations", type=int)
    p.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    {"ref": cmd_ref, "run": cmd_run, "parity": cmd_parity, "compare": cmd_compare, "fd": cmd_fd, "stand": cmd_stand, "force": cmd_force,
     "time": cmd_time}[a.cmd](a)


if __name__ == "__main__":
    main()

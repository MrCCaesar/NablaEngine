"""Per-element sand under the feet (TerrainConfig(terrain_sand=True): the terrain elements' sand fraction k_e scales the
per-world foot_offset; forest_engine.foot_offset, docs/terrain.md "Sand elements") on the forest engine's G1.

A foot contact on element e takes the distance d + k_e s(x) (s(x) the foot_offset interpolation); MuJoCo's equivalent is
that contact's margin -k_e s(x). The scenes are the terrain fixture's scene (fixtures/mjlab/g1_terrain_feet.mjb) with
its 8 boxes moved into a flat 4 x 2 grid of cells (tops at z = 0, the corpus states' floor height) and the heightfield
moved away; the engine takes Terrain.from_model of that scene, MuJoCo C / Warp the scene itself (explicit foot-terrain
pairs).

  dump     (JAX) one control step of --nworld corpus states on --scene with --build sand (the terrain_sand build) / foff
           (the foot_offset terrain build without it) / sandmat, foffmat (the same with terrain_material) / plain (no
           parameters), offsets U(0, --smax) (--offs rand) or 0; the outputs, the contact table (the words the foff
           build has too), foot_normal and the VJP of fixed random cotangents (qpos, qvel, ctrl, the parameter columns;
           with --contact-grad also foot_normal's) -> --out .npz
  compare  bitwise equality of two dumps (the keys both hold; --skip KEYS)
  mjw      (JAX + MuJoCo Warp) one control step on the mixed scene against MuJoCo Warp with each foot contact's
           includemargin -k_e s(x) (set between Warp's collision and make_constraint)
  ledge    (JAX + MuJoCo Warp) the foothold with the sand cells' tops at --dz (+-20 mm) against the hard cell at 0: the
           G1 settled with its left foot across the ledge, then one control step against MuJoCo Warp (margin -k_e s(x)),
           the k words, the split forces and the parameters
  force    (JAX + float64 MuJoCo C) foot_normal_split: sand + hard = foot_normal; forward against the same split of C's
           last-substep forces; the VJP of a random cotangent on both parts (--loss split; --loss state: a functional
           of the state instead) (contact_grad, exact mode) against C central differences (qpos, qvel, ctrl,
           offsets); --floor-k K: the model's floor plane at sand_other k = K instead of the terrain; --accept X: exit 1
           when a block's median relative error exceeds X (a gate; row_overflow_worlds and contact_overflow_worlds
           report worlds beyond the build's capacity, --max-rows)
  params   (JAX) foot_sand_params / foot_sand_presence on the standing G1: a mixed foothold (left heel on a sand cell,
           left toe on hard ground, right foot on another sand cell) and two sand cells with different parameters under
           the two feet; against a NumPy evaluation of the definition from the contact table
  boundary (JAX + float64 MuJoCo C) the gradient across a sand / hard boundary: the G1 settled on hard ground, shifted
           along x by delta (--span, --step) so that the left foot crosses the boundary (--scene k0: no sand, the
           control); L(delta) a fixed random functional of the state after --steps control steps; per delta the
           engine's dL/d delta (VJP) against central differences of the engine's L over --window and of C's L, and
           the engine's L against C's (forward_vs_c). --max-rows (default 400): a foot across a seam between two boxes
           has contacts with both, and with the default capacity (192 rows) the rows beyond it are dropped, so the
           engine's step is not C's (the all-hard control's VJP-vs-C median 6.6e-2 at 192 rows, 1e-5 at 400);
           row_overflow_worlds reports such worlds
  time     (JAX) forward and forward + backward at --nworld worlds on the terrain fixture's course: --build sand
           (--sand-k half: every other element sand, or one k for all; offsets U(0, --smax)) / foff (no sand fields) /
           plain

  python scripts/forest/sand_elem_check.py dump --build sand --scene k1 --out runs/sand_elem/sand_k1.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "forest"))
import foot_offset_check as F

SUB = 4
SCENE = "fixtures/mjlab/g1_terrain_feet.mjb"
COURSE_STATES = "fixtures/mjlab/mjw317_terrain_feet.npz"


def out_json(res, path):
    s = json.dumps(res, indent=1)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(s)
    print(s[:6000])


def stats(x):
    x = np.asarray(x, np.float64)
    if not x.size:
        return None
    return dict(n=int(x.size), median=float(np.median(x)), p90=float(np.quantile(x, 0.9)), max=float(x.max()))


def grid_scene(xs_l, xs_r, rows_l, rows_r, tops=None):
    """The fixture scene with its 8 boxes as cells: left row (y in [0, 1.2]) at x edges xs_l (5 edges, 4 cells), right
    row (y in [-1.2, 0]) at xs_r; rows_*: the cells' sand fields (4 x 4); tops: the 8 cells' top heights (default 0;
    a ledge between cells: different tops). Returns (scene MjModel, Terrain with the
    material rows, {box geom id: element index})."""
    import mujoco
    from nablaengine import terrain as T
    s = mujoco.MjModel.from_binary_path(str(REPO / SCENE))
    boxes = [g for g in range(s.ngeom) if int(s.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX)]
    hfs = [g for g in range(s.ngeom) if int(s.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_HFIELD)]
    assert len(boxes) == 8 and all(int(s.geom_bodyid[g]) == 0 for g in boxes + hfs)
    cells = [(xs_l[i], xs_l[i + 1], 0.0, 1.2) for i in range(4)] + [(xs_r[i], xs_r[i + 1], -1.2, 0.0) for i in range(4)]
    for g, (x0, x1, y0, y1) in zip(boxes, cells):
        h = np.asarray([(x1 - x0) / 2, (y1 - y0) / 2, 0.05])
        top = 0.0 if tops is None else float(tops[boxes.index(g)])
        s.geom_pos[g] = [(x0 + x1) / 2, (y0 + y1) / 2, top - 0.05]
        s.geom_quat[g] = [1, 0, 0, 0]
        s.geom_size[g] = h
        s.geom_rbound[g] = np.linalg.norm(h)
        s.geom_aabb[g] = np.r_[0, 0, 0, h]
    for g in hfs:
        s.geom_pos[g] = [40.0, 40.0, -3.0]
    ter = T.Terrain.from_model(s)
    rows = list(rows_l) + list(rows_r)
    mat = [T.terrain_material(1.0, "none", sand=r) for r in rows] + [T.terrain_material(1.0, "none")[:8].tolist()
                                                                   + [0, 0, 0, 0] for _ in hfs]
    ter.material = np.asarray(mat, np.float32)
    emap = {g: i for i, g in enumerate(boxes)}
    for p in range(s.npair):
        s.pair_friction[p, :2] = 1.0
    return s, ter, emap


def mixed_rows(seed=0, k=None):
    rng = np.random.default_rng(seed)
    ks = [0.0, 1.0, 0.5, 1.0, 0.0, 0.25, 1.0, 0.0] if k is None else [float(k)] * 8
    return [(ks[i], rng.uniform(0.005, 0.03), rng.uniform(0.3, 0.8), rng.uniform(0.2, 1.0)) for i in range(8)]


def scene(kind, seed=0, bx=0.04, dz=0.0):
    """kind: mixed (cells' k 0 / 1 / 0.5 / 0.25), k0 / k1 (every cell), foothold (left row: sand A for x < bx, hard
    beyond; right row: sand B), twocell (left row sand A, right row sand B); dz: the foothold's sand cells' tops (the
    left row's sand A cells and the right row's sand B), the hard cells' at 0 (a ledge of dz at the boundary)."""
    xs = [-1.2, -0.5, 0.0, 0.5, 1.2]
    if kind == "flat":
        xf = [-1.2, 1.2, 1.3, 1.4, 1.5]
        r = mixed_rows(seed)
        return grid_scene(xf, xf, [(1.0,) + tuple(r[0][1:])] * 4, [(0.0,) + tuple(r[4][1:])] * 4)
    if kind in ("mixed", "k0", "k1"):
        r = mixed_rows(seed, None if kind == "mixed" else (0.0 if kind == "k0" else 1.0))
        return grid_scene(xs, xs, r[:4], r[4:])
    A, B = (1.0, 0.020, 0.50, 0.40), (1.0, 0.012, 0.70, 0.90)
    H = (0.0, 0.0, 0.0, 0.0)
    xl = [-1.2, -0.6, bx, 0.6, 1.2]
    if kind == "foothold":
        return grid_scene(xl, xs, [A, A, H, H], [B] * 4, tops=[dz, dz, 0, 0, dz, dz, dz, dz])
    if kind == "twocell":
        return grid_scene(xl, xs, [A] * 4, [B] * 4)
    raise ValueError(kind)


def course_terrain(k="half"):
    import mujoco
    from nablaengine import terrain as T
    s = mujoco.MjModel.from_binary_path(str(REPO / SCENE))
    ter = T.Terrain.from_model(s)
    n = len(ter.boxes) + len(ter.hf_head)
    rng = np.random.default_rng(3)
    ter.material = np.asarray([T.terrain_material(1.0, "none", sand=(float(i % 2) if k == "half" else float(k),
                                                                     rng.uniform(0.005, 0.03), 0.5, 0.5))
                               for i in range(n)], np.float32)
    return s, ter


def engine(build, foot_normal=False, contact_grad=False, api="jax", floor_k=None, max_rows=None):
    from nablaengine import fixtures, terrain as T
    from nablaengine.params import PhysicsParams
    m, mt = fixtures.load_model()
    fp = [int(p) for p in mt["foot_pairs"]]
    P = None if build == "plain" else PhysicsParams(("foot_offset",))
    sand = build in ("sand", "sandmat")
    cfg = T.TerrainConfig(geoms="feet", terrain_sand=sand,
                          terrain_material=build in ("foffmat", "sandmat"),
                          **({} if floor_k is None else dict(floor=True, sand_other=(floor_k, 0.01, 0.6, 0.5))))
    kw = dict(model=m, foot_pairs=fp, com_body="torso_link", terrain=cfg)
    if P is not None:
        kw["params"] = P
    if foot_normal:
        kw["foot_normal"] = True
    if contact_grad:
        kw["contact_grad"] = True
    if max_rows:
        kw["max_rows"] = int(max_rows)
    if api == "torch":
        from nablaengine import torch_ffi as ffi
    else:
        from nablaengine import jax_ffi as ffi
    info = ffi.init(**kw)
    assert info["engine"] == "forest"
    return m, fp, P, info, ffi


def com_of(m, P, offs, B):
    com = np.zeros((B, 3), np.float32)
    if P is None:
        return com
    return np.concatenate([com, P.pack(m.nv, foot_offset=np.broadcast_to(offs, (B, 4)))], 1)


def corpus(B, seed):
    from nablaengine import fixtures
    z = fixtures.load_corpus("mf_s0")
    idx = np.random.default_rng(seed).permutation(len(z["qpos"]))[:B]
    return {k: np.asarray(z[k][idx]) for k in ("qpos", "qvel", "ctrl", "qacc_warmstart")}


def cmd_dump(a):
    import jax
    import jax.numpy as jnp
    m, fp, P, info, ffi = engine(a.build, a.foot_normal, a.contact_grad)
    s, ter, _ = scene(a.scene, a.seed)
    if a.build not in ("sand", "sandmat"):
        ter.material = None
    ffi.set_terrain(ter)
    B = a.nworld
    z = corpus(B, a.seed)
    rng = np.random.default_rng(a.seed + 100)
    offs = rng.uniform(0.0, a.smax, (B, 4)).astype(np.float32) if a.offs == "rand" else np.zeros((B, 4), np.float32)
    f = lambda x: jnp.asarray(np.asarray(x, np.float32))
    com = f(com_of(m, P, offs, B))
    step = ffi.make_step(a.iterations, SUB, a.clamp_kv, a.normal, contacts=a.foot_normal)
    fr = jnp.ones((B,), jnp.float32)
    out = step(f(z["qpos"]), f(z["qvel"]), f(z["qacc_warmstart"]), f(z["ctrl"]), fr, com)
    q1, v1 = np.asarray(out[0]), np.asarray(out[1])
    rng = np.random.default_rng(2)
    yq, yv = f(rng.standard_normal(q1.shape)), f(rng.standard_normal(v1.shape))
    yf = f(rng.standard_normal((B, 4)))

    def loss(q, v, u, c):
        o = step(q, v, f(z["qacc_warmstart"]), u, fr, c)
        L = (yq * o[0]).sum() + (yv * o[1]).sum()
        if a.foot_normal and a.contact_grad:
            L = L + (yf * ffi.contact_table(o[3], SUB)["foot_normal"]).sum()
        return L
    g = jax.grad(loss, argnums=(0, 1, 2, 3))(f(z["qpos"]), f(z["qvel"]), f(z["ctrl"]), com)
    dump = dict(q1=q1, v1=v1, g_qpos=np.asarray(g[0]), g_qvel=np.asarray(g[1]), g_ctrl=np.asarray(g[2]),
                g_param=np.asarray(g[3])[:, 3:], offsets=offs)
    if a.foot_normal:
        tab = ffi.contact_table(out[3], SUB)
        cw = int(info["con_words"])
        dump.update(foot_normal=np.asarray(tab["foot_normal"]),
                    con=np.asarray(out[3]).reshape(B, -1, cw)[..., :15])
        if "foot_normal_split" in tab:
            dump["split"] = np.asarray(tab["foot_normal_split"])
    np.savez(a.out, **dump)
    print(json.dumps(dict(out=a.out, build=a.build, scene=a.scene, offs=a.offs, nworld=B,
                          mean_contacts=float((np.asarray(out[3]).reshape(B, -1, int(info["con_words"]))[..., 0] >= 0)
                                              .sum(1).mean()) if a.foot_normal else None)))


def cmd_compare(a):
    x, y = np.load(a.a), np.load(a.b)
    keys = [k for k in x.files if k in y.files and k not in a.skip]
    res = {}
    for k in keys:
        u, v = x[k], y[k]
        same = u.shape == v.shape and np.array_equal(u.view(np.uint32) if u.dtype == np.float32 else u,
                                                     v.view(np.uint32) if v.dtype == np.float32 else v)
        d = float(np.abs(u.astype(np.float64) - v.astype(np.float64)).max()) if u.shape == v.shape and u.size else None
        res[k] = dict(bitwise=bool(same), max_abs_diff=d)
    out_json(dict(a=a.a, b=a.b, keys=res, all_bitwise=all(r["bitwise"] for r in res.values())), a.out)
    sys.exit(0 if all(r["bitwise"] for r in res.values()) else 1)


def k_of(ter):
    """k(element, p) as the engine's foff_k: the element's k (piecewise per element)."""
    sd = ter.sand_array()
    return lambda e, p: (float(sd[e, 0]), np.zeros(3))


def contact_k(s, emap, kfn, geom, pos):
    """Per contact (MuJoCo's geom pairs) k: the box element's k(p), 0 for contacts without a terrain box."""
    out = np.zeros(len(pos))
    for i in range(len(pos)):
        for g in (int(geom[i][0]), int(geom[i][1])):
            if g in emap:
                out[i] = kfn(emap[g], pos[i])[0]
    return out


def c_substep_k(s, d, fg, offs, emap, kfn, rec=None):
    """F.c_substep with each foot contact's distance d + k s(x)."""
    import mujoco
    mujoco.mj_checkPos(s, d)
    mujoco.mj_checkVel(s, d)
    mujoco.mj_kinematics(s, d)
    mujoco.mj_comPos(s, d)
    mujoco.mj_camlight(s, d)
    mujoco.mj_flex(s, d)
    mujoco.mj_tendon(s, d)
    mujoco.mj_makeM(s, d)
    mujoco.mj_factorM(s, d)
    mujoco.mj_collision(s, d)
    side = w = kk = None
    if d.ncon:
        geom = np.asarray([d.contact.geom1[:d.ncon], d.contact.geom2[:d.ncon]]).T
        pos = d.contact.pos[:d.ncon].copy()
        sv, side, w = F.contact_offsets(s, fg, geom, pos, d.xpos, d.xmat, offs)
        kk = contact_k(s, emap, kfn, geom, pos)
        for i in range(d.ncon):
            if side[i] >= 0:
                d.contact.dist[i] += kk[i] * sv[i]
                if not d.contact.dist[i] < 0:
                    d.contact.exclude[i] = 1
    mujoco.mj_makeConstraint(s, d)
    mujoco.mj_island(s, d)
    mujoco.mj_transmission(s, d)
    mujoco.mj_projectConstraint(s, d)
    mujoco.mj_forwardSkip(s, d, mujoco.mjtStage.mjSTAGE_POS, 0)
    mujoco.mj_checkAcc(s, d)
    if rec is not None:
        f6 = np.zeros(6)
        fn = []
        for i in range(d.ncon):
            if d.contact.efc_address[i] < 0:
                fn.append(0.0)
                continue
            mujoco.mj_contactForce(s, d, i, f6)
            fn.append(f6[0])
        rec.append((np.asarray(fn), side, w, kk))
    mujoco.mj_implicit(s, d)


def c_run(s, fg, emap, kfn, q, v, u, warm, offs, steps, rec=None):
    import mujoco
    d = mujoco.MjData(s)
    d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q, v, u, warm
    for _ in range(steps * SUB):
        c_substep_k(s, d, fg, offs, emap, kfn, rec)
    return d.qpos.copy(), d.qvel.copy()


def c_split(s, fg, emap, kfn, q, v, u, warm, offs):
    rec = []
    c_run(s, fg, emap, kfn, q, v, u, warm, offs, 1, rec)
    fn, side, w, kk = rec[-1]
    out = np.zeros((2, 4))
    if side is not None:
        for i in range(len(fn)):
            if side[i] >= 0:
                for part, kp in ((0, kk[i]), (1, 1 - kk[i])):
                    out[part, 2 * side[i]] += fn[i] * kp * (1 - w[i])
                    out[part, 2 * side[i] + 1] += fn[i] * kp * w[i]
    return out


def scene_feet(s, m, fp):
    names = [m.geom(g).name for g in F.foot_geoms(m, fp)]
    return F.foot_geoms(s, None, names)


def mjw_step(s, emap, kfn, m, fp, z, offs, iterations):
    """One control step of MuJoCo Warp on scene s from the states z (qpos, qvel, qacc_warmstart, ctrl), each foot
    contact's includemargin -k s(x) (set between Warp's collision and make_constraint): (qvel', foot-contact events on
    sand (k > 0), on hard elements (k = 0))."""
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import constraint
    B = len(z["qpos"])
    s.opt.iterations = iterations
    fg = scene_feet(s, m, fp)
    mw = mjw.put_model(s)
    dw = mjw.put_data(s, mujoco.MjData(s), nworld=B, nconmax=64, njmax=256)
    orig = constraint.make_constraint
    cnt = [0, 0]

    def patched(mm, dd):
        n = int(dd.nacon.numpy()[0])
        if n:
            geom, pos = dd.contact.geom.numpy()[:n], dd.contact.pos.numpy()[:n]
            wid, im = dd.contact.worldid.numpy()[:n], dd.contact.includemargin.numpy()
            xpos, xmat = dd.xpos.numpy(), dd.xmat.numpy().reshape(B, -1, 9)
            for wi in np.unique(wid):
                sel = np.nonzero(wid == wi)[0]
                sv, side, _ = F.contact_offsets(s, fg, geom[sel], pos[sel], xpos[wi], xmat[wi], offs[wi])
                kk = contact_k(s, emap, kfn, geom[sel], pos[sel])
                cnt[0] += int(((side >= 0) & (kk > 0)).sum())
                cnt[1] += int(((side >= 0) & (kk == 0)).sum())
                im[sel] = im[sel] - kk * sv
            wp.copy(dd.contact.includemargin, wp.array(im.astype(np.float32)))
        return orig(mm, dd)
    constraint.make_constraint = patched
    try:
        for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
            wp.copy(getattr(dw, k), wp.array(np.ascontiguousarray(z[k], np.float32)))
        for _ in range(SUB):
            mjw.step(mw, dw)
    finally:
        constraint.make_constraint = orig
    return dw.qvel.numpy().astype(np.float64), cnt[0], cnt[1]


def cmd_mjw(a):
    import jax.numpy as jnp
    m, fp, P, info, ffi = engine("sand")
    s, ter, emap = scene("mixed", a.seed)
    ffi.set_terrain(ter)
    B = a.nworld
    z = corpus(B, a.seed)
    rng = np.random.default_rng(a.seed + 100)
    offs = rng.uniform(0.0, a.smax, (B, 4)).astype(np.float32)
    f = lambda x: jnp.asarray(np.asarray(x, np.float32))
    r = ffi.forward(f(z["qpos"]), f(z["qvel"]), f(z["qacc_warmstart"]), f(z["ctrl"]), jnp.ones((B,), jnp.float32),
                    f(com_of(m, P, offs, B)), iterations=a.iterations, substeps=SUB)
    ev, en = np.asarray(r[1], np.float64), np.asarray(r[3])[:, 1]
    rv, ns, nh = mjw_step(s, emap, k_of(ter), m, fp, z, offs, a.iterations)
    e = np.linalg.norm(ev - rv, axis=1) / np.maximum(np.linalg.norm(rv, axis=1), 1e-6)
    res = dict(check="sand elements vs MuJoCo Warp (margin -k_e s(x))", nworld=B, smax=a.smax,
               iterations=a.iterations, qvel_rel=stats(e), foot_contacts_sand=ns, foot_contacts_hard=nh,
               engine_contacts_mean=float(en.mean()))
    out_json(res, a.out)


def cmd_ledge(a):
    """The foothold with the sand cells' tops at dz (+-20 mm) against the hard cell at 0: the G1 settled with the left
    foot across the ledge (heel on sand A, toe on hard ground) and the right foot on sand B, offsets U(0, --smax) per
    world; then one control step against MuJoCo Warp (margin -k s(x)), the split forces and the parameters."""
    import jax.numpy as jnp
    import mujoco
    m, fp, P, info, ffi = engine("sand", True, False)
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    mujoco.mj_kinematics(m, d)
    ax = float(d.xpos[F.foot_bodies(m)[0]][0])
    res = dict(smax=a.smax, nworld=a.nworld, cases={})
    for dz in a.dz:
        bx = ax + (0.5 * (F.LINE[0] + F.LINE[1]) if dz >= 0 else a.bx_lower)
        s, ter, emap = scene("foothold", bx=bx, dz=dz)
        ffi.set_terrain(ter)
        B = a.nworld
        offs = np.random.default_rng(11).uniform(0.0, a.smax, (B, 4)).astype(np.float32)
        q = np.tile(m.key_qpos[0], (B, 1)).astype(np.float32)
        q[:, 2] += max(dz, 0.0) + 0.002
        qj, vj, wj = jnp.asarray(q), jnp.zeros((B, m.nv), jnp.float32), jnp.zeros((B, m.nv), jnp.float32)
        u = jnp.asarray(np.tile(m.key_ctrl[0], (B, 1)), jnp.float32)
        com = jnp.asarray(com_of(m, P, offs, B))
        fr = jnp.ones((B,), jnp.float32)
        step = ffi.make_step(a.iterations, SUB, 0, 3, contacts=True)
        for _ in range(int(round(a.seconds / (SUB * m.opt.timestep)))):
            qj, vj, wj, con = step(qj, vj, wj, u, fr, com)
        z = dict(qpos=np.asarray(qj), qvel=np.asarray(vj), qacc_warmstart=np.asarray(wj), ctrl=np.asarray(u))
        r = ffi.forward(*(jnp.asarray(z[k]) for k in ("qpos", "qvel", "qacc_warmstart", "ctrl")), fr, com,
                        iterations=a.iterations, substeps=SUB, contacts=True)
        ev = np.asarray(r[1], np.float64)
        T = {k: np.asarray(x) for k, x in ffi.contact_table(r[4], SUB).items()}
        rv, ns, nh = mjw_step(s, emap, k_of(ter), m, fp, z, offs, a.iterations)
        e = np.linalg.norm(ev - rv, axis=1) / np.maximum(np.linalg.norm(rv, axis=1), 1e-6)
        sd = ter.sand_array()
        Pr, pres = params_ref(T, sd)
        S, FN = T["foot_normal_split"], T["foot_normal"]
        fs, el, kw = T["foot_side"], T["element"].astype(int), T["sand_k"]
        lf = fs == 0
        A, Bp = sd[0, 1:], sd[4, 1:]
        kel = np.where((el >= 0) & (el < len(sd)), sd[np.clip(el, 0, len(sd) - 1), 0], 1.0)
        straddle = np.mean([(set(el[b][lf[b]]) >= {0, 1} or set(el[b][lf[b]]) >= {1, 2}) and
                            (kw[b][lf[b]] > 0).any() and (kw[b][lf[b]] == 0).any() for b in range(B)])
        res["cases"][f"dz={dz:+.3f}"] = dict(
            bx_from_ankle=bx - ax, qvel_rel_vs_mjw=stats(e), foot_contacts_sand=ns, foot_contacts_hard=nh,
            left_foot_straddles_share=float(straddle),
            k_word_equals_element_k=bool(np.all((kw == kel)[fs >= 0])),
            split_sum_abs_err_max=float(np.abs(S.sum(1) - FN).max()), foot_normal_max=float(FN.max()),
            left_heel_sand_N_median=float(np.median(S[:, 0, 0])), left_heel_hard_N_median=float(np.median(S[:, 1, 0])),
            left_toe_sand_N_median=float(np.median(S[:, 0, 1])), left_toe_hard_N_median=float(np.median(S[:, 1, 1])),
            params_ref_max_abs=float(np.abs(T["foot_sand_params"] - Pr).max()),
            presence_ref_max_abs=float(np.abs(T["foot_sand_presence"] - pres).max()),
            left_heel_params_is_A=float(np.mean(np.all(np.abs(T["foot_sand_params"][:, 0, 0] - A) < 1e-6, -1))),
            right_params_is_B=float(np.mean(np.all(np.abs(T["foot_sand_params"][:, 1] - Bp) < 1e-6, (-1, -2)))),
            base_z_median=float(np.median(z["qpos"][:, 2])))
    out_json(res, a.out)


def cmd_force(a):
    import jax
    import jax.numpy as jnp
    m, fp, P, info, ffi = engine("sand", True, True, floor_k=a.floor_k, max_rows=a.max_rows)
    if a.floor_k is None:
        s, ter, emap = scene(a.scene, a.seed)
    else:
        from nablaengine import terrain as T
        s = m
        ter = T.Terrain(boxes=np.asarray([T.box_object((30.0, 30.0, -1.0), np.eye(3), (0.1, 0.1, 0.1))], np.float64))
        emap = {0: 0}
        for p_ in fp:
            s.pair_friction[p_, :2] = 1.0
    ffi.set_terrain(ter)
    B = a.nworld
    z = corpus(B, a.seed + 7)
    rng = np.random.default_rng(a.seed + 1)
    offs = rng.uniform(0.0, a.smax, (B, 4))
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    step = ffi.make_step(a.iterations, SUB, 1, 0, contacts=True)
    warm, u, fr = f32(z["qacc_warmstart"]), f32(z["ctrl"]), jnp.ones((B,), jnp.float32)
    com = f32(com_of(m, P, offs, B))
    ffi.read_diagnostics()
    out = step(f32(z["qpos"]), f32(z["qvel"]), warm, u, fr, com)
    dg = ffi.read_diagnostics()
    tab = ffi.contact_table(out[3], SUB)
    S, FN = np.asarray(tab["foot_normal_split"], np.float64), np.asarray(tab["foot_normal"], np.float64)
    res = dict(nworld=B, scene=a.scene if a.floor_k is None else "floor", floor_k=a.floor_k, smax=a.smax,
               sum_abs_err_max=float(np.abs(S.sum(1) - FN).max()), foot_normal_max=float(np.abs(FN).max()),
               sand_share=float(S[:, 0].sum() / max(S.sum(), 1e-9)), nefc=int(info["nefc"]),
               row_overflow_worlds=dg["fwd_row_overflow_worlds"],
               contact_overflow_worlds=dg["fwd_contact_overflow_worlds"])
    yf = rng.standard_normal((B, 2, 4))
    yq, yv = rng.standard_normal((B, m.nq)), rng.standard_normal((B, m.nv))
    state = a.loss == "state"

    def loss(q, v, uu, c):
        o = step(q, v, warm, uu, fr, c)
        if state:
            return (f32(yq) * o[0]).sum() + (f32(yv) * o[1]).sum()
        return (f32(yf) * ffi.contact_table(o[3], SUB)["foot_normal_split"]).sum()
    g = jax.grad(loss, argnums=(0, 1, 2, 3))(f32(z["qpos"]), f32(z["qvel"]), u, com)
    gs = [np.asarray(x, np.float64) for x in g]
    gs[3] = gs[3][:, 3:]
    s.opt.iterations, s.opt.tolerance, s.opt.ls_tolerance = a.c_iterations, 1e-12, 1e-12
    fg, kfn = scene_feet(s, m, fp), k_of(ter)
    if a.floor_k is not None:
        kfn = lambda e, p_: (a.floor_k, np.zeros(3))
    rows = {"qpos": [], "qvel": [], "ctrl": [], "offsets": []}
    fwd = []
    for i in range(B):
        base = [np.float64(np.float32(np.asarray(z[k][i], np.float64))) for k in ("qpos", "qvel", "ctrl",
                                                                                   "qacc_warmstart")]
        o = np.float64(np.float32(offs[i]))
        Sc = c_split(s, fg, emap, kfn, *base, o)
        fwd.append(np.abs(Sc - S[i]).max() / max(np.abs(Sc).max(), 1.0))
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
            if state:
                Lc = lambda x: (yq[i] @ (r := c_run(s, fg, emap, kfn, *x[:4], x[4], 1))[0] + yv[i] @ r[1])
            else:
                Lc = lambda x: (yf[i] * c_split(s, fg, emap, kfn, *x)).sum()
            fd = (Lc(xp) - Lc(xm)) / (2 * a.eps)
            gg = an @ dv
            rows[name].append(abs(fd - gg) / max(abs(fd), abs(gg), 1e-9))
    fwd = np.asarray(fwd)
    agree = fwd < a.agree
    res.update(forward_rel_median=float(np.median(fwd)), forward_agree_share=float(agree.mean()))
    for k, v in rows.items():
        res[k] = stats(np.asarray(v)[agree])
    if a.accept is not None:
        res["accept"] = a.accept
        res["accepted"] = all(r is None or r["median"] <= a.accept for r in (res[k] for k in rows))
    out_json(res, a.out)
    if a.accept is not None and not res["accepted"]:
        sys.exit(1)


def stand_states(m, fp, ffi, P, offs, seconds):
    """The G1 from its keyframe for `seconds` (the model's servos; short: it stands for about a second) on the terrain;
    returns the final (qpos, qvel, warm) and the step function."""
    import jax.numpy as jnp
    B = len(offs)
    q = jnp.asarray(np.tile(m.key_qpos[0], (B, 1)), jnp.float32)
    v = w = jnp.zeros((B, m.nv), jnp.float32)
    u = jnp.asarray(np.tile(m.key_ctrl[0], (B, 1)), jnp.float32)
    com = jnp.asarray(com_of(m, P, offs, B))
    step = ffi.make_step(m.opt.iterations, SUB, 0, 3, contacts=True)
    fr = jnp.ones((B,), jnp.float32)
    for _ in range(int(round(seconds / (SUB * m.opt.timestep)))):
        q, v, w, con = step(q, v, w, u, fr, com)
    return q, v, w, u, fr, com, step, con


def params_ref(tab, sand, other=(1.0, 0.0, 0.0, 0.0), eps=1e-3):
    """foot_sand_params, foot_sand_presence from the table (NumPy, float64), per world."""
    side, w, fn, k, e = (np.asarray(tab[x], np.float64) for x in ("foot_side", "toe_weight", "force_normal", "sand_k",
                                                                  "element"))
    tabp = np.concatenate([np.asarray(sand, np.float64), [other]], 0)
    n = len(sand)
    B = len(fn)
    P = np.zeros((B, 2, 2, 3))
    pres = np.zeros((B, 4))
    for b in range(B):
        for f in (0, 1):
            for h, wt in ((0, 1 - w[b]), (1, w[b])):
                on = (side[b] == f) * wt
                wf, wk = fn[b] * k[b] * on, k[b] * on
                pres[b, 2 * f + h] = wk.sum()
                ww = wf if wf.sum() >= eps else wk
                if ww.sum() > 0:
                    idx = np.where((e[b] >= 0) & (e[b] < n), e[b], n).astype(int)
                    P[b, f, h] = (ww[:, None] * tabp[idx, 1:]).sum(0) / ww.sum()
    return P, pres


def cmd_params(a):
    import mujoco
    m, fp, P, info, ffi = engine("sand", True, False)
    res = {}
    for kind in ("foothold", "twocell"):
        d = mujoco.MjData(m)
        d.qpos[:] = m.key_qpos[0]
        mujoco.mj_kinematics(m, d)
        ax = float(d.xpos[F.foot_bodies(m)[0]][0])
        bx = ax + 0.5 * (F.LINE[0] + F.LINE[1])
        s, ter, emap = scene(kind, bx=bx, dz=a.sh)
        ffi.set_terrain(ter)
        offs = np.tile([a.sh, a.sh, a.sh, a.sh], (1, 1)).astype(np.float32)
        q, v, w, u, fr, com, step, con = stand_states(m, fp, ffi, P, offs, a.seconds)
        tab = ffi.contact_table(con, SUB)
        T = {k: np.asarray(x) for k, x in tab.items()}
        Pr, pres = params_ref(T, ter.sand_array())
        Pe, pe = T["foot_sand_params"], T["foot_sand_presence"]
        split = T["foot_normal_split"][0]
        A, Bp = ter.sand_array()[0, 1:], ter.sand_array()[4, 1:]
        r = dict(bx=bx, params_engine=Pe[0].tolist(), presence=pe[0].tolist(), split_sand_hard=split.tolist(),
                 foot_normal=T["foot_normal"][0].tolist(), params_ref_max_abs=float(np.abs(Pe - Pr).max()),
                 presence_ref_max_abs=float(np.abs(pe - pres).max()), cell_A=A.tolist(), cell_B=Bp.tolist(),
                 left_elements=sorted(set(T["element"][0][T["foot_side"][0] == 0].astype(int).tolist())),
                 right_elements=sorted(set(T["element"][0][T["foot_side"][0] == 1].astype(int).tolist())),
                 base_z=float(np.asarray(q)[0, 2]))
        if kind == "foothold":
            r["checks"] = dict(left_heel_is_A=bool(np.allclose(Pe[0, 0, 0], A, atol=1e-6)),
                               right_is_B=bool(np.allclose(Pe[0, 1], Bp, atol=1e-6)),
                               left_toe_hard_force_gt_sand=bool(split[1, 1] > split[0, 1]),
                               left_heel_sand_force_gt_hard=bool(split[0, 0] > split[1, 0]))
        else:
            r["checks"] = dict(left_is_A=bool(np.allclose(Pe[0, 0], A, atol=1e-6)),
                               right_is_B=bool(np.allclose(Pe[0, 1], Bp, atol=1e-6)),
                               hard_part_zero=bool(np.abs(split[1]).max() == 0.0))
        res[kind] = r
    res["accepted"] = all(all(r["checks"].values()) and r["params_ref_max_abs"] < 1e-5 for r in res.values()
                          if isinstance(r, dict))
    out_json(res, a.out)


def cmd_boundary(a):
    import jax
    import jax.numpy as jnp
    import mujoco
    m, fp, P, info, ffi = engine("sand", False, False, max_rows=a.max_rows)
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    mujoco.mj_kinematics(m, d)
    ax = float(d.xpos[F.foot_bodies(m)[0]][0])
    bx = ax + 0.5 * (F.LINE[0] + F.LINE[1])
    s, ter, emap = scene(a.scene, bx=bx)
    sh = a.sh
    import copy
    hard = copy.copy(ter)
    hard.material = None
    ffi.set_terrain(hard)
    off1 = np.full((1, 4), sh, np.float32)
    q, v, w, u, fr, com, step, _ = stand_states(m, fp, ffi, P, off1, a.settle)
    ffi.set_terrain(ter)
    deltas = np.arange(-a.span, a.span + 1e-12, a.step)
    B = len(deltas)
    q0 = np.repeat(np.asarray(q, np.float64), B, 0)
    q0[:, 0] += deltas
    v0, w0 = np.repeat(np.asarray(v), B, 0), np.repeat(np.asarray(w), B, 0)
    u0 = np.repeat(np.asarray(u), B, 0)
    comB = jnp.asarray(com_of(m, P, np.repeat(off1, B, 0), B))
    frB = jnp.ones((B,), jnp.float32)
    rng = np.random.default_rng(5)
    yq, yv = rng.standard_normal(m.nq), rng.standard_normal(m.nv)
    yq[3:7] = 0.0
    stepx = ffi.make_step(a.iterations, SUB, 1, 0)

    def loss(qq):
        x, vv, ww = qq, jnp.asarray(v0, jnp.float32), jnp.asarray(w0, jnp.float32)
        for _ in range(a.steps):
            x, vv, ww = stepx(x, vv, ww, jnp.asarray(u0, jnp.float32), frB, comB)[:3]
        return (jnp.asarray(yq, jnp.float32) * x).sum(1) + (jnp.asarray(yv, jnp.float32) * vv).sum(1)
    ffi.read_diagnostics()
    Ls = np.asarray(loss(jnp.asarray(q0, jnp.float32)), np.float64)
    dg = ffi.read_diagnostics()
    g = np.asarray(jax.grad(lambda qq: loss(qq).sum())(jnp.asarray(q0, jnp.float32)), np.float64)[:, 0]
    hw = max(1, int(round(a.window / a.step)))
    fdw = np.full(B, np.nan)
    fdw[hw:B - hw] = (Ls[2 * hw:] - Ls[:B - 2 * hw]) / (2 * hw * a.step)
    s.opt.iterations, s.opt.tolerance, s.opt.ls_tolerance = a.c_iterations, 1e-12, 1e-12
    fg, kfn = scene_feet(s, m, fp), k_of(ter)
    oo = np.float64(off1[0])
    Lc0 = np.zeros(B)
    fdc = np.zeros(B)
    sel = list(range(0, B, max(1, a.c_stride)))
    for i in sel:
        base = [np.float64(np.float32(x)) for x in (q0[i], v0[i], u0[i], w0[i])]

        def Lc(dx):
            qq = base[0].copy()
            qq[0] += dx
            qe, ve = c_run(s, fg, emap, kfn, qq, base[1], base[2], base[3], oo, a.steps)
            return yq @ qe + yv @ ve
        Lc0[i] = Lc(0.0)
        fdc[i] = (Lc(1e-6) - Lc(-1e-6)) / 2e-6
    jumps = np.abs(np.diff(Ls))
    rel = np.abs(g[sel] - fdc[sel]) / np.maximum(np.maximum(np.abs(fdc[sel]), np.abs(g[sel])), 1e-9)
    ok = ~np.isnan(fdw)
    relw = np.abs(g[ok] - fdw[ok]) / np.maximum(np.maximum(np.abs(fdw[ok]), np.abs(g[ok])), 1e-9)
    res = dict(scene=a.scene, sh=sh, steps=a.steps, span=a.span, step=a.step, window=a.window, bx=bx,
               nefc=int(info["nefc"]), row_overflow_worlds=dg["fwd_row_overflow_worlds"],
               contact_overflow_worlds=dg["fwd_contact_overflow_worlds"],
               forward_vs_c=stats(np.abs(Ls[sel] - Lc0[sel]) / np.maximum(np.abs(Lc0[sel]), 1e-9)),
               L_range=float(Ls.max() - Ls.min()), L_step_abs=stats(jumps), L_step_abs_max=float(jumps.max()),
               L_step_p99_over_median=float(np.quantile(jumps, 0.99) / max(np.median(jumps), 1e-12)),
               grad_abs=stats(np.abs(g)), vjp_vs_c_fd=stats(rel), vjp_vs_window_fd=stats(relw),
               window_fd_abs_max=float(np.nanmax(np.abs(fdw))), vjp_abs_max=float(np.abs(g).max()),
               curve=dict(delta=deltas.tolist(), L=Ls.tolist(), vjp=g.tolist(), fd_window=np.nan_to_num(fdw).tolist(),
                          fd_c=[float(fdc[i]) for i in sel], c_idx=sel))
    out_json(res, a.out)


def cmd_time(a):
    import time
    import jax
    import jax.numpy as jnp
    m, fp, P, info, ffi = engine(a.build)
    s, ter = course_terrain(a.sand_k)
    if a.build != "sand":
        ter.material = None
    ffi.set_terrain(ter)
    r = dict(np.load(REPO / COURSE_STATES))
    idx = np.arange(a.nworld) % len(r["qpos"])
    f = lambda k: jnp.asarray(np.asarray(r[k], np.float32)[idx])
    offs = np.random.default_rng(0).uniform(0.0, a.smax, (a.nworld, 4))
    com = (jnp.asarray(com_of(m, P, offs, a.nworld)) if P is not None
           else jnp.zeros((a.nworld, info["com_words"]), jnp.float32))
    it = 10 if a.iterations is None else a.iterations
    step = ffi.make_step(it, SUB)
    q, v, w, u = f("qpos"), f("qvel"), f("warm"), f("ctrl")
    fr = jnp.ones((a.nworld,), jnp.float32)
    fwd = jax.jit(lambda q, v, w, u, fr, c: step(q, v, w, u, fr, c)[1])
    both = jax.jit(jax.grad(lambda q, v, w, u, fr, c: step(q, v, w, u, fr, c)[1].sum(), argnums=(0, 1, 3)))
    res = dict(build=a.build, sand_k=a.sand_k if a.build == "sand" else None, nworld=a.nworld, iterations=it, smax=a.smax,
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
    p = sp.add_parser("dump")
    p.add_argument("--build", choices=("sand", "sandmat", "foff", "foffmat", "plain"), default="sand")
    p.add_argument("--scene", default="mixed")
    p.add_argument("--offs", choices=("rand", "zero"), default="rand")
    p.add_argument("--foot-normal", action="store_true")
    p.add_argument("--contact-grad", action="store_true")
    p.add_argument("--nworld", type=int, default=512)
    p.add_argument("--smax", type=float, default=0.02)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--clamp-kv", type=int, default=1)
    p.add_argument("--normal", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sp.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--skip", nargs="*", default=())
    p.add_argument("--out")
    p = sp.add_parser("mjw")
    p.add_argument("--nworld", type=int, default=512)
    p.add_argument("--smax", type=float, default=0.02)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p = sp.add_parser("ledge")
    p.add_argument("--dz", type=float, nargs="*", default=[0.02, -0.02])
    p.add_argument("--bx-lower", type=float, default=0.11)
    p.add_argument("--nworld", type=int, default=64)
    p.add_argument("--smax", type=float, default=0.02)
    p.add_argument("--seconds", type=float, default=0.4)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--out")
    p = sp.add_parser("force")
    p.add_argument("--scene", default="mixed")
    p.add_argument("--loss", choices=("split", "state"), default="split")
    p.add_argument("--floor-k", type=float)
    p.add_argument("--nworld", type=int, default=48)
    p.add_argument("--smax", type=float, default=0.02)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--c-iterations", type=int, default=200)
    p.add_argument("--agree", type=float, default=1e-3)
    p.add_argument("--max-rows", type=int, help="the build's constraint-row capacity (init max_rows; default: the "
                   "engine's)")
    p.add_argument("--accept", type=float, help="exit 1 when a block's median relative error exceeds this")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out")
    p = sp.add_parser("params")
    p.add_argument("--sh", type=float, default=0.01)
    p.add_argument("--seconds", type=float, default=0.5)
    p.add_argument("--out")
    p = sp.add_parser("boundary")
    p.add_argument("--scene", default="foothold")
    p.add_argument("--sh", type=float, default=0.02)
    p.add_argument("--settle", type=float, default=0.3)
    p.add_argument("--span", type=float, default=0.06)
    p.add_argument("--step", type=float, default=0.0005)
    p.add_argument("--window", type=float, default=0.002)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--c-iterations", type=int, default=200)
    p.add_argument("--c-stride", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=400,
                   help="the build's constraint-row capacity: a foot across a seam has contacts with both boxes, more "
                        "rows than the default 192 (rows beyond it are dropped; reported as row_overflow_worlds)")
    p.add_argument("--out")
    p = sp.add_parser("time")
    p.add_argument("--build", choices=("sand", "foff", "plain"), default="sand")
    p.add_argument("--sand-k", default="half")
    p.add_argument("--smax", type=float, default=0.02)
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--iterations", type=int)
    p.add_argument("--reps", type=int, default=60)
    a = ap.parse_args()
    {"dump": cmd_dump, "compare": cmd_compare, "mjw": cmd_mjw, "ledge": cmd_ledge, "force": cmd_force, "params": cmd_params,
     "boundary": cmd_boundary, "time": cmd_time}[a.cmd](a)


if __name__ == "__main__":
    main()

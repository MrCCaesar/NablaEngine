"""Gradients of models with connect equalities or fixed tendons in the forest engine (the connect adjoint,
forest_engine.equality; the tendon rows and forces, forest_engine.tendon), against float64 MuJoCo C (the arbiter):"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import connect_check as cc

J = mujoco.mjtJoint


def free_quats(m):
    """The qpos indices of the free joints' quaternions."""
    return [int(m.jnt_qposadr[j]) + 3 + k for j in range(m.njnt) if int(m.jnt_type[j]) == J.mjJNT_FREE
            for k in range(4)]


def q_to_tangent(m, q, g):
    """The qpos cotangent g (nq) as a dof-tangent gradient (nv): hinge / slide / translations as they are, a free
    joint's rotation G_k = g_quat . (quat (0, e_k)) / 2."""
    G = np.zeros(m.nv)
    for j in range(m.njnt):
        qa, da, t = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), int(m.jnt_type[j])
        if t == J.mjJNT_FREE:
            G[da:da + 3] = g[qa:qa + 3]
            qt = q[qa + 3:qa + 7]
            for k in range(3):
                e = np.zeros(4); e[1 + k] = 1.0
                r = np.zeros(4)
                mujoco.mju_mulQuat(r, qt, e)
                G[da + 3 + k] = 0.5 * float(g[qa + 3:qa + 7] @ r)
        elif t in (J.mjJNT_HINGE, J.mjJNT_SLIDE):
            G[da] = g[qa]
        else:
            raise ValueError("ball joints are not handled")
    return G


def gq_tangent_out(m, gq):
    """L's qpos cotangent (zero on quaternion entries) as an output tangent cotangent (nv)."""
    G = np.zeros(m.nv)
    for j in range(m.njnt):
        qa, da, t = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), int(m.jnt_type[j])
        if t == J.mjJNT_FREE:
            G[da:da + 3] = gq[qa:qa + 3]
        else:
            G[da] = gq[qa]
    return G


def tight(m):
    mm = mujoco.MjModel.from_binary_path(m) if isinstance(m, str) else m.__copy__()
    mm.opt.tolerance = 1e-12
    mm.opt.ls_tolerance = 1e-12
    mm.opt.iterations = max(int(mm.opt.iterations), 200)
    mm.opt.ls_iterations = 100
    return mm


def rollout(mm, d, q, v, u, S):
    mujoco.mj_resetData(mm, d)
    d.qpos[:], d.qvel[:], d.ctrl[:] = q, v, u
    nc = 0
    for _ in range(S):
        mujoco.mj_step(mm, d)
        nc = max(nc, d.ncon)
    return d.qpos.copy(), d.qvel.copy(), nc


def fd_grad(mm, q, v, u, gq, gv, S, eps=1e-6):
    d = mujoco.MjData(mm)

    def L(qq, vv, uu):
        qn, vn, _ = rollout(mm, d, qq, vv, uu, S)
        return float(gq @ qn + gv @ vn)

    out = []
    for i in range(mm.nv):
        dv = np.zeros(mm.nv); dv[i] = 1.0
        qp, qm = q.copy(), q.copy()
        mujoco.mj_integratePos(mm, qp, dv, eps)
        mujoco.mj_integratePos(mm, qm, dv, -eps)
        out.append((L(qp, v, u) - L(qm, v, u)) / (2 * eps))
    for i in range(mm.nv):
        e = np.zeros(mm.nv); e[i] = eps
        out.append((L(q, v + e, u) - L(q, v - e, u)) / (2 * eps))
    for i in range(mm.nu):
        e = np.zeros(mm.nu); e[i] = eps
        out.append((L(q, v, u + e) - L(q, v, u - e)) / (2 * eps))
    return np.array(out)


def tfd_grad(mm, q, v, u, gq, gv, S, eps=1e-6):
    """[G_q, G_v, G_u] from the product of mjd_transitionFD's Jacobians along the trajectory."""
    d = mujoco.MjData(mm)
    mujoco.mj_resetData(mm, d)
    d.qpos[:], d.qvel[:], d.ctrl[:] = q, v, u
    nx = 2 * mm.nv + mm.na
    As, Bs = [], []
    for _ in range(S):
        A, B = np.zeros((nx, nx)), np.zeros((nx, mm.nu))
        mujoco.mjd_transitionFD(mm, d, eps, True, A, B, None, None)
        As.append(A); Bs.append(B)
        mujoco.mj_step(mm, d)
    lam = np.concatenate([gq_tangent_out(mm, gq), gv, np.zeros(mm.na)])
    gu = np.zeros(mm.nu)
    for A, B in zip(As[::-1], Bs[::-1]):
        gu += lam @ B
        lam = lam @ A
    return np.concatenate([lam[:2 * mm.nv], gu])


def engine_vjp(m, info, q, v, u, gq, gv, S, **step_kw):
    """The engine's VJP of (qpos', qvel') after S substeps; step_kw: jax_ffi.step's adjoint options (clamp_kv, normal)."""
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    B = len(q)
    cw = int(info.get("com_words", 3))
    w, fr, com = jnp.zeros((B, m.nv), jnp.float32), jnp.ones((B,), jnp.float32), jnp.zeros((B, cw), jnp.float32)

    def f(qq, vv, uu):
        o = jax_ffi.step(qq, vv, w, uu, fr, com, substeps=S, iterations=int(m.opt.iterations), **step_kw)
        return o[0], o[1]

    (qe, ve), pull = jax.vjp(f, jnp.asarray(q), jnp.asarray(v), jnp.asarray(u))
    g = pull((jnp.asarray(gq, jnp.float32), jnp.asarray(gv, jnp.float32)))
    return np.asarray(qe), np.asarray(ve), [np.asarray(x, np.float64) for x in g]




def tape_bitwise(m, info, q, v, u, S):
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    B = len(q)
    cw = int(info.get("com_words", 3))
    args = (jnp.asarray(q), jnp.asarray(v), jnp.zeros((B, m.nv), jnp.float32), jnp.asarray(u),
            jnp.ones((B,), jnp.float32), jnp.zeros((B, cw), jnp.float32))
    a = jax_ffi.step(*args, substeps=S, iterations=int(m.opt.iterations))
    o = jax_ffi.forward(*args, substeps=S, tape=True, iterations=int(m.opt.iterations))
    return bool(np.array_equal(np.asarray(o[0]), np.asarray(a[0])) and np.array_equal(np.asarray(o[1]),
                                                                                     np.asarray(a[1])))


def err(g, ref, sl=slice(None)):
    return float(np.abs(g[sl] - ref[sl]).max() / max(np.abs(ref[sl]).max(), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--states", type=int, default=32)
    ap.add_argument("--substeps", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--tendon-to-joint", default=None)
    ap.add_argument("--no-fluid", action="store_true")
    ap.add_argument("--uniform-limits", action="store_true",
                    help="every joint limit on the first set (connect_check.py; earlier runs only: the engine reads per-joint sets)")
    ap.add_argument("--drop-spatial", action="store_true", help="connect_check.py's option")
    ap.add_argument("--no-tfd", action="store_true", help="skip the mjd_transitionFD product")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cv = cc.load(a.model, a.tendon_to_joint, a.no_fluid, a.uniform_limits, a.drop_spatial)
    m, mo = cv.model, cv.original
    assert m.nq == mo.nq and m.nv == mo.nv and m.nu == mo.nu
    for x in (mo.geom_margin, mo.geom_gap, mo.pair_margin, mo.pair_gap, mo.jnt_margin):
        x[:] = 0
    from nablaengine.forest_engine import equality, tendon
    rep = dict(model=str(Path(a.model).resolve().relative_to(Path(__file__).resolve().parents[2]))
               if str(Path(a.model).resolve()).startswith(str(Path(__file__).resolve().parents[2])) else Path(a.model).name,
               nv=int(m.nv), connects=len(equality.connects(m)), joint_eqs=len(equality.rows(m)),
               tendon_eqs=len(tendon.tendon_eqs(m)), limited_tendons=len(tendon.limited(m)),
               passive_tendons=len(tendon.passive(m)), armature_tendons=len(tendon.armature(m)),
               states=a.states, mujoco=mujoco.__version__, seed=a.seed, by_substeps={})
    q, v, u = cc.corpus(mo, a.states, a.seed)
    info = cc.engine_init(m)
    mm = tight(mo)
    rng = np.random.default_rng(a.seed + 7)
    nv, nu = m.nv, m.nu
    blocks = dict(q=slice(0, nv), v=slice(nv, 2 * nv), u=slice(2 * nv, 2 * nv + nu))
    for S in a.substeps:
        gq = rng.normal(size=(len(q), m.nq))
        gq[:, free_quats(m)] = 0.0
        gv = rng.normal(size=(len(q), nv))
        _, _, (bq, bv, bu) = engine_vjp(m, info, q, v, u, gq, gv, S)
        res = dict(fd=[], tfd=[], contacts=[], fd_blocks={k: [] for k in blocks}, fd_vs_tfd=[])
        dd = mujoco.MjData(mm)
        for w in range(len(q)):
            q64, v64, u64 = (x[w].astype(np.float64) for x in (q, v, u))
            ge = np.concatenate([q_to_tangent(m, q64, bq[w]), bv[w], bu[w]])
            gf = fd_grad(mm, q64, v64, u64, gq[w], gv[w], S)
            res["fd"].append(err(ge, gf))
            for k, sl in blocks.items():
                if sl.stop > sl.start:
                    res["fd_blocks"][k].append(err(ge, gf, sl))
            res["contacts"].append(rollout(mm, dd, q64, v64, u64, S)[2])
            if not a.no_tfd:
                gt = tfd_grad(mm, q64, v64, u64, gq[w], gv[w], S)
                res["tfd"].append(err(ge, gt))
                res["fd_vs_tfd"].append(err(gt, gf))
        nc = np.array(res["contacts"]) == 0
        out = dict(engine_vs_fd=cc.stats(res["fd"]), engine_vs_fd_no_contact=cc.stats(np.array(res["fd"])[nc]),
                   engine_vs_fd_blocks={k: cc.stats(x) for k, x in res["fd_blocks"].items()},
                   states_with_contacts=int((~nc).sum()), tape_bitwise=tape_bitwise(m, info, q, v, u, S))
        if not a.no_tfd:
            out.update(engine_vs_tfd=cc.stats(res["tfd"]), engine_vs_tfd_no_contact=cc.stats(np.array(res["tfd"])[nc]),
                       fd_vs_tfd=cc.stats(res["fd_vs_tfd"]))
        rep["by_substeps"][str(S)] = out
    txt = json.dumps(rep, indent=1)
    print(txt)
    if a.out:
        o = Path(a.out)
        if o.exists():
            sys.exit(f"{o} exists (not overwritten)")
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(txt)


if __name__ == "__main__":
    main()

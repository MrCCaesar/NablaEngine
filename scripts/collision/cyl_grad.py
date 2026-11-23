"""Gradients of a model with cylinder contacts in the forest engine (forest_engine.cylinder) against float64 MuJoCo C
central differences, on the columns where those are a derivative:"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eq"))
import connect_check as cc
import connect_grad as cg


def rollout(mm, d, q, v, u, S):
    """(qpos_S, qvel_S, per-substep contact counts per geom pair)."""
    mujoco.mj_resetData(mm, d)
    d.qpos[:], d.qvel[:], d.ctrl[:] = q, v, u
    sets = []
    for _ in range(S):
        mujoco.mj_step(mm, d)
        sets.append(Counter(tuple(sorted(int(g) for g in c.geom)) for c in d.contact[:d.ncon]))
    return d.qpos.copy(), d.qvel.copy(), sets


def fd_columns(mm, q, v, u, gq, gv, S):
    """(central differences at eps 1e-6, validity) per column (qpos tangent nv, qvel nv, ctrl nu)."""
    d = mujoco.MjData(mm)
    _, _, base = rollout(mm, d, q, v, u, S)

    def L(qq, vv, uu):
        qn, vn, sets = rollout(mm, d, qq, vv, uu, S)
        return float(gq @ qn + gv @ vn), sets == base

    def col(kind, i, eps):
        if kind == 0:
            dv = np.zeros(mm.nv); dv[i] = 1.0
            qp, qm = q.copy(), q.copy()
            mujoco.mj_integratePos(mm, qp, dv, eps)
            mujoco.mj_integratePos(mm, qm, dv, -eps)
            a, sa = L(qp, v, u); b, sb = L(qm, v, u)
        elif kind == 1:
            e = np.zeros(mm.nv); e[i] = eps
            a, sa = L(q, v + e, u); b, sb = L(q, v - e, u)
        else:
            e = np.zeros(mm.nu); e[i] = eps
            a, sa = L(q, v, u + e); b, sb = L(q, v, u - e)
        return (a - b) / (2 * eps), sa and sb

    g, ok = [], []
    for kind, n in ((0, mm.nv), (1, mm.nv), (2, mm.nu)):
        for i in range(n):
            a, s1 = col(kind, i, 1e-6)
            b, s2 = col(kind, i, 1e-7)
            g.append(a)
            ok.append(s1 and s2 and abs(a - b) <= 1e-4 * abs(a) + 1e-6)
    return np.array(g), np.array(ok)


def selffd(m, info, q, v, u, S, eps, seed):
    """The VJP against the engine's own float32 central differences along a random direction per world (qpos without
    the free joints' quaternions, qvel, ctrl), steps eps and 2 eps: rel = |fd - vjp| / max(|fd|, |vjp|) per world over
    the worlds whose contact count is the same at x +- eps, and the two steps' fd against each other (the noise)."""
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    B = len(q)
    cw = int(info.get("com_words", 3))
    z, fr, com = (jnp.zeros((B, m.nv), jnp.float32), jnp.ones((B,), jnp.float32), jnp.zeros((B, cw), jnp.float32))
    rng = np.random.default_rng(seed)
    yq = jnp.asarray(rng.standard_normal((B, m.nq)), jnp.float32)
    yv = jnp.asarray(rng.standard_normal((B, m.nv)), jnp.float32)
    x0 = [jnp.asarray(q), jnp.asarray(v), jnp.asarray(u)]
    it = int(m.opt.iterations)

    def out(qq, vv, uu):
        o = jax_ffi.forward(qq, vv, z, uu, fr, com, iterations=it, substeps=S)
        return np.asarray(jnp.sum(yq * o[0], 1) + jnp.sum(yv * o[1], 1), np.float64), np.asarray(o[3][:, 1])

    def loss(qq, vv, uu):
        o = jax_ffi.step(qq, vv, z, uu, fr, com, iterations=it, substeps=S)
        return jnp.sum(yq * o[0]) + jnp.sum(yv * o[1])

    grads = jax.grad(loss, argnums=(0, 1, 2))(*x0)
    res = {}
    for k, name in enumerate(("qpos", "qvel", "ctrl")):
        if x0[k].shape[1] == 0:
            continue
        d = rng.standard_normal(np.asarray(x0[k]).shape)
        if name == "qpos":
            d[:, cg.free_quats(m)] = 0.0
        d = jnp.asarray(d / np.linalg.norm(d, axis=1, keepdims=True), jnp.float32)
        fds, same = [], np.ones(B, bool)
        for h in (eps, 2 * eps):
            xp, xm = list(x0), list(x0)
            xp[k], xm[k] = x0[k] + h * d, x0[k] - h * d
            (lp, cp), (lm, cm) = out(*xp), out(*xm)
            fds.append((lp - lm) / (2 * h))
            same &= cp == cm
        an = np.asarray(jnp.sum(grads[k] * d, 1), np.float64)
        rel = lambda a_, b_: np.abs(a_ - b_) / np.maximum(np.maximum(np.abs(a_), np.abs(b_)), 1e-12)
        res[name] = dict(worlds=int(same.sum()), vjp_vs_fd=cc.stats(rel(an, fds[0])[same]),
                         fd_noise=cc.stats(rel(fds[0], fds[1])[same]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--states", type=int, default=32)
    ap.add_argument("--substeps", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--selffd-states", type=int, default=512, help="worlds of the self-FD check (0: skip)")
    ap.add_argument("--selffd-eps", type=float, default=5e-4)
    ap.add_argument("--reference", choices=("converted", "source"), default="converted",
                    help="the model MuJoCo C runs (cyl_check.py's option)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-contacts", type=int, default=0, help="contact capacity per world (0: the engine's default)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cv = cc.load(a.model)
    m, mo = cv.model, cv.original
    for x in (mo.geom_margin, mo.geom_gap, mo.pair_margin, mo.pair_gap, mo.jnt_margin):
        x[:] = 0
    if a.reference == "converted":
        mo = m
    root = Path(__file__).resolve().parents[2]
    pm = Path(a.model).resolve()
    rep = dict(model=os.path.relpath(pm, root) if str(pm).startswith(str(root)) else "/".join(pm.parts[-2:]),
               nv=int(m.nv), reference=a.reference, states=a.states, seed=a.seed, mujoco=mujoco.__version__, by_substeps={})
    q, v, u = cc.corpus(mo, max(a.states, 1), a.seed)
    if a.max_contacts:
        from nablaengine import engines, jax_ffi
        engines.prefer_forest(m)
        info = jax_ffi.init(model=m, foot_pairs=[], max_contacts=a.max_contacts)
    else:
        info = cc.engine_init(m)
    mm = cg.tight(mo)
    rng = np.random.default_rng(a.seed + 7)
    for S in (a.substeps if a.states else ()):
        gq = rng.normal(size=(len(q), m.nq))
        gq[:, cg.free_quats(m)] = 0.0
        gv = rng.normal(size=(len(q), m.nv))
        qe, ve, (bq, bv, bu) = cg.engine_vjp(m, info, q, v, u, gq, gv, S)
        dd = mujoco.MjData(mm)
        fwd = []
        agree = []
        err, err_all_valid, frac, err_all = [], [], [], []
        for w in range(len(q)):
            q64, v64, u64 = (x[w].astype(np.float64) for x in (q, v, u))
            ge = np.concatenate([cg.q_to_tangent(m, q64, bq[w]), bv[w], bu[w]])
            gf, ok = fd_columns(mm, q64, v64, u64, gq[w], gv[w], S)
            frac.append(float(ok.mean()))
            err_all.append(cg.err(ge, gf))
            vc = rollout(mm, dd, q64, v64, u64, S)[1]
            fwd.append(float(np.max(np.abs(ve[w] - vc) / np.maximum(np.abs(vc), 1.0))))
            if ok.any():
                e = float(np.abs(ge[ok] - gf[ok]).max() / max(np.abs(gf[ok]).max(), 1e-12))
                err.append(e)
                if ok.all():
                    err_all_valid.append(e)
                if fwd[-1] < 1e-4:
                    agree.append(e)
        rep["by_substeps"][str(S)] = dict(engine_vs_fd_valid_columns=cc.stats(err), valid_fraction=cc.stats(frac),
                                          engine_vs_fd_states_all_valid=cc.stats(err_all_valid),
                                          states_all_valid=len(err_all_valid), engine_vs_fd_all_columns=cc.stats(err_all),
                                          forward_vs_C=cc.stats(fwd), engine_vs_fd_valid_forward_agrees=cc.stats(agree),
                                          tape_bitwise=cg.tape_bitwise(m, info, q, v, u, S))
    if a.selffd_states:
        qs, vs, us = cc.corpus(mo, a.selffd_states, a.seed + 3)
        rep["selffd"] = {str(S): selffd(m, info, qs, vs, us, S, a.selffd_eps, a.seed + 11) for S in a.substeps}
        rep["selffd_eps"] = a.selffd_eps
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

"""The forest engine's VJP on the nv128 scenes against float64 MuJoCo C differences that respect the active set
(sided differences at the columns whose central differences straddle a kink; position, velocity and control columns):"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "scripts/eq", "scripts/forest"):
    sys.path.insert(0, str(ROOT / p))

HS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


def _c_model(scene, tol, it):
    import connect_grad as cg
    import nv128_check as n
    m = n.load(scene)
    mm = cg.tight(m)
    if tol:
        mm.opt.tolerance = mm.opt.ls_tolerance = tol
    if it:
        mm.opt.iterations = it
    return m, mm


def _c_state(args):
    scene, tol, it, q, v, u, gq, gv, S = args
    import mujoco
    m, mm = _c_model(scene, tol, it)
    d = mujoco.MjData(mm)
    nv, nu = mm.nv, mm.nu
    stats = []

    def roll(qq, vv, uu, rec=False):
        mujoco.mj_resetData(mm, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = qq, vv, uu
        for _ in range(S):
            mujoco.mj_step(mm, d)
            if rec:
                lim = int((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT).sum())
                stats.append((int(d.ncon), lim, int(d.nefc), int(np.max(d.solver_niter[:max(d.nisland, 1)]))))
        return float(gq @ d.qpos + gv @ d.qvel), d.qvel.copy(), d.qpos.copy()

    L0, v1, q1 = roll(q, v, u, rec=True)
    F = np.zeros((2 * nv + nu, 2 * len(HS)))
    for i in range(2 * nv + nu):
        for k, h in enumerate(HS):
            for s, sg in enumerate((1.0, -1.0)):
                qq, vv, uu = q.copy(), v.copy(), u.copy()
                if i < nv:
                    dv = np.zeros(nv); dv[i] = 1.0
                    mujoco.mj_integratePos(mm, qq, dv, sg * h)
                elif i < 2 * nv:
                    vv[i - nv] += sg * h
                else:
                    uu[i - 2 * nv] += sg * h
                F[i, 2 * k + s] = roll(qq, vv, uu)[0]
    return L0, F, np.array(stats), v1, q1


def references(L0, F, ge):
    """Central 1e-6 (old), the sided reference for the engine gradient ge, kink mask, unresolved mask, side gap."""
    c = {h: (F[:, 2 * k] - F[:, 2 * k + 1]) / (2 * h) for k, h in enumerate(HS)}
    one = {(h, sg): sg * (F[:, 2 * k + (0 if sg > 0 else 1)] - L0) / h for k, h in enumerate(HS) for sg in (1, -1)}
    c6, c7 = c[1e-6], c[1e-7]
    kink = np.abs(c6 - c7) > 1e-4 * np.maximum(np.abs(c6), 1.0)
    ref = c6.copy()
    unres = np.zeros(len(c6), bool)
    gap = np.zeros(len(c6))
    nsides = np.full(len(c6), 2)
    for i in np.nonzero(kink)[0]:
        sides = []
        for sg in (1, -1):
            o7, o8 = one[(1e-7, sg)][i], one[(1e-8, sg)][i]
            if abs(o7 - o8) <= 1e-3 * max(abs(o7), 1.0):
                sides.append(o7)
        gap[i] = abs(one[(1e-7, 1)][i] - one[(1e-7, -1)][i])
        nsides[i] = len(sides)
        if sides:
            ref[i] = min(sides, key=lambda x: abs(x - ge[i]))
        else:
            ref[i], unres[i] = np.nan, True
    return c6, ref, kink, unres, gap, nsides


def err(g, ref, sel):
    ok = sel & ~np.isnan(ref)
    if not ok.any():
        return None
    return float(np.abs(g[ok] - ref[ok]).max() / max(np.abs(ref[ok]).max(), 1e-12))


def analyze(z, out=None):
    """Tables from a saved npz (keys per substep count S: L0_S, F_S, ge_S_<variant>, cstats_S, ...)."""
    import connect_check as cc
    meta = json.loads(str(z["meta"]))
    nv, nu = meta["nv"], meta["nu"]
    blocks = dict(q=slice(0, nv), v=slice(nv, 2 * nv), u=slice(2 * nv, 2 * nv + nu))
    res = dict(meta=meta, by_substeps={})
    for S in meta["substeps"]:
        L0, F = z[f"L0_{S}"], z[f"F_{S}"]
        B, ncol = F.shape[:2]
        rs = {}
        for var in meta["variants"]:
            ge = z[f"ge_{S}_{var}"]
            rows = []
            for w in range(B):
                c6, ref, kink, unres, gap, nsides = references(L0[w], F[w], ge[w])
                allc = np.ones(ncol, bool)
                r = dict(central=err(ge[w], c6, allc), sided=err(ge[w], ref, allc),
                         smooth_only=err(ge[w], ref, ~kink), kink_cols=int(kink.sum()), unresolved=int(unres.sum()))
                for b, sl in blocks.items():
                    sel = np.zeros(ncol, bool); sel[sl] = True
                    r[f"central_{b}"] = err(ge[w], c6, sel)
                    r[f"sided_{b}"] = err(ge[w], ref, sel)
                    r[f"kink_{b}"] = int(kink[sl].sum())
                if meta.get("nv_robot") and nv > meta["nv_robot"]:
                    nr = meta["nv_robot"]
                    rob = np.zeros(ncol, bool); rob[:nr] = True; rob[nv:nv + nr] = True
                    obj = np.zeros(ncol, bool); obj[nr:nv] = True; obj[nv + nr:2 * nv] = True
                    r["sided_robot_qv"] = err(ge[w], ref, rob)
                    r["sided_object_qv"] = err(ge[w], ref, obj)
                d_ = np.where(np.isnan(ref), 0, np.abs(ge[w] - ref))
                j = int(d_.argmax())
                r["worst_col"] = j
                r["worst"] = dict(engine=float(ge[w][j]), ref=float(ref[j]), c6=float(c6[j]), kink=bool(kink[j]),
                                  nsides=int(nsides[j]), gap=float(gap[j]))
                rows.append(r)
            st = lambda k: cc.stats(np.array([x[k] for x in rows if x[k] is not None]))
            keys = [k for k in rows[0] if k not in ("worst_col", "worst") and not k.startswith("kink") and k != "unresolved"]
            rs[var] = dict({k: st(k) for k in keys}, states_with_kink=int(sum(x["kink_cols"] > 0 for x in rows)),
                           kink_cols_total=int(sum(x["kink_cols"] for x in rows)),
                           unresolved_total=int(sum(x["unresolved"] for x in rows)), per_state=rows)
        cs = z[f"cstats_{S}"]
        tp = z[f"tape_{S}"]
        res["by_substeps"][str(S)] = dict(
            variants=rs, c_ncon_max=cs[:, :, 0].max(1).tolist(), c_nlim_max=cs[:, :, 1].max(1).tolist(),
            c_niter_max=cs[:, :, 3].max(1).tolist(), engine_ncon_max=tp[:, :, 0].max(1).tolist(),
            engine_nlim_max=tp[:, :, 1].max(1).tolist(), engine_niter=tp[:, :, 2].tolist(),
            fwd_qvel_rel=z[f"fwd_{S}"].tolist())
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


def summary(res):
    for S, b in res["by_substeps"].items():
        for var, r in b["variants"].items():
            f = lambda k: "/".join(f"{r[k][x]:.2g}" for x in ("median", "p90", "max")) if r.get(k) else "-"
            print(f"S={S} {var:12s} central {f('central')}  sided {f('sided')}  [q {f('sided_q')} v {f('sided_v')} "
                  f"u {f('sided_u')}]  smooth-only {f('smooth_only')}  kink states {r['states_with_kink']} cols "
                  f"{r['kink_cols_total']} unresolved {r['unresolved_total']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?")
    ap.add_argument("--states", type=int, default=16)
    ap.add_argument("--substeps", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--variants", nargs="+", default=["dflt:0:0:3", "exact:0:1:0", "exact_it100:100:1:0"])
    ap.add_argument("--c-tol", type=float, default=None)
    ap.add_argument("--c-iter", type=int, default=None)
    ap.add_argument("--pool", type=int, default=0, help="select --states forward-consistent states from a corpus of "
                    "this size (0: the first --states corpus states, as nv128_check)")
    ap.add_argument("--fwd-thr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None, help="json (the npz with the raw evaluations next to it)")
    ap.add_argument("--analyze", default=None)
    a = ap.parse_args()
    if a.analyze:
        res = analyze(np.load(a.analyze, allow_pickle=False), a.out)
        summary(res)
        return
    import mujoco
    import connect_check as cc
    import connect_grad as cg
    import nv128_check as n
    from nablaengine.forest_engine import tape as ftape
    m = n.load(a.scene)
    info = n.init(m)
    picked = None
    if a.pool:
        q, v, u = cc.corpus(m, a.pool, 0)
        ok = np.ones(len(q), bool)
        for S in a.substeps:
            mm, dd = cg.tight(m), None
            dd = mujoco.MjData(mm)
            vc = np.stack([cg.rollout(mm, dd, *(x[w].astype(np.float64) for x in (q, v, u)), S)[1] for w in range(len(q))])
            ve = cc.engine_step(m, info, q, v, u, substeps=S)[1]
            ok &= np.abs(np.asarray(ve) - vc).max(1) / np.maximum(np.abs(vc).max(1), 1.0) < a.fwd_thr
        picked = np.nonzero(ok)[0][:a.states]
        q, v, u = q[picked], v[picked], u[picked]
        print(f"pool {a.pool}: {int(ok.sum())} forward-consistent (< {a.fwd_thr}), using {len(picked)}", flush=True)
    else:
        q, v, u = cc.corpus(m, a.states, 0)
    B, nv, nu = len(q), m.nv, m.nu
    nr = None
    fj = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
    if len(fj) > 1:
        nr = int(m.jnt_dofadr[fj[1]])
    variants = []
    for s in a.variants:
        nm, it, ck, nn = s.split(":")
        variants.append((nm, int(it), int(ck), int(nn)))
    meta = dict(scene=a.scene, states=B, substeps=a.substeps, variants=[x[0] for x in variants], variant_spec=a.variants,
                nv=int(nv), nu=int(nu), nv_robot=nr, model_iterations=int(m.opt.iterations), c_tol=a.c_tol,
                c_iter=a.c_iter, mujoco=mujoco.__version__, hs=list(HS), pool=a.pool, fwd_thr=a.fwd_thr,
                picked=None if picked is None else [int(x) for x in picked])
    save = {"meta": np.array(json.dumps(meta))}
    pool = mp.get_context("spawn").Pool(a.workers)
    rng = np.random.default_rng(7)
    for S in a.substeps:
        rng = np.random.default_rng(7)
        gq = rng.normal(size=(B, m.nq))
        gq[:, cg.free_quats(m)] = 0.0
        gv = rng.normal(size=(B, nv))
        jobs = [(a.scene, a.c_tol, a.c_iter, *(x[w].astype(np.float64) for x in (q, v, u)), gq[w], gv[w], S)
                for w in range(B)]
        cres = pool.map_async(_c_state, jobs)
        for nm, it, ck, nn in variants:
            mi = int(m.opt.iterations)
            if it:
                m.opt.iterations = it
            qe, ve, (bq, bv, bu) = cg.engine_vjp(m, info, q, v, u, gq, gv, S, clamp_kv=ck, normal=nn)
            m.opt.iterations = mi
            save[f"ge_{S}_{nm}"] = np.stack([np.concatenate([cg.q_to_tangent(m, q[w].astype(np.float64), bq[w]),
                                                             bv[w], bu[w]]) for w in range(B)])
            if nm == variants[0][0]:
                save[f"qe_{S}"], save[f"ve_{S}"] = qe, ve
        import jax.numpy as jnp
        from nablaengine import jax_ffi
        o = jax_ffi.forward(jnp.asarray(q), jnp.asarray(v), jnp.zeros((B, nv), jnp.float32), jnp.asarray(u),
                            jnp.ones((B,), jnp.float32), jnp.zeros((B, int(info.get("com_words", 3))), jnp.float32),
                            substeps=S, iterations=int(m.opt.iterations), tape=True)
        rec = ftape.decode(np.asarray(o[-1]).reshape(B, S, -1), m, info.get("max_contacts") or info.get("maxcon"),
                           info.get("max_rows") or info.get("nefc"))
        save[f"tape_{S}"] = np.stack([rec["ncon"], rec["nlim"], rec["niter"]], -1)
        cr = cres.get()
        save[f"L0_{S}"] = np.array([x[0] for x in cr])
        save[f"F_{S}"] = np.stack([x[1] for x in cr])
        save[f"cstats_{S}"] = np.stack([x[2] for x in cr])
        vc = np.stack([x[3] for x in cr])
        save[f"fwd_{S}"] = np.abs(save[f"ve_{S}"] - vc).max(1) / np.maximum(np.abs(vc).max(1), 1.0)
        print(f"S={S} done", flush=True)
    pool.close()
    if a.out:
        o = Path(a.out)
        if o.exists() or o.with_suffix(".npz").exists():
            sys.exit(f"{o} exists (not overwritten)")
        o.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(o.with_suffix(".npz"), **save)
        res = analyze(np.load(o.with_suffix(".npz")), str(o))
    else:
        res = analyze(save)
    summary(res)


if __name__ == "__main__":
    main()

"""Contact and constraint-row capacities (jax_ffi.init / torch_ffi.init max_contacts, max_rows): builds with larger
capacities against the default build and against MuJoCo Warp."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load(model):
    """(MjModel, foot pairs, COM body, substeps, corpus dict or None)."""
    import mujoco
    if model is None:
        from nablaengine import fixtures
        m, mt = fixtures.load_model()
        return m, list(mt["foot_pairs"]), mt.get("torso_body", "torso_link"), 4, fixtures.load_corpus("mf_s0")
    meta = json.loads(Path(model).with_suffix(".json").read_text())
    m = mujoco.MjModel.from_xml_path(str(Path(model).with_suffix(".xml")))
    return m, meta["foot_pairs"], meta["com_body"], meta["substeps_per_control_step"], None


def cmd_sink(a):
    m, fp, com_body, sub, z = load(a.model)
    if a.corpus:
        z = dict(np.load(a.corpus))
    B = min(a.nworld, len(z["qpos"]))
    q = np.asarray(z["qpos"][:B], np.float64).copy()
    depth = np.asarray(a.depths, np.float64)[np.arange(B) % len(a.depths)]
    q[:, 2] -= depth
    out = dict(qpos=q, qvel=z["qvel"][:B], ctrl=z["ctrl"][:B], qacc_warmstart=np.zeros((B, m.nv)), depth=depth,
               friction=np.asarray(z["friction"][:B]) if "friction" in z else np.ones(B),
               com=np.zeros((B, 3)), substeps=sub)
    np.savez(a.out, **out)
    print(f"sink: {B} states, depths {sorted(set(a.depths))}; {a.out}")


def cmd_ref(a):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    m, fp, com_body, sub, _ = load(a.model)
    z = np.load(a.corpus)
    B = len(z["qpos"])
    m.opt.iterations = a.iterations
    fr = np.asarray(z["friction"], np.float64)
    mw = mjw.put_model(m)
    pfr = np.tile(m.pair_friction, (B, 1, 1)).astype(np.float32)
    for p in fp:
        pfr[:, p, :2] = fr[:, None]
    mw.pair_friction = wp.array(pfr, dtype=mw.pair_friction.dtype)
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=a.nconmax, njmax=a.njmax)
    for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
        wp.copy(getattr(dw, k), wp.array(np.ascontiguousarray(z[k], np.float32)))
    ncon, nefc = np.zeros(B, np.int64), np.zeros(B, np.int64)
    for _ in range(int(z["substeps"])):
        mjw.step(mw, dw)
    nacon = int(dw.nacon.numpy()[0])
    ncon = np.bincount(dw.contact.worldid.numpy()[:nacon], minlength=B)
    nefc = dw.nefc.numpy().astype(np.int64)
    over = bool(ncon.max() >= a.nconmax or nefc.max() >= a.njmax)
    np.savez(a.out, **{k: z[k] for k in z.files}, out_qpos=dw.qpos.numpy(), out_qvel=dw.qvel.numpy(), ncon=ncon,
             nefc=nefc, iterations=a.iterations)
    print(json.dumps(dict(nworld=B, ncon_max=int(ncon.max()), nefc_max=int(nefc.max()), mjw_capacity_reached=over,
                          out=a.out)))
    if over:
        sys.exit("ref: MuJoCo Warp reached --nconmax / --njmax: raise them")


def cmd_check(a):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    m, fp, com_body, sub, _ = load(a.model)
    r = np.load(a.ref)
    info = jax_ffi.init(model=m, foot_pairs=fp, com_body=com_body, max_contacts=a.max_contacts, max_rows=a.max_rows)
    f = lambda k: jnp.asarray(np.ascontiguousarray(r[k], np.float32))
    it, S = int(r["iterations"]), int(r["substeps"])
    q1, v1, _, inf = jax_ffi.forward(f("qpos"), f("qvel"), f("qacc_warmstart"), f("ctrl"), f("friction"), f("com"),
                                     iterations=it, substeps=S)
    q1, v1, inf = np.asarray(q1), np.asarray(v1), np.asarray(inf)
    ev = np.linalg.norm(v1 - r["out_qvel"], axis=1) / np.maximum(np.linalg.norm(r["out_qvel"], axis=1), 1e-6)
    fits = (r["ncon"] <= info["maxcon"]) & (r["nefc"] <= info["nefc"])
    res = dict(nworld=len(q1), maxcon=info["maxcon"], nefc=info["nefc"], fits_share=float(fits.mean()))
    for name, sel in (("fits", fits), ("beyond", ~fits)):
        if not sel.any():
            res[name] = dict(worlds=0)
            continue
        fl = inf[sel, 3]
        res[name] = dict(worlds=int(sel.sum()), qvel_rel_median=float(np.median(ev[sel])),
                         qvel_rel_p99=float(np.quantile(ev[sel], 0.99)), qvel_rel_max=float(ev[sel].max()),
                         contact_overflow=int(((fl & 4) != 0).sum()), row_overflow=int(((fl & 16) != 0).sum()),
                         ncon_equal=float((inf[sel, 1] == r["ncon"][sel]).mean()),
                         nefc_equal=float((inf[sel, 2] == r["nefc"][sel]).mean()))
    if "depth" in r.files:
        res["by_depth"] = {f"{d:g}": dict(worlds=int((r["depth"] == d).sum()),
                                          fits=float(fits[r["depth"] == d].mean()),
                                          flagged=int((inf[r["depth"] == d, 3] & 20 != 0).sum()),
                                          qvel_rel_median_fits=(float(np.median(ev[(r["depth"] == d) & fits]))
                                                                if ((r["depth"] == d) & fits).any() else None))
                           for d in np.unique(r["depth"])}
    if a.dump:
        rng = np.random.default_rng(0)
        yq = jnp.asarray(rng.standard_normal(q1.shape), jnp.float32)
        yv = jnp.asarray(rng.standard_normal(v1.shape), jnp.float32)

        def loss(u):
            qs, vs, _ = jax_ffi.step(f("qpos"), f("qvel"), f("qacc_warmstart"), u, f("friction"), f("com"),
                                     iterations=it, substeps=S)
            return jnp.sum(yq * qs) + jnp.sum(yv * vs)

        g = np.asarray(jax.jit(jax.grad(loss))(f("ctrl")))
        np.savez(a.dump, q1=q1, v1=v1, info=inf, g=g)
    print(json.dumps(res, indent=1))


def cmd_grad64(a):
    """The VJP (JAX, the build's capacities, --iterations) of one control step against float64 central differences of
    MuJoCo C (tolerance 1e-12, the same iterations) along a random direction of qpos (not the root quaternion), qvel and
    ctrl per world with a random output cotangent, on --nworld of REF's worlds (--deepest: those with the most rows in
    MuJoCo Warp), reported over the worlds whose forward agrees with MuJoCo C (qvel relative < 1e-4)."""
    import jax
    import jax.numpy as jnp
    import mujoco
    from nablaengine import jax_ffi
    m, fp, com_body, sub, _ = load(a.model)
    r = np.load(a.ref)
    if a.indices:
        idx = np.load(a.indices)[:a.nworld]
    else:
        idx = np.argsort(-r["nefc"], kind="stable")[:a.nworld] if a.deepest else np.arange(min(a.nworld, len(r["qpos"])))
    B, S = len(idx), int(r["substeps"])
    jax_ffi.init(model=m, foot_pairs=fp, com_body=com_body, max_contacts=a.max_contacts, max_rows=a.max_rows)
    f64 = lambda k: np.asarray(r[k][idx], np.float64)
    rng = np.random.default_rng(1)
    yq, yv = rng.standard_normal((B, m.nq)), rng.standard_normal((B, m.nv))
    t = lambda x: jnp.asarray(np.asarray(x, np.float32))

    def loss(q, v, u):
        q1, v1, _ = jax_ffi.step(q, v, t(f64("qacc_warmstart")), u, t(f64("friction")), t(f64("com")),
                                 iterations=a.iterations, substeps=S)
        return jnp.sum(t(yq) * q1) + jnp.sum(t(yv) * v1), v1

    (_, v1), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(
        t(f64("qpos")), t(f64("qvel")), t(f64("ctrl")))
    v1 = np.asarray(v1, np.float64)
    m.opt.iterations, m.opt.tolerance, m.opt.ls_tolerance = a.iterations, 1e-12, 1e-12
    d = mujoco.MjData(m)
    fr = f64("friction")

    def run(i, q, v, u):
        mujoco.mj_resetData(m, d)
        for p in fp:
            m.pair_friction[p, :2] = fr[i]
        d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q, v, u, 0.0
        for _ in range(S):
            mujoco.mj_step(m, d)
        return d.qpos.copy(), d.qvel.copy()

    base = [(f64("qpos")[i], f64("qvel")[i], f64("ctrl")[i]) for i in range(B)]
    vc = np.array([run(i, *base[i])[1] for i in range(B)])
    agree = np.linalg.norm(v1 - vc, axis=1) / np.maximum(np.linalg.norm(vc, axis=1), 1e-6) < 1e-4
    res = dict(nworld=B, nefc_mjw_median=float(np.median(r["nefc"][idx])), forward_agree_share=float(agree.mean()),
               iterations=a.iterations)
    for k, name in enumerate(("qpos", "qvel", "ctrl")):
        an = np.asarray(grads[k], np.float64)
        rel = []
        for i in range(B):
            u = rng.standard_normal(an.shape[1])
            if name == "qpos":
                u[3:7] = 0.0
            u /= np.linalg.norm(u)
            lp, lm = [x.copy() for x in base[i]], [x.copy() for x in base[i]]
            lp[k] += a.eps * u
            lm[k] -= a.eps * u
            (qp, vp), (qm, vm) = run(i, *lp), run(i, *lm)
            fd = (yq[i] @ (qp - qm) + yv[i] @ (vp - vm)) / (2 * a.eps)
            g = an[i] @ u
            rel.append(abs(fd - g) / max(abs(fd), abs(g), 1e-12))
        rel = np.asarray(rel)
        res[name] = dict(agree_rel_median=float(np.median(rel[agree])) if agree.any() else None,
                         agree_rel_p90=float(np.quantile(rel[agree], 0.9)) if agree.any() else None)
    print(json.dumps(res, indent=1))


def cmd_selffd(a):
    """The VJP against central differences of the same build (float32, all of REF's worlds at once) along a random
    direction of qpos (not the root quaternion), qvel and ctrl per world with a random output cotangent: rel = |fd - an|
    / max(|fd|, |an|) per world, summarized over the worlds whose contact count is the same on both sides, per group
    of --groups (an .npz of boolean masks over REF's worlds; default: all)."""
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    m, fp, com_body, sub, _ = load(a.model)
    r = np.load(a.ref)
    B, S = len(r["qpos"]), int(r["substeps"])
    jax_ffi.init(model=m, foot_pairs=fp, com_body=com_body, max_contacts=a.max_contacts, max_rows=a.max_rows)
    t = lambda k: jnp.asarray(np.ascontiguousarray(r[k], np.float32))
    x0 = [t("qpos"), t("qvel"), t("ctrl")]
    rng = np.random.default_rng(0)
    yq = jnp.asarray(rng.standard_normal((B, m.nq)), jnp.float32)
    yv = jnp.asarray(rng.standard_normal((B, m.nv)), jnp.float32)
    it = a.iterations

    def out(q, v, u):
        q1, v1, _, info = jax_ffi.forward(q, v, t("qacc_warmstart"), u, t("friction"), t("com"), iterations=it,
                                          substeps=S)
        return jnp.sum(yq * q1, 1) + jnp.sum(yv * v1, 1), info[:, 1]

    def loss(q, v, u):
        q1, v1, _ = jax_ffi.step(q, v, t("qacc_warmstart"), u, t("friction"), t("com"), iterations=it, substeps=S)
        return jnp.sum(yq * q1) + jnp.sum(yv * v1)

    grads = jax.grad(loss, argnums=(0, 1, 2))(*x0)
    groups = dict(np.load(a.groups)) if a.groups else {"all": np.ones(B, bool)}
    res = dict(nworld=B, maxcon=jax_ffi.init()["maxcon"], nefc=jax_ffi.init()["nefc"])
    for k, (name, eps) in enumerate((("qpos", a.eps_q), ("qvel", a.eps_v), ("ctrl", a.eps_u))):
        d = rng.standard_normal(np.asarray(x0[k]).shape)
        if name == "qpos":
            d[:, 3:7] = 0.0
        d = jnp.asarray(d / np.linalg.norm(d, axis=1, keepdims=True), jnp.float32)
        xp, xm = list(x0), list(x0)
        xp[k], xm[k] = x0[k] + eps * d, x0[k] - eps * d
        (lp, np_), (lm, nm) = out(*xp), out(*xm)
        fd = (np.asarray(lp, np.float64) - np.asarray(lm, np.float64)) / (2 * eps)
        an = np.asarray(jnp.sum(grads[k] * d, 1), np.float64)
        rel = np.abs(fd - an) / np.maximum(np.maximum(np.abs(fd), np.abs(an)), 1e-12)
        same = np.asarray(np_) == np.asarray(nm)
        res[name] = {g: dict(worlds=int((msk & same).sum()), rel_median=float(np.median(rel[msk & same])),
                             rel_p90=float(np.quantile(rel[msk & same], 0.9)))
                     for g, msk in groups.items() if (msk & same).any()}
    print(json.dumps(res, indent=1))


def cmd_time(a):
    """Time per control step (PyTorch, CUDA events, median of --reps after warm-up) of the forward and of the forward
    with the backward at --nworld worlds of the build with --max-contacts / --max-rows: REF's states tiled, or without
    --ref the G1 corpus mf_s0 (its friction and COM offsets)."""
    import torch
    from nablaengine import torch_ffi
    m, fp, com_body, sub, z = load(a.model)
    if a.ref:
        z = dict(np.load(a.ref))
        sub = int(z["substeps"])
    torch_ffi.init(model=m, foot_pairs=fp, com_body=com_body, max_contacts=a.max_contacts, max_rows=a.max_rows)
    info = torch_ffi.init()
    idx = np.arange(a.nworld) % len(z["qpos"])
    t = lambda k: torch.from_numpy(np.ascontiguousarray(np.asarray(z[k])[idx], np.float32)).cuda()
    qpos, qvel, warm, ctrl, fr, com = t("qpos"), t("qvel"), t("qacc_warmstart"), t("ctrl"), t("friction"), t("com")
    u = ctrl.clone().requires_grad_()

    def fwd():
        torch_ffi.forward(qpos, qvel, warm, ctrl, fr, com, iterations=a.iterations, substeps=sub)

    def both():
        q1, v1, _ = torch_ffi.step(qpos, qvel, warm, u, fr, com, iterations=a.iterations, substeps=sub)
        (q1.sum() + v1.sum()).backward()

    res = dict(maxcon=info["maxcon"], nefc=info["nefc"], nworld=a.nworld, iterations=a.iterations,
               states=a.ref or "mf_s0", gpu=torch.cuda.get_device_name())
    for key, f in (("forward_ms", fwd), ("forward_backward_ms", both)):
        for _ in range(3):
            f()
        ts = []
        for _ in range(a.reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            f()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        res[key] = round(float(np.median(ts)), 3)
    print(json.dumps(res))


def cmd_compare(a):
    x, y = np.load(a.a), np.load(a.b)
    res = {k: dict(bitwise=bool(np.array_equal(x[k], y[k])), max_abs=float(np.abs(x[k] - y[k]).max()))
           for k in x.files if k in y.files and k != "info"}
    res["info_counts_equal"] = bool(np.array_equal(x["info"][:, 1:3], y["info"][:, 1:3]))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sink")
    p.add_argument("--model", default=None)
    p.add_argument("--corpus", default=None, help="states (qpos, qvel, ctrl[, friction]) instead of the G1's mf_s0")
    p.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4])
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--out", required=True)
    p = sub.add_parser("ref")
    p.add_argument("--model", default=None)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--nconmax", type=int, default=256, help="MuJoCo Warp's contact capacity per world")
    p.add_argument("--njmax", type=int, default=1024, help="MuJoCo Warp's constraint-row capacity per world")
    p = sub.add_parser("check")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", required=True)
    p.add_argument("--max-contacts", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--dump", default=None)
    p = sub.add_parser("grad64")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", required=True)
    p.add_argument("--max-contacts", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--nworld", type=int, default=64)
    p.add_argument("--deepest", action="store_true", help="REF's worlds with the most constraint rows")
    p.add_argument("--indices", default=None, help="an .npy file of REF's world indices instead")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-6)
    p = sub.add_parser("selffd")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", required=True)
    p.add_argument("--max-contacts", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--groups", default=None, help="an .npz of boolean masks over REF's worlds")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--eps-q", type=float, default=1e-4)
    p.add_argument("--eps-v", type=float, default=1e-3)
    p.add_argument("--eps-u", type=float, default=1e-3)
    p = sub.add_parser("time")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", default=None, help="states (default: the G1 corpus mf_s0)")
    p.add_argument("--max-contacts", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--reps", type=int, default=30)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    a = ap.parse_args()
    {"sink": cmd_sink, "ref": cmd_ref, "check": cmd_check, "grad64": cmd_grad64, "selffd": cmd_selffd,
     "time": cmd_time, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Elliptic friction cones: the engine on a model with opt.cone = elliptic against MuJoCo Warp."""
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


def set_condim(m, condim, torsional=None):
    """The frictional pairs' condim (3 or 4) in place; condim 4: torsional friction pair_friction[2] = torsional if
    given, else the pair's own (0.005 x the sliding one where that is 0)."""
    for p in range(m.npair):
        if m.pair_dim[p] >= 3:
            m.pair_dim[p] = condim
            if condim >= 4 and torsional is not None:
                m.pair_friction[p, 2] = torsional
            elif condim >= 4 and m.pair_friction[p, 2] == 0.0:
                m.pair_friction[p, 2] = 0.005 * m.pair_friction[p, 0]
    return m


def elliptic_model(m, impratio, condim, torsional=None):
    """m with the elliptic cone, opt.impratio and the frictional pairs' condim (set_condim), in place."""
    import mujoco
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    m.opt.impratio = impratio
    return set_condim(m, condim, torsional)


def cmd_ref(a):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    m, fp, com_body, sub, z = load(a.model)
    if a.corpus:
        z = dict(np.load(a.corpus))
    if a.pyramidal:
        m.opt.impratio = a.impratio
        set_condim(m, a.condim, a.torsional)
    else:
        m = elliptic_model(m, a.impratio, a.condim, a.torsional)
    m.opt.iterations = a.iterations
    if a.ls_iterations:
        m.opt.ls_iterations = a.ls_iterations
    sub = a.substeps or sub
    B = min(a.nworld, len(z["qpos"]))
    fr = np.asarray(z["friction"][:B], np.float64) if "friction" in z else np.full(B, m.pair_friction[fp[0], 0])
    mw = mjw.put_model(m)
    pfr = np.tile(m.pair_friction, (B, 1, 1)).astype(np.float32)
    for p in fp:
        pfr[:, p, :2] = fr[:, None]
    mw.pair_friction = wp.array(pfr, dtype=mw.pair_friction.dtype)
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=a.nconmax, njmax=a.njmax)
    for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
        wp.copy(getattr(dw, k), wp.array(np.ascontiguousarray(z[k][:B], np.float32)))
    for _ in range(sub):
        mjw.step(mw, dw)
    nacon = int(dw.nacon.numpy()[0])
    ncon = np.bincount(dw.contact.worldid.numpy()[:nacon], minlength=B)
    nefc = dw.nefc.numpy().astype(np.int64)
    np.savez(a.out, qpos=z["qpos"][:B], qvel=z["qvel"][:B], qacc_warmstart=z["qacc_warmstart"][:B],
             ctrl=z["ctrl"][:B], friction=fr, com=np.zeros((B, 3)), out_qpos=dw.qpos.numpy(),
             out_qvel=dw.qvel.numpy(), ncon=ncon, nefc=nefc, substeps=sub, iterations=a.iterations,
             impratio=a.impratio, condim=a.condim, foot_pairs=np.asarray(fp), com_body=str(com_body),
             cone="pyramidal" if a.pyramidal else "elliptic")
    mujoco.mj_saveModel(m, str(Path(a.out).with_suffix(".mjb")), None)
    print(json.dumps(dict(nworld=B, ncon_mean=float(ncon.mean()), ncon_max=int(ncon.max()), nefc_max=int(nefc.max()),
                          impratio=a.impratio, condim=a.condim, out=a.out)))


def cmd_check(a):
    import jax.numpy as jnp
    import mujoco
    from nablaengine import jax_ffi
    r = np.load(a.ref)
    m = mujoco.MjModel.from_binary_path(str(Path(a.ref).with_suffix(".mjb")))
    jax_ffi.init(model=m, foot_pairs=[int(p) for p in r["foot_pairs"]], com_body=str(r["com_body"]))
    f = lambda k: jnp.asarray(np.ascontiguousarray(r[k], np.float32))
    q1, v1, _, info = jax_ffi.forward(f("qpos"), f("qvel"), f("qacc_warmstart"), f("ctrl"), f("friction"), f("com"),
                                      iterations=int(r["iterations"]), substeps=int(r["substeps"]))
    q1, v1, info = np.asarray(q1), np.asarray(v1), np.asarray(info)
    ev = np.linalg.norm(v1 - r["out_qvel"], axis=1) / np.maximum(np.linalg.norm(r["out_qvel"], axis=1), 1e-6)
    res = dict(nworld=len(q1), impratio=float(r["impratio"]), condim=int(r["condim"]),
               qvel_rel_median=float(np.median(ev)), qvel_rel_p99=float(np.quantile(ev, 0.99)),
               qvel_rel_max=float(ev.max()), qpos_abs_max=float(np.abs(q1 - r["out_qpos"]).max()),
               ncon_equal=float((info[:, 1] == r["ncon"]).mean()), nefc_equal=float((info[:, 2] == r["nefc"]).mean()),
               flags=int((info[:, 3] != 0).sum()))
    print(json.dumps(res, indent=1))


def _engine(a):
    import mujoco
    from nablaengine import jax_ffi
    r = np.load(a.ref)
    m = mujoco.MjModel.from_binary_path(str(Path(a.ref).with_suffix(".mjb")))
    fp = [int(p) for p in r["foot_pairs"]]
    jax_ffi.init(model=m, foot_pairs=fp, com_body=str(r["com_body"]))
    return r, m, fp


def cmd_selffd(a):
    """The VJP against central differences of the engine itself (float32, all worlds at once) along a random direction
    of qpos (not the root quaternion), qvel and ctrl per world with a random output cotangent; rel = |fd - an| /
    max(|fd|, |an|) over the worlds whose contact count is the same on both sides."""
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    r, m, fp = _engine(a)
    B, S, it = len(r["qpos"]), int(r["substeps"]), a.iterations or int(r["iterations"])
    t = lambda k: jnp.asarray(np.ascontiguousarray(r[k], np.float32))
    x0 = [t("qpos"), t("qvel"), t("ctrl")]
    rng = np.random.default_rng(0)
    yq = jnp.asarray(rng.standard_normal((B, m.nq)), jnp.float32)
    yv = jnp.asarray(rng.standard_normal((B, m.nv)), jnp.float32)

    def out(q, v, u):
        q1, v1, _, info = jax_ffi.forward(q, v, t("qacc_warmstart"), u, t("friction"), t("com"), iterations=it,
                                          substeps=S)
        return jnp.sum(yq * q1, 1) + jnp.sum(yv * v1, 1), info[:, 1]

    def loss(q, v, u):
        q1, v1, _ = jax_ffi.step(q, v, t("qacc_warmstart"), u, t("friction"), t("com"), iterations=it, substeps=S)
        return jnp.sum(yq * q1) + jnp.sum(yv * v1)

    grads = jax.grad(loss, argnums=(0, 1, 2))(*x0)
    res = dict(nworld=B, iterations=it, impratio=float(r["impratio"]))
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
        res[name] = dict(rel_median=float(np.median(rel[same])), rel_p90=float(np.quantile(rel[same], 0.9)),
                         same_contacts=float(same.mean()))
    print(json.dumps(res, indent=1))


def cmd_grad64(a):
    """The VJP (--iterations) against float64 central differences of MuJoCo C (REF.mjb, tolerance 1e-12, the same
    iterations) along a random direction of qpos (not the root quaternion), qvel and ctrl per world with a random output
    cotangent, over the worlds whose forward agrees with MuJoCo C (qvel relative < 1e-4)."""
    import jax
    import jax.numpy as jnp
    import mujoco
    from nablaengine import jax_ffi
    r, m, fp = _engine(a)
    B, S = min(a.nworld, len(r["qpos"])), int(r["substeps"])
    f64 = lambda k: np.asarray(r[k][:B], np.float64)
    t = lambda x: jnp.asarray(np.asarray(x, np.float32))
    rng = np.random.default_rng(1)
    yq, yv = rng.standard_normal((B, m.nq)), rng.standard_normal((B, m.nv))

    def loss(q, v, u):
        q1, v1, _ = jax_ffi.step(q, v, t(f64("qacc_warmstart")), u, t(f64("friction")), t(f64("com")),
                                 iterations=a.iterations, substeps=S)
        return jnp.sum(t(yq) * q1) + jnp.sum(t(yv) * v1), v1

    (_, v1), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(
        t(f64("qpos")), t(f64("qvel")), t(f64("ctrl")))
    v1 = np.asarray(v1, np.float64)
    m.opt.iterations, m.opt.tolerance, m.opt.ls_tolerance, m.opt.ls_iterations = a.iterations, 1e-12, 1e-12, 100
    d = mujoco.MjData(m)
    fr = f64("friction")

    def run(i, q, v, u):
        mujoco.mj_resetData(m, d)
        for p in fp:
            m.pair_friction[p, :2] = fr[i]
        d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q, v, u, f64("qacc_warmstart")[i]
        for _ in range(S):
            mujoco.mj_step(m, d)
        return d.qpos.copy(), d.qvel.copy()

    base = [(f64("qpos")[i], f64("qvel")[i], f64("ctrl")[i]) for i in range(B)]
    vc = np.array([run(i, *base[i])[1] for i in range(B)])
    agree = np.linalg.norm(v1 - vc, axis=1) / np.maximum(np.linalg.norm(vc, axis=1), 1e-6) < 1e-4
    res = dict(nworld=B, impratio=float(r["impratio"]), iterations=a.iterations, forward_agree_share=float(agree.mean()))
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ref")
    p.add_argument("--model", default=None)
    p.add_argument("--corpus", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--impratio", type=float, default=1.0)
    p.add_argument("--condim", type=int, default=3, choices=(3, 4))
    p.add_argument("--pyramidal", action="store_true", help="keep the pyramidal cone (a baseline reference)")
    p.add_argument("--torsional", type=float, default=None,
                   help="condim 4: the frictional pairs' torsional friction (default: their own)")
    p.add_argument("--substeps", type=int, default=None, help="substeps of the control step (default: the model's)")
    p.add_argument("--ls-iterations", type=int, default=None, help="MuJoCo Warp's line search iterations (default: "
                   "the model's; the engine's line search is exact)")
    p.add_argument("--nconmax", type=int, default=96)
    p.add_argument("--njmax", type=int, default=512)
    p = sub.add_parser("check")
    p.add_argument("--ref", required=True)
    p = sub.add_parser("selffd")
    p.add_argument("--ref", required=True)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--eps-q", type=float, default=1e-4)
    p.add_argument("--eps-v", type=float, default=1e-3)
    p.add_argument("--eps-u", type=float, default=1e-3)
    p = sub.add_parser("grad64")
    p.add_argument("--ref", required=True)
    p.add_argument("--nworld", type=int, default=128)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-6)
    a = ap.parse_args()
    {"ref": cmd_ref, "check": cmd_check, "selffd": cmd_selffd, "grad64": cmd_grad64}[a.cmd](a)


if __name__ == "__main__":
    main()

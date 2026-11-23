"""Engine vs MuJoCo Warp (and float64 MuJoCo) on another robot: MODEL.xml / MODEL.json from export_robot.py."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def load(model):
    """(MjModel, meta). $ROBOT_DROP_BOX=1: without the pairs of box geoms."""
    import mujoco
    meta = json.loads(Path(model).with_suffix(".json").read_text())
    spec = mujoco.MjSpec.from_file(str(Path(model).with_suffix(".xml")))
    if os.environ.get("ROBOT_DROP_BOX") == "1":
        box = {g.name for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_BOX}
        names = [p.name for p in spec.pairs]
        for p in list(spec.pairs):
            if p.geomname1 in box or p.geomname2 in box:
                spec.delete(p)
        kept = [p.name for p in spec.pairs]
        meta["foot_pairs"] = [kept.index(names[i]) for i in meta["foot_pairs"]]
    return spec.compile(), meta


def load_reference(model):
    """(MjModel, meta) of MuJoCo's equivalent of model: meta's reference_model (the export without fusestatic, beside
    model) where the model carries per-geom contact inverse weights, else load(model)."""
    m, meta = load(model)
    if not meta.get("reference_model"):
        return m, meta
    if os.environ.get("ROBOT_DROP_BOX") == "1":
        raise SystemExit("ROBOT_DROP_BOX with a reference_model: not supported")
    import mujoco
    mr = mujoco.MjModel.from_xml_path(str(Path(model).parent / (meta["reference_model"] + ".xml")))
    assert (mr.nq, mr.nv, mr.nu, mr.npair) == (m.nq, m.nv, m.nu, m.npair), "reference_model: layouts differ"
    return mr, meta


def mjw_setup(m, B, nconmax=64, njmax=256):
    import mujoco
    import mujoco_warp as mjw
    return mjw.put_model(m), mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=nconmax, njmax=njmax)


class Engine:
    """The engine on one model (None: the G1 fixture) through the PyTorch interface or the JAX one (--api): one control
    step's forward and VJP on numpy inputs (the two interfaces are bitwise the same)."""

    def __init__(self, api, m=None, foot_pairs=None, com_body=None):
        kw = {} if m is None else dict(model=m, foot_pairs=foot_pairs, com_body=com_body)
        if api == "jax":
            from nablaengine import jax_ffi as ffi
        else:
            from nablaengine import torch_ffi as ffi
        ffi.init(**kw)
        self.api, self.ffi = api, ffi

    def _in(self, x):
        if self.api == "jax":
            import jax.numpy as jnp
            return jnp.asarray(np.ascontiguousarray(x, np.float32))
        import torch
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)).cuda()

    def forward(self, x, iterations, substeps):
        """(qpos', qvel', info) of the states x (qpos, qvel, qacc_warmstart, ctrl, friction, com)."""
        args = [self._in(x[k]) for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com")]
        q1, v1, _, info = self.ffi.forward(*args, iterations=iterations, substeps=substeps)[:4]
        return [np.asarray(y) if self.api == "jax" else y.cpu().numpy() for y in (q1, v1, info)]

    def vjp(self, x, yq, yv, iterations, substeps, clamp_kv, normal):
        """(qpos', qvel', g_qpos, g_qvel, g_ctrl) of L = yq . qpos' + yv . qvel'."""
        args = [self._in(x[k]) for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com")]
        if self.api == "jax":
            q1, v1, _, _, tp = self.ffi.forward(*args, iterations=iterations, substeps=substeps, tape=True)
            g = self.ffi.backward(tp, args[4], args[5], self._in(yq), self._in(yv), substeps=substeps,
                                  clamp_kv=clamp_kv, normal=normal)
            return [np.asarray(y) for y in (q1, v1, g[0], g[1], g[2])]
        for i in (0, 1, 3):
            args[i].requires_grad_()
        q1, v1, _ = self.ffi.step(*args, iterations=iterations, substeps=substeps, clamp_kv=clamp_kv, normal=normal)
        ((self._in(yq) * q1).sum() + (self._in(yv) * v1).sum()).backward()
        return [y.detach().cpu().numpy() for y in (q1, v1, args[0].grad, args[1].grad, args[3].grad)]


def cmd_corpus(a):
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    m, meta = load_reference(a.model)
    m.opt.iterations = a.iterations
    rng = np.random.default_rng(a.seed)
    B, sub = a.nworld, meta["substeps_per_control_step"]
    q0 = np.asarray(meta["init_qpos"])

    def init(n):
        q = np.tile(q0, (n, 1))
        q[:, 2] += rng.uniform(-0.05, 0.1, n)
        ax = rng.normal(size=(n, 3)) * [1, 1, 0.3]
        ang = rng.uniform(0, a.tilt, n)
        ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        q[:, 3:7] = np.concatenate([np.cos(ang / 2)[:, None], np.sin(ang / 2)[:, None] * ax], 1)
        return q, rng.normal(0, 0.3, (n, m.nv))

    q, v = init(B)
    mw, dw = mjw_setup(m, B)
    wp.copy(dw.qpos, wp.array(q.astype(np.float32)))
    wp.copy(dw.qvel, wp.array(v.astype(np.float32)))
    target = np.tile(q0[7:], (B, 1))
    lim = m.actuator_ctrllimited.astype(bool) & (not (m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)))
    rec = {k: [] for k in ("qpos", "qvel", "qacc_warmstart", "ctrl")}
    when = rng.integers(5, a.steps, (B, a.per_world))
    for t in range(a.steps):
        target += rng.normal(0, a.drift, target.shape)
        target = q0[7:] + np.clip(target - q0[7:], -a.span, a.span)
        ctrl = np.where(lim, np.clip(target, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]), target)
        ctrl = ctrl.astype(np.float32)
        wp.copy(dw.ctrl, wp.array(ctrl))
        for _ in range(sub):
            mjw.step(mw, dw)
        sel = np.nonzero((when == t).any(1))[0]
        if len(sel):
            for k in rec:
                rec[k].append((dw.ctrl if k == "ctrl" else getattr(dw, k)).numpy()[sel])
        qn = dw.qpos.numpy()
        fall = np.nonzero(~(qn[:, 2] >= a.reset * q0[2]) | ~np.isfinite(dw.qvel.numpy()).all(1))[0]
        if len(fall):
            vn, wn = dw.qvel.numpy(), dw.qacc_warmstart.numpy()
            qn[fall], vn[fall] = init(len(fall))
            wn[fall] = 0
            target[fall] = q0[7:]
            for x, y in ((dw.qpos, qn), (dw.qvel, vn), (dw.qacc_warmstart, wn)):
                wp.copy(x, wp.array(y.astype(np.float32)))
    out = {k: np.concatenate(v).astype(np.float32) for k, v in rec.items()}
    ok = np.all([np.isfinite(x).all(1) for x in out.values()], 0)
    out = {k: x[ok] for k, x in out.items()}
    if not ok.all():
        print(f"corpus: dropped {np.sum(~ok)} non-finite states")
    mu = {float(m.pair_friction[p, 0]) for p in meta["foot_pairs"]}
    assert len(mu) == 1, mu
    out["friction"] = np.full(len(out["qpos"]), mu.pop(), np.float32)
    out["com"] = np.zeros((len(out["qpos"]), 3), np.float32)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    z = out["qpos"][:, 2]
    print(f"corpus: {len(z)} states, base height {np.quantile(z, 0.05):.3f} .. {np.quantile(z, 0.95):.3f}; {a.out}")


def cmd_ref(a):
    import warp as wp
    import mujoco_warp as mjw
    m, meta = load_reference(a.model)
    m.opt.iterations = a.iterations
    z = np.load(a.corpus)
    B = min(a.nworld, len(z["qpos"]))
    mw, dw = mjw_setup(m, B)
    for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
        wp.copy(getattr(dw, k), wp.array(np.ascontiguousarray(z[k][:B])))
    for i in range(meta["substeps_per_control_step"]):
        mjw.step(mw, dw)
        if i == meta["substeps_per_control_step"] - 1:
            nacon = int(dw.nacon.numpy()[0])
            ncon = np.bincount(dw.contact.worldid.numpy()[:nacon], minlength=B)
    np.savez(a.out, **{k: z[k][:B] for k in z.files}, out_qpos=dw.qpos.numpy(), out_qvel=dw.qvel.numpy(), ncon=ncon,
             iterations=a.iterations)
    print(f"ref: {B} worlds, mean contacts {ncon.mean():.2f}, max {ncon.max()}; {a.out}")


def cmd_check(a):
    m, meta = load(a.model)
    r = np.load(a.ref)
    eng = Engine(a.api, m, meta["foot_pairs"], meta["com_body"])
    q1, v1, info = eng.forward(r, int(r["iterations"]), meta["substeps_per_control_step"])
    it_max = int(info[:, 0].max())
    ev = np.linalg.norm(v1 - r["out_qvel"], axis=1) / np.maximum(np.linalg.norm(r["out_qvel"], axis=1), 1e-6)
    eq = np.abs(q1 - r["out_qpos"]).max(1)
    if a.dump:
        np.savez(a.dump, q1=q1, v1=v1, info=info, ev=ev)
    print(json.dumps(dict(nworld=len(ev), qvel_rel_median=float(np.median(ev)), qvel_rel_p99=float(np.quantile(ev, 0.99)),
                          qvel_rel_max=float(ev.max()), qpos_abs_median=float(np.median(eq)), qpos_abs_max=float(eq.max()),
                          ncon_equal=float((info[:, 1] == r["ncon"]).mean()), ncon_mean=float(info[:, 1].mean()),
                          flags=int((info[:, 3] != 0).sum()), newton_iterations_max=it_max,
                          at_iteration_limit=float((info[:, 0] >= int(r["iterations"])).mean())), indent=1))


def cmd_grad(a):
    """The engine's VJP against its own float32 central differences, per world along a random direction of qpos (not
    the root quaternion), qvel and ctrl with a random output cotangent (as scripts/terrain_check.py grad); --g1: the
    G1 fixture on its corpus (the differences' noise level)."""
    import torch
    from nablaengine import fixtures, torch_ffi
    if a.g1:
        z = fixtures.load_corpus("mf_s0")
        r = {k: z[k][:a.nworld] for k in ("qpos", "qvel", "ctrl", "qacc_warmstart", "friction", "com")}
        sub = 4
        torch_ffi.init()
    else:
        m, meta = load(a.model)
        r = dict(np.load(a.ref))
        sub = meta["substeps_per_control_step"]
        torch_ffi.init(model=m, foot_pairs=meta["foot_pairs"], com_body=meta["com_body"])
    B = min(a.nworld, len(r["qpos"]))
    t = lambda k: torch.from_numpy(np.ascontiguousarray(r[k][:B], np.float32)).cuda()
    qpos, qvel, ctrl, warm, fr, com = t("qpos"), t("qvel"), t("ctrl"), t("qacc_warmstart"), t("friction"), t("com")
    gen = torch.Generator(device="cuda").manual_seed(0)
    yq = torch.randn(B, qpos.shape[1], device="cuda", generator=gen)
    yv = torch.randn(B, qvel.shape[1], device="cuda", generator=gen)

    def out(q, v, u):
        q1, v1, _, info = torch_ffi.forward(q, v, warm, u, fr, com, iterations=a.iterations, substeps=sub)
        return (yq * q1).sum(1).double() + (yv * v1).sum(1).double(), info[:, 1]

    xs = [qpos.clone().requires_grad_(), qvel.clone().requires_grad_(), ctrl.clone().requires_grad_()]
    q1, v1, _ = torch_ffi.step(xs[0], xs[1], warm, xs[2], fr, com, iterations=a.iterations, substeps=sub)
    ((yq * q1).sum() + (yv * v1).sum()).backward()
    res = dict(nworld=B, model="g1" if a.g1 else a.model)
    for k, (name, x, eps) in enumerate((("qpos", qpos, a.eps_q), ("qvel", qvel, a.eps_v), ("ctrl", ctrl, a.eps_u))):
        d = torch.randn(x.shape, device="cuda", generator=gen)
        if name == "qpos":
            d[:, 3:7] = 0.0
        d /= d.norm(dim=1, keepdim=True)
        ap, am = [qpos, qvel, ctrl], [qpos, qvel, ctrl]
        ap[k], am[k] = x + eps * d, x - eps * d
        lp, np_ = out(*ap)
        lm, nm = out(*am)
        fd = ((lp - lm) / (2 * eps)).cpu().numpy()
        an = (xs[k].grad * d).sum(1).double().cpu().numpy()
        rel = np.abs(fd - an) / np.maximum(np.maximum(np.abs(fd), np.abs(an)), 1e-12)
        same = (np_ == nm).cpu().numpy()
        res[name] = dict(rel_median=float(np.median(rel[same])), rel_p90=float(np.quantile(rel[same], 0.9)),
                         same_contacts=float(same.mean()), nonfinite=int((~np.isfinite(an)).sum()))
    print(json.dumps(res, indent=1))


def cmd_grad64(a):
    """The engine's VJP (--iterations, default 50) against float64 central differences of MuJoCo C (mj_step, the same
    model, or its reference_model (load_reference), with tolerance 1e-12 and the same iterations) along a random
    direction of qpos (not the root quaternion), qvel and ctrl per world with a random output cotangent; --g1: the G1
    fixture on its corpus."""
    import mujoco
    from nablaengine import fixtures
    if a.g1:
        m, mt = fixtures.load_model()
        z = fixtures.load_corpus("mf_s0")
        r = {k: z[k] for k in ("qpos", "qvel", "ctrl", "qacc_warmstart", "friction", "com")}
        sub, fp = 4, mt["foot_pairs"]
        eng = Engine(a.api)
    else:
        me, meta = load(a.model)
        r = dict(np.load(a.ref))
        sub, fp, com = meta["substeps_per_control_step"], meta["foot_pairs"], meta["com_body"]
        eng = Engine(a.api, me, fp, com)
        m = load_reference(a.model)[0]
        fp = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PAIR, me.pair(int(p)).name) for p in fp]
        assert min(fp, default=0) >= 0
    rng = np.random.default_rng(1)
    idx = rng.choice(len(r["qpos"]), min(a.nworld, len(r["qpos"])), replace=False)
    B = len(idx)
    yq, yv = rng.standard_normal((B, m.nq)), rng.standard_normal((B, m.nv))
    q1, v1, *grads = eng.vjp({k: r[k][idx] for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com")},
                             yq, yv, a.iterations, sub, a.clamp_kv, a.normal)
    m.opt.iterations, m.opt.tolerance, m.opt.ls_tolerance = a.iterations, 1e-12, 1e-12
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    d = mujoco.MjData(m)
    fpairs = set(int(p) for p in fp)

    def run(q, v, u, w, fr):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q, v, u, w
        mu = m.pair_friction.copy()
        for p in fpairs:
            m.pair_friction[p, :2] = fr
        for _ in range(sub):
            mujoco.mj_step(m, d)
        m.pair_friction[:] = mu
        return d.qpos.copy(), d.qvel.copy()

    res = dict(nworld=B, model="g1" if a.g1 else a.model, iterations=a.iterations, clamp_kv=a.clamp_kv, normal=a.normal,
               api=a.api)
    q1n, v1n = q1.astype(np.float64), v1.astype(np.float64)
    fq, fv = [], []
    for i in range(B):
        qc, vc = run(*(np.asarray(r[x][idx[i]], np.float64) for x in ("qpos", "qvel", "ctrl", "qacc_warmstart")),
                     float(r["friction"][idx[i]]))
        fv.append(np.linalg.norm(v1n[i] - vc) / max(np.linalg.norm(vc), 1e-6))
        fq.append(np.abs(q1n[i] - qc).max())
    res["forward"] = dict(qvel_rel_median=float(np.median(fv)), qvel_rel_p99=float(np.quantile(fv, 0.99)),
                          qpos_abs_median=float(np.median(fq)), qpos_abs_max=float(np.max(fq)))
    agree = np.asarray(fv) < 1e-4
    res["forward"]["agree_share"] = float(agree.mean())
    rel = {k: [] for k in ("qpos", "qvel", "ctrl")}
    for k, name in enumerate(("qpos", "qvel", "ctrl")):
        an = grads[k].astype(np.float64)
        for i in range(B):
            q, v, u = (np.asarray(r[x][idx[i]], np.float64) for x in ("qpos", "qvel", "ctrl"))
            w, fr = np.asarray(r["qacc_warmstart"][idx[i]], np.float64), float(r["friction"][idx[i]])
            dd = rng.standard_normal([m.nq, m.nv, m.nu][k])
            if k == 0:
                dd[3:7] = 0.0
            dd /= np.linalg.norm(dd)
            eps = a.eps
            args_p, args_m = [q.copy(), v.copy(), u.copy()], [q.copy(), v.copy(), u.copy()]
            args_p[k] += eps * dd
            args_m[k] -= eps * dd
            qp, vp = run(*args_p, w, fr)
            qm, vm = run(*args_m, w, fr)
            fd = (yq[i] @ (qp - qm) + yv[i] @ (vp - vm)) / (2 * eps)
            g = an[i] @ dd
            rel[name].append(abs(fd - g) / max(abs(fd), abs(g), 1e-12))
        x = np.asarray(rel[name])
        nan = float("nan")
        res[name] = dict(rel_median=float(np.median(x)), rel_p90=float(np.quantile(x, 0.9)),
                         rel_p99=float(np.quantile(x, 0.99)),
                         agree_rel_median=float(np.median(x[agree])) if agree.any() else nan,
                         agree_rel_p90=float(np.quantile(x[agree], 0.9)) if agree.any() else nan)
    if not agree.any():
        print(f"grad64: no world's forward agrees with MuJoCo C (relative qvel difference < 1e-4 in none of {B}): "
              "the agree_rel_* entries are NaN", file=sys.stderr)
    print(json.dumps(res, indent=1))


def cmd_adjdebug(a):
    """The debug adjoint (-DDK_ADJ_DEBUG; nablaengine.checks' binary) on --nworld reference states: its gradients
    bitwise the production build's, and the built-in residual |r(a*)| / max(|tau|, |M a*|) per world and substep
    (r = ID + armature a* - tau - J'f of the replayed rows: frictionloss, limits and contacts), at the float32
    rounding level when the adjoint's replay of the forward is complete."""
    import tempfile
    from nablaengine import checks
    from nablaengine.g1_engine import adjoint, tape
    m, meta = load(a.model)
    r = np.load(a.ref)
    nw, nsub = min(a.nworld, len(r["qpos"])), meta["substeps_per_control_step"]
    ec = adjoint.engine_constants
    adjoint.engine_constants = lambda m_, fp, terrain=None, com_body=None: ec(m_, fp, terrain,
                                                                             com_body or meta["com_body"])
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        arrs = dict(qpos=r["qpos"][:nw], qvel=r["qvel"][:nw], warm=r["qacc_warmstart"][:nw], ctrl=r["ctrl"][:nw],
                    fric=r["friction"][:nw], com=r["com"][:nw])
        for c in range(3):
            arrs[f"cot{c}_qpos"] = rng.standard_normal((nw, m.nq))
            arrs[f"cot{c}_qvel"] = rng.standard_normal((nw, m.nv))
        checks.write_inputs(work, arrs)
        mt = dict(foot_pairs=meta["foot_pairs"])
        prod = checks.build(m, mt, work, "prod", [])
        dbgb = checks.build(m, mt, work, "dbg", ["DK_ADJ_DEBUG"])
        _, out = checks.run(prod, work, m, nw, a.iterations, nsub, 3, 0, False, a.timeout)
        _, outd = checks.run(dbgb, work, m, nw, a.iterations, nsub, 3, 0, True, a.timeout)
    names = [f"grad{c}_{n}" for c in range(3) for n in ("qpos", "qvel", "ctrl")]
    d = outd["dbg0"]
    v = (d["resid_norm"] / np.maximum(np.maximum(d["tau_norm"], d["ma_norm"]), 1e-30)).ravel()
    rec = tape.decode(out["tape"], m)
    res = dict(model=a.model, nbody=int(m.nbody), nworld=nw, substeps=nsub, iterations=a.iterations,
               records_with_rows=float(((rec["ncon"] > 0) | (rec["nefc"] > 0)).mean()),
               debug_build_gradients_bitwise_equal=all(np.array_equal(out[n], outd[n]) for n in names),
               resid_vs_max_tau_Ma=dict(median=float(np.median(v)), p99=float(np.quantile(v, 0.99)), max=float(v.max())),
               adjoint_flagged_worlds=int((out["adj_info0"] != 0).sum()))
    print(json.dumps(res, indent=1))


def cmd_time(a):
    """Time per control step at --nworld worlds (the reference's states tiled; --g1: the G1 fixture's corpus): the
    forward and the forward with the backward, CUDA events, median of --reps."""
    import torch
    from nablaengine import fixtures, torch_ffi
    if a.g1:
        z = fixtures.load_corpus("mf_s0")
        r = {k: z[k] for k in ("qpos", "qvel", "ctrl", "qacc_warmstart", "friction", "com")}
        sub = 4
        torch_ffi.init()
        name = "g1"
    else:
        m, meta = load(a.model)
        r = dict(np.load(a.ref))
        sub = meta["substeps_per_control_step"]
        torch_ffi.init(model=m, foot_pairs=meta["foot_pairs"], com_body=meta["com_body"])
        name = meta["robot"]
    idx = np.arange(a.nworld) % len(r["qpos"])
    t = lambda k: torch.from_numpy(np.ascontiguousarray(r[k][idx], np.float32)).cuda()
    qpos, qvel, ctrl, warm, fr, com = t("qpos"), t("qvel"), t("ctrl"), t("qacc_warmstart"), t("friction"), t("com")
    u = ctrl.clone().requires_grad_()

    def fwd():
        torch_ffi.forward(qpos, qvel, warm, ctrl, fr, com, iterations=a.iterations, substeps=sub)

    def both():
        q1, v1, _ = torch_ffi.step(qpos, qvel, warm, u, fr, com, iterations=a.iterations, substeps=sub)
        (q1.sum() + v1.sum()).backward()

    res = dict(robot=name, nworld=a.nworld, substeps=sub, iterations=a.iterations, gpu=torch.cuda.get_device_name())
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

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("corpus")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nworld", type=int, default=512)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--per-world", type=int, default=8)
    p.add_argument("--tilt", type=float, default=0.6)
    p.add_argument("--drift", type=float, default=0.05)
    p.add_argument("--span", type=float, default=0.6)
    p.add_argument("--reset", type=float, default=0.5, help="restart worlds whose base is below this x initial height")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=10)
    p = sub.add_parser("ref")
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=10)
    p = sub.add_parser("check")
    p.add_argument("--api", choices=("torch", "jax"), default="torch", help="the engine interface")
    p.add_argument("--model", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--dump", default=None)
    p = sub.add_parser("grad")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", default=None)
    p.add_argument("--g1", action="store_true")
    p.add_argument("--nworld", type=int, default=2048)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--eps-q", type=float, default=1e-4)
    p.add_argument("--eps-v", type=float, default=1e-3)
    p.add_argument("--eps-u", type=float, default=1e-3)
    p = sub.add_parser("grad64")
    p.add_argument("--api", choices=("torch", "jax"), default="torch", help="the engine interface")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", default=None)
    p.add_argument("--g1", action="store_true")
    p.add_argument("--nworld", type=int, default=256)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--clamp-kv", type=int, default=1, help="the gradient mode (1, normal 0: the exact derivative)")
    p.add_argument("--normal", type=int, default=0)
    p = sub.add_parser("time")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", default=None)
    p.add_argument("--g1", action="store_true")
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--reps", type=int, default=30)
    p = sub.add_parser("adjdebug")
    p.add_argument("--model", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--nworld", type=int, default=512)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    {"corpus": cmd_corpus, "ref": cmd_ref, "check": cmd_check, "grad": cmd_grad, "grad64": cmd_grad64,
     "time": cmd_time, "adjdebug": cmd_adjdebug}[a.cmd](a)


if __name__ == "__main__":
    main()

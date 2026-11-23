"""Per-world physics parameters (nablaengine.params): the engine with random per-world foot solref / solimp and dof
damping against MuJoCo Warp with the same values as batched model fields, and its gradients against float64 MuJoCo C.

  ref     (MuJoCo Warp env): one control step of --nworld corpus states, each world with its own random parameters
          (foot pairs' solref: time constant U(0.008, 0.04), damping ratio U(0.8, 1.2), or with --direct MuJoCo's direct
          form (-stiffness U(500, 20000), -damping U(40, 300)); solimp dmin U(0.8, 0.95), dmax
          U(0.95, 0.99), width U(0.001, 0.02), midpoint 0.5; damping U(0, 3) on the hinges) and the corpus's foot
          friction (pair_friction); writes inputs, parameters and outputs to OUT.npz
  check   (PyTorch): the engine (torch_ffi built with the parameters) on REF's inputs: qvel / qpos and contact counts;
          --defaults: every world with the model's own values instead, its outputs written (--dump) for a bitwise
          comparison with --plain (the build without parameters) by `compare`
  grad64  (PyTorch + MuJoCo): the VJP against float64 central differences of MuJoCo C with each world's parameters
  pgrad64 (PyTorch + MuJoCo): the parameters' own cotangents (dof_damping, dof_armature, foot_solref / solimp, torso_mass /
          torso_inertia, friction; foot_torsional with ref --condim4) against the same differences
  ref --condim4 T: the frictional pairs as condim 4 (torsional friction T) and per-world foot_torsional U(0.001, 0.1);
          check / grad64 / pgrad64 / time read it from the reference
  compare A.npz B.npz: bitwise equality of two dumps
  time    (PyTorch): time per control step (CUDA events, median of --reps) of the forward and of the forward with the
          backward at --nworld worlds, REF's inputs and parameters tiled; --plain: the build without parameters

The model: the fixture's G1 (default) or --model, an exported robot (scripts/robots/export_robot.py MODEL.xml / .json).

  python scripts/params_check.py ref --out runs/params/ref.npz
  python scripts/params_check.py check --ref runs/params/ref.npz
  python scripts/params_check.py check --ref runs/params/ref.npz --defaults --dump runs/params/def.npz
  python scripts/params_check.py check --ref runs/params/ref.npz --plain --dump runs/params/plain.npz
  python scripts/params_check.py compare runs/params/def.npz runs/params/plain.npz
  python scripts/params_check.py grad64 --ref runs/params/ref.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FIELDS = ("foot_solref", "foot_solimp", "dof_damping", "dof_armature", "torso_mass", "torso_inertia")


def load(model, condim4=None):
    """(MjModel, foot pairs, COM body, substeps, corpus dict or None); condim4: the frictional pairs set to condim 4 with
    that torsional friction (a reference made with ref --condim4; the build then has foot_torsional)."""
    import mujoco
    if model is None:
        from nablaengine import fixtures
        m, mt = fixtures.load_model()
        z = fixtures.load_corpus("mf_s0")
        out = m, list(mt["foot_pairs"]), mt.get("torso_body", "torso_link"), 4, z
    else:
        meta = json.loads(Path(model).with_suffix(".json").read_text())
        m = mujoco.MjModel.from_xml_path(str(Path(model).with_suffix(".xml")))
        out = m, meta["foot_pairs"], meta["com_body"], meta["substeps_per_control_step"], None
    if condim4 is not None:
        for p in range(m.npair):
            if m.pair_dim[p] >= 3:
                m.pair_dim[p] = 4
                m.pair_friction[p, 2] = condim4
    return out


def ref_condim4(r):
    """The torsional friction a reference was made with (ref --condim4), else None."""
    return float(r["condim4"]) if "condim4" in r.files else None


def fields(c4):
    """The build's parameter fields: FIELDS, and foot_torsional for a condim-4 reference."""
    return FIELDS + (("foot_torsional",) if c4 is not None else ())


def body_id(m, name):
    import mujoco
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def draw(m, B, seed, cb, direct=False, torsional=False):
    """Random per-world parameters {field: (B, width)} (cb: the COM-offset body's id). direct: foot_solref in MuJoCo's
    direct form (-stiffness, -damping): stiffness U(500, 20000), damping U(40, 300) (the standard draw's range)."""
    rng = np.random.default_rng(seed)
    damp = np.zeros((B, m.nv))
    damp[:, 6:] = rng.uniform(0.0, 3.0, (B, m.nv - 6))
    arm = np.tile(np.asarray(m.dof_armature, np.float64), (B, 1)) * rng.uniform(0.5, 1.5, (B, m.nv))
    out = dict(dof_armature=arm, foot_solref=np.stack([rng.uniform(0.008, 0.04, B), rng.uniform(0.8, 1.2, B)], 1),
                foot_solimp=np.stack([rng.uniform(0.8, 0.95, B), rng.uniform(0.95, 0.99, B), rng.uniform(0.001, 0.02, B),
                                      np.full(B, 0.5)], 1),
                dof_damping=damp,
                torso_mass=(m.body_mass[cb] * rng.uniform(0.8, 1.2, B))[:, None],
                torso_inertia=m.body_inertia[cb] * rng.uniform(0.7, 1.3, (B, 3)))
    if direct:
        out["foot_solref"] = -np.stack([rng.uniform(500.0, 20000.0, B), rng.uniform(40.0, 300.0, B)], 1)
    if torsional:
        out["foot_torsional"] = rng.uniform(0.001, 0.1, B)[:, None]
    return out


def world_model(m, fp, r, i, cb):
    """m's arrays set to world i's parameters and foot friction (MuJoCo C, in place; cb: the COM-offset body's id);
    returns the originals."""
    keep = (m.pair_solref.copy(), m.pair_solimp.copy(), m.dof_damping.copy(), m.pair_friction.copy(),
            m.dof_armature.copy(), m.body_mass.copy(), m.body_inertia.copy())
    for p in fp:
        m.pair_solref[p] = r["foot_solref"][i]
        m.pair_solimp[p, :4] = r["foot_solimp"][i]
        m.pair_friction[p, :2] = r["friction"][i]
        if "foot_torsional" in r:
            m.pair_friction[p, 2] = np.asarray(r["foot_torsional"][i]).ravel()[0]
    m.dof_damping[:] = r["dof_damping"][i]
    m.dof_armature[:] = r["dof_armature"][i]
    m.body_mass[cb] = np.asarray(r["torso_mass"][i]).ravel()[0]
    m.body_inertia[cb] = r["torso_inertia"][i]
    return keep


def restore(m, keep):
    (m.pair_solref[:], m.pair_solimp[:], m.dof_damping[:], m.pair_friction[:], m.dof_armature[:], m.body_mass[:],
     m.body_inertia[:]) = keep


def cmd_ref(a):
    import warp as wp
    import mujoco_warp as mjw
    import mujoco
    m, fp, com_body, sub, z = load(a.model, a.condim4)
    if z is None:
        z = dict(np.load(a.corpus))
    m.opt.iterations = a.iterations
    B = min(a.nworld, len(z["qpos"]))
    cb = body_id(m, com_body)
    par = draw(m, B, a.seed, cb, a.direct, a.condim4 is not None)
    fr = np.asarray(z["friction"][:B], np.float64) if "friction" in z else np.full(B, m.pair_friction[fp[0], 0])
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=64, njmax=256)
    sref = np.tile(m.pair_solref, (B, 1, 1)).astype(np.float32)
    simp = np.tile(m.pair_solimp, (B, 1, 1)).astype(np.float32)
    pfr = np.tile(m.pair_friction, (B, 1, 1)).astype(np.float32)
    for p in fp:
        sref[:, p] = par["foot_solref"]
        simp[:, p, :4] = par["foot_solimp"]
        pfr[:, p, :2] = fr[:, None]
        if a.condim4 is not None:
            pfr[:, p, 2] = par["foot_torsional"][:, 0]
    mw.pair_solref = wp.array(sref, dtype=wp.vec2)
    mw.pair_solimp = wp.array(simp, dtype=mw.pair_solimp.dtype)
    mw.pair_friction = wp.array(pfr, dtype=mw.pair_friction.dtype)
    mw.dof_damping = wp.array(par["dof_damping"].astype(np.float32), dtype=float)
    mw.dof_armature = wp.array(par["dof_armature"].astype(np.float32), dtype=float)
    bm = np.tile(m.body_mass, (B, 1)).astype(np.float32)
    bi = np.tile(m.body_inertia, (B, 1, 1)).astype(np.float32)
    bm[:, cb] = par["torso_mass"][:, 0]
    bi[:, cb] = par["torso_inertia"]
    mw.body_mass = wp.array(bm, dtype=float)
    mw.body_inertia = wp.array(bi, dtype=wp.vec3)
    com = np.zeros((B, 3), np.float32)
    for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
        wp.copy(getattr(dw, k), wp.array(np.ascontiguousarray(z[k][:B], np.float32)))
    for i in range(sub):
        mjw.step(mw, dw)
    nacon = int(dw.nacon.numpy()[0])
    ncon = np.bincount(dw.contact.worldid.numpy()[:nacon], minlength=B)
    np.savez(a.out, qpos=z["qpos"][:B], qvel=z["qvel"][:B], qacc_warmstart=z["qacc_warmstart"][:B], ctrl=z["ctrl"][:B],
             friction=fr, com=com, out_qpos=dw.qpos.numpy(), out_qvel=dw.qvel.numpy(), ncon=ncon, substeps=sub,
             iterations=a.iterations, **par, **({"condim4": a.condim4} if a.condim4 is not None else {}))
    print(f"ref: {B} worlds, mean contacts {ncon.mean():.2f}; {a.out}")


def engine(a, m, fp, com_body, plain=False, flds=FIELDS):
    from nablaengine import torch_ffi
    from nablaengine.params import PhysicsParams
    P = PhysicsParams(flds)
    kw = dict(model=m, foot_pairs=fp, com_body=com_body)
    torch_ffi.init(**kw) if plain else torch_ffi.init(**kw, params=P)
    return P


def inputs(a, r, m, fp, P, plain, defaults, com_body=None):
    import torch
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32)).cuda()
    B = len(r["qpos"])
    com = np.asarray(r["com"], np.float32)
    if not plain:
        vals = P.defaults(m, fp, com_body) if defaults else {f: r[f] for f in P.fields}
        com = np.concatenate([com, np.broadcast_to(P.pack(m.nv, **vals), (B, P.width(m.nv)))], 1)
    return t, t(com)


def cmd_check(a):
    r = np.load(a.ref)
    c4 = ref_condim4(r)
    m, fp, com_body, sub, _ = load(a.model, c4)
    P = engine(a, m, fp, com_body, a.plain, fields(c4))
    from nablaengine import torch_ffi
    t, com = inputs(a, r, m, fp, P, a.plain, a.defaults, com_body)
    q1, v1, _, info = torch_ffi.forward(t(r["qpos"]), t(r["qvel"]), t(r["qacc_warmstart"]), t(r["ctrl"]),
                                        t(r["friction"]), com, iterations=int(r["iterations"]), substeps=sub)
    q1, v1, info = q1.cpu().numpy(), v1.cpu().numpy(), info.cpu().numpy()
    if a.dump:
        rng = np.random.default_rng(2)
        xs = [t(r["qpos"]).requires_grad_(), t(r["qvel"]).requires_grad_(), t(r["ctrl"]).requires_grad_()]
        qs, vs, _ = torch_ffi.step(xs[0], xs[1], t(r["qacc_warmstart"]), xs[2], t(r["friction"]), com,
                                   iterations=int(r["iterations"]), substeps=sub)
        ((t(rng.standard_normal(qs.shape)) * qs).sum() + (t(rng.standard_normal(vs.shape)) * vs).sum()).backward()
        np.savez(a.dump, q1=q1, v1=v1, info=info, **{f"g_{n}": x.grad.cpu().numpy()
                                                      for n, x in zip(("qpos", "qvel", "ctrl"), xs)})
    res = dict(nworld=len(q1), plain=a.plain, defaults=a.defaults, flags=int((info[:, 3] != 0).sum()))
    if not (a.plain or a.defaults):
        ev = np.linalg.norm(v1 - r["out_qvel"], axis=1) / np.maximum(np.linalg.norm(r["out_qvel"], axis=1), 1e-6)
        res.update(qvel_rel_median=float(np.median(ev)), qvel_rel_p99=float(np.quantile(ev, 0.99)),
                   qvel_rel_max=float(ev.max()), qpos_abs_max=float(np.abs(q1 - r["out_qpos"]).max()),
                   ncon_equal=float((info[:, 1] == r["ncon"]).mean()))
    print(json.dumps(res, indent=1))


def cmd_time(a):
    import torch
    from nablaengine import torch_ffi
    r = dict(np.load(a.ref))
    c4 = float(r["condim4"]) if "condim4" in r else None
    m, fp, com_body, sub, _ = load(a.model, c4)
    idx = np.arange(a.nworld) % len(r["qpos"])
    for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com", *fields(c4)):
        r[k] = np.asarray(r[k])[idx]
    P = engine(a, m, fp, com_body, a.plain, fields(c4))
    t, com = inputs(a, r, m, fp, P, a.plain, False, com_body)
    qpos, qvel, warm, ctrl, fr = t(r["qpos"]), t(r["qvel"]), t(r["qacc_warmstart"]), t(r["ctrl"]), t(r["friction"])
    u = ctrl.clone().requires_grad_()
    it = int(r["iterations"])

    def fwd():
        torch_ffi.forward(qpos, qvel, warm, ctrl, fr, com, iterations=it, substeps=sub)

    def both():
        q1, v1, _ = torch_ffi.step(qpos, qvel, warm, u, fr, com, iterations=it, substeps=sub)
        (q1.sum() + v1.sum()).backward()

    res = dict(plain=a.plain, nworld=a.nworld, iterations=it, substeps=sub, gpu=torch.cuda.get_device_name())
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
    same = {k: bool(np.array_equal(x[k].view(np.uint32) if x[k].dtype == np.float32 else x[k],
                                   y[k].view(np.uint32) if y[k].dtype == np.float32 else y[k])) for k in x.files}
    print(json.dumps(dict(bitwise_equal=same, all=all(same.values()))))
    sys.exit(0 if all(same.values()) else 1)


def cmd_grad64(a):
    import mujoco
    import torch
    from nablaengine import torch_ffi
    r = np.load(a.ref)
    c4 = ref_condim4(r)
    m, fp, com_body, sub, _ = load(a.model, c4)
    P = engine(a, m, fp, com_body, False, fields(c4))
    B = min(a.nworld, len(r["qpos"]))
    rs = {k: r[k][:B] for k in r.files if r[k].ndim and len(r[k]) == len(r["qpos"])}
    t, com = inputs(a, rs, m, fp, P, False, False)
    rng = np.random.default_rng(1)
    xs = [t(rs["qpos"]).requires_grad_(), t(rs["qvel"]).requires_grad_(), t(rs["ctrl"]).requires_grad_()]
    yq, yv = rng.standard_normal((B, m.nq)), rng.standard_normal((B, m.nv))
    q1, v1, _ = torch_ffi.step(xs[0], xs[1], t(rs["qacc_warmstart"]), xs[2], t(rs["friction"]), com,
                               iterations=a.iterations, substeps=sub)
    ((t(yq) * q1).sum() + (t(yv) * v1).sum()).backward()
    m.opt.iterations, m.opt.tolerance, m.opt.ls_tolerance = a.iterations, 1e-12, 1e-12
    d = mujoco.MjData(m)

    def run(i, q, v, u, w):
        keep = world_model(m, fp, rs, i, body_id(m, com_body))
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q, v, u, w
        for _ in range(sub):
            mujoco.mj_step(m, d)
        restore(m, keep)
        return d.qpos.copy(), d.qvel.copy()

    f64 = lambda k, i: np.asarray(rs[k][i], np.float64)
    v1n = v1.detach().cpu().numpy().astype(np.float64)
    fv = []
    for i in range(B):
        _, vc = run(i, *(f64(k, i) for k in ("qpos", "qvel", "ctrl", "qacc_warmstart")))
        fv.append(np.linalg.norm(v1n[i] - vc) / max(np.linalg.norm(vc), 1e-6))
    agree = np.asarray(fv) < 1e-4
    res = dict(nworld=B, iterations=a.iterations, forward_agree_share=float(agree.mean()),
               forward_qvel_rel_median=float(np.median(fv)))
    for k, name in enumerate(("qpos", "qvel", "ctrl")):
        an = xs[k].grad.cpu().numpy().astype(np.float64)
        rel = []
        for i in range(B):
            u = rng.standard_normal(an.shape[1])
            if name == "qpos":
                u[3:7] = 0.0
            u /= np.linalg.norm(u)
            base = [f64(x, i).copy() for x in ("qpos", "qvel", "ctrl", "qacc_warmstart")]
            lp, lm = [x.copy() for x in base], [x.copy() for x in base]
            lp[k] += a.eps * u
            lm[k] -= a.eps * u
            (qp, vp), (qm, vm) = run(i, *lp), run(i, *lm)
            fd = (yq[i] @ (qp - qm) + yv[i] @ (vp - vm)) / (2 * a.eps)
            g = an[i] @ u
            rel.append(abs(fd - g) / max(abs(fd), abs(g), 1e-12))
        x = np.asarray(rel)
        res[name] = dict(agree_rel_median=float(np.median(x[agree])), agree_rel_p90=float(np.quantile(x[agree], 0.9)))
    print(json.dumps(res, indent=1))


def cmd_pgrad64(a):
    """The parameters' cotangents (com's parameter columns, and the friction input's) against float64 central
    differences of MuJoCo C, per world along a random direction of each field (dof_damping, dof_armature: the hinge
    dofs; foot_solref, foot_solimp; friction), with a random output cotangent; on the worlds whose forward agrees with
    MuJoCo C. Relative errors also split by whether the world has a foot contact (stance)."""
    import mujoco
    import torch
    from nablaengine import torch_ffi
    r = np.load(a.ref)
    c4 = ref_condim4(r)
    m, fp, com_body, sub, _ = load(a.model, c4)
    P = engine(a, m, fp, com_body, False, fields(c4))
    B = min(a.nworld, len(r["qpos"]))
    rs = {k: r[k][:B] for k in r.files if r[k].ndim and len(r[k]) == len(r["qpos"])}
    t, com = inputs(a, rs, m, fp, P, False, False)
    com = com.clone().requires_grad_()
    fr_t = t(rs["friction"]).requires_grad_()
    rng = np.random.default_rng(3)
    yq, yv = rng.standard_normal((B, m.nq)), rng.standard_normal((B, m.nv))
    q1, v1, _ = torch_ffi.step(t(rs["qpos"]), t(rs["qvel"]), t(rs["qacc_warmstart"]), t(rs["ctrl"]), fr_t,
                               com, iterations=a.iterations, substeps=sub)
    ((t(yq) * q1).sum() + (t(yv) * v1).sum()).backward()
    gp = com.grad.cpu().numpy().astype(np.float64)[:, 3:]
    gfr = fr_t.grad.cpu().numpy().astype(np.float64)
    lay = P.layout(m.nv)
    m.opt.iterations, m.opt.tolerance, m.opt.ls_tolerance = a.iterations, 1e-12, 1e-12
    d = mujoco.MjData(m)
    f64 = lambda k, i: np.asarray(rs[k][i], np.float64)

    def run(i, rr):
        keep = world_model(m, fp, rr, i, body_id(m, com_body))
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = (f64(k, i) for k in ("qpos", "qvel", "ctrl", "qacc_warmstart"))
        for _ in range(sub):
            mujoco.mj_step(m, d)
        restore(m, keep)
        return d.qpos.copy(), d.qvel.copy()

    v1n = v1.detach().cpu().numpy().astype(np.float64)
    agree = np.array([np.linalg.norm(v1n[i] - run(i, rs)[1]) / max(np.linalg.norm(run(i, rs)[1]), 1e-6) < 1e-4
                      for i in range(B)])
    res = dict(nworld=B, iterations=a.iterations, forward_agree_share=float(agree.mean()))
    pfields = ("dof_damping", "dof_armature", "foot_solref", "foot_solimp", "torso_mass", "torso_inertia", "friction")
    for f in pfields + (("foot_torsional",) if "foot_torsional" in lay else ()):
        o, w = lay[f] if f != "friction" else (0, 1)
        rel = []
        for i in range(B):
            u = np.zeros(w)
            if f.startswith("dof_"):
                u[6:] = rng.standard_normal(w - 6)
            else:
                u[:] = rng.standard_normal(w)
            u /= np.linalg.norm(u)
            x0 = np.atleast_1d(rs[f][i]).astype(np.float64)
            eps = a.eps * (np.abs(x0).min() if not f.startswith("dof_") else max(1.0, float(np.abs(x0).max())))
            rp = {k: v.copy() for k, v in rs.items()}; rm = {k: v.copy() for k, v in rs.items()}
            rp[f][i] = (x0 + eps * u).reshape(np.shape(rs[f][i])); rm[f][i] = (x0 - eps * u).reshape(np.shape(rs[f][i]))
            (qp, vp), (qm, vm) = run(i, rp), run(i, rm)
            fd = (yq[i] @ (qp - qm) + yv[i] @ (vp - vm)) / (2 * eps)
            g = (gp[i, o:o + w] @ u) if f != "friction" else gfr[i] * u[0]
            rel.append(abs(fd - g) / max(abs(fd), abs(g), 1e-12))
        x = np.asarray(rel)
        res[f] = dict(agree_rel_median=float(np.median(x[agree])), agree_rel_p90=float(np.quantile(x[agree], 0.9)),
                      rel_median=float(np.median(x)))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ref")
    p.add_argument("--model", default=None)
    p.add_argument("--corpus", default=None, help="with --model: a robot_check corpus")
    p.add_argument("--out", required=True)
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--direct", action="store_true", help="foot_solref in the direct form (-stiffness, -damping)")
    p.add_argument("--condim4", type=float, default=None,
                   help="the frictional pairs as condim 4 with this torsional friction, and per-world foot_torsional")
    p = sub.add_parser("check")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", required=True)
    p.add_argument("--defaults", action="store_true")
    p.add_argument("--plain", action="store_true")
    p.add_argument("--dump", default=None)
    p = sub.add_parser("time")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", required=True)
    p.add_argument("--plain", action="store_true")
    p.add_argument("--nworld", type=int, default=4096)
    p.add_argument("--reps", type=int, default=30)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p = sub.add_parser("grad64")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", required=True)
    p.add_argument("--nworld", type=int, default=256)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-6)
    p = sub.add_parser("pgrad64")
    p.add_argument("--model", default=None)
    p.add_argument("--ref", required=True)
    p.add_argument("--nworld", type=int, default=128)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-5)
    a = ap.parse_args()
    {"ref": cmd_ref, "check": cmd_check, "compare": cmd_compare, "grad64": cmd_grad64, "pgrad64": cmd_pgrad64, "time": cmd_time}[a.cmd](a)


if __name__ == "__main__":
    main()

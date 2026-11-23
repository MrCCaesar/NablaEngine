"""Joint equality constraints in the forest engine (forest_engine.equality): forward against MuJoCo Warp, gradients
against float64 MuJoCo C central differences, on a model with joint equalities (scripts/eq/models, or a Menagerie scene)."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

J = mujoco.mjtJoint


def load(path, obj, condim6_to=None):
    from nablaengine.envs import convert
    p = Path(path).resolve()
    xml = p.read_text()
    assets = {str(q.relative_to(p.parent)): q.read_bytes() for q in p.parent.rglob("*.xml")}
    cwd = os.getcwd()
    os.chdir(p.parent)
    try:
        if obj:
            xml = add_objects(xml, assets)
        cv = convert.to_engine(xml, assets, condim6_to=condim6_to)
    finally:
        os.chdir(cwd)
    return cv, p.parent


def home(m):
    k = next((i for i in range(m.nkey) if m.key(i).name == "home"), 0 if m.nkey else None)
    return (m.key_qpos[k].copy(), m.key_ctrl[k].copy()) if k is not None else (m.qpos0.copy(), np.zeros(m.nu))


def finger_pairs(m):
    """[(eq, j1, j2)] of the joint equalities between two slides."""
    out = []
    for e in range(m.neq):
        j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT) and j2 >= 0 and all(
                int(m.jnt_type[j]) == J.mjJNT_SLIDE for j in (j1, j2)):
            out.append((e, j1, j2))
    return out


def _pair_pose(m, q, e, j1, j2, x):
    q = q.copy()
    q[m.jnt_qposadr[j2]] = x
    q[m.jnt_qposadr[j1]] = np.polynomial.Polynomial(m.eq_data[e, :5])(x)
    return q


def open_closed(m, q, e, j1, j2):
    """(open, closed) values of joint2 of finger pair e: the ends of its range with the larger / smaller distance of
    the two finger bodies (joint1 by the equality)."""
    d = mujoco.MjData(m)
    gaps = []
    for x in m.jnt_range[j2]:
        d.qpos[:] = _pair_pose(m, q, e, j1, j2, x)
        mujoco.mj_kinematics(m, d)
        gaps.append(np.linalg.norm(d.xpos[m.jnt_bodyid[j1]] - d.xpos[m.jnt_bodyid[j2]]))
    lo, hi = m.jnt_range[j2]
    return (lo, hi) if gaps[0] > gaps[1] else (hi, lo)


def open_pose(m, q):
    """q with every finger pair open (joint1 by the equality)."""
    for e, j1, j2 in finger_pairs(m):
        q = _pair_pose(m, q, e, j1, j2, open_closed(m, q, e, j1, j2)[0])
    return q


def add_objects(xml, assets):
    m = mujoco.MjModel.from_xml_string(xml, assets)
    d = mujoco.MjData(m)
    d.qpos[:] = open_pose(m, home(m)[0])
    mujoco.mj_forward(m, d)
    bodies = []
    for e, j1, j2 in finger_pairs(m):
        b1, b2 = int(m.jnt_bodyid[j1]), int(m.jnt_bodyid[j2])
        c1, c2 = d.subtree_com[b1], d.subtree_com[b2]
        gap = float(np.linalg.norm(c1 - c2))
        mid = 0.5 * (c1 + c2)
        u = (c1 - c2) / max(gap, 1e-9)
        z = np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.9 else np.cross(u, [0, 1.0, 0])
        z /= np.linalg.norm(z)
        y = np.cross(z, u)
        R = np.stack([u, y, z], 1)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.ravel())
        hs = [0.35 * gap, 0.015, 0.015]
        f = lambda x: " ".join(repr(float(t)) for t in x)
        bodies.append(f'<body name="eq_obj{e}" pos="{f(mid)}" quat="{f(quat)}"><freejoint name="eq_obj{e}"/>'
                      f'<geom name="eq_obj{e}" type="box" size="{f(hs)}" mass="0.05" contype="1" conaffinity="1" '
                      f'condim="3"/></body>')
    i = xml.rindex("</worldbody>")
    return xml[:i] + "".join(bodies) + xml[i:]


def corpus(m, n_roll, per_roll, seed, t_close=0.1, t_min=0.2, t_end=1.0, hold=0.05):
    """(qpos, qvel, ctrl) float32 states from MuJoCo C rollouts (module docstring)."""
    rng = np.random.default_rng(seed)
    q0, u0 = home(m)
    q0 = open_pose(m, q0)
    oc = {}
    for e, j1, j2 in finger_pairs(m):
        o, c = open_closed(m, q0, e, j1, j2)
        P = np.polynomial.Polynomial(m.eq_data[e, :5])
        c = o + 0.9 * (c - o)
        oc[j2], oc[j1] = (o, c), (P(o), P(c))
    lo, hi = m.actuator_ctrlrange.T.copy()
    lim = m.actuator_ctrllimited.astype(bool)
    lo, hi = np.where(lim, lo, u0 - 0.5), np.where(lim, hi, u0 + 0.5)
    grip = []
    for a in range(m.nu):
        j = int(m.actuator_trnid[a, 0])
        if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT) and j in oc:
            grip.append((a, *oc[j]))
        elif int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_TENDON):
            w = range(int(m.tendon_adr[j]), int(m.tendon_adr[j]) + int(m.tendon_num[j]))
            js = [(int(m.wrap_objid[k]), float(m.wrap_prm[k])) for k in w]
            kp, kq = float(m.actuator_gainprm[a, 0]), -float(m.actuator_biasprm[a, 1])
            if all(int(m.wrap_type[k]) == int(mujoco.mjtWrap.mjWRAP_JOINT) for k in w) and js and \
                    all(jj in oc for jj, _ in js) and kp > 0 and kq > 0:
                grip.append((a, *(kq / kp * sum(c * oc[jj][s] for jj, c in js) for s in (0, 1))))
    d = mujoco.MjData(m)
    Q, V, U = [], [], []
    nsteps, every = int(t_end / m.opt.timestep), int(hold / m.opt.timestep)
    for r in range(n_roll):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q0
        u = u0.copy()
        for g, o, _ in grip:
            u[g] = o
        when = set(rng.choice(np.arange(int(t_min / m.opt.timestep), nsteps), per_roll, replace=False).tolist())
        for t in range(nsteps):
            if t % every == 0:
                u = np.clip(u0 + rng.normal(0, 0.15, m.nu) * (hi - lo), lo + 0.01 * (hi - lo), hi - 0.01 * (hi - lo))
                for g, o, c in grip:
                    u[g] = o if t * m.opt.timestep < t_close else c
            d.ctrl[:] = u
            if t in when:
                Q.append(d.qpos.copy()); V.append(d.qvel.copy()); U.append(u.copy())
            mujoco.mj_step(m, d)
    return (np.asarray(Q, np.float32), np.asarray(V, np.float32), np.asarray(U, np.float32))


def eq_residual(m, q):
    """|pos| of every active joint equality, (B, neq) (float64, the definition)."""
    out = np.zeros((len(q), m.neq))
    for e in range(m.neq):
        j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        x1 = q[:, m.jnt_qposadr[j1]].astype(np.float64)
        rhs = (np.polynomial.Polynomial(m.eq_data[e, :5])(q[:, m.jnt_qposadr[j2]].astype(np.float64)) if j2 >= 0
               else m.eq_data[e, 0])
        out[:, e] = np.abs(x1 - rhs)
    return out


def clamped_dofs(m, q, v, u):
    """(nv,) bool: the dofs of actuators whose force is at its forcerange (MuJoCo C at the state)."""
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:] = q, v, u
    mujoco.mj_forward(m, d)
    out = np.zeros(m.nv, bool)
    for a in range(m.nu):
        if m.actuator_forcelimited[a] and int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            lo, hi = m.actuator_forcerange[a]
            if d.actuator_force[a] <= lo + 1e-9 or d.actuator_force[a] >= hi - 1e-9:
                out[int(m.jnt_dofadr[m.actuator_trnid[a, 0]])] = True
    return out


def robot_dofs(m):
    """The dofs not of an added object (eq_obj* bodies)."""
    return np.array([i for i in range(m.nv) if not m.body(int(m.dof_bodyid[i])).name.startswith("eq_obj")], int)


def robot_qpos(m):
    return np.array([int(m.jnt_qposadr[j]) + k for j in range(m.njnt)
                     if not m.body(int(m.jnt_bodyid[j])).name.startswith("eq_obj")
                     for k in range({0: 7, 1: 4, 2: 1, 3: 1}[int(m.jnt_type[j])])], int)


def ncon_c(m, q):
    """MuJoCo C's contact count at each qpos (mj_forward)."""
    d = mujoco.MjData(m)
    out = np.zeros(len(q), int)
    for i, qq in enumerate(q):
        d.qpos[:] = qq
        mujoco.mj_forward(m, d)
        out[i] = d.ncon
    return out


def stats(x):
    x = np.asarray(x, np.float64).ravel()
    if not x.size:
        return None
    return dict(median=float(np.median(x)), p90=float(np.percentile(x, 90)), p99=float(np.percentile(x, 99)),
                max=float(x.max()))


def engine_init(m):
    from nablaengine import engines, jax_ffi
    engines.prefer_forest(m)
    return jax_ffi.init(model=m, foot_pairs=[])


def engine_step(m, info, q, v, u, substeps):
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    B = len(q)
    cw = int(info.get("com_words", 3))
    out = jax_ffi.step(jnp.asarray(q), jnp.asarray(v), jnp.zeros((B, m.nv), jnp.float32), jnp.asarray(u),
                       jnp.ones((B,), jnp.float32), jnp.zeros((B, cw), jnp.float32), substeps=substeps,
                       iterations=int(m.opt.iterations))
    return np.asarray(out[0]), np.asarray(out[1])


def tape_bitwise(m, info, q, v, u, substeps, qe, ve):
    """The forward with the tape (the gradients' forward) gives bitwise step()'s state."""
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    B = len(q)
    cw = int(info.get("com_words", 3))
    o = jax_ffi.forward(jnp.asarray(q), jnp.asarray(v), jnp.zeros((B, m.nv), jnp.float32), jnp.asarray(u),
                        jnp.ones((B,), jnp.float32), jnp.zeros((B, cw), jnp.float32), substeps=substeps, tape=True,
                        iterations=int(m.opt.iterations))
    return bool(np.array_equal(np.asarray(o[0]), qe) and np.array_equal(np.asarray(o[1]), ve))


def row_reference(m, q, v):
    """The equality rows (pos, D, aref) by MuJoCo Warp's definitions in float64, (B, neq, 3)."""
    out = np.full((len(q), max(m.neq, 1), 3), np.nan)
    h = m.opt.timestep
    for e in range(m.neq):
        j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        d1, d2 = int(m.jnt_dofadr[j1]), int(m.jnt_dofadr[j2]) if j2 >= 0 else -1
        c = np.polynomial.Polynomial(m.eq_data[e, :5])
        x1 = q[:, m.jnt_qposadr[j1]].astype(np.float64)
        if j2 >= 0:
            x2 = q[:, m.jnt_qposadr[j2]].astype(np.float64)
            pos, jv = x1 - c(x2), v[:, d1] - c.deriv()(x2) * v[:, d2]
            iw = m.dof_invweight0[d1] + m.dof_invweight0[d2]
        else:
            pos, jv, iw = x1 - m.eq_data[e, 0], v[:, d1].astype(np.float64), m.dof_invweight0[d1]
        sr, si = m.eq_solref[e], m.eq_solimp[e]
        tc, dr = max(sr[0], 2 * h), sr[1]
        dmin, dmax, width, mid, pw = np.clip(si[0], 1e-4, .9999), np.clip(si[1], 1e-4, .9999), si[2], si[3], si[4]
        k, b = 1 / (dmax * dmax * tc * tc * dr * dr), 2 / (dmax * tc)
        x = np.abs(pos) / width
        y = np.where(x <= mid, mid ** (1 - pw) * x ** pw, 1 - (1 - mid) ** (1 - pw) * np.clip(1 - x, 0, None) ** pw)
        imp = np.where(x >= 1, dmax, dmin + y * (dmax - dmin))
        D = 1 / np.maximum(iw * (1 - imp) / imp, mujoco.mjMINVAL)
        out[:, e] = np.stack([pos, D, -k * imp * pos - b * jv], 1)
    return out


def run_mjw(m, q, v, u, substeps, py, work):
    mjb, st, out = work / "m.mjb", work / "states.npz", work / "mjw.npz"
    mujoco.mj_saveModel(m, str(mjb), None)
    np.savez(st, qpos=q, qvel=v, ctrl=u)
    here = Path(__file__).resolve().parent
    env = {k: val for k, val in os.environ.items() if k not in ("PYTHONPATH",)}
    r = subprocess.run([py, str(here / "mjw_step.py"), str(mjb), str(st), str(out), "--substeps", str(substeps)],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return dict(np.load(out)), json.loads(r.stdout.strip().splitlines()[-1])


def fd_grad(m, q, v, u, gq, gv, substeps, eps=1e-6):
    """Central differences of L = gq . qpos' + gv . qvel' (float64 MuJoCo C, tight solver) over the hinge / slide qpos
    entries, qvel and ctrl: (g_qpos (nq; free-joint entries nan), g_qvel, g_ctrl)."""
    mm = mujoco.MjModel.from_binary_path(m) if isinstance(m, str) else m
    mm.opt.tolerance = 1e-12
    mm.opt.ls_tolerance = 1e-12
    mm.opt.iterations = 200
    mm.opt.ls_iterations = 100
    d = mujoco.MjData(mm)

    def L(qq, vv, uu):
        mujoco.mj_resetData(mm, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = qq, vv, uu
        for _ in range(substeps):
            mujoco.mj_step(mm, d)
        return float(gq @ d.qpos + gv @ d.qvel)

    q, v, u = q.astype(np.float64), v.astype(np.float64), u.astype(np.float64)
    scal = [int(mm.jnt_qposadr[j]) for j in range(mm.njnt) if int(mm.jnt_type[j]) in (J.mjJNT_HINGE, J.mjJNT_SLIDE)]
    g_q = np.full(mm.nq, np.nan)
    for i in scal:
        e = np.zeros(mm.nq); e[i] = eps
        g_q[i] = (L(q + e, v, u) - L(q - e, v, u)) / (2 * eps)
    g_v = np.zeros(mm.nv)
    for i in range(mm.nv):
        e = np.zeros(mm.nv); e[i] = eps
        g_v[i] = (L(q, v + e, u) - L(q, v - e, u)) / (2 * eps)
    g_u = np.zeros(mm.nu)
    for i in range(mm.nu):
        e = np.zeros(mm.nu); e[i] = eps
        g_u[i] = (L(q, v, u + e) - L(q, v, u - e)) / (2 * eps)
    return g_q, g_v, g_u


def engine_vjp(m, info, q, v, u, gq, gv, substeps, normal=None):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    B = len(q)
    cw = int(info.get("com_words", 3))
    w, fr, com = jnp.zeros((B, m.nv), jnp.float32), jnp.ones((B,), jnp.float32), jnp.zeros((B, cw), jnp.float32)

    def f(qq, vv, uu):
        o = jax_ffi.step(qq, vv, w, uu, fr, com, substeps=substeps, iterations=int(m.opt.iterations),
                         **({} if normal is None else {"normal": normal}))
        return o[0], o[1]

    _, pull = jax.vjp(f, jnp.asarray(q), jnp.asarray(v), jnp.asarray(u))
    g = pull((jnp.asarray(gq, jnp.float32), jnp.asarray(gv, jnp.float32)))
    return [np.asarray(x) for x in g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mjw-python", default=None, help="a Python with mujoco_warp (forward check); none: skipped")
    ap.add_argument("--object", action="store_true")
    ap.add_argument("--states", type=int, default=512)
    ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--grad", type=int, default=0)
    ap.add_argument("--grad-substeps", type=int, default=1)
    ap.add_argument("--normal", type=int, default=None, help="the adjoint's self-contact normal option (jax_ffi.step "
                    "normal=; default: the interface's, 3: frozen where nu < 1e-3 m; 0: exact)")
    ap.add_argument("--condim6-to", type=int, default=None, choices=(3,),
                    help="convert condim-6 pairs to condim 3 (convert.to_engine(condim6_to=3); default: kept, and the "
                         "forest engine refuses them, capability H17)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strip-eq", action="store_true",
                    help="a control: the corpus of the model with its equalities, then the equalities removed for the "
                         "checks (the same states without the equality rows)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cv, mdir = load(a.model, a.object, a.condim6_to)
    m = cv.model
    q, v, u = corpus(m, a.rollouts, max(1, a.states // a.rollouts), a.seed)
    if a.strip_eq:
        spec = mujoco.MjSpec.from_string(cv.xml)
        for e in list(spec.equalities):
            spec.delete(e)
        cwd = os.getcwd()
        os.chdir(mdir)
        try:
            m = spec.compile()
        finally:
            os.chdir(cwd)
    info = engine_init(m)
    res = dict(model=a.model, object=a.object, strip_eq=a.strip_eq, nq=m.nq, nv=m.nv, nu=m.nu, neq=m.neq, npair=m.npair,
               cone=int(m.opt.cone), integrator=int(m.opt.integrator), timestep=m.opt.timestep, nstates=len(q),
               substeps=a.substeps, engine=info.get("engine"), maxcon=info.get("maxcon"), nefc=info.get("nefc"),
               notes=cv.notes, corpus_eq_residual=stats(eq_residual(m, q)))
    with tempfile.TemporaryDirectory(prefix="eq_check_") as tmp:
        work = Path(tmp)
        qe, ve = engine_step(m, info, q, v, u, a.substeps)
        res["engine_finite"] = bool(np.isfinite(qe).all() and np.isfinite(ve).all())
        res["tape_forward_bitwise"] = tape_bitwise(m, info, q, v, u, a.substeps, qe, ve)
        res["engine_eq_residual"] = stats(eq_residual(m, qe))
        if a.mjw_python:
            cwd = os.getcwd()
            os.chdir(mdir)
            try:
                mj, ver = run_mjw(m, q, v, u, a.substeps, a.mjw_python, work)
            finally:
                os.chdir(cwd)
            res["mjw"] = ver
            dq, dv = np.abs(qe - mj["qpos"]), np.abs(ve - mj["qvel"])
            res["qpos_abs"] = stats(dq.max(1))
            res["qvel_abs"] = stats(dv.max(1))
            rel = (dv / np.maximum(np.abs(mj["qvel"]), 1.0)).max(1)
            res["qvel_rel"] = stats(rel)
            rd = robot_dofs(m)
            if len(rd) < m.nv:
                res["qvel_rel_robot"] = stats((dv / np.maximum(np.abs(mj["qvel"]), 1.0))[:, rd].max(1))
            nc = ncon_c(m, q)
            for key, sel in (("no_contact", nc == 0), ("contact", nc > 0)):
                if sel.any():
                    res[f"qvel_rel_{key}"] = dict(stats(rel[sel]), n=int(sel.sum()))
            res["mjw_eq_residual"] = stats(eq_residual(m, mj["qpos"]))
            res["contacts_mjw"] = dict(mean=float(mj["ncon"].mean()), max=int(mj["ncon"].max()))
            if "eq_rows" in mj:
                ref = row_reference(m, q.astype(np.float64), v.astype(np.float64))
                rows = mj["eq_rows"][:, :m.neq]
                ok = np.isfinite(rows).all(2)
                res["rows_mjw_vs_definition"] = {
                    k: stats(np.abs(rows[..., i] - ref[..., i])[ok] / np.maximum(np.abs(ref[..., i])[ok], 1e-3))
                    for i, k in enumerate(("pos", "D", "aref"))}
                res["rows_mjw_count"] = int(ok.sum())
        if a.grad:
            rng = np.random.default_rng(a.seed + 1)
            idx = rng.choice(len(q), min(a.grad, len(q)), replace=False)
            gq = rng.normal(size=(len(idx), m.nq)).astype(np.float32)
            gv = rng.normal(size=(len(idx), m.nv)).astype(np.float32)
            rd, rq = robot_dofs(m), robot_qpos(m)
            if a.object:
                gq[:, np.setdiff1d(np.arange(m.nq), rq)] = 0
                gv[:, np.setdiff1d(np.arange(m.nv), rd)] = 0
            ge = engine_vjp(m, info, q[idx], v[idx], u[idx], gq, gv, a.grad_substeps, a.normal)
            res["normal"] = a.normal
            errs, comps, nclamp = [], {"qpos": [], "qvel": [], "ctrl": []}, []
            for t, i in enumerate(idx):
                fq, fv, fu = fd_grad(m, q[i], v[i], u[i], gq[t].astype(np.float64), gv[t].astype(np.float64),
                                     a.grad_substeps)
                sel = np.isfinite(fq)
                vs = ~clamped_dofs(m, q[i], v[i], u[i])
                nclamp.append(int((~vs).sum()))
                eg = np.concatenate([ge[0][t][sel], ge[1][t][vs], ge[2][t]])
                fg = np.concatenate([fq[sel], fv[vs], fu])
                errs.append(np.linalg.norm(eg - fg) / max(np.linalg.norm(fg), 1e-12))
                for k, e_, f_ in (("qpos", ge[0][t][sel], fq[sel]), ("qvel", ge[1][t][vs], fv[vs]), ("ctrl", ge[2][t], fu)):
                    comps[k].append(np.linalg.norm(e_ - f_) / max(np.linalg.norm(f_), 1e-12))
            res["grad_rel"] = stats(errs)
            nc = ncon_c(m, q[idx])
            for key, sel in (("no_contact", nc == 0), ("contact", nc > 0)):
                if sel.any():
                    res[f"grad_rel_{key}"] = dict(stats(np.asarray(errs)[sel]), n=int(sel.sum()))
            res["grad_rel_per_state"] = [float(x) for x in errs]
            res["grad_rel_by_input"] = {k: stats(x) for k, x in comps.items()}
            res["grad_states"] = len(idx)
            res["grad_qvel_clamped_dofs_excluded"] = int(sum(nclamp))
            res["grad_substeps"] = a.grad_substeps
    s = json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    print(s)
    if a.out:
        out = Path(a.out)
        if out.exists():
            raise SystemExit(f"{out} exists (not overwritten)")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(s)


if __name__ == "__main__":
    sys.exit(main())

"""General actuators in the forest engine (forest_engine.muscle: MuJoCo's muscles and activation dynamics): unit scenes
against MuJoCo Warp 3.17 and float64 MuJoCo C, exact gradients (state, ctrl and activation) against float64 MuJoCo C
central differences.

Scenes (SCENES; the arm of scripts/forest/models/muscle_arm.xml with a tendon / actuator block, through
nablaengine.envs.convert.to_engine): joint (muscles on hinge / slide joints, gear 1 / 0.5 / 2, one with force -1:
scale / acc0), fixed (muscles on fixed tendons), spatial (muscles on spatial tendons with sphere / cylinder wraps, side
sites, a pulley, tausmooth > 0 on one), actlim (actlimited: a muscle, a filter, an integrator; ctrllimited on some),
early (actearly on a muscle, a filterexact and an integrator), implicit (implicitfast: filter / integrator actuators
with an affine bias kv next to a position actuator). States: MuJoCo C rollouts with random velocities, controls in
[-0.2, 1.2] and their activations (corpus()); the activation input of a state is the rollout's, and with --act-noise
a uniform draw in [-0.1, 1.1] (outside the clamp ranges too). One engine library per process: one scene per run
(--scene), or --all (subprocesses).

  fwd   one control step of 1 and 4 substeps from a zero warm start: qvel relative error per world (|dv| / max(|v|, 1)),
        act' absolute error, and the first substep's actuator force (the tape's, relative to max(|f|, 1)) of the engine
        against MuJoCo Warp (mjw_act_step.py run by --mjw-python) and MuJoCo C; the tape's forward bitwise the plain
        step's (qpos', qvel', act')
  fd    the VJP of one control step (--substeps) in the exact mode (clamp_kv 1, normal 0) with an act' cotangent
        against central differences of MuJoCo C (tight solver, eps 1e-6): relative error per world, overall and by
        block (q, v, ctrl, act)

  PYTHONPATH=src python scripts/forest/muscle_check.py fwd --scene spatial --mjw-python <venv>/bin/python
  PYTHONPATH=src python scripts/forest/muscle_check.py fd --all --out runs/muscle/unit_fd.json

Exit status 1 when a median exceeds its bound (fwd: qvel vs MuJoCo C 1e-4, act' 1e-5; fd: 1e-3).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eq"))
import connect_check as cc
import connect_grad as cg

ARM = REPO / "scripts" / "forest" / "models" / "muscle_arm.xml"

SPATIAL_TENDONS = """<tendon>
  <spatial name="t1"><site site="w0"/><site site="a1"/><site site="b1"/><site site="c1"/><site site="d1"/></spatial>
  <spatial name="t2"><site site="a2"/><geom geom="wcyl" sidesite="side_out"/><site site="b1"/><site site="c1"/></spatial>
  <spatial name="t3"><site site="a1"/><geom geom="wsph"/><site site="b1"/><site site="d1"/></spatial>
  <spatial name="t4"><site site="w1"/><site site="a3"/><pulley divisor="2"/><site site="a3"/><site site="b2"/><site site="d2"/></spatial>
</tendon>"""

SCENES = {
    "joint": """<actuator>
  <muscle name="mh1" joint="h1" force="30" lengthrange="-1.5 1.5" ctrlrange="0 1" ctrllimited="true"/>
  <muscle name="mh2" joint="h2" force="-1" scale="200" lengthrange="-2 2"/>
  <muscle name="mh4" joint="h4" gear="0.5" force="8" lengthrange="-0.5 0.5" lmin="0.4" lmax="1.7" fpmax="1.5" fvmax="1.4"/>
  <muscle name="ms3" joint="s3" gear="2" force="20" lengthrange="-0.1 0.1"/>
</actuator>""",
    "fixed": """<tendon>
  <fixed name="f1"><joint joint="h1" coef="1"/><joint joint="h2" coef="0.5"/></fixed>
  <fixed name="f2"><joint joint="h2" coef="1"/><joint joint="h4" coef="-0.7"/></fixed>
  <fixed name="f3"><joint joint="s3" coef="3"/><joint joint="h4" coef="0.2"/></fixed>
</tendon>
<actuator>
  <muscle name="mf1" tendon="f1" force="40" lengthrange="-2.5 2.5"/>
  <muscle name="mf2" tendon="f2" force="15" lengthrange="-2.7 2.7" ctrlrange="0 1" ctrllimited="true"/>
  <muscle name="mf3" tendon="f3" force="10" lengthrange="-0.35 0.35" timeconst="0.02 0.05"/>
</actuator>""",
    "spatial": SPATIAL_TENDONS + """
<actuator>
  <muscle name="m1" tendon="t1" force="60"/>
  <muscle name="m2" tendon="t2" force="30" tausmooth="0.2"/>
  <muscle name="m3" tendon="t3" force="40" ctrlrange="0 1" ctrllimited="true"/>
  <muscle name="m4" tendon="t4" force="25" gear="1.5"/>
</actuator>""",
    "actlim": SPATIAL_TENDONS + """
<actuator>
  <general name="m1" tendon="t1" dyntype="muscle" gaintype="muscle" biastype="muscle" dynprm="0.01 0.04 0"
           gainprm="0.75 1.05 60 200 0.5 1.6 1.5 1.3 1.2" biasprm="0.75 1.05 60 200 0.5 1.6 1.5 1.3 1.2"
           actlimited="true" actrange="0.05 0.6"/>
  <general name="g2" tendon="t2" dyntype="filter" dynprm="0.03" gainprm="20" actlimited="true" actrange="-0.5 0.5"
           ctrlrange="-1 1" ctrllimited="true"/>
  <general name="g3" joint="h2" dyntype="integrator" gainprm="4" biastype="affine" biasprm="0 -6 -0.3"
           actlimited="true" actrange="-0.6 0.6"/>
  <general name="m3" tendon="t3" dyntype="muscle" gaintype="muscle" biastype="muscle" dynprm="0.02 0.06 0"
           gainprm="0.75 1.05 40 200 0.5 1.6 1.5 1.3 1.2" biasprm="0.75 1.05 40 200 0.5 1.6 1.5 1.3 1.2"
           actlimited="true" actrange="0 1" ctrlrange="0 1" ctrllimited="true"/>
</actuator>""",
    "early": SPATIAL_TENDONS + """
<actuator>
  <general name="m1" tendon="t1" dyntype="muscle" gaintype="muscle" biastype="muscle" dynprm="0.01 0.04 0"
           gainprm="0.75 1.05 60 200 0.5 1.6 1.5 1.3 1.2" biasprm="0.75 1.05 60 200 0.5 1.6 1.5 1.3 1.2" actearly="true"/>
  <general name="g2" tendon="t2" dyntype="filterexact" dynprm="0.02" gainprm="15" actearly="true"/>
  <general name="g3" joint="h4" dyntype="integrator" gainprm="2" biastype="affine" biasprm="0 -3 -0.1"
           actearly="true" actlimited="true" actrange="-0.8 0.8"/>
  <general name="m4" tendon="t4" dyntype="muscle" gaintype="muscle" biastype="muscle" dynprm="0.01 0.04 0.1"
           gainprm="0.75 1.05 25 200 0.5 1.6 1.5 1.3 1.2" biasprm="0.75 1.05 25 200 0.5 1.6 1.5 1.3 1.2" actearly="true"
           gear="1.5"/>
</actuator>""",
    "implicit": SPATIAL_TENDONS + """
<actuator>
  <position name="p1" joint="h1" kp="20" kv="1" ctrlrange="-1 1"/>
  <general name="g2" tendon="t2" dyntype="filter" dynprm="0.03" gainprm="20" biastype="affine" biasprm="0 0 -2"/>
  <general name="g3" joint="h2" dyntype="integrator" gainprm="4" biastype="affine" biasprm="0 -6 -0.3"
           actlimited="true" actrange="-0.6 0.6"/>
  <general name="g4" tendon="t3" dyntype="filterexact" dynprm="0.05" gainprm="10" biastype="affine" biasprm="0 0 -1"
           forcelimited="true" forcerange="-3 2"/>
</actuator>""",
}
IMPLICIT = {"implicit"}


def scene_xml(name):
    base = ARM.read_text()
    if name in IMPLICIT:
        base = base.replace('integrator="Euler"', 'integrator="implicitfast"')
    return base.replace("</mujoco>", SCENES[name] + "\n</mujoco>")


def load_scene(name):
    from nablaengine.envs import convert
    return convert.to_engine(scene_xml(name))


def corpus(m, n, seed, t_min=0.1, every=10, act_noise=False, act_floor=None):
    """(qpos, qvel, ctrl, act) float32: MuJoCo C rollouts with random velocities and controls in [-0.2, 1.2] (new every
    50 steps), sampled every `every` steps after t_min; act_noise: the activations drawn uniformly in [-0.1, 1.1];
    act_floor: the rollouts' activations raised to at least this (a muscle at rest has act exactly 0, a kink of
    MuJoCo's activation dynamics that central differences straddle)."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    Q, V, U, A = [], [], [], []
    while len(Q) < n:
        mujoco.mj_resetData(m, d)
        d.qvel[:] = rng.normal(0, 0.5, m.nv)
        t, k = 0.0, 0
        while t < t_min + 0.4 and len(Q) < n:
            if k % 50 == 0:
                d.ctrl[:] = rng.uniform(-0.2, 1.2, m.nu)
            mujoco.mj_step(m, d)
            t, k = t + m.opt.timestep, k + 1
            if t >= t_min and k % every == 0:
                Q.append(d.qpos.copy()); V.append(d.qvel.copy()); U.append(rng.uniform(-0.2, 1.2, m.nu))
                A.append(rng.uniform(-0.1, 1.1, m.na) if act_noise else d.act.copy())
    f = lambda x: np.asarray(x, np.float32)
    A = f(A)
    if act_floor is not None:
        A = np.maximum(A, np.float32(act_floor))
    return f(Q), f(V), f(U), A


def c_steps(m, q, v, u, a, n):
    """MuJoCo C after n substeps: (qvel, act, actuator_force of the first substep) per world."""
    d = mujoco.MjData(m)
    V, A, F = [], [], []
    for w in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:], d.act[:] = q[w], v[w], u[w], a[w]
        for k in range(n):
            mujoco.mj_step(m, d)
            if k == 0:
                F.append(d.actuator_force.copy())
        V.append(d.qvel.copy())
        A.append(d.act.copy())
    return np.array(V), np.array(A), np.array(F)


def mjw_steps(m, q, v, u, a, n, py):
    with tempfile.TemporaryDirectory() as tmp:
        mujoco.mj_saveModel(m, os.path.join(tmp, "m.mjb"), None)
        np.savez(os.path.join(tmp, "s.npz"), qpos=q, qvel=v, ctrl=u, act=a)
        r = subprocess.run([py, "-I", str(REPO / "scripts" / "forest" / "mjw_act_step.py"), os.path.join(tmp, "m.mjb"),
                            os.path.join(tmp, "s.npz"), os.path.join(tmp, "o.npz"), "--substeps", str(n),
                            "--iterations", str(int(m.opt.iterations))], capture_output=True, text=True,
                           env={k: x for k, x in os.environ.items() if k != "PYTHONPATH"})
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
        z = np.load(os.path.join(tmp, "o.npz"))
        return z["qvel"], z["act"], z["force1"], json.loads(r.stdout.strip().splitlines()[-1])


def engine_init(m):
    from nablaengine import engines, jax_ffi
    engines.prefer_forest(m)
    return jax_ffi.init(model=m, foot_pairs=[])


def _args(m, info, q, v, u):
    import jax.numpy as jnp
    B = len(q)
    cw = int(info.get("com_words", 3))
    return (jnp.asarray(q), jnp.asarray(v), jnp.zeros((B, m.nv), jnp.float32), jnp.asarray(u),
            jnp.ones((B,), jnp.float32), jnp.zeros((B, cw), jnp.float32))


def engine_step(m, info, q, v, u, a, substeps):
    """(qpos', qvel', act', tape) of one control step (forward with the tape) and the plain step's bitwise equality."""
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    args = _args(m, info, q, v, u)
    act = jnp.asarray(a)
    it = int(m.opt.iterations)
    o = jax_ffi.forward(*args, substeps=substeps, tape=True, iterations=it, act=act)
    p = jax_ffi.step(*args, substeps=substeps, iterations=it, act=act)
    same = all(np.array_equal(np.asarray(o[i]), np.asarray(p[i])) for i in (0, 1, 3))
    return np.asarray(o[0]), np.asarray(o[1]), np.asarray(o[3]), np.asarray(o[-1]), same


def tape_forces(m, tape, substeps):
    """The first substep's actuator forces from the tape (the tendon actuators' gear x force / gear; the per-dof joint
    actuators: NaN)."""
    from nablaengine.forest_engine import tape as T, tendon
    from nablaengine import jax_ffi
    info = jax_ffi.init()
    lay = T.layout(m, info["maxcon"])
    off = lay["tact"][0]
    rw = info["rec_words"]
    acts = tendon.actuators(m)
    F = np.full((tape.shape[0], m.nu), np.nan)
    for k, (a, _) in enumerate(acts):
        F[:, a] = tape[:, off + k] / float(m.actuator_gear[a, 0])
    return F


def cmd_fwd(a, cv):
    m = cv.model
    q, v, u, act = corpus(cv.original, a.states, a.seed, act_noise=a.act_noise)
    info = engine_init(m)
    out, ok = {}, True
    for n in (1, 4):
        qe, ve, ae, tp, same = engine_step(m, info, q, v, u, act, n)
        ve, ae = np.asarray(ve, np.float64), np.asarray(ae, np.float64)
        Fe = tape_forces(m, tp, n)
        vc, ac, fc = c_steps(m, q, v, u, act, n)
        vs, as_, fs = c_steps(cv.original, q, v, u, act, n)
        fmask = np.isfinite(Fe[0])
        r = dict(engine_vs_c=cc.stats(cc.rel(ve, vc)), act_vs_c=cc.stats(np.abs(ae - ac).max(-1)),
                 c_converted_vs_source=cc.stats(cc.rel(vc, vs)), c_converted_vs_source_act=cc.stats(
                     np.abs(ac - as_).max(-1)), finite=bool(np.isfinite(ve).all() and np.isfinite(ae).all()),
                 tape_forward_bitwise=bool(same))
        if n == 1:
            r["force_vs_c"] = cc.stats(cc.rel(Fe[:, fmask], fc[:, fmask]))
        if a.mjw_python:
            vw, aw, fw, ver = mjw_steps(m, q, v, u, act, n, a.mjw_python)
            r.update(engine_vs_mjw=cc.stats(cc.rel(ve, vw)), mjw_vs_c=cc.stats(cc.rel(vw, vc)),
                     act_vs_mjw=cc.stats(np.abs(ae - aw).max(-1)), act_mjw_vs_c=cc.stats(np.abs(aw - ac).max(-1)),
                     versions=ver)
            if n == 1:
                r.update(force_vs_mjw=cc.stats(cc.rel(Fe[:, fmask], fw[:, fmask])),
                         force_mjw_vs_c=cc.stats(cc.rel(fw[:, fmask], fc[:, fmask])))
        ok &= r["engine_vs_c"]["median"] < 1e-4 and r["act_vs_c"]["median"] < 1e-5 and same
        out[f"substeps_{n}"] = r
    return out, ok


def rollout(mm, d, q, v, u, a, S):
    mujoco.mj_resetData(mm, d)
    d.qpos[:], d.qvel[:], d.ctrl[:], d.act[:] = q, v, u, a
    for _ in range(S):
        mujoco.mj_step(mm, d)
    return d.qpos.copy(), d.qvel.copy(), d.act.copy()


def fd_grad(mm, q, v, u, a, gq, gv, ga, S, eps=1e-6):
    """Central differences of gq . qpos' + gv . qvel' + ga . act' in (q tangent, v, ctrl, act)."""
    d = mujoco.MjData(mm)

    def L(qq, vv, uu, aa):
        qn, vn, an = rollout(mm, d, qq, vv, uu, aa, S)
        return float(gq @ qn + gv @ vn + ga @ an)

    out = []
    for i in range(mm.nv):
        dv = np.zeros(mm.nv); dv[i] = 1.0
        qp, qm = q.copy(), q.copy()
        mujoco.mj_integratePos(mm, qp, dv, eps)
        mujoco.mj_integratePos(mm, qm, dv, -eps)
        out.append((L(qp, v, u, a) - L(qm, v, u, a)) / (2 * eps))
    for i in range(mm.nv):
        e = np.zeros(mm.nv); e[i] = eps
        out.append((L(q, v + e, u, a) - L(q, v - e, u, a)) / (2 * eps))
    for i in range(mm.nu):
        e = np.zeros(mm.nu); e[i] = eps
        out.append((L(q, v, u + e, a) - L(q, v, u - e, a)) / (2 * eps))
    for i in range(mm.na):
        e = np.zeros(mm.na); e[i] = eps
        out.append((L(q, v, u, a + e) - L(q, v, u, a - e)) / (2 * eps))
    return np.array(out)


def engine_vjp(m, info, q, v, u, a, gq, gv, ga, S):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    B = len(q)
    cw = int(info.get("com_words", 3))
    w, fr, com = jnp.zeros((B, m.nv), jnp.float32), jnp.ones((B,), jnp.float32), jnp.zeros((B, cw), jnp.float32)

    def f(qq, vv, uu, aa):
        o = jax_ffi.step(qq, vv, w, uu, fr, com, substeps=S, iterations=int(m.opt.iterations), clamp_kv=1, normal=0,
                         act=aa)
        return o[0], o[1], o[3]

    _, pull = jax.vjp(f, *(jnp.asarray(x) for x in (q, v, u, a)))
    g = pull(tuple(jnp.asarray(x, jnp.float32) for x in (gq, gv, ga)))
    return [np.asarray(x, np.float64) for x in g]


def cmd_fd(a, cv):
    m = cv.model
    q, v, u, act = corpus(cv.original, a.states, a.seed, act_noise=a.act_noise)
    info = engine_init(m)
    mm = cg.tight(m)
    out, ok = {}, True
    nv, nu, na = m.nv, m.nu, m.na
    for S in a.substeps:
        rng = np.random.default_rng(a.seed + 7)
        gq = rng.normal(size=(len(q), m.nq))
        gq[:, cg.free_quats(m)] = 0.0
        gv = rng.normal(size=(len(q), m.nv))
        ga = rng.normal(size=(len(q), na))
        bq, bv, bu, ba = engine_vjp(m, info, q, v, u, act, gq, gv, ga, S)
        e, eq_, ev, eu, ea = [], [], [], [], []
        for w in range(len(q)):
            q64, v64, u64, a64 = (x[w].astype(np.float64) for x in (q, v, u, act))
            ge = np.concatenate([cg.q_to_tangent(m, q64, bq[w]), bv[w], bu[w], ba[w]])
            gf = fd_grad(mm, q64, v64, u64, a64, gq[w], gv[w], ga[w], S)
            e.append(cg.err(ge, gf))
            eq_.append(cg.err(ge, gf, slice(0, nv)))
            ev.append(cg.err(ge, gf, slice(nv, 2 * nv)))
            eu.append(cg.err(ge, gf, slice(2 * nv, 2 * nv + nu)))
            ea.append(cg.err(ge, gf, slice(2 * nv + nu, 2 * nv + nu + na)))
        r = dict(engine_vs_fd=cc.stats(e), q=cc.stats(eq_), v=cc.stats(ev), ctrl=cc.stats(eu), act=cc.stats(ea))
        ok &= r["engine_vs_fd"]["median"] < 1e-3
        out[f"substeps_{S}"] = r
    return out, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("fwd", "fd"))
    ap.add_argument("--scene", default="spatial", choices=sorted(SCENES))
    ap.add_argument("--all", action="store_true", help="every scene, one subprocess each")
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--substeps", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--act-noise", action="store_true", help="activations drawn in [-0.1, 1.1] (not the rollout's)")
    ap.add_argument("--mjw-python", default="", help="a Python with mujoco_warp (3.17) for the fwd reference")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.all:
        res, ok = {}, True
        for sc in SCENES:
            argv = [sys.executable, __file__, a.cmd, "--scene", sc, "--states", str(a.states), "--seed", str(a.seed),
                    "--substeps", *map(str, a.substeps)] + (["--mjw-python", a.mjw_python] if a.mjw_python else []) \
                + (["--act-noise"] if a.act_noise else [])
            r = subprocess.run(argv, capture_output=True, text=True)
            try:
                res[sc] = json.loads(r.stdout)
            except json.JSONDecodeError:
                res[sc] = dict(error=(r.stdout + r.stderr)[-3000:], **{"pass": False})
            ok &= bool(res[sc].get("pass"))
            print(sc, json.dumps(res[sc].get("result", res[sc]))[:600], file=sys.stderr, flush=True)
        rep = dict(command=a.cmd, act_noise=a.act_noise, scenes=res)
    else:
        cv = load_scene(a.scene)
        res, ok = dict(fwd=cmd_fwd, fd=cmd_fd)[a.cmd](a, cv)
        rep = dict(command=a.cmd, scene=a.scene, states=a.states, seed=a.seed, act_noise=a.act_noise, result=res,
                   notes=cv.notes)
    from nablaengine import provenance
    rep["provenance"] = provenance.provenance()
    rep["pass"] = bool(ok)
    txt = json.dumps(rep, indent=1, default=str)
    print(txt)
    if a.out:
        o = Path(a.out)
        if o.exists():
            sys.exit(f"{o} exists (not overwritten)")
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(txt)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

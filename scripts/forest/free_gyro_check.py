"""MuJoCo 3.17's implicit block of standalone free bodies in the forest engine (forest_engine.free_gyro): unit scenes
against float64 MuJoCo C and MuJoCo Warp 3.17, exact gradients against float64 MuJoCo C central differences.

Scenes (SCENES, generated MJCF, implicitfast; states: qpos0 with random unit quaternions on the free joints, qvel
N(0, 3), zero controls):
  spin      a free capsule, no contacts, no gravity
  spin_off  a free body of a box and a sphere (COM off the origin, a full inertia), gravity, and a hinged arm beside it
  floor     a box on the floor (one condim-3 pair; small tilts, heights about its half size), spinning, and a free-root
            tree of two capsules with a hinge (a free root with a jointed child: no such block in MuJoCo either)
One engine library per process: one scene per run (--scene), or --all (subprocesses).

  fwd   qvel relative error per world after 1 and 4 substeps against MuJoCo C (and MuJoCo Warp with --mjw-python)
  fd    the VJP of gq . qpos' + gv . qvel' (exact mode: clamp_kv 1, normal 0) in (qpos tangent, qvel) against MuJoCo C
        central differences (eps 1e-6; tolerance 1e-12, 200 iterations)

  PYTHONPATH=src python scripts/forest/free_gyro_check.py fwd --all --mjw-python <venv>/bin/python --out runs/x/fwd.json
  PYTHONPATH=src python scripts/forest/free_gyro_check.py fd --all --out runs/x/fd.json

Exit status 1 when a median exceeds its bound (fwd: engine vs MuJoCo C 1e-5; fd: 1e-4).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eq"))
import connect_check as cc
import connect_grad as cg

SCENES = {
    "spin": """<mujoco model="fg_spin"><option timestep="0.002" integrator="implicitfast" gravity="0 0 0"/>
<default><geom contype="0" conaffinity="0" density="600"/></default><worldbody>
<body name="cap" pos="0 0 0.5"><freejoint/><geom type="capsule" size="0.02 0.06" euler="0 20 0"/></body>
</worldbody></mujoco>""",
    "spin_off": """<mujoco model="fg_spin_off"><option timestep="0.002" integrator="implicitfast"/>
<default><geom contype="0" conaffinity="0" density="600"/></default><worldbody>
<body name="b" pos="0 0 0.5"><freejoint/><geom type="box" size="0.05 0.03 0.02" pos="0.02 0.01 -0.01" euler="10 20 30"/>
<geom type="sphere" size="0.02" pos="-0.03 0.04 0"/></body>
<body name="arm" pos="1 0 0.5"><joint type="hinge" axis="0 1 0" damping="0.1"/>
<geom type="capsule" size="0.02 0.1" fromto="0 0 0 0.2 0 0"/></body>
</worldbody></mujoco>""",
    "floor": """<mujoco model="fg_floor"><option timestep="0.002" integrator="implicitfast"/>
<worldbody><geom name="floor" type="plane" size="0 0 1"/>
<body name="box" pos="0 0 0.05"><freejoint/><geom name="box" type="box" size="0.05 0.04 0.03" pos="0.01 0 0" density="500"/></body>
<body name="r" pos="1 0 0.5"><freejoint/><geom name="r" type="capsule" size="0.02 0.1" contype="0" conaffinity="0"/>
<body pos="0 0 0.2"><joint type="hinge" damping="0.3"/><geom type="capsule" size="0.02 0.1" contype="0" conaffinity="0"/></body></body>
</worldbody><contact><pair geom1="floor" geom2="box" condim="3"/></contact></mujoco>""",
}


def states(m, scene, n, seed):
    rng = np.random.default_rng(seed)
    q = np.tile(m.qpos0, (n, 1))
    v = rng.normal(0, 3.0, (n, m.nv))
    for j in range(m.njnt):
        if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        a = int(m.jnt_qposadr[j])
        qt = rng.normal(size=(n, 4))
        if scene == "floor" and m.body(int(m.jnt_bodyid[j])).name == "box":
            qt = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)) + 0.02 * rng.normal(size=(n, 4))
            q[:, a + 2] = 0.03 + 0.001 * rng.normal(size=n)
            v[:, int(m.jnt_dofadr[j]) + 2] *= 0.05
        q[:, a + 3:a + 7] = qt / np.linalg.norm(qt, axis=1, keepdims=True)
    return q.astype(np.float32), v.astype(np.float32)


def c_steps(m, q, v, n):
    d = mujoco.MjData(m)
    out, nc = [], []
    for w in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:] = q[w], v[w]
        for _ in range(n):
            mujoco.mj_step(m, d)
        out.append(d.qvel.copy())
        nc.append(d.ncon)
    return np.array(out), float(np.mean(nc))


def init(m):
    from nablaengine import engines, jax_ffi
    engines.prefer_forest(m)
    return jax_ffi.init(model=m, foot_pairs=[])


def cmd_fwd(a, m):
    from nablaengine.forest_engine import free_gyro
    q, v = states(m, a.scene, a.states, a.seed)
    u = np.zeros((len(q), m.nu), np.float32)
    info = init(m)
    out, ok = dict(free_bodies=free_gyro.bodies(m)), True
    for n in a.substeps:
        _, ve, _ = cc.engine_step(m, info, q, v, u, substeps=n)
        ve = np.asarray(ve, np.float64)
        vc, nc = c_steps(m, q, v, n)
        r = dict(engine_vs_c=cc.stats(cc.rel(ve, vc)), mean_ncon_c=nc, finite=bool(np.isfinite(ve).all()))
        if a.mjw_python:
            from spatial_check import mjw_steps
            vw = mjw_steps(m, q, v, u, n, a.mjw_python)
            r.update(engine_vs_mjw=cc.stats(cc.rel(ve, vw)), mjw_vs_c=cc.stats(cc.rel(vw, vc)))
        ok &= r["engine_vs_c"]["median"] < 1e-5 and r["finite"]
        out[f"substeps_{n}"] = r
    return out, ok


def cmd_fd(a, m):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    q, v = states(m, a.scene, a.states, a.seed)
    B = len(q)
    u = jnp.zeros((B, m.nu), jnp.float32)
    info = init(m)
    mm = cg.tight(m)
    d = mujoco.MjData(mm)
    out, ok = {}, True
    cw = int(info.get("com_words", 3))
    w0, fr, com = jnp.zeros((B, m.nv), jnp.float32), jnp.ones((B,), jnp.float32), jnp.zeros((B, cw), jnp.float32)
    for S in a.substeps:
        g = np.random.default_rng(a.seed + 7)
        gq = g.normal(size=(B, m.nq))
        gq[:, cg.free_quats(m)] = 0.0
        gv = g.normal(size=(B, m.nv))

        def f(qq, vv):
            return jax_ffi.step(qq, vv, w0, u, fr, com, substeps=S, iterations=int(m.opt.iterations), clamp_kv=1,
                                normal=0)[:2]

        _, pull = jax.vjp(f, jnp.asarray(q), jnp.asarray(v))
        bq, bv = [np.asarray(x, np.float64) for x in pull((jnp.asarray(gq, jnp.float32), jnp.asarray(gv, jnp.float32)))]
        E, EQ, EV = [], [], []
        for w in range(B):
            def L(qq, vv):
                mujoco.mj_resetData(mm, d)
                d.qpos[:], d.qvel[:] = qq, vv
                for _ in range(S):
                    mujoco.mj_step(mm, d)
                return float(gq[w] @ d.qpos + gv[w] @ d.qvel)
            q64, v64 = q[w].astype(np.float64), v[w].astype(np.float64)
            fd, eps = [], 1e-6
            for i in range(m.nv):
                dv = np.zeros(m.nv)
                dv[i] = 1.0
                qp, qm = q64.copy(), q64.copy()
                mujoco.mj_integratePos(mm, qp, dv, eps)
                mujoco.mj_integratePos(mm, qm, dv, -eps)
                fd.append((L(qp, v64) - L(qm, v64)) / (2 * eps))
            for i in range(m.nv):
                e = np.zeros(m.nv)
                e[i] = eps
                fd.append((L(q64, v64 + e) - L(q64, v64 - e)) / (2 * eps))
            fd = np.array(fd)
            ge = np.concatenate([cg.q_to_tangent(m, q64, bq[w]), bv[w]])
            E.append(cg.err(ge, fd))
            EQ.append(cg.err(ge, fd, slice(0, m.nv)))
            EV.append(cg.err(ge, fd, slice(m.nv, 2 * m.nv)))
        r = dict(engine_vs_fd=cc.stats(E), q=cc.stats(EQ), v=cc.stats(EV))
        ok &= r["engine_vs_fd"]["median"] < 1e-4
        out[f"substeps_{S}"] = r
    return out, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("fwd", "fd", "xml"))
    ap.add_argument("--scene", default="spin", choices=SCENES)
    ap.add_argument("--all", action="store_true", help="every scene, one subprocess each")
    ap.add_argument("--states", type=int, default=0, help="default: fwd 64, fd 16")
    ap.add_argument("--substeps", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mjw-python", default="", help="a Python with mujoco_warp (3.17) for the fwd reference")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    a.states = a.states or (64 if a.cmd == "fwd" else 16)
    if a.cmd == "xml":
        print(SCENES[a.scene])
        return 0
    if a.all:
        res, ok = {}, True
        for sc in SCENES:
            argv = [sys.executable, __file__, a.cmd, "--scene", sc, "--states", str(a.states), "--seed", str(a.seed),
                    "--substeps", *map(str, a.substeps)] + (["--mjw-python", a.mjw_python] if a.mjw_python else [])
            r = subprocess.run(argv, capture_output=True, text=True)
            try:
                res[sc] = json.loads(r.stdout)
            except json.JSONDecodeError:
                res[sc] = dict(error=(r.stdout + r.stderr)[-3000:], **{"pass": False})
            ok &= bool(res[sc].get("pass"))
            print(sc, json.dumps(res[sc].get("result", res[sc]))[:900], file=sys.stderr, flush=True)
        rep = dict(command=a.cmd, scenes=res)
    else:
        from nablaengine.envs import convert
        m = convert.to_engine(SCENES[a.scene]).model
        res, ok = dict(fwd=cmd_fwd, fd=cmd_fd)[a.cmd](a, m)
        rep = dict(command=a.cmd, scene=a.scene, states=a.states, seed=a.seed, result=res)
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

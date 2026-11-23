"""Per-geom contact inverse weights (nablaengine.model.invweight): the engines read every contact's inverse weight from
the per-geom table. Check: a model whose table gives each geom its body's value times a per-body factor runs bitwise
like the same model with body_invweight0 scaled by those factors (no table), forward (qpos', qvel', warm', info) and
VJP (d/dqpos, d/dqvel, d/dctrl), on contact states. Each variant runs in a child process (one library per process);
exits non-zero on a difference.

  --engine forest   an MJCF scene built here (a two-link capsule arm, a free box and a free sphere on a plane;
                    --cone elliptic, condim 4, or pyramidal, condim 3), the table carried by the custom numeric
                    nablaengine_geom_invweight (MuJoCo's <custom><numeric>); states from MuJoCo C rollouts
  --engine g1       the fixtures' G1 model (fixtures/mjlab; pyramidal, or --cone elliptic: the G1 class's elliptic
                    cone) and the first --nworld states of a G1 corpus (--corpus, default nablaengine.fixtures'
                    default corpus, mf_s0); a binary model cannot take a custom field, so the child replaces
                    invweight.geom_values (the carrier's reader)

  python scripts/invweight_check.py --engine forest [--cone pyramidal] [--nworld 256] [--keep DIR]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCENE = """<mujoco>
  <option timestep="0.005" cone="{cone}" impratio="5" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="link0" pos="0 0 0.18">
      <joint name="j0" type="hinge" axis="0 1 0" range="-1.6 1.6"/>
      <geom name="c0" type="capsule" fromto="0 0 0 0.2 0 0" size="0.025" mass="0.4"/>
      <body name="link1" pos="0.2 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
        <geom name="c1" type="capsule" fromto="0 0 0 0.16 0 0" size="0.02" mass="0.2"/>
      </body>
    </body>
    <body name="box" pos="0.3 0 0.04"><freejoint/><geom name="box" type="box" size="0.04 0.04 0.04" mass="0.1"/></body>
    <body name="ball" pos="0.1 0.02 0.05"><freejoint/><geom name="ball" type="sphere" size="0.035" mass="0.05"/></body>
  </worldbody>
  <contact>
    <pair geom1="floor" geom2="box" condim="{dim}" friction="0.8 0.8 0.01 0.0001 0.0001"/>
    <pair geom1="floor" geom2="ball" condim="{dim}" friction="0.8 0.8 0.01 0.0001 0.0001"/>
    <pair geom1="floor" geom2="c1" condim="{dim}" friction="1 1 0.01 0.0001 0.0001"/>
    <pair geom1="floor" geom2="c0" condim="{dim}" friction="1 1 0.01 0.0001 0.0001"/>
    <pair geom1="c1" geom2="box" condim="{dim}" friction="1 1 0.01 0.0001 0.0001"/>
    <pair geom1="c0" geom2="box" condim="{dim}" friction="1 1 0.01 0.0001 0.0001"/>
    <pair geom1="c1" geom2="ball" condim="{dim}" friction="1 1 0.01 0.0001 0.0001"/>
    <pair geom1="c0" geom2="ball" condim="{dim}" friction="1 1 0.01 0.0001 0.0001"/>
    <pair geom1="box" geom2="ball" condim="{dim}" friction="0.6 0.6 0.01 0.0001 0.0001"/>
  </contact>
  <actuator>
    <position joint="j0" kp="20" ctrlrange="-1.6 1.6"/>
    <position joint="j1" kp="10" ctrlrange="-2.2 2.2"/>
  </actuator>
</mujoco>
"""
SUBSTEPS = 4


def factors(m):
    """A factor per body (1 for the world), different between the bodies."""
    return np.array([0.0 if b == 0 else 1.0 + 0.37 * (b % 5) + 0.11 * b for b in range(m.nbody)])


def forest_models(cone):
    """(model with the per-geom table as the custom numeric, model with body_invweight0 scaled)."""
    import mujoco
    from nablaengine.model import invweight
    xml = SCENE.format(cone=cone, dim=4 if cone == "elliptic" else 3)
    mb = mujoco.MjModel.from_xml_string(xml)
    f = factors(mb)
    vals = mb.body_invweight0[mb.geom_bodyid, 0] * f[mb.geom_bodyid]
    spec = mujoco.MjSpec.from_string(xml)
    spec.add_numeric(name=invweight.NUMERIC, size=mb.ngeom, data=list(map(float, vals)))
    mt = spec.compile()
    mb.body_invweight0[:, 0] *= f
    return mt, mb


def forest_states(m, n, seed):
    """n contact states of MuJoCo C rollouts with random position targets (qpos, qvel, warm, ctrl)."""
    import mujoco
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    out = []
    while len(out) < n:
        mujoco.mj_resetData(m, d)
        d.qpos[0], d.qpos[1] = rng.uniform(-0.3, 0.6), rng.uniform(-1.5, 1.5)
        u = rng.uniform(-1, 1, 2)
        for t in range(200):
            if t % 20 == 0:
                u = np.clip(u + rng.normal(0, 0.8, 2), m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            d.ctrl[:] = u
            mujoco.mj_step(m, d)
            if t > 20 and t % 7 == 0 and d.ncon >= 2:
                out.append((d.qpos.copy(), d.qvel.copy(), d.qacc_warmstart.copy(), d.ctrl.copy()))
    return [np.array([s[k] for s in out[:n]]) for k in range(4)]


def child(a):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    from nablaengine.model import invweight
    if a.engine == "forest":
        mt, mb = forest_models(a.cone)
        m = mt if a.child == "table" else mb
        q, v, w, u = forest_states(mb, a.nworld, a.seed)
        fr, fp = np.ones(len(q)), []
    else:
        from nablaengine import fixtures
        m, mt = fixtures.load_model(check=None)
        fp = mt["foot_pairs"]
        if a.cone == "elliptic":
            import mujoco
            m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        f = factors(m)
        if a.child == "table":
            base = m.body_invweight0[:, 0].copy()
            invweight.geom_values = lambda mm: base[mm.geom_bodyid] * f[mm.geom_bodyid]
        else:
            m.body_invweight0[:, 0] *= f
        z = fixtures.load_corpus(a.corpus)
        sel = slice(0, a.nworld)
        q, v, w, u, fr = (z[k][sel] for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction"))
    info = jax_ffi.init(model=m, foot_pairs=fp)
    B = len(q)
    t = lambda x: jnp.asarray(np.asarray(x, np.float32))
    fr, com = t(fr), jnp.zeros((B, info["com_words"]), jnp.float32)
    q1, v1, w1, inf = jax_ffi.forward(t(q), t(v), t(w), t(u), fr, com, iterations=a.iterations, substeps=SUBSTEPS)
    step = jax_ffi.make_step(iterations=a.iterations, substeps=SUBSTEPS)
    rng = np.random.default_rng(a.seed + 1)
    yq, yv = t(rng.standard_normal(q.shape)), t(rng.standard_normal(v.shape))

    def loss(qq, vv, uu):
        a1, b1, _ = step(qq, vv, t(w), uu, fr, com)
        return jnp.sum(yq * a1) + jnp.sum(yv * b1)

    g = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(t(q), t(v), t(u))
    np.savez(a.child_out, q1=np.asarray(q1), v1=np.asarray(v1), w1=np.asarray(w1), info=np.asarray(inf),
             gq=np.asarray(g[0]), gv=np.asarray(g[1]), gu=np.asarray(g[2]),
             table=np.array(invweight.table(m, _collision_geoms(m)) is not None))


def _collision_geoms(m):
    from nablaengine import engines
    return engines.modules(m).step.collision_geoms(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=("forest", "g1"), required=True)
    ap.add_argument("--cone", choices=("elliptic", "pyramidal"), default=None,
                    help="default: elliptic for the forest scene, the model's (pyramidal) for g1")
    ap.add_argument("--nworld", type=int, default=256)
    ap.add_argument("--corpus", default=None, help="g1: the G1 corpus (default: nablaengine.fixtures' default, mf_s0)")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", default=None, help="keep the children's outputs in this directory")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.cone is None:
        a.cone = "elliptic" if a.engine == "forest" else "pyramidal"
    if a.child:
        return child(a)
    if a.engine == "forest":
        os.environ.setdefault("DK_ADJ_STAGE", "3")
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="invweight_check_"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        out = {}
        for kind in ("table", "body"):
            p = work / f"{a.engine}_{a.cone}_{kind}.npz"
            cmd = [sys.executable, __file__, "--engine", a.engine, "--cone", a.cone, "--nworld", str(a.nworld),
                   "--iterations", str(a.iterations), "--seed", str(a.seed), "--child", kind, "--child-out", str(p)]
            if a.corpus:
                cmd += ["--corpus", a.corpus]
            subprocess.run(cmd, check=True)
            out[kind] = dict(np.load(p))
        t, b = out["table"], out["body"]
        assert bool(t["table"]) and not bool(b["table"]), "the table variant must have a table, the other none"
        res = dict(engine=a.engine, cone=a.cone, nworld=int(len(t["q1"])),
                   contacts_mean=float(t["info"][:, 1].mean()), contacts_max=int(t["info"][:, 1].max()),
                   bitwise={k: bool(np.array_equal(t[k], b[k])) for k in ("q1", "v1", "w1", "info", "gq", "gv", "gu")},
                   finite=bool(all(np.isfinite(t[k]).all() for k in ("gq", "gv", "gu"))))
        print(json.dumps(res, indent=1))
        if not (all(res["bitwise"].values()) and res["finite"] and res["contacts_mean"] > 0):
            sys.exit("invweight_check: FAILED")
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

"""MuJoCo's direct solref form (-stiffness, -damping) in the forest engine (forest_engine.solref): unit scenes against
MuJoCo Warp 3.17 and float64 MuJoCo C, exact gradients against float64 MuJoCo C differences.

Scenes (SCENES, generated MJCF, explicit pairs only, implicitfast; each with standard solrefs beside the direct ones):
  pairs     a free sphere (a light ball) and box and a capsule on a slide (z) and a hinge (y: it rocks onto one end) on a
            floor plane: ball-floor (-80000, -1) (MyoSuite table tennis's ball pairs), capsule-floor (-20000, -60) condim 4, ball-capsule (-80000, -1), capsule-box
            (-5000, -40) condim 1, box-floor standard (0.02, 1); a fixed-base two-hinge arm with motors whose capsule
            tip has a direct pair with the floor (-30000, -80)
  pairse    pairs with the elliptic cone (condim 1 / 3 pairs: the capsule-floor pair condim 3)
  limits    a fixed-base arm of four hinges and a slide with motors: limits (-3000, -30), (-800, -12), standard (0.02, 1),
            slide (-500, -20) (per-dof limit sets); frictionloss on two hinges, solreffriction (-1, -25) and standard
            (0.02, 1) (per-dof frictionloss sets); its tip against the floor (-20000, -50), condim 1 (with friction the
            five-dof arm pressed on the floor is near-indeterminate: float32 MuJoCo Warp and the engine agree, float64
            MuJoCo C differs by up to 3e-2 in a few states, with the standard form as well)
  equality  two hinge chains with motors: a joint equality (-4000, -60), a connect from a free body to a chain tip
            (-6000, -90), a fixed tendon over the second chain with a limit (-2500, -35) and a tendon equality
            (-3000, -50) to a second fixed tendon; standard joint limits (0.02, 1) and a direct one (-1500, -25)
States: kdof64_check's corpus (MuJoCo C rollouts from qpos0, random velocities N(0, 0.1) and piecewise-constant
controls). One engine library per process: one scene per run (--scene), or --all (subprocesses).

  fwd   kdof64_check.cmd_fwd: qvel relative error per world after 1 and 4 substeps against MuJoCo Warp and MuJoCo C, the
        tape's forward bitwise the plain step's; the engine against itself on a twin model whose direct solrefs are
        replaced by the standard pairs with the same k and b (twin(); a second process); MuJoCo C with the direct
        solrefs against MuJoCo C with the engine's earlier misreading (the size of the fix); the rows MuJoCo C builds at the states by form (direct /
        standard) and the solref sets the engine reads
  fd    kdof64_check.cmd_fd: the VJP (exact mode) against the active-set-consistent float64 MuJoCo C differences

  PYTHONPATH=src python scripts/forest/solref_check.py fwd --all --mjw-python <venv>/bin/python --out runs/x/fwd.json
  PYTHONPATH=src python scripts/forest/solref_check.py fd --all --out runs/x/fd.json

Exit status 1 when a median exceeds its bound (fwd: engine vs MuJoCo C 1e-4; fd: sided 1e-3) or a scene has no
direct-form row active at its states.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "forest"))
import kdof64_check as K

HEAD = """<mujoco model="solref_{name}">
<option timestep="0.002" integrator="implicitfast"{cone}/>
<default><geom contype="0" conaffinity="0" density="600"/><joint armature="0.002" damping="0.01"/></default>
<worldbody>
<geom name="floor" type="plane" size="0 0 1"/>
"""


def scene_xml(name):
    cone = ' cone="elliptic"' if name == "pairse" else ""
    if name in ("pairs", "pairse"):
        c4 = "3" if name == "pairse" else '4" friction="0.8 0.8 0.01 0.0001 0.0001'
        body = """<body name="ball" pos="0 0 0.0195"><freejoint/><geom name="ball" type="sphere" size="0.02" mass="0.0027"/></body>
<body name="cap" pos="0.05 0.08 0.0195"><joint name="cz" type="slide" axis="0 0 1"/><joint name="cy" axis="0 1 0"/>
  <geom name="cap" type="capsule" size="0.02 0.06" euler="0 90 0"/></body>
<body name="box" pos="0.05 0.2 0.0295"><freejoint/><geom name="box" type="box" size="0.03 0.03 0.03"/></body>
<body name="arm0" pos="-0.25 0 0.12">
  <joint name="s1" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.015"/>
  <body name="arm1" pos="0.1 0 0"><joint name="s2" axis="0 1 0"/>
    <geom name="tip" type="capsule" fromto="0 0 0 0.08 0 -0.08" size="0.015"/></body>
</body>
</worldbody>
<contact>
<pair geom1="floor" geom2="ball" solref="-80000 -1"/>
<pair geom1="floor" geom2="cap" solref="-20000 -60" condim="{c4}"/>
<pair geom1="ball" geom2="cap" solref="-80000 -1"/>
<pair geom1="cap" geom2="box" solref="-5000 -40" condim="1"/>
<pair geom1="floor" geom2="box"/>
<pair geom1="floor" geom2="tip" solref="-30000 -80"/>
</contact>
<actuator><motor joint="s1" ctrlrange="-0.3 0.3"/><motor joint="s2" ctrlrange="-0.3 0.3"/></actuator>
</mujoco>
""".replace("{c4}", c4)
    elif name == "limits":
        body = """<body name="l0" pos="0 0 0.25">
  <joint name="j0" axis="0 0 1" range="-0.2 0.2" limited="true" solreflimit="-3000 -30" frictionloss="0.02"
         solreffriction="-1 -25"/>
  <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.015"/>
  <body name="l1" pos="0.08 0 0"><joint name="j1" axis="0 1 0" range="-0.15 0.3" limited="true" solreflimit="-800 -12"/>
    <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.015"/>
    <body name="l2" pos="0.08 0 0"><joint name="j2" axis="0 1 0" range="-0.25 0.25" limited="true"
           frictionloss="0.01" solreffriction="0.02 1"/>
      <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.015"/>
      <body name="l3" pos="0.08 0 0"><joint name="j3" axis="1 0 0" range="-0.3 0.3" limited="true" solreflimit="-2000 -40"/>
        <geom type="capsule" fromto="0 0 0 0.06 0 0" size="0.015"/>
        <body name="l4" pos="0.06 0 0"><joint name="j4" type="slide" axis="0 0 -1" range="-0.02 0.02" limited="true"
               solreflimit="-500 -20"/>
          <geom name="tip" type="capsule" fromto="0 0 0 0 0 -0.22" size="0.015"/></body>
      </body>
    </body>
  </body>
</body>
</worldbody>
<contact><pair geom1="floor" geom2="tip" solref="-20000 -50" condim="1"/></contact>
<actuator><motor joint="j0" ctrlrange="-0.4 0.4"/><motor joint="j1" ctrlrange="-0.4 0.4"/>
<motor joint="j2" ctrlrange="-0.3 0.3"/><motor joint="j3" ctrlrange="-0.2 0.2"/><motor joint="j4" ctrlrange="-1 1"/></actuator>
</mujoco>
"""
    elif name == "equality":
        body = """<body name="a0" pos="0 0 0.3"><joint name="a0" axis="0 1 0" range="-0.3 0.3" limited="true"/>
  <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.015"/>
  <body name="a1" pos="0.1 0 0"><joint name="a1" axis="0 1 0" range="-0.4 0.4" limited="true" solreflimit="-1500 -25"/>
    <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.015"/>
    <body name="a2" pos="0.1 0 0"><joint name="a2" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.015"/>
      <site name="atip" pos="0.08 0 0"/></body>
  </body>
</body>
<body name="b0" pos="0 0.2 0.3"><joint name="b0" axis="0 1 0"/>
  <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.015"/>
  <body name="b1" pos="0.1 0 0"><joint name="b1" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.015"/>
    <body name="b2" pos="0.1 0 0"><joint name="b2" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.015"/></body>
  </body>
</body>
<body name="w" pos="0.28 0 0.3"><freejoint/><geom type="sphere" size="0.02"/><site name="wc" pos="0 0 0"/></body>
</worldbody>
<tendon>
<fixed name="tb" limited="true" range="-0.25 0.25" solreflimit="-2500 -35"><joint joint="b1" coef="1"/><joint joint="b2" coef="0.5"/></fixed>
<fixed name="tc"><joint joint="b0" coef="1"/></fixed>
</tendon>
<equality>
<joint joint1="a2" joint2="a0" polycoef="0 0.5 0 0 0" solref="-4000 -60"/>
<connect site1="wc" site2="atip" solref="-6000 -90"/>
<tendon tendon1="tc" tendon2="tb" polycoef="0 0.8 0 0 0" solref="-3000 -50"/>
</equality>
<actuator><motor joint="a0" ctrlrange="-0.5 0.5"/><motor joint="a1" ctrlrange="-0.5 0.5"/>
<motor joint="b1" ctrlrange="-0.4 0.4"/><motor joint="b2" ctrlrange="-0.3 0.3"/></actuator>
</mujoco>
"""
    else:
        raise ValueError(name)
    return HEAD.format(name=name, cone=cone) + body


SCENES = ("pairs", "pairse", "limits", "equality")


def load_scene(name):
    from nablaengine.envs import convert
    return convert.to_engine(scene_xml(name))


def active_rows(m, q, v):
    """The constraint rows MuJoCo C builds at the states, by kind and solref form: {kind: {direct, standard}} (an
    equality row counts once per scalar row; a contact once)."""
    from nablaengine.forest_engine import solref
    d = mujoco.MjData(m)
    T = mujoco.mjtConstraint
    kinds = {int(T.mjCNSTR_EQUALITY): "equality", int(T.mjCNSTR_FRICTION_DOF): "frictionloss",
             int(T.mjCNSTR_LIMIT_JOINT): "joint limit", int(T.mjCNSTR_LIMIT_TENDON): "tendon limit"}
    out = {}
    for w in range(len(q)):
        d.qpos[:], d.qvel[:] = q[w], v[w]
        mujoco.mj_forward(m, d)
        for c in d.contact[:d.ncon]:
            k = out.setdefault("contact", dict(direct=0, standard=0))
            k[solref.form(c.solref)] += 1
        for r in range(d.nefc):
            t, i = int(d.efc_type[r]), int(d.efc_id[r])
            if t not in kinds:
                continue
            sr = {"equality": m.eq_solref, "frictionloss": m.dof_solref, "joint limit": m.jnt_solref,
                  "tendon limit": m.tendon_solref_lim}[kinds[t]][i]
            k = out.setdefault(kinds[t], dict(direct=0, standard=0))
            k[solref.form(sr)] += 1
    return out


ARRAYS = ("pair_solref", "jnt_solref", "dof_solref", "eq_solref", "tendon_solref_lim")


def twin(m, mode):
    """A copy of m with every direct solref (both entries <= 0) replaced: mode "equivalent" by the standard pair with
    the same k and b (timeconst 2 / -solref[1], dampratio -solref[1] / (2 sqrt(-solref[0])): b = 2 / (dmax tc), k =
    1 / (dmax tc dr)^2), "misread" by what the engine read before forest_engine.solref (kbimp's standard formula on
    the direct entries: timeconst max(solref[0], 2 h) = 2 h, the damping ratio's square solref[1]^2, i.e. (2 h,
    |solref[1]|))."""
    mm = m.__copy__()
    for f in ARRAYS:
        x = getattr(mm, f)
        for i in range(len(x)):
            s0, s1 = float(x[i, 0]), float(x[i, 1])
            if s0 <= 0 and s1 <= 0:
                if mode == "equivalent":
                    tc = 2.0 / -s1
                    x[i] = (tc, -s1 / (2.0 * np.sqrt(-s0)))
                else:
                    x[i] = (2.0 * m.opt.timestep, abs(s1))
    return mm


def cmd_twinrun(a, cv):
    """The engine on twin(cv.model, "equivalent") at the scene's corpus: qvel after 1 and 4 substeps (to --out .npz)."""
    import connect_check as cc
    m = twin(cv.model, "equivalent")
    q, v, u = K.corpus(cv.original, a.states, a.seed)
    info = cc.engine_init(m)
    np.savez(a.out, **{f"v{n}": np.asarray(cc.engine_step(m, info, q, v, u, substeps=n)[1]) for n in (1, 4)})
    return {}, True


def cmd_fwd(a, cv):
    import tempfile
    import connect_check as cc
    from nablaengine.forest_engine import solref
    out, ok = K.cmd_fwd(a, cv)
    q, v, u = K.corpus(cv.original, a.states, a.seed)
    info = cc.engine_init(cv.model)
    with tempfile.TemporaryDirectory() as tmp:
        argv = [sys.executable, __file__, "twinrun", "--scene", a.scene, "--states", str(a.states), "--seed",
                str(a.seed), "--out", str(Path(tmp) / "t.npz")]
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-3000:])
        tw = dict(np.load(Path(tmp) / "t.npz"))
    for n in (1, 4):
        ve = np.asarray(cc.engine_step(cv.model, info, q, v, u, substeps=n)[1], np.float64)
        vc, _ = K.c_steps(cv.model, q, v, u, n)
        vm, _ = K.c_steps(twin(cv.model, "misread"), q, v, u, n)
        out[f"substeps_{n}"].update(engine_vs_equivalent_standard=cc.stats(cc.rel(ve, tw[f"v{n}"])),
                                    c_misread_vs_c=cc.stats(cc.rel(vm, vc)))
    out["rows_at_states"] = act = active_rows(cv.model, q, v)
    out["solref_rows"] = {f"{k} {n}": [float(x) for x in sr] for k, n, sr in solref.rows(cv.model)}
    out["direct_text"] = bool(solref.needed(cv.model))
    ok &= out["direct_text"] and sum(r["direct"] for r in act.values()) > 0
    return out, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("fwd", "fd", "xml", "twinrun"))
    ap.add_argument("--scene", default="pairs", choices=SCENES)
    ap.add_argument("--all", action="store_true", help="every scene, one subprocess each")
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--substeps", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mjw-python", default="", help="a Python with mujoco_warp (3.17) for the fwd reference")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "xml":
        print(scene_xml(a.scene))
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
        cv = load_scene(a.scene)
        if a.cmd == "twinrun":
            cmd_twinrun(a, cv)
            return 0
        res, ok = dict(fwd=cmd_fwd, fd=K.cmd_fd)[a.cmd](a, cv)
        rep = dict(command=a.cmd, scene=a.scene, states=a.states, seed=a.seed, result=res, notes=cv.notes)
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

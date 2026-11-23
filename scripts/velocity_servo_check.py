"""Velocity servos (MuJoCo's <velocity kv>) on the forest engine against float64 MuJoCo C: forward, and the VJP against
float64 central differences, with the actuator forcerange clamp, for both adjoint rules of the clamped kv (clamp_kv).

MuJoCo's <velocity kv> is gaintype fixed, gainprm[0] = kv, biastype affine, biasprm = (0, 0, -kv): f = kv (u - qdot)
with ctrl the target velocity. On the forest engine it is its affine actuator form (f = kp u - kq q - kv qdot with
kp = kv, kq = 0), clamped to ctrlrange, then at the actuator forcerange, then at jnt_actfrcrange; under implicitfast kv
enters M + h kv, and is dropped from it while the force sits at the forcerange (MuJoCo Warp's rule). The tree engine
refuses velocity servos (capability code H11v); a single-tree robot with them runs on the forest engine through
nablaengine.engines.prefer_forest(m).

Cases (--case; "all" runs each in a process of its own, one library per process):
  arm       a slide cart under the world with three hinges in a chain: velocity servos on the slide (kv 20) and two
            hinges (kv 3; kv 0.2 with forcerange +-0.3 on the last), a position servo (kp 10, kv 0.5) on the middle one;
            joint damping, no contacts
  floating  a free-floating base (the G1 class) with four hinges driven by velocity servos (one with forcerange
            +-0.8), falling under gravity, no contacts: first the tree engine's refusal (jax_ffi.init raises
            UnsupportedModel with H11v), then the same model through engines.prefer_forest on the forest engine

Per case: --nworld random states (qpos, qvel, ctrl uniform; ctrl wide enough that a share of the worlds reaches the
forcerange), one control step of --substeps substeps. Forward: qvel against MuJoCo C (mj_step, float64), relative
error |d| / (|ref| + 1e-2). Gradients: loss = w . qvel' + c . qpos' (fixed random w, c; qpos of the free joint left
out), d loss / d (ctrl, qvel, qpos) from jax_ffi.make_step against float64 central differences of MuJoCo C over the
first --fd-worlds worlds, relative error |g - fd| / (|fd| + 1e-3), for clamp_kv=1 (the exact rule: the clamped kv's
derivative is 0) and clamp_kv=0 (the FFI default, MuJoCo Warp's adjoint: keeps -kv lambda on a clamped actuator). A
world counts as clamped when the force-limited servo sits at its forcerange in any substep of MuJoCo C's step.

Gates (exit 1 on failure): forward median <= 1e-5 and max <= 5e-4 (float32); clamp_kv=1 every gradient's max <= 1e-3
on every world; clamp_kv=0 every gradient's max <= 1e-3 over the worlds without a clamp (on the clamped worlds the
surrogate's -kv lambda differs from the exact derivative by design, and with several substeps that reaches every input
through the later substeps: reported, not gated). Each case needs clamped and unclamped worlds among the --fd-worlds.

  PYTHONPATH=src python scripts/velocity_servo_check.py [--case all] [--out runs/velocity_servo_check.json]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ARM = """<mujoco model="velocity_arm"><option timestep="0.005" integrator="implicitfast"/>
<default><geom contype="0" conaffinity="0"/></default>
<worldbody><body name="cart"><joint name="s" type="slide" axis="1 0 0" damping="0.1"/>
 <geom type="box" size=".2 .1 .05" mass="2"/>
 <body name="l1"><joint name="h1" type="hinge" axis="0 1 0" damping="0.05"/>
  <geom type="capsule" fromto="0 0 0 0 0 .5" size=".03" mass=".5"/>
  <body name="l2" pos="0 0 .5"><joint name="h2" type="hinge" axis="0 1 0"/>
   <geom type="capsule" fromto="0 0 0 0 0 .4" size=".03" mass=".3"/>
   <body name="w" pos="0 0 .4"><joint name="h3" type="hinge" axis="1 0 0"/>
    <geom type="cylinder" size=".08 .02" euler="0 90 0" mass=".2"/></body></body></body></body></worldbody>
<actuator><velocity joint="s" kv="20"/><velocity joint="h1" kv="3"/><position joint="h2" kp="10" kv="0.5"/>
 <velocity name="h3_lim" joint="h3" kv="0.2" forcerange="-0.3 0.3" forcelimited="true"/></actuator></mujoco>"""

FLOATING = """<mujoco model="velocity_floating"><option timestep="0.004" integrator="implicitfast"/>
<default><geom contype="0" conaffinity="0"/></default>
<worldbody><body name="base" pos="0 0 1"><freejoint/><geom type="box" size=".15 .1 .05" mass="3"/>
 <body name="a1" pos=".15 0 0"><joint name="j1" type="hinge" axis="0 1 0" damping="0.02"/>
  <geom type="capsule" fromto="0 0 0 .3 0 0" size=".03" mass=".4"/>
  <body name="a2" pos=".3 0 0"><joint name="j2" type="hinge" axis="0 0 1"/>
   <geom type="capsule" fromto="0 0 0 .25 0 0" size=".03" mass=".3"/></body></body>
 <body name="b1" pos="-.15 0 0"><joint name="j3" type="hinge" axis="0 1 0"/>
  <geom type="capsule" fromto="0 0 0 -.3 0 0" size=".03" mass=".4"/>
  <body name="wheel" pos="-.3 0 0"><joint name="j4" type="hinge" axis="1 0 0"/>
   <geom type="cylinder" size=".1 .02" euler="0 90 0" mass=".3"/></body></body></body></worldbody>
<actuator><velocity joint="j1" kv="2"/><velocity joint="j2" kv="1"/><velocity joint="j3" kv="2"/>
 <velocity name="j4_lim" joint="j4" kv="0.8" forcerange="-0.8 0.8" forcelimited="true"/></actuator></mujoco>"""

CASES = {"arm": (ARM, "h3_lim", 2.0, 2.0), "floating": (FLOATING, "j4_lim", 1.5, 1.5)}
GATE_FWD_MED, GATE_FWD_MAX, GATE_GRAD = 1e-5, 5e-4, 1e-3


def _rel(a, b, floor):
    return np.abs(np.asarray(a, np.float64) - b) / (np.abs(b) + floor)


def run_case(name, nworld, substeps, fd_worlds, eps, seed):
    import mujoco
    import jax
    import jax.numpy as jnp
    from nablaengine import engines, jax_ffi
    from nablaengine.model import capability
    xml, lim_name, vs, us = CASES[name]
    m = mujoco.MjModel.from_xml_string(xml)
    res = dict(case=name, nq=int(m.nq), nv=int(m.nv), nu=int(m.nu), substeps=substeps, nworld=nworld)
    free = engines.name(m) == engines.G1
    if free:
        try:
            jax_ffi.init(model=m, foot_pairs=[], com_body=None)
            res["tree_refusal"] = None
        except capability.UnsupportedModel as e:
            res["tree_refusal"] = sorted({c for c in ("H11a", "H11v") if c in str(e)})
        engines.prefer_forest(m)
    rng = np.random.default_rng(seed)
    qpos = np.tile(m.qpos0, (nworld, 1))
    hinge_q = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    qpos[:, hinge_q] = rng.uniform(-0.6, 0.6, (nworld, len(hinge_q)))
    if free:
        qv = rng.normal(size=(nworld, 4))
        qpos[:, 3:7] = qv / np.linalg.norm(qv, axis=1, keepdims=True)
    qvel = rng.uniform(-vs, vs, (nworld, m.nv))
    ctrl = rng.uniform(-us, us, (nworld, m.nu))
    lim = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, lim_name)
    frc = float(m.actuator_forcerange[lim, 1])
    d = mujoco.MjData(m)

    def ref(qp, qv, u, track=False):
        d.qpos[:], d.qvel[:], d.ctrl[:] = qp, qv, u
        mujoco.mj_forward(m, d)
        hit = False
        for _ in range(substeps):
            mujoco.mj_step(m, d)
            hit |= track and abs(abs(float(d.actuator_force[lim])) - frc) < 1e-12
        return d.qpos.copy(), d.qvel.copy(), hit

    R = [ref(qpos[b], qvel[b], ctrl[b], True) for b in range(nworld)]
    rq, rv = np.array([r[0] for r in R]), np.array([r[1] for r in R])
    clamped = np.array([r[2] for r in R])
    info = jax_ffi.init(model=m, foot_pairs=[], com_body=None)
    res.update(engine=info["engine"], actuator=info.get("actuator"), clamped_worlds=int(clamped.sum()))
    f32 = lambda x: jnp.asarray(x, jnp.float32)
    B = nworld
    com = jnp.zeros((B, info["com_words"]), jnp.float32)
    fr = jnp.ones((B,), jnp.float32)
    warm = jnp.zeros((B, m.nv), jnp.float32)
    _, v1, _, _ = jax_ffi.forward(f32(qpos), f32(qvel), warm, f32(ctrl), fr, com, substeps=substeps)
    ev = _rel(v1, rv, 1e-2)
    res["forward_qvel"] = dict(median=float(np.median(ev)), max=float(ev.max()))
    wv, wq = rng.normal(size=m.nv), rng.normal(size=m.nq)
    qmask = np.zeros(m.nq)
    qmask[hinge_q] = 1.0
    wq = wq * qmask

    def L64(qp, qv, u):
        a, b, _ = ref(qp, qv, u)
        return float(wv @ b + wq @ a)

    F = min(fd_worlds, B)
    fd = {"ctrl": [], "qvel": [], "qpos": []}
    for b in range(F):
        fd["ctrl"].append([(L64(qpos[b], qvel[b], ctrl[b] + eps * e) - L64(qpos[b], qvel[b], ctrl[b] - eps * e))
                           / (2 * eps) for e in np.eye(m.nu)])
        fd["qvel"].append([(L64(qpos[b], qvel[b] + eps * e, ctrl[b]) - L64(qpos[b], qvel[b] - eps * e, ctrl[b]))
                           / (2 * eps) for e in np.eye(m.nv)])
        fd["qpos"].append([(L64(qpos[b] + eps * np.eye(m.nq)[i], qvel[b], ctrl[b])
                            - L64(qpos[b] - eps * np.eye(m.nq)[i], qvel[b], ctrl[b])) / (2 * eps) for i in hinge_q])
    fd = {k: np.array(v) for k, v in fd.items()}
    for ckv in (1, 0):
        step = jax_ffi.make_step(substeps=substeps, clamp_kv=ckv)

        def loss(qp, qv, u):
            a, b_, _ = step(qp, qv, warm, u, fr, com)
            return jnp.sum(b_ @ f32(wv)) + jnp.sum(a @ f32(wq))
        gq, gv, gu = jax.grad(loss, argnums=(0, 1, 2))(f32(qpos), f32(qvel), f32(ctrl))
        g = {"ctrl": np.asarray(gu)[:F], "qvel": np.asarray(gv)[:F], "qpos": np.asarray(gq)[:F][:, hinge_q]}
        out = {}
        for k in g:
            e = _rel(g[k], fd[k], 1e-3)
            free_w, clamp_w = e[~clamped[:F]], e[clamped[:F]]
            out[k] = dict(median=float(np.median(e)), max=float(e.max()),
                          max_unclamped=float(free_w.max()) if free_w.size else None,
                          max_clamped=float(clamp_w.max()) if clamp_w.size else None)
        res[f"grad_clamp_kv{ckv}"] = out
    res["fd_worlds"], res["fd_clamped_worlds"] = F, int(clamped[:F].sum())
    return res


def gates(r):
    ok = r["forward_qvel"]["median"] <= GATE_FWD_MED and r["forward_qvel"]["max"] <= GATE_FWD_MAX
    ok &= all(v["max"] <= GATE_GRAD for v in r["grad_clamp_kv1"].values())
    ok &= all((v["max_unclamped"] or 0.0) <= GATE_GRAD for v in r["grad_clamp_kv0"].values())
    ok &= 0 < r["fd_clamped_worlds"] < r["fd_worlds"] and r["engine"] == "forest"
    if "tree_refusal" in r:
        ok &= r["tree_refusal"] == ["H11a", "H11v"]
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="all", choices=["all"] + sorted(CASES))
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--fd-worlds", type=int, default=24)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="result JSON (not written by default)")
    ap.add_argument("--json-line", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.case == "all":
        results = []
        for c in sorted(CASES):
            cmd = [sys.executable, __file__, "--case", c, "--json-line", "--nworld", str(a.nworld), "--substeps",
                   str(a.substeps), "--fd-worlds", str(a.fd_worlds), "--eps", str(a.eps), "--seed", str(a.seed)]
            p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))
            line = [x for x in p.stdout.splitlines() if x.startswith("{")]
            if p.returncode not in (0, 1) or not line:
                print(p.stdout[-2000:], p.stderr[-4000:])
                results.append(dict(case=c, error=f"exit {p.returncode}", ok=False))
                continue
            results.append(json.loads(line[-1]))
    else:
        r = run_case(a.case, a.nworld, a.substeps, a.fd_worlds, a.eps, a.seed)
        r["ok"] = gates(r)
        if a.json_line:
            print(json.dumps(r))
            return 0 if r["ok"] else 1
        results = [r]
    from nablaengine import provenance
    for r in results:
        if "error" in r:
            print(f"{r['case']:9s} ERROR {r['error']}")
            continue
        fw = r["forward_qvel"]
        print(f"{r['case']:9s} engine {r['engine']} nv {r['nv']} clamped worlds {r['clamped_worlds']}/{r['nworld']} "
              f"(fd {r['fd_clamped_worlds']}/{r['fd_worlds']}) tree refusal {r.get('tree_refusal', '-')}")
        print(f"          forward qvel median {fw['median']:.2e} max {fw['max']:.2e}")
        for ckv in (1, 0):
            g = r[f"grad_clamp_kv{ckv}"]
            print(f"          clamp_kv={ckv} " + "  ".join(
                f"{k} med {v['median']:.1e} max {v['max']:.1e} (unclamped {v['max_unclamped'] or 0:.1e}, clamped "
                f"{v['max_clamped'] or 0:.1e})" for k, v in g.items()))
        print(f"          {'PASS' if r['ok'] else 'FAIL'}")
    ok = all(r["ok"] for r in results)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(results=results, ok=ok, provenance=provenance.provenance()),
                                          indent=1, default=str) + "\n")
    print("velocity servo check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

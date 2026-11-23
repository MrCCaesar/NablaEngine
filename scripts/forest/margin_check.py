"""Contact margins on the forest engine (forest_engine.margin): forward against MuJoCo Warp and float64 MuJoCo C, exact
gradients against float64 MuJoCo C central differences.

Model: scripts/forest/models/margin_prims.xml (free bodies on a floor, one explicit pair per margin-capable primitive:
plane against sphere, capsule, box, ellipsoid, cylinder; sphere-sphere with a gap, sphere-capsule, capsule-capsule
parallel and crossed, sphere-cylinder; margins 4-12 mm), or --model (an engine-ready MJCF / .mjb with margins): the
convex pairs' unit scenes (forest_engine.margin_convex) margin_convex.xml (box-box, sphere-, capsule-box,
capsule-cylinder, cylinder-cylinder, cylinder-box, box-, capsule-, ellipsoid-, cylinder-ellipsoid), margin_mesh.xml
(plane-mesh, a mesh with a capsule, mesh, box, sphere, cylinder, ellipsoid; box-mesh) and margin_boxprim.xml (box-box
by the separating-axis primitive, NATIVECCD disabled): stacks of two free bodies on the floor, pair margins 6-9 mm,
resting in their margin bands; run them with --max-contacts 64 (some 40 contacts at rest).
States: scripts/eq/connect_check.py's corpus (MuJoCo C rollouts with random velocities; many contacts in the margin
band). One engine library per process.

  fwd   one control step of 1 and 4 substeps from a zero warm start: qvel relative error per world (|dv| / max(|v|, 1),
        max over coordinates) of the engine against MuJoCo Warp (scripts/eq/mjw_step.py run by --mjw-python, a Python
        with mujoco_warp; skipped without it) and against MuJoCo C; contact counts of MuJoCo C and MuJoCo Warp
  fd    the VJP of one control step (--substeps) in the exact mode (clamp_kv 1, normal 0) against central differences
        of MuJoCo C (scripts/eq/connect_grad.py's fd_grad: tight solver, eps 1e-6), relative error per world

  PYTHONPATH=src python scripts/forest/margin_check.py fwd --mjw-python <venv>/bin/python --out runs/margin/fwd.json
  PYTHONPATH=src python scripts/forest/margin_check.py fd --out runs/margin/fd.json

Exit status 1 when a median exceeds its bound (fwd: engine vs MuJoCo Warp 1e-4 with --mjw-python, the acceptance
reference; without it engine vs MuJoCo C 1e-4, which the convex / mesh scenes cannot meet: MuJoCo C uses another
box-box contact model inside the margin band and MuJoCo Warp vs C is itself ~1e-3 there; fd: 1e-3).
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

MODEL = REPO / "scripts" / "forest" / "models" / "margin_prims.xml"


def load(path):
    p = str(path)
    return mujoco.MjModel.from_binary_path(p) if p.endswith(".mjb") else mujoco.MjModel.from_xml_path(p)


def c_steps(m, q, v, u, n):
    d = mujoco.MjData(m)
    V, N = [], []
    for w in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = q[w], v[w], u[w]
        for _ in range(n):
            mujoco.mj_step(m, d)
        V.append(d.qvel.copy())
        N.append(d.ncon)
    return np.array(V), np.array(N)


def mjw(m, q, v, u, n, py):
    with tempfile.TemporaryDirectory() as tmp:
        mujoco.mj_saveModel(m, os.path.join(tmp, "m.mjb"), None)
        np.savez(os.path.join(tmp, "s.npz"), qpos=q, qvel=v, ctrl=u)
        r = subprocess.run([py, "-I", str(REPO / "scripts" / "eq" / "mjw_step.py"), os.path.join(tmp, "m.mjb"),
                            os.path.join(tmp, "s.npz"), os.path.join(tmp, "o.npz"), "--substeps", str(n), "--iterations",
                            str(int(m.opt.iterations)), "--nconmax", "128"], capture_output=True, text=True,
                           env={k: x for k, x in os.environ.items() if k != "PYTHONPATH"})
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
        z = np.load(os.path.join(tmp, "o.npz"))
        return z["qvel"], z["ncon"], json.loads(r.stdout.strip().splitlines()[-1])


def engine_init(m, max_contacts=None):
    """cc.engine_init with a contact capacity per world (None: step.maxcon's default; the convex unit scenes rest on
    some 40 contacts in their margin bands, beyond the default 32 of a model with at most 64 pairs)."""
    if not max_contacts:
        return cc.engine_init(m)
    from nablaengine import engines, jax_ffi
    engines.prefer_forest(m)
    return jax_ffi.init(model=m, foot_pairs=[], max_contacts=max_contacts)


def cmd_fwd(a, m):
    q, v, u = cc.corpus(m, a.states, a.seed)
    info = engine_init(m, a.max_contacts)
    out, ok = {}, True
    for n in (1, 4):
        _, ve, _ = cc.engine_step(m, info, q, v, u, substeps=n)
        ve = np.asarray(ve, np.float64)
        vc, nc = c_steps(m, q, v, u, n)
        r = dict(engine_vs_c=cc.stats(cc.rel(ve, vc)), contacts_c=dict(mean=float(nc.mean()), max=int(nc.max())))
        if a.mjw_python:
            vw, nw, ver = mjw(m, q, v, u, n, a.mjw_python)
            r.update(engine_vs_mjw=cc.stats(cc.rel(ve, vw)), mjw_vs_c=cc.stats(cc.rel(vw, vc)),
                     contacts_mjw=dict(mean=float(nw.mean()), max=int(nw.max())),
                     contact_count_worlds_equal=int((nw == nc).sum()), versions=ver)
        gate = "engine_vs_mjw" if a.mjw_python else "engine_vs_c"
        r["gate"] = gate
        ok &= r[gate]["median"] < 1e-4
        out[f"substeps_{n}"] = r
    return out, ok


def cmd_fd(a, m):
    q, v, u = cc.corpus(m, a.states, a.seed)
    info = engine_init(m, a.max_contacts)
    mm = cg.tight(m)
    rng = np.random.default_rng(a.seed + 7)
    S = a.substeps
    gq = rng.normal(size=(len(q), m.nq))
    gq[:, cg.free_quats(m)] = 0.0
    gv = rng.normal(size=(len(q), m.nv))
    _, _, (bq, bv, bu) = cg.engine_vjp(m, info, q, v, u, gq, gv, S, clamp_kv=1, normal=0)
    e, nc = [], []
    dd = mujoco.MjData(mm)
    for w in range(len(q)):
        q64, v64, u64 = (x[w].astype(np.float64) for x in (q, v, u))
        ge = np.concatenate([cg.q_to_tangent(m, q64, bq[w]), bv[w], bu[w]])
        e.append(cg.err(ge, cg.fd_grad(mm, q64, v64, u64, gq[w], gv[w], S)))
        nc.append(cg.rollout(mm, dd, q64, v64, u64, S)[2])
    out = dict(substeps=S, engine_vs_fd=cc.stats(e), contacts_max=int(max(nc)))
    return out, out["engine_vs_fd"]["median"] < 1e-3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("fwd", "fd"))
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-contacts", type=int, default=0, help="contact capacity per world (0: the engine's default)")
    ap.add_argument("--mjw-python", default="", help="a Python with mujoco_warp (3.17) for the fwd reference")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    m = load(a.model)
    res, ok = (cmd_fwd if a.cmd == "fwd" else cmd_fd)(a, m)
    from nablaengine import provenance
    mp = Path(a.model).resolve()
    rep = dict(command=a.cmd, model=str(mp.relative_to(REPO)) if mp.is_relative_to(REPO) else mp.name,
               states=a.states, seed=a.seed, result=res, provenance=provenance.provenance())
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

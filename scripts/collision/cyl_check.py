"""A model with cylinder contacts in the forest engine (forest_engine.cylinder), against float64 MuJoCo C and MuJoCo
Warp (the forward's reference for contacts):"""
import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eq"))
import connect_check as cc


def c_pair_types(m, q):
    """MuJoCo C's contacts at the states, by sorted geom-type pair name."""
    d = mujoco.MjData(m)
    out, ncon = Counter(), []
    for w in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q[w]
        mujoco.mj_forward(m, d)
        ncon.append(d.ncon)
        for i in range(d.ncon):
            a, b = sorted(mujoco.mjtGeom(int(m.geom_type[g])).name[7:].lower() for g in d.contact.geom[i])
            out[f"{a}-{b}"] += 1
    return dict(sorted(out.items())), np.array(ncon)


def zero_margins(m):
    """The source model's contact margins / gaps and joint-limit margins set to 0, as the converter does
    (to_engine's notes): the references then run the model the engine runs. Returns whether any was nonzero."""
    any_ = bool(m.geom_margin.any() or m.geom_gap.any() or m.pair_margin.any() or m.pair_gap.any() or m.jnt_margin.any())
    for x in (m.geom_margin, m.geom_gap, m.pair_margin, m.pair_gap, m.jnt_margin):
        x[:] = 0
    return any_


def run_mjw(m, q, v, u, py, work):
    import subprocess
    work = Path(work)
    mujoco.mj_saveModel(m, str(work / "model.mjb"))
    np.savez(work / "states.npz", qpos=q, qvel=v, ctrl=u)
    script = Path(__file__).resolve().parents[1] / "eq" / "mjw_step.py"
    r = subprocess.run([py, str(script), str(work / "model.mjb"), str(work / "states.npz"), str(work / "mjw.npz"),
                        "--substeps", "1", "--iterations", str(int(m.opt.iterations)), "--nconmax", "256"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ""})
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return dict(np.load(work / "mjw.npz")), json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mjw-python", default=None)
    ap.add_argument("--states", type=int, default=256)
    ap.add_argument("--rollout-worlds", type=int, default=16)
    ap.add_argument("--rollout-steps", type=int, default=500)
    ap.add_argument("--grad-substeps", type=int, default=4)
    ap.add_argument("--no-fluid", action="store_true")
    ap.add_argument("--uniform-limits", action="store_true",
                    help="every joint limit on the first set (connect_check.py; earlier runs only: the engine reads per-joint sets)")
    ap.add_argument("--drop-spatial", action="store_true")
    ap.add_argument("--reference", choices=("converted", "source"), default="converted",
                    help="the model MuJoCo C / Warp run: the converted one (default) or the source with zero margins")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    cv = cc.load(a.model, None, a.no_fluid, a.uniform_limits, a.drop_spatial)
    m, mo = cv.model, cv.original
    zeroed = zero_margins(mo)
    if a.reference == "converted":
        mo = m
    root = Path(__file__).resolve().parents[2]
    pm = Path(a.model).resolve()
    name = os.path.relpath(pm, root) if str(pm).startswith(str(root)) else "/".join(pm.parts[-2:])
    cyl = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    rep = dict(model=name, nv=int(m.nv), nbody=int(m.nbody), npair=int(m.npair),
               cylinder_pairs=int(sum(cyl in (int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]]))
                                      for p in range(m.npair))),
               uniform_limits=a.uniform_limits, no_fluid=a.no_fluid, drop_spatial=a.drop_spatial,
               reference=a.reference, source_margins_nonzero=zeroed, notes=list(cv.notes)[:20],
               mujoco=mujoco.__version__, seed=a.seed, states=a.states)
    q, v, u = cc.corpus(mo, a.states, a.seed)
    info = cc.engine_init(m)
    rep["maxcon"], rep["nefc"] = int(info.get("maxcon", -1)), int(info.get("nefc", -1))
    B = len(q)
    cw = int(info.get("com_words", 3))
    z = jnp.zeros((B, m.nv), jnp.float32)
    fr, com = jnp.ones((B,), jnp.float32), jnp.zeros((B, cw), jnp.float32)
    o = jax_ffi.forward(jnp.asarray(q), jnp.asarray(v), z, jnp.asarray(u), fr, com, substeps=1,
                        iterations=int(m.opt.iterations), contacts=True)
    ve, inf = np.asarray(o[1], np.float64), np.asarray(o[3])
    ne = np.asarray(jax_ffi.contact_table(o[4])["valid"]).sum(axis=1)
    qc, vc, ac = cc.c_step(mo, q, v, u)
    types, nc = c_pair_types(mo, q)
    rep["c_contacts_by_type"] = types
    rep["forward_vs_C"] = dict(qvel=cc.stats(cc.rel(ve, vc)), contacts_equal=float(np.mean(ne == nc)),
                               contacts_engine_mean=float(ne.mean()), contacts_C_mean=float(nc.mean()))
    rep["unsupported_worlds"] = int((inf[:, 3] != 0).sum())
    if a.mjw_python:
        with tempfile.TemporaryDirectory(prefix="cyl_check_") as work:
            zz, ver = run_mjw(mo, q, v, u, a.mjw_python, work)
        rep["mjw"] = ver
        rep["forward_vs_mjw"] = dict(qvel=cc.stats(cc.rel(ve, zz["qvel"])),
                                     contacts_equal=float(np.mean(ne == zz["ncon"])),
                                     contacts_mjw_mean=float(np.mean(zz["ncon"])))
        rep["mjw_vs_C"] = dict(qvel=cc.stats(cc.rel(zz["qvel"], vc)), contacts_equal=float(np.mean(zz["ncon"] == nc)))
    nr = min(a.rollout_worlds, B)
    rng = np.random.default_rng(a.seed + 1)
    lo, hi = cc.ctrl_range(mo)
    ur = jnp.asarray(rng.uniform(lo, hi, (nr, mo.nu)).astype(np.float32))
    qq, vv, ww = jnp.asarray(q[:nr]), jnp.asarray(v[:nr]), jnp.zeros((nr, m.nv), jnp.float32)
    finite, unsup, cmax = True, np.zeros(nr, bool), 0
    for _ in range(a.rollout_steps // 10):
        o = jax_ffi.forward(qq, vv, ww, ur, fr[:nr], com[:nr], substeps=10, iterations=int(m.opt.iterations))
        qq, vv, ww = o[0], o[1], o[2]
        ii = np.asarray(o[3])
        finite &= bool(np.isfinite(np.asarray(qq)).all() and np.isfinite(np.asarray(vv)).all())
        unsup |= ii[:, 3] != 0
        cmax = max(cmax, int(ii[:, 1].max()))
    rep["rollout"] = dict(worlds=nr, steps=a.rollout_steps, finite=finite, unsupported_worlds=int(unsup.sum()),
                          contacts_max=cmax)
    S = a.grad_substeps
    rg = np.random.default_rng(a.seed + 7)
    gq = rg.normal(size=(B, m.nq)).astype(np.float32)
    gv = rg.normal(size=(B, m.nv)).astype(np.float32)

    def f(qq_, vv_, uu_):
        r = jax_ffi.step(qq_, vv_, z, uu_, fr, com, substeps=S, iterations=int(m.opt.iterations))
        return r[0], r[1]
    try:
        _, pull = jax.vjp(f, jnp.asarray(q), jnp.asarray(v), jnp.asarray(u))
        g = [np.asarray(x) for x in pull((jnp.asarray(gq), jnp.asarray(gv)))]
        tape = cc_tape(m, info, q, v, u, S)
        rep["backward"] = dict(substeps=S, finite_worlds=int(np.all([np.isfinite(x).all(axis=1) for x in g if x.size],
                                                                    axis=0).sum()), worlds=B, tape_bitwise=tape)
    except Exception as e:
        rep["backward"] = dict(substeps=S, error=f"{type(e).__name__}: {str(e)[:300]}")
    d = jax_ffi.read_diagnostics(reset=True)
    rep["diagnostics"] = {k: int(d[k]) for k in ("fwd_unsupported_worlds", "replay_mismatch_worlds", "nonfinite_rows",
                                                 "fwd_contact_overflow_worlds", "fwd_row_overflow_worlds")}
    txt = json.dumps(rep, indent=1)
    print(txt)
    if a.out:
        out = Path(a.out)
        if out.exists():
            sys.exit(f"{out} exists (not overwritten)")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt)


def cc_tape(m, info, q, v, u, S):
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    B = len(q)
    cw = int(info.get("com_words", 3))
    args = (jnp.asarray(q), jnp.asarray(v), jnp.zeros((B, m.nv), jnp.float32), jnp.asarray(u),
            jnp.ones((B,), jnp.float32), jnp.zeros((B, cw), jnp.float32))
    a = jax_ffi.step(*args, substeps=S, iterations=int(m.opt.iterations))
    o = jax_ffi.forward(*args, substeps=S, tape=True, iterations=int(m.opt.iterations))
    return bool(np.array_equal(np.asarray(o[0]), np.asarray(a[0])) and np.array_equal(np.asarray(o[1]), np.asarray(a[1])))


if __name__ == "__main__":
    main()

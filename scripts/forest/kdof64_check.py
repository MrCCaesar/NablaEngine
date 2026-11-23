"""Contact supports of more than 32 dofs in the forest engine (forest_engine.kdof64): unit scenes against MuJoCo Warp
3.17 and float64 MuJoCo C, exact gradients against float64 MuJoCo C differences.

Scenes (SCENES, generated MJCF: hinge chains of capsules lying on a floor plane, ball "feet" at some joints carry them;
explicit pairs only; implicitfast, joint springs and damping, motors on a few hinges, a few joint limits):
  floor40    one chain, free root + 34 hinges (nv 40, 36 bodies): feet against the floor, supports up to 40
  cross40    two chains (free root + 14 hinges each, 31 bodies), B's tip lying across A's tip: the tip pairs' support
             is every dof of both chains (40), the floor pairs' up to 20
  cross40c4  cross40 with condim 4 on the tip pairs (the torsional row)
  cross40e   cross40 with the elliptic cone
  cross64    two chains of free root + 26 hinges (nv 64, 55 bodies): tip support 64 (the capacity)
  nv128      two chains of free root + 24 hinges crossing (tip support 60) and a fixed-base chain of 20 hinges on the
             floor (nv 80, 71 bodies: the nv128 and nb128 paths)
  deep70     floor40's chain (40 dofs deep) and a second one (free root + 24 hinges): nv 70 with a dof 39 ancestors
             deep (the dense factors of the dense factor: nv128.tree_ok), floor supports up to 40
  floor40s, cross32   controls of the same construction with supports of at most 32 (no kdof64 edits): floor40 with
             feet up to link 24 (support <= 30), two chains of free root + 10 hinges (tip support 32)
States: scripts/eq/connect_check.py's corpus (MuJoCo C rollouts with random velocities and controls). One engine
library per process: one scene per run (--scene), or --all (subprocesses).

  fwd   one control step of 1 and 4 substeps from a zero warm start: qvel relative error per world (|dv| / max(|v|, 1))
        of the engine against MuJoCo Warp (scripts/eq/mjw_step.py) and MuJoCo C; the tape's forward bitwise the plain
        step's; the support sizes of the contacts MuJoCo C has at the states
  fd    the VJP of one control step (--substeps) in the exact mode (clamp_kv 1, normal 0) against float64 MuJoCo C:
        central differences (connect_grad.fd_grad, tight solver, eps 1e-6) and the active-set-consistent reference
        (a qpos / qvel column whose 1e-6 and 1e-7 central differences disagree crosses a kink: the one-sided 1e-7
        differences that agree with the 1e-8 ones, the nearer side), relative error per world (max over columns /
        max |reference|)

  PYTHONPATH=src python scripts/forest/kdof64_check.py fwd --all --mjw-python <venv>/bin/python --out runs/x/fwd.json
  PYTHONPATH=src python scripts/forest/kdof64_check.py fd --all --out runs/x/fd.json

Exit status 1 when a median exceeds its bound (fwd: engine vs MuJoCo C 1e-4; fd: sided 1e-3).
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
sys.path.insert(0, str(REPO / "scripts" / "forest"))
import connect_check as cc
import connect_grad as cg

R, L, RF = 0.02, 0.07, 0.024


def _chain(name, n, pos, yaw, free=True, feet=(), motors=(), limits=()):
    c, s = np.cos(yaw), np.sin(yaw)
    out, links, fts, acts = [], [], [], []
    q = f"{np.cos(yaw / 2):.9f} 0 0 {np.sin(yaw / 2):.9f}"
    out.append(f'<body name="{name}0" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" quat="{q}">')
    out.append('<freejoint/>' if free else
               f'<joint name="{name}j0" type="hinge" axis="0 0 1" damping="0.05" armature="0.002"/>')
    for k in range(n + 1):
        if k:
            ax = "0 0 1" if k % 2 else "0 1 0"
            lim = ' limited="true" range="-0.35 0.35"' if k in limits else ""
            out.append(f'<body name="{name}{k}" pos="{L} 0 0"><joint name="{name}j{k}" type="hinge" axis="{ax}" '
                       f'stiffness="0.6" damping="0.02" armature="0.002"{lim}/>')
            if k in motors:
                acts.append(f'<motor joint="{name}j{k}" gear="1" ctrlrange="-0.3 0.3" ctrllimited="true"/>')
        out.append(f'<geom name="{name}g{k}" type="capsule" fromto="0 0 0 {L} 0 0" size="{R}"/>')
        links.append(f"{name}g{k}")
        if k in feet:
            out.append(f'<geom name="{name}f{k}" type="sphere" size="{RF}" pos="0 0 {R - RF:.4f}"/>')
            fts.append(f"{name}f{k}")
    out.append("</body>" * (n + 1))
    return "\n".join(out), "\n".join(acts), links, fts


def scene_xml(name):
    cone = ' cone="elliptic"' if name.endswith("e") else ""
    bodies, acts, pairs = [], [], []
    z = RF
    if name.startswith("floor40") or name == "deep70":
        feet = (0, 4, 9, 14, 19, 24) if name == "floor40s" else (0, 4, 9, 14, 19, 24, 29, 33, 34)
        b, a, links, fts = _chain("a", 34, (0, 0, z), 0.0, feet=feet,
                                  motors=(3, 11, 20, 30), limits=(5, 17))
        bodies.append(b); acts.append(a)
        pairs += [f'<pair geom1="floor" geom2="{g}"/>' for g in fts]
        if name == "deep70":
            b, a, links, fts = _chain("b", 24, (0, 0.4, z), 0.0, feet=(0, 6, 12, 18, 24), motors=(5,))
            bodies.append(b); acts.append(a)
            pairs += [f'<pair geom1="floor" geom2="{g}"/>' for g in fts]
    else:
        n = {"cross64": 26, "nv128": 24, "cross32": 10}.get(name, 14)
        tip = (n + 1) * L
        ax = tip - 0.5 * L
        b1, a1, l1, f1 = _chain("a", n, (0, 0, z), 0.0, feet=tuple(range(0, n, 4)) + (n - 1,), motors=(3, n // 2),
                                limits=(5,))
        b2, a2, l2, f2 = _chain("b", n, (ax, -tip + 0.5 * L, z + 2 * R), np.pi / 2,
                                feet=tuple(range(0, n - 2, 4)), motors=(2, n - 3), limits=(7,))
        bodies += [b1, b2]; acts += [a1, a2]
        pairs += [f'<pair geom1="floor" geom2="{g}"/>' for g in f1 + f2]
        dim = ' condim="4" friction="1 1 0.01 0.0001 0.0001"' if name == "cross40c4" else ""
        pairs += [f'<pair geom1="{x}" geom2="{y}"{dim}/>' for x in l1[-3:] for y in l2[-3:]]
        if name == "nv128":
            b3, a3, l3, f3 = _chain("c", 19, (0, -0.6, z), 0.0, free=False, feet=(0, 5, 10, 15, 19), motors=(4,))
            bodies.append(b3); acts.append(a3)
            pairs += [f'<pair geom1="floor" geom2="{g}"/>' for g in f3]
    return f"""<mujoco model="kdof64_{name}">
<option timestep="0.002" integrator="implicitfast"{cone}/>
<default><geom contype="0" conaffinity="0" density="600"/></default>
<worldbody>
<geom name="floor" type="plane" size="0 0 1"/>
{chr(10).join(bodies)}
</worldbody>
<contact>
{chr(10).join(pairs)}
</contact>
<actuator>
{chr(10).join(x for x in acts if x)}
</actuator>
</mujoco>
"""


SCENES = ("floor40", "cross40", "cross40c4", "cross40e", "cross64", "nv128", "deep70", "floor40s", "cross32")


def corpus(m, n, seed, vel=0.1, t_min=0.1, every=10):
    """connect_check.corpus with initial velocities N(0, vel) (its 0.5 rad/s per hinge throws the long chains' tips
    apart): MuJoCo C rollouts from qpos0 with random piecewise-constant controls, float32 states after t_min."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    lo, hi = cc.ctrl_range(m)
    Q, V, U = [], [], []
    while len(Q) < n:
        mujoco.mj_resetData(m, d)
        d.qvel[:] = rng.normal(0, vel, m.nv)
        t, k = 0.0, 0
        while t < t_min + 0.4 and len(Q) < n:
            if k % 50 == 0:
                d.ctrl[:] = rng.uniform(lo, hi)
            mujoco.mj_step(m, d)
            t, k = t + m.opt.timestep, k + 1
            if t >= t_min and k % every == 0:
                Q.append(d.qpos.copy()); V.append(d.qvel.copy()); U.append(d.ctrl.copy())
    f = lambda x: np.asarray(x, np.float32)
    return f(Q), f(V), f(U)


def load_scene(name):
    from nablaengine.envs import convert
    return convert.to_engine(scene_xml(name))


def c_steps(m, q, v, u, n):
    d = mujoco.MjData(m)
    V, N = [], []
    for w in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = q[w], v[w], u[w]
        for _ in range(n):
            mujoco.mj_step(m, d)
        V.append(d.qvel.copy()); N.append(d.ncon)
    return np.array(V), np.array(N)


def support_hist(m, q):
    """The support sizes (dofs on exactly one chain) of the contacts MuJoCo C has at the states: {size: count}."""
    from nablaengine.forest_engine import step
    dofbody, mask = step.extra_constants(m)
    d = mujoco.MjData(m)
    h = {}
    for w in range(len(q)):
        d.qpos[:] = q[w]
        mujoco.mj_forward(m, d)
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            k = sum(1 for i in range(m.nv) if ((mask[b1] >> int(dofbody[i])) ^ (mask[b2] >> int(dofbody[i]))) & 1)
            h[k] = h.get(k, 0) + 1
    return dict(sorted(h.items()))


def cmd_fwd(a, cv):
    from spatial_check import mjw_steps
    from nablaengine.forest_engine import step
    m = cv.model
    q, v, u = corpus(cv.original, a.states, a.seed)
    info = cc.engine_init(m)
    out, ok = dict(nv=int(m.nv), nbody=int(m.nbody), npair=int(m.npair), max_support=int(step.max_support(m)),
                   contact_supports=support_hist(m, q)), True
    for n in (1, 4):
        _, ve, _ = cc.engine_step(m, info, q, v, u, substeps=n)
        ve = np.asarray(ve, np.float64)
        vc, nc = c_steps(m, q, v, u, n)
        r = dict(engine_vs_c=cc.stats(cc.rel(ve, vc)), finite=bool(np.isfinite(ve).all()),
                 ncon_c=dict(mean=float(nc.mean()), max=int(nc.max())))
        if a.mjw_python:
            vw = mjw_steps(m, q, v, u, n, a.mjw_python)
            r.update(engine_vs_mjw=cc.stats(cc.rel(ve, vw)), mjw_vs_c=cc.stats(cc.rel(vw, vc)))
        ok &= r["engine_vs_c"]["median"] < 1e-4 and r["finite"]
        out[f"substeps_{n}"] = r
    out["tape_forward_bitwise"] = bool(cg.tape_bitwise(m, info, q, v, u, 4))
    ok &= out["tape_forward_bitwise"]
    return out, ok


def _sided(L, ge, i, L0, f):
    c6 = (f[1e-6] - f[-1e-6]) / 2e-6
    c7 = (f[1e-7] - f[-1e-7]) / 2e-7
    if abs(c6 - c7) <= 1e-4 * max(abs(c6), 1.0):
        return c6, c6, False
    sides = []
    for sg in (1, -1):
        o7 = sg * (f[sg * 1e-7] - L0) / 1e-7
        o8 = sg * (L(i, sg * 1e-8) - L0) / 1e-8
        if abs(o7 - o8) <= 1e-3 * max(abs(o7), 1.0):
            sides.append(o7)
    return c6, (min(sides, key=lambda x: abs(x - ge)) if sides else np.nan), True


def cmd_fd(a, cv):
    m = cv.model
    q, v, u = corpus(cv.original, a.states, a.seed)
    info = cc.engine_init(m)
    mm = cg.tight(cv.original)
    d = mujoco.MjData(mm)
    out, ok = dict(nv=int(m.nv)), True
    for S in a.substeps:
        rng = np.random.default_rng(a.seed + 7)
        gq = rng.normal(size=(len(q), m.nq))
        gq[:, cg.free_quats(m)] = 0.0
        gv = rng.normal(size=(len(q), m.nv))
        _, _, (bq, bv, bu) = cg.engine_vjp(m, info, q, v, u, gq, gv, S, clamp_kv=1, normal=0)
        rows = []
        for w in range(len(q)):
            q64, v64, u64 = (x[w].astype(np.float64) for x in (q, v, u))
            ge = np.concatenate([cg.q_to_tangent(m, q64, bq[w]), bv[w], bu[w]])

            def Lf(i, h):
                qq, vv, uu = q64.copy(), v64.copy(), u64.copy()
                if h and i < m.nv:
                    dv = np.zeros(m.nv); dv[i] = 1.0
                    mujoco.mj_integratePos(mm, qq, dv, h)
                elif h and i < 2 * m.nv:
                    vv[i - m.nv] += h
                elif h:
                    uu[i - 2 * m.nv] += h
                qn, vn, _ = cg.rollout(mm, d, qq, vv, uu, S)
                return float(gq[w] @ qn + gv[w] @ vn)
            L0 = Lf(0, 0)
            nc = 2 * m.nv + m.nu
            c6, ref, kink = np.zeros(nc), np.zeros(nc), np.zeros(nc, bool)
            for i in range(nc):
                f = {h: Lf(i, h) for h in (1e-6, -1e-6, 1e-7, -1e-7)}
                c6[i], ref[i], kink[i] = _sided(Lf, ge[i], i, L0, f)
            okc = ~np.isnan(ref)
            if not okc.any() or not np.isfinite(c6).all():
                rows.append(dict(central=np.nan, sided=np.nan, q=np.nan, v=np.nan, kink_cols=int(kink.sum()),
                                 kink_unresolved=int((~okc).sum()), ncon=-1, reference_failed=True))
                print(S, w, "reference failed", file=sys.stderr, flush=True)
                continue
            sc = np.abs(ref[okc]).max()
            dev = np.abs(ge - ref)
            blk = lambda sl: float(dev[sl][okc[sl]].max() / sc) if okc[sl].any() else np.nan
            rows.append(dict(central=float(np.abs(ge - c6).max() / np.abs(c6).max()),
                             sided=float(dev[okc].max() / sc), q=blk(slice(0, m.nv)), v=blk(slice(m.nv, 2 * m.nv)),
                             kink_cols=int(kink.sum()), kink_unresolved=int((~okc).sum()),
                             ncon=int(cg.rollout(mm, d, q64, v64, u64, S)[2])))
            print(S, w, json.dumps(rows[-1]), file=sys.stderr, flush=True)
        st = lambda k: cc.stats(np.array([r[k] for r in rows if not r.get("reference_failed")
                                          and np.isfinite(r[k])]))
        r = dict(central=st("central"), sided=st("sided"), q=st("q"), v=st("v"),
                 states_with_kink=int(sum(x["kink_cols"] > 0 for x in rows)),
                 reference_failed=int(sum(bool(x.get("reference_failed")) for x in rows)),
                 kink_unresolved_total=int(sum(x["kink_unresolved"] for x in rows)),
                 ncon=dict(mean=float(np.mean([x["ncon"] for x in rows])), max=int(max(x["ncon"] for x in rows))))
        ok &= r["sided"]["median"] < 1e-3
        out[f"substeps_{S}"] = r
    return out, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("fwd", "fd", "xml"))
    ap.add_argument("--scene", default="cross40", choices=SCENES)
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
        res, ok = dict(fwd=cmd_fwd, fd=cmd_fd)[a.cmd](a, cv)
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

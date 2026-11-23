"""The forest engine's collision mid-phase (nablaengine.forest_engine.midphase) against the plain broad phase.

  run   build the forest engine's check binary (engine_step_check.forest_binary_source) of a scene three times: main
        (DK_MIDPHASE=0), main with its contacts sorted by (pair, slot) after the collision (main_sorted, the order the
        mid-phase builds use) and the mid-phase (DK_MIDPHASE=1, or --midphase 2); run them on the same states and check
        (pass criteria) that the mid-phase's contact set equals main's in every world after one substep and that every
        output is bitwise main_sorted's after one substep and after one control step (qpos, qvel, warm start, info
        words, the last substep's contact table); against main itself the worlds that differ (the contact order) and
        the largest qvel difference are reported. --profile: DK_PROFILE builds of main and the mid-phase, the per-phase
        cycles; --time R: R interleaved timing rounds of main and the mid-phase (ms per control step of the forward,
        and of k_pre alone via DK_SPLIT_ONLY=1; the host program's CUDA events, --reps launches per round).
  ffi   the JAX and PyTorch interfaces (jax_ffi / torch_ffi) on the scene, one process each: a control step's outputs
        and the VJP of a random cotangent with respect to (qpos, qvel, ctrl); with the mid-phase PyTorch = JAX bit for
        bit (the pass criterion), and against JAX without it (DK_MIDPHASE=0: main's unsorted contact order) the worlds
        that differ and the largest relative difference per output (float32 order effects where the order differs).

Scenes (--scene): arm (scripts/collision/models/midphase_arm_box.xml: a 7-joint arm with a slide gripper and a free
box), clutter_<n> (the arm scene with n more free boxes of mixed sizes piled in front of it, every box-box, floor-box,
pad-box, hand-box and wrist-box pair explicit: many pairs over few bodies), xml:<path> (any forest model); --full-pairs
adds every pair of collision geoms on two bodies that are not parent and child (full self-contact). States (--states):
random (the first keyframe, else qpos0, with noise on the hinge / slide joints and the free bodies lifted), settle (those
stepped 0.6 s by MuJoCo C: the free bodies at rest on the floor and on each other); or --corpus NPZ with qpos, qvel,
ctrl.

  python scripts/collision/midphase_check.py run --scene clutter_8 --states settle --nworld 1024 --profile --time 5 --out clutter.json
  python scripts/collision/midphase_check.py ffi --scene arm --nworld 256 --out arm_ffi.json

Exit status 1 when an output differs.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
ARM_XML = REPO / "scripts" / "collision" / "models" / "midphase_arm_box.xml"
try:
    sys.path.insert(0, str(REPO / "scripts" / "internal"))
    import midphase_scenes as INTERNAL
except ImportError:
    INTERNAL = None


def clutter_xml(n, seed=0):
    """MJCF text: the arm scene with n more free boxes (half sizes 0.015-0.025 m, masses 0.05-0.15 kg) on a 3 x 3 grid
    in front of the arm, layer above layer, and the explicit pairs floor-box, box-box (all), pad-box, hand-box,
    wrist-box (link6) for each; the keyframe extended with the boxes' poses."""
    import re
    rng = np.random.default_rng(seed)
    x = ARM_XML.read_text()
    bodies, pairs, qk = [], [], []
    for i in range(n):
        h = rng.uniform(0.015, 0.025, 3)
        lay, k = divmod(i, 9)
        p = (0.62 + 0.07 * (k // 3), -0.07 + 0.07 * (k % 3), h[2] + 0.065 * lay + 0.002)
        bodies.append(f'<body name="c{i}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"><freejoint name="c{i}"/>'
                      f'<geom name="c{i}" type="box" size="{h[0]:.4f} {h[1]:.4f} {h[2]:.4f}" '
                      f'mass="{rng.uniform(0.05, 0.15):.3f}"/></body>')
        qk.append(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} 1 0 0 0")
        for g in ("floor", "box", "left_finger_pad", "right_finger_pad", "hand_capsule", "link6"):
            pairs.append(f'<pair geom1="{g}" geom2="c{i}"/>')
        pairs += [f'<pair geom1="c{j}" geom2="c{i}"/>' for j in range(i)]
    x = x.replace("  </worldbody>", "    " + "\n    ".join(bodies) + "\n  </worldbody>", 1)
    x = x.replace("  </contact>", "    " + "\n    ".join(pairs) + "\n  </contact>", 1)
    x, c = re.subn(r'(<key name="home" qpos="[^"]*)"', lambda mt: mt.group(1) + " " + " ".join(qk) + '"', x)
    assert c == 1
    return x.replace('<mujoco model="arm_box">', f'<mujoco model="arm_clutter{n}">')


def full_pairs(m):
    """The geom index pairs to add for full self-contact: every two collision geoms (geoms in an explicit pair) on two
    bodies that are not parent and child, not yet a pair (added with MuJoCo's default pair parameters)."""
    cg = sorted(set(int(g) for g in m.pair_geom1) | set(int(g) for g in m.pair_geom2))
    have = {tuple(sorted((int(x), int(y)))) for x, y in zip(m.pair_geom1, m.pair_geom2)}
    gb, par = m.geom_bodyid, m.body_parentid
    add = []
    for i, x in enumerate(cg):
        for y in cg[i + 1:]:
            bx, by = int(gb[x]), int(gb[y])
            if bx == 0 or by == 0 or bx == by or par[bx] == by or par[by] == bx or (x, y) in have:
                continue
            add.append((x, y))
    return add


def ctrl_of(m, Q):
    """Position setpoints = the state (actuators with a joint transmission and a bias), else 0; inside ctrlrange."""
    C = np.zeros((len(Q), m.nu))
    for a in range(m.nu):
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_biastype[a] != 0:
            C[:, a] = Q[:, m.jnt_qposadr[m.actuator_trnid[a, 0]]]
        if m.actuator_ctrllimited[a]:
            lo, hi = m.actuator_ctrlrange[a]
            C[:, a] = np.clip(C[:, a], lo + 1e-3, hi - 1e-3)
    return C



def random_states(m, B, seed=0):
    """qpos0 (the first keyframe if any) with N(0, 0.05) on the hinge / slide joints and free bodies lifted 0-0.3 m,
    qvel N(0, 0.2) on those joints; setpoints = state."""
    rng = np.random.default_rng(seed)
    Q = np.tile(m.key_qpos[0] if m.nkey else m.qpos0, (B, 1))
    V = np.zeros((B, m.nv))
    for j in range(m.njnt):
        a, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
        if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            Q[:, a] += rng.normal(0, 0.05, B)
            if m.jnt_limited[j]:
                Q[:, a] = np.clip(Q[:, a], *m.jnt_range[j])
            V[:, v] = rng.normal(0, 0.2, B)
        elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            Q[:, a + 2] += rng.uniform(0, 0.3, B)
    return (Q, V, ctrl_of(m, Q)), None


def settled_states(m, B, seed=0, seconds=0.6):
    """random_states, then MuJoCo C steps each world `seconds` under its setpoints (the free bodies dropped and come to
    rest on the floor and on each other: resting contacts); (qpos, qvel, ctrl)."""
    (Q, V, C), _ = random_states(m, B, seed)
    d = mujoco.MjData(m)
    n = int(round(seconds / float(m.opt.timestep)))
    for i in range(B):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = Q[i], V[i], C[i]
        for _ in range(n):
            mujoco.mj_step(m, d)
        Q[i], V[i] = d.qpos, d.qvel
    return (Q, V, C), dict(settle_seconds=seconds)


def states_of(m, kind, B, seed):
    if kind == "random":
        return random_states(m, B, seed)
    if kind == "settle":
        return settled_states(m, B, seed)
    raise SystemExit(f"--states {kind}: random or settle")


def load_model(a):
    s = a.scene
    if s == "arm":
        return mujoco.MjModel.from_xml_path(str(ARM_XML)), ARM_XML.read_text()
    if s.startswith("clutter_"):
        x = clutter_xml(int(s[len("clutter_"):]), a.seed)
        return mujoco.MjModel.from_xml_string(x), x
    if s.startswith("xml:"):
        return mujoco.MjModel.from_xml_path(s[4:]), None
    return None, None


def scene(a, B):
    """(model, (qpos, qvel, ctrl), substeps, state info)"""
    m, xml = load_model(a)
    if m is None:
        r = INTERNAL.scene(a, B) if INTERNAL is not None else None
        if r is None:
            raise SystemExit(f"--scene {a.scene}: unknown (arm, clutter_<n>, xml:<path>)")
        return r
    if a.full_pairs:
        add = full_pairs(m)
        if xml is None:
            xml = Path(a.scene[4:]).read_text()
        names = lambda g: m.geom(g).name
        assert all(names(x) and names(y) for x, y in add), "--full-pairs: unnamed collision geoms"
        k = xml.rindex("</contact>")
        xml = xml[:k] + "".join(f'<pair geom1="{names(x)}" geom2="{names(y)}"/>' for x, y in add) + xml[k:]
        m = mujoco.MjModel.from_xml_string(xml) if not a.scene.startswith("xml:") else _from_xml_at(xml, a.scene[4:])
    if a.corpus:
        return m, corpus(a.corpus, B), 4, None
    st, info = states_of(m, a.states, B, a.seed)
    return m, st, 4, info


def _from_xml_at(xml, path):
    cwd = os.getcwd()
    os.chdir(Path(path).resolve().parent)
    try:
        return mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(cwd)


def corpus(path, B):
    z = np.load(path)
    idx = np.arange(B) % len(z["qpos"])
    return z["qpos"][idx], z["qvel"][idx], z["ctrl"][idx]


VARIANTS = ("main", "main_sorted", "midphase")


def build(m, variant, profile, work, midphase="1"):
    """The forest engine's check binary of m for a variant of VARIANTS (and -DDK_PROFILE); (dir, source sha256).
    main_sorted: main's text with the contacts sorted by (pair, slot) after the collision, as the mid-phase builds
    (and the builds with deferred pairs) do: the bitwise reference of the mid-phase. midphase: the DK_MIDPHASE value of
    the mid-phase variant (1, or 2 for the group level)."""
    import engine_step_check as E
    from nablaengine.cuda import compile_cuda
    old = os.environ.get("DK_MIDPHASE")
    os.environ["DK_MIDPHASE"] = midphase if variant == "midphase" else "0"
    try:
        src, consts = E.forest_binary_source(m, [], None)
    finally:
        if old is None:
            os.environ.pop("DK_MIDPHASE")
        else:
            os.environ["DK_MIDPHASE"] = old
    assert ("#define MIDPHASE 1" in src) == (variant == "midphase"), "DK_MIDPHASE did not take effect"
    if variant == "main_sorted":
        x = "#if HAS_DEFER\n  sort_contacts(s, lane);"
        assert src.count(x) == 1
        src = src.replace(x, "#if 1\n  sort_contacts(s, lane);")
    if profile:
        x = "  __syncwarp();\n  substep_pre<1>(s, ctrl + (size_t)w * NU, lane, nullptr);  // (no tape: k_pre writes it)\n"
        if src.count(x) == 1:
            src = src.replace(x, "  if (lane < NPROF) s.prof[lane] = 0;\n" + x + "  __syncwarp();\n  if (lane == 2 || lane"
                              " == 26 || lane == 29) atomicAdd(&g_prof[lane], (unsigned long long)s.prof[lane]);\n")
    sha = hashlib.sha256(src.encode()).hexdigest()
    d = work / f"{variant}_p{int(profile)}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "step.cu").write_text(src)
    compile_cuda(d / "step.cu", d / "step", ["-DDK_PROFILE"] if profile else [], unit="forward")
    consts.tofile(d / "consts.bin")
    return d, sha


def write_inputs(m, d, Q, V, C):
    from nablaengine.forest_engine import step
    nw = len(Q)
    for k, x in dict(qpos=Q, qvel=V, warm=np.zeros_like(V), ctrl=C, fric=np.ones(nw),
                     com=np.zeros((nw, step.com_words(m)))).items():
        np.ascontiguousarray(x, np.float32).tofile(d / f"{k}.bin")


def run(m, d, nw, nsub, iters, reps=0, env=None):
    from nablaengine.forest_engine import step
    o = subprocess.run([str(d / "step"), str(d), str(nw), str(iters), str(reps), str(nsub), "0", "0"], check=True,
                       timeout=3600, stdout=subprocess.PIPE, universal_newlines=True,
                       env=dict(os.environ, **(env or {}))).stdout
    r = json.loads(next(x for x in o.splitlines() if x.startswith('{"nworld"')))
    if r["err"] != "no error":
        raise RuntimeError(f"CUDA error: {r['err']}")
    prof = [json.loads(x) for x in o.splitlines() if x.startswith('{"prof_cycles"')]
    if prof:
        cyc = np.asarray(prof[0]["prof_cycles"], np.float64) / (nw * nsub)
        r["prof"] = dict(zip(step.PROF_PHASES, cyc.round(1).tolist()))
    mc = step.maxcon(m)
    f = lambda n, dt=np.float32: np.fromfile(d / n, dt)
    out = dict(qpos=f("out_qpos.bin").reshape(nw, m.nq), qvel=f("out_qvel.bin").reshape(nw, m.nv),
               warm=f("out_warm.bin").reshape(nw, m.nv), info=f("out_info.bin", np.int32).reshape(nw, 4),
               con=f("out_con.bin").reshape(nw, mc, step.CON_WORDS))
    return r, out


def compare(x, y):
    """{output: number of worlds whose bits differ}"""
    res = {}
    for k in x:
        a, b = x[k], y[k]
        if a.dtype == np.float32:
            a, b = a.view(np.uint32), b.view(np.uint32)
        res[k] = int((a != b).reshape(len(a), -1).any(1).sum())
    return res


def contact_sets_equal(x, y):
    """Per world, the contacts' (pair, dist, pos, normal) words of x and y as sets (sorted rows): worlds differing."""
    nw = len(x["info"])
    bad = 0
    for w in range(nw):
        n0, n1 = min(x["info"][w, 1], x["con"].shape[1]), min(y["info"][w, 1], y["con"].shape[1])
        a, b = x["con"][w, :n0, :8].view(np.uint32), y["con"][w, :n1, :8].view(np.uint32)
        if n0 != n1 or not np.array_equal(a[np.lexsort(a.T[::-1])], b[np.lexsort(b.T[::-1])]):
            bad += 1
    return dict(worlds=int(nw), worlds_differing=int(bad))


def rel_max(a, b):
    """Largest |a - b| over the worlds, relative to each world's largest |a| (1e-6 floor)."""
    d = np.abs(a.astype(np.float64) - b) / np.maximum(np.abs(a.astype(np.float64)).max(1, keepdims=True), 1e-6)
    return float(np.nanmax(d)) if d.size else 0.0


def summary(prof):
    if not prof:
        return None
    part = ("smooth", "qacc_smooth", "collision", "contact_jac", "rows", "newton_init", "newton_grad", "newton_hessian",
            "newton_chol", "newton_linesearch", "newton_update", "integrate")
    return dict(total=round(sum(prof[k] for k in part)), collision=prof["collision"], geom_poses=prof["col_geom"],
                broad_phase=prof["col_broad"])


def cmd_run(a):
    from nablaengine import provenance, workdir
    from nablaengine.forest_engine import midphase as MP
    m, (Q, V, C), nsub, sinfo = scene(a, a.nworld)
    nsub = a.substeps or nsub
    work = workdir.work_dir("midphase_")
    t = MP.tables(m)
    res = dict(scene=a.scene, states=a.corpus and Path(a.corpus).name or a.states, nworld=a.nworld, substeps=nsub,
               iterations=a.iterations, npair=int(m.npair), nbody=int(m.nbody), nv=int(m.nv), state_info=sinfo,
               midphase=dict(nodes=len(t["nodes"]), groups=len(t["gfirst"]) - 1, group_pairs=len(t["gp"]),
                             runs=len(t["runs"]), group_level=MP.group_level(t),
                             auto=MP.enabled(m) if MP.mode() == "auto" else None,
                             groups_named=[n for _, n in MP.describe(m) if len(n) > 1]),
               provenance=dict(commit=provenance.source_commit(), gpu=provenance.gpu(),
                               nvcc=provenance.nvcc_version()))
    ok = True
    bins = {}
    for v in VARIANTS:
        d, sha = build(m, v, False, work, a.midphase)
        write_inputs(m, d, Q, V, C)
        bins[v] = d
        res[f"source_sha256_{v}"] = sha
    for label, ns in (("substep", 1), ("control_step", nsub)):
        outs = {v: run(m, bins[v], a.nworld, ns, a.iterations)[1] for v in VARIANTS}
        if label == "substep":
            nc = outs["main"]["info"][:, 1]
            res["contacts"] = dict(p50=float(np.median(nc)), max=int(nc.max()),
                                   unsupported_worlds=int((outs["main"]["info"][:, 3] != 0).sum()))
            res["contact_sets_equal_main"] = contact_sets_equal(outs["main"], outs["midphase"])
            ok &= res["contact_sets_equal_main"]["worlds_differing"] == 0
        diff = compare(outs["main_sorted"], outs["midphase"])
        res[f"bitwise_vs_main_sorted_{label}"] = diff
        ok &= not any(diff.values())
        res[f"vs_main_{label}"] = dict(worlds_differing=compare(outs["main"], outs["midphase"]),
                                       qvel_rel_max=rel_max(outs["main"]["qvel"], outs["midphase"]["qvel"]))
    if a.profile:
        for v in ("main", "midphase"):
            d, _ = build(m, v, True, work, a.midphase)
            write_inputs(m, d, Q, V, C)
            r, _ = run(m, d, a.nworld, nsub, a.iterations)
            res[f"profile_{v}"] = dict(summary(r["prof"]), phases=r["prof"])
    if a.time:
        ms = {v: dict(forward=[], k_pre=[]) for v in ("main", "midphase")}
        for _ in range(a.time):
            for v in ("main", "midphase"):
                ms[v]["forward"].append(run(m, bins[v], a.nworld, nsub, a.iterations, a.reps)[0]["ms"])
                ms[v]["k_pre"].append(run(m, bins[v], a.nworld, nsub, a.iterations, a.reps,
                                          env={"DK_SPLIT_ONLY": "1"})[0]["ms"])
        res["time_ms"] = ms
        res["time_ratio_midphase_over_main"] = {}
        for k in ("forward", "k_pre"):
            ratio = np.asarray(ms["midphase"][k]) / np.asarray(ms["main"][k])
            res["time_ratio_midphase_over_main"][k] = dict(median=float(np.median(ratio)), min=float(ratio.min()),
                                                           max=float(ratio.max()))
    res["pass"] = bool(ok)
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1))
    if a.out:
        out = Path(a.out)
        if out.exists() and not a.overwrite:
            raise SystemExit(f"{out} exists (--overwrite to replace it)")
        out.write_text(json.dumps(res, indent=1))
    return 0 if ok else 1


def cmd_ffi_one(a):
    """(subprocess) one interface with the current DK_MIDPHASE: outputs and the VJP into --dump."""
    m, (Q, V, C), nsub, _ = scene(a, a.nworld)
    from nablaengine import engines
    engines.prefer_forest(m)
    rng = np.random.default_rng(1)
    gq, gv = rng.normal(size=(a.nworld, m.nq)), rng.normal(size=(a.nworld, m.nv))
    kw = dict(max_contacts=a.max_contacts, max_rows=a.max_rows)
    if a.api == "jax":
        import jax
        import jax.numpy as jnp
        from nablaengine import jax_ffi
        jax_ffi.init(model=m, foot_pairs=[], **kw)
        cw = int(jax_ffi.init().get("com_words", 3))
        f32 = lambda x: jnp.asarray(x, jnp.float32)
        fr, com, w0 = f32(np.ones(a.nworld)), f32(np.zeros((a.nworld, cw))), f32(np.zeros((a.nworld, m.nv)))

        def f(q, v, u):
            q1, v1, _ = jax_ffi.step(q, v, w0, u, fr, com, substeps=nsub, iterations=a.iterations)
            return q1, v1
        (q1, v1), vjp = jax.vjp(f, f32(Q), f32(V), f32(C))
        g = vjp((f32(gq), f32(gv)))
        out = dict(qpos=q1, qvel=v1, g_qpos=g[0], g_qvel=g[1], g_ctrl=g[2])
        out = {k: np.asarray(x) for k, x in out.items()}
    else:
        import torch
        from nablaengine import torch_ffi
        torch_ffi.init(model=m, foot_pairs=[], **kw)
        cw = int(torch_ffi.init().get("com_words", 3))
        t = lambda x, g=False: torch.tensor(np.asarray(x, np.float32), device="cuda", requires_grad=g)
        q, v, u = t(Q, True), t(V, True), t(C, True)
        q1, v1, _ = torch_ffi.step(q, v, t(np.zeros((a.nworld, m.nv))), u, t(np.ones(a.nworld)),
                                   t(np.zeros((a.nworld, cw))), substeps=nsub, iterations=a.iterations)
        (q1 * t(gq)).sum().add((v1 * t(gv)).sum()).backward()
        out = dict(qpos=q1, qvel=v1, g_qpos=q.grad, g_qvel=v.grad, g_ctrl=u.grad)
        out = {k: x.detach().cpu().numpy() for k, x in out.items()}
    np.savez(a.dump, **out)


def cmd_ffi(a):
    from nablaengine import workdir
    work = workdir.work_dir("midphase_ffi_")
    files = {}
    argv = sys.argv[2:]
    for opt in ("--out", "--torch-python"):
        if opt in argv:
            i = argv.index(opt)
            argv = argv[:i] + argv[i + 2:]
    for api, mp in (("jax", "0"), ("jax", a.midphase), ("torch", a.midphase)):
        f = work / f"{api}_{mp}.npz"
        env = dict(os.environ, DK_MIDPHASE=mp)
        exe = a.torch_python if api == "torch" and a.torch_python else sys.executable
        subprocess.run([exe, __file__, "ffi-one"] + argv + ["--api", api, "--dump", str(f)], check=True, env=env)
        files[(api, mp)] = dict(np.load(f))
    mp = a.midphase
    res = {"torch_vs_jax_midphase": compare(files[("jax", mp)], files[("torch", mp)])}
    ok = not any(res["torch_vs_jax_midphase"].values())
    x, y = files[("jax", "0")], files[("jax", mp)]
    res["jax_midphase_vs_main"] = {k: dict(worlds_differing=int((x[k] != y[k]).reshape(len(x[k]), -1).any(1).sum()),
                                           rel_max=rel_max(x[k], y[k])) for k in x}
    res["finite"] = bool(all(np.isfinite(v).all() for f in files.values() for v in f.values()))
    res["pass"] = bool(ok)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(scene=a.scene, states=a.states, nworld=a.nworld, **res), indent=1))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "ffi", "ffi-one"))
    ap.add_argument("--scene", default="clutter_8", help="arm, clutter_<n> or xml:<path>")
    ap.add_argument("--states", default="random", help="random or settle")
    ap.add_argument("--corpus", default=None, help="states from an npz (qpos, qvel, ctrl) instead of --states")
    ap.add_argument("--full-pairs", action="store_true", help="add every non-parent-child pair of collision geoms")
    if INTERNAL is not None:
        INTERNAL.add_arguments(ap)
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--substeps", type=int, default=0, help="per control step (default: the scene's)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--midphase", default="1", choices=("1", "2"),
                    help="DK_MIDPHASE of the mid-phase build (2: the group level)")
    ap.add_argument("--time", type=int, default=0, help="interleaved timing rounds")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--api", default="jax")
    ap.add_argument("--torch-python", default=None, help="ffi: the interpreter of the PyTorch run (an environment with "
                    "torch; default this one)")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--max-contacts", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    from nablaengine import workdir
    workdir.add_argument(ap)
    a = ap.parse_args()
    workdir.configure(a)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "ffi":
        return cmd_ffi(a)
    cmd_ffi_one(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())

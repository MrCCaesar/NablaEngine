"""Smoke test for a new device (e.g. Jetson Orin): does the engine build, load and step, and does it agree with a stored
reference?

Run this first on the device. It builds (or loads) the engine library for the GPU JAX sees, steps 64 worlds of the
mjlab G1 from the stored input states and compares with the reference recorded on another GPU
(scripts/orin/data/smoke_<engine>_g1.npz):
  step1     one control step (4 substeps): qpos', qvel' (forward, no tape)
  roll15    15 control steps (0.3 s): the final qpos, qvel
  vjp1      the VJP of one control step for fixed output cotangents: g_qpos, g_qvel, g_ctrl
  mujoco    one control step of MuJoCo C (float64, CPU, mj_step x 4) from the same states: the engine's distance to it
            here and on the reference GPU (context: the float32 engine against a float64 solver)
Another architecture or nvcc compiles the same source to other machine code, so bitwise equality is reported but not
required (it held on an RTX 3090, sm_86 with nvcc 12.6, against the V100 reference, sm_70 with nvcc 12.4). Tolerances (max abs over worlds and coordinates): step1 qpos 1e-4, qvel 1e-2; vjp1 relative norm 1e-2;
roll15 is reported only (a 15-step rollout amplifies ulp differences: chaotic contacts). Also: the forward and the
backward repeated give bitwise the same results (determinism), and no world hits a capacity overflow.

  python scripts/orin/smoke.py [--engine forest|tree] [--json out.json]
  python scripts/orin/smoke.py --write    # (re)record the reference on this GPU

Exit status 0 when every check passes. No GPU index or path is assumed (JAX's default device; fixtures from
$DK_FIXTURES, else fixtures/mjlab of this checkout).
"""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]


def setup_env(engine):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["DK_ENGINE"] = engine
    if not os.environ.get("DK_FIXTURES") and (REPO / "fixtures" / "mjlab").is_dir():
        os.environ["DK_FIXTURES"] = str(REPO / "fixtures" / "mjlab")
    try:
        import nablaengine
    except ImportError:
        sys.path.insert(0, str(REPO / "src"))


def versions(jax, dev):
    v = dict(python=platform.python_version(), machine=platform.machine(), jax=jax.__version__)
    for p in ("jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda13-plugin", "jax-cuda13-pjrt", "mujoco",
              "numpy", "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvcc-cu12"):
        try:
            v[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pass
    v["device"] = dev.device_kind
    v["platform"] = dev.platform
    v["compute_capability"] = getattr(dev, "compute_capability", None)
    v["core_count"] = getattr(dev, "core_count", None)
    try:
        v["platform_version"] = dev.client.platform_version
    except Exception:
        pass
    try:
        from jax._src.lib import cuda_versions
        if cuda_versions is not None:
            v["cuda_runtime"] = cuda_versions.cuda_runtime_get_version()
            v["cuda_driver"] = cuda_versions.cuda_driver_get_version()
    except Exception:
        pass
    tegra = Path("/etc/nv_tegra_release")
    if tegra.exists():
        v["l4t"] = tegra.read_text().splitlines()[0].strip()
    try:
        from nablaengine import cuda, provenance
        nv = cuda.nvcc()
        out = subprocess.run([nv, "--version"], stdout=subprocess.PIPE, universal_newlines=True).stdout
        v["nvcc"] = nv
        v["nvcc_version"] = out.strip().splitlines()[-1]
        c, dirty = provenance.source_commit()
        v["engine_commit"] = f"{c}{'-dirty' if dirty else ''}"
    except Exception as e:
        v["nvcc"] = f"unavailable: {e}"
    return v


def initial_state(m, fp, B, com_words, seed):
    """Per-world inputs: the keyframe knees_bent (else home, else qpos0) with uniform joint perturbations of +-0.05 rad,
    root height +-1 cm, velocities N(0, 0.1); ctrl0 the keyframe's joint positions; the model's foot friction; no COM
    offset; zero warm start."""
    import mujoco
    import numpy as np
    rng = np.random.default_rng(seed)
    names = [m.key(i).name for i in range(m.nkey)]
    k = names.index("knees_bent") if "knees_bent" in names else (names.index("home") if "home" in names else None)
    q = (m.key_qpos[k] if k is not None else m.qpos0).astype(np.float64).copy()
    qpos = np.tile(q, (B, 1))
    qpos[:, 7:] += rng.uniform(-0.05, 0.05, (B, m.nq - 7))
    qpos[:, 2] += rng.uniform(-0.01, 0.01, B)
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    lim = m.jnt_limited[1:].astype(bool)
    qpos[:, 7:] = np.where(lim, np.clip(qpos[:, 7:], lo, hi), qpos[:, 7:])
    qvel = rng.normal(0, 0.1, (B, m.nv))
    ctrl0 = np.tile(q[7:7 + m.nu], (B, 1))
    if k is not None and m.key_ctrl.shape[1] == m.nu and np.any(m.key_ctrl[k]):
        ctrl0 = np.tile(m.key_ctrl[k], (B, 1))
    f = np.float32
    return dict(qpos=qpos.astype(f), qvel=qvel.astype(f), warm=np.zeros((B, m.nv), f), ctrl0=ctrl0.astype(f),
                friction=np.full(B, m.pair_friction[fp[0], 0], f), com=np.zeros((B, com_words), f))

TOL = dict(step1_qpos=1e-4, step1_qvel=1e-2, vjp1_rel=1e-2)
B, H = 64, 15


def mujoco_step(m, s, nsub):
    """One control step of MuJoCo C per world (float64): qpos', qvel'."""
    import mujoco
    import numpy as np
    d = mujoco.MjData(m)
    out_q, out_v = [], []
    for i in range(s["qpos"].shape[0]):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = s["qpos"][i]
        d.qvel[:] = s["qvel"][i]
        d.ctrl[:] = s["ctrl0"][i]
        for _ in range(nsub):
            mujoco.mj_step(m, d)
        out_q.append(d.qpos.copy())
        out_v.append(d.qvel.copy())
    return np.array(out_q), np.array(out_v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", default="forest", choices=("forest", "tree"))
    ap.add_argument("--ref", default=None, help="reference npz (default scripts/orin/data/smoke_<engine>_g1.npz)")
    ap.add_argument("--write", action="store_true", help="record the reference on this GPU instead of checking")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    setup_env(args.engine)
    ref_path = Path(args.ref or HERE / "data" / f"smoke_{args.engine}_g1.npz")

    import numpy as np
    import jax
    import jax.numpy as jnp
    from nablaengine import fixtures, jax_ffi as F

    dev = jax.devices()[0]
    ver = versions(jax, dev)
    print("versions:", json.dumps(ver))
    t = time.perf_counter()
    info = F.init()
    print(f"engine {info['engine']}, arch {info['arch']}, flags {info['arch_flags']}, built now {info['built']} "
          f"({time.perf_counter() - t:.1f} s)")
    mt = fixtures.meta()
    m, _ = fixtures.load_model(check=None)
    S = int(mt.get("substeps_per_control_step", 4))
    if args.write:
        s = initial_state(m, mt["foot_pairs"], B, int(info.get("com_words", 3)), seed=123)
        rng = np.random.default_rng(7)
        s["cot_qpos"] = rng.standard_normal((B, m.nq)).astype(np.float32)
        s["cot_qvel"] = rng.standard_normal((B, m.nv)).astype(np.float32)
    else:
        z = np.load(ref_path)
        s = {k: z[k] for k in ("qpos", "qvel", "warm", "ctrl0", "friction", "com", "cot_qpos", "cot_qvel")}
    q0, v0, w0, u0, fr, com = (jnp.asarray(s[k]) for k in ("qpos", "qvel", "warm", "ctrl0", "friction", "com"))
    cq, cv = jnp.asarray(s["cot_qpos"]), jnp.asarray(s["cot_qvel"])

    fwd = jax.jit(lambda q, v, w: F.forward(q, v, w, u0, fr, com, substeps=S))

    def roll(q, v, w):
        def body(c, _):
            return F.step(*c, u0, fr, com, substeps=S), None
        return jax.lax.scan(body, (q, v, w), None, length=H)[0]
    roll = jax.jit(roll)

    def vjp1(q, v, u):
        _, pull = jax.vjp(lambda a, b, c: F.step(a, b, w0, c, fr, com, substeps=S)[:2], q, v, u)
        return pull((cq, cv))
    vjp1 = jax.jit(vjp1)

    out = {}
    q1, v1, w1, inf1 = (np.asarray(x) for x in fwd(q0, v0, w0))
    out["step1_qpos"], out["step1_qvel"] = q1, v1
    qr, vr, _ = (np.asarray(x) for x in roll(q0, v0, w0))
    out["roll15_qpos"], out["roll15_qvel"] = qr, vr
    g = [np.asarray(x) for x in vjp1(q0, v0, u0)]
    out["vjp1_qpos"], out["vjp1_qvel"], out["vjp1_ctrl"] = g
    q1b, v1b = (np.asarray(x) for x in fwd(q0, v0, w0)[:2])
    gb = [np.asarray(x) for x in vjp1(q0, v0, u0)]
    deterministic = bool(np.array_equal(q1, q1b) and np.array_equal(v1, v1b)
                         and all(np.array_equal(a, b) for a, b in zip(g, gb)))
    overflow = int(np.count_nonzero(inf1[:, 3] & (F.OVERFLOW_CONTACTS | F.OVERFLOW_ROWS)))
    mq, mv = mujoco_step(m, s, S)
    mj = dict(qpos=float(np.max(np.abs(q1 - mq))), qvel=float(np.max(np.abs(v1 - mv))))
    finite = all(np.all(np.isfinite(x)) for x in out.values())

    if args.write:
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(arch=info["arch"], engine=info["engine"], device=ver.get("device"), jax=ver.get("jax"),
                    nvcc=ver.get("nvcc_version"), commit=ver.get("engine_commit"), mujoco_dist=mj)
        np.savez_compressed(ref_path, **s, **out, meta=json.dumps(meta))
        print(f"wrote {ref_path} ({ref_path.stat().st_size} B): {meta}; deterministic {deterministic}, overflow "
              f"worlds {overflow}, finite {finite}")
        sys.exit(0 if deterministic and finite and not overflow else 1)

    meta = json.loads(str(z["meta"]))
    rows, ok = {}, deterministic and finite and overflow == 0
    for k in out:
        d = float(np.max(np.abs(out[k] - z[k])))
        rel = float(np.linalg.norm(out[k] - z[k]) / max(np.linalg.norm(z[k]), 1e-30))
        rows[k] = dict(max_abs=d, rel_norm=rel, bitwise=bool(np.array_equal(out[k], z[k])))
    checks = {"step1_qpos": rows["step1_qpos"]["max_abs"] <= TOL["step1_qpos"],
              "step1_qvel": rows["step1_qvel"]["max_abs"] <= TOL["step1_qvel"],
              "vjp1": all(rows[k]["rel_norm"] <= TOL["vjp1_rel"] for k in ("vjp1_qpos", "vjp1_qvel", "vjp1_ctrl"))}
    ok = ok and all(checks.values())
    print(f"reference: {meta['device']} ({meta['arch']}, {meta['nvcc']}, commit {meta['commit']})")
    for k, r in rows.items():
        print(f"  {k:<12} max abs {r['max_abs']:.3e}  rel norm {r['rel_norm']:.3e}  bitwise {r['bitwise']}")
    print(f"  vs MuJoCo C (1 step, max abs): here qpos {mj['qpos']:.3e} qvel {mj['qvel']:.3e}; on the reference "
          f"GPU qpos {meta['mujoco_dist']['qpos']:.3e} qvel {meta['mujoco_dist']['qvel']:.3e}")
    print(f"  deterministic {deterministic}, overflow worlds {overflow}, finite {finite}, checks {checks}")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(versions=ver, arch=info["arch"], reference=meta, rows=rows,
                                                   mujoco_dist=mj, deterministic=deterministic, overflow=overflow,
                                                   checks=checks, ok=ok), indent=1))
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

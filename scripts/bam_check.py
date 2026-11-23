"""The BAM servo model's forward on the tree engine (init(actuator=Bam(...))) against the same model on MuJoCo C in float64
(nablaengine.actuators.bam.Reference).

--model g1: the mjlab G1 with an XL330 on every hinge, lifted 3 m (free flight, no contacts);
--model microduck: Pollen Robotics' Microduck, the robot these servos are in (fixtures/robots/microduck, 4 spheres per
foot), standing on the floor from its STAND keyframe, the model's own solver settings."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nablaengine import fixtures, provenance
from nablaengine.actuators import Bam, bam

GATES = {"g1": dict(qvel_med=1e-5, qvel_max=1e-3, torque_max=1e-3, roll1_max=1e-3),
         # microduck: the G1's gates until a CUDA run sets them (the Mac port's Metal build of the same model and servo
         # model without the current limit, 1024 states: qvel median 5.7e-7, max 3.9e-6)
         "microduck": dict(qvel_med=1e-5, qvel_max=1e-3, torque_max=1e-3, roll1_max=1e-3)}
MICRODUCK = ROOT / "fixtures" / "robots" / "microduck" / "microduck_spheres4"


def load(name):
    """(raw MjModel, foot pairs, COM body, initial qpos, substeps per control step)."""
    if name == "microduck":
        mt = json.loads(Path(f"{MICRODUCK}.json").read_text())
        m = mujoco.MjModel.from_xml_path(f"{MICRODUCK}.xml")
        return m, mt["foot_pairs"], mt["com_body"], np.array(mt["init_qpos"], float), int(mt["substeps_per_control_step"])
    m, mt = fixtures.load_model(None, check=None)
    q0 = m.key_qpos[0].copy() if m.nkey else m.qpos0.copy()
    if not m.nkey:
        q0[2] = 0.8
    return m, mt["foot_pairs"], mt.get("torso_body", "torso_link"), q0, int(mt.get("substeps_per_control_step", 4))


class Engine:
    def __init__(self, backend, m, fp, com, b):
        self.backend = backend
        if backend == "jax":
            import jax.numpy as jnp
            from nablaengine import jax_ffi
            self.info = jax_ffi.init(model=m, foot_pairs=fp, com_body=com, actuator=b)
            self.mod, self.cv = jax_ffi, lambda x: jnp.asarray(np.ascontiguousarray(x, np.float32))
        else:
            import torch
            from nablaengine import torch_ffi
            self.info = torch_ffi.init(model=m, foot_pairs=fp, com_body=com, actuator=b)
            self.mod = torch_ffi
            self.cv = lambda x: torch.as_tensor(np.ascontiguousarray(x, np.float32), device="cuda")

    def forward(self, qpos, qvel, warm, ctrl, fric, act, iterations, substeps):
        B = qpos.shape[0]
        r = self.mod.forward(*map(self.cv, (qpos, qvel, warm, ctrl, fric, np.zeros((B, 3)))), act=self.cv(act),
                             iterations=iterations, substeps=substeps)
        return [np.asarray(x.cpu() if hasattr(x, "cpu") else x) for x in r[:5]]


def corpus(ref, q0, sub, n, rng):
    """States of reference rollouts (the module docstring), the targets, and the reference's control step from each."""
    m, d = ref.m, ref.d
    nu, nv = m.nu, m.nv
    Z = {k: [] for k in ("qpos", "qvel", "warm", "ctrl", "act", "qpos1", "qvel1", "warm1", "act1")}
    while len(Z["qpos"]) < n:
        q, v, w = q0.copy(), np.zeros(nv), np.zeros(nv)
        q[7:] += rng.normal(0, 0.05, nu)
        a = bam.initial_state(nv, 1, rng.uniform(6.5, 8.2), rng.uniform(0.0, 0.2))[0].astype(float)
        f, ph, amp = rng.uniform(0.3, 2.5, (4, nu)), rng.uniform(0, 6.28, (4, nu)), rng.normal(0, 0.08, (4, nu))
        for k in range(30):
            t = k * sub * m.opt.timestep
            u = q0[7:] + (amp * np.sin(2 * np.pi * f * t + ph)).sum(0)
            take = k >= 4 and k % 3 == 0
            if take:
                u = q[7:] + rng.normal(0, 0.25, nu)
                if rng.random() < 0.3:
                    u[rng.choice(nu, 3, replace=False)] += rng.choice([-2.5, 2.5], 3)
            q, v, w, u = (np.float32(x).astype(float) for x in (q, v, w, u))
            a = np.float32(a).astype(float)
            q1, v1, w1, a1 = ref.step(q, v, w, u, a, sub)
            if take:
                for key, x in zip(Z, (q, v, w, u, a, q1, v1, w1, a1)):
                    Z[key].append(x)
                if len(Z["qpos"]) >= n:
                    break
            q, v, w, a = q1, v1, w1, a1
            up = np.zeros(9)
            mujoco.mju_quat2Mat(up, q[3:7])
            if up[8] < 0.3 or not np.isfinite(v).all():
                break
    return {k: np.array(x) for k, x in Z.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="g1", choices=tuple(GATES))
    ap.add_argument("--backend", default="jax", choices=("jax", "torch"))
    ap.add_argument("--max-current", default="firmware")
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=None,
                    help="the solver's iterations and line-search iterations in both (default: the model's)")
    ap.add_argument("--lift", type=float, default=None,
                    help="raise the initial root height by this (m; default 0, g1: 3, free flight)")
    ap.add_argument("--out")
    ap.add_argument("--dump")
    a = ap.parse_args()
    mc = {"firmware": "firmware", "none": None}.get(a.max_current, None)
    if a.max_current not in ("firmware", "none"):
        mc = float(a.max_current)
    b = Bam(max_current=mc)
    m0, fp, com, q0, sub = load(a.model)
    if a.model == "g1":
        a.lift = 3.0 if a.lift is None else a.lift
        a.iterations = a.iterations or 50
    q0[2] += a.lift or 0.0
    if a.iterations:
        m0.opt.iterations, m0.opt.ls_iterations = a.iterations, a.iterations
    ref = bam.Reference(m0, b)
    m, sv, nv = ref.m, bam.servo(b), ref.m.nv
    t0 = time.perf_counter()
    eng = Engine(a.backend, m0, fp, com, b)
    t_init = time.perf_counter() - t0
    it = int(m.opt.iterations)
    fric = np.full(a.worlds, m.pair_friction[fp[0], 0])
    rng = np.random.default_rng(a.seed)

    Z = corpus(ref, q0, sub, a.worlds, rng)
    n = len(Z["qpos"])
    q1, v1, w1, a1, info = eng.forward(Z["qpos"], Z["qvel"], Z["warm"], Z["ctrl"], fric[:n], Z["act"], it, sub)
    dv = np.linalg.norm(v1 - Z["qvel1"], axis=1) / np.maximum(np.linalg.norm(Z["qvel1"], axis=1), 1e-9)
    dq = np.abs(q1 - Z["qpos1"]).max(1)
    et = np.abs(a1[:, 6:nv] - Z["act1"][:, 6:nv])
    ee = np.abs(a1[:, nv + 6:2 * nv] - Z["act1"][:, nv + 6:2 * nv])
    passthrough = bool(np.array_equal(a1[:, -2:], np.float32(Z["act"][:, -2:])))
    e = Z["ctrl"] - Z["qpos"][:, 7:]
    vv = Z["qvel"][:, 6:]
    sc = np.maximum(Z["act"][:, 2 * nv] - Z["act"][:, 2 * nv + 1] * np.abs(Z["act"][:, 6:nv]).sum(1),
                    sv["vin_min"]) / sv["vin_nom"]
    raw = sc[:, None] * sv["kp"] * e
    cur = int(((raw < sv["kd"] * vv - sv["ilim"]) | (raw > sv["kd"] * vv + sv["ilim"])).sum()) if sv["ilim"] else 0
    volt = int((np.abs(raw) > sc[:, None] * sv["frclim"]).sum())
    one = dict(worlds=n, qvel_rel_median=float(np.median(dv)), qvel_rel_p90=float(np.percentile(dv, 90)),
               qvel_rel_max=float(dv.max()), qpos_abs_max=float(dq.max()),
               tau_abs_err_median=float(np.median(et)), tau_abs_err_max=float(et.max()),
               ext_abs_err_median=float(np.median(ee)), ext_abs_err_max=float(ee.max()),
               tau_p95=float(np.percentile(np.abs(Z["act1"][:, 6:nv]), 95)),
               ext_p95=float(np.percentile(np.abs(Z["act1"][:, nv + 6:2 * nv]), 95)),
               servo_dofs_at_current_window_substep0=cur, servo_dofs_at_voltage_limit_substep0=volt,
               supply_passed_through=passthrough, unsupported_worlds=int((info[:, 3] != 0).sum()))

    R, H = min(a.rollouts, n), a.steps
    eq, ev, ea = Z["qpos"][:R].copy(), Z["qvel"][:R].copy(), Z["act"][:R].copy()
    ew = Z["warm"][:R].copy()
    rs = [(Z["qpos"][r].copy(), Z["qvel"][r].copy(), Z["warm"][r].copy(), Z["act"][r].copy()) for r in range(R)]
    targ = Z["ctrl"][:R] + rng.normal(0, 0.1, (H, R, m.nu))
    targ = np.float32(targ).astype(float)
    roll_q, roll_v, roll_t = [], [], []
    for k in range(H):
        eq, ev, ew, ea, _ = eng.forward(eq, ev, ew, targ[k], fric[:R], ea, it, sub)
        rs = [ref.step(*rs[r][:3], targ[k, r], rs[r][3], sub) for r in range(R)]
        rq = np.array([x[0] for x in rs]); rv = np.array([x[1] for x in rs]); ra = np.array([x[3] for x in rs])
        roll_q.append(float(np.abs(eq - rq).max()))
        roll_v.append(float((np.linalg.norm(ev - rv, axis=1) / np.maximum(np.linalg.norm(rv, axis=1), 1e-9)).max()))
        roll_t.append(float(np.abs(ea[:, 6:nv] - ra[:, 6:nv]).max()))
    roll = dict(rollouts=R, steps=H, qpos_abs_max_per_step=roll_q, qvel_rel_max_per_step=roll_v,
                tau_abs_max_per_step=roll_t)

    g = GATES[a.model]
    rows = [("qvel rel median", one["qvel_rel_median"], g["qvel_med"]), ("qvel rel max", one["qvel_rel_max"], g["qvel_max"]),
            ("tau max (N m)", one["tau_abs_err_max"], g["torque_max"]), ("ext max (N m)", one["ext_abs_err_max"], g["torque_max"]),
            ("rollout step 1 qvel rel max", roll_v[0], g["roll1_max"])]
    ok = all(v <= t for _, v, t in rows) and passthrough and one["unsupported_worlds"] == 0
    res = dict(check="bam_check", model=a.model, backend=a.backend,
               engine=eng.info.get("engine"), bam=b.key(), servo=sv, substeps=sub,
               iterations=it, lift=a.lift or 0.0, seed=a.seed, init_seconds=t_init, one_step=one, rollout=roll,
               gates={k: dict(value=v, limit=t, ok=v <= t) for k, v, t in rows}, ok=ok,
               provenance=provenance.provenance())
    print(json.dumps({k: res[k] for k in ("model", "backend", "bam", "one_step", "gates", "ok")}, indent=1))
    print("rollout qpos max per step:", " ".join(f"{x:.1e}" for x in roll_q))
    print("rollout qvel rel max per step:", " ".join(f"{x:.1e}" for x in roll_v))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    if a.dump:
        np.savez(a.dump, q1=q1, v1=v1, w1=w1, a1=a1, **{"in_" + k: v for k, v in Z.items()})
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()

"""Forest terrain check: a terrain build on the forest engine against the same build on the tree engine (the parity
reference; not bitwise: the engines' solver code differs).

  dump  one engine (--engine tree | forest; forest: engines.prefer_forest on the fixture's G1, else $DK_ENGINE's
        routing is not used) on a terrain_check reference's states (REF.npz, the terrain from --scene or REF.mjb):
        one control step with the contact table, the tape and the output kinematics, and the VJP of a fixed random
        output cotangent; the kinematics (and its VJP) of the input state; --sensors: depth / height scan
        (nablaengine.sensors.g1) on the input state's kinematics and the IMU; --objects K: K object slots per world;
        --model MODEL: an exported robot (MODEL.xml / MODEL.json, as terrain_check --model) instead of the G1;
        --course COURSE.npz with --ref STATES.npz: a course and its states instead; writes OUT.npz. --time N: the
        forward and forward + backward times instead (medians of N; --no-terrain: the build without terrain)
  cmp   two dumps (A the tree's, B the forest's): per world the contact sets after the last substep (pair, terrain
        element, position within --tol), qpos / qvel relative difference, the kinematics of B's output state against
        A's kinematics function on B's state (bitwise expected: the same text), the VJP's relative difference; --gate
        exits 1 unless the contact sets agree in at least --min-sets of the worlds and the VJP medians are below
        --max-grad

  python scripts/forest_terrain_check.py dump --engine forest --ref fixtures/mjlab/mjw317_terrain_feet.npz \\
      --scene fixtures/mjlab/g1_terrain_feet.mjb --mode analytic --out runs/forest_terrain/forest.npz
  python scripts/forest_terrain_check.py cmp runs/forest_terrain/tree.npz runs/forest_terrain/forest.npz --gate
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def cmd_dump(a):
    import jax
    import jax.numpy as jnp
    import mujoco
    from nablaengine import engines, fixtures, jax_ffi, terrain as T
    if a.course:
        r = dict(np.load(a.ref))
        r.setdefault("iterations", np.int64(a.iterations))
        r.setdefault("geoms", np.str_(a.geoms or "body"))
        ter = T.Terrain.load(a.course)
    else:
        r = dict(np.load(a.ref))
        scene = a.scene or str(Path(a.ref).with_suffix(".mjb"))
        ter = T.Terrain.from_model(mujoco.MjModel.from_binary_path(scene))
    if a.model:
        meta = json.loads(Path(a.model).with_suffix(".json").read_text())
        m = mujoco.MjModel.from_xml_path(str(Path(a.model).with_suffix(".xml")))
        mt = {"foot_pairs": meta["foot_pairs"], "torso_body": meta["com_body"]}
    else:
        m, mt = fixtures.load_model(check=None)
    if a.engine == "forest":
        engines.prefer_forest(m)
    else:
        engines.prefer_tree(m)
    cfg = T.TerrainConfig(geoms=a.geoms or str(r["geoms"]), mode=a.mode, objects=a.objects, maxcon=a.maxcon)
    info = jax_ffi.init(model=m, foot_pairs=mt["foot_pairs"], com_body=mt.get("torso_body", "torso_link"), terrain=cfg,
                        max_rows=a.max_rows, contact_forces=a.contact_forces) if not a.no_terrain else \
        jax_ffi.init(model=m, foot_pairs=mt["foot_pairs"], com_body=mt.get("torso_body", "torso_link"),
                     max_rows=a.max_rows)
    assert engines.name(m) == ("forest" if a.engine == "forest" else "g1"), info["engine"]
    if not a.no_terrain:
        jax_ffi.set_terrain(ter)
    B = min(a.nworld, len(r["qpos"]))
    f = lambda k: jnp.asarray(np.ascontiguousarray(r[k][:B], np.float32))
    fr = f("friction") if "friction" in r else jnp.ones(B, jnp.float32)
    com = np.zeros((B, int(info.get("com_words", 3))), np.float32)
    if "com" in r:
        com[:, :3] = r["com"][:B]
    if a.objects:
        rng = np.random.default_rng(3)
        q = np.asarray(r["qpos"][:B], np.float64)
        obj = np.zeros((B, a.objects, T.OBJ_WORDS), np.float32)
        obj[:, 0, :3] = q[:, :3] + np.c_[rng.uniform(-0.3, 0.3, (B, 2)), -q[:, 2:3] + 0.05]
        obj[:, 0, 3:12] = np.eye(3).ravel()
        obj[:, 0, 12:15] = np.where(rng.random((B, 1)) < 0.5, 0.0, 0.15)
        com = T.pack_com(com[:, :3], obj)
    com = jnp.asarray(com)
    S, it = int(r["substeps"]) if "substeps" in r else 4, int(r["iterations"])
    if a.time:
        import time
        x = [f(k) for k in ("qpos", "qvel", "warm", "ctrl")]
        fwd = jax.jit(lambda q, v, w, u: jax_ffi.forward(q, v, w, u, fr, com, iterations=it, substeps=S)[:3])

        def lossb(q, v, u):
            q1, v1, _ = jax_ffi.step(q, v, x[2], u, fr, com, iterations=it, substeps=S)
            return jnp.sum(q1) + jnp.sum(v1)
        bwd = jax.jit(jax.grad(lossb, (0, 1, 2)))
        res = {}
        for name, fn, args in (("forward_ms", fwd, x), ("forward_backward_ms", bwd, (x[0], x[1], x[3]))):
            jax.block_until_ready(fn(*args))
            ts = []
            for _ in range(a.time):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(*args))
                ts.append(1e3 * (time.perf_counter() - t0))
            res[name] = float(np.median(ts))
        print(json.dumps(dict(engine=engines.name(m), nworld=B, reps=a.time, **res)))
        return
    out = jax_ffi.forward(f("qpos"), f("qvel"), f("warm"), f("ctrl"), fr, com, iterations=it, substeps=S, tape=True,
                          kinematics=True, contacts=True)
    q1, v1, w1, inf, con, tape = out[:6]
    kin = out[6:]
    ct = jax_ffi.contact_table(con)
    rng = np.random.default_rng(7)
    cq = jnp.asarray(rng.standard_normal((B, m.nq)).astype(np.float32))
    cv = jnp.asarray(rng.standard_normal((B, m.nv)).astype(np.float32))

    def loss(qpos, qvel, ctrl):
        q, v, _ = jax_ffi.step(qpos, qvel, f("warm"), ctrl, fr, com, iterations=it, substeps=S)
        return jnp.sum(q * cq) + jnp.sum(v * cv)
    gq, gv, gu = jax.grad(loss, (0, 1, 2))(f("qpos"), f("qvel"), f("ctrl"))
    kq = jax_ffi.kinematics_forward(q1, v1, com)
    kin_in = jax_ffi.kinematics_forward(f("qpos"), f("qvel"), com)
    kc = [jnp.asarray(rng.standard_normal(x.shape).astype(np.float32)) for x in kin_in]
    kin_vjp = jax_ffi.kinematics_backward(f("qpos"), f("qvel"), com, *kc)
    res = dict(engine=engines.name(m), qpos=np.asarray(q1), qvel=np.asarray(v1), info=np.asarray(inf),
               pair=np.asarray(ct["pair"]), dist=np.asarray(ct["dist"]), pos=np.asarray(ct["pos"]),
               normal=np.asarray(ct["normal"]), gq=np.asarray(gq), gv=np.asarray(gv), gu=np.asarray(gu),
               npair=m.npair, q0=np.asarray(f("qpos")), v0=np.asarray(f("qvel")), com=np.asarray(com))
    for i, x in enumerate(kin):
        res[f"kin{i}"] = np.asarray(x)
    for i, x in enumerate(kq):
        res[f"kinf{i}"] = np.asarray(x)
    for i, x in enumerate(kin_in):
        res[f"kinin{i}"] = np.asarray(x)
    for i, x in enumerate(kin_vjp):
        res[f"kinvjp{i}"] = np.asarray(x)
    if a.rollout:
        q, v, w_ = f("qpos"), f("qvel"), f("warm")
        traj, ncon_t = [], []
        for _ in range(a.rollout):
            q, v, w_, inf_r, con_r = jax_ffi.forward(q, v, w_, f("ctrl"), fr, com, iterations=it, substeps=S,
                                                     contacts=True)[:5]
            traj.append(np.asarray(q))
            ncon_t.append((np.asarray(jax_ffi.contact_table(con_r)["pair"]) >= 0).sum(1))
        res["roll_qpos"], res["roll_ncon"] = np.stack(traj, 1), np.stack(ncon_t, 1)
    if a.sensors:
        from nablaengine.sensors import g1 as sg1
        sens = sg1.G1Sensors(m, terrain=ter)
        objs = jnp.asarray(np.asarray(com)[:, 3:].reshape(B, a.objects, T.OBJ_WORDS)) if a.objects else None
        so = sens(kin_in[1], kin_in[2], objects=objs)
        for k in ("depth", "seg", "height_scan"):
            if k in so:
                res[f"sens_{k}"] = np.asarray(so[k])
        imu = sg1.Imu(m, dt=0.02)
        r_imu = imu.read(tuple(kq[1:]), tuple(kin_in[1:]))
        res["sens_gyro"], res["sens_accel"] = np.asarray(r_imu["gyro"]), np.asarray(r_imu["accel"])
    if "element" in ct:
        res["element"] = np.asarray(ct["element"])
        res["force"] = np.asarray(ct["force"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **res)
    print(json.dumps(dict(engine=engines.name(m), nworld=B, ncon_mean=float((res["pair"] >= 0).sum(1).mean()))))


def _rel(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return np.linalg.norm((x - y).reshape(len(x), -1), axis=1) / np.maximum(
        np.linalg.norm(y.reshape(len(y), -1), axis=1), 1e-12)


def cmd_cmp(a):
    A, B = np.load(a.a), np.load(a.b)
    n = len(A["qvel"])
    same_in = np.array_equal(A["q0"], B["q0"]) and np.array_equal(A["v0"], B["v0"])
    sets = np.zeros(n, bool)
    order = np.zeros(n, bool)
    for w in range(n):
        ka = [(int(p), *np.round(x / a.tol).astype(int)) for p, x in zip(A["pair"][w], A["pos"][w]) if p >= 0]
        kb = [(int(p), *np.round(x / a.tol).astype(int)) for p, x in zip(B["pair"][w], B["pos"][w]) if p >= 0]
        pa = [k[0] for k in ka]
        pb = [k[0] for k in kb]
        order[w] = pa == pb
        if len(ka) == len(kb):
            da = np.array([x for p, x in zip(A["pair"][w], A["pos"][w]) if p >= 0]).reshape(-1, 3)
            db = np.array([x for p, x in zip(B["pair"][w], B["pos"][w]) if p >= 0]).reshape(-1, 3)
            sets[w] = order[w] and (len(da) == 0 or np.abs(da - db).max() < a.tol)
    kin = {}
    for k in sorted(x for x in B.files if x[:3] == "kin" and x[3:].isdigit()):
        kin[k] = bool(np.array_equal(B[k], B["kinf" + k[3:]]))
    kin_ab = {k: bool(np.array_equal(A[k], B[k])) for k in sorted(B.files)
              if (k.startswith("kinin") or k.startswith("kinvjp")) and k in A.files}
    sens_ab = {k: bool(np.array_equal(A[k], B[k])) for k in sorted(B.files) if k.startswith("sens_") and k in A.files}
    sens_rel = {k: float(np.median(_rel(B[k], A[k]))) for k in ("sens_gyro", "sens_accel")
                if k in A.files and k in B.files}
    out = dict(nworld=n, same_inputs=bool(same_in), contact_pairs_in_order=float(order.mean()),
               contact_sets_equal=float(sets.mean()),
               ncon_mean=[float((A["pair"] >= 0).sum(1).mean()), float((B["pair"] >= 0).sum(1).mean())],
               qpos_rel_median=float(np.median(_rel(B["qpos"], A["qpos"]))),
               qvel_rel_median=float(np.median(_rel(B["qvel"], A["qvel"]))),
               qvel_rel_p99=float(np.percentile(_rel(B["qvel"], A["qvel"]), 99)),
               qvel_rel_median_sets_equal=float(np.median(_rel(B["qvel"], A["qvel"])[sets])) if sets.any() else None,
               kinematics_fused_vs_function_bitwise_B=kin,
               kinematics_input_state_A_eq_B_bitwise=kin_ab,
               sensors_A_eq_B_bitwise=sens_ab,
               imu_rel_median=sens_rel,
               grad_rel_median={k: float(np.median(_rel(B[k], A[k]))) for k in ("gq", "gv", "gu")},
               grad_rel_median_sets_equal={k: float(np.median(_rel(B[k], A[k])[sets])) for k in ("gq", "gv", "gu")}
               if sets.any() else None)
    if "roll_qpos" in A.files and "roll_qpos" in B.files:
        qa, qb = A["roll_qpos"], B["roll_qpos"]
        fa, fb = np.isfinite(qa).all((1, 2)), np.isfinite(qb).all((1, 2))
        ok = fa & fb
        T_ = qa.shape[1]
        d = [float(np.median(np.abs(qa[ok, k, :3] - qb[ok, k, :3]).max(1))) for k in range(T_)]
        out["rollout"] = dict(steps=T_, finite=[float(fa.mean()), float(fb.mean())],
                              base_pos_absdiff_median={str(k + 1): d[k] for k in sorted({0, 4, 9, 24, T_ - 1}) if k < T_},
                              ncon_mean=[float(A["roll_ncon"].mean()), float(B["roll_ncon"].mean())],
                              ncon_equal_step1=float((A["roll_ncon"][:, 0] == B["roll_ncon"][:, 0]).mean()),
                              base_z_mean_last=[float(qa[ok, -1, 2].mean()), float(qb[ok, -1, 2].mean())])
    if "element" in A.files and "element" in B.files:
        out["elements_equal"] = float(np.mean([np.array_equal(A["element"][w][A["pair"][w] >= 0],
                                                              B["element"][w][B["pair"][w] >= 0]) for w in range(n)]))
        out["force_rel_median"] = float(np.median(_rel(B["force"], A["force"])))
    ok = out["contact_sets_equal"] >= a.min_sets and max(out["grad_rel_median"].values()) <= a.max_grad
    out["pass"] = bool(ok)
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    if a.gate and not ok:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dump")
    d.add_argument("--engine", choices=("tree", "forest"), required=True)
    d.add_argument("--ref", required=True, help="terrain_check's REF.npz, or with --course the states")
    d.add_argument("--course", default=None, help="a course (Terrain.save); --ref then holds its states")
    d.add_argument("--iterations", type=int, default=10, help="(--course)")
    d.add_argument("--max-rows", type=int, default=None)
    d.add_argument("--scene", default=None)
    d.add_argument("--mode", default="analytic")
    d.add_argument("--geoms", default=None)
    d.add_argument("--model", default=None, help="an exported robot (MODEL without extension) instead of the G1")
    d.add_argument("--objects", type=int, default=0)
    d.add_argument("--maxcon", type=int, default=None)
    d.add_argument("--contact-forces", action="store_true", help="contact_forces=True (the table's terrain element)")
    d.add_argument("--nworld", type=int, default=1024)
    d.add_argument("--rollout", type=int, default=0, help="also a rollout of N control steps (ctrl held)")
    d.add_argument("--sensors", action="store_true", help="depth / height scan on the input state's kinematics, IMU")
    d.add_argument("--no-terrain", action="store_true", help="the build without terrain (the floor; --time baseline)")
    d.add_argument("--time", type=int, default=0, help="time N repetitions instead (no OUT written)")
    d.add_argument("--out", default=None)
    c = sp.add_parser("cmp")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--tol", type=float, default=1e-4, help="contact position tolerance (m)")
    c.add_argument("--min-sets", type=float, default=0.99)
    c.add_argument("--max-grad", type=float, default=1e-3)
    c.add_argument("--gate", action="store_true")
    c.add_argument("--out", default=None)
    a = p.parse_args()
    {"dump": cmd_dump, "cmp": cmd_cmp}[a.cmd](a)


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()

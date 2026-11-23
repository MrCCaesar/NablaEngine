"""Dynamic comparison of the sole mesh against the sphere fits (MuJoCo C, CPU).

Test A (foot drops): the ankle body alone (its own collision geoms, mass set to half the
robot) dropped on the floor from random orientations / velocities; compare its trajectory
and its rest pose with the mesh foot's.

Test B (walking policy): the official alpha_walking.onnx (pollen-robotics/microduck-policies)
on the whole robot through microduck_rl's scripts/infer_policy.py PolicyInference, headless,
for a set of velocity commands; compare velocity tracking, falls, trunk height, foot contact
timing. Actuator: the XML position servos (infer_policy --no-bam), NOT the BAM servo model the
policy was made with (bam not installed), the same in every scene.

Usage: python compare.py <microduck_rl checkout> <alpha_walking.onnx>
"""
import importlib.util
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
SCENES = ["scene_mesh.xml", "scene_spheres4.xml", "scene_spheres6.xml", "scene_spheres8.xml",
          "scene_spheres4_pairs.xml", "scene_spheres8_pairs.xml"]


def quat_tilt_deg(q):
    m = np.zeros(9); mujoco.mju_quat2Mat(m, q)
    return float(np.degrees(np.arccos(np.clip(m[8], -1, 1))))


# --------------------------------------------------------------------------- test A
def foot_only_model(scene, side="left", mass=0.37):
    """World with the floor + one free body holding the ankle body's collision geoms."""
    src = mujoco.MjSpec.from_file(str(MODEL / scene))
    body_name = "ankle_left" if side == "left" else "ankle_right"
    spec = mujoco.MjSpec()
    spec.option.timestep = src.option.timestep
    spec.option.integrator = src.option.integrator
    spec.option.cone = src.option.cone
    spec.option.iterations = src.option.iterations
    spec.option.ls_iterations = src.option.ls_iterations
    spec.option.ccd_iterations = src.option.ccd_iterations
    spec.compiler.meshdir = str(MODEL / "assets")
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01])
    fsrc = src.geom("floor")
    floor.solref, floor.solimp, floor.friction = fsrc.solref, fsrc.solimp, fsrc.friction
    b = spec.worldbody.add_body(name="foot")
    b.add_freejoint(name="free")
    geoms = [g for g in src.geoms if g.parent.name == body_name and g.name and "foot" in g.name]
    pts = []
    for g in geoms:
        ng = b.add_geom()
        for k in ("name", "type", "size", "pos", "quat", "condim", "priority", "friction", "solref", "solimp",
                  "contype", "conaffinity", "margin", "gap"):
            setattr(ng, k, getattr(g, k))
        ng.mass = 0; ng.density = 0
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            ng.meshname = g.meshname
            if not any(a.name == g.meshname for a in spec.meshes):
                ms = src.mesh(g.meshname)
                nm = spec.add_mesh(); nm.name = ms.name; nm.file = ms.file
        pts.append(np.array(g.pos))
    # common inertia for every variant: same mass/inertia, at the ankle body's own COM
    m0 = src.compile()
    bid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, body_name)
    b.ipos = m0.body_ipos[bid]
    b.mass = mass
    b.inertia = [mass * 0.03 ** 2 / 3] * 3                 # ~ a 6 cm solid block
    explicit = [(p.geomname1, p.geomname2) for p in src.pairs if p.geomname2 in [g.name for g in geoms]]
    for g1, g2 in explicit:
        sp = next(p for p in src.pairs if p.geomname2 == g2)
        p = spec.add_pair(); p.geomname1 = "floor"; p.geomname2 = g2
        p.condim, p.friction, p.solref, p.solimp = sp.condim, sp.friction, sp.solref, sp.solimp
    if explicit:
        floor.contype = 0; floor.conaffinity = 0
    return spec.compile()


def _sole_down_quat():
    """Rotation taking the left sole's normal (ankle frame, from fit_results.json) to world -z."""
    dn = np.array(json.loads((HERE / "fit_results.json").read_text())["left"]["down"])
    tgt = np.array([0, 0, -1.0])
    ax = np.cross(dn, tgt); s = np.linalg.norm(ax); c = float(dn @ tgt)
    q = np.zeros(4)
    if s < 1e-9:
        q[:] = [1, 0, 0, 0]
    else:
        mujoco.mju_axisAngle2Quat(q, ax / s, np.arctan2(s, c))
    return q


Q_SOLE_DOWN = _sole_down_quat()


def foot_drops(n=200, seed=0, T=1.0):
    rng = np.random.default_rng(seed)
    inits = []
    for _ in range(n):
        ax = rng.normal(size=3); ax[2] = 0; ax /= np.linalg.norm(ax)
        ang = np.radians(rng.uniform(0, 30))
        qt = np.zeros(4); mujoco.mju_axisAngle2Quat(qt, ax, ang)
        q = np.zeros(4); mujoco.mju_mulQuat(q, qt, Q_SOLE_DOWN)   # sole down, then tilted
        v = np.concatenate([rng.normal(0, 0.15, 2), [rng.uniform(-0.4, 0)]])
        w = rng.normal(0, 2.0, 3)
        inits.append((q, v, w, rng.uniform(0.0, 0.01)))
    out = {}
    for scene in SCENES:
        m = foot_only_model(scene); d = mujoco.MjData(m)
        # rest height reference: drop flat, slowly
        trajs, rest, ncon = [], [], []
        for q, v, w, h0 in inits:
            mujoco.mj_resetData(m, d)
            d.qpos[3:7] = q
            mujoco.mj_forward(m, d)
            # put the lowest point h0 above the floor
            lowest = min(d.contact[i].dist for i in range(d.ncon)) if d.ncon else None
            d.qpos[2] = 0.05; mujoco.mj_forward(m, d)
            zlow = _lowest_point(m, d)
            d.qpos[2] += h0 - zlow
            d.qvel[:3] = v; d.qvel[3:] = w
            tr = []
            nc = []
            for k in range(int(T / m.opt.timestep)):
                mujoco.mj_step(m, d)
                tr.append(np.concatenate([d.qpos[:7]]))
                nc.append(d.ncon)
            trajs.append(np.array(tr)); ncon.append(np.mean(nc[-50:]))
            rest.append((d.qpos[2], quat_tilt_deg(d.qpos[3:7]), np.linalg.norm(d.qvel)))
        out[scene] = dict(trajs=np.array(trajs), rest=np.array(rest), ncon=np.array(ncon))
    return out


def _lowest_point(m, d):
    z = np.inf
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = min(z, d.geom_xpos[g][2] - m.geom_size[g][0])
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            z = min(z, w[:, 2].min())
    return z


def summarize_drops(res):
    ref = res["scene_mesh.xml"]
    rows = {}
    for scene, r in res.items():
        dp = np.linalg.norm(r["trajs"][:, :, :3] - ref["trajs"][:, :, :3], axis=2) * 1e3       # mm
        # orientation difference
        qa, qb = r["trajs"][:, :, 3:7], ref["trajs"][:, :, 3:7]
        dq = np.degrees(2 * np.arccos(np.clip(np.abs((qa * qb).sum(-1)), 0, 1)))
        rows[scene] = dict(
            pos_diff_mm_median_end=float(np.median(dp[:, -1])), pos_diff_mm_p90_end=float(np.percentile(dp[:, -1], 90)),
            rot_diff_deg_median_end=float(np.median(dq[:, -1])), rot_diff_deg_p90_end=float(np.percentile(dq[:, -1], 90)),
            pos_diff_mm_median_100ms=float(np.median(dp[:, 19])),
            rest_z_mm=float(np.median(r["rest"][:, 0]) * 1e3), rest_tilt_deg_median=float(np.median(r["rest"][:, 1])),
            still_frac=float(np.mean(r["rest"][:, 2] < 1e-2)), ncon_rest=float(np.mean(r["ncon"])))
    return rows


# --------------------------------------------------------------------------- test B
def load_infer(repo):
    spec = importlib.util.spec_from_file_location("infer_policy", Path(repo) / "scripts" / "infer_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def walk(ip, onnx, scene, cmd, T=10.0, push=None, seed=0):
    m = mujoco.MjModel.from_xml_path(str(MODEL / scene))
    d = mujoco.MjData(m)
    pol = ip.PolicyInference(m, d, walking_onnx_path=str(onnx), new_cmd_obs=True, use_projected_gravity=True)
    pol.set_vel_cmd(*cmd)
    fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa, va = m.jnt_qposadr[fj], m.jnt_dofadr[fj]
    d.qpos[qa:qa + 3] = [0, 0, 0.125]; d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
    for i, qi in enumerate(pol.joint_qpos_indices):
        d.qpos[qi] = pol.default_pose[i]
    pol.set_position_targets(pol.default_pose)
    mujoco.mj_forward(m, d)
    feet = {s: [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"{s}_foot")]
            for s in ("left", "right")}
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    rng = np.random.default_rng(seed)
    log = dict(v=[], z=[], tilt=[], contact=[], ncon=[], fz=[], selfcon=[])
    fell = None
    n = int(T / (4 * m.opt.timestep))
    f6 = np.zeros(6)
    for k in range(n):
        a = pol.infer(); pol.apply_action(a)
        if push is not None and k % 100 == 50:            # lateral/fwd shove every 2 s, 0.1 s long
            d.xfrc_applied[trunk, :2] = rng.normal(0, push, 2)
        if push is not None and k % 100 == 55:
            d.xfrc_applied[trunk, :2] = 0
        for _ in range(4):
            mujoco.mj_step(m, d)
        q = d.qpos[qa + 3:qa + 7]
        vb = pol.quat_rotate_inverse(q.astype(np.float32), d.qvel[va:va + 3].astype(np.float32))
        tilt = quat_tilt_deg(q)
        c = {"left": 0, "right": 0}; fz = {"left": 0.0, "right": 0.0}; sc = 0
        for i in range(d.ncon):
            con = d.contact[i]
            gs = {con.geom1, con.geom2}
            if floor not in gs:
                sc += 1
                continue
            for s in c:
                if gs & set(feet[s]):
                    c[s] += 1
                    mujoco.mj_contactForce(m, d, i, f6); fz[s] += f6[0]
        log["v"].append([vb[0], vb[1], d.qvel[va + 5]]); log["z"].append(d.qpos[qa + 2]); log["tilt"].append(tilt)
        log["contact"].append([c["left"] > 0, c["right"] > 0]); log["ncon"].append(d.ncon)
        log["fz"].append([fz["left"], fz["right"]]); log["selfcon"].append(sc)
        if tilt > 60 and fell is None:
            fell = k * 4 * m.opt.timestep
            break
    L = {k: np.array(v) for k, v in log.items()}
    s = slice(len(L["z"]) // 5, None)                    # skip the first 20 %
    cont = L["contact"][s]
    # stance/swing switching: touchdowns per foot per second
    td = (np.diff(cont.astype(int), axis=0) == 1).sum(0) / (len(cont) * 4 * m.opt.timestep)
    return dict(fell_at=fell, v_mean=L["v"][s].mean(0).tolist(), v_err=(L["v"][s].mean(0) - np.array(cmd)).tolist(),
                z_mean_mm=float(L["z"][s].mean() * 1e3), tilt_mean=float(L["tilt"][s].mean()),
                double_support=float(cont.all(1).mean()), flight=float((~cont.any(1)).mean()),
                touchdowns_per_s=td.tolist(), ncon_mean=float(L["ncon"][s].mean()), ncon_max=int(L["ncon"].max()),
                selfcon_frac=float((L["selfcon"][s] > 0).mean()))


# --------------------------------------------------------------------------- test C
# Whole robot, our own PD at the policies' DEFAULT_POSE (numbers from microduck_rl's
# scripts/infer_policy.py), scripted squat + sway, then random pushes. No policy, no BAM.
DEFAULT_POSE = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
                         0.0, 0.0873, 0.4579, 0.0049, -0.4530])
KP = 2.0   # infer_policy's standby stiffness (XML kp 0.55 ~ kp_fw 200; standby 2.0)


def scripted(scene, seed=0, T=14.0, push_n=1.5):
    m = mujoco.MjModel.from_xml_path(str(MODEL / scene)); d = mujoco.MjData(m)
    m.actuator_gainprm[:, 0] = KP; m.actuator_biasprm[:, 1] = -KP
    d.qpos[:3] = [0, 0, 0.125]; d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[7:] = DEFAULT_POSE
    d.ctrl[:] = DEFAULT_POSE
    mujoco.mj_forward(m, d)
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    side_of = {}
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if n.startswith("left_foot"): side_of[g] = 0
        if n.startswith("right_foot"): side_of[g] = 1
    rng = np.random.default_rng(seed)
    dt = m.opt.timestep
    L = dict(t=[], qpos=[], w=[], ncon=[], fz=[], slip=[], fell=None)
    f6 = np.zeros(6); vel6 = np.zeros(6)
    import time
    t0 = time.perf_counter()
    for k in range(int(T / dt)):
        t = k * dt
        u = DEFAULT_POSE.copy()
        if 2.0 <= t < 8.0:                                   # squat 0.5 Hz + sway 0.8 Hz
            s = 0.06 * np.sin(2 * np.pi * 0.5 * (t - 2)); r = 0.06 * np.sin(2 * np.pi * 0.8 * (t - 2))
            u[[2, 3, 4]] += [s, -s, -s]; u[[11, 12, 13]] -= [s, -s, -s]   # signs found by test: trunk bobs, no fall
            u[1] += r; u[10] -= r
        d.ctrl[:] = u
        if t >= 8.0 and k % int(1.0 / dt) == 0:
            d.xfrc_applied[trunk, :2] = rng.normal(0, push_n, 2)
        if t >= 8.0 and k % int(1.0 / dt) == int(0.1 / dt):
            d.xfrc_applied[trunk, :2] = 0
        mujoco.mj_step(m, d)
        nc = [0, 0]; fz = [0.0, 0.0]; sl = []
        for i in range(d.ncon):
            c = d.contact[i]
            g = c.geom2 if c.geom1 == floor else c.geom1 if c.geom2 == floor else None
            if g is None or g not in side_of:
                continue
            sd = side_of[g]; nc[sd] += 1
            mujoco.mj_contactForce(m, d, i, f6); fz[sd] += f6[0]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, g, vel6, 0)
            # contact-point tangential speed
            p = c.pos - d.geom_xpos[g]
            v = vel6[3:] + np.cross(vel6[:3], p)
            sl.append(np.linalg.norm(v[:2]))
        L["t"].append(t); L["qpos"].append(d.qpos[:7].copy()); L["w"].append(d.qvel[3:6].copy())
        L["ncon"].append(nc); L["fz"].append(fz); L["slip"].append(max(sl) if sl else 0.0)
        if quat_tilt_deg(d.qpos[3:7]) > 45 and L["fell"] is None:
            L["fell"] = t
    wall = time.perf_counter() - t0
    A = {k: np.array(v) for k, v in L.items() if k != "fell"}
    A["fell"] = L["fell"]; A["wall_per_step_us"] = wall / len(L["t"]) * 1e6
    return A


def summarize_scripted(runs):
    ref = runs["scene_mesh.xml"]
    rows = {}
    for s, A in runs.items():
        t = A["t"]
        quiet = (t > 0.5) & (t < 2.0)
        phases = {"stand": (t > 0.5) & (t < 2), "squat_sway": (t >= 2) & (t < 8), "push": t >= 8}
        row = dict(fell=A["fell"], wall_per_step_us=A["wall_per_step_us"],
                   quiet_angvel_rms=float(np.sqrt((A["w"][quiet] ** 2).sum(1).mean())),
                   quiet_z_mm=float(A["qpos"][quiet, 2].mean() * 1e3))
        for ph, sel in phases.items():
            nc = A["ncon"][sel]
            row[f"{ph}_ncon_per_foot"] = float(nc.mean())
            row[f"{ph}_ncon_max_per_foot"] = int(nc.max())
            row[f"{ph}_slip_p95_mm_s"] = float(np.percentile(A["slip"][sel], 95) * 1e3)
            row[f"{ph}_tilt_mean"] = float(np.mean([quat_tilt_deg(q[3:7]) for q in A["qpos"][sel]]))
        dpos = np.linalg.norm(A["qpos"][:, :3] - ref["qpos"][:, :3], axis=1) * 1e3
        for tt in (1.0, 2.0, 5.0, 8.0, 14.0):
            i = min(int(tt / 0.005) - 1, len(dpos) - 1)
            row[f"trunk_diff_mm_t{tt:g}"] = float(dpos[i])
        row["final_xy_mm"] = (A["qpos"][-1, :2] * 1e3).tolist()
        rows[s] = row
    return rows


def main():
    res = {}
    print("== Test A: foot drops (200 random drops, 1 s each)")
    drops = summarize_drops(foot_drops())
    res["foot_drops"] = drops
    for s, r in drops.items():
        print(f"  {s:26s} end pos diff vs mesh med {r['pos_diff_mm_median_end']:6.2f} mm p90 {r['pos_diff_mm_p90_end']:6.2f} | "
              f"rot med {r['rot_diff_deg_median_end']:5.2f} p90 {r['rot_diff_deg_p90_end']:5.2f} deg | "
              f"rest z {r['rest_z_mm']:6.2f} mm still {r['still_frac']:.2f} ncon {r['ncon_rest']:.1f}")
    print("== Test C: whole robot, PD kp 2.0, stand / squat+sway / pushes, 3 seeds")
    res["scripted"] = {}
    for seed in range(3):
        rows = summarize_scripted({s: scripted(s, seed=seed) for s in SCENES})
        res["scripted"][seed] = rows
        for s, r in rows.items():
            print(f"  seed {seed} {s:26s} " + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    (HERE / "compare_results.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", HERE / "compare_results.json")
    if len(sys.argv) < 3:
        return
    repo, onnx = sys.argv[1], sys.argv[2]
    print("== Test B: alpha_walking.onnx, XML servos, 10 s per command")
    ip = load_infer(repo)
    cmds = [(0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (0.3, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.15, 0.0), (0.0, 0.0, 1.0), (0.2, 0.0, 0.8)]
    res["walk"] = {}
    for scene in SCENES:
        res["walk"][scene] = {}
        for cmd in cmds:
            r = walk(ip, onnx, scene, cmd)
            res["walk"][scene][str(cmd)] = r
            print(f"  {scene:26s} cmd {cmd}: fell {r['fell_at']} v {np.round(r['v_mean'], 3).tolist()} z {r['z_mean_mm']:.1f} mm "
                  f"DS {r['double_support']:.2f} TD/s {np.round(r['touchdowns_per_s'], 2).tolist()} ncon {r['ncon_mean']:.1f}/{r['ncon_max']} "
                  f"self {r['selfcon_frac']:.2f}")
        r = walk(ip, onnx, scene, (0.15, 0.0, 0.0), T=20.0, push=1.0)
        res["walk"][scene]["push"] = r
        print(f"  {scene:26s} pushes 1 N rms: fell {r['fell_at']} v {np.round(r['v_mean'], 3).tolist()}")
    (HERE / "compare_results.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", HERE / "compare_results.json")


if __name__ == "__main__":
    main()

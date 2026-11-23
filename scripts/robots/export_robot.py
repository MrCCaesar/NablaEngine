"""Export a robot scene as an engine model: an mjlab robot's flat velocity scene (run in an mjlab environment)
or a scene MJCF such as MuJoCo Playground's (robot + floor plane)."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ROBOTS = {
    "go1": ("mjlab.tasks.velocity.config.go1.env_cfgs:unitree_go1_flat_env_cfg",
            [f"{n}_foot_collision" for n in ("FR", "FL", "RR", "RL")], "trunk"),
}


def mixed(m, g1, g2):
    """MuJoCo's contact parameters of dynamic geom pair (g1, g2): condim, friction (5), solref, solimp, margin, gap."""
    p1, p2 = m.geom_priority[g1], m.geom_priority[g2]
    f1, f2 = m.geom_friction[g1], m.geom_friction[g2]
    if p1 != p2:
        g = g1 if p1 > p2 else g2
        condim, fr = m.geom_condim[g], m.geom_friction[g]
        solref, solimp = m.geom_solref[g], m.geom_solimp[g]
    else:
        condim, fr = max(m.geom_condim[g1], m.geom_condim[g2]), np.maximum(f1, f2)
        s1, s2 = m.geom_solmix[g1], m.geom_solmix[g2]
        mix = 0.5 if s1 + s2 < mujoco.mjMINVAL else s1 / (s1 + s2)
        r1, r2 = m.geom_solref[g1], m.geom_solref[g2]
        solref = mix * r1 + (1 - mix) * r2 if (r1[0] > 0 and r2[0] > 0) else np.minimum(r1, r2)
        solimp = mix * m.geom_solimp[g1] + (1 - mix) * m.geom_solimp[g2]
    margin = m.geom_margin[g1] + m.geom_margin[g2]
    gap = m.geom_gap[g1] + m.geom_gap[g2]
    return int(condim), [fr[0], fr[0], fr[1], fr[2], fr[2]], list(solref), list(solimp), float(margin), float(gap)


def convert(spec, out, name, source, foot_geoms, com_body, substeps, init_qpos, feet_only=False, locomotion=False,
            servo=None, fusestatic=False, invweight="geom", write_meta=True, drop_unpaired_meshes=False):
    """The engine model of scene spec (robot + floor) written to OUT.xml / OUT.json: visual meshes dropped, the static
    bodies' geoms onto the world body, actuators in joint order with their force limits on the joints, dynamic
    contacts as explicit pairs (MuJoCo's mixed parameters); init_qpos(m): the initial state as qpos; feet_only: only
    the foot geoms' pairs with the floor (MuJoCo Playground's feet-only collision, for full-collision models);
    locomotion: the usual locomotion contact model (pyramidal cone, condim 3, no margin) for models such as MuJoCo
    Menagerie's (elliptic, condim 6); servo (kp, kd): torque motors as position servos (kd as joint damping);
    fusestatic: MuJoCo's compiler option fusestatic (bodies without joints merged into their parents, e.g. Apollo's
    camera frames welded to the base: the engine wants one hinge per body after the root); invweight (with
    fusestatic): the fused bodies' contact softness, "geom" (the default: the unfused model's body_invweight0 per geom,
    in the custom numeric nablaengine.model.invweight.NUMERIC that the engines read, written where some geom's value
    differs from its body's; MuJoCo C / MuJoCo Warp ignore it, the model exported without fusestatic is their
    equivalent) or "converted" (MuJoCo's fusestatic values: as nablaengine.envs.convert.to_engine's options);
    drop_unpaired_meshes: collision meshes in no contact pair dropped (feet_only models with mesh collision geoms;
    checked not to change any body's inertia). Returns the metadata (written to OUT.json unless write_meta is False)."""
    if invweight not in ("geom", "converted"):
        raise ValueError(f"invweight={invweight!r}: 'geom' or 'converted'")
    if fusestatic:
        spec.compiler.fusestatic = True
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.contype == 0 and g.conaffinity == 0:
            spec.delete(g)
    for b in spec.bodies:
        for i, g in enumerate(b.geoms):
            if not g.name and (g.contype or g.conaffinity):
                g.name = f"{b.name or 'world'}_collision{i}"
    keep = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
    for x in list(spec.meshes):
        if x.name not in keep:
            spec.delete(x)
    for t in list(spec.textures):
        if t.file:
            for mat in spec.materials:
                mat.textures = ["" if x == t.name else x for x in mat.textures]
            spec.delete(t)
    for body in list(spec.worldbody.bodies):
        if body.joints or body.bodies:
            continue
        assert not np.any(body.pos) and np.allclose(body.quat, [1, 0, 0, 0]), body.name
        for g in list(body.geoms):
            ng = spec.worldbody.add_geom()
            for k in ("type", "size", "pos", "quat", "rgba", "material", "friction", "condim", "solref", "solimp",
                      "solmix", "priority", "margin", "gap", "contype", "conaffinity", "group"):
                setattr(ng, k, getattr(g, k))
            gname = g.name
            spec.delete(g)
            ng.name = gname
        for light in list(body.lights):
            spec.delete(light)
        spec.delete(body)
    order = [j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
    acts = {act.target: act for act in spec.actuators}
    assert sorted(acts) == sorted(order), (sorted(acts), order)
    keep = ("name", "trntype", "target", "dyntype", "gaintype", "biastype", "dynprm", "gainprm", "biasprm", "gear",
            "ctrllimited", "ctrlrange", "forcelimited", "forcerange", "actlimited", "actrange", "group")
    saved = [{k: (list(v) if hasattr(v, "__len__") and not isinstance(v, str) else v)
              for k in keep for v in [getattr(acts[n], k)]} for n in order]
    for act in list(spec.actuators):
        spec.delete(act)
    for d in saved:
        if servo and d["biastype"] == mujoco.mjtBias.mjBIAS_NONE:
            j = spec.joint(d["target"])
            d["forcelimited"], d["forcerange"] = True, list(d["ctrlrange"])
            lim = j.range[0] < j.range[1]
            d["ctrlrange"], d["ctrllimited"] = (list(j.range), True) if lim else ([0.0, 0.0], False)
            d["biastype"] = mujoco.mjtBias.mjBIAS_AFFINE
            d["gainprm"] = [servo[0]] + [0.0] * (len(d["gainprm"]) - 1)
            d["biasprm"] = [0.0, -servo[0]] + [0.0] * (len(d["biasprm"]) - 2)
            j.damping = [j.damping[0] + servo[1], *j.damping[1:]]
        act = spec.add_actuator()
        for k, v in d.items():
            setattr(act, k, v)
    for act in spec.actuators:
        if act.forcelimited and act.forcerange[0] < act.forcerange[1]:
            j = spec.joint(act.target)
            j.actfrcrange, j.actfrclimited = list(act.forcerange), True
            act.forcelimited, act.forcerange = False, [0.0, 0.0]
    for s in list(spec.sensors):
        spec.delete(s)
    filled = None
    if fusestatic and invweight == "geom":
        from nablaengine.model import invweight as iw
        filled = iw.fill_spec(spec)
    m = spec.compile()
    weld = m.body_weldid
    pairs = []
    for g1 in range(m.ngeom):
        for g2 in range(g1 + 1, m.ngeom):
            if not ((m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1])):
                continue
            b1, b2 = weld[m.geom_bodyid[g1]], weld[m.geom_bodyid[g2]]
            if b1 == b2:
                continue
            if b1 != 0 and b2 != 0 and (m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1):
                continue
            if feet_only and not ((b1 == 0 and m.geom(g2).name in foot_geoms) or
                                  (b2 == 0 and m.geom(g1).name in foot_geoms)):
                continue
            pairs.append((g1, g2))
    names = [m.geom(g).name for g in range(m.ngeom)]
    for i, (g1, g2) in enumerate(pairs):
        condim, fr, solref, solimp, margin, gap = mixed(m, g1, g2)
        if locomotion:
            condim, margin, gap = min(condim, 3), 0.0, 0.0
        spec.add_pair(name=f"p{i}_{names[g1]}__{names[g2]}", geomname1=names[g1], geomname2=names[g2], condim=condim,
                      friction=fr, solref=solref, solimp=solimp, margin=margin, gap=gap)
    for g in spec.geoms:
        g.contype = g.conaffinity = 0
    if drop_unpaired_meshes:
        paired = {names[g] for p in pairs for g in p}
        for g in list(spec.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.name not in paired:
                spec.delete(g)
        used = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
        for x in list(spec.meshes):
            if x.name not in used:
                spec.delete(x)
        md = spec.compile()
        for k in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
            assert np.array_equal(getattr(md, k), getattr(m, k)), f"drop_unpaired_meshes changes {k}"
    if locomotion:
        spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if spec.meshes:
        import shutil
        mdir = Path(source).parent / (spec.meshdir or "")
        (out.parent / "meshes").mkdir(exist_ok=True)
        for x in spec.meshes:
            if x.file:
                shutil.copyfile(mdir / x.file, out.parent / "meshes" / Path(x.file).name)
                x.file = str((out.parent / "meshes" / Path(x.file).name).resolve())
        spec.meshdir = spec.compiler.meshdir = ""
    m2, ginvw = iw.attach(spec, filled) if filled is not None else (spec.compile(), None)
    spec.to_file(str(out.with_suffix(".xml")))
    xml = out.with_suffix(".xml").read_text()
    xml = xml.replace(str((out.parent / "meshes").resolve()) + "/", "meshes/")
    out.with_suffix(".xml").write_text("\n".join(l for l in xml.split("\n") if l.strip() != "<default/>"))
    if ginvw is not None:
        _full_precision_numeric(out.with_suffix(".xml"), m2)
    feet = foot_geoms
    is_foot = lambda g: m2.geom(int(g)).name.split("/")[-1] in feet
    static = lambda g: m2.geom_bodyid[int(g)] == 0
    foot_pairs = [p for p in range(m2.npair) if (is_foot(m2.pair_geom1[p]) and static(m2.pair_geom2[p]))
                  or (is_foot(m2.pair_geom2[p]) and static(m2.pair_geom1[p]))]
    q0 = init_qpos(m2)
    iwmeta = dict(invweight=invweight, geom_invweight=ginvw is not None) if fusestatic else {}
    dropm = dict(drop_unpaired_meshes=True) if drop_unpaired_meshes else {}
    meta = dict(robot=name, source=source, feet_only=feet_only, locomotion=locomotion, servo=servo, fusestatic=fusestatic, **iwmeta, **dropm, init_qpos=[float(x) for x in q0], foot_pairs=foot_pairs, foot_geoms=feet,
                com_body=next(m2.body(b).name for b in range(m2.nbody) if m2.body(b).name.split("/")[-1] == com_body), substeps_per_control_step=int(substeps), nq=m2.nq, nv=m2.nv, nu=m2.nu,
                nbody=m2.nbody, npair=m2.npair, timestep=float(m2.opt.timestep), mujoco_version=mujoco.__version__)
    if write_meta:
        _write_meta(out, meta)
    return meta


def _write_meta(out, meta):
    Path(out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


def _full_precision_numeric(path, m):
    import re
    from nablaengine.model import invweight as iw
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, iw.NUMERIC)
    adr = int(m.numeric_adr[i])
    data = " ".join(repr(float(x)) for x in m.numeric_data[adr:adr + m.ngeom])
    xml = path.read_text()
    pat = re.compile(r'(<numeric name="' + iw.NUMERIC + r'"[^>]*data=")[^"]*(")')
    assert len(pat.findall(xml)) == 1
    path.write_text(pat.sub(lambda mo: mo.group(1) + data + mo.group(2), xml))
    m1 = mujoco.MjModel.from_xml_path(str(path))
    j = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_NUMERIC, iw.NUMERIC)
    assert np.array_equal(m1.numeric_data[m1.numeric_adr[j]:m1.numeric_adr[j] + m1.ngeom], m.numeric_data[adr:adr + m.ngeom])


def from_mjlab(robot, out):
    import importlib
    import re
    from mjlab.scene import Scene
    mod, fn = ROBOTS[robot][0].split(":")
    cfg = getattr(importlib.import_module(mod), fn)()
    cfg.scene.num_envs = 1
    scene = Scene(cfg.scene, device="cpu")
    spec = scene.spec
    mo = cfg.sim.mujoco
    o = spec.option
    o.timestep, o.iterations, o.ls_iterations, o.tolerance, o.ls_tolerance = (mo.timestep, mo.iterations,
                                                                             mo.ls_iterations, mo.tolerance,
                                                                             mo.ls_tolerance)
    o.impratio, o.gravity, o.ccd_iterations = mo.impratio, list(mo.gravity), mo.ccd_iterations
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST if mo.integrator == "implicitfast" else o.integrator
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL if mo.cone == "pyramidal" else mujoco.mjtCone.mjCONE_ELLIPTIC
    o.solver = mujoco.mjtSolver.mjSOL_NEWTON
    ent = cfg.scene.entities["robot"]

    def init_qpos(m):
        q0 = m.qpos0.copy()
        q0[:3] = ent.init_state.pos
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            nm = m.joint(j).name.split("/")[-1]
            for pat, val in ent.init_state.joint_pos.items():
                if re.fullmatch(pat, nm):
                    q0[m.jnt_qposadr[j]] = val
        return q0

    convert(spec, out, robot, ROBOTS[robot][0], ROBOTS[robot][1], ROBOTS[robot][2], cfg.decimation, init_qpos)


def from_mjcf(path, out, name, foot_geoms, com_body, substeps, key=None, feet_only=False, z=None, foot_class=None,
              locomotion=False, servo=None, fusestatic=False, invweight="geom", drop_unpaired_meshes=False,
              overrides=None):
    """A scene MJCF (robot + floor plane), e.g. MuJoCo Playground's; key: the keyframe of the initial state; z: the
    root height of the initial state instead (models without a keyframe); foot_class: the foot geoms are those of these
    default classes (comma-separated; MuJoCo Menagerie's unnamed foot geoms, named BODY_foot here, BODY_foot2, ... for a
    body's further ones); overrides: the model changes a training environment makes in code after loading the scene
    (e.g. MuJoCo Playground's Barkour joystick), a dict of kp (every actuator's gainprm[0] = -biasprm[1]), damping
    (every hinge's damping) and timestep, None entries unchanged."""
    spec = mujoco.MjSpec.from_file(str(path))
    ov = overrides or {}
    if ov.get("kp") is not None:
        for act in spec.actuators:
            act.gainprm = [ov["kp"], *act.gainprm[1:]]
            act.biasprm = [act.biasprm[0], -ov["kp"], *act.biasprm[2:]]
    if ov.get("damping") is not None:
        for j in spec.joints:
            if j.type == mujoco.mjtJoint.mjJNT_HINGE:
                j.damping = [ov["damping"], *j.damping[1:]]
    if ov.get("timestep") is not None:
        spec.option.timestep = ov["timestep"]
    if foot_class:
        foot_geoms, seen = [], {}
        classes = foot_class.split(",")
        for g in spec.geoms:
            if g.classname and g.classname.name in classes:
                k = seen[g.parent.name] = seen.get(g.parent.name, 0) + 1
                g.name = g.name or f"{g.parent.name}_foot" + (str(k) if k > 1 else "")
                foot_geoms.append(g.name)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON

    def init_qpos(m):
        q0 = m.qpos0.copy() if key is None else m.key(key).qpos.copy()
        if z is not None:
            q0[2] = z
        return q0

    ref = spec.copy() if fusestatic and invweight == "geom" else None
    args = (name, str(path), foot_geoms, com_body, substeps, init_qpos, feet_only, locomotion, servo)
    meta = convert(spec, out, *args, fusestatic, invweight, write_meta=False, drop_unpaired_meshes=drop_unpaired_meshes)
    if any(v is not None for v in ov.values()):
        meta["overrides"] = {k: v for k, v in ov.items() if v is not None}
    if ref is not None and meta["geom_invweight"]:
        rout = Path(str(out) + "_reference")
        convert(ref, rout, *args, False, write_meta=False, drop_unpaired_meshes=drop_unpaired_meshes)
        meta["reference_model"] = rout.name
    _write_meta(out, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=None, help="an mjlab robot (ROBOTS)")
    ap.add_argument("--mjcf", default=None, help="a scene MJCF instead (robot + floor plane)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--foot-geoms", default="", help="comma-separated (with --mjcf)")
    ap.add_argument("--com-body", default="torso_link")
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--key", default=None)
    ap.add_argument("--foot-class", default=None, help="the foot geoms by default class, comma-separated (with --mjcf)")
    ap.add_argument("--feet-only", action="store_true", help="only the foot geoms' pairs with the floor")
    ap.add_argument("--z", type=float, default=None, help="the initial root height (with --mjcf, no keyframe)")
    ap.add_argument("--locomotion", action="store_true", help="pyramidal cone, condim <= 3, no margin (Menagerie)")
    ap.add_argument("--servo", default=None, help="KP,KD: torque motors as position servos (with --mjcf)")
    ap.add_argument("--fusestatic", action="store_true", help="merge bodies without joints into their parents")
    ap.add_argument("--invweight", choices=("geom", "converted"), default="geom",
                    help="with --fusestatic: the contact softness of the fused bodies' geoms and their parents': the "
                         "unfused model's per geom (geom, the default) or MuJoCo's fusestatic values (converted)")
    ap.add_argument("--drop-unpaired-meshes", action="store_true",
                    help="drop the collision meshes in no contact pair (e.g. with --feet-only)")
    ap.add_argument("--kp", type=float, default=None, help="every actuator's position gain (with --mjcf)")
    ap.add_argument("--damping", type=float, default=None, help="every hinge's damping (with --mjcf)")
    ap.add_argument("--timestep", type=float, default=None, help="the model's timestep (with --mjcf)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mjcf:
        from_mjcf(a.mjcf, a.out, a.name or Path(a.mjcf).stem, [g for g in a.foot_geoms.split(",") if g], a.com_body,
                  a.substeps, a.key, a.feet_only, a.z, a.foot_class, a.locomotion,
                  [float(x) for x in a.servo.split(",")] if a.servo else None, a.fusestatic, a.invweight,
                  a.drop_unpaired_meshes, dict(kp=a.kp, damping=a.damping, timestep=a.timestep))
    else:
        from_mjlab(a.robot, a.out)


if __name__ == "__main__":
    main()

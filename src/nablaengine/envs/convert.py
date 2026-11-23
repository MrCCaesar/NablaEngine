"""MJCF models as the engine takes them (nablaengine.envs): the same physics in the structure the forest engine needs."""
import copy
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco
import numpy as np

D = mujoco.mjtDisableBit
J = mujoco.mjtJoint
TINY = 1e-12
_FRAME = ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis")


@dataclass
class Converted:
    model: mujoco.MjModel
    original: mujoco.MjModel
    xml: str
    body_map: np.ndarray
    notes: list = field(default_factory=list)
    fused: dict = field(default_factory=dict)
    geom_invweight: np.ndarray = None


def _name_all(root):
    nb = 0
    for body in root.iter("body"):
        if not body.get("name"):
            body.set("name", f"b{nb}")
        nb += 1
    for body in [root.find("worldbody")] + list(root.iter("body")):
        bname = body.get("name", "world")
        for kind in ("joint", "freejoint", "geom"):
            for i, el in enumerate(body.findall(kind)):
                if not el.get("name"):
                    el.set("name", f"{bname}_{'joint' if kind != 'geom' else 'geom'}{i}")
    used = {el.get("name") for el in root.iter() if el.get("name")}
    parents = {c: p for p in root.iter() for c in p}
    count = {}
    for wb in root.findall("worldbody"):
        for el in wb.iter():
            if el.tag not in ("joint", "freejoint", "geom") or el.get("name"):
                continue
            p = parents[el]
            while p.tag not in ("body", "worldbody"):
                p = parents[p]
            stem = f"{p.get('name', 'world')}_frame_{'joint' if el.tag != 'geom' else 'geom'}"
            while f"{stem}{count.get(stem, 0)}" in used:
                count[stem] = count.get(stem, 0) + 1
            el.set("name", f"{stem}{count.get(stem, 0)}")
            used.add(el.get("name"))
    return root


def _inline_includes(root, assets):
    norm = {k[2:] if k.startswith("./") else k: v for k, v in assets.items()}
    while True:
        parents = {c: p for p in root.iter() for c in p}
        inc = next(root.iter("include"), None)
        if inc is None:
            return root
        f = inc.get("file")
        f = f[2:] if f.startswith("./") else f
        sub = ET.fromstring(norm[f])
        parent = parents[inc]
        i = list(parent).index(inc)
        parent.remove(inc)
        for k, child in enumerate(sub):
            parent.insert(i + k, child)


def _fmt(v):
    return " ".join(repr(float(x)) for x in v)


def _shift_contents(body, c):
    for el in body:
        if el.tag not in ("geom", "site", "inertial", "camera", "light", "frame", "body"):
            continue
        if el.get("fromto"):
            v = np.array(el.get("fromto").split(), float)
            el.set("fromto", _fmt(v - np.concatenate([c, c])))
            continue
        p = np.array(el.get("pos", "0 0 0").split(), float)
        el.set("pos", _fmt(p - c))


def _split_bodies(root, m0):
    parents = {c: p for p in root.iter() for c in p}
    shifted = {}
    for body in list(root.iter("body")):
        joints = [el for el in body if el.tag in ("joint", "freejoint")]
        if not joints or any(el.tag == "freejoint" for el in joints):
            continue
        ids = [mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, el.get("name")) for el in joints]
        types = [int(m0.jnt_type[j]) for j in ids]
        if J.mjJNT_FREE in types:
            continue
        assert all(t in (J.mjJNT_HINGE, J.mjJNT_SLIDE) for t in types), f"body {body.get('name')}: ball joint"
        anchors, c = [], np.zeros(3)
        for j, t in zip(ids, types):
            if t == J.mjJNT_HINGE:
                c = np.array(m0.jnt_pos[j], float)
            anchors.append(c.copy())
        for el in joints:
            el.set("pos", "0 0 0")
        if len(joints) == 1 and not anchors[0].any():
            continue
        b = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, body.get("name"))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m0.body_quat[b])
        parent = parents[body]
        idx = list(parent).index(body)
        prev = None
        for k, (el, a) in enumerate(zip(joints, anchors)):
            last = k == len(joints) - 1
            w = body if last else ET.Element("body", name=f"{body.get('name')}__{el.get('name')}")
            if not last:
                ET.SubElement(w, "inertial", pos="0 0 0", mass=repr(TINY), diaginertia=_fmt([TINY] * 3))
                body.remove(el)
                w.append(el)
                if body.get("childclass"):
                    w.set("childclass", body.get("childclass"))
            if k == 0:
                for at in _FRAME:
                    w.attrib.pop(at, None)
                w.set("pos", _fmt(m0.body_pos[b] - shifted.get(parent, 0.0) + R.reshape(3, 3) @ a))
                w.set("quat", _fmt(m0.body_quat[b]))
            else:
                for at in _FRAME:
                    w.attrib.pop(at, None)
                if (a - anchors[k - 1]).any():
                    w.set("pos", _fmt(a - anchors[k - 1]))
            if last:
                if a.any():
                    _shift_contents(body, a)
                    shifted[body] = a
                if prev is not None:
                    prev.append(body)
            elif prev is None:
                parent.remove(body)
                parent.insert(idx, w)
            else:
                prev.append(w)
            prev = w
    return root


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _zero_refs(root, m0):
    parents = {c: p for p in root.iter() for c in p}
    for el in list(root.iter("joint")):
        if parents[el].tag != "body" or not el.get("name"):
            continue
        j = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, el.get("name"))
        t, q0 = int(m0.jnt_type[j]), float(m0.qpos0[m0.jnt_qposadr[j]])
        if t not in (J.mjJNT_HINGE, J.mjJNT_SLIDE) or q0 == 0.0:
            continue
        body = parents[el]
        if body.get("quat"):
            quat = np.array(body.get("quat").split(), float)
        elif any(body.get(a) for a in _FRAME[2:]):
            quat = np.array(m0.body_quat[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, body.get("name"))], float)
        else:
            quat = np.array([1.0, 0, 0, 0])
        axis = np.array(m0.jnt_axis[j], float)
        if t == J.mjJNT_HINGE:
            r = np.zeros(4)
            mujoco.mju_axisAngle2Quat(r, axis, -q0)
            quat = _quat_mul(quat, r)
        else:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, quat)
            pos = np.array(body.get("pos", "0 0 0").split(), float)
            body.set("pos", _fmt(pos - q0 * (R.reshape(3, 3) @ axis)))
        for a in _FRAME[1:]:
            body.attrib.pop(a, None)
        body.set("quat", _fmt(quat))
        el.set("ref", "0")
    _shift_joint_equalities(root, m0)
    return root


def _shift_joint_equalities(root, m0):
    els = [el for eq in root.iter("equality") for el in eq]
    if len(els) != m0.neq:
        return
    for e, el in enumerate(els):
        o1, o2 = int(m0.eq_obj1id[e]), int(m0.eq_obj2id[e])
        if el.tag == "joint" and int(m0.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT):
            j1, j2 = o1, o2
            r1 = float(m0.qpos0[m0.jnt_qposadr[j1]])
            r2 = float(m0.qpos0[m0.jnt_qposadr[j2]]) if j2 >= 0 else 0.0
        elif el.tag == "tendon" and int(m0.eq_type[e]) == int(mujoco.mjtEq.mjEQ_TENDON):
            j2 = o2
            r1 = float(m0.tendon_length0[o1])
            r2 = float(m0.tendon_length0[o2]) if o2 >= 0 else 0.0
        else:
            continue
        if r1 == 0.0 and r2 == 0.0:
            continue
        c = np.array(m0.eq_data[e, :5], float)
        if j2 < 0:
            c[0] += r1
        else:
            P = np.polynomial.Polynomial(c)
            c = np.zeros(5)
            sh = P(np.polynomial.Polynomial([-r2, 1.0])).coef
            c[:len(sh)] = sh
            c[0] += r1
        el.set("polycoef", _fmt(c))


def mixed(m, g1, g2):
    """MuJoCo's contact parameters of the dynamic geom pair (g1, g2): condim, friction (5), solref, solimp, margin,
    gap (scripts/robots/export_robot.py's rule; margin and gap: the sums of the geoms', as MuJoCo 3.17's
    mj_collideGeoms and MuJoCo Warp's contact_margin_gap)."""
    p1, p2 = m.geom_priority[g1], m.geom_priority[g2]
    if p1 != p2:
        g = g1 if p1 > p2 else g2
        condim, fr, solref, solimp = m.geom_condim[g], m.geom_friction[g], m.geom_solref[g], m.geom_solimp[g]
    else:
        condim, fr = max(m.geom_condim[g1], m.geom_condim[g2]), np.maximum(m.geom_friction[g1], m.geom_friction[g2])
        s1, s2 = m.geom_solmix[g1], m.geom_solmix[g2]
        mix = 0.5 if s1 + s2 < mujoco.mjMINVAL else s1 / (s1 + s2)
        r1, r2 = m.geom_solref[g1], m.geom_solref[g2]
        solref = mix * r1 + (1 - mix) * r2 if (r1[0] > 0 and r2[0] > 0) else np.minimum(r1, r2)
        solimp = mix * m.geom_solimp[g1] + (1 - mix) * m.geom_solimp[g2]
    return (int(condim), [fr[0], fr[0], fr[1], fr[2], fr[2]], list(solref), list(solimp),
            float(m.geom_margin[g1] + m.geom_margin[g2]), float(m.geom_gap[g1] + m.geom_gap[g2]))


def _keep_bounded_inertia(spec, m0):
    spec.compiler.boundmass = 0.0
    spec.compiler.boundinertia = 0.0
    mt = spec.compile()
    d0, d1 = mujoco.MjData(m0), mujoco.MjData(mt)
    d0.qpos[:] = m0.qpos0
    d1.qpos[:] = m0.qpos0
    mujoco.mj_kinematics(m0, d0)
    mujoco.mj_kinematics(mt, d1)
    for b0 in range(1, m0.nbody):
        b1 = mujoco.mj_name2id(mt, mujoco.mjtObj.mjOBJ_BODY, m0.body(b0).name)
        if b1 < 0 or (m0.body_mass[b0] == mt.body_mass[b1] and np.array_equal(m0.body_inertia[b0], mt.body_inertia[b1])):
            continue
        R1 = d1.xmat[b1].reshape(3, 3)
        q1inv, qi, iq = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(q1inv, d1.xquat[b1])
        mujoco.mju_mat2Quat(qi, d0.ximat[b0])
        mujoco.mju_mulQuat(iq, q1inv, qi)
        sb = spec.body(m0.body(b0).name)
        sb.explicitinertial = True
        sb.mass = float(m0.body_mass[b0])
        sb.inertia = [float(x) for x in m0.body_inertia[b0]]
        sb.ipos = [float(x) for x in R1.T @ (d0.xipos[b0] - d1.xpos[b1])]
        sb.iquat = [float(x) for x in iq]
        sb.fullinertia = [float("nan")] * 6


def _margin_ok(m, g1, g2, margin, gap):
    from nablaengine._optional import FOREST, NotInRelease, load
    try:
        mg = load("nablaengine.forest_engine", FOREST, "margin")
    except NotInRelease:
        return False
    return margin >= 0 and gap >= 0 and mg.supported_types(m.geom_type[g1], m.geom_type[g2])


def to_engine(xml, assets=None, keep=(), drop_types=(), cone="source", invweight="geom", condim6_to=None):
    """Converted (module docstring) of the MJCF text xml (assets: MuJoCo's assets dict for its includes / files);
    drop_types: geom types (mujoco.mjtGeom) whose contact pairs are left out (listed in notes; for pairs the caller
    knows never touch, e.g. decorations); cone: "source" (the source's friction cone where the engine covers it) or
    "pyramidal". invweight: the fused bodies' contact softness, "geom" (the default; where bodies were fused: the source's value of each geom's body,
    per geom, in the custom numeric nablaengine.model.invweight.NUMERIC that the engines read: then the contacts of the
    fused geoms and of their parents' own geoms have the source's softness; MuJoCo C / MuJoCo Warp ignore the field,
    the source model is their equivalent) or "converted" (MuJoCo's fusestatic values: the fused geoms take their parent's
    body_invweight0). Without fused bodies both give the same model. condim6_to: None (the default) keeps every pair's
    condim as the source has it, so a condim-6 pair (rolling friction, which the engines do not have) stays condim 6
    and capability.check_forest refuses the model (H17); 3 turns the condim-6 pairs into condim-3 pairs (a change to
    the dynamics: the rolling and torsional friction dropped; listed in notes, with a RuntimeWarning), an explicit
    opt-in for reproducing runs made before the forest engine checked condim (2026-11-13)."""
    if condim6_to not in (None, 3):
        raise ValueError(f"condim6_to={condim6_to!r}: None or 3")
    if cone not in ("source", "pyramidal"):
        raise ValueError(f"cone={cone!r}: 'source' or 'pyramidal'")
    if invweight not in ("converted", "geom"):
        raise ValueError(f"invweight={invweight!r}: 'converted' or 'geom'")
    from nablaengine.model.capability import filter_pairs
    assets = dict(assets or {})
    root = _name_all(_inline_includes(ET.fromstring(xml), assets))
    named = ET.tostring(root, encoding="unicode")
    m0 = mujoco.MjModel.from_xml_string(named, assets)
    dis = int(m0.opt.disableflags)
    no_contact = bool(dis & (D.mjDSBL_CONTACT | D.mjDSBL_CONSTRAINT))
    pairs = [] if no_contact else filter_pairs(m0)
    if not no_contact:
        pairs += [(int(a), int(b)) for a, b in zip(m0.pair_geom1, m0.pair_geom2)]
    params = [mixed(m0, a, b) for a, b in pairs[:len(pairs) - m0.npair]]
    params += [(int(m0.pair_dim[p]), list(m0.pair_friction[p]), list(m0.pair_solref[p]), list(m0.pair_solimp[p]),
                float(m0.pair_margin[p]), float(m0.pair_gap[p])) for p in range(m0.npair)]
    gname = [m0.geom(g).name for g in range(m0.ngeom)]
    wid = np.asarray(m0.body_weldid)
    static = lambda g: wid[m0.geom_bodyid[g]] == 0 or m0.body_mocapid[wid[m0.geom_bodyid[g]]] >= 0
    keep_s = [i for i, (a, b) in enumerate(pairs) if not (static(a) and static(b))]
    if len(keep_s) < len(pairs):
        pairs, params = [pairs[i] for i in keep_s], [params[i] for i in keep_s]
    dropped = [(a, b) for a, b in pairs if {int(m0.geom_type[a]), int(m0.geom_type[b])} & {int(x) for x in drop_types}]
    if dropped:
        keep_i = [i for i, pr in enumerate(pairs) if pr not in dropped]
        pairs, params = [pairs[i] for i in keep_i], [params[i] for i in keep_i]

    root = _zero_refs(_split_bodies(_connect_sites(copy.deepcopy(root), m0, notes0 := []), m0), m0)
    comp = root.find("compiler")
    if comp is not None and comp.get("inertiafromgeom") == "true":
        assert next(root.iter("inertial"), None) is None or all(
            b.get("name", "").count("__") for b in root.iter("body") if b.find("inertial") is not None)
        comp.set("inertiafromgeom", "auto")
    for c in root.iter("contact"):
        for p in list(c):
            if p.tag == "pair":
                c.remove(p)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"), assets=assets)
    notes = [f"contact pairs left out: {', '.join(gname[a] + '-' + gname[b] for a, b in dropped)}"] if dropped else []
    notes += notes0
    assert len(spec.actuators) == m0.nu
    for aid, a in enumerate(spec.actuators):
        if _is_general(m0, aid):
            a.lengthrange = list(map(float, m0.actuator_lengthrange[aid]))
            for prm, typ in (("gainprm", int(m0.actuator_gaintype[aid]) == int(mujoco.mjtGain.mjGAIN_MUSCLE)),
                             ("biasprm", int(m0.actuator_biastype[aid]) == int(mujoco.mjtBias.mjBIAS_MUSCLE))):
                p = np.array(getattr(a, prm), float)
                if typ and p[2] < 0:
                    p[2] = p[3] / max(1e-15, float(m0.actuator_acc0[aid]))
                    setattr(a, prm, list(p))
            continue
        g = float(a.gear[0])
        assert not np.any(a.gear[1:]), f"actuator {a.name}: gear {list(a.gear)}"
        assert a.trntype in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_TENDON) \
            and a.dyntype == mujoco.mjtDyn.mjDYN_NONE, f"actuator {a.name}: joint / tendon transmission, no dynamics"
        assert a.gaintype == mujoco.mjtGain.mjGAIN_FIXED, f"actuator {a.name}: fixed gain only"
        if g != 1.0:
            a.gainprm[0] *= g
            a.biasprm[0] *= g
            a.biasprm[1] *= g * g
            a.biasprm[2] *= g * g
            a.forcerange = np.sort(np.asarray(a.forcerange) * g)
            a.gear = [1.0, 0, 0, 0, 0, 0]
    from nablaengine.forest_engine import spatial as _sp
    for t in spec.tendons:
        tid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_TENDON, t.name)
        if tid >= 0 and _sp.is_spatial(m0, tid):
            t.springlength = list(map(float, m0.tendon_lengthspring[tid]))
    for g in spec.geoms:
        g.contype = g.conaffinity = 0
    for p in spec.pairs:
        spec.delete(p)
    for e in spec.excludes:
        spec.delete(e)
    c6 = []
    for i, ((g1, g2), (condim, fr, solref, solimp, margin, gap)) in enumerate(zip(pairs, params)):
        if int(m0.geom_type[g1]) > int(m0.geom_type[g2]):
            g1, g2 = g2, g1
        if condim == 6 and condim6_to is not None:
            c6.append(f"{gname[g1]}-{gname[g2]}")
            condim = condim6_to
        if (margin or gap) and not _margin_ok(m0, g1, g2, margin, gap):
            if margin:
                notes.append(f"pair {gname[g1]}-{gname[g2]}: margin {margin} / gap {gap} -> 0 (the forest engine has "
                             "no margin on this pair type: forest_engine.margin)")
            margin = gap = 0.0
        spec.add_pair(name=f"p{i}_{gname[g1]}__{gname[g2]}", geomname1=gname[g1], geomname2=gname[g2], condim=condim,
                      friction=fr, solref=solref, solimp=solimp, margin=margin, gap=gap)
    if c6:
        notes.append(f"condim 6 -> {condim6_to} (condim6_to={condim6_to}: rolling / torsional friction dropped) on "
                     f"{len(c6)} pairs: {', '.join(c6[:8])}{' ...' if len(c6) > 8 else ''}")
        warnings.warn(f"nablaengine.envs.convert: condim6_to={condim6_to}: {len(c6)} condim-6 contact pairs converted "
                      f"to condim {condim6_to} (rolling and torsional friction dropped: not the source's dynamics)",
                      RuntimeWarning, stacklevel=2)
    o = spec.option
    if dis & D.mjDSBL_CONSTRAINT or dis & D.mjDSBL_LIMIT:
        for j in list(spec.joints) + list(spec.tendons):
            j.limited = mujoco.mjtLimited.mjLIMITED_FALSE
    if dis & D.mjDSBL_CONSTRAINT or dis & D.mjDSBL_FRICTIONLOSS:
        for j in list(spec.joints) + list(spec.tendons):
            j.frictionloss = 0.0
    for j in spec.joints:
        if j.margin:
            jid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, j.name)
            if m0.jnt_limited[jid] and not dis & (D.mjDSBL_CONSTRAINT | D.mjDSBL_LIMIT):
                notes.append(f"joint {j.name}: limit margin {j.margin} -> 0")
            j.margin = 0.0
    if dis & D.mjDSBL_GRAVITY:
        o.gravity = [0.0, 0.0, 0.0]
    if dis & D.mjDSBL_SPRING:
        for j in list(spec.joints) + list(spec.tendons):
            j.stiffness = 0.0
    if dis & D.mjDSBL_DAMPER:
        for j in list(spec.joints) + list(spec.tendons):
            j.damping = 0.0
    if dis & D.mjDSBL_CONSTRAINT or dis & D.mjDSBL_EQUALITY:
        for e in list(spec.equalities):
            spec.delete(e)
    folded = (D.mjDSBL_CONTACT | D.mjDSBL_CONSTRAINT | D.mjDSBL_GRAVITY | D.mjDSBL_LIMIT | D.mjDSBL_FRICTIONLOSS
              | D.mjDSBL_SPRING | D.mjDSBL_DAMPER | D.mjDSBL_EQUALITY)
    o.disableflags = dis & ~int(folded)
    o.enableflags = int(o.enableflags) & ~int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    I = mujoco.mjtIntegrator
    if "integrator" not in keep and int(o.integrator) not in (int(I.mjINT_EULER), int(I.mjINT_IMPLICITFAST)):
        notes.append(f"integrator {I(int(o.integrator)).name} -> mjINT_EULER")
        o.integrator = I.mjINT_EULER
    if int(o.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC) and spec.pairs and cone == "pyramidal":
        notes.append("cone elliptic -> pyramidal (cone='pyramidal')")
        o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    mocap = [b.name for b in spec.bodies[1:] if b.mocap]
    for b in spec.bodies[1:]:
        if b.mocap:
            b.mocap = False
    if mocap:
        notes.append(f"mocap bodies frozen at their model pose (static, fused into their parents): {mocap} (the "
                     "source's dynamics while mocap_pos / mocap_quat stay at the body pose; the engine has no mocap "
                     "input)")
    if spec.compiler.boundmass > 0 or spec.compiler.boundinertia > 0:
        _keep_bounded_inertia(spec, m0)
    static = [b.name for b in spec.bodies[1:] if not b.joints and b.name]
    from nablaengine.model import gravcomp as _gc
    gmix = _gc.mixed(m0, static) if static else {}
    for r, (mem, gce) in gmix.items():
        for b in mem:
            spec.body(m0.body(b).name).gravcomp = gce
    sens = _retarget_sensors(spec, m0, set(static), notes) if static else None
    if static:
        spec.compiler.fusestatic = True
        notes.append(f"static bodies fused into their parents: {static} (their geoms and sites stay, on the parent)")
    m = spec.compile()
    if int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC) and m.npair:
        from nablaengine.g1_engine import elliptic
        ok, why = elliptic.supported(m)
        if not ok:
            notes.append(f"cone elliptic -> pyramidal ({why})")
            spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
            m = spec.compile()
    if gmix:
        field = _gc.offset_field(m0, m, gmix)
        nb = m.nbody
        spec.add_numeric(name=_gc.NUMERIC, size=field.size, data=list(map(float, field.ravel())))
        m = spec.compile()
        assert m.nbody == nb
        for r, (mem, gce) in gmix.items():
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, m0.body(r).name)
            notes.append(f"gravcomp of {m0.body(r).name} with its fused bodies "
                         f"{[m0.body(k).name for k in mem[1:]]} ({[float(m0.body_gravcomp[k]) for k in mem]}): "
                         f"{gce:.6g} at the source's point, offset {np.round(field[b], 6).tolist()} from the COM "
                         f"(custom numeric {_gc.NUMERIC}; the engine reads it, MuJoCo C / MuJoCo Warp apply the force "
                         "at the COM)")

    fused = {}
    for name in static:
        b0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, name)
        par = int(m0.body_parentid[b0])
        while par and m0.body(par).name in static:
            par = int(m0.body_parentid[par])
        pname = m0.body(par).name
        fused[name] = (pname, m0.body_invweight0[b0].copy(), m0.body_invweight0[par].copy())
        new = m.body_invweight0[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, pname)] if par else None
        if par and not np.allclose(new, m0.body_invweight0[b0], rtol=1e-9, atol=0.0):
            notes.append(f"contact softness of {name}'s geoms: body_invweight0 {m0.body_invweight0[b0, 0]:.4g} -> "
                         f"{new[0]:.4g} (now {pname}'s, whose own was {m0.body_invweight0[par, 0]:.4g}; translational)")

    ginvw = None
    if fused and invweight == "geom":
        from nablaengine.model import invweight as _iw
        filled = _iw.fill(m0, m)
        if filled is not None:
            field, ginvw = filled
            _iw.add(spec, field)
            ngeom = m.ngeom
            names_before = [m.geom(g).name for g in range(m.ngeom)]
            m = spec.compile()
            assert m.ngeom == ngeom and [m.geom(g).name for g in range(m.ngeom)] == names_before
            assert np.array_equal(_iw.geom_values(m), ginvw)
            touched = set(fused) | {v[0] for v in fused.values()}
            other = sorted({m.body(int(m.geom_bodyid[g])).name for g in np.nonzero(field >= 0)[0]
                            if m.geom_bodyid[g]} - touched)
            notes.append(f"invweight='geom': the source's body_invweight0 per geom (custom numeric {_iw.NUMERIC}) for "
                         f"the geoms of {sorted(touched)}" + (f" and of {other} (values changed by fusing)" if other else "")
                         + "; the engines read it, MuJoCo C / MuJoCo Warp do not")

    if invweight == "geom" and _connect_invweights(m0, m, spec):
        nb = m.nbody
        m = spec.compile()
        assert m.nbody == nb
        notes.append("connect equalities on fused bodies: the source's invweights (custom numeric "
                     "nablaengine_eq_invweight; the engine reads it, MuJoCo C / MuJoCo Warp do not)")

    names = lambda mm, obj, n: [mujoco.mj_id2name(mm, obj, i) for i in range(n)]
    assert names(m, mujoco.mjtObj.mjOBJ_JOINT, m.njnt) == names(m0, mujoco.mjtObj.mjOBJ_JOINT, m0.njnt)
    assert (m.nq, m.nv, m.nu) == (m0.nq, m0.nv, m0.nu)
    assert names(m, mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu) == names(m0, mujoco.mjtObj.mjOBJ_ACTUATOR, m0.nu)
    body_map = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, m0.body(b).name) if b else 0
                         for b in range(m0.nbody)])
    if m.neq:
        _check_equalities(m0, m, notes)
    if m.ntendon:
        _check_tendons(m0, m, notes)
    if any(_is_general(m0, a) for a in range(m0.nu)):
        _check_general_actuators(m0, m)
    if sens and sens[1]:
        _check_sensors(m0, m, *sens)
    return Converted(m, m0, spec.to_xml(), body_map, notes, fused, ginvw)


_FRAME_SENSORS = {int(getattr(mujoco.mjtSensor, f"mjSENS_FRAME{k}")) for k in (
    "POS", "QUAT", "XAXIS", "YAXIS", "ZAXIS", "LINVEL", "ANGVEL", "LINACC", "ANGACC")}


def _retarget_sensors(spec, m0, static, notes):
    O = mujoco.mjtObj
    bodyish = (int(O.mjOBJ_BODY), int(O.mjOBJ_XBODY))
    cut = (int(mujoco.mjtSensor.mjSENS_FORCE), int(mujoco.mjtSensor.mjSENS_TORQUE))
    kept, moved, dropped, sites = [], [], [], {}
    for i, se in enumerate(list(spec.sensors)):
        slots = [k for k, (ot, on) in enumerate(((se.objtype, se.objname), (se.reftype, se.refname)))
                 if int(ot) in bodyish and on in static]
        if int(se.type) in cut and m0.body(int(m0.site_bodyid[mujoco.mj_name2id(m0, O.mjOBJ_SITE, se.objname)])
                                           ).name in static:
            slots = [None]
        if not slots:
            kept.append(i)
            continue
        if int(se.type) not in _FRAME_SENSORS or None in slots:
            dropped.append(se.name or str(i))
            spec.delete(se)
            continue
        for k in slots:
            ot, on = (se.objtype, se.objname) if k == 0 else (se.reftype, se.refname)
            inertial = int(ot) == int(O.mjOBJ_BODY)
            key = (on, inertial)
            if key not in sites:
                b = mujoco.mj_name2id(m0, O.mjOBJ_BODY, on)
                nm = f"{on}__{'inertial' if inertial else 'frame'}"
                sites[key] = nm
                spec.body(on).add_site(name=nm, pos=list(map(float, m0.body_ipos[b])) if inertial else [0.0] * 3,
                                       quat=list(map(float, m0.body_iquat[b])) if inertial else [1.0, 0, 0, 0],
                                       group=5)
            if k == 0:
                se.objtype, se.objname = O.mjOBJ_SITE, sites[key]
            else:
                se.reftype, se.refname = O.mjOBJ_SITE, sites[key]
        kept.append(i)
        moved.append(i)
    if moved:
        notes.append(f"sensors on jointless bodies: {len(moved)} frame sensors now on sites at the same frames ("
                     f"{', '.join(sorted(sites.values()))}), so the bodies fuse")
    if dropped:
        notes.append(f"sensors on jointless bodies dropped (subtree, distance or force / torque sensors: the bodies "
                     f"fuse): {dropped}")
    return kept, moved


def _check_sensors(m0, m, kept, moved, nstate=8, seed=3):
    assert m.nsensor == len(kept), "sensors: the converted model's count differs"
    rng = np.random.default_rng(seed)
    d0, d1 = mujoco.MjData(m0), mujoco.MjData(m)
    for _ in range(nstate):
        q = m0.qpos0.copy()
        for j in range(m0.njnt):
            if int(m0.jnt_type[j]) in (J.mjJNT_HINGE, J.mjJNT_SLIDE):
                lo, hi = m0.jnt_range[j] if m0.jnt_limited[j] else (-1.0, 1.0)
                q[m0.jnt_qposadr[j]] = rng.uniform(lo, hi)
        v = rng.normal(size=m0.nv)
        for dd in (d0, d1):
            dd.qpos[:], dd.qvel[:] = q, v
            dd.ctrl[:] = 0.0
        mujoco.mj_forward(m0, d0)
        mujoco.mj_forward(m, d1)
        for i in moved:
            k = kept.index(i)
            a0, a1, n = int(m0.sensor_adr[i]), int(m.sensor_adr[k]), int(m0.sensor_dim[i])
            x0, x1 = d0.sensordata[a0:a0 + n], d1.sensordata[a1:a1 + n]
            if int(m0.sensor_type[i]) == int(mujoco.mjtSensor.mjSENS_FRAMEQUAT) and np.dot(x0, x1) < 0:
                x1 = -x1
            assert np.allclose(x0, x1, rtol=1e-7, atol=1e-9), \
                f"sensor {m0.sensor(i).name or i}: the retargeted sensor's values differ"


def _is_general(m, a):
    return (int(m.actuator_dyntype[a]) != int(mujoco.mjtDyn.mjDYN_NONE)
            or int(m.actuator_gaintype[a]) == int(mujoco.mjtGain.mjGAIN_MUSCLE)
            or int(m.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_MUSCLE))


def _check_general_actuators(m0, m, nstate=16, seed=2):
    gen = [a for a in range(m0.nu) if _is_general(m0, a)]
    for f in ("actuator_lengthrange", "actuator_dynprm", "actuator_gear", "actuator_actadr", "actuator_actnum"):
        x0, x1 = np.asarray(getattr(m0, f))[gen], np.asarray(getattr(m, f))[gen]
        assert np.array_equal(x0, x1), f"to_engine: {f} of the general actuators differs"
    rng = np.random.default_rng(seed)
    d0, d1 = mujoco.MjData(m0), mujoco.MjData(m)
    for _ in range(nstate):
        mujoco.mj_resetData(m0, d0)
        for j in range(m0.njnt):
            if int(m0.jnt_type[j]) in (int(J.mjJNT_HINGE), int(J.mjJNT_SLIDE)):
                d0.qpos[m0.jnt_qposadr[j]] += 0.2 * rng.standard_normal()
        d0.qvel[:] = 0.5 * rng.standard_normal(m0.nv)
        d0.ctrl[:] = rng.uniform(-0.2, 1.2, m0.nu)
        d0.act[:] = rng.uniform(-0.1, 1.1, m0.na)
        mujoco.mj_resetData(m, d1)
        d1.qpos[:], d1.qvel[:], d1.ctrl[:], d1.act[:] = d0.qpos, d0.qvel, d0.ctrl, d0.act
        mujoco.mj_forward(m0, d0)
        mujoco.mj_forward(m, d1)
        for nm, x0, x1 in (("actuator_length", d0.actuator_length[gen], d1.actuator_length[gen]),
                           ("actuator_velocity", d0.actuator_velocity[gen], d1.actuator_velocity[gen]),
                           ("actuator_force", d0.actuator_force[gen], d1.actuator_force[gen]),
                           ("act_dot", d0.act_dot, d1.act_dot)):
            sc = max(1.0, float(np.abs(x0).max(initial=0.0)))
            assert np.abs(x0 - x1).max(initial=0.0) <= 1e-7 * sc, f"to_engine: {nm} of the general actuators differs"


def _connect_sites(root, m0, notes):
    els = [el for eq in root.iter("equality") for el in eq]
    if len(els) != m0.neq or not any(int(t) == int(mujoco.mjtEq.mjEQ_CONNECT) for t in m0.eq_type):
        return root
    bodies = {el.get("name"): el for el in root.iter("body")}
    world = root.find("worldbody")
    refs = any(int(m0.jnt_type[j]) in (J.mjJNT_HINGE, J.mjJNT_SLIDE) and m0.qpos0[m0.jnt_qposadr[j]] != 0
               for j in range(m0.njnt))

    def moved(b):
        if b == 0:
            return False
        n, j0 = int(m0.body_jntnum[b]), int(m0.body_jntadr[b])
        return n != 1 or (int(m0.jnt_type[j0]) == J.mjJNT_HINGE and bool(np.any(m0.jnt_pos[j0])))

    for e, el in enumerate(els):
        if el.tag != "connect" or int(m0.eq_objtype[e]) != int(mujoco.mjtObj.mjOBJ_BODY):
            continue
        b1, b2 = int(m0.eq_obj1id[e]), max(int(m0.eq_obj2id[e]), 0)
        if not (refs or moved(b1) or moved(b2)):
            continue
        name = el.get("name") or f"connect{e}"
        sites = []
        for k, (b, a) in enumerate(((b1, m0.eq_data[e, 0:3]), (b2, m0.eq_data[e, 3:6]))):
            host = world if b == 0 else bodies[m0.body(b).name]
            sname = f"{name}__site{k + 1}"
            ET.SubElement(host, "site", name=sname, pos=_fmt(a))
            sites.append(sname)
        for at in ("body1", "body2", "anchor"):
            el.attrib.pop(at, None)
        el.set("site1", sites[0])
        el.set("site2", sites[1])
        notes.append(f"connect {name}: as a site connect ({sites[0]}, {sites[1]}: the anchors; the same constraint)")
    return root


def _connect_invweights(m0, m, spec):
    from nablaengine.forest_engine import equality
    if m0.neq != m.neq:
        return False
    vals, need = np.full(m.neq, -1.0), False
    for e in range(m.neq):
        if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_CONNECT):
            continue
        src = equality.connect_invweight(m0, e)
        vals[e] = src
        need |= not np.isclose(src, equality.connect_invweight(m, e), rtol=1e-9, atol=0.0)
    if need:
        spec.add_numeric(name=equality.INVW_NUMERIC, size=m.neq, data=list(map(float, vals)))
    return need


def _check_equalities(m0, m, notes, nstate=16, seed=0):
    if m0.neq != m.neq or int(m0.opt.disableflags) & int(D.mjDSBL_EQUALITY | D.mjDSBL_CONSTRAINT):
        return
    rng = np.random.default_rng(seed)
    d0, d = mujoco.MjData(m0), mujoco.MjData(m)
    for _ in range(nstate):
        q = m0.qpos0.copy()
        for j in range(m0.njnt):
            if int(m0.jnt_type[j]) in (J.mjJNT_HINGE, J.mjJNT_SLIDE):
                lo, hi = m0.jnt_range[j] if m0.jnt_limited[j] else (-1.0, 1.0)
                q[m0.jnt_qposadr[j]] = rng.uniform(lo, hi)
        d0.qpos[:] = q
        d.qpos[:] = q
        mujoco.mj_forward(m0, d0)
        mujoco.mj_forward(m, d)
        e0 = d0.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        e1 = d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        assert e0.sum() == e1.sum(), "equality rows: the converted model's count differs"
        assert np.allclose(d0.efc_pos[e0], d.efc_pos[e1], rtol=1e-9, atol=1e-9), \
            f"equality rows: efc_pos {d0.efc_pos[e0]} -> {d.efc_pos[e1]}"
    from nablaengine.forest_engine import equality
    for e in range(m.neq):
        if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_CONNECT):
            a, b = equality.connect_invweight(m0, e), equality.connect_invweight(m, e)
            if not np.isclose(a, b, rtol=1e-6, atol=0.0):
                notes.append(f"equality {e}: connect invweight (body_invweight0 sum) {a:.6g} -> {b:.6g}")
            continue
        if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        dofs = [int(m.jnt_dofadr[j]) for j in (m.eq_obj1id[e], m.eq_obj2id[e]) if j >= 0]
        a, b = m0.dof_invweight0[dofs].sum(), m.dof_invweight0[dofs].sum()
        if not np.isclose(a, b, rtol=1e-6, atol=0.0):
            notes.append(f"equality {e}: invweight (dof_invweight0 sum) {a:.6g} -> {b:.6g}")


def _efc_J_dense(m, d):
    if mujoco.mj_isSparse(m):
        out = np.zeros((d.nefc, m.nv))
        if d.nefc:
            mujoco.mju_sparse2dense(out, d.efc_J, d.efc_J_rownnz[:d.nefc], d.efc_J_rowadr[:d.nefc], d.efc_J_colind)
        return out
    return np.asarray(d.efc_J[:d.nefc * m.nv]).reshape(d.nefc, m.nv)


def _ten_J_dense(m, d):
    J = np.zeros((m.ntendon, m.nv))
    for t in range(m.ntendon):
        for k in range(int(m.ten_J_rownnz[t])):
            J[t, int(m.ten_J_colind[m.ten_J_rowadr[t] + k])] = d.ten_J[m.ten_J_rowadr[t] + k]
    return J


def _check_tendons(m0, m, notes, nstate=16, seed=1):
    if m0.ntendon != m.ntendon:
        return
    fixed = list(range(m.ntendon))
    if not fixed:
        return
    rng = np.random.default_rng(seed)
    pairs = []
    for mm in (m0, m):
        mc = copy.copy(mm)
        mc.jnt_stiffness[:] = 0.0
        mc.dof_damping[:] = 0.0
        mc.body_gravcomp[:] = 0.0
        mc.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        pairs.append((mc, mujoco.MjData(mc)))
    (a0, d0), (a1, d1) = pairs
    lim = mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON
    for _ in range(nstate):
        q = m0.qpos0.copy()
        for j in range(m0.njnt):
            if int(m0.jnt_type[j]) in (J.mjJNT_HINGE, J.mjJNT_SLIDE):
                lo, hi = m0.jnt_range[j] if m0.jnt_limited[j] else (-1.0, 1.0)
                q[m0.jnt_qposadr[j]] = rng.uniform(lo, hi)
        v = rng.normal(size=m0.nv)
        for dd in (d0, d1):
            dd.qpos[:] = q
            dd.qvel[:] = v
        mujoco.mj_forward(a0, d0)
        mujoco.mj_forward(a1, d1)
        assert np.allclose(d0.ten_length[fixed], d1.ten_length[fixed], rtol=1e-9, atol=1e-9), "tendon lengths differ"
        assert np.allclose(_ten_J_dense(a0, d0), _ten_J_dense(a1, d1), rtol=1e-9, atol=1e-10), "tendon Jacobians differ"
        e0, e1 = d0.efc_type == lim, d1.efc_type == lim
        assert e0.sum() == e1.sum(), "tendon limit rows: the converted model's count differs"
        assert np.allclose(d0.efc_pos[e0], d1.efc_pos[e1], rtol=1e-9, atol=1e-9), "tendon limit rows: efc_pos differs"
        if e0.any():
            J0, J1 = _efc_J_dense(a0, d0)[e0], _efc_J_dense(a1, d1)[e1]
            assert np.allclose(J0, J1, rtol=1e-9, atol=1e-12), "tendon limit rows: efc_J differs"
        assert np.allclose(d0.qfrc_passive, d1.qfrc_passive, rtol=1e-7, atol=1e-9), "tendon passive forces differ"
    for t in fixed:
        a, b = float(m0.tendon_invweight0[t]), float(m.tendon_invweight0[t])
        if not np.isclose(a, b, rtol=1e-6, atol=0.0):
            notes.append(f"tendon {t}: tendon_invweight0 {a:.6g} -> {b:.6g}")

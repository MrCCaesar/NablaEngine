"""Per-world physics parameters (nablaengine.params.PhysicsParams) in the forest engine's text: apply() edits the engine
(adjoint.cuda_source), the host kernels (host.KERNEL) and the kinematics (kinematics.KIN_DEVICE) of a build with
parameters; a build without them is unchanged. The device code and most anchors are the G1 engine's
(g1_engine.world_params); this module holds the forest engine's differences:"""
from nablaengine.g1_engine import world_params as G

FOREST_FIELDS = ("foot_solref", "foot_solimp", "dof_damping", "dof_armature", "dof_frictionloss", "frictionloss_solref",
                 "frictionloss_solimp", "torso_mass", "torso_inertia", "foot_torsional", "object_size", "gravity",
                 "xfrc", "body_inertial", "sole_thickness", "foot_offset")

ADJ_GCTRL = ("  for (int i = lane; i < NU; i += 32) gctrl[(size_t)w * NU + i] = s.ubar[i];\n",
             "  for (int i = lane; i < NU; i += 32) gctrl[(size_t)w * (NU + NPARAM + 1) + i] = s.ubar[i];  // (world_params)\n"
             "  __syncwarp();\n"
             "  for (int i = lane; i <= NPARAM; i += 32) gctrl[(size_t)w * (NU + NPARAM + 1) + NU + i] = s.gpw[i];\n", 1)
ADJ_COMPOSE = (G.ADJ[1][0], G.ADJ[1][1].replace("- h * y * aadv;", "- (PW_DAMP_Q ? h * y * aadv : 0.f);"), 1)
assert ADJ_COMPOSE[1] != G.ADJ[1][1]
ARMATURE = (
    G.ARMATURE[0],
    ("        s.M[TRI(r0 + j, r0 + i)] = dot3(kw, n) + dot3(kv, f) + (i == j ? c_armature[r0 + j] : 0.f);",
     "        s.M[TRI(r0 + j, r0 + i)] = dot3(kw, n) + dot3(kv, f) + (i == j ? s.pw[(PW_ARM >= 0 ? PW_ARM : 0) + r0 + j] : 0.f);"
     "  // (world_params)", 1),
    G.ARMATURE[2],
)
PARTIALS_CALL = (
    "  cot[1] = cot[2] = cot[3] = 0.f;\n  contact_pbar_motion(Wv, Wa, Wl, Ov, Oa, Ol, cot + 1);\n",
    "  if (PFOOT(pair) && zone) {  // the foot pair's parameters (world_params)\n"
    "#if HAS_CONDIM4\n"
    "    contact_param_vjp4(Vv, Va, Vl, fr, mu, dim, zone, d, D, k, b, imp, contact_invw(pair, mu), sref, simp, pp, Ov, Oa,"
    " Ol, dim == 4 ? PMUS(pair) : 0.f);\n"
    "#else\n"
    "    contact_param_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, k, b, imp, contact_invw(pair, mu), sref, simp, pp);\n"
    "#endif\n"
    "  }\n"
    "  cot[1] = cot[2] = cot[3] = 0.f;\n  contact_pbar_motion(Wv, Wa, Wl, Ov, Oa, Ol, cot + 1);\n", 1)
PARTIALS = (
    (G.PARTIALS[0][0], G.PARTIALS[0][1].replace(
        G.PARTIALS_DEVICE, "#if HAS_CONDIM4\n" + G.PARTIALS4_DEVICE + "#else\n" + G.PARTIALS_DEVICE + "#endif\n"), 1),
    PARTIALS_CALL,
    (G.PARTIALS[2][0], G.PARTIALS[2][1].replace("float pp[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};",
                                                "float pp[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};"), 1),
    G.PARTIALS[3],
    G.PARTIALS[4],
)
_CC_HEAD = ("float* dbg CC_REC_PARAM) {", "float* dbg) {", "float* dbg, float* pp) {", "float* dbg, float* pp CC_REC_PARAM) {")
_CC_CALL = ("cw, dbg CC_REC_ARG)", "cw, dbg)", "cw, dbg, pp)", "cw, dbg, pp CC_REC_ARG)")


def _coop_safe(src, fn):
    if _CC_HEAD[0] not in src:
        return fn(src)
    for a, b, _, _ in (_CC_HEAD, _CC_CALL):
        if src.count(a) != 1 or src.count(b) != 0:
            raise RuntimeError(f"world_params (forest): ccd_coop's {a!r} found {src.count(a)} times")
        src = src.replace(a, b)
    src = fn(src)
    for _, _, c, d in (_CC_HEAD, _CC_CALL):
        if src.count(c) != 1:
            raise RuntimeError(f"world_params (forest): {c!r} found {src.count(c)} times after the edits")
        src = src.replace(c, d)
    return src


MUS_FOOT = "(PFOOT({p}) ? s.musw : PMUS({p}))"
TORS_STATE = (
    ("  float com[3], fric;\n", "  float com[3], fric;\n  float musw;  // the foot pairs' spin friction (world_params)\n",
     3),
    ("  const float *qsm, *qacc0, *con_mu, *efc_aref;  // in Hand\n",
     "  const float *qsm, *qacc0, *con_mu, *efc_aref;  // in Hand\n  float musw;  // (world_params: foot_torsional)\n",
     1),
    ("struct Hand {\n  float M[NTRI], qsm[NV], qacc0[NV];\n",
     "struct Hand {\n  float M[NTRI], qsm[NV], qacc0[NV];\n  float musw;  // (world_params: foot_torsional)\n", 1),
)
TORS_LOAD = "  if (lane == 0) s.dampw = s.pw + (PW_DAMP >= 0 ? PW_DAMP : 0);\n"
TORS_LOAD_NEW = (TORS_LOAD + "  if (lane == 0) s.musw = fmaxf(com[(size_t)w * COM_STRIDE + PW_OFF + (PW_TORS >= 0 ? PW_TORS : 0)],"
                 " 1e-5f);  // (foot_torsional)\n")
assert G.LOAD[1].count(TORS_LOAD) == 1
TORS_SPLIT = (
    (G.SPLIT[0][1], G.SPLIT[0][1] + "  if (lane == 0) g.musw = s.musw;  // (world_params: foot_torsional)\n", 1),
    ("s.qsm = g.qsm; s.dampw = g.dampw;", "s.qsm = g.qsm; s.dampw = g.dampw; s.musw = g.musw;", 1),
)
TORS_SUM = (G.PARTIALS[4][1], G.PARTIALS[4][1].replace("for (int q = 0; q < 7; q++)", "for (int q = 0; q < 8; q++)")
            .replace("        else if (q == 6) s.gpw[NPARAM] += v;\n",
                     "        else if (q == 6) s.gpw[NPARAM] += v;\n"
                     "        else if (q == 7) s.gpw[PW_TORS >= 0 ? PW_TORS : 0] += v;  // (foot_torsional)\n"), 1)
assert TORS_SUM[1] != G.PARTIALS[4][1] and "q < 8" in TORS_SUM[1]
TORSO_FORCE = ("  float m = c_mass[b], gc = c_gcomp[b];",
               "  float m = ipos[3], gc = c_gcomp[b];  // (world_params)", 1)
TORSO_VJP_GC = ("  for (int k = 0; k < 3; k++) f1[k] = A[k] + t[k] + wwc[k];\n",
                "  for (int k = 0; k < 3; k++) f1[k] = A[k] + t[k] + wwc[k] + c_gcomp[b] * c_grav[k];  // (gravcomp)\n")
assert G.TORSO_VJP.count(TORSO_VJP_GC[0]) == 1
TORSO_ENGINE = tuple(
    TORSO_FORCE if h[0].startswith("  float m = c_mass[b];\n  for (int k = 0; k < 3; k++) F[k]")
    else (h[0], h[1].replace(G.TORSO_VJP, G.TORSO_VJP.replace(*TORSO_VJP_GC)), h[2]) if h[1].startswith(G.TORSO_VJP)
    else h for h in G.TORSO_ENGINE)
assert TORSO_FORCE in TORSO_ENGINE and sum(TORSO_VJP_GC[1] in h[1] for h in TORSO_ENGINE) == 1
KIN_LOAD = ("  if (lane < 3) s.com[lane] = com[(size_t)w * COM_STRIDE + lane];  // (fixtures: not in the kinematics "
            "outputs)\n")


def check(m, params, foot_pairs, com_body):
    """ValueError where params do not apply to forest model m (foot_pairs: the build's, com_body: init's)."""
    import mujoco
    if not params:
        return
    bad = [f for f in params.fields if f not in FOREST_FIELDS]
    if bad:
        raise ValueError(f"PhysicsParams {bad}: not on the forest engine (it has {FOREST_FIELDS})")
    if int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        raise ValueError("per-world physics parameters with the elliptic friction cone: not supported yet")
    if "foot_torsional" in params.fields:
        if not list(foot_pairs):
            raise ValueError("PhysicsParams foot_torsional on the forest engine: init(foot_pairs=) must name the "
                             "condim-4 contact pairs it applies to")
        if not any(int(m.pair_dim[p]) == 4 for p in foot_pairs):
            raise ValueError("PhysicsParams foot_torsional: the foot pairs have no condim-4 pair")
    if "object_size" in params.fields:
        from nablaengine.forest_engine import object_size
        object_size.check(m, params, com_body)
    from nablaengine.forest_engine import world_forces, body_params, floss
    world_forces.check(m, params)
    floss.check(m, params)
    body_params.check(m, params, foot_pairs, com_body)
    if ("foot_solref" in params.fields or "foot_solimp" in params.fields) and not list(foot_pairs):
        raise ValueError("PhysicsParams foot_solref / foot_solimp on the forest engine: init(foot_pairs=) must name the "
                         "contact pairs they apply to")
    if "torso_mass" in params.fields or "torso_inertia" in params.fields:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, com_body) if com_body else -1
        if b < 1:
            raise ValueError("PhysicsParams torso_mass / torso_inertia on the forest engine: init(com_body=) must name "
                             f"the (moving) body they apply to (got {com_body!r})")


def damped(m, params):
    """Whether an Euler build with params solves the damped update although the model has no damping (model_hooks'
    "explicit" text replaced by "euler"): dof_damping with eulerdamp enabled."""
    from nablaengine.forest_engine import model_hooks
    import mujoco
    return (bool(params) and "dof_damping" in params.fields
            and int(m.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_EULER)
            and not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
            and model_hooks.integrator(m) == "explicit")


def _kin(src):
    if src.count(KIN_LOAD) != 1:
        raise RuntimeError("world_params (forest): the kinematics' com load not found")
    return src.replace(KIN_LOAD, G.TORSO_LOAD[1].replace(G.TORSO_LOAD[0], KIN_LOAD))


def apply(src, m, params, part, pw_off):
    """apply_params(src, m, params, part, pw_off), also on a build with motor limits (g1_engine.world_params's
    plain_compose: motor_hooks adds the record to adj_compose's signature, which ADJ_COMPOSE anchors on)."""
    if not params:
        return src
    return G.plain_compose(src, lambda s: apply_params(s, m, params, part, pw_off))


def apply_params(src, m, params, part, pw_off):
    """src with the edits of a forest build with parameters params (unchanged without); part: "engine"
    (adjoint.cuda_source, last), "host" (host.KERNEL) or "kinematics" (model_hooks' kinematics text). pw_off: the
    parameters' offset in a com row (step.com_words(m); a terrain build: forest_engine.terrain.com_words(m, terrain))."""
    if not params:
        return src
    from nablaengine.forest_engine import model_hooks
    from nablaengine.forest_engine import body_params
    for a, b in G.STRIDE:
        src = src.replace(a, b)
    inertial = body_params.inertial_active(params)
    torso = "torso_mass" in params.fields or "torso_inertia" in params.fields or inertial
    if part == "kinematics":
        if not torso:
            return src
        src = src.replace("float ip[3]", "float ip[7]")
        src = G._apply(src, G.TORSO_KIN)
        for a in ("    Mb = c_mass[b];\n", "    const float m = c_mass[b];\n"):
            if src.count(a) != 1:
                raise RuntimeError(f"world_params (forest): kinematics mass read {a.strip()!r} not found once")
            src = src.replace(a, a.replace("c_mass[b]", "ip[3]"))
        if "c_mass[" in src:
            raise RuntimeError("world_params (forest): the kinematics read c_mass outside the torso edits")
        return body_params.inertial(_kin(src), part) if inertial else _kin(src)
    tors = "foot_torsional" in params.fields
    n = 1 if part == "engine" else 3
    if src.count(G.LOAD[0]) != n:
        raise RuntimeError(f"world_params (forest): com loads found {src.count(G.LOAD[0])} times, expected {n}")
    src = src.replace(*(G.LOAD_ADJ if part == "engine" else G.LOAD))
    if tors:
        src = G._apply(src, ((TORS_LOAD, TORS_LOAD_NEW, n),))
    from nablaengine.forest_engine import floss
    if part == "host":
        src = G._apply(src, G.SPLIT + (TORS_SPLIT if tors else ()))
        src = src.replace(*G.TORSO_LOAD) if torso else src
        src = body_params.inertial(src, part) if inertial else src
        src = floss.apply(src, m, params, part)
        return body_params.iw_host(src) if body_params.iw_active(params) else src
    integ = "euler" if damped(m, params) else model_hooks.integrator(m)
    explicit = integ == "explicit"
    lay = params.layout(m.nv)
    src = (G.header(m, params, pw_off)
           + f"#define PW_DAMP_Q {0 if explicit else 1}  // the damping in Q (world_params)\n"
           + (f"#define PW_OSIZE {lay['object_size'][0]}  // the object's size (forest_engine.object_size)\n"
              if "object_size" in lay else "")
           + _coop_safe(src, lambda t: G._apply(t, G.STATE + (G.ADJ[0], ADJ_COMPOSE, ADJ_GCTRL) + G.FOOT + PARTIALS)))
    if tors:
        src = G._apply(src, TORS_STATE + (TORS_SUM,))
        for a in ("s.con_pair[a]", "s.con_pair[c]"):
            if f"PMUS({a})" not in src:
                raise RuntimeError(f"world_params (forest): PMUS({a}) not found (foot_torsional)")
            src = src.replace(f"PMUS({a})", MUS_FOOT.format(p=a))
        old = "dim == 4 ? PMUS(pair) : 0.f"
        if src.count(old) != 2:
            raise RuntimeError(f"world_params (forest): {old!r} found {src.count(old)} times, expected 2")
        src = src.replace(old, "dim == 4 ? " + MUS_FOOT.format(p="pair") + " : 0.f")
    if "foot_solref" in params.fields:
        src = G._apply(src, G.DIRECT_SOLREF)
    if "dof_damping" in params.fields:
        nd, want = src.count("c_damp[i]"), {"explicit": 2, "euler": 4, "implicitfast": 5}[integ]
        if nd != want:
            raise RuntimeError(f"world_params (forest): c_damp[i] found {nd} times, expected {want}")
        src = src.replace("c_damp[i]", "s.dampw[i]")
    if "dof_armature" in params.fields:
        src = G._apply(src, ARMATURE)
    src = floss.apply(src, m, params, part)
    if torso:
        src = src.replace("float ip[3]", "float ip[7]")
        src = G._apply(src, TORSO_ENGINE)
        src = src.replace(*G.TORSO_LOAD)
        if inertial:
            src = body_params.inertial(src, part)
    return src

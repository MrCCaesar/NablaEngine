"""Per-world body inertials, sole thickness and per-world constraint inverse weights (nablaengine.params: the fields
body_inertial and sole_thickness, PhysicsParams(invweight0="per_world")) in the forest engine's text. A build without
them is unchanged (every function here returns its text as is)."""
import numpy as np

INERTIAL, SOLE = "body_inertial", "sole_thickness"


def inertial_active(params):
    return bool(params) and INERTIAL in params.fields


def sole_active(params):
    return bool(params) and SOLE in params.fields


def iw_active(params):
    return bool(params) and getattr(params, "invweight0", "model") in ("per_world", "per_world_mass")


def iw_mass(params):
    """invweight0="per_world_mass" (module docstring)."""
    return bool(params) and getattr(params, "invweight0", "model") == "per_world_mass"


def iw_mass_targets(m, params, com_body):
    """[(body, mass word)] of invweight0="per_world_mass": the targets of iw_targets (the torso fields' body: com_body)
    with a mass word; ValueError where a target is not a free-floating single body (module docstring)."""
    import mujoco
    cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, com_body) if isinstance(com_body, str) else (
        -1 if com_body is None else int(com_body))
    out = []
    for b, om, _, _ in iw_targets(m, params, None):
        b = cb if b == -1 else b
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) if b >= 0 else None
        ja = int(m.body_jntadr[b]) if b >= 1 else -1
        ok = (b >= 1 and int(m.body_parentid[b]) == 0 and int(m.body_jntnum[b]) == 1
              and int(m.jnt_type[ja]) == mujoco.mjtJoint.mjJNT_FREE
              and not any(int(m.body_parentid[k]) == b for k in range(1, m.nbody))
              and not np.any(m.dof_armature[m.jnt_dofadr[ja]:m.jnt_dofadr[ja] + 6]))
        if not ok:
            raise ValueError(f"PhysicsParams invweight0='per_world_mass': body {name!r} (a parameter target) is not a "
                             "free-floating single body (free joint, parent the world, no child bodies, zero armature); "
                             "use invweight0='per_world'")
        if om >= 0:
            out.append((b, om))
    if not out:
        raise ValueError("PhysicsParams invweight0='per_world_mass': no per-world mass (torso_mass or body_inertial)")
    return out


def active(params):
    return inertial_active(params) or sole_active(params) or iw_active(params)


def inertial_ids(m, params):
    """The model ids of PhysicsParams.inertial_bodies (ValueError where one is not a moving body)."""
    import mujoco
    out = []
    for b in params.inertial_bodies:
        bid = b if isinstance(b, int) else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if not 1 <= bid < m.nbody:
            raise ValueError(f"PhysicsParams inertial_bodies: {b!r} is not a moving body of the model")
        out.append(int(bid))
    if len(set(out)) != len(out):
        raise ValueError(f"PhysicsParams inertial_bodies {params.inertial_bodies}: a body more than once")
    return out


def sole_ids(m, params, foot_pairs):
    """The sole geoms' model ids: PhysicsParams.sole_geoms, else the geoms on moving bodies of the foot pairs (ValueError
    where one is not a collision geom on a moving body, or there is none)."""
    import mujoco
    from nablaengine.forest_engine import step
    cg = set(step.collision_geoms(m))
    if params.sole_geoms:
        out = []
        for g in params.sole_geoms:
            gid = g if isinstance(g, int) else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if not 0 <= gid < m.ngeom:
                raise ValueError(f"PhysicsParams sole_geoms: no geom {g!r}")
            out.append(int(gid))
    else:
        out = sorted({int(g) for p in foot_pairs for g in (m.pair_geom1[p], m.pair_geom2[p])
                      if int(m.geom_bodyid[g]) != 0})
        if not out:
            raise ValueError("PhysicsParams sole_thickness: no sole geoms (name them with PhysicsParams(sole_geoms=...) "
                             "or pass the foot pairs as init(foot_pairs=...))")
    for g in out:
        if g not in cg:
            raise ValueError(f"PhysicsParams sole_geoms: geom {g} is in no contact pair")
        if int(m.geom_bodyid[g]) == 0:
            raise ValueError(f"PhysicsParams sole_geoms: geom {g} is a world geom")
    if len(set(out)) != len(out):
        raise ValueError(f"PhysicsParams sole_geoms {params.sole_geoms}: a geom more than once")
    return out


def check(m, params, foot_pairs, com_body, terrain=None, surrogate=False):
    """ValueError where these options do not apply to forest model m (surrogate: a contact_surrogate build)."""
    import mujoco
    if not active(params):
        return
    if inertial_active(params):
        ids = inertial_ids(m, params)
        torso = "torso_mass" in params.fields or "torso_inertia" in params.fields
        cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, com_body) if com_body else -1
        if torso and cb in ids:
            raise ValueError(f"PhysicsParams body_inertial: body {com_body!r} is also the torso fields' body (init's "
                             "com_body with torso_mass / torso_inertia); give it in one of them")
    if sole_active(params):
        sole_ids(m, params, foot_pairs)
        if terrain is not None:
            raise ValueError("PhysicsParams sole_thickness: not in a terrain build yet")
        from nablaengine.forest_engine import midphase
        if midphase.enabled(m):
            raise ValueError("PhysicsParams sole_thickness: not with the collision mid-phase (its bounds are the model's "
                             "geom positions)")
    if iw_active(params) and surrogate:
        raise ValueError("PhysicsParams invweight0='per_world' with contact_surrogate: not supported (the invweight0 "
                         "derivative takes the exact rows' active set, the surrogates relax it)")
    if iw_mass(params):
        if "dof_armature" in params.fields:
            raise ValueError("PhysicsParams invweight0='per_world_mass': not with dof_armature (use 'per_world')")
        iw_mass_targets(m, params, com_body)
    if iw_active(params):
        from nablaengine.forest_engine import equality, tendon
        from nablaengine.model import invweight
        if m.nbody > 64:
            raise ValueError("PhysicsParams invweight0='per_world': at most 64 bodies (its subtree sets are 64-bit "
                             "masks)")
        if m.neq:
            raise ValueError("PhysicsParams invweight0='per_world': not with equality constraints (their inverse "
                             "weights stay the model's)")
        if np.any(m.dof_frictionloss > 0):
            raise ValueError("PhysicsParams invweight0='per_world': not with frictionloss (its rows' invweight0 path "
                             "has no gradient yet)")
        if any(m.tendon_limited):
            raise ValueError("PhysicsParams invweight0='per_world': not with limited tendons (tendon_invweight0)")
        if invweight.table(m, step_geoms(m)) is not None:
            raise ValueError("PhysicsParams invweight0='per_world': not with per-geom inverse weights "
                             "(nablaengine.model.invweight)")
        del equality, tendon


def step_geoms(m):
    from nablaengine.forest_engine import step
    return step.collision_geoms(m)


def resolve(m, params, foot_pairs):
    """params with sole_geoms as the sole geoms' model ids (sole_ids()); params itself without sole_thickness."""
    if not sole_active(params):
        return params
    import dataclasses
    return dataclasses.replace(params, sole_geoms=tuple(sole_ids(m, params, foot_pairs)))


def defines(m, params, foot_pairs=()):
    """#defines of a build with these options (prepended to the engine's text; empty without)."""
    if not active(params):
        return ""
    lay = params.layout(m.nv)
    out = ""
    if inertial_active(params):
        ids = inertial_ids(m, params)
        slot = "".join(f"(b) == {b} ? {k} : " for k, b in enumerate(ids)) + "-1"
        out += (f"#define NIB {len(ids)}  // the per-world inertial bodies (forest_engine.body_params)\n"
                f"#define PW_INERTIAL {lay[INERTIAL][0]}  // their words in the world's parameters\n"
                f"#define CIPW (7 + 7 * NIB)  // the state's com array: COM offset, torso, the bodies' inertials\n"
                f"#define IB_SLOT(b) ({slot})  // body -> its inertial slot, -1 none\n")
        if "torso_mass" not in lay and "torso_inertia" not in lay:
            out += "#define PW_TMASS -1\n#define PW_TINERT -1\n"
    if sole_active(params):
        from nablaengine.forest_engine import step
        cg = step.collision_geoms(m)
        ids = [cg.index(g) for g in sole_ids(m, params, foot_pairs)]
        slot = "".join(f"(g) == {g} ? {k} : " for k, g in enumerate(ids)) + "-1"
        n = ", ".join(f"{x!r}f" for x in [float(np.float32(v)) for v in params.sole_normal] * len(ids))
        out += (f"#define PW_SOLE {lay[SOLE][0]}  // the sole thickness (forest_engine.body_params)\n"
                f"#define SOLE_SLOT(g) ({slot})  // engine geom -> its sole slot, -1 none\n"
                f"__device__ const float c_sole_n[{3 * len(ids)}] = {{{n}}};  // the soles' normals (body frame)\n")
    return out


BODY_IPOS_TORSO = r"""__device__ __forceinline__ void body_ipos(int b, const float* com, float* out) {  // (world_params: out[7], com[7])
  for (int k = 0; k < 3; k++) out[k] = c_ipos[3 * b + k] + (b == (int)c_misc[1] ? com[k] : 0.f);
  out[3] = b == (int)c_misc[1] ? com[3] : c_mass[b];  // mass
  for (int k = 0; k < 3; k++) out[4 + k] = b == (int)c_misc[1] ? com[4 + k] : c_inert[3 * b + k];  // principal moments
}
"""
BODY_IPOS_INERTIAL = r"""__device__ __forceinline__ void body_ipos(int b, const float* com, float* out) {  // (world_params: out[7], com[CIPW])
  for (int k = 0; k < 3; k++) out[k] = c_ipos[3 * b + k] + (b == (int)c_misc[1] ? com[k] : 0.f);
  out[3] = b == (int)c_misc[1] ? com[3] : c_mass[b];  // mass
  for (int k = 0; k < 3; k++) out[4 + k] = b == (int)c_misc[1] ? com[4 + k] : c_inert[3 * b + k];  // principal moments
  const int j = IB_SLOT(b);  // (body_params: a per-world inertial body: mass, moments, COM offset)
  if (j >= 0) {
    const float* x = com + 7 + 7 * j;
    out[3] = x[0];
    for (int k = 0; k < 3; k++) { out[4 + k] = x[1 + k]; out[k] = out[k] + x[4 + k]; }
  }
}
"""
TORSO_PARAM_GUARD = ("  const int b = (int)c_misc[1];\n  if (k == 0) return PW_TMASS",
                     "  const int b = (int)c_misc[1];\n  if (b < 1) return 0.f;  // (body_params: no torso body)\n"
                     "  if (k == 0) return PW_TMASS")
TORSO_LOAD_LINE = ("  if (lane < 4) s.com[3 + lane] = torso_param(com + (size_t)w * COM_STRIDE + PW_OFF, lane);  "
                   "// (world_params)\n")
INERTIAL_LOAD_LINE = ("  for (int i = lane; i < 7 * NIB; i += 32) s.com[7 + i] = com[(size_t)w * COM_STRIDE + PW_OFF + "
                      "PW_INERTIAL + i];  // (body_params)\n")
COM_VJP = r"""// body_params: d psi_b / d ipos of a body's COM offset (body frame): body_local_vjp's c bar (the cotangent of
// c = R_b ipos) rotated into the body frame, psi_b = F1.Vl + N1.Wl + F2.Vy + N2.Wy as there
__device__ void inertial_com_vjp(int b, const float* Q, const float* ipos, const float* W, const float* AL,
                                 const float* A, const float* AL2, const float* A2, const float* Wl, const float* Vl,
                                 const float* Wy, const float* Vy, float* dc) {
  float c[3], t[3], u[3], wc[3], wwc[3], F1[3], F2[3], g[3], gw[3], cb[3];
  const float m = ipos[3];
  qrot(Q, ipos, c);
  cross(W, c, wc);
  cross(W, wc, wwc);
  cross(AL, c, t);
  for (int k = 0; k < 3; k++) F1[k] = m * (A[k] + t[k] + wwc[k] + c_gcomp[b] * c_grav[k]);
  cross(AL2, c, t);
  for (int k = 0; k < 3; k++) F2[k] = m * (A2[k] + t[k]);
  cross(Wl, c, t);
  for (int k = 0; k < 3; k++) g[k] = m * (Vl[k] + t[k]);
  cross(g, W, gw);
  cross(F1, Wl, t);
  cross(g, AL, u);
  for (int k = 0; k < 3; k++) cb[k] = t[k] + u[k];
  cross(gw, W, t);
  for (int k = 0; k < 3; k++) cb[k] += t[k];
  cross(Wy, c, t);
  for (int k = 0; k < 3; k++) g[k] = m * (Vy[k] + t[k]);
  cross(F2, Wy, t);
  cross(g, AL2, u);
  for (int k = 0; k < 3; k++) cb[k] += t[k] + u[k];
  qrot_inv(Q, cb, dc);
}

"""
TORSO_GRAD_TAIL = ("      if (PW_TINERT >= 0) for (int k = 0; k < 3; k++) s.gpw[(PW_TINERT >= 0 ? PW_TINERT : 0) + k] -= dI[k];\n"
                   "    }\n")
INERTIAL_GRAD = r"""    if (IB_SLOT(b) >= 0) {  // a per-world inertial body's values: -d psi (body_params)
      float dm, dI[3], dc[3];
      torso_param_vjp(b, s.Q[b], ip, s.W[b], s.AL[b], s.A[b], s.AL2[b], s.A2[b], s.Wl[b], s.Vl[b], s.Wy[b], s.Vy[b],
                      &dm, dI);
      inertial_com_vjp(b, s.Q[b], ip, s.W[b], s.AL[b], s.A[b], s.AL2[b], s.A2[b], s.Wl[b], s.Vl[b], s.Wy[b], s.Vy[b],
                       dc);
      float* gi = s.gpw + PW_INERTIAL + 7 * IB_SLOT(b);
      gi[0] -= dm;
      for (int k = 0; k < 3; k++) { gi[1 + k] -= dI[k]; gi[4 + k] -= dc[k]; }
    }
"""


_W2_SYNCS = ("  if (DK_PRE2W) __syncthreads(); else __syncwarp();\n",
             "  if (DK_ADJ_WARPS == 2) __syncthreads(); else __syncwarp();\n")


def _sub(src, a, b, n=1, what=""):
    if src.count(a) == 0 and a.startswith("  __syncwarp();\n") and b.count("  __syncwarp();\n") == 1:
        for w in _W2_SYNCS:
            a2, b2 = w + a[len("  __syncwarp();\n"):], b.replace("  __syncwarp();\n", w)
            if src.count(a2) == n:
                return src.replace(a2, b2)
    if src.count(a) != n:
        raise RuntimeError(f"body_params: {what or a[:70]!r} found {src.count(a)} times, expected {n}")
    return src.replace(a, b)


def inertial(src, part):
    """The body_inertial edits of the engine / host / kinematics text after world_params' torso edits."""
    if part == "host":
        return _sub(src, TORSO_LOAD_LINE, TORSO_LOAD_LINE + INERTIAL_LOAD_LINE, 3, "the host's torso loads")
    if part == "kinematics":
        src = _sub(src, "float qpos[NQ], qvel[NV], com[7];", "float qpos[NQ], qvel[NV], com[CIPW];", 2)
        return _sub(src, TORSO_LOAD_LINE, TORSO_LOAD_LINE + INERTIAL_LOAD_LINE, 1, "the kinematics' torso loads")
    src = _sub(src, "  float com[7], fric;", "  float com[CIPW], fric;", 3)
    src = _sub(src, BODY_IPOS_TORSO, BODY_IPOS_INERTIAL)
    src = _sub(src, *TORSO_PARAM_GUARD)
    src = _sub(src, TORSO_LOAD_LINE, TORSO_LOAD_LINE + INERTIAL_LOAD_LINE, 1, "the adjoint's torso load")
    src = _sub(src, "__device__ void body_local_vjp(", COM_VJP + "__device__ void body_local_vjp(")
    return _sub(src, TORSO_GRAD_TAIL, TORSO_GRAD_TAIL + INERTIAL_GRAD)


SOLE_DEVICE = r"""
// body_params: a geom's pose with the world's sole thickness th: a sole geom (SOLE_SLOT) at geom_pos + th n (body frame)
__device__ __forceinline__ void geom_pose_sole(int g, const float* Qb, const float* Xb, float* gx, float* gq, float th) {
  const int k = SOLE_SLOT(g);
  if (k < 0) { geom_pose(g, Qb, Xb, gx, gq); return; }
  float p[3], t[3];
  for (int i = 0; i < 3; i++) p[i] = RFMA(th, c_sole_n[3 * k + i], c_gpos[3 * g + i]);
  qrot_rn(Qb, p, t);
  for (int i = 0; i < 3; i++) gx[i] = RADD(Xb[i], t[i]);
  qmul_rn(Qb, c_gquat + 4 * g, gq);
}
"""
SOLE_POSE = ("    geom_pose(g, s.Q[b], s.X[b], s.gx[g], s.gq[g]);\n",
             "    geom_pose_sole(g, s.Q[b], s.X[b], s.gx[g], s.gq[g], s.pw[PW_SOLE]);  // (body_params)\n")
SOLE_VJP_DEVICE = r"""
// body_params: a contact's geom-position cotangent xbar on geom g of body b into the sole thickness's (this lane's sum)
__device__ __forceinline__ void sole_vjp(Adj& s, int g, int b, const float* xbar) {
  const int k = SOLE_SLOT(g);
  if (k < 0) return;
  float n[3];
  qrot(s.Q[b], c_sole_n + 3 * k, n);
  s.sole_acc[threadIdx.x & 31] += dot3(xbar, n);
}
"""
SOLE_GTB = (("if (b1) geom_to_body_vjp(s.gx[g1], s.X[b1], gb, gb + 3, fg, fg + 3);",
             "if (b1) { geom_to_body_vjp(s.gx[g1], s.X[b1], gb, gb + 3, fg, fg + 3); sole_vjp(s, g1, b1, gb); }"),
            ("geom_to_body_vjp(s.gx[g2], s.X[b2], gb + 6, gb + 9, fg + 6, fg + 9);",
             "geom_to_body_vjp(s.gx[g2], s.X[b2], gb + 6, gb + 9, fg + 6, fg + 9); sole_vjp(s, g2, b2, gb + 6);"))
SOLE_ADJ_STATE = ("  float gpw[NPARAM + 1];",
                  "  float sole_acc[32];  // the sole thickness's cotangent per contact lane (body_params)\n"
                  "  float gpw[NPARAM + 1];")
SOLE_ADJ_ZERO = ("  for (int i = lane; i <= NPARAM; i += 32) s.gpw[i] = 0.f;  // (world_params)\n",
                 "  for (int i = lane; i <= NPARAM; i += 32) s.gpw[i] = 0.f;  // (world_params)\n"
                 "  s.sole_acc[lane] = 0.f;  // (body_params)\n")
SOLE_ADJ_OUT = ("  __syncwarp();\n  for (int i = lane; i <= NPARAM; i += 32) gctrl[(size_t)w * (NU + NPARAM + 1) + NU + i] = s.gpw[i];\n",
                "  __syncwarp();\n"
                "  if (lane == 0) {  // the sole thickness's cotangent: -sum of the lanes' P terms, in lane order (body_params)\n"
                "    float t = 0.f;\n"
                "    for (int l = 0; l < 32; l++) t += s.sole_acc[l];\n"
                "    s.gpw[PW_SOLE] -= t;\n"
                "  }\n"
                "  __syncwarp();\n"
                "  for (int i = lane; i <= NPARAM; i += 32) gctrl[(size_t)w * (NU + NPARAM + 1) + NU + i] = s.gpw[i];\n")
ADJ_CONTACTS = "template <bool DBG>\n__device__ __noinline__ void adj_contacts("


def sole(src):
    """The sole_thickness edits of the final engine text."""
    src = _sub(src, "// world poses of the collision geoms (position, rotation quaternion) from the body poses\n",
               SOLE_DEVICE + "// world poses of the collision geoms (position, rotation quaternion) from the body poses\n")
    src = _sub(src, *SOLE_POSE, 2, "the geom poses (forward, adjoint)")
    n = src.count(SOLE_GTB[0][0])
    if n < 1 or src.count(SOLE_GTB[1][0]) != n:
        raise RuntimeError("body_params: the contact VJP's geom_to_body_vjp calls not found in pairs")
    for a, b in SOLE_GTB:
        src = src.replace(a, b)
    i = src.find("__device__ __forceinline__ float adj_contact_one(")
    if i < 0:
        raise RuntimeError("body_params: adj_contact_one not found")
    src = src[:i] + SOLE_VJP_DEVICE.lstrip("\n") + "\n" + src[i:]
    src = _sub(src, *SOLE_ADJ_STATE)
    src = _sub(src, *SOLE_ADJ_ZERO)
    return _sub(src, *SOLE_ADJ_OUT)


def iw_tables(m):
    """qpos0's kinematics and the dof structure for iw0_world (float64 / int arrays)."""
    import mujoco
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_kinematics(m, d)
    nv, nb = m.nv, m.nbody
    cdof = np.zeros((nv, 6))
    for j in range(m.njnt):
        b, a, t = int(m.jnt_bodyid[j]), int(m.jnt_dofadr[j]), int(m.jnt_type[j])
        if t == int(mujoco.mjtJoint.mjJNT_FREE):
            for k in range(3):
                cdof[a + k, 3 + k] = 1.0
                w = d.xmat[b].reshape(3, 3)[:, k]
                cdof[a + 3 + k, :3] = w
                cdof[a + 3 + k, 3:] = np.cross(d.xpos[b], w)
        elif t == int(mujoco.mjtJoint.mjJNT_HINGE):
            cdof[a, :3] = d.xaxis[j]
            cdof[a, 3:] = np.cross(d.xanchor[j], d.xaxis[j])
        elif t == int(mujoco.mjtJoint.mjJNT_SLIDE):
            cdof[a, 3:] = d.xaxis[j]
        else:
            raise ValueError("invweight0='per_world': ball joints are not supported")
    par = m.dof_parentid.astype(int)
    chains = []
    for i in range(nv):
        c, k = [], i
        while k >= 0:
            c.append(k)
            k = int(par[k])
        chains.append(c)
    maxdep = max(len(c) for c in chains)
    chain = np.zeros((nv, maxdep), int)
    for i, c in enumerate(chains):
        chain[i, :len(c)] = c
    dd = np.array([len(c) - 1 for c in chains])
    blast = np.zeros(nb, int)
    for b in range(1, nb):
        assert m.body_dofnum[b] >= 1, "invweight0='per_world': every moving body has a joint (the forest engine's)"
        blast[b] = int(m.body_dofadr[b] + m.body_dofnum[b] - 1)
    sub = np.zeros(nb, np.uint64)
    for c in range(1, nb):
        k = c
        while k >= 1:
            sub[k] |= np.uint64(1) << np.uint64(c)
            k = int(m.body_parentid[k])
    tasks = []
    for j in range(m.njnt):
        a, t = int(m.jnt_dofadr[j]), int(m.jnt_type[j])
        if t == int(mujoco.mjtJoint.mjJNT_FREE):
            tasks += [(a + 2, a, 3, a, 3), (a + 5, a + 3, 3, a + 3, 3)]
        else:
            tasks.append((a, a, 1, a, 1))
    return dict(xpos=d.xpos.copy(), xmat=d.xmat.copy(), ximat=d.ximat.copy(), cdof=cdof, dbody=m.dof_bodyid.astype(int),
                mass=m.body_mass.copy(), inert=m.body_inertia.copy(), ipos=m.body_ipos.copy(), arm=m.dof_armature.copy(),
                chain=chain, dd=dd, maxdep=maxdep, blast=blast, sub=sub, tasks=np.array(tasks, int))


def _dtab(name, x):
    v = np.asarray(x, np.float64).ravel()
    return f"__device__ const double {name}[{max(v.size, 1)}] = {{{', '.join(repr(float(t)) for t in v) or '0.0'}}};\n"


def _itab(name, x, kind="int"):
    v = np.asarray(x).ravel()
    suf = "ull" if kind == "unsigned long long" else ""
    return f"__device__ const {kind} {name}[{max(v.size, 1)}] = {{{', '.join(f'{int(t)}{suf}' for t in v) or '0'}}};\n"


def iw_header(m):
    """The constants and #defines of iw0_world."""
    t = iw_tables(m)
    return ("// invweight0 per world (forest_engine.body_params): qpos0's kinematics, the model's inertials (double) and\n"
            "// the dof chains\n"
            f"#define IW_MAXDEP {t['maxdep']}\n#define IW_NTD {len(t['tasks'])}\n"
            "#define IW_NH ((NV + 31) / 32)\n#define IW_NBH ((NB + 31) / 32)\n"
            "#ifdef NIB\n#define IW_INR (PW_OFF + PW_INERTIAL)\n#else\n#define IW_INR 0\n#endif\n"
            + _dtab("c_iw_mass", t["mass"]) + _dtab("c_iw_inert", t["inert"]) + _dtab("c_iw_ipos", t["ipos"])
            + _dtab("c_iw_arm", t["arm"]) + _dtab("c_iw_xpos", t["xpos"]) + _dtab("c_iw_xmat", t["xmat"]) + _dtab("c_iw_ximat", t["ximat"])
            + _dtab("c_iw_cdof", t["cdof"]) + _itab("c_iw_dbody", t["dbody"]) + _itab("c_iw_chain", t["chain"])
            + _itab("c_iw_dd", t["dd"]) + _itab("c_iw_blast", t["blast"])
            + _itab("c_iw_sub", t["sub"], "unsigned long long") + _itab("c_iw_task", t["tasks"])
            + "__shared__ float dk_biw[NB], dk_diw[NV];  // the world's body / dof inverse weights (iw0_world)\n")


def iw_header_mass(m, params):
    """The constants and #defines of the light iw0_world / iw0_grad (invweight0="per_world_mass")."""
    t = [(b, om) for b, om, _, _ in iw_targets(m, params, None) if om >= 0]
    return ("// invweight0 per world, 1 / m of free-floating bodies (forest_engine.body_params: per_world_mass)\n"
            "#define IW_MAXDEP 1\n#define IWM_N " + str(len(t)) + "\n"
            "#ifdef NIB\n#define IW_INR (PW_OFF + PW_INERTIAL)\n#else\n#define IW_INR 0\n#endif\n"
            + _dtab("c_iw_mass", np.asarray(m.body_mass, np.float64))
            + _itab("c_iwm_b", [x for bt in t for x in bt])
            + "__shared__ float dk_biw[NB], dk_diw[NV];  // the world's body / dof inverse weights (iw0_world)\n")


IW_DEVICE_MASS = r"""
// body_params: invweight0 per world, per_world_mass: the model's inverse weights, and 1 / m of the free-floating
// target bodies (c_iwm_b: body (-1: the torso fields' body), mass word), m their world's mass (the torso fields' or body_inertial's, else the
// model's, as iw0_factor). Whole warp; begins and ends with __syncwarp
__device__ __forceinline__ double iw0_mass(int b, const float* inr, const float* pw) {
  double m = c_iw_mass[b];
#if defined(PW_TMASS)
  if (b == (int)c_misc[1] && PW_TMASS >= 0) m = pw[PW_TMASS >= 0 ? PW_TMASS : 0];
#endif
#if defined(NIB)
  if (IB_SLOT(b) >= 0) m = inr[7 * IB_SLOT(b)];
#endif
  (void)inr;
  (void)pw;
  return m;
}
__device__ __noinline__ void iw0_world(const float* cof, const float* inr, const float* pw, int lane, float* biw,
                                       float* diw, double* sh) {
  (void)cof;
  (void)sh;
  __syncwarp();
  for (int i = lane; i < NB; i += 32) biw[i] = c_binvw[i];
  for (int i = lane; i < NV; i += 32) diw[i] = c_dinvw[i];
  __syncwarp();
  if (lane < IWM_N) {
    const int b = c_iwm_b[2 * lane] >= 0 ? c_iwm_b[2 * lane] : (int)c_misc[1];
    biw[b] = (float)(1.0 / iw0_mass(b, inr, pw));
  }
  __syncwarp();
}
"""
IW_GRAD_DEVICE_MASS = r"""
// body_params: per_world_mass's cotangents: d (1 / m) / d m = -1 / m^2 on each target's mass word (Adj.giw: dL / d iw)
__device__ __noinline__ void iw0_grad(Adj& s, int lane) {
  __syncwarp();
  if (lane == 0)
    for (int j = 0; j < IWM_N; j++) {
      const int b = c_iwm_b[2 * j] >= 0 ? c_iwm_b[2 * j] : (int)c_misc[1];
      const double m = iw0_mass(b, s.com + 7, s.pw);
      s.gpw[c_iwm_b[2 * j + 1]] += (float)(-(double)s.giw[b] / (m * m));
    }
  __syncwarp();
}
"""
IW_DEVICE = r"""
// body_params: invweight0 per world (MuJoCo's mj_setConst at qpos0, double precision). iw0_factor: M at qpos0 with the
// world's inertials (com: the state's com array, the COM offset and the inertial bodies' words; pw: the world's
// parameters; the model's values in double, c_iw_*, where the world does not set them, so that a world at a model's
// values gets that model's invweight0; cof: the COM offset input, inr: the inertial bodies' 7 words each) and its LDL'
// factor; out: the lane's bodies' COMs cp (world) and masses mb, the
// lane's dof rows (M = L' D L: row[.][0] = D, row[.][p] = L to the chain's p-th ancestor). Whole warp
__device__ __forceinline__ void iw0_factor(const float* cof, const float* inr, const float* pw, int lane,
                                           double (*cp)[3], double* mb, double (*row)[IW_MAXDEP]) {
  (void)inr;
  (void)pw;
  double ci[IW_NBH][10];  // the lane's bodies' spatial inertias about the origin (m, m c, I_o)
  for (int h = 0; h < IW_NBH; h++) {
    const int b = lane + 32 * h;
    for (int k = 0; k < 10; k++) ci[h][k] = 0.0;
    for (int k = 0; k < 3; k++) cp[h][k] = 0.0;
    mb[h] = 0.0;
    if (b >= 1 && b < NB) {
      double m = c_iw_mass[b], d0 = c_iw_inert[3 * b], d1 = c_iw_inert[3 * b + 1], d2 = c_iw_inert[3 * b + 2];
      double ip[3] = {c_iw_ipos[3 * b], c_iw_ipos[3 * b + 1], c_iw_ipos[3 * b + 2]};
      if (b == (int)c_misc[1]) {  // init's COM offset; the torso fields
        for (int k = 0; k < 3; k++) ip[k] += (double)cof[k];
#if defined(PW_TMASS)
        if (PW_TMASS >= 0) m = pw[PW_TMASS >= 0 ? PW_TMASS : 0];
        if (PW_TINERT >= 0) {
          d0 = pw[(PW_TINERT >= 0 ? PW_TINERT : 0)]; d1 = pw[(PW_TINERT >= 0 ? PW_TINERT : 0) + 1];
          d2 = pw[(PW_TINERT >= 0 ? PW_TINERT : 0) + 2];
        }
#endif
      }
#if defined(NIB)
      if (IB_SLOT(b) >= 0) {  // body_inertial
        const float* x = inr + 7 * IB_SLOT(b);
        m = x[0]; d0 = x[1]; d1 = x[2]; d2 = x[3];
        for (int k = 0; k < 3; k++) ip[k] += (double)x[4 + k];
      }
#endif
      const double* R = c_iw_xmat + 9 * b;
      const double* Ri = c_iw_ximat + 9 * b;
      double c[3];
      for (int k = 0; k < 3; k++)
        c[k] = c_iw_xpos[3 * b + k] + (R[3 * k] * ip[0] + R[3 * k + 1] * ip[1] + R[3 * k + 2] * ip[2]);
      double I[6];  // R diag(d) R' (xx yy zz xy xz yz)
      const int ia[6] = {0, 1, 2, 0, 0, 1}, ib[6] = {0, 1, 2, 1, 2, 2};
      for (int e = 0; e < 6; e++)
        I[e] = d0 * Ri[3 * ia[e]] * Ri[3 * ib[e]] + d1 * Ri[3 * ia[e] + 1] * Ri[3 * ib[e] + 1]
             + d2 * Ri[3 * ia[e] + 2] * Ri[3 * ib[e] + 2];
      const double cc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
      ci[h][0] = m;
      mb[h] = m;
      for (int k = 0; k < 3; k++) { ci[h][1 + k] = m * c[k]; cp[h][k] = c[k]; }
      for (int e = 0; e < 6; e++) ci[h][4 + e] = I[e] + m * ((ia[e] == ib[e] ? cc : 0.0) - c[ia[e]] * c[ib[e]]);
    }
  }
  double crb[IW_NBH][10];  // composite: the subtree sums (body order)
  for (int h = 0; h < IW_NBH; h++)
    for (int k = 0; k < 10; k++) crb[h][k] = 0.0;
  for (int c = 1; c < NB; c++) {
    double v[10];
    for (int k = 0; k < 10; k++) v[k] = __shfl_sync(0xffffffffu, ci[c >> 5][k], c & 31);
    for (int h = 0; h < IW_NBH; h++) {
      const int b = lane + 32 * h;
      if (b >= 1 && b < NB && ((c_iw_sub[b] >> c) & 1ull))
        for (int k = 0; k < 10; k++) crb[h][k] += v[k];
    }
  }
  for (int h = 0; h < IW_NH; h++) {
    const int i = lane + 32 * h, bi = i < NV ? c_iw_dbody[i] : 0;
    double cr[10];
    for (int k = 0; k < 10; k++) {
      double v = 0.0;
      for (int hb = 0; hb < IW_NBH; hb++) {
        const double x = __shfl_sync(0xffffffffu, crb[hb][k], bi & 31);
        if ((bi >> 5) == hb) v = x;
      }
      cr[k] = v;
    }
    for (int p = 0; p < IW_MAXDEP; p++) row[h][p] = 0.0;
    if (i < NV) {
      const double* s = c_iw_cdof + 6 * i;  // (w, v) about the origin
      const double m = cr[0], hx = cr[1], hy = cr[2], hz = cr[3];
      // L = I_o w + h x v, P = m v - h x w
      const double L0 = cr[4] * s[0] + cr[7] * s[1] + cr[8] * s[2] + (hy * s[5] - hz * s[4]);
      const double L1 = cr[7] * s[0] + cr[5] * s[1] + cr[9] * s[2] + (hz * s[3] - hx * s[5]);
      const double L2 = cr[8] * s[0] + cr[9] * s[1] + cr[6] * s[2] + (hx * s[4] - hy * s[3]);
      const double P0 = m * s[3] - (hy * s[2] - hz * s[1]);
      const double P1 = m * s[4] - (hz * s[0] - hx * s[2]);
      const double P2 = m * s[5] - (hx * s[1] - hy * s[0]);
      for (int p = 0; p <= c_iw_dd[i]; p++) {
        const double* t = c_iw_cdof + 6 * c_iw_chain[i * IW_MAXDEP + p];
        row[h][p] = t[0] * L0 + t[1] * L1 + t[2] * L2 + t[3] * P0 + t[4] * P1 + t[5] * P2;
      }
#if defined(PW_ARM) && PW_ARM >= 0
      row[h][0] += (double)pw[PW_ARM + i];
#else
      row[h][0] += c_iw_arm[i];
#endif
    }
  }
  for (int k = NV - 1; k >= 0; k--) {  // MuJoCo's sparse LDL' (mj_factorI): rows from the leaves
    double rk[IW_MAXDEP];
    for (int p = 0; p < IW_MAXDEP; p++) rk[p] = __shfl_sync(0xffffffffu, row[k >> 5][p], k & 31);
    const int ddk = c_iw_dd[k];
    for (int h = 0; h < IW_NH; h++) {
      const int i = lane + 32 * h;
      if (i >= NV || i == k) continue;
      const int p = ddk - c_iw_dd[i];
      if (p < 1 || c_iw_chain[k * IW_MAXDEP + p] != i) continue;
      const double t = rk[p] / rk[0];
      for (int q = 0; q <= c_iw_dd[i]; q++) row[h][q] -= t * rk[p + q];
    }
    if (lane == (k & 31))
      for (int p = 1; p <= ddk; p++) row[k >> 5][p] = rk[p] / rk[0];
  }
}

// the lane's task t (bodies 1 .. NB - 1: the COM Jacobian's 3 translational rows; then the dof tasks: unit rows):
// its rows y over the chain of dof st; ob: its body (-1), od / no: its dofs; returns the row count (0: no task).
// Whole warp (the body's COM from its lane)
__device__ __forceinline__ int iw0_task(int t, const double (*cp)[3], double (*y)[IW_MAXDEP], int* st, int* ob, int* od,
                                        int* no) {
  const int nt = NB - 1 + IW_NTD;
  int nr = 0;
  *st = 0; *ob = -1; *od = -1; *no = 0;
  for (int r = 0; r < 3; r++)
    for (int p = 0; p < IW_MAXDEP; p++) y[r][p] = 0.0;
  const int bt = t + 1 < NB ? t + 1 : 0;  // (the COM of a body task's body from its lane)
  double c[3];
  for (int k = 0; k < 3; k++) {
    double v = 0.0;
    for (int hb = 0; hb < IW_NBH; hb++) {
      const double x = __shfl_sync(0xffffffffu, cp[hb][k], bt & 31);
      if ((bt >> 5) == hb) v = x;
    }
    c[k] = v;
  }
  if (t < NB - 1) {
    *ob = t + 1; *st = c_iw_blast[*ob]; nr = 3;
    for (int p = 0; p <= c_iw_dd[*st]; p++) {  // rows: v + w x c of the chain's dofs
      const double* s = c_iw_cdof + 6 * c_iw_chain[*st * IW_MAXDEP + p];
      y[0][p] = s[3] + (s[1] * c[2] - s[2] * c[1]);
      y[1][p] = s[4] + (s[2] * c[0] - s[0] * c[2]);
      y[2][p] = s[5] + (s[0] * c[1] - s[1] * c[0]);
    }
  } else if (t < nt) {
    const int* q = c_iw_task + 5 * (t - (NB - 1));
    *st = q[0]; nr = q[2]; *od = q[3]; *no = q[4];
    for (int r = 0; r < nr; r++) y[r][c_iw_dd[*st] - c_iw_dd[q[1] + r]] = 1.0;
  }
  return nr;
}

// y = L^-T y (L' y = J') over the chain of st in place, the rows broadcast from their lanes (whole warp); dg: D at the
// chain's positions; returns sum y' D^-1 y (the rows' diag(J M^-1 J') summed)
__device__ __forceinline__ double iw0_lt(const double (*row)[IW_MAXDEP], int st, int nr, double (*y)[IW_MAXDEP],
                                         double* dg) {
  double acc = 0.0;
  for (int k = NV - 1; k >= 0; k--) {
    double rk[IW_MAXDEP];
    for (int p = 0; p < IW_MAXDEP; p++) rk[p] = __shfl_sync(0xffffffffu, row[k >> 5][p], k & 31);
    if (!nr) continue;
    const int p = c_iw_dd[st] - c_iw_dd[k];
    if (p < 0 || c_iw_chain[st * IW_MAXDEP + p] != k) continue;
    dg[p] = rk[0];
    for (int r = 0; r < nr; r++) {
      const double yp = y[r][p];
      acc += yp * yp / rk[0];
      for (int q = 1; q <= c_iw_dd[k]; q++) y[r][p + q] -= rk[q] * yp;
    }
  }
  return acc;
}

// iw0_lt with the rows in shared memory (sh: NV x IW_MAXDEP), each lane on its own task's chain alone: the same
// operations in the same order (the chain's dofs from st up are iw0_lt's k from NV - 1 down that act), so the same bits
__device__ __forceinline__ double iw0_lt_sh(const double* sh, int st, int nr, double (*y)[IW_MAXDEP], double* dg) {
  double acc = 0.0;
  if (!nr) return acc;
  for (int p = 0; p <= c_iw_dd[st]; p++) {
    const int k = c_iw_chain[st * IW_MAXDEP + p];
    const double* rk = sh + k * IW_MAXDEP;
    dg[p] = rk[0];
    for (int r = 0; r < nr; r++) {
      const double yp = y[r][p];
      acc += yp * yp / rk[0];
      for (int q = 1; q <= c_iw_dd[k]; q++) y[r][p + q] -= rk[q] * yp;
    }
  }
  return acc;
}

// the world's body_invweight0[:, 0] and dof_invweight0 into biw / diw (shared or global); sh: null, or the warp's
// NV x IW_MAXDEP doubles of shared memory for the rows (k_iw: no shuffles in the solves). Whole warp; begins and ends
// with __syncwarp
__device__ __noinline__ void iw0_world(const float* cof, const float* inr, const float* pw, int lane, float* biw,
                                       float* diw, double* sh) {
  __syncwarp();
  double cp[IW_NBH][3], mb[IW_NBH], row[IW_NH][IW_MAXDEP];
  iw0_factor(cof, inr, pw, lane, cp, mb, row);
  if (sh) {
    for (int h = 0; h < IW_NH; h++)
      if (lane + 32 * h < NV)
        for (int p = 0; p < IW_MAXDEP; p++) sh[(lane + 32 * h) * IW_MAXDEP + p] = row[h][p];
    __syncwarp();
  }
  const int nt = NB - 1 + IW_NTD;
  for (int t0 = 0; t0 < nt; t0 += 32) {
    int st, ob, od, no;
    double y[3][IW_MAXDEP], dg[IW_MAXDEP];
    const int nr = iw0_task(t0 + lane, cp, y, &st, &ob, &od, &no);
    const double acc = sh ? iw0_lt_sh(sh, st, nr, y, dg) : iw0_lt(row, st, nr, y, dg);
    if (ob >= 0) biw[ob] = (float)(acc / 3.0);
    for (int o = 0; o < no; o++) diw[od + o] = (float)(acc / (double)nr);
  }
  if (lane == 0) biw[0] = c_binvw[0];
  __syncwarp();
}
"""
IW_GRAD_DEVICE = r"""
// body_params: the cotangents of the per-world invweight0 on the parameters (the derivative of mj_setConst's values).
// dL / d iw (Adj.giw per body: the contacts' rows, R = iw (1 - imp) / imp; Adj.gdiw per dof: the limit rows) is summed
// over the substeps; here, with L_iw = sum_b G_b tr(Jp_b M^-1 Jp_b') / 3 + sum_t H_t sum_r e_r' M^-1 e_r at qpos0:
// dL = sum_tasks w [2 tr(dJ X') - tr(X' dM X)], X = M^-1 J' (the task's rows), with dM: mass m_k: Jp_k' Jp_k (COM
// Jacobian), principal moment j: Jr_k' u_j u_j' Jr_k, COM offset (world e_l): m_k (dJp' Jp + Jp' dJp), dJp = -[e_l]x Jr
// (the columns w_i x e_l), armature i: e_i e_i'; and the own term 2 w tr(dJp_b X') of a body task's COM. Added to
// Adj.gpw (the targets c_iw_tg: body (-1: the torso fields' body), mass, moments, COM offset words; PW_ARM)
__device__ __forceinline__ double iw_wsum(double v) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __noinline__ void iw0_grad(Adj& s, int lane) {
  __syncwarp();
  double cp[IW_NBH][3], mb[IW_NBH], row[IW_NH][IW_MAXDEP];
  iw0_factor(s.com, s.com + 7, s.pw, lane, cp, mb, row);
  for (int h = 0; h < IW_NH; h++)  // the rows into shared memory: the solves need no shuffles
    if (lane + 32 * h < NV)
      for (int p = 0; p < IW_MAXDEP; p++) s.iwsh[(lane + 32 * h) * IW_MAXDEP + p] = row[h][p];
  __syncwarp();
  double tc[IW_NTG][3], tm[IW_NTG], gt[IW_NTG][7];  // the targets' COMs (world), masses; their cotangents
  for (int j = 0; j < IW_NTG; j++) {
    const int kb = c_iw_tg[4 * j] >= 0 ? c_iw_tg[4 * j] : (int)c_misc[1];
    for (int k = 0; k < 3; k++) {
      double v = 0.0;
      for (int hb = 0; hb < IW_NBH; hb++) {
        const double x = __shfl_sync(0xffffffffu, cp[hb][k], kb & 31);
        if ((kb >> 5) == hb) v = x;
      }
      tc[j][k] = v;
    }
    double v = 0.0;
    for (int hb = 0; hb < IW_NBH; hb++) {
      const double x = __shfl_sync(0xffffffffu, mb[hb], kb & 31);
      if ((kb >> 5) == hb) v = x;
    }
    tm[j] = v;
    for (int k = 0; k < 7; k++) gt[j][k] = 0.0;
  }
  double ga[NV];
  for (int i = 0; i < NV; i++) ga[i] = 0.0;
  const int nt = NB - 1 + IW_NTD;
  for (int t0 = 0; t0 < nt; t0 += 32) {
    int st, ob, od, no;
    double y[3][IW_MAXDEP], dg[IW_MAXDEP], X[3][NV];
    const int nr = iw0_task(t0 + lane, cp, y, &st, &ob, &od, &no);
    iw0_lt_sh(s.iwsh, st, nr, y, dg);
    for (int r = 0; r < 3; r++)
      for (int i = 0; i < NV; i++) X[r][i] = 0.0;
    if (nr)
      for (int p = 0; p <= c_iw_dd[st]; p++)
        for (int r = 0; r < nr; r++) X[r][c_iw_chain[st * IW_MAXDEP + p]] = y[r][p] / dg[p];
    for (int k = 0; k < NV; k++) {  // X = L^-1 (D^-1 y): L x = z from the roots
      const double* rk = s.iwsh + k * IW_MAXDEP;
      for (int r = 0; r < nr; r++) {
        double x = X[r][k];
        for (int q = 1; q <= c_iw_dd[k]; q++) x -= rk[q] * X[r][c_iw_chain[k * IW_MAXDEP + q]];
        X[r][k] = x;
      }
    }
    double w = 0.0;  // the task's weight: dL / d (its sum)
    if (ob >= 0) w = (double)s.giw[ob] / 3.0;
    for (int o = 0; o < no; o++) w += (double)s.gdiw[od + o] / (double)nr;
    if (w == 0.0) continue;
    for (int j = 0; j < IW_NTG; j++) {
      const int kb = c_iw_tg[4 * j] >= 0 ? c_iw_tg[4 * j] : (int)c_misc[1];
      const int sk = c_iw_blast[kb];
      double P[3][3], Q[3][3];  // Jp_k X', Jr_k X' (row a, task row r)
      for (int a = 0; a < 3; a++)
        for (int r = 0; r < 3; r++) P[a][r] = Q[a][r] = 0.0;
      for (int p = 0; p <= c_iw_dd[sk]; p++) {
        const int i = c_iw_chain[sk * IW_MAXDEP + p];
        const double* sv = c_iw_cdof + 6 * i;
        const double jp[3] = {sv[3] + (sv[1] * tc[j][2] - sv[2] * tc[j][1]), sv[4] + (sv[2] * tc[j][0] - sv[0] * tc[j][2]),
                              sv[5] + (sv[0] * tc[j][1] - sv[1] * tc[j][0])};
        for (int r = 0; r < nr; r++)
          for (int a = 0; a < 3; a++) { P[a][r] += jp[a] * X[r][i]; Q[a][r] += sv[a] * X[r][i]; }
      }
      const double* Ri = c_iw_ximat + 9 * kb;
      for (int r = 0; r < nr; r++) {
        gt[j][0] -= w * (P[0][r] * P[0][r] + P[1][r] * P[1][r] + P[2][r] * P[2][r]);
        for (int a = 0; a < 3; a++) {
          const double u = Ri[a] * Q[0][r] + Ri[3 + a] * Q[1][r] + Ri[6 + a] * Q[2][r];  // (principal axis a: column a)
          gt[j][1 + a] -= w * u * u;
        }
        // -2 m_k w P_r x Q_r (M's COM dependence), and a body task of the target body: 2 w e_r x Q_r
        gt[j][4] -= 2.0 * tm[j] * w * (P[1][r] * Q[2][r] - P[2][r] * Q[1][r]);
        gt[j][5] -= 2.0 * tm[j] * w * (P[2][r] * Q[0][r] - P[0][r] * Q[2][r]);
        gt[j][6] -= 2.0 * tm[j] * w * (P[0][r] * Q[1][r] - P[1][r] * Q[0][r]);
        if (ob == kb) {
          const double e[3] = {r == 0 ? 1.0 : 0.0, r == 1 ? 1.0 : 0.0, r == 2 ? 1.0 : 0.0};
          gt[j][4] += 2.0 * w * (e[1] * Q[2][r] - e[2] * Q[1][r]);
          gt[j][5] += 2.0 * w * (e[2] * Q[0][r] - e[0] * Q[2][r]);
          gt[j][6] += 2.0 * w * (e[0] * Q[1][r] - e[1] * Q[0][r]);
        }
      }
    }
    for (int i = 0; i < NV; i++)
      for (int r = 0; r < nr; r++) ga[i] -= w * X[r][i] * X[r][i];
  }
  for (int j = 0; j < IW_NTG; j++) {
    const int kb = c_iw_tg[4 * j] >= 0 ? c_iw_tg[4 * j] : (int)c_misc[1];
    double g[7];
    for (int k = 0; k < 7; k++) g[k] = iw_wsum(gt[j][k]);
    if (lane == 0) {
      const int om = c_iw_tg[4 * j + 1], oi = c_iw_tg[4 * j + 2], oc = c_iw_tg[4 * j + 3];
      if (om >= 0) s.gpw[om] += (float)g[0];
      if (oi >= 0) for (int a = 0; a < 3; a++) s.gpw[oi + a] += (float)g[1 + a];
      if (oc >= 0) {  // world -> the body frame (xmat at qpos0)
        const double* R = c_iw_xmat + 9 * kb;
        for (int a = 0; a < 3; a++) s.gpw[oc + a] += (float)(R[a] * g[4] + R[3 + a] * g[5] + R[6 + a] * g[6]);
      }
    }
  }
#if defined(PW_ARM) && PW_ARM >= 0
  for (int i = 0; i < NV; i++) {
    const double g = iw_wsum(ga[i]);
    if (lane == 0) s.gpw[PW_ARM + i] += (float)g;
  }
#endif
  __syncwarp();
}
"""
IW_CONTACT_DEVICE = r"""
// body_params: sum_e (u_e . V_lam) x_e over the contact's rows in its zone, times D (contact_param_vjp's rows)
__device__ float contact_iw_vjp(const float* Vv, const float* Va, const float* Vl, const float* fr, float mu, int dim,
                                unsigned zone, float d, float D, float k, float b, float imp, const float* Ov,
                                const float* Oa, const float* Ol, float mus) {
  if (!(D < 1e15f)) return 0.f;
  float vf[3], af[3], lf[3], acc = 0.f;
  for (int j = 0; j < 3; j++) { vf[j] = dot3(fr + 3 * j, Vv); af[j] = dot3(fr + 3 * j, Va); lf[j] = dot3(fr + 3 * j, Vl); }
  for (int e = 0; e < 4; e++) {
    if (!((zone >> e) & 1u) || (e > 0 && dim < 3)) continue;
    const int t = 1 + (e >> 1);
    const float se = dim >= 3 ? ((e & 1) ? -1.f : 1.f) : 0.f, sm = se * mu;
    const float uv = vf[0] + sm * vf[t], ua = af[0] + sm * af[t], ul = lf[0] + sm * lf[t];
    acc += ul * (ua + k * imp * d + b * uv);
  }
  if (dim == 4) {
    const float wv = dot3(fr, Ov), wa = dot3(fr, Oa), wl = dot3(fr, Ol);
    for (int e = 4; e < 6; e++) {
      if (!((zone >> e) & 1u)) continue;
      const float sm = ((e & 1) ? -1.f : 1.f) * mus;
      const float uv = vf[0] + sm * wv, ua = af[0] + sm * wa, ul = lf[0] + sm * wl;
      acc += ul * (ua + k * imp * d + b * uv);
    }
  }
  return D * acc;
}
"""
IW_CONTACT_CALL = ("  cot[1] = cot[2] = cot[3] = 0.f;\n  contact_pbar_motion(Wv, Wa, Wl, Ov, Oa, Ol, cot + 1);\n",
                   "  if (zone) {  // dL / d (iw_b1 + iw_b2) of this contact (body_params)\n"
                   "    s.iwg[threadIdx.x & 31] = contact_iw_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, k, b, imp, Ov, Oa, Ol,\n"
                   "                                             dim == 4 ? IW_MUS(pair) : 0.f) / RADD(dk_biw[b1], dk_biw[b2]);\n"
                   "    s.iwp[threadIdx.x & 31] = pair;\n"
                   "  }\n"
                   "  cot[1] = cot[2] = cot[3] = 0.f;\n  contact_pbar_motion(Wv, Wa, Wl, Ov, Oa, Ol, cot + 1);\n")
IW_BATCH = ("    const unsigned skip = __ballot_sync(0xffffffffu, on && !use), mis = __ballot_sync(0xffffffffu, bad);\n",
            "    __syncwarp();  // the batch's invweight0 cotangents onto the pairs' bodies, in lane order (body_params)\n"
            "    if (lane == 0)\n"
            "      for (int l = 0; l < 32; l++)\n"
            "        if (s.iwg[l] != 0.f) {\n"
            "          const int pr = s.iwp[l];\n"
            "          s.giw[c_gbody[PG1(pr)]] += s.iwg[l];\n"
            "          s.giw[c_gbody[PG2(pr)]] += s.iwg[l];\n"
            "        }\n"
            "    __syncwarp();\n"
            "    s.iwg[lane] = 0.f;\n"
            "    const unsigned skip = __ballot_sync(0xffffffffu, on && !use), mis = __ballot_sync(0xffffffffu, bad);\n")
IW_LIMIT = ("        s.Pq[i] += pq;\n        s.Pv[i] += pv;\n",
            "        s.Pq[i] += pq;\n        s.Pv[i] += pv;\n"
            "        if (D < 1e15f)  // dL / d dof_invweight0 (body_params)\n"
            "          s.gdiw[i] += sg * s.lam[i] * (sg * s.a[i] + k * imp * pos + b * (sg * s.v[i])) * D / dk_diw[i];\n")
IW_ADJ_STATE = ("  float gpw[NPARAM + 1];",
                "  double iwsh[NV * IW_MAXDEP];  // the invweight0 factor's rows (iw0_world / iw0_grad, body_params)\n"
                "  float giw[NB], gdiw[NV], iwg[32];  // dL / d invweight0 per body / dof; a batch's per contact lane\n"
                "  int iwp[32];  // (body_params)\n  float gpw[NPARAM + 1];")
IW_ADJ_ZERO = ("  for (int i = lane; i <= NPARAM; i += 32) s.gpw[i] = 0.f;  // (world_params)\n",
               "  for (int i = lane; i <= NPARAM; i += 32) s.gpw[i] = 0.f;  // (world_params)\n"
               "  for (int i = lane; i < NB; i += 32) s.giw[i] = 0.f;  // (body_params)\n"
               "  for (int i = lane; i < NV; i += 32) s.gdiw[i] = 0.f;\n"
               "  s.iwg[lane] = 0.f;\n")
IW_ADJ_OUT = ("  for (int i = lane; i <= NPARAM; i += 32) gctrl[(size_t)w * (NU + NPARAM + 1) + NU + i] = s.gpw[i];\n",
              "  iw0_grad(s, lane);  // (body_params: the invweight0 path's cotangents)\n"
              "  for (int i = lane; i <= NPARAM; i += 32) gctrl[(size_t)w * (NU + NPARAM + 1) + NU + i] = s.gpw[i];\n")


def iw_targets(m, params, com_body_fields):
    """[(body, mass word, moments word, COM word)] of the invweight0 gradient's targets (-1: none; body -1: the torso
    fields' body)."""
    lay = params.layout(m.nv)
    out = []
    if "torso_mass" in lay or "torso_inertia" in lay:
        out.append((-1, lay["torso_mass"][0] if "torso_mass" in lay else -1,
                    lay["torso_inertia"][0] if "torso_inertia" in lay else -1, -1))
    if INERTIAL in lay:
        o = lay[INERTIAL][0]
        out += [(b, o + 7 * k, o + 7 * k + 1, o + 7 * k + 4) for k, b in enumerate(inertial_ids(m, params))]
    return out


def iw_grad(src, m, params):
    """The invweight0 gradient's edits of the final engine text (after iw()); per_world_mass: its light iw0_grad."""
    tg = iw_targets(m, params, None)
    tors = "foot_torsional" in params.fields
    head = (f"#define IW_NTG {max(len(tg), 1)}  // the invweight0 gradient's target bodies (body_params)\n"
            + _itab("c_iw_tg", [x for t in (tg or [(0, -1, -1, -1)]) for x in t])
            + ("#define IW_MUS(p) (PFOOT(p) ? s.musw : PMUS(p))\n" if tors else
               "#define IW_MUS(p) (PMUS(p))\n"))
    src = _sub(src, "template <int PHASE = 0, class S>\n__device__ void substep_pre(",
               head + "template <int PHASE = 0, class S>\n__device__ void substep_pre(")
    src = _sub(src, "__device__ __forceinline__ float adj_contact_one(",
               IW_CONTACT_DEVICE.lstrip("\n") + "\n__device__ __forceinline__ float adj_contact_one(")
    src = _sub(src, *IW_CONTACT_CALL)
    src = _sub(src, *IW_BATCH)
    src = _sub(src, *IW_LIMIT)
    src = _sub(src, *IW_ADJ_STATE)
    src = _sub(src, *IW_ADJ_ZERO)
    dev = IW_GRAD_DEVICE_MASS if iw_mass(params) else IW_GRAD_DEVICE
    src = _sub(src, "__global__ void k_adjoint(", dev.lstrip("\n") + "\n__global__ void k_adjoint(")
    return _sub(src, *IW_ADJ_OUT)


IW_SUBSTEP = ("  __syncwarp();\n  int nit = 0;\n  for (int k = 0; k < nsub; k++) {\n",
              "  __syncwarp();\n  iw0_world(s.com, s.com + 7, s.pw, lane, dk_biw, dk_diw, nullptr);  // (body_params)\n  int nit = 0;\n"
              "  for (int k = 0; k < nsub; k++) {\n")
IW_PRE = ("  __syncwarp();\n#if CV_DEFER\n  {  // the contacts of k_pre_a and k_ccd\n",
          "  for (int i = lane; i < NB; i += 32) dk_biw[i] = g.iwb[i];  // the world's inverse weights (body_params: k_iw)\n"
          "  for (int i = lane; i < NV; i += 32) dk_diw[i] = g.iwd[i];\n"
          "  __syncwarp();\n#if CV_DEFER\n  {  // the contacts of k_pre_a and k_ccd\n")
IW_KERNEL = ("static inline size_t dk_order_bytes(int nw) {",
             "__global__ void __launch_bounds__(128) k_iw(int nw, const float* com, Hand* work) {  // (body_params)\n"
             "  const int lane = threadIdx.x & 31, w = blockIdx.x * 4 + (threadIdx.x >> 5);\n"
             "  if (w >= nw) return;  // (a whole warp)\n"
             "  __shared__ double sh[4][NV * IW_MAXDEP];  // the warps' factor rows\n"
             "  const float* row = com + (size_t)w * COM_STRIDE;\n"
             "  iw0_world(row, row + IW_INR, row + PW_OFF, lane, work[w].iwb, work[w].iwd, sh[threadIdx.x >> 5]);\n"
             "}\n"
             "static inline size_t dk_order_bytes(int nw) {")
IW_LAUNCH = ("  for (int k = 0; k < nsub; k++) {\n#if CV_DEFER\n    if (dk_split_only != 2) {",
             "  if (dk_split_only != 2) k_iw<<<(nw + 3) / 4, 128, 0, st>>>(nw, com, hand);  // (body_params)\n"
             "  for (int k = 0; k < nsub; k++) {\n#if CV_DEFER\n    if (dk_split_only != 2) {")
IW_ADJ = ("  __syncwarp();\n  APROF(APROF_IO);\n  for (int k = nsub - 1; k >= 0; k--) {\n",
          "  __syncwarp();\n  iw0_world(s.com, s.com + 7, s.pw, lane, dk_biw, dk_diw, s.iwsh);  // (body_params)\n  APROF(APROF_IO);\n"
          "  for (int k = nsub - 1; k >= 0; k--) {\n")
IW_ADJ_W2 = ("  if (DK_ADJ_WARPS == 2) __syncthreads(); else __syncwarp();\n  APROF(APROF_IO);\n"
             "  for (int k = nsub - 1; k >= 0; k--) {\n",
             "  if (DK_ADJ_WARPS == 2) __syncthreads(); else __syncwarp();\n"
             "  if (threadIdx.x < 32) iw0_world(s.com, s.com + 7, s.pw, lane, dk_biw, dk_diw, s.iwsh);  // (body_params)\n"
             "  if (DK_ADJ_WARPS == 2) __syncthreads(); else __syncwarp();\n  APROF(APROF_IO);\n"
             "  for (int k = nsub - 1; k >= 0; k--) {\n")
IW_HAND = ("struct Hand {\n  float M[NTRI], qsm[NV], qacc0[NV];\n",
           "struct Hand {\n  float M[NTRI], qsm[NV], qacc0[NV];\n"
           "  float iwb[NB], iwd[NV];  // the world's inverse weights (body_params: invweight0 per world)\n")
IW_DECL = "DKC float c_binvw[NB], c_dinvw[NV],"


def iw(src, m, params=None):
    """The per-world invweight0 edits of the final engine text: the constants, iw0_world, the reads, k_adjoint's call,
    struct Hand's copy (per_world_mass: its light constants and iw0_world)."""
    if src.count(IW_DECL) != 1:
        raise RuntimeError("body_params: c_binvw / c_dinvw's declaration not found")
    i = src.index(IW_DECL)
    rest = src[i + len(IW_DECL):]
    n = rest.count("c_binvw[") + rest.count("c_dinvw[")
    if n < 2:
        raise RuntimeError(f"body_params: {n} inverse-weight reads")
    rest = rest.replace("c_binvw[", "dk_biw[").replace("c_dinvw[", "dk_diw[")
    src = src[:i + len(IW_DECL)] + rest
    j = src.index("\n", src.index(";", src.index(IW_DECL))) + 1
    mass = iw_mass(params)
    src = src[:j] + (iw_header_mass(m, params) if mass else iw_header(m)) + src[j:]
    src = _sub(src, "template <int PHASE = 0, class S>\n__device__ void substep_pre(",
               (IW_DEVICE_MASS if mass else IW_DEVICE).lstrip("\n") + "\ntemplate <int PHASE = 0, class S>\n__device__ void substep_pre(")
    src = _sub(src, *IW_HAND)
    if src.count(IW_ADJ[0]) == 0 and src.count(IW_ADJ_W2[0]) == 1:
        return _sub(src, *IW_ADJ_W2)
    return _sub(src, *IW_ADJ)


def iw_host(src):
    src = _sub(_sub(src, *IW_SUBSTEP), *IW_PRE)
    if src.count(IW_LAUNCH[0]) == 1:
        return _sub(_sub(src, *IW_KERNEL), *IW_LAUNCH)
    return _sub(_sub(src, *IW_KERNEL), IW_LAUNCH[0].replace("\n#if CV_DEFER\n    if (dk_split_only != 2) {", "\n"),
                IW_LAUNCH[1].replace("\n#if CV_DEFER\n    if (dk_split_only != 2) {", "\n"))


def engine(src, m, params, foot_pairs=()):
    """The final engine text's edits (adjoint.cuda_source, after world_forces): sole_thickness, invweight0; the
    defines. Unchanged without these options (body_inertial's edits come with world_params' torso edits)."""
    if not active(params):
        return src
    if sole_active(params):
        src = sole(src)
    if iw_active(params):
        src = iw_grad(iw(src, m, params), m, params)
    return defines(m, params, foot_pairs) + src

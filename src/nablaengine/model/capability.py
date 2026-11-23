"""Which MuJoCo models the engine supports: input validation of a model before anything is built."""

import os
import warnings
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from nablaengine.model import fields as F

ALLOW_VAR = "DK_ALLOW_UNSUPPORTED"
COM_BODY = "torso_link"
SCOPES = {"forward": ("forward",), "engine": ("forward", "adjoint"), "kinematics": ("kinematics",),
          "forest": ("forest",)}
J, G, T = mujoco.mjtJoint, mujoco.mjtGeom, mujoco.mjtTrn
GEOM_CODE = {int(G.mjGEOM_PLANE): 0, int(G.mjGEOM_SPHERE): 1, int(G.mjGEOM_CAPSULE): 2, int(G.mjGEOM_BOX): 3,
             int(G.mjGEOM_CYLINDER): 5, int(G.mjGEOM_MESH): 6, int(G.mjGEOM_ELLIPSOID): 7}
FORWARD_TYPES = {0, 1, 2, 3, 5, 6}
ADJOINT_TYPES = {0, 1, 2}
DISPATCHED = {(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3),
              (0, 5), (1, 5), (1, 3), (2, 5), (5, 5), (3, 5),
              (0, 6), (1, 6), (2, 6), (3, 6), (5, 6), (6, 6)}
CONTACTS_PER_PAIR = {(0, 1): 1, (0, 2): 2, (0, 3): 4, (1, 1): 1, (1, 2): 1, (2, 2): 2, (2, 3): 2, (3, 3): 8,
                     (0, 5): 4, (1, 5): 1, (1, 3): 1, (2, 5): 4, (5, 5): 4, (3, 5): 4, (0, 6): 4, (1, 6): 1, (2, 6): 4,
                     (3, 6): 4, (5, 6): 4, (6, 6): 4}
D, E = mujoco.mjtDisableBit, mujoco.mjtEnableBit
DISABLE_OK = (D.mjDSBL_SENSOR, D.mjDSBL_EULERDAMP, D.mjDSBL_AUTORESET, D.mjDSBL_MIDPHASE, D.mjDSBL_ISLAND,
              D.mjDSBL_FILTERPARENT, D.mjDSBL_CLAMPCTRL)
DISABLE_OK_NO_BOX = (D.mjDSBL_NATIVECCD, D.mjDSBL_MULTICCD)
DISABLE_NOOP = (D.mjDSBL_SPRING, D.mjDSBL_DAMPER, D.mjDSBL_EQUALITY)
ENABLE_OK = (E.mjENBL_ENERGY, E.mjENBL_FWDINV, E.mjENBL_INVDISCRETE)
STATUSES = ("error", "warn", "info", "ok")
CTRLSPEC_OK = (0, int(mujoco.mjtCtrlInput.mjINPUT_POS))


def _bits(flags, enum):
    return [k for k, v in enum.__members__.items() if not k.startswith("mjN") and flags & int(v)]


def _mask(members):
    out = 0
    for v in members:
        out |= int(v)
    return out


@dataclass(frozen=True)
class Finding:
    code: str
    status: str
    what: str
    value: object = None
    scopes: tuple = ("forward",)
    applies: bool = True

    def line(self):
        v = "" if self.value is None else f" [{_short(self.value)}]"
        return f"{self.code:5s} {self.status:5s} {self.what}{v}"


def _short(v, n=160):
    s = str(v)
    return s if len(s) <= n else s[:n - 3] + "..."


class CapabilityReport:
    """The findings of check() for one model and scope."""

    def __init__(self, scope, findings):
        self.scope, self.findings = scope, list(findings)

    def _of(self, status):
        return [f for f in self.findings if f.applies and f.status == status]

    def errors(self):
        return self._of("error")

    def warnings(self):
        return self._of("warn")

    def infos(self):
        return self._of("info")

    def codes(self, status="error"):
        return sorted({f.code for f in self._of(status)})

    def ok(self):
        return not self.errors()

    def table(self, all_rows=False):
        """One line per finding that applies and is not ok (all_rows: every finding)."""
        rows = [f for f in self.findings if all_rows or (f.applies and f.status != "ok")]
        return "\n".join(f.line() + ("" if f.applies else "  (other scope)") for f in rows)

    def summary(self):
        return (f"scope {self.scope}: {len(self.errors())} errors {self.codes('error')}, {len(self.warnings())} "
                f"warnings {self.codes('warn')}, {len(self.infos())} infos {self.codes('info')}")

    def to_dict(self):
        return dict(scope=self.scope, errors=self.codes("error"), warnings=self.codes("warn"), infos=self.codes("info"),
                    findings=[{k: (_jsonable(v)) for k, v in asdict(f).items()} for f in self.findings])


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


class UnsupportedModel(ValueError, AssertionError):
    """A model the engine does not support (capability errors). A ValueError (as jax_ffi.check_model's model
    mismatch) and an AssertionError (as the code generators' asserts, which jax_ffi.init documents)."""

    def __init__(self, report, context=""):
        self.report = report
        lines = "\n  ".join(f.line() for f in report.errors())
        eng = "the forest engine" if report.scope == "forest" else "the engine"
        super().__init__(f"nablaengine: the model is not supported by {eng}{' (' + context + ')' if context else ''}"
                         f": {report.summary()}\n  {lines}\n(nablaengine.model.capability documents the codes; "
                         f"${ALLOW_VAR}=<codes> overrides them, for experiments only)")


def filter_pairs(m):
    """Geom pairs that MuJoCo's broadphase lets collide besides the explicit <pair>s (contype / conaffinity, not the
    same weld body, the parent filter unless disabled, not excluded, not already an explicit pair; measured on mujoco
    3.17: an explicit pair replaces only its own geom pair, exclude signatures are body1 << 16 | body2): the contacts
    the engine, which generates only explicit pairs, would leave out."""
    ex = {int(s) for s in m.exclude_signature} if m.nexclude else set()
    explicit = {frozenset((int(a), int(b))) for a, b in zip(m.pair_geom1, m.pair_geom2)}
    filt_parent = not (m.opt.disableflags & int(D.mjDSBL_FILTERPARENT))
    g = [i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
    out = []
    for a in range(len(g)):
        for b in range(a + 1, len(g)):
            g1, g2 = g[a], g[b]
            if not ((m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1])):
                continue
            w1, w2 = int(m.body_weldid[m.geom_bodyid[g1]]), int(m.body_weldid[m.geom_bodyid[g2]])
            if w1 == w2:
                continue
            p1, p2 = int(m.body_weldid[m.body_parentid[w1]]), int(m.body_weldid[m.body_parentid[w2]])
            if filt_parent and w1 and w2 and (p1 == w2 or p2 == w1):
                continue
            b1, b2 = sorted((int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])))
            if (b1 << 16) + b2 in ex or frozenset((g1, g2)) in explicit:
                continue
            out.append((g1, g2))
    return out


def collision_geoms(m):
    return sorted({int(g) for g in m.pair_geom1} | {int(g) for g in m.pair_geom2})


def _pair_codes(m, p):
    a, b = int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]])
    return tuple(sorted((GEOM_CODE.get(a, -1), GEOM_CODE.get(b, -1))))


def _name(m, objtype, i):
    return mujoco.mj_id2name(m, objtype, int(i)) or str(int(i))


def check(m, foot_pairs=None, com_body=COM_BODY, scope="engine"):
    """The CapabilityReport of model m for a build scope ("forward", "engine", "kinematics": the G1 engine's;
    "forest": the forest engine's, check_forest; module docstring). foot_pairs: the contact pairs that take the
    per-world friction (jax_ffi's foot_pairs; None: not checked, and U4 counts the plane pairs); com_body: the body that
    takes the per-world COM offset (None: not checked; the forest scope reads the G1 default torso_link as None)."""
    from nablaengine.g1_engine import step
    if scope not in SCOPES:
        raise ValueError(f"scope {scope!r}: one of {', '.join(SCOPES)}")
    if scope == "forest":
        return check_forest(m, foot_pairs, None if com_body == COM_BODY else com_body)
    want = set(SCOPES[scope])
    out = []

    def add(code, ok, status, what, value=None, scopes=("forward",)):
        out.append(Finding(code, "ok" if ok else status, what, _jsonable(value), tuple(scopes),
                           bool(want & set(scopes))))

    nb, nv, nu = int(m.nbody), int(m.nv), int(m.nu)
    parent = [int(x) for x in m.body_parentid]
    children = [[] for _ in range(nb)]
    for b in range(1, nb):
        children[parent[b]].append(b)
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == J.mjJNT_HINGE]

    add("H1", nb <= 63, "error", "at most 63 bodies (bodies l and l + 32 on lane l beyond 32: model_hooks.py; 64-bit "
        "ancestor masks)", nb,
        ("forward", "kinematics"))
    add("H2", nb <= 63, "error", "at most 63 bodies in the adjoint (lane 31's second body slot carries a root rotation "
        "dof)", nb,
        ("adjoint",))
    add("H3", nv >= 7, "error", "nv >= 7 (a free root and at least one hinge)", nv)
    add("H4", nv <= 64, "error", "nv <= 64 (two rows per lane)", nv)
    add("H5", 33 <= nv <= 35, "warn", "33 <= nv <= 35: the tuned CHOL_BCAST / NEWTON_REUSE / blocked triangular solve "
        "(else the register Cholesky fallback: untested on a corpus, slower)", nv, ("forward", "adjoint"))
    free1 = nb > 1 and m.body_jntnum[1] == 1 and m.jnt_type[m.body_jntadr[1]] == J.mjJNT_FREE
    one_tree = nb > 1 and all(int(m.body_rootid[b]) == 1 for b in range(1, nb))
    add("H6", free1 and children[0] == [1] and one_tree, "error",
        "body 1 has the only joint of its body, a free joint, and is the world's only child (one tree)",
        dict(world_children=children[0], body1_free=bool(free1)), ("forward", "kinematics"))
    bad7 = [b for b in range(2, nb) if not (m.body_jntnum[b] == 1 and m.body_jntadr[b] >= 0
                                            and m.jnt_type[m.body_jntadr[b]] == J.mjJNT_HINGE
                                            and not np.any(m.jnt_pos[m.body_jntadr[b]]))]
    add("H7", not bad7, "error", "every body after the root has one hinge at its origin (no welded, multi-joint, ball "
        "or slide bodies, no hinge anchor offset)", [_name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bad7[:8]],
        ("forward", "kinematics"))
    order = nb > 2 and m.body_dofadr[1] == 0 and m.body_dofnum[1] == 6 and all(
        m.body_dofnum[b] == 1 and m.body_dofadr[b] == b + int(m.body_dofadr[2]) - 2 for b in range(2, nb))
    add("H7b", bool(order), "error", "dofs in body order: the root's 0-5, then body b's hinge at b + const "
        "(adjoint.header's ADJ_HINGE_DOF0)", None, ("adjoint",))
    q0 = [j for j in hinge if m.qpos0[m.jnt_qposadr[j]] != 0]
    add("H8", not q0, "error", "hinge qpos0 = 0 (the engine uses qpos as the angle, MuJoCo qpos - qpos0)",
        {_name(m, mujoco.mjtObj.mjOBJ_JOINT, j): float(m.qpos0[m.jnt_qposadr[j]]) for j in q0[:8]},
        ("forward", "kinematics"))
    nch = max((len(c) for c in children[1:]), default=0)
    add("H9", nch <= 8, "error", "at most 8 children per body (c_child; model_hooks.py beyond 3)", nch)

    def nanc(b):
        k, n = parent[b], 0
        while k >= 2:
            n, k = n + 1, parent[k]
        return n
    na_max = max((nanc(b) for b in range(2, nb)), default=0)
    add("H10", na_max <= 12, "error", "at most 12 hinge ancestors per body (c_anc)", na_max)

    bad11 = []
    for a in range(int(m.nactuator)):
        j = int(m.actuator_trnid[a, 0])
        ok = (m.actuator_trntype[a] == T.mjTRN_JOINT and 0 <= j < m.njnt and m.jnt_type[j] == J.mjJNT_HINGE
              and m.jnt_dofadr[j] == 6 + a and m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
              and m.actuator_gear[a, 0] == 1.0 and not np.any(m.actuator_gear[a, 1:])
              and m.actuator_gainprm[a, 0] == -m.actuator_biasprm[a, 1] and not m.actuator_forcelimited[a]
              and m.actuator_ctrladr[a] == a and m.actuator_ctrlnum[a] == 1 and m.actuator_outadr[a] == a
              and m.actuator_outnum[a] == 1)
        if not ok:
            bad11.append(_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a))
    add("H11a", nu == m.nactuator == m.nout == nv - 6 and not bad11, "error",
        "one position servo per hinge in joint order (actuator a on dof 6 + a, one control and force each), joint "
        "transmission, affine bias, gear 1, gainprm[0] = -biasprm[1], not force-limited",
        dict(nu=nu, nactuator=int(m.nactuator), hinges=nv - 6, bad=bad11[:8]))
    vel = [_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(int(m.nactuator))
           if m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
           and m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and m.actuator_gainprm[a, 0] != 0
           and m.actuator_biasprm[a, 1] == 0 and m.actuator_biasprm[a, 2] == -m.actuator_gainprm[a, 0]]
    add("H11v", not vel, "error", "no velocity servos (MuJoCo <velocity>: gainprm[0] = kv, biasprm = (0, 0, -kv)): "
        "the tree engine simulates position servos only; the forest engine simulates velocity servos, route the model "
        "there with nablaengine.engines.prefer_forest(m) before init (docs/engines.md)", dict(velocity=vel[:8]))
    gt = [_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(int(m.nactuator))
          if m.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED or m.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE
          or m.actuator_biasprm[a, 0] != 0 or np.any(m.actuator_biasprm[a, 3:]) or np.any(m.actuator_gainprm[a, 1:])
          or int(m.actuator_ctrlspec[a]) not in CTRLSPEC_OK]
    add("H11b", not gt and m.na == 0, "error", "fixed gain with one position input (ctrlspec 0 or mjINPUT_POS), no "
        "activation dynamics (na 0), bias (0, -kp, -kv) only", dict(bad=gt[:8], na=int(m.na)))
    clamped = int(np.asarray(m.actuator_ctrllimited).sum()) if not (m.opt.disableflags & int(D.mjDSBL_CLAMPCTRL)) else 0
    add("H11c", clamped == 0, "info", "ctrl is not clamped to ctrlrange by the engine (MuJoCo and MuJoCo Warp clamp "
        "it; the interface's documented semantics)", dict(ctrllimited=clamped))
    asym = [_name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)
            if m.jnt_actfrclimited[j] and m.jnt_actfrcrange[j, 0] != -m.jnt_actfrcrange[j, 1]]
    add("H12", not asym, "error", "symmetric actuator force ranges (jnt_actfrcrange)", asym[:8])
    if com_body is not None:
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, com_body)
        add("H13", tid >= 1, "error", f"the COM-offset body {com_body!r} exists", tid)

    cg = collision_geoms(m)
    types = {GEOM_CODE.get(int(m.geom_type[g]), -1) for g in cg}
    tnames = sorted({mujoco.mjtGeom(int(m.geom_type[g])).name for g in cg})
    add("H14a", types <= FORWARD_TYPES, "error", "contact pair geoms are planes, spheres, capsules, boxes, cylinders or "
        "meshes", tnames)
    planes_off_world = [g for g in cg if m.geom_type[g] == G.mjGEOM_PLANE and m.geom_bodyid[g] != 0]
    box_ok = all(_pair_codes(m, p)[0] in (0, 1, 2) or _pair_codes(m, p) in ((3, 3), (3, 5), (3, 6))
                 for p in range(m.npair) if 3 in _pair_codes(m, p))
    cyl_ok = all(_pair_codes(m, p) in ((0, 5), (1, 5), (2, 5), (5, 5), (3, 5), (5, 6)) for p in range(m.npair)
                 if 5 in _pair_codes(m, p))
    mesh_ok = True
    add("H14b", types - {3, 5, 6} <= ADJOINT_TYPES and (3 not in types or box_ok) and (5 not in types or cyl_ok)
        and (6 not in types or mesh_ok)
        and not planes_off_world, "error",
        "contact pair geoms are planes (on the world body), spheres or capsules in the adjoint, boxes in pairs "
        "with planes, spheres, capsules or boxes (model_hooks.py), cylinders (g1_engine.cylinder, cylccd) and meshes "
        "(g1_engine.mesh, cylccd)", dict(types=tnames, planes_off_world=planes_off_world[:4]), ("adjoint",))
    combos = sorted({_pair_codes(m, p) for p in range(m.npair)} - DISPATCHED)
    add("H14c", not combos, "error", "every pair's type combination is one the forward collides (not plane-plane)",
        combos)
    d_it, d_tol = F.option_default("ccd_iterations"), F.option_default("ccd_tolerance")
    boxbox = sum(1 for p in range(m.npair) if _pair_codes(m, p) == (3, 3))
    add("H14d", not boxbox or (m.opt.ccd_iterations >= d_it and m.opt.ccd_tolerance <= d_tol), "error",
        f"box-box pairs: native CCD at least as converged as MuJoCo's default (ccd_iterations >= {d_it}, ccd_tolerance "
        f"<= {d_tol:g}), whose converged contact the forward computes",
        dict(boxbox_pairs=boxbox, ccd_iterations=int(m.opt.ccd_iterations), ccd_tolerance=float(m.opt.ccd_tolerance)))
    fp = filter_pairs(m)
    add("H15", not fp, "error", "no contype / conaffinity contacts (the engine generates only the explicit pairs)",
        dict(n=len(fp), first=[(_name(m, mujoco.mjtObj.mjOBJ_GEOM, a), _name(m, mujoco.mjtObj.mjOBJ_GEOM, b))
                               for a, b in fp[:4]]))
    mg = [p for p in range(m.npair) if m.pair_margin[p] or m.pair_gap[p]
          or m.pair_friction[p, 0] != m.pair_friction[p, 1]]
    add("H16", not mg, "error", "pair margin = gap = 0 and friction[0] = friction[1]",
        [_name(m, mujoco.mjtObj.mjOBJ_PAIR, p) for p in mg[:8]])
    cd = sorted({int(x) for x in m.pair_dim} - {1, 3, 4})
    add("H17", not cd, "error", "pair condim 1, 3 or 4 (others would be simulated as frictionless)", cd)
    from nablaengine.g1_engine import elliptic as _ell
    ell_ok = _ell.applies(m) and _ell.supported(m)[0]
    add("H18", m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL or ell_ok, "error",
        "pyramidal friction cone, or elliptic with condim 1 / 3 pairs, isotropic sliding friction and no "
        "solreffriction (g1_engine.elliptic: forward and adjoint)",
        mujoco.mjtCone(int(m.opt.cone)).name + ("" if not _ell.applies(m) or ell_ok else f" ({_ell.supported(m)[1]})"))
    neg = dict(pairs=int((m.pair_solref <= 0).any(axis=1).sum()), joints=int((m.jnt_solref[1:] <= 0).any(axis=1).sum()),
               dofs=int((m.dof_solref <= 0).any(axis=1).sum()))
    add("H19", not any(neg.values()), "error", "solref in the standard form, both entries > 0 (time constant, damping "
        "ratio; a direct, non-positive stiffness or damping entry is unsupported)", neg)
    lim_free = [j for j in range(m.njnt) if m.jnt_limited[j] and m.jnt_type[j] != J.mjJNT_HINGE]
    uni = (m.njnt < 2 or ((m.jnt_solref[1:] == m.jnt_solref[1]).all() and (m.jnt_solimp[1:] == m.jnt_solimp[1]).all()))
    dof_uni = nv == 0 or ((m.dof_solref == m.dof_solref[0]).all() and (m.dof_solimp == m.dof_solimp[0]).all())
    add("H20", bool(uni and not m.jnt_margin.any() and dof_uni and not lim_free), "error",
        "one solref / solimp for every joint limit and every dof, jnt_margin 0, only hinge limits",
        dict(limited_non_hinge=lim_free[:4]))
    add("U10", not (m.dof_frictionloss > 0).any(), "warn", "no frictionloss rows (the tree engine's adjoint of them, "
        "g1_engine/model_hooks.py, is not checked against finite differences; the forest engine's is, with per-dof "
        "solreffriction / solimpfriction: scripts/forest/floss_check.py, docs/engines.md)",
        int((m.dof_frictionloss > 0).sum()), ("adjoint",))

    medium = dict(density=float(m.opt.density), viscosity=float(m.opt.viscosity), wind=[float(x) for x in m.opt.wind])
    add("H22", m.opt.density == 0 and m.opt.viscosity == 0 and not np.any(m.opt.wind), "error",
        "no fluid forces (opt density, viscosity, wind 0)", medium)
    add("H23", m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST, "error", "implicitfast integrator",
        mujoco.mjtIntegrator(int(m.opt.integrator)).name)
    add("H24a", m.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON, "error", "Newton solver",
        mujoco.mjtSolver(int(m.opt.solver)).name)
    has_box = any(m.geom_type[g] == G.mjGEOM_BOX for g in cg)
    noop = DISABLE_NOOP + ((D.mjDSBL_FRICTIONLOSS,) if not (m.dof_frictionloss > 0).any() else ()) + (
        (D.mjDSBL_LIMIT,) if not np.asarray(m.jnt_limited).any() else ())
    dis = int(m.opt.disableflags) & ~_mask(DISABLE_OK + noop + (() if has_box else DISABLE_OK_NO_BOX))
    en = int(m.opt.enableflags) & ~_mask(ENABLE_OK)
    add("H24b", dis == 0 and en == 0 and m.opt.noslip_iterations == 0 and m.opt.disableactuator == 0, "error",
        "no disable / enable flags that change the physics, noslip off (the forest engine has the noslip pass: "
        "nablaengine.engines.prefer_forest), no disabled actuator groups",
        dict(disable=_bits(dis, D), enable=_bits(en, E), noslip_iterations=int(m.opt.noslip_iterations),
             disableactuator=int(m.opt.disableactuator)))
    add("H25", m.ntendon == 0 and m.neq == 0 and m.nmocap == 0 and m.nflex == 0 and m.nplugin == 0, "error",
        "no tendons, equality constraints, mocap bodies, flexes or plugins",
        dict(ntendon=int(m.ntendon), neq=int(m.neq), nmocap=int(m.nmocap), nflex=int(m.nflex),
             nplugin=int(m.nplugin)), ("forward", "kinematics"))

    gen = types <= FORWARD_TYPES and not bad7 and nb > 1
    kdof = step.max_support(m) if m.npair and gen else 0
    maxcon = step.maxcon(m)
    npset = len(step.pair_param_sets(m)[0])
    add("H26a", len(cg) <= 255 and npset <= 255 and kdof <= 32 and maxcon <= 64 and m.npair < 65536, "error",
        "<= 255 collision geoms and parameter sets (the pair word), KDOF <= 32, MAXCON <= 64",
        dict(ngeomc=len(cg), param_sets=npset, kdof=kdof, maxcon=maxcon))
    add("H26b", m.npair < 65536, "error", "fewer than 65536 contact pairs (the tape's pair index: 9 bits below 512 "
        "pairs, 16 bits from there, g1_engine.step.capacity_hooks)", int(m.npair), ("adjoint",))
    if foot_pairs is not None:
        fps = [int(p) for p in np.asarray(foot_pairs).ravel()]
        add("H30", set(fps) <= set(range(m.npair)), "error", "the friction (foot) pairs are contact pair indices",
            dict(n=len(fps), outside=sorted(set(fps) - set(range(m.npair)))[:8]))

    sub = np.array(m.body_mass, np.float64)
    for b in range(nb - 1, 0, -1):
        sub[parent[b]] += sub[b]
    add("H31", not np.any(m.body_pos[0]) and np.array_equal(m.body_quat[0], [1.0, 0.0, 0.0, 0.0])
        and m.body_mass[0] == 0 and (nb < 2 or sub[1:].min() >= mujoco.mjMINVAL), "error",
        "kinematics outputs: the world body at the origin and massless, every subtree mass >= mjMINVAL", None,
        ("kinematics",))

    add("U8", nv >= 32, "warn", "nv >= 32 (the G1's; below, the solves' spare lanes repeat the last row: model_hooks.py, "
        "untested on a held-out corpus)", nv)
    add("U9", nch <= 3, "warn", "at most 3 children per body (the G1's; beyond, model_hooks.py widens c_child: untested "
        "on a held-out corpus)", nch)
    add("U1", not (33 <= nv <= 34), "warn", "CHOL_BCAST's tail of nv - 32 rows: only the G1's 3-row tail (nv 35) ran on "
        "a corpus", nv, ("forward", "adjoint"))
    add("U2", not gen or kdof == 16, "warn", "KDOF = 16 (the G1's): otherwise DK_JV_VEC's 128-bit con_jv loads are off and the "
        "scalar path, never run on a corpus, is used", kdof)
    add("U3", maxcon == 64, "warn", "MAXCON 64 in the adjoint (npair > 64; the adjoint never ran with MAXCON 32)",
        maxcon, ("adjoint",))
    worst = 0
    dr = ([int(p) for p in np.asarray(foot_pairs).ravel()] if foot_pairs is not None
          else [p for p in range(m.npair) if 0 in _pair_codes(m, p)])
    for p in dr:
        if 0 <= p < m.npair:
            worst += CONTACTS_PER_PAIR.get(_pair_codes(m, p), 1)
    add("U4", worst <= maxcon, "warn", "the friction (foot) pairs' worst-case contacts fit MAXCON (else contacts are "
        "dropped and the world flagged while standing)", dict(worst_case=worst, maxcon=maxcon))
    floor = step.floor_geom(m) if gen else None
    add("U5", floor is None or floor >= 0 or not m.npair, "warn", "one plane among the collision geoms (FLOOR_G; else "
        "the general plane path, untested on a corpus)", floor)
    add("U6", bool(step.dof_body_root6(m)), "warn", "bodies and dofs in the same order (DOF_BODY_ROOT6; else the general "
        "support mask, untested on a corpus)", bool(step.dof_body_root6(m)))
    pw = sorted({float(np.float32(x)) for x in list(m.pair_solimp[:, 4]) + list(m.jnt_solimp[1:, 4])})
    add("U7", bool(step.solimp_pow2(m)) and len(pw) <= 1, "warn", "every solimp power 2 (SOLIMP_POW2, ADJ_SOLIMP_POWER; "
        "else the general power paths, untested on a corpus)", pw, ("forward", "adjoint"))
    add("I1", True, "info", "the timestep (baked into the constants)", float(m.opt.timestep))

    defaults = F.default_rows()
    for name in F.default_fields():
        f = F.FIELDS[name]
        bad = F.non_default(m, name, defaults)
        if bad:
            kind = name.split("_", 1)[0]
            obj = dict(body=mujoco.mjtObj.mjOBJ_BODY, jnt=mujoco.mjtObj.mjOBJ_JOINT, dof=None,
                       geom=mujoco.mjtObj.mjOBJ_GEOM, pair=mujoco.mjtObj.mjOBJ_PAIR,
                       actuator=mujoco.mjtObj.mjOBJ_ACTUATOR)[kind]
            rows = [(_name(m, obj, i) if obj is not None else int(i)) for i in bad[:6]]
            add(f.code, False, "error", f"{name} at its default ({f.note})", dict(rows=len(bad), first=rows))
    unknown = F.unclassified(m)
    add("F0", not unknown, "error", "every MjModel field is classified (nablaengine.model.fields; a MuJoCo version "
        "with new fields needs their classification)", unknown[:8], ("forward", "adjoint", "kinematics"))
    return CapabilityReport(scope, out)


FOREST_TYPES = {0, 1, 2, 3, 5, 6, 7}
FOREST_CONDIM = {1, 3, 4}


def check_forest(m, foot_pairs=None, com_body=None):
    """The forest engine's CapabilityReport of model m (scope "forest"; nablaengine.forest_engine, docs/engines.md):
    the codes of check() where the requirement is the same (H4, H12, H13, H15, H17, H22, H24a, H25, H30; H25 without the
    equality constraints, which H25e checks, and the fixed tendons, which H25t checks: forest_engine.equality /
    tendon), T1-T14 for the
    forest engine's own requirements on structure, actuators, geoms, integrator, options, contact margins, contact
    supports, solref forms and the elliptic cone, and T0 for an assumption of
    its constant packing (forest_engine.source / step) that the explicit checks do not name. com_body: the body that
    takes the per-world COM offset (None: none, the forest engine's default). Unlike check()'s scopes, physics fields
    the forest engine does not model are checked only where listed here."""
    out = []

    def add(code, ok, what, value=None, status="error"):
        out.append(Finding(code, "ok" if ok else status, what, _jsonable(value), ("forest",), True))

    nb, nv = int(m.nbody), int(m.nv)
    parent = [int(x) for x in m.body_parentid]
    children = [[] for _ in range(nb)]
    for b in range(1, nb):
        children[parent[b]].append(b)

    def body(b):
        return _name(m, mujoco.mjtObj.mjOBJ_BODY, b)

    add("T1", nb <= 128, "at most 128 bodies, the world included (bodies l, l + 32, l + 64, l + 96 on lane l; beyond 32: "
        "forest_engine.nb64, beyond 64: forest_engine.nb128)", nb)
    nch = max((len(c) for c in children[1:]), default=0)
    add("T2", nch <= 8, "at most 8 children per body (c_child)", nch)
    add("H4", nv <= 128, "nv <= 128 (dofs l, l + 32, l + 64, l + 96 on lane l; beyond 64: forest_engine.nv128)", nv)
    bad = {}
    for b in range(1, nb):
        if m.body_jntnum[b] != 1:
            bad[body(b)] = f"{int(m.body_jntnum[b])} joints"
            continue
        j = int(m.body_jntadr[b])
        jt = int(m.jnt_type[j])
        if jt not in (int(J.mjJNT_FREE), int(J.mjJNT_HINGE), int(J.mjJNT_SLIDE)):
            bad[body(b)] = mujoco.mjtJoint(jt).name
        elif jt == int(J.mjJNT_FREE) and parent[b] != 0:
            bad[body(b)] = "a free joint below the tree root"
        elif jt == int(J.mjJNT_FREE) and m.jnt_stiffness[j] != 0:
            bad[body(b)] = "a free-joint spring"
        elif jt != int(J.mjJNT_FREE) and np.any(m.jnt_pos[j]):
            bad[body(b)] = "joint anchor off the body origin"
        elif jt != int(J.mjJNT_FREE) and m.qpos0[m.jnt_qposadr[j]] != 0:
            bad[body(b)] = "joint ref != 0 (the engine's angle is qpos)"
    add("T3", not bad, "every body has exactly one joint: a free joint (tree roots only, no spring) or a hinge / slide "
        "at the body origin with ref 0 (no welded, multi-joint or ball bodies)", dict(list(bad.items())[:8]))

    def nanc(b):
        k, n = parent[b], 0
        while k >= 1:
            if m.body_jntnum[k] == 1 and m.jnt_type[m.body_jntadr[k]] != J.mjJNT_FREE:
                n += 1
            k = parent[k]
        return n
    na_max = max((nanc(b) for b in range(1, nb)), default=0)
    from nablaengine._optional import FOREST, load
    _src = load("nablaengine.forest_engine", FOREST, "source")
    add("T4", na_max <= _src.MAX_ANC, f"at most {_src.MAX_ANC} hinge / slide ancestors per body (c_anc)", na_max)

    bad, seen = [], set()
    from nablaengine._optional import FOREST, load
    _ten = load("nablaengine.forest_engine", FOREST, "tendon")
    tbad_act = {n for n, why in _ten.check_actuators(m)}
    _mus = load("nablaengine.forest_engine", FOREST, "muscle")
    general = _mus.general(m)
    gbad = _mus.check(m) if general else []
    for a in range(int(m.nu)):
        if a in general:
            continue
        j = int(m.actuator_trnid[a, 0])
        affine = int(m.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
        if m.actuator_trntype[a] == T.mjTRN_TENDON:
            if _name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) in tbad_act or \
                    (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or str(a)) in tbad_act:
                bad.append(_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a))
            continue
        joint = m.actuator_trntype[a] == T.mjTRN_JOINT and 0 <= j < m.njnt
        dof = int(m.jnt_dofadr[j]) if joint else -1
        ok = (joint and int(m.jnt_type[j]) in (int(J.mjJNT_HINGE), int(J.mjJNT_SLIDE)) and dof not in seen
              and m.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE
              and m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
              and (affine or int(m.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_NONE))
              and (not affine or m.actuator_biasprm[a, 0] == 0.0)
              and m.actuator_gear[a, 0] == 1.0 and not np.any(m.actuator_gear[a, 1:])
              and (not m.actuator_forcelimited[a] or m.actuator_forcerange[a, 0] == -m.actuator_forcerange[a, 1]))
        seen.add(dof)
        if not ok:
            bad.append(_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a))
    add("T5", not bad, "actuators without activation dynamics or muscle curves: a joint transmission on a hinge or "
        "slide, at most one per dof, or a tendon transmission (fixed: with a support dof free of joint actuators, unless "
        "the model has spatial tendons or general actuators; spatial: forest_engine.spatial); fixed gain, affine "
        "(biasprm[0] = 0) or no bias, gear 1, symmetric forcerange (force = kp ctrl - kq q - kv qdot, q the joint "
        "position or the tendon length); no tendon actfrcrange", dict(bad=bad[:8]))
    add("T12", not gbad, "general actuators (activation dynamics or MuJoCo's muscle: forest_engine.muscle): dyntype "
        "none / integrator / filter / filterexact / muscle (one activation each), gain fixed / affine / muscle, bias "
        "none / affine / muscle, on a hinge / slide joint or a tendon, gear on its first component; with implicitfast "
        "no velocity-dependent gain (the muscle's FV); their tendons' and joints' actfrcrange as T5",
        dict(n=len(general), bad=[f"{n}: {why}" for n, why in gbad[:4]]))
    asym = [_name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)
            if m.jnt_actfrclimited[j] and m.jnt_actfrcrange[j, 0] != -m.jnt_actfrcrange[j, 1]]
    add("H12", not asym, "symmetric actuator force ranges (jnt_actfrcrange)", asym[:8])
    add("T6", not np.asarray(m.jnt_actgravcomp).any(), "gravity compensation as a passive force only (no "
        "actuatorgravcomp)", int(np.asarray(m.jnt_actgravcomp).sum()))
    if com_body is not None:
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, com_body)
        add("H13", tid >= 1, f"the COM-offset body {com_body!r} exists", tid)

    cg = collision_geoms(m)
    tnames = sorted({mujoco.mjtGeom(int(m.geom_type[g])).name for g in cg})
    types = {GEOM_CODE.get(int(m.geom_type[g]), -1) for g in cg}
    add("T7", types <= FOREST_TYPES, "contact pair geoms are planes, spheres, capsules, ellipsoids, cylinders, boxes "
        "or meshes (convex hulls); a heightfield: nablaengine.terrain (Terrain.from_model, TerrainConfig: sphere, capsule "
        "and box geoms against it), or envs.convert.to_engine(drop_types=(mjGEOM_HFIELD,)) where its pairs are "
        "decoration (e.g. a flat heightfield under the floor plane)", tnames)
    fp = filter_pairs(m)
    add("H15", not fp, "no contype / conaffinity contacts (the engine generates only the explicit pairs; "
        "nablaengine.envs.convert.to_engine turns them into pairs)",
        dict(n=len(fp), first=[(_name(m, mujoco.mjtObj.mjOBJ_GEOM, a), _name(m, mujoco.mjtObj.mjOBJ_GEOM, b))
                               for a, b in fp[:4]]))
    cdbad = [f"{_name(m, mujoco.mjtObj.mjOBJ_PAIR, p)}: {int(m.pair_dim[p])}" for p in range(m.npair)
             if int(m.pair_dim[p]) not in FOREST_CONDIM]
    add("H17", not cdbad, "pair condim 1, 3 or 4 (the forest engine has no rolling friction: condim 6 would run as "
        "condim 3; nablaengine.envs.convert.to_engine(condim6_to=3) converts it explicitly)",
        dict(n=len(cdbad), pairs=cdbad[:8]))
    _mg = load("nablaengine.forest_engine", FOREST, "margin")
    mgbad = _mg.problems(m)
    add("T10", not mgbad, "no pair adhesion; contact margins / gaps >= 0 on pair types the forest engine collides "
        "(forest_engine.margin, forest_engine.margin_convex: not two pairs of the same geoms with different margins, "
        "no box-box margin on a G1-class model)", dict(n=len(mgbad), pairs=mgbad[:8]))
    if m.npair and not types - FOREST_TYPES and nb <= 128 and nv <= 128:
        _st = load("nablaengine.forest_engine", FOREST, "step")
        ksup = int(_st.max_support(m))
        _kd = load("nablaengine.forest_engine", FOREST, "kdof64")
        add("T11", ksup <= _kd.MAX_KDOF, "a contact pair's Jacobian support (the dofs on exactly one of its two bodies' "
            f"chains) at most {_kd.MAX_KDOF} dofs (KDOF; beyond 32: forest_engine.kdof64)", ksup)
    fps = [int(p) for p in np.asarray(foot_pairs).ravel()] if foot_pairs is not None else []
    if foot_pairs is not None:
        add("H30", set(fps) <= set(range(m.npair)), "the friction (foot) pairs are contact pair indices",
            dict(n=len(fps), outside=sorted(set(fps) - set(range(m.npair)))[:8]))

    medium = dict(density=float(m.opt.density), viscosity=float(m.opt.viscosity), wind=[float(x) for x in m.opt.wind])
    add("H22", m.opt.density == 0 and m.opt.viscosity == 0 and not np.any(m.opt.wind),
        "no fluid forces (opt density, viscosity, wind 0)", medium)
    add("T8", int(m.opt.integrator) in (int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
                                        int(mujoco.mjtIntegrator.mjINT_EULER)),
        "implicitfast or Euler integrator", mujoco.mjtIntegrator(int(m.opt.integrator)).name)
    add("H24a", m.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON, "Newton solver",
        mujoco.mjtSolver(int(m.opt.solver)).name)
    nsbad = load("nablaengine.forest_engine", FOREST, "noslip").problems(m)
    add("T9", not nsbad and m.opt.disableactuator == 0, "no disabled actuator groups; noslip (opt.noslip_iterations "
        "> 0) only where forest_engine.noslip takes it (pyramidal cone, no equality constraints or tendons, at most 31 "
        "bodies and 64 dofs, islands on)", dict(noslip_iterations=int(m.opt.noslip_iterations),
                                               disableactuator=int(m.opt.disableactuator), noslip=nsbad[:4]))
    tendon = load("nablaengine.forest_engine", FOREST, "tendon")
    tbad = tendon.check(m)
    spatial = [t for t, why in tbad if why.startswith("spatial")]
    add("H25", not spatial and m.nmocap == 0 and m.nflex == 0 and m.nplugin == 0,
        "no mocap bodies, flexes or plugins; spatial tendons (site paths, pulleys, sphere / cylinder wrapping with side "
        "sites: forest_engine.spatial) with a Jacobian support of at most 32 dofs, path bodies with one joint each, "
        "no armature (fixed tendons: H25t; equality constraints: H25e)",
        dict(spatial_tendons=spatial[:8], nmocap=int(m.nmocap), nflex=int(m.nflex), nplugin=int(m.nplugin)))
    add("H25t", not [b for b in tbad if not b[1].startswith("spatial")],
        "tendons: fixed ones over hinge / slide joints with length0 0 (joint refs folded by the converter); limits (at most "
        f"{tendon.MAX_LIMITED} limited tendons, margin >= 0), stiffness with "
        "its deadband, damping, armature, tendon equalities; no tendon frictionloss (forest_engine.tendon; tendon "
        "actuators: T5)", dict(ntendon=int(m.ntendon), bad=[b for b in tbad if not b[1].startswith("spatial")][:8]))
    equality = load("nablaengine.forest_engine", FOREST, "equality")
    eqbad = equality.check(m)
    add("H25e", not eqbad, "equality constraints: joint equalities (mjEQ_JOINT) between hinge / slide joints with ref 0, "
        "connect equalities (mjEQ_CONNECT, body or site objects, active at build, at most 32 dofs on their chains) and "
        "tendon equalities on fixed tendons (mjEQ_TENDON, active at build) "
        "(forest_engine.equality, forest_engine.tendon; no weld or flex equalities yet)",
        dict(neq=int(m.neq), bad=eqbad[:8]))

    _sr = load("nablaengine.forest_engine", FOREST, "solref")
    srbad = _sr.problems(m)
    add("T13", not srbad, "solref in MuJoCo's standard (time constant, damping ratio: both > 0) or direct (-stiffness, "
        "-damping: both <= 0) form on every row the engine builds (contact pairs, joint limits, frictionloss rows, "
        "equalities, tendon limits; forest_engine.solref), not mixed (MuJoCo replaces a mixed solref by the default, "
        "MuJoCo Warp reads each entry on its own); refsafe on (or every time constant >= 2 h); no contact override "
        "(mjENBL_OVERRIDE)", dict(n=len(srbad), bad=srbad[:8]))
    if int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        from nablaengine.g1_engine import elliptic as _ell
        ell_ok, ell_why = _ell.supported(m)
        add("T14", ell_ok, "elliptic cone: condim 1, 3 or 4 pairs, isotropic sliding friction, no solreffriction "
            "(forest_engine.elliptic: the friction rows take the pair's solref)", ell_why)

    if not any(f.status == "error" for f in out):
        import traceback
        adjoint = load("nablaengine.forest_engine", FOREST, "adjoint")
        try:
            adjoint.engine_constants(m, fps, com_body)
            add("T0", True, "the forest engine's constant packing accepts the model")
        except AssertionError as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            add("T0", False, "the forest engine's constant packing accepts the model",
                str(e) or f"{tb.name}: {tb.line}")
    return CapabilityReport("forest", out)


def allowed_codes():
    """The codes $DK_ALLOW_UNSUPPORTED lets through (a set; {"all"} for every code)."""
    v = os.environ.get(ALLOW_VAR, "").strip()
    return {c.strip() for c in v.split(",") if c.strip()}


def require(m, foot_pairs=None, com_body=COM_BODY, scope="engine", context=""):
    """check(), then raise UnsupportedModel if it has errors (except the codes $DK_ALLOW_UNSUPPORTED lets through,
    which become a RuntimeWarning). Returns the report."""
    rep = check(m, foot_pairs, com_body, scope)
    errs = rep.errors()
    if errs:
        allow = allowed_codes()
        kept = [f for f in errs if not ("all" in allow or f.code in allow)]
        if kept:
            raise UnsupportedModel(CapabilityReport(scope, [f for f in rep.findings if f not in errs or f in kept]),
                                   context)
        warnings.warn(f"nablaengine: ${ALLOW_VAR}={os.environ.get(ALLOW_VAR)}: building for a model the engine does "
                      f"not support ({', '.join(sorted({f.code for f in errs}))}): its physics is not MuJoCo's"
                      + (f" ({context})" if context else ""), RuntimeWarning, stacklevel=2)
    return rep


def diff(a, b):
    """Codes whose status or value differ between two reports (e.g. of a model and of a modified copy of it): [{code,
    what, a: (status, value), b: (status, value)}] (codes present in one report only too)."""
    def by_code(r):
        out = {}
        for f in r.findings:
            out.setdefault(f.code, []).append(f)
        return out
    ia, ib = by_code(a), by_code(b)
    res = []
    for code in sorted(set(ia) | set(ib), key=lambda c: (c[0], int("".join(ch for ch in c[1:] if ch.isdigit()) or 0), c)):
        fa, fb = ia.get(code, []), ib.get(code, [])
        sa = [(f.status, f.value) for f in fa]
        sb = [(f.status, f.value) for f in fb]
        if sa != sb:
            res.append(dict(code=code, what=(fa or fb)[0].what, a=sa, b=sb))
    return res


def main(argv=None):
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser(description="The engine's capability report of a MuJoCo model (nablaengine.model."
                                             "capability: the input validation of jax_ffi.init and the checks)")
    ap.add_argument("--model", required=True, help="an .mjb or MJCF (.xml) file")
    ap.add_argument("--foot-pairs", default="", help="comma-separated friction (foot) pair indices, or a g1_meta.json")
    ap.add_argument("--com-body", default=COM_BODY)
    ap.add_argument("--scope", default="engine", choices=sorted(SCOPES))
    ap.add_argument("--all", action="store_true", help="print the passing checks too")
    ap.add_argument("--json", default="", help="write the report here")
    args = ap.parse_args(argv)
    p = Path(args.model)
    m = mujoco.MjModel.from_binary_path(str(p)) if p.suffix == ".mjb" else mujoco.MjModel.from_xml_path(str(p))
    fp = None
    if args.foot_pairs.endswith(".json"):
        fp = json.loads(Path(args.foot_pairs).read_text())["foot_pairs"]
    elif args.foot_pairs:
        fp = [int(x) for x in args.foot_pairs.split(",")]
    rep = check(m, fp, args.com_body, args.scope)
    print(rep.table(args.all) or "(no findings)")
    print(rep.summary())
    if args.json:
        Path(args.json).write_text(json.dumps(rep.to_dict(), indent=1) + "\n")
    return 0 if rep.ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())

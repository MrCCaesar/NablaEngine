# Third-party code: the collision primitives in STEP_DEVICE (plane against sphere / capsule / box, sphere-sphere,
# sphere-capsule, capsule-capsule, sphere-box, capsule-box, box-box) are translated to CUDA from MuJoCo Warp,
# mujoco_warp/_src/collision_primitive_core.py, and make_frame from mujoco_warp/_src/math.py (github.com/google-deepmind/
# mujoco_warp; the version used here: etaoxing/mujoco_warp 357a75d, the head of google-deepmind/mujoco_warp pull
# request 1535), Copyright 2025 The Newton Developers, Apache License 2.0; parts of those follow MuJoCo's C
# implementation (Copyright DeepMind Technologies Limited, Apache License 2.0). Modified: float32 with explicit
# roundings where the adjoint replays them, the pairs spread over a warp's lanes, margins by forest_engine.margin. The constraint rows and
# the Newton solver implement MuJoCo's / MuJoCo Warp's definitions (constraint.py, solver.py) and are this engine's own
# code.
"""CUDA source of one G1 physics substep (block per world): collision, constraints, Newton solver, implicitfast update."""

import math
import os
import warnings

import mujoco
import numpy as np

from nablaengine.forest_engine import convex
from nablaengine import engines as _engines
from nablaengine.model import invweight as _invw

NEFC = 192


MAX_CONTACTS = 255
MAX_ROWS = 1023


def rows_needed(max_contacts, m=None):
    """The constraint rows max_contacts contacts can take, uncapped: 4 rows per contact (condim 3 pyramids; 6 with
    condim-4 pairs in the model m) and the dof rows' reserve (the limits and frictionloss of up to 64 dofs: 64; beyond
    64 dofs, forest_engine.nv128, a frictionloss and a limit row per dof: 2 nv). A row capacity below it can fill
    before the contact capacity does (the rows beyond are dropped, unsupported bit 16)."""
    per = 6 if m is not None and bool((m.pair_dim == 4).any()) else 4
    return per * int(max_contacts) + (2 * m.nv if m is not None and m.nv > 64 else 64)


def nefc(max_contacts, m=None, max_rows=None):
    """Constraint row capacity per world for a contact capacity: max_rows when given, else rows_needed (at least 192)
    up to 1023 (the tape header's 10 bits; rows beyond are dropped, unsupported bit 16). A G1-class model (m None, or
    engines.is_g1_class) without condim-4 pairs keeps 192 rows up to 64 contacts, the tree engine's capacity: its
    contacts (feet, hands) stay well below the 32 that fill 192 rows. Multi-tree models get rows for every contact
    (they once had 192 rows up to 64 contacts too: a G1 in a 9-piece mesh tub ran out of rows at ~48
    contacts and its hand went through the wall)."""
    if max_rows is not None:
        return int(max_rows)
    if m is not None and (m.nv > 64 or bool((m.pair_dim == 4).any())):
        return min(rows_needed(max_contacts, m), MAX_ROWS)
    if max_contacts <= 64 and (m is None or _engines.is_g1_class(m)):
        return NEFC
    return min(max(rows_needed(max_contacts, m), NEFC), MAX_ROWS)

def check_capacity(m, max_contacts=None, max_rows=None, terrain=None):
    """Raise ValueError unless max_contacts (1 .. MAX_CONTACTS) and max_rows (1 .. MAX_ROWS) fit this engine (terrain:
    (TerrainConfig, robot geoms), forest_engine.terrain.check)."""
    if terrain is not None:
        from nablaengine.forest_engine import terrain as _terrain
        _terrain.check(m, terrain)
    if max_contacts is not None and not 1 <= int(max_contacts) <= MAX_CONTACTS:
        raise ValueError(f"max_contacts={max_contacts}: the forest engine holds 1 .. {MAX_CONTACTS} contacts per world")
    if max_rows is not None and not 1 <= int(max_rows) <= MAX_ROWS:
        raise ValueError(f"max_rows={max_rows}: the forest engine holds 1 .. {MAX_ROWS} constraint rows per world")


CON_WORDS = 12
PROF_PHASES = ("smooth", "qacc_smooth", "collision", "contact_jac", "rows", "newton_init", "newton_grad",
               "newton_hessian", "newton_chol", "newton_linesearch", "newton_update", "integrate", "ls_jv_Mp",
               "count_ls_steps",
               "sm_kin", "sm_force", "sm_accum", "sm_crb",
               "ni_matvec", "ni_rowsjv",
               "ng_rows", "ng_jtf_conf", "ng_jtf_dof", "ng_jtf_scatter",
               "ls_rowsjv", "ls_matvec",
               "col_geom", "nh_conC", "nh_copy_dof", "col_broad")
PROF_PARTITION = 12
PAIR_LAYOUT = (
    ("c_pword", "u", 1),
)
PSET_WIDTH = 9
GEOM_LAYOUT = (
    ("c_gbody", "i", 1), ("c_gtype", "i", 1), ("c_gsize", "f", 3), ("c_gpos", "f", 3), ("c_gquat", "f", 4),
    ("c_grbound", "f", 1), ("c_gaabb", "f", 6),
)
SOLVER_LAYOUT = (
    ("c_binvw", "f", "nb"), ("c_dinvw", "f", "nv"), ("c_floss", "f", "nv"), ("c_jrange", "f", "nv2"),
    ("c_jlim", "f", "nv"), ("c_jsolref", "f", 2), ("c_jsolimp", "f", 5), ("c_dsolref", "f", 2), ("c_dsolimp", "f", 5),
    ("c_sopt", "f", 4))
GEOM_TYPE = {int(mujoco.mjtGeom.mjGEOM_PLANE): 0, int(mujoco.mjtGeom.mjGEOM_SPHERE): 1,
             int(mujoco.mjtGeom.mjGEOM_CAPSULE): 2, int(mujoco.mjtGeom.mjGEOM_BOX): 3,
             int(mujoco.mjtGeom.mjGEOM_MESH): convex.MESH,
             int(mujoco.mjtGeom.mjGEOM_CYLINDER): 5,
             int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): 7}


def pair_order(m, g1, g2):
    """The pair's geoms (g1, g2) in MuJoCo's collision-table order (by MuJoCo geom type: plane, sphere, capsule,
    ellipsoid, cylinder, box, mesh), the order the engine's contact code and MuJoCo Warp's collision functions take them
    in (an ellipsoid or a cylinder before a box or a mesh; the engine codes are monotone in MuJoCo's types except the
    cylinder's 5 and the ellipsoid's 7)."""
    return (g2, g1) if int(m.geom_type[g1]) > int(m.geom_type[g2]) else (g1, g2)


def maxcon(m):
    """Contact capacity per world: 32 up to 64 pairs, 64 up to 4096 pairs, 255 beyond (scenes of objects composed
    of boxes: some 90 contacts at rest and up to 239 while the objects land after a reset; at 128 the dropped contacts
    let objects sink, up to 7 cm). $DK_MAXCON overrides it (experiments); it changes every
    build, the FFI library included, so it warns."""
    default = 32 if m.npair <= 64 else 64 if m.npair <= 4096 else 255
    if os.environ.get("DK_MAXCON"):
        warnings.warn(f"DK_MAXCON={os.environ['DK_MAXCON']} overrides the contact capacity (default {default}) of the "
                      "engine builds and the FFI library (an experiment switch)", RuntimeWarning, stacklevel=2)
        return int(os.environ["DK_MAXCON"])
    return default


def max_support(m):
    """Largest number of dofs a contact Jacobian can touch: the dofs on exactly one of the two bodies' chains, over all
    pairs (16 for both G1 models: a hand on the floor, root 6 + waist 3 + arm 7)."""
    dofbody, mask = extra_constants(m)
    return max((sum(1 for i in range(m.nv) if ((mask[b1] >> int(dofbody[i])) ^ (mask[b2] >> int(dofbody[i]))) & 1)
                for b1, b2 in ((m.geom_bodyid[m.pair_geom1[p]], m.geom_bodyid[m.pair_geom2[p]])
                               for p in range(m.npair))), default=0)


def collision_geoms(m):
    """The geoms that appear in a contact pair, in model order (index into the geom table)."""
    return sorted(set(int(g) for g in m.pair_geom1) | set(int(g) for g in m.pair_geom2))


GROUP_PAIRS = 2048
NCAND = 2048


def broad_groups(m):
    """The pair groups of the two-level broad phase (models with more than GROUP_PAIRS pairs, else None): runs of
    consecutive pairs with the same first geom and the same body of the second geom, as (words g1 | body << 8 | plane
    << 16 (g1: geom table index), first pairs (n + 1: the last is NPAIR), body bounding radii about the body frame's
    origin (NB: the largest |local position| + bounding radius of the body's collision geoms))."""
    if m.npair <= GROUP_PAIRS:
        return None
    table = {g: i for i, g in enumerate(collision_geoms(m))}
    words, start = [], []
    for p in range(m.npair):
        g1, b2 = table[int(m.pair_geom1[p])], int(m.geom_bodyid[m.pair_geom2[p]])
        assert b2 != 0, "pair groups: the second geom on a body (explicit pairs: world geoms first)"
        w = g1 | b2 << 8 | int(m.geom_type[m.pair_geom1[p]] == mujoco.mjtGeom.mjGEOM_PLANE) << 16
        if not words or words[-1] != w:
            words.append(w)
            start.append(p)
    start.append(m.npair)
    rad = np.zeros(m.nbody)
    for g in collision_geoms(m):
        b = m.geom_bodyid[g]
        rad[b] = max(rad[b], float(np.linalg.norm(m.geom_pos[g])) + float(m.geom_rbound[g]))
    return words, start, rad


def pair_word(g1, g2, pset, dim, foot, plane=0, frozen=0):
    """A contact pair as one 32-bit word: geom table indices g1, g2 (8 bits each), parameter set (8 bits), condim (4 bits),
    foot-floor flag (1 bit), plane flag (1 bit: g1 is a plane), frozen flag (1 bit: the adjoint has no derivative of
    this geom-type pair, its contacts are frozen there; every pair type has its VJP now: 0); decoded on the device by
    PG1 / PG2 / PSET / PDIM / PFOOT / PPLANE / PFROZEN."""
    assert g1 < 256 and g2 < 256 and pset < 256 and dim < 16
    return g1 | g2 << 8 | pset << 16 | dim << 24 | foot << 28 | plane << 29 | frozen << 30


def pair_param_sets(m):
    """(unique (mu, solref, solimp) rows as an (n, PSET_WIDTH) array, index of each pair's row): the 502 mjlab pairs
    share a handful of parameter sets, so the pair table is one word per pair."""
    keys, index = {}, []
    ell = int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    fl = (lambda x: max(float(x), 1e-5)) if ell else float
    for i in range(m.npair):
        key = (fl(m.pair_friction[i, 0]), *map(float, m.pair_solref[i]), *map(float, m.pair_solimp[i]),
               fl(m.pair_friction[i, 2]) if m.pair_dim[i] >= 4 else 0.0)
        index.append(keys.setdefault(key, len(keys)))
    return np.asarray(list(keys), np.float64).reshape(len(keys), PSET_WIDTH), index


def const_layout(m):
    """[(symbol, kind, length)] of pair_constants(m, ...) in packing order."""
    ng = len(collision_geoms(m))
    sizes = {"nb": m.nbody, "nv": m.nv, "nv2": 2 * m.nv}
    npset = len(pair_param_sets(m)[0])
    return ([(s, k, m.npair * w) for s, k, w in PAIR_LAYOUT] + [("c_pset", "f", npset * PSET_WIDTH)]
            + [(s, k, ng * w) for s, k, w in GEOM_LAYOUT]
            + [(s, k, sizes[w] if isinstance(w, str) else w) for s, k, w in SOLVER_LAYOUT]
            + convex.layout(m, collision_geoms(m))
            + ([("c_ginvw", "f", ng)] if _invw.table(m, collision_geoms(m)) is not None else []))


def pair_constants(m, foot_pairs):
    """Contact-pair arrays (geoms ordered by type, as MuJoCo's collision table expects), the collision-geom table and
    the solver constants, as one float32 array in const_layout(m) order."""
    geoms = collision_geoms(m)
    gid = {g: i for i, g in enumerate(geoms)}
    sets, words = pair_param_sets(m), []
    foot = set(int(x) for x in foot_pairs)
    assert foot <= set(range(m.npair)), f"foot pairs {sorted(foot)}: not all contact pair indices (0..{m.npair - 1})"
    from nablaengine.forest_engine import margin as _margin
    mbad = _margin.problems(m)
    assert not mbad, f"contact margins / gaps: {mbad[:4]}"
    for i in range(m.npair):
        g1, g2 = pair_order(m, int(m.pair_geom1[i]), int(m.pair_geom2[i]))
        assert m.pair_friction[i, 0] == m.pair_friction[i, 1]
        t1, t2 = GEOM_TYPE[int(m.geom_type[g1])], GEOM_TYPE[int(m.geom_type[g2])]
        words.append(pair_word(gid[g1], gid[g2], sets[1][i], int(m.pair_dim[i]), int(i in foot), int(t1 == 0),
                               0))
    words = np.asarray(words, np.int64)
    out = [np.stack([words & 0xFFFF, words >> 16], 1).ravel(), np.asarray(sets[0], np.float64).ravel()]
    out += [m.geom_bodyid[geoms], [GEOM_TYPE[int(m.geom_type[g])] for g in geoms], m.geom_size[geoms],
            m.geom_pos[geoms], m.geom_quat[geoms], m.geom_rbound[geoms], m.geom_aabb[geoms]]
    jr = np.zeros((m.nv, 2))
    jlim = np.zeros(m.nv)
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            jr[m.jnt_dofadr[j]], jlim[m.jnt_dofadr[j]] = m.jnt_range[j], m.jnt_limited[j]
    lj = limit_joints(m)
    assert not m.jnt_margin.any()
    fd = floss_dof0(m)
    if m.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
        from nablaengine.g1_engine import elliptic as _ell
        ok, why = _ell.supported(m)
        assert ok, f"elliptic cone: {why}"
    out += [m.body_invweight0[:, 0], m.dof_invweight0, m.dof_frictionloss, jr, jlim, m.jnt_solref[lj[0]],
            m.jnt_solimp[lj[0]],
            m.dof_solref[fd], m.dof_solimp[fd],
            [m.opt.impratio, m.stat.meaninertia, m.opt.tolerance, m.npair]]
    out += convex.constants(m, geoms)
    tab = _invw.table(m, geoms)
    if tab is not None:
        out += [tab]
    return np.concatenate([np.asarray(x, np.float64).ravel() for x in out]).astype(np.float32)


def limit_joints(m):
    """The joints whose solref / solimp the joint-limit rows use: the limited hinge / slide joints, else [1] (joint 1,
    the G1 class's first hinge; or 0 in a one-joint model). The first one's set is c_jsolref / c_jsolimp; when they
    have more than one set, limit_hooks gives every dof its own (limit_sets)."""
    lim = [j for j in range(m.njnt) if m.jnt_limited[j] and m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    return lim or [min(1, m.njnt - 1)]


def limit_sets(m):
    """Whether the limited hinge / slide joints have more than one solref / solimp set (limit_hooks; one set: the text
    and the constants unchanged)."""
    lj = limit_joints(m)
    return any((m.jnt_solref[j] != m.jnt_solref[lj[0]]).any() or (m.jnt_solimp[j] != m.jnt_solimp[lj[0]]).any()
               for j in lj)


def limit_constants(m):
    """(c_jsolrefv and c_jsolimpv as one float32 array, their layout entries) for limit_hooks' models: each hinge /
    slide dof's joint solref (2) and solimp (5); dofs without a limit row (free joints, unlimited joints) the first
    limited joint's set (unused). None for one set."""
    if not limit_sets(m):
        return None
    lj = limit_joints(m)
    ref, imp = np.tile(m.jnt_solref[lj[0]], (m.nv, 1)), np.tile(m.jnt_solimp[lj[0]], (m.nv, 1))
    for j in lj:
        ref[m.jnt_dofadr[j]], imp[m.jnt_dofadr[j]] = m.jnt_solref[j], m.jnt_solimp[j]
    return (np.concatenate([ref.ravel(), imp.ravel()]).astype(np.float32),
            [("c_jsolrefv", "f", "nv", 2), ("c_jsolimpv", "f", "nv", 5)])


def limit_hooks(m):
    """The (anchor, replacement, count) edits of the engine text for joint limits with more than one solref / solimp
    set (limit_sets; none for one set: the text unchanged): c_jsolrefv / c_jsolimpv (limit_constants), read per dof by
    the forward's limit row (substep_pre) and the adjoint's (adj_limits)."""
    if not limit_sets(m):
        return ()
    return (("    c_dsolref[2], c_dsolimp[5], c_sopt[4];  // impratio, meaninertia, tolerance, npair\n",
             "    c_dsolref[2], c_dsolimp[5], c_sopt[4];  // impratio, meaninertia, tolerance, npair\n"
             "DKC float c_jsolrefv[NV * 2], c_jsolimpv[NV * 5];  // joint-limit solref / solimp per dof "
             "(step.limit_hooks)\n", 1),
            ("        sref = fl ? c_dsolref : c_jsolref; simp = fl ? c_dsolimp : c_jsolimp;\n",
             "        sref = fl ? c_dsolref : c_jsolrefv + 2 * i;  // (step.limit_hooks: per dof)\n"
             "        simp = fl ? c_dsolimp : c_jsolimpv + 5 * i;\n", 1),
            ("      adj_row_impedance(c_jsolref, c_jsolimp, pos, c_dinvw[i], ri);\n",
             "      adj_row_impedance(c_jsolrefv + 2 * i, c_jsolimpv + 5 * i, pos, c_dinvw[i], ri);  // (step.limit_hooks)\n",
             1))


def floss_dof0(m):
    """The dof whose solref / solimp is c_dsolref / c_dsolimp: the first dof with a frictionloss row (0 without one; no
    other row reads dof_solref / dof_solimp, so the other dofs' values are not used)."""
    fl = np.flatnonzero(m.dof_frictionloss > 0)
    return int(fl[0]) if len(fl) else 0


def floss_sets(m):
    """Whether the frictionloss rows' dofs have more than one dof_solref / dof_solimp set (MuJoCo's solreffriction /
    solimpfriction per joint; floss_hooks). One set: the text and the constants unchanged."""
    fl = np.flatnonzero(m.dof_frictionloss > 0)
    return len(fl) > 1 and bool((m.dof_solref[fl] != m.dof_solref[fl[0]]).any()
                                or (m.dof_solimp[fl] != m.dof_solimp[fl[0]]).any())


def floss_constants(m):
    """(c_dsolrefv and c_dsolimpv as one float32 array, their layout entries) for floss_hooks' models: each dof's
    dof_solref (2) and dof_solimp (5) (the dofs without a frictionloss row: the first row's set, unused). None for one
    set."""
    if not floss_sets(m):
        return None
    ref, imp = np.tile(m.dof_solref[floss_dof0(m)], (m.nv, 1)), np.tile(m.dof_solimp[floss_dof0(m)], (m.nv, 1))
    fl = np.flatnonzero(m.dof_frictionloss > 0)
    ref[fl], imp[fl] = m.dof_solref[fl], m.dof_solimp[fl]
    return (np.concatenate([ref.ravel(), imp.ravel()]).astype(np.float32),
            [("c_dsolrefv", "f", "nv", 2), ("c_dsolimpv", "f", "nv", 5)])


def floss_hooks(m):
    """The edits of the engine text for frictionloss rows with more than one solref / solimp set (floss_sets; none for
    one set): c_dsolrefv / c_dsolimpv (floss_constants), read per dof by the forward's frictionloss row (substep_pre)
    and the adjoint's (model_hooks.ENGINE_FLOSS). Composes with limit_hooks (either order)."""
    if not floss_sets(m):
        return ()
    return (("    c_dsolref[2], c_dsolimp[5], c_sopt[4];  // impratio, meaninertia, tolerance, npair\n",
             "    c_dsolref[2], c_dsolimp[5], c_sopt[4];  // impratio, meaninertia, tolerance, npair\n"
             "DKC float c_dsolrefv[NV * 2], c_dsolimpv[NV * 5];  // frictionloss solref / solimp per dof "
             "(step.floss_hooks)\n", 1),
            ("sref = fl ? c_dsolref : ", "sref = fl ? c_dsolrefv + 2 * i : ", 1),
            ("simp = fl ? c_dsolimp : ", "simp = fl ? c_dsolimpv + 5 * i : ", 1),
            ("        adj_row_impedance(c_dsolref, c_dsolimp, 0.f, c_dinvw[i], ri);\n",
             "        adj_row_impedance(c_dsolrefv + 2 * i, c_dsolimpv + 5 * i, 0.f, c_dinvw[i], ri);  // (step.floss_hooks)\n",
             1))


def hess_columns(kdof, passes=3):
    """Support pairs (a, b), b <= a < kdof, of the Hessian assembly per lane: one fixed column b per lane and set, so
    that a lane computes G_b = C J_b once per contact. Set k covers the rows lo_k <= a < lo_(k+1), the largest range
    whose columns (each split over ceil(rows / passes) lanes, rows interleaved) fit 32 lanes. Returns [(lo, hi, table)],
    table[pass][lane] = (a, b, thr), the pair valid when a < nd, i.e. nd > thr (thr = 255: none)."""
    sets, lo = [], 0
    while lo < kdof:
        def lanes(hi):
            return sum(math.ceil((hi - max(lo, b)) / passes) for b in range(hi))
        if lanes(lo + 1) > 32:
            from nablaengine.forest_engine import kdof64
            return sets + kdof64.hess_columns_wide(kdof, lo, passes)
        hi = max(h for h in range(lo + 1, kdof + 1) if lanes(h) <= 32)
        tab = [[(0, 0, 255)] * 32 for _ in range(passes)]
        lane = 0
        for b in range(hi):
            rows = math.ceil((hi - max(lo, b)) / passes)
            for j in range(rows):
                for t in range(passes):
                    a = max(lo, b) + j + rows * t
                    tab[t][lane] = (a, b, a) if a < hi else (b, b, 255)
                lane += 1
        sets.append((lo, hi, tab))
        lo = hi
    return sets


def defer_boxes():
    """$DK_DEFER_BOX (default 0): 1 also the box-box and capsule-box pairs in the split forward's k_ccd (V100: 9-13%
    faster at 16384 worlds; RTX PRO 6000: 5-12% slower on the box scenes: the extra launches and the recomputed smooth
    phase); 0 only the mesh pairs."""
    return os.environ.get("DK_DEFER_BOX", "0") not in ("", "0")


def has_defer(m):
    """Whether the model has pairs the split forward computes in k_ccd (host.py): mesh pairs, and with defer_boxes()
    box-box and capsule-box pairs."""
    for p in range(m.npair):
        t = sorted((GEOM_TYPE[int(m.geom_type[m.pair_geom1[p]])], GEOM_TYPE[int(m.geom_type[m.pair_geom2[p]])]))
        if convex.MESH in t or (defer_boxes() and t in ([3, 3], [2, 3])):
            return True
    return False


def floor_geom(m):
    """Index in the collision-geom table of the model's only plane, or -1 (none or several)."""
    planes = [i for i, g in enumerate(collision_geoms(m)) if GEOM_TYPE[int(m.geom_type[g])] == 0]
    return planes[0] if len(planes) == 1 else -1


def solimp_pow2(m):
    """Whether every solimp power the rows use (contact pairs, joint limits, dof frictionloss) is 2."""
    pw = list(m.pair_solimp[:, 4]) + list(m.jnt_solimp[1:, 4]) + list(m.dof_solimp[m.dof_frictionloss > 0, 4])
    if m.neq or m.ntendon:
        from nablaengine.forest_engine import equality
        pw += equality.solimp_powers(m)
    return all(float(x) == 2.0 for x in pw)


def dof_body_root6(m):
    """True when body 1 (the free root) has dofs 0-5 and every later dof i is the only dof of body i - 4 (bodies and
    dofs in the same order): a contact's support mask is then a shift of its bodies' chain mask (support_mask)."""
    d = np.asarray(m.dof_bodyid)
    return m.nv >= 6 and bool((d[:6] == 1).all() and (d[6:] == np.arange(6, m.nv) - 4).all())


def fixture_groups(m):
    """The model's fixtures with per-world placements (a text element nablaengine_fixtures): [dict(name, geoms: collision-geom table indices, roots: body ids, export)], [] without."""
    import json
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXT, "nablaengine_fixtures")
    if k < 0:
        return []
    raw = bytes(m.text_data[m.text_adr[k]:m.text_adr[k] + m.text_size[k] - 1]).decode()
    table = {g: i for i, g in enumerate(collision_geoms(m))}
    out = []
    for f in json.loads(raw):
        gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in f["geoms"]]
        out.append(dict(name=f["name"], export=f["export"], geoms=[table[g] for g in gid if g in table],
                        roots=[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in f["roots"]]))
        assert all(b > 0 and m.body_parentid[b] == 0 for b in out[-1]["roots"]), "fixture roots: children of the world"
    return out


def com_words(m):
    """Per-world com input words: the COM offset (3), then per fixture its placement relative to the exported one
    (translation 3, quaternion 4; zeros: none)."""
    return 3 + 7 * len(fixture_groups(m))


def _fixtures_header(m):
    fx = fixture_groups(m)
    nb, ng = m.nbody, len(collision_geoms(m))
    fb, fg = [-1] * nb, [-1] * ng
    for i, f in enumerate(fx):
        for b in f["roots"]:
            fb[b] = i
        for g in f["geoms"]:
            fg[g] = i
    h = f"#define NFIX {len(fx)}  // fixtures with per-world placements (step.fixture_groups)\n#define COM_WORDS {3 + 7 * len(fx)}\n"
    if fx:
        h += (f"__device__ const int c_fixb[{nb}] = {{{', '.join(map(str, fb))}}};  // body -> fixture (its parent frame)\n"
              f"__device__ const int c_fixg[{ng}] = {{{', '.join(map(str, fg))}}};  // collision geom -> fixture\n")
    return h


def _groups_header(m):
    from nablaengine.forest_engine import midphase
    if midphase.enabled(m):
        return midphase.header(m)
    bg = broad_groups(m)
    if bg is None:
        return "#define BROAD_GROUPS 0  // the broad phase over every pair (step.broad_groups)\n"
    w, st, rad = bg
    return (f"#define BROAD_GROUPS 1  // the two-level broad phase (step.broad_groups)\n#define NGRP {len(w)}\n"
            f"#define NCAND {min(NCAND, m.npair)}\n"
            f"__device__ const unsigned c_grp[{len(w)}] = {{{', '.join(map(str, w))}}};  // g1 | body << 8 | plane << 16\n"
            f"__device__ const unsigned short c_gstart[{len(st)}] = {{{', '.join(map(str, st))}}};\n"
            f"__device__ const float c_brad[{len(rad)}] = {{{', '.join(repr(float(np.float32(r * (1 + 1e-5) + 1e-6))) + 'f' for r in rad)}}};\n")


def clamp_implicit():
    """(mode, delta) of DK_CLAMP_IMPLICIT / DK_CLAMP_IMPLICIT_DELTA (g1_engine.step.clamp_implicit). The forest engine
    already drops kv at an actuator's forcerange (MuJoCo Warp up to 3.17 too); the switch is about the joint's
    actfrcrange: 0 keeps kv there, 1 drops it (MuJoCo 4.0), 2 fades it (kv w, w = exp(-o / delta), o the relative
    overshoot of the force over the joint's limit; the tape records sign(f) w per dof)."""
    from nablaengine.g1_engine import step as g1_step
    return g1_step.clamp_implicit()


def step_header(m, max_contacts=None, max_rows=None):
    from nablaengine.g1_engine import step as g1_step
    p, g, kdof = m.npair, len(collision_geoms(m)), max(max_support(m), 1)
    hs = hess_columns(kdof)
    hcol = [a | b << 8 | thr << 16 for _, _, tab in hs for row in tab for a, b, thr in row]
    return g1_step.clamp_implicit_defines() + f"""
#define NPAIR {p}
#define NGEOMC {g}
#define MAXCON {max_contacts or maxcon(m)}
#define CPL ((MAXCON + 31) / 32)  // contacts per lane (lane l: contacts l, l + 32, ..)
#define HAS_CONDIM4 {int(bool((m.pair_dim == 4).any()))}  // pairs with torsional friction: 6 pyramid rows, a 4th frame row
#define CD (HAS_CONDIM4 ? 4 : 3)  // frame rows per contact: normal, two tangents (, the normal's angular row)
#define CC (HAS_CONDIM4 ? 7 : 5)  // Hessian coefficients per contact
#define CON_BITS (MAXCON > 127 ? 8 : 7)  // uniform_int's widths of a contact count (at most MAXCON) and a row count
#define KDOF {kdof}
#define HSETS {len(hs)}  // Hessian assembly (hess_columns(KDOF)): sets of 3 passes, their first rows, pairs per lane
#define HSET_LO {{{", ".join(str(lo) for lo, _, _ in hs)}}}
__device__ const unsigned c_hcol[{len(hcol or [0])}] = {{{", ".join(map(str, hcol or [0]))}}};  // a | b << 8 | thr << 16
{_groups_header(m)}{_fixtures_header(m)}#define HAS_BOX {int(any(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in collision_geoms(m)))}  // compile box code
#define BOX_BOX_CCD {0 if m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD else 1}  // the model's box-box: native CCD (MuJoCo Warp's default) or the separating-axis primitive
#define FLOOR_G {floor_geom(m)}  // the geom table index of the only plane (-1: none or several)
{convex.header(m, collision_geoms(m))}#define HAS_DEFER {int(has_defer(m))}  // pairs for the split forward's k_ccd (step.has_defer)
#define DEFER_BOX {int(defer_boxes())}  // box-box and capsule-box pairs among them ($DK_DEFER_BOX)
#define NEFC {nefc(max_contacts or maxcon(m), m, max_rows)}
#define EFC_BITS (NEFC > 255 ? 10 : 8)
#define CON_WORDS 12  // the forward's contact table per contact: pair, dist, pos 3, normal 3, force 3 (world), normal force
#define DOF_BODY_ROOT6 {int(dof_body_root6(m))}  // step.dof_body_root6 (support_mask in closed form)
#define NFLOSS {int((m.dof_frictionloss > 0).sum())}  // dofs with a frictionloss row (the first rows)
#define SOLIMP_POW2 {int(solimp_pow2(m))}  // every solimp power of the model is 2 (kbimp without the general powers)
#define NPROF {len(PROF_PHASES)}
DKC unsigned int c_pword[{max(p, 1)}];  // pair_word(): g1 | g2 << 8 | pset << 16 | dim << 24 | foot << 28 | plane << 29 | frozen << 30
DKC float c_pset[{max(len(pair_param_sets(m)[0]), 1) * PSET_WIDTH}];  // per parameter set: mu, solref[2], solimp[5]
#define PG1(pr) ((int)(c_pword[pr] & 255u))
#define PG2(pr) ((int)((c_pword[pr] >> 8) & 255u))
#define PSET(pr) ((int)((c_pword[pr] >> 16) & 255u))
#define PDIM(pr) ((int)((c_pword[pr] >> 24) & 15u))
#define PFOOT(pr) ((int)((c_pword[pr] >> 28) & 1u))
#define PPLANE(pr) ((int)((c_pword[pr] >> 29) & 1u))
#define PFROZEN(pr) ((int)((c_pword[pr] >> 30) & 1u))
#define PMU(pr) (c_pset[PSET_WIDTH * PSET(pr)])
#define PSOLREF(pr) (c_pset + PSET_WIDTH * PSET(pr) + 1)
#define PSOLIMP(pr) (c_pset + PSET_WIDTH * PSET(pr) + 3)
#define PSET_WIDTH {PSET_WIDTH}
#define PMUS(pr) (c_pset[PSET_WIDTH * PSET(pr) + 8])  // spin friction (condim 4)
DKC int c_gbody[{max(g, 1)}], c_gtype[{max(g, 1)}];  // (sizes >= 1: a model without contact pairs)
DKC float c_gsize[{max(g, 1) * 3}], c_gpos[{max(g, 1) * 3}], c_gquat[{max(g, 1) * 4}], c_grbound[{max(g, 1)}],
    c_gaabb[{max(g, 1) * 6}];
DKC float c_binvw[NB], c_dinvw[NV], c_floss[NV], c_jrange[NV * 2], c_jlim[NV], c_jsolref[2], c_jsolimp[5],
    c_dsolref[2], c_dsolimp[5], c_sopt[4];  // impratio, meaninertia, tolerance, npair
{_invw.header(m, collision_geoms(m))}"""


STEP_DEVICE = r"""
// DK_CLAMP_IMPLICIT (clamp_implicit): the implicit kv of a dof whose force is clamped at its joint's actfrcrange
#ifndef DK_CLAMP_IMPLICIT
#define DK_CLAMP_IMPLICIT 0
#endif
static_assert(DK_CLAMP_IMPLICIT >= 0 && DK_CLAMP_IMPLICIT <= 2, "DK_CLAMP_IMPLICIT: 0, 1 or 2");
// NEWTON_REUSE: a Newton iteration whose rows all kept their zones (a third of them on a 4096-state corpus) re-uses
// the last factor (chol_resolve). Split forward, ms per control step (that corpus, 10 iterations): RTX 3090 3.07
// against 3.34 without (1.66 / 1.77 at 1787 worlds, the waves of 4096 worlds on the target GPUs); V100 2% slower
// (6.34 / 6.20)
#ifndef NEWTON_REUSE
#define NEWTON_REUSE (NV > 32 && NV <= 35)
#endif
// NEWTON_REUSE_WIDE: the same for 36 <= NV <= 48 with chol_solve_bcast_wide's factor (on with CHOL_BCAST_WIDE)
#ifndef NEWTON_REUSE_WIDE
#define NEWTON_REUSE_WIDE CHOL_BCAST_WIDE
#endif
struct Sim {
  float Q[NB][4], X[NB][3];  // body poses
  union {  // the smooth phase's body quantities and the constraint rows are never alive at the same time
    struct {
      float W[NB][3], V[NB][3];
      union {  // the collision's candidate pairs (after the mass matrix)
        struct { float AL[NB][3], A[NB][3], F[NB][3], N[NB][3], J[NB][10]; };
#if BROAD_GROUPS
        struct { unsigned short cand[NCAND], gcand[NGRP]; };
#else
        unsigned short cand[NPAIR];
#endif
      };
    };
    struct {
      unsigned char efc_kind[NEFC], efc_a[NEFC];
      signed char efc_b[NEFC];
      float efc_D[NEFC], efc_aref[NEFC], efc_jar[NEFC], efc_jv[NEFC];  // efc_jv: J p in the line search, else forces
    };
  };
  float M[NTRI], L[NTRI];  // mass matrix and Cholesky scratch, packed lower triangles TRI(i, j)
  float qpos[NQ], qvel[NV], zero[NV], qsm[NV], qacc0[NV], qacc[NV], Ma[NV], grad[NV], p[NV], Mp[NV], tmp[NV];
  unsigned kvoff[(NV + 31) / 32];  // the dofs whose implicit kv is off (actuator forcerange)
#if DK_CLAMP_IMPLICIT == 2
  float kvw[NV];  // the joint-limit weights w of the implicit kv (DK_CLAMP_IMPLICIT=2; 1 unclamped)
#endif
  float con_fo[MAXCON][3];  // the contacts' frame forces (oforce of the last substep)
  float (*oforce)[3];  // null, or con_fo
  alignas(16) float cbuf[CHOL_CB];  // CHOL_SOLVE's scratch (chol_solve_bcast)
  float com[3], fric;
  float fixq[NFIX > 0 ? NFIX : 1][4], fixx[NFIX > 0 ? NFIX : 1][3];  // the fixtures' placements (com words)
  float gx[NGEOMC][3], gq[NGEOMC][4];  // world poses of the collision geoms
  int ncon, nefc, unsupported;
  int min_nu_bits;  // tape: smallest centre / axis distance (before the radii) of a sphere or capsule self contact, as
                    // float bits (non-negative: the integer order is the float order, so atomicMin works); +inf if none
  int con_pair[MAXCON], con_dim[MAXCON], con_row[MAXCON], con_nrow[MAXCON], nefc_dof;
  union {  // per-contact scratch, each consumed right after it is written
    float con_dot[MAXCON][CD];  // J_c,d v (rows_jv)
    float con_f[MAXCON][CD];    // a contact's row forces combined per frame direction (rows_jtf)
    float con_C[MAXCON][CC];    // Hessian coefficients on (n n, n t1, t1 t1, n t2, t2 t2[, n r, r r]), see the Newton loop
  };
  // contact Jacobian rows (normal, two tangents) on the contact's support: the con_nd[c] dofs con_dof[c][k] (ascending)
  float con_dist[MAXCON], con_pos[MAXCON][3], con_frame[MAXCON][9], con_mu[MAXCON];
  alignas(16) float con_J[MAXCON][CD][KDOF];  // (16-byte aligned: con_jv's 128-bit loads)
  alignas(16) unsigned char con_dof[MAXCON][KDOF];
  unsigned char con_slot[MAXCON];  // which contact of its pair (tape): plane-capsule end, parallel capsule-capsule candidate
  int con_nd[MAXCON];
  unsigned long long con_mask[MAXCON];  // the support as dof bits (bit i: dof i; con_dof[c][k] is the k-th set bit)
#ifdef DK_PROFILE
  long long prof[NPROF], prof_t, prof_u;
#endif
#ifdef DK_TAPE_VERIFY
  float Lv[NTRI];  // debug: the factor of the last Newton assembly, compared with a re-assembly at the final zones
#endif
};

// a condition that is the same on every lane (from warp_sum or the like), as a vote: the compiler then knows the branch
// is uniform. Without it the Newton loop counts as divergent control flow and every shuffle and __syncwarp() in it, and
// in the functions it calls (CHOL_SOLVE), gets a divergence check (BRA.DIV / BRA.CONV, and no scheduling across it):
// chol_solve_bcast 8.2k cycles from a uniform caller, 12.9k from a divergent one (V100). The other source of such
// checks is add_contact's shared atomicAdd, whose result is used in the divergent collision code: in k_substep it costs
// the whole kernel its convergence, in the split forward only k_pre. Split forward, ms per control step (a 4096-state
// corpus, 10 iterations, alternating runs): V100 6.34 against 7.27 without (-DDK_UNIFORM=0), RTX 3090 3.07 against
// 3.14
#ifndef DK_UNIFORM
#define DK_UNIFORM 1
#endif
#if DK_UNIFORM
#define UNIFORM(c) __all_sync(0xffffffffu, (c))
#else
#define UNIFORM(c) (c)
#endif
// the same for an integer that is the same on every lane (a count in shared memory): its bits through votes, for loop
// bounds around shuffles and ballots
__device__ __forceinline__ int uniform_int(int x, int bits) {
#if DK_UNIFORM
  int u = 0;
#pragma unroll
  for (int b = 0; b < bits; b++) u |= (int)__any_sync(0xffffffffu, (x >> b) & 1) << b;
  return u;
#else
  return x;
#endif
}

// phase boundary: a __syncwarp() in every build (the code relies on it), plus per-phase cycle counters of one world
// (lane 0) with -DDK_PROFILE; phases: PROF_PHASES in step.py
#ifdef DK_PROFILE
#define PROF_START() do { __syncwarp(); if (lane == 0) s.prof_t = s.prof_u = clock64(); } while (0)
#define PROF(k) do { __syncwarp(); if (lane == 0) { long long t_ = clock64(); s.prof[k] += t_ - s.prof_t; s.prof_t = s.prof_u = t_; } __syncwarp(); } while (0)
#define SUBPROF(k) do { __syncwarp(); if (lane == 0) { long long t_ = clock64(); s.prof[k] += t_ - s.prof_u; s.prof_u = t_; } __syncwarp(); } while (0)
#else
#define SUBPROF(k)
#define PROF_START() __syncwarp()
#define PROF(k) __syncwarp()
#endif

// inlining of the larger helpers: code size (instruction fetch) against call overhead (an A / B switch, e.g.
// -DDK_INL_MATVEC=__noinline__). newton_grad not inlined (one copy instead of two): RTX 3090 1.3% faster, V100 0.2%;
// matvec, total_cost or refresh not inlined: no gain beyond the noise on the RTX 3090
// DK_JTF_GATHER: rows_jtf's contact part J' f as a per-dof gather over the contacts' support masks (con_mask, written
// by substep_pre), in registers and without a __syncwarp() per contact, instead of a scatter of each contact over its
// support (0). Split forward, ms per control step (a 4096-state corpus, 10 iterations, alternating runs, the other
// switches of this group on):
// V100 6.22 against 6.31 for the scatter (1787 worlds 2.60 / 2.64), RTX 3090 -1.3% (1787 worlds -0.7%)
#ifndef DK_JTF_GATHER
#define DK_JTF_GATHER 1
#endif
// DK_JV_VEC: con_jv reads a contact row's Jacobian entries and support dofs from Hand with 128-bit loads (con_J,
// con_dof 16-byte aligned; RTX 3090 -1.2 to -1.5%, V100 -0.4%); DK_GRAD_COST: the line search's new cost is summed
// in newton_grad's |grad| reduction (one interleaved butterfly; -0.1 to -0.2%); DK_HESS_DOFPAR: the Hessian's dof-row
// curvatures added to L by one lane per row (the dof rows' diagonals are distinct when there are no frictionloss rows:
// one limit row per dof) instead of one row at a time with two shuffles each (RTX 3090 -0.6%, V100 -0.3%)
#ifndef DK_JV_VEC
#define DK_JV_VEC 1
#endif
#ifndef DK_GRAD_COST
#define DK_GRAD_COST 1
#endif
#ifndef DK_HESS_DOFPAR
#define DK_HESS_DOFPAR 1
#endif
#ifndef DK_INL_COLLIDE
#define DK_INL_COLLIDE
#endif
#ifndef DK_INL_ROWSJV
#define DK_INL_ROWSJV
#endif
#ifndef DK_INL_GRAD
#define DK_INL_GRAD __noinline__
#endif
#ifndef DK_INL_MATVEC
#define DK_INL_MATVEC __forceinline__
#endif
#ifndef DK_INL_COST
#define DK_INL_COST
#endif

__device__ __forceinline__ void normalize3_rn(float* v) {
  float n = sqrtf(dot3_rn(v, v));
  if (n > 0.f) { v[0] /= n; v[1] /= n; v[2] /= n; }
}

// MuJoCo's orthogonals(a) / make_frame(a): rows n, b, c (explicit roundings, as all of the contact geometry below:
// the adjoint's replay computes the same bits, see source.DEVICE)
__device__ __forceinline__ void make_frame(const float* n, float* fr) {
  float b[3] = {0.f, 0.f, 0.f};
  if (-0.5f < n[1] && n[1] < 0.5f) b[1] = 1.f; else b[2] = 1.f;
  float d = dot3_rn(n, b);
  for (int k = 0; k < 3; k++) b[k] = RFMS(n[k], d, b[k]);
  normalize3_rn(b);
  float c[3];
  cross_rn(n, b, c);
  for (int k = 0; k < 3; k++) { fr[k] = n[k]; fr[3 + k] = b[k]; fr[6 + k] = c[k]; }
}

// slot: which contact of the pair this is (the tape's contact table, tape.py): the end e of a plane-capsule pair
// (0: +axis, 1: -axis), the candidate e = 0..3 of a parallel capsule-capsule pair, a box corner / clip point; else 0.
// (The atomicAdd's result in divergent code: see UNIFORM)
template <class S>
__device__ __forceinline__ bool add_contact(S& s, int pair, int slot, float dist, const float* pos,
                                            const float* frame) {
  int c = atomicAdd(&s.ncon, 1);
  if (c >= MAXCON) return false;
  s.con_pair[c] = pair;
  s.con_slot[c] = (unsigned char)slot;
  s.con_dim[c] = PDIM(pair);
  s.con_dist[c] = dist;
  for (int k = 0; k < 3; k++) s.con_pos[c][k] = pos[k];
  for (int k = 0; k < 9; k++) s.con_frame[c][k] = frame[k];
  s.con_mu[c] = PFOOT(pair) ? s.fric : PMU(pair);
  return true;
}

// a contact from the sphere core (sphere-sphere, sphere-capsule, capsule-capsule): frame from the normal n, and the tape's
// min_nu (nu: the centre or axis distance)
template <class S>
__device__ __forceinline__ void add_sphere_contact(S& s, int pair, int slot, float nu, float dist, const float* pos,
                                                   const float* n) {
  float fr[9];
  make_frame(n, fr);
  if (add_contact(s, pair, slot, dist, pos, fr)) atomicMin(&s.min_nu_bits, __float_as_int(nu));
}

// A pair's contacts collected before they enter a world's table: the narrow phases of the pairs the split forward
// computes in k_ccd (box-box, capsule-box, convex) write into a CSink in every path (one instantiation: the fused and
// the split forward run the same code), then sink_flush (or k_ccd) adds them
#define SINK_MAX 8
struct CSink {
  int n, unsupported;
  int slot[SINK_MAX];
  float dist[SINK_MAX], pos[SINK_MAX][3], fr[SINK_MAX][9];
};
__device__ __forceinline__ bool add_contact(CSink& o, int pair, int slot, float dist, const float* pos,
                                            const float* frame) {
  (void)pair;
  if (o.n >= SINK_MAX) { o.unsupported |= 4; return false; }
  o.slot[o.n] = slot;
  o.dist[o.n] = dist;
  for (int k = 0; k < 3; k++) o.pos[o.n][k] = pos[k];
  for (int k = 0; k < 9; k++) o.fr[o.n][k] = frame[k];
  o.n++;
  return true;
}
template <class S>
__device__ __forceinline__ void flag_unsupported(S& s, int bits) { atomicOr(&s.unsupported, bits); }
__device__ __forceinline__ void flag_unsupported(CSink& o, int bits) { o.unsupported |= bits; }
template <class S>
__device__ __forceinline__ void sink_flush(S& s, int pair, const CSink& o) {
  for (int i = 0; i < o.n; i++) add_contact(s, pair, o.slot[i], o.dist[i], o.pos[i], o.fr[i]);
  if (o.unsupported) atomicOr(&s.unsupported, o.unsupported);
}

// world pose (centre gx, rotation quaternion gq) of collision geom g on its body's pose (Qb, Xb): the forward's
// geom_poses and the adjoint's replay (adjoint.py adj_contacts)
__device__ __forceinline__ void geom_pose(int g, const float* Qb, const float* Xb, float* gx, float* gq) {
  float t[3];
  qrot_rn(Qb, c_gpos + 3 * g, t);
  for (int i = 0; i < 3; i++) gx[i] = RADD(Xb[i], t[i]);
  qmul_rn(Qb, c_gquat + 4 * g, gq);
}

// the broad phase's per-pair test: collide_pair's first test (bounding spheres, a plane against the other geom's sphere)
// with a relative slack of 1e-5, so that it keeps every pair collide_pair keeps; fn: the only plane's normal (FLOOR_G)
template <class S>
__device__ __forceinline__ bool pair_bound(const S& s, int pr, const float* fn) {
  int g1 = PG1(pr), g2 = PG2(pr);
  float dif[3] = {RSUB(s.gx[g2][0], s.gx[g1][0]), RSUB(s.gx[g2][1], s.gx[g1][1]), RSUB(s.gx[g2][2], s.gx[g1][2])};
  if (PPLANE(pr)) {
#if FLOOR_G >= 0
    const float* n = fn;
#else
    float n[3];
    zaxis_rn(s.gq[g1], n);
#endif
    return dot3(dif, n) <= RFMA(c_grbound[g2], 1.f + 1e-5f, 1e-6f);
  }
  float rb = RADD(c_grbound[g1], c_grbound[g2]);
  return dot3(dif, dif) <= RFMA(RMUL(rb, rb), 1.f + 1e-5f, 1e-12f);
}

// world poses of the collision geoms (position, rotation quaternion) from the body poses
template <class S>
__device__ __forceinline__ void geom_poses(S& s, int lane) {
  for (int g = lane; g < NGEOMC; g += 32) {
    int b = c_gbody[g];
#if NFIX
    if (c_fixg[g] >= 0) { geom_pose(g, s.fixq[c_fixg[g]], s.fixx[c_fixg[g]], s.gx[g], s.gq[g]); continue; }
#endif
    geom_pose(g, s.Q[b], s.X[b], s.gx[g], s.gq[g]);
  }
}

// a fixture's placement from its 7 com words (translation, quaternion; an all-zero quaternion: identity)
__device__ __forceinline__ void fix_load(const float* c, float* q, float* x) {
  for (int k = 0; k < 3; k++) x[k] = c[k];
  float n = c[3] * c[3] + c[4] * c[4] + c[5] * c[5] + c[6] * c[6];
  if (n == 0.f) { q[0] = 1.f; q[1] = q[2] = q[3] = 0.f; return; }
  n = rsqrtf(n);
  for (int k = 0; k < 4; k++) q[k] = c[3 + k] * n;
}
template <class S>
__device__ __forceinline__ void load_fixtures(S& s, const float* com, int w, int lane) {
#if NFIX
  for (int i = lane; i < NFIX; i += 32) fix_load(com + (size_t)w * COM_WORDS + 3 + 7 * i, s.fixq[i], s.fixx[i]);
#endif
}

// returns the centre distance nu = |p2 - p1| (the tape's min_nu: the contact normal's derivative is of order 1 / nu)
__device__ __forceinline__ float sphere_sphere(const float* p1, float r1, const float* p2, float r2, float* dist,
                                               float* pos, float* n) {
  float dir[3] = {RSUB(p2[0], p1[0]), RSUB(p2[1], p1[1]), RSUB(p2[2], p1[2])};
  float d = sqrtf(dot3_rn(dir, dir));
  if (d == 0.f) { n[0] = 1.f; n[1] = 0.f; n[2] = 0.f; } else { for (int k = 0; k < 3; k++) n[k] = dir[k] / d; }
  *dist = RSUB(d, RADD(r1, r2));
  float hp = RFMA(0.5f, *dist, r1);
  for (int k = 0; k < 3; k++) pos[k] = RFMA(n[k], hp, p1[k]);
  return d;
}

// The contact geometry of the mjlab model's primitives (MuJoCo Warp's collision_primitive_core), shared by the
// forward's collide_pair and the adjoint's replay (adjoint_contact.adj_contact_geom): the intermediates the VJP needs
// are outputs (the forward drops them).

// a sphere of radius r at c on the plane through p1 with normal n (plane-sphere; plane-capsule: an end of the
// capsule): the distance, and the contact point pos, midway in the penetration
__device__ __forceinline__ float plane_point(const float* p1, const float* n, const float* c, float r, float* pos) {
  float dist = RSUB(RSUB(dot3_rn(c, n), dot3_rn(p1, n)), r), hp = RFMA(0.5f, dist, r);
  for (int k = 0; k < 3; k++) pos[k] = RFMS(n[k], hp, c[k]);
  return dist;
}

// the frame of a plane-capsule contact: rows n, b = the capsule axis projected on the plane and normalized (make_frame's
// basis vector where the projection is shorter than 0.5), n x b; returns the projection's length
__device__ __forceinline__ float plane_capsule_frame(const float* n, const float* axis, float* fr) {
  float na = dot3_rn(n, axis), b[3];
  for (int k = 0; k < 3; k++) b[k] = RFMS(n[k], na, axis[k]);
  float bn = sqrtf(dot3_rn(b, b));
  if (bn > 0.f) for (int k = 0; k < 3; k++) b[k] /= bn;
  if (bn < 0.5f) {
    b[0] = 0.f; b[1] = 0.f; b[2] = 0.f;
    if (-0.5f < n[1] && n[1] < 0.5f) b[1] = 1.f; else b[2] = 1.f;
  }
  float c[3];
  cross_rn(n, b, c);
  for (int k = 0; k < 3; k++) { fr[k] = n[k]; fr[3 + k] = b[k]; fr[6 + k] = c[k]; }
  return bn;
}

// end e of a capsule (centre p, unit axis, half length hl): e = 0 the +axis end, 1 the -axis end
__device__ __forceinline__ void capsule_end(const float* p, const float* axis, float hl, int e, float* ce) {
  for (int k = 0; k < 3; k++) {
    float l = RMUL(axis[k], hl);
    ce[k] = e == 0 ? RADD(p[k], l) : RSUB(p[k], l);
  }
}

// sphere-capsule (MuJoCo Warp's sphere_capsule): the closest point pt = a + t ab of the capsule's segment (centre p2,
// unit axis, half length hl; a = p2 - hl axis, ab = 2 hl axis) to the sphere's centre p1, t = clamp(raw, 0, 1),
// raw = ap . ab / gam with ap = p1 - a and MuJoCo Warp's regularizer gam = ab . ab + 1e-6; returns raw
__device__ __forceinline__ float segment_point(const float* p1, const float* p2, const float* axis, float hl, float* a,
                                               float* ab, float* ap, float* gam, float* t, float* pt) {
  for (int k = 0; k < 3; k++) {
    a[k] = RFMS(axis[k], hl, p2[k]); ab[k] = RMUL(RMUL(2.f, axis[k]), hl); ap[k] = RSUB(p1[k], a[k]);
  }
  *gam = RADD(dot3_rn(ab, ab), 1e-6f);
  float raw = dot3_rn(ap, ab) / *gam;
  *t = fminf(fmaxf(raw, 0.f), 1.f);
  for (int k = 0; k < 3; k++) pt[k] = RFMA(*t, ab[k], a[k]);
  return raw;
}

// capsule-capsule (MuJoCo Warp's capsule_capsule): the segments p_k + x_k a_k, x_k in [-1, 1], with the scaled axes
// a_k = hl_k R_k e_z, dif = p1 - p2, the normal equations [ma mb; mb mc] x = (u, v) and det = ma mc - mb^2
struct CapCap { float a1[3], a2[3], dif[3], ma, mb, mc, u, v, det; };
__device__ __forceinline__ void capcap_setup(const float* p1, const float* q1, float hl1, const float* p2,
                                             const float* q2, float hl2, CapCap& c) {
  zaxis_rn(q1, c.a1);
  zaxis_rn(q2, c.a2);
  for (int k = 0; k < 3; k++) { c.a1[k] = RMUL(c.a1[k], hl1); c.a2[k] = RMUL(c.a2[k], hl2); c.dif[k] = RSUB(p1[k], p2[k]); }
  c.ma = dot3_rn(c.a1, c.a1); c.mb = -dot3_rn(c.a1, c.a2); c.mc = dot3_rn(c.a2, c.a2);
  c.u = -dot3_rn(c.a1, c.dif); c.v = dot3_rn(c.a2, c.dif);
  c.det = RFMA(c.ma, c.mc, -RMUL(c.mb, c.mb));
}
// the segment parameters where |det| >= 1e-15: the unconstrained minimum, then the clamp cascade. f1, f2: x1 / x2 free
// (not fixed at +-1; wp.clamp's gradient rule lo <= raw < hi for the clamp of x1 in the cascade's second stage), raw:
// x1 before that clamp (0 if it did not happen)
__device__ __forceinline__ void capcap_params(const CapCap& c, float* x1o, float* x2o, bool* f1, bool* f2, float* raw) {
  float x1 = RFMA(c.mc, c.u, -RMUL(c.mb, c.v)) / c.det, x2 = RFMA(c.ma, c.v, -RMUL(c.mb, c.u)) / c.det;
  *f1 = *f2 = true;
  *raw = 0.f;
  if (x1 > 1.f) { x1 = 1.f; x2 = RSUB(c.v, c.mb) / c.mc; *f1 = false; }
  else if (x1 < -1.f) { x1 = -1.f; x2 = RADD(c.v, c.mb) / c.mc; *f1 = false; }
  if (x2 > 1.f || x2 < -1.f) {
    bool hi = x2 > 1.f;
    x2 = hi ? 1.f : -1.f;
    *raw = hi ? RSUB(c.u, c.mb) / c.ma : RADD(c.u, c.mb) / c.ma;
    x1 = fminf(fmaxf(*raw, -1.f), 1.f);
    *f1 = *raw >= -1.f && *raw < 1.f;
    *f2 = false;
  }
  *x1o = x1; *x2o = x2;
}
// candidate e = 0..3 of the parallel case (|det| < 1e-15): an end of one segment (e = 0, 1: x1 = +1, -1; 2, 3: x2 =
// +1, -1) and the other's clamped closest parameter (raw before its clamp); f1, f2 as capcap_params
__device__ __forceinline__ void capcap_candidate(const CapCap& c, int e, float* x1, float* x2, bool* f1, bool* f2,
                                                 float* raw) {
  if (e < 2) {
    *x1 = e == 0 ? 1.f : -1.f;
    *raw = e == 0 ? RSUB(c.v, c.mb) / c.mc : RADD(c.v, c.mb) / c.mc;
    *x2 = fminf(fmaxf(*raw, -1.f), 1.f);
    *f1 = false;
    *f2 = *raw >= -1.f && *raw < 1.f;
  } else {
    *x2 = e == 2 ? 1.f : -1.f;
    *raw = e == 2 ? RSUB(c.u, c.mb) / c.ma : RADD(c.u, c.mb) / c.ma;
    *x1 = fminf(fmaxf(*raw, -1.f), 1.f);
    *f2 = false;
    *f1 = *raw >= -1.f && *raw < 1.f;
  }
}
// the sphere core on the closest points p_k + x_k a_k (radii r1, r2): returns nu, as sphere_sphere
__device__ __forceinline__ float capcap_core(const float* p1, const float* p2, const CapCap& c, float x1, float x2,
                                             float r1, float r2, float* dist, float* pos, float* n) {
  float v1[3], v2[3];
  for (int k = 0; k < 3; k++) { v1[k] = RFMA(c.a1[k], x1, p1[k]); v2[k] = RFMA(c.a2[k], x2, p2[k]); }
  return sphere_sphere(v1, r1, v2, r2, dist, pos, n);
}

// 3x3 helpers (row-major float[9]) for the box collisions, MuJoCo Warp's quat_to_mat convention
__device__ __forceinline__ void quat2mat(const float* q, float* R) {
  float q00 = q[0] * q[0], q01 = q[0] * q[1], q02 = q[0] * q[2], q03 = q[0] * q[3], q11 = q[1] * q[1];
  float q12 = q[1] * q[2], q13 = q[1] * q[3], q22 = q[2] * q[2], q23 = q[2] * q[3], q33 = q[3] * q[3];
  R[0] = q00 + q11 - q22 - q33; R[1] = 2.f * (q12 - q03); R[2] = 2.f * (q13 + q02);
  R[3] = 2.f * (q12 + q03); R[4] = q00 - q11 + q22 - q33; R[5] = 2.f * (q23 - q01);
  R[6] = 2.f * (q13 - q02); R[7] = 2.f * (q23 + q01); R[8] = q00 - q11 - q22 + q33;
}
__device__ __forceinline__ void mv3(const float* A, const float* v, float* o) {
  for (int i = 0; i < 3; i++) o[i] = A[3 * i] * v[0] + A[3 * i + 1] * v[1] + A[3 * i + 2] * v[2];
}
__device__ __forceinline__ void mtv3(const float* A, const float* v, float* o) {
  for (int i = 0; i < 3; i++) o[i] = A[i] * v[0] + A[3 + i] * v[1] + A[6 + i] * v[2];
}
__device__ __forceinline__ void mm3(const float* A, const float* B, float* C) {  // C = A B
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
}
__device__ __forceinline__ void mmt3(const float* A, const float* B, float* C) {  // C = A B'
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) C[3 * i + j] = A[3 * i] * B[3 * j] + A[3 * i + 1] * B[3 * j + 1] + A[3 * i + 2] * B[3 * j + 2];
}
__device__ __forceinline__ float safe_div(float x, float y) { return x / (y != 0.f ? y : 1e-15f); }

// sphere-box (MuJoCo Warp's sphere_box): distance, position, normal (from the sphere towards the box)
__device__ void sphere_box(const float* sp, float r, const float* bp, const float* R, const float* bs, float* dist,
                           float* pos, float* n) {
  float d[3] = {sp[0] - bp[0], sp[1] - bp[1], sp[2] - bp[2]}, c[3], cl[3], dir[3], pl[3], nl[3] = {0.f, 0.f, 0.f};
  mtv3(R, d, c);
  for (int k = 0; k < 3; k++) { cl[k] = fmaxf(-bs[k], fminf(bs[k], c[k])); dir[k] = cl[k] - c[k]; }
  float dd = sqrtf(dot3(dir, dir));
  if (dd <= 1e-15f) {
    float closest = 2.f * (bs[0] + bs[1] + bs[2]);
    int kk = 0;
    for (int i = 0; i < 6; i++) {
      float fd = fabsf(((i % 2) ? 1.f : -1.f) * bs[i / 2] - c[i / 2]);
      if (closest > fd) { closest = fd; kk = i; }
    }
    nl[kk / 2] = (kk % 2) ? -1.f : 1.f;
    for (int k = 0; k < 3; k++) pl[k] = c[k] + nl[k] * (r - closest) / 2.f;
    *dist = -closest - r;
  } else {
    for (int k = 0; k < 3; k++) { nl[k] = dir[k] / dd; pl[k] = 0.5f * (cl[k] + c[k] + nl[k] * r); }
    *dist = dd - r;
  }
  mv3(R, nl, n);
  float t[3];
  mv3(R, pl, t);
  for (int k = 0; k < 3; k++) pos[k] = bp[k] + t[k];
}

// capsule-box (MuJoCo Warp's capsule_box_witness + capsule_box): up to two sphere-box contacts along the segment
template <class S>
__device__ void capsule_box(S& s, int pair, const float* cp, const float* axisw, float cr, float hl, const float* bp,
                            const float* R, const float* bs) {
  float d[3] = {cp[0] - bp[0], cp[1] - bp[1], cp[2] - bp[2]}, pos[3], axis[3], ha[3];
  mtv3(R, d, pos);
  mtv3(R, axisw, axis);
  for (int k = 0; k < 3; k++) ha[k] = axis[k] * hl;
  int axisdir = int(ha[0] > 0.f) + 2 * int(ha[1] > 0.f) + 4 * int(ha[2] > 0.f);
  float bestdist = 1e32f, bestseg = -12.f, bestbox = 0.f;
  int cltype = -4, clface = -12, clcorner = -123, cledge = -123;
  for (int i = -1; i < 2; i += 2) {  // a capsule end closest to a face
    float tip[3], bpnt[3];
    int nout = 0, axout = -1;
    for (int j = 0; j < 3; j++) {
      tip[j] = pos[j] + float(i) * ha[j];
      bpnt[j] = tip[j];
      if (bpnt[j] < -bs[j]) { nout++; axout = j; bpnt[j] = -bs[j]; }
      else if (bpnt[j] > bs[j]) { nout++; axout = j; bpnt[j] = bs[j]; }
    }
    if (nout > 1) continue;
    float dist = 0.f;
    for (int j = 0; j < 3; j++) dist += (bpnt[j] - tip[j]) * (bpnt[j] - tip[j]);
    if (dist < bestdist) { bestdist = dist; bestseg = float(i); cltype = -2 + i; clface = axout; }
  }
  for (int i = 0; i < 8; i++)  // a box edge closest
    for (int j = 0; j < 3; j++) {
      if (i & (1 << j)) continue;
      float bpt[3] = {((i & 1) ? 1.f : -1.f) * bs[0], ((i & 2) ? 1.f : -1.f) * bs[1], ((i & 4) ? 1.f : -1.f) * bs[2]};
      bpt[j] = 0.f;
      float dif[3] = {bpt[0] - pos[0], bpt[1] - pos[1], bpt[2] - pos[2]};
      float u = -bs[j] * dif[j], v = dot3(ha, dif), ma = bs[j] * bs[j], mb = -bs[j] * ha[j], mc = hl * hl;
      float det = ma * mc - mb * mb;
      if (fabsf(det) < 1e-15f) continue;
      float idet = 1.f / det, x1 = (mc * u - mb * v) * idet, x2 = (ma * v - mb * u) * idet;
      int s1 = 1, s2 = 1;
      if (x1 > 1.f) { x1 = 1.f; s1 = 2; x2 = safe_div(v - mb, mc); }
      else if (x1 < -1.f) { x1 = -1.f; s1 = 0; x2 = safe_div(v + mb, mc); }
      bool over = x2 > 1.f;
      if (over || x2 < -1.f) {
        if (over) { x2 = 1.f; s2 = 2; x1 = safe_div(u - mb, ma); }
        else { x2 = -1.f; s2 = 0; x1 = safe_div(u + mb, ma); }
        if (x1 > 1.f) { x1 = 1.f; s1 = 2; } else if (x1 < -1.f) { x1 = -1.f; s1 = 0; }
      }
      for (int k = 0; k < 3; k++) dif[k] -= ha[k] * x2;
      dif[j] += bs[j] * x1;
      int ct = s1 * 3 + s2;
      float dsq = dot3(dif, dif);
      if (dsq < bestdist - 1e-15f) {
        bestdist = dsq; bestseg = x2; bestbox = x1;
        clcorner = i + (1 << j) * (ct / 6); cledge = j; cltype = ct;
      }
    }
  if (cltype == -4) return;
  float second = -4.f;
  if (cltype >= 0 && cltype / 3 != 1) {  // closest to a box corner
    int c1 = axisdir ^ clcorner;
    if (c1 != 0 && c1 != 7) {
      int mul, ax = 0, ax1 = 1, ax2 = 2;
      if (c1 == 1 || c1 == 2 || c1 == 4) mul = 1; else { mul = -1; c1 = 7 - c1; }
      if (c1 == 1) { ax = 0; ax1 = 1; ax2 = 2; } else if (c1 == 2) { ax = 1; ax1 = 2; ax2 = 0; }
      else if (c1 == 4) { ax = 2; ax1 = 0; ax2 = 1; }
      if (axis[ax] * axis[ax] > 0.5f) {
        float m = 2.f * safe_div(bs[ax], fabsf(ha[ax]));
        second = fminf(1.f - float(mul) * bestseg, m);
      } else {
        float m = 2.f * fminf(safe_div(bs[ax1], fabsf(ha[ax1])), safe_div(bs[ax2], fabsf(ha[ax2])));
        second = -fminf(1.f + float(mul) * bestseg, m);
      }
      second *= float(mul);
    }
  } else if (cltype >= 0 && cltype / 3 == 1) {  // on a box edge
    int c1 = (axisdir ^ clcorner) & (7 - (1 << cledge));
    if (c1 == 1 || c1 == 2 || c1 == 4) {
      int ax1 = 1, ax2 = 2, ax = cledge, mul;
      if (cledge == 0) { ax1 = 1; ax2 = 2; }
      if (cledge == 1) { ax1 = 2; ax2 = 0; }
      if (cledge == 2) { ax1 = 0; ax2 = 1; }
      if (fabsf(axis[ax1]) > fabsf(axis[ax2])) ax1 = ax2;
      ax2 = 3 - ax - ax1;
      if (c1 & (1 << ax2)) { mul = 1; second = 1.f - bestseg; } else { mul = -1; second = 1.f + bestseg; }
      float e1 = 2.f * safe_div(bs[ax2], fabsf(ha[ax2]));
      second = fminf(e1, second);
      float e2 = (((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0)) ? 1.f - bestbox : 1.f + bestbox;
      e1 = bs[ax] * safe_div(e2, fabsf(ha[ax]));
      second = fminf(e1, second);
      second *= float(mul);
    }
  } else if (clface != -1) {  // a capsule end closest to a face
    int mul = cltype == -3 ? 1 : -1;
    second = 2.f;
    float t1[3];
    for (int k = 0; k < 3; k++) t1[k] = pos[k] - ha[k] * float(mul);
    for (int i = 0; i < 3; i++) {
      if (i == clface) continue;
      float har = safe_div(float(mul), ha[i]);
      float e1 = (bs[i] - t1[i]) * har;
      if (0.f < e1 && e1 < second) second = e1;
      e1 = (-bs[i] - t1[i]) * har;
      if (0.f < e1 && e1 < second) second = e1;
    }
    second *= float(mul);
  }
  for (int e = 0; e < 2; e++) {
    if (e == 1 && !(second > -3.f)) break;
    float seg = e == 0 ? bestseg : bestseg + second, ce[3], dist, pc[3], n[3], fr[9];
    for (int k = 0; k < 3; k++) ce[k] = bp[k];
    float lc[3], t[3];
    for (int k = 0; k < 3; k++) lc[k] = pos[k] + ha[k] * seg;
    mv3(R, lc, t);
    for (int k = 0; k < 3; k++) ce[k] += t[k];
    sphere_box(ce, cr, bp, R, bs, &dist, pc, n);
    if (dist < 0.f) { make_frame(n, fr); add_contact(s, pair, e, dist, pc, fr); }
  }
}

__device__ __forceinline__ void rotmore_of(int f, float* r) {
  for (int k = 0; k < 9; k++) r[k] = 0.f;
  if (f == 0) { r[2] = -1.f; r[4] = 1.f; r[6] = 1.f; }
  else if (f == 1) { r[0] = 1.f; r[5] = -1.f; r[7] = 1.f; }
  else if (f == 2) { r[0] = 1.f; r[4] = 1.f; r[8] = 1.f; }
  else if (f == 3) { r[2] = 1.f; r[4] = 1.f; r[6] = -1.f; }
  else if (f == 4) { r[0] = 1.f; r[5] = 1.f; r[7] = -1.f; }
  else if (f == 5) { r[0] = -1.f; r[4] = 1.f; r[8] = -1.f; }
}

// box-box (MuJoCo Warp's box_box, margin 0): separating-axis test, then face-vertex or edge-edge clipping
#define BB_MAXPT 24
#ifndef BOX_BOX_CCD
#define BOX_BOX_CCD 1  // 1: MuJoCo Warp default (native CCD); 0: the separating-axis primitive (NATIVECCD disabled)
#endif
template <class S>
__device__ void box_box(S& s, int pair, const float* p1, const float* R1, const float* s1, const float* p2,
                        const float* R2, const float* s2) {
  const float margin = 0.f;
  float d[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]}, dn[3] = {-d[0], -d[1], -d[2]}, pos21[3], pos12[3];
  mtv3(R1, d, pos21);
  mtv3(R2, dn, pos12);
  float rot21[9], rot12[9], a21[9], a12[9], plen2[3], plen1[3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) rot21[3 * i + j] = R1[i] * R2[j] + R1[3 + i] * R2[3 + j] + R1[6 + i] * R2[6 + j];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { rot12[3 * i + j] = rot21[3 * j + i]; a21[3 * i + j] = fabsf(rot21[3 * i + j]); }
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) a12[3 * i + j] = a21[3 * j + i];
  mv3(a21, s2, plen2);
  mv3(a12, s1, plen1);
  float sep = margin + 3.f * (s1[0] + s2[0]) + 3.f * (s1[1] + s2[1]) + 3.f * (s1[2] + s2[2]);
  int code = -1;
  for (int i = 0; i < 3; i++) {
    float c1 = -fabsf(pos21[i]) + s1[i] + plen2[i], c2 = -fabsf(pos12[i]) + s2[i] + plen1[i];
    if (c1 < -margin || c2 < -margin) return;
    if (c1 < sep) { sep = c1; code = i + 3 * int(pos21[i] < 0.f); }
    if (c2 < sep) { sep = c2; code = i + 3 * int(pos12[i] < 0.f) + 6; }
  }
  float clnorm[3] = {0.f, 0.f, 0.f};
  bool inv = false;
  int cle1 = 0, cle2 = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      float ca[3];
      if (i == 0) { ca[0] = 0.f; ca[1] = -rot12[3 * j + 2]; ca[2] = rot12[3 * j + 1]; }
      else if (i == 1) { ca[0] = rot12[3 * j + 2]; ca[1] = 0.f; ca[2] = -rot12[3 * j]; }
      else { ca[0] = -rot12[3 * j + 1]; ca[1] = rot12[3 * j]; ca[2] = 0.f; }
      float cl = sqrtf(dot3(ca, ca));
      if (cl < 1e-15f) continue;
      for (int k = 0; k < 3; k++) ca[k] /= cl;
      float bd = dot3(pos21, ca), c3 = 0.f;
      for (int k = 0; k < 3; k++) {
        if (k != i) c3 += s1[k] * fabsf(ca[k]);
        if (k != j) c3 += s2[k] * a21[3 * i + 3 - k - j] / cl;
      }
      c3 -= fabsf(bd);
      if (c3 < -margin) return;
      if (c3 < sep * (1.f - 1e-12f)) {
        sep = c3; cle1 = 0; cle2 = 0;
        for (int k = 0; k < 3; k++) {
          if (k != i && (int(ca[k] > 0.f) ^ int(bd < 0.f))) cle1 += 1 << k;
          if (k != j && (int(rot21[3 * i + 3 - k - j] > 0.f) ^ int(bd < 0.f) ^ int((k - j + 3) % 3 == 1))) cle2 += 1 << k;
        }
        code = 12 + i * 3 + j;
        for (int k = 0; k < 3; k++) clnorm[k] = ca[k];
        inv = bd < 0.f;
      }
    }
  if (code == -1) return;
  float pts[BB_MAXPT][3], depth[BB_MAXPT], rw[9], pw[3], nrm[3], hz;
  int n = 0;
  if (code < 12) {  // face-vertex
    int face = code % 6, bi = code / 6;
    float rm[9], r[9], p[3], ss[3], sv[3], t[3];
    rotmore_of(face, rm);
    mm3(rm, bi ? rot12 : rot21, r);
    mv3(rm, bi ? pos12 : pos21, p);
    mv3(rm, bi ? s2 : s1, t);
    for (int k = 0; k < 3; k++) { ss[k] = fabsf(t[k]); sv[k] = bi ? s1[k] : s2[k]; }
    float lx = ss[0], ly = ss[1];
    hz = ss[2];
    p[2] -= hz;
    int clc = 0;
    for (int i = 0; i < 3; i++) if (r[6 + i] < 0.f) clc += 1 << i;
    float lp[3] = {p[0], p[1], p[2]};
    for (int i = 0; i < 3; i++)
      for (int k = 0; k < 3; k++) lp[k] += r[3 * k + i] * sv[i] * ((clc & (1 << i)) ? 1.f : -1.f);
    int dirs = 0;
    float cn1[3] = {0.f, 0.f, 0.f}, cn2[3] = {0.f, 0.f, 0.f};
    for (int i = 0; i < 3; i++)
      if (fabsf(r[6 + i]) < 0.5f) {
        float* cn = dirs ? cn2 : cn1;
        for (int k = 0; k < 3; k++) cn[k] = r[3 * k + i] * sv[i] * ((clc & (1 << i)) ? -2.f : 2.f);
        dirs++;
      }
    for (int i = 0; i < dirs * dirs; i++)
      for (int q = 0; q < 2; q++) {
        float la[3], lb[3];
        for (int k = 0; k < 3; k++) {
          la[k] = lp[k] + (i < 2 ? 0.f : (i == 2 ? cn1[k] : cn2[k]));
          lb[k] = (i == 0 || i == 3) ? cn1[k] : cn2[k];
        }
        if (fabsf(lb[q]) > 1e-15f) {
          float br = 1.f / lb[q];
          for (int j = -1; j < 2; j += 2) {
            float l = ss[q] * float(j), c1 = (l - la[q]) * br;
            if (c1 < 0.f || c1 > 1.f) continue;
            float c2 = la[1 - q] + lb[1 - q] * c1;
            if (fabsf(c2) > ss[1 - q]) continue;
            if (n < BB_MAXPT) { for (int k = 0; k < 3; k++) pts[n][k] = la[k] + c1 * lb[k]; }
            n++;
          }
        }
      }
    if (dirs == 2) {
      float ax = cn1[0], bx = cn2[0], ay = cn1[1], by = cn2[1], C = safe_div(1.f, ax * by - bx * ay);
      for (int i = 0; i < 4; i++) {
        float llx = (i / 2) ? lx : -lx, lly = (i % 2) ? ly : -ly, x = llx - lp[0], y = lly - lp[1];
        float u = (x * by - y * bx) * C, v = (y * ax - x * ay) * C;
        if (u > 0.f && v > 0.f && u < 1.f && v < 1.f) {
          if (n < BB_MAXPT) { pts[n][0] = llx; pts[n][1] = lly; pts[n][2] = lp[2] + u * cn1[2] + v * cn2[2]; }
          n++;
        }
      }
    }
    for (int i = 0; i < (1 << dirs); i++) {
      float tv[3];
      for (int k = 0; k < 3; k++) tv[k] = lp[k] + float(i & 1) * cn1[k] + float((i & 2) != 0) * cn2[k];
      if (tv[0] > -lx && tv[0] < lx && tv[1] > -ly && tv[1] < ly) {
        if (n < BB_MAXPT) { for (int k = 0; k < 3; k++) pts[n][k] = tv[k]; }
        n++;
      }
    }
    if (n > BB_MAXPT) flag_unsupported(s, 2);
    int m = min(n, BB_MAXPT);
    n = 0;
    for (int i = 0; i < m; i++) {
      if (pts[i][2] > margin) continue;
      for (int k = 0; k < 3; k++) pts[n][k] = pts[i][k];
      depth[n] = pts[n][2];
      pts[n][2] *= 0.5f;
      n++;
    }
    mmt3(bi ? R2 : R1, rm, rw);
    for (int k = 0; k < 3; k++) { pw[k] = bi ? p2[k] : p1[k]; nrm[k] = (bi ? -1.f : 1.f) * rw[3 * k + 2]; }
  } else {  // edge-edge
    int e1 = (code - 12) / 3, e2 = (code - 12) % 3;
    int ax1 = 1 - (e2 & 1), ax2 = 2 - (e2 & 2), pax1 = 1 - (e1 & 1), pax2 = 2 - (e1 & 2);
    if (a21[3 * e1 + ax1] < a21[3 * e1 + ax2]) { int t = ax1; ax1 = ax2; ax2 = t; }
    if (a12[3 * e2 + pax1] < a12[3 * e2 + pax2]) { int t = pax1; pax1 = pax2; pax2 = t; }
    float rm[9], p[3], rn[3], r[9], sv[3], t[3];
    rotmore_of((cle1 & (1 << pax2)) ? pax2 : pax2 + 3, rm);
    mv3(rm, pos21, p);
    mv3(rm, clnorm, rn);
    mm3(rm, rot21, r);
    mtv3(rm, s1, t);
    for (int k = 0; k < 3; k++) sv[k] = fabsf(t[k]);
    float lx = sv[0], ly = sv[1];
    hz = sv[2];
    p[2] -= hz;
    float pu[4][3];
    float g1 = (cle2 & (1 << ax1)) ? 1.f : -1.f, g2 = (cle2 & (1 << ax2)) ? 1.f : -1.f;
    for (int k = 0; k < 3; k++) {
      float b0 = p[k] + r[3 * k + ax1] * s2[ax1] * g1 + r[3 * k + ax2] * s2[ax2] * g2;
      float b2 = p[k] - r[3 * k + ax1] * s2[ax1] * g1 + r[3 * k + ax2] * s2[ax2] * g2;
      float de = r[3 * k + e2] * s2[e2];
      pu[0][k] = b0 + de; pu[1][k] = b0 - de; pu[2][k] = b2 + de; pu[3][k] = b2 - de;
    }
    float alp[3], acn1[3], acn2[3];
    for (int k = 0; k < 3; k++) { alp[k] = pu[0][k]; acn1[k] = pu[1][k] - pu[0][k]; acn2[k] = pu[2][k] - pu[0][k]; }
    if (fabsf(rn[2]) < 1e-15f) return;
    float sg = inv ? -1.f : 1.f, innorm = sg / rn[2], pp[4][3];
    for (int i = 0; i < 4; i++) {
      float cs = pu[i][2] * sg * innorm;
      for (int k = 0; k < 3; k++) pp[i][k] = pu[i][k] - rn[k] * cs;
    }
    float plp[3], pcn1[3], pcn2[3];
    for (int k = 0; k < 3; k++) { plp[k] = pp[0][k]; pcn1[k] = pp[1][k] - pp[0][k]; pcn2[k] = pp[2][k] - pp[0][k]; }
    for (int i = 0; i < 4; i++)
      for (int q = 0; q < 2; q++) {
        float la = plp[q] + (i < 2 ? 0.f : (i == 2 ? pcn1[q] : pcn2[q]));
        float lb = (i == 0 || i == 3) ? pcn1[q] : pcn2[q];
        float lc = plp[1 - q] + (i < 2 ? 0.f : (i == 2 ? pcn1[1 - q] : pcn2[1 - q]));
        float ld = (i == 0 || i == 3) ? pcn1[1 - q] : pcn2[1 - q];
        float lua[3], lub[3];
        for (int k = 0; k < 3; k++) {
          lua[k] = alp[k] + (i < 2 ? 0.f : (i == 2 ? acn1[k] : acn2[k]));
          lub[k] = (i == 0 || i == 3) ? acn1[k] : acn2[k];
        }
        if (fabsf(lb) > 1e-15f) {
          float br = 1.f / lb;
          for (int j = -1; j < 2; j += 2) {
            if (n == 8) break;
            float l = sv[q] * float(j), c1 = (l - la) * br;
            if (c1 < 0.f || c1 > 1.f) continue;
            float c2 = lc + ld * c1;
            if (fabsf(c2) > sv[1 - q]) continue;
            if ((lua[2] + lub[2] * c1) * innorm > margin) continue;
            for (int k = 0; k < 3; k++) pts[n][k] = lua[k] * 0.5f + c1 * lub[k] * 0.5f;
            pts[n][q] += 0.5f * l;
            pts[n][1 - q] += 0.5f * c2;
            depth[n] = pts[n][2] * innorm * 2.f;
            n++;
          }
        }
      }
    int nl = n;
    float ax = pcn1[0], bx = pcn2[0], ay = pcn1[1], by = pcn2[1], C = safe_div(1.f, ax * by - bx * ay);
    for (int i = 0; i < 4; i++) {
      if (n == 8) break;
      float llx = (i / 2) ? lx : -lx, lly = (i % 2) ? ly : -ly, x = llx - plp[0], y = lly - plp[1];
      float u = (x * by - y * bx) * C, v = (y * ax - x * ay) * C;
      if (nl == 0) { if ((u < 0.f || u > 1.f) && (v < 0.f || v > 1.f)) continue; }
      else if (u < 0.f || v < 0.f || u > 1.f || v > 1.f) continue;
      u = fminf(fmaxf(u, 0.f), 1.f);
      v = fminf(fmaxf(v, 0.f), 1.f);
      float w = 1.f - u - v, vt[3], q2[3] = {llx, lly, 0.f};
      for (int k = 0; k < 3; k++) vt[k] = pu[0][k] * w + pu[1][k] * u + pu[2][k] * v;
      float tc1 = 0.f;
      for (int k = 0; k < 3; k++) tc1 += (q2[k] - vt[k]) * (q2[k] - vt[k]);
      if (vt[2] > 0.f && tc1 > margin * margin) continue;
      for (int k = 0; k < 3; k++) pts[n][k] = 0.5f * (q2[k] + vt[k]);
      depth[n] = sqrtf(tc1) * (vt[2] < 0.f ? -1.f : 1.f);
      n++;
    }
    int nf = n;
    for (int i = 0; i < 4; i++) {
      if (n >= 8) break;
      float x = pu[i][0], y = pu[i][1];
      if (nl == 0 && nf != 0) { if ((x < -lx || x > lx) && (y < -ly || y > ly)) continue; }
      else if (x < -lx || x > lx || y < -ly || y > ly) continue;
      float c1 = 0.f;
      for (int j = 0; j < 2; j++) {
        if (pu[i][j] < -sv[j]) c1 += (pu[i][j] + sv[j]) * (pu[i][j] + sv[j]);
        else if (pu[i][j] > sv[j]) c1 += (pu[i][j] - sv[j]) * (pu[i][j] - sv[j]);
      }
      c1 += pu[i][2] * innorm * pu[i][2] * innorm;
      if (pu[i][2] > 0.f && c1 > margin * margin) continue;
      float tp[3] = {pu[i][0], pu[i][1], 0.f};
      for (int j = 0; j < 2; j++) {
        if (pu[i][j] < -sv[j]) tp[j] = -sv[j] * 0.5f;
        else if (pu[i][j] > sv[j]) tp[j] = sv[j] * 0.5f;
      }
      for (int k = 0; k < 3; k++) pts[n][k] = (tp[k] + pu[i][k]) * 0.5f;
      depth[n] = sqrtf(c1) * (pu[i][2] < 0.f ? -1.f : 1.f);
      n++;
    }
    mmt3(R1, rm, rw);
    float t2[3];
    mv3(rw, rn, t2);
    for (int k = 0; k < 3; k++) { pw[k] = p1[k]; nrm[k] = sg * t2[k]; }
  }
  if (n > 8) { flag_unsupported(s, 2); n = 8; }
  {  // no point deeper than the separating-axis overlap sep (the boxes' largest penetration): near-parallel faces
     // can select an edge-edge axis from a nearly vanishing cross product (float32), whose clipping then reports
     // corner distances (4.7 cm) as depths. MuJoCo C gives the face's 4 points there; MuJoCo Warp has the same issue
    const float tol = 1e-4f * fmaxf(fmaxf(fmaxf(s1[0], s1[1]), s1[2]), fmaxf(fmaxf(s2[0], s2[1]), s2[2]));
    int k = 0;
    for (int i = 0; i < n; i++) {
      if (depth[i] < -(sep + tol)) continue;
      if (k != i) { depth[k] = depth[i]; for (int t = 0; t < 3; t++) pts[k][t] = pts[i][t]; }
      k++;
    }
    n = k;
  }
  float fr[9];
  make_frame(nrm, fr);
#if BOX_BOX_CCD
  // MuJoCo Warp collides box-box with GJK/EPA (native CCD): one contact at the minimum-penetration axis, which is the
  // separating-axis result above, unless multiccd finds two faces within 0.0889 rad of facing each other or an edge
  // within that angle of lying on a face; those near-aligned cases are flagged (bit 8), not emulated
  const float ftol = 0.99605107f, etol = 0.08878297f;  // cos(0.0889), sin(0.0889)
  bool multi = false;
  if (code < 12) {  // face of one box against a vertex of the other: any axis of the vertex box near-perpendicular to n
    const float* Rv = (code / 6) ? R1 : R2;
    for (int k = 0; k < 3; k++) {
      float c = fabsf(Rv[k] * nrm[0] + Rv[3 + k] * nrm[1] + Rv[6 + k] * nrm[2]);
      if (c > ftol || c < etol) multi = true;
    }
    int best = -1;
    for (int i = 0; i < n; i++) if (depth[i] < 0.f && (best < 0 || depth[i] < depth[best])) best = i;
    if (multi) flag_unsupported(s, 8);
    if (best < 0) return;
    float pl[3] = {pts[best][0], pts[best][1], pts[best][2] + hz}, pc[3];
    mv3(rw, pl, pc);
    for (int k = 0; k < 3; k++) pc[k] += pw[k];
    add_contact(s, pair, 0, depth[best], pc, fr);
  } else {  // edge-edge: the midpoint of the two edges' closest points
    int e1 = (code - 12) / 3, e2 = (code - 12) % 3;
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        float c = fabsf(R1[a] * R2[b] + R1[3 + a] * R2[3 + b] + R1[6 + a] * R2[6 + b]);
        if (a != e1 && b != e2 && c > ftol) multi = true;      // adjacent faces facing each other
        if (a == e1 && b != e2 && c < etol) multi = true;      // edge 1 lying on a face of box 2
        if (b == e2 && a != e1 && c < etol) multi = true;      // edge 2 lying on a face of box 1
      }
    if (multi) flag_unsupported(s, 8);
    float l1[3], l2[3], c1[3], c2[3], d1[3], d2[3];
    for (int k = 0; k < 3; k++) {
      l1[k] = k == e1 ? 0.f : ((cle1 & (1 << k)) ? s1[k] : -s1[k]);
      l2[k] = k == e2 ? 0.f : ((cle2 & (1 << k)) ? s2[k] : -s2[k]);
    }
    mv3(R1, l1, c1);
    mv3(R2, l2, c2);
    for (int k = 0; k < 3; k++) {
      c1[k] += p1[k]; c2[k] += p2[k];
      d1[k] = R1[3 * k + e1] * s1[e1]; d2[k] = R2[3 * k + e2] * s2[e2];
    }
    // closest points of the segments c1 + t1 d1, c2 + t2 d2 with t in [-1, 1]
    float r[3] = {c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2]};
    float a11 = dot3(d1, d1), a12 = dot3(d1, d2), a22 = dot3(d2, d2), b1 = dot3(d1, r), b2 = dot3(d2, r);
    float det = a11 * a22 - a12 * a12, t1 = 0.f, t2 = 0.f;
    if (det > 1e-12f) t1 = fminf(fmaxf((a12 * b2 - a22 * b1) / det, -1.f), 1.f);
    t2 = fminf(fmaxf((b2 + a12 * t1) / a22, -1.f), 1.f);
    t1 = fminf(fmaxf((a12 * t2 - b1) / a11, -1.f), 1.f);
    float pc[3];
    for (int k = 0; k < 3; k++) pc[k] = 0.5f * (c1[k] + t1 * d1[k] + c2[k] + t2 * d2[k]);
    if (sep > 0.f) add_contact(s, pair, 0, -sep, pc, fr);
  }
#else
  for (int i = 0; i < n; i++) {
    if (!(depth[i] < 0.f)) continue;
    float pl[3] = {pts[i][0], pts[i][1], pts[i][2] + hz}, pc[3];
    mv3(rw, pl, pc);
    for (int k = 0; k < 3; k++) pc[k] += pw[k];
    add_contact(s, pair, i, depth[i], pc, fr);
  }
#endif
}

#if HAS_BOX
// capsule-box (t1 2) or box-box (t1 3) into o: one non-template function (the fused and the split forward's k_ccd)
__device__ __noinline__ void box_pair(int g1, int t1, const float* p1, const float* q1, int g2, const float* p2,
                                      const float* q2, CSink& o) {
  o.n = 0; o.unsupported = 0;
  float R2[9];
  quat2mat(q2, R2);
  if (t1 == 2) {
    float a1[3];
    zaxis_rn(q1, a1);
    capsule_box(o, 0, p1, a1, c_gsize[3 * g1], c_gsize[3 * g1 + 1], p2, R2, c_gsize + 3 * g2);
  } else {
    float R1[9];
    quat2mat(q1, R1);
    box_box(o, 0, p1, R1, c_gsize + 3 * g1, p2, R2, c_gsize + 3 * g2);
  }
}
#endif
@@OBB_DEVICE@@
@@CONVEX_DEVICE@@
// the split forward's k_pre_a queues a convex pair for k_ccd (host.py: its overload for SimP); else computed here
template <class S>
__device__ __forceinline__ bool cv_defer(S&, int) { return false; }

// the contacts in canonical order: by (pair, slot) (unique keys), so that the order does not depend on the order in
// which they were found (k_ccd's atomics); lane l moves its contacts l, l + 32, .. through registers
template <class S>
__device__ void sort_contacts(S& s, int lane) {
  const int n = min(s.ncon, MAXCON);
  int key[CPL], rank[CPL], pr[CPL], dm[CPL];
  unsigned char sl[CPL];
  float mu[CPL], ds[CPL], ps[CPL][3], fr[CPL][9];
#pragma unroll
  for (int h = 0; h < CPL; h++) {
    const int c = lane + 32 * h;
    key[h] = c < n ? (s.con_pair[c] << 5 | s.con_slot[c]) : 0x7fffffff;
    rank[h] = 0;
    if (c < n) {
      pr[h] = s.con_pair[c]; sl[h] = s.con_slot[c]; dm[h] = s.con_dim[c]; mu[h] = s.con_mu[c]; ds[h] = s.con_dist[c];
      for (int k = 0; k < 3; k++) ps[h][k] = s.con_pos[c][k];
      for (int k = 0; k < 9; k++) fr[h][k] = s.con_frame[c][k];
    }
  }
  for (int j = 0; j < n; j++) {
    const int kj = s.con_pair[j] << 5 | s.con_slot[j];
#pragma unroll
    for (int h = 0; h < CPL; h++) rank[h] += (kj < key[h] || (kj == key[h] && j < lane + 32 * h)) ? 1 : 0;
  }
  __syncwarp();
#pragma unroll
  for (int h = 0; h < CPL; h++) {
    if (lane + 32 * h < n) {
      const int r = rank[h];
      s.con_pair[r] = pr[h]; s.con_slot[r] = sl[h]; s.con_dim[r] = dm[h]; s.con_mu[r] = mu[h]; s.con_dist[r] = ds[h];
      for (int k = 0; k < 3; k++) s.con_pos[r][k] = ps[h][k];
      for (int k = 0; k < 9; k++) s.con_frame[r][k] = fr[h][k];
    }
  }
  __syncwarp();
}

template <class S>
__device__ DK_INL_COLLIDE void collide_pair(S& s, int pair) {
  int g1 = PG1(pair), g2 = PG2(pair), t1 = c_gtype[g1], t2 = c_gtype[g2];
  float p1[3], q1[4], p2[3], q2[4];
  for (int k = 0; k < 3; k++) { p1[k] = s.gx[g1][k]; p2[k] = s.gx[g2][k]; }
  for (int k = 0; k < 4; k++) { q1[k] = s.gq[g1][k]; q2[k] = s.gq[g2][k]; }
  const float* sz1 = c_gsize + 3 * g1;
  const float* sz2 = c_gsize + 3 * g2;
  if (t1 == 0) {  // plane (the floor): normal = its z axis
    float n[3];
    zaxis_rn(q1, n);
    float dif[3] = {RSUB(p2[0], p1[0]), RSUB(p2[1], p1[1]), RSUB(p2[2], p1[2])};
    if (dot3_rn(dif, n) > c_grbound[g2]) return;  // the other geom's bounding sphere is above the plane
    if (t2 == 1) {
      float pos[3], fr[9];
      float dist = plane_point(p1, n, p2, sz2[0], pos);
      make_frame(n, fr);
      if (dist < 0.f) add_contact(s, pair, 0, dist, pos, fr);
    } else if (t2 == 2) {
      float axis[3], fr[9];
      zaxis_rn(q2, axis);
      plane_capsule_frame(n, axis, fr);
      for (int e = 0; e < 2; e++) {
        float ce[3], pos[3];
        capsule_end(p2, axis, sz2[1], e, ce);
        float dist = plane_point(p1, n, ce, sz2[0], pos);
        if (dist < 0.f) add_contact(s, pair, e, dist, pos, fr);
      }
    }
#if HAS_BOX
    else if (t2 == 3) {
      float fr[9], cd = dot3(p2, n) - dot3(p1, n);
      make_frame(n, fr);
      for (int i = 0; i < 8; i++) {
        float cl[3] = {(i & 1) ? sz2[0] : -sz2[0], (i & 2) ? sz2[1] : -sz2[1], (i & 4) ? sz2[2] : -sz2[2]}, cw[3];
        qrot(q2, cl, cw);
        float dist = cd + dot3(n, cw), pos[3];
        for (int k = 0; k < 3; k++) pos[k] = cw[k] + p2[k] - 0.5f * n[k] * dist;
        if (dist < 0.f) add_contact(s, pair, i, dist, pos, fr);
      }
    }
#endif
#if HAS_MESH
    else if (t2 == 4 && !cv_defer(s, pair)) {
      CSink o;
      CV_PLANE(p1, n, g2, p2, q2, o);
      sink_flush(s, pair, o);
    }
#endif
    return;
  }
  {  // bounding spheres (geom_rbound) do not overlap: no contact at zero margin
    float dif[3] = {RSUB(p2[0], p1[0]), RSUB(p2[1], p1[1]), RSUB(p2[2], p1[2])}, rb = RADD(c_grbound[g1], c_grbound[g2]);
    if (dot3_rn(dif, dif) > RMUL(rb, rb)) return;
  }
  if (t1 == 1 && t2 == 1) {  // sphere-sphere
    float dist, pos[3], n[3];
    float nu = sphere_sphere(p1, sz1[0], p2, sz2[0], &dist, pos, n);
    if (dist < 0.f) add_sphere_contact(s, pair, 0, nu, dist, pos, n);
    return;
  }
  if (t1 == 1 && t2 == 2) {  // sphere-capsule (MuJoCo Warp's sphere_capsule: closest point of the segment)
    float axis[3], a[3], ab[3], ap[3], gam, t, pt[3], dist, pos[3], n[3];
    zaxis_rn(q2, axis);
    segment_point(p1, p2, axis, sz2[1], a, ab, ap, &gam, &t, pt);
    float nu = sphere_sphere(p1, sz1[0], pt, sz2[0], &dist, pos, n);
    if (dist < 0.f) add_sphere_contact(s, pair, 0, nu, dist, pos, n);
    return;
  }
  if (t1 == 2 && t2 == 2) {  // capsule-capsule
    CapCap cc;
    capcap_setup(p1, q1, sz1[1], p2, q2, sz2[1], cc);
    float dist, pos[3], n[3], x1, x2, raw;
    bool f1, f2;
    if (fabsf(cc.det) >= 1e-15f) {
      capcap_params(cc, &x1, &x2, &f1, &f2, &raw);
      float nu = capcap_core(p1, p2, cc, x1, x2, sz1[0], sz2[0], &dist, pos, n);
      if (dist < 0.f) add_sphere_contact(s, pair, 0, nu, dist, pos, n);
    } else {
      int cnt = 0;
      for (int e = 0; e < 4 && cnt < 2; e++) {
        capcap_candidate(cc, e, &x1, &x2, &f1, &f2, &raw);
        float nu = capcap_core(p1, p2, cc, x1, x2, sz1[0], sz2[0], &dist, pos, n);
        if (dist <= 0.f) {
          cnt++;
          if (dist < 0.f) add_sphere_contact(s, pair, e, nu, dist, pos, n);
        }
      }
    }
    return;
  }
#if HAS_BOX
  if ((t1 == 2 || t1 == 3) && t2 == 3) {  // capsule-box, box-box: past the OBB test, in k_ccd (the split forward)
    if (obb_separated(g1, p1, q1, g2, p2, q2) || (DEFER_BOX && cv_defer(s, pair))) return;
    CSink o;
    box_pair(g1, t1, p1, q1, g2, p2, q2, o);
    sink_flush(s, pair, o);
    return;
  }
#endif
#if HAS_MESH
  if (t2 == 4) {  // a convex mesh against a sphere, capsule, box or mesh (forest_engine.convex), past an OBB test
    if (obb_separated(g1, p1, q1, g2, p2, q2) || cv_defer(s, pair)) return;  // (the split forward: k_ccd)
    CSink o;
    CV_PAIR(g1, t1, p1, q1, g2, t2, p2, q2, o);
    sink_flush(s, pair, o);
    return;
  }
#endif
  atomicOr(&s.unsupported, 1);  // other type combinations do not occur in the G1 model
}

// x^p for the solimp power: the integer powers 1 and 2 (both G1 models use 2) without powf (x * x is the correctly
// rounded square, powf(x, 2) is within 1 ulp of it)
__device__ __forceinline__ float pow_imp(float x, float p) { return p == 2.f ? RMUL(x, x) : (p == 1.f ? x : powf(x, p)); }

// the impedance curve's raw (unclamped) value for 0 < x < 1 and a power p != 1, on the branch lo = (x <= mid): the
// forward's kbimp and the adjoint's kbimp_d (its clamp test) compute the same bits
__device__ __forceinline__ float imp_raw(float x, float dmin, float dmax, float mid, float p, bool lo) {
  return lo ? RFMA(pow_imp(x, p) / pow_imp(mid, p - 1.f), RSUB(dmax, dmin), dmin)
            : RFMA(RSUB(1.f, pow_imp(RSUB(1.f, x), p) / pow_imp(RSUB(1.f, mid), p - 1.f)), RSUB(dmax, dmin), dmin);
}

// impedance (k, b, imp) at pos_imp with refsafe (timeconst >= 2 h); explicit roundings: the adjoint's kbimp_d calls it
__device__ __forceinline__ void kbimp(const float* solref, const float* solimp, float pos_imp, float h, float* k,
                                      float* b, float* imp) {
  float tc = fmaxf(solref[0], RMUL(2.f, h)), dr = solref[1];
  float dmin = fminf(fmaxf(solimp[0], 1e-4f), 0.9999f), dmax = fminf(fmaxf(solimp[1], 1e-4f), 0.9999f);
  float width = fmaxf(1e-15f, solimp[2]), mid = fminf(fmaxf(solimp[3], 1e-4f), 0.9999f);
  const float power = SOLIMP_POW2 ? 2.f : fmaxf(1.f, solimp[4]);  // a constant: pow_imp folds to x * x and mid
  *k = 1.f / fmaxf(1e-15f, RMUL(RMUL(RMUL(RMUL(RMUL(dmax, dmax), tc), tc), dr), dr));
  *b = 2.f / fmaxf(1e-15f, RMUL(dmax, tc));
  float x = fabsf(pos_imp) / width;
  if (dmin == dmax || solimp[2] <= 1e-15f) *imp = RMUL(0.5f, RADD(dmin, dmax));
  else if (x <= 0.f) *imp = dmin;
  else if (x >= 1.f) *imp = dmax;
  else if (power == 1.f) *imp = RFMA(x, RSUB(dmax, dmin), dmin);
  else *imp = fminf(fmaxf(imp_raw(x, dmin, dmax, mid, power, x <= mid), dmin), dmax);
}

// the distances of hinge dof i at angle q to its lower and upper limit (the limit rows' test and pos: one IEEE
// subtraction each; the adjoint's adj_limits replays the test)
__device__ __forceinline__ void limit_dists(int i, float q, float* dmin, float* dmax) {
  *dmin = RSUB(q, c_jrange[2 * i]);
  *dmax = RSUB(c_jrange[2 * i + 1], q);
}

// the invweight of a contact of pair pr (friction mu): body 1's + body 2's, for condim 3 MuJoCo's pyramid scaling
__device__ __forceinline__ float contact_invw(int pr, float mu) {
  float iw = RADD(c_binvw[c_gbody[PG1(pr)]], c_binvw[c_gbody[PG2(pr)]]);
  if (PDIM(pr) >= 3) iw = RMUL(RMUL(RMUL(RFMA(RMUL(mu, mu), iw, iw), 2.f), mu), mu) / c_sopt[0];
  return iw;
}

// a row's D = 1 / max(R, 1e-15), R = invw (1 - imp) / imp
__device__ __forceinline__ float efc_D(float iw, float imp) { return 1.f / fmaxf(RMUL(iw, RSUB(1.f, imp)) / imp, 1e-15f); }

#ifndef DK_PRE2W  // the latency mode's k_pre (host.py): two warps per world, warp 0 the kinematics and the smooth
#define DK_PRE2W 0  // dynamics, warp 1 the collision, Jacobians and rows (substep_pre); for small batches
#endif
#if DK_PRE2W && defined(DK_PROFILE)
#error "DK_PRE2W: no phase profile (two warps would share the counters)"
#endif
template <class T> struct dk_is_ptr { static constexpr bool value = false; };
template <class T> struct dk_is_ptr<T*> { static constexpr bool value = true; };

// con_dot[c][d] = J_c,d v for every contact and frame direction (one lane per pair; v a shared NV vector). DK_JV_VEC's
// 128-bit loads only where con_J is a pointer (to struct Hand: SimP, SimN): in k_substep (struct Sim) they made ptxas
// contract other mul / add pairs of that kernel differently (its mass matrix), and k_substep must stay bitwise the
// split forward
template <class S>
__device__ void con_jv(S& s, const float* v, int lane) {
  int ncon = min(s.ncon, MAXCON);
  for (int q = lane; q < CD * ncon; q += 32) {
    int c = q / CD, nd = s.con_nd[c];
    const float* J = s.con_J[c][q % CD];
    const unsigned char* dof = s.con_dof[c];
    float a0 = 0.f, a1 = 0.f;  // even / odd support entries (unrolled and predicated: the loads are independent)
#if DK_JV_VEC && KDOF == 16
    if constexpr (dk_is_ptr<decltype(s.con_J)>::value) {
      const uint4 dw = *reinterpret_cast<const uint4*>(dof);  // the KDOF support dofs, 4 per word
      const unsigned dws[4] = {dw.x, dw.y, dw.z, dw.w};
#pragma unroll
      for (int k4 = 0; k4 < KDOF; k4 += 4) {
        float4 j4 = make_float4(0.f, 0.f, 0.f, 0.f);
        if (k4 < nd) j4 = *reinterpret_cast<const float4*>(J + k4);
        const float jk[4] = {j4.x, j4.y, j4.z, j4.w};
#pragma unroll
        for (int t = 0; t < 4; t += 2) {
          const int k = k4 + t;
          if (k < nd) a0 += jk[t] * v[(dws[k4 >> 2] >> (8 * t)) & 255u];
          if (k + 1 < nd) a1 += jk[t + 1] * v[(dws[k4 >> 2] >> (8 * t + 8)) & 255u];
        }
      }
    } else
#endif
    {
#pragma unroll
      for (int k = 0; k < KDOF; k += 2) {
        if (k < nd) a0 += J[k] * v[dof[k]];
        if (k + 1 < nd) a1 += J[k + 1] * v[dof[k + 1]];
      }
    }
    s.con_dot[c][q % CD] = a0 + a1;
  }
  __syncwarp();
}

// J v for every row (v a shared NV vector) into out[] (NEFC entries); contact rows from con_jv
template <class S>
__device__ DK_INL_ROWSJV void rows_jv(S& s, const float* v, float* out, int lane) {
  con_jv(s, v, lane);
  for (int r = lane; r < s.nefc; r += 32) {
    int kind = s.efc_kind[r], a = s.efc_a[r], b = s.efc_b[r];
    float val;
    if (kind == 0) val = v[a];
    else if (kind == 1) val = b * v[a];
    else if (kind == 2) {
      float mu = s.con_mu[a], sg = (b & 1) ? -1.f : 1.f;
#if HAS_CONDIM4
      if (b >= 4) val = s.con_dot[a][0] + sg * PMUS(s.con_pair[a]) * s.con_dot[a][3];  // (the spin rows)
      else
#endif
      val = s.con_dot[a][0] + sg * mu * s.con_dot[a][1 + (b >> 1)];
    } else val = s.con_dot[a][0];
    out[r] = val;
  }
  __syncwarp();
}

// The piece of row r's cost at x (0 inactive, 1 quadratic, 2 / 3 the linear frictionloss pieces), with the cost
// derivative s'(x) and the curvature (D on the quadratic piece, else 0); D the row's, F its frictionloss bound. The
// frictionloss rows are the first NFLOSS rows (kind 0), so without them (mjlab model) the test is compiled out
__device__ __forceinline__ int row_piece(int r, float x, float D, float F, float* ds, float* dd) {
  if (NFLOSS > 0 && r < NFLOSS) {
    float R = 1.f / D;
    if (x <= -R * F) { *ds = -F; *dd = 0.f; return 2; }
    if (x >= R * F) { *ds = F; *dd = 0.f; return 3; }
    *ds = D * x; *dd = D; return 1;
  }
  if (x < 0.f) { *ds = D * x; *dd = D; return 1; }
  *ds = 0.f; *dd = 0.f; return 0;
}
__device__ __forceinline__ float row_cost(int z, float x, float D, float F) {  // the row's cost s(x) on piece z
  if (NFLOSS > 0 && z >= 2) { float R = 1.f / D; return (z == 2 ? -F * x : F * x) - 0.5f * R * F * F; }
  return z == 1 ? 0.5f * D * x * x : 0.f;
}
template <class S>
__device__ __forceinline__ float row_floss(const S& s, int r) {  // frictionloss bound of row r (0 for the others)
  return NFLOSS > 0 && r < NFLOSS ? c_floss[s.efc_a[r]] : 0.f;
}

// row cost derivative s'(x), curvature (D on the quadratic zone, else 0) and cost s(x)
template <class S>
__device__ __forceinline__ void row_eval(const S& s, int r, float x, float* ds, float* dd, float* cost) {
  float D = s.efc_D[r], F = row_floss(s, r);
  *cost = row_cost(row_piece(r, x, D, F, ds, dd), x, D, F);
}

// J^T f into out (f per row): a contact's rows combined into one force per frame direction (pyramid rows
// n +- mu t1, n +- mu t2), then per dof the dof rows plus the contacts
template <class S>
__device__ __forceinline__ void con_force(const S& s, const float* f, int c, float* g) {  // contact c's, per direction
  const float* fr = f + s.con_row[c];
  int nr = s.con_nrow[c];
  float f0 = nr > 0 ? fr[0] : 0.f, f1 = nr > 1 ? fr[1] : 0.f, f2 = nr > 2 ? fr[2] : 0.f, f3 = nr > 3 ? fr[3] : 0.f;
  float mu = s.con_dim[c] >= 3 ? s.con_mu[c] : 0.f;
  g[0] = f0 + f1 + f2 + f3; g[1] = mu * (f0 - f1); g[2] = mu * (f2 - f3);
#if HAS_CONDIM4
  float f4 = nr > 4 ? fr[4] : 0.f, f5 = nr > 5 ? fr[5] : 0.f;
  g[0] += f4 + f5; g[3] = PMUS(s.con_pair[c]) * (f4 - f5);
#endif
}
template <class S>
__device__ void rows_jtf(S& s, const float* f, float* out, int lane) {
  int ncon = uniform_int(min(s.ncon, MAXCON), CON_BITS);
#if DK_JTF_GATHER
  // per dof in registers (lane l: dofs l and l + 32): the dof rows, then the contacts in contact order (the sums of the
  // scatter below, bitwise). Lane i of a block of 32 contacts holds contact c0 + i's forces and support mask
  // (con_mask), shuffled to all lanes per contact; dof d is in contact c's support when bit d of the mask is set, at
  // support position k = the set bits below d. No __syncwarp() between the contacts and no shared memory
  float acc[2];
#pragma unroll
  for (int t = 0; t < 2; t++) {
    const int i = lane + 32 * t;
    acc[t] = 0.f;
    if (i < NV)
      for (int r = 0; r < s.nefc_dof; r++) {
        if (s.efc_a[r] != i) continue;
        acc[t] += r < NFLOSS ? f[r] : s.efc_b[r] * f[r];  // frictionloss rows (kind 0) first, then limits (kind 1)
      }
  }
  SUBPROF(21);
  SUBPROF(22);
  const unsigned below = (1u << lane) - 1u;
  const auto CJ = s.con_J;  // (SimP / SimN: the pointer into Hand, loaded once)
  for (int c0 = 0; c0 < ncon; c0 += 32) {
    float g[CD] = {};
    unsigned glo = 0u, ghi = 0u;
    if (c0 + lane < ncon) {
      con_force(s, f, c0 + lane, g);
      const unsigned long long m = s.con_mask[c0 + lane];
      glo = (unsigned)m; ghi = (unsigned)(m >> 32);
    }
    const int n = min(32, ncon - c0);
    int i0 = 0;
#if MAXCON > 64
    // many contacts (tabletop scenes): the loads of JTF_BATCH contacts first, then their sums in contact order (the
    // loop below waits on each contact's loads from Hand: some 550 cycles per contact on V100)
    constexpr int JTF_BATCH = 8;
#pragma unroll 1
    for (; i0 + JTF_BATCH <= n; i0 += JTF_BATCH) {
      float jv[JTF_BATCH][CD], fv[JTF_BATCH][CD];
      bool on[JTF_BATCH];
#pragma unroll
      for (int u = 0; u < JTF_BATCH; u++) {
        const int i = i0 + u;
        const unsigned lo = __shfl_sync(0xffffffffu, glo, i);
        fv[u][0] = __shfl_sync(0xffffffffu, g[0], i); fv[u][1] = __shfl_sync(0xffffffffu, g[1], i);
        fv[u][2] = __shfl_sync(0xffffffffu, g[2], i);
#if HAS_CONDIM4
        fv[u][3] = __shfl_sync(0xffffffffu, g[3], i);
#endif
        on[u] = (lo >> lane) & 1u;
        const int k = on[u] ? __popc(lo & below) : 0;
        const float* J = CJ[c0 + i][0];
        jv[u][0] = J[k]; jv[u][1] = J[KDOF + k]; jv[u][2] = J[2 * KDOF + k];
#if HAS_CONDIM4
        jv[u][3] = J[3 * KDOF + k];
#endif
      }
#pragma unroll
      for (int u = 0; u < JTF_BATCH; u++)
        if (on[u]) acc[0] += fv[u][0] * jv[u][0] + fv[u][1] * jv[u][1] + fv[u][2] * jv[u][2]
#if HAS_CONDIM4
                             + fv[u][3] * jv[u][3]
#endif
            ;
    }
#endif
#pragma unroll 1
    for (int i = i0; i < n; i++) {  // the dofs 0-31 (not unrolled: 2 / 4 times were 1.7% / 4.8% slower on V100)
      const unsigned lo = __shfl_sync(0xffffffffu, glo, i);
      const float f0 = __shfl_sync(0xffffffffu, g[0], i), f1 = __shfl_sync(0xffffffffu, g[1], i),
                  f2 = __shfl_sync(0xffffffffu, g[2], i);
#if HAS_CONDIM4
      const float f3 = __shfl_sync(0xffffffffu, g[3], i);
#endif
      const float* J = CJ[c0 + i][0];
      if ((lo >> lane) & 1u) {
        const int k = __popc(lo & below);
        acc[0] += f0 * J[k] + f1 * J[KDOF + k] + f2 * J[2 * KDOF + k]
#if HAS_CONDIM4
                  + f3 * J[3 * KDOF + k]
#endif
            ;
      }
    }
    // the dofs 32.. (lanes 0 ..): only the contacts with support there (few: a right wrist), in contact order. In the
    // loop above they cost every contact 3 predicated loads and 7 more instructions (ptxas if-converts a uniform
    // branch around them), and the gather was then 1-2% slower than the scatter on the RTX 3090
    if (NV > 32)
      for (unsigned hm = __ballot_sync(0xffffffffu, ghi != 0u); hm; hm &= hm - 1u) {
        const int i = __ffs(hm) - 1;
        const unsigned lo = __shfl_sync(0xffffffffu, glo, i), hi = __shfl_sync(0xffffffffu, ghi, i);
        const float f0 = __shfl_sync(0xffffffffu, g[0], i), f1 = __shfl_sync(0xffffffffu, g[1], i),
                    f2 = __shfl_sync(0xffffffffu, g[2], i);
#if HAS_CONDIM4
        const float f3 = __shfl_sync(0xffffffffu, g[3], i);
#endif
        const float* J = CJ[c0 + i][0];
        if ((hi >> lane) & 1u) {
          const int k = __popc(lo) + __popc(hi & below);
          acc[1] += f0 * J[k] + f1 * J[KDOF + k] + f2 * J[2 * KDOF + k]
#if HAS_CONDIM4
                    + f3 * J[3 * KDOF + k]
#endif
              ;
        }
      }
  }
#pragma unroll
  for (int t = 0; t < 2; t++)
    if (lane + 32 * t < NV) out[lane + 32 * t] = acc[t];
  __syncwarp();
#else
  for (int c = lane; c < ncon; c += 32) con_force(s, f, c, s.con_f[c]);
  __syncwarp();
  SUBPROF(21);
  for (int i = lane; i < NV; i += 32) {
    float acc = 0.f;
    for (int r = 0; r < s.nefc_dof; r++) {
      if (s.efc_a[r] != i) continue;
      acc += r < NFLOSS ? f[r] : s.efc_b[r] * f[r];  // frictionloss rows (kind 0) first, then limits (kind 1)
    }
    out[i] = acc;
  }
  __syncwarp();
  SUBPROF(22);
  for (int c = 0; c < ncon; c++) {  // scatter each contact over its support (one dof per lane)
    if (lane < s.con_nd[c])
      out[s.con_dof[c][lane]] += s.con_f[c][0] * s.con_J[c][0][lane] + s.con_f[c][1] * s.con_J[c][1][lane]
                                 + s.con_f[c][2] * s.con_J[c][2][lane]
#if HAS_CONDIM4
                                 + s.con_f[c][3] * s.con_J[c][3][lane]
#endif
          ;
    __syncwarp();
  }
#endif
  SUBPROF(23);
}

// zone of row r at x (the piece of its cost): 0 inactive, 1 quadratic, 2 / 3 the linear frictionloss pieces
template <class S>
__device__ __forceinline__ int row_zone(const S& s, int r, float x) {
  float ds, dd;
  return row_piece(r, x, s.efc_D[r], row_floss(s, r), &ds, &dd);
}

// The tape helpers are not inlined: the forward without a tape then keeps its compact code (V100, a 4096-state corpus,
// 4096 worlds: tape off 20.49 ms against 20.60 ms with the helpers inlined, 20.41 ms without any tape code; tape on
// 20.84 / 20.88 ms)

// tape: the rows on their quadratic zone at the current jar (row_zone 1, the rows that put D into the Newton Hessian),
// as bits; lane j returns rows 32 j .. 32 j + 31 (0 past nefc and on lanes j >= ceil(nefc / 32))
template <class S>
__device__ __noinline__ unsigned zone_word(const S& s, int lane) {
  unsigned z = 0u;
  const int nefc = uniform_int(s.nefc, EFC_BITS);
  for (int r0 = 0; r0 < nefc; r0 += 32) {
    int r = r0 + lane;
    unsigned b = __ballot_sync(0xffffffffu, r < nefc && row_zone(s, r, s.efc_jar[r]) == 1);
    if (lane == r0 >> 5) z = b;
  }
  return z;
}

// tape: n words from shared memory into the record and zeros up to word end (the padding to the next field, so a record
// does not depend on what the buffer held before), lane-strided (each warp store is one coalesced 128-byte line when
// dst is aligned), streaming stores (__stcs: the adjoint reads the tape much later)
__device__ __noinline__ void tape_put(float* dst, const float* src, int n, int end, int lane) {
  for (int i = lane; i < end; i += 32) __stcs(dst + i, i < n ? src[i] : 0.f);
}

// constraint forces at the current jar, J^T f into tmp and grad = M qacc - qfrc_smooth - J^T f; returns |grad| (and
// with DK_GRAD_COST in .y the warp sum of extra, reduced together with it)
template <class S>
// (forces: compute the row forces into efc_jv; false: the line search's update has stored them there)
#if DK_GRAD_COST
__device__ DK_INL_GRAD float2 newton_grad(S& s, int lane, bool forces = true, float extra = 0.f) {
#else
__device__ DK_INL_GRAD float newton_grad(S& s, int lane, bool forces = true) {
#endif
  if (forces)
    for (int r = lane; r < s.nefc; r += 32) {
      float ds, dd, c;
      row_eval(s, r, s.efc_jar[r], &ds, &dd, &c);
      s.efc_jv[r] = -ds;  // the line search's J p is not needed any more
    }
  __syncwarp();
  SUBPROF(20);
  rows_jtf(s, s.efc_jv, s.tmp, lane);
  float gn = 0.f;
  for (int i = lane; i < NV; i += 32) { float g = s.Ma[i] - s.qsm[i] - s.tmp[i]; s.grad[i] = g; gn += g * g; }
  __syncwarp();
#if DK_GRAD_COST
  for (int o = 16; o > 0; o >>= 1) {  // warp_sum of both (the same sums, interleaved)
    gn += __shfl_xor_sync(0xffffffffu, gn, o);
    extra += __shfl_xor_sync(0xffffffffu, extra, o);
  }
  return make_float2(sqrtf(gn), extra);
#else
  return sqrtf(warp_sum(gn));
#endif
}

// out = A v for a symmetric shared NV x NV matrix stored as its packed lower triangle
// (a split of rows 32.. over the warp with shuffle sums, instead of their second trip, was 0.4% / 2.4% slower on the
// V100 / RTX 3090; the element as `j <= i ? A[ri + j] : A[TRI(j, 0) + i]`, meant as two predicated loads at immediate
// offsets, compiles to the same address select; a padded full copy of M does not fit SimN at 11 worlds per SM)
__device__ DK_INL_MATVEC void matvec(const float* A, const float* v, float* out, int lane) {
  for (int i = lane; i < NV; i += 32) {  // one trip count for all lanes: element (i, j) from row max(i, j)
    const int ri = TRI(i, 0);
    float acc = 0.f, acc2 = 0.f;  // even / odd columns
#pragma unroll
    for (int j = 0; j < NV; j++) {  // fully unrolled, no branches: TRI(j, 0) is a constant
      const float* e = A + (j <= i ? ri + j : TRI(j, 0) + i);
      if (j & 1) acc2 += *e * v[j]; else acc += *e * v[j];
    }
    out[i] = acc + acc2;
  }
  __syncwarp();
}

__device__ __forceinline__ float vdot(const float* a, const float* b, int lane) {
  float acc = 0.f;
  for (int i = lane; i < NV; i += 32) acc += a[i] * b[i];
  return warp_sum(acc);
}

// total cost at the current qacc (needs Ma and efc_jar)
template <class S>
__device__ DK_INL_COST float total_cost(S& s, int lane) {
  float g = 0.f;
  for (int i = lane; i < NV; i += 32) g += 0.5f * (s.Ma[i] - s.qsm[i]) * (s.qacc[i] - s.qacc0[i]);
  for (int r = lane; r < s.nefc; r += 32) {
    float ds, dd, c;
    row_eval(s, r, s.efc_jar[r], &ds, &dd, &c);
    g += c;
  }
  return warp_sum(g);
}

template <class S>
__device__ void refresh(S& s, int lane) {  // Ma, jar at the current qacc
  matvec(s.M, s.qacc, s.Ma, lane);
  SUBPROF(18);
  rows_jv(s, s.qacc, s.efc_jar, lane);
  SUBPROF(19);
  for (int r = lane; r < s.nefc; r += 32) s.efc_jar[r] -= s.efc_aref[r];
  __syncwarp();
}

// the free joint (lane 0): position, then the orientation rotated by h times the angular velocity. Not inlined: ptxas
// fuses some of its mul / add pairs depending on the surrounding code, and k_substep and k_post must agree bitwise
__device__ __noinline__ void integrate_root(float* qpos, const float* qvel, float h) {
  for (int k = 0; k < 3; k++) qpos[k] += h * qvel[k];
  float w[3] = {qvel[3], qvel[4], qvel[5]}, nw = sqrtf(dot3(w, w));
  if (nw > 0.f) {
    float ang = h * nw, sa, ca;
    sincosf(0.5f * ang, &sa, &ca);
    float dq[4] = {ca, sa * w[0] / nw, sa * w[1] / nw, sa * w[2] / nw};
    float q[4] = {qpos[3], qpos[4], qpos[5], qpos[6]}, o[4];
    qmul(q, dq, o);
    float on = rsqrtf(o[0] * o[0] + o[1] * o[1] + o[2] * o[2] + o[3] * o[3]);
    for (int k = 0; k < 4; k++) qpos[3 + k] = o[k] * on;
  }
}
"""

STEP_DEVICE = STEP_DEVICE.replace("@@OBB_DEVICE@@", convex.COMMON).replace("@@CONVEX_DEVICE@@", convex.DEVICE + convex.rounded_source())

STEP_KERNEL = r"""
// a contact's support as dof bits (con_mask) from x = the body mask of its two chains' symmetric difference (bit b:
// body b on exactly one of them): the dofs whose body is in x
__device__ __forceinline__ unsigned long long support_mask(unsigned x) {
#if DOF_BODY_ROOT6
  return (unsigned long long)(x >> 2) << 6 | ((x & 2u) ? 0x3full : 0ull);
#else
  unsigned long long m = 0ull;
  for (int i = 0; i < NV; i++)
    if ((x >> c_dofbody[i]) & 1u) m |= 1ull << i;
  return m;
#endif
}
// the Newton Hessian into s.L (see below); world, grp: the contacts against the world and their groups by body, per 32
// contacts (substep_post)
template <class S>
__device__ __forceinline__ void hessian(S& s, int ncon, const unsigned (&world)[CPL], const unsigned (&grp)[CPL],
                                        int lane) {
  // H = M + J' D J on the quadratic rows (lower triangle into L), curvatures D from the rows' zones at jar. The pyramid
  // rows of a contact, u_k = n +- mu t1, n +- mu t2, add C_nn n n' + C_n1 (n t1' + t1 n') + C_11 t1 t1' + (same for t2)
  // = J' C J with C = [C_nn C_n1 C_n2; C_n1 C_11 0; C_n2 0 C_22] (only C_nn for condim 1) and J the frame rows
  float mv[(NTRI + 31) / 32];  // L = M: all loads first (entry e by lane e mod 32)
#pragma unroll
  for (int k = 0; k < (NTRI + 31) / 32; k++) if (lane + 32 * k < NTRI) mv[k] = s.M[lane + 32 * k];
  int ha[HSETS][3], hb[HSETS], hthr[HSETS][3];  // this lane's support pairs (ha, hb) of one column hb per set, the
#pragma unroll                                   // same for every contact, valid when nd > hthr (step.hess_columns)
  for (int k = 0; k < HSETS; k++)
#pragma unroll
    for (int t = 0; t < 3; t++) {
      unsigned e = c_hcol[(3 * k + t) * 32 + lane];
      ha[k][t] = e & 255; hb[k] = (e >> 8) & 255; hthr[k][t] = e >> 16;
    }
#ifndef DK_FOREST_HESS_UNROLL  // forest hessian(): unrolled contact-mask loops (nablaengine.forest_engine.step; 0 on the
#define DK_FOREST_HESS_UNROLL 1  // build targets sm_90 / sm_90a by default: nablaengine.cuda.forest_hold_defines)
#endif
#ifndef DK_HESS_UNROLLED
#define DK_HESS_UNROLLED DK_FOREST_HESS_UNROLL
#endif
#if DK_HESS_UNROLLED
  unsigned act[CPL];  // contacts with a quadratic row
#pragma unroll
  for (int q = 0; q < CPL; q++) {
    const int c0 = 32 * q;
    int c = c0 + lane;
#else
  unsigned act[CPL];  // contacts with a quadratic row
#pragma unroll
  for (int q = 0; q < CPL; q++) act[q] = 0u;
  for (int c0 = 0; c0 < ncon; c0 += 32) {
    int c = c0 + lane;
#endif
    bool on = false;
    if (c < ncon) {
      int r0 = s.con_row[c], nr = s.con_nrow[c];
      float d[2 * CD - 2];
#pragma unroll
      for (int k = 0; k < 2 * CD - 2; k++) {  // curvatures; the loads read a valid row either way (con_row is set only when
        int r = nr > k ? r0 + k : 0;  // nr > 0), so there are no branches
        float jar = s.efc_jar[r], D = s.efc_D[r];
        d[k] = nr > k && jar < 0.f ? D : 0.f;
      }
      float d0 = d[0], d1 = d[1], d2 = d[2], d3 = d[3], mu = s.con_dim[c] >= 3 ? s.con_mu[c] : 0.f;
      s.con_C[c][0] = d0 + d1 + d2 + d3;
      s.con_C[c][1] = mu * (d0 - d1); s.con_C[c][2] = mu * mu * (d0 + d1);
      s.con_C[c][3] = mu * (d2 - d3); s.con_C[c][4] = mu * mu * (d2 + d3);
#if HAS_CONDIM4  // the spin rows n +- mus r (r: the normal's angular row)
      {
        const float mus = s.con_dim[c] == 4 ? PMUS(s.con_pair[c]) : 0.f;
        s.con_C[c][0] += d[4] + d[5];
        s.con_C[c][5] = mus * (d[4] - d[5]); s.con_C[c][6] = mus * mus * (d[4] + d[5]);
      }
#endif
      on = s.con_C[c][0] != 0.f;  // C_nn is the sum of the curvatures
    }
#if DK_HESS_UNROLLED
    act[q] = __ballot_sync(0xffffffffu, on);
  }
#else
    unsigned b = __ballot_sync(0xffffffffu, on);
#pragma unroll
    for (int q = 0; q < CPL; q++)
      if (c0 >> 5 == q) act[q] = b;
  }
#endif
  SUBPROF(27);
#pragma unroll
  for (int k = 0; k < (NTRI + 31) / 32; k++) if (lane + 32 * k < NTRI) s.L[lane + 32 * k] = mv[k];
  const int ndof = uniform_int(s.nefc_dof, 8);
#if DK_HESS_DOFPAR
  if (NFLOSS == 0) {  // limit rows only: one per dof, distinct diagonal entries, each added by its row's lane
    __syncwarp();     // (after the copy)
    for (int r = lane; r < ndof; r += 32) {
      float ds, dd, c;
      row_eval(s, r, s.efc_jar[r], &ds, &dd, &c);
      s.L[TRI(s.efc_a[r], s.efc_a[r])] += dd;
    }
  } else
#endif
  for (int r0 = 0; r0 < ndof; r0 += 32) {  // dof rows on the diagonal: frictionloss (quadratic zone) and limits, each
    int r = r0 + lane, e = 0;              // added in row order by the lane that copied its entry
    float dd = 0.f;
    if (r < ndof) {
      float ds, c;
      row_eval(s, r, s.efc_jar[r], &ds, &dd, &c);
      e = TRI(s.efc_a[r], s.efc_a[r]);
    }
    for (int k = 0; k < min(32, ndof - r0); k++) {
      int ek = __shfl_sync(0xffffffffu, e, k);
      float dk = __shfl_sync(0xffffffffu, dd, k);
      if ((ek & 31) == lane) s.L[ek] += dk;
    }
  }
  __syncwarp();
  SUBPROF(28);
  // the contacts on the lower triangles of their supports (dofs ascending: dof[a] >= dof[b]): first the world
  // contacts, a group (one body, one support) at a time, then each contact between two bodies. Per lane and set: the
  // pairs (a, b) of one column b, H_ab = J_a' G_b with G_b = C J_b once per contact, summed over the group before the
  // update of L
  for (int pass = 0; pass < 2; pass++)  // the world contacts (all of them), then the others
#if DK_HESS_UNROLLED
#pragma unroll
#else
#pragma unroll 1
#endif
  for (int q = 0; q < CPL; q++)
  for (unsigned m = act[q] & (pass ? ~world[q] : world[q]); m;) {
    const bool w = pass == 0;
    const int c0 = 32 * q;  // contacts c0 + i, i in gm
    int c = __ffs(m) - 1;
    unsigned gw = __ballot_sync(0xffffffffu, (grp[q] >> c) & 1u), gm = w ? m & gw : 1u << c;  // c's group
    m &= ~gm;
    if (!w) __syncwarp();  // between two bodies: fenced
    int nd = s.con_nd[c0 + c];
    const unsigned char* dof = s.con_dof[c0 + c];
    constexpr int hlo[HSETS] = HSET_LO;
#pragma unroll
    for (int k = 0; k < HSETS; k++) {
      if (nd <= hlo[k]) break;  // no rows of this set in the support
      int db = dof[hb[k]], he[3];
      float hv[3], lv[3];
#pragma unroll
      for (int t = 0; t < 3; t++) {  // the entries of one support are distinct: loads first, stores after the sum
        int da = dof[ha[k][t]];
        he[t] = da * (da + 1) / 2 + db;
        hv[t] = 0.f;
        if (nd > hthr[k][t]) lv[t] = s.L[he[t]];
      }
      unsigned mm = gm;
      int in = __ffs(mm) - 1;  // the next contact, found one iteration ahead
      mm &= mm - 1u;
      for (int i = in; i >= 0; i = in) {
        in = __ffs(mm) - 1;
        mm &= mm - 1u;
        int cc = c0 + i, b = hb[k];
        const float *C = s.con_C[cc], *Jn = s.con_J[cc][0], *Ja = s.con_J[cc][1], *Jb = s.con_J[cc][2];
        float j0 = Jn[b], j1 = Ja[b], j2 = Jb[b];  // G_b (condim 1: C_nn only, the rest adds exact zeros)
        float g0 = C[0] * j0 + C[1] * j1 + C[3] * j2, g1 = C[1] * j0 + C[2] * j1, g2 = C[3] * j0 + C[4] * j2;
#if HAS_CONDIM4
        const float* Jr = s.con_J[cc][3];
        const float j3 = Jr[b], g3 = C[5] * j0 + C[6] * j3;
        g0 += C[5] * j3;
#endif
#pragma unroll
        for (int t = 0; t < 3; t++) {
          int a = ha[k][t];
          hv[t] = fmaf(Jb[a], g2, fmaf(Ja[a], g1, fmaf(Jn[a], g0, hv[t])));
#if HAS_CONDIM4
          hv[t] = fmaf(Jr[a], g3, hv[t]);
#endif
        }
      }
#pragma unroll
      for (int t = 0; t < 3; t++) if (nd > hthr[k][t]) s.L[he[t]] = lv[t] + hv[t];
    }
  }
}

// one substep for the world in s (qpos, qvel, qacc warm start, com, fric set); ctrl: this world's NV-6 setpoints.
// Two halves (substep_pre: smooth dynamics, collision, Jacobians, rows; substep_post: Newton, implicitfast) so that they can
// also run as separate kernels with the state in global memory (engine_step_check.py --split).
// rec: this world's tape record of the substep (tape.py), or null (no tape); the same
// pointer goes to both halves. The record is written at five points: the input state (here), the actuator clamp mask,
// the contact table (end of substep_pre), the final zones with their Hessian factor L_H, and after the implicitfast
// solve L_Q, a*, a_adv and the header (substep_post). The forward results do not depend on rec.
// PHASE (the split forward with deferred pairs, host.py): 0 everything (k_substep, k_pre); 1 the kinematics and the
// collision only, the deferred pairs queued for k_ccd (k_pre_a; no tape); 2 everything, the collision's contacts read
// from the handoff instead (k_pre after k_ccd: the caller loads them; the smooth phase recomputed: the same bits, where
// handing k_pre_a's smooth state on was 2-4% faster but not bitwise k_substep's)
template <int PHASE = 0, class S>
__device__ void substep_pre(S& s, const float* ctrl, int lane, float* rec) {
  const float h = c_misc[0];
  int b = lane, d = (b < NB) ? c_depth[b] : -1, p = (b < NB) ? c_parent[b] : 0;
  // DK_PRE2W (k_pre of the latency mode, SimP: M in Hand): warp 0 the kinematics, then the smooth dynamics and
  // qacc_smooth; warp 1 the collision, then (after warp 0's hinge subspaces W, V: named barrier 1) the contact
  // Jacobians and rows. The two branches share no arrays (SimP's unions split), so the results are the same bits
  constexpr bool TWO = DK_PRE2W && PHASE == 0 && dk_is_ptr<decltype(s.M)>::value;
  const int warp = TWO ? (int)(threadIdx.x >> 5) : 0;
  if (warp == 0) {
  if (rec) {  // tape: the substep's input state
    tape_put(rec + OFF_QPOS, s.qpos, NQ, NQ, lane);
    tape_put(rec + OFF_QVEL, s.qvel, NV, NV, lane);
  }
  if (lane == 0) {
    s.Q[0][0] = 1.f; s.Q[0][1] = s.Q[0][2] = s.Q[0][3] = 0.f;
    for (int k = 0; k < 3; k++) { s.X[0][k] = s.W[0][k] = s.V[0][k] = s.AL[0][k] = 0.f; s.A[0][k] = -c_grav[k]; }  // gravity
    // as the world's acceleration: the trees with a hinge / slide root
    s.nefc = 0;  // s.unsupported accumulates over substeps; the caller clears it
    if (PHASE != 2) {  // (PHASE 2: the caller loaded the collision's contacts)
      s.ncon = 0;
      s.min_nu_bits = 0x7f800000;  // +inf
    }
  }
  for (int i = lane; i < NV; i += 32) s.zero[i] = 0.f;
  for (int i = lane; i < NTRI; i += 32) s.M[i] = 0.f;
  PROF_START();
  // The tree passes below read what they need from shared memory into registers before they store anything: the
  // compiler cannot tell a body's arrays from its parent's or children's (same arrays, variable indices), so a load
  // after a store would wait for it (one shared-memory latency per element). The arithmetic is unchanged
  for (int L = 1; L <= MAXD; L++) {  // kinematics and velocities (zero acceleration: RNE gives the bias forces)
    if (d == L) {
      float Qp[4], Xp[3], Wp[3], Vp[3], ALp[3], Ap[3], Qb[4], Xb[3], Wb[3], Vb[3], ALb[3], Ab[3];
      for (int k = 0; k < 4; k++) Qp[k] = s.Q[p][k];
      for (int k = 0; k < 3; k++) {
        Xp[k] = s.X[p][k]; Wp[k] = s.W[p][k]; Vp[k] = s.V[p][k]; ALp[k] = s.AL[p][k]; Ap[k] = s.A[p][k];
      }
#if NFIX
      if (c_fixb[b] >= 0) {  // a fixture's moving part: its parent frame is the fixture's placement (at rest)
        for (int k = 0; k < 4; k++) Qp[k] = s.fixq[c_fixb[b]][k];
        for (int k = 0; k < 3; k++) Xp[k] = s.fixx[c_fixb[b]][k];
      }
#endif
      fwd_body(b, Qp, Xp, Wp, Vp, ALp, Ap, s.qpos, s.qvel, s.zero, Qb, Xb, Wb, Vb, ALb, Ab);
      for (int k = 0; k < 4; k++) s.Q[b][k] = Qb[k];
      for (int k = 0; k < 3; k++) {
        s.X[b][k] = Xb[k]; s.W[b][k] = Wb[k]; s.V[b][k] = Vb[k]; s.AL[b][k] = ALb[k]; s.A[b][k] = Ab[k];
      }
    }
    __syncwarp();
  }
  SUBPROF(14);
  }  // warp 0
  if constexpr (TWO) __syncthreads();
  if constexpr (PHASE != 1) {  // (PHASE 1: kinematics and collision only)
  if (warp == 0) {
  const int rt = (b >= 1 && b < NB) ? c_root[b] : 1;  // the body's tree root, whose origin is the tree's point O
  const float O[3] = {s.X[rt][0], s.X[rt][1], s.X[rt][2]};
  if (b >= 1 && b < NB) {
    float ip[3], Qb[4], Xb[3], Wb[3], ALb[3], Ab[3], Fb[3], Nb[3], Jb[10];
    for (int k = 0; k < 4; k++) Qb[k] = s.Q[b][k];
    for (int k = 0; k < 3; k++) { Xb[k] = s.X[b][k]; Wb[k] = s.W[b][k]; ALb[k] = s.AL[b][k]; Ab[k] = s.A[b][k]; }
    body_ipos(b, s.com, ip);
    body_force(b, Qb, Wb, ALb, Ab, ip, Fb, Nb);
    body_inertia(b, Qb, Xb, O, ip, Jb);
    for (int k = 0; k < 3; k++) { s.F[b][k] = Fb[k]; s.N[b][k] = Nb[k]; }
    for (int k = 0; k < 10; k++) s.J[b][k] = Jb[k];
    if (JT_OF(b) != JFREE) {  // the joint's motion subspace about O into W, V (no longer needed), for the mass matrix and the
      float w[3], v[3];  // contact Jacobians
      dof_motion(b, 0, Qb, Xb, O, w, v);
      for (int k = 0; k < 3; k++) { s.W[b][k] = w[k]; s.V[b][k] = v[k]; }
    }
  }
  __syncwarp();
  SUBPROF(15);
  if constexpr (TWO) { __threadfence_block(); asm volatile("bar.arrive 1, 64;" ::: "memory"); }  // W, V for warp 1
  for (int L = MAXD - 1; L >= 1; L--) {
    if (d == L && c_nchild[b] > 0) {
      int nch = c_nchild[b];
      float Xb[3], Fb[3], Nb[3], Jb[10];
      for (int k = 0; k < 3; k++) { Xb[k] = s.X[b][k]; Fb[k] = s.F[b][k]; Nb[k] = s.N[b][k]; }
      for (int k = 0; k < 10; k++) Jb[k] = s.J[b][k];
      for (int i = 0; i < nch; i++) {
        int c = c_child[NCH * b + i];
        float Xc[3], Fc[3], Nc[3];
        for (int k = 0; k < 3; k++) { Xc[k] = s.X[c][k]; Fc[k] = s.F[c][k]; Nc[k] = s.N[c][k]; }
        accumulate(Xb, Xc, Fc, Nc, Fb, Nb);
        for (int k = 0; k < 10; k++) Jb[k] += s.J[c][k];
      }
      for (int k = 0; k < 3; k++) { s.F[b][k] = Fb[k]; s.N[b][k] = Nb[k]; }
      for (int k = 0; k < 10; k++) s.J[b][k] = Jb[k];
    }
    __syncwarp();
  }
  SUBPROF(16);
  if (b >= 1 && b < NB) torques(b, s.Q[b], s.F[b], s.N[b], s.qsm);  // qsm holds the bias for now
  float sw[3], sv[3], kw[3], kv[3], n[3], f[3];
  const float q1[4] = {s.Q[rt][0], s.Q[rt][1], s.Q[rt][2], s.Q[rt][3]};  // the root's rotation (its dofs' subspace)
  const bool rfree = b >= 1 && b < NB && JT_OF(rt) == JFREE;
  if (b >= 1 && b < NB && JT_OF(b) != JFREE) {
    int j = c_dadr[b], na = c_nanc[b], ad[12];
    float Jb[10], aw[12][3], av[12][3];
    for (int k = 0; k < 10; k++) Jb[k] = s.J[b][k];
    for (int k = 0; k < 3; k++) { sw[k] = s.W[b][k]; sv[k] = s.V[b][k]; }
#pragma unroll
    for (int a = 0; a < 12; a++)  // the hinge ancestors' motion subspaces, all loaded before the first store
      if (a < na) {
        int k = c_anc[12 * b + a];
        ad[a] = c_dadr[k];
        for (int t = 0; t < 3; t++) { aw[a][t] = s.W[k][t]; av[a][t] = s.V[k][t]; }
      }
    apply_inertia(Jb, sw, sv, n, f);
    s.M[TRI(j, j)] = dot3(sw, n) + dot3(sv, f) + c_armature[j];
#pragma unroll
    for (int a = 0; a < 12; a++)
      if (a < na) s.M[TRI(j, ad[a])] = dot3(aw[a], n) + dot3(av[a], f);
    if (rfree) {
      const int r0 = c_dadr[rt];
      for (int t = 0; t < 6; t++) {
        dof_motion(rt, t, q1, O, O, kw, kv);
        s.M[TRI(j, r0 + t)] = dot3(kw, n) + dot3(kv, f);
      }
    }
  }
  if (b >= 1 && b < NB && JT_OF(b) == JFREE) {
    float J1[10];
    const int r0 = c_dadr[b];
    for (int k = 0; k < 10; k++) J1[k] = s.J[b][k];
    for (int j = 0; j < 6; j++) {
      dof_motion(b, j, q1, O, O, sw, sv);
      apply_inertia(J1, sw, sv, n, f);
      for (int i = 0; i <= j; i++) {
        dof_motion(b, i, q1, O, O, kw, kv);
        s.M[TRI(r0 + j, r0 + i)] = dot3(kw, n) + dot3(kv, f) + (i == j ? c_armature[r0 + j] : 0.f);
      }
    }
  }
  __syncwarp();
  SUBPROF(17);
  for (int i0 = 0; i0 < NV; i0 += 32) {  // qfrc_smooth = clamped actuator force - bias
    int i = i0 + lane;
    bool clamp = false;
    bool aclamp = false, uclamp = false;
#if DK_CLAMP_IMPLICIT == 2
    float wv = 1.f;  // the joint limit's weight of the implicit kv, with the force's sign (1: not clamped there)
#endif
    if (i < NV) {
      float act = 0.f;
      const int a = ACT_OF(i);
      if (a >= 0) {  // MuJoCo Warp's order: ctrl clamped to ctrlrange, gain ctrl + bias, actuator forcerange, then the
        // joint's actfrcrange
        float u = fminf(fmaxf(ctrl[a], c_ctrlr[2 * i]), c_ctrlr[2 * i + 1]);
        uclamp = u != ctrl[a];  // (the adjoint: no ctrl gradient)
        act = c_kp[i] * u - c_kq[i] * s.qpos[DQADR(i)] - c_kv[i] * s.qvel[i];
        aclamp = act <= -c_afrc[i] || act >= c_afrc[i];  // MuJoCo Warp: no implicit kv at the forcerange
        act = fminf(fmaxf(act, -c_afrc[i]), c_afrc[i]);
        clamp = aclamp || act <= -c_frclim[i] || act >= c_frclim[i];  // at the limit: MuJoCo Warp's actuator_lam test
#if DK_CLAMP_IMPLICIT == 2
        if (clamp && !aclamp)  // w = exp(-o / delta), o the overshoot over the joint's limit (adjoint: dw/df)
          wv = copysignf(expf(-fmaxf((fabsf(act) - c_frclim[i]) / c_frclim[i], 0.f) / DK_CLAMP_IMPLICIT_DELTA), act);
#endif
        act = fminf(fmaxf(act, -c_frclim[i]), c_frclim[i]);
      }
#if DK_CLAMP_IMPLICIT == 2
      s.kvw[i] = wv;
      if (rec) __stcs(rec + OFF_KVW + i, wv);
#endif
      s.qsm[i] = act - c_damp[i] * s.qvel[i] - s.qsm[i];
    }
    {  // the dofs whose implicit kv is off (actuator force at its forcerange), for substep_post; tape: that mask and
       // the ctrl-clamp mask
#if DK_CLAMP_IMPLICIT == 1  // MuJoCo 4.0: also off at the joint's actfrcrange
      unsigned bal = __ballot_sync(0xffffffffu, clamp), ubal = __ballot_sync(0xffffffffu, uclamp);
#else
      unsigned bal = __ballot_sync(0xffffffffu, aclamp), ubal = __ballot_sync(0xffffffffu, uclamp);
#endif
      if (lane == 0) s.kvoff[i0 >> 5] = bal;
      if (rec && lane == 0) {
        __stcs(rec + OFF_KVOFF + (i0 >> 5), __uint_as_float(bal));
        __stcs(rec + OFF_UCLAMP + (i0 >> 5), __uint_as_float(ubal));
      }
    }
    if (rec) {  // tape: the clamp mask (dof bits), stored rather than recomputed: the adjoint's FMA contraction of the
      unsigned bal = __ballot_sync(0xffffffffu, clamp);  // force could differ and flip a dof at the limit
      if (lane == 0) __stcs(rec + OFF_CLAMP + (i0 >> 5), __uint_as_float(bal));
    }
  }
  PROF(0);
  for (int i = lane; i < NV; i += 32) s.qacc0[i] = s.qsm[i];
  __syncwarp();
  CHOL_M_SOLVE(s.M, s.L, s.qacc0, lane, s.cbuf);
  PROF(1);
  }  // warp 0

  }  // PHASE != 1

  if (!TWO || warp == 1) {
  // collision
  if constexpr (PHASE == 2) {  // the contacts of k_pre_a and k_ccd (the caller loaded them), in canonical order
    __syncwarp();
    sort_contacts(s, lane);
  } else {
  geom_poses(s, lane);
  __syncwarp();
  SUBPROF(26);
  {  // broad phase: the pairs whose bounding test passes (collide_pair's first test, here with a relative slack of 1e-5
     // so that it keeps every pair collide_pair keeps), in pair order into s.cand; then collide_pair on them, one per
     // lane (rather than 16 pairs per lane, most of them rejected and the narrow phases of the others diverging)
    static_assert(NPAIR < 65536, "candidate pairs as 16-bit indices");
    const unsigned lanes_below = (1u << lane) - 1u;
#if FLOOR_G >= 0
    float fn[3];  // the only plane's normal
    zaxis_rn(s.gq[FLOOR_G], fn);
#define FN fn
#else
#define FN nullptr
#endif
    int ncand = 0;
#if BROAD_GROUPS
    // first the pair groups (step.broad_groups: one geom against the bounding sphere of a body), the passing ones into
    // s.gcand; then their pairs in pair order, 32 groups at a time (lane j: the group's pairs at stream positions
    // exc_j .. inc_j - 1), with the per-pair test below
    int ngc = 0;
    for (int k = 0; k < (NGRP + 31) / 32; k++) {
      int gi = lane + 32 * k;
      bool keep = false;
      if (gi < NGRP) {
        const unsigned e = c_grp[gi];
        const int g1 = e & 255u, b = (e >> 8) & 255u;
        float dif[3] = {RSUB(s.X[b][0], s.gx[g1][0]), RSUB(s.X[b][1], s.gx[g1][1]), RSUB(s.X[b][2], s.gx[g1][2])};
        if (e >> 16) {
          float n[3];
          zaxis_rn(s.gq[g1], n);
          keep = dot3(dif, n) <= c_brad[b];
        } else {
          float rb = RADD(c_grbound[g1], c_brad[b]);
          keep = dot3(dif, dif) <= RFMA(RMUL(rb, rb), 1.f + 1e-5f, 1e-12f);
        }
      }
      unsigned bal = __ballot_sync(0xffffffffu, keep);
      if (keep) s.gcand[ngc + __popc(bal & lanes_below)] = (unsigned short)gi;
      ngc += __popc(bal);
    }
    __syncwarp();
    for (int j0 = 0; j0 < ngc; j0 += 32) {
      const int j = j0 + lane, gi = j < ngc ? s.gcand[j] : 0;
      const int first = j < ngc ? c_gstart[gi] : 0, cnt = j < ngc ? c_gstart[gi + 1] - first : 0;
      int inc = cnt;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
      }
      const int total = __shfl_sync(0xffffffffu, inc, 31), exc = inc - cnt;
      for (int t0 = 0; t0 < total; t0 += 32) {
        const int t = t0 + lane;
        int lo = 0;  // the last lane whose stream start is <= t (exc ascending; the lanes past ngc start at total)
#pragma unroll
        for (int st = 16; st; st >>= 1) {
          int e = __shfl_sync(0xffffffffu, exc, (lo + st) & 31);
          if (lo + st < 32 && e <= t) lo += st;
        }
        const int pr = __shfl_sync(0xffffffffu, first, lo) + t - __shfl_sync(0xffffffffu, exc, lo);
        const bool keep = t < total && pair_bound(s, pr, FN);
        const unsigned bal = __ballot_sync(0xffffffffu, keep);
        const int at = ncand + __popc(bal & lanes_below);
        if (keep && at < NCAND) s.cand[at] = (unsigned short)pr;
        ncand += __popc(bal);
      }
    }
    if (ncand > NCAND) {  // candidates beyond capacity: their contacts are dropped
      if (lane == 0) s.unsupported |= 4;
      ncand = NCAND;
    }
#else
#pragma unroll 4
    for (int k = 0; k < (NPAIR + 31) / 32; k++) {
      int pr = lane + 32 * k;
      bool keep = pr < NPAIR && pair_bound(s, pr, FN);
      unsigned bal = __ballot_sync(0xffffffffu, keep);
      if (keep) s.cand[ncand + __popc(bal & lanes_below)] = (unsigned short)pr;
      ncand += __popc(bal);
    }
#endif
    __syncwarp();
    SUBPROF(29);
    for (int i = lane; i < ncand; i += 32) collide_pair(s, s.cand[i]);
#undef FN
  }
  __syncwarp();
  if constexpr (PHASE == 1) return;  // (k_pre_a: the caller hands the contacts and geom poses to k_ccd)
#if HAS_DEFER
  sort_contacts(s, lane);  // (the order of the split forward's, whose k_ccd appends in no fixed order)
#endif
  }  // PHASE != 2
  PROF(2);
  if constexpr (TWO) asm volatile("bar.sync 1, 64;" ::: "memory");  // warp 0's hinge subspaces W, V
  if (lane == 0 && s.ncon > MAXCON) s.unsupported |= 4;
  int ncon = uniform_int(min(s.ncon, MAXCON), CON_BITS);
  {  // contact Jacobians: frame rows of J(body2, pos) - J(body1, pos) on the dofs on exactly one chain, packed in
     // ascending order. Lane l keeps the motion subspaces (w, v about O) of dofs l and l + 32, and the two bodies'
     // ancestor masks of contacts l, l + 32, .. (CPL of them; shuffled to all lanes per contact)
    static_assert(NV <= 64 && MAXCON <= 255, "two dofs per lane; contact counts in 8 bits");
    float dw[2][3], dv[2][3], dO[2][3];
    int dbody[2];
    unsigned am1[CPL], am2[CPL];
#pragma unroll
    for (int h = 0; h < CPL; h++) {
      int c = lane + 32 * h, pr = c < ncon ? s.con_pair[c] : 0;
      am1[h] = c_ancmask[c_gbody[PG1(pr)]];
      am2[h] = c_ancmask[c_gbody[PG2(pr)]];
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
      int i = lane + 32 * h;
      dbody[h] = i < NV ? c_dofbody[i] : 0;
      const int db = dbody[h], dr = db >= 1 ? c_root[db] : 1;  // the dof's body and its tree root (point O)
      for (int t = 0; t < 3; t++) dO[h][t] = s.X[dr][t];
      if (db >= 1 && JT_OF(db) == JFREE) {
        const float qr[4] = {s.Q[db][0], s.Q[db][1], s.Q[db][2], s.Q[db][3]};
        dof_motion(db, i - c_dadr[db], qr, dO[h], dO[h], dw[h], dv[h]);
      } else if (db >= 1)
        for (int t = 0; t < 3; t++) { dw[h][t] = s.W[db][t]; dv[h][t] = s.V[db][t]; }
    }
    for (int c = 0; c < ncon; c++) {
      unsigned a1 = am1[0], a2 = am2[0];
#pragma unroll
      for (int h = 1; h < CPL; h++)
        if (c >> 5 == h) { a1 = am1[h]; a2 = am2[h]; }
      unsigned m1 = __shfl_sync(0xffffffffu, a1, c & 31);
      unsigned m2 = __shfl_sync(0xffffffffu, a2, c & 31);
      float pos[3], fr[9];
      for (int t = 0; t < 3; t++) pos[t] = s.con_pos[c][t];
      for (int t = 0; t < 9; t++) fr[t] = s.con_frame[c][t];
      int base = 0;
#pragma unroll
      for (int h = 0; h < 2; h++) {
        int i = lane + 32 * h, bi = dbody[h];
        int in2 = (m2 >> bi) & 1, in1 = (m1 >> bi) & 1;
        bool on = i < NV && in1 != in2;
        unsigned bal = __ballot_sync(0xffffffffu, on);
        if (on) {
          int k = base + __popc(bal & ((1u << lane) - 1u));
          float r[3], u[3], jp[3];
          for (int t = 0; t < 3; t++) r[t] = pos[t] - dO[h][t];
          cross(dw[h], r, u);
          float sg = in2 ? 1.f : -1.f;
          for (int t = 0; t < 3; t++) jp[t] = sg * (dv[h][t] + u[t]);
          for (int dd = 0; dd < 3; dd++) s.con_J[c][dd][k] = dot3(fr + 3 * dd, jp);
#if HAS_CONDIM4  // the normal's angular row: sg (w_i . n), 0 for a slide
          s.con_J[c][3][k] = sg * dot3(fr, dw[h]);
#endif
          s.con_dof[c][k] = (unsigned char)i;
        }
        base += __popc(bal);
      }
      if (lane == 0) s.con_nd[c] = base;
    }
#pragma unroll
    for (int h = 0; h < CPL; h++)
      if (lane + 32 * h < ncon) s.con_mask[lane + 32 * h] = support_mask(am1[h] ^ am2[h]);  // (the ballots' bits)
  }
  PROF(3);

  // constraint rows: dof frictionloss rows (dof order), active hinge limits (dof order), then per contact (contact order)
  // 4 pyramid rows (condim 3) or 1 normal row (condim 1); row indices from ballot prefix counts. First the dof rows'
  // (kind, dof, sign), one lane per dof
  const unsigned below = (1u << lane) - 1u;
  int ndof = 0;
  for (int i0 = 0; i0 < NV; i0 += 32) {
    int i = i0 + lane;
    bool on = i < NV && c_floss[i] > 0.f;
    unsigned bal = __ballot_sync(0xffffffffu, on);
    if (on) { int r = ndof + __popc(bal & below); s.efc_kind[r] = 0; s.efc_a[r] = i; s.efc_b[r] = 1; }
    ndof += __popc(bal);
  }
  for (int i0 = 0; i0 < NV; i0 += 32) {
    int i = min(i0 + lane, NV - 1);  // clamped so that the loads need no branch
    float dmin, dmax;
    limit_dists(i, s.qpos[DQADR(i)], &dmin, &dmax);
    bool on = i0 + lane < NV && c_jlim[i] != 0.f && fminf(dmin, dmax) < 0.f;  // (free dofs: c_jlim 0)
    unsigned bal = __ballot_sync(0xffffffffu, on);
    if (on) {
      int r = ndof + __popc(bal & below);
      s.efc_kind[r] = 1; s.efc_a[r] = i; s.efc_b[r] = dmin < dmax ? 1 : -1;
    }
    ndof += __popc(bal);
  }
  con_jv(s, s.qvel, lane);  // contact J qvel for the contact rows' velocities (its __syncwarp also publishes the above)
  // then one lane per job, a dof row (job j < ndof) or a contact (c = j - ndof): impedance, D and aref of its rows;
  // contact c's first row is ndof + c + 3 * (the number of condim-3 contacts before c)
  int ntri = 0, nquad = 0;
  for (int j0 = 0; j0 < ndof + ncon; j0 += 32) {
    int j = j0 + lane, c = j - ndof;
    bool con = c >= 0 && c < ncon, tri = con && s.con_dim[c] >= 3;
    unsigned b3 = __ballot_sync(0xffffffffu, tri);
    int r0 = ndof + c + 3 * (ntri + __popc(b3 & below));
    ntri += __popc(b3);
#if HAS_CONDIM4  // condim 4: two more rows (after its 4 pyramid rows)
    const bool quad = con && s.con_dim[c] == 4;
    const unsigned b4 = __ballot_sync(0xffffffffu, quad);
    r0 += 2 * (nquad + __popc(b4 & below));
    nquad += __popc(b4);
#endif
    if (j < ndof + ncon) {
      const float *sref, *simp;
      float iw, pos = 0.f, mu = 0.f;
      int i = 0;
      if (!con) {  // dof row j: frictionloss (at 0) or limit
        i = s.efc_a[j];
        bool fl = s.efc_kind[j] == 0;
        float dmin, dmax;
        limit_dists(i, s.qpos[DQADR(i)], &dmin, &dmax);
        iw = c_dinvw[i];
        pos = fl ? 0.f : fminf(dmin, dmax);
        sref = fl ? c_dsolref : c_jsolref; simp = fl ? c_dsolimp : c_jsolimp;
      } else {
        int pr = s.con_pair[c];
        mu = s.con_mu[c];
        iw = contact_invw(pr, mu);
        pos = s.con_dist[c]; sref = PSOLREF(pr); simp = PSOLIMP(pr);
      }
      float k, bb, imp;
      kbimp(sref, simp, pos, h, &k, &bb, &imp);
      float D = efc_D(iw, imp);
      if (!con) {
        s.efc_D[j] = D; s.efc_aref[j] = -k * imp * pos - bb * (s.efc_b[j] * s.qvel[i]);
      } else {
        int nr = max(0, min(tri ? (HAS_CONDIM4 && s.con_dim[c] == 4 ? 6 : 4) : 1, NEFC - r0));  // rows beyond NEFC: dropped
        s.con_row[c] = r0; s.con_nrow[c] = nr;
#pragma unroll
        for (int t = 0; t < 2 * CD - 2; t++) {
          if (t >= nr) break;
          float sg = (t & 1) ? -1.f : 1.f;
          float vel = tri ? s.con_dot[c][0] + sg * mu * s.con_dot[c][1 + (t >> 1)] : s.con_dot[c][0];
#if HAS_CONDIM4
          if (t >= 4) vel = s.con_dot[c][0] + sg * PMUS(s.con_pair[c]) * s.con_dot[c][3];
#endif
          s.efc_kind[r0 + t] = tri ? 2 : 3; s.efc_a[r0 + t] = c; s.efc_b[r0 + t] = t;
          s.efc_D[r0 + t] = D; s.efc_aref[r0 + t] = -k * imp * pos - bb * vel;
        }
      }
    }
  }
  if (lane == 0) {
    int need = ndof + ncon + 3 * ntri + 2 * nquad;
    if (need > NEFC) s.unsupported |= 16;  // constraint rows beyond capacity are dropped
    s.nefc = min(need, NEFC);
    s.nefc_dof = ndof;  // dof rows (frictionloss, limits) come first; contact rows follow
  }
  PROF(4);
  if (rec)  // tape: the contact table in contact (= row) order, one entry pair | slot << 16 per word
    for (int i = lane; i < MAXCON; i += 32) {
      unsigned e = i < ncon ? (unsigned)s.con_pair[i] | (unsigned)s.con_slot[i] << 16 : TAPE_CON_NONE;
      __stcs(rec + OFF_CON + i, __uint_as_float(e));
    }
  }  // warp 1 (or the only warp)
  if constexpr (TWO) __syncthreads();
}

template <bool NARROW> struct ZWord { typedef unsigned T; };  // the line search's row piece codes
template <> struct ZWord<false> { typedef unsigned long long T; };

template <class S>
__device__ void substep_post(S& s, int iterations, int lane, int* niter_out, float* rec) {
  const float h = c_misc[0];
  int ncon = uniform_int(min(s.ncon, MAXCON), CON_BITS);
  PROF_START();
  // Newton solver from the warm start (or qacc_smooth when that is cheaper)
  const float scale = 1.f / (c_sopt[1] * NV), tol = c_sopt[2];
  refresh(s, lane);
  float cost = total_cost(s, lane);
  // the warm start's qacc, Ma and jar kept in tmp, p and efc_jv (all free here; SimN's Mp shares tmp's storage) in
  // case it stays the start
  for (int i = lane; i < NV; i += 32) { s.tmp[i] = s.qacc[i]; s.qacc[i] = s.qacc0[i]; s.p[i] = s.Ma[i]; }
  for (int r = lane; r < s.nefc; r += 32) s.efc_jv[r] = s.efc_jar[r];
  __syncwarp();
  refresh(s, lane);
  float cost0 = total_cost(s, lane);
  if (UNIFORM(cost0 > cost)) {
    for (int i = lane; i < NV; i += 32) { s.qacc[i] = s.tmp[i]; s.Ma[i] = s.p[i]; }
    for (int r = lane; r < s.nefc; r += 32) s.efc_jar[r] = s.efc_jv[r];
    __syncwarp();
  } else cost = cost0;
  PROF(5);
  newton_grad(s, lane);
  PROF(6);
  // contacts against the world, for the Hessian assembly: the support of such a contact is the whole chain of the
  // other body, so the dofs two of them share are their common ancestors' dofs, at the same positions in both supports,
  // and an entry of L they share comes from the same support pair (a, b), which is always on the same lane: no
  // __syncwarp() between them. The world contacts on one body have the same support and are summed in registers before
  // one update of L (grp[q]: lane masks of the contacts 32 q .. 32 q + 31 on the same body). A contact between two
  // bodies (support: the dofs on exactly one chain) is fenced on both sides.
  static_assert(KDOF <= 32, "hess_columns fits 32 lanes");
  unsigned world[CPL], grp[CPL];
#pragma unroll
  for (int q = 0; q < CPL; q++) {
    int c = 32 * q + lane, pr = c < ncon ? s.con_pair[c] : 0, b1 = c_gbody[PG1(pr)], b2 = c_gbody[PG2(pr)];
    bool w = c < ncon && (b1 == 0 || b2 == 0);
    world[q] = __ballot_sync(0xffffffffu, w);
    grp[q] = __match_any_sync(0xffffffffu, w ? b1 + b2 : -1 - lane);
  }
  // tape (rec): L_H must be the Hessian factor at the final zones. The last factorization was assembled at the jar
  // before the last step; when that step moved a row to another zone (chg, tested where jar is updated), or no step was
  // made, the loop runs once more without a step (refact): assembly and factorization at the final jar, so the tape
  // needs no second copy of the assembly code. Only the scratch s.L, s.p and s.con_C change, so the forward results are
  // the same with and without the tape
  int it = 0;
  bool conv = false, zchanged = rec && iterations <= 0, refact = zchanged, zsame = false;
  while (it < iterations || refact) {
#if NEWTON_REUSE || NEWTON_REUSE_WIDE
    if (zsame && !refact) {  // no row changed its zone in the last step: H and its factor in s.L are the same
      for (int i = lane; i < NV; i += 32) s.p[i] = -s.grad[i];
      __syncwarp();
      chol_resolve(s.L, s.p, lane, s.cbuf);
    } else
#endif
    {
      hessian(s, ncon, world, grp, lane);
      PROF(7);
      for (int i = lane; i < NV; i += 32) s.p[i] = -s.grad[i];
      __syncwarp();
      CHOL_H_SOLVE(s.L, s.L, s.p, lane, s.cbuf);
    }
    PROF(8);
    if (refact) break;  // the tape's re-factorization: no step
    // exact line search along p on the piecewise-quadratic cost: 1-D Newton, stopping when the step stays on its piece
    rows_jv(s, s.p, s.efc_jv, lane);
    SUBPROF(24);
    matvec(s.M, s.p, s.Mp, lane);
    SUBPROF(25);
#ifdef DK_PROFILE
    __syncwarp(); if (lane == 0) s.prof[12] += clock64() - s.prof_t;
#endif
    float a1 = 0.f, a2 = 0.f;
    for (int i = lane; i < NV; i += 32) { a1 += (s.Ma[i] - s.qsm[i]) * s.p[i]; a2 += s.p[i] * s.Mp[i]; }
    a1 = warp_sum(a1); a2 = warp_sum(a2);
    // the lane's rows r = lane + 32 t stay in registers for all steps (jar, J p, D, frictionloss bound; zeros past
    // nefc: inactive, no contribution), their pieces as 2-bit codes; a Newton step's test whether it kept every row on
    // its piece (then it landed on the exact minimizer) is done by the next pass over the rows
    constexpr int RT = (NEFC + 31) / 32;
    typedef typename ZWord<(RT <= 16)>::T ZT;  // 2 bits per row
    static_assert(RT <= 32, "the rows' piece codes in 64 bits");
    const int nefc = s.nefc;
    float rx[RT], rj[RT], rD[RT], rF[RT];
#pragma unroll
    for (int t = 0; t < RT; t++) {
      int r = lane + 32 * t;
      bool v = r < nefc;
      rx[t] = v ? s.efc_jar[r] : 0.f; rj[t] = v ? s.efc_jv[r] : 0.f; rD[t] = v ? s.efc_D[r] : 0.f;
      rF[t] = row_floss(s, r);
    }
    float alpha = 0.f, lo = 0.f, hi = 1e30f, d0 = 0.f;
    ZT zprev = 0u;
    bool newton = false;
    for (int k = 0; k <= 40; k++) {
      float fd = 0.f, fdd = 0.f;
      ZT z = 0u;
#pragma unroll
      for (int t = 0; t < RT; t++) {
        if (32 * t >= nefc) break;
        float ds, dd, jv = rj[t];
        z |= (ZT)row_piece(lane + 32 * t, rx[t] + alpha * jv, rD[t], rF[t], &ds, &dd) << (2 * t);
        fd += ds * jv; fdd += dd * jv * jv;
      }
      if (__all_sync(0xffffffffu, newton && z == zprev)) break;  // the last Newton step kept every row on its piece
      if (k == 40) break;
#ifdef DK_PROFILE
      if (lane == 0) s.prof[13]++;
#endif
      fd = warp_sum(fd) + a1 + alpha * a2;
      fdd = warp_sum(fdd) + a2;
      if (k == 0) { d0 = fd; if (UNIFORM(d0 >= 0.f)) break; }
      if (UNIFORM(fabsf(fd) <= 1e-7f * fabsf(d0))) break;
      if (fd < 0.f) lo = alpha; else hi = alpha;
      float nx = alpha - fd / fdd;
      newton = nx > lo && nx < hi;
      if (!newton) nx = hi < 1e29f ? 0.5f * (lo + hi) : 2.f * fmaxf(alpha, 1e-6f);
      zprev = z;
      alpha = nx;
    }
    // move along p: qacc, M qacc and J qacc - aref are linear in alpha; with the new cost (as total_cost) and the rows'
    // forces for newton_grad
    float g = 0.f;
    for (int i = lane; i < NV; i += 32) {
      float qa = s.qacc[i] + alpha * s.p[i], ma = s.Ma[i] + alpha * s.Mp[i];
      s.qacc[i] = qa; s.Ma[i] = ma;
      g += 0.5f * (ma - s.qsm[i]) * (qa - s.qacc0[i]);
    }
    bool chg = false;  // a row of this lane changes its zone (tape: the Hessian factored above is then out of date)
#pragma unroll
    for (int t = 0; t < RT; t++) {
      int r = lane + 32 * t;
      if (32 * t >= nefc) break;
      if (r < nefc) {
        float xn = rx[t] + alpha * rj[t], ds, dd;
        int z0 = row_piece(r, rx[t], rD[t], rF[t], &ds, &dd), z1 = row_piece(r, xn, rD[t], rF[t], &ds, &dd);
        chg |= z0 != z1;
        g += row_cost(z1, xn, rD[t], rF[t]);
        s.efc_jar[r] = xn;
        s.efc_jv[r] = -ds;  // the line search's J p is not needed any more
      }
    }
    zsame = !__any_sync(0xffffffffu, chg);  // then the next H is this one
    PROF(9);
#if DK_GRAD_COST
    PROF(10);
    const float2 gc = newton_grad(s, lane, false, g);  // gradient at the new point (the stopping test and the next
    const float gn = gc.x, newcost = gc.y;             // step), the new cost
    float improvement = scale * (cost - newcost);
    cost = newcost;
#else
    float newcost = warp_sum(g);
    PROF(10);
    float improvement = scale * (cost - newcost);
    cost = newcost;
    float gn = newton_grad(s, lane, false);  // gradient at the new point: the stopping test and the next step
#endif
    PROF(6);
    it++;
    conv = UNIFORM(improvement < tol || scale * gn < tol);
    if (conv || it == iterations) {
      if (rec) {  // tape: did the last step change a zone?
        zchanged = __any_sync(0xffffffffu, chg);
#ifdef DK_TAPE_VERIFY
        for (int e = lane; e < NTRI; e += 32) s.Lv[e] = s.L[e];  // debug: always re-assemble, compare below
        refact = true;
#else
        refact = zchanged;
#endif
      }
      if (!refact) break;
    }
  }
  if (niter_out && lane == 0) *niter_out = it;
  if (s.oforce) {  // the contacts' forces at the solution (the last substep): the rows' forces -s'(J a* - aref) on
    // their pieces, per contact in its frame (con_force: normal, t1, t2); recomputed here, as a re-assembly of H for
    // the tape reuses their memory
    for (int r = lane; r < s.nefc; r += 32) {
      float ds, dd, cst;
      row_eval(s, r, s.efc_jar[r], &ds, &dd, &cst);
      s.efc_jv[r] = -ds;
    }
    __syncwarp();
    const int nc = min(s.ncon, MAXCON);
    for (int c = lane; c < nc; c += 32) {
      float g[CD];
      con_force(s, s.efc_jv, c, g);
      for (int k = 0; k < 3; k++) s.oforce[c][k] = g[k];
    }
    __syncwarp();
  }
  if (rec) {  // tape: the final zones and their Hessian factor, before the implicitfast solve overwrites L
    unsigned z = zone_word(s, lane);
    if (lane < (NEFC + 31) / 32) __stcs(rec + OFF_ZONE + lane, __uint_as_float(z));
    tape_put(rec + OFF_LH, s.L, NTRI, OFF_LQ - OFF_LH, lane);
  }
#ifdef DK_TAPE_VERIFY
  bool vdiff = false;  // the factor of the last Newton assembly differs from the one at the final zones
  if (rec && it > 0) {
    bool d = false;
    for (int e = lane; e < NTRI; e += 32) d |= __float_as_uint(s.L[e]) != __float_as_uint(s.Lv[e]);
    vdiff = __any_sync(0xffffffffu, d);
  }
#endif

  // implicitfast: (M + h kv) qacc' = M qacc; then qvel, qpos
  for (int e = lane; e < NTRI; e += 32) s.L[e] = s.M[e];
  __syncwarp();
  for (int i = lane; i < NV; i += 32) {  // kv off at the actuator forcerange (substep_pre's kvoff), damping always
    const bool off = (s.kvoff[i >> 5] >> (i & 31)) & 1u;
#if DK_CLAMP_IMPLICIT == 2
    s.L[TRI(i, i)] += h * ((off ? 0.f : c_kv[i] * fabsf(s.kvw[i])) + c_damp[i]);
#else
    s.L[TRI(i, i)] += h * ((off ? 0.f : c_kv[i]) + c_damp[i]);
#endif
    s.tmp[i] = s.Ma[i];
  }
  __syncwarp();
  CHOL_M_SOLVE(s.L, s.L, s.tmp, lane, s.cbuf);
  if (rec) {  // tape: L_Q = chol(M + h kv), a* and a_adv (qvel' = qvel + h a_adv), header
    tape_put(rec + OFF_LQ, s.L, NTRI, REC_WORDS - OFF_LQ, lane);
    tape_put(rec + OFF_QACC, s.qacc, NV, NV, lane);
#if DK_CLAMP_IMPLICIT == 2  // (the weights after qacc_adv, written by substep_pre: pad only behind them)
    tape_put(rec + OFF_QADV, s.tmp, NV, NV, lane);
    for (int i = OFF_KVW + NV + lane; i < OFF_LH; i += 32) __stcs(rec + i, 0.f);
#else
    tape_put(rec + OFF_QADV, s.tmp, NV, OFF_LH - OFF_QADV, lane);
#endif
    if (lane == 0) {
      float nu = __int_as_float(s.min_nu_bits);
      unsigned fl = (unsigned)(s.unsupported & 31) | (zchanged ? TAPE_REFACT : 0u) | (conv ? 0u : TAPE_NONCONV)
                    | (nu < TAPE_SINGULAR_NU ? TAPE_SINGULAR : 0u);
#ifdef DK_TAPE_VERIFY
      if (vdiff) fl |= TAPE_VERIFY_DIFF;
#endif
      unsigned hd = (unsigned)ncon | (unsigned)s.nefc << 8 | (unsigned)(s.nefc_dof - NFLOSS) << 18
                    | (unsigned)min(it, 255) << 24;
      __stcs(rec + OFF_HDR, __uint_as_float(hd));
      __stcs(rec + OFF_FLAGS, __uint_as_float(fl));
      __stcs(rec + OFF_MIN_NU, nu);
      __stcs(rec + OFF_RESERVED, 0.f);
    }
  }
  for (int i = lane; i < NV; i += 32) s.qvel[i] += h * s.tmp[i];
  __syncwarp();
  for (int b = 1 + lane; b < NB; b += 32)  // free joints: integrate_root; hinge / slide: q += h qdot
    if (JT_OF(b) == JFREE) integrate_root(s.qpos + c_qadr[b], s.qvel + c_dadr[b], h);
    else s.qpos[c_qadr[b]] += h * s.qvel[c_dadr[b]];
  __syncwarp();
  PROF(11);
}

template <class S>
__device__ __forceinline__ void substep(S& s, const float* ctrl, int iterations, int lane, int* niter_out,
                                        float* rec) {
  substep_pre(s, ctrl, lane, rec);
  substep_post(s, iterations, lane, niter_out, rec);
}

// The split forward (host.py: k_pre, k_post, dk_forward_split): substep_pre and substep_post as two kernels, each
// with its own state (SimP, SimN: only what that half uses, so more worlds fit an SM, and each kernel's code is about
// half of k_substep's). Their members have struct Sim's names, so the device code above runs unchanged on all three;
// arrays that live in global memory are pointers there (struct Hand, one per world: what substep_pre hands to
// substep_post).
// substep_post runs on SimNT<SPLIT_NCON, SPLIT_NEFC> in shared memory when the world's contacts and rows fit (corpora:
// at most 31 contacts and 122 rows), else on a full-capacity SimN in Hand (global memory, the same code).
// Beyond 64 contacts (tabletop scenes: some 90 contacts at rest), the full capacity (SimN of 17 KB at 128 contacts).
#ifndef SPLIT_NCON
#define SPLIT_NCON (MAXCON > 64 ? MAXCON : MAXCON < 32 ? MAXCON : 32)  // k_post's shared-memory capacity: contacts
#endif
#ifndef SPLIT_NEFC
#define SPLIT_NEFC (MAXCON > 64 ? NEFC : NEFC < 128 ? NEFC : 128)  // and constraint rows
#endif
template <int NC, int NR>
struct SimNT {  // substep_post: Newton, implicitfast; capacity NC contacts, NR rows. Members whose lifetimes do not
  float M[NTRI], L[NTRI];  // overlap share storage (unions), the read-only ones the loop seldom reads stay in Hand
  float qacc[NV], Ma[NV];
  union { float grad[NV], p[NV]; };  // grad is last read by p = -grad (in place)
  union { float Mp[NV], tmp[NV]; };  // Mp: line search to update; tmp: J'f (newton_grad), start's qacc, implicitfast
  float efc_D[NR], efc_jar[NR];
  unsigned char efc_kind[NR], efc_a[NR];
  signed char efc_b[NR];
  int ncon, nefc, unsupported, min_nu_bits, nefc_dof;
  union {
    float con_C[NC][CC];  // the Hessian assembly only (J p, the forces and their per-contact sums are dead there)
    alignas(16) float cbuf[CHOL_CB];  // CHOL_SOLVE's scratch: the factorizations (con_C is dead from there on)
    struct {
      union { float con_dot[NC][CD], con_f[NC][CD]; };
      float efc_jv[NR];
    };
  };
  unsigned short con_row[NC];
  unsigned char con_dim[NC], con_nrow[NC], con_nd[NC];
  float *qpos, *qvel;  // the kernel's outputs (implicitfast only)
  const float *qsm, *qacc0, *con_mu, *efc_aref;  // in Hand
  const int* con_pair;
  const float (*con_J)[CD][KDOF];
  const unsigned char (*con_dof)[KDOF];
  const unsigned long long* con_mask;
  const unsigned* kvoff;  // in Hand
#if DK_CLAMP_IMPLICIT == 2
  const float* kvw;  // in Hand
#endif
  float (*oforce)[3];  // the contacts' frame forces at the solution (null: not wanted; Hand::con_force)
#ifdef DK_PROFILE
  long long prof[NPROF], prof_t, prof_u;
#endif
#ifdef DK_TAPE_VERIFY
  float Lv[NTRI];
#endif
};
typedef SimNT<SPLIT_NCON, SPLIT_NEFC> SimN;  // k_post's shared memory
typedef SimNT<MAXCON, NEFC> SimNfull;  // the worlds beyond it (Hand::big)

struct Hand {
  float M[NTRI], qsm[NV], qacc0[NV];
#if HAS_DEFER
  float gx[NGEOMC][3], gq[NGEOMC][4];  // k_pre_a's geom poses, for k_ccd
  unsigned char con_slot[MAXCON];
#endif
  float efc_D[NEFC], efc_aref[NEFC];
  alignas(16) float con_J[MAXCON][CD][KDOF];  // (16-byte aligned: con_jv's 128-bit loads)
  float con_mu[MAXCON];
  int con_pair[MAXCON], con_dim[MAXCON], con_row[MAXCON], con_nrow[MAXCON], con_nd[MAXCON];
  int ncon, nefc, nefc_dof, unsupported, min_nu_bits;
  unsigned char efc_kind[NEFC], efc_a[NEFC];
  signed char efc_b[NEFC];
  alignas(16) unsigned char con_dof[MAXCON][KDOF];
  unsigned long long con_mask[MAXCON];
  unsigned kvoff[(NV + 31) / 32];
#if DK_CLAMP_IMPLICIT == 2
  float kvw[NV];
#endif
  float con_force[MAXCON][3];  // the last substep's contact forces in the contact frames (substep_post)
  float con_dist[MAXCON], con_pos[MAXCON][3], con_frame[MAXCON][9];  // substep_pre's (collision, Jacobians, rows)
  SimNfull big;  // k_post's state of a world whose contacts or rows exceed SimN's capacity
};

struct SimP {  // substep_pre: smooth dynamics, qacc_smooth, collision, contact Jacobians (into Hand), rows (into Hand)
  float Q[NB][4], X[NB][3];
  float W[NB][3], V[NB][3];  // body velocities, then (body forces to contact Jacobians) the hinge motion subspaces
  union {  // the smooth phase's body quantities, then qacc_smooth's factor, then the collision's geoms and candidates
    struct { float AL[NB][3], A[NB][3], F[NB][3], N[NB][3], J[NB][10]; };
    struct { float L[NTRI]; alignas(16) float cbuf[CHOL_CB]; };  // CHOL_SOLVE's scratch
#if DK_PRE2W  // the collision runs on warp 1 concurrently with the smooth phase: its arrays are not shared
  };
  union {
#endif
#if BROAD_GROUPS
    struct { float gx[NGEOMC][3], gq[NGEOMC][4], con_dot[MAXCON][CD]; unsigned short cand[NCAND], gcand[NGRP]; };
#else
    struct { float gx[NGEOMC][3], gq[NGEOMC][4], con_dot[MAXCON][CD]; unsigned short cand[NPAIR]; };
#endif
  };
  float* M;  // in Hand (the stores of the CRB, the loads of qacc_smooth's factorization)
  float qpos[NQ], qvel[NV], zero[NV], qsm[NV], qacc0[NV];
  float com[3], fric;
  float fixq[NFIX > 0 ? NFIX : 1][4], fixx[NFIX > 0 ? NFIX : 1][3];  // the fixtures' placements (com words)
  int ncon, nefc, unsupported, min_nu_bits, nefc_dof;
  int con_pair[MAXCON], con_dim[MAXCON], con_row[MAXCON], con_nrow[MAXCON], con_nd[MAXCON];
  float con_mu[MAXCON];
  unsigned char con_slot[MAXCON];
  float *con_dist, (*con_pos)[3], (*con_frame)[9];  // in Hand
  float (*con_J)[CD][KDOF];
  unsigned char (*con_dof)[KDOF];
  unsigned long long* con_mask;
  unsigned* kvoff;  // in Hand
#if DK_CLAMP_IMPLICIT == 2
  float* kvw;  // in Hand
#endif
  unsigned char *efc_kind, *efc_a;
  signed char* efc_b;
  float *efc_D, *efc_aref;
#if HAS_DEFER
  int2* cvq;  // k_pre_a: the queue of convex pairs (world, pair) for k_ccd, its counter and capacity; null: none
  int *cvn, cvcap, world;
#endif
#ifdef DK_PROFILE
  long long prof[NPROF], prof_t, prof_u;
#endif
};
"""

_v = (os.environ.get("DK_FOREST_HESS_UNROLL") or "").strip()
if _v not in ("", "0", "1"):
    raise ValueError(f"$DK_FOREST_HESS_UNROLL={_v!r}: must be 0 or 1 (or unset)")
if _v:
    STEP_KERNEL = f"#define DK_FOREST_HESS_UNROLL {_v}\n" + STEP_KERNEL


def extra_constants(m):
    """dof -> body map and per-body ancestor bitmasks (bit k: body k is the body or one of its ancestors)."""
    dofbody = np.asarray(m.dof_bodyid, np.float64)
    mask = [0] * m.nbody
    for b in range(m.nbody):
        k = b
        while k > 0:
            mask[b] |= 1 << k
            k = int(m.body_parentid[k])
    if m.nbody > 64:
        return dofbody, np.asarray(mask, object)
    return dofbody, np.asarray(mask, np.uint64 if m.nbody > 32 else np.int64)


EXTRA_HEADER = "DKC int c_dofbody[NV];\nDKC unsigned int c_ancmask[NB];\n"


OBJECT_HOST_HOOKS, OBJECT_ADJ_HOOKS, OBJECT_KIN_HOOKS = "host", "adjoint", "kinematics"


def object_hooks(src, hooks, terrain, m=None):
    """src with the object hooks (OBJECT_*_HOOKS) of a terrain build with object slots, else unchanged (m: the model;
    the host text of a terrain build with deferred pairs gets forest_engine.terrain's Hand hooks too)."""
    if terrain is None:
        return src
    from nablaengine.forest_engine import terrain as _terrain
    return _terrain.object_hooks(src, hooks, terrain, m)

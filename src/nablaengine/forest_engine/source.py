"""CUDA source of the forest engine's smooth dynamics (block per world), generated from the MuJoCo model."""

import mujoco
import numpy as np

CONST_LAYOUT = (
    ("c_parent", "i", "nb", 1), ("c_depth", "i", "nb", 1), ("c_qadr", "i", "nb", 1), ("c_dadr", "i", "nb", 1),
    ("c_nchild", "i", "nb", 1), ("c_child", "i", "nb", 3), ("c_pos", "f", "nb", 3), ("c_quat", "f", "nb", 4),
    ("c_axis", "f", "nb", 3), ("c_ipos", "f", "nb", 3), ("c_iquat", "f", "nb", 4), ("c_mass", "f", "nb", 1),
    ("c_inert", "f", "nb", 3), ("c_grav", "f", "one", 3), ("c_armature", "f", "nv", 1), ("c_nanc", "i", "nb", 1),
    ("c_anc", "i", "nb", 12), ("c_kp", "f", "nv", 1), ("c_kv", "f", "nv", 1), ("c_frclim", "f", "nv", 1),
    ("c_misc", "f", "one", 2),
    ("c_jtype", "i", "nb", 1), ("c_root", "i", "nb", 1), ("c_kq", "f", "nv", 1), ("c_afrc", "f", "nv", 1),
    ("c_act", "i", "nv", 1), ("c_ctrlr", "f", "nv", 2), ("c_dqadr", "i", "nv", 1), ("c_damp", "f", "nv", 1),
    ("c_gcomp", "f", "nb", 1),
)
JFREE, JHINGE, JSLIDE = 0, 1, 2


MAX_CHILDREN = 8
MIN_NV = 1


def children_of(m):
    """[[child body ids] per body]."""
    children = [[] for _ in range(m.nbody)]
    for b in range(1, m.nbody):
        children[m.body_parentid[b]].append(b)
    return children


def child_width(m):
    """c_child's entries per body: 3 (the G1's layout), or the model's most children per body."""
    return max(3, max(len(c) for c in children_of(m)[1:]))


def has_damping(m):
    """(the forest engine: c_damp is always in the layout, model_hooks.py's damping hooks are not applied)"""
    return False


def const_layout(m):
    """CONST_LAYOUT with c_child's and c_anc's widths for model m (child_width, anc_width)."""
    w, wa = child_width(m), anc_width(m)
    return tuple((sym, kind, n, w if sym == "c_child" else wa if sym == "c_anc" else width)
                 for sym, kind, n, width in CONST_LAYOUT)


MAX_ANC = 40
ANC_UNROLLED = 16


def hinge_ancestors(m):
    """[[hinge / slide ancestor body ids, nearest first] per body] (a free joint's body is no such ancestor)."""
    out = [[] for _ in range(m.nbody)]
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    for b in range(1, m.nbody):
        k = m.body_parentid[b]
        while k >= 1:
            if not (m.body_jntnum[k] == 1 and int(m.jnt_type[m.body_jntadr[k]]) == free):
                out[b].append(int(k))
            k = m.body_parentid[k]
    return out


def anc_width(m):
    """c_anc's entries per body: 12 (the G1's layout), 16 for 13-16 hinge ancestors per body, MAX_ANC beyond."""
    na = max((len(a) for a in hinge_ancestors(m)), default=0)
    return 12 if na <= 12 else ANC_UNROLLED if na <= ANC_UNROLLED else MAX_ANC


def anc_hooks(m):
    """The (anchor, replacement, count) edits of the engine text for c_anc's width beyond 12 (the mass matrix's rows
    over a body's hinge ancestors; none for at most 12: the text unchanged). Up to ANC_UNROLLED the ancestors' motion
    subspaces are held in registers (the loops unrolled, all loaded before the first store); beyond, a rolled loop
    loads one ancestor and stores its entry (the same arithmetic per entry: dot3(w_a, n) + dot3(v_a, f))."""
    w = anc_width(m)
    if w == 12:
        return ()
    decl = ("    c_anc[NB * 12];\n", f"    c_anc[NB * {w}];  // (source.anc_hooks: {w} hinge ancestors per body)\n", 1)
    if w <= ANC_UNROLLED:
        return (decl,
                ("    int j = c_dadr[b], na = c_nanc[b], ad[12];\n    float Jb[10], aw[12][3], av[12][3];\n",
                 f"    int j = c_dadr[b], na = c_nanc[b], ad[{w}];  // (source.anc_hooks)\n"
                 f"    float Jb[10], aw[{w}][3], av[{w}][3];\n", 1),
                ("    for (int a = 0; a < 12; a++)  // the hinge ancestors' motion subspaces, all loaded before the first store\n"
                 "      if (a < na) {\n        int k = c_anc[12 * b + a];\n",
                 f"    for (int a = 0; a < {w}; a++)  // the hinge ancestors' motion subspaces, all loaded before the first store\n"
                 f"      if (a < na) {{\n        int k = c_anc[{w} * b + a];\n", 1),
                ("    for (int a = 0; a < 12; a++)\n      if (a < na) s.M[TRI(j, ad[a])]",
                 f"    for (int a = 0; a < {w}; a++)\n      if (a < na) s.M[TRI(j, ad[a])]", 1))
    return (decl,
            ("    int j = c_dadr[b], na = c_nanc[b], ad[12];\n    float Jb[10], aw[12][3], av[12][3];\n",
             "    int j = c_dadr[b], na = c_nanc[b];  // (source.anc_hooks: the ancestors one at a time below)\n"
             "    float Jb[10];\n", 1),
            ("#pragma unroll\n    for (int a = 0; a < 12; a++)  // the hinge ancestors' motion subspaces, all loaded before the "
             "first store\n      if (a < na) {\n        int k = c_anc[12 * b + a];\n        ad[a] = c_dadr[k];\n"
             "        for (int t = 0; t < 3; t++) { aw[a][t] = s.W[k][t]; av[a][t] = s.V[k][t]; }\n      }\n", "", 1),
            ("#pragma unroll\n    for (int a = 0; a < 12; a++)\n      if (a < na) s.M[TRI(j, ad[a])] = dot3(aw[a], n) + "
             "dot3(av[a], f);\n",
             "#pragma unroll 1\n    for (int a = 0; a < na; a++) {  // (source.anc_hooks: more than "
             f"{ANC_UNROLLED} hinge ancestors)\n"
             f"      const int k = c_anc[{w} * b + a];\n"
             "      const float aw[3] = {s.W[k][0], s.W[k][1], s.W[k][2]}, av[3] = {s.V[k][0], s.V[k][1], s.V[k][2]};\n"
             "      s.M[TRI(j, c_dadr[k])] = dot3(aw, n) + dot3(av, f);\n    }\n", 1))


COM_BODY = None


def max_depth(m):
    """The kinematic tree's maximum depth (constants' second result)."""
    depth = [0] * m.nbody
    for b in range(1, m.nbody):
        depth[b] = depth[m.body_parentid[b]] + 1
    return max(depth)


def gcoff_hooks(m):
    """The (anchor, replacement, count) edits of the engine text for gravcomp applied off the COM (fused bodies with
    different gravcomp: nablaengine.model.gravcomp; none without: the text unchanged): c_gcoff (gcoff_constants), the
    moment (R_b d_b) x gc m g in body_force and its VJP in the adjoint's body_local_vjp (N1's extra term: its cotangent
    bWl, and the rotation's e x (f x Wl), e = R_b d_b, f = gc m g)."""
    from nablaengine.model import gravcomp
    if gravcomp.offsets(m) is None:
        return ()
    return (("DKC float c_kq[NV], c_afrc[NV], c_ctrlr[NV * 2], c_damp[NV], c_gcomp[NB];\n",
             "DKC float c_kq[NV], c_afrc[NV], c_ctrlr[NV * 2], c_damp[NV], c_gcomp[NB];\n"
             "DKC float c_gcoff[NB * 3];  // gravcomp's point: offset from the COM, body frame (source.gcoff_hooks)\n", 1),
            ("  for (int k = 0; k < 3; k++) N[k] = Ia[k] + u[k] + cf[k];\n",
             "  for (int k = 0; k < 3; k++) N[k] = Ia[k] + u[k] + cf[k];\n"
             "  {  // gravcomp at its point (source.gcoff_hooks): + (R d) x gc m g\n"
             "    float e[3], f[3], t[3];\n"
             "    qrot(Q, c_gcoff + 3 * b, e);\n"
             "    for (int k = 0; k < 3; k++) f[k] = m * (gc * c_grav[k]);\n"
             "    cross(e, f, t);\n"
             "    for (int k = 0; k < 3; k++) N[k] += t[k];\n"
             "  }\n", 1),
            ("  for (int k = 0; k < 3; k++) { bVl[k] = F1[k]; bWl[k] = IL[k] + t[k] + u[k]; }\n",
             "  for (int k = 0; k < 3; k++) { bVl[k] = F1[k]; bWl[k] = IL[k] + t[k] + u[k]; }\n"
             "  float ge[3], gf[3];  // gravcomp at its point (source.gcoff_hooks): N1 += e x f, e = R d, f = gc m g\n"
             "  qrot(Q, c_gcoff + 3 * b, ge);\n"
             "  for (int k = 0; k < 3; k++) gf[k] = m * (c_gcomp[b] * c_grav[k]);\n"
             "  cross(ge, gf, t);\n"
             "  for (int k = 0; k < 3; k++) bWl[k] += t[k];\n", 1),
            ("  for (int k = 0; k < 3; k++) tb[k] = tau[k] + t[k];\n",
             "  for (int k = 0; k < 3; k++) tb[k] = tau[k] + t[k];\n"
             "  cross(gf, Wl, t);  // (source.gcoff_hooks) e rotates with the body: e x (f x Wl)\n"
             "  cross(ge, t, u);\n"
             "  for (int k = 0; k < 3; k++) tb[k] += u[k];\n", 1))


def gcoff_constants(m):
    """(c_gcoff values float32, its layout entry) for gcoff_hooks' models (None otherwise)."""
    from nablaengine.model import gravcomp
    off = gravcomp.offsets(m)
    if off is None:
        return None
    return np.asarray(off, np.float32).ravel(), ("c_gcoff", "f", "nb", 3)


def constants(m, com_body=None):
    """Model arrays in const_layout(m) order (float32), and the tree's maximum depth; com_body: the name of the body
    that takes the per-world COM offset (default COM_BODY: none)."""
    nb = m.nbody
    depth, children = [0] * nb, children_of(m)
    for b in range(1, nb):
        depth[b] = depth[m.body_parentid[b]] + 1
    assert nb <= 128 and max(len(c) for c in children[1:]) <= MAX_CHILDREN
    assert m.nmocap == 0, "no mocap bodies"
    from nablaengine.forest_engine import tendon
    assert not tendon.check(m), f"unsupported tendons: {tendon.check(m)[:4]}"
    from nablaengine.forest_engine import equality
    assert not equality.check(m), f"unsupported equality constraints: {equality.check(m)[:4]}"
    assert not m.jnt_actgravcomp.any(), "gravcomp as a passive force (no actuatorgravcomp)"
    qadr, dadr, axis, jtype, root = (np.zeros(nb) for _ in range(5))
    axis = np.zeros((nb, 3))
    kinds = {int(mujoco.mjtJoint.mjJNT_FREE): JFREE, int(mujoco.mjtJoint.mjJNT_HINGE): JHINGE,
             int(mujoco.mjtJoint.mjJNT_SLIDE): JSLIDE}
    dqadr = np.zeros(m.nv)
    from nablaengine.forest_engine import model_hooks
    model_hooks.integrator(m)
    for b in range(1, nb):
        j = m.body_jntadr[b]
        assert m.body_jntnum[b] == 1, f"body {b}: exactly one joint per body"
        jt = kinds[int(m.jnt_type[j])]
        assert jt != JFREE or m.body_parentid[b] == 0, f"body {b}: a free joint on a tree root only"
        assert jt == JFREE or not m.jnt_pos[j].any(), f"body {b}: the joint at the body origin"
        assert jt == JFREE or m.qpos0[m.jnt_qposadr[j]] == 0, f"body {b}: joint ref 0 (the engine's angle is qpos)"
        assert jt != JFREE or m.jnt_stiffness[j] == 0, f"body {b}: no free-joint spring"
        qadr[b], dadr[b], axis[b], jtype[b] = m.jnt_qposadr[j], m.jnt_dofadr[j], m.jnt_axis[j], jt
        if jt != JFREE:
            dqadr[m.jnt_dofadr[j]] = m.jnt_qposadr[j]
        k = b
        while m.body_parentid[k] != 0:
            k = m.body_parentid[k]
        root[b] = k
    child = np.zeros((nb, child_width(m)))
    for b in range(1, nb):
        child[b, :len(children[b])] = children[b]
    nchild = np.array([len(children[b]) if b else 0 for b in range(nb)])
    anc, nanc = np.zeros((nb, anc_width(m))), np.zeros(nb)
    for b in range(1, nb):
        k, lst = m.body_parentid[b], []
        while k >= 1:
            if jtype[k] != JFREE:
                lst.append(k)
            k = m.body_parentid[k]
        assert len(lst) <= MAX_ANC
        anc[b, :len(lst)], nanc[b] = lst, len(lst)
    from nablaengine.forest_engine import muscle as _muscle
    kp, kq, kv, frclim, afrc = np.zeros(m.nv), np.zeros(m.nv), np.zeros(m.nv), np.full(m.nv, 1e30), np.full(m.nv, 1e30)
    act, ctrlr = np.full(m.nv, -1.0), np.tile([-1e30, 1e30], (m.nv, 1))
    for a in range(m.nu):
        j = m.actuator_trnid[a, 0]
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON:
            continue
        if _muscle.is_general(m, a):
            continue
        assert m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT, f"actuator {a}: joint / fixed-tendon transmissions"
        assert int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)), f"actuator {a}"
        assert m.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE and m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
        assert int(m.actuator_biastype[a]) in (int(mujoco.mjtBias.mjBIAS_AFFINE), int(mujoco.mjtBias.mjBIAS_NONE))
        assert m.actuator_gear[a, 0] == 1.0 and not m.actuator_gear[a, 1:].any()
        dof = m.jnt_dofadr[j]
        assert act[dof] < 0, f"actuator {a}: one actuator per dof"
        bias = m.actuator_biasprm[a] if m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE else np.zeros(10)
        assert bias[0] == 0.0
        act[dof], kp[dof], kq[dof], kv[dof] = a, m.actuator_gainprm[a, 0], -bias[1], -bias[2]
        if m.actuator_ctrllimited[a]:
            ctrlr[dof] = m.actuator_ctrlrange[a]
        if m.actuator_forcelimited[a]:
            afrc[dof] = m.actuator_forcerange[a, 1]
            assert m.actuator_forcerange[a, 0] == -m.actuator_forcerange[a, 1]
        if m.jnt_actfrclimited[j]:
            frclim[dof] = m.jnt_actfrcrange[j, 1]
            assert m.jnt_actfrcrange[j, 0] == -m.jnt_actfrcrange[j, 1]
    com_body = com_body or COM_BODY
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, com_body) if com_body else -1
    assert com_body is None or torso >= 1, f'no body named "{com_body}": the per-world COM offset (com)'
    from nablaengine.model import gravcomp as _gc
    _off = _gc.offsets(m)
    assert torso < 1 or _off is None or not _off[torso].any(), \
        "the per-world COM offset's body has a gravcomp offset (fused bodies with different gravcomp)"
    parts = [m.body_parentid, depth, qadr, dadr, nchild, child, m.body_pos, m.body_quat, axis, m.body_ipos,
             m.body_iquat, m.body_mass, m.body_inertia, m.opt.gravity, m.dof_armature, nanc, anc, kp, kv, frclim,
             [m.opt.timestep, torso], jtype, root, kq, afrc, act, ctrlr, dqadr, m.dof_damping, m.body_gravcomp]
    return np.concatenate([np.asarray(x, np.float64).ravel() for x in parts]).astype(np.float32), max(depth)


def tree_header(m):
    """The kinematic trees' dof blocks (M is block diagonal over them): NTREE, MAXTD (the largest block), TREE_OF(t, o,
    n) (block t: first dof o, n dofs), TREE_START(i) (the first dof of dof i's block) and TREE_ID(i) (its tree), for
    chol_solve_trees and chol_solve_comps."""
    adr, num = [int(x) for x in m.tree_dofadr], [int(x) for x in m.tree_dofnum]
    of = " ".join(f"if ((t) == {t}) {{ (o) = {a}; (n) = {k}; }}" for t, (a, k) in enumerate(zip(adr, num)))
    start = "0"
    for a in adr[1:]:
        start = f"((i) >= {a} ? {a} : {start})"
    tid = "0"
    for t, a in enumerate(adr[1:], 1):
        tid = f"((i) >= {a} ? {t} : {tid})"
    return (f"#define NTREE {len(adr)}\n#define MAXTD {max(num) if num else 0}\n#define TREE_ID(i) {tid}\n"
            f"#define TREE_OF(t, o, n) do {{ (o) = 0; (n) = 0; {of} }} while (0)\n#define TREE_START(i) {start}\n")


def _body_set(bs, var="b"):
    if not bs:
        return "0"
    if len(bs) <= 3:
        return "(" + " || ".join(f"({var}) == {b}" for b in bs) + ")"
    if max(bs) < 64:
        return f"((int)((0x{sum(1 << b for b in bs):x}ULL >> ({var})) & 1ULL))"
    lo, hi = sum(1 << b for b in bs if b < 64), sum(1 << (b - 64) for b in bs if b >= 64)
    return (f"((int)(({var}) < 64 ? (0x{lo:x}ULL >> ({var})) & 1ULL : (0x{hi:x}ULL >> (({var}) - 64)) & 1ULL))")


def joint_header(m):
    """The model's joint types at compile time (the forest specialization): JT_OF(b), the joint type of body b >= 1
    as a constant expression of b (JT_FREE_B / JT_SLIDE_B: the model's free / slide bodies, 0 when it has none), so the
    branches on joint types the model does not have fold away; DQADR(i) / ACT_OF(i), the qpos address of hinge / slide dof
    i and its joint actuator (-1: none), in closed form where the model's layout allows it (c_dqadr / c_act otherwise).
    -DDK_FOREST_SPEC=0 reads all of them from the tables (the generic build, for bisection; the same bits)."""
    jt = {int(mujoco.mjtJoint.mjJNT_FREE): JFREE, int(mujoco.mjtJoint.mjJNT_HINGE): JHINGE,
          int(mujoco.mjtJoint.mjJNT_SLIDE): JSLIDE}
    kind = [None] + [jt.get(int(m.jnt_type[m.body_jntadr[b]])) if m.body_jntnum[b] == 1 else None
                     for b in range(1, m.nbody)]
    free = [b for b in range(1, m.nbody) if kind[b] == JFREE]
    slide = [b for b in range(1, m.nbody) if kind[b] == JSLIDE]
    fd = sorted(int(m.jnt_dofadr[m.body_jntadr[b]]) for b in free)
    dq = {int(m.jnt_dofadr[j]): int(m.jnt_qposadr[j]) for j in range(m.njnt) if int(m.jnt_type[j]) != 0}
    dqadr, free_dof = "(c_dqadr[i])", ""
    if len(fd) <= 3 and all(q == i + sum(i >= f + 6 for f in fd) for i, q in dq.items()):
        dqadr = "((i)" + "".join(f" + ((i) >= {f + 6})" for f in fd) + ")"
        pred = " || ".join(f"((i) >= {f} && (i) < {f + 6})" for f in fd) or "0"
        free_dof = f"#define JT_FREE_DOF(i) ({pred})  // (with the closed-form DQADR only)\n"
    act = {}
    from nablaengine.forest_engine import muscle
    for a in range(m.nu):
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and not muscle.is_general(m, a):
            act[int(m.jnt_dofadr[m.actuator_trnid[a, 0]])] = a
    act_of = "(c_act[i])"
    if act:
        d0, a0, n = min(act), act[min(act)], len(act)
        if sorted(act) == list(range(d0, d0 + n)) and all(act[d] == d - d0 + a0 for d in act):
            hi = "" if d0 + n == m.nv else f" && (i) < {d0 + n}"
            act_of = f"((i) >= {d0}{hi} ? (i) - {d0 - a0} : -1)" if d0 or hi else f"((i) - {d0 - a0})"
    else:
        act_of = "(-1)"
    return f"""#ifndef DK_FOREST_SPEC
#define DK_FOREST_SPEC 1  // the model's joint types and dof maps at compile time (source.joint_header); 0: the tables
#endif
#if DK_FOREST_SPEC
#define JT_FREE_B(b) {_body_set(free)}
#define JT_SLIDE_B(b) {_body_set(slide)}
#define JT_OF(b) (JT_FREE_B(b) ? JFREE : JT_SLIDE_B(b) ? JSLIDE : JHINGE)
#define DQADR(i) {dqadr}
{free_dof}#define ACT_OF(i) {act_of}
#else
#define JT_OF(b) (c_jtype[b])
#define DQADR(i) (c_dqadr[i])
#define ACT_OF(i) (c_act[i])
#endif
"""


def header(m, maxd, wpb):
    return f"""
#include <cuda_runtime.h>
#define NB {m.nbody}
#define NQ {m.nq}
#define NV {m.nv}
#define NU {m.nu}
#define MAXD {maxd}
#define WPB {wpb}
#define NTRI (NV * (NV + 1) / 2)
#define NCH {child_width(m)}
#define TRI(i, j) ((i) * ((i) + 1) / 2 + (j))  // packed lower triangle, j <= i
""" + tree_header(m) + r"""
#ifndef DKC
#define DKC __device__  // where the model arrays live: global memory (L1-cached), as lanes index them by pair, body or
                       // dof (divergent addresses serialize in __constant__ memory); -DDKC=__constant__ for comparison
#endif
DKC int c_parent[NB], c_depth[NB], c_qadr[NB], c_dadr[NB], c_nchild[NB], c_child[NB * NCH], c_nanc[NB],
    c_anc[NB * 12];
DKC float c_pos[NB * 3], c_quat[NB * 4], c_axis[NB * 3], c_ipos[NB * 3], c_iquat[NB * 4], c_mass[NB],
    c_inert[NB * 3], c_grav[3], c_armature[NV], c_kp[NV], c_kv[NV], c_frclim[NV], c_misc[2];
DKC int c_jtype[NB], c_root[NB], c_act[NV], c_dqadr[NV];  // the forest's (CONST_LAYOUT)
DKC float c_kq[NV], c_afrc[NV], c_ctrlr[NV * 2], c_damp[NV], c_gcomp[NB];
#define JFREE 0
#define JHINGE 1
#define JSLIDE 2
""" + joint_header(m)


DEVICE = r"""
__device__ __forceinline__ void qmul(const float* a, const float* b, float* o) {
  float w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  float x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  float y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  float z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
  o[0] = w; o[1] = x; o[2] = y; o[3] = z;
}
__device__ __forceinline__ void qrot(const float* q, const float* v, float* o) {  // o = q v q^-1
  float tx = 2.f * (q[2] * v[2] - q[3] * v[1]), ty = 2.f * (q[3] * v[0] - q[1] * v[2]), tz = 2.f * (q[1] * v[1] - q[2] * v[0]);
  o[0] = v[0] + q[0] * tx + q[2] * tz - q[3] * ty;
  o[1] = v[1] + q[0] * ty + q[3] * tx - q[1] * tz;
  o[2] = v[2] + q[0] * tz + q[1] * ty - q[2] * tx;
}
__device__ __forceinline__ void qrot_inv(const float* q, const float* v, float* o) {
  float qc[4] = {q[0], -q[1], -q[2], -q[3]};
  qrot(qc, v, o);
}
__device__ __forceinline__ void cross(const float* a, const float* b, float* o) {
  o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
}
__device__ __forceinline__ float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Explicit roundings. The adjoint replays the forward's body poses, collision-geom poses and contact geometry (dist,
// pos, frame) and needs them bitwise (adjoint.py adj_poses / adj_contacts), in another kernel and, in the JAX library,
// another compilation unit. Written as plain C (qmul, qrot, ... above), the mul / add pairs are fused into FMAs by NVVM
// and ptxas depending on the surrounding code (the uses of a product, register pressure, -fmad, the architecture), so
// the same source can give results an ulp apart in two kernels. These intrinsics are never contracted (and __fmaf_rn
// is never split): an expression written with them has its IEEE value, the same in every kernel, compilation unit and
// -fmad setting (on one GPU: rsqrtf and sincosf are the architecture's; IEEE division and sqrtf, nvcc's default
// -prec-div / -prec-sqrt, are correctly rounded and need nothing). Every operation the replay reproduces is written
// with them: the *_rn helpers here and fwd_body (the poses, and with them the twists and accelerations), step.py's
// geom_pose, the contact geometry that collide_pair shares with the adjoint's adj_contact_geom (make_frame,
// sphere_sphere, plane_point, plane_capsule_frame, capsule_end, segment_point, capcap_*), the bounding tests, the limit
// test (limit_dists) and the rows' impedance (kbimp, imp_raw, contact_invw, efc_D). The forms are the ones ptxas chose
// for the plain C on sm_70, so the change left the V100 results bitwise as they were: a b + c d = fma(a, b, c d),
// a b - c d = fma(a, b, -(c d)), x +- a b = fma(+-a, b, x), sums from the left; a product with a second use is rounded
// first (W = Wp + round(qd aw) in fwd_body, p +- round(l a) for the two ends of a capsule). The rest of the engine
// (dynamics, solver, the adjoint's VJPs) keeps the plain C helpers above: it is not replayed. Test: the forward or the
// adjoint unit built with -fmad=false: the replay stays bitwise the forward's
#define RMUL(a, b) __fmul_rn((a), (b))
#define RADD(a, b) __fadd_rn((a), (b))
#define RSUB(a, b) __fsub_rn((a), (b))
#define RFMA(a, b, c) __fmaf_rn((a), (b), (c))   // a b + c, one rounding
#define RFMS(a, b, c) __fmaf_rn(-(a), (b), (c))  // c - a b, one rounding

__device__ __forceinline__ void qmul_rn(const float* a, const float* b, float* o) {
  float w = RFMS(a[3], b[3], RFMS(a[2], b[2], RFMA(a[0], b[0], -RMUL(a[1], b[1]))));
  float x = RFMS(a[3], b[2], RFMA(a[2], b[3], RFMA(a[0], b[1], RMUL(a[1], b[0]))));
  float y = RFMA(a[3], b[1], RFMA(a[2], b[0], RFMA(a[0], b[2], -RMUL(a[1], b[3]))));
  float z = RFMA(a[3], b[0], RFMS(a[2], b[1], RFMA(a[0], b[3], RMUL(a[1], b[2]))));
  o[0] = w; o[1] = x; o[2] = y; o[3] = z;
}
// o = q v q^-1: t = 2 qv x v, o = v + q0 t + qv x t
__device__ __forceinline__ void qrot_rn(const float* q, const float* v, float* o) {
  float tx = RMUL(2.f, RFMA(q[2], v[2], -RMUL(q[3], v[1])));
  float ty = RMUL(2.f, RFMA(q[3], v[0], -RMUL(q[1], v[2])));
  float tz = RMUL(2.f, RFMA(q[1], v[1], -RMUL(q[2], v[0])));
  o[0] = RFMS(q[3], ty, RFMA(q[2], tz, RFMA(q[0], tx, v[0])));
  o[1] = RFMS(q[1], tz, RFMA(q[3], tx, RFMA(q[0], ty, v[1])));
  o[2] = RFMS(q[2], tx, RFMA(q[1], ty, RFMA(q[0], tz, v[2])));
}
__device__ __forceinline__ void cross_rn(const float* a, const float* b, float* o) {
  o[0] = RFMA(a[1], b[2], -RMUL(a[2], b[1]));
  o[1] = RFMA(a[2], b[0], -RMUL(a[0], b[2]));
  o[2] = RFMA(a[0], b[1], -RMUL(a[1], b[0]));
}
__device__ __forceinline__ float dot3_rn(const float* a, const float* b) {
  return RFMA(a[2], b[2], RFMA(a[0], b[0], RMUL(a[1], b[1])));
}
__device__ __forceinline__ void zaxis_rn(const float* q, float* a) {  // a = q e_z: a plane's normal, a capsule's axis
  const float ez[3] = {0.f, 0.f, 1.f};
  qrot_rn(q, ez, a);
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// fwd_body's three steps for a hinge body (b >= 2), in its order: the origin X and the parent-composed rotation q0 from
// the parent's pose and the body's constant frame (qb = c_quat + 4 b, pb = c_pos + 3 b); the hinge rotation ha at the
// joint angle (axis ax = c_axis + 3 b), which alone does not depend on the parent; the rest (Q = q0 ha, the twist, the
// accelerations). Split so that a sweep can compute fwd_hinge_rot (and load the constants) for every body before it
// starts (g1_engine.kinematics: the kinematics outputs, the same bits as fwd_body); fwd_body runs them as one
__device__ __forceinline__ void fwd_hinge_pos(const float* Qp, const float* Xp, const float* qb, const float* pb,
                                              float* q0, float* X) {
  float t[3];
  qmul_rn(Qp, qb, q0);
  qrot_rn(Qp, pb, t);
  for (int k = 0; k < 3; k++) X[k] = RADD(Xp[k], t[k]);
}
__device__ __forceinline__ void fwd_hinge_rot(float ang, const float* ax, float* ha) {
  float s, c;
  sincosf(RMUL(0.5f, ang), &s, &c);  // bitwise sinf, cosf with one argument reduction (V100: 4% faster forward)
  ha[0] = c; ha[1] = RMUL(s, ax[0]); ha[2] = RMUL(s, ax[1]); ha[3] = RMUL(s, ax[2]);
}
__device__ __forceinline__ void fwd_hinge_rest(const float* q0, const float* ha, const float* ax, float qd, float qdd,
                                               const float* Xp, const float* Wp, const float* Vp, const float* ALp,
                                               const float* Ap, float* Q, const float* X, float* W, float* V,
                                               float* AL, float* A) {
  qmul_rn(q0, ha, Q);
  float aw[3], awd[3], r[3], u[3], u2[3], wr[3];
  qrot_rn(Q, ax, aw);
  for (int k = 0; k < 3; k++) { r[k] = RSUB(X[k], Xp[k]); awd[k] = RMUL(aw[k], qd); W[k] = RADD(Wp[k], awd[k]); }
  cross_rn(Wp, r, u);
  for (int k = 0; k < 3; k++) V[k] = RADD(Vp[k], u[k]);
  cross_rn(Wp, awd, u);
  for (int k = 0; k < 3; k++) AL[k] = RADD(RFMA(aw[k], qdd, ALp[k]), u[k]);
  cross_rn(ALp, r, u);
  cross_rn(Wp, r, wr);
  cross_rn(Wp, wr, u2);
  for (int k = 0; k < 3; k++) A[k] = RADD(RADD(Ap[k], u[k]), u2[k]);
}

// forward kinematics and velocity / acceleration propagation of body b from its parent's state. Explicit roundings
// (RMUL ...): the forward (substep_pre, with qacc = 0) and the adjoint (adj_poses, at a*) compute the same bits
__device__ __forceinline__ void fwd_body(int b, const float* Qp, const float* Xp, const float* Wp, const float* Vp,
                                         const float* ALp, const float* Ap, const float* qpos, const float* qvel,
                                         const float* qacc, float* Q, float* X, float* W, float* V, float* AL,
                                         float* A) {
  const int jt = JT_OF(b);
  if (jt == JFREE) {  // free root: world position, quaternion, world linear / local angular velocity
    const float* qp = qpos + c_qadr[b];
    const float* qv = qvel + c_dadr[b];
    const float* qa = qacc + c_dadr[b];
    float n = rsqrtf(RFMA(qp[6], qp[6], RFMA(qp[5], qp[5], RFMA(qp[3], qp[3], RMUL(qp[4], qp[4])))));
    for (int k = 0; k < 4; k++) Q[k] = RMUL(qp[3 + k], n);
    for (int k = 0; k < 3; k++) { X[k] = qp[k]; V[k] = qv[k]; A[k] = RSUB(qa[k], c_grav[k]); }
    qrot_rn(Q, qv + 3, W);
    qrot_rn(Q, qa + 3, AL);
    return;
  }
  float q0[4];
  fwd_hinge_pos(Qp, Xp, c_quat + 4 * b, c_pos + 3 * b, q0, X);
  float qj = qpos[c_qadr[b]], qd = qvel[c_dadr[b]], qdd = qacc[c_dadr[b]];
  if (jt == JSLIDE) {  // slide: X += R ax q; W = Wp, AL = ALp; V, A of the moving origin
    float aw[3], r[3], u[3], u2[3], wr[3], awd[3], c2[3];
    for (int k = 0; k < 4; k++) Q[k] = q0[k];
    qrot_rn(Q, c_axis + 3 * b, aw);
    for (int k = 0; k < 3; k++) { X[k] = RFMA(aw[k], qj, X[k]); r[k] = RSUB(X[k], Xp[k]); awd[k] = RMUL(aw[k], qd); }
    for (int k = 0; k < 3; k++) { W[k] = Wp[k]; AL[k] = ALp[k]; }
    cross_rn(Wp, r, u);
    for (int k = 0; k < 3; k++) V[k] = RADD(RADD(Vp[k], u[k]), awd[k]);
    cross_rn(ALp, r, u);
    cross_rn(Wp, r, wr);
    cross_rn(Wp, wr, u2);
    cross_rn(Wp, awd, c2);
    for (int k = 0; k < 3; k++)
      A[k] = RFMA(aw[k], qdd, RADD(RADD(RADD(Ap[k], u[k]), u2[k]), RMUL(2.f, c2[k])));
    return;
  }
  float ha[4];
  fwd_hinge_rot(qj, c_axis + 3 * b, ha);
  fwd_hinge_rest(q0, ha, c_axis + 3 * b, qd, qdd, Xp, Wp, Vp, ALp, Ap, Q, X, W, V, AL, A);
}

// force F and moment N about the body origin that body b alone needs (Newton-Euler at its COM)
__device__ __forceinline__ void body_force(int b, const float* Q, const float* W, const float* AL, const float* A,
                                           const float* ipos, float* F, float* N) {
  float cvec[3], u[3], u2[3], wc[3];
  qrot(Q, ipos, cvec);
  cross(AL, cvec, u);
  cross(W, cvec, wc);
  cross(W, wc, u2);
  float m = c_mass[b], gc = c_gcomp[b];  // gravcomp: the force m (a - g) + gc m g (A holds a - g)
  for (int k = 0; k < 3; k++) F[k] = m * (A[k] + u[k] + u2[k] + gc * c_grav[k]);
  float qi[4], l[3], Iw[3], Ia[3], cf[3];
  qmul(Q, c_iquat + 4 * b, qi);
  qrot_inv(qi, W, l);
  for (int k = 0; k < 3; k++) l[k] *= c_inert[3 * b + k];
  qrot(qi, l, Iw);
  qrot_inv(qi, AL, l);
  for (int k = 0; k < 3; k++) l[k] *= c_inert[3 * b + k];
  qrot(qi, l, Ia);
  cross(W, Iw, u);
  cross(cvec, F, cf);
  for (int k = 0; k < 3; k++) N[k] = Ia[k] + u[k] + cf[k];
}

__device__ __forceinline__ void accumulate(const float* Xb, const float* Xc, const float* Fc, const float* Nc, float* F,
                                           float* N) {
  float r[3] = {Xc[0] - Xb[0], Xc[1] - Xb[1], Xc[2] - Xb[2]}, u[3];
  cross(r, Fc, u);
  for (int k = 0; k < 3; k++) { F[k] += Fc[k]; N[k] += Nc[k] + u[k]; }
}

__device__ __forceinline__ void torques(int b, const float* Q, const float* F, const float* N, float* qfrc) {
  const int jt = JT_OF(b);
  if (jt == JFREE) {
    float l[3];
    qrot_inv(Q, N, l);
    float* o = qfrc + c_dadr[b];
    for (int k = 0; k < 3; k++) { o[k] = F[k]; o[3 + k] = l[k]; }
  } else {
    float aw[3];
    qrot(Q, c_axis + 3 * b, aw);
    qfrc[c_dadr[b]] = dot3(aw, jt == JSLIDE ? F : N);
  }
}

// spatial inertia of body b about the point O, packed as J (xx yy zz xy xz yz), h = m c (3), m; c = com - O
__device__ __forceinline__ void body_inertia(int b, const float* Q, const float* X, const float* O, const float* ipos,
                                             float* J) {
  float qi[4], R[9], c[3], t[3];
  qmul(Q, c_iquat + 4 * b, qi);
  for (int k = 0; k < 3; k++) {
    float e[3] = {0.f, 0.f, 0.f};
    e[k] = 1.f;
    qrot(qi, e, R + 3 * k);  // column k of the inertia frame
  }
  const float* I = c_inert + 3 * b;
  J[0] = I[0] * R[0] * R[0] + I[1] * R[3] * R[3] + I[2] * R[6] * R[6];
  J[1] = I[0] * R[1] * R[1] + I[1] * R[4] * R[4] + I[2] * R[7] * R[7];
  J[2] = I[0] * R[2] * R[2] + I[1] * R[5] * R[5] + I[2] * R[8] * R[8];
  J[3] = I[0] * R[0] * R[1] + I[1] * R[3] * R[4] + I[2] * R[6] * R[7];
  J[4] = I[0] * R[0] * R[2] + I[1] * R[3] * R[5] + I[2] * R[6] * R[8];
  J[5] = I[0] * R[1] * R[2] + I[1] * R[4] * R[5] + I[2] * R[7] * R[8];
  qrot(Q, ipos, t);
  for (int k = 0; k < 3; k++) c[k] = X[k] + t[k] - O[k];
  float m = c_mass[b], cc = dot3(c, c);
  J[0] += m * (cc - c[0] * c[0]); J[1] += m * (cc - c[1] * c[1]); J[2] += m * (cc - c[2] * c[2]);
  J[3] -= m * c[0] * c[1]; J[4] -= m * c[0] * c[2]; J[5] -= m * c[1] * c[2];
  J[6] = m * c[0]; J[7] = m * c[1]; J[8] = m * c[2]; J[9] = m;
}

// (n, f) = I (w, v) for a packed spatial inertia about O; n = J w + h x v, f = m v - h x w
__device__ __forceinline__ void apply_inertia(const float* J, const float* w, const float* v, float* n, float* f) {
  float hv[3], hw[3];
  cross(J + 6, v, hv);
  cross(J + 6, w, hw);
  n[0] = J[0] * w[0] + J[3] * w[1] + J[4] * w[2] + hv[0];
  n[1] = J[3] * w[0] + J[1] * w[1] + J[5] * w[2] + hv[1];
  n[2] = J[4] * w[0] + J[5] * w[1] + J[2] * w[2] + hv[2];
  for (int k = 0; k < 3; k++) f[k] = J[9] * v[k] - hw[k];
}

// motion subspace of dof t of body k about O: w (angular), v (velocity of the point O); a free body's O is its origin
__device__ __forceinline__ void dof_motion(int k, int t, const float* Q, const float* X, const float* O, float* w,
                                           float* v) {
  if (JT_OF(k) == JSLIDE) {
    qrot(Q, c_axis + 3 * k, v);
    w[0] = w[1] = w[2] = 0.f;
    return;
  }
  if (JT_OF(k) == JFREE) {
    float e[3] = {0.f, 0.f, 0.f};
    if (t < 3) { e[t] = 1.f; w[0] = w[1] = w[2] = 0.f; v[0] = e[0]; v[1] = e[1]; v[2] = e[2]; return; }
    e[t - 3] = 1.f;
    qrot(Q, e, w);  // local angular axes about the root origin = O
    v[0] = v[1] = v[2] = 0.f;
    return;
  }
  float r[3] = {O[0] - X[0], O[1] - X[1], O[2] - X[2]};
  qrot(Q, c_axis + 3 * k, w);
  cross(w, r, v);
}

// COM offset of body b in its frame, with the per-world torso offset (domain randomization) applied
__device__ __forceinline__ void body_ipos(int b, const float* com, float* out) {
  for (int k = 0; k < 3; k++) out[k] = c_ipos[3 * b + k] + (b == (int)c_misc[1] ? com[k] : 0.f);
}

// in-place Cholesky A = L L^T (lower triangle of a dense NV x NV shared matrix) by one warp, left-looking (Crout):
// column k from the finished columns to its left; lane l computes rows k+1+l, k+33+l, ... as read-only dot products
// (row stride NV = 35 keeps the lanes on distinct banks; row k is a broadcast)
__device__ __forceinline__ void chol_warp(float* A, int lane) {
  for (int k = 0; k < NV; k++) {
    const float* Ak = A + k * NV;
    float part = 0.f;
    for (int j = lane; j < k; j += 32) part += Ak[j] * Ak[j];
    float d = sqrtf(fmaxf(Ak[k] - warp_sum(part), 1e-30f)), inv = 1.f / d;
    for (int i = k + 1 + lane; i < NV; i += 32) {
      const float* Ai = A + i * NV;
      float a0 = Ai[k], a1 = 0.f, a2 = 0.f, a3 = 0.f;
      int j = 0;
      for (; j + 4 <= k; j += 4) {
        a0 -= Ai[j] * Ak[j]; a1 -= Ai[j + 1] * Ak[j + 1]; a2 -= Ai[j + 2] * Ak[j + 2]; a3 -= Ai[j + 3] * Ak[j + 3];
      }
      for (; j < k; j++) a0 -= Ai[j] * Ak[j];
      A[i * NV + k] = ((a0 + a1) + (a2 + a3)) * inv;
    }
    __syncwarp();
    if (lane == 0) A[k * NV + k] = d;
    __syncwarp();
  }
}

// Cholesky factorization and solve in registers: L L^T = A (A: packed lower triangle TRI(i, j) in shared memory, read
// before anything is written, so L may alias A), then x = A^-1 b in place (b: shared NV vector). Lane l holds rows l and
// l + 32 (NV <= 64) in registers; the factorization is right-looking with column k broadcast by __shfl_sync, all loops
// unrolled at compile time. L (packed lower triangle, with the true diagonal) is written to L for the back substitution
// and for callers that keep it.
#ifndef CHOL_INLINE
#define CHOL_INLINE __noinline__  // one copy of the (large, fully unrolled) code: -DCHOL_INLINE=__forceinline__ inlines it
#endif
__device__ CHOL_INLINE void chol_solve_reg(const float* A, float* L, float* b, int lane) {
  static_assert(NV <= 64, "two rows per lane");
  const int r0 = lane, r1 = lane + 32;
  const bool has1 = r1 < NV;
  float a0[NV], a1[NV], dinv[NV];
#pragma unroll
  for (int j = 0; j < NV; j++) { a0[j] = j <= r0 ? A[TRI(r0, j)] : 0.f; a1[j] = has1 && j <= r1 ? A[TRI(r1, j)] : 0.f; }
  __syncwarp();
#pragma unroll
  for (int k = 0; k < NV; k++) {
    float dkk = fmaxf(__shfl_sync(0xffffffffu, k < 32 ? a0[k] : a1[k], k & 31), 1e-30f);
    float inv = rsqrtf(dkk);
    inv = inv * (1.5f - 0.5f * dkk * inv * inv);  // one Newton step: about full float precision
    float d = dkk * inv;
    dinv[k] = inv;
    if (r0 > k) a0[k] *= inv; else if (r0 == k) a0[k] = d;
    if (r1 > k) a1[k] *= inv; else if (r1 == k) a1[k] = d;
#pragma unroll
    for (int j = k + 1; j < NV; j++) {  // unpredicated: above the diagonal this updates the symmetric Schur complement
      float ljk = __shfl_sync(0xffffffffu, j < 32 ? a0[k] : a1[k], j & 31);  // (bounded, never read as L)
      a0[j] -= a0[k] * ljk;
      a1[j] -= a1[k] * ljk;
    }
  }
#pragma unroll
  for (int j = 0; j < NV; j++) { if (j <= r0) L[TRI(r0, j)] = a0[j]; if (has1 && j <= r1) L[TRI(r1, j)] = a1[j]; }
  float y0 = b[r0], y1 = has1 ? b[r1] : 0.f;
#pragma unroll
  for (int k = 0; k < NV; k++) {  // forward: L y = b, column-wise
    float yk = __shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31) * dinv[k];
    if (r0 == k) y0 = yk; else if (r0 > k) y0 -= a0[k] * yk;
    if (r1 == k) y1 = yk; else if (r1 > k) y1 -= a1[k] * yk;
  }
  __syncwarp();
  // backward: L^T x = y; lane i holds y_i and reads row k of L (contiguous) to update its entries i < k
#pragma unroll
  for (int k = NV - 1; k >= 0; k--) {
    float xk = __shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31) * dinv[k];
    if (r0 == k) y0 = xk; else if (r0 < k) y0 -= L[TRI(k, r0)] * xk;
    if (r1 == k) y1 = xk; else if (r1 < k) y1 -= L[TRI(k, r1)] * xk;
  }
  b[r0] = y0;
  if (has1) b[r1] = y1;
  __syncwarp();
}

// Compact variant of chol_solve_reg (same contract): the column loop is not unrolled. Lane registers a0 / a1 hold the
// trailing part of rows lane / lane + 32, shifted left by one entry per step (a0[m] = row(k + m) at step k), so every
// register index stays a compile-time constant; finished columns of L go to shared memory as they are produced. The
// trailing update runs in groups of 8 columns and stops after the last group that holds a column. Small code (no
// instruction-fetch stalls) at the price of more executed instructions than the fully unrolled version.
__device__ __noinline__ void chol_solve_compact(const float* A, float* L, float* b, int lane) {
  static_assert(NV <= 64, "two rows per lane");
  const int r0 = lane, r1 = lane + 32;
  const bool has1 = r1 < NV;
  float a0[NV], a1[NV];
#pragma unroll
  for (int j = 0; j < NV; j++) { a0[j] = j <= r0 ? A[TRI(r0, j)] : 0.f; a1[j] = has1 && j <= r1 ? A[TRI(r1, j)] : 0.f; }
  __syncwarp();
  for (int k = 0; k < NV; k++) {
    float dkk = fmaxf(__shfl_sync(0xffffffffu, k < 32 ? a0[0] : a1[0], k & 31), 1e-30f);
    float inv = rsqrtf(dkk);
    inv = inv * (1.5f - 0.5f * dkk * inv * inv);
    float d = dkk * inv;
    float l0 = r0 > k ? a0[0] * inv : (r0 == k ? d : 0.f), l1 = r1 > k ? a1[0] * inv : (r1 == k ? d : 0.f);
    if (r0 >= k) L[TRI(r0, k)] = l0;
    if (has1 && r1 >= k) L[TRI(r1, k)] = l1;
    if (r0 == k) l0 = 0.f;  // the pivot row takes no part in the trailing update
    if (r1 == k) l1 = 0.f;
#pragma unroll
    for (int g = 1; g < NV; g += 8) {
      if (k + g >= NV) break;  // uniform: no column left in this group
#pragma unroll
      for (int m = g; m < g + 8 && m < NV; m++) {
        int j = k + m;
        float ljk = __shfl_sync(0xffffffffu, j < 32 ? l0 : l1, j & 31);
        a0[m - 1] = a0[m] - l0 * ljk;
        a1[m - 1] = a1[m] - l1 * ljk;
      }
    }
  }
  __syncwarp();
  float y0 = b[r0], y1 = has1 ? b[r1] : 0.f;
  for (int k = 0; k < NV; k++) {  // forward: L y = b, column-wise
    float yk = __shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31) / L[TRI(k, k)];
    if (r0 == k) y0 = yk; else if (r0 > k) y0 -= L[TRI(r0, k)] * yk;
    if (r1 == k) y1 = yk; else if (has1 && r1 > k) y1 -= L[TRI(r1, k)] * yk;
  }
  for (int k = NV - 1; k >= 0; k--) {  // backward: L^T x = y
    float xk = __shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31) / L[TRI(k, k)];
    if (r0 == k) y0 = xk; else if (r0 < k) y0 -= L[TRI(k, r0)] * xk;
    if (r1 == k) y1 = xk; else if (has1 && r1 < k) y1 -= L[TRI(k, r1)] * xk;
  }
  b[r0] = y0;
  if (has1) b[r1] = y1;
  __syncwarp();
}

// Broadcast variant (same contract, plus cb: 72 floats of 16-byte aligned shared scratch), for 32 < NV <= 35. The
// shuffles of the column broadcast (one per entry of the trailing update in chol_solve_reg / _compact) are the cost of
// those two; here column k goes through shared memory once (one store per lane) and is read back as float4 broadcasts,
// four entries per load. Rows 0..31 are in registers (lane r: row r, shifted left by one entry per step as in
// chol_solve_compact); the NT = NV - 32 tail rows are held transposed (lane j: A[32 + t][j]) and their own NT x NT
// block redundantly on every lane, so a lane updates its row's entries left of column 32 only. The forward solve runs
// inside the factorization (lane k broadcasts y_k with column k), and the pivot chain does not wait for the broadcast:
// lane k + 1 forms its next diagonal from its own entries. The backward solve reads L back by rows (one load per step).
// L is the same factor as chol_solve_compact's (same update order), the solve multiplies by the refined reciprocal
// pivots.
// Cycles per factor + solve of a 35 x 35 matrix, 4096 worlds (a standalone microbenchmark): V100 at 3 per SM 8.2k
// (chol_solve_compact 31.6k, _reg 16.2k), RTX 3090 at 2 per SM 8.9k (51.9k, 19.3k); in the engine (V100) the
// qacc_smooth phase went from 44.8k to 10.8k cycles.
// CHOL_BCAST_WIDE: chol_solve_bcast_wide (below) for 36 <= NV <= 48; with it chol_back32 / chol_resolve (NEWTON_REUSE_WIDE)
#ifndef CHOL_BCAST_WIDE
#define CHOL_BCAST_WIDE (NV >= 36 && NV <= 48)
#endif
#if NV > 32 && NV <= 35 || CHOL_BCAST_WIDE
#define CHOL_NT (NV - 32)  // tail rows
// The arithmetic below is written with explicit roundings (FMS: c - a b in one fma; MUL): ptxas otherwise fuses
// mul / add pairs depending on the calling kernel, and k_substep, k_pre and k_post must agree bitwise
#define FMS(a, b, c) __fmaf_rn(-(a), (b), (c))
#define MUL(a, b) __fmul_rn((a), (b))
// rsqrtf for a normal argument (x >= 2^-126): the bare MUFU.RSQ without rsqrtf's scaling of denormals (same result)
__device__ __forceinline__ float rsqrt_normal(float x) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
// backward substitution L' x = y on rows 0..31 in blocks of 8 (lane r: y_r in, x_r out; the contributions of rows 32..
// already subtracted): a block's own triangle is solved redundantly on every lane (broadcast loads of L; rinv: the
// reciprocal pivots in shared memory), then each lane updates its row: one round of shuffles per block, not per row
__device__ __forceinline__ float chol_back32(const float* L, const float* rinv, float y, int lane) {
#pragma unroll
  for (int kb = 24; kb >= 0; kb -= 8) {
    float xb[8];
#pragma unroll
    for (int j = 0; j < 8; j++) xb[j] = __shfl_sync(0xffffffffu, y, kb + j);
#pragma unroll
    for (int j = 7; j >= 0; j--) {
      float t = xb[j];
#pragma unroll
      for (int i = j + 1; i < 8; i++) t = FMS(L[TRI(kb + i, kb + j)], xb[i], t);
      xb[j] = MUL(t, rinv[kb + j]);
    }
#pragma unroll
    for (int j = 0; j < 8; j++)
      y = lane < kb ? FMS(L[TRI(kb + j, min(lane, kb + j))], xb[j], y) : (lane == kb + j ? xb[j] : y);
  }
  return y;
}

#if NV <= 35
__device__ __noinline__ void chol_solve_bcast(const float* A, float* L, float* b, int lane, float* cb) {
  static_assert(NV > 32 && NV <= 35, "a tail of 1..3 rows");
  const int r = lane;
  float a[32];  // a[m] = A[r][k + m] at step k (entries right of the diagonal: bounded garbage, never read as L)
#pragma unroll
  for (int j = 0; j < 32; j++) a[j] = j <= r ? A[TRI(r, j)] : 0.f;
  float T[CHOL_NT], tb[CHOL_NT * (CHOL_NT + 1) / 2], bt[CHOL_NT];  // A[32 + t][r]; the tail block (packed); b tail
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) {
    T[t] = A[TRI(32 + t, r)];
    bt[t] = b[32 + t];
#pragma unroll
    for (int u = 0; u <= t; u++) tb[t * (t + 1) / 2 + u] = A[TRI(32 + t, 32 + u)];
  }
  float y = b[r], myinv = 0.f;
  __syncwarp();
  float dn = a[0];  // the next pivot (lane k at step k): the diagonal after the updates so far
  // unrolled by 2 (unroll 1: V100 13.7 against 12.7 ms per control step, RTX 3090 7.59 against 7.46; a test per group
  // of the row update, or phases with fewer groups as the trailing part shrinks, i.e. more code: V100 13.0 / 15.2 ms)
#pragma unroll 2
  for (int k = 0; k < 32; k++) {
    float* buf = cb + (k & 1) * 36;  // double buffered: [0, 4) L[32 + t][k], y_k; 4 + (j - k - 1): L[j][k], k < j < 32
    float dkk = fmaxf(__shfl_sync(0xffffffffu, dn, k), 1e-30f);
    float inv = rsqrt_normal(dkk);
    inv = MUL(inv, FMS(MUL(MUL(0.5f, dkk), inv), inv, 1.5f));  // one Newton step: about full float precision
    float d = MUL(dkk, inv);
    float l = r > k ? MUL(a[0], inv) : (r == k ? d : 0.f);
    if (r >= k) L[TRI(r, k)] = l;
    if (r == k) {
      myinv = inv;
      y = MUL(y, inv);
      float tw[4] = {0.f, 0.f, 0.f, y};
#pragma unroll
      for (int t = 0; t < CHOL_NT; t++) { T[t] = MUL(T[t], inv); L[TRI(32 + t, k)] = T[t]; tw[t] = T[t]; }
      *reinterpret_cast<float4*>(buf) = make_float4(tw[0], tw[1], tw[2], tw[3]);
      l = 0.f;  // the pivot row takes no part in the trailing update
    }
    if (r > k) buf[4 + r - k - 1] = l;
    dn = FMS(l, l, a[1]);  // lane k + 1: its next diagonal (= a[0] after the shift below)
    __syncwarp();
    float4 tv = *reinterpret_cast<const float4*>(buf);
    float tl[3] = {tv.x, tv.y, tv.z};
    y = FMS(l, tv.w, y);
#pragma unroll
    for (int t = 0; t < CHOL_NT; t++) {
      T[t] = FMS(tl[t], l, T[t]);
      bt[t] = FMS(tl[t], tv.w, bt[t]);
#pragma unroll
      for (int u = 0; u <= t; u++) tb[t * (t + 1) / 2 + u] = FMS(tl[t], tl[u], tb[t * (t + 1) / 2 + u]);
    }
#pragma unroll
    for (int g = 1; g < 32; g += 4) {  // all groups every step (skipping the second half once k >= 16, one uniform
      float4 c = *reinterpret_cast<const float4*>(buf + 4 + g - 1);  // branch, was slower: RTX 3090 9.9k against 8.9k)
      a[g - 1] = FMS(l, c.x, a[g]);
      if (g + 1 < 32) a[g] = FMS(l, c.y, a[g + 1]);
      if (g + 2 < 32) a[g + 1] = FMS(l, c.z, a[g + 2]);
      if (g + 3 < 32) a[g + 2] = FMS(l, c.w, a[g + 3]);
    }
  }
  float dt[CHOL_NT], it[CHOL_NT], xt[CHOL_NT];  // the tail block's factor and solve, on every lane
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) {
    float s = tb[t * (t + 1) / 2 + t];
#pragma unroll
    for (int u = 0; u < t; u++) s = FMS(tb[t * (t + 1) / 2 + u], tb[t * (t + 1) / 2 + u], s);
    float dkk = fmaxf(s, 1e-30f), inv = rsqrt_normal(dkk);
    inv = MUL(inv, FMS(MUL(MUL(0.5f, dkk), inv), inv, 1.5f));
    dt[t] = MUL(dkk, inv); it[t] = inv;
#pragma unroll
    for (int v = t + 1; v < CHOL_NT; v++) {
      float e = tb[v * (v + 1) / 2 + t];
#pragma unroll
      for (int u = 0; u < t; u++) e = FMS(tb[v * (v + 1) / 2 + u], tb[t * (t + 1) / 2 + u], e);
      tb[v * (v + 1) / 2 + t] = MUL(e, inv);
    }
    float yt = bt[t];
#pragma unroll
    for (int u = 0; u < t; u++) yt = FMS(tb[t * (t + 1) / 2 + u], xt[u], yt);
    xt[t] = MUL(yt, inv);  // forward
  }
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < CHOL_NT; t++) {
#pragma unroll
      for (int u = 0; u < t; u++) L[TRI(32 + t, 32 + u)] = tb[t * (t + 1) / 2 + u];
      L[TRI(32 + t, 32 + t)] = dt[t];
    }
  }
#pragma unroll
  for (int t = CHOL_NT - 1; t >= 0; t--) {  // backward on the tail
    float x = xt[t];
#pragma unroll
    for (int v = t + 1; v < CHOL_NT; v++) x = FMS(tb[v * (v + 1) / 2 + t], xt[v], x);
    xt[t] = MUL(x, it[t]);
  }
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) y = FMS(T[t], xt[t], y);
  __syncwarp();
  // backward on rows 0..31: L' x = y, column by column, unrolled by 4 (fully: V100 12.73 against 12.44 ms, RTX 3090 the
  // same; chol_back32 was slower here)
#pragma unroll 4
  for (int k = 31; k >= 0; k--) {
    float xk = __shfl_sync(0xffffffffu, MUL(y, myinv), k);
    float lk = r < k ? L[TRI(k, r)] : 0.f;
    y = r == k ? xk : FMS(lk, xk, y);
  }
  b[r] = y;
  if (lane == 0) {  // (lane t writing entry t became a dynamically indexed local array)
#pragma unroll
    for (int t = 0; t < CHOL_NT; t++) b[32 + t] = xt[t];
  }
  __syncwarp();
}
#endif  // NV <= 35

// x = (L L')^-1 b in place (b: shared NV vector) with a factor L from CHOL_SOLVE (packed lower triangle in shared
// memory, true diagonal), for a Newton iteration whose Hessian did not change; cb: scratch as for chol_solve_bcast.
// Blocks of 8 rows as in chol_back32 (forward likewise), the tail rows 32.. on every lane. V100: 3.0k cycles (column by
// column with one broadcast per row: 6.3k)
// CHOL_BCAST_WIDE: inlined. As a call, ptxas 12.4 (sm_70, sm_86) restored a register the callee needed by recomputing
// the value one call site had in it (the lane) also for k_post's other site (Hand::big), where it held the upper half
// of a pointer: an illegal address after the re-use
#if CHOL_BCAST_WIDE
__device__ __forceinline__ void chol_resolve(const float* L, float* b, int lane, float* cb) {
#else
__device__ __noinline__ void chol_resolve(const float* L, float* b, int lane, float* cb) {
#endif
  const int r = lane;
  float* rinv = cb;
  rinv[r] = 1.f / L[TRI(r, r)];
  if (r < CHOL_NT) rinv[32 + r] = 1.f / L[TRI(32 + r, 32 + r)];
  const int r1 = min(32 + r, NV - 1);  // lanes 0 .. CHOL_NT - 1: tail row 32 + r (y1); others a valid dummy row
  float y = b[r], y1 = b[r1];
  const int tr = TRI(r, 0), tr1 = TRI(r1, 0);
  __syncwarp();
#pragma unroll
  for (int kb = 0; kb < 32; kb += 8) {  // forward: L y = b on rows 0..31
    float yb[8];
#pragma unroll
    for (int j = 0; j < 8; j++) yb[j] = __shfl_sync(0xffffffffu, y, kb + j);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float t = yb[j];
#pragma unroll
      for (int i = 0; i < j; i++) t = FMS(L[TRI(kb + j, kb + i)], yb[i], t);
      yb[j] = MUL(t, rinv[kb + j]);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      y = r >= kb + 8 ? FMS(L[tr + min(kb + j, r)], yb[j], y) : (r == kb + j ? yb[j] : y);
      y1 = FMS(L[tr1 + kb + j], yb[j], y1);
    }
  }
  float yt[CHOL_NT];  // the tail rows on every lane: forward, backward, then their part of rows 0..31
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) {
    float u = __shfl_sync(0xffffffffu, y1, t);
#pragma unroll
    for (int i = 0; i < t; i++) u = FMS(L[TRI(32 + t, 32 + i)], yt[i], u);
    yt[t] = MUL(u, rinv[32 + t]);
  }
#pragma unroll
  for (int t = CHOL_NT - 1; t >= 0; t--) {
    float u = yt[t];
#pragma unroll
    for (int i = t + 1; i < CHOL_NT; i++) u = FMS(L[TRI(32 + i, 32 + t)], yt[i], u);
    yt[t] = MUL(u, rinv[32 + t]);
  }
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) y = FMS(L[TRI(32 + t, r)], yt[t], y);
  y = chol_back32(L, rinv, y, lane);
  b[r] = y;
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < CHOL_NT; t++) b[32 + t] = yt[t];
  }
  __syncwarp();
}
#undef FMS
#undef MUL
#endif

// Wide broadcast variant (same contract as chol_solve_bcast; cb: CHOL_CB floats of 16-byte aligned shared scratch) for
// 36 <= NV <= 48, e.g. a humanoid with a free object. Rows 0..31 as in chol_solve_bcast (lane r: row r in registers,
// shifted, column k broadcast through the double-buffered scratch together with the tail entries L[32 + t][k] and
// y_k), the WT = NV - 32 tail rows held transposed (lane r: A[32 + t][r]), but without the tail block on every lane:
// after the 32 steps the Schur complement S = A22 - L21 L21' and the tail's right-hand side b2 - L21 y1 are formed once,
// one entry per lane and round (dot products of length 32 over the rows of L21 in shared memory, in the order of the
// rank-1 updates: the same roundings as updating them step by step), written in place of A22 / b2; then lane t < WT
// factors and solves row t of S as chol_solve_small does (column broadcast through the scratch), and the backward
// solve runs column by column (tail, then rows 0..31). -DCHOL_BCAST_WIDE=0: the previous CHOL_SOLVE for these sizes
// (chol_solve_compact below Ampere, chol_solve_reg from Ampere on)
#if CHOL_BCAST_WIDE
#define CHOL_WT (NV - 32)                      // tail rows
#define CHOL_WH ((CHOL_WT + 4) / 4 * 4)        // a step's header: L[32 + t][k] (t < WT), y_k, padded to float4
#define CHOL_WS (CHOL_WH + 32)                 // one buffer: the header, then L[j][k] at CHOL_WH + (j - k - 1)
#define CHOL_CB (2 * CHOL_WS)                  // the scratch (Sim::cbuf and the like)
#define WFMS(a, b, c) __fmaf_rn(-(a), (b), (c))  // explicit roundings, as chol_solve_bcast
#define WMUL(a, b) __fmul_rn((a), (b))
__device__ __forceinline__ float rsqrt_normal_w(float x) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __noinline__ void chol_solve_bcast_wide(const float* A, float* L, float* b, int lane, float* cb) {
  static_assert(NV >= 36 && NV <= 48, "a tail of 4..16 rows");
  const int r = lane;
  float a[32];  // a[m] = A[r][k + m] at step k (entries right of the diagonal: bounded garbage, never read as L)
#pragma unroll
  for (int j = 0; j < 32; j++) a[j] = j <= r ? A[TRI(r, j)] : 0.f;
  float T[CHOL_WT];  // A[32 + t][r], updated to L[32 + t][r]
#pragma unroll
  for (int t = 0; t < CHOL_WT; t++) T[t] = A[TRI(32 + t, r)];
  float y = b[r], myinv = 0.f;
  __syncwarp();
  float dn = a[0];  // the next pivot (lane k at step k)
#pragma unroll 2
  for (int k = 0; k < 32; k++) {
    float* buf = cb + (k & 1) * CHOL_WS;
    float dkk = fmaxf(__shfl_sync(0xffffffffu, dn, k), 1e-30f);
    float inv = rsqrt_normal_w(dkk);
    inv = WMUL(inv, WFMS(WMUL(WMUL(0.5f, dkk), inv), inv, 1.5f));  // one Newton step
    float d = WMUL(dkk, inv);
    float l = r > k ? WMUL(a[0], inv) : (r == k ? d : 0.f);
    if (r >= k) L[TRI(r, k)] = l;
    if (r == k) {
      myinv = inv;
      y = WMUL(y, inv);
      float tw[CHOL_WH];
#pragma unroll
      for (int t = 0; t < CHOL_WH; t++) tw[t] = 0.f;
#pragma unroll
      for (int t = 0; t < CHOL_WT; t++) { T[t] = WMUL(T[t], inv); L[TRI(32 + t, k)] = T[t]; tw[t] = T[t]; }
      tw[CHOL_WT] = y;
#pragma unroll
      for (int t = 0; t < CHOL_WH; t += 4)
        *reinterpret_cast<float4*>(buf + t) = make_float4(tw[t], tw[t + 1], tw[t + 2], tw[t + 3]);
      l = 0.f;  // the pivot row takes no part in the trailing update
    }
    if (r > k) buf[CHOL_WH + r - k - 1] = l;
    dn = WFMS(l, l, a[1]);  // lane k + 1: its next diagonal
    __syncwarp();
    float tl[CHOL_WH];
#pragma unroll
    for (int t = 0; t < CHOL_WH; t += 4) {
      float4 v = *reinterpret_cast<const float4*>(buf + t);
      tl[t] = v.x; tl[t + 1] = v.y; tl[t + 2] = v.z; tl[t + 3] = v.w;
    }
    y = WFMS(l, tl[CHOL_WT], y);
#pragma unroll
    for (int t = 0; t < CHOL_WT; t++) T[t] = WFMS(tl[t], l, T[t]);
#pragma unroll
    for (int g = 1; g < 32; g += 4) {
      float4 c = *reinterpret_cast<const float4*>(buf + CHOL_WH + g - 1);
      a[g - 1] = WFMS(l, c.x, a[g]);
      if (g + 1 < 32) a[g] = WFMS(l, c.y, a[g + 1]);
      if (g + 2 < 32) a[g + 1] = WFMS(l, c.z, a[g + 2]);
      if (g + 3 < 32) a[g + 2] = WFMS(l, c.w, a[g + 3]);
    }
  }
  b[r] = y;  // y_r of rows 0..31, read by the tail's right-hand side
  __syncwarp();
  // S = A22 - L21 L21' (packed, entry e < WT (WT + 1) / 2) and b2 - L21 y1 (entry WT (WT + 1) / 2 + t): each entry read
  // and written by one lane, in place (A22 / b2 are read nowhere else, L21 and y1 are in columns / entries < 32)
  constexpr int NS = CHOL_WT * (CHOL_WT + 1) / 2, NE = NS + CHOL_WT;
#pragma unroll
  for (int e0 = 0; e0 < NE; e0 += 32) {
    const int e = e0 + lane;
    if (e < NE) {
      int t = 0;
#pragma unroll
      for (int i = 1; i < CHOL_WT; i++) t = e >= i * (i + 1) / 2 ? i : t;
      const bool rhs = e >= NS;
      const int tr = rhs ? e - NS : t, u = e - t * (t + 1) / 2;
      const float* p = L + TRI(32 + tr, 0);
      const float* q = rhs ? b : L + TRI(32 + u, 0);
      float* out = rhs ? b + 32 + tr : L + TRI(32 + t, 32 + u);
      float s = rhs ? b[32 + tr] : A[TRI(32 + t, 32 + u)];
#pragma unroll
      for (int j = 0; j < 32; j++) s = WFMS(p[j], q[j], s);
      *out = s;
    }
  }
  __syncwarp();
  // the tail: lane t < WT holds row t of S (unshifted; the loop is unrolled), the forward solve inside the factorization
  const bool own = r < CHOL_WT;
  float s[CHOL_WT];
#pragma unroll
  for (int j = 0; j < CHOL_WT; j++) s[j] = own && j <= r ? L[TRI(32 + r, 32 + j)] : 0.f;
  float yt = own ? b[32 + r] : 0.f, tinv = 0.f;
  __syncwarp();
  dn = s[0];
#pragma unroll
  for (int k = 0; k < CHOL_WT; k++) {
    float* buf = cb + (k & 1) * 20;  // [0]: y_k; [j - k]: L[32 + j][32 + k], k < j < WT
    float dkk = fmaxf(__shfl_sync(0xffffffffu, dn, k), 1e-30f);
    float inv = rsqrt_normal_w(dkk);
    inv = WMUL(inv, WFMS(WMUL(WMUL(0.5f, dkk), inv), inv, 1.5f));
    float d = WMUL(dkk, inv);
    float l = r > k ? WMUL(s[k], inv) : (r == k ? d : 0.f);
    if (own && r >= k) L[TRI(32 + r, 32 + k)] = l;
    if (r == k) {
      tinv = inv;
      yt = WMUL(yt, inv);
      buf[0] = yt;
      l = 0.f;
    }
    if (own && r > k) buf[r - k] = l;
    if (k + 1 < CHOL_WT) dn = WFMS(l, l, s[k + 1]);
    __syncwarp();
    yt = WFMS(l, buf[0], yt);
#pragma unroll
    for (int j = k + 1; j < CHOL_WT; j++) s[j] = WFMS(l, buf[j - k], s[j]);
  }
  __syncwarp();
#pragma unroll
  for (int k = CHOL_WT - 1; k >= 0; k--) {  // backward on the tail: L22' x2 = y2, column by column
    float xk = __shfl_sync(0xffffffffu, WMUL(yt, tinv), k);
    float lk = r < k ? L[TRI(32 + k, 32 + r)] : 0.f;
    yt = r == k ? xk : WFMS(lk, xk, yt);
  }
#pragma unroll
  for (int t = CHOL_WT - 1; t >= 0; t--) y = WFMS(T[t], __shfl_sync(0xffffffffu, yt, t), y);  // y1 - L21' x2
  // backward on rows 0..31, as chol_solve_bcast
#pragma unroll 4
  for (int k = 31; k >= 0; k--) {
    float xk = __shfl_sync(0xffffffffu, WMUL(y, myinv), k);
    float lk = r < k ? L[TRI(k, r)] : 0.f;
    y = r == k ? xk : WFMS(lk, xk, y);
  }
  b[r] = y;
  if (own) b[32 + r] = yt;
  __syncwarp();
}
#undef WFMS
#undef WMUL
#endif
#ifndef CHOL_CB
#define CHOL_CB 72  // CHOL_SOLVE's scratch (floats): chol_solve_bcast's two buffers of 36
#endif

#if NV <= 32
// chol_solve_small: as chol_solve_bcast (lane r: row r of A in registers, shifted one entry per step so that the
// entry the step needs is a[0]; the column of L broadcast through the double-buffered scratch cb; the forward solve
// during the factorization, then the backward one column by column), for NV <= 32 rows and no tail. Explicit roundings
// (k_substep, k_pre and k_post must agree bitwise)
#define SFMS(a, b, c) __fmaf_rn(-(a), (b), (c))
#define SMUL(a, b) __fmul_rn((a), (b))
__device__ __noinline__ void chol_solve_small(const float* A, float* L, float* b, int lane, float* cb) {
  const int r = lane;
  const bool own = r < NV;
  float a[32];  // a[m] = A[r][k + m] at step k (entries right of the diagonal: bounded garbage, never read as L)
#pragma unroll
  for (int j = 0; j < 32; j++) a[j] = (own && j <= r && j < NV) ? A[TRI(r, j)] : 0.f;
  float y = own ? b[r] : 0.f, myinv = 0.f;
  __syncwarp();
  float dn = a[0];  // the next pivot (lane k at step k)
#pragma unroll 2
  for (int k = 0; k < NV; k++) {
    float* buf = cb + (k & 1) * 36;  // [0]: y_k; [1 + (j - k - 1)]: L[j][k], k < j < 32
    float dkk = fmaxf(__shfl_sync(0xffffffffu, dn, k), 1e-30f);
    float inv;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(dkk));
    inv = SMUL(inv, SFMS(SMUL(SMUL(0.5f, dkk), inv), inv, 1.5f));  // one Newton step
    float d = SMUL(dkk, inv);
    float l = r > k ? SMUL(a[0], inv) : (r == k ? d : 0.f);
    if (own && r >= k) L[TRI(r, k)] = l;
    if (r == k) {
      myinv = inv;
      y = SMUL(y, inv);
      buf[0] = y;
      l = 0.f;  // the pivot row takes no part in the trailing update
    }
    if (r > k) buf[r - k] = l;
    dn = SFMS(l, l, a[1]);
    __syncwarp();
    y = SFMS(l, buf[0], y);
#pragma unroll
    for (int g = 1; g < 32; g++) a[g - 1] = SFMS(l, buf[g], a[g]);  // (buf[g] past the rows: zeros from those lanes)
  }
  __syncwarp();
#pragma unroll 4
  for (int k = NV - 1; k >= 0; k--) {  // backward: L' x = y
    float xk = __shfl_sync(0xffffffffu, SMUL(y, myinv), k);
    float lk = (r < k) ? L[TRI(k, r)] : 0.f;
    y = r == k ? xk : SFMS(lk, xk, y);
  }
  if (own) b[r] = y;
  __syncwarp();
}
#undef SFMS
#undef SMUL
#endif

// chol_solve_trees: the solve with a block-diagonal A, one block per kinematic tree (M and M + h kv: no coupling
// between trees without equality constraints or tendons), same contract as CHOL_SOLVE (A == L allowed). Lane t factors
// block t in registers (fully unrolled to MAXTD rows, a shorter block padded with identity rows: the same instructions
// on every lane), solves its part of b, and writes its block of L; the entries of L outside the blocks are zero.
// Explicit roundings (k_substep, k_pre and k_post must agree bitwise)
#if NTREE <= 32 && MAXTD <= 12
#define TFMS(a, b, c) __fmaf_rn(-(a), (b), (c))
#define TMUL(a, b) __fmul_rn((a), (b))
__device__ __noinline__ void chol_solve_trees(const float* A, float* L, float* b, int lane) {
  int o, n;
  TREE_OF(lane, o, n);
  float a[MAXTD * (MAXTD + 1) / 2], y[MAXTD], rinv[MAXTD];
#pragma unroll
  for (int i = 0; i < MAXTD; i++) {
#pragma unroll
    for (int j = 0; j <= i; j++) a[TRI(i, j)] = i < n ? A[TRI(o + i, o + j)] : (i == j ? 1.f : 0.f);
    y[i] = i < n ? b[o + i] : 0.f;
  }
  __syncwarp();
  for (int i = lane; i < NV; i += 32)  // the entries outside the blocks (zero in A too: A == L is safe)
    for (int j = 0, j1 = TREE_START(i); j < j1; j++) L[TRI(i, j)] = 0.f;
#pragma unroll
  for (int j = 0; j < MAXTD; j++) {  // factor (Cholesky-Crout, column j), with the forward solve
    float s = a[TRI(j, j)];
#pragma unroll
    for (int k = 0; k < j; k++) s = TFMS(a[TRI(j, k)], a[TRI(j, k)], s);
    float dkk = fmaxf(s, 1e-30f), inv;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(dkk));
    inv = TMUL(inv, TFMS(TMUL(TMUL(0.5f, dkk), inv), inv, 1.5f));  // one Newton step
    a[TRI(j, j)] = TMUL(dkk, inv);
    rinv[j] = inv;
#pragma unroll
    for (int i = j + 1; i < MAXTD; i++) {
      float e = a[TRI(i, j)];
#pragma unroll
      for (int k = 0; k < j; k++) e = TFMS(a[TRI(i, k)], a[TRI(j, k)], e);
      a[TRI(i, j)] = TMUL(e, inv);
    }
    float t = y[j];
#pragma unroll
    for (int k = 0; k < j; k++) t = TFMS(a[TRI(j, k)], y[k], t);
    y[j] = TMUL(t, inv);
  }
#pragma unroll
  for (int i = MAXTD - 1; i >= 0; i--) {  // backward: L' x = y
    float t = y[i];
#pragma unroll
    for (int k = i + 1; k < MAXTD; k++) t = TFMS(a[TRI(k, i)], y[k], t);
    y[i] = TMUL(t, rinv[i]);
  }
#pragma unroll
  for (int i = 0; i < MAXTD; i++) {
    if (i < n) {
      b[o + i] = y[i];
#pragma unroll
      for (int j = 0; j <= i; j++) L[TRI(o + i, o + j)] = a[TRI(i, j)];
    }
  }
  __syncwarp();
}
#undef TFMS
#undef TMUL
#endif

// which Cholesky the engine uses for M (CHOL_M_SOLVE: the smooth acceleration, implicitfast): chol_solve_trees by
// default where the trees allow it (two or more, at most 12 dofs each; -DCHOL_TREES=0: CHOL_SOLVE)
#ifndef CHOL_TREES
#define CHOL_TREES (NTREE >= 2 && NTREE <= 32 && MAXTD <= 12)
#endif

// which Cholesky the engine uses: CHOL_SOLVE(A, L, b, lane, cb) (cb: scratch for chol_solve_bcast). -DCHOL_BCAST=1
// (the default where NV allows it) selects chol_solve_bcast (V100 20.5 -> 12.4 ms per mjlab control step at 10
// iterations, RTX 3090 11.1 -> 7.5 ms, with NEWTON_REUSE); with -DCHOL_BCAST=0, -DCHOL_COMPACT=1 / =0 forces one of the
// others, by default the compact one on Volta (V100: 26 vs 34 ms) and the unrolled one from Ampere on (RTX 3090: 14.7
// vs 17.9 ms)
#ifndef CHOL_BCAST
#define CHOL_BCAST (NV > 32 && NV <= 35)
#endif
#ifndef CHOL_COMPACT
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define CHOL_COMPACT 1
#else
#define CHOL_COMPACT 0
#endif
#endif
#ifndef CHOL_SMALL  // chol_solve_small below Ampere (a 22-dof two-arm robot: the Cholesky phases 3x faster than _compact),
// from Ampere on chol_solve_reg as the tree engine (RTX 3090: forest / tree forward 1.01-1.07 -> 0.91-0.98 on go1, go2, h1,
// spot), except the build targets sm_90 / sm_90a until verified there
// (DK_FOREST_SM90_SMALL: nablaengine.cuda.forest_hold_defines)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(DK_FOREST_SM90_SMALL)
#define CHOL_SMALL 0
#else
#define CHOL_SMALL (NV <= 32)
#endif
#endif
#if CHOL_BCAST
#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_bcast(A, L, b, lane, cb)
#elif CHOL_BCAST_WIDE
#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_bcast_wide(A, L, b, lane, cb)
#elif CHOL_SMALL
#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_small(A, L, b, lane, cb)
#elif CHOL_COMPACT
#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_compact(A, L, b, lane)
#else
#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_reg(A, L, b, lane)
#endif
// chol_solve_comps: CHOL_SOLVE for a Hessian whose trees are coupled by some contacts only (same contract, A == L
// allowed). The coupling is read from A (tree blocks with a nonzero entry), its transitive closure gives the connected
// components of trees; the natural-order factor of A is zero between components and, within one, the factor of the
// component's submatrix in ascending dof order, so each component is factored on its own: by its lowest tree's lane in
// registers (gathered, unrolled to CB rows, padded with identity rows) when every component has at most CB dofs, else
// the whole matrix by CHOL_SOLVE (a world with a large component). Explicit roundings as chol_solve_trees
#ifndef CB
#define CB 12
#endif
#if NTREE <= 32 && MAXTD <= CB
#define CFMS(a, b, c) __fmaf_rn(-(a), (b), (c))
#define CMUL(a, b) __fmul_rn((a), (b))
__device__ __noinline__ void chol_solve_comps(const float* A, float* L, float* b, int lane, float* cb) {
  unsigned* adj = reinterpret_cast<unsigned*>(cb);  // adj[t]: the trees coupled to tree t
  if (lane < NTREE) adj[lane] = 1u << lane;
  __syncwarp();
#pragma unroll
  for (int i0 = 0; i0 < NV; i0 += 32) {  // row i: the tree blocks left of its own with a nonzero entry
    const int i = i0 + lane;
    if (i < NV) {
      const int ti = TREE_ID(i), j1 = TREE_START(i);
      unsigned bits = 0u;
      for (int j = 0; j < j1; j++)
        if (A[TRI(i, j)] != 0.f) bits |= 1u << (TREE_ID(j));
      if (bits) {
        atomicOr(adj + ti, bits);
        for (unsigned v = bits; v; v &= v - 1) atomicOr(adj + (__ffs(v) - 1), 1u << ti);
      }
    }
  }
  __syncwarp();
  unsigned cm = lane < NTREE ? adj[lane] : 0u;  // closure: cm = the component of tree lane
  for (bool more = true; more;) {
    unsigned c2 = cm;
#pragma unroll
    for (int t = 0; t < NTREE; t++) {
      unsigned v = __shfl_sync(0xffffffffu, cm, t);
      if ((cm >> t) & 1u) c2 |= v;
    }
    more = __any_sync(0xffffffffu, c2 != cm);
    cm = c2;
  }
  const bool rep = lane < NTREE && (cm & ((1u << lane) - 1u)) == 0u;  // the component's lowest tree
  int n = 0;
#pragma unroll
  for (int t = 0; t < NTREE; t++) {
    int o, k;
    TREE_OF(t, o, k);
    if (rep && ((cm >> t) & 1u)) n += k;
  }
  if (__any_sync(0xffffffffu, n > CB)) {  // a large component: the dense factorization
    CHOL_SOLVE(A, L, b, lane, cb);
    return;
  }
  int idx[CB];  // the component's dofs, ascending (trees are contiguous ascending dof ranges)
#pragma unroll
  for (int i = 0; i < CB; i++) {
    int rem = i, d = 0;
    bool found = false;
#pragma unroll
    for (int t = 0; t < NTREE; t++) {
      int o, k;
      TREE_OF(t, o, k);
      if (!found && ((cm >> t) & 1u)) {
        if (rem < k) { d = o + rem; found = true; } else rem -= k;
      }
    }
    idx[i] = d;
  }
  float a[CB * (CB + 1) / 2], y[CB], rinv[CB];
#pragma unroll
  for (int i = 0; i < CB; i++) {
#pragma unroll
    for (int j = 0; j <= i; j++) a[TRI(i, j)] = i < n ? A[TRI(idx[i], idx[j])] : (i == j ? 1.f : 0.f);
    y[i] = i < n ? b[idx[i]] : 0.f;
  }
  if (A != L) {  // the entries between components (zero in A)
    __syncwarp();
#pragma unroll
    for (int i0 = 0; i0 < NV; i0 += 32) {
      const int i = i0 + lane;
      const unsigned ci = __shfl_sync(0xffffffffu, cm, i < NV ? TREE_ID(i) : 0);
      if (i < NV)
        for (int j = 0; j < i; j++)
          if (!((ci >> (TREE_ID(j))) & 1u)) L[TRI(i, j)] = 0.f;
    }
  }
#pragma unroll
  for (int j = 0; j < CB; j++) {
    float s = a[TRI(j, j)];
#pragma unroll
    for (int k = 0; k < j; k++) s = CFMS(a[TRI(j, k)], a[TRI(j, k)], s);
    float dkk = fmaxf(s, 1e-30f), inv;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(dkk));
    inv = CMUL(inv, CFMS(CMUL(CMUL(0.5f, dkk), inv), inv, 1.5f));
    a[TRI(j, j)] = CMUL(dkk, inv);
    rinv[j] = inv;
#pragma unroll
    for (int i = j + 1; i < CB; i++) {
      float e = a[TRI(i, j)];
#pragma unroll
      for (int k = 0; k < j; k++) e = CFMS(a[TRI(i, k)], a[TRI(j, k)], e);
      a[TRI(i, j)] = CMUL(e, inv);
    }
    float t = y[j];
#pragma unroll
    for (int k = 0; k < j; k++) t = CFMS(a[TRI(j, k)], y[k], t);
    y[j] = CMUL(t, inv);
  }
#pragma unroll
  for (int i = CB - 1; i >= 0; i--) {
    float t = y[i];
#pragma unroll
    for (int k = i + 1; k < CB; k++) t = CFMS(a[TRI(k, i)], y[k], t);
    y[i] = CMUL(t, rinv[i]);
  }
  __syncwarp();
#pragma unroll
  for (int i = 0; i < CB; i++) {
    if (i < n) {
      b[idx[i]] = y[i];
#pragma unroll
      for (int j = 0; j <= i; j++) L[TRI(idx[i], idx[j])] = a[TRI(i, j)];
    }
  }
  __syncwarp();
}
#undef CFMS
#undef CMUL
#endif

// the Newton Hessian's (CHOL_H_SOLVE): chol_solve_comps by default where the trees allow it (two or more, NV > 32: the
// smaller models' CHOL_SOLVE is chol_solve_small; -DCHOL_COMPS=0: CHOL_SOLVE)
#ifndef CHOL_COMPS
#define CHOL_COMPS (NTREE >= 2 && NTREE <= 32 && MAXTD <= CB && NV > 32)
#endif
#if CHOL_COMPS
#define CHOL_H_SOLVE(A, L, b, lane, cb) chol_solve_comps(A, L, b, lane, cb)
#else
#define CHOL_H_SOLVE(A, L, b, lane, cb) CHOL_SOLVE(A, L, b, lane, cb)
#endif
#if CHOL_TREES
#define CHOL_M_SOLVE(A, L, b, lane, cb) chol_solve_trees(A, L, b, lane)
#else
#define CHOL_M_SOLVE(A, L, b, lane, cb) CHOL_SOLVE(A, L, b, lane, cb)
#endif

// solve L L^T x = b in place (b in shared memory) with the factor from chol_warp
__device__ __forceinline__ void chol_solve_warp(const float* L, float* b, int lane) {
  for (int k = 0; k < NV; k++) {  // forward: L y = b
    if (lane == 0) b[k] /= L[k * NV + k];
    __syncwarp();
    for (int i = k + 1 + lane; i < NV; i += 32) b[i] -= L[i * NV + k] * b[k];
    __syncwarp();
  }
  for (int k = NV - 1; k >= 0; k--) {  // backward: L^T x = y
    if (lane == 0) b[k] /= L[k * NV + k];
    __syncwarp();
    for (int i = lane; i < k; i += 32) b[i] -= L[k * NV + i] * b[k];
    __syncwarp();
  }
}
"""

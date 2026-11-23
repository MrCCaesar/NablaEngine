"""Fixed tendons in the forest engine: tendon equality rows, tendon limit rows, tendon passive forces (stiffness with
its deadband, damping, implicit in implicitfast) and tendon armature, forward and adjoint."""
import mujoco
import numpy as np

from nablaengine.forest_engine import equality as EQ

TL_KIND, ET_KIND = 7, 8
MAX_LIMITED = 16


def _is_fixed(m, t):
    if t >= m.ntendon:
        return True
    adr, n = int(m.tendon_adr[t]), int(m.tendon_num[t])
    return all(int(m.wrap_type[w]) == int(mujoco.mjtWrap.mjWRAP_JOINT) for w in range(adr, adr + n))


def support_list(m, t):
    """[(dof, coef)] of tendon t's support, ascending: a fixed tendon's coefficients, a spatial tendon's support
    (forest_engine.spatial.support) with coefficient 0 (its J is per world)."""
    from nablaengine.forest_engine import spatial
    if not _is_fixed(m, t):
        return [(d, 0.0) for d in spatial.support(m, t)]
    return sorted(coefs(m, t).items())


def spatial_build(m):
    """Whether the build takes forest_engine.spatial's text: a used tendon is spatial, or a general actuator
    (forest_engine.muscle)."""
    from nablaengine.forest_engine import muscle
    return muscle.applies(m) or any(not _is_fixed(m, t) for t in used(m))


def coefs(m, t):
    """{dof: coef} of fixed tendon t (a joint listed twice: summed)."""
    if t >= m.ntendon:
        from nablaengine.forest_engine import muscle
        return {muscle.virtual_dof(m, t): 1.0}
    out = {}
    for w in range(int(m.tendon_adr[t]), int(m.tendon_adr[t]) + int(m.tendon_num[t])):
        d = int(m.jnt_dofadr[int(m.wrap_objid[w])])
        out[d] = out.get(d, 0.0) + float(m.wrap_prm[w])
    return out


def _eq_on(m):
    return m.neq > 0 and not EQ.disabled(m)


def tendon_eqs(m):
    """The active tendon equalities that get rows, in equality order: [dict(e, t1, t2 (-1: none), poly (5,), solref,
    solimp, invw, dofs (the union support, ascending))]."""
    if not _eq_on(m):
        return []
    out = []
    for e in range(m.neq):
        if not m.eq_active0[e] or int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_TENDON):
            continue
        t1, t2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        if not _is_fixed(m, t1) or (t2 >= 0 and not _is_fixed(m, t2)):
            continue
        dofs = sorted(set(coefs(m, t1)) | (set(coefs(m, t2)) if t2 >= 0 else set()))
        invw = float(m.tendon_invweight0[t1]) + (float(m.tendon_invweight0[t2]) if t2 >= 0 else 0.0)
        out.append(dict(e=e, t1=t1, t2=t2, poly=np.array(m.eq_data[e, :5], float), solref=np.array(m.eq_solref[e]),
                        solimp=np.array(m.eq_solimp[e]), invw=invw, dofs=dofs))
    return out


def _limit_off(m):
    D = mujoco.mjtDisableBit
    return bool(int(m.opt.disableflags) & (int(D.mjDSBL_LIMIT) | int(D.mjDSBL_CONSTRAINT)))


def limited(m):
    """The limited tendons, in tendon order (none when the flags disable limits)."""
    if _limit_off(m):
        return []
    return [t for t in range(m.ntendon) if m.tendon_limited[t]]


def _flag(m, bit):
    return bool(int(m.opt.disableflags) & int(bit))


def passive(m):
    """The tendons with stiffness or damping (the flags' SPRING / DAMPER respected)."""
    D = mujoco.mjtDisableBit
    k = not _flag(m, D.mjDSBL_SPRING)
    d = not _flag(m, D.mjDSBL_DAMPER)
    return [t for t in range(m.ntendon) if (k and m.tendon_stiffness[t] > 0) or (d and m.tendon_damping[t] > 0)]


def stiffness(m, t):
    return 0.0 if _flag(m, mujoco.mjtDisableBit.mjDSBL_SPRING) else float(m.tendon_stiffness[t])


def damping(m, t):
    return 0.0 if _flag(m, mujoco.mjtDisableBit.mjDSBL_DAMPER) else float(m.tendon_damping[t])


def armature(m):
    """The tendons with armature."""
    return [t for t in range(m.ntendon) if m.tendon_armature[t] > 0]


def actuators(m):
    """The actuators on tendons (fixed, and spatial ones: forest_engine.spatial): [(actuator, tendon)]; with the general
    joint actuators on their virtual tendons (forest_engine.muscle), in actuator order."""
    from nablaengine.forest_engine import muscle
    vj = set(muscle.joint_general(m)) if muscle.applies(m) else set()
    return [(a, int(m.actuator_trnid[a, 0]) if a not in vj else muscle.virtual_tendon(m, a)) for a in range(m.nu)
            if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_TENDON) or a in vj]


def carriers(m):
    """{actuator: carrier dof} for the tendon actuators: the first support dof without a joint actuator and not another
    tendon actuator's (None where there is none: check_actuators refuses it)."""
    taken = {int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in range(m.nu)
             if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)}
    if spatial_build(m):
        return {a: None for a, _ in actuators(m)}
    out = {}
    for a, t in actuators(m):
        if not _is_fixed(m, t):
            out[a] = None
            continue
        d = next((d for d in sorted(coefs(m, t)) if d not in taken), None)
        out[a] = d
        if d is not None:
            taken.add(d)
    return out


def check_actuators(m):
    """[(actuator name, reason)] of the tendon actuators the forest engine does not support."""
    bad = []
    car = carriers(m)
    sb = any(not _is_fixed(m, int(m.actuator_trnid[a, 0])) for a in range(m.nu)
             if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_TENDON)) or spatial_build(m)
    for a in range(m.nu):
        if int(m.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_TENDON):
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or str(a)
        t = int(m.actuator_trnid[a, 0])
        affine = int(m.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
        from nablaengine.forest_engine import muscle
        if muscle.is_general(m, a):
            pass
        elif not (m.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE and m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
                and (affine or int(m.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_NONE))
                and (not affine or m.actuator_biasprm[a, 0] == 0.0) and m.actuator_gear[a, 0] == 1.0
                and not np.any(m.actuator_gear[a, 1:])
                and (not m.actuator_forcelimited[a] or m.actuator_forcerange[a, 0] == -m.actuator_forcerange[a, 1])):
            bad.append((nm, "tendon actuator: not fixed gain / affine bias / gear 1 / symmetric forcerange"))
        if m.tendon_actfrclimited[t]:
            bad.append((nm, "tendon actfrcrange"))
        if any(m.jnt_actfrclimited[int(m.dof_jntid[d])] for d, _ in support_list(m, t)):
            bad.append((nm, "tendon actuator through a joint with actuatorfrcrange"))
        if not sb and car.get(a) is None:
            bad.append((nm, "no support dof without a joint actuator (the tape's carrier)"))
    return bad


def used(m):
    """The tendons the engine reads (equality objects, limited, passive, armature, actuators), ascending."""
    s = set(limited(m)) | set(passive(m)) | set(armature(m)) | {t for _, t in actuators(m)}
    for r in tendon_eqs(m):
        s |= {r["t1"]} | ({r["t2"]} if r["t2"] >= 0 else set())
    return sorted(s)


def applies(m):
    """Whether the model has tendons the engine simulates, or general actuators (forest_engine.muscle: then the build
    takes this module's text)."""
    from nablaengine.forest_engine import muscle
    return (m.ntendon > 0 and bool(used(m))) or muscle.applies(m)


def trees(m, t):
    """The kinematic trees tendon t spans."""
    return sorted({int(m.dof_treeid[d]) for d, _ in support_list(m, t)})


def spans_trees(m):
    """The tendons the engine reads that span several kinematic trees."""
    return [t for t in used(m) if len(trees(m, t)) > 1]


def _ancestor(m, i, j):
    while i >= 0:
        if i == j:
            return True
        i = int(m.dof_parentid[i])
    return False


def check(m):
    """[(tendon or equality name, reason)] of the tendon uses the forest engine does not support (empty: all)."""
    bad = []
    J = mujoco.mjtJoint
    name = lambda t: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, t) or str(t)
    from nablaengine.forest_engine import spatial
    for t in range(m.ntendon):
        if not _is_fixed(m, t):
            bad += [(name(t), why) for why in spatial.check_tendon(m, t)]
        else:
            for w in range(int(m.tendon_adr[t]), int(m.tendon_adr[t]) + int(m.tendon_num[t])):
                j = int(m.wrap_objid[w])
                if int(m.jnt_type[j]) not in (int(J.mjJNT_HINGE), int(J.mjJNT_SLIDE)):
                    bad.append((name(t), "a joint other than hinge / slide"))
        if m.tendon_frictionloss[t] > 0 and not _flag(m, mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS) \
                and not _flag(m, mujoco.mjtDisableBit.mjDSBL_CONSTRAINT):
            bad.append((name(t), "tendon frictionloss (no frictionloss rows in the forest adjoint)"))
        if _is_fixed(m, t) and m.tendon_length0[t] != 0:
            bad.append((name(t), "tendon_length0 != 0 (joint refs: the converter folds them)"))
        if t in limited(m):
            if m.tendon_margin[t] < 0:
                bad.append((name(t), "negative limit margin"))
        if t in armature(m) and len(support_list(m, t)) > 32:
            bad.append((name(t), "armature on more than 32 dofs"))
    if spatial_build(m):
        from nablaengine.forest_engine import spatial
        words = spatial.words(m)
        if words > spatial.MAX_TS_WORDS:
            bad.append(("tendons", f"spatial tendons: per-world tendon storage of {words} words (at most "
                        f"{spatial.MAX_TS_WORDS}: the adjoint's shared memory; fewer used tendons or actuators)"))
    if len(limited(m)) > MAX_LIMITED:
        bad.append(("tendons", f"{len(limited(m))} limited tendons (at most {MAX_LIMITED})"))
    for r in tendon_eqs(m):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, r["e"]) or str(r["e"])
        if r["t1"] == r["t2"]:
            bad.append((nm, "tendon1 == tendon2"))
        if len(r["dofs"]) > 32:
            bad.append((nm, "a tendon equality over more than 32 dofs"))
    return bad


def check_eq(m, e):
    """The reasons tendon equality e is not supported (forest_engine.equality.check)."""
    why = []
    t1, t2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
    for t in (t1, t2):
        if t >= 0 and not _is_fixed(m, t):
            why.append("a tendon equality on a spatial tendon")
    if not m.eq_active0[e]:
        why.append("tendon equality inactive at build (eq_active0 = 0)")
    return why



def _f32(x):
    return EQ._f32(x)


def header(m):
    """The tendon tables (compile-time constants, at least one entry each)."""
    ts = used(m)
    tix = {t: k for k, t in enumerate(ts)}
    nt = len(ts)
    sup = [support_list(m, t) for t in ts]
    kt = max(len(x) for x in sup)
    es, ls, ps, ar = tendon_eqs(m), limited(m), passive(m), armature(m)
    acts, car = actuators(m), carriers(m)
    sb = spatial_build(m)
    if sb:
        car = {a: 0 for a, _ in acts}
    bias = lambda a: (m.actuator_biasprm[a] if int(m.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
                      else np.zeros(10))
    from nablaengine.forest_engine import muscle
    gen = {a: muscle.plain_constants(m, a) for a, _ in acts if muscle.is_general(m, a)}
    ke = max([len(r["dofs"]) for r in es], default=1)
    n1 = lambda n: max(n, 1)
    tab = lambda typ, nm, n, vals, note="": (f"__device__ const {typ} {nm}[{n}] = {{{vals}}};"
                                            + (f"  // {note}" if note else "") + "\n")
    I, F = EQ._ints, EQ._floats
    dense = []
    for s_ in sup:
        c = dict(s_)
        dense += [c.get(i, 0.0) for i in range(m.nv)]
    tdof = [x for s in sup for x in [d for d, _ in s] + [0] * (kt - len(s))]
    tco = [x for s in sup for x in [c for _, c in s] + [0.0] * (kt - len(s))]
    pat = []
    for s in sup:
        ds = [d for d, _ in s]
        pat += [sum(1 << b for b in range(a + 1) if _ancestor(m, ds[a], ds[b])) for a in range(len(ds))]
        pat += [0] * (kt - len(ds))
    lsp = [x for t in ts for x in (m.tendon_lengthspring[t] if t < m.ntendon else (0.0, 0.0))]
    etd = [x for r in es for x in r["dofs"] + [0] * (ke - len(r["dofs"]))]
    from nablaengine.forest_engine import model_hooks
    implicit = model_hooks.integrator(m) == "implicitfast"
    nd = sum(1 for t in ps if damping(m, t) > 0)
    return (f"#define NTEN {nt}  // fixed tendons the engine reads (forest_engine.tendon)\n"
            f"#define KTEN {kt}  // the largest tendon support (dofs)\n"
            f"#define NEQT {len(es)}  // tendon equality rows (kind 8), the last equality rows\n"
            f"#define KET {ke}  // the largest tendon equality support\n"
            f"#define NTL {len(ls)}  // limited tendons (kind 7 rows after the equality rows: at most 2 NTL)\n"
            f"#define NTP {len(ps)}  // tendons with stiffness or damping\n"
            f"#define NTA {len(ar)}  // tendons with armature\n"
            f"#define TEN_IMPLICIT {int(implicit and nd > 0)}  // tendon damping in implicitfast's matrix\n"
            + tab("int", "c_tnd", nt, I(len(s) for s in sup), "tendon: support size")
            + tab("unsigned char", "c_tdof", nt * kt, I(tdof), "its dofs, ascending")
            + tab("float", "c_tco", nt * kt, F(tco), "their coefficients")
            + tab("unsigned", "c_tpat", nt * kt, ", ".join(f"{x}u" for x in pat),
                  "support entry a: the entries b <= a on M's sparsity pattern (bits)")
            + tab("float", "c_tcd", nt * m.nv, F(dense), "the coefficient of every dof (0: none)")
            + tab("float", "c_tstiff", nt, F(stiffness(m, t) if t in ps else 0.0 for t in ts))
            + tab("float", "c_tdamp", nt, F(damping(m, t) if t in ps else 0.0 for t in ts))
            + tab("float", "c_tlsp", 2 * nt, F(lsp), "the spring's deadband (tendon_lengthspring)")
            + tab("int", "c_tpas", n1(len(ps)), I(tix[t] for t in ps), "the tendons with a passive force")
            + tab("float", "c_tdampp", n1(len(ps)), F(damping(m, t) for t in ps), "their damping")
            + tab("int", "c_tarmt", n1(len(ar)), I(tix[t] for t in ar), "the tendons with armature")
            + tab("float", "c_tarm", n1(len(ar)), F(m.tendon_armature[t] for t in ar))
            + tab("int", "c_ett", 2 * n1(len(es)), I(x for r in es for x in (tix[r["t1"]], tix[r["t2"]] if r["t2"] >= 0
                                                                              else -1)),
                  "tendon equality: tendons 1, 2 (-1: none)")
            + tab("int", "c_etnd", n1(len(es)), I(len(r["dofs"]) for r in es), "its union support size")
            + tab("unsigned char", "c_etdof", n1(len(es)) * ke, I(etd), "its dofs, ascending")
            + tab("float", "c_etpoly", 5 * n1(len(es)), F(x for r in es for x in r["poly"]), "polycoef")
            + tab("float", "c_etsolref", 2 * n1(len(es)), F(x for r in es for x in r["solref"]))
            + tab("float", "c_etsolimp", 5 * n1(len(es)), F(x for r in es for x in r["solimp"]))
            + tab("float", "c_etinvw", n1(len(es)), F(r["invw"] for r in es))
            + tab("int", "c_tlt", n1(len(ls)), I(tix[t] for t in ls), "limited tendon: the tendon")
            + tab("float", "c_tlr", 2 * n1(len(ls)), F(x for t in ls for x in m.tendon_range[t]), "its range")
            + tab("float", "c_tlmargin", n1(len(ls)), F(m.tendon_margin[t] for t in ls))
            + tab("float", "c_tlsolref", 2 * n1(len(ls)), F(x for t in ls for x in m.tendon_solref_lim[t]))
            + tab("float", "c_tlsolimp", 5 * n1(len(ls)), F(x for t in ls for x in m.tendon_solimp_lim[t]))
            + tab("float", "c_tlinvw", n1(len(ls)), F(m.tendon_invweight0[t] for t in ls))
            + f"#define NTACT {len(acts)}  // actuators on fixed tendons\n"
            + tab("int", "c_tat", n1(len(acts)), I(tix[t] for _, t in acts), "tendon actuator: the tendon")
            + tab("int", "c_taa", n1(len(acts)), I(a for a, _ in acts), "its actuator (ctrl index)")
            + tab("int", "c_tacar", n1(len(acts)), I(car[a] for a, _ in acts), "its carrier dof (the tape's bits)")
            + tab("float", "c_takp", n1(len(acts)), F(gen[a][0] if a in gen else m.actuator_gainprm[a, 0] for a, _ in acts))
            + tab("float", "c_takq", n1(len(acts)), F(gen[a][1] if a in gen else -bias(a)[1] for a, _ in acts))
            + tab("float", "c_takv", n1(len(acts)), F(gen[a][2] if a in gen else -bias(a)[2] for a, _ in acts))
            + tab("float", "c_tafrc", n1(len(acts)), F(gen[a][3] if a in gen else (
                m.actuator_forcerange[a, 1] if m.actuator_forcelimited[a] else 1e30) for a, _ in acts),
                "forcerange (symmetric)")
            + tab("float", "c_tactr", 2 * n1(len(acts)), F(x for a, _ in acts for x in (
                gen[a][4] if a in gen else m.actuator_ctrlrange[a] if m.actuator_ctrllimited[a] else (-1e30, 1e30))),
                "ctrlrange")
            + (_spatial_header(m, ts, kt, acts, implicit) if sb else "")
            + (_muscle_header(m, acts) if sb and _muscle_on(m) else "")
)


def _muscle_on(m):
    from nablaengine.forest_engine import muscle
    return muscle.applies(m)


def _muscle_header(m, acts):
    from nablaengine.forest_engine import muscle
    return muscle.tables(m, acts)


def _spatial_header(m, ts, kt, acts, implicit):
    from nablaengine.forest_engine import spatial
    return (f"#define SP_IMPLICIT {int(implicit)}  // tendon actuators' kv in implicitfast's matrix (forest_engine.spatial)\n"
            + spatial.tables(m, ts, kt, acts))


DEVICE = r"""// fixed tendons (forest_engine.tendon): L = sum c_i q_i over the support, explicitly rounded (the adjoint replays the
// limit rows' test and the equality rows' pos bitwise)
__device__ __forceinline__ float ten_len(const float* qpos, int t) {
  const unsigned char* d = c_tdof + KTEN * t;
  const float* c = c_tco + KTEN * t;
  float L = RMUL(c[0], qpos[c_dqadr[d[0]]]);
  for (int k = 1; k < c_tnd[t]; k++) L = RFMA(c[k], qpos[c_dqadr[d[k]]], L);
  return L;
}
// c . v of tendon t (v an NV vector)
__device__ __forceinline__ float ten_jv(const float* v, int t) {
  const unsigned char* d = c_tdof + KTEN * t;
  const float* c = c_tco + KTEN * t;
  float acc = 0.f;
  for (int k = 0; k < c_tnd[t]; k++) acc += c[k] * v[d[k]];
  return acc;
}
// the part of c . v on M's sparsity pattern relative to dof i (its ancestors and descendants on the support,
// c_tpat): the implicit damping's row i, (h d c c' restricted) y = h d c_i ten_jv_pat(y, t, i)
__device__ __forceinline__ float ten_jv_pat(const float* v, int t, int i) {
  const unsigned char* d = c_tdof + KTEN * t;
  const float* c = c_tco + KTEN * t;
  int a = 0;
  while (d[a] != i) a++;
  float acc = 0.f;
  for (int k = 0; k < c_tnd[t]; k++) {
    const bool on = k <= a ? (c_tpat[KTEN * t + a] >> k) & 1u : (c_tpat[KTEN * t + k] >> a) & 1u;
    if (on) acc += c[k] * v[d[k]];
  }
  return acc;
}
// tendon equality e: J_2 = -P'(L2) (0 without tendon 2), its derivative -P''(L2), pos = L1 - P(L2)
__device__ __forceinline__ float et_j2(const float* qpos, int e) {
  const int t2 = c_ett[2 * e + 1];
  if (t2 < 0) return 0.f;
  const float* c = c_etpoly + 5 * e;
  const float x = ten_len(qpos, t2);
  return -RFMA(x, RFMA(x, RFMA(x, RMUL(4.f, c[4]), RMUL(3.f, c[3])), RMUL(2.f, c[2])), c[1]);
}
__device__ __forceinline__ float et_j2d(const float* qpos, int e) {
  const int t2 = c_ett[2 * e + 1];
  if (t2 < 0) return 0.f;
  const float* c = c_etpoly + 5 * e;
  const float x = ten_len(qpos, t2);
  return -RFMA(x, RFMA(x, RMUL(12.f, c[4]), RMUL(6.f, c[3])), RMUL(2.f, c[2]));
}
__device__ __forceinline__ float et_pos(const float* qpos, int e) {
  const float* c = c_etpoly + 5 * e;
  const float L1 = ten_len(qpos, c_ett[2 * e]);
  const int t2 = c_ett[2 * e + 1];
  if (t2 < 0) return RSUB(L1, c[0]);
  const float x = ten_len(qpos, t2);
  return RSUB(L1, RFMA(x, RFMA(x, RFMA(x, RFMA(x, c[4], c[3]), c[2]), c[1]), c[0]));
}
// its J at dof i (J_2 = et_j2)
__device__ __forceinline__ float et_J(int e, int i, float j2) {
  const int t2 = c_ett[2 * e + 1];
  const float c1 = c_tcd[NV * c_ett[2 * e] + i];
  return t2 < 0 ? c1 : c1 + j2 * c_tcd[NV * t2 + i];
}
__device__ __forceinline__ float et_jv(const float* qpos, int e, const float* v) {
  const float j2 = et_j2(qpos, e);
  const unsigned char* d = c_etdof + KET * e;
  float acc = 0.f;
  for (int k = 0; k < c_etnd[e]; k++) acc += et_J(e, d[k], j2) * v[d[k]];
  return acc;
}
// limited tendon l, side 0 (lower: dist = L - range[0], J = +c) or 1 (upper: dist = range[1] - L, J = -c)
__device__ __forceinline__ float tl_dist(const float* qpos, int l, int side) {
  const float L = ten_len(qpos, c_tlt[l]);
  return side ? RSUB(c_tlr[2 * l + 1], L) : RSUB(L, c_tlr[2 * l]);
}
// the number of active tendon limit rows at qpos (the forward's test)
__device__ __forceinline__ int tl_rows(const float* qpos) {
  int n = 0;
  for (int l = 0; l < NTL; l++)
    for (int side = 0; side < 2; side++) n += tl_dist(qpos, l, side) < c_tlmargin[l] ? 1 : 0;
  return n;
}
// the passive force of tendon t (index into the tendon table): -k (L - s) - d (c . v), s the deadband's nearer end
// (L inside: no spring force); keff: its dF/dL (k outside the deadband, else 0)
__device__ __forceinline__ float ten_passive_f(const float* qpos, const float* v, int t, float* keff) {
  const float L = ten_len(qpos, t), lo = c_tlsp[2 * t], hi = c_tlsp[2 * t + 1], k = c_tstiff[t];
  const float sp = L > hi ? -k * (L - hi) : L < lo ? -k * (L - lo) : 0.f;
  *keff = L > hi || L < lo ? k : 0.f;
  return sp - c_tdamp[t] * ten_jv(v, t);
}
// the tendons' passive force on dof i
__device__ __forceinline__ float ten_qfrc(const float* qpos, const float* v, int i) {
  float acc = 0.f, ke;
  for (int p = 0; p < NTP; p++) {
    const int t = c_tpas[p];
    const float c = c_tcd[NV * t + i];
    if (c != 0.f) acc += c * ten_passive_f(qpos, v, t, &ke);
  }
  return acc;
}
// A += w_t c_t c_t' over the tendons in list[0 .. n) (w = scale * wt[p]) on M's sparsity pattern (c_tpat: MuJoCo's
// sparse qM); lane a: the entries (a, b <= a) of the support, tendons one after another (fenced)
__device__ __forceinline__ void ten_outer(float* A, const int* list, const float* wt, int n, float scale, int lane) {
  for (int p = 0; p < n; p++) {
    const int t = list[p], nd = c_tnd[t];
    const unsigned char* d = c_tdof + KTEN * t;
    const float* c = c_tco + KTEN * t;
    if (lane < nd) {
      const float g = scale * wt[p] * c[lane];
      const unsigned pm = c_tpat[KTEN * t + lane];
      for (int b = 0; b <= lane; b++)
        if ((pm >> b) & 1u) A[TRI(d[lane], d[b])] += g * c[b];
    }
    __syncwarp();
  }
}
// the tendon actuators' force on dof i: f = kp u - kq L - kv (c . v), u = ctrl clamped to ctrlrange, f clamped to
// the forcerange; on the carrier dof also the flags (ctrl clamped, force at its forcerange) for the tape's bits
__device__ __forceinline__ float tact_qfrc(const float* qpos, const float* v, const float* ctrl, int i, bool* aclamp,
                                           bool* uclamp, bool* clamp) {
  float acc = 0.f;
  for (int k = 0; k < NTACT; k++) {
    const int t = c_tat[k], a = c_taa[k];
    const float c = c_tcd[NV * t + i];
    const bool car = c_tacar[k] == i;
    if (c == 0.f && !car) continue;
    const float u = fminf(fmaxf(ctrl[a], c_tactr[2 * k]), c_tactr[2 * k + 1]);
    float f = c_takp[k] * u - c_takq[k] * ten_len(qpos, t) - c_takv[k] * ten_jv(v, t);
    const bool fc = f <= -c_tafrc[k] || f >= c_tafrc[k];
    f = fminf(fmaxf(f, -c_tafrc[k]), c_tafrc[k]);
    if (car) { *uclamp = *uclamp || u != ctrl[a]; *aclamp = *aclamp || fc; *clamp = *clamp || fc; }
    acc += c * f;
  }
  return acc;
}
// L += h kv c c' of the tendon actuators whose kv is on (the carrier dof's kvoff bit), on M's sparsity pattern
template <class S>
__device__ void tact_implicit(S& s, float h, int lane) {
  for (int k = 0; k < NTACT; k++) {
    const int t = c_tat[k], cr = c_tacar[k], nd = c_tnd[t];
    if ((s.kvoff[cr >> 5] >> (cr & 31)) & 1u) continue;
    const unsigned char* d = c_tdof + KTEN * t;
    const float* c = c_tco + KTEN * t;
    if (lane < nd) {
      const float g = h * c_takv[k] * c[lane];
      const unsigned pm = c_tpat[KTEN * t + lane];
      for (int b = 0; b <= lane; b++)
        if ((pm >> b) & 1u) s.L[TRI(d[lane], d[b])] += g * c[b];
    }
    __syncwarp();
  }
}
// J' f of the tendon rows on dof i, added to acc: the tendon equality rows (after the connect rows), then the tendon
// limit rows (kind 7, after the equality rows)
template <class S>
__device__ __forceinline__ void ten_jtf(const S& s, const float* f, int i, float& acc) {
  for (int e = 0; e < NEQT; e++) {
    const float fe = f[s.nefc_dof + NEQJ + 3 * NEQC + e];
    const float J = et_J(e, i, et_j2(s.qpos, e));
    if (J != 0.f) acc += J * fe;
  }
  for (int r = s.nefc_dof + NEQR; r < s.nefc && s.efc_kind[r] == 7; r++) {
    const float c = c_tcd[NV * c_tlt[s.efc_a[r]] + i];
    if (c != 0.f) acc += s.efc_b[r] * c * f[r];
  }
}
// H += J' D J of the tendon rows (lane a: the entries (a, b <= a) of the row's support; rows one after another):
// the equality rows always, the limit rows on their quadratic piece
template <class S>
__device__ void ten_hessian(S& s, int lane) {
  for (int e = 0; e < NEQT; e++) {
    const int r = s.nefc_dof + NEQJ + 3 * NEQC + e, nd = c_etnd[e];
    const unsigned char* d = c_etdof + KET * e;
    const float j2 = et_j2(s.qpos, e), D = s.efc_D[r];
    if (lane < nd) {
      const float g = D * et_J(e, d[lane], j2);
      for (int b = 0; b <= lane; b++) s.L[TRI(d[lane], d[b])] += g * et_J(e, d[b], j2);
    }
    __syncwarp();
  }
  for (int r = s.nefc_dof + NEQR; r < s.nefc && s.efc_kind[r] == 7; r++) {
    float ds, dd, cost;
    row_eval(s, r, s.efc_jar[r], &ds, &dd, &cost);
    if (dd != 0.f) {
      const int t = c_tlt[s.efc_a[r]], nd = c_tnd[t];
      const unsigned char* d = c_tdof + KTEN * t;
      const float* c = c_tco + KTEN * t;
      if (lane < nd) {
        const float g = dd * c[lane];
        for (int b = 0; b <= lane; b++) s.L[TRI(d[lane], d[b])] += g * c[b];
      }
    }
    __syncwarp();
  }
}

"""

ADJ_DEVICE = r"""
// A5 / A6, tendon rows (forest_engine.tendon), lane 0 in row order (deterministic): the tendon equality rows (always on
// their quadratic piece) as the joint rows with J = c_1 + J_2 c_2 (dJ/dq_m = -P''(L2) c2_m c2), then the tendon limit
// rows replayed with the forward's test (tl_dist), the tape's zone bits, limit_row_vjp in the tendon coordinate times
// c. jtf (debug build): the rows' J' f into s.jtf
__device__ __noinline__ void adj_tendon(Adj& s, int lane, bool jtf) {
  if (lane == 0) {
    for (int e = 0; e < NEQT; e++) {
      const int t2 = c_ett[2 * e + 1], nd = c_etnd[e];
      const unsigned char* d = c_etdof + KET * e;
      const float pos = et_pos(s.q, e), j2 = et_j2(s.q, e), j2d = et_j2d(s.q, e);
      float ri[6];
      adj_row_impedance(c_etsolref + 2 * e, c_etsolimp + 5 * e, pos, c_etinvw[e], ri);
      const float k = ri[0], b = ri[1], imp = ri[2], dimp = ri[3], D = ri[4], dD = ri[5];
      float ja = 0.f, jv = 0.f, jl = 0.f, c2a = 0.f, c2v = 0.f, c2l = 0.f;
      for (int q = 0; q < nd; q++) {
        const int i = d[q];
        const float J = et_J(e, i, j2);
        ja += J * s.a[i]; jv += J * s.v[i]; jl += J * s.lam[i];
        if (t2 >= 0) {
          const float c2 = c_tcd[NV * t2 + i];
          c2a += c2 * s.a[i]; c2v += c2 * s.v[i]; c2l += c2 * s.lam[i];
        }
      }
      const float x = ja + k * imp * pos + b * jv;
      const float sp = jl * (dD * x + D * (k * (imp + dimp * pos))), pv = D * b * jl;
      const float sn = t2 >= 0 ? j2d * (D * ((c2a + b * c2v) * jl + x * c2l)) : 0.f;
      for (int q = 0; q < nd; q++) {
        const int i = d[q];
        const float J = et_J(e, i, j2);
        s.Pq[i] += sp * J + (t2 >= 0 ? sn * c_tcd[NV * t2 + i] : 0.f);
        s.Pv[i] += pv * J;
#ifdef DK_ADJ_DEBUG
        if (jtf) s.jtf[i] += -D * x * J;
#endif
      }
    }
    int r = NFLOSS + (((unsigned)s.hdr >> 18) & 63u) + NEQR;  // the first tendon limit row
    for (int l = 0; l < NTL; l++)
      for (int side = 0; side < 2; side++) {
        const float dist = tl_dist(s.q, l, side);
        if (!(dist < c_tlmargin[l])) continue;
        const bool act = (s.zone[r >> 5] >> (r & 31)) & 1u;
        r++;
        if (!act) continue;
        const int t = c_tlt[l], nd = c_tnd[t];
        const unsigned char* d = c_tdof + KTEN * t;
        const float* c = c_tco + KTEN * t;
        const float pos = RSUB(dist, c_tlmargin[l]), sg = side ? -1.f : 1.f;
        float ri[6], pq, pv, ca = 0.f, cv = 0.f, cl = 0.f;
        adj_row_impedance(c_tlsolref + 2 * l, c_tlsolimp + 5 * l, pos, c_tlinvw[l], ri);
        for (int q = 0; q < nd; q++) { ca += c[q] * s.a[d[q]]; cv += c[q] * s.v[d[q]]; cl += c[q] * s.lam[d[q]]; }
        limit_row_vjp(sg, pos, ca, cv, cl, ri[0], ri[1], ri[2], ri[3], ri[4], ri[5], &pq, &pv);
        for (int q = 0; q < nd; q++) { s.Pq[d[q]] += pq * c[q]; s.Pv[d[q]] += pv * c[q]; }
#ifdef DK_ADJ_DEBUG
        if (jtf) {
          const float x = sg * ca + ri[0] * ri[2] * pos + ri[1] * (sg * cv);
          for (int q = 0; q < nd; q++) s.jtf[d[q]] += -ri[4] * x * sg * c[q];
        }
#endif
      }
  }
  (void)jtf;
  __syncwarp();
}
"""


def _sub(text, a, b, count=1):
    assert text.count(a) == count, f"forest tendon: {text.count(a)} x {a[:80]!r}"
    return text.replace(a, b)


TL_ROWS = """  int ntl = 0;  // the tendon limit rows (kind 7) after the equality rows (forest_engine.tendon): candidate (tendon
  {             // lane >> 1, side lane & 1), in (tendon, side) order
    const int l = min(lane >> 1, NTL > 0 ? NTL - 1 : 0), side = lane & 1;
    const bool on = lane < 2 * NTL && tl_dist(s.qpos, l, side) < c_tlmargin[l];
    const unsigned bal = __ballot_sync(0xffffffffu, on);
    if (on) {
      const int r = ndof + __popc(bal & below);
      s.efc_kind[r] = 7; s.efc_a[r] = l; s.efc_b[r] = side ? -1 : 1;
    }
    ntl = __popc(bal);
    ndof += ntl;
  }
"""

JOB_TL = """      const bool tlr = !con && j >= ndof - ntl;  // a tendon limit row (forest_engine.tendon)
      const bool eqr = !con && j >= ndof - ntl - NEQR;  // an equality row (forest_engine.equality) or a tendon limit
      float eqvel = 0.f, eqpos = 0.f, eqjd = 0.f;
      if (tlr) {
        const int l = s.efc_a[j], side = s.efc_b[j] < 0;
        pos = RSUB(tl_dist(s.qpos, l, side), c_tlmargin[l]); eqpos = pos;
        eqvel = s.efc_b[j] * ten_jv(s.qvel, c_tlt[l]);
        iw = c_tlinvw[l];
        sref = c_tlsolref + 2 * l; simp = c_tlsolimp + 5 * l;
      } else if (eqr) {
        const int e = j - (ndof - ntl - NEQR);
        if (e >= NEQJ + 3 * NEQC) {  // tendon equality
          const int et = e - NEQJ - 3 * NEQC;
          pos = et_pos(s.qpos, et); eqpos = pos;
          eqvel = et_jv(s.qpos, et, s.qvel);
          iw = c_etinvw[et];
          sref = c_etsolref + 2 * et; simp = c_etsolimp + 5 * et;
        } else if (e < NEQJ) {  // joint
"""


def hooks(m):
    """The forward's edits for a tendon build: forest_engine.equality's connect-build hooks (NEQC possibly 0) with the
    tendon rows, then the passive force, the armature and the implicit damping."""
    base = list(EQ.hooks_connect(m, extra_rows=len(tendon_eqs(m))))
    out = []
    for a, b in base:
        if a == "#define NFLOSS ":
            b = b.replace("#define NFLOSS ", header(m) + "#define NFLOSS ")
        elif a.startswith("template <class T> struct dk_is_ptr"):
            b = b.replace("template <class T> struct dk_is_ptr", DEVICE + "template <class T> struct dk_is_ptr", 1)
        elif "eq_jtf(s, f, lane + 32 * t, acc[t]);" in b:
            b = _sub(b, "eq_jtf(s, f, lane + 32 * t, acc[t]);",
                     "{ eq_jtf(s, f, lane + 32 * t, acc[t]); ten_jtf(s, f, lane + 32 * t, acc[t]); }")
        elif "    eq_jtf(s, f, i, acc);\n" in b:
            b = _sub(b, "    eq_jtf(s, f, i, acc);\n", "    eq_jtf(s, f, i, acc);\n    ten_jtf(s, f, i, acc);\n")
        elif "eq_hessian(s, lane);" in b:
            b = _sub(b, "  __syncwarp();\n  SUBPROF(28);\n",
                     "  __syncwarp();\n  ten_hessian(s, lane);  // the tendon rows (forest_engine.tendon)\n"
                     "  SUBPROF(28);\n")
        elif "s.nefc_dof = ndof - NEQR;" in b:
            b = _sub(b, "s.nefc_dof = ndof - NEQR;", "s.nefc_dof = ndof - NEQR - ntl;")
        elif a == "    if (kind == 0) val = v[a];\n":
            b = ("    if (kind == 7) val = b * ten_jv(v, c_tlt[a]);  // tendon limit row (forest_engine.tendon)\n"
                 "    else if (kind == 8) val = et_jv(s.qpos, a, v);  // tendon equality row\n    else " + b.lstrip())
        elif a == EQ.ROWS_INSERT_AT:
            b = _sub(b, "    else { s.efc_kind[r] = 6; s.efc_a[r] = c; s.efc_b[r] = e - NEQJ - 3 * c; }\n",
                     "    else if (e < NEQJ + 3 * NEQC) { s.efc_kind[r] = 6; s.efc_a[r] = c; s.efc_b[r] = e - NEQJ - 3 * c; }\n"
                     "    else { s.efc_kind[r] = 8; s.efc_a[r] = e - NEQJ - 3 * NEQC; s.efc_b[r] = 0; }  // tendon\n")
            i = b.index(EQ.ROWS_INSERT_AT)
            b = b[:i] + TL_ROWS + b[i:]
        elif a == EQ.JOB_OLD:
            head = ("      const bool eqr = !con && j >= ndof - NEQR;  // an equality row (forest_engine.equality)\n"
                    "      float eqvel = 0.f, eqpos = 0.f, eqjd = 0.f;\n      if (eqr) {\n"
                    "        const int e = j - (ndof - NEQR);\n        if (e < NEQJ) {  // joint\n")
            b = _sub(b, head, JOB_TL)
        out.append((a, b))
    from nablaengine.forest_engine import model_hooks
    implicit = model_hooks.integrator(m) == "implicitfast"
    if actuators(m):
        out.append(("      s.qsm[i] = act - c_damp[i] * s.qvel[i]",
                    "      act += tact_qfrc(s.qpos, s.qvel, ctrl, i, &aclamp, &uclamp, &clamp);  // the tendon actuators "
                    "(forest_engine.tendon)\n      s.qsm[i] = act - c_damp[i] * s.qvel[i]"))
    if passive(m):
        out.append(("      s.qsm[i] = act - c_damp[i] * s.qvel[i]",
                    "      act += ten_qfrc(s.qpos, s.qvel, i);  // the tendons' passive force (forest_engine.tendon)\n"
                    "      s.qsm[i] = act - c_damp[i] * s.qvel[i]"))
    if armature(m):
        out.append(("  __syncwarp();\n  SUBPROF(17);\n",
                    "  __syncwarp();\n  ten_outer(s.M, c_tarmt, c_tarm, NTA, 1.f, lane);  // tendon armature: M += a c c' "
                    "(forest_engine.tendon)\n  SUBPROF(17);\n"))
    damp = implicit and any(damping(m, t) > 0 for t in passive(m))
    if damp or (implicit and actuators(m)):
        add = ""
        if damp:
            add += ("  ten_outer(s.L, c_tpas, c_tdampp, NTP, h, lane);  // + h d c c' of the tendons' damping "
                    "(forest_engine.tendon)\n")
        if implicit and actuators(m):
            add += ("  tact_implicit(s, h, lane);  // + h kv c c' of the tendon actuators (kv off at the forcerange; "
                    "forest_engine.tendon)\n")
        out.append(("  __syncwarp();\n  CHOL_M_SOLVE(s.L, s.L, s.tmp, lane, s.cbuf);\n",
                    "  __syncwarp();\n" + add + "  CHOL_M_SOLVE(s.L, s.L, s.tmp, lane, s.cbuf);\n"))
    return out


def adj_hooks(m, src):
    """The adjoint's edits: equality's (with adj_tendon after adj_connect, the contacts' first row after the tendon limit
    rows), the implicit tendon damping and actuator kv in A1, the passive force and the tendon actuators in
    adj_compose."""
    out = []
    for a, b in EQ.adj_hooks_connect(m, src):
        if a.startswith("template <bool DBG>\n__device__ __noinline__ void adj_limits("):
            b = _sub(b, "  adj_connect(s, lane, jtf);\n}\n", "  adj_connect(s, lane, jtf);\n  adj_tendon(s, lane, jtf);\n}\n")
            b = ADJ_DEVICE + b
        elif "nlim + NEQR + c + " in b:
            b = _sub(b, "nlim + NEQR + c + ", "nlim + NEQR + tl_rows(s.q) + c + ")
        out.append((a, b))
    from nablaengine.forest_engine import model_hooks
    implicit = model_hooks.integrator(m) == "implicitfast"
    damp = implicit and any(damping(m, t) > 0 for t in passive(m))
    if damp or (implicit and actuators(m)):
        out.append(("    s.ny[i] = -s.y[i];\n  }\n  __syncwarp();\n  APROF(APROF_SOLVE_Q);\n",
                    "    s.ny[i] = -s.y[i];\n  }\n"
                    "  for (int i = lane; i < NV; i += 32) {  // - h (d_t c (c . y) + kv c (c . y)) on M's pattern: the "
                    "tendons'\n"
                    "    float acc = 0.f;  // implicit damping and actuator kv (forest_engine.tendon)\n"
                    + ("    for (int p = 0; p < NTP; p++) {\n"
                       "      const int t = c_tpas[p];\n"
                       "      const float c = c_tcd[NV * t + i];\n"
                       "      if (c != 0.f) acc += c_tdamp[t] * c * ten_jv_pat(s.y, t, i);\n"
                       "    }\n" if damp else "")
                    + ("    for (int k = 0; k < NTACT; k++) {\n"
                       "      const int t = c_tat[k], cr = c_tacar[k];\n"
                       "      const float c = c_tcd[NV * t + i];\n"
                       "      if (c != 0.f && !((s.kvoff[cr >> 5] >> (cr & 31)) & 1u)) acc += c_takv[k] * c * "
                       "ten_jv_pat(s.y, t, i);\n"
                       "    }\n" if implicit and actuators(m) else "")
                    + "    s.abar[i] -= h * acc;\n"
                    "  }\n  __syncwarp();\n  APROF(APROF_SOLVE_Q);\n"))
    if passive(m) or actuators(m):
        body = ""
        if passive(m):
            body += ("    for (int p = 0; p < NTP; p++) {  // the passive force: P_q += k_eff c (c . lam), P_v += d c (c . lam)\n"
                     "      const int t = c_tpas[p];\n"
                     "      const float c = c_tcd[NV * t + i];\n"
                     "      if (c == 0.f) continue;\n"
                     "      float ke;\n"
                     "      (void)ten_passive_f(s.q, s.v, t, &ke);\n"
                     "      const float cl = c * ten_jv(s.lam, t);\n"
                     "      s.Pq[i] += ke * cl;\n"
                     "      s.Pv[i] += c_tdamp[t] * cl;\n"
                     "    }\n")
        if actuators(m):
            body += ("    for (int k = 0; k < NTACT; k++) {  // the tendon actuators: f = kp u - kq L - kv (c . v), lam~ = "
                     "c . lam\n"
                     "      const int t = c_tat[k], cr = c_tacar[k];  // (0 where f is at its forcerange: the carrier dof's "
                     "clamp bit)\n"
                     "      const float c = c_tcd[NV * t + i];\n"
                     "      if (c == 0.f) continue;\n"
                     "      const float cl = ten_jv(s.lam, t), lt = ((s.clamp[cr >> 5] >> (cr & 31)) & 1u) ? 0.f : cl;\n"
                     "      s.Pq[i] += c_takq[k] * c * lt;\n"
                     "      s.Pv[i] += c_takv[k] * c * (DK_ADJ_CLAMP_KV ? lt : cl);\n"
                     "    }\n")
        ub = ""
        if actuators(m):
            ub = ("  if (lane == 0)  // u bar += kp lam~ (ctrl at its ctrlrange: none; the carrier dof's uclamp bit)\n"
                  "    for (int k = 0; k < NTACT; k++) {\n"
                  "      const int t = c_tat[k], cr = c_tacar[k];\n"
                  "      if (((s.clamp[cr >> 5] >> (cr & 31)) & 1u) || ((s.uclamp[cr >> 5] >> (cr & 31)) & 1u)) continue;\n"
                  "      s.ubar[c_taa[k]] += c_takp[k] * ten_jv(s.lam, t);\n"
                  "    }\n")
        out.append(("  for (int i = lane; i < NV; i += 32) s.vbar[i] -= s.Pv[i];\n",
                    "  for (int i = lane; i < NV; i += 32) {  // the tendons (forest_engine.tendon)\n" + body + "  }\n"
                    + ub + "  __syncwarp();\n"
                    "  for (int i = lane; i < NV; i += 32) s.vbar[i] -= s.Pv[i];\n"))
    return out


def nrows_max(m):
    """The tendon rows at most: the equality rows and 2 per limited tendon."""
    return len(tendon_eqs(m)) + 2 * len(limited(m))


def solimp_powers(m):
    return ([float(np.float32(r["solimp"][4])) for r in tendon_eqs(m)]
            + [float(np.float32(m.tendon_solimp_lim[t, 4])) for t in limited(m)])


DEV_AT = "template <class T> struct dk_is_ptr { static constexpr bool value = false; };\n"
SP_CONJ = ("  float ts_store[TS_WORDS];  // the tendons' L, J and actuator forces (forest_engine.spatial ts_base: right "
           "after ec_Jrows)\n")


def apply(src, m, nefc_rows):
    """src (adjoint.cuda_source's unit, after model_hooks and elliptic) with the tendon edits (and the equality edits
    of the model's joint / connect equalities) for a model with fixed tendons the engine uses."""
    bad = check(m) + check_actuators(m) + EQ.check(m)
    if bad:
        raise ValueError(f"tendons / equality constraints the forest engine does not support: {bad[:4]}")
    EQ.check_rows(m, nefc_rows, extra=nrows_max(m))
    has_adj = "__device__ __noinline__ void adj_limits(" in src
    pairs = hooks(m) + (adj_hooks(m, src) if has_adj else [])
    if spatial_build(m):
        from nablaengine.forest_engine import spatial
        pairs = spatial.transform(m, pairs, EQ.EC_CONJ, SP_CONJ, DEV_AT) + (spatial.adj_extra() if has_adj else [])
    for a, b in pairs:
        k = src.count(a)
        if a == EQ.EC_CONJ and k == 2:
            src = src.replace(a, b)
            continue
        if k != 1:
            raise RuntimeError(f"forest tendon: anchor found {k} times, expected 1: {a[:100]!r}")
        src = src.replace(a, b)
    from nablaengine.forest_engine import muscle
    if muscle.applies(m):
        src = muscle.apply(src, m)
    if spatial_build(m):
        from nablaengine.forest_engine import spatial
        if spatial.compact_on(m):
            src = spatial.compact(src, m)
    return src

"""Equality constraints in the forest engine: joint equalities (mjEQ_JOINT) and connect equalities (mjEQ_CONNECT),
forward rows and their adjoint."""
import mujoco
import numpy as np

EQ_KIND = 5


def _f32(x):
    return repr(float(np.float32(x))) + "f"


def disabled(m):
    """The model's DisableBit flags remove every equality row (EQUALITY or CONSTRAINT)."""
    D = mujoco.mjtDisableBit
    return bool(int(m.opt.disableflags) & (int(D.mjDSBL_EQUALITY) | int(D.mjDSBL_CONSTRAINT)))


def check(m):
    """[(eq name or index, reason)] of the equalities the forest engine does not support (empty: all supported)."""
    bad = []
    J = mujoco.mjtJoint
    for e in range(m.neq):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e) or str(e)
        t = int(m.eq_type[e])
        if t == int(mujoco.mjtEq.mjEQ_CONNECT):
            bad += [(name, why) for why in _check_connect(m, e)]
            continue
        if t == int(mujoco.mjtEq.mjEQ_TENDON):
            from nablaengine.forest_engine import tendon
            bad += [(name, why) for why in tendon.check_eq(m, e)]
            continue
        if t != int(mujoco.mjtEq.mjEQ_JOINT):
            bad.append((name, mujoco.mjtEq(t).name))
            continue
        j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        for j in (j1, j2):
            if j < 0:
                continue
            if int(m.jnt_type[j]) not in (int(J.mjJNT_HINGE), int(J.mjJNT_SLIDE)):
                bad.append((name, "a joint other than hinge / slide"))
            elif m.qpos0[m.jnt_qposadr[j]] != 0:
                bad.append((name, "joint ref != 0"))
        if j1 == j2:
            bad.append((name, "joint1 == joint2"))
    return bad


def _connect_bodies(m, e):
    if int(m.eq_objtype[e]) == int(mujoco.mjtObj.mjOBJ_SITE):
        s1, s2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        return (int(m.site_bodyid[s1]), int(m.site_bodyid[s2]), np.array(m.site_pos[s1], float),
                np.array(m.site_pos[s2], float))
    return (int(m.eq_obj1id[e]), max(int(m.eq_obj2id[e]), 0), np.array(m.eq_data[e, 0:3], float),
            np.array(m.eq_data[e, 3:6], float))


def _chain_dofs(m, b):
    out = []
    while b > 0:
        out += range(int(m.body_dofadr[b]), int(m.body_dofadr[b]) + int(m.body_dofnum[b]))
        b = int(m.body_parentid[b])
    return out


def _check_connect(m, e):
    why = []
    if int(m.eq_objtype[e]) not in (int(mujoco.mjtObj.mjOBJ_BODY), int(mujoco.mjtObj.mjOBJ_XBODY),
                                    int(mujoco.mjtObj.mjOBJ_SITE)):
        return ["connect: objects other than bodies / sites"]
    if not m.eq_active0[e]:
        why.append("connect inactive at build (eq_active0 = 0)")
    b1, b2, _, _ = _connect_bodies(m, e)
    if b1 == b2:
        why.append("connect: both objects on one body")
    nd = len(set(_chain_dofs(m, b1)) | set(_chain_dofs(m, b2)))
    if nd > 32:
        why.append(f"connect: {nd} dofs on its chains (at most 32)")
    return why


def rows(m):
    """The active joint equalities that get rows, in equality order: [(eq index, d1, d2 (-1: none), polycoef (5,),
    solref (2,), solimp (5,))] (none for a model whose flags disable equalities)."""
    if m.neq == 0 or disabled(m):
        return []
    out = []
    for e in range(m.neq):
        if not m.eq_active0[e] or int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        out.append((e, int(m.jnt_dofadr[j1]), int(m.jnt_dofadr[j2]) if j2 >= 0 else -1, np.array(m.eq_data[e, :5]),
                    np.array(m.eq_solref[e]), np.array(m.eq_solimp[e])))
    return out


INVW_NUMERIC = "nablaengine_eq_invweight"


def _invw_override(m):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, INVW_NUMERIC)
    if i < 0:
        return None
    adr, n = int(m.numeric_adr[i]), int(m.numeric_size[i])
    if n != m.neq:
        raise ValueError(f"custom numeric {INVW_NUMERIC}: size {n}, the model has {m.neq} equalities")
    return np.array(m.numeric_data[adr:adr + n], np.float64)


def connect_invweight(m, e):
    """Connect e's invweight: body_invweight0[b1, 0] + body_invweight0[b2, 0] (MuJoCo C: the bodies themselves, also
    for a site's body), or the custom numeric INVW_NUMERIC's value."""
    ov = _invw_override(m)
    if ov is not None and ov[e] >= 0:
        return float(ov[e])
    b1, b2, _, _ = _connect_bodies(m, e)
    return float(m.body_invweight0[b1, 0] + m.body_invweight0[b2, 0])


def connects(m):
    """The connect equalities that get rows, in equality order (none for a model whose flags disable equalities):
    [dict(e, b1, b2, a1, a2, solref, solimp, invw, dofs (the support: both chains' dofs, ascending), side (bit 0: on
    chain 1, bit 1: on chain 2), par (support index of each dof's parent dof, -1: none), tw (support index of the twist
    whose cross product gives dS/dt: the parent's, a free root's rotations the root's last dof; -1: dS/dt = 0), k1, k2
    (support index of the twist of body 1 / 2: its last dof; -1: the world))]."""
    if m.neq == 0 or disabled(m):
        return []
    out = []
    for e in range(m.neq):
        if not m.eq_active0[e] or int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_CONNECT):
            continue
        b1, b2, a1, a2 = _connect_bodies(m, e)
        c1, c2 = set(_chain_dofs(m, b1)), set(_chain_dofs(m, b2))
        dofs = sorted(c1 | c2)
        pos = {d: k for k, d in enumerate(dofs)}
        par, tw, side = [], [], []
        for d in dofs:
            p = int(m.dof_parentid[d])
            par.append(pos[p] if p >= 0 else -1)
            side.append(int(d in c1) | 2 * int(d in c2))
            b = int(m.dof_bodyid[d])
            if int(m.jnt_type[m.dof_jntid[d]]) == int(mujoco.mjtJoint.mjJNT_FREE):
                t = d - int(m.body_dofadr[b])
                tw.append(-1 if t < 3 else pos[int(m.body_dofadr[b]) + 5])
            else:
                tw.append(par[-1])
        last = lambda b: pos[int(m.body_dofadr[b]) + int(m.body_dofnum[b]) - 1] if b > 0 else -1
        out.append(dict(e=e, b1=b1, b2=b2, a1=a1, a2=a2, solref=np.array(m.eq_solref[e]),
                        solimp=np.array(m.eq_solimp[e]), invw=connect_invweight(m, e), dofs=dofs, side=side, par=par,
                        tw=tw, k1=last(b1), k2=last(b2)))
    return out


def nrows(m):
    """The equality rows of the model: 1 per joint equality, 3 per connect."""
    return len(rows(m)) + 3 * len(connects(m))


def applies(m):
    """Whether the model has equality rows (then apply() edits the text)."""
    return bool(rows(m)) or bool(connects(m))


def linear(m):
    """Every row's J is constant: polycoef c2 = c3 = c4 = 0 (or no joint2)."""
    return all(d2 < 0 or not np.any(c[2:]) for _, _, d2, c, _, _ in rows(m))


def solimp_powers(m):
    """The equality rows' solimp powers (float32 values), for step.solimp_pow2 / adjoint.solimp_powers."""
    out = [float(np.float32(si[4])) for *_, si in rows(m)] + [float(np.float32(c["solimp"][4])) for c in connects(m)]
    if m.ntendon:
        from nablaengine.forest_engine import tendon
        out += tendon.solimp_powers(m)
    return out


def header(m):
    rs = rows(m)
    n = len(rs)
    poly = [x for r in rs for x in r[3]]
    sref = [x for r in rs for x in r[4]]
    simp = [x for r in rs for x in r[5]]
    return (f"#define NEQR {n}  // joint equality rows (forest_engine.equality), after the dof rows\n"
            f"#define EQ_LINEAR {int(linear(m))}  // every equality's polycoef is linear: J constant\n"
            f"__device__ const int c_eqd1[{n}] = {{{', '.join(str(r[1]) for r in rs)}}};  // joint1's dof\n"
            f"__device__ const int c_eqd2[{n}] = {{{', '.join(str(r[2]) for r in rs)}}};  // joint2's dof (-1: none)\n"
            f"__device__ const float c_eqpoly[{5 * n}] = {{{', '.join(map(_f32, poly))}}};  // polycoef\n"
            f"__device__ const float c_eqsolref[{2 * n}] = {{{', '.join(map(_f32, sref))}}};\n"
            f"__device__ const float c_eqsolimp[{5 * n}] = {{{', '.join(map(_f32, simp))}}};\n")


def _ints(xs):
    return ", ".join(str(int(x)) for x in xs) or "0"


def _floats(xs):
    return ", ".join(map(_f32, xs)) or "0.f"


def header_connect(m, extra_rows=0):
    """header() of a model with connect equalities: the joint tables (at least one entry) and the connect tables (a
    tendon build, forest_engine.tendon: also without connects, every table at least one entry; extra_rows: its tendon
    equality rows, the last NEQR rows)."""
    rs, cs = rows(m), connects(m)
    nj, nc = len(rs), len(cs)
    kc = max((len(c["dofs"]) for c in cs), default=1)
    n1 = max(nj, 1)
    c1 = max(nc, 1)

    def pad(c, key, fill):
        return list(c[key]) + [fill] * (kc - len(c[key]))

    ecpos = []
    for c in cs:
        where = {d: k for k, d in enumerate(c["dofs"])}
        ecpos += [where.get(i, -1) for i in range(m.nv)]
    if not cs:
        ecpos = [-1] * m.nv
    tab = lambda typ, name, n, vals, note="": (f"__device__ const {typ} {name}[{n}] = {{{vals}}};"
                                               + (f"  // {note}" if note else "") + "\n")
    return (f"#define NEQR {nj + 3 * nc + extra_rows}  // equality rows (forest_engine.equality) after the dof rows: "
            + ("joint, connect, tendon\n" if extra_rows else "joint, connect\n") +
            f"#define NEQJ {nj}  // joint equality rows (kind 5)\n"
            f"#define NEQC {nc}  // connect equalities (3 rows each, kind 6)\n"
            f"#define KEQC {kc}  // the largest connect support (dofs)\n"
            f"#define EQ_LINEAR {int(linear(m))}  // every equality's polycoef is linear: J constant\n"
            + tab("int", "c_eqd1", n1, _ints(r[1] for r in rs), "joint1's dof")
            + tab("int", "c_eqd2", n1, _ints(r[2] for r in rs), "joint2's dof (-1: none)")
            + tab("float", "c_eqpoly", 5 * n1, _floats(x for r in rs for x in r[3]), "polycoef")
            + tab("float", "c_eqsolref", 2 * n1, _floats(x for r in rs for x in r[4]))
            + tab("float", "c_eqsolimp", 5 * n1, _floats(x for r in rs for x in r[5]))
            + tab("int", "c_ecnd", c1, _ints(len(c["dofs"]) for c in cs), "connect: support size")
            + tab("unsigned char", "c_ecdof", c1 * kc, _ints(x for c in cs for x in pad(c, "dofs", 0)),
                  "its dofs, ascending")
            + tab("unsigned char", "c_ecside", c1 * kc, _ints(x for c in cs for x in pad(c, "side", 0)),
                  "bit 0: on chain 1, bit 1: on chain 2")
            + tab("signed char", "c_ecpar", c1 * kc, _ints(x for c in cs for x in pad(c, "par", -1)),
                  "the parent dof's support index (-1: none)")
            + tab("signed char", "c_ectw", c1 * kc, _ints(x for c in cs for x in pad(c, "tw", -1)),
                  "the twist of dS/dt (-1: dS/dt = 0)")
            + tab("signed char", "c_ecpos", c1 * m.nv, _ints(ecpos), "support index of dof i (-1: none)")
            + tab("int", "c_ecb", 2 * c1, _ints(x for c in cs for x in (c["b1"], c["b2"])), "bodies 1, 2 (0: world)")
            + tab("signed char", "c_eck", 2 * c1, _ints(x for c in cs for x in (c["k1"], c["k2"])),
                  "their twists' support index (-1: the world)")
            + tab("float", "c_ecanc", 6 * c1, _floats(x for c in cs for x in list(c["a1"]) + list(c["a2"])),
                  "anchors in the bodies' frames")
            + tab("float", "c_ecinvw", c1, _floats(c["invw"] for c in cs))
            + tab("float", "c_ecsolref", 2 * c1, _floats(x for c in cs for x in c["solref"]))
            + tab("float", "c_ecsolimp", 5 * c1, _floats(x for c in cs for x in c["solimp"])))


DEVICE = r"""// joint equality e (forest_engine.equality; MuJoCo Warp's _equality_joint): the coefficient J_2 of dof d2 in its row
// J = e_d1 + J_2 e_d2 (J_2 = -P'(q2); 0 without joint2), from qpos
__device__ __forceinline__ float eq_j2(const float* qpos, int e) {
  const float* c = c_eqpoly + 5 * e;
  if (c_eqd2[e] < 0) return 0.f;
#if EQ_LINEAR
  (void)qpos;
  return -c[1];
#else
  const float x = qpos[c_dqadr[c_eqd2[e]]];
  return -RFMA(x, RFMA(x, RFMA(x, RMUL(4.f, c[4]), RMUL(3.f, c[3])), RMUL(2.f, c[2])), c[1]);
#endif
}
// its dJ_2 / dq2 = -P''(q2) (the adjoint)
__device__ __forceinline__ float eq_j2d(const float* qpos, int e) {
#if EQ_LINEAR
  (void)qpos; (void)e;
  return 0.f;
#else
  const float* c = c_eqpoly + 5 * e;
  if (c_eqd2[e] < 0) return 0.f;
  const float x = qpos[c_dqadr[c_eqd2[e]]];
  return -RFMA(x, RFMA(x, RMUL(12.f, c[4]), RMUL(6.f, c[3])), RMUL(2.f, c[2]));
#endif
}
// its position pos = q1 - P(q2) (qpos0 = 0: capability T3); explicit roundings: the adjoint replays it bitwise
__device__ __forceinline__ float eq_pos(const float* qpos, int e) {
  const float* c = c_eqpoly + 5 * e;
  const float q1 = qpos[c_dqadr[c_eqd1[e]]];
  if (c_eqd2[e] < 0) return RSUB(q1, c[0]);
  const float x = qpos[c_dqadr[c_eqd2[e]]];
#if EQ_LINEAR
  return RSUB(q1, RFMA(x, c[1], c[0]));
#else
  return RSUB(q1, RFMA(x, RFMA(x, RFMA(x, RFMA(x, c[4], c[3]), c[2]), c[1]), c[0]));
#endif
}
// its invweight: dof_invweight0[d1] + dof_invweight0[d2]
__device__ __forceinline__ float eq_invw(int e) {
  return c_eqd2[e] < 0 ? c_dinvw[c_eqd1[e]] : RADD(c_dinvw[c_eqd1[e]], c_dinvw[c_eqd2[e]]);
}
// J v of equality e's row (v an NV vector)
__device__ __forceinline__ float eq_jv(const float* qpos, int e, const float* v) {
  const int d2 = c_eqd2[e];
  return d2 < 0 ? v[c_eqd1[e]] : v[c_eqd1[e]] + eq_j2(qpos, e) * v[d2];
}
// J' f of the equality rows on dof i, added to acc in equality order (f per row; rows from nefc_dof)
template <class S>
__device__ __forceinline__ void eq_jtf(const S& s, const float* f, int i, float& acc) {
#pragma unroll
  for (int e = 0; e < NEQR; e++) {
    const float fe = f[s.nefc_dof + e];
    if (c_eqd1[e] == i) acc += fe;
    if (c_eqd2[e] == i) acc += eq_j2(s.qpos, e) * fe;
  }
}
// H += J' D J of the equality rows (lane 0, in equality order: rows may share entries; the caller fences)
template <class S>
__device__ __forceinline__ void eq_hessian(S& s, int lane) {
  if (lane == 0)
    for (int e = 0; e < NEQR; e++) {
      const int d1 = c_eqd1[e], d2 = c_eqd2[e];
      const float D = s.efc_D[s.nefc_dof + e];
      s.L[TRI(d1, d1)] += D;
      if (d2 >= 0) {
        const float j2 = eq_j2(s.qpos, e), dj = D * j2;
        s.L[TRI(d2, d2)] += dj * j2;
        s.L[d1 > d2 ? TRI(d1, d2) : TRI(d2, d1)] += dj;
      }
    }
}

"""

EC_DEVICE = r"""// connect equalities (forest_engine.equality; MuJoCo C 3.17's mjEQ_CONNECT rows, Jdot v at the current velocities)
// explicitly rounded scalar operations (D1 has its own; the adjoint replays ec_eval over D1)
__device__ __forceinline__ float ec_m(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float ec_a(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float ec_s(float a, float b) { return __fsub_rn(a, b); }
template <class T>
__device__ __forceinline__ void ec_cross(const T* a, const T* b, T* o) {
  o[0] = ec_s(ec_m(a[1], b[2]), ec_m(a[2], b[1]));
  o[1] = ec_s(ec_m(a[2], b[0]), ec_m(a[0], b[2]));
  o[2] = ec_s(ec_m(a[0], b[1]), ec_m(a[1], b[0]));
}
// o = q v q^-1 (q a unit quaternion): v + q0 t + u x t, t = 2 u x v
template <class T>
__device__ __forceinline__ void ec_qrot(const T* q, const T* v, T* o) {
  T t[3], c[3];
  ec_cross(q + 1, v, t);
  for (int k = 0; k < 3; k++) t[k] = ec_m(T(2.f), t[k]);
  ec_cross(q + 1, t, c);
  for (int k = 0; k < 3; k++) o[k] = ec_a(ec_a(v[k], ec_m(q[0], t[k])), c[k]);
}
// the world position of the point a (body frame) of body b (0: the world)
template <class T>
__device__ __forceinline__ void ec_point(int b, const float* a, const T (*Q)[4], const T (*X)[3], T* p) {
  const T av[3] = {T(a[0]), T(a[1]), T(a[2])};
  if (b == 0) { for (int k = 0; k < 3; k++) p[k] = av[k]; return; }
  T r[3];
  ec_qrot(Q[b], av, r);
  for (int k = 0; k < 3; k++) p[k] = ec_a(X[b][k], r[k]);
}
// connect c at the body poses Q, X (per body) and the velocities v (per dof): pos = p1 - p2, vel = J v and jdv = Jdot v
// (3 each, world frame), and J's row rsel over the support into Jout (KEQC entries; rsel < 0: all three rows, 3 KEQC).
// Spatial motion vectors about O = p1 (a fixed point at this instant): the support dof k's subspace S_k = (w, u), the
// twists T_k = T_parent + S_k v_k, dS_k/dt = T_tw x S_k, A_i = sum over chain i of v_k dS_k/dt; Jdot v = A_1(p1) +
// w_1 x pdot_1 - A_2(p2) - w_2 x pdot_2. Only the support bodies' poses and v enter (the adjoint's replay seeds them)
template <class T>
__device__ void ec_eval(int c, const T (*Q)[4], const T (*X)[3], const T* v, T* pos, T* vel, T* jdv, T* Jout,
                        int rsel) {
  const int nd = c_ecnd[c], b1 = c_ecb[2 * c], b2 = c_ecb[2 * c + 1];
  const unsigned char* dof = c_ecdof + KEQC * c;
  T p1[3], p2[3], r2[3];
  ec_point(b1, c_ecanc + 6 * c, Q, X, p1);
  ec_point(b2, c_ecanc + 6 * c + 3, Q, X, p2);
  for (int k = 0; k < 3; k++) { pos[k] = ec_s(p1[k], p2[k]); r2[k] = -pos[k]; }  // r2 = p2 - O
  T S[KEQC][6], Tw[KEQC][6];
  for (int k = 0; k < nd; k++) {
    const int i = dof[k], b = c_dofbody[i], jt = JT_OF(b), t = i - c_dadr[b];
    T w[3] = {T(0.f), T(0.f), T(0.f)}, u[3];
    if (jt == JSLIDE || (jt == JFREE && t < 3)) {
      const T e[3] = {T(t == 0 ? 1.f : 0.f), T(t == 1 ? 1.f : 0.f), T(t == 2 ? 1.f : 0.f)};
      if (jt == JSLIDE) {
        const T ax[3] = {T(c_axis[3 * b]), T(c_axis[3 * b + 1]), T(c_axis[3 * b + 2])};
        ec_qrot(Q[b], ax, u);
      } else for (int q = 0; q < 3; q++) u[q] = e[q];
    } else {
      T ax[3];
      if (jt == JFREE) { for (int q = 0; q < 3; q++) ax[q] = T(q == t - 3 ? 1.f : 0.f); }
      else { for (int q = 0; q < 3; q++) ax[q] = T(c_axis[3 * b + q]); }
      ec_qrot(Q[b], ax, w);
      T d[3];
      for (int q = 0; q < 3; q++) d[q] = ec_s(p1[q], X[b][q]);
      ec_cross(w, d, u);
    }
    const int pa = c_ecpar[KEQC * c + k];
    const T vi = v[i];
    for (int q = 0; q < 3; q++) {
      S[k][q] = w[q]; S[k][3 + q] = u[q];
      Tw[k][q] = ec_a(pa >= 0 ? Tw[pa][q] : T(0.f), ec_m(w[q], vi));
      Tw[k][3 + q] = ec_a(pa >= 0 ? Tw[pa][3 + q] : T(0.f), ec_m(u[q], vi));
    }
  }
  T A1[3] = {T(0.f), T(0.f), T(0.f)}, A2w[3] = {T(0.f), T(0.f), T(0.f)}, A2u[3] = {T(0.f), T(0.f), T(0.f)};
  for (int q = 0; q < 3; q++) vel[q] = T(0.f);
  for (int k = 0; k < nd; k++) {
    const int sd = c_ecside[KEQC * c + k], tw = c_ectw[KEQC * c + k];
    const T vi = v[dof[k]];
    if (tw >= 0) {  // dS/dt = T x S = (om x w, om x u + vo x w)
      T dw[3], du[3], x1[3];
      ec_cross(Tw[tw], S[k], dw);
      ec_cross(Tw[tw], S[k] + 3, du);
      ec_cross(Tw[tw] + 3, S[k], x1);
      for (int q = 0; q < 3; q++) du[q] = ec_a(du[q], x1[q]);
      for (int q = 0; q < 3; q++) {
        if (sd & 1) A1[q] = ec_a(A1[q], ec_m(vi, du[q]));
        if (sd & 2) { A2w[q] = ec_a(A2w[q], ec_m(vi, dw[q])); A2u[q] = ec_a(A2u[q], ec_m(vi, du[q])); }
      }
    }
    T col[3], wr[3];  // J's column: chain 1 u (at O = p1), chain 2 -(u + w x r2), both w x pos
    ec_cross(S[k], r2, wr);
    for (int q = 0; q < 3; q++)
      col[q] = sd == 3 ? -wr[q] : sd == 1 ? S[k][3 + q] : -ec_a(S[k][3 + q], wr[q]);
    for (int q = 0; q < 3; q++) vel[q] = ec_a(vel[q], ec_m(col[q], vi));
    if (rsel < 0) { for (int q = 0; q < 3; q++) Jout[KEQC * q + k] = col[q]; }
    else Jout[k] = col[rsel];
  }
  const int k1 = c_eck[2 * c], k2 = c_eck[2 * c + 1];
  T j1[3] = {T(0.f), T(0.f), T(0.f)}, j2[3], t2[3], pd2[3] = {T(0.f), T(0.f), T(0.f)};
  if (k1 >= 0) {  // A_1 at p1 = O, plus w_1 x pdot_1 (pdot_1 = vo_1)
    ec_cross(Tw[k1], Tw[k1] + 3, j1);
  }
  for (int q = 0; q < 3; q++) j1[q] = ec_a(A1[q], j1[q]);
  ec_cross(A2w, r2, j2);  // A_2 at p2, plus w_2 x pdot_2 (pdot_2 = vo_2 + w_2 x r2)
  for (int q = 0; q < 3; q++) j2[q] = ec_a(A2u[q], j2[q]);
  if (k2 >= 0) {
    ec_cross(Tw[k2], r2, t2);
    for (int q = 0; q < 3; q++) pd2[q] = ec_a(Tw[k2][3 + q], t2[q]);
    ec_cross(Tw[k2], pd2, t2);
    for (int q = 0; q < 3; q++) j2[q] = ec_a(j2[q], t2[q]);
  }
  for (int q = 0; q < 3; q++) jdv[q] = ec_s(j1[q], j2[q]);
}
// the stored J of the connect rows (per world, KEQC entries per row): right after con_J in struct Sim / struct Hand
template <class S>
__device__ __forceinline__ float* ec_J(const S& s) {
  return (float*)(s.con_J + MAXCON);
}
// J v of connect c's row r (v an NV vector)
template <class S>
__device__ __forceinline__ float ec_jv(const S& s, int c, int r, const float* v) {
  const float* J = ec_J(s) + (3 * c + r) * KEQC;
  const unsigned char* dof = c_ecdof + KEQC * c;
  float acc = 0.f;
  for (int k = 0; k < c_ecnd[c]; k++) acc += J[k] * v[dof[k]];
  return acc;
}

"""

EC_JTF_OLD = """    if (c_eqd2[e] == i) acc += eq_j2(s.qpos, e) * fe;
  }
}
"""
EC_JTF_NEW = """    if (c_eqd2[e] == i) acc += eq_j2(s.qpos, e) * fe;
  }
  const float* EJ = ec_J(s);  // then the connect rows, in connect order
  for (int c = 0; c < NEQC; c++) {
    const int k = c_ecpos[NV * c + i];
    if (k >= 0) {
      const int r0 = s.nefc_dof + NEQJ + 3 * c;
      const float* J = EJ + 3 * c * KEQC;
      acc += J[k] * f[r0] + J[KEQC + k] * f[r0 + 1] + J[2 * KEQC + k] * f[r0 + 2];
    }
  }
}
"""
EC_HESS_OLD = """        s.L[d1 > d2 ? TRI(d1, d2) : TRI(d2, d1)] += dj;
      }
    }
}
"""
EC_HESS_NEW = """        s.L[d1 > d2 ? TRI(d1, d2) : TRI(d2, d1)] += dj;
      }
    }
  __syncwarp();
  for (int c = 0; c < NEQC; c++) {  // the connect rows: lane a adds the entries (a, b <= a) of its support (distinct per
    const int r0 = s.nefc_dof + NEQJ + 3 * c, nd = c_ecnd[c];  // lane), the connects one after another
    const float* J = ec_J(s) + 3 * c * KEQC;
    const unsigned char* dof = c_ecdof + KEQC * c;
    if (lane < nd) {
      const int da = dof[lane];
      const float g0 = s.efc_D[r0] * J[lane], g1 = s.efc_D[r0 + 1] * J[KEQC + lane],
                  g2 = s.efc_D[r0 + 2] * J[2 * KEQC + lane];
      for (int b = 0; b <= lane; b++) s.L[TRI(da, dof[b])] += g0 * J[b] + g1 * J[KEQC + b] + g2 * J[2 * KEQC + b];
    }
    __syncwarp();
  }
}
"""

ROWS_INSERT_AT = "  con_jv(s, s.qvel, lane);  // contact J qvel for the contact rows' velocities"
ROWS_INSERT = """  for (int e = lane; e < NEQR; e += 32) {  // the equality rows (kind 5) after the dof rows (forest_engine.equality)
    const int r = ndof + e;
    s.efc_kind[r] = 5; s.efc_a[r] = e; s.efc_b[r] = 0;
  }
  ndof += NEQR;  // (jobs j in [ndof - NEQR, ndof): equality j - ndof + NEQR; s.nefc_dof below: the dof rows only)
"""
JOB_OLD = """      if (!con) {  // dof row j: frictionloss (at 0) or limit
        i = s.efc_a[j];
"""
JOB_NEW = """      const bool eqr = !con && j >= ndof - NEQR;  // an equality row (forest_engine.equality)
      float eqvel = 0.f;
      if (eqr) {
        const int e = j - (ndof - NEQR);
        pos = eq_pos(s.qpos, e);
        eqvel = eq_jv(s.qpos, e, s.qvel);
        iw = eq_invw(e);
        sref = c_eqsolref + 2 * e; simp = c_eqsolimp + 5 * e;
      } else if (!con) {  // dof row j: frictionloss (at 0) or limit
        i = s.efc_a[j];
"""
AREF_OLD = "        s.efc_D[j] = D; s.efc_aref[j] = -k * imp * pos - bb * (s.efc_b[j] * s.qvel[i]);\n"
AREF_NEW = ("        s.efc_D[j] = D; s.efc_aref[j] = -k * imp * pos - bb * (eqr ? eqvel : s.efc_b[j] * s.qvel[i]);\n")


EC_ROWS_INSERT = """  for (int e = lane; e < NEQR; e += 32) {  // the equality rows after the dof rows (forest_engine.equality): joint
    const int r = ndof + e, c = (e - NEQJ) / 3;  // (kind 5), then 3 per connect (kind 6, efc_b the row)
    if (e < NEQJ) { s.efc_kind[r] = 5; s.efc_a[r] = e; s.efc_b[r] = 0; }
    else { s.efc_kind[r] = 6; s.efc_a[r] = c; s.efc_b[r] = e - NEQJ - 3 * c; }
  }
  ndof += NEQR;  // (jobs j in [ndof - NEQR, ndof): equality row j - ndof + NEQR; s.nefc_dof below: the dof rows only)
"""
EC_JOB_NEW = """      const bool eqr = !con && j >= ndof - NEQR;  // an equality row (forest_engine.equality)
      float eqvel = 0.f, eqpos = 0.f, eqjd = 0.f;
      if (eqr) {
        const int e = j - (ndof - NEQR);
        if (e < NEQJ) {  // joint
          pos = eq_pos(s.qpos, e); eqpos = pos;
          eqvel = eq_jv(s.qpos, e, s.qvel);
          iw = eq_invw(e);
          sref = c_eqsolref + 2 * e; simp = c_eqsolimp + 5 * e;
        } else {  // connect c's row r: J row r into ec_J, the impedance at |pos|
          const int c = (e - NEQJ) / 3, r = e - NEQJ - 3 * c;
          float p3[3], v3[3], jd3[3];
          ec_eval<float>(c, s.Q, s.X, s.qvel, p3, v3, jd3, ec_J(s) + (3 * c + r) * KEQC, r);
          pos = __fsqrt_rn(RFMA(p3[2], p3[2], RFMA(p3[1], p3[1], RMUL(p3[0], p3[0]))));
          eqpos = p3[r]; eqvel = v3[r]; eqjd = jd3[r];
          iw = c_ecinvw[c];
          sref = c_ecsolref + 2 * c; simp = c_ecsolimp + 5 * c;
        }
      } else if (!con) {  // dof row j: frictionloss (at 0) or limit
        i = s.efc_a[j];
"""
EC_AREF_NEW = ("        s.efc_D[j] = D; s.efc_aref[j] = eqr ? -k * imp * eqpos - bb * eqvel - eqjd\n"
               "                                            : -k * imp * pos - bb * (s.efc_b[j] * s.qvel[i]);\n")
EC_CONJ = "  alignas(16) float con_J[MAXCON][CD][KDOF];  // (16-byte aligned: con_jv's 128-bit loads)\n"


def _shared_hooks():
    return (
        ("__device__ __forceinline__ int row_piece(int r, float x, float D, float F, float* ds, float* dd) {\n",
         "__device__ __forceinline__ int row_piece(int r, float x, float D, float F, float* ds, float* dd) {\n"
         "  if (F < 0.f) { *ds = D * x; *dd = D; return 1; }  // an equality row (row_floss -1): always quadratic\n"),
        ("  return NFLOSS > 0 && r < NFLOSS ? c_floss[s.efc_a[r]] : 0.f;\n",
         "  if ((unsigned)(r - s.nefc_dof) < (unsigned)NEQR) return -1.f;  // equality rows: row_piece's quadratic piece\n"
         "  return NFLOSS > 0 && r < NFLOSS ? c_floss[s.efc_a[r]] : 0.f;\n"),
        ("  SUBPROF(21);\n  SUBPROF(22);\n",
         "#pragma unroll\n  for (int t = 0; t < 2; t++)\n    if (lane + 32 * t < NV) eq_jtf(s, f, lane + 32 * t, acc[t]);\n"
         "  SUBPROF(21);\n  SUBPROF(22);\n"),
        ("    out[i] = acc;\n  }\n  __syncwarp();\n  SUBPROF(22);\n",
         "    eq_jtf(s, f, i, acc);\n    out[i] = acc;\n  }\n  __syncwarp();\n  SUBPROF(22);\n"),
        ("  __syncwarp();\n  SUBPROF(28);\n",
         "  __syncwarp();\n  eq_hessian(s, lane);  // the equality rows (forest_engine.equality)\n  __syncwarp();\n"
         "  SUBPROF(28);\n"),
        ("    s.nefc_dof = ndof;  // dof rows (frictionloss, limits) come first; contact rows follow\n",
         "    s.nefc_dof = ndof - NEQR;  // dof rows (frictionloss, limits) come first, then the equality rows, then the "
         "contacts\n"),
    )


def connect_device():
    """DEVICE of a model with connects: the joint loops over NEQJ, eq_jtf / eq_hessian with the connect rows."""
    d = DEVICE.replace("e < NEQR;", "e < NEQJ;")
    for a, b in ((EC_JTF_OLD, EC_JTF_NEW), (EC_HESS_OLD, EC_HESS_NEW)):
        assert d.count(a) == 1
        d = d.replace(a, b)
    assert "NEQR" not in d
    return EC_DEVICE + d


def hooks(m):
    """The forward's edits (anchor, replacement); each anchor must occur exactly once (the connect build's con_J line:
    twice, struct Sim and struct Hand)."""
    if connects(m):
        return hooks_connect(m)
    return (
        ("#define NFLOSS ", header(m) + "#define NFLOSS "),
        ("template <class T> struct dk_is_ptr { static constexpr bool value = false; };\n",
         DEVICE + "template <class T> struct dk_is_ptr { static constexpr bool value = false; };\n"),
    ) + _shared_hooks() + (
        ("    if (kind == 0) val = v[a];\n",
         "    if (kind == 5) val = eq_jv(s.qpos, a, v);  // equality row (forest_engine.equality)\n"
         "    else if (kind == 0) val = v[a];\n"),
        (ROWS_INSERT_AT, ROWS_INSERT + ROWS_INSERT_AT),
        (JOB_OLD, JOB_NEW),
        (AREF_OLD, AREF_NEW),
    )


def hooks_connect(m, extra_rows=0):
    """hooks() of a model with connects; a tendon build (forest_engine.tendon) takes it also without connects
    (extra_rows: its tendon equality rows)."""
    conj = ("  float ec_Jrows[NEQC * 3 * KEQC];  // the connect rows' J (forest_engine.equality: ec_J, right after "
            "con_J)\n")
    if not connects(m):
        conj = conj.replace("[NEQC * 3 * KEQC]", "[3 * KEQC]")
    return (
        ("#define NFLOSS ", header_connect(m, extra_rows) + "#define NFLOSS "),
        ("template <class T> struct dk_is_ptr { static constexpr bool value = false; };\n",
         connect_device() + "template <class T> struct dk_is_ptr { static constexpr bool value = false; };\n"),
    ) + _shared_hooks() + (
        ("    if (kind == 0) val = v[a];\n",
         "    if (kind == 6) val = ec_jv(s, a, b, v);  // connect row (forest_engine.equality)\n"
         "    else if (kind == 5) val = eq_jv(s.qpos, a, v);  // joint equality row\n"
         "    else if (kind == 0) val = v[a];\n"),
        (ROWS_INSERT_AT, EC_ROWS_INSERT + ROWS_INSERT_AT),
        (JOB_OLD, EC_JOB_NEW),
        (AREF_OLD, EC_AREF_NEW),
        (EC_CONJ, EC_CONJ + conj),
    )


ADJ_DEVICE = r"""
// A5 / A6, joint equalities (forest_engine.equality): every row is on its quadratic piece (no zone bit needed); lane 0
// in equality order (rows may share dofs: deterministic, no atomics). P_q, P_v of the row's dofs; jtf (debug build):
// also the rows' J' f into s.jtf
__device__ __noinline__ void adj_equality(Adj& s, int lane, bool jtf) {
  if (lane == 0)
    for (int e = 0; e < NEQR; e++) {
      const int d1 = c_eqd1[e], d2 = c_eqd2[e];
      const float pos = eq_pos(s.q, e), j2 = eq_j2(s.q, e), j2d = eq_j2d(s.q, e);  // the forward's pos, bitwise
      float ri[6];
      adj_row_impedance(c_eqsolref + 2 * e, c_eqsolimp + 5 * e, pos, eq_invw(e), ri);
      const float k = ri[0], b = ri[1], imp = ri[2], dimp = ri[3], D = ri[4], dD = ri[5];
      float ja = s.a[d1], jv = s.v[d1], jl = s.lam[d1];
      if (d2 >= 0) { ja += j2 * s.a[d2]; jv += j2 * s.v[d2]; jl += j2 * s.lam[d2]; }
      const float x = ja + k * imp * pos + b * jv;
      const float sp = jl * (dD * x + D * (k * (imp + dimp * pos))), pv = D * b * jl;
      s.Pq[d1] += sp;
      s.Pv[d1] += pv;
      if (d2 >= 0) {
        s.Pq[d2] += sp * j2 + j2d * (D * ((s.a[d2] + b * s.v[d2]) * jl + x * s.lam[d2]));
        s.Pv[d2] += pv * j2;
      }
#ifdef DK_ADJ_DEBUG
      if (jtf) { s.jtf[d1] += -D * x; if (d2 >= 0) s.jtf[d2] += -D * x * j2; }
#else
      (void)jtf;
#endif
    }
  __syncwarp();
}
"""


ADJ_EC_DEVICE = r"""
// A5 / A6, connect equalities (forest_engine.equality): Psi_c = sum over the 3 rows of D(|pos|) x_r (J lam)_r with
// x = J a* + k imp pos + b J v + Jdot v (a*, lam fixed), differentiated exactly by forward-mode replays of the forward's
// ec_eval over D1: support lane k replays it once with the pose tangent of dof k (the bodies of its subtree, c_ecsub,
// turn about its axis or translate along it: dX_b = w x (X_b - x_k) + t, dQ_b = (0, w) Q_b / 2; the axes and anchors
// of the dofs below follow from the seeded rotations) for P_q,k and once with qvel_k's tangent for P_v,k. The replay's
// values are bitwise the forward's (the same explicitly rounded operations on the same poses). Lanes write distinct
// dofs; the connects one after another (deterministic). d|pos| = 0 at pos = 0 (a subgradient)
__device__ __forceinline__ D1 ec_m(D1 a, D1 b) { return a * b; }
__device__ __forceinline__ D1 ec_a(D1 a, D1 b) { return a + b; }
__device__ __forceinline__ D1 ec_s(D1 a, D1 b) { return a - b; }
// the directional derivative of Psi_c of connect c with the poses Q, X and the velocities v (D1: their tangents);
// jtf (debug build): the rows' J' f onto the support (the values)
__device__ __noinline__ float adj_connect_dpsi(Adj& s, int c, const D1 (*Q)[4], const D1 (*X)[3], const D1* v, int kj) {
  D1 pos[3], vel[3], jdv[3], J[3 * KEQC];
  ec_eval<D1>(c, Q, X, v, pos, vel, jdv, J, -1);
  const float n = __fsqrt_rn(RFMA(pos[2].v, pos[2].v, RFMA(pos[1].v, pos[1].v, RMUL(pos[0].v, pos[0].v))));
  float ri[6];
  adj_row_impedance(c_ecsolref + 2 * c, c_ecsolimp + 5 * c, n, c_ecinvw[c], ri);
  const float k = ri[0], b = ri[1];
  const float dn = n > 0.f ? (pos[0].v * pos[0].d + pos[1].v * pos[1].d + pos[2].v * pos[2].d) / n : 0.f;
  const D1 imp(ri[2], ri[3] * dn), D(ri[4], ri[5] * dn);
  const unsigned char* dof = c_ecdof + KEQC * c;
  D1 psi(0.f);
  for (int r = 0; r < 3; r++) {
    D1 ja(0.f), jl(0.f);
    for (int q = 0; q < c_ecnd[c]; q++) {
      ja = ja + J[KEQC * r + q] * s.a[dof[q]];
      jl = jl + J[KEQC * r + q] * s.lam[dof[q]];
    }
    const D1 x = ja + k * imp * pos[r] + b * vel[r] + jdv[r];
    psi = psi + D * x * jl;
#ifdef DK_ADJ_DEBUG
    if (kj >= 0) s.jtf[dof[kj]] += -D.v * x.v * J[KEQC * r + kj].v;
#endif
  }
  (void)kj;
  return psi.d;
}
__device__ __noinline__ void adj_connect(Adj& s, int lane, bool jtf) {
  for (int c = 0; c < NEQC; c++) {
    const int nd = c_ecnd[c];
    if (lane < nd) {
      const unsigned char* dof = c_ecdof + KEQC * c;
      D1 Q[NB][4], X[NB][3], v[NV];
      for (int i = 0; i < NV; i++) v[i] = D1(s.v[i]);
      for (int q = 0; q < nd; q++) {  // the support bodies at rest
        const int b = c_dofbody[dof[q]];
        for (int t = 0; t < 4; t++) Q[b][t] = D1(s.Q[b][t]);
        for (int t = 0; t < 3; t++) X[b][t] = D1(s.X[b][t]);
      }
      const int i = dof[lane];
      v[i].d = 1.f;  // qvel_i's tangent
      s.Pv[i] += adj_connect_dpsi(s, c, Q, X, v, jtf ? lane : -1);
      v[i].d = 0.f;
      const int bk = c_dofbody[i], jt = JT_OF(bk), ti = i - c_dadr[bk];
      float w[3] = {0.f, 0.f, 0.f}, tr[3] = {0.f, 0.f, 0.f};  // dof i's pose tangent: turn w about X_bk, translate tr
      if (jt == JFREE && ti < 3) tr[ti] = 1.f;
      else {
        float ax[3] = {0.f, 0.f, 0.f};
        if (jt == JFREE) ax[ti - 3] = 1.f;
        else for (int t = 0; t < 3; t++) ax[t] = c_axis[3 * bk + t];
        qrot(s.Q[bk], ax, jt == JSLIDE ? tr : w);
      }
      const unsigned sub = c_ecsub[KEQC * c + lane];
      for (int q = 0; q < nd; q++) {
        if (!((sub >> q) & 1u)) continue;
        const int b = c_dofbody[dof[q]];
        const float* Qb = s.Q[b];
        const float d[3] = {s.X[b][0] - s.X[bk][0], s.X[b][1] - s.X[bk][1], s.X[b][2] - s.X[bk][2]};
        X[b][0].d = w[1] * d[2] - w[2] * d[1] + tr[0];
        X[b][1].d = w[2] * d[0] - w[0] * d[2] + tr[1];
        X[b][2].d = w[0] * d[1] - w[1] * d[0] + tr[2];
        Q[b][0].d = -0.5f * (w[0] * Qb[1] + w[1] * Qb[2] + w[2] * Qb[3]);
        Q[b][1].d = 0.5f * (Qb[0] * w[0] + w[1] * Qb[3] - w[2] * Qb[2]);
        Q[b][2].d = 0.5f * (Qb[0] * w[1] + w[2] * Qb[1] - w[0] * Qb[3]);
        Q[b][3].d = 0.5f * (Qb[0] * w[2] + w[0] * Qb[2] - w[1] * Qb[1]);
      }
      s.Pq[i] += adj_connect_dpsi(s, c, Q, X, v, -1);
    }
    __syncwarp();
  }
  (void)jtf;
}
"""


def adj_connect_device(m):
    """ADJ_EC_DEVICE with the subtree table c_ecsub: per connect and support dof k, bit q set when dof q's body moves
    with dof k (its body or a descendant)."""
    from nablaengine.dual import DUAL_DEVICE
    cs = connects(m)
    kc = max((len(c["dofs"]) for c in cs), default=1)
    sub = []
    for c in cs:
        for d in c["dofs"]:
            bk = int(m.dof_bodyid[d])
            mask = 0
            for q, dq in enumerate(c["dofs"]):
                b = int(m.dof_bodyid[dq])
                while b > 0 and b != bk:
                    b = int(m.body_parentid[b])
                mask |= (b == bk) << q
            sub.append(mask)
        sub += [0] * (kc - len(c["dofs"]))
    tab = (f"__device__ const unsigned c_ecsub[{max(len(cs), 1) * kc}] = {{{', '.join(f'{x}u' for x in sub) or '0u'}}};  "
           "// connect, support dof k: the support dofs whose bodies move with dof k\n")
    return DUAL_DEVICE + tab + ADJ_EC_DEVICE


def adj_hooks(m, src):
    """The adjoint's edits: adj_equality after adj_limits (production and the debug build's limit group), the
    contacts' first row after the equality rows."""
    return _adj_hooks(m, src, bool(connects(m)))


def adj_hooks_connect(m, src):
    """adj_hooks() of the connect build (a tendon build: also without connects)."""
    return _adj_hooks(m, src, True)


def _adj_hooks(m, src, conn):
    k = src.index("  // the contact's first row")
    i, j = src.rindex("\n", 0, k) + 1, src.index("\n", k)
    line = src[i:j]
    assert line.count("nlim + c + ") == 1, f"the adjoint's contact row offset: {line!r}"
    dev = ADJ_DEVICE
    if conn:
        dev = dev.replace("e < NEQR;", "e < NEQJ;")
        a = "  __syncwarp();\n}\n"
        assert dev.endswith(a)
        dev = adj_connect_device(m) + dev[:-len(a)] + "  __syncwarp();\n  adj_connect(s, lane, jtf);\n}\n"
    return (
        ("template <bool DBG>\n__device__ __noinline__ void adj_limits(",
         dev + "template <bool DBG>\n__device__ __noinline__ void adj_limits("),
        (line, line.replace("nlim + c + ", "nlim + NEQR + c + ")),
        ("  adj_limits<false>(s, lane, nullptr);  // A5, A6: P_q, P_v of the limit rows' dofs (marks limits)\n",
         "  adj_limits<false>(s, lane, nullptr);  // A5, A6: P_q, P_v of the limit rows' dofs (marks limits)\n"
         "  adj_equality(s, lane, false);  // A5, A6: the equality rows (forest_engine.equality)\n"),
        ("    adj_limits<true>(s, lane, dbg);\n",
         "    adj_limits<true>(s, lane, dbg);\n    adj_equality(s, lane, false);  // (in the limit group)\n"),
        ("    adj_jtf(s, dbg, lane);\n",
         "    adj_jtf(s, dbg, lane);\n    for (int i = lane; i < NV; i += 32) { s.Pq[i] = 0.f; s.Pv[i] = 0.f; }\n"
         "    __syncwarp();\n    adj_equality(s, lane, true);  // (the equality rows' J'f)\n"),
    )


def check_rows(m, max_rows_total, extra=0):
    """Raise ValueError unless the dof rows and the equality rows (and extra: a tendon build's tendon rows at most,
    forest_engine.tendon) fit the row capacity (they are never dropped)."""
    ndof = int((m.dof_frictionloss > 0).sum()) + int(sum(
        1 for j in range(m.njnt) if m.jnt_limited[j] and int(m.jnt_type[j]) in (2, 3)))
    neq = nrows(m) + extra
    reserve = 64 if m.nv <= 32 else max_rows_total
    if ndof + neq > min(reserve, max_rows_total):
        raise ValueError(f"equality rows: {ndof} dof rows + {neq} equality rows exceed the row reserve "
                         f"min({reserve}, max_rows={max_rows_total}) (forest_engine.equality)")
    if (connects(m) or extra) and neq + 1 > max_rows_total:
        raise ValueError(f"equality rows: max_rows={max_rows_total} leaves no row beyond the {neq} equality rows")


def apply(src, m, nefc_rows):
    """src (adjoint.cuda_source's unit, after model_hooks and elliptic) with the equality edits (a model with
    equality rows), else src unchanged. nefc_rows: the build's row capacity (step.nefc)."""
    from nablaengine.forest_engine import tendon
    if tendon.applies(m):
        return tendon.apply(src, m, nefc_rows)
    if not applies(m):
        return src
    bad = check(m)
    if bad:
        raise ValueError(f"equality constraints the forest engine does not support: {bad[:4]}")
    check_rows(m, nefc_rows)
    has_adj = "__device__ __noinline__ void adj_limits(" in src
    for a, b in hooks(m) + (adj_hooks(m, src) if has_adj else ()):
        k = src.count(a)
        if a == EC_CONJ and k == 2:
            src = src.replace(a, b)
            continue
        if k != 1:
            raise RuntimeError(f"forest equality: anchor found {k} times, expected 1: {a[:100]!r}")
        src = src.replace(a, b)
    return src

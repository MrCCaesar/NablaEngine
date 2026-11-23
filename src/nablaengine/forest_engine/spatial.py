# The tendon path code here (sp_wrap_circle, sp_wrap_inside, sp_wrap_local, the segment rules of sp_compute) follows
# MuJoCo Warp's util_misc.wrap / wrap_circle / wrap_inside and smooth._spatial_site_tendon / _spatial_geom_tendon
# (mujoco_warp 3.17, Copyright 2025 The Newton Developers, Apache License 2.0), which mirror MuJoCo's mju_wrap /
# mju_wrapCircle / mju_wrapInside and mj_tendon; the templated form, the explicit roundings and the derivatives are this
# engine's.
"""Spatial tendons in the forest engine (with forest_engine.tendon): length and Jacobian of site paths with pulleys and
sphere / cylinder wrapping (side sites, inside side sites), forward and adjoint."""
import mujoco
import numpy as np

W = mujoco.mjtWrap
KIND_SITE, KIND_SPHERE, KIND_CYL = 0, 1, 2
MAX_KTEN = 32


MAX_TS_WORDS = 6144


def storage_words(m, ts, kten, nact):
    """TS_WORDS of a spatial build (ts the used tendons) with J in KTEN-strided rows."""
    return len(ts) + len(ts) * kten + 2 * nact


def _sizes(m):
    from nablaengine.forest_engine import muscle
    from nablaengine.forest_engine import tendon as TEN
    ts = TEN.used(m)
    return ts, [len(TEN.support_list(m, t)) for t in ts], len(TEN.actuators(m)), muscle.act_words(m)


def compact_on(m):
    """The packed layout (compact()): a spatial build whose KTEN-strided storage exceeds MAX_TS_WORDS."""
    from nablaengine.forest_engine import tendon as TEN
    if not TEN.spatial_build(m):
        return False
    ts, sup, nact, na = _sizes(m)
    return storage_words(m, ts, max(sup), nact) + na > MAX_TS_WORDS


def words(m):
    """TS_WORDS of a spatial build (the activation words included): KTEN-strided J rows, or the packed layout's (each
    tendon's J over its own support) when the strided one exceeds MAX_TS_WORDS."""
    ts, sup, nact, na = _sizes(m)
    if compact_on(m):
        return len(ts) + max(sum(sup), 1) + 2 * nact + na
    return storage_words(m, ts, max(sup), nact) + na


def is_spatial(m, t):
    if t >= m.ntendon:
        return False
    adr, n = int(m.tendon_adr[t]), int(m.tendon_num[t])
    return any(int(m.wrap_type[w]) != int(W.mjWRAP_JOINT) for w in range(adr, adr + n))


def path(m, t):
    """The segments of spatial tendon t: [dict(kind, s0, s1, g (-1), side (-1), scale)] in path order (MuJoCo's rules:
    a pulley sets the scale of what follows and breaks the path; site-site is straight; site-geom-site may wrap).
    Raises ValueError for a path MuJoCo would not compile (kept as a guard)."""
    adr, n = int(m.tendon_adr[t]), int(m.tendon_num[t])
    typ = [int(m.wrap_type[w]) for w in range(adr, adr + n)]
    obj = [int(m.wrap_objid[w]) for w in range(adr, adr + n)]
    prm = [float(m.wrap_prm[w]) for w in range(adr, adr + n)]
    out, scale, j = [], 1.0, 0
    while j < n - 1:
        t0, t1 = typ[j], typ[j + 1]
        if t0 == int(W.mjWRAP_PULLEY):
            scale = 1.0 / prm[j]
            j += 1
            continue
        if t1 == int(W.mjWRAP_PULLEY):
            j += 1
            continue
        if t0 != int(W.mjWRAP_SITE):
            raise ValueError(f"tendon {t}: wrap element {j} of type {t0} where a site is expected")
        if t1 == int(W.mjWRAP_SITE):
            out.append(dict(kind=KIND_SITE, s0=obj[j], s1=obj[j + 1], g=-1, side=-1, scale=scale))
            j += 1
            continue
        if t1 in (int(W.mjWRAP_SPHERE), int(W.mjWRAP_CYLINDER)):
            if j + 2 >= n or typ[j + 2] != int(W.mjWRAP_SITE):
                raise ValueError(f"tendon {t}: a wrap geom not followed by a site")
            g = obj[j + 1]
            gt = int(m.geom_type[g])
            kind = KIND_SPHERE if gt == int(mujoco.mjtGeom.mjGEOM_SPHERE) else (
                KIND_CYL if gt == int(mujoco.mjtGeom.mjGEOM_CYLINDER) else -1)
            if kind < 0:
                raise ValueError(f"tendon {t}: wrap geom {g} is neither a sphere nor a cylinder")
            out.append(dict(kind=kind, s0=obj[j], s1=obj[j + 2], g=g, side=int(round(prm[j + 1])), scale=scale))
            j += 2
            continue
        raise ValueError(f"tendon {t}: wrap element {j + 1} of type {t1}")
    return out


def bodies(m, t):
    """The bodies spatial tendon t's path touches (its sites' and wrap geoms' bodies; side sites only decide)."""
    bs = set()
    for s in path(m, t):
        bs |= {int(m.site_bodyid[s["s0"]]), int(m.site_bodyid[s["s1"]])}
        if s["g"] >= 0:
            bs.add(int(m.geom_bodyid[s["g"]]))
    return bs


def _chain(m, b):
    out = []
    while b > 0:
        out += range(int(m.body_dofadr[b]), int(m.body_dofadr[b]) + int(m.body_dofnum[b]))
        b = int(m.body_parentid[b])
    return out


def support(m, t):
    """The dofs J of spatial tendon t can be nonzero on, ascending: the chains of the bodies its path touches minus the
    chain they all share."""
    chains = [set(_chain(m, b)) for b in bodies(m, t)]
    if not chains:
        return []
    return sorted(set.union(*chains) - set.intersection(*chains))


def check_tendon(m, t):
    """[reason] spatial tendon t is not supported (empty: supported); the reasons start with "spatial tendon"."""
    try:
        p = path(m, t)
    except ValueError as e:
        return [f"spatial tendon: {e}"]
    why = []
    if not p:
        why.append("spatial tendon: an empty path")
    if len(support(m, t)) > MAX_KTEN:
        why.append(f"spatial tendon: Jacobian support of {len(support(m, t))} dofs (at most {MAX_KTEN})")
    for b in bodies(m, t):
        if b > 0 and int(m.body_jntnum[b]) != 1:
            why.append("spatial tendon: a path body without exactly one joint")
            break
    if m.tendon_armature[t] > 0:
        why.append("spatial tendon: armature (not supported on spatial tendons)")
    return why


def _f32(x):
    return f"{float(np.float32(x))!r}f"


def tables(m, ts, kten, acts):
    """The spatial tables' text: ts the used tendons (forest_engine.tendon.used, the table order), kten = KTEN, acts:
    the tendon actuators [(actuator, tendon)]."""
    tix = {t: k for k, t in enumerate(ts)}
    sites, geoms, segs, sg0, sgn = [], [], [], [], []
    for t in ts:
        sg0.append(len(segs))
        if not is_spatial(m, t):
            sgn.append(0)
            continue
        p = path(m, t)
        sgn.append(len(p))
        for s in p:
            for key in ("s0", "s1"):
                if s[key] not in sites:
                    sites.append(s[key])
            if s["side"] >= 0 and s["side"] not in sites:
                sites.append(s["side"])
            if s["g"] >= 0 and s["g"] not in geoms:
                geoms.append(s["g"])
            segs.append(s)
    n1 = lambda n: max(n, 1)
    I = lambda xs: ", ".join(str(int(x)) for x in xs) or "0"
    F = lambda xs: ", ".join(_f32(x) for x in xs) or "0.f"
    tab = lambda typ, nm, n, vals, note="": (f"__device__ const {typ} {nm}[{n}] = {{{vals}}};"
                                            + (f"  // {note}" if note else "") + "\n")
    tpos = []
    from nablaengine.forest_engine import tendon as TEN
    for t in ts:
        sup = [d for d, _ in TEN.support_list(m, t)]
        row = [-1] * m.nv
        for k, d in enumerate(sup):
            row[d] = k
        tpos += row
    nsp = sum(1 for t in ts if is_spatial(m, t))
    return (f"#define NTS_SPATIAL {nsp}  // spatial tendons among the NTEN (forest_engine.spatial)\n"
            f"#define NSPSEG {len(segs)}  // their path segments\n"
            f"#define TS_WORDS (NTEN + NTEN * KTEN + 2 * NTACT)  // per-world storage: L, J (KTEN per tendon), actuator "
            "force and flags\n"
            + tab("int", "c_tsg0", len(ts), I(sg0), "tendon: its first segment")
            + tab("int", "c_tsgn", len(ts), I(sgn), "its number of segments (0: a fixed tendon)")
            + tab("signed char", "c_tpos", len(ts) * m.nv, I(tpos), "dof -> its support entry (-1: none)")
            + tab("int", "c_sgk", n1(len(segs)), I(s["kind"] for s in segs), "segment: 0 site-site, 1 sphere, 2 cylinder")
            + tab("int", "c_sgs0", n1(len(segs)), I(sites.index(s["s0"]) for s in segs), "its first site")
            + tab("int", "c_sgs1", n1(len(segs)), I(sites.index(s["s1"]) for s in segs), "its last site")
            + tab("int", "c_sgg", n1(len(segs)), I(geoms.index(s["g"]) if s["g"] >= 0 else -1 for s in segs),
                  "its wrap geom (-1: none)")
            + tab("int", "c_sgsd", n1(len(segs)), I(sites.index(s["side"]) if s["side"] >= 0 else -1 for s in segs),
                  "its side site (-1: none)")
            + tab("float", "c_sgsc", n1(len(segs)), F(s["scale"] for s in segs), "its scale (1 / the pulley divisor)")
            + tab("int", "c_stb", n1(len(sites)), I(m.site_bodyid[s] for s in sites), "site: body")
            + tab("float", "c_stp", 3 * n1(len(sites)), F(x for s in sites for x in m.site_pos[s]),
                  "its position in the body frame")
            + tab("int", "c_wgb", n1(len(geoms)), I(m.geom_bodyid[g] for g in geoms), "wrap geom: body")
            + tab("float", "c_wgp", 3 * n1(len(geoms)), F(x for g in geoms for x in m.geom_pos[g]), "its position")
            + tab("float", "c_wgq", 4 * n1(len(geoms)), F(x for g in geoms for x in m.geom_quat[g]), "its quaternion")
            + tab("float", "c_wgr", n1(len(geoms)), F(m.geom_size[g, 0] for g in geoms), "its radius")
            + tab("int", "c_tatix", n1(len(acts)), I(tix[t] for _, t in acts), "tendon actuator: its tendon (table)"))


DEVICE = r"""
// ---------------------------------------------------------------- spatial tendons (forest_engine.spatial)
#if DK_PRE2W
#error "spatial tendons: not with DK_PRE2W (the latency mode)"
#endif
#define SP_MINVAL 1e-15f
// the per-world storage (sp_compute): L[NTEN], J[NTEN][KTEN] (over the tendon's support), the tendon actuators'
// force[NTACT] and flags[NTACT] (bit 0 force at its forcerange, bit 1 ctrl at its ctrlrange); dk_tsp points at it
// (lane 0 sets it at the start of substep_pre, substep_post and adj_substep)
__shared__ float* dk_tsp;
#define TS_L(t) (dk_tsp[(t)])
#define TS_J(t) (dk_tsp + NTEN + KTEN * (t))
#define TS_F(k) (dk_tsp[NTEN + NTEN * KTEN + (k)])
#define TS_FL(k) (__float_as_int(dk_tsp[NTEN + NTEN * KTEN + NTACT + (k)]))
__device__ __forceinline__ float ts_cd(int t, int i) {  // J of tendon t at dof i (0 off its support)
  const int k = c_tpos[NV * t + i];
  return k >= 0 ? TS_J(t)[k] : 0.f;
}
// the storage of a state: right after the connect rows' J (forest_engine.equality ec_J)
template <class S>
__device__ __forceinline__ float* ts_base(const S& s) {
  return (float*)(s.con_J + MAXCON) + (NEQC > 0 ? NEQC : 1) * 3 * KEQC;
}

// SpR: a float whose operations are never contracted (explicit roundings: the same bits in every kernel and unit)
struct SpR {
  float v;
  __device__ __forceinline__ SpR(float x = 0.f) : v(x) {}
};
__device__ __forceinline__ SpR operator+(SpR a, SpR b) { return SpR(__fadd_rn(a.v, b.v)); }
__device__ __forceinline__ SpR operator-(SpR a, SpR b) { return SpR(__fsub_rn(a.v, b.v)); }
__device__ __forceinline__ SpR operator*(SpR a, SpR b) { return SpR(__fmul_rn(a.v, b.v)); }
__device__ __forceinline__ SpR operator/(SpR a, SpR b) { return SpR(__fdiv_rn(a.v, b.v)); }
__device__ __forceinline__ SpR operator-(SpR a) { return SpR(-a.v); }
__device__ __forceinline__ float spv(SpR a) { return a.v; }
__device__ __forceinline__ SpR sp_sqrt(SpR a) { return SpR(__fsqrt_rn(a.v)); }
__device__ __forceinline__ SpR sp_atan2(SpR y, SpR x) { return SpR(atan2f(y.v, x.v)); }
__device__ __forceinline__ SpR sp_acos(SpR a) { return SpR(acosf(a.v)); }
__device__ __forceinline__ SpR sp_asin(SpR a) { return SpR(asinf(a.v)); }
__device__ __forceinline__ SpR sp_sin(SpR a) { return SpR(sinf(a.v)); }
__device__ __forceinline__ SpR sp_cos(SpR a) { return SpR(cosf(a.v)); }
__device__ __forceinline__ SpR sp_abs(SpR a) { return SpR(fabsf(a.v)); }
// SpD: the same value arithmetic with six partials (forward mode)
struct SpD {
  float v, d[6];
  __device__ __forceinline__ SpD(float x = 0.f) : v(x) {
    for (int k = 0; k < 6; k++) d[k] = 0.f;
  }
};
__device__ __forceinline__ SpD operator+(SpD a, SpD b) {
  SpD r(__fadd_rn(a.v, b.v));
  for (int k = 0; k < 6; k++) r.d[k] = a.d[k] + b.d[k];
  return r;
}
__device__ __forceinline__ SpD operator-(SpD a, SpD b) {
  SpD r(__fsub_rn(a.v, b.v));
  for (int k = 0; k < 6; k++) r.d[k] = a.d[k] - b.d[k];
  return r;
}
__device__ __forceinline__ SpD operator*(SpD a, SpD b) {
  SpD r(__fmul_rn(a.v, b.v));
  for (int k = 0; k < 6; k++) r.d[k] = a.d[k] * b.v + a.v * b.d[k];
  return r;
}
__device__ __forceinline__ SpD operator/(SpD a, SpD b) {
  SpD r(__fdiv_rn(a.v, b.v));
  for (int k = 0; k < 6; k++) r.d[k] = (a.d[k] - r.v * b.d[k]) / b.v;
  return r;
}
__device__ __forceinline__ SpD operator-(SpD a) {
  SpD r(-a.v);
  for (int k = 0; k < 6; k++) r.d[k] = -a.d[k];
  return r;
}
__device__ __forceinline__ float spv(SpD a) { return a.v; }
__device__ __forceinline__ SpD sp_chain(float v, const SpD& a, float g) {  // f(a) with f(a.v) = v, f'(a.v) = g
  SpD r(v);
  for (int k = 0; k < 6; k++) r.d[k] = g * a.d[k];
  return r;
}
__device__ __forceinline__ SpD sp_sqrt(SpD a) {
  const float v = __fsqrt_rn(a.v);
  return sp_chain(v, a, v > 0.f ? 0.5f / v : 0.f);
}
__device__ __forceinline__ SpD sp_atan2(SpD y, SpD x) {
  SpD r(atan2f(y.v, x.v));
  const float n = x.v * x.v + y.v * y.v;
  if (n > 0.f)
    for (int k = 0; k < 6; k++) r.d[k] = (x.v * y.d[k] - y.v * x.d[k]) / n;
  return r;
}
__device__ __forceinline__ SpD sp_acos(SpD a) {
  const float s = 1.f - a.v * a.v;
  return sp_chain(acosf(a.v), a, s > 0.f ? -rsqrtf(s) : 0.f);
}
__device__ __forceinline__ SpD sp_asin(SpD a) {
  const float s = 1.f - a.v * a.v;
  return sp_chain(asinf(a.v), a, s > 0.f ? rsqrtf(s) : 0.f);
}
__device__ __forceinline__ SpD sp_sin(SpD a) { return sp_chain(sinf(a.v), a, cosf(a.v)); }
__device__ __forceinline__ SpD sp_cos(SpD a) { return sp_chain(cosf(a.v), a, -sinf(a.v)); }
__device__ __forceinline__ SpD sp_abs(SpD a) { return a.v < 0.f ? -a : a; }

template <class T> __device__ __forceinline__ T sp_sdiv(T x, T y) { return x / (spv(y) != 0.f ? y : T(SP_MINVAL)); }
template <class T> __device__ __forceinline__ T sp_nrm2(T x, T y) { return sp_sqrt(x * x + y * y); }
template <class T> __device__ __forceinline__ T sp_nrm3(const T* a) { return sp_sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }

// 2D segment intersection (util_misc.is_intersect), on the values
__device__ __forceinline__ bool sp_intersect(SpR p1x, SpR p1y, SpR p2x, SpR p2y, SpR p3x, SpR p3y, SpR p4x, SpR p4y) {
  const SpR ux = p2x - p1x, uy = p2y - p1y, vx = p4x - p3x, vy = p4y - p3y;
  const SpR det = vy * ux - vx * uy;
  if (fabsf(det.v) < SP_MINVAL) return false;
  const SpR s = SpR(det.v > 0.f ? 1.f : -1.f);
  const SpR wx = p1x - p3x, wy = p1y - p3y, dx = p2x - p4x, dy = p2y - p4y;
  const SpR a0 = (vx * wy - vy * wx) * s, a1 = (vx * dy - vy * dx) * s;
  const SpR b0 = (ux * wy - uy * wx) * s, b1 = (ux * dy - uy * dx) * s;
  return a0.v >= 0.f && a1.v <= 0.f && b0.v >= 0.f && b1.v <= 0.f;
}

// util_misc.length_circle
template <class T>
__device__ __forceinline__ T sp_length_circle(const T* p0, const T* p1, int ind, float radius) {
  const T cr = p0[1] * p1[0] - p0[0] * p1[1];
  T ang = sp_atan2(sp_abs(cr), p0[0] * p1[0] + p0[1] * p1[1]);
  if ((spv(cr) > 0.f && ind != 0) || (spv(cr) < 0.f && ind == 0)) ang = T(6.2831853071795862f) - ang;
  return T(radius) * ang;
}

// util_misc.wrap_circle: e = (end0, end1), side (sd, valid: vs) projected and scaled; the wrap length (< 0: none) and
// the two points
template <class T>
__device__ T sp_wrap_circle(const T* e, bool vs, const T* sd, float radius, T* pnt0, T* pnt1) {
  const T sqlen0 = e[0] * e[0] + e[1] * e[1], sqlen1 = e[2] * e[2] + e[3] * e[3];
  const T sqrad = T(radius) * T(radius);
  if (spv(sqlen0) < spv(sqrad) || spv(sqlen1) < spv(sqrad) || radius < SP_MINVAL) return T(-1.f);
  const T dx = e[2] - e[0], dy = e[3] - e[1];
  const T dd = dx * dx + dy * dy;
  if (spv(dd) < SP_MINVAL) return T(-1.f);
  T a = -(dx * e[0] + dy * e[1]) / dd;
  if (spv(a) < 0.f) a = T(0.f);
  if (spv(a) > 1.f) a = T(1.f);
  const T tx = a * dx + e[0], ty = a * dy + e[1];
  if (spv(tx * tx + ty * ty) > spv(sqrad) && (!vs || spv(sd[0] * tx + sd[1] * ty) >= 0.f)) return T(-1.f);
  const T s0 = sp_sqrt(sqlen0 - sqrad), s1 = sp_sqrt(sqlen1 - sqrad), R = T(radius);
  T sol00[2] = {sp_sdiv(e[0] * sqrad + R * e[1] * s0, sqlen0), sp_sdiv(e[1] * sqrad - R * e[0] * s0, sqlen0)};
  T sol01[2] = {sp_sdiv(e[2] * sqrad - R * e[3] * s1, sqlen1), sp_sdiv(e[3] * sqrad + R * e[2] * s1, sqlen1)};
  T sol10[2] = {sp_sdiv(e[0] * sqrad - R * e[1] * s0, sqlen0), sp_sdiv(e[1] * sqrad + R * e[0] * s0, sqlen0)};
  T sol11[2] = {sp_sdiv(e[2] * sqrad + R * e[3] * s1, sqlen1), sp_sdiv(e[3] * sqrad - R * e[2] * s1, sqlen1)};
  SpR good0, good1;
  if (vs) {
    SpR ax = SpR(spv(sol00[0])) + SpR(spv(sol01[0])), ay = SpR(spv(sol00[1])) + SpR(spv(sol01[1]));
    SpR n = sp_sqrt(ax * ax + ay * ay);
    if (n.v != 0.f) { ax = ax / n; ay = ay / n; }
    good0 = ax * SpR(spv(sd[0])) + ay * SpR(spv(sd[1]));
    ax = SpR(spv(sol10[0])) + SpR(spv(sol11[0])); ay = SpR(spv(sol10[1])) + SpR(spv(sol11[1]));
    n = sp_sqrt(ax * ax + ay * ay);
    if (n.v != 0.f) { ax = ax / n; ay = ay / n; }
    good1 = ax * SpR(spv(sd[0])) + ay * SpR(spv(sd[1]));
  } else {
    SpR ax = SpR(spv(sol00[0])) - SpR(spv(sol01[0])), ay = SpR(spv(sol00[1])) - SpR(spv(sol01[1]));
    good0 = -(ax * ax + ay * ay);
    ax = SpR(spv(sol10[0])) - SpR(spv(sol11[0])); ay = SpR(spv(sol10[1])) - SpR(spv(sol11[1]));
    good1 = -(ax * ax + ay * ay);
  }
  const bool i0 = sp_intersect(SpR(spv(e[0])), SpR(spv(e[1])), SpR(spv(sol00[0])), SpR(spv(sol00[1])), SpR(spv(e[2])),
                               SpR(spv(e[3])), SpR(spv(sol01[0])), SpR(spv(sol01[1])));
  const bool i1 = sp_intersect(SpR(spv(e[0])), SpR(spv(e[1])), SpR(spv(sol10[0])), SpR(spv(sol10[1])), SpR(spv(e[2])),
                               SpR(spv(e[3])), SpR(spv(sol11[0])), SpR(spv(sol11[1])));
  if (i0) good0 = SpR(-10000.f);
  if (i1) good1 = SpR(-10000.f);
  int ind;
  if (good0.v > good1.v) {
    if (i0) return T(-1.f);
    pnt0[0] = sol00[0]; pnt0[1] = sol00[1]; pnt1[0] = sol01[0]; pnt1[1] = sol01[1];
    ind = 0;
  } else {
    if (i1) return T(-1.f);
    pnt0[0] = sol10[0]; pnt0[1] = sol10[1]; pnt1[0] = sol11[0]; pnt1[1] = sol11[1];
    ind = 1;
  }
  return sp_length_circle(pnt0, pnt1, ind, radius);
}

// util_misc.wrap_inside (side site inside the geom): 0 and the single point (pnt0 = pnt1), or -1 (no wrap)
template <class T>
__device__ T sp_wrap_inside(const T* e, float radius, T* pnt0, T* pnt1) {
  const T len0 = sp_nrm2(e[0], e[1]), len1 = sp_nrm2(e[2], e[3]);
  const T dx = e[2] - e[0], dy = e[3] - e[1];
  const T dd = dx * dx + dy * dy;
  if (spv(len0) <= radius || spv(len1) <= radius || radius < SP_MINVAL || spv(len0) < SP_MINVAL
      || spv(len1) < SP_MINVAL)
    return T(-1.f);
  if (spv(dd) > SP_MINVAL) {
    const T a = -(dx * e[0] + dy * e[1]) / dd;
    if (spv(a) > 0.f && spv(a) < 1.f) {
      const T tx = e[0] + a * dx, ty = e[1] + a * dy;
      if (spv(sp_nrm2(tx, ty)) <= radius) return T(-1.f);
    }
  }
  T px = T(0.5f) * (e[0] + e[2]), py = T(0.5f) * (e[1] + e[3]);
  {
    const T n = sp_nrm2(px, py);
    if (spv(n) != 0.f) { px = px / n; py = py / n; }
  }
  px = px * T(radius); py = py * T(radius);
  pnt0[0] = pnt1[0] = px; pnt0[1] = pnt1[1] = py;
  const T R = T(radius);
  const T A = sp_sdiv(R, len0), B = sp_sdiv(R, len1);
  const T sqA = A * A, sqB = B * B;
  const T cosG = sp_sdiv(len0 * len0 + len1 * len1 - dd, T(2.f) * len0 * len1);
  if (spv(cosG) < -1.f + SP_MINVAL) return T(-1.f);
  if (spv(cosG) > 1.f - SP_MINVAL) return T(0.f);
  const T G = sp_acos(cosG);
  T phi = T(1.5707963267948966f), z = T(1.f);
  T asA = sp_asin(A), asB = sp_asin(B);
  T f = asA + asB - T(3.1415926535897931f) + G;
  if (spv(f) > 0.f) return T(0.f);
  int it = 0;
  while (it < 20 && fabsf(spv(f)) > 1e-6f) {
    const T sqz = z * z;
    T r1 = sp_sqrt(T(1.f) - sqz * sqA), r2 = sp_sqrt(T(1.f) - sqz * sqB);
    if (spv(r1) < SP_MINVAL) r1 = T(SP_MINVAL);
    if (spv(r2) < SP_MINVAL) r2 = T(SP_MINVAL);
    const T df = sp_cos(phi) * (A / r1 + B / r2) - T(2.f);
    if (spv(df) > -SP_MINVAL) return T(0.f);
    const T phi1 = phi - sp_sdiv(f, df);
    if (spv(phi1) > spv(phi)) return T(0.f);
    phi = phi1;
    z = sp_sin(phi);
    asA = sp_asin(A * z);
    asB = sp_asin(B * z);
    f = asA + asB - T(2.f) * phi + G;
    if (spv(f) > 1e-6f) return T(0.f);
    it++;
  }
  if (it >= 20) return T(0.f);
  T vx, vy, ang;
  if (spv(e[0] * e[3] - e[1] * e[2]) > 0.f) { vx = e[0]; vy = e[1]; ang = phi - asA; }
  else { vx = e[2]; vy = e[3]; ang = phi - asB; }
  {
    const T n = sp_nrm2(vx, vy);
    if (spv(n) != 0.f) { vx = vx / n; vy = vy / n; }
  }
  const T ca = sp_cos(ang), sa = sp_sin(ang);
  px = R * (ca * vx - sa * vy);
  py = R * (sa * vx + ca * vy);
  pnt0[0] = pnt1[0] = px; pnt0[1] = pnt1[1] = py;
  return T(0.f);
}

// util_misc.wrap in the geom's frame: p0, p1 the sites, side the side site (null: none); the wrap length (< 0: no wrap)
// and the tangent points r0, r1 (geom frame); cyl: a cylinder (else a sphere)
template <class T>
__device__ T sp_wrap_local(const T* p0, const T* p1, const SpR* side, float radius, bool cyl, T* r0, T* r1) {
  if (spv(sp_nrm3(p0)) < SP_MINVAL || spv(sp_nrm3(p1)) < SP_MINVAL) return T(-1.f);
  T a0[3], a1[3];
  if (!cyl) {
    T n0 = sp_nrm3(p0);
    for (int k = 0; k < 3; k++) a0[k] = spv(n0) != 0.f ? p0[k] / n0 : p0[k];
    T nm[3] = {p0[1] * p1[2] - p0[2] * p1[1], p0[2] * p1[0] - p0[0] * p1[2], p0[0] * p1[1] - p0[1] * p1[0]};
    const T nrm = sp_nrm3(nm);
    if (spv(nrm) != 0.f)
      for (int k = 0; k < 3; k++) nm[k] = nm[k] / nrm;
    if (spv(nrm) < SP_MINVAL) {
      const float b0 = fabsf(spv(a0[0])), b1 = fabsf(spv(a0[1])), b2 = fabsf(spv(a0[2]));
      int i = 0;
      if (b1 > b0 && b1 > b2) i = 1;
      if (b2 > b0 && b2 > b1) i = 2;
      T ax[3] = {T(1.f), T(1.f), T(1.f)};
      ax[i] = T(0.f);
      nm[0] = a0[1] * ax[2] - a0[2] * ax[1]; nm[1] = a0[2] * ax[0] - a0[0] * ax[2]; nm[2] = a0[0] * ax[1] - a0[1] * ax[0];
      const T n2 = sp_nrm3(nm);
      if (spv(n2) != 0.f)
        for (int k = 0; k < 3; k++) nm[k] = nm[k] / n2;
    }
    a1[0] = nm[1] * a0[2] - nm[2] * a0[1]; a1[1] = nm[2] * a0[0] - nm[0] * a0[2]; a1[2] = nm[0] * a0[1] - nm[1] * a0[0];
    const T n1 = sp_nrm3(a1);
    if (spv(n1) != 0.f)
      for (int k = 0; k < 3; k++) a1[k] = a1[k] / n1;
  } else {
    a0[0] = T(1.f); a0[1] = T(0.f); a0[2] = T(0.f);
    a1[0] = T(0.f); a1[1] = T(1.f); a1[2] = T(0.f);
  }
  T e[4] = {p0[0] * a0[0] + p0[1] * a0[1] + p0[2] * a0[2], p0[0] * a1[0] + p0[1] * a1[1] + p0[2] * a1[2],
            p1[0] * a0[0] + p1[1] * a0[1] + p1[2] * a0[2], p1[0] * a1[0] + p1[1] * a1[1] + p1[2] * a1[2]};
  T pn0[2], pn1[2], wlen;
  if (side) {  // the side site: projected and scaled (decisions only: its values)
    T sd[2] = {T(spv(side[0])) * a0[0] + T(spv(side[1])) * a0[1] + T(spv(side[2])) * a0[2],
               T(spv(side[0])) * a1[0] + T(spv(side[1])) * a1[1] + T(spv(side[2])) * a1[2]};
    const T n = sp_nrm2(sd[0], sd[1]);
    if (spv(n) != 0.f) { sd[0] = sd[0] / n; sd[1] = sd[1] / n; }
    sd[0] = sd[0] * T(radius); sd[1] = sd[1] * T(radius);
    if (spv(sp_nrm3(side)) < radius) wlen = sp_wrap_inside(e, radius, pn0, pn1);
    else wlen = sp_wrap_circle(e, true, sd, radius, pn0, pn1);
  } else {
    const T sd[2] = {T(0.f), T(0.f)};
    wlen = sp_wrap_circle(e, false, sd, radius, pn0, pn1);
  }
  if (spv(wlen) < 0.f) return T(-1.f);
  for (int k = 0; k < 3; k++) {
    r0[k] = a0[k] * pn0[0] + a1[k] * pn0[1];
    r1[k] = a0[k] * pn1[0] + a1[k] * pn1[1];
  }
  if (cyl) {
    const T d0x = p0[0] - r0[0], d0y = p0[1] - r0[1], d1x = p1[0] - r1[0], d1y = p1[1] - r1[1];
    const T L0 = sp_sqrt(d0x * d0x + d0y * d0y), L1 = sp_sqrt(d1x * d1x + d1y * d1y);
    const T tot = L0 + wlen + L1;
    r0[2] = p0[2] + (p1[2] - p0[2]) * sp_sdiv(L0, tot);
    r1[2] = p0[2] + (p1[2] - p0[2]) * sp_sdiv(L0 + wlen, tot);
    const T hgt = sp_abs(r1[2] - r0[2]);
    wlen = sp_sqrt(wlen * wlen + hgt * hgt);
  }
  return wlen;
}

// world position of site s
__device__ __forceinline__ void sp_site(const float (*Q)[4], const float (*X)[3], int s, float* x) {
  const int b = c_stb[s];
  float o[3];
  qrot_rn(Q[b], c_stp + 3 * s, o);
  for (int k = 0; k < 3; k++) x[k] = RADD(X[b][k], o[k]);
}
// world pose of wrap geom g (centre xg, quaternion qg)
__device__ __forceinline__ void sp_geom(const float (*Q)[4], const float (*X)[3], int g, float* xg, float* qg) {
  const int b = c_wgb[g];
  float o[3];
  qrot_rn(Q[b], c_wgp + 3 * g, o);
  for (int k = 0; k < 3; k++) xg[k] = RADD(X[b][k], o[k]);
  qmul_rn(Q[b], c_wgq + 4 * g, qg);
}
// x (world) in the frame (xg, qg)
__device__ __forceinline__ void sp_tolocal(const float* xg, const float* qg, const float* x, float* l) {
  const float qc[4] = {qg[0], -qg[1], -qg[2], -qg[3]};
  const float d[3] = {RSUB(x[0], xg[0]), RSUB(x[1], xg[1]), RSUB(x[2], xg[2])};
  qrot_rn(qc, d, l);
}
// segment sg's wrap at sites x0, x1 (world): the wrap length (< 0: none, straight), the tangent points t0, t1 (world),
// and the sites in the geom frame p0l, p1l and the geom's pose (for the adjoint)
__device__ float sp_wrap(const float (*Q)[4], const float (*X)[3], int sg, const float* x0, const float* x1, float* t0,
                         float* t1, float* p0l, float* p1l, float* xg, float* qg) {
  const int g = c_sgg[sg];
  sp_geom(Q, X, g, xg, qg);
  sp_tolocal(xg, qg, x0, p0l);
  sp_tolocal(xg, qg, x1, p1l);
  SpR side[3], P0[3] = {p0l[0], p0l[1], p0l[2]}, P1[3] = {p1l[0], p1l[1], p1l[2]}, R0[3], R1[3];
  const int sd = c_sgsd[sg];
  if (sd >= 0) {
    float xs[3], sl[3];
    sp_site(Q, X, sd, xs);
    sp_tolocal(xg, qg, xs, sl);
    for (int k = 0; k < 3; k++) side[k] = SpR(sl[k]);
  }
  const SpR wl = sp_wrap_local(P0, P1, sd >= 0 ? side : (const SpR*)nullptr, c_wgr[g], c_sgk[sg] == 2, R0, R1);
  if (wl.v < 0.f) return -1.f;
  const float r0[3] = {R0[0].v, R0[1].v, R0[2].v}, r1[3] = {R1[0].v, R1[1].v, R1[2].v};
  float o[3];
  qrot_rn(qg, r0, o);
  for (int k = 0; k < 3; k++) t0[k] = RADD(xg[k], o[k]);
  qrot_rn(qg, r1, o);
  for (int k = 0; k < 3; k++) t1[k] = RADD(xg[k], o[k]);
  return wl.v;
}
__device__ __forceinline__ SpR sp_dist(const float* a, const float* b) {
  const SpR dx = SpR(b[0]) - SpR(a[0]), dy = SpR(b[1]) - SpR(a[1]), dz = SpR(b[2]) - SpR(a[2]);
  return sp_sqrt(dx * dx + dy * dy + dz * dz);
}
// the motion column of dof i of body bb at world point p (hinge z x (p - X), slide z, free: translation e_j, rotation
// (R e_j) x (p - X)); j: the free joint's dof index (0..5). Explicit roundings (as all of J: the fused and the split
// forward compute the same bits)
__device__ __forceinline__ void sp_col(const float (*Q)[4], const float (*X)[3], int bb, int j, const float* p,
                                       float* col, float* z) {
  const int jt = JT_OF(bb);
  const float r[3] = {RSUB(p[0], X[bb][0]), RSUB(p[1], X[bb][1]), RSUB(p[2], X[bb][2])};
  if (jt == JFREE) {
    if (j < 3) { col[0] = col[1] = col[2] = 0.f; col[j] = 1.f; z[0] = z[1] = z[2] = 0.f; return; }
    float e[3] = {0.f, 0.f, 0.f};
    e[j - 3] = 1.f;
    qrot_rn(Q[bb], e, z);
  } else {
    qrot_rn(Q[bb], c_axis + 3 * bb, z);
    if (jt == JSLIDE) { col[0] = z[0]; col[1] = z[1]; col[2] = z[2]; return; }
  }
  cross_rn(z, r, col);
}
// J[t] += w u . (motion column of each support dof on body b's chain at p)
__device__ void sp_chain_add(const float (*Q)[4], const float (*X)[3], float* J, int t, int b, const float* p,
                             const float* u, float w) {
  for (int bb = b; bb > 0; bb = c_parent[bb]) {
    const int i0 = c_dadr[bb], nd = JT_OF(bb) == JFREE ? 6 : 1;
    for (int j = 0; j < nd; j++) {
      const int k = c_tpos[NV * t + i0 + j];
      if (k < 0) continue;
      float col[3], z[3];
      sp_col(Q, X, bb, nd == 6 ? j : 0, p, col, z);
      J[k] = RFMA(w, dot3_rn(u, col), J[k]);
    }
  }
}
// a straight piece pa (body ba) -> pb (body bb) with scale sc: J += sc u'(J_pb - J_pa) when the bodies differ
__device__ __forceinline__ void sp_piece(const float (*Q)[4], const float (*X)[3], float* J, int t, const float* pa,
                                         int ba, const float* pb, int bb, float sc) {
  if (ba == bb) return;
  float u[3] = {RSUB(pb[0], pa[0]), RSUB(pb[1], pa[1]), RSUB(pb[2], pa[2])};
  const float n = __fsqrt_rn(dot3_rn(u, u));
  if (n < SP_MINVAL) { u[0] = 1.f; u[1] = u[2] = 0.f; }
  else for (int k = 0; k < 3; k++) u[k] = __fdiv_rn(u[k], n);
  sp_chain_add(Q, X, J, t, bb, pb, u, sc);
  sp_chain_add(Q, X, J, t, ba, pa, u, -sc);
}
// L and J of every tendon (lane per tendon) into T (TS_WORDS: L, J), from the body poses Q, X (bitwise the forward's in
// the adjoint) and qpos (fixed tendons: L = sum c q, the fixed text's ten_len)
__device__ void sp_compute(const float (*Q)[4], const float (*X)[3], const float* qpos, float* T, int lane) {
  for (int t = lane; t < NTEN; t += 32) {
    float* J = T + NTEN + KTEN * t;
    if (c_tsgn[t] == 0) {
      const unsigned char* d = c_tdof + KTEN * t;
      const float* c = c_tco + KTEN * t;
      float L = RMUL(c[0], qpos[c_dqadr[d[0]]]);
      for (int k = 1; k < c_tnd[t]; k++) L = RFMA(c[k], qpos[c_dqadr[d[k]]], L);
      T[t] = L;
      for (int k = 0; k < KTEN; k++) J[k] = c[k];
      continue;
    }
    for (int k = 0; k < KTEN; k++) J[k] = 0.f;
    SpR L(0.f);
    for (int sg = c_tsg0[t]; sg < c_tsg0[t] + c_tsgn[t]; sg++) {
      const int s0 = c_sgs0[sg], s1 = c_sgs1[sg], b0 = c_stb[s0], b1 = c_stb[s1];
      const float sc = c_sgsc[sg];
      float x0[3], x1[3];
      sp_site(Q, X, s0, x0);
      sp_site(Q, X, s1, x1);
      if (c_sgk[sg] != 0) {
        float t0[3], t1[3], p0l[3], p1l[3], xg[3], qg[4];
        const float wl = sp_wrap(Q, X, sg, x0, x1, t0, t1, p0l, p1l, xg, qg);
        if (wl >= 0.f) {
          const int bg = c_wgb[c_sgg[sg]];
          const SpR tot = sp_dist(x0, t0) + SpR(wl) + sp_dist(t1, x1);
          if (tot.v != 0.f) L = L + tot * SpR(sc);
          sp_piece(Q, X, J, t, x0, b0, t0, bg, sc);
          sp_piece(Q, X, J, t, t1, bg, x1, b1, sc);
          continue;
        }
      }
      const SpR len = sp_dist(x0, x1);
      if (len.v != 0.f) L = L + len * SpR(sc);
      sp_piece(Q, X, J, t, x0, b0, x1, b1, sc);
    }
    T[t] = L.v;
  }
}
// the tendon actuators (lane per actuator, after sp_compute): f = kp u - kq L - kv (J v), u = ctrl clamped to
// ctrlrange, f clamped to the forcerange; force and flags (bit 0 force clamped, bit 1 ctrl clamped) into the storage and
// the tape (rec: null for none)
__device__ void sp_actuators(const float* qvel, const float* ctrl, float* T, float* rec, int lane) {
  for (int k = lane; k < NTACT; k += 32) {
    const int t = c_tat[k], a = c_taa[k];
    const unsigned char* d = c_tdof + KTEN * t;
    const float* J = T + NTEN + KTEN * t;
    float jv = 0.f;
    for (int q = 0; q < c_tnd[t]; q++) jv = RFMA(J[q], qvel[d[q]], jv);
    const float u = fminf(fmaxf(ctrl[a], c_tactr[2 * k]), c_tactr[2 * k + 1]);
    float f = RFMS(c_takv[k], jv, RFMS(c_takq[k], T[t], RMUL(c_takp[k], u)));
    const bool fc = f <= -c_tafrc[k] || f >= c_tafrc[k];
    f = fminf(fmaxf(f, -c_tafrc[k]), c_tafrc[k]);
    const int fl = (fc ? 1 : 0) | (u != ctrl[a] ? 2 : 0);
    T[NTEN + NTEN * KTEN + k] = f;
    T[NTEN + NTEN * KTEN + NTACT + k] = __int_as_float(fl);
    if (rec) { __stcs(rec + OFF_TACT + k, f); __stcs(rec + OFF_TACT + NTACT + k, __int_as_float(fl)); }
  }
}
// the tendon actuators' force on dof i (J f)
__device__ __forceinline__ float sp_tact_qfrc(int i) {
  float acc = 0.f;
  for (int k = 0; k < NTACT; k++) {
    const float c = ts_cd(c_tat[k], i);
    if (c != 0.f) acc = RFMA(c, TS_F(k), acc);
  }
  return acc;
}
// L += h kv J J' of the tendon actuators whose force is not clamped, on M's sparsity pattern
template <class S>
__device__ void sp_tact_implicit(S& s, float h, int lane) {
  for (int k = 0; k < NTACT; k++) {
    if ((TS_FL(k) & 1) || c_takv[k] == 0.f) continue;
    const int t = c_tat[k], nd = c_tnd[t];
    const unsigned char* d = c_tdof + KTEN * t;
    const float* c = TS_J(t);
    if (lane < nd) {
      const float g = h * c_takv[k] * c[lane];
      const unsigned pm = c_tpat[KTEN * t + lane];
      for (int b = 0; b <= lane; b++)
        if ((pm >> b) & 1u) s.L[TRI(d[lane], d[b])] += g * c[b];
    }
    __syncwarp();
  }
}
"""

ADJ_DEVICE = r"""
#if DK_ADJ_WARPS == 2
#error "spatial tendons: not with DK_ADJ_WARPS=2 (the latency mode)"
#endif
// the limit row's weight on J (forest_engine.spatial): W += D (J lam) (a + b v) + sg D x lam on the support
__device__ __forceinline__ void sp_row_w(Adj& s, int t, float D, float cl, float x, float b, float sg) {
  if (c_tsgn[t] == 0) return;
  const unsigned char* d = c_tdof + KTEN * t;
  float* w = s.tw + KTEN * t;
  for (int q = 0; q < c_tnd[t]; q++) w[q] += D * cl * (s.a[d[q]] + b * s.v[d[q]]) + sg * D * x * s.lam[d[q]];
}
// body b's pose cotangent of a point cotangent pb at world point p (b 0: none)
__device__ __forceinline__ void sp_pt_bar(Adj& s, int b, const float* p, const float* pb) {
  if (b <= 0) return;
  float t[3], r[3] = {p[0] - s.X[b][0], p[1] - s.X[b][1], p[2] - s.X[b][2]};
  cross(r, pb, t);
  for (int k = 0; k < 3; k++) { s.fb[b][k] += pb[k]; s.tb[b][k] += t[k]; }
}
// the end p (body b) of a piece with g = coefficient x u (J . W's term g . V_W(p)): the support dofs' axes / anchors
// (pose cotangents of their bodies) and the point's own cotangent pbar += g x omega_W; also V_W(p) into vw
__device__ void sp_end_bar(Adj& s, int t, int b, const float* p, const float* g, const float* w, float* pbar,
                           float* vw) {
  for (int k = 0; k < 3; k++) vw[k] = 0.f;
  for (int bb = b; bb > 0; bb = c_parent[bb]) {
    const int i0 = c_dadr[bb], nd = JT_OF(bb) == JFREE ? 6 : 1;
    for (int j = 0; j < nd; j++) {
      const int k = c_tpos[NV * t + i0 + j];
      if (k < 0) continue;
      const float wk = w[k];
      float col[3], z[3], t3[3], u3[3];
      sp_col(s.Q, s.X, bb, nd == 6 ? j : 0, p, col, z);
      for (int c = 0; c < 3; c++) vw[c] += wk * col[c];
      const int jt = JT_OF(bb);
      if (jt == JFREE && j < 3) continue;
      if (jt == JSLIDE) {  // g . z w: z rotates with bb
        for (int c = 0; c < 3; c++) u3[c] = wk * g[c];
        cross(z, u3, t3);
        for (int c = 0; c < 3; c++) s.tb[bb][c] += t3[c];
        continue;
      }
      // g . (z x (p - X)) w: X bar = -w (g x z), z bar = w (p - X) x g, p bar = w (g x z)
      float gz[3], r[3] = {p[0] - s.X[bb][0], p[1] - s.X[bb][1], p[2] - s.X[bb][2]};
      cross(g, z, gz);
      cross(r, g, u3);
      for (int c = 0; c < 3; c++) u3[c] *= wk;
      cross(z, u3, t3);
      for (int c = 0; c < 3; c++) { s.fb[bb][c] -= wk * gz[c]; s.tb[bb][c] += t3[c]; pbar[c] += wk * gz[c]; }
    }
  }
}
// P_q += d/dq [J_t(q) . W] of spatial tendon t (W = s.tw's row on the support), as pose cotangents (s.fb, s.tb)
__device__ void sp_hvp(Adj& s, int t) {
  const float* w = s.tw + KTEN * t;
  for (int sg = c_tsg0[t]; sg < c_tsg0[t] + c_tsgn[t]; sg++) {
    const int s0 = c_sgs0[sg], s1 = c_sgs1[sg], b0 = c_stb[s0], b1 = c_stb[s1];
    const float sc = c_sgsc[sg];
    float x0[3], x1[3];
    sp_site(s.Q, s.X, s0, x0);
    sp_site(s.Q, s.X, s1, x1);
    // the pieces: (pa, ba, tangent index of pa or -1, pb, bb, tangent index of pb or -1)
    float P[2][2][3];
    int Bd[2][2], Tg[2][2], np = 1;
    float t0[3], t1[3], p0l[3], p1l[3], xg[3], qg[4];
    int bg = 0;
    float wl = -1.f;
    if (c_sgk[sg] != 0) wl = sp_wrap(s.Q, s.X, sg, x0, x1, t0, t1, p0l, p1l, xg, qg);
    if (wl >= 0.f) {
      bg = c_wgb[c_sgg[sg]];
      np = 2;
      for (int k = 0; k < 3; k++) { P[0][0][k] = x0[k]; P[0][1][k] = t0[k]; P[1][0][k] = t1[k]; P[1][1][k] = x1[k]; }
      Bd[0][0] = b0; Bd[0][1] = bg; Bd[1][0] = bg; Bd[1][1] = b1;
      Tg[0][0] = -1; Tg[0][1] = 0; Tg[1][0] = 1; Tg[1][1] = -1;
    } else {
      for (int k = 0; k < 3; k++) { P[0][0][k] = x0[k]; P[0][1][k] = x1[k]; }
      Bd[0][0] = b0; Bd[0][1] = b1; Tg[0][0] = Tg[0][1] = -1;
    }
    float tbar[2][3] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};  // the tangent points' world cotangents (sliding part)
    for (int pc = 0; pc < np; pc++) {
      const float *pa = P[pc][0], *pb = P[pc][1];
      const int ba = Bd[pc][0], bb = Bd[pc][1];
      if (ba == bb) continue;
      float u[3] = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
      const float n = sqrtf(dot3(u, u));
      const bool fix = n < SP_MINVAL;
      if (fix) { u[0] = 1.f; u[1] = u[2] = 0.f; }
      else for (int k = 0; k < 3; k++) u[k] /= n;
      float ga[3], gb[3], pba[3] = {0.f, 0.f, 0.f}, pbb[3] = {0.f, 0.f, 0.f}, va[3], vb[3];
      for (int k = 0; k < 3; k++) { gb[k] = sc * u[k]; ga[k] = -sc * u[k]; }
      sp_end_bar(s, t, bb, pb, gb, w, pbb, vb);
      sp_end_bar(s, t, ba, pa, ga, w, pba, va);
      if (!fix) {  // u's cotangent sc (V_W(pb) - V_W(pa)) onto the ends
        float ub[3] = {sc * (vb[0] - va[0]), sc * (vb[1] - va[1]), sc * (vb[2] - va[2])};
        const float uu = dot3(u, ub);
        for (int k = 0; k < 3; k++) {
          const float db = (ub[k] - u[k] * uu) / n;
          pbb[k] += db; pba[k] -= db;
        }
      }
      sp_pt_bar(s, bb, pb, pbb);
      sp_pt_bar(s, ba, pa, pba);
      if (Tg[pc][1] >= 0) for (int k = 0; k < 3; k++) tbar[Tg[pc][1]][k] += pbb[k];
      if (Tg[pc][0] >= 0) for (int k = 0; k < 3; k++) tbar[Tg[pc][0]][k] += pba[k];
    }
    if (np == 2) {  // the tangent points' motion over the geom: through the sites' local coordinates p0l, p1l
      float tl[2][3];
      const float qc[4] = {qg[0], -qg[1], -qg[2], -qg[3]};
      qrot(qc, tbar[0], tl[0]);
      qrot(qc, tbar[1], tl[1]);
      SpD P0[3], P1[3], R0[3], R1[3];
      for (int k = 0; k < 3; k++) {
        P0[k] = SpD(p0l[k]); P0[k].d[k] = 1.f;
        P1[k] = SpD(p1l[k]); P1[k].d[3 + k] = 1.f;
      }
      SpR side[3];
      const int sd = c_sgsd[sg];
      if (sd >= 0) {
        float xs[3], sl[3];
        sp_site(s.Q, s.X, sd, xs);
        sp_tolocal(xg, qg, xs, sl);
        for (int k = 0; k < 3; k++) side[k] = SpR(sl[k]);
      }
      const SpD wd = sp_wrap_local(P0, P1, sd >= 0 ? side : (const SpR*)nullptr, c_wgr[c_sgg[sg]], c_sgk[sg] == 2, R0,
                                   R1);
      if (wd.v >= 0.f) {
        float l0[3], l1[3], g0[3], g1[3];
        for (int c = 0; c < 3; c++) {
          l0[c] = l1[c] = 0.f;
          for (int k = 0; k < 3; k++) {
            l0[c] += tl[0][k] * R0[k].d[c] + tl[1][k] * R1[k].d[c];
            l1[c] += tl[0][k] * R0[k].d[3 + c] + tl[1][k] * R1[k].d[3 + c];
          }
        }
        qrot(qg, l0, g0);  // the sites' world cotangents; the geom's body pays the opposite (p_l = R'(x - x_g))
        qrot(qg, l1, g1);
        float m0[3] = {-g0[0], -g0[1], -g0[2]}, m1[3] = {-g1[0], -g1[1], -g1[2]};
        sp_pt_bar(s, b0, x0, g0);
        sp_pt_bar(s, bg, x0, m0);
        sp_pt_bar(s, b1, x1, g1);
        sp_pt_bar(s, bg, x1, m1);
      }
    }
  }
}
// A5 / A6 (after adj_tendon, lane 0): the weights of the passive force, the actuators and the implicit damping / kv,
// then the pose cotangents of d/dq [J . W] for every spatial tendon; s.tw zeroed for the next substep
__device__ __noinline__ void adj_spatial(Adj& s, int lane) {
  const float h = c_misc[0];
  if (lane == 0)
    for (int t = 0; t < NTEN; t++) {
      if (c_tsgn[t] == 0) continue;
      const unsigned char* d = c_tdof + KTEN * t;
      const float* J = TS_J(t);
      float* w = s.tw + KTEN * t;
      const int nd = c_tnd[t];
      float cl = 0.f;
      for (int q = 0; q < nd; q++) cl += J[q] * s.lam[d[q]];
      float cimp = 0.f;  // the implicit matrix's coefficient of J J' (damping, actuator kv)
#if NTP > 0
      for (int p = 0; p < NTP; p++) {
        if (c_tpas[p] != t) continue;
        float ke;
        const float f = ten_passive_f(s.q, s.v, t, &ke);
        for (int q = 0; q < nd; q++) w[q] += -f * s.lam[d[q]] + c_tdamp[t] * cl * s.v[d[q]];
        if (TEN_IMPLICIT) cimp += c_tdamp[t];
      }
#endif
      for (int k = 0; k < NTACT; k++) {
        if (c_tat[k] != t) continue;
        const int fl = s.tfl[k];
        const float f = s.tfrc[k], lt = (fl & 1) ? 0.f : cl;
        for (int q = 0; q < nd; q++) w[q] += -f * s.lam[d[q]] + c_takv[k] * (DK_ADJ_CLAMP_KV ? lt : cl) * s.v[d[q]];
        if (SP_IMPLICIT && !(fl & 1)) cimp += c_takv[k];
      }
      if (cimp != 0.f)  // h c (y_i (J a_adv)_pat(i) + a_adv,i (J y)_pat(i)), y = -ny, a_adv = a - at
        for (int q = 0; q < nd; q++) {
          const int i = d[q];
          float ja = 0.f, jy = 0.f;
          for (int r = 0; r < nd; r++) {
            const bool on = r <= q ? (c_tpat[KTEN * t + q] >> r) & 1u : (c_tpat[KTEN * t + r] >> q) & 1u;
            if (on) { ja += J[r] * (s.a[d[r]] - s.at[d[r]]); jy -= J[r] * s.ny[d[r]]; }
          }
          w[q] += h * cimp * (-s.ny[i] * ja + (s.a[i] - s.at[i]) * jy);
        }
      sp_hvp(s, t);
      for (int q = 0; q < KTEN; q++) w[q] = 0.f;
    }
  __syncwarp();
}
"""


TEN_LEN_FIXED = """__device__ __forceinline__ float ten_len(const float* qpos, int t) {
  const unsigned char* d = c_tdof + KTEN * t;
  const float* c = c_tco + KTEN * t;
  float L = RMUL(c[0], qpos[c_dqadr[d[0]]]);
  for (int k = 1; k < c_tnd[t]; k++) L = RFMA(c[k], qpos[c_dqadr[d[k]]], L);
  return L;
}
"""
TEN_LEN_SPATIAL = """__device__ __forceinline__ float ten_len(const float* qpos, int t) {  // (forest_engine.spatial: the storage)
  (void)qpos;
  return TS_L(t);
}
"""


def _sub_tcd(b):
    key = "c_tcd[NV * "
    out, pos = [], 0
    while True:
        k = b.find(key, pos)
        if k < 0:
            return "".join(out) + b[pos:]
        j, depth = k + len(key), 1
        while depth:
            depth += {"[": 1, "]": -1}.get(b[j], 0)
            j += 1
        inner = b[k + len(key):j - 1]
        assert inner.endswith(" + i"), inner
        out += [b[pos:k], f"ts_cd({inner[:-4]}, i)"]
        pos = j


def _rewrite(b):
    b = b.replace(TEN_LEN_FIXED, TEN_LEN_SPATIAL)
    b = b.replace("c_tco + KTEN * t", "TS_J(t)")
    b = _sub_tcd(b)
    b = b.replace(", cr = c_tacar[k];", ";")
    b = b.replace("!((s.kvoff[cr >> 5] >> (cr & 31)) & 1u)", "!(s.tfl[k] & 1)")
    b = b.replace("((s.clamp[cr >> 5] >> (cr & 31)) & 1u) ? 0.f : cl", "(s.tfl[k] & 1) ? 0.f : cl")
    b = b.replace("if (((s.clamp[cr >> 5] >> (cr & 31)) & 1u) || ((s.uclamp[cr >> 5] >> (cr & 31)) & 1u)) continue;",
                  "if (s.tfl[k] & 3) continue;")
    b = b.replace("act += tact_qfrc(s.qpos, s.qvel, ctrl, i, &aclamp, &uclamp, &clamp);",
                  "act += sp_tact_qfrc(i);")
    b = b.replace("  tact_implicit(s, h, lane);", "  sp_tact_implicit(s, h, lane);")
    return b


SP_FWD_AT = "  __syncwarp();\n  SUBPROF(17);\n"
SP_FWD = ("  __syncwarp();\n"
          "  if (lane == 0) dk_tsp = ts_base(s);  // the tendons' L, J and actuator forces (forest_engine.spatial)\n"
          "  __syncwarp();\n"
          "  sp_compute(s.Q, s.X, s.qpos, dk_tsp, lane);\n"
          "  __syncwarp();\n"
          "  sp_actuators(s.qvel, ctrl, dk_tsp, rec, lane);\n")
SP_POST_AT = ("__device__ void substep_post(S& s, int iterations, int lane, int* niter_out, float* rec) {\n"
              "  const float h = c_misc[0];\n")
SP_ADJ_MEMBERS_AT = "  int flags;  // union of the records' flags (tape.py), ADJ_ROWS_SKIPPED, ADJ_REPLAY_MISMATCH\n"
SP_ADJ_MEMBERS = ("  float ts_store[TS_WORDS];  // the tendons' L, J (forest_engine.spatial), replayed per substep\n"
                  "  float tw[NTEN * KTEN];  // the spatial tendons' weights W of d/dq [J . W] (adj_spatial)\n"
                  "  float tfrc[NTACT > 0 ? NTACT : 1];  // the record's tendon actuator forces\n"
                  "  int tfl[NTACT > 0 ? NTACT : 1];  // and flags (bit 0 force clamped, bit 1 ctrl clamped)\n")
SP_INIT_AT = "__device__ void adj_init(Adj& s, int lane) {\n"
SP_POSES_AT = "  adj_poses(s, lane);  // A3: poses and channel 1\n  APROF(APROF_POSES);\n"
SP_POSES = ("  if (lane == 0) dk_tsp = s.ts_store;  // the tendons' L, J from the replayed poses (forest_engine.spatial)\n"
            "  __syncwarp();\n"
            "  sp_compute(s.Q, s.X, s.q, s.ts_store, lane);\n"
            "  for (int k = lane; k < NTACT; k += 32) {\n"
            "    s.tfrc[k] = __ldcs(rec + OFF_TACT + k);\n"
            "    s.tfl[k] = __float_as_int(__ldcs(rec + OFF_TACT + NTACT + k));\n"
            "  }\n"
            "  __syncwarp();\n")
SP_ROW_AT = "        limit_row_vjp(sg, pos, ca, cv, cl, ri[0], ri[1], ri[2], ri[3], ri[4], ri[5], &pq, &pv);\n"
SP_ROW = "        sp_row_w(s, t, ri[4], cl, sg * ca + ri[0] * ri[2] * pos + ri[1] * (sg * cv), ri[1], sg);\n"


def transform(m, pairs, conj_at, conj_extra, dev_at):
    """forest_engine.tendon's (anchor, replacement) edits for a spatial build: the replacements rewritten (_rewrite),
    this module's device text ahead of the tendon text (dev_at: the anchor whose replacement carries it), the storage
    after the connect rows' J (conj_at: the struct Sim / Hand anchor, conj_extra its line), and the spatial edits
    (sp_compute after the mass matrix, dk_tsp in substep_post and the adjoint, the adjoint's members, replay and
    adj_spatial)."""
    out = [(SP_FWD_AT, SP_FWD + SP_FWD_AT)]
    for a, b in pairs:
        b = _rewrite(b)
        if a == conj_at:
            b = b + conj_extra
        if a == dev_at:
            mark = "// fixed tendons (forest_engine.tendon): L = sum c_i q_i"
            assert b.count(mark) == 1
            b = b.replace(mark, DEVICE + mark, 1)
        if "__device__ __noinline__ void adj_tendon(Adj& s, int lane, bool jtf)" in b:
            assert b.count(SP_ROW_AT) == 1
            b = b.replace(SP_ROW_AT, SP_ROW_AT + SP_ROW)
            b = b.replace("// A5 / A6, tendon rows (forest_engine.tendon)", ADJ_DEVICE + "// A5 / A6, tendon rows "
                          "(forest_engine.tendon)", 1)
            assert "  adj_tendon(s, lane, jtf);\n}\n" in b
            b = b.replace("  adj_tendon(s, lane, jtf);\n}\n", "  adj_tendon(s, lane, jtf);\n  adj_spatial(s, lane);\n}\n")
        out.append((a, b))
    out.append((SP_POST_AT, SP_POST_AT + "  if (lane == 0) dk_tsp = ts_base(s);  // (forest_engine.spatial)\n"
                "  __syncwarp();\n"))
    return out


def adj_extra():
    """The adjoint's own spatial edits (with transform's)."""
    return [(SP_ADJ_MEMBERS_AT, SP_ADJ_MEMBERS_AT + SP_ADJ_MEMBERS),
            (SP_INIT_AT, SP_INIT_AT + "  for (int i = lane; i < NTEN * KTEN; i += 32) s.tw[i] = 0.f;  // "
             "(forest_engine.spatial)\n"),
            (SP_POSES_AT, SP_POSES_AT + SP_POSES)]


PACK = (
    ("#define TS_WORDS (NTEN + NTEN * KTEN + 2 * NTACT", "#define TS_WORDS (NTEN + TS_JW + 2 * NTACT", 1),
    ("#define TS_J(t) (dk_tsp + NTEN + KTEN * (t))", "#define TS_J(t) (dk_tsp + NTEN + c_tjo[t])", 1),
    ("per-world storage: L, J (KTEN per tendon)", "per-world storage: L, J (packed: c_tjo)", 1),
    ("L[NTEN], J[NTEN][KTEN] (over the tendon's support)", "L[NTEN], J[TS_JW] (packed: each tendon's support)", 1),
    ("T + NTEN + KTEN * t", "T + NTEN + c_tjo[t]", None),
    ("NTEN + NTEN * KTEN", "NTEN + TS_JW", None),
    ("s.tw + KTEN * t", "s.tw + c_tjo[t]", None),
    ("  float tw[NTEN * KTEN];", "  float tw[TS_JW];", None),
    ("for (int i = lane; i < NTEN * KTEN; i += 32) s.tw[i] = 0.f;", "for (int i = lane; i < TS_JW; i += 32) s.tw[i] = 0.f;",
     None),
    ("      for (int k = 0; k < KTEN; k++) J[k] = c[k];\n", "      for (int k = 0; k < c_tnd[t]; k++) J[k] = c[k];\n", 1),
    ("    for (int k = 0; k < KTEN; k++) J[k] = 0.f;\n", "    for (int k = 0; k < c_tnd[t]; k++) J[k] = 0.f;\n", 1),
    ("      for (int q = 0; q < KTEN; q++) w[q] = 0.f;\n", "      for (int q = 0; q < c_tnd[t]; q++) w[q] = 0.f;\n", None),
)


def compact(src, m):
    """src (a spatial build's unit text, after forest_engine.tendon / spatial / muscle) in the packed layout: TS_JW (the
    J words: the used tendons' support sizes summed) and c_tjo (each tendon's first J word) ahead of TS_WORDS."""
    ts, sup, _, _ = _sizes(m)
    jo = np.concatenate([[0], np.cumsum(sup)[:-1]]).astype(int) if sup else [0]
    key = "#define TS_WORDS (NTEN + NTEN * KTEN + 2 * NTACT"
    if src.count(key) != 1:
        raise RuntimeError(f"forest spatial (packed layout): {src.count(key)} x {key!r}")
    head = (f"#define TS_JW {max(sum(sup), 1)}  // the J words of the packed layout (forest_engine.spatial.compact: each "
            "tendon's J over its own support)\n"
            f"__device__ const int c_tjo[{len(ts)}] = {{{', '.join(str(int(x)) for x in jo)}}};  // tendon: its first J "
            "word\n")
    src = src.replace(key, head + key, 1)
    for a, b, n in PACK:
        k = src.count(a)
        if (n is not None and k != n) or k == 0:
            raise RuntimeError(f"forest spatial (packed layout): {k} x {a!r}")
        src = src.replace(a, b)
    for left in ("NTEN * KTEN", "NTEN + KTEN", "tw + KTEN", "k < KTEN; k++) J[k]", "q < KTEN; q++) w[q]"):
        if left in src:
            raise RuntimeError(f"forest spatial (packed layout): {left!r} left in the text")
    return src

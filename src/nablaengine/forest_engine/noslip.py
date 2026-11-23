"""MuJoCo's noslip post-pass on the forest engine (opt.noslip_iterations > 0, opt.noslip_tolerance): forward and adjoint.

The pass follows MuJoCo 3.17's mj_solNoSlip and its use in mj_fwdConstraint (engine_solver.c, engine_forward.c;
github.com/google-deepmind/mujoco, Apache License 2.0; the algorithm is reimplemented here in float32, no code is
copied). After the Newton solve, with the Newton forces f* of every row:"""

import mujoco
import numpy as np

NS_FLAG_NONCONV = 1 << 8
NS_FLAG_OVERFLOW = 1 << 9
NS_FLAG_SKIPPED = 1 << 10
ADJ_NOSLIP = 1 << 14
NS_STATUS = dict(free=0, upper=1, lower=2, newton=3)
CON_WORDS = 13


def enabled(m):
    """True when model m asks for the noslip pass."""
    return int(m.opt.noslip_iterations) > 0


def ns_cap(m, max_contacts=None):
    """Friction-direction capacity per world: the frictionloss rows plus (2 or, with condim-4 pairs, 3) per contact,
    at most 128."""
    from nablaengine.forest_engine import step
    nc = max_contacts or step.maxcon(m)
    per = 3 if bool((m.pair_dim == 4).any()) else 2
    return int(min(int((m.dof_frictionloss > 0).sum()) + per * nc, 128))


def problems(m):
    """Reasons the forest engine's noslip pass does not take model m (empty: supported, or noslip off)."""
    if not enabled(m):
        return []
    out = []
    if int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        out.append("elliptic cone (the noslip pass is implemented for the pyramidal cone)")
    if m.neq:
        out.append("equality constraints (their rows join islands and need the adjoint's split; not verified)")
    if m.ntendon:
        out.append("tendons (tendon limit / frictionloss rows; not verified)")
    if m.nbody > 32:
        out.append("more than 31 bodies (forest_engine.nb64; not verified)")
    if m.nv > 64:
        out.append("more than 64 dofs (forest_engine.nv128; not verified)")
    if mujoco.mjtDisableBit.mjDSBL_ISLAND & int(m.opt.disableflags):
        out.append("islands disabled (the pass runs per island, MuJoCo's default)")
    return out


def check(m):
    p = problems(m)
    if p:
        raise ValueError("noslip (opt.noslip_iterations > 0) on the forest engine: not supported for this model: "
                         + "; ".join(p))


PARAM_FIELDS = ("object_size", "torso_mass", "torso_inertia", "dof_damping", "foot_solref", "foot_solimp",
                "foot_torsional")


def resolve_pairs(m, spec):
    """noslip_pairs to a sorted tuple of pair indices, or None when it selects every pair (the unrestricted text).
    spec: "objects" (the pairs with a geom on an unactuated tree rooted in a free joint and the other geom on another
    body that is not the world: robot / object and object / object contacts, not object / floor), or an iterable of
    pair indices, pair names, or (name, name) of two bodies or two geoms."""
    import mujoco
    if spec is None:
        return None
    O = mujoco.mjtObj
    if isinstance(spec, str):
        if spec != "objects":
            raise ValueError(f"noslip_pairs={spec!r}: a rule is 'objects', else a list of pairs")
        act_trees = {int(m.body_rootid[m.jnt_bodyid[m.actuator_trnid[a, 0]]]) for a in range(m.nu)
                     if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}

        def obj(b):
            r = int(m.body_rootid[b])
            return (b > 0 and r not in act_trees and m.body_jntnum[r] == 1
                    and m.jnt_type[m.body_jntadr[r]] == mujoco.mjtJoint.mjJNT_FREE)
        sel = []
        for p in range(m.npair):
            b1, b2 = int(m.geom_bodyid[m.pair_geom1[p]]), int(m.geom_bodyid[m.pair_geom2[p]])
            if b1 > 0 and b2 > 0 and (obj(b1) or obj(b2)) and m.body_rootid[b1] != m.body_rootid[b2]:
                sel.append(p)
    else:
        sel = set()
        for e in spec:
            if isinstance(e, (int, np.integer)):
                if not 0 <= int(e) < m.npair:
                    raise ValueError(f"noslip_pairs: pair index {e} outside 0 .. {m.npair - 1}")
                sel.add(int(e))
            elif isinstance(e, str):
                p = mujoco.mj_name2id(m, O.mjOBJ_PAIR, e)
                if p < 0:
                    raise ValueError(f"noslip_pairs: no pair named {e!r}")
                sel.add(p)
            else:
                a, b = e
                ga, gb = mujoco.mj_name2id(m, O.mjOBJ_GEOM, a), mujoco.mj_name2id(m, O.mjOBJ_GEOM, b)
                if ga >= 0 and gb >= 0:
                    hit = [p for p in range(m.npair) if {int(m.pair_geom1[p]), int(m.pair_geom2[p])} == {ga, gb}]
                else:
                    ba, bb = mujoco.mj_name2id(m, O.mjOBJ_BODY, a), mujoco.mj_name2id(m, O.mjOBJ_BODY, b)
                    if ba < 0 or bb < 0:
                        raise ValueError(f"noslip_pairs: {e!r} names neither two geoms nor two bodies")
                    hit = [p for p in range(m.npair) if {int(m.geom_bodyid[m.pair_geom1[p]]),
                                                         int(m.geom_bodyid[m.pair_geom2[p]])} == {ba, bb}]
                if not hit:
                    raise ValueError(f"noslip_pairs: no contact pair between {a!r} and {b!r}")
                sel.update(hit)
    sel = tuple(sorted(sel))
    if not sel:
        raise ValueError("noslip_pairs selects no contact pair")
    return None if len(sel) == m.npair else sel


def mask_text(m, pairs):
    """The pair mask of a restricted build (c_nsmask, NS_PAIR) or, for None, NS_PAIR true."""
    if pairs is None:
        return "#define NS_PAIR(pr) true  // (every pair: MuJoCo's noslip)\n"
    w = [0] * ((m.npair + 31) // 32)
    for p in pairs:
        w[p >> 5] |= 1 << (p & 31)
    return (f"__device__ const unsigned c_nsmask[{len(w)}] = {{{', '.join(f'{x}u' for x in w)}}};  // noslip_pairs\n"
            "#define NS_PAIR(pr) ((c_nsmask[(pr) >> 5] >> ((pr) & 31)) & 1u)\n")


def check_foot(m, params, foot_pairs, pairs):
    """ValueError for foot parameters (PhysicsParams foot_*) with foot pairs in the noslip rows."""
    if not params or not any(f.startswith("foot_") for f in params.fields):
        return
    fp = {int(p) for p in np.asarray(foot_pairs if foot_pairs is not None else [], int).ravel()}
    sel = set(range(m.npair)) if pairs is None else set(pairs)
    both = sorted(fp & {p for p in sel if int(m.pair_dim[p]) >= 3})
    if both:
        raise ValueError(f"noslip: the foot parameters ({[f for f in params.fields if f.startswith('foot_')]}) need the"
                         f" foot pairs outside the noslip rows (noslip_pairs); in them: {both[:8]}")


def check_params(m, params, pairs=None):
    """ValueError for PhysicsParams fields the noslip build does not take (params: a PhysicsParams or None)."""
    if not params:
        return
    bad = [f for f in params.fields if f not in PARAM_FIELDS]
    if bad:
        raise ValueError(f"noslip (opt.noslip_iterations > 0): PhysicsParams {bad} not supported (it takes "
                         f"{PARAM_FIELDS}; dof_armature, gravity, xfrc: their lam1 terms of the second pass are missing)")


PFOOT_PARTIALS = "  if (PFOOT(pair) && zone) {  // the foot pair's parameters (world_params)\n"


def params_hooks(src):
    """The world_params text with noslip: the foot pairs' parameter partials in pass A only."""
    return _rep(src, PFOOT_PARTIALS, "  if (PFOOT(pair) && zone && !s.ns_pass) {  // the foot pair's parameters "
                "(world_params; noslip: pass A only)\n")


def defines(m, max_contacts=None):
    tol = float(np.float32(m.opt.noslip_tolerance))
    return (f"#define NOSLIP 1  // forest_engine.noslip: MuJoCo's noslip post-pass\n"
            f"#define NS_ITER {int(m.opt.noslip_iterations)}  // opt.noslip_iterations\n"
            f"#define NS_TOL {tol!r}f  // opt.noslip_tolerance\n"
            f"#define NS_CAP {ns_cap(m, max_contacts)}  // friction directions per world\n"
            f"#define NS_NSR ((NS_CAP + 31) / 32)\n"
            f"#define NS_FLAG_NONCONV {NS_FLAG_NONCONV}u\n#define NS_FLAG_OVERFLOW {NS_FLAG_OVERFLOW}u\n"
            f"#define NS_FLAG_SKIPPED {NS_FLAG_SKIPPED}u\n#define NS_CON_WORDS {CON_WORDS}\n")



HAND = """  // forest_engine.noslip: the pass's scratch (k_post)
  float ns_f[NEFC];  // the rows' forces (Newton, then noslip)
  float ns_f0[NEFC];  // the Newton forces f*
  float ns_Z[NS_CAP][NV];  // L^-1 g_k of direction k
  float ns_A[NS_CAP][NS_CAP];  // A = Z' Z = G M^-1 G'
  int ns_dir[NS_CAP];  // direction k: its first row | pair t << 10 (3: a frictionloss row) | contact << 12
  int ns_dst[NS_CAP];  // its final status (noslip.NS_STATUS)
  int ns_k0[MAXCON];  // contact c's first direction
  int ns_isl[NB];  // tree (root body) -> island representative, -1: no rows
  float ns_scale[NB];  // island representative -> 1 / trace(M) over its dofs
"""

FWD_DECL = """
#if NOSLIP
template <class S> __device__ void noslip_pass(S& s, int lane, float* rec);  // (forest_engine.noslip)
#endif
"""

FWD_DEVICE = r"""
// ---------------------------------------------------------------- noslip (forest_engine.noslip)
__device__ __forceinline__ int ns_tree(int b) { return b > 0 ? c_root[b] : -1; }
__device__ __forceinline__ int ns_find(int* par, int x) { while (par[x] != x) x = par[x]; return x; }

// L x = b, then L' x = x (in place; packed lower L, shared vector b)
__device__ __noinline__ void ns_llt_solve(const float* L, float* b, int lane) {
  for (int k = 0; k < NV; k++) {
    const float bk = b[k] / L[TRI(k, k)];
    __syncwarp();
    if (lane == 0) b[k] = bk;
    for (int i = k + 1 + lane; i < NV; i += 32) b[i] -= L[TRI(i, k)] * bk;
    __syncwarp();
  }
  for (int k = NV - 1; k >= 0; k--) {
    const float bk = b[k] / L[TRI(k, k)];
    __syncwarp();
    if (lane == 0) b[k] = bk;
    for (int i = lane; i < k; i += 32) b[i] -= L[TRI(k, i)] * bk;
    __syncwarp();
  }
}

// direction k's row g (pair t of contact c: g = J_{2t} - J_{2t+1} = 2 mu_t T_t; a frictionloss row: e_dof) dotted with
// the shared vector v
template <class S>
__device__ __forceinline__ float ns_gdot(const S& s, int dir, const float* v) {
  const int r0 = dir & 1023, t = (dir >> 10) & 3, c = dir >> 12;
  if (t == 3) return v[s.efc_a[r0]];
  const float mu = t == 2 ? 0.f : s.con_mu[c];
  float mt = mu;
#if HAS_CONDIM4
  if (t == 2) mt = PMUS(s.con_pair[c]);
#endif
  float acc = 0.f;
  const int nd = s.con_nd[c];
  for (int k = 0; k < nd; k++) acc += s.con_J[c][1 + t][k] * v[s.con_dof[c][k]];
  return 2.f * mt * acc;
}

template <class S>
__device__ void noslip_pass(S& s, int lane, float* rec) {
  if constexpr (!dk_is_ptr<decltype(s.qsm)>::value) {  // the fused forward: no scratch, no pass (flagged)
    if (rec && lane == 0) __stcs(rec + OFF_NSHDR, __uint_as_float(NS_FLAG_SKIPPED));
    return;
  } else {
  // the world's struct Hand from k_post's pointer into it (post_world_on: s.qsm = g.qsm, also for Hand::big)
  Hand& g = *reinterpret_cast<Hand*>(reinterpret_cast<char*>(const_cast<float*>(s.qsm)) - offsetof(Hand, qsm));
  const unsigned FULL = 0xffffffffu, below = (1u << lane) - 1u;
  const int nefc = s.nefc, ncon = uniform_int(min(s.ncon, MAXCON), CON_BITS);
  float* F = g.ns_f;
  // the Newton forces f* (the rows' -s'(jar)), the tape's a*
  for (int r = lane; r < nefc; r += 32) {
    float ds, dd, c;
    row_eval(s, r, s.efc_jar[r], &ds, &dd, &c);
    F[r] = -ds;
    g.ns_f0[r] = -ds;
  }
  if (rec) tape_put(rec + OFF_NSQN, s.qacc, NV, OFF_NSHDR - OFF_NSQN, lane);
  __syncwarp();
  // L = chol(M); tmp = qacc of the dual (MuJoCo's AR f + b): M^-1 (qfrc_smooth + J' f*)
  rows_jtf(s, F, s.tmp, lane);
  for (int i = lane; i < NV; i += 32) s.tmp[i] += s.qsm[i];
  for (int e = lane; e < NTRI; e += 32) s.L[e] = s.M[e];
  __syncwarp();
  CHOL_M_SOLVE(s.L, s.L, s.tmp, lane, s.cbuf);
  if (rec) tape_put(rec + OFF_LM, s.L, NTRI, REC_WORDS - OFF_LM, lane);
  // directions: the frictionloss rows (the first NFLOSS rows), then per contact of condim >= 3 its row pairs
  int nd = NFLOSS;
  for (int k = lane; k < NFLOSS && k < NS_CAP; k += 32) g.ns_dir[k] = k | 3 << 10;
  for (int c0 = 0; c0 < ncon; c0 += 32) {
    const int c = c0 + lane;
    const int np = c < ncon && s.con_dim[c] >= 3 && NS_PAIR(s.con_pair[c]) ? min(s.con_dim[c] - 1, s.con_nrow[c] / 2) : 0;
    int inc = np;
    for (int o = 1; o < 32; o <<= 1) {
      const int t = __shfl_up_sync(FULL, inc, o);
      if (lane >= o) inc += t;
    }
    const int k0 = nd + inc - np;
    if (c < ncon) g.ns_k0[c] = k0;
    for (int t = 0; t < np; t++)
      if (k0 + t < NS_CAP) g.ns_dir[k0 + t] = (s.con_row[c] + 2 * t) | t << 10 | c << 12;
    nd += __shfl_sync(FULL, inc, 31);
  }
  const int ndir = min(nd, NS_CAP);
  unsigned hdr = nd > NS_CAP ? NS_FLAG_OVERFLOW : 0u;
  // islands (lane 0): trees joined by the contacts; a tree with a row is constrained; scale = 1 / trace(M) per island
  if (lane == 0) {
    int* par = g.ns_isl;
    unsigned long long used = 0ull;
    for (int b = 0; b < NB; b++) par[b] = b;
    for (int r = 0; r < nefc; r++) {
      const int kind = s.efc_kind[r], a = s.efc_a[r];
      if (kind <= 1) { used |= 1ull << ns_tree(c_dofbody[a]); continue; }
      const int pr = s.con_pair[a], t1 = ns_tree(c_gbody[PG1(pr)]), t2 = ns_tree(c_gbody[PG2(pr)]);
      if (t1 >= 0) used |= 1ull << t1;
      if (t2 >= 0) used |= 1ull << t2;
      if (t1 >= 0 && t2 >= 0) {
        const int x = ns_find(par, t1), y = ns_find(par, t2);
        if (x != y) par[max(x, y)] = min(x, y);
      }
    }
    for (int b = 0; b < NB; b++) g.ns_scale[b] = 0.f;
    for (int i = 0; i < NV; i++) {
      const int t = ns_tree(c_dofbody[i]);
      if ((used >> t) & 1ull) g.ns_scale[ns_find(par, t)] += s.M[TRI(i, i)];
    }
    for (int b = 0; b < NB; b++) {
      g.ns_scale[b] = 1.f / fmaxf(1e-15f, g.ns_scale[b]);
      par[b] = (used >> b) & 1ull ? ns_find(par, b) : -1;  // (ascending b: a parent is below its child)
    }
  }
  __syncwarp();
  // per direction (lane k & 31, register k >> 5): Z = L^-1 g in global scratch, residual, mid / bound, island
  float rr[NS_NSR], yy[NS_NSR], mm[NS_NSR];
  int isl[NS_NSR], st[NS_NSR];
#pragma unroll
  for (int t = 0; t < NS_NSR; t++) {
    const int k = lane + 32 * t;
    rr[t] = yy[t] = mm[t] = 0.f; isl[t] = -1; st[t] = 0;
    if (k >= ndir) continue;
    const int dir = g.ns_dir[k], r0 = dir & 1023, tt = (dir >> 10) & 3, c = dir >> 12;
    float* z = g.ns_Z[k];
    for (int i = 0; i < NV; i++) z[i] = 0.f;
    if (tt == 3) {
      z[s.efc_a[r0]] = 1.f;
      isl[t] = g.ns_isl[ns_tree(c_dofbody[s.efc_a[r0]])];
      rr[t] = s.tmp[s.efc_a[r0]] - s.efc_aref[r0];
      yy[t] = F[r0]; mm[t] = c_floss[s.efc_a[r0]];
    } else {
      float mt = s.con_mu[c];
#if HAS_CONDIM4
      if (tt == 2) mt = PMUS(s.con_pair[c]);
#endif
      const int ndc = s.con_nd[c];
      for (int q = 0; q < ndc; q++) z[s.con_dof[c][q]] = 2.f * mt * s.con_J[c][1 + tt][q];
      const int pr = s.con_pair[c], t1 = ns_tree(c_gbody[PG1(pr)]);
      isl[t] = g.ns_isl[t1 >= 0 ? t1 : ns_tree(c_gbody[PG2(pr)])];
      rr[t] = ns_gdot(s, dir, s.tmp) - (s.efc_aref[r0] - s.efc_aref[r0 + 1]);
      mm[t] = 0.5f * (F[r0] + F[r0 + 1]); yy[t] = 0.5f * (F[r0] - F[r0 + 1]);
    }
    for (int i = 0; i < NV; i++) {  // forward substitution in place
      float x = z[i];
      for (int j = 0; j < i; j++) x -= s.L[TRI(i, j)] * z[j];
      z[i] = x / s.L[TRI(i, i)];
    }
  }
  __syncwarp();
  for (int q = lane; q < ndir * ndir; q += 32) {  // A = Z' Z (each pair once, both halves: symmetric)
    const int a = q / ndir, b = q - a * ndir;
    if (b < a) continue;
    float acc = 0.f;
    for (int i = 0; i < NV; i++) acc += g.ns_Z[a][i] * g.ns_Z[b][i];
    g.ns_A[a][b] = acc;
    g.ns_A[b][a] = acc;
  }
  __syncwarp();
  // the sweeps; islands (representative body ids < 64) by bits; the improvement of island j on lane j & 31
  unsigned long long live = 0ull;
#pragma unroll
  for (int t = 0; t < NS_NSR; t++) if (isl[t] >= 0) live |= 1ull << isl[t];
  for (int o = 16; o > 0; o >>= 1) live |= __shfl_xor_sync(FULL, live, o);
  // every island with rows takes part (MuJoCo: also the ones without friction directions; they stop after 2 sweeps)
  unsigned long long all = 0ull;
  for (int r = lane; r < nefc; r += 32) {
    const int kind = s.efc_kind[r], a = s.efc_a[r];
    int tr;
    if (kind <= 1) tr = ns_tree(c_dofbody[a]);
    else { const int pr = s.con_pair[a]; tr = ns_tree(c_gbody[PG1(pr)]); if (tr < 0) tr = ns_tree(c_gbody[PG2(pr)]); }
    all |= 1ull << g.ns_isl[tr];
  }
  for (int o = 16; o > 0; o >>= 1) all |= __shfl_xor_sync(FULL, all, o);
  unsigned long long done = 0ull;
  int niter = 0;
  for (int it = 0; it < NS_ITER; it++) {
    if ((all & ~done) == 0ull) break;
    float imp[2] = {0.f, 0.f};
    if (it == 0)  // 0.5 sum f^2 R over the island's rows
      for (int r0 = 0; r0 < nefc; r0 += 32) {
        const int r = r0 + lane;
        float term = 0.f;
        int j = -1;
        if (r < nefc) {
          const int kind = s.efc_kind[r], a = s.efc_a[r];
          int tr;
          if (kind <= 1) tr = ns_tree(c_dofbody[a]);
          else { const int pr = s.con_pair[a]; tr = ns_tree(c_gbody[PG1(pr)]); if (tr < 0) tr = ns_tree(c_gbody[PG2(pr)]); }
          j = g.ns_isl[tr];
          term = 0.5f * F[r] * F[r] / s.efc_D[r];
        }
        for (int q = 0; q < 32; q++) {
          const float tq = __shfl_sync(FULL, term, q);
          const int jq = __shfl_sync(FULL, j, q);
          if (jq >= 0 && (jq & 31) == lane) imp[jq >> 5] += tq;
        }
      }
#pragma unroll 1
    for (int k = 0; k < ndir; k++) {
      const int ow = k & 31, tk = k >> 5;
      float r = rr[0], y = yy[0], m = mm[0];
      int ik = isl[0];
#pragma unroll
      for (int t = 1; t < NS_NSR; t++)
        if (tk == t) { r = rr[t]; y = yy[t]; m = mm[t]; ik = isl[t]; }
      r = __shfl_sync(FULL, r, ow); y = __shfl_sync(FULL, y, ow); m = __shfl_sync(FULL, m, ow);
      ik = __shfl_sync(FULL, ik, ow);
      if ((done >> ik) & 1ull) continue;
      const float akk = g.ns_A[k][k];
      const bool fl = k < NFLOSS;
      float yn, chg;
      int bnd = 0;
      if (fl) {
        const float ainv = 1.f / fmaxf(1e-15f, akk);
        yn = y - r * ainv;
        if (yn < -m) yn = -m; else if (yn > m) yn = m;
        const float d = yn - y;
        chg = 0.5f * d * d / ainv + d * r;
      } else {
        if (akk < 1e-15f) yn = 0.f;
        else {
          yn = y - r / akk;
          if (yn < -m) { yn = -m; bnd = 2; } else if (yn > m) { yn = m; bnd = 1; }
        }
        const float d = yn - y;
        chg = 0.5f * akk * d * d + d * r;
        if (chg > 1e-10f) { yn = y; chg = 0.f; bnd = -1; }  // (the cost went up: restored, status kept)
      }
      if ((ik & 31) == lane) imp[ik >> 5] -= chg;
      const float d = yn - y;
      if (d != 0.f) {
        const float* Ak = g.ns_A[k];
#pragma unroll
        for (int t = 0; t < NS_NSR; t++) {
          const int kp = lane + 32 * t;
          if (kp < ndir) rr[t] += Ak[kp] * d;
        }
      }
      if (lane == ow)
#pragma unroll
        for (int t = 0; t < NS_NSR; t++)
          if (tk == t) { yy[t] = yn; if (bnd >= 0) st[t] = bnd; }
    }
    niter = it + 1;
    // per island: stop when the scaled improvement is below the tolerance
    unsigned long long stop = 0ull;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int j = lane + 32 * h;
      if (j < NB && ((all & ~done) >> j) & 1ull && imp[h] * g.ns_scale[j] < NS_TOL) stop |= 1ull << j;
    }
    for (int o = 16; o > 0; o >>= 1) stop |= __shfl_xor_sync(FULL, stop, o);
    done |= stop;
  }
  if ((all & ~done) != 0ull) hdr |= NS_FLAG_NONCONV;
  // the final forces and statuses
#pragma unroll
  for (int t = 0; t < NS_NSR; t++) {
    const int k = lane + 32 * t;
    if (k >= ndir) continue;
    const int dir = g.ns_dir[k], r0 = dir & 1023;
    if (((dir >> 10) & 3) == 3) F[r0] = yy[t];
    else { F[r0] = mm[t] + yy[t]; F[r0 + 1] = mm[t] - yy[t]; }
    g.ns_dst[k] = st[t];
  }
  __syncwarp();
  if (rec) {
    for (int c = lane; c < MAXCON; c += 32) {
      unsigned w = 0u;
      if (c < ncon && s.con_dim[c] >= 3) {
        const int np = min(s.con_dim[c] - 1, s.con_nrow[c] / 2), k0 = g.ns_k0[c];
        const bool sel = NS_PAIR(s.con_pair[c]);  // (a pair outside noslip_pairs: its rows keep the Newton forces)
        for (int t = 0; t < np; t++) w |= (unsigned)(sel && k0 + t < ndir ? g.ns_dst[k0 + t] : 3) << (2 * t);
      }
      __stcs(rec + OFF_NSST + c, __uint_as_float(w));
    }
    tape_put(rec + OFF_NSF, F, nefc, OFF_NSDF - OFF_NSF, lane);
    tape_put(rec + OFF_NSJAR, s.efc_jar, nefc, OFF_LM - OFF_NSJAR, lane);  // (the Newton x = J a* - aref of the rows)
    for (int r = lane; r < OFF_NSJAR - OFF_NSDF; r += 32) __stcs(rec + OFF_NSDF + r, r < nefc ? F[r] - g.ns_f0[r] : 0.f);
    if (lane == 0) __stcs(rec + OFF_NSHDR, __uint_as_float(hdr | (unsigned)min(niter, 255)));
  }
  // qacc = M^-1 (qfrc_smooth + J' f), M qacc, and the rows' forces for the contact forces
  rows_jtf(s, F, s.tmp, lane);
  for (int i = lane; i < NV; i += 32) s.tmp[i] += s.qsm[i];
  __syncwarp();
  ns_llt_solve(s.L, s.tmp, lane);
  for (int i = lane; i < NV; i += 32) s.qacc[i] = s.tmp[i];
  __syncwarp();
  matvec(s.M, s.qacc, s.Ma, lane);
  for (int r = lane; r < nefc; r += 32) s.efc_jv[r] = F[r];
  __syncwarp();
  }
}
"""

PRE_ANCHOR = ("      __stcs(rec + OFF_CON + i, __uint_as_float(e));\n    }\n}\n")
PRE_ADD = ("      __stcs(rec + OFF_CON + i, __uint_as_float(e));\n    }\n"
           "  if (rec) {  // tape (forest_engine.noslip): each contact's position, frame and its rows' D\n"
           "    __syncwarp();\n"
           "    for (int i = lane; i < MAXCON; i += 32) {\n"
           "      float* o = rec + OFF_NSCON + NS_CON_WORDS * i;\n"
           "      const bool v = i < ncon;\n"
           "      for (int t = 0; t < 3; t++) __stcs(o + t, v ? s.con_pos[i][t] : 0.f);\n"
           "      for (int t = 0; t < 9; t++) __stcs(o + 3 + t, v ? s.con_frame[i][t] : 0.f);\n"
           "      __stcs(o + 12, v && s.con_nrow[i] > 0 ? s.efc_D[s.con_row[i]] : 0.f);\n"
           "    }\n"
           "  }\n}\n")

PRE_ANCHOR_W1 = ("      __stcs(rec + OFF_CON + i, __uint_as_float(e));\n    }\n  }  // warp 1 (or the only warp)\n")

ZONE_BLOCK = ("  if (rec) {  // tape: the final zones and their Hessian factor, before the implicitfast solve overwrites L\n"
              "    unsigned z = zone_word(s, lane);\n"
              "    if (lane < (NEFC + 31) / 32) __stcs(rec + OFF_ZONE + lane, __uint_as_float(z));\n"
              "    tape_put(rec + OFF_LH, s.L, NTRI, OFF_LQ - OFF_LH, lane);\n"
              "  }\n")
NITER = "  if (niter_out && lane == 0) *niter_out = it;\n"
OFORCE_EVAL = ("    for (int r = lane; r < s.nefc; r += 32) {\n"
               "      float ds, dd, cst;\n"
               "      row_eval(s, r, s.efc_jar[r], &ds, &dd, &cst);\n"
               "      s.efc_jv[r] = -ds;\n"
               "    }\n"
               "    __syncwarp();\n"
               "    const int nc = min(s.ncon, MAXCON);\n")
LQ_PUT = "    tape_put(rec + OFF_LQ, s.L, NTRI, REC_WORDS - OFF_LQ, lane);\n"
POST_ANCHOR = "template <class S>\n__device__ void substep_post(S& s, int iterations, int lane, int* niter_out, float* rec) {\n"
HAND_ANCHOR = "  SimNfull big;  // k_post's state of a world whose contacts or rows exceed SimN's capacity\n"


def _rep(src, a, b, n=1):
    assert src.count(a) == n, f"noslip anchor found {src.count(a)} times (want {n}): {a[:80]!r}"
    return src.replace(a, b)


def apply_forward(src, m, max_contacts=None, pairs=None):
    """The engine text (adjoint.cuda_source's) with the forward pass; asserts every anchor."""
    src = _rep(src, "#define REC_WORDS", defines(m, max_contacts) + mask_text(m, pairs) + "#define REC_WORDS")
    src = _rep(src, HAND_ANCHOR, HAND_ANCHOR + HAND)
    if src.count(PRE_ANCHOR) == 0 and src.count(PRE_ANCHOR_W1) == 1:
        src = _rep(src, PRE_ANCHOR_W1, PRE_ADD[:-len("}\n")] + "  }  // warp 1 (or the only warp)\n")
    else:
        src = _rep(src, PRE_ANCHOR, PRE_ADD)
    src = _rep(src, ZONE_BLOCK, "")
    src = _rep(src, NITER, NITER + ZONE_BLOCK + "  noslip_pass(s, lane, rec);  // (forest_engine.noslip)\n")
    src = _rep(src, OFORCE_EVAL, "    const int nc = min(s.ncon, MAXCON);  // (noslip_pass left the rows' forces)\n")
    src = _rep(src, LQ_PUT, "    tape_put(rec + OFF_LQ, s.L, NTRI, OFF_NSQN - OFF_LQ, lane);\n")
    src = _rep(src, POST_ANCHOR, FWD_DECL + POST_ANCHOR)
    i = src.index("struct SimP {")
    j = src.index("\n};\n", i) + 4
    src = src[:j] + FWD_DEVICE + src[j:]
    return src



ADJ_DOC = """
The adjoint of a noslip substep (implicit function theorem at the converged point, the active set frozen: the Newton
rows' zones as the engine's adjoint, and the pyramid pairs' statuses of the noslip pass: free, at +m, at -m).

Unknowns: the Newton acceleration a* (Newton's stationarity r_N(a*) = M a* - tau - J' f*(a*) = 0, f* = -D x on the
quadratic zones) and the noslip acceleration a with the free pairs' y_S:
  R1 = M a - tau - J' f_ns = 0,   f_ns = f* on the other rows, (m + y, m - y) on the pairs (m = (f*_0 + f*_1) / 2,
       y = y_S free, y = +-m bound),
  R2 = g_k . (a + b v) = 0 for the free pairs (g_k = J_0 - J_1 = 2 mu T_k; aref_0 - aref_1 = -b g_k v).
Given a bar (from the integrator: a enters implicitfast), the KKT adjoint [M G_S'; G_S 0] [lam1; lam2] = [a bar; 0]
gives lam1 = L^-T (I - Q Q') L^-1 a bar and lam2 = R^-1 Q' L^-1 a bar from the CGS2 QR Z = Q R of the free directions
Z = L^-1 G_S' (L = chol(M) from the tape; directions with |z| <= 1e-3 |z0| after the projection are dependent and get
lam2 = 0: the step is consistent for any lam2 of the redundant set). f* enters R1 through the mids and the fixed rows:
f*_bar_e = phi_e = l_n + sigma_t mu_t l_t on the rows of pair t (sigma 0 free, +-1 bound, l = J lam1), J_e lam1 on
every other row (limits, frictionless contacts, untouched pairs). With psi_e = phi_e - J_e lam1 (nonzero on pyramid
rows only), lam_N = H^-1 a*_bar, a*_bar = -sum_zone D_e J_e' phi_e, so lam_N + lam1 = H^-1 (M lam1 - sum_zone D_e J_e'
psi_e) (L_H, L_M from the tape).

The explicit partials, without the cancellation of two large terms (with pairs at the cone's bound lam1 and lamN
can be 100x the gradient and opposite): lam1 (R1 at a_ns) + lamN (r_N at a*) = lam_tot (r_N at a*) + lam1 (R1 at a_ns
minus r_N at a*), the latter's bias, actuator and Jacobian-value terms cancelling exactly:
  pass A  the engine's residual VJP at (a*, lam_tot, -y): RNE and actuator terms, the integrator's M(q) term at
          a_ns - a_adv (channel 2), the limit rows with lam_tot, the contacts' Psi = sum_zone D x_e (u_e . V_lam_tot +
          psi_e) (psi_e = phi_e - J_e lam1: the noslip cotangent beyond lam1's own);
  pass B  lam1' M(q) (a_ns - a*) (channel 2 with seed lam1, no channel 1), the contacts' geometry terms
          Psi = -sum_e (f_ns,e - f*_e) (u_e . V_lam1) + sum_t kappa_t T_t . (V_a + b V_v) (V_a of a_ns; kappa_t = 2 mu_t
          lam2_t); no actuator, damping or limit terms.
Not differentiated (refused: forest_engine.noslip.problems / adjoint.cuda_source): the elliptic cone, frictionloss
rows (the forest adjoint has none), equality and tendon rows, per-world parameters, the surrogates, terrain.
"""

ADJ_STRUCT_ANCHOR = "#if DK_ADJ_CLAMP_KV == 2\n  float u[NU], soft;"
ADJ_STRUCT = ("  // forest_engine.noslip: a*, a_ns, lam1, lam_tot, scratch, the free directions' QR in L^-1 space, per contact\n"
              "  // the rows' f_ns - f*, psi and kappa (2 mu lam2), and the pass (1: B, 0: A)\n"
              "  float ns_qn[NV], ns_an[NV], ns_l1[NV], ns_lN[NV], ns_u[NV], ns_w[NV], ns_g[NV], ns_zero[NV], ns_h[NS_QC];\n"
              "  float ns_B[32][NV], ns_Q[NS_QC][NV], ns_R[NS_QC * (NS_QC + 1) / 2];  // a batch of L^-1 g, Q, R\n"
              "  float ns_Hb[32][NS_QC], ns_n0[32];  // the batch's Gram-Schmidt coefficients, squared norms\n  int ns_dk[32];  // Q column -> its direction; direction -> contact | pair << 8\n"
              "  float ns_fns[MAXCON][6], ns_phi[MAXCON][6], ns_kap[MAXCON][3];\n"
              "  int ns_col[MAXCON][3], ns_pass;\n" + ADJ_STRUCT_ANCHOR)

ADJ_DEVICE = r"""
// ---------------------------------------------------------------- noslip adjoint (forest_engine.noslip)
// contact rows on dof i: J_u,i = s_i u . col_i, and the angular one s_i u . z_i (0 for translations and slides)
__device__ __forceinline__ void ns_dof_col(const Adj& s, int pair, const float* p, int i, float* col, float* ang) {
  const int b1 = c_gbody[PG1(pair)], b2 = c_gbody[PG2(pair)];
  const unsigned m1 = c_ancmask[b1], m2 = c_ancmask[b2];
  const int bi = c_dofbody[i];
  for (int t = 0; t < 3; t++) col[t] = ang[t] = 0.f;
  if (!(((m1 ^ m2) >> bi) & 1u)) return;
  const float sg = (m2 >> bi) & 1u ? 1.f : -1.f;
  const int jt = JT_OF(bi), k = i - c_dadr[bi];
  if (jt == JFREE && k < 3) { col[k] = sg; return; }
  float z[3], r[3];
  if (jt == JFREE) {
    const float ek[3] = {k == 3 ? 1.f : 0.f, k == 4 ? 1.f : 0.f, k == 5 ? 1.f : 0.f};
    qrot(s.Q[bi], ek, z);
  } else qrot(s.Q[bi], c_axis + 3 * bi, z);
  if (jt == JSLIDE) { for (int t = 0; t < 3; t++) col[t] = sg * z[t]; return; }
  for (int t = 0; t < 3; t++) r[t] = p[t] - s.X[bi][t];
  cross(z, r, col);
  for (int t = 0; t < 3; t++) { col[t] *= sg; ang[t] = sg * z[t]; }
}

// b = L^-1 b in place (packed lower L, shared b)
__device__ __noinline__ void ns_fwd_solve(const float* L, float* b, int lane) {
  for (int k = 0; k < NV; k++) {
    const float bk = b[k] / L[TRI(k, k)];
    __syncwarp();
    if (lane == 0) b[k] = bk;
    for (int i = k + 1 + lane; i < NV; i += 32) b[i] -= L[TRI(i, k)] * bk;
    __syncwarp();
  }
}
__device__ __noinline__ void ns_bwd_solve(const float* L, float* b, int lane) {
  for (int k = NV - 1; k >= 0; k--) {
    const float bk = b[k] / L[TRI(k, k)];
    __syncwarp();
    if (lane == 0) b[k] = bk;
    for (int i = lane; i < k; i += 32) b[i] -= L[TRI(k, i)] * bk;
    __syncwarp();
  }
}

// the noslip stage after adj_solves (s.abar = a bar of the noslip acceleration): lam1 (s.ns_l1), lamN (s.ns_lN), and
// per contact ns_fns, ns_phi, ns_kap (noslip.ADJ_DOC)
__device__ __noinline__ void adj_noslip(Adj& s, const float* rec, int lane) {
  LANE_BOUND(lane);
  const int ncon = s.hdr & 255, nlim = ((unsigned)s.hdr >> 18) & 63;
  for (int i = lane; i < NV; i += 32) s.ns_u[i] = s.abar[i];
  for (int e = lane; e < NTRI; e += 32) s.L[e] = __ldcs(rec + OFF_LM + e);
  __syncwarp();
  ns_fwd_solve(s.L, s.ns_u, lane);
  // the free directions (contact by contact, pair by pair) in batches of 32, one lane per direction: their rows g
  // (ns_B), Z = L^-1 g, classical Gram-Schmidt twice against the columns so far (Q in ns_Q, coefficients in ns_Hb),
  // then right-looking modified Gram-Schmidt within the batch. A direction whose residual norm is above 1e-3 of its
  // own norm becomes the next unit column (R in ns_R); dependent directions get lam2 = 0. (Was: one direction at a time
  // on the whole warp, 16 of the backward's 42 ms on the G1.)
  int rank = 0;
  for (int b0 = 0;; b0 += 32) {
    int nS = 0, r0 = nlim;
    for (int i = lane; i < 32 * NV; i += 32) (&s.ns_B[0][0])[i] = 0.f;
    __syncwarp();
    for (int c = 0; c < ncon; c++) {
      const unsigned e = s.con[c];
      const int pair = e & 0xFFFFu, dim = PDIM(pair);
      const int nr = max(0, min(dim >= 3 ? (dim == 4 ? 6 : 4) : 1, NEFC - r0));
      const int rc = r0;
      r0 += dim >= 3 ? (dim == 4 ? 6 : 4) : 1;
      unsigned zone = 0u;
      if (nr > 0) {
        const int w = rc >> 5;
        unsigned long long zz = s.zone[w];
        if (w + 1 < (NEFC + 31) / 32) zz |= (unsigned long long)s.zone[w + 1] << 32;
        zone = (unsigned)(zz >> (rc & 31)) & ((1u << nr) - 1u);
      }
      if (b0 == 0) {
        if (lane < 6) s.ns_fns[c][lane] = lane < nr ? __ldcs(rec + OFF_NSDF + rc + lane) : 0.f;  // (f_ns - f*)
        if (lane < 3) { s.ns_col[c][lane] = -1; s.ns_kap[c][lane] = 0.f; }
        if (lane < 6) s.ns_phi[c][lane] = 0.f;
      }
      if (!zone || dim < 3) continue;
      const unsigned st = __float_as_uint(__ldcs(rec + OFF_NSST + c));
      const int np = min(dim - 1, nr / 2);
      unsigned fm = 0u;  // the free pairs
      for (int t = 0; t < np; t++)
        if (((st >> (2 * t)) & 3u) == 0u) fm |= 1u << t;
      if (!fm) continue;
      const int n = __popc(fm);
      if (nS + n > b0 && nS < b0 + 32) {  // a direction of this batch
        const float* cw = rec + OFF_NSCON + NS_CON_WORDS * c;
        float p[3], fr[9];
        for (int t = 0; t < 3; t++) p[t] = __ldcs(cw + t);
        for (int t = 0; t < 9; t++) fr[t] = __ldcs(cw + 3 + t);
        const float mu = PFOOT(pair) ? s.fric : PMU(pair);
#if HAS_CONDIM4
        const float mus = dim == 4 ? PMUS(pair) : 0.f;
#else
        const float mus = 0.f;
#endif
        for (int i = lane; i < NV; i += 32) {
          float col[3], ang[3];
          ns_dof_col(s, pair, p, i, col, ang);
          int k = nS;
          for (int t = 0; t < 3; t++)
            if ((fm >> t) & 1u) {
              if (k >= b0 && k < b0 + 32)
                s.ns_B[k - b0][i] = t == 2 ? 2.f * mus * dot3(fr, ang) : 2.f * mu * dot3(fr + 3 * (1 + t), col);
              k++;
            }
        }
        if (lane == 0)
          for (int t = 0, k = nS; t < 3; t++)
            if ((fm >> t) & 1u) { if (k >= b0 && k < b0 + 32) s.ns_dk[k - b0] = c | t << 8; k++; }
      }
      nS += n;
    }
    const int nb = min(32, nS - b0);
    if (nb <= 0) break;
    __syncwarp();
    if (lane < nb) {  // this lane's direction: Z = L^-1 g, |z|^2, CGS2 against the columns so far
      float* z = s.ns_B[lane];
      for (int i = 0; i < NV; i++) {
        float x = z[i];
        for (int j = 0; j < i; j++) x -= s.L[TRI(i, j)] * z[j];
        z[i] = x / s.L[TRI(i, i)];
      }
      float n0 = 0.f;
      for (int i = 0; i < NV; i++) n0 += z[i] * z[i];
      s.ns_n0[lane] = n0;
      for (int j = 0; j < rank; j++) s.ns_Hb[lane][j] = 0.f;
      for (int pass = 0; pass < 2; pass++)
        for (int j = 0; j < rank; j++) {
          const float* q = s.ns_Q[j];
          float h = 0.f;
          for (int i = 0; i < NV; i++) h += q[i] * z[i];
          for (int i = 0; i < NV; i++) z[i] -= h * q[i];
          s.ns_Hb[lane][j] += h;
        }
    }
    __syncwarp();
    for (int k = 0; k < nb; k++) {  // within the batch: right-looking MGS
      const float* z = s.ns_B[k];
      float n1 = 0.f;
      for (int i = lane; i < NV; i += 32) n1 += z[i] * z[i];
      n1 = sqrtf(warp_sum(n1));
      if (!(rank < NS_QC && n1 > 1e-3f * sqrtf(s.ns_n0[k]))) continue;  // (uniform: warp_sum)
      for (int i = lane; i < NV; i += 32) s.ns_Q[rank][i] = z[i] / n1;
      for (int j = lane; j < rank; j += 32) s.ns_R[TRI(rank, j)] = s.ns_Hb[k][j];
      if (lane == 0) {
        s.ns_R[TRI(rank, rank)] = n1;
        s.ns_col[s.ns_dk[k] & 255][s.ns_dk[k] >> 8] = rank;
      }
      __syncwarp();
      const int k2 = k + 1 + lane;
      if (k2 < nb) {  // the later directions of the batch against q
        float* z2 = s.ns_B[k2];
        const float* q = s.ns_Q[rank];
        float h = 0.f;
        for (int i = 0; i < NV; i++) h += q[i] * z2[i];
        for (int i = 0; i < NV; i++) z2[i] -= h * q[i];
        s.ns_Hb[k2][rank] = h;
      }
      rank++;
      __syncwarp();
    }
    if (nb < 32) break;
  }
  // lam2 (on the independent columns) = R^-1 Q' u; w = u - Q Q' u; lam1 = L^-T w
  for (int q = 0; q < 2; q++) {
    const int j = lane + 32 * q;
    if (j < rank) { const float* qj = s.ns_Q[j]; float acc = 0.f; for (int i = 0; i < NV; i++) acc += qj[i] * s.ns_u[i]; s.ns_h[j] = acc; }
  }
  __syncwarp();
  for (int i = lane; i < NV; i += 32) {
    float acc = s.ns_u[i];
    for (int j = 0; j < rank; j++) acc -= s.ns_h[j] * s.ns_Q[j][i];
    s.ns_l1[i] = acc;
  }
  if (lane == 0)
    for (int j = rank - 1; j >= 0; j--) {
      float x = s.ns_h[j];
      for (int k = j + 1; k < rank; k++) x -= s.ns_R[TRI(k, j)] * s.ns_w[k];
      s.ns_w[j] = x / s.ns_R[TRI(j, j)];
    }
  __syncwarp();
  ns_bwd_solve(s.L, s.ns_l1, lane);
  // kappa, phi, and g = M lam1 - sum_zone D_e J_e' psi_e
  for (int c = 0; c < ncon; c++) {
    const unsigned e = s.con[c];
    const int pair = e & 0xFFFFu, dim = PDIM(pair);
    if (lane < 3 && s.ns_col[c][lane] >= 0) {
      const float mu = PFOOT(pair) ? s.fric : PMU(pair);
#if HAS_CONDIM4
      const float mt = lane == 2 ? PMUS(pair) : mu;
#else
      const float mt = mu;
#endif
      s.ns_kap[c][lane] = 2.f * mt * s.ns_w[s.ns_col[c][lane]];
    }
  }
  for (int i = lane; i < NV; i += 32) s.ns_g[i] = 0.f;
  __syncwarp();
  int r0 = nlim;
  for (int c = 0; c < ncon; c++) {
    const unsigned e = s.con[c];
    const int pair = e & 0xFFFFu, dim = PDIM(pair);
    const int nr = max(0, min(dim >= 3 ? (dim == 4 ? 6 : 4) : 1, NEFC - r0));
    const int rc = r0;
    r0 += dim >= 3 ? (dim == 4 ? 6 : 4) : 1;
    unsigned zone = 0u;
    if (nr > 0) {
      const int w = rc >> 5;
      unsigned long long zz = s.zone[w];
      if (w + 1 < (NEFC + 31) / 32) zz |= (unsigned long long)s.zone[w + 1] << 32;
      zone = (unsigned)(zz >> (rc & 31)) & ((1u << nr) - 1u);
    }
    if (!zone && !NS_ALL_ROWS) continue;  // (cone_mu: every contact's rows have a slope weight)
    const unsigned st = dim >= 3 ? __float_as_uint(__ldcs(rec + OFF_NSST + c)) : 0u;
    const float* cw = rec + OFF_NSCON + NS_CON_WORDS * c;
    float p[3], fr[9];
    for (int t = 0; t < 3; t++) p[t] = __ldcs(cw + t);
    for (int t = 0; t < 9; t++) fr[t] = __ldcs(cw + 3 + t);
    const float D = __ldcs(cw + 12), mu = PFOOT(pair) ? s.fric : PMU(pair);
#if HAS_CONDIM4
    const float mus = dim == 4 ? PMUS(pair) : 0.f;
#else
    const float mus = 0.f;
#endif
    float l[4] = {0.f, 0.f, 0.f, 0.f};  // n, t1, t2, r rows times lam1
    for (int i = lane; i < NV; i += 32) {
      float col[3], ang[3];
      ns_dof_col(s, pair, p, i, col, ang);
      const float li = s.ns_l1[i];
      l[0] += dot3(fr, col) * li; l[1] += dot3(fr + 3, col) * li; l[2] += dot3(fr + 6, col) * li;
      l[3] += dot3(fr, ang) * li;
    }
    for (int q = 0; q < 4; q++) l[q] = warp_sum(l[q]);
    float phi[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, psi[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (dim < 3) phi[0] = l[0];
    else
      for (int t = 0; t < nr / 2; t++) {
        const float mt = t == 2 ? mus : mu, lt = t == 2 ? l[3] : l[1 + t];
        const float le0 = l[0] + mt * lt, le1 = l[0] - mt * lt;
        const unsigned sx = (st >> (2 * t)) & 3u;
        const float ph = sx == 0u ? l[0] : sx == 1u ? le0 : le1;
        phi[2 * t] = sx == 3u ? le0 : ph;
        phi[2 * t + 1] = sx == 3u ? le1 : ph;
        psi[2 * t] = phi[2 * t] - le0;
        psi[2 * t + 1] = phi[2 * t + 1] - le1;
      }
    if (lane < 6) s.ns_phi[c][lane] = psi[lane];  // (pass A's offsets: psi = phi - J lam1)
    float cn = 0.f, ct[3] = {0.f, 0.f, 0.f};
    for (int q = 0; q < nr; q++) {
      const float wq = NS_ROW_W(rc + q, (zone >> q) & 1u, (zone >> (q & ~1)) & 3u);  // (a row's slope weight)
      if (wq == 0.f) continue;
      cn += wq * psi[q];
      const int t = q >> 1;
      if (dim >= 3) ct[t] += wq * ((q & 1) ? -1.f : 1.f) * (t == 2 ? mus : mu) * psi[q];
    }
    if (cn != 0.f || ct[0] != 0.f || ct[1] != 0.f || ct[2] != 0.f)
      for (int i = lane; i < NV; i += 32) {
        float col[3], ang[3];
        ns_dof_col(s, pair, p, i, col, ang);
        s.ns_g[i] += D * (cn * dot3(fr, col) + ct[0] * dot3(fr + 3, col) + ct[1] * dot3(fr + 6, col)
                          + ct[2] * dot3(fr, ang));
      }
    __syncwarp();
  }
  // w = M lam1 - g (M = L L'), lam_tot = H^-1 w, lamN = lam_tot - lam1
  for (int i = lane; i < NV; i += 32) {
    float acc = 0.f;
    for (int k = i; k < NV; k++) acc += s.L[TRI(k, i)] * s.ns_l1[k];
    s.ns_u[i] = acc;
  }
  __syncwarp();
  for (int i = lane; i < NV; i += 32) {
    float acc = 0.f;
    for (int k = 0; k <= i; k++) acc += s.L[TRI(i, k)] * s.ns_u[k];
    s.ns_w[i] = acc - s.ns_g[i];
  }
  __syncwarp();
  for (int e = lane; e < NTRI; e += 32) s.L[e] = __ldcs(rec + OFF_LH + e);
  __syncwarp();
  NS_TOT_SOLVE(s.L, s.ns_w, lane);  // (lam_tot = H^-1 w; the surrogate's cone_mu: H_mu, surrogate_hooks)
  for (int i = lane; i < NV; i += 32) s.ns_lN[i] = s.ns_w[i];  // lam_tot = lamN + lam1
  __syncwarp();
}
"""

PSI_DEVICE = r"""
// pass A: contact_psi_vjp with the offset psi_e of each row added to u_e . V_lam in the force-value terms
__device__ __forceinline__ void contact_psi_phi(const float* Vv, const float* Va, const float* Vl, const float* fr,
                                                float mu, int dim, unsigned zone, float d, float D, float dD, float k,
                                                float b, float imp, float dimp, const float* phi, float* Wv, float* Wa,
                                                float* Wl, float* dbar, float* frbar
#if HAS_CONDIM4
                                                , const float* Ov, const float* Oa, const float* Ol, float mus,
                                                float* AWv, float* AWa, float* AWl
#endif
                                                ) {
  float vf[3], af[3], lf[3];
#pragma unroll
  for (int j = 0; j < 3; j++) {
    vf[j] = dot3(fr + 3 * j, Vv); af[j] = dot3(fr + 3 * j, Va); lf[j] = dot3(fr + 3 * j, Vl);
  }
  float g = k * imp * d, gd = k * (imp + dimp * d);
  float cl[3] = {0.f, 0.f, 0.f}, ca[3] = {0.f, 0.f, 0.f}, db = 0.f;
#pragma unroll
  for (int e = 0; e < 4; e++) {
    if (!((zone >> e) & 1u) || (e > 0 && dim < 3)) continue;
    const int t = 1 + (e >> 1);
    float sm = (e & 1) ? -mu : mu;
    float uv = vf[0], ua = af[0], ul = lf[0];
    if (dim >= 3) { uv += sm * vf[t]; ua += sm * af[t]; ul += sm * lf[t]; }
    const float uw = ul + phi[e];
    float x = ua + g + b * uv, Dx = D * x, Dl = D * uw;
    cl[0] += Dx; ca[0] += Dl;
    if (dim >= 3) { cl[t] += sm * Dx; ca[t] += sm * Dl; }
    db += uw * (dD * x + D * gd);
  }
#if HAS_CONDIM4
  float awl = 0.f, awa = 0.f;
  if (dim == 4) {
    const float wv = dot3(fr, Ov), wa = dot3(fr, Oa), wl = dot3(fr, Ol);
#pragma unroll
    for (int e = 4; e < 6; e++) {
      if (!((zone >> e) & 1u)) continue;
      const float sm = (e & 1) ? -mus : mus;
      const float uv = vf[0] + sm * wv, ua = af[0] + sm * wa, ul = lf[0] + sm * wl, uw = ul + phi[e];
      const float x = ua + g + b * uv, Dx = D * x, Dl = D * uw;
      cl[0] += Dx; ca[0] += Dl;
      awl += sm * Dx; awa += sm * Dl;
      db += uw * (dD * x + D * gd);
    }
  }
#pragma unroll
  for (int i = 0; i < 3; i++) { AWl[i] = awl * fr[i]; AWa[i] = awa * fr[i]; AWv[i] = b * AWa[i]; }
#endif
#pragma unroll
  for (int i = 0; i < 3; i++) {
    Wl[i] = cl[0] * fr[i] + cl[1] * fr[3 + i] + cl[2] * fr[6 + i];
    Wa[i] = ca[0] * fr[i] + ca[1] * fr[3 + i] + ca[2] * fr[6 + i];
    Wv[i] = b * Wa[i];
  }
  float ab[3] = {Va[0] + b * Vv[0], Va[1] + b * Vv[1], Va[2] + b * Vv[2]};
#pragma unroll
  for (int j = 0; j < 3; j++)
#pragma unroll
    for (int i = 0; i < 3; i++) frbar[3 * j + i] = cl[j] * Vl[i] + ca[j] * ab[i];
#if HAS_CONDIM4
#pragma unroll
  for (int i = 0; i < 3; i++) frbar[i] += awl * Ol[i] + awa * (Oa[i] + b * Ov[i]);
#endif
  *dbar = db;
}

// pass B: Psi = -sum_e f_e (u_e . V_lam) + sum_t kappa_t T_t . (V_a + b V_v) (f: f_ns - f* of the rows, V_lam of
// lam1, V_a of the noslip acceleration); no distance term
__device__ __forceinline__ void contact_psi_ns(const float* Vv, const float* Va, const float* Vl, const float* fr,
                                               float mu, int dim, float b, const float* fns, const float* kap,
                                               float* Wv, float* Wa, float* Wl, float* dbar, float* frbar
#if HAS_CONDIM4
                                               , const float* Ov, const float* Oa, const float* Ol, float mus,
                                               float* AWv, float* AWa, float* AWl
#endif
                                               ) {
  float cl[3] = {0.f, 0.f, 0.f}, ca[3] = {0.f, 0.f, 0.f};
#pragma unroll
  for (int e = 0; e < 4; e++) {
    if (e > 0 && dim < 3) continue;
    const int t = 1 + (e >> 1);
    const float sm = (e & 1) ? -mu : mu, f = fns[e];
    cl[0] -= f;
    if (dim >= 3) cl[t] -= sm * f;
  }
  if (dim >= 3) { ca[1] = kap[0]; ca[2] = kap[1]; }
#if HAS_CONDIM4
  float awl = 0.f, awa = 0.f;
  if (dim == 4) {
#pragma unroll
    for (int e = 4; e < 6; e++) {
      const float sm = (e & 1) ? -mus : mus, f = fns[e];
      cl[0] -= f;
      awl -= sm * f;
    }
    awa = kap[2];
  }
#pragma unroll
  for (int i = 0; i < 3; i++) { AWl[i] = awl * fr[i]; AWa[i] = awa * fr[i]; AWv[i] = b * AWa[i]; }
#endif
#pragma unroll
  for (int i = 0; i < 3; i++) {
    Wl[i] = cl[0] * fr[i] + cl[1] * fr[3 + i] + cl[2] * fr[6 + i];
    Wa[i] = ca[0] * fr[i] + ca[1] * fr[3 + i] + ca[2] * fr[6 + i];
    Wv[i] = b * Wa[i];
  }
  float ab[3] = {Va[0] + b * Vv[0], Va[1] + b * Vv[1], Va[2] + b * Vv[2]};
#pragma unroll
  for (int j = 0; j < 3; j++)
#pragma unroll
    for (int i = 0; i < 3; i++) frbar[3 * j + i] = cl[j] * Vl[i] + ca[j] * ab[i];
#if HAS_CONDIM4
#pragma unroll
  for (int i = 0; i < 3; i++) frbar[i] += awl * Ol[i] + awa * (Oa[i] + b * Ov[i]);
#endif
  *dbar = 0.f;
}
"""

PSI_CALL4 = ("  contact_psi_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4, Ov, Oa, Ol,\n"
             "                  dim == 4 ? PMUS(pair) : 0.f, AWv, AWa, AWl);\n")
PSI_CALL3 = "  contact_psi_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4);\n"
PSI_NEW4 = ("  if (s.ns_pass)  // (forest_engine.noslip: the two passes' contact terms)\n"
            "    contact_psi_ns(Vv, Va, Vl, fr, mu, dim, b, s.ns_fns[c], s.ns_kap[c], Wv, Wa, Wl, cot, cot + 4, Ov, Oa,\n"
            "                   Ol, dim == 4 ? PMUS(pair) : 0.f, AWv, AWa, AWl);\n"
            "  else\n"
            "    contact_psi_phi(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, s.ns_phi[c], Wv, Wa, Wl, cot,\n"
            "                    cot + 4, Ov, Oa, Ol, dim == 4 ? PMUS(pair) : 0.f, AWv, AWa, AWl);\n")
PSI_NEW3 = ("  if (s.ns_pass)  // (forest_engine.noslip: the two passes' contact terms)\n"
            "    contact_psi_ns(Vv, Va, Vl, fr, mu, dim, b, s.ns_fns[c], s.ns_kap[c], Wv, Wa, Wl, cot, cot + 4);\n"
            "  else\n"
            "    contact_psi_phi(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, s.ns_phi[c], Wv, Wa, Wl, cot,\n"
            "                    cot + 4);\n")

FLAGS_ANCHOR = "    s.flags |= (int)__float_as_uint(__ldcs(rec + OFF_FLAGS));\n"
FLAGS_ADD = FLAGS_ANCHOR + ("    if (__float_as_uint(__ldcs(rec + OFF_NSHDR)) & (NS_FLAG_NONCONV | NS_FLAG_OVERFLOW | NS_FLAG_SKIPPED))\n"
                            f"      s.flags |= {ADJ_NOSLIP};  // (noslip.ADJ_NOSLIP)\n")
A0_ANCHOR = "  if (lane < (NEFC + 31) / 32) s.zone[lane] = __float_as_uint(__ldcs(rec + OFF_ZONE + lane));\n"
A0_ADD = A0_ANCHOR + ("  for (int i = lane; i < NV; i += 32) {  // (noslip: a_ns kept, the main pass at the Newton a*; at =\n"
                      "    s.ns_an[i] = s.a[i]; s.ns_qn[i] = __ldcs(rec + OFF_NSQN + i); s.a[i] = s.ns_qn[i];  // a_ns - a_adv)\n"
                      "  }\n")
PASSES_OLD = "  adj_tree_down(s, s.lam, s.ny, lane);  // A3 (channel 2, twists), A4\n"
PASSES_NEW = """  // forest_engine.noslip: lam1, lam_tot = lamN + lam1, psi, kappa, f_ns - f*; pass A: the Newton residual's VJP at a*
  // with lam_tot (contacts: psi offsets) and the integrator's -y at a_ns - a_adv; pass B (after A8): lam1's terms
  adj_noslip(s, rec, lane);
  for (int i = lane; i < NV; i += 32) s.lam[i] = s.ns_lN[i];
  if (lane == 0) s.ns_pass = 0;
  __syncwarp();
  adj_tree_down(s, s.lam, s.ny, lane);  // A3 (channel 2, twists), A4
"""
COMPOSE_OLD = "  adj_compose(s, lane);  // A8\n  APROF(APROF_COMPOSE);\n"
COMPOSE_NEW = COMPOSE_OLD + """  // pass B (forest_engine.noslip): lam1' M(q) (a_ns - a*) as channel 2 (seed lam1, accelerations a_ns - a*), the
  // contacts' -lam1' J' (f_ns - f*) and lam2' G (a_ns + b v) (geometry only); no channel 1, actuator or limit terms
  for (int i = lane; i < NV; i += 32) {
    s.at[i] = s.ns_an[i] - s.ns_qn[i]; s.a[i] = s.ns_an[i]; s.lam[i] = s.ns_l1[i]; s.ny[i] = 0.f; s.ns_zero[i] = 0.f;
  }
  if (lane == 0) s.ns_pass = 1;
  __syncwarp();
  adj_poses(s, lane);
  adj_tree_down(s, s.ns_zero, s.ns_l1, lane);
#if DK_ADJ_STAGE >= 2 && DK_ADJ_CONTACTS
  adj_contacts<false>(s, lane, nullptr);
#endif
  adj_tree_up(s, s.ns_zero, s.ns_l1, lane);
  for (int i = lane; i < NV; i += 32) s.lam[i] = 0.f;
  __syncwarp();
  adj_compose(s, lane);
"""
SUBSTEP_ANCHOR = "// one substep's VJP (A0-A8)"


def apply_adjoint(src, m, surrogate=False):
    src = _rep(src, "#define REC_WORDS", "#define NS_QC (NV < NS_CAP ? NV : NS_CAP)  // (noslip adjoint) QR columns\n"
               "#define NS_ROW_W(r, z, zp) ((z) ? 1.f : 0.f)  // a contact row's weight in lam_tot's psi term: its zone\n"
               "#define NS_TOT_SOLVE(L, w, lane) tri_solve_packed(L, w, lane)\n#define NS_ALL_ROWS 0\n"
               "#ifdef DK_ADJ_DEBUG\n#error \"forest_engine.noslip: no -DDK_ADJ_DEBUG build\"\n#endif\n"
               "#if DK_CLAMP_IMPLICIT == 2\n#error \"forest_engine.noslip: not with DK_CLAMP_IMPLICIT=2\"\n#endif\n"
               "#define REC_WORDS")
    src = _rep(src, ADJ_STRUCT_ANCHOR, ADJ_STRUCT)
    src = _rep(src, "__device__ __forceinline__ float adj_contact_one(", PSI_DEVICE
               + "__device__ __forceinline__ float adj_contact_one(")
    if not surrogate:
        if src.count(PSI_CALL4):
            src = _rep(src, PSI_CALL4, PSI_NEW4)
        src = _rep(src, PSI_CALL3, PSI_NEW3)
    src = _rep(src, A0_ANCHOR, A0_ADD)
    src = _rep(src, FLAGS_ANCHOR, FLAGS_ADD)
    src = _rep(src, PASSES_OLD, PASSES_NEW)
    src = _rep(src, COMPOSE_OLD, COMPOSE_NEW)
    src = _rep(src, SUBSTEP_ANCHOR, ADJ_DEVICE + SUBSTEP_ANCHOR)
    return src


def apply(src, m, max_contacts=None, pairs=None, surrogate=False):
    """The engine text with the noslip pass, forward and adjoint (adjoint.cuda_source); pairs: resolve_pairs'.
    surrogate: a contact_surrogate build (surrogate_hooks follows surrogate.apply)."""
    src = apply_forward(src, m, max_contacts, pairs)
    return apply_adjoint(src, m, surrogate)


SUR_DOC = """
noslip with the contact-gradient surrogates (init(contact_surrogate=True), backward cone_mu > 0): the main (Newton)
solve's contact rows take the surrogate's central-path relaxation exactly as without noslip (surrogate.py): pass A's
contact partials weight the slope terms by w_e and the force-value terms by the zone bit, and lam_tot = H_mu^-1 (M lam1
- sum_e w_e D_e J_e' psi_e) with the surrogate's relaxed Hessian H_mu (every row of a contact with a quadratic row;
psi_e = phi_e - J_e lam1 on every row, also of the contacts without a quadratic row: as the main solve's linearized
law, the noslip pass sees their forces move). The noslip pass's own discrete state
(the pairs at the cone's bound or free) stays exact. impulse_clip scales pass A's contact partials by the Newton
forces' impulse; pass B is not clipped. cone_mu = 0 and no clip: the exact noslip adjoint's arithmetic.
Not with per-world parameters (not verified).
"""


def surrogate_hooks(src):
    """A noslip build's text after surrogate.apply (SUR_DOC): the two passes' dispatch inside the surrogate's wrapper,
    contact_psi_vjp_sur_phi, the relaxed weights and Hessian of lam_tot."""
    import re
    i = src.index("__device__ __forceinline__ float contact_psi_vjp_sur(")
    j = src.index("\n}\n", i) + 3
    f = src[i:j].replace("contact_psi_vjp_sur(", "contact_psi_vjp_sur_phi(", 1)
    f = f.replace("float* Wl, float* dbar, float* frbar, float smu", "float* Wl, float* dbar, float* frbar, float smu,\n"
                  "                                                     const float* phi", 1)
    assert f.count("const float Dx = D * x, Dl = D * ul;") == 2 and f.count("db += ul * (") == 2
    f = f.replace("const float Dx = D * x, Dl = D * ul;", "const float Dx = D * x, Dl = D * (ul + phi[e]);")
    f = f.replace("db += ul * (", "db += (ul + phi[e]) * (")
    src = src[:j] + "// pass A with the surrogates (forest_engine.noslip.SUR_DOC): psi offsets as contact_psi_phi\n" + f + src[j:]
    for call, new, a4 in ((PSI_CALL4, PSI_NEW4, True), (PSI_CALL3, PSI_NEW3, False)):
        inner = "".join("  " + line + "\n" for line in call.rstrip("\n").split("\n"))
        if not src.count(inner):
            continue
        src = _rep(src, inner, "".join("  " + line + "\n" for line in new.rstrip("\n").split("\n")))
    pat = re.compile(r"    const float fm = contact_psi_vjp_sur\(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot,\n"
                     r"\s+cot \+ 4, s\.sur_mu(?P<a4>[^;]*)\);\n")
    def sub(mt):
        a4 = mt.group("a4")
        ns4 = ", Ov, Oa, Ol, dim == 4 ? PMUS(pair) : 0.f, AWv, AWa, AWl" if a4 else ""
        return ("    float fm = 0.f;  // (noslip: pass B's contact terms unclipped, fm 0)\n"
                "    if (s.ns_pass)\n"
                f"      contact_psi_ns(Vv, Va, Vl, fr, mu, dim, b, s.ns_fns[c], s.ns_kap[c], Wv, Wa, Wl, cot, cot + 4{ns4});\n"
                "    else\n"
                "      fm = contact_psi_vjp_sur_phi(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot,\n"
                f"                                   cot + 4, s.sur_mu, s.ns_phi[c]{a4});\n")
    src, n = pat.subn(sub, src)
    assert n >= 1, "noslip.surrogate_hooks: the surrogate's psi call"
    src = _rep(src, "#define NS_ROW_W(r, z, zp) ((z) ? 1.f : 0.f)  // a contact row's weight in lam_tot's psi term: its zone\n",
               "#define NS_ROW_W(r, z, zp) (s.sur_mu > 0.f ? sur_weight(__ldcs(rec + OFF_NSJAR + (r)), s.sur_mu) "
               ": ((z) ? 1.f : 0.f))  // (cone_mu: the central-path weight)\n")
    src = _rep(src, "#define NS_ALL_ROWS 0\n", "#define NS_ALL_ROWS (s.sur_mu > 0.f)\n")
    src = _rep(src, "#define NS_TOT_SOLVE(L, w, lane) tri_solve_packed(L, w, lane)\n",
               "#define NS_TOT_SOLVE(L, w, lane) do { if (s.sur_mu > 0.f) { __syncwarp(); adj_sur_hessian(s, lane); "
               "chol_solve_compact(L, L, w, lane); } else tri_solve_packed(L, w, lane); } while (0)\n")
    return src

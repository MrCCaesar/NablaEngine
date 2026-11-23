"""Elliptic friction cones in the G1-class engine: text edits of the forward for a model with
opt.cone = elliptic (none otherwise: the pyramidal text unchanged). MuJoCo Warp 3.17's semantics (mujoco_warp/_src
constraint.py _efc_contact, solver.py _eval_constraint, _eval_elliptic_*, curvature_terms,
_elliptic_hessian_entry_from_projections; Copyright 2025 The Newton Developers, Apache License 2.0), translated to this
engine's row-oriented Newton solver:"""
import mujoco
import numpy as np


def applies(m):
    return int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)


def supported(m):
    """(ok, reason): the elliptic builds' model conditions."""
    for p in range(m.npair):
        d = int(m.pair_dim[p])
        if d not in (1, 3, 4):
            return False, f"pair {p}: condim {d} with the elliptic cone (1, 3 and 4 only)"
        if d >= 3 and m.pair_friction[p, 0] != m.pair_friction[p, 1]:
            return False, f"pair {p}: anisotropic sliding friction"
        if np.any(m.pair_solreffriction[p] != 0.0):
            return False, f"pair {p}: solreffriction"
    return True, ""


def isq(m):
    """MuJoCo Warp's opt.impratio_invsqrt as float32."""
    return np.float32(1.0 / np.sqrt(max(float(m.opt.impratio), mujoco.mjMINVAL)))


DEVICE = r"""
// ---- elliptic friction cones (g1_engine.elliptic; MuJoCo Warp 3.17's solver.py) ----
// MuJoCo Warp's _eval_constraint for elliptic row r (kind 4) at the rows' jar in s: its force and cost (the cone's on
// the normal row), returns the contact's zone (0 top, 1 bottom: quadratic, 2 cone)
template <class S>
__device__ __forceinline__ int ell_row(const S& s, int r, float* force, float* cost) {
  const int c = s.efc_a[r], t = s.efc_b[r], r0 = r - t, nr = s.con_nrow[c];
  const float f = s.con_mu[c], mu = f * ELL_ISQ, N = s.efc_jar[r0] * mu;
  float TT = 0.f, uf = 0.f;
#pragma unroll
  for (int j = 1; j < ELL_NT; j++)
    if (j < nr) {
#if ELL_NT == 4
      const float fj = j == 3 ? PMUS(s.con_pair[c]) : f;  // (condim 4: the torsional row's friction)
#else
      const float fj = f;
#endif
      const float u = s.efc_jar[r0 + j] * fj;
      TT += u * u;
      if (j == t) uf = u * fj;
    }
  const float T = TT <= 0.f ? 0.f : sqrtf(TT), x = s.efc_jar[r], D = s.efc_D[r];
  if (N >= mu * T || (T <= 0.f && N >= 0.f)) { *force = 0.f; *cost = 0.f; return 0; }
  if (mu * N + T <= 0.f || (T <= 0.f && N < 0.f)) { *force = -D * x; *cost = 0.5f * D * x * x; return 1; }
  const float dm = s.efc_D[r0] / (mu * mu * (1.f + mu * mu)), nmt = N - mu * T, fn = -dm * nmt * mu;
  if (t == 0) { *force = fn; *cost = 0.5f * dm * nmt * nmt; }
  else { *force = -(fn / T) * uf; *cost = 0.f; }
  return 2;
}

// contact c's Hessian coefficients C (n n, n t1, t1 t1, n t2, t2 t2, t1 t2; condim 4 also n r, r r, t1 r, t2 r) at the
// rows' jar: condim 1 the normal row's D when active; elliptic: the bottom zone's diagonal D, the cone's block
// (curvature_terms and _elliptic_hessian_entry_from_projections on the frame rows). Returns whether C is nonzero
template <class S>
__device__ __forceinline__ bool ell_coef(const S& s, int c, float* C) {
  const int r0 = s.con_row[c], nr = s.con_nrow[c];
#pragma unroll
  for (int k = 0; k < ELL_NC; k++) C[k] = 0.f;
  if (nr <= 0) return false;
  const float D0 = s.efc_D[r0];
  if (s.con_dim[c] < 3) {
    C[0] = s.efc_jar[r0] < 0.f ? D0 : 0.f;
    return C[0] != 0.f;
  }
  float fo, co;
  const int z = ell_row(s, r0, &fo, &co);
  if (z == 0) return false;
  if (z == 1) {
    C[0] = D0;
    C[2] = nr > 1 ? s.efc_D[r0 + 1] : 0.f;
    C[4] = nr > 2 ? s.efc_D[r0 + 2] : 0.f;
#if ELL_NT == 4
    C[7] = nr > 3 ? s.efc_D[r0 + 3] : 0.f;
#endif
    return true;
  }
  const float f = s.con_mu[c], mu = f * ELL_ISQ, mu2 = mu * mu, dm = D0 / (mu2 * (1.f + mu2));
  if (dm == 0.f) return false;
  const float n = s.efc_jar[r0] * mu, u1 = nr > 1 ? s.efc_jar[r0 + 1] * f : 0.f, u2 = nr > 2 ? s.efc_jar[r0 + 2] * f : 0.f;
#if ELL_NT == 4
  const float f3 = nr > 3 ? PMUS(s.con_pair[c]) : 0.f, u3 = nr > 3 ? s.efc_jar[r0 + 3] * f3 : 0.f;
  const float t = fmaxf(sqrtf(u1 * u1 + u2 * u2 + u3 * u3), 1e-15f), it = 1.f / t, w1 = u1 * it, w2 = u2 * it;
#else
  const float t = fmaxf(sqrtf(u1 * u1 + u2 * u2), 1e-15f), it = 1.f / t, w1 = u1 * it, w2 = u2 * it;
#endif
  const float mnt = mu * n * it, td = mu2 - mnt, ff = f * f;
  C[0] = dm * mu2;
  C[1] = -dm * mu2 * f * w1;
  C[3] = -dm * mu2 * f * w2;
  C[2] = dm * ff * (mnt * w1 * w1 + td);
  C[4] = dm * ff * (mnt * w2 * w2 + td);
  C[5] = dm * ff * mnt * w1 * w2;
#if ELL_NT == 4
  if (nr > 3) {  // the torsional row r: scale f3 (MuJoCo Warp's friction[2])
    const float w3 = u3 * it;
    C[6] = -dm * mu2 * f3 * w3;
    C[7] = dm * f3 * f3 * (mnt * w3 * w3 + td);
    C[8] = dm * f * f3 * mnt * w1 * w3;
    C[9] = dm * f * f3 * mnt * w2 * w3;
  }
#endif
  return true;
}

// an elliptic contact in the line search: its rows' jar x and J p j at alpha = 0, their D, the cone's mu, the friction f
struct EllLs {
  float x[ELL_NT], j[ELL_NT], D[ELL_NT], mu, f;
#if ELL_NT == 4
  float f3;  // the torsional row's friction (condim 4)
#endif
  int nr;  // 0: not an elliptic contact (or none)
};
template <class S>
__device__ __forceinline__ void ell_ls_load(const S& s, int c, int ncon, EllLs& e) {
  e.nr = 0;
  e.mu = e.f = 0.f;
#if ELL_NT == 4
  e.f3 = 0.f;
#endif
#pragma unroll
  for (int k = 0; k < ELL_NT; k++) e.x[k] = e.j[k] = e.D[k] = 0.f;
  if (c >= ncon || s.con_dim[c] < 3 || s.con_nrow[c] <= 0) return;
  const int r0 = s.con_row[c];
  e.nr = s.con_nrow[c];
  e.f = s.con_mu[c];
  e.mu = e.f * ELL_ISQ;
#if ELL_NT == 4
  e.f3 = PMUS(s.con_pair[c]);
#endif
#pragma unroll
  for (int k = 0; k < ELL_NT; k++)
    if (k < e.nr) { e.x[k] = s.efc_jar[r0 + k]; e.j[k] = s.efc_jv[r0 + k]; e.D[k] = s.efc_D[r0 + k]; }
}
// the contact's cost derivative d1, curvature d2 and cost at alpha (MuJoCo Warp's _eval_elliptic_*); returns its zone
__device__ __forceinline__ int ell_ls_eval(const EllLs& e, float alpha, float* d1, float* d2, float* cost) {
  *d1 = *d2 = *cost = 0.f;
  if (e.nr == 0) return 0;
  float xa[ELL_NT];
#pragma unroll
  for (int k = 0; k < ELL_NT; k++) xa[k] = e.x[k] + alpha * e.j[k];
#if ELL_NT == 4
  const float N = xa[0] * e.mu, u1 = xa[1] * e.f, u2 = xa[2] * e.f, u3 = xa[3] * e.f3, TT = u1 * u1 + u2 * u2 + u3 * u3;
#else
  const float N = xa[0] * e.mu, u1 = xa[1] * e.f, u2 = xa[2] * e.f, TT = u1 * u1 + u2 * u2;
#endif
  const float T = TT <= 0.f ? 0.f : sqrtf(TT);
  if (N >= e.mu * T || (T <= 0.f && N >= 0.f)) return 0;
  if (e.mu * N + T <= 0.f || (T <= 0.f && N < 0.f)) {
#pragma unroll
    for (int k = 0; k < ELL_NT; k++) {
      *d1 += e.D[k] * xa[k] * e.j[k];
      *d2 += e.D[k] * e.j[k] * e.j[k];
      *cost += 0.5f * e.D[k] * xa[k] * xa[k];
    }
    return 1;
  }
  const float dm = e.D[0] / (e.mu * e.mu * (1.f + e.mu * e.mu));
#if ELL_NT == 4
  const float v1 = e.j[1] * e.f, v2 = e.j[2] * e.f, v3 = e.j[3] * e.f3, T1 = (u1 * v1 + u2 * v2 + u3 * v3) / T,
              T2 = (v1 * v1 + v2 * v2 + v3 * v3 - T1 * T1) / T;
#else
  const float v1 = e.j[1] * e.f, v2 = e.j[2] * e.f, T1 = (u1 * v1 + u2 * v2) / T, T2 = (v1 * v1 + v2 * v2 - T1 * T1) / T;
#endif
  const float r = N - e.mu * T, r1 = e.j[0] * e.mu - e.mu * T1;
  *d1 = dm * r * r1;
  *d2 = dm * (r1 * r1 - e.mu * r * T2);
  *cost = 0.5f * dm * r * r;
  return 2;
}
// contact c's rows' forces at the rows' jar into efc_jv (ell_row each)
template <class S>
__device__ __forceinline__ void ell_forces(S& s, int c, int ncon) {
  if (c >= ncon || s.con_dim[c] < 3) return;
  const int r0 = s.con_row[c], nr = s.con_nrow[c];
  for (int k = 0; k < nr; k++) {
    float fo, co;
    ell_row(s, r0 + k, &fo, &co);
    s.efc_jv[r0 + k] = fo;
  }
}
// any contact on its cone zone at the rows' jar (warp vote)
template <class S>
__device__ __forceinline__ bool ell_any_cone(const S& s, int lane) {
  bool cone = false;
  const int ncon = min(s.ncon, MAXCON);
  for (int c = lane; c < ncon; c += 32)
    if (s.con_dim[c] >= 3 && s.con_nrow[c] > 0) {
      float fo, co;
      cone |= ell_row(s, s.con_row[c], &fo, &co) == 2;
    }
  return __any_sync(0xffffffffu, cone);
}

"""

ROWS3 = """        int nr = max(0, min(tri ? 3 : 1, NEFC - r0));  // rows beyond NEFC are dropped
        s.con_row[c] = r0; s.con_nrow[c] = nr;
        const float Dt = D / RMUL(ELL_ISQ, ELL_ISQ);  // friction rows: invweight / impratio (MuJoCo Warp's elliptic D)
#pragma unroll
        for (int t = 0; t < 3; t++) {
          if (t >= nr) break;
          s.efc_kind[r0 + t] = tri ? 4 : 3; s.efc_a[r0 + t] = c; s.efc_b[r0 + t] = t;
          s.efc_D[r0 + t] = t ? Dt : D;
          s.efc_aref[r0 + t] = t ? -bb * s.con_dot[c][t] : -k * imp * pos - bb * s.con_dot[c][0];
        }
"""
ROWS4 = """        int nr = max(0, min(tri ? (s.con_dim[c] == 4 ? 4 : 3) : 1, NEFC - r0));  // rows beyond NEFC are dropped
        s.con_row[c] = r0; s.con_nrow[c] = nr;
        const float Dt = D / RMUL(ELL_ISQ, ELL_ISQ);  // friction rows: invweight / impratio (MuJoCo Warp's elliptic D)
        const float f3 = PMUS(s.con_pair[c]);
        const float Dr = D / RMUL(RMUL(ELL_ISQ, ELL_ISQ), RMUL(mu, mu) / RMUL(f3, f3));  // the torsional row's
#pragma unroll
        for (int t = 0; t < 4; t++) {
          if (t >= nr) break;
          s.efc_kind[r0 + t] = tri ? 4 : 3; s.efc_a[r0 + t] = c; s.efc_b[r0 + t] = t;
          s.efc_D[r0 + t] = t == 3 ? Dr : (t ? Dt : D);
          s.efc_aref[r0 + t] = t ? -bb * s.con_dot[c][t] : -k * imp * pos - bb * s.con_dot[c][0];
        }
"""

LS_OLD = None

LS_NEW = r"""    constexpr int RT = (NEFC + 31) / 32;
    const int nefc = s.nefc;
    float rx[RT], rj[RT], rD[RT], rF[RT];
    unsigned ellm = 0u;  // the lane's elliptic rows (kind 4): their cost per contact (EllLs) below
#pragma unroll
    for (int t = 0; t < RT; t++) {
      int r = lane + 32 * t;
      bool v = r < nefc;
      rx[t] = v ? s.efc_jar[r] : 0.f; rj[t] = v ? s.efc_jv[r] : 0.f; rD[t] = v ? s.efc_D[r] : 0.f;
      rF[t] = row_floss(s, r);
      if (v && s.efc_kind[r] == 4) ellm |= 1u << t;
    }
    const int ncon_e = min(s.ncon, MAXCON);
    EllLs el[2];  // the lane's contacts lane, lane + 32
#pragma unroll
    for (int h2 = 0; h2 < 2; h2++) ell_ls_load(s, lane + 32 * h2, ncon_e, el[h2]);
    float alpha = 0.f, lo = 0.f, hi = 1e30f, d0 = 0.f;
    unsigned long long zprev = 0ull;  // the rows' pieces, 2 bits each (up to 32 per lane)
    unsigned zeprev = 0u;  // the contacts' zones
    bool newton = false;
    for (int k = 0; k <= 40; k++) {
      float fd = 0.f, fdd = 0.f;
      unsigned long long z = 0ull;
      bool cone = false;
#pragma unroll
      for (int t = 0; t < RT; t++) {
        if (32 * t >= nefc) break;
        if ((ellm >> t) & 1u) continue;
        float ds, dd, jv = rj[t];
        z |= (unsigned long long)row_piece(lane + 32 * t, rx[t] + alpha * jv, rD[t], rF[t], &ds, &dd) << (2 * t);
        fd += ds * jv; fdd += dd * jv * jv;
      }
      unsigned ze = 0u;
#pragma unroll
      for (int h2 = 0; h2 < 2; h2++) {
        float e1, e2, ec;
        const int zz = ell_ls_eval(el[h2], alpha, &e1, &e2, &ec);
        ze |= (unsigned)zz << (2 * h2);
        cone |= zz == 2;
        fd += e1; fdd += e2;
      }
      if (__all_sync(0xffffffffu, newton && z == zprev && ze == zeprev && !cone)) break;  // every row / contact kept
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
      zprev = z; zeprev = ze;
      alpha = nx;
    }
    // move along p: qacc, M qacc and J qacc - aref are linear in alpha; with the new cost (as total_cost) and the rows'
    // forces for newton_grad (the elliptic rows' per contact, after their rows' jar are all written)
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
        s.efc_jar[r] = xn;
        if ((ellm >> t) & 1u) continue;
        int z0 = row_piece(r, rx[t], rD[t], rF[t], &ds, &dd), z1 = row_piece(r, xn, rD[t], rF[t], &ds, &dd);
        chg |= z0 != z1;
        g += row_cost(z1, xn, rD[t], rF[t]);
        s.efc_jv[r] = -ds;  // the line search's J p is not needed any more
      }
    }
    __syncwarp();
    bool econe = false;
#pragma unroll
    for (int h2 = 0; h2 < 2; h2++) {
      float e1, e2, ec0, ec;
      const int z0 = ell_ls_eval(el[h2], 0.f, &e1, &e2, &ec0), z1 = ell_ls_eval(el[h2], alpha, &e1, &e2, &ec);
      chg |= z0 != z1;
      econe |= z1 == 2;
      g += ec;
      ell_forces(s, lane + 32 * h2, ncon_e);
    }
    zsame = !__any_sync(0xffffffffu, chg || econe);  // then the next H is this one (a cone contact's changes with jar)
"""


def hooks(m, step_text):
    """(anchor, replacement) edits of the composed forward + adjoint text for an elliptic model (step_text: step.py's
    STEP_DEVICE + STEP_KERNEL, the line search's anchor); () otherwise."""
    if not applies(m):
        return ()
    ok, why = supported(m)
    if not ok:
        raise ValueError(f"elliptic cone: {why}")
    i = step_text.index("    constexpr int RT = (NEFC + 31) / 32;\n    const int nefc = s.nefc;\n")
    j = step_text.index("    zsame = !__any_sync(0xffffffffu, chg);  // then the next H is this one\n", i)
    ls_old = step_text[i:j] + "    zsame = !__any_sync(0xffffffffu, chg);  // then the next H is this one\n"
    v = isq(m)
    c4 = bool((np.asarray(m.pair_dim) == 4).any())
    nt, nc = (4, 10) if c4 else (3, 6)
    return (
        ("#define NPAIR ", f"#define ELLIPTIC 1  // elliptic friction cones (g1_engine.elliptic)\n"
                           f"#define ELL_ISQ {float(v)!r}f  // MuJoCo Warp's impratio_invsqrt (float32)\n"
                           f"#define ELL_NT {nt}  // frame rows per contact (4: condim 4)\n"
                           f"#define ELL_NC {nc}  // Hessian coefficients per contact\n"
                           f"#define NPAIR "),
        ("    float con_C[MAXCON][5];    // Hessian coefficients on (n n, n t1, t1 t1, n t2, t2 t2), see the Newton loop\n",
         f"    float con_C[MAXCON][{nc}];    // Hessian coefficients (elliptic: ell_coef's order)\n"),
        ("    float con_C[NC][5];  // the Hessian assembly only",
         f"    float con_C[NC][{nc}];  // (elliptic: ell_coef's coefficients) the Hessian assembly only"),
        ("// row cost derivative s'(x), curvature (D on the quadratic zone, else 0) and cost s(x)\n",
         DEVICE + "// row cost derivative s'(x), curvature (D on the quadratic zone, else 0) and cost s(x)\n"),
        ("  float D = s.efc_D[r], F = row_floss(s, r);\n  *cost = row_cost(row_piece(r, x, D, F, ds, dd), x, D, F);\n",
         "  if (s.efc_kind[r] == 4) {  // elliptic: per contact (at the rows' jar in s)\n"
         "    float f;\n    ell_row(s, r, &f, cost);\n    *ds = -f; *dd = 0.f;\n    return;\n  }\n"
         "  float D = s.efc_D[r], F = row_floss(s, r);\n  *cost = row_cost(row_piece(r, x, D, F, ds, dd), x, D, F);\n"),
        ("  float mu = s.con_dim[c] == 3 ? s.con_mu[c] : 0.f;\n  g[0] = f0 + f1 + f2 + f3; g[1] = mu * (f0 - f1); "
         "g[2] = mu * (f2 - f3);\n",
         ("  g[0] = f0; g[1] = f1; g[2] = f2; g[3] = nr > 3 ? fr[3] : 0.f;  // elliptic rows: one per frame direction\n"
          if c4 else "  (void)f3;\n  g[0] = f0; g[1] = f1; g[2] = f2;  // elliptic rows: one per frame direction\n")),
        ("    } else val = s.con_dot[a][0];\n",
         "    } else if (kind == 4) val = s.con_dot[a][b];  // elliptic: the frame row\n    else val = s.con_dot[a][0];\n"),
        ("  float ds, dd;\n  return row_piece(r, x, s.efc_D[r], row_floss(s, r), &ds, &dd);\n",
         "  if (s.efc_kind[r] == 4) {  // elliptic (tape bits): normal row = bottom zone, first tangent row = cone zone\n"
         "    float f, c;\n    const int z = ell_row(s, r, &f, &c), t = s.efc_b[r];\n"
         "    return (t == 0 && z == 1) || (t == 1 && z == 2) ? 1 : 0;\n  }\n"
         "  float ds, dd;\n  return row_piece(r, x, s.efc_D[r], row_floss(s, r), &ds, &dd);\n"),
        ("    int r0 = ndof + c + 3 * (ntri + __popc(b3 & below));\n",
         "    int r0 = ndof + c + 2 * (ntri + __popc(b3 & below));  // (elliptic: 3 rows per condim-3 contact)\n"),
        ("    int need = ndof + ncon + 3 * ntri;\n", "    int need = ndof + ncon + 2 * ntri;\n"),
        ("        iw = contact_invw(pr, mu);\n",
         "        iw = RADD(c_binvw[c_gbody[PG1(pr)]], c_binvw[c_gbody[PG2(pr)]]);  // elliptic: no pyramid scaling\n"),
        ("""        int nr = max(0, min(tri ? 4 : 1, NEFC - r0));  // rows beyond NEFC are dropped
        s.con_row[c] = r0; s.con_nrow[c] = nr;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          if (t >= nr) break;
          float sg = (t & 1) ? -1.f : 1.f;
          float vel = tri ? s.con_dot[c][0] + sg * mu * s.con_dot[c][1 + (t >> 1)] : s.con_dot[c][0];
          s.efc_kind[r0 + t] = tri ? 2 : 3; s.efc_a[r0 + t] = c; s.efc_b[r0 + t] = t;
          s.efc_D[r0 + t] = D; s.efc_aref[r0 + t] = -k * imp * pos - bb * vel;
        }
""",
         ROWS4 if c4 else ROWS3),
        ("""      int r0 = s.con_row[c], nr = s.con_nrow[c];
      float d[4];
#pragma unroll
      for (int k = 0; k < 4; k++) {  // curvatures; the loads read a valid row either way (con_row is set only when
        int r = nr > k ? r0 + k : 0;  // nr > 0), so there are no branches
        float jar = s.efc_jar[r], D = s.efc_D[r];
        d[k] = nr > k && jar < 0.f ? D : 0.f;
      }
      float d0 = d[0], d1 = d[1], d2 = d[2], d3 = d[3], mu = s.con_dim[c] == 3 ? s.con_mu[c] : 0.f;
      s.con_C[c][0] = d0 + d1 + d2 + d3;
      s.con_C[c][1] = mu * (d0 - d1); s.con_C[c][2] = mu * mu * (d0 + d1);
      s.con_C[c][3] = mu * (d2 - d3); s.con_C[c][4] = mu * mu * (d2 + d3);
      on = d0 + d1 + d2 + d3 != 0.f;  // C_nn is the sum of the curvatures
""",
         """      on = ell_coef(s, c, s.con_C[c]);  // elliptic: the contact's block (g1_engine.elliptic)
"""),
        ("        float g0 = C[0] * j0 + C[1] * j1 + C[3] * j2, g1 = C[1] * j0 + C[2] * j1, g2 = C[3] * j0 + C[4] * j2;\n",
         ("        const float* Jr = s.con_J[cc][3];  // condim 4: the torsional row's couplings (C[6..9])\n"
          "        const float j3 = Jr[b];\n"
          "        float g0 = C[0] * j0 + C[1] * j1 + C[3] * j2 + C[6] * j3, g1 = C[1] * j0 + C[2] * j1 + C[5] * j2 + C[8] * j3,\n"
          "              g2 = C[3] * j0 + C[4] * j2 + C[5] * j1 + C[9] * j3, g3 = C[6] * j0 + C[8] * j1 + C[9] * j2 + C[7] * j3;\n")
         if c4 else
         ("        float g0 = C[0] * j0 + C[1] * j1 + C[3] * j2, g1 = C[1] * j0 + C[2] * j1 + C[5] * j2,\n"
          "              g2 = C[3] * j0 + C[4] * j2 + C[5] * j1;  // (elliptic: the cone's t1 t2 coupling)\n")),
        (ls_old, LS_NEW),
        ("        refact = zchanged;\n",
         "        refact = zchanged || ell_any_cone(s, lane);  // (a cone contact's curvature: re-factor at the final point)\n"),
    )


ADJ_DEVICE = r"""
// the elliptic contact's Psi_c VJP (contact_psi_vjp's contract; fric: the friction coefficient, zone: the tape's bits
// of the contact's rows, bit 0 = bottom zone, bit 1 = cone zone; top zone: zone 0, everything 0):
// Psi_c = -sum_k f_k(x) l_k over the frame rows k (l_k = fr_k . V_lam), x_k = fr_k . V_a + b fr_k . V_v (+ k imp d on
// the normal row), f the forward's forces (ell_row); dPsi_c / dx = C l with C the contact's Hessian block (ell_coef):
// the bottom zone's diagonal D (tangent rows D / impratio), the cone's block
__device__ __forceinline__ void contact_psi_vjp_ell(const float* Vv, const float* Va, const float* Vl, const float* fr,
                                                    float fric, int dim, unsigned zone, float d, float D, float dD,
                                                    float k, float b, float imp, float dimp, float* Wv, float* Wa,
                                                    float* Wl, float* dbar, float* frbar) {
  float vf[3], af[3], lf[3];
#pragma unroll
  for (int j = 0; j < 3; j++) { vf[j] = dot3(fr + 3 * j, Vv); af[j] = dot3(fr + 3 * j, Va); lf[j] = dot3(fr + 3 * j, Vl); }
  const float g = k * imp * d, gd = k * (imp + dimp * d);
  const float x[3] = {af[0] + g + b * vf[0], af[1] + b * vf[1], af[2] + b * vf[2]};
  float cl[3] = {0.f, 0.f, 0.f}, ca[3] = {0.f, 0.f, 0.f}, db = 0.f;  // -f and C l (frame components)
  if (dim != 3) {  // condim 1: the frictionless normal row
    if (zone & 1u) { cl[0] = D * x[0]; ca[0] = D * lf[0]; db = lf[0] * (dD * x[0] + D * gd); }
  } else if (zone & 1u) {  // bottom zone: every row quadratic
    const float s2 = RMUL(ELL_ISQ, ELL_ISQ), Dt = D / s2, dDt = dD / s2;
    cl[0] = D * x[0]; cl[1] = Dt * x[1]; cl[2] = Dt * x[2];
    ca[0] = D * lf[0]; ca[1] = Dt * lf[1]; ca[2] = Dt * lf[2];
    db = lf[0] * (dD * x[0] + D * gd) + dDt * (lf[1] * x[1] + lf[2] * x[2]);
  } else if (zone & 2u) {  // cone zone
    const float mu = fric * ELL_ISQ, mu2 = mu * mu, dm = D / (mu2 * (1.f + mu2));
    const float N = x[0] * mu, u1 = x[1] * fric, u2 = x[2] * fric, T = sqrtf(u1 * u1 + u2 * u2);
    if (T > 0.f && dm != 0.f) {
      const float r = N - mu * T, sv = dm * r * mu, w1 = u1 / T, w2 = u2 / T;  // f_0 = -sv, f_j = sv fric w_j
      cl[0] = sv; cl[1] = -sv * fric * w1; cl[2] = -sv * fric * w2;
      const float tt = fmaxf(T, 1e-15f), v1 = u1 / tt, v2 = u2 / tt, mnt = mu * N / tt, td = mu2 - mnt, ff = fric * fric;
      const float C00 = dm * mu2, C01 = -dm * mu2 * fric * v1, C02 = -dm * mu2 * fric * v2;
      const float C11 = dm * ff * (mnt * v1 * v1 + td), C22 = dm * ff * (mnt * v2 * v2 + td), C12 = dm * ff * mnt * v1 * v2;
      ca[0] = C00 * lf[0] + C01 * lf[1] + C02 * lf[2];
      ca[1] = C01 * lf[0] + C11 * lf[1] + C12 * lf[2];
      ca[2] = C02 * lf[0] + C12 * lf[1] + C22 * lf[2];
      const float psi = cl[0] * lf[0] + cl[1] * lf[1] + cl[2] * lf[2];
      db = ca[0] * gd + psi * dD / D;  // (Psi_c is proportional to D through dm)
    }
  }
#pragma unroll
  for (int i = 0; i < 3; i++) {
    Wl[i] = cl[0] * fr[i] + cl[1] * fr[3 + i] + cl[2] * fr[6 + i];
    Wa[i] = ca[0] * fr[i] + ca[1] * fr[3 + i] + ca[2] * fr[6 + i];
    Wv[i] = b * Wa[i];
  }
  const float ab[3] = {Va[0] + b * Vv[0], Va[1] + b * Vv[1], Va[2] + b * Vv[2]};
#pragma unroll
  for (int j = 0; j < 3; j++)
#pragma unroll
    for (int i = 0; i < 3; i++) frbar[3 * j + i] = cl[j] * Vl[i] + ca[j] * ab[i];
  *dbar = db;
}

"""

ADJ_DEVICE4 = r"""
// contact_psi_vjp_ell with condim 4 (g1_engine.condim4): the torsional row 3 on the contact's relative angular motions
// Om (x_3 = n . Om_a + b n . Om_v, l_3 = n . Om_lam; scale f3 = friction[2], D_3 = D / (impratio^-1 f0^2 / f3^2)); its
// weights on the angular columns AW = (aw n) and n's cotangent in frbar
__device__ __forceinline__ void contact_psi_vjp_ell4(const float* Vv, const float* Va, const float* Vl, const float* fr,
                                                     float fric, int dim, unsigned zone, float d, float D, float dD,
                                                     float k, float b, float imp, float dimp, float* Wv, float* Wa,
                                                     float* Wl, float* dbar, float* frbar, const float* Ov,
                                                     const float* Oa, const float* Ol, float f3, float* AWv, float* AWa,
                                                     float* AWl) {
  float vf[4], af[4], lf[4];
#pragma unroll
  for (int j = 0; j < 3; j++) { vf[j] = dot3(fr + 3 * j, Vv); af[j] = dot3(fr + 3 * j, Va); lf[j] = dot3(fr + 3 * j, Vl); }
  vf[3] = dot3(fr, Ov); af[3] = dot3(fr, Oa); lf[3] = dot3(fr, Ol);
  const float g = k * imp * d, gd = k * (imp + dimp * d);
  const float x[4] = {af[0] + g + b * vf[0], af[1] + b * vf[1], af[2] + b * vf[2], af[3] + b * vf[3]};
  float cl[4] = {0.f, 0.f, 0.f, 0.f}, ca[4] = {0.f, 0.f, 0.f, 0.f}, db = 0.f;  // -f and C l (frame components)
  const bool tor = dim == 4;
  if (dim < 3) {  // condim 1: the frictionless normal row
    if (zone & 1u) { cl[0] = D * x[0]; ca[0] = D * lf[0]; db = lf[0] * (dD * x[0] + D * gd); }
  } else if (zone & 1u) {  // bottom zone: every row quadratic
    const float s2 = RMUL(ELL_ISQ, ELL_ISQ), Dt = D / s2, dDt = dD / s2;
    cl[0] = D * x[0]; cl[1] = Dt * x[1]; cl[2] = Dt * x[2];
    ca[0] = D * lf[0]; ca[1] = Dt * lf[1]; ca[2] = Dt * lf[2];
    db = lf[0] * (dD * x[0] + D * gd) + dDt * (lf[1] * x[1] + lf[2] * x[2]);
    if (tor) {
      const float sr = RMUL(s2, RMUL(fric, fric) / RMUL(f3, f3)), Dr = D / sr, dDr = dD / sr;
      cl[3] = Dr * x[3]; ca[3] = Dr * lf[3];
      db += dDr * lf[3] * x[3];
    }
  } else if (zone & 2u) {  // cone zone
    const float mu = fric * ELL_ISQ, mu2 = mu * mu, dm = D / (mu2 * (1.f + mu2));
    const float N = x[0] * mu, u1 = x[1] * fric, u2 = x[2] * fric, u3 = tor ? x[3] * f3 : 0.f;
    const float T = sqrtf(u1 * u1 + u2 * u2 + u3 * u3);
    if (T > 0.f && dm != 0.f) {
      const float r = N - mu * T, sv = dm * r * mu, w1 = u1 / T, w2 = u2 / T, w3 = u3 / T;
      cl[0] = sv; cl[1] = -sv * fric * w1; cl[2] = -sv * fric * w2; cl[3] = -sv * f3 * w3;
      const float tt = fmaxf(T, 1e-15f), v1 = u1 / tt, v2 = u2 / tt, v3 = u3 / tt, mnt = mu * N / tt, td = mu2 - mnt;
      const float ff = fric * fric;
      const float C00 = dm * mu2, C01 = -dm * mu2 * fric * v1, C02 = -dm * mu2 * fric * v2, C03 = -dm * mu2 * f3 * v3;
      const float C11 = dm * ff * (mnt * v1 * v1 + td), C22 = dm * ff * (mnt * v2 * v2 + td), C12 = dm * ff * mnt * v1 * v2;
      const float C33 = tor ? dm * f3 * f3 * (mnt * v3 * v3 + td) : 0.f, C13 = dm * fric * f3 * mnt * v1 * v3,
                  C23 = dm * fric * f3 * mnt * v2 * v3;
      ca[0] = C00 * lf[0] + C01 * lf[1] + C02 * lf[2] + C03 * lf[3];
      ca[1] = C01 * lf[0] + C11 * lf[1] + C12 * lf[2] + C13 * lf[3];
      ca[2] = C02 * lf[0] + C12 * lf[1] + C22 * lf[2] + C23 * lf[3];
      ca[3] = C03 * lf[0] + C13 * lf[1] + C23 * lf[2] + C33 * lf[3];
      const float psi = cl[0] * lf[0] + cl[1] * lf[1] + cl[2] * lf[2] + cl[3] * lf[3];
      db = ca[0] * gd + psi * dD / D;  // (Psi_c is proportional to D through dm)
    }
  }
  if (!tor) { cl[3] = 0.f; ca[3] = 0.f; }
#pragma unroll
  for (int i = 0; i < 3; i++) {
    Wl[i] = cl[0] * fr[i] + cl[1] * fr[3 + i] + cl[2] * fr[6 + i];
    Wa[i] = ca[0] * fr[i] + ca[1] * fr[3 + i] + ca[2] * fr[6 + i];
    Wv[i] = b * Wa[i];
    AWl[i] = cl[3] * fr[i]; AWa[i] = ca[3] * fr[i]; AWv[i] = b * AWa[i];
  }
  const float ab[3] = {Va[0] + b * Vv[0], Va[1] + b * Vv[1], Va[2] + b * Vv[2]};
#pragma unroll
  for (int j = 0; j < 3; j++)
#pragma unroll
    for (int i = 0; i < 3; i++) frbar[3 * j + i] = cl[j] * Vl[i] + ca[j] * ab[i];
#pragma unroll
  for (int i = 0; i < 3; i++) frbar[i] += cl[3] * Ol[i] + ca[3] * (Oa[i] + b * Ov[i]);
  *dbar = db;
}

"""


def adj_hooks(m):
    """The adjoint's edits for an elliptic model (adjoint.cuda_source, with hooks()); () otherwise."""
    if not applies(m):
        return ()
    c4 = bool((np.asarray(m.pair_dim) == 4).any())
    call = ("  float AWv[3], AWa[3], AWl[3];  // (condim 4: the torsional row's angular weights, condim4's cs words)\n"
            "  contact_psi_vjp_ell4(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4,"
            " Ov, Oa, Ol, dim == 4 ? PMUS(pair) : 0.f, AWv, AWa, AWl);\n" if c4 else
            "  contact_psi_vjp_ell(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4);\n")
    return (
        ("__device__ __forceinline__ float adj_contact_one(Adj& s, int c, int pair, int slot, unsigned zone, float* cs,\n",
         (ADJ_DEVICE4 if c4 else ADJ_DEVICE) + "__device__ __forceinline__ float adj_contact_one(Adj& s, int c, int pair,"
         " int slot, unsigned zone, float* cs,\n"),
        ("  adj_row_impedance(PSOLREF(pair), PSOLIMP(pair), d, contact_invw(pair, mu), ri);\n",
         "  adj_row_impedance(PSOLREF(pair), PSOLIMP(pair), d, RADD(c_binvw[b1], c_binvw[b2]), ri);  // (elliptic invweight)\n"),
        ("  contact_psi_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4);\n",
         call),
        ("    const int r0 = nlim + c + 3 * (ntri + __popc(b3 & below));  // the contact's first row\n",
         "    const int r0 = nlim + c + 2 * (ntri + __popc(b3 & below));  // the contact's first row (elliptic: 3 rows)\n"),
        ("      const int nr = min(dim == 3 ? 4 : 1, NEFC - r0), w = r0 >> 5;\n",
         "      const int nr = min(dim == 3 ? 3 : 1, NEFC - r0), w = r0 >> 5;\n"),
    )

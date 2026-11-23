"""Backward-only contact-gradient surrogates of the forest engine's adjoint (an opt-in build: init(contact_surrogate=True))."""

ADJ_PARAMS = "  float sur_mu, sur_clip;  // the surrogates' run-time parameters (cone_mu, impulse_clip; surrogate.py)\n"

ADJ_STRUCT = (
    "  float u[NU], soft;  // the world's ctrl and delta (clamp_soft) for lam_soft (120 B: still 9 worlds per SM)\n#endif\n};",
    "  float u[NU], soft;  // the world's ctrl and delta (clamp_soft) for lam_soft (120 B: still 9 worlds per SM)\n#endif\n"
    + ADJ_PARAMS +
    "  int sur_on;  // cone_mu > 0 or impulse_clip finite\n"
    "  float sgx[NGEOMC][3], sgq[NGEOMC][4];  // adj_sur_hessian: the collision geoms' poses\n"
    "  float sjr[6][NV];  // adj_sur_hessian: the rows J_e of one contact (pyramid rows 0-3, spin rows 4-5)\n"
    "};")

DEVICE = r"""
// ---------------------------------------------------------------- contact-gradient surrogates (surrogate.py)
#define DK_ADJ_SURR 1
// the central-path slope weight of a row at x = J a* - aref: w = (1 - x / sqrt(x^2 + 4 mu)) / 2
__device__ __forceinline__ float sur_weight(float x, float mu) { return 0.5f * (1.f - x / sqrtf(x * x + 4.f * mu)); }

// contact_psi_vjp with the surrogates' row weights: the force-value terms (W_lam, the frame through u_e . V_lam,
// dD x) take the zone bit z_e, the slope terms (W_a, W_v, the frame through x_e, D dx / dd) w_e (smu > 0: sur_weight;
// smu = 0: z_e, the exact partials). Returns |F|, the contact's force vector's norm from the forward's rows
// (sum over the rows on their zone of -D x_e u_e)
__device__ __forceinline__ float contact_psi_vjp_sur(const float* Vv, const float* Va, const float* Vl, const float* fr,
                                                     float mu, int dim, unsigned zone, float d, float D, float dD,
                                                     float k, float b, float imp, float dimp, float* Wv, float* Wa,
                                                     float* Wl, float* dbar, float* frbar, float smu
#if HAS_CONDIM4
                                                     , const float* Ov, const float* Oa, const float* Ol, float mus,
                                                     float* AWv, float* AWa, float* AWl
#endif
                                                     ) {
  float vf[3], af[3], lf[3];
  for (int j = 0; j < 3; j++) { vf[j] = dot3(fr + 3 * j, Vv); af[j] = dot3(fr + 3 * j, Va); lf[j] = dot3(fr + 3 * j, Vl); }
  const float g = k * imp * d, gd = k * (imp + dimp * d);
  float cl[3] = {0.f, 0.f, 0.f}, ca[3] = {0.f, 0.f, 0.f}, db = 0.f, F[3] = {0.f, 0.f, 0.f};
  for (int e = 0; e < 4; e++) {
    if (e > 0 && dim < 3) continue;
    const int t = 1 + (e >> 1);
    const float sm = (e & 1) ? -mu : mu;
    float uv = vf[0], ua = af[0], ul = lf[0];
    if (dim >= 3) { uv += sm * vf[t]; ua += sm * af[t]; ul += sm * lf[t]; }
    const float x = ua + g + b * uv, zf = ((zone >> e) & 1u) ? 1.f : 0.f, wz = smu > 0.f ? sur_weight(x, smu) : zf;
    const float Dx = D * x, Dl = D * ul;
    cl[0] += zf * Dx; ca[0] += wz * Dl; F[0] -= zf * Dx;
    if (dim >= 3) { cl[t] += sm * (zf * Dx); ca[t] += sm * (wz * Dl); F[t] -= sm * (zf * Dx); }
    db += ul * (zf * (dD * x) + wz * (D * gd));
  }
#if HAS_CONDIM4
  float awl = 0.f, awa = 0.f;
  if (dim == 4) {
    const float wv = dot3(fr, Ov), wa = dot3(fr, Oa), wl = dot3(fr, Ol);
    for (int e = 4; e < 6; e++) {
      const float sm = (e & 1) ? -mus : mus;
      const float uv = vf[0] + sm * wv, ua = af[0] + sm * wa, ul = lf[0] + sm * wl;
      const float x = ua + g + b * uv, zf = ((zone >> e) & 1u) ? 1.f : 0.f, wz = smu > 0.f ? sur_weight(x, smu) : zf;
      const float Dx = D * x, Dl = D * ul;
      cl[0] += zf * Dx; ca[0] += wz * Dl; F[0] -= zf * Dx;
      awl += sm * (zf * Dx); awa += sm * (wz * Dl);
      db += ul * (zf * (dD * x) + wz * (D * gd));
    }
  }
  for (int i = 0; i < 3; i++) { AWl[i] = awl * fr[i]; AWa[i] = awa * fr[i]; AWv[i] = b * AWa[i]; }
#endif
  for (int i = 0; i < 3; i++) {
    Wl[i] = cl[0] * fr[i] + cl[1] * fr[3 + i] + cl[2] * fr[6 + i];
    Wa[i] = ca[0] * fr[i] + ca[1] * fr[3 + i] + ca[2] * fr[6 + i];
    Wv[i] = b * Wa[i];
  }
  const float ab[3] = {Va[0] + b * Vv[0], Va[1] + b * Vv[1], Va[2] + b * Vv[2]};
  for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) frbar[3 * j + i] = cl[j] * Vl[i] + ca[j] * ab[i];
#if HAS_CONDIM4
  for (int i = 0; i < 3; i++) frbar[i] += awl * Ol[i] + awa * (Oa[i] + b * Ov[i]);
#endif
  *dbar = db;
  return sqrtf(F[0] * F[0] + F[1] * F[1] + F[2] * F[2]);  // (the frame is orthonormal)
}
"""

PSI_CALL4 = (
    "  contact_psi_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4, Ov, Oa, Ol,\n"
    "                  dim == 4 ? PMUS(pair) : 0.f, AWv, AWa, AWl);\n")
PSI_CALL3 = "  contact_psi_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4);\n"
SUR_CALL = r"""  if (s.sur_on) {  // the surrogates (surrogate.py): relaxed partials, then the impulse clip
    const float fm = contact_psi_vjp_sur(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot,
                                         cot + 4, s.sur_mu@ARGS4@);
    const float imp_c = c_misc[0] * fm;  // the contact's impulse over the substep, h |F|
    if (imp_c > s.sur_clip) {
      const float kap = s.sur_clip / imp_c;
      for (int t = 0; t < 3; t++) { Wv[t] *= kap; Wa[t] *= kap; Wl[t] *= kap; }
      cot[0] *= kap;
      for (int t = 4; t < 13; t++) cot[t] *= kap;
@CLIP4@    }
  } else {
@CALL@  }
"""
ARGS4 = ", Ov, Oa, Ol, dim == 4 ? PMUS(pair) : 0.f, AWv, AWa, AWl"
CLIP4 = "      for (int t = 0; t < 3; t++) { AWv[t] *= kap; AWa[t] *= kap; AWl[t] *= kap; }\n"


def _sur_call(call, cond4):
    body = "".join("  " + line + "\n" for line in call.rstrip("\n").split("\n"))
    return (SUR_CALL.replace("@ARGS4@", ARGS4 if cond4 else "").replace("@CLIP4@", CLIP4 if cond4 else "")
            .replace("@CALL@", body))


PRIMITIVE = ("  if (zone) {  // the primitive's VJP (it replays again) and the geoms' pose cotangents on their bodies\n",
             "  if (zone || s.sur_mu > 0.f) {  // the primitive's VJP (it replays again) and the geoms' pose cotangents on "
             "their bodies (surrogate: also contacts without a quadratic row)\n")
CONTACTS_USE = (
    "    if (use && (zone || dbg)) {  // a contact without a quadratic row adds nothing (MuJoCo Warp's _contact_any_active)\n"
    "      bad = !(adj_contact_one(s, c, pair, slot, zone, cw, dbg) < 0.f);\n"
    "      if (zone) cb = c_gbody[PG1(pair)] | c_gbody[PG2(pair)] << 8;\n"
    "    }\n",
    "    if (use && (zone || dbg || s.sur_mu > 0.f)) {  // a contact without a quadratic row adds nothing (MuJoCo Warp's\n"
    "      // _contact_any_active), except with the cone relaxation (surrogate.py: every row has a slope weight)\n"
    "      const float d_ = adj_contact_one(s, c, pair, slot, zone, cw, dbg);\n"
    "      bad = (zone || dbg) && !(d_ < 0.f);  // (the replay check: contacts with a quadratic row, as without it)\n"
    "      if (zone || s.sur_mu > 0.f) cb = c_gbody[PG1(pair)] | c_gbody[PG2(pair)] << 8;\n"
    "    }\n")

DECL_ANCHOR = "__device__ __noinline__ void adj_solves(Adj& s, const float* rec, int lane) {\n"
DECL = ("#if DK_ADJ_STAGE >= 2 && DK_ADJ_CONTACTS\n__device__ __noinline__ void adj_sur_hessian(Adj& s, int lane);  "
        "// (surrogate.py)\n#endif\n")
SOLVE = ("  for (int e = lane; e < NTRI; e += 32) s.L[e] = __ldcs(rec + OFF_LH + e);\n"
         "  for (int i = lane; i < NV; i += 32) s.lam[i] = s.abar[i];\n"
         "  tri_solve_packed(s.L, s.lam, lane);\n}\n")
SOLVE_SUR = ("  for (int e = lane; e < NTRI; e += 32) s.L[e] = __ldcs(rec + OFF_LH + e);\n"
             "  for (int i = lane; i < NV; i += 32) s.lam[i] = s.abar[i];\n"
             "#if DK_ADJ_STAGE >= 2 && DK_ADJ_CONTACTS\n"
             "  if (s.sur_mu > 0.f) {  // the cone relaxation: lambda = H_mu^-1 a* bar (surrogate.py)\n"
             "    __syncwarp();\n"
             "    adj_sur_hessian(s, lane);\n"
             "    chol_solve_compact(s.L, s.L, s.lam, lane);\n"
             "    return;\n"
             "  }\n"
             "#endif\n"
             "  tri_solve_packed(s.L, s.lam, lane);\n}\n")

HESSIAN_ANCHOR = "\n#ifdef DK_ADJ_DEBUG\n__device__ void dbg_put(float* dst, const float* src, int n, float sg, int lane) {\n"
HESSIAN = r"""
#if DK_ADJ_STAGE >= 2 && DK_ADJ_CONTACTS
// the cone relaxation's row coefficients of one contact (one lane per contact): replay (the forward's dist, point,
// frame), impedance and x_e as adj_contact_one; cw = p (0-2), frame (3-11), c_e = (w_e - z_e) D_e of rows 0-5 (12-17;
// 0 for rows past nr)
__device__ __forceinline__ void sur_contact_coef(const Adj& s, int pair, int slot, unsigned zone, int nr, float* cw) {
  const int g1 = PG1(pair), g2 = PG2(pair), b1 = c_gbody[g1], b2 = c_gbody[g2], dim = PDIM(pair);
  float d, p[3], fr[9], nu;
  adj_pair_geom(pair, slot, s.sgx, s.sgq, &d, p, fr, &nu, nullptr, nullptr@MVID@);
  const float mu = PFOOT(pair) ? s.fric : PMU(pair);
  float ri[6];
  adj_row_impedance(PSOLREF(pair), PSOLIMP(pair), d, contact_invw(pair, mu), ri);
  const float k = ri[0], b = ri[1], imp = ri[2], D = ri[4];
  float Vv[3] = {0.f, 0.f, 0.f}, Va[3] = {0.f, 0.f, 0.f}, Ov[3] = {0.f, 0.f, 0.f}, Oa[3] = {0.f, 0.f, 0.f};
  const unsigned m1 = c_ancmask[b1], m2 = c_ancmask[b2];
  unsigned hb = (m1 ^ m2) & ~1u;
#pragma unroll 1
  while (hb) {
    const int bi = __ffs(hb) - 1;
    hb &= hb - 1u;
    const float sg = (m2 >> bi) & 1u ? 1.f : -1.f;
    const int jt = JT_OF(bi), d0 = c_dadr[bi];
    float r[3];
    for (int t = 0; t < 3; t++) r[t] = p[t] - s.X[bi][t];
    if (jt == JFREE)
      for (int t = 0; t < 3; t++) { Vv[t] += sg * s.v[d0 + t]; Va[t] += sg * s.a[d0 + t]; }
    const int nrot = jt == JFREE ? 3 : 1;
    for (int e = 0; e < nrot; e++) {
      const int i = jt == JFREE ? d0 + 3 + e : d0;
      const float vi = sg * s.v[i], ai = sg * s.a[i];
      float z[3], col[3];
      if (jt == JFREE) {
        const float ek[3] = {e == 0 ? 1.f : 0.f, e == 1 ? 1.f : 0.f, e == 2 ? 1.f : 0.f};
        qrot(s.Q[bi], ek, z);
      } else {
        qrot(s.Q[bi], c_axis + 3 * bi, z);
      }
      if (jt == JSLIDE) {
        for (int t = 0; t < 3; t++) { Vv[t] += vi * z[t]; Va[t] += ai * z[t]; }
        continue;
      }
      cross(z, r, col);
      for (int t = 0; t < 3; t++) { Vv[t] += vi * col[t]; Va[t] += ai * col[t]; Ov[t] += vi * z[t]; Oa[t] += ai * z[t]; }
    }
  }
  float vf[3], af[3];
  for (int j = 0; j < 3; j++) { vf[j] = dot3(fr + 3 * j, Vv); af[j] = dot3(fr + 3 * j, Va); }
  const float g = k * imp * d;
  for (int t = 0; t < 3; t++) cw[t] = p[t];
  for (int t = 0; t < 9; t++) cw[3 + t] = fr[t];
  for (int e = 0; e < 6; e++) cw[12 + e] = 0.f;
  for (int e = 0; e < 4 && e < nr; e++) {
    if (e > 0 && dim < 3) break;
    const int t = 1 + (e >> 1);
    const float sm = (e & 1) ? -mu : mu;
    float uv = vf[0], ua = af[0];
    if (dim >= 3) { uv += sm * vf[t]; ua += sm * af[t]; }
    const float x = ua + g + b * uv, zf = ((zone >> e) & 1u) ? 1.f : 0.f;
    cw[12 + e] = (sur_weight(x, s.sur_mu) - zf) * D;
  }
#if HAS_CONDIM4
  if (dim == 4) {
    const float wv = dot3(fr, Ov), wa = dot3(fr, Oa), mus = PMUS(pair);
    for (int e = 4; e < 6 && e < nr; e++) {
      const float sm = (e & 1) ? -mus : mus;
      const float x = af[0] + sm * wa + g + b * (vf[0] + sm * wv), zf = ((zone >> e) & 1u) ? 1.f : 0.f;
      cw[12 + e] = (sur_weight(x, s.sur_mu) - zf) * D;
    }
  }
#endif
  (void)nu;
}

// the cone relaxation's Hessian, in s.L (which holds L_H): H = L_H L_H' in place, then per differentiated contact of
// the tape's table (adj_contacts' selection and row offsets) the rank update sum_e c_e J_e' J_e, c_e = (w_e - z_e) D_e:
// one lane per contact for the coefficients, then per contact the rows J_e on the dof lanes (s.sjr: J_e,i = s_i u_e .
// col_i, spin rows s_i (n . col_i + mus_e n . z_i)) and the update on the lanes of the triangle's columns. Before
// chol_solve_compact (H_mu = M + sum_e w_e D_e J_e' J_e is positive definite)
__device__ __noinline__ void adj_sur_hessian(Adj& s, int lane) {
  LANE_BOUND(lane);
  const int ncon = uniform_int(s.hdr & 255, 8), nlim = ((unsigned)s.hdr >> 18) & 63;
  for (int g = lane; g < NGEOMC; g += 32) {  // geom_poses (step.py), as adj_contacts
    const int b = c_gbody[g];
#if NFIX
    if (c_fixg[g] >= 0) { geom_pose(g, s.fixq[c_fixg[g]], s.fixx[c_fixg[g]], s.sgx[g], s.sgq[g]); continue; }
#endif
    geom_pose(g, s.Q[b], s.X[b], s.sgx[g], s.sgq[g]);
  }
  for (int i = NV - 1; i >= 0; i--) {  // H = L L' in place: row i from rows <= i of L (rows < i still L), cols <= j
    float h[2] = {0.f, 0.f};
    for (int t = 0; t < 2; t++) {
      const int j = lane + 32 * t;
      if (j <= i) {
        float acc = 0.f;
        for (int q = 0; q <= j; q++) acc += s.L[TRI(i, q)] * s.L[TRI(j, q)];
        h[t] = acc;
      }
    }
    __syncwarp();
    for (int t = 0; t < 2; t++) {
      const int j = lane + 32 * t;
      if (j <= i) s.L[TRI(i, j)] = h[t];
    }
    __syncwarp();
  }
  const unsigned below = (1u << lane) - 1u;
  int ntri = 0, nquad = 0;
  for (int c0 = 0; c0 < ncon; c0 += 32) {
    const int c = c0 + lane, nb = min(32, ncon - c0);
    const bool on = c < ncon;
    const unsigned e = on ? s.con[c] : 0u;
    const int pair = e & 0xFFFFu, slot = e >> 16, dim = on ? PDIM(pair) : 0;
    const unsigned b3 = __ballot_sync(0xffffffffu, dim >= 3);
@ROW0@
    ntri += __popc(b3);
#if HAS_CONDIM4
    const unsigned b4 = __ballot_sync(0xffffffffu, dim == 4);
    const int r0x = r0 + 2 * (nquad + __popc(b4 & below));
    nquad += __popc(b4);
#else
    const int r0x = r0;
#endif
    const bool use = on && DK_ADJ_CONTACTS && (DK_ADJ_STAGE >= 3 || c_gbody[PG1(pair)] == 0 || c_gbody[PG2(pair)] == 0)
                     && !PFROZEN(pair);
    float cw[18];
#pragma unroll
    for (int q = 0; q < 18; q++) cw[q] = 0.f;
    int cb = -1;  // body 1 | body 2 << 8 | dim << 16
    if (use && r0x < NEFC) {
      const int nr = min(dim >= 3 ? (dim == 4 ? 6 : 4) : 1, NEFC - r0x), w = r0x >> 5;
      unsigned long long zz = s.zone[w];
      if (w + 1 < (NEFC + 31) / 32) zz |= (unsigned long long)s.zone[w + 1] << 32;
      const unsigned zone = (unsigned)(zz >> (r0x & 31)) & ((1u << nr) - 1u);
      sur_contact_coef(s, pair, slot, zone, nr, cw);
      cb = c_gbody[PG1(pair)] | c_gbody[PG2(pair)] << 8 | dim << 16;
    }
#pragma unroll 1
    for (int j = 0; j < nb; j++) {
      const int cj = __shfl_sync(0xffffffffu, cb, j);
      if (UNIFORM(cj < 0)) continue;
      float w[18];
#pragma unroll
      for (int q = 0; q < 18; q++) w[q] = __shfl_sync(0xffffffffu, cw[q], j);
      const int c1 = cj & 255, c2 = (cj >> 8) & 255, dm = cj >> 16;
      const unsigned m1 = c_ancmask[c1], m2 = c_ancmask[c2];
      for (int i = lane; i < NV; i += 32) {  // the rows J_e on dof i
        const int bi = c_dofbody[i];
        float col[3] = {0.f, 0.f, 0.f}, ang[3] = {0.f, 0.f, 0.f}, sg = 0.f;
        if (((m1 ^ m2) >> bi) & 1u) {
          sg = (m2 >> bi) & 1u ? 1.f : -1.f;
          const int jt = JT_OF(bi), k = i - c_dadr[bi];
          if (jt == JFREE && k < 3) col[k] = 1.f;
          else {
            float z[3], r[3];
            if (jt == JFREE) {
              const float ek[3] = {k == 3 ? 1.f : 0.f, k == 4 ? 1.f : 0.f, k == 5 ? 1.f : 0.f};
              qrot(s.Q[bi], ek, z);
            } else {
              qrot(s.Q[bi], c_axis + 3 * bi, z);
            }
            if (jt == JSLIDE) {
              for (int t = 0; t < 3; t++) col[t] = z[t];
            } else {
              for (int t = 0; t < 3; t++) { r[t] = w[t] - s.X[bi][t]; ang[t] = z[t]; }
              cross(z, r, col);
            }
          }
        }
        const float cn = w[3] * col[0] + w[4] * col[1] + w[5] * col[2];
        const float c1t = w[6] * col[0] + w[7] * col[1] + w[8] * col[2];
        const float c2t = w[9] * col[0] + w[10] * col[1] + w[11] * col[2];
        if (dm >= 3) {
          // n, t1, t2 components; the pyramid rows u_e = n +- mu t_k are formed in the update (below)
          s.sjr[0][i] = sg * cn; s.sjr[1][i] = sg * c1t; s.sjr[2][i] = sg * c2t;
        } else {
          s.sjr[0][i] = sg * cn; s.sjr[1][i] = 0.f; s.sjr[2][i] = 0.f;
        }
        s.sjr[3][i] = sg * (w[3] * ang[0] + w[4] * ang[1] + w[5] * ang[2]);  // n . z (spin rows)
      }
      __syncwarp();
      // the contact's friction coefficients (the pair of contact c0 + j: its table entry)
      const unsigned ej = s.con[c0 + j];
      const int pj = ej & 0xFFFFu;
      const float muj = PFOOT(pj) ? s.fric : PMU(pj);
#if HAS_CONDIM4
      const float musj = dm == 4 ? PMUS(pj) : 0.f;
#else
      const float musj = 0.f;
#endif
      for (int a = 0; a < NV; a++) {
        const float na = s.sjr[0][a], t1a = s.sjr[1][a], t2a = s.sjr[2][a], sa = s.sjr[3][a];
        for (int bb = lane; bb <= a; bb += 32) {
          const float nb_ = s.sjr[0][bb], t1b = s.sjr[1][bb], t2b = s.sjr[2][bb], sb = s.sjr[3][bb];
          float acc = 0.f;
          if (dm >= 3) {
            const float u0a = na + muj * t1a, u0b = nb_ + muj * t1b, u1a = na - muj * t1a, u1b = nb_ - muj * t1b;
            const float u2a = na + muj * t2a, u2b = nb_ + muj * t2b, u3a = na - muj * t2a, u3b = nb_ - muj * t2b;
            acc = w[12] * u0a * u0b + w[13] * u1a * u1b + w[14] * u2a * u2b + w[15] * u3a * u3b;
            if (dm == 4) {
              const float u4a = na + musj * sa, u4b = nb_ + musj * sb, u5a = na - musj * sa, u5b = nb_ - musj * sb;
              acc += w[16] * u4a * u4b + w[17] * u5a * u5b;
            }
          } else {
            acc = w[12] * na * nb_;
          }
          s.L[TRI(a, bb)] += acc;
        }
      }
      __syncwarp();
    }
  }
}
#endif
"""

KERNEL_SIG = ("                          const float* ctrl ADJ_SOFT_PARAM) {\n",
              "                          const float* ctrl ADJ_SOFT_PARAM, float cone_mu, float impulse_clip) {\n")
KERNEL_INIT = ("  if (lane == 0) s.fric = fric ? fric[w] : 0.f;\n",
               "  if (lane == 0) s.fric = fric ? fric[w] : 0.f;\n"
               "  if (lane == 0) {  // the surrogates' parameters (surrogate.py): both off run the exact adjoint's code\n"
               "    s.sur_mu = cone_mu;\n    s.sur_clip = impulse_clip;\n"
               "    s.sur_on = cone_mu > 0.f || impulse_clip < INFINITY;\n  }\n")


import re


def _rep(src, a, b, n=1):
    assert src.count(a) == n, f"surrogate anchor found {src.count(a)} times (want {n}): {a[:80]!r}"
    return src.replace(a, b)


PARAM_EDITS3 = (
    ("__device__ __noinline__ void contact_param_vjp(", "__device__ __noinline__ void contact_param_vjp_sur("),
    ("                                               float* pp) {",
     "                                               float* pp, float smu) {"),
    ("    if (!((zone >> e) & 1u) || (e > 0 && dim != 3)) continue;\n", "    if (e > 0 && dim != 3) continue;\n"),
)
PARAM_EDITS4 = (
    ("__device__ __noinline__ void contact_param_vjp4(", "__device__ __noinline__ void contact_param_vjp4_sur("),
    ("                                               const float* Ol, float mus) {",
     "                                               const float* Ol, float mus, float smu) {"),
    ("    if (!((zone >> e) & 1u) || (e > 0 && dim < 3)) continue;\n", "    if (e > 0 && dim < 3) continue;\n"),
    ("      if (!((zone >> e) & 1u)) continue;\n", ""),
    ("      pp[6] += dDdmu * xe * ul;\n      pp[7] += D * se * ((wa + b * wv) * ul + xe * wl);\n",
     "      pp[6] += zf * (dDdmu * xe * ul);\n      pp[7] += D * se * (wz * ((wa + b * wv) * ul) + zf * (xe * wl));\n"),
)
PARAM_XE = "const float xe = ua + k * imp * d + b * uv;\n"
PARAM_ROW = ("for (int q = 0; q < 6; q++) pp[q] += ul * (dDdi * di[q] * xe + D * (d * (dk[q] * imp + k * di[q]) + db[q] "
             "* uv));\n")


def _param_row(ind):
    return ((ind + PARAM_XE + ind + PARAM_ROW),
            (ind + PARAM_XE + ind + "const float zf = ((zone >> e) & 1u) ? 1.f : 0.f, wz = smu > 0.f ? sur_weight(xe, smu)"
             " : zf;\n" + ind + "for (int q = 0; q < 6; q++)\n" + ind + "  pp[q] += ul * (zf * (dDdi * di[q] * xe) + wz * "
             "(D * (d * (dk[q] * imp + k * di[q]) + db[q] * uv)));\n"))


PARAM_MU = ("    pp[6] += dDdmu * xe * ul + D * se * ((af[t] + b * vf[t]) * ul + xe * lf[t]);\n",
            "    pp[6] += zf * (dDdmu * xe * ul) + D * se * (wz * ((af[t] + b * vf[t]) * ul) + zf * (xe * lf[t]));\n")
PARAM_ANCHOR = "__device__ __forceinline__ float adj_contact_one(Adj& s, int c, int pair, int slot, unsigned zone,"
PARAM_CALL_HEAD = "  if (PFOOT(pair) && zone) {  // the foot pair's parameters (world_params)\n"
PARAM_CALL_SUR = (
    "  if (PFOOT(pair) && (zone || s.sur_mu > 0.f)) {  // the foot pair's parameters (world_params; surrogate.py)\n"
    "   if (s.sur_on) {  // the surrogates' row weights, the impulse clip's factor\n"
    "    float pq[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};\n"
    "#if HAS_CONDIM4\n"
    "    contact_param_vjp4_sur(Vv, Va, Vl, fr, mu, dim, zone, d, D, k, b, imp, contact_invw(pair, mu), sref, simp, pq,"
    " Ov, Oa, Ol, dim == 4 ? @MUS@ : 0.f, s.sur_mu);\n"
    "#else\n"
    "    contact_param_vjp_sur(Vv, Va, Vl, fr, mu, dim, zone, d, D, k, b, imp, contact_invw(pair, mu), sref, simp, pq,"
    " s.sur_mu);\n"
    "#endif\n"
    "    for (int q = 0; q < 8; q++) pp[q] += sur_kap * pq[q];\n"
    "   } else {\n")
PARAM_CALL_TAIL = ("  }\n  cot[1] = cot[2] = cot[3] = 0.f;\n  contact_pbar_motion(Wv, Wa, Wl, Ov, Oa, Ol, cot + 1);\n",
                   "   }\n  }\n  cot[1] = cot[2] = cot[3] = 0.f;\n  contact_pbar_motion(Wv, Wa, Wl, Ov, Oa, Ol, cot + 1);\n")


def _param_device(src):
    from nablaengine.g1_engine import world_params as G
    out = ""
    for dev, edits, n in ((G.PARTIALS_DEVICE, PARAM_EDITS3, 1), (G.PARTIALS4_DEVICE, PARAM_EDITS4, 2)):
        if dev not in src:
            continue
        for a, b in edits:
            dev = _rep(dev, a, b)
        for ind in ("    ", "      ")[:n]:
            dev = _rep(dev, *_param_row(ind))
        dev = _rep(dev, *PARAM_MU)
        assert "zone >> e" not in dev.replace("((zone >> e) & 1u) ? 1.f : 0.f", ""), "surrogate: a zone test left"
        out += dev
    assert out, "surrogate: world_params' contact_param_vjp not found"
    return out


def _params(src, m, params):
    tors = "foot_torsional" in params.fields
    mus = "(PFOOT(pair) ? s.musw : PMUS(pair))" if tors else "PMUS(pair)"
    src = _rep(src, PARAM_ANCHOR, _param_device(src) + PARAM_ANCHOR)
    i = src.index(PARAM_CALL_HEAD)
    j = src.index(PARAM_CALL_TAIL[0], i)
    body = src[i + len(PARAM_CALL_HEAD):j]
    assert body.count("contact_param_vjp") == 2 and src.count(PARAM_CALL_HEAD) == 1
    src = src[:i] + PARAM_CALL_SUR.replace("@MUS@", mus) + body + PARAM_CALL_TAIL[1] + src[j + len(PARAM_CALL_TAIL[0]):]
    return src


def _kap(call, params):
    if not params:
        return call
    call = _rep(call, "  if (s.sur_on) {", "  float sur_kap = 1.f;  // the impulse clip's factor (the parameter partials)\n"
                "  if (s.sur_on) {")
    return _rep(call, "      const float kap = s.sur_clip / imp_c;\n",
                "      const float kap = s.sur_clip / imp_c;\n      sur_kap = kap;\n")


def apply(src, m, params=None):
    """The engine source (adjoint.cuda_source's full text, after forest_engine.world_params' edits of a build with
    per-world parameters params) with the surrogates; asserts every anchor once."""
    import mujoco
    if int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        raise ValueError("contact_surrogate: the pyramidal friction cone only (the model has opt.cone = elliptic)")
    tors = bool(params) and "foot_torsional" in params.fields
    mus = "(PFOOT(pair) ? s.musw : PMUS(pair))" if tors else "PMUS(pair)"
    src = _rep(src, *ADJ_STRUCT)
    src = _rep(src, DECL_ANCHOR, DEVICE + DECL + DECL_ANCHOR)
    src = _rep(src, SOLVE, SOLVE_SUR)
    call4 = PSI_CALL4.replace("PMUS(pair)", mus)
    if src.count(call4):
        src = _rep(src, call4, _kap(_sur_call(call4, True).replace(ARGS4, ARGS4.replace("PMUS(pair)", mus)), params))
    src = _rep(src, PSI_CALL3, _kap(_sur_call(PSI_CALL3, False), params))
    src = _rep(src, *PRIMITIVE)
    cu = CONTACTS_USE if not params else tuple(x.replace("cw, dbg)", "cw, dbg, pp)") for x in CONTACTS_USE)
    src = _rep(src, *cu)
    if params:
        src = _params(src, m, params)
    mt = re.search(r"  float d, p\[3\], fr\[9\], nu;\n((?:  [^\n]*;[^\n]*\n)*?)  const int br = adj_pair_geom\(pair, slot, "
                   r"s\.gx, s\.gq, &d, p, fr, &nu, nullptr, nullptr([^)]*)\);\n", src)
    assert mt, "surrogate: adj_contact_one's replay call"
    from nablaengine.forest_engine import adjoint
    hess = HESSIAN.replace("@MVID@", mt.group(2)).replace("@ROW0@", adjoint.contact_row_line(src))
    if params:
        from nablaengine.g1_engine import world_params as G
        hess = _rep(hess, G.FOOT[2][0], G.FOOT[2][1])
        if tors:
            hess = _rep(hess, "const float wv = dot3(fr, Ov), wa = dot3(fr, Oa), mus = PMUS(pair);",
                        f"const float wv = dot3(fr, Ov), wa = dot3(fr, Oa), mus = {mus};")
            hess = _rep(hess, "const float musj = dm == 4 ? PMUS(pj) : 0.f;",
                        "const float musj = dm == 4 ? (PFOOT(pj) ? s.musw : PMUS(pj)) : 0.f;")
    hess = hess.replace("  float d, p[3], fr[9], nu;\n  adj_pair_geom(", "  float d, p[3], fr[9], nu;\n" + mt.group(1)
                        + "  adj_pair_geom(")
    src = _rep(src, HESSIAN_ANCHOR, hess + HESSIAN_ANCHOR)
    src = _rep(src, *KERNEL_SIG)
    src = _rep(src, *KERNEL_INIT)
    return src

"""Elliptic friction cones in the forest engine: g1_engine.elliptic ported (its device code, ROWS3 / ROWS4 and the
adjoint's contact_psi_vjp_ell / _ell4 verbatim; forest-specific anchors where the forest text differs: the CC / CD
macros, the condim-4 text the forest has in its base text under HAS_CONDIM4, the ZWord line-search words and up to CPL
contacts per lane, the adjoint's r0x / nr lines). Applied by adjoint.cuda_source after model_hooks, only for a model with
opt.cone = elliptic (the pyramidal text unchanged). MuJoCo Warp 3.17's semantics as g1_engine.elliptic's docstring.
step.pair_param_sets floors the friction values at MJ_MINMU = 1e-5 for elliptic models (as MuJoCo Warp's contacts are)."""
import numpy as np
from nablaengine.g1_engine import elliptic as E
LS_START = "    constexpr int RT = (NEFC + 31) / 32;\n    typedef typename ZWord<(RT <= 16)>::T ZT;  // 2 bits per row\n"
LS_END = "    zsame = !__any_sync(0xffffffffu, chg);  // then the next H is this one\n"

LS_NEW = r"""    constexpr int RT = (NEFC + 31) / 32;
    typedef typename ZWord<(RT <= 16)>::T ZT;  // 2 bits per row
    static_assert(RT <= 32, "the rows' piece codes in 64 bits");
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
    EllLs el[CPL];  // the lane's contacts lane + 32 h (forest: up to 8 per lane at MAXCON 255)
#pragma unroll
    for (int h2 = 0; h2 < CPL; h2++) ell_ls_load(s, lane + 32 * h2, ncon_e, el[h2]);
    float alpha = 0.f, lo = 0.f, hi = 1e30f, d0 = 0.f;
    ZT zprev = 0u;
    unsigned zeprev = 0u;  // the contacts' zones, 2 bits each (CPL <= 8)
    bool newton = false;
    for (int k = 0; k <= 40; k++) {
      float fd = 0.f, fdd = 0.f;
      ZT z = 0u;
      bool cone = false;
#pragma unroll
      for (int t = 0; t < RT; t++) {
        if (32 * t >= nefc) break;
        if ((ellm >> t) & 1u) continue;
        float ds, dd, jv = rj[t];
        z |= (ZT)row_piece(lane + 32 * t, rx[t] + alpha * jv, rD[t], rF[t], &ds, &dd) << (2 * t);
        fd += ds * jv; fdd += dd * jv * jv;
      }
      unsigned ze = 0u;
#pragma unroll
      for (int h2 = 0; h2 < CPL; h2++) {
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
        if ((ellm >> t) & 1u) { s.efc_jar[r] = xn; continue; }
        int z0 = row_piece(r, rx[t], rD[t], rF[t], &ds, &dd), z1 = row_piece(r, xn, rD[t], rF[t], &ds, &dd);
        chg |= z0 != z1;
        g += row_cost(z1, xn, rD[t], rF[t]);
        s.efc_jar[r] = xn;
        s.efc_jv[r] = -ds;  // the line search's J p is not needed any more
      }
    }
    __syncwarp();
    bool econe = false;
#pragma unroll
    for (int h2 = 0; h2 < CPL; h2++) {
      float e1, e2, ec0, ec;
      const int z0 = ell_ls_eval(el[h2], 0.f, &e1, &e2, &ec0), z1 = ell_ls_eval(el[h2], alpha, &e1, &e2, &ec);
      chg |= z0 != z1;
      econe |= z1 == 2;
      g += ec;
      ell_forces(s, lane + 32 * h2, ncon_e);
    }
    zsame = !__any_sync(0xffffffffu, chg || econe);  // then the next H is this one (a cone contact's changes with jar)
"""

CON_FORCE_OLD = """  float mu = s.con_dim[c] >= 3 ? s.con_mu[c] : 0.f;
  g[0] = f0 + f1 + f2 + f3; g[1] = mu * (f0 - f1); g[2] = mu * (f2 - f3);
#if HAS_CONDIM4
  float f4 = nr > 4 ? fr[4] : 0.f, f5 = nr > 5 ? fr[5] : 0.f;
  g[0] += f4 + f5; g[3] = PMUS(s.con_pair[c]) * (f4 - f5);
#endif
"""
CON_FORCE_NEW = """  g[0] = f0; g[1] = f1; g[2] = f2;  // elliptic rows: one per frame direction
#if HAS_CONDIM4
  g[3] = f3;  // (condim 4: the torsional row)
#else
  (void)f3;
#endif
"""

COEF_OLD = """      int r0 = s.con_row[c], nr = s.con_nrow[c];
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
"""
COEF_NEW = "      on = ell_coef(s, c, s.con_C[c]);  // elliptic: the contact's block (g1_engine.elliptic)\n"

ASM_OLD = """        float g0 = C[0] * j0 + C[1] * j1 + C[3] * j2, g1 = C[1] * j0 + C[2] * j1, g2 = C[3] * j0 + C[4] * j2;
#if HAS_CONDIM4
        const float* Jr = s.con_J[cc][3];
        const float j3 = Jr[b], g3 = C[5] * j0 + C[6] * j3;
        g0 += C[5] * j3;
#endif
"""
ROWS_OLD = """        int nr = max(0, min(tri ? (HAS_CONDIM4 && s.con_dim[c] == 4 ? 6 : 4) : 1, NEFC - r0));  // rows beyond NEFC: dropped
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
"""


def hooks(m):
    v = E.isq(m)
    c4 = bool((np.asarray(m.pair_dim) == 4).any())
    nt, nc = (4, 10) if c4 else (3, 6)
    asm_new = (("        const float* Jr = s.con_J[cc][3];  // condim 4: the torsional row's couplings (C[6..9])\n"
                "        const float j3 = Jr[b];\n"
                "        float g0 = C[0] * j0 + C[1] * j1 + C[3] * j2 + C[6] * j3, g1 = C[1] * j0 + C[2] * j1 + C[5] * j2 + C[8] * j3,\n"
                "              g2 = C[3] * j0 + C[4] * j2 + C[5] * j1 + C[9] * j3, g3 = C[6] * j0 + C[8] * j1 + C[9] * j2 + C[7] * j3;\n")
               if c4 else
               ("        float g0 = C[0] * j0 + C[1] * j1 + C[3] * j2, g1 = C[1] * j0 + C[2] * j1 + C[5] * j2,\n"
                "              g2 = C[3] * j0 + C[4] * j2 + C[5] * j1;  // (elliptic: the cone's t1 t2 coupling)\n"))
    return (
        ("#define NPAIR ", f"#define ELLIPTIC 1  // elliptic friction cones (forest port of g1_engine.elliptic)\n"
                           f"#define ELL_ISQ {float(v)!r}f  // MuJoCo Warp's impratio_invsqrt (float32)\n"
                           f"#define ELL_NT {nt}  // frame rows per contact (4: condim 4)\n"
                           f"#define ELL_NC {nc}  // Hessian coefficients per contact\n#define NPAIR "),
        ("#define CC (HAS_CONDIM4 ? 7 : 5)  // Hessian coefficients per contact\n",
         "#define CC ELL_NC  // Hessian coefficients per contact (elliptic: ell_coef's order)\n"),
        ("// row cost derivative s'(x), curvature (D on the quadratic zone, else 0) and cost s(x)\n",
         E.DEVICE + "// row cost derivative s'(x), curvature (D on the quadratic zone, else 0) and cost s(x)\n"),
        ("  float D = s.efc_D[r], F = row_floss(s, r);\n  *cost = row_cost(row_piece(r, x, D, F, ds, dd), x, D, F);\n",
         "  if (s.efc_kind[r] == 4) {  // elliptic: per contact (at the rows' jar in s)\n"
         "    float f;\n    ell_row(s, r, &f, cost);\n    *ds = -f; *dd = 0.f;\n    return;\n  }\n"
         "  float D = s.efc_D[r], F = row_floss(s, r);\n  *cost = row_cost(row_piece(r, x, D, F, ds, dd), x, D, F);\n"),
        (CON_FORCE_OLD, CON_FORCE_NEW),
        ("    } else val = s.con_dot[a][0];\n",
         "    } else if (kind == 4) val = s.con_dot[a][b];  // elliptic: the frame row\n    else val = s.con_dot[a][0];\n"),
        ("  float ds, dd;\n  return row_piece(r, x, s.efc_D[r], row_floss(s, r), &ds, &dd);\n",
         "  if (s.efc_kind[r] == 4) {  // elliptic (tape bits): normal row = bottom zone, first tangent row = cone zone\n"
         "    float f, c;\n    const int z = ell_row(s, r, &f, &c), t = s.efc_b[r];\n"
         "    return (t == 0 && z == 1) || (t == 1 && z == 2) ? 1 : 0;\n  }\n"
         "  float ds, dd;\n  return row_piece(r, x, s.efc_D[r], row_floss(s, r), &ds, &dd);\n"),
        ("    int r0 = ndof + c + 3 * (ntri + __popc(b3 & below));\n",
         "    int r0 = ndof + c + 2 * (ntri + __popc(b3 & below));  // (elliptic: 3 rows per condim-3 contact)\n"),
        ("    r0 += 2 * (nquad + __popc(b4 & below));\n",
         "    r0 += nquad + __popc(b4 & below);  // (elliptic condim 4: the torsional row)\n"),
        ("    int need = ndof + ncon + 3 * ntri + 2 * nquad;\n", "    int need = ndof + ncon + 2 * ntri + nquad;\n"),
        ("        iw = contact_invw(pr, mu);\n",
         "        iw = RADD(c_binvw[c_gbody[PG1(pr)]], c_binvw[c_gbody[PG2(pr)]]);  // elliptic: no pyramid scaling\n"),
        (ROWS_OLD, E.ROWS4 if c4 else E.ROWS3),
        (COEF_OLD, COEF_NEW),
        (ASM_OLD, asm_new),
        ("LS", None),
        ("        refact = zchanged;\n",
         "        refact = zchanged || ell_any_cone(s, lane);  // (a cone contact's curvature: re-factor at the final point)\n"),
    )


ADJ_CALL_OLD = """#if HAS_CONDIM4
  float AWv[3], AWa[3], AWl[3];
  contact_psi_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4, Ov, Oa, Ol,
"""


def adj_hooks(m, src):
    """The adjoint's edits (the elliptic contact VJP, invweight, row offsets)."""
    c4 = bool((np.asarray(m.pair_dim) == 4).any())
    i = src.index(ADJ_CALL_OLD)
    j = src.index("#endif\n", i) + len("#endif\n")
    call_old = src[i:j]
    call_new = ("#if HAS_CONDIM4\n  float AWv[3], AWa[3], AWl[3];\n"
                "  contact_psi_vjp_ell4(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4,"
                " Ov, Oa, Ol, dim == 4 ? PMUS(pair) : 0.f, AWv, AWa, AWl);\n#else\n"
                "  contact_psi_vjp_ell(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4);\n"
                "#endif\n")
    hdr = "__device__ __forceinline__ float adj_contact_one("
    k = src.index(hdr)
    return (
        (hdr, (E.ADJ_DEVICE4 if c4 else E.ADJ_DEVICE) + hdr),
        ("  adj_row_impedance(PSOLREF(pair), PSOLIMP(pair), d, contact_invw(pair, mu), ri);\n",
         "  adj_row_impedance(PSOLREF(pair), PSOLIMP(pair), d, RADD(c_binvw[b1], c_binvw[b2]), ri);  // (elliptic invweight)\n"),
        (call_old, call_new),
        ("c + 3 * (ntri + __popc(b3 & below));  // the contact's first row",
         "c + 2 * (ntri + __popc(b3 & below));  // the contact's first row (elliptic: 3 rows)"),
        ("    const int r0x = r0 + 2 * (nquad + __popc(b4 & below));",
         "    const int r0x = r0 + nquad + __popc(b4 & below);  // (elliptic condim 4: the torsional row)"),
        ("      const int nr = min(dim >= 3 ? (dim == 4 ? 6 : 4) : 1, NEFC - r0x), w = r0x >> 5;\n",
         "      const int nr = min(dim >= 3 ? (dim == 4 ? 4 : 3) : 1, NEFC - r0x), w = r0x >> 5;\n"),
    )


def applies(m):
    return E.applies(m)


def apply(src, m):
    """src (adjoint.cuda_source's unit, after model_hooks) with the elliptic edits (m elliptic), else src unchanged."""
    if not E.applies(m):
        return src
    ok, why = E.supported(m)
    if not ok:
        raise ValueError(f"elliptic cone: {why}")
    for a, b in hooks(m) + adj_hooks(m, src):
        if a == "LS":
            assert src.count(LS_START) == 1 and src.count(LS_END) == 1, "the line search's span"
            i = src.index(LS_START)
            j = src.index(LS_END, i) + len(LS_END)
            src = src[:i] + LS_NEW + src[j:]
            continue
        k = src.count(a)
        if k != 1:
            raise RuntimeError(f"forest elliptic: anchor found {k} times, expected 1: {a[:100]!r}")
        src = src.replace(a, b)
    return src

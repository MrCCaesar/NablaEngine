"""Condim-4 contacts (torsional friction) in the G1-class engine, pyramidal cone."""

import numpy as np


def applies(m):
    return bool((np.asarray(m.pair_dim) == 4).any())


def check(m, terrain=None, params=None):
    """Refuse the combinations not supported yet (ValueError); condim must be 1, 3 or 4."""
    if not applies(m):
        return
    bad = sorted(set(int(x) for x in m.pair_dim) - {1, 3, 4})
    if bad:
        raise ValueError(f"pair condim {bad}: the G1-class engine supports condim 1, 3 and 4")
    if terrain is not None:
        raise ValueError("condim-4 contacts in a terrain build: not supported yet")


def _lit(x):
    return f"{float(np.float32(x))!r}f"


ADJ_DEVICE = r"""
// contact_psi_vjp with condim 4 (g1_engine/condim4.py; forest_engine/adjoint_contact.py's HAS_CONDIM4 version): the
// spin rows u = n +- mus r (rows 4, 5) on the contact's relative angular motions Om (n . Om_v, n . Om_a, n . Om_lam);
// AW_* = (aw n): their weights on the angular columns, and n's cotangent in frbar
__device__ __forceinline__ void contact_psi_vjp4(const float* Vv, const float* Va, const float* Vl, const float* fr,
                                                 float mu, int dim, unsigned zone, float d, float D, float dD, float k,
                                                 float b, float imp, float dimp, float* Wv, float* Wa, float* Wl,
                                                 float* dbar, float* frbar, const float* Ov, const float* Oa,
                                                 const float* Ol, float mus, float* AWv, float* AWa, float* AWl) {
  float vf[3], af[3], lf[3];
#pragma unroll
  for (int j = 0; j < 3; j++) { vf[j] = dot3(fr + 3 * j, Vv); af[j] = dot3(fr + 3 * j, Va); lf[j] = dot3(fr + 3 * j, Vl); }
  float g = k * imp * d, gd = k * (imp + dimp * d);
  float cl[3] = {0.f, 0.f, 0.f}, ca[3] = {0.f, 0.f, 0.f}, db = 0.f;
#pragma unroll
  for (int e = 0; e < 4; e++) {
    if (!((zone >> e) & 1u) || (e > 0 && dim < 3)) continue;
    const int t = 1 + (e >> 1);
    float sm = (e & 1) ? -mu : mu;
    float uv = vf[0], ua = af[0], ul = lf[0];
    if (dim >= 3) { uv += sm * vf[t]; ua += sm * af[t]; ul += sm * lf[t]; }
    float x = ua + g + b * uv, Dx = D * x, Dl = D * ul;
    cl[0] += Dx; ca[0] += Dl;
    if (dim >= 3) { cl[t] += sm * Dx; ca[t] += sm * Dl; }
    db += ul * (dD * x + D * gd);
  }
  float awl = 0.f, awa = 0.f;
  if (dim == 4) {
    const float wv = dot3(fr, Ov), wa = dot3(fr, Oa), wl = dot3(fr, Ol);
#pragma unroll
    for (int e = 4; e < 6; e++) {
      if (!((zone >> e) & 1u)) continue;
      const float sm = (e & 1) ? -mus : mus;
      const float uv = vf[0] + sm * wv, ua = af[0] + sm * wa, ul = lf[0] + sm * wl;
      const float x = ua + g + b * uv, Dx = D * x, Dl = D * ul;
      cl[0] += Dx; ca[0] += Dl;
      awl += sm * Dx; awa += sm * Dl;
      db += ul * (dD * x + D * gd);
    }
  }
#pragma unroll
  for (int i = 0; i < 3; i++) { AWl[i] = awl * fr[i]; AWa[i] = awa * fr[i]; AWv[i] = b * AWa[i]; }
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
#pragma unroll
  for (int i = 0; i < 3; i++) frbar[i] += awl * Ol[i] + awa * (Oa[i] + b * Ov[i]);
  *dbar = db;
}

"""


def edits(m):
    """[(anchor, replacement, count)] in application order."""
    mus = [max(float(m.pair_friction[p, 2]), 1e-5) if int(m.pair_dim[p]) == 4 else 0.0 for p in range(m.npair)]
    ell = int(m.opt.cone) == 1
    rfms = "#define RFMS(a, b, c) __fmaf_rn(-(a), (b), (c))  // c - a b, one rounding\n"
    E = [
        (rfms, rfms + "#define HAS_CONDIM4 1  // (g1_engine/condim4.py)\n"
         f"__device__ const float c_pmus[NPAIR] = {{{', '.join(_lit(x) for x in mus)}}};  // pair_friction[2] of the "
         "condim-4 pairs\n#define PMUS(pr) c_pmus[pr]\n", 1),
        ("    float con_dot[MAXCON][3];  // J_c,d v (rows_jv)\n", "    float con_dot[MAXCON][4];  // J_c,d v (rows_jv)\n", 1),
        ("    float con_f[MAXCON][3];    //", "    float con_f[MAXCON][4];    //", 1),
        ("    float con_C[MAXCON][5];    //", "    float con_C[MAXCON][7];    //", 1),
        ("  alignas(16) float con_J[MAXCON][3][KDOF];", "  alignas(16) float con_J[MAXCON][4][KDOF];", 2),
        ("    float con_C[NC][5];", "    float con_C[NC][7];", 1),
        ("      union { float con_dot[NC][3], con_f[NC][3]; };", "      union { float con_dot[NC][4], con_f[NC][4]; };", 1),
        ("  const float (*con_J)[3][KDOF];", "  const float (*con_J)[4][KDOF];", 1),
        ("con_dot[MAXCON][3]; unsigned short cand[NPAIR]; };", "con_dot[MAXCON][4]; unsigned short cand[NPAIR]; };", 1),
        ("\n  float (*con_J)[3][KDOF];", "\n  float (*con_J)[4][KDOF];", 1),
        ("  if (PDIM(pr) == 3) iw = RMUL(", "  if (PDIM(pr) >= 3) iw = RMUL(", 1),
        ("  for (int q = lane; q < 3 * ncon; q += 32) {\n    int c = q / 3, nd = s.con_nd[c];\n"
         "    const float* J = s.con_J[c][q % 3];\n",
         "  for (int q = lane; q < 4 * ncon; q += 32) {\n    int c = q / 4, nd = s.con_nd[c];\n"
         "    const float* J = s.con_J[c][q % 4];\n", 1),
        ("    s.con_dot[c][q % 3] = a0 + a1;\n", "    s.con_dot[c][q % 4] = a0 + a1;\n", 1),
        ("      val = s.con_dot[a][0] + sg * mu * s.con_dot[a][1 + (b >> 1)];\n",
         "      if (b >= 4) val = s.con_dot[a][0] + sg * PMUS(s.con_pair[a]) * s.con_dot[a][3];  // (the spin rows)\n"
         "      else val = s.con_dot[a][0] + sg * mu * s.con_dot[a][1 + (b >> 1)];\n", 1),
        ("  float mu = s.con_dim[c] == 3 ? s.con_mu[c] : 0.f;\n"
         "  g[0] = f0 + f1 + f2 + f3; g[1] = mu * (f0 - f1); g[2] = mu * (f2 - f3);\n",
         "  float mu = s.con_dim[c] >= 3 ? s.con_mu[c] : 0.f;\n"
         "  g[0] = f0 + f1 + f2 + f3; g[1] = mu * (f0 - f1); g[2] = mu * (f2 - f3);\n"
         "  const float f4 = nr > 4 ? fr[4] : 0.f, f5 = nr > 5 ? fr[5] : 0.f;\n"
         "  g[0] += f4 + f5; g[3] = PMUS(s.con_pair[c]) * (f4 - f5);  // (condim 4)\n", 1),
        ("    float g[3] = {0.f, 0.f, 0.f};\n    unsigned glo = 0u, ghi = 0u;\n",
         "    float g[4] = {0.f, 0.f, 0.f, 0.f};\n    unsigned glo = 0u, ghi = 0u;\n", 1),
        ("      const float f0 = __shfl_sync(0xffffffffu, g[0], i), f1 = __shfl_sync(0xffffffffu, g[1], i),\n"
         "                  f2 = __shfl_sync(0xffffffffu, g[2], i);\n",
         "      const float f0 = __shfl_sync(0xffffffffu, g[0], i), f1 = __shfl_sync(0xffffffffu, g[1], i),\n"
         "                  f2 = __shfl_sync(0xffffffffu, g[2], i), f3 = __shfl_sync(0xffffffffu, g[3], i);\n", 1),
        ("        acc[0] += f0 * J[k] + f1 * J[KDOF + k] + f2 * J[2 * KDOF + k];\n",
         "        acc[0] += f0 * J[k] + f1 * J[KDOF + k] + f2 * J[2 * KDOF + k] + f3 * J[3 * KDOF + k];\n", 1),
        ("        const float f0 = __shfl_sync(0xffffffffu, g[0], i), f1 = __shfl_sync(0xffffffffu, g[1], i),\n"
         "                    f2 = __shfl_sync(0xffffffffu, g[2], i);\n",
         "        const float f0 = __shfl_sync(0xffffffffu, g[0], i), f1 = __shfl_sync(0xffffffffu, g[1], i),\n"
         "                    f2 = __shfl_sync(0xffffffffu, g[2], i), f3 = __shfl_sync(0xffffffffu, g[3], i);\n", 1),
        ("          acc[1] += f0 * J[k] + f1 * J[KDOF + k] + f2 * J[2 * KDOF + k];\n",
         "          acc[1] += f0 * J[k] + f1 * J[KDOF + k] + f2 * J[2 * KDOF + k] + f3 * J[3 * KDOF + k];\n", 1),
        ("                                 + s.con_f[c][2] * s.con_J[c][2][lane];\n",
         "                                 + s.con_f[c][2] * s.con_J[c][2][lane] + s.con_f[c][3] * s.con_J[c][3][lane];\n", 1),
        ("      float d[4];\n#pragma unroll\n      for (int k = 0; k < 4; k++) {",
         "      float d[6];\n#pragma unroll\n      for (int k = 0; k < 6; k++) {", 1),
        ("      float d0 = d[0], d1 = d[1], d2 = d[2], d3 = d[3], mu = s.con_dim[c] == 3 ? s.con_mu[c] : 0.f;\n",
         "      float d0 = d[0], d1 = d[1], d2 = d[2], d3 = d[3], mu = s.con_dim[c] >= 3 ? s.con_mu[c] : 0.f;\n", 1),
        ("      on = d0 + d1 + d2 + d3 != 0.f;  // C_nn is the sum of the curvatures\n",
         "      {  // condim 4: the spin rows n +- mus r\n"
         "        const float mus = s.con_dim[c] == 4 ? PMUS(s.con_pair[c]) : 0.f;\n"
         "        s.con_C[c][0] += d[4] + d[5];\n"
         "        s.con_C[c][5] = mus * (d[4] - d[5]); s.con_C[c][6] = mus * mus * (d[4] + d[5]);\n"
         "      }\n"
         "      on = s.con_C[c][0] != 0.f;  // C_nn is the sum of the curvatures\n", 1),
        ("        float g0 = C[0] * j0 + C[1] * j1 + C[3] * j2, g1 = C[1] * j0 + C[2] * j1, g2 = C[3] * j0 + C[4] * j2;\n",
         "        float g0 = C[0] * j0 + C[1] * j1 + C[3] * j2, g1 = C[1] * j0 + C[2] * j1, g2 = C[3] * j0 + C[4] * j2;\n"
         "        const float* Jr = s.con_J[cc][3];  // condim 4: (n, r) and (r, r)\n"
         "        const float j3 = Jr[b], g3 = C[5] * j0 + C[6] * j3;\n"
         "        g0 += C[5] * j3;\n", 1),
        ("          hv[t] = fmaf(Jb[a], g2, fmaf(Ja[a], g1, fmaf(Jn[a], g0, hv[t])));\n",
         "          hv[t] = fmaf(Jr[a], g3, fmaf(Jb[a], g2, fmaf(Ja[a], g1, fmaf(Jn[a], g0, hv[t]))));\n", 1),
        ("          for (int dd = 0; dd < 3; dd++) s.con_J[c][dd][k] = dot3(fr + 3 * dd, jp);\n",
         "          for (int dd = 0; dd < 3; dd++) s.con_J[c][dd][k] = dot3(fr + 3 * dd, jp);\n"
         "          s.con_J[c][3][k] = sg * dot3(fr, dw[h]);  // the normal's angular row (condim 4)\n", 1),
        ("  int ntri = 0;\n  for (int j0 = 0; j0 < ndof + ncon; j0 += 32) {\n",
         "  int ntri = 0, nquad = 0;  // (condim 4: 2 more rows each)\n"
         "  for (int j0 = 0; j0 < ndof + ncon; j0 += 32) {\n", 1),
        ("    bool con = c >= 0 && c < ncon, tri = con && s.con_dim[c] == 3;\n",
         "    bool con = c >= 0 && c < ncon, tri = con && s.con_dim[c] >= 3, quad = con && s.con_dim[c] == 4;\n", 1),
        ("    int r0 = ndof + c + 3 * (ntri + __popc(b3 & below));\n    ntri += __popc(b3);\n",
         "    int r0 = ndof + c + 3 * (ntri + __popc(b3 & below));\n    ntri += __popc(b3);\n"
         "    const unsigned b4 = __ballot_sync(0xffffffffu, quad);\n"
         "    r0 += 2 * (nquad + __popc(b4 & below));\n    nquad += __popc(b4);\n", 1),
        ("        int nr = max(0, min(tri ? 4 : 1, NEFC - r0));",
         "        int nr = max(0, min(tri ? (quad ? 6 : 4) : 1, NEFC - r0));", 1),
        ("#pragma unroll\n        for (int t = 0; t < 4; t++) {\n          if (t >= nr) break;\n",
         "#pragma unroll\n        for (int t = 0; t < 6; t++) {\n          if (t >= nr) break;\n", 1),
        ("          float vel = tri ? s.con_dot[c][0] + sg * mu * s.con_dot[c][1 + (t >> 1)] : s.con_dot[c][0];\n",
         "          float vel = tri ? (t >= 4 ? s.con_dot[c][0] + sg * PMUS(s.con_pair[c]) * s.con_dot[c][3]\n"
         "                                    : s.con_dot[c][0] + sg * mu * s.con_dot[c][1 + (t >> 1)])\n"
         "                          : s.con_dot[c][0];\n", 1),
        ("    int need = ndof + ncon + 3 * ntri;", "    int need = ndof + ncon + 3 * ntri + 2 * nquad;", 1),
        ("#define CS_N 24 ", "#define CS_N 33 ", 1),
        ("__device__ __forceinline__ float adj_contact_one(Adj& s, int c, int pair, int slot, unsigned zone, float* cs,\n",
         ADJ_DEVICE + "__device__ __forceinline__ float adj_contact_one(Adj& s, int c, int pair, int slot, unsigned zone,"
         " float* cs,\n", 1),
        ("  contact_psi_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4);\n",
         "  float AWv[3], AWa[3], AWl[3];\n"
         "  contact_psi_vjp4(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4, Ov, Oa,"
         " Ol, dim == 4 ? PMUS(pair) : 0.f, AWv, AWa, AWl);\n", 1),
        ("  for (int t = 0; t < 12; t++) cs[12 + t] = fg[t];\n",
         "  for (int t = 0; t < 12; t++) cs[12 + t] = fg[t];\n"
         "  for (int t = 0; t < 3; t++) { cs[24 + t] = AWv[t]; cs[27 + t] = AWa[t]; cs[30 + t] = AWl[t]; }\n", 1),
        ("  int ntri = 0;  // condim-3 contacts before the batch\n",
         "  int ntri = 0, nquad = 0;  // condim-3 / 4 and condim-4 contacts before the batch\n", 1),
        ("    const unsigned b3 = __ballot_sync(0xffffffffu, dim == 3);\n",
         "    const unsigned b3 = __ballot_sync(0xffffffffu, dim >= 3);\n", 1),
        ("    ntri += __popc(b3);\n    const bool use = on",
         "    ntri += __popc(b3);\n    const unsigned b4 = __ballot_sync(0xffffffffu, dim == 4);\n"
         "    const int r0x = r0 + 2 * (nquad + __popc(b4 & below));  // (condim 4)\n    nquad += __popc(b4);\n"
         "    const bool use = on", 1),
        ("    if (use && r0 < NEFC) {\n      const int nr = min(dim == 3 ? 4 : 1, NEFC - r0), w = r0 >> 5;\n",
         "    if (use && r0x < NEFC) {\n      const int nr = min(dim >= 3 ? (dim == 4 ? 6 : 4) : 1, NEFC - r0x), w = r0x >> 5;\n", 1),
        ("      zone = (unsigned)(zz >> (r0 & 31)) & ((1u << nr) - 1u);\n",
         "      zone = (unsigned)(zz >> (r0x & 31)) & ((1u << nr) - 1u);\n", 1),
        ("      for (int k = 0; k < 12; k++) w[k] = __shfl_sync(0xffffffffu, cw[k], j);\n",
         "      for (int k = 0; k < 12; k++) w[k] = __shfl_sync(0xffffffffu, cw[k], j);\n"
         "#pragma unroll\n      for (int k = 24; k < 33; k++) w[k] = __shfl_sync(0xffffffffu, cw[k], j);\n", 1),
        ("        for (int k = 0; k < 3; k++) { f[k] += u[k]; t[k] += y[k]; }\n",
         "        for (int k = 0; k < 3; k++) { f[k] += u[k]; t[k] += y[k]; }\n"
         "        {  // condim 4: the spin rows on the dof's angular column s (n . z)\n"
         "          float Ga[3], ua[3];\n"
         "          for (int k = 0; k < 3; k++) Ga[k] = vi * w[24 + k] + ai * w[27 + k] + li * w[30 + k];\n"
         "          pv += sg * dot3(z, w + 24);\n"
         "          cross(z, Ga, ua);\n"
         "          for (int k = 0; k < 3; k++) t[k] += ua[k];\n"
         "        }\n", 1),
    ]
    if not ell:
        return E
    pyr = ("    float con_C[MAXCON][5];", "    float con_C[NC][5];", "  float mu = s.con_dim[c] == 3 ? s.con_mu[c] : 0.f;\n",
           "      float d[4];", "      float d0 = d[0]", "      on = d0 + d1", "        float g0 = C[0] * j0",
           "    int r0 = ndof + c + 3 *", "        int nr = max(0, min(tri ? 4 : 1", "#pragma unroll\n        for (int t = 0; t < 4;",
           "          float vel = tri ?", "    int need = ndof + ncon + 3 * ntri;", "__device__ __forceinline__ float adj_contact_one",
           "  contact_psi_vjp(", "    ntri += __popc(b3);\n    const bool use = on", "    if (use && r0 < NEFC) {")
    E = [e for e in E if not e[0].startswith(pyr)]
    E += [
        ("    int r0 = ndof + c + 2 * (ntri + __popc(b3 & below));  // (elliptic: 3 rows per condim-3 contact)\n"
         "    ntri += __popc(b3);\n",
         "    int r0 = ndof + c + 2 * (ntri + __popc(b3 & below));  // (elliptic: 3 rows per condim-3 contact)\n"
         "    ntri += __popc(b3);\n"
         "    const unsigned b4 = __ballot_sync(0xffffffffu, quad);\n"
         "    r0 += nquad + __popc(b4 & below);  // (elliptic condim 4: the torsional row)\n    nquad += __popc(b4);\n", 1),
        ("    int need = ndof + ncon + 2 * ntri;", "    int need = ndof + ncon + 2 * ntri + nquad;", 1),
        ("    ntri += __popc(b3);\n    const bool use = on",
         "    ntri += __popc(b3);\n    const unsigned b4 = __ballot_sync(0xffffffffu, dim == 4);\n"
         "    const int r0x = r0 + nquad + __popc(b4 & below);  // (elliptic condim 4)\n    nquad += __popc(b4);\n"
         "    const bool use = on", 1),
        ("    if (use && r0 < NEFC) {\n      const int nr = min(dim == 3 ? 3 : 1, NEFC - r0), w = r0 >> 5;\n",
         "    if (use && r0x < NEFC) {\n      const int nr = min(dim >= 3 ? (dim == 4 ? 4 : 3) : 1, NEFC - r0x), w = r0x >> 5;\n", 1),
    ]
    return E


def apply(src, m):
    """src with the condim-4 edits (m has condim-4 pairs), else src unchanged."""
    if not applies(m):
        return src
    for a, b, n in edits(m):
        k = src.count(a)
        if k != n:
            raise RuntimeError(f"condim4: anchor found {k} times, expected {n}: {a[:90]!r}")
        src = src.replace(a, b)
    return src

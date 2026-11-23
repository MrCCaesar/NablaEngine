"""The cooperative VJP pass of the forest engine's native-CCD box-box contacts (DK_CCD_COOP, default 1)."""

HEAD = r"""#ifndef DK_CCD_COOP
#define DK_CCD_COOP 1  // (forest_engine.ccd_coop; sm_90: nablaengine.cuda.forest_hold_defines)
#endif
#define CC_COOP_ON (DK_CCD_COOP && CY_NT > 1 && CY_REC)
#ifndef CC_TEST_RECHECK
#define CC_TEST_RECHECK 0  // (tests: 1 fails every record's check in cc_dir: cc_full's path)
#endif
#define CC_COOP_PAIR(pair) (BOX_BOX_CCD && c_gtype[PG1(pair)] == 3 && c_gtype[PG2(pair)] == 3)
#if CC_COOP_ON
#define CC_REC_PARAM , CyRec& crec
#define CC_REC_ARG , crec
#else
#define CC_REC_PARAM
#define CC_REC_ARG
#endif
"""

DEFER = """#if CC_COOP_ON
    if (cot && crec && crec->kind == 1 && t1 == 3 && t2 == 3) return br;  // (ADJ_CCD_DEFER: forest_engine.ccd_coop's
    // pass, cc_coop, takes its VJP)
#endif
"""

DEVICE = r"""#if CC_COOP_ON
#define CC_NJ (12 / CY_NT)  // jobs (direction groups) per pair
// (forest_engine.ccd_coop) job jj of box-box pair `pair`: one DN pass from the record rec (kind 1) along pose
// directions CY_NT jj .. CY_NT jj + CY_NT - 1 (cyc_adj's seeds and pass), out[k] = sum over the pair's contacts i < nm
// (slots sl[i]; cr[i]: d bar, p bar, nbar) of cot . (dist', pos', nrm') of direction CY_NT jj + k
__device__ __noinline__ bool cc_dir(const float (*gx)[3], const float (*gq)[4], int pair, int jj, CyRec* rec, int nm,
                                    const int* sl, const float (*cr)[7], const float (*cb)[13], float* out) {
  const int g1 = PG1(pair), g2 = PG2(pair);
  const float *p1 = gx[g1], *p2 = gx[g2], *s1 = c_gsize + 3 * g1, *s2 = c_gsize + 3 * g2;
  float R1[9], R2[9];
  quat2mat(gq[g1], R1);
  quat2mat(gq[g2], R2);
  const int j0 = CY_NT * jj;
  DN P1[3], Q1[9], S1[3], P2[3], Q2[9], S2[3];
  cyc_geom<DN>(3, p1, R1, s1, P1, Q1, S1);
  cyc_geom<DN>(3, p2, R2, s2, P2, Q2, S2);
  for (int k = 0; k < CY_NT; k++) {
    const int j = j0 + k, g = j / 6, jd = j % 6;  // geom g, direction jd
    const float* R = g == 0 ? R1 : R2;
    float ea[3] = {0.f, 0.f, 0.f};
    ea[jd % 3] = 1.f;
    DN* P = g == 0 ? P1 : P2;
    DN* Q = g == 0 ? Q1 : Q2;
    if (jd < 3) DN_D(P[jd], k) = 1.f;
    else
      for (int cc = 0; cc < 3; cc++) {  // d col = e x col
        float col[3] = {R[cc], R[3 + cc], R[6 + cc]}, dc[3];
        cross(ea, col, dc);
        for (int r = 0; r < 3; r++) DN_D(Q[3 * r + cc], k) = dc[r];
      }
  }
  cycd::Sink d;
  cycd::cy_pair(cyc_cgeom<cycd::CGeom, DN>(3, g1, P1, Q1, S1), cyc_cgeom<cycd::CGeom, DN>(3, g2, P2, Q2, S2), d, rec,
                true);
  for (int k = 0; k < CY_NT; k++) out[k] = 0.f;
  bool same = true;
  for (int i = 0; i < nm; i++) {
    int cd = -1;
    for (int q = 0; q < d.n; q++) if (d.slot[q] == sl[i]) cd = q;
    if (cd < 0) { same = false; continue; }
    // the record's check (cyc_adj's RF replay of the record, here from the pass' values: DN's values are RF's bits)
    same = same && __float_as_int(d.dist[cd].v) == __float_as_int(cb[i][0]);
    for (int t = 0; t < 3; t++) same = same && __float_as_int(d.pos[cd][t].v) == __float_as_int(cb[i][1 + t]);
    for (int t = 0; t < 9; t++) same = same && __float_as_int(d.fr[cd][t].v) == __float_as_int(cb[i][4 + t]);
    const float *cot = cr[i], *nbar = cr[i] + 4;
    for (int k = 0; k < CY_NT; k++) {
      float gj = cot[0] * DN_D(d.dist[cd], k);
      for (int t = 0; t < 3; t++) gj += cot[1 + t] * DN_D(d.pos[cd][t], k) + nbar[t] * DN_D(d.nrm[cd][t], k);
      out[k] = i ? out[k] + gj : gj;
    }
  }
  return same && !CC_TEST_RECHECK;
}
// (forest_engine.ccd_coop) the contact `slot` of box-box pair `pair` whose record failed cc_dir's check: cyc_adj's full
// replay passes over DN (no record: the search in each pass) on this lane, gb[12] (cr: d bar, p bar, nbar)
__device__ __noinline__ void cc_full(const float (*gx)[3], const float (*gq)[4], int pair, int slot, const float* cr,
                                     float* gb) {
  const int g1 = PG1(pair), g2 = PG2(pair);
  const float *p1 = gx[g1], *p2 = gx[g2], *s1 = c_gsize + 3 * g1, *s2 = c_gsize + 3 * g2;
  const float *cot = cr, *nbar = cr + 4;
  float R1[9], R2[9];
  quat2mat(gq[g1], R1);
  quat2mat(gq[g2], R2);
  for (int j0 = 0; j0 < 12; j0 += CY_NT) {
    DN P1[3], Q1[9], S1[3], P2[3], Q2[9], S2[3];
    cyc_geom<DN>(3, p1, R1, s1, P1, Q1, S1);
    cyc_geom<DN>(3, p2, R2, s2, P2, Q2, S2);
    for (int k = 0; k < CY_NT; k++) {
      const int j = j0 + k, g = j / 6, jd = j % 6;
      const float* R = g == 0 ? R1 : R2;
      float ea[3] = {0.f, 0.f, 0.f};
      ea[jd % 3] = 1.f;
      DN* P = g == 0 ? P1 : P2;
      DN* Q = g == 0 ? Q1 : Q2;
      if (jd < 3) DN_D(P[jd], k) = 1.f;
      else
        for (int cc = 0; cc < 3; cc++) {
          float col[3] = {R[cc], R[3 + cc], R[6 + cc]}, dc[3];
          cross(ea, col, dc);
          for (int r = 0; r < 3; r++) DN_D(Q[3 * r + cc], k) = dc[r];
        }
    }
    cycd::Sink d;
    cycd::cy_pair(cyc_cgeom<cycd::CGeom, DN>(3, g1, P1, Q1, S1), cyc_cgeom<cycd::CGeom, DN>(3, g2, P2, Q2, S2), d,
                  nullptr, false);
    int cd = -1;
    for (int i = 0; i < d.n; i++) if (d.slot[i] == slot) cd = i;
    for (int k = 0; k < CY_NT; k++) {
      if (cd < 0) { gb[j0 + k] = 0.f; continue; }
      float gj = cot[0] * DN_D(d.dist[cd], k);
      for (int t = 0; t < 3; t++) gj += cot[1 + t] * DN_D(d.pos[cd][t], k) + nbar[t] * DN_D(d.nrm[cd][t], k);
      gb[j0 + k] = gj;
    }
  }
}
// (forest_engine.ccd_coop) the batch's deferred box-box contacts (mine: this lane's contact (pair, slot), its record
// crec and cotangent in cw[12..18]): their pose cotangents on the bodies into cw[12..23] (the pair's leader lane: the
// pair's sum; its other contacts' lanes: zeros)
__device__ __forceinline__ void cc_coop(Adj& s, int lane, bool mine, int pair, int slot, float* cw, const CyRec& crec) {
  if (!__ballot_sync(0xffffffffu, mine)) return;  // (uniform)
  const unsigned grp = __match_any_sync(0xffffffffu, mine ? pair : -1 - lane);  // this lane's pair's lanes
  const bool lead = mine && __ffs(grp) - 1 == lane;
  const unsigned L = __ballot_sync(0xffffffffu, lead);
  const int nj = CC_NJ * __popc(L), u0 = __popc(L & ((1u << lane) - 1u));  // jobs; this leader's pair index
  float gb[12];
#pragma unroll
  for (int k = 0; k < 12; k++) gb[k] = 0.f;
  bool bad = false;  // (the leader: a job of its pair failed the record's check)
#pragma unroll 1
  for (int i0 = 0; i0 < nj; i0 += 32) {  // (uniform)
    const int t = i0 + lane, u = t < nj ? t / CC_NJ : 0, jj = t - CC_NJ * u;
    int c = 0;  // the leader lane of pair u: the last lane c with leaders(lanes < c) <= u
#pragma unroll
    for (int b = 16; b; b >>= 1)
      if (__popc(L & ((1u << (c + b)) - 1u)) <= u) c += b;
    const unsigned gm = __shfl_sync(0xffffffffu, grp, c);
    const int pr = __shfl_sync(0xffffffffu, pair, c);
    CyRec r;  // the leader's record (the fields cy_pair reads with kind 1)
    r.kind = __shfl_sync(0xffffffffu, crec.kind, c);
    r.n = __shfl_sync(0xffffffffu, crec.n, c);
#pragma unroll
    for (int k = 0; k < 3; k++) {
      r.v1[k] = __shfl_sync(0xffffffffu, crec.v1[k], c);
      r.v2[k] = __shfl_sync(0xffffffffu, crec.v2[k], c);
      r.fv[k] = __shfl_sync(0xffffffffu, crec.fv[k], c);
      r.c[k] = __shfl_sync(0xffffffffu, crec.c[k], c);
    }
    r.v1[3] = r.v2[3] = 0;
    int sl[4], nm = 0;
    float cr[4][7], cb[4][13];
    unsigned mm = gm;
#pragma unroll
    for (int i = 0; i < 4; i++) {  // the pair's contacts in lane order (CY_SINK_MAX 4)
      const int src = mm ? __ffs(mm) - 1 : c;
      sl[i] = __shfl_sync(0xffffffffu, slot, src);
#pragma unroll
      for (int k = 0; k < 7; k++) cr[i][k] = __shfl_sync(0xffffffffu, cw[12 + k], src);
      cb[i][0] = __shfl_sync(0xffffffffu, crec.dist, src);  // (its bits: the forward's contact)
#pragma unroll
      for (int k = 0; k < 3; k++) cb[i][1 + k] = __shfl_sync(0xffffffffu, crec.pos[k], src);
#pragma unroll
      for (int k = 0; k < 9; k++) cb[i][4 + k] = __shfl_sync(0xffffffffu, crec.fr[k], src);
      if (mm) { nm++; mm &= mm - 1u; }
    }
    float g[CY_NT];
    bool ok = true;
    if (t < nj) ok = cc_dir(s.gx, s.gq, pr, jj, &r, nm, sl, cr, cb, g);
    else
      for (int k = 0; k < CY_NT; k++) g[k] = 0.f;
    const unsigned okm = __ballot_sync(0xffffffffu, ok);
#pragma unroll
    for (int k = 0; k < 12; k++) {  // direction k of this leader's pair: job CC_NJ u0 + k / CY_NT, its tangent k % CY_NT
      const int sk = CC_NJ * u0 + k / CY_NT - i0;
      const float x = __shfl_sync(0xffffffffu, g[k % CY_NT], sk & 31);
      if (lead && sk >= 0 && sk < 32) {
        gb[k] = x;
        bad = bad || !((okm >> sk) & 1u);
      }
    }
  }
  bad = __shfl_sync(0xffffffffu, bad, __ffs(grp) - 1);  // (the pair's leader's verdict)
  if (__ballot_sync(0xffffffffu, mine && bad)) {  // (rare) a record that does not reproduce the forward's contact:
    // each of the pair's contacts takes cyc_adj's full replay passes on its lane
    if (mine && bad) cc_full(s.gx, s.gq, pair, slot, cw + 12, gb);
  }
  if (mine) {  // adj_contact_one's geom_to_body_vjp, once per pair
    const int g1 = PG1(pair), g2 = PG2(pair), b1 = c_gbody[g1], b2 = c_gbody[g2];
    float fg[12];
    for (int t = 0; t < 12; t++) fg[t] = 0.f;
    if (lead || bad) {
      if (b1) geom_to_body_vjp(s.gx[g1], s.X[b1], gb, gb + 3, fg, fg + 3);
      geom_to_body_vjp(s.gx[g2], s.X[b2], gb + 6, gb + 9, fg + 6, fg + 9);
    }
    for (int t = 0; t < 12; t++) cw[12 + t] = fg[t];
  }
}
#endif

"""

G2B = ("    if (b1) geom_to_body_vjp(s.gx[g1], s.X[b1], gb, gb + 3, fg, fg + 3);\n"
       "    geom_to_body_vjp(s.gx[g2], s.X[b2], gb + 6, gb + 9, fg + 6, fg + 9);\n")
G2B_NEW = ("#if CC_COOP_ON\n"
           "    if (CC_COOP_PAIR(pair) && crec.kind == 1) {  // the record for cc_coop (forest_engine.ccd_coop)\n"
           "      for (int t = 0; t < 4; t++) fg[t] = cot[t];\n"
           "      make_frame_vjp(fr, cot + 4, fg + 4);\n"
           "    } else\n"
           "#endif\n"
           "    {\n" + G2B + "    }\n")

ONE_HEAD = ("__device__ __forceinline__ float adj_contact_one(Adj& s, int c, int pair, int slot, unsigned zone, float* cs,\n"
            "                                                 float* dbg) {\n")
REC_DECL = ("  CyRec crec;  // (g1_engine.cylccd: the replay's record for the VJP's call)\n"
            "  crec.kind = -1;\n")
CALL = "adj_contact_one(s, c, pair, slot, zone, cw, dbg)"
BATCH = "    bool bad = false;\n    int cb = -1;"
SKIP = "    const unsigned skip = __ballot_sync(0xffffffffu, on && !use), mis = __ballot_sync(0xffffffffu, bad);\n"


def apply(src):
    """src (an adjoint text after g1_engine.cylccd.apply_rec, with model_hooks' ADJ_CCD branch for box-box) with the
    pass: HEAD and DEVICE ahead of adj_contacts, DEFER in the branch, the record hoisted to adj_contacts' batch, the
    deferred contact's record in adj_contact_one and cc_coop after the batch's calls."""
    def one(s, a, b):
        if s.count(a) != 1:
            raise RuntimeError(f"ccd_coop.apply: anchor found {s.count(a)} times: {a[:70]!r}")
        return s.replace(a, b)
    ver = "    if (rec->kind > 0) {  // the record's contact `slot` over RF: the forward's bits, else the full replay\n"
    src = one(src, ver, "#if CC_COOP_ON  // (box-box with adj_contact_one's record: cc_dir checks it, forest_engine.ccd_coop)\n"
                        "    if (rec->kind > 0 && !(!cot && rec != &lrec && t1 == 3 && t2 == 3 && BOX_BOX_CCD)) {\n#else\n"
                        + ver + "#endif\n")
    cyc = "    if (cyc_adj(CYT(t1), p1, q1, sz1, CYT(t2), p2, q2, sz2, slot, cot, gb, dist, pos, fr, crec) < 0) return -1;\n"
    src = one(src, cyc, DEFER + cyc)
    fn = "// the forward's contact `slot` of a convex pair with a cylinder (dist, pos, fr: its bits) and, with cot, its VJP"
    src = one(src, fn, HEAD + fn)
    src = one(src, ONE_HEAD, ONE_HEAD.replace("float* dbg) {", "float* dbg CC_REC_PARAM) {"))
    src = one(src, REC_DECL, "#if !CC_COOP_ON\n" + REC_DECL + "#endif\n")
    src = one(src, G2B, G2B_NEW)
    src = one(src, CALL, "adj_contact_one(s, c, pair, slot, zone, cw, dbg CC_REC_ARG)")
    src = one(src, BATCH, "#if CC_COOP_ON\n    CyRec crec;  // (forest_engine.ccd_coop: adj_contact_one's record, read by "
                          "cc_coop)\n    crec.kind = -1;\n#endif\n" + BATCH)
    src = one(src, SKIP, "#if CC_COOP_ON\n    cc_coop(s, lane, cb >= 0 && CC_COOP_PAIR(pair) && crec.kind == 1, pair, slot, "
                         "cw, crec);  // (forest_engine.ccd_coop)\n#endif\n" + SKIP)
    adj = "template <bool DBG>\n__device__ __noinline__ void adj_contacts(Adj& s, int lane, float* dbg_in) {\n"
    return one(src, adj, DEVICE + adj)

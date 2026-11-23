"""Edits of the kernels' text for models outside the G1's structure (nv < 32, more than 3 children per body, more
than 32 bodies, box pairs)."""
from nablaengine.g1_engine import source

ENGINE_CHILD = (
    ("DKC int c_parent[NB], c_depth[NB], c_qadr[NB], c_dadr[NB], c_nchild[NB], c_child[NB * 3], c_nanc[NB],",
     "DKC int c_parent[NB], c_depth[NB], c_qadr[NB], c_dadr[NB], c_nchild[NB], c_child[NB * NCH], c_nanc[NB],", 1),
    ("c_child[3 * b + i]", "c_child[NCH * b + i]", 3),
)
KIN_CHILD = (
    ("  for (int i = 0; i < 3; i++) ch[i] = i < n ? c_child[3 * b + i] : 0;",
     "  for (int i = 0; i < NCH; i++) ch[i] = i < n ? c_child[NCH * b + i] : 0;", 1),
    ("  int ch[3];", "  int ch[NCH];", 2),
)
ENGINE_SMALL_NV = (
    ("  const int r0 = lane, r1 = lane + 32;\n  const bool has1 = r1 < NV;",
     "  const int r0 = min(lane, NV - 1), r1 = lane + 32;  // lanes >= NV repeat row NV - 1 (model_hooks)\n"
     "  const bool has1 = r1 < NV;", 2),
    ("  const int r0 = lane, r1 = min(lane + 32, NV - 1);",
     "  const int r0 = min(lane, NV - 1), r1 = min(lane + 32, NV - 1);  // (model_hooks)", 1),
)

ENGINE_ADJ_SCOPE = (
    ('static_assert(!HAS_BOX && NFLOSS == 0, "the contact adjoint is for the mjlab model: no boxes, no frictionloss rows");',
     '// (model_hooks: plane-box contacts and frictionloss rows in the contact adjoint)', 1),
    ('static_assert(HAS_BOX == 0 && NFLOSS == 0, "the adjoint is for the mjlab model (no boxes, no frictionloss rows)");',
     '// (model_hooks: plane-box contacts and frictionloss rows in the adjoint)', 1),
)
ENGINE_PLANE_BOX = (
    ("    if (t2 != 2) return -1;\n",
     """    if (t2 == 3) {  // plane-box corner `slot` (model_hooks): the forward's collide_pair arithmetic, then its VJP
      float cl[3] = {(slot & 1) ? sz2[0] : -sz2[0], (slot & 2) ? sz2[1] : -sz2[1], (slot & 4) ? sz2[2] : -sz2[2]}, cw[3];
      qrot(q2, cl, cw);
      float cd = dot3(p2, n) - dot3(p1, n);
      *dist = cd + dot3(n, cw);
      for (int k = 0; k < 3; k++) pos[k] = cw[k] + p2[k] - 0.5f * n[k] * (*dist);
      make_frame(n, fr);
      if (!(-0.5f < n[1] && n[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
      if (cot) {  // c = x + cw: xbar = cbar, tbar = cw x cbar; cbar = pbar - n (n . pbar) / 2 + n dbar
        float pn = dot3(n, cot + 1);
        for (int k = 0; k < 3; k++) gb[6 + k] = cot[1 + k] + n[k] * (cot[0] - 0.5f * pn);
        cross(cw, gb + 6, gb + 9);
      }
      return br;
    }
    if (t2 != 2) return -1;\n""", 1),
)

ENGINE_DAMPING = (
    ("    c_inert[NB * 3], c_grav[3], c_armature[NV], c_kp[NV], c_kv[NV], c_frclim[NV], c_misc[2];\n",
     "    c_inert[NB * 3], c_grav[3], c_armature[NV], c_kp[NV], c_kv[NV], c_frclim[NV], c_misc[2];\n"
     "DKC float c_damp[NV];  // joint damping (model_hooks)\n", 1),
    ("      s.qsm[i] = act - s.qsm[i];",
     "      s.qsm[i] = act - c_damp[i] * s.qvel[i] - s.qsm[i];  // the passive damping force (model_hooks)", 1),
    ("{ s.L[TRI(i, i)] += h * KVI(i); s.tmp[i] = s.Ma[i]; }",
     "{ s.L[TRI(i, i)] += h * (KVI(i) + c_damp[i]); s.tmp[i] = s.Ma[i]; }", 1),
    ("{ s.abar[i] -= h * KVI(i) * s.y[i]; s.ny[i] = -s.y[i]; }",
     "{ s.abar[i] -= h * (KVI(i) + c_damp[i]) * s.y[i]; s.ny[i] = -s.y[i]; }", 1),
    ("{ float x = s.did[i] - h * KVI(i) * s.a[i]; acc += x * x; }",
     "{ float x = s.did[i] - h * (KVI(i) + c_damp[i]) * s.a[i]; acc += x * x; }", 1),
    ("    s.ubar[i - 6] += kl;\n  }\n  __syncwarp();\n",
     "    s.ubar[i - 6] += kl;\n  }\n  __syncwarp();\n"
     "  for (int i = lane; i < NV; i += 32) s.Pv[i] += c_damp[i] * s.lam[i];  // joint damping (model_hooks)\n"
     "  __syncwarp();\n", 1),
    ("    float r = s.did[i] + c_armature[i] * s.a[i] - tau - s.jtf[i];",
     "    float r = s.did[i] + c_armature[i] * s.a[i] - tau + c_damp[i] * s.v[i] - s.jtf[i];", 1),
)

ENGINE_FLOSS = (
    ("      const int r = nl + __popc(bal & below);",
     "      const int r = NFLOSS + nl + __popc(bal & below);  // after the frictionloss rows (model_hooks)", 1),
    ("    const int r0 = nlim + c + 3 * (ntri + __popc(b3 & below));  // the contact's first row",
     "    const int r0 = NFLOSS + nlim + c + 3 * (ntri + __popc(b3 & below));  // the contact's first row (model_hooks)",
     1),
    ("""  const unsigned below = (1u << lane) - 1u;
  int nl = 0;
#pragma unroll 1
  for (int i0 = 0; i0 < NV; i0 += 32) {
    const int i = min(i0 + lane, NV - 1);  // clamped so that the loads need no branch (as the forward)""",
     """  const unsigned below = (1u << lane) - 1u;
  // the frictionloss rows (model_hooks): row r = the number of frictionloss dofs before (dof order, the first rows),
  // J = e_i, pos 0, aref = -b v_i (substep_pre), constant D; on its quadratic zone (the tape's bit) the row force
  // -D (a_i + b v_i) gives P_v += D b lam (limit_row_vjp without the position terms); its linear pieces are constant
  int nf = 0;
#pragma unroll 1
  for (int i0 = 0; i0 < NV; i0 += 32) {
    const int i = i0 + lane;
    const bool on = i < NV && c_floss[i] > 0.f;
    const unsigned bal = __ballot_sync(0xffffffffu, on);
    if (on) {
      const int r = nf + __popc(bal & below);
      if ((s.zone[r >> 5] >> (r & 31)) & 1u) {
        float ri[6];
        adj_row_impedance(c_dsolref, c_dsolimp, 0.f, c_dinvw[i], ri);
        s.Pv[i] += ri[4] * ri[1] * s.lam[i];
      }
    }
    nf += __popc(bal);
  }
  int nl = 0;
#pragma unroll 1
  for (int i0 = 0; i0 < NV; i0 += 32) {
    const int i = min(i0 + lane, NV - 1);  // clamped so that the loads need no branch (as the forward)""", 1),
)


ADJ_DEBUG_FLOSS = (
    ("      acc = (q - c_jrange[2 * i] < c_jrange[2 * i + 1] - q ? 1.f : -1.f) * dbg[DBG_LIM_FORCE + i];\n    }\n",
     """      acc = (q - c_jrange[2 * i] < c_jrange[2 * i + 1] - q ? 1.f : -1.f) * dbg[DBG_LIM_FORCE + i];
    }
    if (c_floss[i] > 0.f) {  // its frictionloss row (model_hooks.ADJ_DEBUG_FLOSS): row r = frictionloss dofs before i
      int r = 0;
      for (int j = 0; j < i; j++) r += c_floss[j] > 0.f;
      float ri[6];
      adj_row_impedance(c_dsolref, c_dsolimp, 0.f, c_dinvw[i], ri);
      const float x = s.a[i] + ri[1] * s.v[i];
      acc += ((s.zone[r >> 5] >> (r & 31)) & 1u) ? -ri[4] * x : (x > 0.f ? -c_floss[i] : c_floss[i]);
    }
""", 1),
)


def box_box_hook():
    """(the text of step.py's box_box, its replacement): MuJoCo Warp's native CCD for box-box (nablaengine.ccd,
    ccd_box_box: up to 4 contacts by multicontact) through a CSink into the world's contacts."""
    from nablaengine import ccd
    from nablaengine.g1_engine import step
    src = step.STEP_DEVICE
    head = ("template <class S>\n__device__ void box_box(S& s, int pair, const float* p1, const float* R1, "
            "const float* s1, const float* p2,\n                        const float* R2, const float* s2) {\n")
    i = src.index(head)
    k = src.index("#else\n  for (int i = 0; i < n; i++) {\n    if (!(depth[i] < 0.f)) continue;", i)
    wrapper = """
// two boxes (centres p, rotations R row major, half sizes h) are separated: a separating-axis test on the 15 axes
// with a slack of 1e-6 (+1e-6 on |R|: parallel edges), conservative, so a pair it drops has no contact at zero margin
// (the CCD's result is then empty too: dropping it changes no output)
__device__ bool boxes_separated(const float* p1, const float* R1, const float* h1, const float* p2, const float* R2,
                                const float* h2) {
  const float tw[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
  float t[3], R[3][3], A[3][3];
  for (int i = 0; i < 3; i++) t[i] = R1[i] * tw[0] + R1[3 + i] * tw[1] + R1[6 + i] * tw[2];  // in frame 1
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      R[i][j] = R1[i] * R2[j] + R1[3 + i] * R2[3 + j] + R1[6 + i] * R2[6 + j];
      A[i][j] = fabsf(R[i][j]) + 1e-6f;
    }
  const float slack = 1e-6f;
  for (int i = 0; i < 3; i++)
    if (fabsf(t[i]) > h1[i] + h2[0] * A[i][0] + h2[1] * A[i][1] + h2[2] * A[i][2] + slack) return true;
  for (int j = 0; j < 3; j++)
    if (fabsf(t[0] * R[0][j] + t[1] * R[1][j] + t[2] * R[2][j]) > h1[0] * A[0][j] + h1[1] * A[1][j] + h1[2] * A[2][j] + h2[j] + slack)
      return true;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      const int i1 = (i + 1) % 3, i2 = (i + 2) % 3, j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      const float ra = h1[i1] * A[i2][j] + h1[i2] * A[i1][j], rb = h2[j1] * A[i][j2] + h2[j2] * A[i][j1];
      if (fabsf(t[i2] * R[i1][j] - t[i1] * R[i2][j]) > ra + rb + slack) return true;
    }
  return false;
}
template <class S>
__device__ void box_box(S& s, int pair, const float* p1, const float* R1, const float* s1, const float* p2,
                        const float* R2, const float* s2) {  // (model_hooks: MuJoCo Warp's native CCD, nablaengine.ccd)
  if (boxes_separated(p1, R1, s1, p2, R2, s2)) return;  // (no contact: the CCD is skipped)
  CSink o;
  ccd_box_box(p1, R1, s1, p2, R2, s2, o);
  for (int i = 0; i < o.n; i++) add_contact(s, pair, o.slot[i], o.dist[i], o.pos[i], o.fr[i]);
  if (o.unsupported) atomicOr(&s.unsupported, o.unsupported);
}
"""
    return src[i:src.index("\n}\n", k) + 3], ccd.DEVICE + wrapper


ADJ_BOX_BOX = """  if (t1 == 3 && t2 == 3) {  // box-box (model_hooks): the forward's CCD contact `slot` and its exact VJP
    // (nablaengine.ccd ccd_box_box_vjp: clipped points as plane intersections, EPA vertex-face and edge-edge contacts)
    float R1[9], R2[9];
    quat2mat(q1, R1);
    quat2mat(q2, R2);
    if (!ccd_box_box_vjp(p1, R1, sz1, p2, R2, sz2, slot, cot, gb, dist, pos, fr)) return -1;
    if (!(-0.5f < fr[1] && fr[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
    return br;
  }
"""


ENGINE_NB64 = (
    ("DKC unsigned int c_ancmask[NB];\n", "DKC unsigned long long c_ancmask[NB];  // (model_hooks: NB > 32)\n", 1),
    ("    if (d == L) {\n      float Qp[4], Xp[3], Wp[3], Vp[3], ALp[3], Ap[3], Qb[4], Xb[3], Wb[3], Vb[3], ALb[3], Ab[3];\n",
     "    for (int b = lane; b < NB; b += 32) if (c_depth[b] == L) {  // (model_hooks: NB > 32)\n"
     "      const int p = c_parent[b];\n"
     "      float Qp[4], Xp[3], Wp[3], Vp[3], ALp[3], Ap[3], Qb[4], Xb[3], Wb[3], Vb[3], ALb[3], Ab[3];\n", 1),
    ("  if (b >= 1 && b < NB) {\n    float ip[3], Qb[4], Xb[3], Wb[3], ALb[3], Ab[3], Fb[3], Nb[3], Jb[10];\n",
     "  for (int b = lane; b < NB; b += 32) if (b >= 1) {  // (model_hooks: NB > 32)\n"
     "    float ip[3], Qb[4], Xb[3], Wb[3], ALb[3], Ab[3], Fb[3], Nb[3], Jb[10];\n", 1),
    ("    if (d == L && c_nchild[b] > 0) {\n      int nch = c_nchild[b];\n",
     "    for (int b = lane; b < NB; b += 32) if (c_depth[b] == L && c_nchild[b] > 0) {  // (model_hooks: NB > 32)\n"
     "      int nch = c_nchild[b];\n", 1),
    ("  if (b >= 1 && b < NB) torques(b, s.Q[b], s.F[b], s.N[b], s.qsm);",
     "  for (int b = lane; b < NB; b += 32) if (b >= 1) torques(b, s.Q[b], s.F[b], s.N[b], s.qsm);  // (model_hooks)", 1),
    ("  if (b >= 2 && b < NB) {\n    int j = c_dadr[b], na = c_nanc[b], ad[12];\n",
     "  for (int b = lane; b < NB; b += 32) if (b >= 2) {  // (model_hooks: NB > 32)\n"
     "    int j = c_dadr[b], na = c_nanc[b], ad[12];\n", 1),
    ("support_mask(unsigned x) {", "support_mask(unsigned long long x) {  // (model_hooks: NB > 32)", 1),
    ("    unsigned am1[2], am2[2];\n", "    unsigned long long am1[2], am2[2];  // (model_hooks: NB > 32)\n", 1),
    ("      unsigned m1 = __shfl_sync(0xffffffffu, c < 32 ? am1[0] : am1[1], c & 31);\n"
     "      unsigned m2 = __shfl_sync(0xffffffffu, c < 32 ? am2[0] : am2[1], c & 31);\n",
     "      unsigned long long m1 = __shfl_sync(0xffffffffu, c < 32 ? am1[0] : am1[1], c & 31);  // (model_hooks)\n"
     "      unsigned long long m2 = __shfl_sync(0xffffffffu, c < 32 ? am2[0] : am2[1], c & 31);\n", 1),
    ("  unsigned hb = sup & ~3u;  // the support's hinge bodies (bit 0: the world, 1: the root)\n"
     "  int i = (sup >> 1) & 1u ? 3 : (hb ? __ffs(hb) - 1 + ADJ_HINGE_DOF0 : NV);\n"
     "#pragma unroll 1\n"
     "  for (; i < NV; i = i < 5 ? i + 1 : (hb ? __ffs(hb) - 1 + ADJ_HINGE_DOF0 : NV)) {\n",
     "  unsigned long long hb = sup & ~3ull;  // the support's hinge bodies (model_hooks: NB > 32)\n"
     "  int i = (sup >> 1) & 1u ? 3 : (hb ? __ffsll(hb) - 1 + ADJ_HINGE_DOF0 : NV);\n"
     "#pragma unroll 1\n"
     "  for (; i < NV; i = i < 5 ? i + 1 : (hb ? __ffsll(hb) - 1 + ADJ_HINGE_DOF0 : NV)) {\n", 1),
    ("  const unsigned m1 = c_ancmask[b1], m2 = c_ancmask[b2], sup = m1 ^ m2;",
     "  const unsigned long long m1 = c_ancmask[b1], m2 = c_ancmask[b2], sup = m1 ^ m2;  // (model_hooks: NB > 32)", 1),
    ("      const unsigned m1 = c_ancmask[c_gbody[PG1(pair)]], m2 = c_ancmask[c_gbody[PG2(pair)]];",
     "      const unsigned long long m1 = c_ancmask[c_gbody[PG1(pair)]], m2 = c_ancmask[c_gbody[PG2(pair)]];  // (model_hooks)",
     1),
)


ADJ_NB64 = (
    ('static_assert(NB <= 31, "the contact reduction: lanes 0 and 31 take two of the root rotation dofs");',
     'static_assert(NB <= 63, "the contact reduction: lane 31\'s second slot takes a root rotation dof (model_hooks: NB > 32)");', 1),
    ('  const bool body = lane >= 1 && lane < NB, rot = lane == 0 || lane == 31;\n  const int rb = body ? lane : 1, rk = lane == 0 ? 1 : (lane == 31 ? 2 : 0), di = rb >= 2 ? c_dadr[rb] : 3 + rk;\n',
     "  // (model_hooks: NB > 32) two body slots per lane, slot h: body lane + 32 h; lane 0's slot 0 takes the root\n  // rotation dof 4, lane 31's slot 1 the dof 5 (body 63 never exists), lane 1's slot 0 (body 1) the dof 3\n  bool body[2], rot[2];\n  int rb[2], rk[2], di[2];\n#pragma unroll\n  for (int h = 0; h < 2; h++) {\n    const int bb = lane + 32 * h;\n    body[h] = bb >= 1 && bb < NB;\n    rot[h] = h == 0 ? lane == 0 : lane == 31;\n    rb[h] = body[h] ? bb : 1;\n    rk[h] = rot[h] ? 1 + h : 0;\n    di[h] = rb[h] >= 2 ? c_dadr[rb[h]] : 3 + rk[h];\n  }\n", 1),
    ("    float f[3] = {0.f, 0.f, 0.f}, t[3] = {0.f, 0.f, 0.f}, pt[3] = {0.f, 0.f, 0.f}, pv = 0.f;\n#pragma unroll 1\n    for (int j = 0; j < nb; j++) {  // the batch in contact order\n      const int cj = __shfl_sync(0xffffffffu, cb, j);\n      if (UNIFORM(cj < 0)) continue;\n      const int c1 = cj & 255, c2 = cj >> 8;\n      float w[CS_N];\n#pragma unroll\n      for (int k = 0; k < 12; k++) w[k] = __shfl_sync(0xffffffffu, cw[k], j);\n      const unsigned i2 = (c_ancmask[c2] >> rb) & 1u, i1 = (c_ancmask[c1] >> rb) & 1u;\n      if ((body || rot) && i1 != i2) {  // this lane's dof in the support (contact_dof_vjp)\n        const float sg = i2 ? 1.f : -1.f, vi = sg * s.v[di], ai = sg * s.a[di], li = sg * s.lam[di];\n        const float* z = rb >= 2 ? s.z[rb] : s.zr[rk];\n        float G[3], r[3], wv[3], col[3], u[3], x[3], y[3];\n        for (int k = 0; k < 3; k++) {\n          wv[k] = w[k];\n          G[k] = vi * wv[k] + ai * w[3 + k] + li * w[6 + k];\n          r[k] = w[9 + k] - s.X[rb][k];\n        }\n        cross(z, r, col);\n        pv += sg * dot3(col, wv);\n        cross(z, G, u);\n        cross(r, G, x);\n        cross(z, x, y);\n        for (int k = 0; k < 3; k++) { f[k] += u[k]; t[k] += y[k]; }\n        if (lane == 1)\n          for (int k = 0; k < 3; k++) pt[k] += sg * wv[k];  // the root translations: columns e_k, no pose term\n      }\n      if (UNIFORM(c1 != 0)) {  // the floor, body 0, takes no pose cotangent\n#pragma unroll\n        for (int k = 12; k < 18; k++) w[k] = __shfl_sync(0xffffffffu, cw[k], j);\n        if (body && rb == c1)\n          for (int k = 0; k < 3; k++) { f[k] += w[12 + k]; t[k] += w[15 + k]; }\n      }\n#pragma unroll\n      for (int k = 18; k < 24; k++) w[k] = __shfl_sync(0xffffffffu, cw[k], j);\n      if (body && rb == c2)\n        for (int k = 0; k < 3; k++) { f[k] += w[18 + k]; t[k] += w[21 + k]; }\n    }\n    float o[7] = {f[0], f[1], f[2], t[0], t[1], t[2], pv}, o0[7], o31[7];  // root rotations 4, 5: to lane 1\n    for (int k = 0; k < 7; k++) { o0[k] = __shfl_sync(0xffffffffu, o[k], 0); o31[k] = __shfl_sync(0xffffffffu, o[k], 31); }\n    if (body) {\n      if (lane == 1) {\n        for (int k = 0; k < 3; k++) {\n          f[k] += o0[k] + o31[k]; t[k] += o0[3 + k] + o31[3 + k]; s.Pv[k] += pt[k];\n        }\n        s.Pv[4] += o0[6];\n        s.Pv[5] += o31[6];\n      }\n      for (int k = 0; k < 3; k++) { s.fb[rb][k] += f[k]; s.tb[rb][k] += t[k]; }\n      s.Pv[di] += pv;\n    }\n",
     "    // (model_hooks: NB > 32) per slot h: the sums of body rb[h] (or the root rotation dof 3 + rk[h])\n    float f[2][3] = {}, t[2][3] = {}, pt[3] = {0.f, 0.f, 0.f}, pv[2] = {0.f, 0.f};\n#pragma unroll 1\n    for (int j = 0; j < nb; j++) {  // the batch in contact order\n      const int cj = __shfl_sync(0xffffffffu, cb, j);\n      if (UNIFORM(cj < 0)) continue;\n      const int c1 = cj & 255, c2 = cj >> 8;\n      float w[CS_N];\n#pragma unroll\n      for (int k = 0; k < 12; k++) w[k] = __shfl_sync(0xffffffffu, cw[k], j);\n      const unsigned long long a2 = c_ancmask[c2], a1 = c_ancmask[c1];\n#pragma unroll\n      for (int h = 0; h < 2; h++) {\n        const unsigned i2 = (a2 >> rb[h]) & 1u, i1 = (a1 >> rb[h]) & 1u;\n        if ((body[h] || rot[h]) && i1 != i2) {  // this slot's dof in the support (contact_dof_vjp)\n          const float sg = i2 ? 1.f : -1.f, vi = sg * s.v[di[h]], ai = sg * s.a[di[h]], li = sg * s.lam[di[h]];\n          const float* z = rb[h] >= 2 ? s.z[rb[h]] : s.zr[rk[h]];\n          float G[3], r[3], wv[3], col[3], u[3], x[3], y[3];\n          for (int k = 0; k < 3; k++) {\n            wv[k] = w[k];\n            G[k] = vi * wv[k] + ai * w[3 + k] + li * w[6 + k];\n            r[k] = w[9 + k] - s.X[rb[h]][k];\n          }\n          cross(z, r, col);\n          pv[h] += sg * dot3(col, wv);\n          cross(z, G, u);\n          cross(r, G, x);\n          cross(z, x, y);\n          for (int k = 0; k < 3; k++) { f[h][k] += u[k]; t[h][k] += y[k]; }\n          if (h == 0 && lane == 1)\n            for (int k = 0; k < 3; k++) pt[k] += sg * wv[k];  // the root translations: columns e_k, no pose term\n        }\n      }\n      if (UNIFORM(c1 != 0)) {  // the floor, body 0, takes no pose cotangent\n#pragma unroll\n        for (int k = 12; k < 18; k++) w[k] = __shfl_sync(0xffffffffu, cw[k], j);\n#pragma unroll\n        for (int h = 0; h < 2; h++)\n          if (body[h] && rb[h] == c1)\n            for (int k = 0; k < 3; k++) { f[h][k] += w[12 + k]; t[h][k] += w[15 + k]; }\n      }\n#pragma unroll\n      for (int k = 18; k < 24; k++) w[k] = __shfl_sync(0xffffffffu, cw[k], j);\n#pragma unroll\n      for (int h = 0; h < 2; h++)\n        if (body[h] && rb[h] == c2)\n          for (int k = 0; k < 3; k++) { f[h][k] += w[18 + k]; t[h][k] += w[21 + k]; }\n    }\n    // root rotations 4 (lane 0, slot 0) and 5 (lane 31, slot 1): to lane 1\n    float o0[7], o31[7];\n    {\n      const float a[7] = {f[0][0], f[0][1], f[0][2], t[0][0], t[0][1], t[0][2], pv[0]};\n      const float c[7] = {f[1][0], f[1][1], f[1][2], t[1][0], t[1][1], t[1][2], pv[1]};\n      for (int k = 0; k < 7; k++) { o0[k] = __shfl_sync(0xffffffffu, a[k], 0); o31[k] = __shfl_sync(0xffffffffu, c[k], 31); }\n    }\n#pragma unroll\n    for (int h = 0; h < 2; h++)\n      if (body[h]) {\n        if (h == 0 && lane == 1) {\n          for (int k = 0; k < 3; k++) {\n            f[0][k] += o0[k] + o31[k]; t[0][k] += o0[3 + k] + o31[3 + k]; s.Pv[k] += pt[k];\n          }\n          s.Pv[4] += o0[6];\n          s.Pv[5] += o31[6];\n        }\n        for (int k = 0; k < 3; k++) { s.fb[rb[h]][k] += f[h][k]; s.tb[rb[h]][k] += t[h][k]; }\n        s.Pv[di[h]] += pv[h];\n      }\n", 1),
    ('    if (d == L)\n      fwd_body(b, s.Q[p], s.X[p], s.W[p], s.V[p], s.AL[p], s.A[p], s.q, s.v, s.a, s.Q[b], s.X[b], s.W[b], s.V[b],\n               s.AL[b], s.A[b]);\n',
     '    for (int b = lane; b < NB; b += 32)  // (model_hooks: NB > 32)\n      if (c_depth[b] == L) {\n        const int p = c_parent[b];\n        fwd_body(b, s.Q[p], s.X[p], s.W[p], s.V[p], s.AL[p], s.A[p], s.q, s.v, s.a, s.Q[b], s.X[b], s.W[b], s.V[b],\n                 s.AL[b], s.A[b]);\n      }\n', 1),
    ('  const int i = b >= 2 && b < NB ? c_dadr[b] : 0;\n  float r[3];\n  for (int k = 0; k < 3; k++) r[k] = b < NB ? s.X[b][k] - s.X[p][k] : 0.f;\n  const float sa = s.at[i], sl = lam[i], sy = ny[i];\n  for (int L = 1; L <= MAXD; L++) {\n    if (d == L) {\n      if (b >= 2) qrot(',
     '  for (int L = 1; L <= MAXD; L++) {\n    for (int b = lane; b < NB; b += 32) if (c_depth[b] == L) {  // (model_hooks: NB > 32)\n      const int p = c_parent[b], i = b >= 2 ? c_dadr[b] : 0;\n      float r[3];\n      for (int k = 0; k < 3; k++) r[k] = s.X[b][k] - s.X[p][k];\n      const float sa = s.at[i], sl = lam[i], sy = ny[i];\n      if (b >= 2) qrot(', 1),
    ('  if (b >= 1 && b < NB) {\n    float ip[3];\n    body_ipos(b, s.com, ip);\n    body_local_vjp(',
     '  for (int b = lane; b < NB; b += 32) if (b >= 1) {  // (model_hooks: NB > 32)\n    float ip[3];\n    body_ipos(b, s.com, ip);\n    body_local_vjp(', 1),
    ('    if (d == L) {\n      float bW[3], bAL[3], bA[3], bAL2[3], bA2[3], bWl[3], bVl[3], bWy[3], bVy[3], F[3], N[3], t[3], u[3];\n',
     '    for (int b = lane; b < NB; b += 32) if (c_depth[b] == L) {  // (model_hooks: NB > 32)\n      const int p = c_parent[b];\n      float bW[3], bAL[3], bA[3], bAL2[3], bA2[3], bWl[3], bVl[3], bWy[3], bVy[3], F[3], N[3], t[3], u[3];\n', 1),
)

ADJ_DEBUG_NB64 = (
    ("  if (b >= 1 && b < NB)\n    for (int k = 0; k < 3; k++) { s.dF[b][k] = s.bVl[b][k]; s.dN[b][k] = s.bWl[b][k]; }\n",
     "  for (int b = lane; b < NB; b += 32)  // (model_hooks: NB > 32)\n    if (b >= 1)\n"
     "      for (int k = 0; k < 3; k++) { s.dF[b][k] = s.bVl[b][k]; s.dN[b][k] = s.bWl[b][k]; }\n", 1),
    ("    if (d == L)\n      for (int i = 0; i < c_nchild[b]; i++) {\n",
     "    for (int b = lane; b < NB; b += 32) if (c_depth[b] == L)  // (model_hooks: NB > 32)\n"
     "      for (int i = 0; i < c_nchild[b]; i++) {\n", 1),
    ("  if (b >= 1 && b < NB) torques(b, s.Q[b], s.dF[b], s.dN[b], s.did);\n",
     "  for (int b = lane; b < NB; b += 32) if (b >= 1) torques(b, s.Q[b], s.dF[b], s.dN[b], s.did);  // (model_hooks)\n", 1),
)


def kin_nb64_hooks():
    from nablaengine.g1_engine import kinematics
    src = kinematics.KIN_DEVICE

    def span(start, end, keep_end=False):
        a = src.index(start)
        e = src.index(end, a)
        assert src.count(start) == 1
        return src[a:e + (len(end) if keep_end else 0)]
    return ((span("template <class S>\n__device__ __forceinline__ void kin_state(", "\n}\n", True)[:-1], KIN_STATE_NB64, 1),
            (span("  const int b = lane;\n  float o[6]", "  float* oq = qn"), KIN_OUT_NB64, 1),
            (span("  const int b = lane, d = b < NB ? c_depth[b] : -1, p = b < NB ? c_parent[b] : 0;\n  const bool body",
                  "  const float* cq = gqn"), KIN_VJP_NB64, 1))


KIN_STATE_NB64 = r'''template <class S>
__device__ __forceinline__ void kin_state(S& s, int lane) {  // (model_hooks.KIN_NB64: bodies lane and lane + 32)
  if (lane == 0) {
    s.Q[0][0] = 1.f; s.Q[0][1] = s.Q[0][2] = s.Q[0][3] = 0.f;
    for (int k = 0; k < 3; k++) s.X[0][k] = s.W[0][k] = s.V[0][k] = 0.f;
  }
  __syncwarp();
  for (int L = 1; L <= MAXD; L++) {
    for (int b = lane; b < NB; b += 32) if (c_depth[b] == L) {
      const int p = c_parent[b];
      float Qp[4], Xp[3], Wp[3], Vp[3], Qb[4], Xb[3], Wb[3], Vb[3], ALb[3], Ab[3], q0[4];
      const float z3[3] = {0.f, 0.f, 0.f};
      for (int k = 0; k < 4; k++) Qp[k] = s.Q[p][k];
      for (int k = 0; k < 3; k++) { Xp[k] = s.X[p][k]; Wp[k] = s.W[p][k]; Vp[k] = s.V[p][k]; }
      if (L == 1) {  // body 1, the free root
        fwd_body(1, Qp, Xp, Wp, Vp, z3, z3, s.qpos, s.qvel, s.qvel, Qb, Xb, Wb, Vb, ALb, Ab);
      } else {
        float qb[4], pb[3], ax[3], ha[4];
        for (int k = 0; k < 4; k++) qb[k] = c_quat[4 * b + k];
        for (int k = 0; k < 3; k++) { pb[k] = c_pos[3 * b + k]; ax[k] = c_axis[3 * b + k]; }
        const float qd = s.qvel[c_dadr[b]];
        fwd_hinge_rot(s.qpos[c_qadr[b]], ax, ha);
        fwd_hinge_pos(Qp, Xp, qb, pb, q0, Xb);
        fwd_hinge_rest(q0, ha, ax, qd, 0.f, Xp, Wp, Vp, z3, z3, Qb, Xb, Wb, Vb, ALb, Ab);
      }
      for (int k = 0; k < 4; k++) s.Q[b][k] = Qb[k];
      for (int k = 0; k < 3; k++) { s.X[b][k] = Xb[k]; s.W[b][k] = Wb[k]; s.V[b][k] = Vb[k]; }
    }
    __syncwarp();
  }
  for (int b = lane; b < NB; b += 32) if (b >= 1) {
    float ip[3], c[3];
    body_ipos(b, s.com, ip);
    qrot(s.Q[b], ip, c);
    const float Mb = c_mass[b];
    for (int k = 0; k < 3; k++) s.C[b][k] = (s.X[b][k] + c[k]) * Mb;
    s.Mt[b] = Mb;
  }
  __syncwarp();
  for (int L = MAXD - 1; L >= 1; L--) {
    for (int b = lane; b < NB; b += 32) if (c_depth[b] == L) {
      int ch[3];
      const int nch = kin_children(b, ch);
      float S3[3] = {s.C[b][0], s.C[b][1], s.C[b][2]}, Mb = s.Mt[b];
#pragma unroll
      for (int i = 0; i < 3; i++)
        if (i < nch) {
          for (int k = 0; k < 3; k++) S3[k] += s.C[ch[i]][k];
          Mb += s.Mt[ch[i]];
        }
      for (int k = 0; k < 3; k++) s.C[b][k] = S3[k];
      s.Mt[b] = Mb;
    }
    __syncwarp();
  }
  for (int b = lane; b < NB; b += 32)
    if (b >= 1)
      for (int k = 0; k < 3; k++) s.C[b][k] = s.C[b][k] / s.Mt[b];
  __syncwarp();
  if (lane == 0) {
    for (int k = 0; k < 3; k++) s.C[0][k] = s.C[1][k];
    s.Mt[0] = s.Mt[1];
  }
  __syncwarp();
}
'''

KIN_OUT_NB64 = r'''  float o[2][6] = {};  // cvel_b = [W_b; V_b + W_b x (C_1 - X_b)], body 0: zeros (model_hooks.KIN_NB64)
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int b = lane + 32 * h;
    if (b >= 1 && b < NB) {
      float r[3], u[3];
      for (int k = 0; k < 3; k++) r[k] = s.C[1][k] - s.X[b][k];
      cross(s.W[b], r, u);
      for (int k = 0; k < 3; k++) { o[h][k] = s.W[b][k]; o[h][3 + k] = s.V[b][k] + u[k]; }
    }
  }
  __syncwarp();  // (cv aliases W, V)
#pragma unroll
  for (int h = 0; h < 2; h++)
    if (lane + 32 * h < NB)
      for (int k = 0; k < 6; k++) s.cv[lane + 32 * h][k] = o[h][k];
'''

KIN_VJP_NB64 = r'''  // (model_hooks.KIN_NB64: bodies lane and lane + 32; F, N hold the bodies' f bar, tau bar and rb the subtree sums'
  // cotangents S bar until the sweeps)
  const float C1[3] = {s.C[1][0], s.C[1][1], s.C[1][2]};
  float gr[3] = {0.f, 0.f, 0.f};
  for (int b = lane; b < NB; b += 32) if (b >= 1) {
    const size_t row = (size_t)w * NB + b;
    float wb[3], vb[3], r[3], t[3], g[3], tb[3] = {0.f, 0.f, 0.f};
    for (int k = 0; k < 3; k++) { wb[k] = gcvel[row * 6 + k]; vb[k] = gcvel[row * 6 + 3 + k]; r[k] = C1[k] - s.X[b][k]; }
    cross(r, vb, t);
    for (int k = 0; k < 3; k++) { s.bW[b][k] = wb[k] + t[k]; s.bV[b][k] = vb[k]; }
    cross(vb, s.W[b], g);
    for (int k = 0; k < 3; k++) { gr[k] += g[k]; s.F[b][k] = gxpos[row * 3 + k] - g[k]; s.rb[b][k] = gsc[row * 3 + k]; }
    if (b >= 2) {
      const float* q = s.Q[b];  // vec(g q*)
      const float g0 = gxquat[row * 4], g1 = gxquat[row * 4 + 1], g2 = gxquat[row * 4 + 2], g3 = gxquat[row * 4 + 3];
      tb[0] = 0.5f * (-g0 * q[1] + g1 * q[0] - g2 * q[3] + g3 * q[2]);
      tb[1] = 0.5f * (-g0 * q[2] + g1 * q[3] + g2 * q[0] - g3 * q[1]);
      tb[2] = 0.5f * (-g0 * q[3] - g1 * q[2] + g2 * q[1] + g3 * q[0]);
    }
    for (int k = 0; k < 3; k++) s.N[b][k] = tb[k];
  }
  float gc1[3];  // C_1 bar from the cvel reference point (every lane: the same sum)
  for (int k = 0; k < 3; k++) gc1[k] = warp_sum(gr[k]);
  __syncwarp();
  for (int b = lane; b < NB; b += 32)
    if (b >= 1)
      for (int k = 0; k < 3; k++)
        s.rb[b][k] = (b == 1 ? s.rb[b][k] + gsc[(size_t)w * NB * 3 + k] + gc1[k] : s.rb[b][k]) / s.Mt[b];
  __syncwarp();  // (every lane has read C_1: G overwrites C)
  for (int L = 1; L <= MAXD; L++) {
    for (int b = lane; b < NB; b += 32)
      if (c_depth[b] == L)
        for (int k = 0; k < 3; k++) s.G[b][k] = s.rb[b][k] + (b >= 2 ? s.G[c_parent[b]][k] : 0.f);
    __syncwarp();
  }
  for (int b = lane; b < NB; b += 32) if (b >= 1) {  // xipos_b = X_b + c_b, c_b = R_b ipos_b: f bar += g, tau bar += c x g
    float ip[3], c[3], g[3], t[3];
    body_ipos(b, s.com, ip);
    qrot(s.Q[b], ip, c);
    const float m = c_mass[b];
    for (int k = 0; k < 3; k++) { g[k] = m * s.G[b][k]; s.F[b][k] += g[k]; }
    cross(c, g, t);
    for (int k = 0; k < 3; k++) s.N[b][k] += t[k];
  }
  __syncwarp();
  // leaf to root: the twists' recursion reversed with the pose sweep (as below for NB <= 32)
  for (int L = MAXD; L >= 1; L--) {
    for (int b = lane; b < NB; b += 32) if (c_depth[b] == L) {
      const int p = c_parent[b];
      const size_t row = (size_t)w * NB + b;
      int ch[3];
      const int nch = kin_children(b, ch);
      float bW[3], bV[3], F[3], N[3], t[3], u[3];
      for (int k = 0; k < 3; k++) { bW[k] = s.bW[b][k]; bV[k] = s.bV[b][k]; F[k] = s.F[b][k]; N[k] = s.N[b][k]; }
#pragma unroll
      for (int i = 0; i < 3; i++) {
        if (i >= nch) break;
        const int c = ch[i];
        float r[3];
        for (int k = 0; k < 3; k++) r[k] = s.X[c][k] - s.X[b][k];
        cross(r, s.bV[c], t);  // V_c = V_b + W_b x r_c
        cross(r, s.F[c], u);
        for (int k = 0; k < 3; k++) {
          bW[k] += s.bW[c][k] + t[k]; bV[k] += s.bV[c][k];
          F[k] += s.F[c][k] - s.rb[c][k]; N[k] += s.N[c][k] + u[k];
        }
      }
      if (b >= 2) {
        const int i = c_dadr[b];
        const float qd = s.qvel[i];
        float rb[3], z[3];
        qrot(s.Q[b], c_axis + 3 * b, z);
        s.gv[i] = dot3(z, bW);
        for (int k = 0; k < 3; k++) u[k] = qd * bW[k];
        cross(z, u, t);  // z = R_b axis rotates with the body
        cross(bV, s.W[p], rb);  // V_b = V_p + W_p x r_b
        for (int k = 0; k < 3; k++) { N[k] += t[k]; F[k] += rb[k]; s.rb[b][k] = rb[k]; }
        s.gq[c_qadr[b]] = dot3(z, N);
      } else {  // the root: V_1 = qvel[0:3], W_1 = R_1 qvel[3:6]; X_1 = qpos[0:3], Q_1 = normalize(qpos[3:7])
        float l[3];
        qrot_inv(s.Q[1], bW, l);
        for (int k = 0; k < 3; k++) { s.gv[k] = bV[k]; s.gv[3 + k] = l[k]; s.gq[k] = F[k]; }
        cross(s.W[1], bW, t);
        for (int k = 0; k < 3; k++) N[k] += t[k];
        const float n0[4] = {0.f, N[0], N[1], N[2]}, *u4 = s.Q[1], *q = s.qpos + 3;
        float a[4];
        qmul(n0, u4, a);  // dL/dQ_1 (tangential) = 2 [0; N] Q_1, plus the direct cotangents of Q_1
        for (int k = 0; k < 4; k++) a[k] = 2.f * a[k] + gxquat[row * 4 + k] + gqn[(size_t)w * NQ + 3 + k];
        const float nq = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        const float ua = u4[0] * a[0] + u4[1] * a[1] + u4[2] * a[2] + u4[3] * a[3];
        for (int k = 0; k < 4; k++) s.gq[3 + k] = (a[k] - u4[k] * ua) / nq;  // normalize's VJP
      }
      for (int k = 0; k < 3; k++) { s.bW[b][k] = bW[k]; s.bV[b][k] = bV[k]; s.F[b][k] = F[k]; s.N[b][k] = N[k]; }
    }
    __syncwarp();
  }
'''


TERRAIN_BOX_GEOMS = (
    ("""      if (dist < 0.f) { make_frame(n, fr); add_contact(s, pair, k | e << 8, dist, pos, fr); }
    }
  }
""", """      if (dist < 0.f) { make_frame(n, fr); add_contact(s, pair, k | e << 8, dist, pos, fr); }
    }
  } else if (c_gtype[g] == 3) {  // a box: native CCD (model_hooks.TERRAIN_BOX_GEOMS)
    float R2[9];
    quat2mat(s.gq[g], R2);
    if (boxes_separated(b, b + 3, b + 12, p, R2, sz)) return;
    CSink o;
    ccd_box_box(b, b + 3, b + 12, p, R2, sz, o);
    for (int i = 0; i < o.n; i++) add_contact(s, pair, o.slot[i] | e << 8, o.dist[i], o.pos[i], o.fr[i]);
    if (o.unsupported) atomicOr(&s.unsupported, o.unsupported);
  }
""", 1),
    ("  const int gt = c_gtype[g], npts = gt == 2 ? TERRAIN_CAP_PTS : 1;",
     "  const int gt = c_gtype[g], npts = gt == 2 ? TERRAIN_CAP_PTS : (gt == 3 ? 8 : 1);  // (a box: its corners)", 1),
    ("""  for (int i = 0; i < 3; i++) q[i] += w * sz[1] * al[i];
  float h, n[3];
  if (!hfield_surface(hh + 12, g_ter.hfd + hi[2], hi[0], hi[1], q[0], q[1], &h, n)) return false;
  *dist = (q[2] - h) * n[2] - sz[0];
  float posl[3];
  for (int i = 0; i < 3; i++) posl[i] = q[i] - (sz[0] + 0.5f * (*dist)) * n[i];""",
     """  for (int i = 0; i < 3; i++) q[i] += w * sz[1] * al[i];
  float rad = sz[0];
  if (gt == 3) {  // a box: corner k (model_hooks.TERRAIN_BOX_GEOMS), radius 0
    float cl[3] = {(k & 1) ? sz[0] : -sz[0], (k & 2) ? sz[1] : -sz[1], (k & 4) ? sz[2] : -sz[2]}, cw[3], ch[3];
    qrot(gq, cl, cw);
    mtv3(R, cw, ch);
    for (int i = 0; i < 3; i++) q[i] += ch[i];
    rad = 0.f;
  }
  float h, n[3];
  if (!hfield_surface(hh + 12, g_ter.hfd + hi[2], hi[0], hi[1], q[0], q[1], &h, n)) return false;
  *dist = (q[2] - h) * n[2] - rad;
  float posl[3];
  for (int i = 0; i < 3; i++) posl[i] = q[i] - (rad + 0.5f * (*dist)) * n[i];""", 1),
    ("""  float n[3], c[3] = {p2[0], p2[1], p2[2]}, axis[3] = {0.f, 0.f, 0.f}, w = 0.f, qp[3];
""", """  if (gt == 3 && box) {  // a box against a terrain box: the CCD contact `k` and its VJP (model_hooks.TERRAIN_BOX_GEOMS)
    const float* b = ter_box(e);
    float R2[9], g12[12];
    quat2mat(q2, R2);
    if (!ccd_box_box_vjp(b, b + 3, b + 12, p2, R2, sz, k, cot, g12, dist, pos, fr)) return -1;
    if (cot)
      for (int t = 0; t < 6; t++) gb[6 + t] = g12[6 + t];
    return (-0.5f < fr[1] && fr[1] < 0.5f) ? 0 : ADJ_BR_FRAME_Z;
  }
  float n[3], c[3] = {p2[0], p2[1], p2[2]}, axis[3] = {0.f, 0.f, 0.f}, w = 0.f, qp[3];
""", 1),
    ("""    if (!terrain_hfield_point(e - nbox, gt, sz, p2, q2, k, dist, pos, n)) return -1;
""", """    if (!terrain_hfield_point(e - nbox, gt, sz, p2, q2, k, dist, pos, n)) return -1;
    if (gt == 3) {  // the box corner k, a material point
      float cl[3] = {(k & 1) ? sz[0] : -sz[0], (k & 2) ? sz[1] : -sz[1], (k & 4) ? sz[2] : -sz[2]}, cw[3];
      qrot(q2, cl, cw);
      for (int t = 0; t < 3; t++) c[t] = p2[t] + cw[t];
    }
""", 1),
    ("""  for (int t = 0; t < 3; t++) gb[6 + t] = cb[t];
  if (gt == 2) {""", """  for (int t = 0; t < 3; t++) gb[6 + t] = cb[t];
  if (gt == 3) {  // the box corner c = x + R c_l: tbar = (c - x) x cbar
    float rc[3] = {c[0] - p2[0], c[1] - p2[1], c[2] - p2[2]};
    cross(rc, cb, gb + 9);
  }
  if (gt == 2) {""", 1),
)


def terrain_box_geom_hooks(m):
    """TERRAIN_BOX_GEOMS, with MuJoCo Warp's box-box CCD for a model without box-box pairs (model_hooks.apply adds it
    with them: box_box_hook, the CCD header, its adjoint)."""
    if box_pairs(m)[2]:
        return TERRAIN_BOX_GEOMS
    from nablaengine import ccd
    orig, fn = box_box_hook()
    return ((orig, fn, 1),
            ("__device__ __noinline__ int adj_contact_geom(", ccd.ADJ_DEVICE + "__device__ __noinline__ int adj_contact_geom(",
             1),
            ("#define NTRI (NV * (NV + 1) / 2)\n", "#define NTRI (NV * (NV + 1) / 2)\n" + ccd.header(m), 1)) + TERRAIN_BOX_GEOMS


def has_floss(m):
    return bool((m.dof_frictionloss > 0).any())


def box_pairs(m):
    """(there is a pair with a box, every such pair is plane-box, box-box, capsule-box, sphere-box (g1_engine.sphbox) or
    cylinder-box (g1_engine.cylccd): the adjoint has them, there is a box-box pair)."""
    import mujoco
    box, plane = int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_PLANE)
    cap = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    pairs = [(int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]])) for p in range(m.npair)]
    with_box = [t for t in pairs if box in t]
    sph, cyl, msh = int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_MESH)
    return (bool(with_box), all(plane in t or t == (box, box) or cap in t or sph in t or cyl in t or msh in t
                                for t in with_box),
            (box, box) in with_box)


def capbox_pairs(m):
    """Whether model m has a capsule-box pair."""
    import mujoco
    box, cap = int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    return any(sorted((int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]]))) == sorted((box, cap))
               for p in range(m.npair))


def capbox_hooks(terrain=False):
    """The hooks for capsule-box pairs: the forward's capsule_box as capbox_segs (step.capbox_segs_source),
    capbox_point and a __noinline__ sphere-box (one copy each: the adjoint's replay computes the forward's bits), and
    the contact adjoint's capsule-box branch ADJ_CAPBOX (the step.py functions over duals, adjoint.capbox_dual_source).
    terrain: a terrain build, whose text has capbox_segs, capbox_point (before collide_pair) and the dual templates
    (before adj_terrain_geom) already: only the sphere-box wrapper is added."""
    import re
    from nablaengine.dual import DUAL_DEVICE
    from nablaengine.g1_engine import adjoint, step
    m = re.search(r"__device__ __noinline__ void capbox_point\(", step.TERRAIN_DEVICE)
    point = step.TERRAIN_DEVICE[m.start():step.TERRAIN_DEVICE.index("\n}\n", m.start()) + 3]
    dev = ("// capsule-box pairs (model_hooks.capbox_hooks): capsule_box's candidates, their sphere centres and the "
           "sphere-box\n// contact as one __noinline__ copy each\n" + ("" if terrain else step.capbox_segs_source() + point)
           + "__device__ __noinline__ void capbox_sphere_box(const float* sp, float r, const float* bp, const float* R,\n"
           "                                               const float* bs, float* dist, float* pos, float* n) {\n"
           "  sphere_box(sp, r, bp, R, bs, dist, pos, n);\n}\n\n")
    fwd_old = "    capsule_box(s, pair, p1, a1, sz1[0], sz1[1], p2, R2, sz2);\n    return;\n"
    fwd_new = """    float B[15], segs[2], ce[3], dist, pc[3], n[3], fr[9];  // (model_hooks.capbox_hooks: capsule_box's contacts)
    for (int k = 0; k < 3; k++) { B[k] = p2[k]; B[12 + k] = sz2[k]; }
    for (int k = 0; k < 9; k++) B[3 + k] = R2[k];
    const int ns = capbox_segs(p1, a1, sz1[1], p2, R2, sz2, segs);
    for (int e = 0; e < ns; e++) {
      capbox_point(p1, a1, sz1[1], B, segs[e], ce);
      capbox_sphere_box(ce, sz1[0], p2, R2, sz2, &dist, pc, n);
      if (dist < 0.f) { make_frame(n, fr); add_contact(s, pair, e, dist, pc, fr); }
    }
    return;
"""
    anchor = "  if (t1 != 2 || t2 != 2) return -1;\n"
    blk = DUAL_DEVICE + adjoint.capbox_dual_source()
    head = "__device__ __noinline__ int adj_contact_geom("
    ter = adjoint.CBTER_DIR_VJP_TREE + "__device__ __noinline__ int adj_terrain_geom("
    duals = (((blk + ter, ter, 1),) if terrain else ()) + ((head, blk + CBT_DIR_VJP + head, 1),)
    return ((fwd_old, fwd_new, 1),
            ("__device__ __forceinline__ void rotmore_of(int f, float* r) {",
             dev + "__device__ __forceinline__ void rotmore_of(int f, float* r) {", 1)) + duals + (
            (anchor, ADJ_CAPBOX + anchor, 1),)


ADJ_CAPBOX = """  if (t1 == 2 && t2 == 3) {  // capsule-box (model_hooks.capbox_hooks): the forward's candidate `slot`, then the
    // contact replayed over duals (D1) once per pose direction of each geom (cbt_dir_vjp)
#if CB_COOP_ON
    if (cot) return 0;  // (the VJP: adj_contacts' cooperative pass, cb_coop)
#endif
    float axis[3], R2[9], B[15], segs[2], ce[3], n[3];
    zaxis_rn(q1, axis);
    quat2mat(q2, R2);
    for (int k = 0; k < 3; k++) { B[k] = p2[k]; B[12 + k] = sz2[k]; }
    for (int k = 0; k < 9; k++) B[3 + k] = R2[k];
    const int ns = capbox_segs(p1, axis, sz1[1], p2, R2, sz2, segs);
    if (slot >= ns) return -1;
    capbox_point(p1, axis, sz1[1], B, segs[slot], ce);
    capbox_sphere_box(ce, sz1[0], p2, R2, sz2, dist, pos, n);
    make_frame(n, fr);
    if (!(-0.5f < n[1] && n[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
    if (cot) {
      float nbar[3];
      make_frame_vjp(fr, cot + 4, nbar);
      for (int j = 0; j < 12; j++) gb[j] = cbt_dir_vjp(slot, j, p1, axis, sz1, p2, R2, sz2, cot, nbar);
    }
    return br;
  }
"""

CBT_DIR_VJP = """// (model_hooks.CBT_DIR_VJP) pose direction j of a capsule-box contact's VJP, gb[j]
__device__ __noinline__ float cbt_dir_vjp(int slot, int j, const float* p1, const float* axis, const float* sz1,
                                          const float* p2, const float* R2, const float* sz2, const float* cot,
                                          const float* nbar) {
  const int g = j / 6, jj = j % 6;  // geom g (0: the capsule, 1: the box), direction jj
  float ea[3] = {0.f, 0.f, 0.f}, ta[3];
  ea[jj % 3] = 1.f;
  D1 P[3], A[3], Bd[15], sg[2], C[3], dd, pp[3], nn[3];
  cross(ea, axis, ta);
  for (int t = 0; t < 3; t++) {
    P[t] = D1(p1[t], g == 0 && jj < 3 && t == jj ? 1.f : 0.f);
    A[t] = D1(axis[t], g == 0 && jj >= 3 ? ta[t] : 0.f);
    Bd[t] = D1(p2[t], g == 1 && jj < 3 && t == jj ? 1.f : 0.f);
    Bd[12 + t] = D1(sz2[t]);
  }
  for (int c = 0; c < 3; c++) {  // R2's columns turn with the box: d col = e x col
    float col[3] = {R2[c], R2[3 + c], R2[6 + c]}, dc[3];
    cross(ea, col, dc);
    for (int r = 0; r < 3; r++) Bd[3 + 3 * r + c] = D1(R2[3 * r + c], g == 1 && jj >= 3 ? dc[r] : 0.f);
  }
  capbox_segs_t<D1>(P, A, D1(sz1[1]), Bd, Bd + 3, Bd + 12, sg);
  capbox_point_t<D1>(P, A, D1(sz1[1]), Bd, sg[slot], C);
  sphere_box_t<D1>(C, D1(sz1[0]), Bd, Bd + 3, Bd + 12, &dd, pp, nn);
  float gj = cot[0] * dd.d;
  for (int t = 0; t < 3; t++) gj += cot[1 + t] * pp[t].d + nbar[t] * nn[t].d;
  return gj;
}

"""


def tree_coop_dispatch(robot, terrain):
    """capbox_coop_hooks' dispatch for the tree engine: the robot's own capsule-box pairs (robot: CBT_DIR_VJP in the
    text) and the terrain's capsule-box contacts (terrain: adjoint.CBTER_DIR_VJP in the text; pair >= NPAIR, a
    terrain box or object slot, directions 6..11 = the capsule's, 0..5 = 0)."""
    conds = []
    body = ""
    if terrain:
        conds.append("((pair) >= NPAIR && c_gtype[PG2(pair)] == 2 && CB_TER_BOX((slot) >> 8))")
        body += """  if (pair >= NPAIR) {  // a terrain capsule-box contact: the capsule's 6 directions (gb[6..11])
    if (j < 6) return 0.f;
    const int g2 = PG2(pair);
    float axis[3];
    zaxis_rn(gq[g2], axis);
    return cbter_dir_vjp(j - 6, slot & 255, ter_box(slot >> 8), gx[g2], axis, c_gsize + 3 * g2, rec, rec + 4);
  }
"""
    if robot:
        conds.append("((pair) < NPAIR && c_gtype[PG1(pair)] == 2 && c_gtype[PG2(pair)] == 3)")
        body += """  {  // the robot's own capsule-box pair
    const int g1 = PG1(pair), g2 = PG2(pair);
    float axis[3], R2[9];
    zaxis_rn(gq[g1], axis);
    quat2mat(gq[g2], R2);
    return cbt_dir_vjp(slot, j, gx[g1], axis, c_gsize + 3 * g1, gx[g2], R2, c_gsize + 3 * g2, rec, rec + 4);
  }
"""
    else:
        body += "  return 0.f;\n"
    head = ("#define CB_TER_BOX(e) ((e) < (g_ter.grid ? int(g_ter.grid[5]) : 0) || (e) >= OBJ_E0)  // (adj_terrain_geom's box)\n"
            if terrain else "")
    return (head + "#define CB_COOP_PAIR(pair, slot) (" + " || ".join(conds) + ")  // (model_hooks.tree_coop_dispatch)\n"
            "#define CB_COOP_WIDE(pair, slot) ((pair) < NPAIR)  // (both geoms' 12 directions; terrain: the capsule's 6)\n"
            "// (model_hooks.tree_coop_dispatch) direction j's gb[j] of contact (pair, slot), rec: cot[0..3], nbar\n"
            "__device__ __forceinline__ float cb_coop_dir(const float (*gx)[3], const float (*gq)[4], int pair, int slot,"
            " int j,\n                                             const float* rec) {\n" + body + "}\n\n")


CAPBOX_COOP_DEV = r"""#if CB_COOP_ON
// (g1_engine.model_hooks.CAPBOX_COOP_DEV) the batch's box contacts whose VJP adj_contact_one deferred (mine: this
// lane's contact (pair, slot), its record in cw[12..18]): their pose cotangents on the bodies into cw[12..23]
__device__ __forceinline__ void cb_coop(Adj& s, int lane, bool mine, int pair, int slot, float* cw) {
  unsigned m = __ballot_sync(0xffffffffu, mine);
  const unsigned wide = __ballot_sync(0xffffffffu, mine && CB_COOP_WIDE(pair, slot));
#pragma unroll 1
  while (m) {
    unsigned gs = 0xFFFFFFFFu, done = 0u;  // group k (lanes 8k..8k+7) -> its contact's lane in bits 8k..8k+7 (0xFF: none)
    int used = 0;
#pragma unroll 1
    while (m) {  // (uniform: the ballots' bits)
      const int c = __ffs(m) - 1, need = (wide >> c) & 1u ? 2 : 1;
      if (used + need > 4) break;
      for (int t = 0; t < need; t++, used++) gs = (gs & ~(0xFFu << (8 * used))) | ((unsigned)c << (8 * used));
      m &= m - 1u;
      done |= 1u << c;
    }
    const int gk = lane >> 3, src = (gs >> (8 * gk)) & 0xFFu;
    const bool has = src != 0xFF;
    const int sc = has ? src : 0;
    const bool w = (wide >> sc) & 1u;
    const int first = gk > 0 && has && (int)((gs >> (8 * (gk - 1))) & 0xFFu) == src ? gk - 1 : gk, sub = lane - 8 * first;
    const int pr = __shfl_sync(0xffffffffu, pair, sc), sl = __shfl_sync(0xffffffffu, slot, sc);
    float rec[7];
#pragma unroll
    for (int k = 0; k < 7; k++) rec[k] = __shfl_sync(0xffffffffu, cw[12 + k], sc);
    float g = 0.f;
    if (has && sub < (w ? 12 : 6)) g = cb_coop_dir(s.gx, s.gq, pr, sl, w ? sub : 6 + sub, rec);
    const bool me = (done >> lane) & 1u, mw = (wide >> lane) & 1u;
    int base = 0;  // this contact lane's first group
    for (int k = 3; k >= 0; k--)
      if ((int)((gs >> (8 * k)) & 0xFFu) == lane) base = 8 * k;
    float gb[12];
#pragma unroll
    for (int k = 0; k < 12; k++) {
      const int sk = mw ? base + k : base + k - 6;
      const float x = __shfl_sync(0xffffffffu, g, sk < 0 ? 0 : (sk > 31 ? 31 : sk));
      gb[k] = mw || k >= 6 ? x : 0.f;
    }
    if (me) {  // adj_contact_one's geom_to_body_vjp
      const int g1 = PG1(pair), g2 = PG2(pair), b1 = c_gbody[g1], b2 = c_gbody[g2];
      float fg[12];
      for (int t = 0; t < 12; t++) fg[t] = 0.f;
      if (b1) geom_to_body_vjp(s.gx[g1], s.X[b1], gb, gb + 3, fg, fg + 3);
      geom_to_body_vjp(s.gx[g2], s.X[b2], gb + 6, gb + 9, fg + 6, fg + 9);
      for (int t = 0; t < 12; t++) cw[12 + t] = fg[t];
    }
  }
}
#endif

"""


TREE_COOP_DEV = r"""#if CB_COOP_ON
// (g1_engine.model_hooks.TREE_COOP_DEV) the batch's box contacts whose VJP adj_contact_one deferred (mine: this lane's
// contact (pair, slot), its record in cw[12..18]): their pose cotangents on the bodies into cw[12..23]
__device__ __forceinline__ void cb_coop(Adj& s, int lane, bool mine, int pair, int slot, float* cw) {
  const unsigned m = __ballot_sync(0xffffffffu, mine);
  if (!m) return;  // (uniform)
  const unsigned wide = CB_COOP_ANY_WIDE ? __ballot_sync(0xffffffffu, mine && CB_COOP_WIDE(pair, slot)) : 0u;
  const unsigned below = (1u << lane) - 1u;
  const int nu = __popc(m) + __popc(wide), nt = 6 * nu;  // units, jobs
  const int u0 = __popc(m & below) + __popc(wide & below);  // this lane's contact's first unit
  const bool mw = (wide >> lane) & 1u;
  float gb[12];
#pragma unroll
  for (int k = 0; k < 12; k++) gb[k] = 0.f;
#pragma unroll 1
  for (int i0 = 0; i0 < nt; i0 += 32) {  // (uniform)
    const int t = i0 + lane, u = t < nt ? t / 6 : 0, j6 = t - 6 * u;
    int c = 0;  // the contact lane of unit u: the last lane c with units(lanes < c) <= u
#pragma unroll
    for (int b = 16; b; b >>= 1) {
      const unsigned lt = (1u << (c + b)) - 1u;
      if (__popc(m & lt) + __popc(wide & lt) <= u) c += b;
    }
    const int half = u - __popc(m & ((1u << c) - 1u)) - __popc(wide & ((1u << c) - 1u));
    const bool w = (wide >> c) & 1u;
    const int pr = __shfl_sync(0xffffffffu, pair, c), sl = __shfl_sync(0xffffffffu, slot, c);
    float rec[7];
#pragma unroll
    for (int k = 0; k < 7; k++) rec[k] = __shfl_sync(0xffffffffu, cw[12 + k], c);
    float g = 0.f;
    if (t < nt) g = cb_coop_dir(s.gx, s.gq, pr, sl, (w ? 6 * half : 6) + j6, rec);
#pragma unroll
    for (int k = CB_COOP_ANY_WIDE ? 0 : 6; k < 12; k++) {  // direction k of this lane's contact: job 6 u0 + its index
      const int sk = 6 * u0 + (mw ? k : k - 6) - i0;
      const float x = __shfl_sync(0xffffffffu, g, sk & 31);
      if (mine && (mw || k >= 6) && sk >= 0 && sk < 32) gb[k] = x;
    }
  }
  if (mine) {  // adj_contact_one's geom_to_body_vjp, once
    const int g1 = PG1(pair), g2 = PG2(pair), b1 = c_gbody[g1], b2 = c_gbody[g2];
    float fg[12];
    for (int t = 0; t < 12; t++) fg[t] = 0.f;
    if (b1) geom_to_body_vjp(s.gx[g1], s.X[b1], gb, gb + 3, fg, fg + 3);  // (a terrain contact: gb[0..5] 0)
    geom_to_body_vjp(s.gx[g2], s.X[b2], gb + 6, gb + 9, fg + 6, fg + 9);
    for (int t = 0; t < 12; t++) cw[12 + t] = fg[t];
  }
}
#endif

"""


def tree_coop_dev(wide):
    """TREE_COOP_DEV with CB_COOP_ANY_WIDE (wide: the model has robot capsule-box pairs, contacts with 12 directions)."""
    return f"#define CB_COOP_ANY_WIDE {int(bool(wide))}  // (model_hooks.tree_coop_dev)\n" + TREE_COOP_DEV


def capbox_coop_hooks(dispatch, dev=CAPBOX_COOP_DEV):
    """The (anchor, replacement, count) edits of DK_CAPBOX_COOP (see CAPBOX_COOP_DEV) for an adjoint text with a box
    contact VJP branch over CB_COOP_ON; dispatch: the engine's #define CB_COOP_PAIR(pair, slot) and cb_coop_dir (text
    placed before adj_contact_one); dev: the pass' cb_coop (CAPBOX_COOP_DEV; the tree engine: tree_coop_dev)."""
    head = "__device__ __noinline__ int adj_contact_geom("
    one = "__device__ __forceinline__ float adj_contact_one("
    adj = "template <bool DBG>\n__device__ __noinline__ void adj_contacts(Adj& s, int lane, float* dbg_in) {\n"
    return (
        (head, "#ifndef DK_CAPBOX_COOP\n#define DK_CAPBOX_COOP 1  // (g1_engine.model_hooks.capbox_coop_hooks)\n#endif\n"
               "#define CB_COOP_ON DK_CAPBOX_COOP\n" + head, 1),
        (one, "#if CB_COOP_ON\n" + dispatch + "#endif\n" + one, 1),
        (adj, dev + adj, 1),
        ("    if (b1) geom_to_body_vjp(s.gx[g1], s.X[b1], gb, gb + 3, fg, fg + 3);\n"
         "    geom_to_body_vjp(s.gx[g2], s.X[b2], gb + 6, gb + 9, fg + 6, fg + 9);\n",
         "#if CB_COOP_ON\n"
         "    if (CB_COOP_PAIR(pair, slot)) {  // the record for cb_coop (model_hooks.capbox_coop_hooks)\n"
         "      for (int t = 0; t < 4; t++) fg[t] = cot[t];\n"
         "      make_frame_vjp(fr, cot + 4, fg + 4);\n"
         "    } else\n"
         "#endif\n"
         "    {\n"
         "    if (b1) geom_to_body_vjp(s.gx[g1], s.X[b1], gb, gb + 3, fg, fg + 3);\n"
         "    geom_to_body_vjp(s.gx[g2], s.X[b2], gb + 6, gb + 9, fg + 6, fg + 9);\n"
         "    }\n", 1),
        ("    const unsigned skip = __ballot_sync(0xffffffffu, on && !use), mis = __ballot_sync(0xffffffffu, bad);\n",
         "#if CB_COOP_ON\n"
         "    cb_coop(s, lane, cb >= 0 && CB_COOP_PAIR(pair, slot), pair, slot, cw);  // (model_hooks.capbox_coop_hooks)\n"
         "#endif\n"
         "    const unsigned skip = __ballot_sync(0xffffffffu, on && !use), mis = __ballot_sync(0xffffffffu, bad);\n",
         1),
    )




def needed(m):
    """True where model m needs edits (False for the G1)."""
    from nablaengine.g1_engine import cylinder, mesh
    return (m.nv < 32 or source.child_width(m) > 3 or box_pairs(m)[0] or source.has_damping(m)
            or has_floss(m) or m.nbody > 32 or cylinder.pairs(m)[0]
            or mesh.pairs(m)[0])


def _apply(src, hooks):
    for a, b, n in hooks:
        if src.count(a) != n:
            raise RuntimeError(f"model hook anchor found {src.count(a)} times, expected {n}: {a[:70]!r}")
        src = src.replace(a, b)
    return src


def apply(src, m, part="engine"):
    """src with the hooks model m needs; part: "engine" (adjoint.cuda_source's text) or "kinematics"
    (kinematics.KIN_DEVICE). Unchanged for the G1."""
    if not needed(m):
        return src
    hooks = ()
    if m.nbody > 32 and part == "kinematics":
        hooks += kin_nb64_hooks()
    if source.child_width(m) > 3:
        hooks += ENGINE_CHILD if part == "engine" else KIN_CHILD
    if m.nv < 32 and part == "engine":
        hooks += ENGINE_SMALL_NV
    if m.nbody > 32 and part == "engine":
        hooks += ENGINE_NB64 + ADJ_NB64 + ADJ_DEBUG_NB64
    if source.has_damping(m) and part == "engine":
        hooks += ENGINE_DAMPING
    has_box, box_ok, boxbox = box_pairs(m)
    if (has_box or has_floss(m)) and part == "engine":
        hooks += ENGINE_ADJ_SCOPE
    if has_floss(m) and part == "engine":
        hooks += ENGINE_FLOSS + ADJ_DEBUG_FLOSS
    if has_box and part == "engine":
        if not box_ok:
            raise ValueError("box contacts with planes, spheres, capsules and boxes only")
        hooks += ENGINE_PLANE_BOX
    if boxbox and part == "engine":
        orig, fn = box_box_hook()
        from nablaengine import ccd
        hooks += ((orig, fn, 1), ("  if (t1 != 2 || t2 != 2) return -1;\n",
                                  ADJ_BOX_BOX + "  if (t1 != 2 || t2 != 2) return -1;\n", 1),
                  ("__device__ __noinline__ int adj_contact_geom(", ccd.ADJ_DEVICE + "__device__ __noinline__ int adj_contact_geom(", 1))
    if capbox_pairs(m) and part == "engine":
        hooks += capbox_hooks(terrain="int capbox_segs(" in src)
    from nablaengine.g1_engine import cylccd, cylinder, sphbox
    if sphbox.pairs(m) and part == "engine":
        hooks += sphbox.hooks()
    has_cyl, cyl_ok = cylinder.pairs(m)
    if has_cyl and part == "engine":
        if not cyl_ok:
            raise ValueError("cylinder contacts with planes, spheres, capsules, cylinders and boxes only")
        hooks += cylinder.hooks()
    cyl_ccd = cylccd.pairs(m)
    if cyl_ccd and part == "engine":
        hooks += cylccd.hooks()
    from nablaengine.g1_engine import mesh
    has_mesh, mesh_ok = mesh.pairs(m)
    if has_mesh and part == "engine":
        if not mesh_ok:
            raise ValueError("mesh contacts with planes, spheres, capsules, cylinders, boxes and meshes only")
        hooks += mesh.hooks(m)
    src = _apply(src, hooks)
    if cyl_ccd and part == "engine":
        src = cylccd.apply_rec(src)
    if part == "engine":
        extra = ""
        if boxbox:
            from nablaengine import ccd
            extra = ccd.header(m)
        if cyl_ccd:
            extra += cylccd.header(m)
        src = src.replace("#define NTRI (NV * (NV + 1) / 2)\n",
                          f"#define NTRI (NV * (NV + 1) / 2)\n#define NCH {source.child_width(m)}  // (model_hooks)\n"
                          + extra, 1)
    from nablaengine import ccd
    return ccd.apply_gjk_mj4(src)

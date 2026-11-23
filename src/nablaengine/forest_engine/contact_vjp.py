"""The VJP of the contact output (init(contact_grad=True), the forest engine): a loss on the contact table's forces and
geometry words and on the per-foot wrenches gets its gradients with respect to the control step's qpos, qvel and ctrl
(and the per-world parameters / object-slot words where the build has them). Applied to the generated text only for a
contact_grad build (jax_ffi.library_sources, after forest_engine.contact_forces); every other build is unchanged."""
import re

import mujoco


def check(m, spec, terrain=None, surrogate=False, bam=None):
    """ValueError where a contact_grad build does not fit forest model m (spec: its contact_forces Spec)."""
    if spec is None:
        raise ValueError("contact_grad: needs the contact output (internal: no contact_forces Spec)")
    if int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        raise ValueError("contact_grad: the elliptic friction cone is not supported (pyramidal only)")
    if any(int(d) == 4 for d in m.pair_dim[:m.npair]):
        raise ValueError("contact_grad: condim-4 (torsional) contacts are not supported (condim 1 and 3)")
    if surrogate:
        raise ValueError("contact_grad: not with contact_surrogate")
    if bam is not None:
        raise ValueError("contact_grad: not with the BAM servo model")
    if int(m.opt.noslip_iterations) > 0:
        raise ValueError("contact_grad: not with the noslip pass (opt.noslip_iterations > 0): the contact output's VJP "
                         "does not include the noslip rows' forces")


def _sub(src, a, b, n, what):
    k = src.count(a)
    if k != n:
        raise RuntimeError(f"contact_vjp: {what}: anchor found {k} times, expected {n}: {a[:80]!r}")
    return src.replace(a, b)


DEVICE = r"""
// ---- contact_vjp (forest_engine.contact_vjp): the VJP of the contact output. gcon per world: the contact output's
// cotangent (forest_engine.contact_forces' layout: CG_T tables of MAXCON * CON_WORDS words, then CF_NFOOT wrenches
// per substep)
#define CG_T(nsub) (CF_SUB ? (size_t)(nsub) : (size_t)1)
__device__ __forceinline__ size_t cg_stride(int nsub) {
  return CG_T(nsub) * MAXCON * CON_WORDS + (CF_NFOOT ? (size_t)nsub * CF_NFOOT * 6 + CF_NFOOT : 0);
}
#if CF_NFOOT
__device__ __forceinline__ int cg_foot(int i) {
  constexpr int fb[CF_NFOOT] = CF_FEET;
  return fb[i];
}
#endif
// substep k's cotangents: its table's (null where the build writes no table for k) and its wrenches' (null without feet)
__device__ __forceinline__ void cg_set(Adj& s, const float* gcon, int w, int k, int nsub, int lane) {
  if (lane == 0) {
    const float* b = gcon ? gcon + (size_t)w * cg_stride(nsub) : nullptr;
    s.cgt = b && (CF_SUB || k == nsub - 1) ? b + (CF_SUB ? (size_t)k * MAXCON * CON_WORDS : (size_t)0) : nullptr;
    s.cgw = b && CF_NFOOT ? b + CG_T(nsub) * MAXCON * CON_WORDS + (size_t)k * CF_NFOOT * 6 : nullptr;
  }
  __syncwarp();
}
// contact c (pair, condim dim, friction mu, frame fr, point p): the cotangent of its world force F (Ft: the table's
// force words, and on a foot sg (wrench force bar + moment bar x (p - X_foot))), of its frame forces (phib: normal,
// t1, t2; the force_normal word's on the normal) and of its pyramid rows' forces (fbe: phib_0 +- mu phib_t; condim 1:
// phib_0)
__device__ __forceinline__ void cg_row(const Adj& s, int c, int pair, int dim, float mu, const float* fr,
                                       const float* p, float* fbe, float* Ft, float* phib) {
  float fnb = 0.f;
  for (int k = 0; k < 3; k++) Ft[k] = 0.f;
  if (s.cgt) {
    const float* t = s.cgt + (size_t)c * CON_WORDS;
    for (int k = 0; k < 3; k++) Ft[k] = t[8 + k];
    fnb = t[11];
  }
#if CF_NFOOT
  if (s.cgw) {
    const int b1 = c_gbody[PG1(pair)], b2 = c_gbody[PG2(pair)];
    for (int i = 0; i < CF_NFOOT; i++) {
      const int fb = cg_foot(i);
      const float sg = b2 == fb ? 1.f : (b1 == fb ? -1.f : 0.f);  // (cf_finish: + on geom 2)
      if (sg == 0.f) continue;
      const float* g = s.cgw + 6 * i;
      const float r[3] = {p[0] - s.X[fb][0], p[1] - s.X[fb][1], p[2] - s.X[fb][2]};
      float mr[3];
      cross(g + 3, r, mr);
      for (int k = 0; k < 3; k++) Ft[k] += sg * (g[k] + mr[k]);
    }
  }
#endif
  for (int j = 0; j < 3; j++) phib[j] = dot3(fr + 3 * j, Ft);
  phib[0] += fnb;
  for (int e = 0; e < 4; e++) {
    const int t = 1 + (e >> 1);
    fbe[e] = dim >= 3 ? phib[0] + ((e & 1) ? -mu : mu) * phib[t] : (e == 0 ? phib[0] : 0.f);
  }
}
// the pre-pass's contact vector w = -D sum_e fbe_e u_e over the quadratic rows (u_e = n +- mu t; condim 1: n)
__device__ __forceinline__ void cg_wa(const float* fr, float mu, int dim, unsigned zone, float D, const float* fbe,
                                      float* w) {
  float a[3] = {0.f, 0.f, 0.f};
  for (int e = 0; e < 4; e++) {
    if (!((zone >> e) & 1u) || (e > 0 && dim < 3)) continue;
    const int t = 1 + (e >> 1);
    a[0] += fbe[e];
    if (dim >= 3) a[t] += ((e & 1) ? -mu : mu) * fbe[e];
  }
  for (int k = 0; k < 3; k++) w[k] = -D * (a[0] * fr[k] + a[1] * fr[3 + k] + a[2] * fr[6 + k]);
}
// the explicit geometry terms of contact c into its geometry cotangent cot (d bar, p bar, frame bar; the adjoint's sign:
// minus the loss's derivative) and the feet's origin cotangents cgx (the adjoint's sign: + sg F x m bar): the table's
// dist / pos / normal / t1 words, F's rotation by the frame (fr_j bar += phi_j F bar) and the moments' arm. The frame
// forces phi from the row forces f_e = -D x_e at the solution (V_v, V_a: the contact point's motions as
// contact_psi_vjp's). Returns the loss's explicit derivative by the friction mu: phib_1 (f_0 - f_1) + phib_2 (f_2 - f_3)
__device__ __forceinline__ float cg_geom(const Adj& s, int c, int pair, int dim, float mu, unsigned zone,
                                         const float* fr, const float* p, const float* Vv, const float* Va, float d,
                                         float D, float k, float b, float imp, const float* Ft, const float* phib,
                                         float* cot, float* cgx) {
  float vf[3], af[3], f[4] = {0.f, 0.f, 0.f, 0.f};
  for (int j = 0; j < 3; j++) { vf[j] = dot3(fr + 3 * j, Vv); af[j] = dot3(fr + 3 * j, Va); }
  const float g = k * imp * d;
  for (int e = 0; e < 4; e++) {
    if (!((zone >> e) & 1u) || (e > 0 && dim < 3)) continue;
    const int t = 1 + (e >> 1);
    const float sm = dim >= 3 ? ((e & 1) ? -mu : mu) : 0.f;
    f[e] = -D * (af[0] + sm * af[t] + g + b * (vf[0] + sm * vf[t]));
  }
  const float phi[3] = {f[0] + f[1] + f[2] + f[3], dim >= 3 ? mu * (f[0] - f[1]) : 0.f,
                        dim >= 3 ? mu * (f[2] - f[3]) : 0.f};
  if (s.cgt) {
    const float* t = s.cgt + (size_t)c * CON_WORDS;
    cot[0] -= t[1];
    for (int q = 0; q < 3; q++) { cot[1 + q] -= t[2 + q]; cot[4 + q] -= t[5 + q]; }
#if CF_FRAME
    for (int q = 0; q < 3; q++) cot[7 + q] -= t[CF_T1 + q];
#endif
  }
  for (int j = 0; j < 3; j++)
    for (int q = 0; q < 3; q++) cot[4 + 3 * j + q] -= phi[j] * Ft[q];
#if CF_NFOOT
  if (s.cgw) {
    const int b1 = c_gbody[PG1(pair)], b2 = c_gbody[PG2(pair)];
    float F[3];
    for (int q = 0; q < 3; q++) F[q] = phi[0] * fr[q] + phi[1] * fr[3 + q] + phi[2] * fr[6 + q];
    for (int i = 0; i < CF_NFOOT; i++) {
      const int fb = cg_foot(i);
      const float sg = b2 == fb ? 1.f : (b1 == fb ? -1.f : 0.f);
      if (sg == 0.f) continue;
      float x[3];
      cross(F, s.cgw + 6 * i + 3, x);
      for (int q = 0; q < 3; q++) { cot[1 + q] -= sg * x[q]; cgx[3 * i + q] += sg * x[q]; }
    }
  }
#endif
  (void)pair;
  return dim >= 3 ? phib[1] * (f[0] - f[1]) + phib[2] * (f[2] - f[3]) : 0.f;
}
"""

PRE = r"""
// (contact_vjp) the contacts' a* bar term into s.cga (zeros without cotangents)
__device__ __noinline__ void cg_pre(Adj& s, int lane) {
  LANE_BOUND(lane);
  for (int i = lane; i < NV; i += 32) s.cga[i] = 0.f;
  __syncwarp();
  if (!(s.cgt || s.cgw)) return;
@GEOM_POSES@
  __syncwarp();
  const int ncon = uniform_int(s.hdr & 255, 8), nlim = ((unsigned)s.hdr >> 18) & 63;
  const unsigned below = (1u << lane) - 1u;
  float acc[(NV + 31) / 32];
  for (int t = 0; t < (NV + 31) / 32; t++) acc[t] = 0.f;
  int ntri = 0, nquad = 0;
  (void)nquad;
  for (int c0 = 0; c0 < ncon; c0 += 32) {
    const int c = c0 + lane, nb = min(32, ncon - c0);
    const bool on = c < ncon;
    const unsigned e = on ? s.con[c] : 0u;
@PAIR_SLOT@
    const unsigned b3 = __ballot_sync(0xffffffffu, dim >= 3);
@ROW0@
    ntri += __popc(b3);
    const int r0x = r0;
    const bool use = on && DK_ADJ_CONTACTS && (DK_ADJ_STAGE >= 3 || c_gbody[PG1(pair)] == 0 || c_gbody[PG2(pair)] == 0)
                     && !PFROZEN(pair);
    unsigned zone = 0u;
    if (use && r0x < NEFC) {
      const int nr = min(dim >= 3 ? 4 : 1, NEFC - r0x), w = r0x >> 5;
      unsigned long long zz = s.zone[w];
      if (w + 1 < (NEFC + 31) / 32) zz |= (unsigned long long)s.zone[w + 1] << 32;
      zone = (unsigned)(zz >> (r0x & 31)) & ((1u << nr) - 1u);
    }
    int cb = -1;
    float cw[CS_N];
#pragma unroll
    for (int k = 0; k < 6; k++) cw[k] = 0.f;
    if (use && zone) {
      float cgx_[CF_NFOOT ? 3 * CF_NFOOT : 1];
@PP_DECL@
      if (adj_contact_one<true>(s, c, pair, slot, zone, cw, nullptr, cgx_@PP_ARG@) < 0.f)
        cb = c_gbody[PG1(pair)] | c_gbody[PG2(pair)] << 8;
    }
#pragma unroll 1
    for (int j = 0; j < nb; j++) {
      const int cj = __shfl_sync(0xffffffffu, cb, j);
      if (UNIFORM(cj < 0)) continue;
      const int b1 = cj & 255, b2 = cj >> 8;
      float w[6];
#pragma unroll
      for (int k = 0; k < 6; k++) w[k] = __shfl_sync(0xffffffffu, cw[k], j);
      const auto m1 = c_ancmask[b1], m2 = c_ancmask[b2];
      for (int t = 0; t < (NV + 31) / 32; t++) {
        const int i = lane + 32 * t;
        if (i >= NV) break;
        const int bi = c_dofbody[i];
        if (!(((m1 ^ m2) >> bi) & 1u)) continue;
        const float sg = ((m2 >> bi) & 1u) ? 1.f : -1.f;
        const int jt = JT_OF(bi), kk = i - c_dadr[bi];
        float col[3];
        if (jt == JFREE && kk < 3) {
          for (int q = 0; q < 3; q++) col[q] = q == kk ? 1.f : 0.f;
        } else {
          float z[3];
          if (jt == JFREE) {
            const float ek[3] = {kk == 3 ? 1.f : 0.f, kk == 4 ? 1.f : 0.f, kk == 5 ? 1.f : 0.f};
            qrot(s.Q[bi], ek, z);
          } else {
            qrot(s.Q[bi], c_axis + 3 * bi, z);
          }
          if (jt == JSLIDE) {
            for (int q = 0; q < 3; q++) col[q] = z[q];
          } else {
            const float r[3] = {w[3] - s.X[bi][0], w[4] - s.X[bi][1], w[5] - s.X[bi][2]};
            cross(z, r, col);
          }
        }
        acc[t] += sg * dot3(col, w);
      }
    }
  }
  for (int t = 0; t < (NV + 31) / 32; t++) {
    const int i = lane + 32 * t;
    if (i < NV) s.cga[i] = acc[t];
  }
  __syncwarp();
}
"""

_GEOM_HEAD = "  for (int g = lane; g < NGEOMC; g += 32) {  // geom_poses (step.py)\n"


def _pair_slot(src):
    i = src.index("void adj_contacts(Adj& s, int lane, float* dbg_in) {")
    a = "    const unsigned e = on ? s.con[c] : 0u;\n"
    j = src.index(a, i) + len(a)
    k = src.index("    const unsigned b3 = __ballot_sync(0xffffffffu, dim >= 3);\n", j)
    out = src[j:k].rstrip("\n")
    if "const int pair = " not in out or "dim = " not in out:
        raise RuntimeError("contact_vjp: adj_contacts' pair / slot lines")
    return out


def _row_line(src):
    from nablaengine.forest_engine import adjoint
    return adjoint.contact_row_line(src)


def _geom_poses(src):
    i = src.index("void adj_contacts(Adj& s, int lane, float* dbg_in) {")
    j = src.index(_GEOM_HEAD, i)
    k = src.index("\n  }\n", j) + len("\n  }\n")
    return src[j:k].rstrip("\n")


def engine(src, spec, m):
    """The engine text (adjoint.cuda_source's unit after forest_engine.contact_forces.engine) with the contact output's
    VJP."""
    assert spec is not None
    if "#define HAS_CONDIM4 1" in src:
        raise ValueError("contact_grad: condim-4 (torsional) contacts are not supported (condim 1 and 3)")
    gl = "          float gx[NGEOMC][3], gq[NGEOMC][4];  // world poses of the collision geoms (the forward's geom_poses)\n"
    src = _sub(src, gl, "          float cg_unused;  // (contact_vjp: gx, gq out of the union)\n", 1, "Adj's geom poses")
    flags = "  int flags;  // union of the records' flags (tape.py), ADJ_ROWS_SKIPPED, ADJ_REPLAY_MISMATCH\n"
    src = _sub(src, flags, flags
               + "  const float *cgt, *cgw;  // (contact_vjp) the substep's table / wrench cotangents (null: none)\n"
               "  float cga[NV];  // (contact_vjp) the contacts' a* bar term (cg_pre)\n"
               "  float gx[NGEOMC][3], gq[NGEOMC][4];  // world poses of the collision geoms (the forward's geom_poses)\n",
               1, "struct Adj")
    src = _sub(src, "                                                float* dbar, float* frbar\n",
               "                                                float* dbar, float* frbar, const float* fbe\n", 1,
               "contact_psi_vjp's signature")
    src = _sub(src, "    float x = ua + g + b * uv, Dx = D * x, Dl = D * ul;\n",
               "    float x = ua + g + b * uv, Dx = D * x, Dl = D * (ul + fbe[e]);  // (contact_vjp: + fbe)\n", 1,
               "contact_psi_vjp's rows")
    src = _sub(src, "    db += ul * (dD * x + D * gd);\n  }\n",
               "    db += (ul + fbe[e]) * (dD * x + D * gd);  // (contact_vjp: + fbe)\n  }\n", 1, "contact_psi_vjp's d bar")
    head = "__device__ __forceinline__ float adj_contact_one(Adj& s, int c, int pair, int slot, unsigned zone,"
    if src.count(head) != 1:
        raise RuntimeError("contact_vjp: adj_contact_one's head")
    src = src.replace(head, DEVICE + "template <bool CGPRE>\n" + head)
    sig = "                                                 float* dbg"
    i = src.index(head)
    j = src.index(sig, i)
    if j - i > 400:
        raise RuntimeError("contact_vjp: adj_contact_one's dbg parameter")
    src = src[:j] + sig + ", float* cgx" + src[j + len(sig):]
    k = src.index(") {", j)
    extra = re.findall(r"float\* (\w+)", src[j + len(sig) + len(", float* cgx"):k])
    has_pp = "pp" in extra
    has_mg = "mg" in extra
    calls = src.count("adj_contact_one(s, c, pair, slot, zone, cw, dbg")
    if calls != 1:
        raise RuntimeError(f"contact_vjp: adj_contact_one's calls ({calls})")
    src = src.replace("adj_contact_one(s, c, pair, slot, zone, cw, dbg",
                      "adj_contact_one<false>(s, c, pair, slot, zone, cw, dbg, cgx")
    wv = "  float Wv[3], Wa[3], Wl[3], cot[13], fg[12];  // cot = (d bar, p bar, frame bar): adj_contact_geom's cotangent\n"
    src = _sub(src, wv,
               "  float cg_fbe[4] = {0.f, 0.f, 0.f, 0.f}, cg_Ft[3] = {0.f, 0.f, 0.f}, cg_pb[3] = {0.f, 0.f, 0.f};\n"
               "  float cg_dmu = 0.f;  // (contact_vjp) the explicit friction derivative\n"
               "  const bool cgo = s.cgt || s.cgw;  // (contact_vjp) the substep has cotangents\n"
               "  if (cgo) cg_row(s, c, pair, dim, mu, fr, p, cg_fbe, cg_Ft, cg_pb);\n"
               "  if (CGPRE) {  // (contact_vjp) the pre-pass: w = -D sum_e fbe_e u_e and the contact point\n"
               "    cg_wa(fr, mu, dim, zone, D, cg_fbe, cs);\n"
               "    for (int t = 0; t < 3; t++) cs[3 + t] = p[t];\n"
               "    (void)br; (void)nu; (void)dimp; (void)dD;\n"
               "    return d;\n"
               "  }\n" + wv, 1, "adj_contact_one's cotangents")
    n = src.count("Wv, Wa, Wl, cot, cot + 4")
    if n not in (1, 2):
        raise RuntimeError(f"contact_vjp: contact_psi_vjp calls ({n})")
    src = src.replace("Wv, Wa, Wl, cot, cot + 4", "Wv, Wa, Wl, cot, cot + 4, cg_fbe")
    pb = "  cot[1] = cot[2] = cot[3] = 0.f;\n  contact_pbar_motion(Wv, Wa, Wl, Ov, Oa, Ol, cot + 1);\n"
    src = _sub(src, pb, pb + "  if (cgo)  // (contact_vjp) the words' geometry, F's frame, the moments' arm\n"
               "    cg_dmu = cg_geom(s, c, pair, dim, mu, zone, fr, p, Vv, Va, d, D, k, b, imp, cg_Ft, cg_pb, cot, cgx);\n"
               + ("  if (PFOOT(pair) && zone" + (" && !em_mu" if "const bool em_mu" in src else "")
                  + ") pp[6] -= cg_dmu;  // (contact_vjp: the frame forces' mu, world_params)\n" if has_pp else "")
               + ("  if (om_e >= 0 && zone) mg[6] -= cg_dmu;  // (contact_vjp: the frame forces' mu, OBJ_MAT)\n"
                  if has_mg else "")
               + "  (void)cg_dmu;\n", 1, "adj_contact_one's geometry cotangent")
    src = _sub(src, "  if (zone) {  // the primitive's VJP", "  if (zone || cgo) {  // the primitive's VJP", 1,
               "adj_contact_one's primitive VJP")
    if has_pp or has_mg:
        for x in ("pp", "mg"):
            a = f"imp, contact_invw(pair, mu), sref, simp, {x});\n"
            if src.count(a) != (1 if x in extra else 0):
                raise RuntimeError(f"contact_vjp: contact_param_vjp's call with {x}")
            src = src.replace(a, a.replace(f"{x});", f"{x}, cg_fbe);"))
        for a_, b_, what in (
                ("const float* solimp,\n                                               float* pp) {",
                 "const float* solimp,\n                                               float* pp, const float* fbe) {",
                 "signature"),
                ("    for (int q = 0; q < 6; q++) pp[q] += ul * (dDdi",
                 "    for (int q = 0; q < 6; q++) pp[q] += (ul + fbe[e]) * (dDdi", "rows"),
                ("    pp[6] += dDdmu * xe * ul + D * se * ((af[t] + b * vf[t]) * ul + xe * lf[t]);\n",
                 "    pp[6] += dDdmu * xe * (ul + fbe[e]) + D * se * ((af[t] + b * vf[t]) * (ul + fbe[e]) + xe * lf[t]);\n",
                 "mu")):
            if src.count(a_) < 1:
                raise RuntimeError(f"contact_vjp: contact_param_vjp's {what}")
            src = src.replace(a_, b_)
    src = _sub(src, "    if (use && (zone || dbg)) {", "    if (use && (zone || dbg || s.cgt || s.cgw)) {", 1,
               "adj_contacts' gate")
    gate = re.findall(r"\n      if \(zone(?:\)| &&)[^\n]*cb = c_gbody\[PG1\(pair\)\] \| c_gbody\[PG2\(pair\)\] << 8;", src)
    if len(gate) != 1:
        raise RuntimeError(f"contact_vjp: adj_contacts' reduction gate found {len(gate)} times")
    src = src.replace(gate[0], gate[0].replace("if (zone", "if ((zone || s.cgt || s.cgw)", 1)
                      .replace("if ((zone || s.cgt || s.cgw))", "if (zone || s.cgt || s.cgw)", 1) + "  // (contact_vjp)")
    cw = "    float cw[CS_N];  // the lane's contact (adj_contact_one), read by the body lanes by shuffles\n"
    src = _sub(src, cw, cw + "    float cgx[CF_NFOOT ? 3 * CF_NFOOT : 1];  // (contact_vjp) the feet's origin cotangents\n"
               "    for (int k = 0; k < (CF_NFOOT ? 3 * CF_NFOOT : 1); k++) cgx[k] = 0.f;\n", 1, "adj_contacts' words")
    red = "    __syncwarp();\n    APROF(APROF_CONTACT_REDUCE);\n"
    src = _sub(src, red, "    __syncwarp();\n"
               "#if CF_NFOOT\n"
               "    if (s.cgw) {  // (contact_vjp) the feet's origin cotangents, summed over the batch (a fixed order)\n"
               "      for (int q = 0; q < 3 * CF_NFOOT; q++) {\n"
               "        const float v = warp_sum(cgx[q]);\n"
               "        if (lane == 0) s.fb[cg_foot(q / 3)][q % 3] += v;\n"
               "      }\n"
               "    }\n"
               "#endif\n" + red, 1, "adj_contacts' batch end")
    pre = (PRE.replace("@GEOM_POSES@", _geom_poses(src)).replace("@PAIR_SLOT@", _pair_slot(src))
           .replace("@ROW0@", _row_line(src))
           .replace("@PP_DECL@", "".join(f"      float {x}_[8] = {{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}};\n"
                                         for x in extra).rstrip("\n"))
           .replace("@PP_ARG@", "".join(f", {x}_" for x in extra)))
    lim = "template <bool DBG>\n__device__ __noinline__ void adj_contacts(Adj& s, int lane, float* dbg_in) {"
    src = _sub(src, lim, lim, 1, "adj_contacts' head")
    ps = "  adj_poses(s, lane);  // A3: poses and channel 1\n  APROF(APROF_POSES);\n"
    src = _sub(src, ps, ps + "  cg_pre(s, lane);  // (contact_vjp) the contacts' a* bar term\n", 1, "adj_substep's poses")
    lam = "  for (int i = lane; i < NV; i += 32) s.lam[i] = s.abar[i];\n"
    src = _sub(src, lam, "  for (int i = lane; i < NV; i += 32)  // (contact_vjp) + the contacts' a* bar term\n"
               "    s.lam[i] = s.cgt || s.cgw ? s.abar[i] + s.cga[i] : s.abar[i];\n", 1, "adj_solves' lambda")
    end = "\n#endif\n\n#ifdef DK_ADJ_DEBUG\n__device__ void dbg_put("
    stub = ("#else\n__device__ void cg_pre(Adj& s, int lane) {  // (contact_vjp; DK_ADJ_STAGE 1: no contacts)\n"
            "  for (int i = lane; i < NV; i += 32) s.cga[i] = 0.f;\n  __syncwarp();\n}\n")
    src = _sub(src, end, "\n" + pre + stub + "#endif\n\n#ifdef DK_ADJ_DEBUG\n__device__ void dbg_put(", 1,
               "cg_pre's place")
    kp = "                          const float* ctrl ADJ_SOFT_PARAM) {"
    src = _sub(src, kp, "                          const float* ctrl ADJ_SOFT_PARAM, const float* gcon) {", 1,
               "k_adjoint's signature")
    kr = "    const size_t r = (size_t)w * nsub + k;\n"
    src = _sub(src, kr, kr + "    cg_set(s, gcon, w, k, nsub, lane);  // (contact_vjp)\n", 1, "k_adjoint's substeps")
    return src


def width_expr(spec):
    """The C++ expression of the contact output's words per world (substeps: the handler's attribute)."""
    nf = len(spec.feet)
    t = "(int64_t)substeps * " if spec.mode == "substeps" else ""
    return f"({t}MAXCON * {spec.con_words}" + (f" + (int64_t)substeps * {6 * nf} + {nf}" if nf else "") + ")"


FFI = r"""
// contact_grad builds (forest_engine.contact_vjp): the backward with the contact output's cotangent gcon (B, width),
// ctrl read for clamp_kv 2 only; with and without the kinematics outputs
static ffi::Error step_bwd_con_impl(cudaStream_t st, BufF tape, BufF fric, BufF com, BufF ctrl, BufF gq_out,
                                    BufF gv_out, BufF gcon, int32_t substeps, int32_t clamp_kv, int32_t normal,
                                    float clamp_soft, RetF gq, RetF gv, RetF gu, ffi::Result<ffi::Buffer<ffi::S32>> info) {
  return bwd_impl(st, tape, fric, com, clamp_kv == 2 ? &ctrl : nullptr, gq_out, gv_out, substeps, clamp_kv, normal,
                  clamp_soft, gq, gv, gu, info, nullptr, nullptr, &gcon);
}
static ffi::Error step_bwd_con_kin_impl(cudaStream_t st, ffi::ScratchAllocator scratch, BufF tape, BufF fric, BufF com,
                                        BufF ctrl, BufF q1, BufF v1, BufF gq_out, BufF gv_out, BufF gqn, BufF gxp,
                                        BufF gxq, BufF gcv, BufF gsc, BufF gcon, int32_t substeps, int32_t clamp_kv,
                                        int32_t normal, float clamp_soft, RetF gq, RetF gv, RetF gu,
                                        ffi::Result<ffi::Buffer<ffi::S32>> info) {
  const KinBwd k{&q1, &v1, &gqn, &gxp, &gxq, &gcv, &gsc};
  return bwd_impl(st, tape, fric, com, clamp_kv == 2 ? &ctrl : nullptr, gq_out, gv_out, substeps, clamp_kv, normal,
                  clamp_soft, gq, gv, gu, info, &k, &scratch, &gcon);
}
#define CON_ATTRS .Attr<int32_t>("substeps").Attr<int32_t>("clamp_kv").Attr<int32_t>("normal").Attr<float>("clamp_soft")
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepBwdCon, step_bwd_con_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG CON_ATTRS
                                  F32RET F32RET F32RET .Ret<ffi::Buffer<ffi::S32>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepBwdConKin, step_bwd_con_kin_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
                                  F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG F32ARG KIN5(F32ARG) F32ARG CON_ATTRS
                                  F32RET F32RET F32RET .Ret<ffi::Buffer<ffi::S32>>());
"""


def units(units_, spec, nvariants, ffi_unit="dk_ffi.cu"):
    """A contact_grad build's edits of the library's units (in place): every variant unit's <unit>_adjoint takes gcon
    after clamp_soft and passes it to k_adjoint; the FFI unit's declarations, bwd_impl (gcon: null for the existing
    handlers) and the handlers FFI (absent: the PyTorch library)."""
    sig = "int* info, const float* ctrl, float clamp_soft) {"
    for u in list(units_):
        t = units_[u]
        if sig in t:
            assert t.count(sig) == 1 and t.count("ADJ_SOFT_ARG(clamp_soft));") == 1
            units_[u] = (t.replace(sig, "int* info, const float* ctrl, float clamp_soft, const float* gcon) {")
                         .replace("ADJ_SOFT_ARG(clamp_soft));", "ADJ_SOFT_ARG(clamp_soft), gcon);"))
    if ffi_unit in units_:
        t = units_[ffi_unit]
        n = t.count("int*, const float*, float);")
        assert n == 1 + nvariants, n
        t = t.replace("int*, const float*, float);", "int*, const float*, float, const float*);")
        a = "ffi::ScratchAllocator* scratch = nullptr) {"
        b = "               clamp_soft);\n  if (rc) return cuda_error((cudaError_t)rc, \"k_adjoint launch\");"
        c = "  AdjointFn adj = kAdjoint[clamp_kv][normal];\n"
        assert t.count(a) == 1 and t.count(b) == 1 and t.count(c) == 1
        t = t.replace(a, "ffi::ScratchAllocator* scratch = nullptr,\n"
                         "                           const ffi::Buffer<ffi::F32>* gcon = nullptr) {  // (contact_vjp)")
        t = t.replace(b, b.replace("clamp_soft);", "clamp_soft, gcon ? gcon->typed_data() : nullptr);"))
        t = t.replace(c, f"  if (gcon && !dims_are(*gcon, {{B, {width_expr(spec)}}}))\n"
                         "    return bad(\"the contact output's cotangent must be (B, contact output width) float32\");\n"
                      + c)
        units_[ffi_unit] = t + FFI

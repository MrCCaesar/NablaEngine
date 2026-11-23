"""Per-world physics parameters (nablaengine.params.PhysicsParams) in the G1 engine's text: apply() edits the engine
(adjoint.cuda_source), the host kernels (host.KERNEL) and the kinematics (kinematics.KIN_DEVICE) of a build with
parameters; a build without them is unchanged."""
from nablaengine.g1_engine import model_hooks, source


def header(m, params, pw_off):
    """#defines of a build with parameters (prepended to the engine's text)."""
    lay = params.layout(m.nv)
    off = lambda f: lay[f][0] if f in lay else -1
    return (f"#define NPARAM {params.width(m.nv)}  // per-world parameters (nablaengine.params: {', '.join(lay)})\n"
            f"#define PW_OFF {pw_off}  // their offset in a world's com row\n"
            f"#define COM_STRIDE (PW_OFF + NPARAM)  // floats per world of the com input\n"
            f"#define PW_SOLREF {off('foot_solref')}\n#define PW_SOLIMP {off('foot_solimp')}\n"
            f"#define PW_DAMP {off('dof_damping')}\n#define PW_ARM {off('dof_armature')}\n"
            + (f"#define PW_TMASS {off('torso_mass')}\n#define PW_TINERT {off('torso_inertia')}\n"
               if "torso_mass" in lay or "torso_inertia" in lay else "")
            + (f"#define PW_TORS {off('foot_torsional')}\n" if "foot_torsional" in lay else ""))


STRIDE = (
    ("com[w * 3 + lane]", "com[(size_t)w * COM_STRIDE + lane]"),
    ("com[(size_t)w * 3 + lane]", "com[(size_t)w * COM_STRIDE + lane]"),
    ("com[(size_t)w * COM_WORDS + lane]", "com[(size_t)w * COM_STRIDE + lane]"),
    ("com + (size_t)w * COM_WORDS + 3", "com + (size_t)w * COM_STRIDE + 3"),
)
LOAD = ("  if (lane < 3) s.com[lane] = com[(size_t)w * COM_STRIDE + lane];\n",
        "  if (lane < 3) s.com[lane] = com[(size_t)w * COM_STRIDE + lane];\n"
        "  for (int i = lane; i < NPARAM; i += 32) s.pw[i] = com[(size_t)w * COM_STRIDE + PW_OFF + i];  // (world_params)\n"
        "  if (lane == 0) s.dampw = s.pw + (PW_DAMP >= 0 ? PW_DAMP : 0);\n")
STATE = (
    ("  float com[3], fric;\n",
     "  float com[3], fric;\n  float pw[NPARAM];  // the world's parameters (world_params)\n  const float* dampw;\n", 3),
    ("  float *qpos, *qvel;  // the kernel's outputs (implicitfast only)\n",
     "  float *qpos, *qvel;  // the kernel's outputs (implicitfast only)\n  const float* dampw;  // in Hand (world_params)\n",
     1),
    ("struct Hand {\n  float M[NTRI], qsm[NV], qacc0[NV];\n",
     "struct Hand {\n  float M[NTRI], qsm[NV], qacc0[NV];\n  float dampw[NV];  // the world's damping (world_params)\n", 1),
)
SPLIT = (
    ("  for (int i = lane; i < NV; i += 32) { g.qsm[i] = s.qsm[i]; g.qacc0[i] = s.qacc0[i]; }\n",
     "  for (int i = lane; i < NV; i += 32) { g.qsm[i] = s.qsm[i]; g.qacc0[i] = s.qacc0[i]; g.dampw[i] = s.dampw[i]; }\n",
     1),
    ("    s.qpos = q; s.qvel = v; s.qsm = g.qsm;", "    s.qpos = q; s.qvel = v; s.qsm = g.qsm; s.dampw = g.dampw;", 1),
)
FOOT = (
    ("        pos = s.con_dist[c]; sref = PSOLREF(pr); simp = PSOLIMP(pr);\n",
     "        pos = s.con_dist[c]; sref = PSOLREF(pr); simp = PSOLIMP(pr);\n"
     "        if (PFOOT(pr)) {  // the world's foot-pair impedance (world_params)\n"
     "          if (PW_SOLREF >= 0) sref = s.pw + (PW_SOLREF >= 0 ? PW_SOLREF : 0);\n"
     "          if (PW_SOLIMP >= 0) {\n"
     "            for (int k = 0; k < 4; k++) fsi[k] = s.pw[(PW_SOLIMP >= 0 ? PW_SOLIMP : 0) + k];\n"
     "            fsi[4] = PSOLIMP(pr)[4];\n"
     "            simp = fsi;\n"
     "          }\n"
     "        }\n", 1),
    ("      const float *sref, *simp;\n      float iw, pos = 0.f, mu = 0.f;\n",
     "      const float *sref, *simp;\n      float iw, pos = 0.f, mu = 0.f, fsi[5];\n", 1),
    ("  adj_row_impedance(PSOLREF(pair), PSOLIMP(pair), d, contact_invw(pair, mu), ri);\n",
     "  const float *sref = PSOLREF(pair), *simp = PSOLIMP(pair);\n"
     "  float fsi[5];\n"
     "  if (PFOOT(pair)) {  // the world's foot-pair impedance (world_params)\n"
     "    if (PW_SOLREF >= 0) sref = s.pw + (PW_SOLREF >= 0 ? PW_SOLREF : 0);\n"
     "    if (PW_SOLIMP >= 0) {\n"
     "      for (int k = 0; k < 4; k++) fsi[k] = s.pw[(PW_SOLIMP >= 0 ? PW_SOLIMP : 0) + k];\n"
     "      fsi[4] = PSOLIMP(pair)[4];\n"
     "      simp = fsi;\n"
     "    }\n"
     "  }\n"
     "  adj_row_impedance(sref, simp, d, contact_invw(pair, mu), ri);\n", 1),
)


LOAD_ADJ = (LOAD[0], LOAD[1] + "  for (int i = lane; i <= NPARAM; i += 32) s.gpw[i] = 0.f;  // (world_params)\n")
ADJ = (
    ("  unsigned clamp[(NV + 31) / 32];  // the record's actuator clamp mask (dof bits)\n",
     "  float gpw[NPARAM + 1];  // the parameters' cotangents and the foot friction's, summed over the substeps (world_params)\n"
     "  unsigned clamp[(NV + 31) / 32];  // the record's actuator clamp mask (dof bits)\n", 1),
    ("__device__ void adj_compose(Adj& s, int lane) {\n",
     """__device__ void adj_compose(Adj& s, int lane) {
  {  // the parameters' cotangents of this substep (world_params)
    const float h = c_misc[0];
    for (int i = lane; i < NV; i += 32) {
      const float y = -s.ny[i], aadv = s.a[i] - s.at[i];
      (void)y; (void)aadv; (void)h;
      if (PW_DAMP >= 0) s.gpw[(PW_DAMP >= 0 ? PW_DAMP : 0) + i] += -s.v[i] * s.lam[i] - h * y * aadv;
      if (PW_ARM >= 0) s.gpw[(PW_ARM >= 0 ? PW_ARM : 0) + i] += y * s.at[i] - s.a[i] * s.lam[i];
    }
  }
""", 1),
    ("  for (int i = lane; i < NV - 6; i += 32) gctrl[(size_t)w * (NV - 6) + i] = s.ubar[i];",
     "  for (int i = lane; i < NV - 6; i += 32) gctrl[(size_t)w * (NV - 5 + NPARAM) + i] = s.ubar[i];  // (world_params)\n"
     "  __syncwarp();\n"
     "  for (int i = lane; i <= NPARAM; i += 32) gctrl[(size_t)w * (NV - 5 + NPARAM) + NV - 6 + i] = s.gpw[i];", 1),
)
PARTIALS_DEVICE = r"""
// world_params: d Psi / d theta of one foot contact's quadratic rows, theta = (time constant, damping ratio, dmin, dmax,
// width, midpoint, mu) into pp[7] (added); the rows and their terms as contact_psi_vjp's dist cotangent
__device__ __noinline__ void contact_param_vjp(const float* Vv, const float* Va, const float* Vl, const float* fr,
                                               float mu, int dim, unsigned zone, float d, float D, float k, float b,
                                               float imp, float invw, const float* solref, const float* solimp,
                                               float* pp) {
  float vf[3], af[3], lf[3];
  for (int j = 0; j < 3; j++) { vf[j] = dot3(fr + 3 * j, Vv); af[j] = dot3(fr + 3 * j, Va); lf[j] = dot3(fr + 3 * j, Vl); }
  const float h = c_misc[0], tc = fmaxf(solref[0], RMUL(2.f, h)), dr = solref[1];
  const float dmin = fminf(fmaxf(solimp[0], 1e-4f), 0.9999f), dmax = fminf(fmaxf(solimp[1], 1e-4f), 0.9999f);
  const float width = fmaxf(1e-15f, solimp[2]), mid = fminf(fmaxf(solimp[3], 1e-4f), 0.9999f);
#ifdef ADJ_SOLIMP_POWER
  const float power = ADJ_SOLIMP_POWER;
#else
  const float power = fmaxf(1.f, solimp[4]);
#endif
  const float fmin_ = solimp[0] > 1e-4f && solimp[0] < 0.9999f ? 1.f : 0.f;  // dmin / dmax / mid not clamped
  const float fmax_ = solimp[1] > 1e-4f && solimp[1] < 0.9999f ? 1.f : 0.f;
  const float fmid = solimp[3] > 1e-4f && solimp[3] < 0.9999f ? 1.f : 0.f;
  float dk[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, db[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, di[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  // the standard form k = 1 / (dmax tc dr)^2, b = 2 / (dmax tc) (tc above its 2 h floor); the direct form, each entry on
  // its own as MuJoCo Warp's _contact_kbimp: solref[0] <= 0: k = -solref[0] / dmax^2, solref[1] <= 0: b = -solref[1] / dmax
  const bool kdir = solref[0] <= 0.f, bdir = solref[1] <= 0.f;
  if (kdir) dk[0] = -1.f / fmaxf(1e-15f, dmax * dmax);
  else {
    if (solref[0] > RMUL(2.f, h)) dk[0] = -2.f * k / tc;
    dk[1] = -2.f * k / dr;
  }
  if (bdir) db[1] = -1.f / fmaxf(1e-15f, dmax);
  else if (solref[0] > RMUL(2.f, h)) db[0] = -b / tc;
  dk[3] = -2.f * k / dmax * fmax_; db[3] = -b / dmax * fmax_;
  const float x = fabsf(d) / width;
  if (dmin == dmax || solimp[2] <= 1e-15f) { di[2] = 0.5f * fmin_; di[3] = 0.5f * fmax_; }
  else if (x <= 0.f) di[2] = fmin_;
  else if (x >= 1.f) di[3] = fmax_;
  else if (power == 1.f) { di[2] = (1.f - x) * fmin_; di[3] = x * fmax_; di[4] = -(dmax - dmin) * x / width; }
  else {
    const bool lo = x <= mid;
    const float raw = imp_raw(x, dmin, dmax, mid, power, lo);
    if (raw <= dmin) di[2] = fmin_;
    else if (raw >= dmax) di[3] = fmax_;
    else {
      const float yv = (raw - dmin) / (dmax - dmin), y_ = lo ? x : 1.f - x, ym = lo ? mid : 1.f - mid;
      const float dydx = power * powf(y_, power - 1.f) / powf(ym, power - 1.f);
      const float dydm = -(power - 1.f) * powf(y_, power) / powf(ym, power) * (lo ? 1.f : 1.f);
      di[2] = (1.f - yv) * fmin_; di[3] = yv * fmax_;
      di[4] = (dmax - dmin) * dydx * (-x / width);
      di[5] = (dmax - dmin) * dydm * fmid;
    }
  }
  const float om = 1.f - imp, R = invw * om / imp;
  const float dDdi = R > 1e-15f ? 1.f / (invw * om * om) : 0.f;
  const float dDdmu = (R > 1e-15f && dim == 3) ? -D * (2.f / mu + 2.f * mu / (1.f + mu * mu)) : 0.f;
  for (int e = 0; e < 4; e++) {
    if (!((zone >> e) & 1u) || (e > 0 && dim != 3)) continue;
    const int t = 1 + (e >> 1);
    const float se = dim == 3 ? ((e & 1) ? -1.f : 1.f) : 0.f, sm = se * mu;
    const float uv = vf[0] + sm * vf[t], ua = af[0] + sm * af[t], ul = lf[0] + sm * lf[t];
    const float xe = ua + k * imp * d + b * uv;
    for (int q = 0; q < 6; q++) pp[q] += ul * (dDdi * di[q] * xe + D * (d * (dk[q] * imp + k * di[q]) + db[q] * uv));
    pp[6] += dDdmu * xe * ul + D * se * ((af[t] + b * vf[t]) * ul + xe * lf[t]);
  }
}
"""
PARTIALS = (
    ("__device__ __forceinline__ float adj_contact_one(Adj& s, int c, int pair, int slot, unsigned zone, float* cs,\n"
     "                                                 float* dbg) {",
     PARTIALS_DEVICE + "__device__ __forceinline__ float adj_contact_one(Adj& s, int c, int pair, int slot, unsigned zone,"
     " float* cs,\n                                                 float* dbg, float* pp) {", 1),
    ("  contact_psi_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4);\n",
     "  contact_psi_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4);\n"
     "  if (PFOOT(pair) && zone)  // the foot pair's parameters (world_params)\n"
     "    contact_param_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, k, b, imp, contact_invw(pair, mu), sref, simp, pp);\n", 1),
    ("    float cw[CS_N];  // the lane's contact (adj_contact_one), read by the body lanes by shuffles\n",
     "    float cw[CS_N];  // the lane's contact (adj_contact_one), read by the body lanes by shuffles\n"
     "    float pp[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};  // its foot-pair parameter terms (world_params)\n", 1),
    ("      bad = !(adj_contact_one(s, c, pair, slot, zone, cw, dbg) < 0.f);\n",
     "      bad = !(adj_contact_one(s, c, pair, slot, zone, cw, dbg, pp) < 0.f);\n", 1),
    ("    const unsigned skip = __ballot_sync(0xffffffffu, on && !use), mis = __ballot_sync(0xffffffffu, bad);\n",
     "    for (int q = 0; q < 7; q++) {  // the batch's foot-pair terms, summed in a fixed order (world_params)\n"
     "      const float v = -warp_sum(pp[q]);  // (Psi's derivatives are the adjoint's P: the gradient is -P)\n"
     "      if (lane == 0) {\n"
     "        if (q < 2 && PW_SOLREF >= 0) s.gpw[(PW_SOLREF >= 0 ? PW_SOLREF : 0) + q] += v;\n"
     "        else if (q >= 2 && q < 6 && PW_SOLIMP >= 0) s.gpw[(PW_SOLIMP >= 0 ? PW_SOLIMP : 0) + q - 2] += v;\n"
     "        else if (q == 6) s.gpw[NPARAM] += v;\n"
     "      }\n"
     "    }\n"
     "    const unsigned skip = __ballot_sync(0xffffffffu, on && !use), mis = __ballot_sync(0xffffffffu, bad);\n", 1),
)
PARTIALS4_DEVICE = (PARTIALS_DEVICE
                    .replace("__device__ __noinline__ void contact_param_vjp(", "__device__ __noinline__ void contact_param_vjp4(")
                    .replace("                                               float* pp) {",
                             "                                               float* pp, const float* Ov, const float* Oa,\n"
                             "                                               const float* Ol, float mus) {")
                    .replace("(R > 1e-15f && dim == 3)", "(R > 1e-15f && dim >= 3)")
                    .replace("if (!((zone >> e) & 1u) || (e > 0 && dim != 3)) continue;",
                             "if (!((zone >> e) & 1u) || (e > 0 && dim < 3)) continue;")
                    .replace("const float se = dim == 3 ? ((e & 1) ? -1.f : 1.f) : 0.f, sm = se * mu;",
                             "const float se = dim >= 3 ? ((e & 1) ? -1.f : 1.f) : 0.f, sm = se * mu;")
                    .replace("""    pp[6] += dDdmu * xe * ul + D * se * ((af[t] + b * vf[t]) * ul + xe * lf[t]);
  }
}
""", """    pp[6] += dDdmu * xe * ul + D * se * ((af[t] + b * vf[t]) * ul + xe * lf[t]);
  }
  if (dim == 4) {  // the spin rows (condim 4): D and the impedance as the pyramid rows', u = n +- mus r
    const float wv = dot3(fr, Ov), wa = dot3(fr, Oa), wl = dot3(fr, Ol);
    for (int e = 4; e < 6; e++) {
      if (!((zone >> e) & 1u)) continue;
      const float se = (e & 1) ? -1.f : 1.f, sm = se * mus;
      const float uv = vf[0] + sm * wv, ua = af[0] + sm * wa, ul = lf[0] + sm * wl;
      const float xe = ua + k * imp * d + b * uv;
      for (int q = 0; q < 6; q++) pp[q] += ul * (dDdi * di[q] * xe + D * (d * (dk[q] * imp + k * di[q]) + db[q] * uv));
      pp[6] += dDdmu * xe * ul;
      pp[7] += D * se * ((wa + b * wv) * ul + xe * wl);
    }
  }
}
"""))
assert PARTIALS4_DEVICE.count("contact_param_vjp4(") == 1 and PARTIALS4_DEVICE.count("pp[7] +=") == 1


def partials4(tors):
    """PARTIALS for a condim-4 model (pp[8]; tors: foot_torsional, the foot pairs' mus from the world's parameters)."""
    mus = "(PFOOT(pair) ? s.pw[PW_TORS >= 0 ? PW_TORS : 0] : PMUS(pair))" if tors else "PMUS(pair)"
    call = ("  contact_psi_vjp4(Vv, Va, Vl, fr, mu, dim, zone, d, D, dD, k, b, imp, dimp, Wv, Wa, Wl, cot, cot + 4, Ov, Oa,"
            " Ol, dim == 4 ? PMUS(pair) : 0.f, AWv, AWa, AWl);\n")
    return (
        (PARTIALS[0][0], PARTIALS[0][1].replace(PARTIALS_DEVICE, PARTIALS4_DEVICE), 1),
        (call, call.replace("PMUS(pair)", mus)
         + "  if (PFOOT(pair) && zone)  // the foot pair's parameters (world_params)\n"
           "    contact_param_vjp4(Vv, Va, Vl, fr, mu, dim, zone, d, D, k, b, imp, contact_invw(pair, mu), sref, simp, pp,"
           f" Ov, Oa, Ol, dim == 4 ? {mus} : 0.f);\n", 1),
        (PARTIALS[2][0], PARTIALS[2][1].replace("float pp[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};",
                                                  "float pp[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};"), 1),
        PARTIALS[3],
        (PARTIALS[4][0], PARTIALS[4][1].replace("for (int q = 0; q < 7; q++)", "for (int q = 0; q < 8; q++)")
         .replace("        else if (q == 6) s.gpw[NPARAM] += v;\n",
                  "        else if (q == 6) s.gpw[NPARAM] += v;\n"
                  + ("        else if (q == 7) s.gpw[PW_TORS >= 0 ? PW_TORS : 0] += v;  // (foot_torsional)\n"
                     if tors else "")), 1),
    )


TORS_ENGINE = (
    ("con_frame[MAXCON][9], con_mu[MAXCON];", "con_frame[MAXCON][9], con_mu[MAXCON], con_mus[MAXCON];", 1),
    ("  float con_mu[MAXCON];\n", "  float con_mu[MAXCON];\n  float con_mus[MAXCON];  // (world_params: foot_torsional)\n", 2),
    ("  const float *qsm, *qacc0, *con_mu, *efc_aref;  // in Hand", "  const float *qsm, *qacc0, *con_mu, *con_mus, *efc_aref;  // in Hand", 1),
    ("  s.con_mu[c] = PFOOT(pair) ? s.fric : PMU(pair);\n",
     "  s.con_mu[c] = PFOOT(pair) ? s.fric : PMU(pair);\n"
     "  s.con_mus[c] = PFOOT(pair) ? fmaxf(s.pw[PW_TORS >= 0 ? PW_TORS : 0], 1e-5f) : PMUS(pair);  // (foot_torsional)\n", 1),
)
TORS_HOST = (
    ("g.con_mu[c] = s.con_mu[c];", "g.con_mu[c] = s.con_mu[c]; g.con_mus[c] = s.con_mus[c];", 1),
    ("s.con_mu = g.con_mu;", "s.con_mu = g.con_mu; s.con_mus = g.con_mus;", 1),
)


ARMATURE = (
    ("    s.M[TRI(j, j)] = dot3(sw, n) + dot3(sv, f) + c_armature[j];",
     "    s.M[TRI(j, j)] = dot3(sw, n) + dot3(sv, f) + s.pw[(PW_ARM >= 0 ? PW_ARM : 0) + j];  // (world_params)", 1),
    ("        s.M[TRI(j, i)] = dot3(kw, n) + dot3(kv, f) + (i == j ? c_armature[j] : 0.f);",
     "        s.M[TRI(j, i)] = dot3(kw, n) + dot3(kv, f) + (i == j ? s.pw[(PW_ARM >= 0 ? PW_ARM : 0) + j] : 0.f);", 1),
    ("float r = s.did[i] + c_armature[i] * s.a[i]", "float r = s.did[i] + s.pw[(PW_ARM >= 0 ? PW_ARM : 0) + i] * s.a[i]",
     1),
)


TORSO_DEVICE = r"""
// the torso's mass (k 0) and principal moments (k 1..3) of a world (pw: its parameters), the model's without the field
__device__ __forceinline__ float torso_param(const float* pw, int k) {
  const int b = (int)c_misc[1];
  if (k == 0) return PW_TMASS >= 0 ? pw[PW_TMASS >= 0 ? PW_TMASS : 0] : c_mass[b];
  return PW_TINERT >= 0 ? pw[(PW_TINERT >= 0 ? PW_TINERT : 0) + k - 1] : c_inert[3 * b + k - 1];
}
"""
BODY_IPOS = (
    r"""__device__ __forceinline__ void body_ipos(int b, const float* com, float* out) {
  for (int k = 0; k < 3; k++) out[k] = c_ipos[3 * b + k] + (b == (int)c_misc[1] ? com[k] : 0.f);
}
""",
    r"""__device__ __forceinline__ void body_ipos(int b, const float* com, float* out) {  // (world_params: out[7], com[7])
  for (int k = 0; k < 3; k++) out[k] = c_ipos[3 * b + k] + (b == (int)c_misc[1] ? com[k] : 0.f);
  out[3] = b == (int)c_misc[1] ? com[3] : c_mass[b];  // mass
  for (int k = 0; k < 3; k++) out[4 + k] = b == (int)c_misc[1] ? com[4 + k] : c_inert[3 * b + k];  // principal moments
}
""" + TORSO_DEVICE, 1)
TORSO_LOAD = ("  if (lane < 3) s.com[lane] = com[(size_t)w * COM_STRIDE + lane];\n",
              "  if (lane < 3) s.com[lane] = com[(size_t)w * COM_STRIDE + lane];\n"
              "  if (lane < 4) s.com[3 + lane] = torso_param(com + (size_t)w * COM_STRIDE + PW_OFF, lane);  // (world_params)\n")
TORSO_VJP = r"""// d psi_b / d theta of the torso's mass and principal moments (world_params): F1 = m f1, N1 = I AL + W x I W + c x F1,
// F2 = m f2, N2 = I AL2 + c x F2, f1 = A + AL x c + W x (W x c), f2 = A2 + AL2 x c, I = sum_k d_k r_k r_k' (r_k: the
// inertia frame's axes); psi_b = F1.Vl + N1.Wl + F2.Vy + N2.Wy
__device__ void torso_param_vjp(int b, const float* Q, const float* ipos, const float* W, const float* AL, const float* A,
                                const float* AL2, const float* A2, const float* Wl, const float* Vl, const float* Wy,
                                const float* Vy, float* dm, float* dI) {
  float c[3], t[3], wc[3], wwc[3], f1[3], f2[3], u[3], qi[4];
  qrot(Q, ipos, c);
  cross(W, c, wc);
  cross(W, wc, wwc);
  cross(AL, c, t);
  for (int k = 0; k < 3; k++) f1[k] = A[k] + t[k] + wwc[k];
  cross(AL2, c, t);
  for (int k = 0; k < 3; k++) f2[k] = A2[k] + t[k];
  *dm = dot3(f1, Vl) + dot3(f2, Vy);
  cross(c, f1, u);
  *dm += dot3(u, Wl);
  cross(c, f2, u);
  *dm += dot3(u, Wy);
  qmul(Q, c_iquat + 4 * b, qi);
  for (int k = 0; k < 3; k++) {
    float e[3] = {0.f, 0.f, 0.f}, r[3], wr[3];
    e[k] = 1.f;
    qrot(qi, e, r);
    cross(W, r, wr);
    dI[k] = dot3(r, AL) * dot3(r, Wl) + dot3(r, W) * dot3(wr, Wl) + dot3(r, AL2) * dot3(r, Wy);
  }
}

"""
TORSO_GRAD = r"""    if (b == (int)c_misc[1] && (PW_TMASS >= 0 || PW_TINERT >= 0)) {  // the torso's parameters: -d psi (world_params)
      float dm, dI[3];
      torso_param_vjp(b, s.Q[b], ip, s.W[b], s.AL[b], s.A[b], s.AL2[b], s.A2[b], s.Wl[b], s.Vl[b], s.Wy[b], s.Vy[b],
                      &dm, dI);
      if (PW_TMASS >= 0) s.gpw[PW_TMASS >= 0 ? PW_TMASS : 0] -= dm;
      if (PW_TINERT >= 0) for (int k = 0; k < 3; k++) s.gpw[(PW_TINERT >= 0 ? PW_TINERT : 0) + k] -= dI[k];
    }
"""
CALL_TAIL = ("                   s.bW[b], s.bAL[b], s.bA[b], s.bAL2[b], s.bA2[b], s.bWl[b], s.bVl[b], s.bWy[b], s.bVy[b],"
             " s.tb[b]);\n")
TORSO_ENGINE = (
    ("  float com[3], fric;\n", "  float com[7], fric;  // (world_params: the COM offset, the torso's mass and moments)\n", 3),
    BODY_IPOS,
    ("  float m = c_mass[b];\n  for (int k = 0; k < 3; k++) F[k] = m * (A[k] + u[k] + u2[k]);",
     "  float m = ipos[3];  // (world_params)\n  for (int k = 0; k < 3; k++) F[k] = m * (A[k] + u[k] + u2[k]);", 1),
    ("  for (int k = 0; k < 3; k++) l[k] *= c_inert[3 * b + k];\n", "  for (int k = 0; k < 3; k++) l[k] *= ipos[4 + k];\n", 2),
    ("  const float* I = c_inert + 3 * b;\n", "  const float* I = ipos + 4;  // (world_params)\n", 1),
    ("  float m = c_mass[b], cc = dot3(c, c);\n", "  float m = ipos[3], cc = dot3(c, c);\n", 1),
    ("  const float* d = c_inert + 3 * b;\n  const float m = c_mass[b];\n",
     "  const float* d = ipos + 4;  // (world_params)\n  const float m = ipos[3];\n", 1),
    ("__device__ void body_local_vjp(", TORSO_VJP + "__device__ void body_local_vjp(", 1),
    (CALL_TAIL, CALL_TAIL + TORSO_GRAD, 1),
)
TORSO_KIN = (
    ("  float qpos[NQ], qvel[NV], com[3];\n", "  float qpos[NQ], qvel[NV], com[7];  // (world_params)\n", 2),
)


def _torso(src, part):
    if "float ip[3]" not in src:
        raise RuntimeError("world_params: no ip[3] for the torso fields")
    src = src.replace("float ip[3]", "float ip[7]")
    if part == "engine":
        src = _apply(src, TORSO_ENGINE)
    else:
        src = _apply(src, TORSO_KIN)
        for a in ("    Mb = c_mass[b];\n", "    const float Mb = c_mass[b];\n", "    const float m = c_mass[b];\n"):
            src = src.replace(a, a.replace("c_mass[b]", "ip[3]"))
        if "c_mass[" in src:
            raise RuntimeError("world_params: the kinematics read c_mass outside the torso edits")
    if src.count(TORSO_LOAD[0]) < 1:
        raise RuntimeError("world_params: no com load for the torso fields")
    return src.replace(*TORSO_LOAD)


DIRECT_SOLREF = (
    ("  *k = 1.f / fmaxf(1e-15f, RMUL(RMUL(RMUL(RMUL(RMUL(dmax, dmax), tc), tc), dr), dr));\n"
     "  *b = 2.f / fmaxf(1e-15f, RMUL(dmax, tc));\n",
     "  *k = solref[0] <= 0.f ? -solref[0] / fmaxf(1e-15f, RMUL(dmax, dmax))  // direct form (world_params: foot_solref)\n"
     "                       : 1.f / fmaxf(1e-15f, RMUL(RMUL(RMUL(RMUL(RMUL(dmax, dmax), tc), tc), dr), dr));\n"
     "  *b = solref[1] <= 0.f ? -solref[1] / fmaxf(1e-15f, dmax) : 2.f / fmaxf(1e-15f, RMUL(dmax, tc));\n", 1),
)


def _apply(src, hooks):
    for a, b, n in hooks:
        if src.count(a) != n:
            raise RuntimeError(f"world_params anchor found {src.count(a)} times, expected {n}: {a[:70]!r}")
        src = src.replace(a, b)
    return src


COMPOSE_MOTOR = "__device__ void adj_compose(Adj& s, int lane, const float* rec) {\n"


def plain_compose(src, edit):
    """edit(src) with motor_hooks' adj_compose signature (the record added: a build with motor limits) taken as the
    plain one that ADJ's anchor names, and put back afterwards; without motor limits edit(src)."""
    plain = ADJ[1][0]
    if COMPOSE_MOTOR not in src:
        return edit(src)
    if src.count(COMPOSE_MOTOR) != 1 or plain in src:
        raise RuntimeError("world_params: adj_compose's signature found more than once")
    out = edit(src.replace(COMPOSE_MOTOR, plain))
    if out.count(plain) != 1:
        raise RuntimeError(f"world_params: adj_compose's signature found {out.count(plain)} times after the edits")
    return out.replace(plain, COMPOSE_MOTOR)


def apply(src, m, params, part, pw_off=3):
    """apply_params(src, m, params, part, pw_off), also on a build with motor limits (plain_compose)."""
    if not params:
        return src
    return plain_compose(src, lambda s: apply_params(s, m, params, part, pw_off))


def apply_params(src, m, params, part, pw_off=3):
    """src with the edits of a build with parameters params (unchanged without); part: "engine" (adjoint.cuda_source,
    after model_hooks), "host" (host.KERNEL) or "kinematics" (kinematics.KIN_DEVICE). pw_off: the parameters' offset in
    a com row (3, or a terrain-object build's com words)."""
    if not params:
        return src
    if int(m.opt.cone) == 1:
        raise ValueError("per-world physics parameters with the elliptic friction cone: not supported yet")
    for a, b in STRIDE:
        src = src.replace(a, b)
    torso = "torso_mass" in params.fields or "torso_inertia" in params.fields
    if part == "kinematics":
        return _torso(src, part) if torso else src
    n = 1 if part == "engine" else 2
    if src.count(LOAD[0]) != n:
        raise RuntimeError(f"world_params: com loads found {src.count(LOAD[0])} times, expected {n}")
    src = src.replace(*(LOAD_ADJ if part == "engine" else LOAD))
    from nablaengine.g1_engine import condim4
    c4, tors = condim4.applies(m), "foot_torsional" in params.fields
    if tors and not c4:
        raise ValueError("PhysicsParams foot_torsional: the model has no condim-4 pairs")
    if part == "host":
        src = _apply(src, SPLIT + (TORS_HOST if tors else ()))
        return src.replace(*TORSO_LOAD) if torso else src
    src = header(m, params, pw_off) + _apply(src, STATE + ADJ + FOOT + (partials4(tors) if c4 else PARTIALS))
    if tors:
        src = _apply(src, TORS_ENGINE)
        for a, b in (("PMUS(s.con_pair[a])", "s.con_mus[a]"), ("PMUS(s.con_pair[c])", "s.con_mus[c]")):
            if a not in src:
                raise RuntimeError(f"world_params: {a} not found (foot_torsional)")
            src = src.replace(a, b)
    if "foot_solref" in params.fields:
        src = _apply(src, DIRECT_SOLREF)
    if "dof_damping" in params.fields:
        if source.has_damping(m):
            if src.count("c_damp[i]") != 6:
                raise RuntimeError(f"world_params: c_damp[i] found {src.count('c_damp[i]')} times, expected 6")
            src = src.replace("c_damp[i]", "s.dampw[i]")
        else:
            src = _apply(src, tuple((a, b.replace("c_damp[i]", "s.dampw[i]"), k)
                                    for a, b, k in model_hooks.ENGINE_DAMPING[1:]))
    if "dof_armature" in params.fields:
        src = _apply(src, ARMATURE)
    if torso:
        src = _torso(src, part)
    return src

"""Text edits of the tree engine (g1_engine) for the BAM servo model (nablaengine.actuators.bam; init(actuator="bam")):
the forward, the adjoint and the state input / output. Without BAM no edits are made (the text unchanged)."""

PRE_BLOCK = """#if DK_BAM
  // BAM (bam_hooks): this substep's voltage scale and friction budget from the previous solve's torques
  float bsc;
  {
    float st = 0.f;
    for (int i = lane; i < NV; i += 32) if (i >= 6) st += fabsf(s.btau[i]);
    st = warp_sum(st);
    bsc = fmaxf(s.bvin - s.bdrop * st, BAM_VIN_MIN) * (1.f / BAM_VIN_NOM);
    for (int i = lane; i < NV; i += 32) s.bfl[i] = i >= 6 ? bam_budget(s.btau[i], s.bext[i], s.qvel[i]) : 0.f;
    if (rec) {  // tape: the scale and the torques the budget came from (then the servo forces, below)
      if (lane == 0) __stcs(rec + OFF_BAM, bsc);
      for (int i = lane; i < NV; i += 32) { __stcs(rec + OFF_BAM + 1 + i, s.btau[i]); __stcs(rec + OFF_BAM + 1 + NV + i, s.bext[i]); }
      for (int j = 1 + 4 * NV + lane; j < BAM_TAPE_WORDS; j += 32) __stcs(rec + OFF_BAM + j, 0.f);  // (the padding)
    }
    __syncwarp();
  }
#endif
"""
POST_BLOCK = """#if DK_BAM
  // BAM (bam_hooks): this solve's external torque, ext = M a - act + d v (substep_pre) - the dof's friction row force
  for (int i = lane; i < NV; i += 32) if (i >= 6) s.bext[i] += s.Ma[i];
  __syncwarp();
  for (int r = lane; r < NFLOSS; r += 32) {
    float ds, dd, c;
    row_eval(s, r, s.efc_jar[r], &ds, &dd, &c);
    s.bext[s.efc_a[r]] += ds;  // (the row's force is -ds)
  }
  __syncwarp();
#endif
"""
ACT_LOOP = "  for (int i0 = 0; i0 < NV; i0 += 32) {  // qfrc_smooth = clamped actuator force - bias\n"
IMPLICITFAST = "  // implicitfast: (M + h kv) qacc' = M qacc; then qvel, qpos\n"
QSM = "      s.qsm[i] = act - c_damp[i] * s.qvel[i] - s.qsm[i];  // the passive damping force (model_hooks)\n"

ENGINE = (
    ("  alignas(16) float cbuf[72];  // CHOL_SOLVE's scratch (chol_solve_bcast)\n  float com[3], fric;\n",
     "  alignas(16) float cbuf[72];  // CHOL_SOLVE's scratch (chol_solve_bcast)\n  float com[3], fric;\n"
     "  float btau[NV], bext[NV], bfl[NV], bvin, bdrop;  // BAM: previous solve's torques, friction budget (bam_hooks)\n", 1),
    ("  float com[3], fric;\n  int ncon, nefc, unsupported, min_nu_bits, nefc_dof;\n",
     "  float com[3], fric;\n  float *btau, *bext, *bfl, bvin, bdrop;  // BAM, in Hand (bam_hooks)\n"
     "  int ncon, nefc, unsupported, min_nu_bits, nefc_dof;\n", 1),
    ("  float *qpos, *qvel;  // the kernel's outputs (implicitfast only)\n",
     "  float *qpos, *qvel;  // the kernel's outputs (implicitfast only)\n  float *bext, *bfl;  // BAM, in Hand (bam_hooks)\n", 1),
    ("struct Hand {\n  float M[NTRI], qsm[NV], qacc0[NV];\n",
     "struct Hand {\n  float M[NTRI], qsm[NV], qacc0[NV];\n  float btau[NV], bext[NV], bfl[NV], bvin, bdrop;  // BAM (bam_hooks)\n", 1),
    (ACT_LOOP, PRE_BLOCK + ACT_LOOP, 1),
    ("        act = c_kp[i] * (ctrl[i - 6] - s.qpos[i + 1]) - c_kv[i] * s.qvel[i];\n",
     "        act = BAM_SCALE(c_kp[i]) * (ctrl[i - 6] - s.qpos[i + 1]) - c_kv[i] * s.qvel[i];\n"
     "#if BAM_CURRENT_LIMIT\n"
     "        {  // the firmware current limit: |duty vin - kt v| <= R I_max, as a force window KD v +- kt I_max (bam_hooks);\n"
     "           // at the window's bound the force is current-limited (the record's flag; at the bound itself too, as the\n"
     "           // clamp mask's test of the voltage limit and motor_limits' curve)\n"
     "          const float bk = BAM_KD * s.qvel[i];\n"
     "          act = fminf(fmaxf(act, bk - BAM_ILIM), bk + BAM_ILIM);\n"
     "          bcur = act <= bk - BAM_ILIM || act >= bk + BAM_ILIM ? 1.f : 0.f;\n"
     "        }\n"
     "#endif\n", 1),
    ("        const float ov = fmaxf((fabsf(act) - c_frclim[i]) / c_frclim[i], 0.f);  // the relative overshoot o\n",
     "        const float ov = fmaxf((fabsf(act) - BAM_SCALE(c_frclim[i])) / BAM_SCALE(c_frclim[i]), 0.f);  // the relative overshoot o\n", 1),
    ("        act = fminf(fmaxf(act, -c_frclim[i]), c_frclim[i]);\n"
     "        clamp = act <= -c_frclim[i] || act >= c_frclim[i];  // at the limit: MuJoCo Warp's actuator_lam test\n",
     "        act = fminf(fmaxf(act, -BAM_SCALE(c_frclim[i])), BAM_SCALE(c_frclim[i]));\n"
     "        clamp = act <= -BAM_SCALE(c_frclim[i]) || act >= BAM_SCALE(c_frclim[i]);  // at the limit: MuJoCo Warp's actuator_lam test\n", 1),
    (QSM, QSM + "#if DK_BAM\n      if (i >= 6) { s.btau[i] = act - BAM_KD * s.qvel[i]; s.bext[i] = c_damp[i] * s.qvel[i] - act; }  // (bam_hooks)\n"
     "      if (rec) { __stcs(rec + OFF_BAM + 1 + 2 * NV + i, act); __stcs(rec + OFF_BAM + 1 + 3 * NV + i, bcur); }\n"
     "#endif\n", 1),
    ("      float act = 0.f;\n",
     "      float act = 0.f;\n#if DK_BAM\n      float bcur = 0.f;  // 1: the current window limits the force (bam_hooks)\n#endif\n", 1),
    ("    tape_put(rec + OFF_LQ, s.L, NTRI, REC_WORDS - OFF_LQ, lane);\n",
     "    tape_put(rec + OFF_LQ, s.L, NTRI, OFF_BAM - OFF_LQ, lane);\n", 1),
    ("  return NFLOSS > 0 && r < NFLOSS ? c_floss[s.efc_a[r]] : 0.f;\n",
     "  return NFLOSS > 0 && r < NFLOSS ? BAM_FLOSS(s, s.efc_a[r]) : 0.f;\n", 1),
    (IMPLICITFAST, POST_BLOCK + IMPLICITFAST, 1),
)

HOST = (
    ("                          float* owarm, int* info, float* ocon, float* tape) {\n",
     "                          float* owarm, int* info, float* ocon, float* tape, const float* bam, float* obam) {\n", 1),
    ("  if (lane == 0) { s.fric = fric[w]; s.unsupported = 0; }\n",
     "  if (lane == 0) { s.fric = fric[w]; s.unsupported = 0; }\n"
     "#if DK_BAM\n"
     "  for (int i = lane; i < NV; i += 32) { s.btau[i] = bam[(size_t)w * BAM_STRIDE + i]; s.bext[i] = bam[(size_t)w * BAM_STRIDE + NV + i]; }\n"
     "  if (lane == 0) { s.bvin = bam[(size_t)w * BAM_STRIDE + 2 * NV]; s.bdrop = bam[(size_t)w * BAM_STRIDE + 2 * NV + 1]; }\n"
     "#endif\n", 1),
    ("  for (int i = lane; i < NV; i += 32) { oqvel[(size_t)w * NV + i] = s.qvel[i]; owarm[(size_t)w * NV + i] = s.qacc[i]; }\n",
     "  for (int i = lane; i < NV; i += 32) { oqvel[(size_t)w * NV + i] = s.qvel[i]; owarm[(size_t)w * NV + i] = s.qacc[i]; }\n"
     "#if DK_BAM\n"
     "  for (int i = lane; i < NV; i += 32) { obam[(size_t)w * BAM_STRIDE + i] = s.btau[i]; obam[(size_t)w * BAM_STRIDE + NV + i] = s.bext[i]; }\n"
     "  if (lane == 0) { obam[(size_t)w * BAM_STRIDE + 2 * NV] = s.bvin; obam[(size_t)w * BAM_STRIDE + 2 * NV + 1] = s.bdrop; }  // (passed through)\n"
     "#endif\n", 1),
    ("    s.efc_kind = g.efc_kind; s.efc_a = g.efc_a; s.efc_b = g.efc_b; s.efc_D = g.efc_D; s.efc_aref = g.efc_aref;\n  }\n",
     "    s.efc_kind = g.efc_kind; s.efc_a = g.efc_a; s.efc_b = g.efc_b; s.efc_D = g.efc_D; s.efc_aref = g.efc_aref;\n"
     "    s.btau = g.btau; s.bext = g.bext; s.bfl = g.bfl;  // BAM (bam_hooks)\n"
     "    if (k == 0) { g.bvin = bam[(size_t)w * BAM_STRIDE + 2 * NV]; g.bdrop = bam[(size_t)w * BAM_STRIDE + 2 * NV + 1]; }\n"
     "    s.bvin = bam[(size_t)w * BAM_STRIDE + 2 * NV]; s.bdrop = bam[(size_t)w * BAM_STRIDE + 2 * NV + 1];\n  }\n"
     "  if (k == 0)\n"
     "    for (int i = lane; i < NV; i += 32) { g.btau[i] = bam[(size_t)w * BAM_STRIDE + i]; g.bext[i] = bam[(size_t)w * BAM_STRIDE + NV + i]; }\n", 1),
    ("                                          const float* ctrl, const float* fric, const float* com, int k, int nsub,\n"
     "                                          float* ocon, float* tape) {\n",
     "                                          const float* ctrl, const float* fric, const float* com, int k, int nsub,\n"
     "                                          float* ocon, float* tape, const float* bam) {\n", 1),
    ("  pre_world(*reinterpret_cast<SimP*>(dk_smem), work[w], w, lane, qpos, qvel, ctrl, fric, com, k, nsub, ocon, tape);\n",
     "  pre_world(*reinterpret_cast<SimP*>(dk_smem), work[w], w, lane, qpos, qvel, ctrl, fric, com, k, nsub, ocon, tape, bam);\n", 1),
    ("                                       const float* fric, const float* com, int k, int nsub, Hand* work, float* ocon,\n"
     "                                       float* tape) {\n",
     "                                       const float* fric, const float* com, int k, int nsub, Hand* work, float* ocon,\n"
     "                                       float* tape, const float* bam) {\n", 1),
    ("    s.qpos = q; s.qvel = v; s.qsm = g.qsm; s.qacc0 = g.qacc0; s.con_mu = g.con_mu; s.efc_aref = g.efc_aref;\n",
     "    s.qpos = q; s.qvel = v; s.qsm = g.qsm; s.qacc0 = g.qacc0; s.con_mu = g.con_mu; s.efc_aref = g.efc_aref;\n"
     "    s.bext = const_cast<float*>(g.bext); s.bfl = const_cast<float*>(g.bfl);  // BAM (bam_hooks)\n", 1),
    ("  for (int i = lane; i < NV; i += 32) owarm[(size_t)w * NV + i] = s.qacc[i];\n  if (k == nsub - 1 && lane == 0) {\n",
     "  for (int i = lane; i < NV; i += 32) owarm[(size_t)w * NV + i] = s.qacc[i];\n"
     "  if (k == nsub - 1)  // BAM (bam_hooks): the state out\n"
     "    for (int i = lane; i < NV; i += 32) { obam[(size_t)w * BAM_STRIDE + i] = g.btau[i]; obam[(size_t)w * BAM_STRIDE + NV + i] = g.bext[i]; }\n"
     "  if (k == nsub - 1 && lane == 0) { obam[(size_t)w * BAM_STRIDE + 2 * NV] = g.bvin; obam[(size_t)w * BAM_STRIDE + 2 * NV + 1] = g.bdrop; }\n"
     "  if (k == nsub - 1 && lane == 0) {\n", 1),
    ("float* oqpos, float* oqvel, float* owarm, int* info, float* tape) {\n",
     "float* oqpos, float* oqvel, float* owarm, int* info, float* tape, float* obam) {\n", 2),
    ("                                         float* owarm, int* info, float* tape) {\n",
     "                                         float* owarm, int* info, float* tape, float* obam) {\n", 1),
    ("    post_world_on(*reinterpret_cast<SimN*>(smem), g, w, lane, qpos, qvel, warm, iterations, k, nsub, oqpos, oqvel,\n"
     "                  owarm, info, tape);\n",
     "    post_world_on(*reinterpret_cast<SimN*>(smem), g, w, lane, qpos, qvel, warm, iterations, k, nsub, oqpos, oqvel,\n"
     "                  owarm, info, tape, obam);\n", 1),
    ("    post_world_on(g.big, g, w, lane, qpos, qvel, warm, iterations, k, nsub, oqpos, oqvel, owarm, info, tape);\n",
     "    post_world_on(g.big, g, w, lane, qpos, qvel, warm, iterations, k, nsub, oqpos, oqvel, owarm, info, tape, obam);\n", 1),
    ("  post_world(dk_smem, work[w], w, lane, qpos, qvel, warm, iterations, k, nsub, oqpos, oqvel, owarm, info, tape);\n",
     "  post_world(dk_smem, work[w], w, lane, qpos, qvel, warm, iterations, k, nsub, oqpos, oqvel, owarm, info, tape, obam);\n", 1),
    ("                                           int* info, float* ocon, float* tape, void* work,\n",
     "                                           int* info, float* ocon, float* tape, const float* bam, float* obam, void* work,\n", 1),
    ("      k_pre<<<nw, 32, smem_pre, st>>>(nw, k ? oqpos : qpos, k ? oqvel : qvel, ctrl, fric, com, k, nsub, hand, ocon,\n"
     "                                      tape);\n",
     "      k_pre<<<nw, 32, smem_pre, st>>>(nw, k ? oqpos : qpos, k ? oqvel : qvel, ctrl, fric, com, k, nsub, hand, ocon,\n"
     "                                      tape, bam);\n", 1),
    ("                                        hand, oqpos, oqvel, owarm, info, tape);\n",
     "                                        hand, oqpos, oqvel, owarm, info, tape, obam);\n", 1),
)

FWD_SHIM = (
    ("float* ov, float* ow, int* info, float* ocon, float* tape, void* work) {\n",
     "float* ov, float* ow, int* info, float* ocon, float* tape, const float* bam, float* obam,\n"
     "                           void* work) {  // bam, obam: the BAM state rows in and out (bam_hooks)\n", 1),
    ("                               tape, work);\n#else\n", "                               tape, bam, obam, work);\n#else\n", 1),
    ("                                         tape);\n  return (int)cudaGetLastError();\n#endif\n",
     "                                         tape, bam, obam);\n  return (int)cudaGetLastError();\n#endif\n", 1),
)

FFI = (
    ("int, int, float*, float*, float*, int*, float*, float*, void*);",
     "int, int, float*, float*, float*, int*, float*, float*, const float*, float*, void*);", 2),
    ("                               RetF* kin = nullptr) {\n",
     "                               RetF* kin = nullptr, const BufF* act = nullptr, RetF* oact = nullptr) {\n", 1),
    ("  if (B > INT_MAX) return bad(\"B too large\");\n  if (B == 0) return ffi::Error::Success();\n"
     "  ffi::Error r = ensure_ready(st);\n  if (r.failure()) return r;\n  float* tp = tape ? (*tape)->typed_data() : nullptr;\n",
     "  if (!act || !oact)  // a BAM build (bam_hooks): the state rows are an input and an output of every forward\n"
     "    return bad(\"this build has the BAM servo model: its state act (B, ACT_WORDS) is an input and an output \"\n"
     "               \"(jax_ffi.forward / step with act=)\");\n"
     "  if (!dims_are(*act, {B, ACT_WORDS}) || !dims_are(**oact, {B, ACT_WORDS}))\n"
     "    return bad(\"act and act' must be (B, ACT_WORDS) float32\");\n"
     "  if (B > INT_MAX) return bad(\"B too large\");\n  if (B == 0) return ffi::Error::Success();\n"
     "  ffi::Error r = ensure_ready(st);\n  if (r.failure()) return r;\n  float* tp = tape ? (*tape)->typed_data() : nullptr;\n", 1),
    ("                    wo->typed_data(), info->typed_data(), con->typed_data(), tp, work);\n",
     "                    wo->typed_data(), info->typed_data(), con->typed_data(), tp, act->typed_data(),\n"
     "                    (*oact)->typed_data(), work);\n", 1),
    ("#define F32ARG .Arg<ffi::Buffer<ffi::F32>>()\n",
     "#define F32ARG .Arg<ffi::Buffer<ffi::F32>>()\n"
     "// a BAM build (bam_hooks): dk_step / dk_step_fwd with the state rows act (B, ACT_WORDS) after com and act' last\n"
     "static ffi::Error step_act_impl(cudaStream_t st, ffi::ScratchAllocator scratch, BufF qpos, BufF qvel, BufF warm,\n"
     "                                BufF ctrl, BufF fric, BufF com, BufF act, int32_t iterations, int32_t substeps,\n"
     "                                RetF qo, RetF vo, RetF wo, ffi::Result<ffi::Buffer<ffi::S32>> info, RetF con,\n"
     "                                RetF oact) {\n"
     "  return forward_impl(st, scratch, qpos, qvel, warm, ctrl, fric, com, iterations, substeps, qo, vo, wo, info, con,\n"
     "                      nullptr, nullptr, &act, &oact);\n"
     "}\n"
     "static ffi::Error step_fwd_act_impl(cudaStream_t st, ffi::ScratchAllocator scratch, BufF qpos, BufF qvel, BufF warm,\n"
     "                                    BufF ctrl, BufF fric, BufF com, BufF act, int32_t iterations, int32_t substeps,\n"
     "                                    RetF qo, RetF vo, RetF wo, ffi::Result<ffi::Buffer<ffi::S32>> info, RetF con,\n"
     "                                    RetF tape, RetF oact) {\n"
     "  return forward_impl(st, scratch, qpos, qvel, warm, ctrl, fric, com, iterations, substeps, qo, vo, wo, info, con,\n"
     "                      &tape, nullptr, &act, &oact);\n"
     "}\n"
     "#define FWD_ACT_BIND(B) B.Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()                    \\\n"
     "    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()                          \\\n"
     "    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()                          \\\n"
     "    .Arg<ffi::Buffer<ffi::F32>>().Attr<int32_t>(\"iterations\").Attr<int32_t>(\"substeps\")                          \\\n"
     "    .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::S32>>() \\\n"
     "    .Ret<ffi::Buffer<ffi::F32>>()\n"
     "XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepAct, step_act_impl, FWD_ACT_BIND(ffi::Ffi::Bind()).Ret<ffi::Buffer<ffi::F32>>());\n"
     "XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepFwdAct, step_fwd_act_impl,\n"
     "                              FWD_ACT_BIND(ffi::Ffi::Bind()).Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>());\n", 1),
)



ADJ_ROWS = """#if DK_BAM
  {  // BAM (bam_hooks): each friction row's piece at the recorded solution: D and D b on its quadratic piece, else the
     // sign of its force +-F (x = a_i + b v_i: force +F for x < 0, -F for x > 0)
    const unsigned below_b = (1u << lane) - 1u;
    int nf = 0;
    for (int i0 = 0; i0 < NV; i0 += 32) {
      const int i = i0 + lane;
      const bool on = i < NV && c_floss[i] > 0.f;
      const unsigned bal = __ballot_sync(0xffffffffu, on);
      if (on) {
        const int r = nf + __popc(bal & below_b);
        float ri[6];
        adj_row_impedance(c_dsolref, c_dsolimp, 0.f, c_dinvw[i], ri);
        const bool quad = (s.zone[r >> 5] >> (r & 31)) & 1u;
        s.bD[i] = quad ? ri[4] : 0.f; s.bDb[i] = quad ? ri[4] * ri[1] : 0.f;
        s.bsg[i] = quad ? 0.f : (s.a[i] + ri[1] * s.v[i] > 0.f ? -1.f : 1.f);
      }
      nf += __popc(bal);
    }
    __syncwarp();
  }
#endif
"""
ADJ_ABAR = """#if DK_BAM
  // BAM (bam_hooks): ext' = M a* - act + d v - (friction row force) sends its cotangent into a* bar: M e bar
  // (M = L_Q L_Q' - h diag(kv + d)) and D e bar on the quadratic friction rows (M's dependence on q: adj_compose)
  tri_mul_llt(s.L, s.beb, s.bw1, s.bw2, lane);
  for (int i = lane; i < NV; i += 32) s.abar[i] += s.bw2[i] - h * (KVI(i) + c_damp[i]) * s.beb[i] + s.bD[i] * s.beb[i];
  __syncwarp();
#endif
"""
ADJ_ACT_OLD = """  for (int i = 6 + lane; i < NV; i += 32) {
    float lt = LAM_ACT(s, i), kl = c_kp[i] * lt;
    s.Pq[i] += kl;
    s.Pv[i] += c_kv[i] * (DK_ADJ_CLAMP_KV ? lt : s.lam[i]);
    s.ubar[i - 6] += kl;
  }
"""
ADJ_ACT_NEW = """#if DK_BAM
  // BAM (bam_hooks): the servo force's cotangent is lam (the dynamics) + tau' bar - ext' bar; the force is
  // clamp(clamp(sc kp (u - q) - kv v, KD v +- kt I_max), +-sc frclim) (kv = 0): free, the scale's cotangent takes
  // (act + kv v) / sc of it; at the voltage limit (the clamp mask) act / sc; at the current limit (the flags) KD of it
  // goes to v (P_v -= KD phi, as the torque-speed limit's slope) and none to u, q or the scale
  float bscb = 0.f;
  for (int i = 6 + lane; i < NV; i += 32) {
    const float phi = s.lam[i] + s.btb[i] - s.beb[i];
    const bool cl = (s.clamp[i >> 5] >> (i & 31)) & 1u;
    const bool cu = !cl && ((s.bcur[i >> 5] >> (i & 31)) & 1u);
    const float lt = cl || cu ? 0.f : phi, kl = s.bsc * c_kp[i] * lt;
    s.Pq[i] += kl;
    s.Pv[i] += c_kv[i] * lt - (cu ? BAM_KD * phi : 0.f);
    s.ubar[i - 6] += kl;
    bscb += cu ? 0.f : (cl ? s.bact[i] : s.bact[i] + c_kv[i] * s.v[i]) / s.bsc * phi;
  }
#else
""" + ADJ_ACT_OLD + "#endif\n"
ADJ_END_OLD = """    for (int k = 0; k < 4; k++) s.qbar[3 + k] -= 2.f * t[k];
  }
  __syncwarp();
}
"""
ADJ_END_NEW = """    for (int k = 0; k < 4; k++) s.qbar[3 + k] -= 2.f * t[k];
  }
  __syncwarp();
#if DK_BAM
  {  // BAM (bam_hooks): the scale's and the friction budget's cotangents onto the substep's input torques, velocity
     // and the world's supply parameters; tau' = act - KD v and ext' = ... + d v - (friction row force) onto v
    float st = 0.f;
    for (int i = 6 + lane; i < NV; i += 32) st += fabsf(s.btau[i]);
    st = warp_sum(st);
    const float scb = warp_sum(bscb);
    const float g = s.bvin - s.bdrop * st > BAM_VIN_MIN ? scb * (1.f / BAM_VIN_NOM) : 0.f;
    if (lane == 0) { s.bgvin += g; s.bgdrop -= st * g; }
    for (int i = 6 + lane; i < NV; i += 32) {
      const float tb = s.btb[i], eb = s.beb[i];
      const float Fb = s.bsg[i] * (s.lam[i] - eb);  // the budget's cotangent: the row's force +-F, and -(that) in ext'
      float dt, de, dv;
      bam_budget_d(s.btau[i], s.bext[i], s.v[i], &dt, &de, &dv);
      s.vbar[i] += -BAM_KD * tb + (c_damp[i] + s.bDb[i]) * eb + Fb * dv;
      const float sg = s.btau[i] > 0.f ? 1.f : (s.btau[i] < 0.f ? -1.f : 0.f);
      s.btb[i] = -s.bdrop * sg * g + Fb * dt;
      s.beb[i] = Fb * de;
      s.bw1[i] = -eb;  // (the seed of the pass below)
    }
    for (int i = lane; i < NV; i += 32) { if (i < 6) s.bw1[i] = 0.f; s.bw2[i] = 0.f; }
    __syncwarp();
  }
  // ext' = M(q) a* - ... also depends on q through M: q bar += ext' bar . d(M a*)/dq, the sweeps' channel 2 (a seed ny
  // against d(M a~)/dq, no velocity, no gravity) run once more with a~ := a*, ny := -ext' bar and the lam channel off
  for (int i = lane; i < NV; i += 32) s.at[i] = s.a[i];
  __syncwarp();
  adj_tree_down_c2(s, s.bw1, lane);  // channel 2 alone: the poses are still the main pass's (Q, X persist)
  adj_tree_up_c2(s, s.bw1, lane);
  for (int i = lane; i < NV; i += 32) {
    if (i < 3) s.qbar[i] -= s.Pq[i];
    else if (i >= 6) s.qbar[i + 1] -= s.Pq[i];
    s.vbar[i] -= s.Pv[i];
  }
  if (lane == 0) {
    float g[4] = {0.f, s.Pq[3], s.Pq[4], s.Pq[5]}, t[4];
    qmul(s.q + 3, g, t);
    for (int k = 0; k < 4; k++) s.qbar[3 + k] -= 2.f * t[k];
  }
  __syncwarp();
#endif
}
"""
ADJ_PARAMS = ", const float* bam, const float* gbam_out, float* gbam_in"
ADJOINT = (
    ("  float com[3], fric;\n  unsigned clamp[(NV + 31) / 32];",
     "  float com[3], fric;\n"
     "  // BAM (bam_hooks): the record's scale, input torques and servo forces; the cotangents of tau, ext (the substep's\n"
     "  // output's, then its input's); the friction rows' D, D b, force sign; scratch; the world's supply and its gradient;\n"
     "  // the record's current-limit flags\n"
     "  float bsc, btau[NV], bext[NV], bact[NV], btb[NV], beb[NV], bD[NV], bDb[NV], bsg[NV], bw1[NV], bw2[NV];\n"
     "  float bvin, bdrop, bgvin, bgdrop;\n"
     "  unsigned bcur[(NV + 31) / 32];\n"
     "  unsigned clamp[(NV + 31) / 32];", 1),
    ("  if (lane < (NV + 31) / 32) s.clamp[lane] = __float_as_uint(__ldcs(rec + OFF_CLAMP + lane));\n",
     "  if (lane < (NV + 31) / 32) s.clamp[lane] = __float_as_uint(__ldcs(rec + OFF_CLAMP + lane));\n"
     "#if DK_BAM\n"
     "  if (lane == 0) s.bsc = __ldcs(rec + OFF_BAM);\n"
     "  for (int i0 = 0; i0 < NV; i0 += 32) {\n"
     "    const int i = i0 + lane;\n"
     "    bool cu = false;\n"
     "    if (i < NV) {\n"
     "      s.btau[i] = __ldcs(rec + OFF_BAM + 1 + i); s.bext[i] = __ldcs(rec + OFF_BAM + 1 + NV + i);\n"
     "      s.bact[i] = __ldcs(rec + OFF_BAM + 1 + 2 * NV + i);\n"
     "      cu = __ldcs(rec + OFF_BAM + 1 + 3 * NV + i) != 0.f;\n"
     "    }\n"
     "    const unsigned bal = __ballot_sync(0xffffffffu, cu);\n"
     "    if (lane == 0) s.bcur[i0 >> 5] = bal;\n"
     "  }\n"
     "#endif\n", 1),
    ("  __syncwarp();\n  APROF(APROF_RECORD);\n", "  __syncwarp();\n" + ADJ_ROWS + "  APROF(APROF_RECORD);\n", 1),
    ("  for (int e = lane; e < NTRI; e += 32) s.L[e] = __ldcs(rec + OFF_LH + e);\n  for (int i = lane; i < NV; i += 32) s.lam[i] = s.abar[i];\n",
     ADJ_ABAR + "  for (int e = lane; e < NTRI; e += 32) s.L[e] = __ldcs(rec + OFF_LH + e);\n"
     "  for (int i = lane; i < NV; i += 32) s.lam[i] = s.abar[i];\n", 1),
    (ADJ_ACT_OLD, ADJ_ACT_NEW, 1),
    (ADJ_END_OLD, ADJ_END_NEW, 1),
    ("                          const float* ctrl ADJ_SOFT_PARAM) {\n",
     "                          const float* ctrl ADJ_SOFT_PARAM" + ADJ_PARAMS + ") {\n", 1),
    ("  for (int i = lane; i < NV - 6; i += 32) s.ubar[i] = 0.f;\n  if (lane < 3) s.com[lane] = com[w * 3 + lane];\n",
     "  for (int i = lane; i < NV - 6; i += 32) s.ubar[i] = 0.f;\n"
     "#if DK_BAM\n"
     "  for (int i = lane; i < NV; i += 32) {  // the cotangent of the step's state output (zero without it)\n"
     "    s.btb[i] = gbam_out && i >= 6 ? gbam_out[(size_t)w * BAM_STRIDE + i] : 0.f;\n"
     "    s.beb[i] = gbam_out && i >= 6 ? gbam_out[(size_t)w * BAM_STRIDE + NV + i] : 0.f;\n"
     "    s.bD[i] = 0.f; s.bDb[i] = 0.f; s.bsg[i] = 0.f;\n"
     "  }\n"
     "  if (lane == 0) {\n"
     "    s.bvin = bam[(size_t)w * BAM_STRIDE + 2 * NV]; s.bdrop = bam[(size_t)w * BAM_STRIDE + 2 * NV + 1];\n"
     "    s.bgvin = gbam_out ? gbam_out[(size_t)w * BAM_STRIDE + 2 * NV] : 0.f;  // (the passed-through supply's cotangent)\n"
     "    s.bgdrop = gbam_out ? gbam_out[(size_t)w * BAM_STRIDE + 2 * NV + 1] : 0.f;\n"
     "  }\n"
     "#endif\n"
     "  if (lane < 3) s.com[lane] = com[w * 3 + lane];\n", 1),
    ("  for (int i = lane; i < NV - 6; i += 32) gctrl[(size_t)w * (NV - 6) + i] = s.ubar[i];\n",
     "  for (int i = lane; i < NV - 6; i += 32) gctrl[(size_t)w * (NV - 6) + i] = s.ubar[i];\n"
     "#if DK_BAM\n"
     "  __syncwarp();\n"
     "  for (int i = lane; i < NV; i += 32) {\n"
     "    gbam_in[(size_t)w * BAM_STRIDE + i] = i >= 6 ? s.btb[i] : 0.f;\n"
     "    gbam_in[(size_t)w * BAM_STRIDE + NV + i] = i >= 6 ? s.beb[i] : 0.f;\n"
     "  }\n"
     "  if (lane == 0) { gbam_in[(size_t)w * BAM_STRIDE + 2 * NV] = s.bgvin; gbam_in[(size_t)w * BAM_STRIDE + 2 * NV + 1] = s.bgdrop; }\n"
     "#endif\n", 1),
)
ADJ_GUARDS = """#ifdef DK_ADJ_DEBUG
#error "BAM (bam_hooks): the DK_ADJ_DEBUG record is not implemented"
#endif
"""
BUDGET_D = """// its derivatives with respect to tm, te and v (piecewise: the signs and the branch of the forward's values)
__device__ __forceinline__ void bam_budget_d(float tm, float te, float v, float* dt, float* de, float* dv) {
  const float x = fabsf(v) * BAM_RDTHETA, pw = powf(x, BAM_ALPHA), sb = expf(-pw);
  const float am = fabsf(tm), ae = fabsf(te);
  const float g1 = te * BAM_LF_EXT - tm * BAM_LF_MOT, g2 = te * BAM_LFS_EXT - tm * BAM_LFS_MOT;
  const float s1 = g1 > 0.f ? 1.f : (g1 < 0.f ? -1.f : 0.f), s2 = g2 > 0.f ? 1.f : (g2 < 0.f ? -1.f : 0.f);
  const bool drive = am > ae;
  *dt = -BAM_LF_MOT * s1 - sb * BAM_LFS_MOT * s2 + (drive ? 0.f : sb * 2.f * BAM_Q_MOT * tm);
  *de = BAM_LF_EXT * s1 + sb * BAM_LFS_EXT * s2 + (drive ? sb * 2.f * BAM_Q_EXT * te : 0.f);
  const float dsb = x > 0.f ? -sb * BAM_ALPHA * (pw / x) * BAM_RDTHETA * (v > 0.f ? 1.f : -1.f) : 0.f;
  *dv = dsb * (BAM_F_STRIB + fabsf(g2) + (drive ? BAM_Q_EXT * ae * ae : BAM_Q_MOT * am * am));
}
"""
ADJ_C2 = r"""#if DK_BAM
// BAM (bam_hooks): the pass for ext's dependence on q as channel 2 alone: adj_tree_down / adj_tree_up / body_local_vjp with the lambda
// channel's seed zero drop to their channel-2 and pose terms (the channel-1 terms are products with exact zeros), and
// the poses Q, X are still those of the main pass's adj_poses (members outside the unions), so it is not run again.
// Same terms in the same order as the full sweeps with lam = 0 (rounding may differ by the compiler's contractions).
__device__ void body_local_vjp_c2(int b, const float* Q, const float* ipos, const float* AL2, const float* A2,
                                  const float* Wy, const float* Vy, float* bAL2, float* bA2, float* bWy, float* bVy,
                                  float* tb) {
  float c[3], qi[4], t[3], u[3];
  qrot(Q, ipos, c);
  qmul(Q, c_iquat + 4 * b, qi);
  const float* d = c_inert + 3 * b;
  const float m = c_mass[b];
  float IL2[3], IWy[3], F2[3], g[3], cb[3], tau[3];
  inert_mul(qi, d, AL2, IL2);
  inert_mul(qi, d, Wy, IWy);
  cross(AL2, c, t);
  for (int k = 0; k < 3; k++) F2[k] = m * (A2[k] + t[k]);
  cross(c, F2, u);
  for (int k = 0; k < 3; k++) { bVy[k] = F2[k]; bWy[k] = IL2[k] + u[k]; }
  cross(Wy, c, t);
  for (int k = 0; k < 3; k++) g[k] = m * (Vy[k] + t[k]);
  cross(c, g, t);
  for (int k = 0; k < 3; k++) { bA2[k] = g[k]; bAL2[k] = t[k] + IWy[k]; }
  cross(F2, Wy, t);
  cross(g, AL2, u);
  for (int k = 0; k < 3; k++) cb[k] = t[k] + u[k];
  cross(IL2, Wy, t);
  cross(AL2, IWy, u);
  for (int k = 0; k < 3; k++) tau[k] = t[k] - u[k];
  cross(c, cb, t);
  for (int k = 0; k < 3; k++) tb[k] = tau[k] + t[k];
}

__device__ __noinline__ void adj_tree_down_c2(Adj& s, const float* ny, int lane) {
  LANE_BOUND(lane);
  const int b = lane, d = b < NB ? c_depth[b] : -1, p = b < NB ? c_parent[b] : 0;
  const int i = b >= 2 && b < NB ? c_dadr[b] : 0;
  float r[3];
  for (int k = 0; k < 3; k++) r[k] = b < NB ? s.X[b][k] - s.X[p][k] : 0.f;
  const float sa = s.at[i], sy = ny[i];
  for (int L = 1; L <= MAXD; L++) {
    if (d == L) {
      if (b >= 2) qrot(s.Q[b], c_axis + 3 * b, s.z[b]);
      adj_twist(b, s.Q[b], s.z[b], r, s.AL2[p], s.A2[p], s.at, sa, s.AL2[b], s.A2[b]);
      adj_twist(b, s.Q[b], s.z[b], r, s.Wy[p], s.Vy[p], ny, sy, s.Wy[b], s.Vy[b]);
    }
    __syncwarp();
  }
  if (b >= 1 && b < NB) {
    float ip[3];
    body_ipos(b, s.com, ip);
    body_local_vjp_c2(b, s.Q[b], ip, s.AL2[b], s.A2[b], s.Wy[b], s.Vy[b], s.bAL2[b], s.bA2[b], s.bWy[b], s.bVy[b], s.tb[b]);
    for (int k = 0; k < 3; k++) s.fb[b][k] = 0.f;
  }
  __syncwarp();
  for (int i = lane; i < NV; i += 32) { s.Pq[i] = 0.f; s.Pv[i] = 0.f; }
  __syncwarp();
}

__device__ __noinline__ void adj_tree_up_c2(Adj& s, const float* ny, int lane) {
  LANE_BOUND(lane);
  const int b = lane, d = b < NB ? c_depth[b] : -1, p = b < NB ? c_parent[b] : 0;
  for (int L = MAXD; L >= 1; L--) {
    if (d == L) {
      float bAL2[3], bA2[3], bWy[3], bVy[3], F[3], N[3], t[3];
      for (int k = 0; k < 3; k++) {
        bAL2[k] = s.bAL2[b][k]; bA2[k] = s.bA2[b][k]; bWy[k] = s.bWy[b][k]; bVy[k] = s.bVy[b][k];
        F[k] = s.fb[b][k]; N[k] = s.tb[b][k];
      }
      for (int j = 0; j < c_nchild[b]; j++) {
        const int c = c_child[3 * b + j];
        float r[3];
        for (int k = 0; k < 3; k++) r[k] = s.X[c][k] - s.X[b][k];
        cross(r, s.bA2[c], t);
        for (int k = 0; k < 3; k++) { bAL2[k] += s.bAL2[c][k] + t[k]; bA2[k] += s.bA2[c][k]; }
        cross(r, s.bVy[c], t);
        for (int k = 0; k < 3; k++) { bWy[k] += s.bWy[c][k] + t[k]; bVy[k] += s.bVy[c][k]; }
        cross(r, s.fb[c], t);
        for (int k = 0; k < 3; k++) { F[k] += s.fb[c][k] - s.rb[c][k]; N[k] += s.tb[c][k] + t[k]; }
      }
      if (b >= 2) {
        const int i = c_dadr[b];
        const float* z = s.z[b];
        float zb[3], rb[3];
        for (int k = 0; k < 3; k++) zb[k] = s.at[i] * bAL2[k] + ny[i] * bWy[k];
        cross(bA2, s.AL2[p], rb);
        cross(bVy, s.Wy[p], t);
        for (int k = 0; k < 3; k++) rb[k] += t[k];
        cross(z, zb, t);
        for (int k = 0; k < 3; k++) { N[k] += t[k]; F[k] += rb[k]; s.rb[b][k] = rb[k]; }
        s.Pq[i] += dot3(z, N);
      } else {  // root: AL2_1 and Wy_1 rotate with the root
        float l[3];
        cross(s.AL2[1], bAL2, t);
        for (int k = 0; k < 3; k++) N[k] += t[k];
        cross(s.Wy[1], bWy, t);
        for (int k = 0; k < 3; k++) N[k] += t[k];
        qrot_inv(s.Q[1], N, l);
        for (int k = 0; k < 3; k++) { s.Pq[k] += F[k]; s.Pq[3 + k] += l[k]; }
      }
      for (int k = 0; k < 3; k++) {
        s.bAL2[b][k] = bAL2[k]; s.bA2[b][k] = bA2[k]; s.bWy[b][k] = bWy[k]; s.bVy[b][k] = bVy[k];
        s.fb[b][k] = F[k]; s.tb[b][k] = N[k];
      }
    }
    __syncwarp();
  }
}
#endif

"""

ADJ_UNIT = (
    ("                           int* info, const float* ctrl, float clamp_soft) {\n",
     "                           int* info, const float* ctrl, float clamp_soft, const float* bam,\n"
     "                           const float* gbam_out, float* gbam_in) {  // the BAM state rows (bam_hooks)\n", 1),
    ("nullptr, DK_ADJ_CLAMP_KV == 2 ? ctrl : nullptr ADJ_SOFT_ARG(clamp_soft));\n",
     "nullptr, DK_ADJ_CLAMP_KV == 2 ? ctrl : nullptr ADJ_SOFT_ARG(clamp_soft),\n"
     "                                             bam, gbam_out, gbam_in);\n", 1),
)

FFI_ADJ = (
    ("int*, const float*, float);",
     "int*, const float*, float, const float*, const float*, float*);", 13),
    ("                           ffi::ScratchAllocator* scratch = nullptr) {\n",
     "                           ffi::ScratchAllocator* scratch = nullptr, const BufF* act = nullptr,\n"
     "                           const BufF* gact_out = nullptr, RetF* gact = nullptr) {\n", 1),
    ("  if (clamp_kv < 0 || clamp_kv > 2 || normal < 0 || normal > 3) return bad(\"clamp_kv in {0, 1, 2}, normal in 0..3\");\n",
     "  if (clamp_kv < 0 || clamp_kv > 2 || normal < 0 || normal > 3) return bad(\"clamp_kv in {0, 1, 2}, normal in 0..3\");\n"
     "  if (!act || !gact_out || !gact)  // a BAM build (bam_hooks): the state's input and output cotangent, its gradient\n"
     "    return bad(\"this build has the BAM servo model: its backward takes the state act (B, ACT_WORDS) and the \"\n"
     "               \"cotangent of act' (jax_ffi.backward with act=, g_act=)\");\n"
     "  if (!dims_are(*act, {B, ACT_WORDS}) || !dims_are(*gact_out, {B, ACT_WORDS}) || !dims_are(**gact, {B, ACT_WORDS}))\n"
     "    return bad(\"act, g_act' and g_act must be (B, ACT_WORDS) float32\");\n"
     "  if (clamp_kv == 2) return bad(\"the BAM servo model: clamp_kv 0 or 1 (its servo force has no kv; no soft clamp)\");\n", 1),
    ("               clamp_soft);\n",
     "               clamp_soft, (*act).typed_data(), (*gact_out).typed_data(), (**gact).typed_data());\n", 1),
    ("#define F32ARG .Arg<ffi::Buffer<ffi::F32>>()\n",
     "#define F32ARG .Arg<ffi::Buffer<ffi::F32>>()\n"
     "// a BAM build (bam_hooks): dk_step_bwd with the state act (B, ACT_WORDS) after com and the cotangent of act' after\n"
     "// g_qvel; its gradient g_act last\n"
     "static ffi::Error step_bwd_act_impl(cudaStream_t st, BufF tape, BufF fric, BufF com, BufF act, BufF gq_out,\n"
     "                                    BufF gv_out, BufF gact_out, int32_t substeps, int32_t clamp_kv, int32_t normal,\n"
     "                                    RetF gq, RetF gv, RetF gu, ffi::Result<ffi::Buffer<ffi::S32>> info, RetF gact) {\n"
     "  return bwd_impl(st, tape, fric, com, nullptr, gq_out, gv_out, substeps, clamp_kv, normal, 0.f, gq, gv, gu, info,\n"
     "                  nullptr, nullptr, &act, &gact_out, &gact);\n"
     "}\n"
     "XLA_FFI_DEFINE_HANDLER_SYMBOL(DkStepBwdAct, step_bwd_act_impl,\n"
     "                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()\n"
     "                                  .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()\n"
     "                                  .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()\n"
     "                                  .Arg<ffi::Buffer<ffi::F32>>()\n"
     "                                  .Attr<int32_t>(\"substeps\").Attr<int32_t>(\"clamp_kv\").Attr<int32_t>(\"normal\")\n"
     "                                  .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()\n"
     "                                  .Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::F32>>());\n", 1),
)


def _apply(src, hooks, what):
    for old, new, count in hooks:
        n = src.count(old)
        if n != count:
            raise RuntimeError(f"bam_hooks ({what}): {old[:90]!r} found {n} times, expected {count}")
        src = src.replace(old, new)
    return src


def engine(src, bam, nv):
    """adjoint.cuda_source's unit with BAM in its forward device code (bam: a nablaengine.actuators.Bam); the tape
    record grows by actuators.bam.tape_words(nv) words at OFF_BAM (its old length)."""
    import re
    from nablaengine.actuators import bam as B
    src = _apply(src, ENGINE, "engine")
    mt = re.search(r"^#define REC_WORDS (\d+)", src, re.M)
    if not mt or src.count("struct Sim {\n") != 1:
        raise RuntimeError("bam_hooks: REC_WORDS / struct Sim")
    old = int(mt.group(1))
    src = src.replace(mt.group(0), f"#define OFF_BAM {old}  // BAM (bam_hooks): scale, tau[NV], ext[NV], act[NV], cur[NV]\n"
                                   f"#define BAM_TAPE_WORDS {B.tape_words(nv)}\n"
                                   f"#define REC_WORDS {old + B.tape_words(nv)}", 1)
    src = _apply(src, ADJOINT, "adjoint")
    a = "__device__ void adj_compose(Adj& s, int lane) {"
    if src.count(a) != 1:
        raise RuntimeError("bam_hooks: adj_compose")
    src = src.replace(a, ADJ_C2 + a, 1)
    if "chol_tree_factor(" in src:
        src = _apply(src, (("  tri_mul_llt(s.L, s.beb, s.bw1, s.bw2, lane);\n",
                            "  tri_mul_rtr(s.L, s.beb, s.bw1, s.bw2, lane);  // (DK_TREE_CHOL: L_Q is R)\n", 1),), "tree_chol")
    return src.replace("struct Sim {\n", B.device_header(bam) + BUDGET_D + ADJ_GUARDS + "struct Sim {\n", 1)


def host(text):
    """host.KERNEL with the BAM state rows read by the first substep and written by the last."""
    return _apply(text, HOST, "host")


def adj_unit(text):
    """A variant unit's text with its adjoint entry <unit>_adjoint taking bam, gbam_out, gbam_in (after clamp_soft)."""
    return _apply(text, ADJ_UNIT, "adjoint entry")


def fwd_shim(text):
    """A forward unit's text with the forward entry taking bam, obam (before work)."""
    return _apply(text, FWD_SHIM, "forward entry")


def ffi(text, nv):
    """The FFI unit with the act input / output (ACT_WORDS) and the handlers DkStepAct, DkStepFwdAct, DkStepBwdAct."""
    text = _apply(_apply(text, FFI, "ffi"), FFI_ADJ, "ffi backward")
    anchor = "#define NQ "
    if text.count(anchor) != 1:
        raise RuntimeError("bam_hooks: the FFI unit's #define NQ")
    return text.replace(anchor, f"#define ACT_WORDS {2 * int(nv) + 2}  // the BAM state row (bam_hooks)\n" + anchor, 1)

"""The BAM servo model (nablaengine.actuators.bam; init(actuator="bam")) on the forest engine: g1_engine.bam_hooks' text
re-anchored onto the forest's actuator code, its split forward and its adjoint. Without BAM no edits are made (the text
unchanged)."""
from nablaengine.g1_engine import bam_hooks as T

ACT_OLD = "        act = c_kp[i] * u - c_kq[i] * s.qpos[DQADR(i)] - c_kv[i] * s.qvel[i];\n"
ACT_NEW = ("        act = BAM_SCALE(c_kp[i]) * u - BAM_SCALE(c_kq[i]) * s.qpos[DQADR(i)] - c_kv[i] * s.qvel[i];  "
           "// (bam_hooks)\n"
           "#if BAM_CURRENT_LIMIT\n"
           "        {  // the firmware current limit: |duty vin - kt v| <= R I_max, as a force window KD v +- kt I_max (bam_hooks);\n"
           "           // at the window's bound the force is current-limited (the record's flag)\n"
           "          const float bk = BAM_KD * s.qvel[i];\n"
           "          act = fminf(fmaxf(act, bk - BAM_ILIM), bk + BAM_ILIM);\n"
           "          bcur = act <= bk - BAM_ILIM || act >= bk + BAM_ILIM ? 1.f : 0.f;\n"
           "        }\n"
           "#endif\n")
QSM = "      s.qsm[i] = act - c_damp[i] * s.qvel[i] - s.qsm[i];\n"
SIMP = "  float qpos[NQ], qvel[NV], zero[NV], qsm[NV], qacc0[NV];\n  float com[3], fric;\n"

def _forest_sim(hook):
    a, b, n = hook
    return a.replace("cbuf[72]", "cbuf[CHOL_CB]"), b.replace("cbuf[72]", "cbuf[CHOL_CB]"), n


ENGINE = (
    _forest_sim(T.ENGINE[0]),
    (SIMP, SIMP + "  float *btau, *bext, *bfl, bvin, bdrop;  // BAM, in Hand (bam_hooks)\n", 1),
    T.ENGINE[2],
    T.ENGINE[3],
    (T.ACT_LOOP, T.PRE_BLOCK + T.ACT_LOOP, 1),
    (ACT_OLD, ACT_NEW, 1),
    ("        clamp = aclamp || act <= -c_frclim[i] || act >= c_frclim[i];  // at the limit: MuJoCo Warp's actuator_lam test\n",
     "        clamp = aclamp || act <= -BAM_SCALE(c_frclim[i]) || act >= BAM_SCALE(c_frclim[i]);  // (bam_hooks: the scaled "
     "limit)\n", 1),
    ("        act = fminf(fmaxf(act, -c_frclim[i]), c_frclim[i]);\n",
     "        act = fminf(fmaxf(act, -BAM_SCALE(c_frclim[i])), BAM_SCALE(c_frclim[i]));\n", 1),
    (QSM, QSM + "#if DK_BAM\n      if (i >= 6) { s.btau[i] = act - BAM_KD * s.qvel[i]; s.bext[i] = c_damp[i] * s.qvel[i] - act; }"
     "  // (bam_hooks)\n      if (rec) { __stcs(rec + OFF_BAM + 1 + 2 * NV + i, act); __stcs(rec + OFF_BAM + 1 + 3 * NV + i, bcur); }\n"
     "#endif\n", 1),
    ("      float act = 0.f;\n",
     "      float act = 0.f;\n#if DK_BAM\n      float bcur = 0.f;  // 1: the current window limits the force (bam_hooks)\n#endif\n", 1),
    T.ENGINE[10],
    T.ENGINE[11],
    (T.IMPLICITFAST, T.POST_BLOCK + T.IMPLICITFAST, 1),
)

ADJ_ACT_OLD = """  for (int i = lane; i < NV; i += 32) {  // force = kp u - kq q - kv v (unclamped dofs), passive -damping v
    const int a = ACT_OF(i);
    s.Pv[i] += c_damp[i] * s.lam[i];
    if (a < 0) continue;
    float lt = LAM_ACT(s, i);
    s.Pq[i] += c_kq[i] * lt;
    s.Pv[i] += c_kv[i] * (DK_ADJ_CLAMP_KV ? lt : s.lam[i]);
    if (!((s.uclamp[i >> 5] >> (i & 31)) & 1u)) s.ubar[a] += c_kp[i] * lt;  // ctrl at its ctrlrange: none
  }
"""
ADJ_ACT_NEW = """#if DK_BAM
  // BAM (bam_hooks): the servo force's cotangent is lam (the dynamics) + tau' bar - ext' bar; the force is
  // clamp(clamp(sc kp u - sc kq q - kv v, KD v +- kt I_max), +-sc frclim): free, the scale's cotangent takes
  // (act + kv v) / sc of it; at the voltage limit (the clamp mask) act / sc; at the current limit (the flags) KD of it
  // goes to v and none to u, q or the scale
  float bscb = 0.f;
  for (int i = lane; i < NV; i += 32) {
    const int a = ACT_OF(i);
    s.Pv[i] += c_damp[i] * s.lam[i];
    if (a < 0) continue;
    const float phi = s.lam[i] + s.btb[i] - s.beb[i];
    const bool cl = (s.clamp[i >> 5] >> (i & 31)) & 1u;
    const bool cu = !cl && ((s.bcur[i >> 5] >> (i & 31)) & 1u);
    const float lt = cl || cu ? 0.f : phi;
    s.Pq[i] += s.bsc * c_kq[i] * lt;
    s.Pv[i] += c_kv[i] * lt - (cu ? BAM_KD * phi : 0.f);
    if (!((s.uclamp[i >> 5] >> (i & 31)) & 1u)) s.ubar[a] += s.bsc * c_kp[i] * lt;  // ctrl at its ctrlrange: none
    bscb += cu ? 0.f : (cl ? s.bact[i] : s.bact[i] + c_kv[i] * s.v[i]) / s.bsc * phi;
  }
#else
""" + ADJ_ACT_OLD + "#endif\n"

ADJ_FOLD = """  for (int i = lane; i < NV; i += 32) s.vbar[i] -= s.Pv[i];
  for (int b = 1 + lane; b < NB; b += 32) {  // dof tangent to qpos: one lane per body
    const int qa = c_qadr[b], d0 = c_dadr[b];
    if (JT_OF(b) != JFREE) {
      s.qbar[qa] -= s.Pq[d0];
      continue;
    }
    for (int k = 0; k < 3; k++) s.qbar[qa + k] -= s.Pq[d0 + k];
    float g[4] = {0.f, s.Pq[d0 + 3], s.Pq[d0 + 4], s.Pq[d0 + 5]}, t[4];
    qmul(s.q + qa + 3, g, t);
    for (int k = 0; k < 4; k++) s.qbar[qa + 3 + k] -= 2.f * t[k];
  }
  __syncwarp();
"""
ADJ_END_OLD = """    for (int k = 0; k < 4; k++) s.qbar[qa + 3 + k] -= 2.f * t[k];
  }
  __syncwarp();
}
"""
ADJ_END_NEW = """    for (int k = 0; k < 4; k++) s.qbar[qa + 3 + k] -= 2.f * t[k];
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
    for (int i = lane; i < NV; i += 32) { if (i < 6) s.bw1[i] = 0.f; s.bw2[i] = 0.f; }  // (bw2: the zero seed)
    __syncwarp();
  }
  // ext' = M(q) a* - ... also depends on q through M: q bar += ext' bar . d(M a*)/dq, the sweeps' channel 2 (seed ny
  // against d(M a~)/dq, no velocity, no gravity) with a~ := a*, ny := -ext' bar and the lambda channel's seed zero;
  // adj_poses first (P_q, P_v share the memory of the poses' A .. Vy)
  for (int i = lane; i < NV; i += 32) s.at[i] = s.a[i];
  __syncwarp();
  adj_poses(s, lane);
  adj_tree_down(s, s.bw2, s.bw1, lane);
  adj_tree_up(s, s.bw2, s.bw1, lane);
  __syncwarp();
""" + ADJ_FOLD + """#endif
}
"""
ADJ_MEMBERS = "  unsigned clamp[(NV + 31) / 32];  // the record's actuator clamp mask (dof bits)\n"
ADJ_REC = "    s.uclamp[lane] = __float_as_uint(__ldcs(rec + OFF_UCLAMP + lane));\n  }\n"
ADJ_UBAR = "  for (int i = lane; i < NU; i += 32) s.ubar[i] = 0.f;\n"
ADJ_GCTRL = "  for (int i = lane; i < NU; i += 32) gctrl[(size_t)w * NU + i] = s.ubar[i];\n"


def _tree_block(old):
    for a, b, _ in T.ADJOINT:
        if a == old:
            assert b.startswith(a) or b.endswith(a), a[:60]
            return b[len(a):] if b.startswith(a) else b[:-len(a)]
    raise RuntimeError(f"forest bam_hooks: no tree hook {old[:60]!r}")


def _adjoint():
    t_members = T.ADJOINT[0][1][len("  float com[3], fric;\n"):-len("  unsigned clamp[(NV + 31) / 32];")]
    t_rec = _tree_block(T.ADJOINT[1][0])
    t_init = T.ADJOINT[7][1]
    t_init = t_init[t_init.index("#if DK_BAM\n"):t_init.index("#endif\n") + len("#endif\n")]
    t_out = _tree_block(T.ADJOINT[8][0])
    return (
        (ADJ_MEMBERS, t_members + ADJ_MEMBERS, 1),
        (ADJ_REC, ADJ_REC + t_rec, 1),
        T.ADJOINT[2],
        T.ADJOINT[3],
        (ADJ_ACT_OLD, ADJ_ACT_NEW, 1),
        (ADJ_END_OLD, ADJ_END_NEW, 1),
        T.ADJOINT[6],
        (ADJ_UBAR, ADJ_UBAR + t_init, 1),
        (ADJ_GCTRL, ADJ_GCTRL + t_out, 1),
    )


PRE_LOAD = ("    s.efc_kind = g.efc_kind; s.efc_a = g.efc_a; s.efc_b = g.efc_b; s.efc_D = g.efc_D; s.efc_aref = g.efc_aref;\n"
            "#if HAS_DEFER\n    s.cvq = nullptr;\n#endif\n  }\n")


def _host():
    tpre = T.HOST[3][1]
    k0 = "    s.btau = g.btau;"
    bam_lines = tpre[tpre.index(k0):tpre.index("  }\n  if (k == 0)\n")]
    after = tpre[tpre.index("  if (k == 0)\n"):]
    return (
        T.HOST[0], T.HOST[1], T.HOST[2],
        (PRE_LOAD, PRE_LOAD[:-len("  }\n")] + bam_lines + "  }\n" + after, 1),
        T.HOST[4], T.HOST[5], T.HOST[6],
        T.HOST[7],
        T.HOST[8],
        ("float* oqpos, float* oqvel, float* owarm, int* info, float* tape,\n"
         "                                              float* ocon, float (*fo)[3]) {\n",
         "float* oqpos, float* oqvel, float* owarm, int* info, float* tape,\n"
         "                                              float* ocon, float (*fo)[3], float* obam) {\n", 1),
        ("                                           float* oqpos, float* oqvel, float* owarm, int* info, float* tape,\n"
         "                                           float* ocon) {\n",
         "                                           float* oqpos, float* oqvel, float* owarm, int* info, float* tape,\n"
         "                                           float* ocon, float* obam) {\n", 1),
        ("                  owarm, info, tape, ocon, g.con_force);\n",
         "                  owarm, info, tape, ocon, g.con_force, obam);\n", 1),
        ("tape, ocon,\n                  g.con_force);\n", "tape, ocon,\n                  g.con_force, obam);\n", 1),
        ("                                         float* owarm, int* info, float* tape, float* ocon) {\n",
         "                                         float* owarm, int* info, float* tape, float* ocon, float* obam) {\n", 1),
        ("  post_world(dk_smem, work[w], w, lane, qpos, qvel, warm, iterations, k, nsub, oqpos, oqvel, owarm, info, tape, ocon);\n",
         "  post_world(dk_smem, work[w], w, lane, qpos, qvel, warm, iterations, k, nsub, oqpos, oqvel, owarm, info, tape, ocon,\n"
         "             obam);\n", 1),
        T.HOST[14],
        (T.HOST[15][0].replace("k_pre<<<nw, 32,", "k_pre<<<nw, PRE_THREADS,"),
         T.HOST[15][1].replace("k_pre<<<nw, 32,", "k_pre<<<nw, PRE_THREADS,"), 1),
        ("                                        hand, oqpos, oqvel, owarm, info, tape, ocon);\n",
         "                                        hand, oqpos, oqvel, owarm, info, tape, ocon, obam);\n", 1),
    )


KVI = """#ifndef KVI  // (bam_hooks: the implicit kv of dof i, off at the actuator forcerange)
#define KVI(i) (((s.kvoff[(i) >> 5] >> ((i) & 31)) & 1u) ? 0.f : c_kv[i])
#endif
#if DK_CLAMP_IMPLICIT == 2
#error "BAM (bam_hooks): not with DK_CLAMP_IMPLICIT=2"
#endif
"""


def check(m):
    """ValueError where the forest build of m cannot have BAM (beyond actuators.check_bam_build's restrictions)."""
    from nablaengine.forest_engine import step
    if step.has_defer(m):
        raise ValueError("BAM on the forest engine: not with the split forward's deferred pairs (meshes, $DK_DEFER_BOX)")
    if m.nv > 64:
        raise ValueError("BAM on the forest engine: at most 64 dofs")


def engine(src, bam, nv):
    """adjoint.cuda_source's unit (the forest engine) with BAM in its forward and adjoint (g1_engine.bam_hooks.engine's
    layout: OFF_BAM at the record's old length, actuators.bam.tape_words(nv) words more)."""
    import re
    from nablaengine.actuators import bam as B
    src = T._apply(src, ENGINE, "forest engine")
    mt = re.search(r"^#define REC_WORDS (\d+)", src, re.M)
    if not mt or src.count("struct Sim {\n") != 1:
        raise RuntimeError("forest bam_hooks: REC_WORDS / struct Sim")
    old = int(mt.group(1))
    src = src.replace(mt.group(0), f"#define OFF_BAM {old}  // BAM (bam_hooks): scale, tau[NV], ext[NV], act[NV], cur[NV]\n"
                                   f"#define BAM_TAPE_WORDS {B.tape_words(nv)}\n"
                                   f"#define REC_WORDS {old + B.tape_words(nv)}", 1)
    src = T._apply(src, _adjoint(), "forest adjoint")
    return src.replace("struct Sim {\n", B.device_header(bam) + T.BUDGET_D + T.ADJ_GUARDS + KVI + "struct Sim {\n", 1)


def host(text):
    """The forest's host.KERNEL with the BAM state rows read by the first substep and written by the last."""
    return T._apply(text, _host(), "forest host")


fwd_shim, adj_unit, ffi = T.fwd_shim, T.adj_unit, T.ffi

# The muscle curves and the activation dynamics here (mu_fl, mu_gain, mu_bias, mu_dyn, the activation update) follow
# MuJoCo Warp's util_misc.muscle_gain / muscle_bias / muscle_dynamics and support.next_act (mujoco_warp 3.17, Copyright
# 2025 The Newton Developers, Apache License 2.0), which mirror MuJoCo's mju_muscleGain / mju_muscleBias /
# mju_muscleDynamics and mj_nextActivation; the derivatives are this engine's.
"""General actuators in the forest engine: MuJoCo's muscles (dyn / gain / bias MUSCLE) and actuators with activation
dynamics (integrator, filter, filterexact), on joint (hinge / slide) and tendon (fixed and spatial) transmissions, with
the activation as a per-world state of the step (forward and adjoint)."""
import mujoco
import numpy as np

Dyn, Gain, Bias, Trn = mujoco.mjtDyn, mujoco.mjtGain, mujoco.mjtBias, mujoco.mjtTrn
DYN_CODE = {int(Dyn.mjDYN_NONE): 0, int(Dyn.mjDYN_INTEGRATOR): 1, int(Dyn.mjDYN_FILTER): 2,
            int(Dyn.mjDYN_FILTEREXACT): 3, int(Dyn.mjDYN_MUSCLE): 4}
GAIN_CODE = {int(Gain.mjGAIN_FIXED): 0, int(Gain.mjGAIN_AFFINE): 1, int(Gain.mjGAIN_MUSCLE): 2}
BIAS_CODE = {int(Bias.mjBIAS_NONE): 0, int(Bias.mjBIAS_AFFINE): 1, int(Bias.mjBIAS_MUSCLE): 2}
MINVAL = 1e-15
NDER = 5


def is_muscle(m, a):
    return (int(m.actuator_dyntype[a]) == int(Dyn.mjDYN_MUSCLE) or int(m.actuator_gaintype[a]) == int(Gain.mjGAIN_MUSCLE)
            or int(m.actuator_biastype[a]) == int(Bias.mjBIAS_MUSCLE))


def is_general(m, a):
    """Actuator a takes this module's path: activation dynamics or a muscle gain / bias."""
    return int(m.actuator_dyntype[a]) != int(Dyn.mjDYN_NONE) or is_muscle(m, a)


def general(m):
    """The general actuators, in actuator order."""
    return [a for a in range(int(m.nu)) if is_general(m, a)]


def applies(m):
    return any(is_general(m, a) for a in range(int(m.nu)))


def joint_general(m):
    """The general actuators on joints (each gets a virtual tendon), in actuator order."""
    return [a for a in general(m) if int(m.actuator_trntype[a]) == int(Trn.mjTRN_JOINT)]


def virtual_dof(m, t):
    """The dof of virtual tendon t (t >= ntendon)."""
    a = joint_general(m)[t - int(m.ntendon)]
    return int(m.jnt_dofadr[int(m.actuator_trnid[a, 0])])


def virtual_tendon(m, a):
    return int(m.ntendon) + joint_general(m).index(a)


def _vel_derivative_constant(m, a):
    gt = int(m.actuator_gaintype[a])
    return gt == int(Gain.mjGAIN_FIXED) or (gt == int(Gain.mjGAIN_AFFINE) and m.actuator_gainprm[a, 2] == 0.0)


def check(m):
    """[(actuator name, reason)] of the general actuators the forest engine does not support (empty: all)."""
    from nablaengine.forest_engine import model_hooks
    bad = []
    implicit = int(m.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    for a in general(m):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or str(a)
        trn, j = int(m.actuator_trntype[a]), int(m.actuator_trnid[a, 0])
        if trn == int(Trn.mjTRN_JOINT):
            if int(m.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                bad.append((nm, "general actuator: a joint transmission on a joint other than hinge / slide"))
        elif trn != int(Trn.mjTRN_TENDON):
            bad.append((nm, f"general actuator: transmission {Trn(trn).name} (joint or tendon only)"))
        if int(m.actuator_dyntype[a]) not in DYN_CODE:
            bad.append((nm, f"dyntype {Dyn(int(m.actuator_dyntype[a])).name} (none, integrator, filter, filterexact, "
                            "muscle)"))
        elif int(m.actuator_dyntype[a]) != int(Dyn.mjDYN_NONE) and int(m.actuator_actnum[a]) != 1:
            bad.append((nm, f"actnum {int(m.actuator_actnum[a])} (one activation per actuator)"))
        if int(m.actuator_gaintype[a]) not in GAIN_CODE:
            bad.append((nm, f"gaintype {Gain(int(m.actuator_gaintype[a])).name} (fixed, affine, muscle)"))
        if int(m.actuator_biastype[a]) not in BIAS_CODE:
            bad.append((nm, f"biastype {Bias(int(m.actuator_biastype[a])).name} (none, affine, muscle)"))
        if np.any(m.actuator_gear[a, 1:]):
            bad.append((nm, "gear beyond its first component"))
        if implicit and not _vel_derivative_constant(m, a):
            bad.append((nm, "implicitfast with a velocity-dependent gain (the muscle's FV, affine gainprm[2]): its "
                            "velocity derivative in the implicit matrix is not supported (Euler is)"))
        if m.actuator_forcelimited[a] and not m.actuator_forcerange[a, 0] <= m.actuator_forcerange[a, 1]:
            bad.append((nm, "forcerange lower > upper"))
        if trn == int(Trn.mjTRN_TENDON) and 0 <= j < m.ntendon:
            from nablaengine.forest_engine import tendon
            if m.tendon_actfrclimited[j]:
                bad.append((nm, "tendon actfrcrange"))
            if any(m.jnt_actfrclimited[int(m.dof_jntid[d])] for d, _ in tendon.support_list(m, j)):
                bad.append((nm, "a tendon through a joint with actuatorfrcrange"))
        elif trn == int(Trn.mjTRN_JOINT) and 0 <= j < m.njnt and m.jnt_actfrclimited[j]:
            bad.append((nm, "a joint with actuatorfrcrange (MuJoCo clamps the summed actuator force)"))
    if model_hooks.integrator(m) not in ("implicitfast", "euler", "explicit"):
        bad.append(("integrator", "general actuators need Euler or implicitfast"))
    return bad


def act_words(m):
    """The per-world activation state's words (MuJoCo's na) of a build with general actuators."""
    return int(m.na) if applies(m) else 0


def _f32(x):
    return f"{float(np.float32(x))!r}f"


def _ctrl_range(m, a):
    if m.actuator_ctrllimited[a] and not int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL):
        return tuple(m.actuator_ctrlrange[a])
    return (-1e30, 1e30)


def _prm(m, a, which):
    p = np.array((m.actuator_gainprm if which == "gain" else m.actuator_biasprm)[a, :9], np.float32)
    typ = int(m.actuator_gaintype[a] if which == "gain" else m.actuator_biastype[a])
    mus = typ == int(Gain.mjGAIN_MUSCLE) if which == "gain" else typ == int(Bias.mjBIAS_MUSCLE)
    if mus and p[2] < 0:
        p[2] = np.float32(p[3]) / np.float32(max(np.float32(MINVAL), np.float32(m.actuator_acc0[a])))
    return p


def plain_constants(m, a):
    """(kp, kq, kv, frc, ctrlrange) a general actuator gets in forest_engine.tendon's tables: only kv is read (the
    implicit matrix's constant velocity derivative -gear^2 biasprm[2], implicitfast; 0 otherwise)."""
    g = float(m.actuator_gear[a, 0])
    kv = -g * g * float(m.actuator_biasprm[a, 2]) if int(m.actuator_biastype[a]) == int(Bias.mjBIAS_AFFINE) else 0.0
    return 0.0, 0.0, kv, 1e30, (-1e30, 1e30)


def tables(m, acts):
    """The general actuators' tables over the tendon actuators acts [(actuator, tendon)] (forest_engine.tendon order)."""
    n1 = lambda n: max(n, 1)
    I = lambda xs: ", ".join(str(int(x)) for x in xs) or "0"
    F = lambda xs: ", ".join(_f32(x) for x in xs) or "0.f"
    tab = lambda typ, nm, n, vals, note="": (f"__device__ const {typ} {nm}[{n}] = {{{vals}}};"
                                            + (f"  // {note}" if note else "") + "\n")
    A = [a for a, _ in acts]
    gen = [is_general(m, a) for a in A]
    dyn = [DYN_CODE.get(int(m.actuator_dyntype[a]), 0) if g else 0 for a, g in zip(A, gen)]
    gt = [GAIN_CODE.get(int(m.actuator_gaintype[a]), 0) if g else 0 for a, g in zip(A, gen)]
    bt = [BIAS_CODE.get(int(m.actuator_biastype[a]), 0) if g else 0 for a, g in zip(A, gen)]
    adr = [int(m.actuator_actadr[a]) if g and d else -1 for a, g, d in zip(A, gen, dyn)]
    fl = [(1 if m.actuator_actearly[a] else 0) | (2 if m.actuator_actlimited[a] else 0) if g else 0
          for a, g in zip(A, gen)]
    ar = [x for a in A for x in (m.actuator_actrange[a] if m.actuator_actlimited[a] else (-1e30, 1e30))]
    fr = [x for a in A for x in (m.actuator_forcerange[a] if m.actuator_forcelimited[a] else (-1e30, 1e30))]
    cr = [x for a in A for x in _ctrl_range(m, a)]
    dp = [x for a in A for x in m.actuator_dynprm[a, :3]]
    gp = [x for a in A for x in _prm(m, a, "gain")]
    bp = [x for a in A for x in _prm(m, a, "bias")]
    lr = [x for a in A for x in m.actuator_lengthrange[a]]
    n = n1(len(A))
    return (f"#define NMACT {int(m.na)}  // the activation state's words (forest_engine.muscle; MuJoCo's act)\n"
            f"#define NMGEN {sum(gen)}  // general actuators among the NTACT\n"
            + tab("int", "c_tgen", n, I(gen), "tendon actuator: general (forest_engine.muscle)")
            + tab("int", "c_tdyn", n, I(dyn), "its dynamics: 0 none, 1 integrator, 2 filter, 3 filterexact, 4 muscle")
            + tab("int", "c_tgt", n, I(gt), "gain: 0 fixed, 1 affine, 2 muscle")
            + tab("int", "c_tbt", n, I(bt), "bias: 0 none, 1 affine, 2 muscle")
            + tab("int", "c_tmadr", n, I(adr), "its activation (-1: none)")
            + tab("int", "c_tmfl", n, I(fl), "bit 0 actearly, bit 1 actlimited")
            + tab("float", "c_tgear", n, F(m.actuator_gear[a, 0] for a in A), "gear")
            + tab("float", "c_tmar", 2 * n, F(ar), "actrange")
            + tab("float", "c_tmfr", 2 * n, F(fr), "forcerange")
            + tab("float", "c_tmcr", 2 * n, F(cr), "ctrlrange")
            + tab("float", "c_tmdp", 3 * n, F(dp), "dynprm[0..2]")
            + tab("float", "c_tmgp", 9 * n, F(gp), "gainprm[0..8] (muscle F0 resolved)")
            + tab("float", "c_tmbp", 9 * n, F(bp), "biasprm[0..8] (muscle F0 resolved)")
            + tab("float", "c_tmlr", 2 * n, F(lr), "lengthrange"))


DEVICE = r"""
// ---------------------------------------------------------------- general actuators (forest_engine.muscle)
#define MU_MINVAL 1e-15f
#define TS_ACT (NTEN + NTEN * KTEN + 2 * NTACT)  // the activation state in the tendon storage
#define OFF_MDER (OFF_TACT + 2 * NTACT)  // the record's slopes, 5 words per tendon actuator
// the muscle's length curve FL and its derivative (util_misc.muscle_gain_length / _deriv)
__device__ __forceinline__ float mu_fl(float L, float lmin, float lmax, float* d) {
  if (lmin > L || L > lmax) { *d = 0.f; return 0.f; }
  const float a = 0.5f * (lmin + 1.f), b = 0.5f * (1.f + lmax);
  if (L <= a) { const float w = fmaxf(MU_MINVAL, a - lmin), x = (L - lmin) / w; *d = x / w; return 0.5f * x * x; }
  if (L <= 1.f) { const float w = fmaxf(MU_MINVAL, 1.f - a), x = (1.f - L) / w; *d = x / w; return 1.f - 0.5f * x * x; }
  if (L <= b) { const float w = fmaxf(MU_MINVAL, b - 1.f), x = (L - 1.f) / w; *d = -x / w; return 1.f - 0.5f * x * x; }
  const float w = fmaxf(MU_MINVAL, lmax - b), x = (lmax - L) / w;
  *d = -x / w;
  return 0.5f * x * x;
}
// the muscle's active force -F0 FL(l) FV(v) and its derivatives in length, velocity (util_misc.muscle_gain, p the
// gainprm with F0 resolved, lr the lengthrange)
__device__ float mu_gain(const float* p, const float* lr, float len, float vel, float* dl, float* dv) {
  const float F = p[2], vmax = p[6], fvmax = p[8];
  const float L0 = (lr[1] - lr[0]) / fmaxf(MU_MINVAL, p[1] - p[0]);
  const float L = p[0] + (len - lr[0]) / fmaxf(MU_MINVAL, L0);
  const float V = vel / fmaxf(MU_MINVAL, L0 * vmax);
  float dFL;
  const float FL = mu_fl(L, p[4], p[5], &dFL);
  const float y = fvmax - 1.f;
  float FV, dFV;
  if (V <= -1.f) { FV = 0.f; dFV = 0.f; }
  else if (V <= 0.f) { FV = (V + 1.f) * (V + 1.f); dFV = 2.f * V + 2.f; }
  else if (V <= y) { FV = fvmax - (y - V) * (y - V) / fmaxf(MU_MINVAL, y); dFV = (-2.f * V + 2.f * y) / fmaxf(MU_MINVAL, y); }
  else { FV = fvmax; dFV = 0.f; }
  *dl = -F * dFL * FV / fmaxf(MU_MINVAL, L0);
  *dv = -F * FL * dFV / fmaxf(MU_MINVAL, L0 * vmax);
  return -F * FL * FV;
}
// the muscle's passive force -F0 FP(l) and its derivative in length (util_misc.muscle_bias)
__device__ float mu_bias(const float* p, const float* lr, float len, float* dl) {
  const float F = p[2], lmax = p[5], fpmax = p[7];
  const float L0 = (lr[1] - lr[0]) / fmaxf(MU_MINVAL, p[1] - p[0]);
  const float L = p[0] + (len - lr[0]) / fmaxf(MU_MINVAL, L0);
  const float b = 0.5f * (1.f + lmax), w = fmaxf(MU_MINVAL, b - 1.f);
  if (L <= 1.f) { *dl = 0.f; return 0.f; }
  if (L <= b) {
    const float x = (L - 1.f) / w;
    *dl = -F * fpmax * (x / w) / fmaxf(MU_MINVAL, L0);
    return -F * fpmax * 0.5f * x * x;
  }
  const float x = (L - b) / w;
  *dl = -F * fpmax * (1.f / w) / fmaxf(MU_MINVAL, L0);
  return -F * fpmax * (0.5f + x);
}
// the muscle's activation dynamics (util_misc.muscle_dynamics, p = dynprm: tau_act, tau_deact, tausmooth) and the
// derivatives of act_dot in act and in u (the ctrl after ctrlrange: 0 where clamp(u, 0, 1) is active)
__device__ float mu_dyn(float u, float act, const float* p, float* da, float* du) {
  const float uc = fminf(fmaxf(u, 0.f), 1.f), ac = fminf(fmaxf(act, 0.f), 1.f);
  const float c = 0.5f + 1.5f * ac;
  const float ta = p[0] * c, td = p[1] / c;
  const bool ain = act >= 0.f && act <= 1.f;  // (at 0 / 1 the one-sided slope inside [0, 1], where act stays)
  const float dta = ain ? 1.5f * p[0] : 0.f, dtd = ain ? -1.5f * p[1] / (c * c) : 0.f;
  const float sw = p[2], dctrl = uc - act;
  float tau, tac, tad;  // tau, d tau / d ac, d tau / d dctrl
  if (sw < MU_MINVAL) {
    const bool on = dctrl > 0.f;
    tau = on ? ta : td; tac = on ? dta : dtd; tad = 0.f;
  } else {
    const float x = dctrl / sw + 0.5f;
    float S = 0.f, Sd = 0.f;
    if (x >= 1.f) S = 1.f;
    else if (x > 0.f) { S = x * x * x * (3.f * x * (2.f * x - 5.f) + 10.f); Sd = 30.f * x * x * (x - 1.f) * (x - 1.f); }
    tau = td + (ta - td) * S; tac = dtd + (dta - dtd) * S; tad = (ta - td) * Sd / sw;
  }
  const float tm = fmaxf(MU_MINVAL, tau);
  const float r = dctrl / tm;
  const bool lim = !(tau > MU_MINVAL);
  *da = lim ? -1.f / tm : (-tm - dctrl * (tac - tad)) / (tm * tm);
  *du = (u > 0.f && u < 1.f) ? (lim ? 1.f / tm : (tm - dctrl * tad) / (tm * tm)) : 0.f;
  return r;
}
// the tendon actuators (lane per actuator, after sp_compute): the affine ones as sp_actuators, the general ones
// (c_tgen) with their activation (T[TS_ACT + c_tmadr], advanced in place) and, with a record, their slopes (OFF_MDER:
// dF/ds, dF/dlength, dF/dvelocity, d act'/d act, d act'/d u); force (times gear) and flags (bit 0 force at its
// forcerange, bit 1 ctrl at its ctrlrange, bit 2 act' at its actrange) into the storage and the tape
__device__ void mu_actuators(const float* qvel, const float* ctrl, float* T, float* rec, int lane) {
  const float h = c_misc[0];
  for (int k = lane; k < NTACT; k += 32) {
    const int t = c_tat[k], a = c_taa[k];
    const unsigned char* d = c_tdof + KTEN * t;
    const float* J = T + NTEN + KTEN * t;
    float jv = 0.f;
    for (int q = 0; q < c_tnd[t]; q++) jv = RFMA(J[q], qvel[d[q]], jv);
    if (!c_tgen[k]) {
      const float u = fminf(fmaxf(ctrl[a], c_tactr[2 * k]), c_tactr[2 * k + 1]);
      float f = RFMS(c_takv[k], jv, RFMS(c_takq[k], T[t], RMUL(c_takp[k], u)));
      const bool fc = f <= -c_tafrc[k] || f >= c_tafrc[k];
      f = fminf(fmaxf(f, -c_tafrc[k]), c_tafrc[k]);
      const int fl = (fc ? 1 : 0) | (u != ctrl[a] ? 2 : 0);
      T[NTEN + NTEN * KTEN + k] = f;
      T[NTEN + NTEN * KTEN + NTACT + k] = __int_as_float(fl);
      if (rec) { __stcs(rec + OFF_TACT + k, f); __stcs(rec + OFF_TACT + NTACT + k, __int_as_float(fl)); }
      continue;
    }
    const float g = c_tgear[k];
    const float u = fminf(fmaxf(ctrl[a], c_tmcr[2 * k]), c_tmcr[2 * k + 1]);
    int fl = u != ctrl[a] ? 2 : 0;
    float s = u, Na = 0.f, Nu = (fl & 2) ? 0.f : 1.f;  // (no dynamics: Nu = d s / d u)
    const int dyn = c_tdyn[k];
    if (dyn) {
      float* A = T + TS_ACT + c_tmadr[k];
      const float act = *A;
      const float* dp = c_tmdp + 3 * k;
      float adot, dda = 0.f, ddu = 0.f;
      if (dyn == 1) { adot = u; ddu = 1.f; }
      else if (dyn == 4) adot = mu_dyn(u, act, dp, &dda, &ddu);
      else { const float tau = fmaxf(dp[0], MU_MINVAL); adot = (u - act) / tau; dda = -1.f / tau; ddu = 1.f / tau; }
      float nx, sc;
      if (dyn == 3) {
        const float tau = fmaxf(MU_MINVAL, dp[0]);
        sc = tau * (1.f - expf(-h / tau));
        nx = act + adot * tau * (1.f - expf(-h / tau));
      } else {
        sc = h;
        nx = act + adot * h;
      }
      Na = 1.f + sc * dda;
      Nu = (fl & 2) ? 0.f : sc * ddu;
      if (c_tmfl[k] & 2) {
        const float cl = fminf(fmaxf(nx, c_tmar[2 * k]), c_tmar[2 * k + 1]);
        if (cl != nx) { Na = 0.f; Nu = 0.f; fl |= 4; }
        nx = cl;
      }
      s = (c_tmfl[k] & 1) ? nx : act;
      *A = nx;
    }
    const float len = g * T[t], vel = g * jv;
    const float* gp = c_tmgp + 9 * k;
    const float* bp = c_tmbp + 9 * k;
    const float* lr = c_tmlr + 2 * k;
    float gain, gl = 0.f, gv = 0.f, bias = 0.f, bl = 0.f, bv = 0.f;
    const int gt = c_tgt[k], bt = c_tbt[k];
    if (gt == 0) gain = gp[0];
    else if (gt == 1) { gain = gp[0] + gp[1] * len + gp[2] * vel; gl = gp[1]; gv = gp[2]; }
    else gain = mu_gain(gp, lr, len, vel, &gl, &gv);
    if (bt == 1) { bias = bp[0] + bp[1] * len + bp[2] * vel; bl = bp[1]; bv = bp[2]; }
    else if (bt == 2) bias = mu_bias(bp, lr, len, &bl);
    float f = gain * s + bias;
    const bool fc = f <= c_tmfr[2 * k] || f >= c_tmfr[2 * k + 1];
    f = fminf(fmaxf(f, c_tmfr[2 * k]), c_tmfr[2 * k + 1]);
    fl |= fc ? 1 : 0;
    const float gf = g * f;
    T[NTEN + NTEN * KTEN + k] = gf;
    T[NTEN + NTEN * KTEN + NTACT + k] = __int_as_float(fl);
    if (rec) {
      __stcs(rec + OFF_TACT + k, gf); __stcs(rec + OFF_TACT + NTACT + k, __int_as_float(fl));
      float* r = rec + OFF_MDER + 5 * k;
      __stcs(r, fc ? 0.f : gain); __stcs(r + 1, fc ? 0.f : gl * s + bl); __stcs(r + 2, fc ? 0.f : gv * s + bv);
      __stcs(r + 3, Na); __stcs(r + 4, Nu);
    }
  }
}
"""

DEV_AT = "// the tendon actuators' force on dof i (J f)\n__device__ __forceinline__ float sp_tact_qfrc(int i) {\n"
FWD_OLD = "  sp_actuators(s.qvel, ctrl, dk_tsp, rec, lane);\n"
FWD_NEW = "  mu_actuators(s.qvel, ctrl, dk_tsp, rec, lane);  // (forest_engine.muscle)\n"
TSW_OLD = "#define TS_WORDS (NTEN + NTEN * KTEN + 2 * NTACT)"
TSW_NEW = "#define TS_WORDS (NTEN + NTEN * KTEN + 2 * NTACT + NMACT)"
ADJ_MEMBER_AT = "  float tw[NTEN * KTEN];  // the spatial tendons' weights W of d/dq [J . W] (adj_spatial)\n"
ADJ_MEMBER = "  const float* mrec;  // the substep's record (forest_engine.muscle: the general actuators' slopes)\n"
ADJ_REPLAY_AT = "    s.tfl[k] = __float_as_int(__ldcs(rec + OFF_TACT + NTACT + k));\n  }\n"
ADJ_REPLAY = "  if (lane == 0) s.mrec = rec;  // (forest_engine.muscle)\n"
ADJ_W_OLD = ("        for (int q = 0; q < nd; q++) w[q] += -f * s.lam[d[q]] + c_takv[k] * (DK_ADJ_CLAMP_KV ? lt : cl) * "
             "s.v[d[q]];\n")
ADJ_W_NEW = ("        const float kvw = c_tgen[k] ? -c_tgear[k] * c_tgear[k] * __ldcs(s.mrec + OFF_MDER + 5 * k + 2) * lt\n"
             "                                    : c_takv[k] * (DK_ADJ_CLAMP_KV ? lt : cl);  // (forest_engine.muscle)\n"
             "        for (int q = 0; q < nd; q++) w[q] += -f * s.lam[d[q]] + kvw * s.v[d[q]];\n")
ADJ_P_OLD = ("      s.Pq[i] += c_takq[k] * c * lt;\n"
             "      s.Pv[i] += c_takv[k] * c * (DK_ADJ_CLAMP_KV ? lt : cl);\n")
ADJ_P_NEW = ("      if (c_tgen[k]) {  // a general actuator (forest_engine.muscle): kq, kv -> -gear^2 dF/dlength, dF/dvelocity\n"
             "        const float g2 = c_tgear[k] * c_tgear[k];\n"
             "        s.Pq[i] -= g2 * __ldcs(s.mrec + OFF_MDER + 5 * k + 1) * c * lt;\n"
             "        s.Pv[i] -= g2 * __ldcs(s.mrec + OFF_MDER + 5 * k + 2) * c * lt;\n"
             "        continue;\n"
             "      }\n" + ADJ_P_OLD)
ADJ_U_OLD = "      if (s.tfl[k] & 3) continue;\n"
ADJ_U_NEW = "      if (c_tgen[k] || (s.tfl[k] & 3)) continue;  // (general actuators: below)\n"
ADJ_ACT_AT = "  __syncwarp();\n  for (int i = lane; i < NV; i += 32) s.vbar[i] -= s.Pv[i];\n"
ADJ_ACT = """  __syncwarp();
  for (int k = lane; k < NTACT; k += 32) {  // general actuators (forest_engine.muscle): s bar = dF/ds gear (J . lam)
    if (!c_tgen[k]) continue;               // onto ctrl and the activation's cotangent (TS_ACT)
    const float* r = s.mrec + OFF_MDER + 5 * k;
    const float lt = (s.tfl[k] & 1) ? 0.f : c_tgear[k] * ten_jv(s.lam, c_tat[k]);
    const float sb = __ldcs(r) * lt, Na = __ldcs(r + 3), Nu = __ldcs(r + 4);
    const int ad = c_tmadr[k];
    if (ad < 0) { s.ubar[c_taa[k]] += sb * Nu; continue; }
    float* ab = s.ts_store + TS_ACT + ad;
    const float nb = (c_tmfl[k] & 1) ? *ab + sb : *ab;
    *ab = (c_tmfl[k] & 1) ? nb * Na : nb * Na + sb;
    s.ubar[c_taa[k]] += nb * Nu;
  }
"""


def _sub(src, a, b, count=1):
    n = src.count(a)
    if n != count:
        raise RuntimeError(f"forest muscle: {n} x {a[:90]!r} (expected {count})")
    return src.replace(a, b)


def apply(src, m):
    """src (the spatial build's unit text, after forest_engine.tendon / spatial) with the general actuators."""
    bad = check(m)
    if bad:
        raise ValueError(f"general actuators the forest engine does not support: {bad[:4]}")
    src = _sub(src, TSW_OLD, TSW_NEW)
    src = _sub(src, DEV_AT, DEVICE + DEV_AT)
    src = _sub(src, FWD_OLD, FWD_NEW)
    if "__device__ __noinline__ void adj_spatial(Adj& s, int lane)" in src:
        src = _sub(src, ADJ_MEMBER_AT, ADJ_MEMBER_AT + ADJ_MEMBER)
        src = _sub(src, ADJ_REPLAY_AT, ADJ_REPLAY_AT + ADJ_REPLAY)
        src = _sub(src, ADJ_W_OLD, ADJ_W_NEW)
        src = _sub(src, ADJ_P_OLD, ADJ_P_NEW)
        src = _sub(src, ADJ_U_OLD, ADJ_U_NEW)
        i = src.index("  for (int i = lane; i < NV; i += 32) {  // the tendons (forest_engine.tendon)\n")
        j = src.index(ADJ_ACT_AT, i)
        src = src[:j] + ADJ_ACT + src[j:]
        src = _act_io(src)
    return src


ADJ_IO = (
    ("  for (int i = lane; i < NU; i += 32) s.ubar[i] = 0.f;\n  if (lane < 3) s.com[lane] = com[(size_t)w * COM_WORDS + lane];\n",
     "  for (int i = lane; i < NU; i += 32) s.ubar[i] = 0.f;\n"
     "  for (int i = lane; i < NMACT; i += 32)  // the activation's cotangent: g_act' (forest_engine.muscle)\n"
     "    s.ts_store[TS_ACT + i] = gbam_out ? gbam_out[(size_t)w * NMACT + i] : 0.f;\n"
     "  if (lane < 3) s.com[lane] = com[(size_t)w * COM_WORDS + lane];\n"),
    ("  for (int i = lane; i < NU; i += 32) gctrl[(size_t)w * NU + i] = s.ubar[i];\n",
     "  for (int i = lane; i < NU; i += 32) gctrl[(size_t)w * NU + i] = s.ubar[i];\n"
     "  for (int i = lane; i < NMACT; i += 32) gbam_in[(size_t)w * NMACT + i] = s.ts_store[TS_ACT + i];  // g_act\n"),
)


def _act_io(src):
    from nablaengine.g1_engine import bam_hooks as T
    a, b, n = T.ADJOINT[6]
    src = _sub(src, a, b, n)
    for a, b in ADJ_IO:
        src = _sub(src, a, b)
    return src


def _host_items():
    from nablaengine.forest_engine import bam_hooks as FB
    from nablaengine.g1_engine import bam_hooks as T
    items = list(FB._host())
    load = ("  { float* T_ = ts_base(s);  // the activation state in (forest_engine.muscle)\n"
            "    for (int i = lane; i < NMACT; i += 32) T_[TS_ACT + i] = bam[(size_t)w * NMACT + i]; }\n")
    store = ("  { const float* T_ = ts_base(s);  // the activation state out (forest_engine.muscle)\n"
             "    for (int i = lane; i < NMACT; i += 32) obam[(size_t)w * NMACT + i] = T_[TS_ACT + i]; }\n")
    a1 = T.HOST[1][0]
    items[1] = (a1, a1 + load, 1)
    a2 = T.HOST[2][0]
    items[2] = (a2, a2 + store, 1)
    pre = "  if (DK_PRE2W) __syncthreads(); else __syncwarp();\n#if CV_DEFER\n"
    items[3] = (pre, "  if (DK_PRE2W) __syncthreads(); else __syncwarp();\n"
                "  if (k == 0) {  // the activation state in, into Hand's tendon storage (forest_engine.muscle)\n"
                "    float* T_ = ts_base(g);\n"
                "    for (int i = lane; i < NMACT; i += 32) T_[TS_ACT + i] = bam[(size_t)w * NMACT + i];\n"
                "    __syncwarp();\n  }\n#if CV_DEFER\n", 1)
    items[7] = None
    a8 = "  for (int i = lane; i < NV; i += 32) owarm[(size_t)w * NV + i] = s.qacc[i];\n  if (k == nsub - 1 && lane == 0) {\n"
    items[8] = (a8, "  for (int i = lane; i < NV; i += 32) owarm[(size_t)w * NV + i] = s.qacc[i];\n"
                "  if (k == nsub - 1) {  // the activation state out (forest_engine.muscle)\n"
                "    const float* T_ = ts_base(g);\n"
                "    for (int i = lane; i < NMACT; i += 32) obam[(size_t)w * NMACT + i] = T_[TS_ACT + i];\n"
                "  }\n  if (k == nsub - 1 && lane == 0) {\n", 1)
    assert sum(1 for x in items if x is not None and x[0].startswith("      k_pre<<<nw, PRE_THREADS, smem_pre, st>>>")) == 1
    return [x for x in items if x is not None]


def host(text):
    """The forest's host.KERNEL with the activation state read by the first substep and written after the last (the
    act / act' pointers of forest_engine.bam_hooks' host signatures)."""
    from nablaengine.g1_engine import bam_hooks as T
    return T._apply(text, _host_items(), "forest muscle host")


def ffi(text, na):
    """The FFI unit with the act input / output (ACT_WORDS = na) and the handlers DkStepAct, DkStepFwdAct, DkStepBwdAct
    (forest_engine.bam_hooks' FFI text, the messages naming the activation state)."""
    from nablaengine.g1_engine import bam_hooks as T
    text = T._apply(T._apply(text, T.FFI, "ffi"), T.FFI_ADJ, "ffi backward")
    text = text.replace("this build has the BAM servo model: its state act (B, ACT_WORDS)",
                        "this build has actuators with activation: its state act (B, ACT_WORDS)")
    text = text.replace("this build has the BAM servo model: its backward takes the state act (B, ACT_WORDS)",
                        "this build has actuators with activation: its backward takes the state act (B, ACT_WORDS)")
    text = text.replace("the BAM servo model: clamp_kv 0 or 1 (its servo force has no kv; no soft clamp)",
                        "actuators with activation: clamp_kv 0 or 1 (no soft clamp)")
    anchor = "#define NQ "
    if text.count(anchor) != 1:
        raise RuntimeError("forest muscle: the FFI unit's #define NQ")
    return text.replace(anchor, f"#define ACT_WORDS {int(na)}  // the activation state (forest_engine.muscle)\n" + anchor, 1)


def fwd_shim(text):
    from nablaengine.g1_engine import bam_hooks as T
    return T.fwd_shim(text)


def adj_unit(text):
    from nablaengine.g1_engine import bam_hooks as T
    return T.adj_unit(text)

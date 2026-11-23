"""Text edits of the forest engine for actuator torque-speed limits (nablaengine.motor; init(motor_limits=...))."""

from nablaengine.g1_engine import motor_hooks as g1_motor_hooks
from nablaengine.g1_engine.motor_hooks import _lit
from nablaengine.forest_engine import step

check = g1_motor_hooks.check


def hooks(m, motor):
    """The (anchor, replacement) edits for motor (a nablaengine.motor.MotorLimits, or None: ())."""
    if motor is None:
        return ()
    check(motor)
    assert step.clamp_implicit()[0] != 2
    t = motor.table(m).ravel()
    lit = ", ".join(_lit(x) for x in t)
    rfms = "#define RFMS(a, b, c) __fmaf_rn(-(a), (b), (c))  // c - a b, one rounding\n"
    frc = "        act = fminf(fmaxf(act, -c_frclim[i]), c_frclim[i]);\n      }\n"
    return (
        (rfms, rfms + f"__device__ const float c_motor[NV * 6] = {{{lit}}};  // nablaengine.motor\n"
         + g1_motor_hooks.DEVICE),
        ("    bool clamp = false;\n    bool aclamp = false, uclamp = false;\n",
         "    float msl = 0.f;  // -df/dqdot of the torque-speed limited force (nablaengine.motor), the record's mslope\n"
         "    bool clamp = false;\n    bool aclamp = false, uclamp = false;\n"),
        (frc,
         "        act = fminf(fmaxf(act, -c_frclim[i]), c_frclim[i]);\n"
         "        {  // the torque-speed limit (nablaengine.motor): |f| <= L(|qdot|)\n"
         "          float sl;\n"
         "          const float lim = motor_lim(i, act, s.qvel[i], &sl);\n"
         "          if (fabsf(act) >= lim) { act = copysignf(lim, act); clamp = true; msl = sl; }\n"
         "        }\n      }\n"),
        ("      s.qsm[i] = act - c_damp[i] * s.qvel[i] - s.qsm[i];\n",
         "      if (rec) __stcs(rec + OFF_MSLOPE + i, msl);\n"
         "      s.qsm[i] = act - c_damp[i] * s.qvel[i] - s.qsm[i];\n"),
        ("    tape_put(rec + OFF_QADV, s.tmp, NV, OFF_LH - OFF_QADV, lane);\n",
         "    tape_put(rec + OFF_QADV, s.tmp, NV, NV, lane);\n"
         "    for (int i = OFF_MSLOPE + NV + lane; i < OFF_LH; i += 32) __stcs(rec + i, 0.f);\n"),
        ("__device__ void adj_compose(Adj& s, int lane) {\n",
         "__device__ void adj_compose(Adj& s, int lane, const float* rec) {\n"),
        ("  adj_compose(s, lane);  // A8\n", "  adj_compose(s, lane, rec);  // A8\n"),
        ("    if (a < 0) continue;\n    float lt = LAM_ACT(s, i);\n",
         "    if (a < 0) continue;\n"
         "    s.Pv[i] += __ldcs(rec + OFF_MSLOPE + i) * s.lam[i];  // the torque-speed limit's slope (nablaengine.motor)\n"
         "    float lt = LAM_ACT(s, i);\n"),
        ("  const float f = c_kp[i] * u - c_kq[i] * s.q[DQADR(i)] - c_kv[i] * s.v[i], fmax = fminf(c_afrc[i], c_frclim[i]);\n",
         "  const float f = c_kp[i] * u - c_kq[i] * s.q[DQADR(i)] - c_kv[i] * s.v[i];\n"
         "  float msl_;\n"
         "  const float fmax = fminf(fminf(c_afrc[i], c_frclim[i]), motor_lim(i, f, s.v[i], &msl_));  // (nablaengine.motor)\n"),
    )

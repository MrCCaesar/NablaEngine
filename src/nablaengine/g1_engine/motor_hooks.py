"""Text edits of the G1-class engine for actuator torque-speed limits (nablaengine.motor; init(motor_limits=...))."""

import numpy as np

from nablaengine.g1_engine import step


def _lit(x):
    return f"{float(np.float32(x))!r}f".replace("inf", "INFINITY")


DEVICE = r"""
// the torque-speed limit of dof i (nablaengine.motor; g1_engine/motor_hooks.py): L(|v|) = min(peak, max(0, c - k |v|))
// of the force's quadrant (driving: f v > 0, else braking); *sl = -df/dv of sign(f) L on a sloped piece
// (0 < c - k |v| < peak), else 0
__device__ __forceinline__ float motor_lim(int i, float f, float v, float* sl) {
  const float* mc = c_motor + 6 * i + (f * v > 0.f ? 0 : 3);
  const float lin = RFMS(mc[2], fabsf(v), mc[1]);
  *sl = lin > 0.f && lin < mc[0] ? RMUL(RMUL(mc[2], copysignf(1.f, f)), copysignf(1.f, v)) : 0.f;
  return fminf(mc[0], fmaxf(lin, 0.f));
}
"""


def check(motor):
    """Refuse the combinations motor limits do not support (ValueError)."""
    if motor is not None and step.clamp_implicit()[0] == 2:
        raise ValueError("motor limits with DK_CLAMP_IMPLICIT=2: not supported (its fading weights use the "
                         "jnt_actfrcrange bound); use 0 or 1")


def hooks(m, motor):
    """The (anchor, replacement) edits for motor (a nablaengine.motor.MotorLimits, or None: ())."""
    if motor is None:
        return ()
    check(motor)
    t = motor.table(m).ravel()
    lit = ", ".join(_lit(x) for x in t)
    rfms = "#define RFMS(a, b, c) __fmaf_rn(-(a), (b), (c))  // c - a b, one rounding\n"
    clamp = "        clamp = act <= -c_frclim[i] || act >= c_frclim[i];  // at the limit: MuJoCo Warp's actuator_lam test\n"
    return (
        (rfms, rfms + f"__device__ const float c_motor[NV * 6] = {{{lit}}};  // nablaengine.motor\n" + DEVICE),
        ("    bool clamp = false;\n",
         "    float msl = 0.f;  // -df/dqdot of the torque-speed limited force (nablaengine.motor), the record's mslope\n"
         "    bool clamp = false;\n"),
        (clamp, clamp +
         "        {  // the torque-speed limit (nablaengine.motor): |f| <= L(|qdot|)\n"
         "          float sl;\n"
         "          const float lim = motor_lim(i, act, s.qvel[i], &sl);\n"
         "          if (fabsf(act) >= lim) { act = copysignf(lim, act); clamp = true; msl = sl; }\n"
         "        }\n"),
        ("#if DK_CLAMP_IMPLICIT\n      s.kvi[i] = kvg;\n",
         "      if (rec) __stcs(rec + OFF_MSLOPE + i, msl);\n#if DK_CLAMP_IMPLICIT\n      s.kvi[i] = kvg;\n"),
        ("    tape_put(rec + OFF_QADV, s.tmp, NV, OFF_LH - OFF_QADV, lane);\n",
         "    tape_put(rec + OFF_QADV, s.tmp, NV, NV, lane);\n"
         "    for (int i = OFF_MSLOPE + NV + lane; i < OFF_LH; i += 32) __stcs(rec + i, 0.f);\n"),
        ("__device__ void adj_compose(Adj& s, int lane) {\n",
         "__device__ void adj_compose(Adj& s, int lane, const float* rec) {\n"),
        ("  adj_compose(s, lane);  // A8\n", "  adj_compose(s, lane, rec);  // A8\n"),
        ("    s.ubar[i - 6] += kl;\n",
         "    s.Pv[i] += __ldcs(rec + OFF_MSLOPE + i) * s.lam[i];  // the torque-speed limit's slope (nablaengine.motor)\n"
         "    s.ubar[i - 6] += kl;\n"),
        ("  const float f = c_kp[i] * (s.u[i - 6] - s.q[i + 1]) - c_kv[i] * s.v[i], fmax = c_frclim[i];\n",
         "  const float f = c_kp[i] * (s.u[i - 6] - s.q[i + 1]) - c_kv[i] * s.v[i];\n"
         "  float msl_;\n"
         "  const float fmax = fminf(c_frclim[i], motor_lim(i, f, s.v[i], &msl_));  // (nablaengine.motor)\n"),
        ("      tau = fminf(fmaxf(tau, -c_frclim[i]), c_frclim[i]);\n",
         "      tau = fminf(fmaxf(tau, -c_frclim[i]), c_frclim[i]);\n"
         "      { float sl; const float lim = motor_lim(i, tau, s.v[i], &sl); if (fabsf(tau) >= lim) tau = copysignf(lim, tau); }\n"),
    )

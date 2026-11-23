"""Actuator torque-speed limits (opt-in): init(motor_limits=...) of nablaengine.jax_ffi / nablaengine.torch_ffi."""

import hashlib
import re
from dataclasses import dataclass

import mujoco
import numpy as np

NO_LIMIT = 1e30


@dataclass(frozen=True)
class MotorCurve:
    """One motor's torque limit against |qdot| per quadrant: L(|qdot|) = min(peak, max(0, intercept - slope |qdot|)),
    drive_* where the force and qdot have the same sign, brake_* otherwise (N m, N m, N m s / rad)."""
    drive_peak: float
    drive_intercept: float
    drive_slope: float
    brake_peak: float
    brake_intercept: float
    brake_slope: float

    @classmethod
    def unitree(cls, y1, y2, x1, x2):
        """Unitree's T-N curve: peak y1 (driving) / y2 (braking) up to the knee speed x1, then linearly to 0 at the
        no-load speed x2 (unitree_rl_lab UnitreeActuator's Y1, Y2, X1, X2)."""
        if not (0 <= x1 < x2 and y1 > 0 and y2 > 0):
            raise ValueError(f"MotorCurve.unitree: needs 0 <= x1 < x2 and y1, y2 > 0 (got {y1}, {y2}, {x1}, {x2})")
        k1, k2 = y1 / (x2 - x1), y2 / (x2 - x1)
        return cls(float(y1), k1 * x2, k1, float(y2), k2 * x2, k2)

    @classmethod
    def dc(cls, saturation_effort, velocity_limit, effort_limit):
        """Isaac Lab's DCMotor: driving min(effort_limit, max(0, tau_sat (1 - |qdot| / qdot_max))), braking
        min(effort_limit, tau_sat (1 + |qdot| / qdot_max))."""
        if not (saturation_effort > 0 and velocity_limit > 0 and effort_limit > 0):
            raise ValueError("MotorCurve.dc: saturation_effort, velocity_limit and effort_limit must be > 0")
        k = saturation_effort / velocity_limit
        return cls(float(effort_limit), float(saturation_effort), k, float(effort_limit), float(saturation_effort), -k)

    def scaled(self, factor):
        """The curve with every torque times factor (e.g. two motors on one joint)."""
        f = float(factor)
        return MotorCurve(self.drive_peak * f, self.drive_intercept * f, self.drive_slope * f,
                          self.brake_peak * f, self.brake_intercept * f, self.brake_slope * f)

    def row(self):
        return (self.drive_peak, self.drive_intercept, self.drive_slope,
                self.brake_peak, self.brake_intercept, self.brake_slope)


@dataclass(frozen=True)
class MotorLimits:
    """Torque-speed curves by joint: curves = ((joint-name regular expression, MotorCurve), ...); each actuated hinge
    takes the curve of the first expression that matches its name in full. require_all: every actuated joint must
    match one (otherwise table() raises); a joint without a curve is limited by jnt_actfrcrange only."""
    curves: tuple
    name: str = "custom"
    require_all: bool = True

    def table(self, m):
        """(nv, 6) float32: each dof's (drive peak, intercept, slope, brake peak, intercept, slope); the unactuated dofs
        (the G1's root) and the dofs without a curve NO_LIMIT, NO_LIMIT, 0 (twice)."""
        out = np.tile(np.array([NO_LIMIT, NO_LIMIT, 0.0] * 2, np.float64), (m.nv, 1))
        pats = [(re.compile(p), c) for p, c in self.curves]
        missing = []
        for a in range(m.nu):
            if int(m.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            j = int(m.actuator_trnid[a, 0])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint {j}"
            hit = next((c for p, c in pats if p.fullmatch(name)), None)
            if hit is None:
                missing.append(name)
                continue
            out[int(m.jnt_dofadr[j])] = hit.row()
        if missing and (self.require_all or len(missing) == m.nu):
            raise ValueError(f"motor limits {self.name!r}: no curve for the actuated joints {missing}")
        t = out.astype(np.float32)
        if not np.isfinite(t).all() or (t[:, [0, 1, 3, 4]] <= 0).any():
            raise ValueError(f"motor limits {self.name!r}: peaks and intercepts must be finite and > 0")
        return t

    def key(self, m):
        """A short identifier of the per-dof table (the build's info["motor_limits"])."""
        return f"{self.name}:{hashlib.sha256(self.table(m).tobytes()).hexdigest()[:12]}"


_N5020 = MotorCurve.unitree(24.8, 31.9, 30.86, 40.13)
_N7520_14 = MotorCurve.unitree(71.0, 83.3, 22.63, 35.52)
_N7520_22 = MotorCurve.unitree(111.0, 131.0, 14.5, 22.7)
_W4010 = MotorCurve.unitree(4.8, 8.6, 15.3, 24.76)

UNITREE_G1 = MotorLimits(curves=(
    (r".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint", _N7520_14),
    (r".*_hip_roll_joint|.*_knee_joint", _N7520_22),
    (r".*_elbow_joint|.*_shoulder_(pitch|roll|yaw)_joint|.*_wrist_roll_joint", _N5020),
    (r".*_wrist_pitch_joint|.*_wrist_yaw_joint", _W4010),
    (r"waist_pitch_joint|waist_roll_joint|.*_ankle_pitch_joint|.*_ankle_roll_joint", _N5020.scaled(2.0)),
), name="unitree_g1")

_DC = {k: MotorCurve.dc(e, v, e) for k, (v, e) in
       {"5020": (37.0, 25.0), "7520_14": (32.0, 88.0), "7520_22": (20.0, 139.0), "4010": (22.0, 5.0)}.items()}
MJLAB_G1_DC = MotorLimits(curves=(
    (r".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint", _DC["7520_14"]),
    (r".*_hip_roll_joint|.*_knee_joint", _DC["7520_22"]),
    (r".*_elbow_joint|.*_shoulder_(pitch|roll|yaw)_joint|.*_wrist_roll_joint", _DC["5020"]),
    (r".*_wrist_pitch_joint|.*_wrist_yaw_joint", _DC["4010"]),
    (r"waist_pitch_joint|waist_roll_joint|.*_ankle_pitch_joint|.*_ankle_roll_joint", _DC["5020"].scaled(2.0)),
), name="mjlab_g1_dc")

PRESETS = {"unitree_g1": UNITREE_G1, "mjlab_g1_dc": MJLAB_G1_DC}


def as_motor(x):
    """None / False -> None; a preset name ("unitree_g1"); a MotorLimits; a {joint regex: MotorCurve} dict."""
    if x is None or x is False:
        return None
    if isinstance(x, MotorLimits):
        return x
    if isinstance(x, str):
        if x not in PRESETS:
            raise ValueError(f"motor_limits={x!r}: one of {sorted(PRESETS)}, a MotorLimits or a dict")
        return PRESETS[x]
    if isinstance(x, dict):
        return MotorLimits(curves=tuple(x.items()))
    raise TypeError(f"motor_limits: None, a preset name, a MotorLimits or a dict, not {type(x).__name__}")


def limit(table, f, qdot):
    """Reference (numpy, float64): the limited force sign(f) min(|f|, L(|qdot|)) and -df/dqdot on the curves' sloped
    pieces (0 elsewhere) for per-dof forces f already clamped at jnt_actfrcrange; table: MotorLimits.table(m)."""
    t = np.asarray(table, np.float64)
    f, qdot = np.asarray(f, np.float64), np.asarray(qdot, np.float64)
    drive = f * qdot > 0
    pk, c, k = (np.where(drive, t[:, i], t[:, 3 + i]) for i in range(3))
    lin = c - k * np.abs(qdot)
    lim = np.minimum(pk, np.maximum(lin, 0.0))
    on = np.abs(f) >= lim
    out = np.where(on, np.sign(f) * lim, f)
    slope = np.where(on & (lin > 0) & (lin < pk), np.sign(f) * np.sign(qdot) * k, 0.0)
    return out, slope

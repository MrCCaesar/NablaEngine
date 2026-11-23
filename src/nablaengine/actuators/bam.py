"""BAM, Rhoban's servo model (Better Actuator Models, model M6), for the XL330 servo: the engine-independent part."""

import copy
import warnings
import json
import math
from pathlib import Path

import numpy as np

PARAMS_DIR = Path(__file__).resolve().parent / "bam_params"
M6_KEYS = ("kt", "R", "armature", "friction_base", "friction_stribeck", "load_friction_motor",
           "load_friction_external", "load_friction_motor_stribeck", "load_friction_external_stribeck",
           "load_friction_motor_quad", "load_friction_external_quad", "dtheta_stribeck", "alpha", "friction_viscous")
ERROR_GAIN = {"xl330": (4096 / (2 * math.pi)) / (256 * 885)}
FIRMWARE_MAX_CURRENT = {"xl330": 1.75}
FRICTION_SOLREF = (0.01, 1.0)
FRICTION_SOLIMP = (0.99, 0.9999, 0.001, 0.5, 2.0)


def load_params(params="xl330/m6"):
    """The M6 parameter dict of a BAM parameter set: "xl330/m6" (bundled) or a path to a BAM json (an XL330 M6 set:
    "actuator": "xl330", "model": "m6"). ValueError for another servo or model."""
    p = Path(params)
    path = PARAMS_DIR / f"{params}.json" if not p.suffix and (PARAMS_DIR / f"{params}.json").exists() else p
    if not path.exists():
        raise ValueError(f"BAM parameters {params!r}: not a bundled set ('xl330/m6') or an existing json file")
    d = json.loads(path.read_text())
    if d.get("model") != "m6" or d.get("actuator") != "xl330":
        raise ValueError(f"BAM parameters {params!r}: an XL330 M6 set is supported (actuator 'xl330', model 'm6'), "
                         f"not actuator {d.get('actuator')!r} model {d.get('model')!r}")
    miss = [k for k in M6_KEYS if k not in d]
    if miss:
        raise ValueError(f"BAM parameters {params!r}: missing {miss}")
    out = {k: float(d[k]) for k in M6_KEYS}
    out["actuator"] = d["actuator"]
    return out


def max_current(bam, p=None):
    """The firmware current limit of bam (a nablaengine.actuators.Bam) in A, None when switched off."""
    mc = bam.max_current
    if mc is None:
        return None
    if mc == "firmware":
        p = p or load_params(bam.params)
        return FIRMWARE_MAX_CURRENT[p["actuator"]]
    mc = float(mc)
    if not (mc > 0 and math.isfinite(mc)):
        raise ValueError(f"Bam(max_current={bam.max_current!r}): a positive current in A, 'firmware' or None")
    return mc


def servo(bam, p=None):
    """The engine model's servo values of bam at its nominal voltage: kp, frclim (the force limit at duty 1), damping
    (KD + friction_viscous), kd (KD = kt^2 / R), armature, frictionloss (friction_base), ilim (kt I_max, the current
    limit's torque; None when off), vin_nom, vin_min."""
    p = p or load_params(bam.params)
    vn = float(bam.vin_nom)
    imax = max_current(bam, p)
    return dict(kp=p["kt"] * vn * ERROR_GAIN[p["actuator"]] * float(bam.kp_fw) / p["R"], frclim=p["kt"] * vn / p["R"],
                damping=p["kt"] ** 2 / p["R"] + p["friction_viscous"], kd=p["kt"] ** 2 / p["R"],
                armature=p["armature"], frictionloss=p["friction_base"],
                ilim=None if imax is None else p["kt"] * imax, vin_nom=vn, vin_min=float(bam.vin_min))


def act_words(nv):
    """Floats of a world's BAM state row: tau (nv), ext (nv), vin0, drop."""
    return 2 * int(nv) + 2


def layout(nv):
    """{name: (offset, width)} of a world's BAM state row."""
    nv = int(nv)
    return {"tau": (0, nv), "ext": (nv, nv), "vin0": (2 * nv, 1), "drop": (2 * nv + 1, 1)}


def initial_state(nv, batch, vin0=7.4, drop=0.0, tau=None, ext=None):
    """A (batch, act_words(nv)) float32 NumPy array of BAM state rows: zero torques (or tau / ext, (batch, nv) or
    (nv,)), the supply voltage vin0 (V) and the drop gain drop (V per N m of summed |tau|), scalars or (batch,)."""
    nv, B = int(nv), int(batch)
    a = np.zeros((B, act_words(nv)), np.float32)
    if tau is not None:
        a[:, :nv] = tau
    if ext is not None:
        a[:, nv:2 * nv] = ext
    a[:, 2 * nv] = vin0
    a[:, 2 * nv + 1] = drop
    return a


def _servo_ids(m):
    import mujoco
    act = np.arange(m.nu)
    if m.nu and (m.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT).any():
        raise ValueError("BAM: every actuator must drive a joint (joint transmission)")
    jnt = m.actuator_trnid[:, 0].copy()
    return act, jnt, m.jnt_dofadr[jnt].copy(), m.jnt_qposadr[jnt].copy()


def check_model(m, bam):
    """Raise ValueError unless the MjModel m fits BAM's engine form (prepare's output): one position servo per hinge
    after a free root (in dof order) without kv, the servo values of bam (kp, the joints' force limit, damping,
    armature), a positive frictionloss on every servo dof, the friction rows' impedance, no joint stiffness, the
    pyramidal cone, no condim-4 pairs."""
    import mujoco
    sv, bad = servo(bam), []
    if m.njnt < 1 or m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or m.nu != m.nv - 6:
        bad.append(f"a free root and one servo per hinge (nu = {m.nu}, nv - 6 = {m.nv - 6})")
    else:
        _, jnt, dof, _ = _servo_ids(m)
        if not (dof == np.arange(6, m.nv)).all() or (m.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_HINGE).any():
            bad.append("the actuators must drive the hinges in dof order")
        else:
            close = lambda a, b: np.allclose(a, b, rtol=1e-6, atol=0)
            if not close(m.actuator_gainprm[:, 0], sv["kp"]) or not close(-m.actuator_biasprm[:, 1], sv["kp"]):
                bad.append(f"actuator kp is not BAM's {sv['kp']:.6g} (gainprm[0] = -biasprm[1])")
            if (m.actuator_biasprm[:, 2] != 0).any():
                bad.append("actuator kv != 0: the back-EMF belongs to dof_damping")
            if (m.actuator_forcelimited != 0).any():
                bad.append("actuator forcerange: BAM's limit is the joints' jnt_actfrcrange")
            if not (m.jnt_actfrclimited[jnt] != 0).all() or not close(m.jnt_actfrcrange[jnt, 1], sv["frclim"]) \
                    or not close(-m.jnt_actfrcrange[jnt, 0], sv["frclim"]):
                bad.append(f"jnt_actfrcrange is not +-{sv['frclim']:.6g} (kt vin_nom / R)")
            if not close(m.dof_damping[6:], sv["damping"]):
                bad.append(f"dof_damping[6:] is not KD + friction_viscous = {sv['damping']:.6g}")
            if not close(m.dof_armature[6:], sv["armature"]):
                bad.append(f"dof_armature[6:] is not the servo's {sv['armature']:.6g}")
            if not (m.dof_frictionloss[6:] > 0).all():
                bad.append("dof_frictionloss[6:] has zeros: every servo dof needs its friction row")
    if (m.jnt_stiffness != 0).any():
        bad.append("joint stiffness")
    if m.nv and (not np.allclose(m.dof_solref, FRICTION_SOLREF) or not np.allclose(m.dof_solimp, FRICTION_SOLIMP)):
        bad.append(f"the friction rows' impedance (dof_solref {FRICTION_SOLREF}, dof_solimp {FRICTION_SOLIMP})")
    if m.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
        bad.append("the elliptic friction cone (not with BAM yet)")
    if m.npair and (m.pair_dim > 3).any():
        bad.append("condim-4 contact pairs (not with BAM yet)")
    if bad:
        raise ValueError("BAM: the model does not fit the servo model's engine form (actuators.bam.prepare makes one "
                         "that does): " + "; ".join(bad))


def prepare(m, bam):
    """A copy of the MjModel m with every actuator made an XL330 under bam (a nablaengine.actuators.Bam) in the
    engine's form (the module docstring): position servos kp without kv or actuator force limit, the joints' force
    limit +-frclim, BAM's damping, armature, frictionloss and friction-row impedance; mj_setConst then recomputes the
    derived constants (the inverse weights). Idempotent. ValueError for a model whose actuators are not one per hinge
    after a free root."""
    import mujoco
    sv = servo(bam)
    m = copy.deepcopy(m)
    if m.njnt < 1 or m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or m.nu != m.nv - 6:
        raise ValueError(f"BAM: a free root and one actuator per hinge are needed (nu = {m.nu}, nv - 6 = {m.nv - 6})")
    act, jnt, dof, _ = _servo_ids(m)
    if not (dof == np.arange(6, m.nv)).all() or (m.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_HINGE).any():
        raise ValueError("BAM: the actuators must drive the hinges in dof order")
    g, bp = m.actuator_gainprm[act, 0], m.actuator_biasprm[act]
    other = (m.actuator_biastype[act] != mujoco.mjtBias.mjBIAS_AFFINE) | (bp[:, 1] != -g) | (g <= 0)
    if other.any():
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a)) or str(int(a)) for a in act[other]]
        warnings.warn(f"BAM replaces {len(names)} actuator(s) that are not position servos (e.g. velocity servos or "
                      f"motors) with XL330 position servos: {', '.join(names[:8])}" + (" ..." if len(names) > 8 else ""),
                      RuntimeWarning, stacklevel=2)
    m.actuator_gaintype[act] = mujoco.mjtGain.mjGAIN_FIXED
    m.actuator_biastype[act] = mujoco.mjtBias.mjBIAS_AFFINE
    m.actuator_gainprm[act] = 0.0
    m.actuator_biasprm[act] = 0.0
    m.actuator_gainprm[act, 0] = sv["kp"]
    m.actuator_biasprm[act, 1] = -sv["kp"]
    m.actuator_dyntype[act] = mujoco.mjtDyn.mjDYN_NONE
    m.actuator_gear[act] = 0.0
    m.actuator_gear[act, 0] = 1.0
    m.actuator_forcelimited[act] = 0
    m.jnt_actfrclimited[jnt] = 1
    m.jnt_actfrcrange[jnt] = (-sv["frclim"], sv["frclim"])
    m.dof_damping[dof] = sv["damping"]
    m.dof_armature[dof] = sv["armature"]
    m.dof_frictionloss[dof] = sv["frictionloss"]
    m.dof_solref[:] = FRICTION_SOLREF
    m.dof_solimp[:] = FRICTION_SOLIMP
    mujoco.mj_setConst(m, mujoco.MjData(m))
    return m



def budget(tm, te, v, p):
    """The M6 friction budget from the previous solve's motor torque tm, external torque te and the velocity v
    (bam.mjlab.BamActuator._compute_friction_budget)."""
    sb = np.exp(-np.abs(v / p["dtheta_stribeck"]) ** p["alpha"])
    am, ae = np.abs(tm), np.abs(te)
    return (p["friction_base"] + sb * p["friction_stribeck"]
            + np.abs(te * p["load_friction_external"] - tm * p["load_friction_motor"])
            + sb * np.abs(te * p["load_friction_external_stribeck"] - tm * p["load_friction_motor_stribeck"])
            + sb * np.where(am > ae, p["load_friction_external_quad"] * ae ** 2, p["load_friction_motor_quad"] * am ** 2))


def force_window(sc, v, sv):
    """(lo, hi) of the servo force after the current limit and the PWM limit, as one window (clamp(clamp(x, a, b),
    c, d) = clamp(x, lo, hi)): the current window KD v +- ilim, then the voltage window +-sc frclim (applied last)."""
    c = sc * sv["frclim"]
    if sv["ilim"] is None:
        return -c * np.ones_like(v), c * np.ones_like(v)
    a, b = sv["kd"] * v - sv["ilim"], sv["kd"] * v + sv["ilim"]
    lo, hi = np.maximum(a, -c), np.minimum(b, c)
    lo = np.where(a > c, c, np.where(b < -c, -c, lo))
    hi = np.where(a > c, c, np.where(b < -c, -c, hi))
    return lo, hi


class Reference:
    """The engine-form BAM servo model on MuJoCo C in float64, one world: the prepared model (prepare), every substep
    the voltage scale on kp, the current and voltage windows in jnt_actfrcrange, the friction budget in
    dof_frictionloss (from the previous solve's tau, ext), then mj_step; tau, ext from the solve. step(qpos, qvel,
    warm, ctrl, act, substeps) runs one control step from a state and returns (qpos', qvel', warm', act')."""

    def __init__(self, m, bam):
        import mujoco
        self.m = prepare(m, bam)
        check_model(self.m, bam)
        self.d = mujoco.MjData(self.m)
        self.p, self.sv = load_params(bam.params), servo(bam)
        self.act, self.jnt, self.dof, self.qadr = _servo_ids(self.m)
        self.kp = self.m.actuator_gainprm[self.act, 0].copy()

    def substep(self, ctrl, tau, ext, vin0, drop):
        import mujoco
        m, d, sv = self.m, self.d, self.sv
        v = d.qvel[self.dof].copy()
        sc = max(vin0 - drop * np.abs(tau).sum(), sv["vin_min"]) / sv["vin_nom"]
        m.actuator_gainprm[self.act, 0] = self.kp * sc
        m.actuator_biasprm[self.act, 1] = -self.kp * sc
        lo, hi = force_window(sc, v, sv)
        m.jnt_actfrcrange[self.jnt, 0], m.jnt_actfrcrange[self.jnt, 1] = lo, hi
        m.dof_frictionloss[self.dof] = budget(tau, ext, v, self.p)
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)
        fr = np.zeros(m.nv)
        for i in range(d.nefc):
            if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF:
                fr[d.efc_id[i]] += d.efc_force[i]
        ext = (-d.qfrc_bias + d.qfrc_constraint - fr)[self.dof]
        tau = d.qfrc_actuator[self.dof] - sv["kd"] * v
        return tau, ext

    def step(self, qpos, qvel, warm, ctrl, act, substeps):
        """One control step of `substeps` substeps from (qpos, qvel, warm (qacc_warmstart), ctrl, act (a state row,
        act_words(nv))): (qpos', qvel', warm', act') in float64."""
        import mujoco
        m, d, nv = self.m, self.d, self.m.nv
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.qacc_warmstart[:] = qpos, qvel, warm
        act = np.asarray(act, np.float64)
        tau, ext = act[6:nv].copy(), act[nv + 6:2 * nv].copy()
        vin0, drop = float(act[2 * nv]), float(act[2 * nv + 1])
        for _ in range(int(substeps)):
            tau, ext = self.substep(ctrl, tau, ext, vin0, drop)
        out = np.zeros(act_words(nv))
        out[6:nv], out[nv + 6:2 * nv], out[2 * nv], out[2 * nv + 1] = tau, ext, vin0, drop
        return d.qpos.copy(), d.qvel.copy(), d.qacc_warmstart.copy(), out



def _f(x):
    return repr(float(np.float32(x))) + "f"


def device_header(bam):
    """The #defines and the friction budget's device functions of a BAM build (the M6 constants of bam's parameter
    set, its nominal and minimum voltage, the current limit)."""
    p, sv = load_params(bam.params), servo(bam)
    ilim = sv["ilim"]
    return f"""#define DK_BAM 1  // the BAM servo model in the kernels (nablaengine.actuators.bam)
#define BAM_STRIDE (2 * NV + 2)  // a world's BAM state row: tau[NV], ext[NV], vin0, drop
#define BAM_VIN_NOM {_f(sv["vin_nom"])}
#define BAM_VIN_MIN {_f(sv["vin_min"])}
#define BAM_KD {_f(sv["kd"])}  // back-EMF damping kt^2 / R
#define BAM_CURRENT_LIMIT {0 if ilim is None else 1}  // the firmware current limit (window KD v +- BAM_ILIM)
#define BAM_ILIM {_f(0.0 if ilim is None else ilim)}  // kt I_max
#define BAM_F_BASE {_f(p["friction_base"])}
#define BAM_F_STRIB {_f(p["friction_stribeck"])}
#define BAM_LF_MOT {_f(p["load_friction_motor"])}
#define BAM_LF_EXT {_f(p["load_friction_external"])}
#define BAM_LFS_MOT {_f(p["load_friction_motor_stribeck"])}
#define BAM_LFS_EXT {_f(p["load_friction_external_stribeck"])}
#define BAM_Q_MOT {_f(p["load_friction_motor_quad"])}
#define BAM_Q_EXT {_f(p["load_friction_external_quad"])}
#define BAM_RDTHETA {_f(1.0 / p["dtheta_stribeck"])}
#define BAM_ALPHA {_f(p["alpha"])}
#define BAM_SCALE(x) (bsc * (x))
#define BAM_FLOSS(s, i) (s).bfl[i]
// the M6 friction budget of a servo dof from the previous solve's motor torque tm and external torque te and the
// dof's velocity v (bam.mjlab.BamActuator._compute_friction_budget)
__device__ __forceinline__ float bam_budget(float tm, float te, float v) {{
  const float sb = expf(-powf(fabsf(v) * BAM_RDTHETA, BAM_ALPHA));
  const float am = fabsf(tm), ae = fabsf(te);
  return BAM_F_BASE + sb * BAM_F_STRIB + fabsf(te * BAM_LF_EXT - tm * BAM_LF_MOT)
         + sb * fabsf(te * BAM_LFS_EXT - tm * BAM_LFS_MOT) + sb * (am > ae ? BAM_Q_EXT * ae * ae : BAM_Q_MOT * am * am);
}}
"""


def tape_words(nv):
    """Words a BAM build adds to a substep's tape record: the scale, tau[NV], ext[NV], the servo forces[NV], the
    current-limit flags[NV], to a multiple of 32."""
    return -(-(1 + 4 * int(nv)) // 32) * 32

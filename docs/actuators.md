# Actuator models

The model's MJCF actuators define the actuator force (MuJoCo's semantics). `init(actuator=...)` adds an actuator model
on top of it (`nablaengine.actuators`):

- `None` (default): the model's actuators as MuJoCo defines them;
- torque-speed limits: a preset (`"unitree_g1"`, `"mjlab_g1_dc"`), a `motor.MotorLimits`, or
  `actuators.Actuators((joint_regex, actuators.TorqueSpeed(curve)), ...)`. Details in [api.md](api.md), "Actuator
  models"; `motor_limits=` is a permanent alias of `actuator=`;
- `"bam"` or `actuators.Bam(...)`: the BAM servo model, below.

`info["actuator"]` is the option's key. Torque-speed limits and BAM cannot be combined in one build.

## The BAM servo model

**BAM** is Rhoban's *Better Actuator Models* (model M6) for the Dynamixel **XL330** servo: the firmware's proportional
position controller, a DC motor with back-EMF, the firmware current limit and a load-dependent friction model. It is
stateful: the motor and external torques of the previous solve and the supply voltage are a per-world actuator state
that goes in and out of every step, differentiably.

```python
from nablaengine import jax_ffi
from nablaengine.actuators import Bam, bam

info = jax_ffi.init(model=m, foot_pairs=fp, com_body="trunk_base", actuator="bam")   # or actuator=Bam(...)
act = bam.initial_state(info["nv"], B, vin0=7.4, drop=0.1)                            # (B, info["act_words"]) float32
qpos, qvel, warm, act = jax_ffi.step(qpos, qvel, warm, ctrl, friction, com, act=act)  # act' is the next step's act
```

`jax.grad` through a rollout of `step` calls (or a `jax.lax.scan` of `make_step`) carries the state's cotangent from
each step to the one before it.

Per physics substep and servo dof (the form of mjlab's `bam.mjlab.BamActuator`):

| | |
|---|---|
| supply | `vin = max(vin0 - drop · Σ_i |τ_i|, vin_min)`, `sc = vin / vin_nom` |
| servo | `act = sc · kp · (u - q)` |
| current limit | `act = clamp(act, KD·v - kt·I_max, KD·v + kt·I_max)` |
| PWM limit | `act = clamp(act, -sc·frclim, sc·frclim)` (applied last) |
| friction | the friction row's bound: Coulomb, Stribeck, load-dependent and quadratic terms |
| motor torque | `τ' = act - KD·v` (`KD = kt² / R`, the back-EMF, part of `dof_damping`) |

**State.** A world's row holds `act_words = 2·nv + 2` floats (`info["act_layout"]`): `τ (nv)`, `e (nv)`, `vin0`,
`drop`. Only the servo dofs are used. `vin0` (V) and `drop` (V per N·m of summed |τ|) pass through unchanged, so a
world's supply can be randomized by setting them in the row (mjlab's ranges: `vin0` 6.5 to 8.2 V, `drop` 0 to 0.2).
They get gradients through `g_act`.

**Options.** `Bam(params="xl330/m6", kp_fw=200, vin_nom=7.4, vin_min=6.0, max_current="firmware")`:

- `params`: the bundled XL330 M6 parameter set (Rhoban/bam's `params/xl330/m6.json`, Apache-2.0), or a path to
  another XL330 M6 json in BAM's format;
- `kp_fw`: the firmware's proportional gain;
- `vin_nom`, `vin_min`: the nominal voltage and the supply's floor (V);
- `max_current`: `"firmware"` (1.75 A, the XL330's default current limit), a current in A, or `None` (off).

`actuator="bam"`, `{"kind": "bam", ...}` and `Actuators((r".*", Bam(...)),)` are equivalent. The M6 parameters (kt,
R, friction constants) are compile-time constants and get no gradient.

**Model preparation.** `init(actuator="bam")` prepares a copy of the model (`bam.prepare(m, bam)`, then
`bam.check_model`): position servos without kv, gear 1 and no forcerange; `jnt_actfrcrange = ±frclim`; the servo's
damping, armature and a friction row per servo dof with BAM's stiff impedance. The model needs a free root and one
actuator per hinge, in dof order. Pass the prepared model to MuJoCo when you compare against it.

## Interfaces

- **JAX.** `forward(..., act=)` returns `(qpos', qvel', warm', act', info[, contacts][, tape])`; `step(..., act=)` and
  `make_step(...)(qpos, qvel, warm, ctrl, friction, com, act)` return `(qpos', qvel', warm', act'[, contacts])`;
  `backward(tape, friction, com, g_qpos, g_qvel, act=, g_act=)` returns `(g_qpos, g_qvel, g_ctrl, g_act, info)`.
  `grad_decay` scales the state's cotangents (qpos, qvel, act). `clamp_kv` 0 and 1 give the same gradients.
- **PyTorch.** `torch_ffi.forward`, `step` and `backward` take the same `act=` / `g_act=` and return the same tuples;
  `step` is a `torch.autograd.Function` when an input requires grad. Results are bitwise those of JAX.
- **`EnginePhysics(..., actuator="bam")`**: `.step(qpos, qvel, warm, ctrl, friction, com, act)` and
  `.initial_act(B, vin0, drop)`. For a robot whose COM body is not `torso_link`, call
  `jax_ffi.init(model=, foot_pairs=, com_body=, actuator="bam")` first, then `EnginePhysics(nworld=B)`.

In a BAM build `act` is required; elsewhere it is refused.

## Limits

- Single-tree robots (a free root, one position servo per hinge) on either engine. On the forest engine: no deferred
  pairs (meshes, `DK_DEFER_BOX`) and at most 64 dofs.
- Not with BAM (`ValueError` at `init`): kinematics outputs, per-world parameters (`params=`), terrain builds,
  `contact_forces=True`, `DK_CLAMP_IMPLICIT=2`, the elliptic cone, condim-4 pairs, joint stiffness, `clamp_kv=2`,
  noslip.
- XL330 servos only; no gradients for the M6 parameters.

Measured (a G1-sized tree with 29 XL330 servos, free flight, 50 solver iterations): qvel against the float64 reference
on MuJoCo C, median relative error 9.5e-7.

## Checks

- `scripts/bam_check.py`: the forward against `bam.Reference` (the same servo model on MuJoCo C in float64);
  `--backend torch`, `--max-current none`.
- `scripts/bam_grad_check.py`: the adjoint against float64 central differences of the reference, including
  backpropagation through several control steps; same options.
- `scripts/motor_check.py`: torque-speed limits.

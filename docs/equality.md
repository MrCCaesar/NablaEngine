# Equality constraints, tendons and muscles (forest engine)

The forest engine simulates and differentiates:

- joint equalities (`mjEQ_JOINT`): a gripper's mimic finger, coupled joints, a joint held at a value;
- connect equalities (`mjEQ_CONNECT`): closed loops, four-bar linkages;
- fixed tendons: tendon equalities (`mjEQ_TENDON`), limits, stiffness, damping, armature, tendon actuators (a
  gripper's "split" tendon);
- spatial tendons (experimental): site paths, pulleys, sphere and cylinder wrapping with side sites;
- muscles and actuators with activation dynamics (experimental), on joints and tendons.

There is nothing to switch on: load the model and call `init(model=m, ...)`. Models without these features build the
same code as before. The G1 engine supports none of them (capability code H25).

Not supported: weld and flex equalities, tendon frictionloss, armature on spatial tendons, per-world toggling of
`eq_active`, per-world equality or tendon parameters.

## Converting a model

Pass MuJoCo models through `nablaengine.envs.convert.to_engine` first. It keeps each constraint where the engine's
structure differs from the source:

- a nonzero joint `ref` is folded into the body pose, and the polycoef of joint and tendon equalities is rewritten so
  the constraint stays the same;
- a body connect becomes a site connect when one of its bodies is fused, split or moved;
- where fusing changes a connect's inverse weight, the source's value is kept in the custom numeric
  `nablaengine_eq_invweight` (`invweight="converted"` leaves it out);
- tendons, sites and wrap geoms are carried to the bodies they land on, spring lengths are made explicit;
- muscle and general actuators keep their gear, parameters and compiled lengthrange.

The converter asserts on random states that the converted model's equality positions, tendon lengths, Jacobians,
limit rows and passive forces are the source's.

## Joint equalities

One constraint row per active equality between joint1 and joint2: pos = q1 - P(q2), with P the `polycoef` polynomial
(degree up to 4; without joint2, pos = q1 - c0). The row uses the equality's own `solref` / `solimp` and is never
clamped, as in MuJoCo Warp 3.17.

- Hinge or slide joints; solref in the standard or the direct form (a mixed solref raises T13).
- Equalities with `eq_active0 = 0` get no rows.
- The dof rows (frictionloss, limits) and the equality rows must fit the 64-row reserve (above 32 dofs: the row
  capacity). A model that does not fit raises `ValueError`; equality rows are never dropped.

Menagerie scenes this makes reachable include `agilex_piper`, `arx_l5`, `i2rt_yam`, `trossen_vx300s`,
`trossen_wx250s` and the `franka_emika_panda` MJX variant.

Measured (gripper scenes holding a box, one substep): robot qvel against MuJoCo Warp, median relative error below
4e-7; gradients against float64 MuJoCo C, median 2e-7 to 1.3e-6.

## Connect equalities

Three rows (x, y, z) per active connect between two bodies or two sites; body 2 may be the world. The forward follows
MuJoCo C 3.17, including the J-dot term from the current velocities; the adjoint differentiates the rows exactly.

- Body or site objects, active at build (an inactive connect raises H25e).
- Standard or direct solref.
- At most 32 dofs on the two chains together.
- A connect between two trees couples them in the Newton solve only.

Measured (four-bar linkages, a free base, two coupled trees, Robotiq 2F-85): gradients against float64 MuJoCo C over
16 substeps, median relative error 4e-7 to 7e-5.

## Fixed tendons

A fixed tendon has the length L = sum_i c_i q_i over hinge / slide joints.

- **Tendon equality**: one row, pos = (L1 - L1_0) - P(L2 - L2_0), never clamped.
- **Tendon limit**: a one-sided row per side where dist < margin, with `tendon_solref_lim` / `solimp_lim`. Both sides
  can be active at once, as in MuJoCo C.
- **Passive force**: stiffness with the `tendon_lengthspring` deadband, and damping (implicit under implicitfast,
  explicit under Euler).
- **Armature and implicit damping** are added on M's sparsity pattern, as MuJoCo C does.
- **Actuators**: fixed gain with an affine or no bias, gear folded by the converter, symmetric forcerange.

Limits (H25t, T5): at most 16 limited tendons; margin >= 0; an actuator on a fixed tendon needs a support dof without a
joint actuator, gear 1 after conversion, a symmetric forcerange, no tendon actfrcrange and no actuatorfrcrange on a
joint the tendon spans. The `DK_CLAMP_IMPLICIT=2` / `DK_ADJ_CLAMP_KV=2` switches act on joint actuators only.
Torque-speed limits (`nablaengine.motor`) act on joint actuators and compose with tendon actuators.

Measured (tendon scenes without contacts): gradients against float64 MuJoCo C over 16 substeps, median relative error
2e-7 to 6e-6.

## Spatial tendons (experimental)

A spatial tendon's path is its wrap elements in order: straight segments between sites, sphere and cylinder wraps
(a side site selects the side; a side site inside the geom takes MuJoCo's inside wrap), and pulleys that scale what
follows by 1 / divisor. Limits, stiffness, damping and actuators act as on fixed tendons, with the path's length and
Jacobian. Gradients include the Jacobian's dependence on the pose.

Limits (H25, T5):

- a Jacobian support of at most 32 dofs;
- every body on the path has exactly one joint (true after `to_engine`);
- no armature on a spatial tendon;
- the per-world tendon storage (lengths, Jacobians, two words per tendon actuator, activations) is at most 6144 words;
- any number of tendon actuators (no carrier dof needed);
- not with `DK_FOREST_LATENCY`.

### Packed storage (experimental)

Where the strided storage (one row of the largest support per tendon) would exceed
6144 words, the build switches to a packed layout (each tendon's Jacobian over its own support). Results are bitwise
the same in both layouts. MS-Human-700 with all 700 muscles needs 16800 words strided and 5598 packed.

Measured (unit scenes of every wrap kind): qvel against float64 MuJoCo C, median relative error below 4e-7; gradients
median 2e-7 to 3e-6.

## Muscles and activation dynamics (experimental)

MuJoCo's muscle actuators and general actuators with activation dynamics, on hinge / slide joints and on fixed and
spatial tendons. The activation is a per-world state that goes in and out of the step:

```python
info = jax_ffi.init(model=m, foot_pairs=[])                       # info["act_words"] == m.na
act = jnp.zeros((B, info["act_words"]), jnp.float32)              # MuJoCo's act layout (d.act)
qpos, qvel, warm, act = jax_ffi.step(qpos, qvel, warm, ctrl, friction, com, act=act)
```

`step` / `make_step` are differentiable in `act` (`jax.grad` through a rollout carries the activation's cotangent
across control steps); `backward(..., act=, g_act=)` returns `(g_qpos, g_qvel, g_ctrl, g_act, info)`.

Supported (T12): dyntype none / integrator / filter / filterexact / muscle; gaintype fixed / affine / muscle; biastype
none / affine / muscle; gear on its first component; any forcerange; actlimited, actearly, tausmooth.

Limits:

- refused (T12): user and dcmotor types, other transmissions, implicitfast with a velocity-dependent gain (the
  muscle's force-velocity curve; use Euler, as MyoSuite models do), actuatorfrcrange on a joint or tendon in the
  actuator's path;
- the activations count toward the 6144-word tendon storage;
- `clamp_kv` 0 and 1 only; no kinematics outputs; PyTorch builds the same library but has not been run with muscles.

Measured (MyoSuite models and MS-Human-700 with their muscles): the activation is bitwise MuJoCo Warp's; qvel against
float64 MuJoCo C, median relative error 3e-7 to 4e-5.

## Checks

- `scripts/eq/eq_check.py`: joint equalities, forward and gradients.
- `scripts/eq/connect_check.py`, `scripts/eq/connect_grad.py`: connect equalities and tendon models, forward, rows and
  gradients.
- `scripts/eq/tendon_check.py`: fixed tendons, forward and rows.
- `scripts/forest/spatial_check.py`: spatial tendons.
- `scripts/forest/muscle_check.py`: muscles and activation dynamics.

Test models are in `scripts/eq/models` and `scripts/forest/models`.

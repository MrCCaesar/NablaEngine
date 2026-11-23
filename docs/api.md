# API reference

NablaEngine runs B worlds of a MuJoCo model on the GPU, one control step per call, with an analytic adjoint. It has a
JAX interface (`nablaengine.jax_ffi`, XLA FFI custom calls with `jax.custom_vjp`) and a PyTorch interface
(`nablaengine.torch_ffi`, `torch.autograd`) over the same CUDA library. This page uses the JAX names; the PyTorch
functions have the same names, arguments and results (see "PyTorch" below).

## Overview

- **Two engines, one interface.** The code is generated from the model by one of two engines: the **forest engine**
  (the default for every model: several kinematic trees, hinge or slide roots, motors, Euler; up to 64 dofs, or 128 as an experimental feature) and the
  **G1 engine** (single free-tree legged robots, opt-in with `engines.prefer_tree(m)` or `DK_ENGINE=tree`). Callers use
  the same functions and options with either; `info["engine"]` (`"g1"` or `"forest"`) says which one was built.
  Details: `docs/engines.md`.
- **Builds.** The library is generated for the model and compiled by nvcc on first use (about 15 s per GPU
  architecture), then loaded from a cache (`$DK_CACHE_DIR`, default `~/.cache/nablaengine`) keyed by the sources, the
  model, the options, the nvcc version and XLA's FFI headers.
- **Precision.** Everything is float32.

## `init`

```python
info = jax_ffi.init(model_dir=None, model=None, foot_pairs=None, terrain=None, com_body=None, params=None,
                    max_contacts=None, max_rows=None, motor_limits=None, contact_forces=False, actuator=None,
                    contact_surrogate=False, foot_wrench=None, contact_frame=False, contact_grad=False,
                    noslip_pairs=None, foot_normal=False, capacity_diag=False)
```

Loads (or builds) the library for a model and registers the FFI targets. Idempotent, and called implicitly by every
traced call. One library per process.

- `model` / `foot_pairs`: an `MjModel` and the indices of the contact pairs that take the per-world `friction` input.
  Without `model`, the bundled fixtures (`model_dir`, default `$DK_FIXTURES`) are used.
- `com_body`: the body that takes the per-world COM offset `com` (default `torso_link` on a G1-class model; the forest
  engine has no default for other models).
- The other arguments are build options described in the sections below (`terrain`: `docs/terrain.md`).

The returned dict holds the library path and cache key, `arch`, `arch_flags`, `engine`, `com_body`, `com_layout`
(`{name: (offset, width)}` of the `com` row), `com_words`, `con_words` (words per contact), `maxcon`, `nefc` and
`built`. Log it with your results.

## Shapes

For the bundled mjlab Unitree G1: `nq = 36`, `nv = 35`, `nu = 29`. B is read from the arrays at run time: one library
serves every batch size.

| argument | shape | meaning |
|---|---|---|
| `qpos` | (B, nq) | joint positions (free joint: position, unit quaternion w x y z) |
| `qvel` | (B, nv) | joint velocities |
| `warm` | (B, nv) | Newton warm start (`qacc_warmstart`); the step returns the next one |
| `ctrl` | (B, nu) | actuator controls, held over the control step |
| `friction` | (B,) | friction of the `foot_pairs` contacts (pass the model's value where it is not randomized) |
| `com` | (B, `info["com_words"]`) | COM offset of `com_body` (3 words), then any per-world parameters |

A control step is `substeps` substeps of the model's timestep (4 x 5 ms for the G1).

## `step` and `make_step`

```python
qpos1, qvel1, warm1 = jax_ffi.step(qpos, qvel, warm, ctrl, friction, com, *, iterations=10, substeps=4,
                                   clamp_kv=0, normal=3, clamp_soft=None, grad_decay=1.0, kinematics=False,
                                   contacts=False, act=None, cone_mu=0.0, impulse_clip=None)
step_fn = jax_ffi.make_step(iterations=10, substeps=4, clamp_kv=0, normal=3, clamp_soft=None, grad_decay=1.0,
                            kinematics=False, contacts=False, cone_mu=0.0, impulse_clip=None)
```

One differentiable control step. Options are static Python values; `make_step` returns the cached `custom_vjp`
function for one combination, so jit caches stay valid. Without differentiation the step runs without a tape. Under
`jax.vjp` / `jax.grad` the forward writes a tape (`tape_bytes(nworld)` bytes) and the backward replays the forward's
own contacts, rows and solver state; the result is deterministic. The VJP returns cotangents for `qpos` (in qpos
form), `qvel` and `ctrl`, and zeros for `warm`, `friction` and `com` (unless per-world parameters are built in).

```python
outs, pull = jax.vjp(lambda q, v, u: jax_ffi.step(q, v, warm, u, friction, com)[:2], qpos, qvel, ctrl)
g_qpos, g_qvel, g_ctrl = pull((cot_qpos, cot_qvel))
```

`iterations` is the number of Newton iterations per substep; `substeps` the substeps per control step. The gradient
options are below; they change only the backward, so one forward and tape serve every gradient variant.

## `forward` and `backward`

```python
qpos1, qvel1, warm1, info, tape = jax_ffi.forward(qpos, qvel, warm, ctrl, friction, com, *, iterations=10,
                                                  substeps=4, tape=True, kinematics=False, contacts=False, act=None)
g_qpos, g_qvel, g_ctrl, binfo = jax_ffi.backward(tape, friction, com, g_qpos1, g_qvel1, *, substeps=4, clamp_kv=0,
                                                 normal=3, ctrl=None, clamp_soft=None, kinematics=None, act=None,
                                                 g_act=None, cone_mu=0.0, impulse_clip=None, g_contacts=None)
```

The same computation without `custom_vjp` (traceable). `info` is (B, 4) int32 per world: Newton iterations of the
last substep, contacts, constraint rows, unsupported/overflow bits. `binfo` (B,) int32 holds the adjoint's flags,
or-ed with `NONFINITE_* << NONFINITE_SHIFT` where a gradient or incoming cotangent is NaN / inf. `clamp_kv=2` needs
the step's `ctrl`. `backward` never applies `grad_decay`.

## Gradient options

| option | default | meaning |
|---|---|---|
| `clamp_kv` | 0 | dofs whose actuator force is clamped at its limit: 0 keeps the damping term -kv (as MuJoCo Warp's adjoint), 1 drops it (exact), 2 soft clamp of width `clamp_soft` |
| `normal` | 3 | contact normal of sphere / capsule self contacts: 3 frozen below 1e-3 m (finite gradients), 0 exact, 1 and 2 other approximations |
| `clamp_soft` | None | width for `clamp_kv=2` (None: 0.2); must stay None for `clamp_kv` 0 and 1 |
| `grad_decay` | 1.0 | multiplies the returned qpos / qvel cotangents (1.0: no change) |

`clamp_kv=1, normal=0` is the exact derivative of the forward ("exact mode"). The defaults differ from it only on
dofs at their force limit and near-singular self contacts. Measured: on a robot whose actuators often saturate
(MuJoCo Playground's H1), median error against finite differences is 1-4% in the default mode and about 1e-6 in exact
mode.

### `scale_grad`

```python
state = jax_ffi.scale_grad(state, decay)
```

Identity on a pytree whose cotangent is multiplied by `decay` (a static float). On the whole carried state at the top
of a `lax.scan` body, every derivative that crosses n steps is scaled by exactly `decay ** n`. On only the physics
call's `qpos`, `qvel` it equals `step(..., grad_decay=decay)`.

### Remat

```python
body = jax.checkpoint(body, policy=jax_ffi.remat_policy())   # = save_only_these_names(*REMAT_NAMES, *extra)
```

The tape and outputs carry checkpoint names (`TAPE_NAME = "dk_tape"`, `OUT_NAME = "dk_out"`, `KIN_NAME = "dk_kin"`,
all in `REMAT_NAMES`). With `remat_policy()` a rematerialized rollout still runs one forward per control step; a plain
`jax.checkpoint(body)` reruns the forward for the tape. `remat_policy(*extra_names)` adds your own names.

## Contact-gradient surrogates (forest engine; experimental)

```python
jax_ffi.init(model=m, foot_pairs=[], contact_surrogate=True)
step_fn = jax_ffi.make_step(..., cone_mu=0.1, impulse_clip=0.02)   # also step / backward / EnginePhysics
```

Two backward-only surrogates for contact-rich scenes (an object pushed, lifted, struck). The forward, tape and
outputs are unchanged. `cone_mu=0`, `impulse_clip=None` (the defaults) give the exact adjoint bitwise.

| option | default | meaning |
|---|---|---|
| `cone_mu` | 0.0 | central-path relaxation of the pyramidal friction cone, barrier `mu` in (m/s^2)^2: inactive rows and stick/slip switches get smooth sensitivities; transition half-width about 2 sqrt(mu) m/s^2, useful range ~1e-3 to 1; mu -> 0 gives the exact gradient |
| `impulse_clip` | None | per contact and substep, where the impulse h\|F\| (N s) exceeds the value, that contact's explicit partials are scaled by `impulse_clip / I` |

They compose with per-world parameters (`init(params=..., contact_surrogate=True)`). Limits: pyramidal cones only
(elliptic raises `ValueError`), forest engine only; joint-limit, equality and tendon rows stay exact. A second `init`
with another `contact_surrogate` value raises `RuntimeError`.

## Kinematics outputs

```python
qpos_n, xpos, xquat, cvel, subtree_com = jax_ffi.kinematics(qpos, qvel, com)
*state, qpos_n, xpos, xquat, cvel, subtree_com = jax_ffi.step(..., kinematics=True)
```

MJX's `kinematics` + `com_pos` + `com_vel`, with VJP: `qpos_n` (B, nq) with normalized free-joint quaternions,
`xpos` (B, nbody, 3), `xquat` (B, nbody, 4) (world body included), `cvel` (B, nbody, 6) twists [angular; linear],
`subtree_com` (B, nbody, 3). `step(..., kinematics=True)` computes them for the output state inside the step;
`kinematics()` is for other states and gives the same bits on a step's output. `kinematics_forward` /
`kinematics_backward` are the versions without `custom_vjp`. On forest models the step library's kernels do not cover
(several trees, slides, free bodies other than body 1) these calls go through `dyn_query.kinematics` automatically.

## Contact forces

```python
info = jax_ffi.init(contact_forces="substeps", foot_wrench=True, contact_frame=True)   # any subset; either engine
q1, v1, w1, con = jax_ffi.step(qpos, qvel, warm, ctrl, friction, com, contacts=True)
c = jax_ffi.contact_table(con)       # dict of arrays; torch_ffi.contact_table gives the same keys
```

| `init` option | values | effect |
|---|---|---|
| `contact_forces` | `False`, `True` = `"last"`, `"substeps"` | contact table with forces (12 words per contact) for the last substep, or one per substep |
| `foot_wrench` | `None`, `True` (the G1's ankle-roll links), or 1-8 body names / ids | per substep and foot: summed contact force and moment; per step: largest total normal force |
| `contact_frame` | `False`, `True` | 3 more words per contact: the frame's first tangent `t1` |

The forces are MuJoCo's `mj_contactForce` at the end of each substep's solve, in world coordinates. The options change
no state, tape or gradient (bitwise). The forest engine's default build already holds the last substep's 12-word
table. `EnginePhysics(..., contacts=True)` makes `.step` return the output too.

Per-contact words: `pair` (contact pair id, -1 past the world's contacts), `dist` (signed distance; on a pair with a
margin (experimental, forest engine) dist - margin), `pos`, `normal` (from the pair's first geom to its second),
`force` (on the second geom), `force_normal`, `tangent1` (with `contact_frame`), `element` (terrain builds).
`contact_table(con, substeps=None)` decodes them into arrays `(B, maxcon, ...)` or `(B, substeps, maxcon, ...)` with
keys `valid`, `pair`, `dist`, `pos`, `normal`, `force`, `force_normal`; with `contact_frame` also `tangent1`,
`tangent2`, `force_t1`, `force_t2`; with `foot_wrench` also `foot_force`, `foot_moment` (B, substeps, nfoot, 3) and
`foot_fn_max`, `foot_fnet_max`, `foot_fup_max` (B, nfoot). `contact_forces.pair_geoms(m, foot_pairs, terrain)` maps
pair ids to geoms. The tangents follow MuJoCo's frame convention (plane-capsule contacts use the capsule axis; use
`contact_frame=True` to get it). The foot moment is about the foot body's origin.

Limits: forward only on the G1 engine; condim-4 torsional moments in `foot_moment` are not validated. Measured
against float64 MuJoCo C on G1 states: per-contact force median relative error 4.7e-6.

### Gradients of the contact output (`contact_grad`, forest engine)

```python
info = jax_ffi.init(contact_forces="substeps", foot_wrench=True, contact_frame=True, contact_grad=True)
def loss(qpos, qvel, ctrl):
    q1, v1, w1, con = jax_ffi.step(qpos, qvel, warm, ctrl, friction, com, contacts=True)
    c = jax_ffi.contact_table(con)
    return jnp.sum(c["foot_force"][..., 2] ** 2) + jnp.sum(c["force_normal"])
g = jax.grad(loss, (0, 1, 2))(qpos, qvel, ctrl)
# lower level: backward(tape, friction, com, g_qpos, g_qvel, g_contacts=g_con)
```

Makes the contact output of `step` differentiable (PyTorch: `torch_ffi.init(contact_grad=True)`). Differentiated:
`dist`, `pos`, `normal`, `force`, `force_normal`, `tangent1`, `foot_force`, `foot_moment` (and `foot_fnet_max` /
`foot_fup_max`, derived from `foot_force`). Not differentiated: `pair`, `element`, `foot_fn_max`. The derivative is
that of the active set at the state: the output jumps where the contact set or a row's zone changes. Without a contact
cotangent results are bitwise those of the build without it.

Limits: pyramidal cone, condim 1 and 3; not with `contact_surrogate` or BAM. Measured (G1, exact mode, float64
MuJoCo C central differences): gradient of last-substep forces, median relative error 1.1e-4 (qpos), 2.6e-5 (qvel),
1.3e-5 (ctrl).

## Diagnostics and capacities

The library keeps exact integer counters per GPU, added to by every call (also inside `jit` and `lax.scan`):
`fwd_calls`, `fwd_worlds`, `fwd_unsupported_worlds`, `fwd_iteration_limit_worlds`, `tape_records`, `refact_records`,
`nonconv_records`, `singular_records`, `bwd_calls`, `bwd_rows`, `nonfinite_rows`, `nonfinite_qpos_rows`,
`nonfinite_qvel_rows`, `nonfinite_ctrl_rows`, `nonfinite_cotangent_rows`, `replay_mismatch_worlds`,
`fwd_contact_overflow_worlds`, `fwd_row_overflow_worlds`.

| call | where | returns |
|---|---|---|
| `read_diagnostics(reset=True)` | outside jit (host sync) | `{name: int}` since the last reset |
| `diagnostics(*deps, reset=True)` | inside jit, after `deps` | (18, 2) uint32; `decode_diagnostics(arr)` -> `{name: int}` |
| `last_forward(nworld=None)` | outside jit | last forward's `worlds`, `contacts`, `contacts_max`, `rows_max` |
| `launch_info(device=0)` | outside jit | adjoint kernel launch per variant: shared memory, worlds per SM |

**Capacity.** By default a G1-class world holds 64 contacts and 192 constraint rows (a frictional contact takes 4
rows, a joint limit 1); a multi-tree forest model gets 4 rows per contact plus 64 (320 at 64 contacts).
`init(max_contacts=, max_rows=)` raises them: up to 64 contacts / 1023 rows for the G1 class, 255 / 1023 on the
forest engine. Rows usually run out first, so raise `max_rows` first. Worlds that fit the default capacity give
bitwise the same results with a larger one; measured cost on the RTX PRO 6000 (G1, `max_rows=1023`): 2% in the
forward.

A world beyond its capacity drops the extra contacts / rows, so its step is not the model's. It is flagged in
`forward()`'s `info[:, 3]` (`OVERFLOW_CONTACTS = 4`, `OVERFLOW_ROWS = 16`) and counted in
`fwd_contact_overflow_worlds` / `fwd_row_overflow_worlds`. `set_overflow_warnings(True)` (or `DK_OVERFLOW_WARN=1`)
makes the diagnostics readers warn when these are non-zero. Mask or reset flagged worlds.

### Capacity diagnostics (`capacity_diag`, forest engine; experimental)

```python
info = jax_ffi.init(..., capacity_diag=True)              # forward()'s info becomes (B, 6)
contacts_found, rows_needed = jax_ffi.capacity_counts(info)   # (B,) int32 each; torch_ffi.capacity_counts likewise
```

Per world, the largest over the call's substeps of the contacts the collision found and the rows the assembly needed,
including those beyond the capacity. Use it to size `max_contacts` / `max_rows` for a new scene. State, tape and
gradients are bitwise unchanged; no measurable cost. Not counted: contacts lost when an internal candidate buffer
fills, and contacts removed by foot offsets.

## `EnginePhysics`

```python
phys = jax_ffi.EnginePhysics(mjm, nworld, foot_pairs, torso_id, *, iterations=None, substeps=4, clamp_kv=0,
                             normal=3, clamp_soft=None, grad_decay=1.0, kinematics=False, use_model=False,
                             params=None, max_contacts=None, max_rows=None, motor_limits=None,
                             contact_forces=False, actuator=None, cone_mu=0.0, impulse_clip=None,
                             foot_wrench=None, contact_frame=False, contacts=False)
qpos1, qvel1, warm1 = phys.step(qpos, qvel, warm, ctrl, friction, com)
phys.stats()          # nacon, naconmax, nefc_max, njmax of the last forward with this batch size
phys.diagnostics()    # read_diagnostics()
```

A batched physics object with MuJoCo Warp's capacity names (`nconmax`, `njmax`: 64 and 192 by default). By default
the constants come from the fixtures and `mjm` must be the same model; with `use_model=True` they come from `mjm`, on
either engine (`torso_id`: the COM-offset body's id, -1 for none). Other keyword options of a MuJoCo Warp based physics
object are accepted and ignored.

## Per-world physics parameters

Domain randomization and system identification beyond `friction` and `com`. Choose the fields at build time with
`params` (`nablaengine.params.PhysicsParams`, or a tuple of names). Each field gets its own gradient.

| field | width | meaning |
|---|---|---|
| `foot_solref` | 2 | solref of the `foot_pairs` (standard or direct form, each entry read as in MuJoCo) |
| `foot_solimp` | 4 | solimp (dmin, dmax, width, midpoint) of the `foot_pairs` |
| `dof_damping` | nv | joint damping per dof |
| `dof_armature` | nv | armature per dof |
| `dof_frictionloss` | nv | forest only: joint dry friction per dof (experimental) |
| `frictionloss_solref` | 2 | forest only: solref of every frictionloss row, standard form (experimental) |
| `frictionloss_solimp` | 4 | forest only: solimp of every frictionloss row (experimental) |
| `torso_mass` | 1 | mass of `com_body` |
| `torso_inertia` | 3 | principal moments of `com_body` (axes stay the model's) |
| `foot_torsional` | 1 | torsional friction of condim-4 `foot_pairs` (pyramidal cone) |
| `object_size` | 3 | forest only: half-extents of one box geom |
| `gravity` | 3 | forest only: gravity vector (world frame, m/s^2) |
| `xfrc` | 9 per body | forest only: force, torque (world frame) and force point (body frame) on each `xfrc_bodies` body |
| `body_inertial` | 7 per body | forest only: mass, principal moments, COM offset of each `inertial_bodies` body |
| `sole_thickness` | 1 | forest only: added thickness of the sole geoms |
| `foot_offset` | 4 | forest only: support offset under each foot (experimental) |

```python
phys = jax_ffi.EnginePhysics(mjm, nworld, foot_pairs, torso_id,
                             params=("foot_solref", "foot_solimp", "dof_damping", "dof_armature"))
rand = phys.pack_rand(foot_solref=sr, foot_solimp=si, dof_damping=d, dof_armature=a)  # G1: (B, 2 + 4 + 35 + 35)
qpos1, qvel1, warm1 = phys.step(qpos, qvel, warm, ctrl, friction, com, rand)
```

With the functional calls, pass `init(..., params=P)` and append `P.pack(m.nv, **values)` to the COM offset in `com`
(`info["com_layout"]` gives the columns). `P.defaults(m, foot_pairs, com_body)` returns the model's own values. The
gradients arrive as the cotangent of `rand` or of `com`'s parameter columns; `backward` returns `g_params` and
`g_friction` as fifth and sixth outputs. With every world at the model's values the results are bitwise those of the
build without `params`. On the forest engine the foot fields apply to whatever pairs `init(foot_pairs=)` names, and the
torso fields need an explicit `init(com_body=)` (any moving body). Measured (G1, float64 MuJoCo C central
differences): median relative error 1.7e-6 (`dof_damping`) to 1.8e-5 (`foot_solimp`).

Notes per field:

- **Mass and inertia fields** act as MuJoCo with `body_mass` / `body_inertia` changed without `mj_setConst`: the
  constraints' `invweight0` stays the model's unless `invweight0="per_world"` (below).
- **`object_size`**: `PhysicsParams(("object_size", ...), size_geom="box_collision", object_size_max=(0.24, 0.23,
  0.26))`. `size_geom` defaults to `com_body`'s single box collision geom; `object_size_max` (required) bounds every
  world (`pack` raises above it). The size is an input with no gradient. Not with a terrain the object touches.
- **`gravity`, `xfrc`**: `PhysicsParams(("gravity", "xfrc"), xfrc_bodies=("torso_link", "box"))`. Gravity replaces
  `opt.gravity` (an accelerating support is equivalent to gravity g - a). Each `xfrc` body takes f and t in the world
  frame and the force point p in the body frame, applied each substep as `mj_applyFT`; `defaults` puts p at the COM.
  Exact gradients for all values. Measured VJP median relative error (an internal G1 + box scene): wrenches 4.2e-6,
  gravity 4.6e-5. `scripts/forest/xfrc_gravity_check.py` runs on `scripts/forest/models/g1box.xml`.
- **`body_inertial`**: `PhysicsParams(..., inertial_bodies=(...))`, per body mass, moments and COM offset added to
  `body_ipos`. A body cannot be both an inertial body and the torso fields' body.
- **`sole_thickness`**: `PhysicsParams(..., sole_geoms=(...), sole_normal=(0, 0, -1))` (default geoms: the moving geoms
  of `foot_pairs`). Moves each sole geom by t along its normal; negative values thin the sole. Not with terrain or the
  collision mid-phase.
- **`invweight0="per_world"`** (on `PhysicsParams`, default `"model"`): recomputes each world's constraint inverse
  weights from its own mass matrix at `qpos0`, as `mj_setConst`, including its gradient. A parameter build then matches
  per-model builds of each world's values bitwise. Not with equality constraints, limited tendons, frictionloss,
  per-geom inverse weights or `contact_surrogate`.
- **Frictionloss fields** (experimental): the model decides which dofs have a frictionloss row; give each such dof a
  positive `dof_frictionloss` (and its `dof_solref` / `dof_solimp`). A world's 0 means no friction on that dof. `pack`
  refuses negative values and non-positive solref entries. Not with noslip or `invweight0="per_world"`. Measured
  `g_params` against MuJoCo C central differences: within 1.1e-5 (frictionloss) and 2.6e-4 (impedance) relative.

  ```python
  m.dof_frictionloss[6:18] = 1.0                       # the 12 leg dofs get rows
  P = PhysicsParams(("dof_frictionloss",))
  info = jax_ffi.init(model=m, foot_pairs=fp, params=P)
  F = np.tile(P.defaults(m, fp)["dof_frictionloss"], (B, 1)); F[:, 6:18] = rng.uniform(lo, hi, (B, 12))
  com = jnp.asarray(np.concatenate([com_offset, P.pack(m.nv, dof_frictionloss=F)], 1))
  ```

The G1 engine raises `ValueError` for every forest-only field; route a G1-class model with `engines.prefer_forest(m)`.

### Foot offsets, foot normal force and sand (forest engine; experimental)

```python
P = PhysicsParams(("foot_offset",))   # feet ("left_ankle_roll_link", "right_ankle_roll_link"),
                                      # foot_line=(-0.064, 0.142) (heel, toe; m), foot_axis=(1, 0, 0)
info = jax_ffi.init(model=m, foot_pairs=fp, params=P, foot_normal=True, contact_grad=True)
com = np.concatenate([com_offset, P.pack(m.nv, foot_offset=s)], 1)   # s (B, 4): heel, toe left; heel, toe right
q1, v1, w1, con = jax_ffi.step(qpos, qvel, warm, ctrl, friction, com, contacts=True)
fn = jax_ffi.contact_table(con)["foot_normal"]   # (B, 4) N: left heel, left toe, right heel, right toe
```

- `foot_offset` (>= 0, m): each foot contact uses distance d + s(x), with s linear from heel to toe along the foot.
  The support sinks by s; 0 is the model, bitwise. Gradients flow to the offsets and through the contact point.
- `foot_normal=True` (needs `foot_offset`): splits each foot's normal force into heel and toe; differentiable with
  `contact_grad=True`.
- Sand: `TerrainConfig(terrain_sand=True)` gives each terrain element a sand fraction, so the offset applies only on
  sand (`docs/terrain.md`).

Limits: sphere or capsule feet; not with noslip or `contact_surrogate`; G1 engine refuses. Measured (G1, exact mode,
float64 MuJoCo C central differences): offset gradient median relative error 3.0e-5.

## Actuator models (`actuator=`)

```python
info = jax_ffi.init(..., actuator="unitree_g1")      # also torch_ffi.init / EnginePhysics
info = jax_ffi.init(..., motor_limits="unitree_g1")  # permanent alias; passing both raises ValueError
```

The model defines the actuators (MuJoCo semantics). `actuator=` adds a model on top (`nablaengine.actuators`):

- `None` (default): the model's actuators unchanged.
- **Torque-speed limits**: a preset (`"unitree_g1"`, `"mjlab_g1_dc"`), a `motor.MotorLimits`, a `{joint regex:
  MotorCurve}` dict, or `actuators.Actuators((joint_regex, actuators.TorqueSpeed(curve)), ...)`. Each force is limited
  by |f| <= min(peak, max(0, c - k |qdot|)), with separate driving and braking curves; the model's force range still
  applies. Custom curves: `motor.MotorCurve.unitree(y1, y2, x1, x2)` or `motor.MotorCurve.dc(saturation_effort,
  velocity_limit, effort_limit)`. Both engines. Not with `DK_CLAMP_IMPLICIT=2`. Measured (G1, forward against MuJoCo
  Warp): qvel median relative error 1.4e-6.
- **BAM**: `"bam"` or `actuators.Bam(params="xl330/m6", kp_fw=200, vin_nom=7.4, vin_min=6.0,
  max_current="firmware")`, the Dynamixel XL330 servo model for single-tree robots. `step` / `forward` then take and
  return the state `act` (B, `info["act_words"]`), differentiable (`backward(..., act=, g_act=)`). See
  `docs/actuators.md`.

`info["actuator"]` names the option (`None` without one).

**Muscles and activation dynamics** (forest engine; experimental): a model with MuJoCo muscles or activation states
needs no option; `act` (MuJoCo's `d.act`, `info["act_words"]` = na) is then required in `forward` / `step` /
`make_step` and `backward(..., act=, g_act=)`. See `docs/equality.md`.

## Unsupported models and capability codes

`init`, `check_model`, `EnginePhysics` and `torch_ffi.build` run the capability checker of the model's engine
(`nablaengine.model.capability`) before building. A model outside the scope raises `UnsupportedModel` (a subclass of
`ValueError` and `AssertionError`) listing the failing codes and values; the engine never silently simulates other
physics. `check_model(mjm, foot_pairs, torso_id)` raises `ValueError` when a model differs from the loaded library's.
`DK_ALLOW_UNSUPPORTED` turns named codes into warnings for experiments (`docs/configuration.md`); such results are not
MuJoCo's.

At call time, buffers of the wrong rank or shape raise `InvalidArgument` (a `JaxRuntimeError`); `make_step` raises
`ValueError` for an unknown variant or a non-finite `grad_decay`.

## Other robots

Pass any supported model as `init(model=m, foot_pairs=[...], com_body=...)`, set `friction` to its foot friction
unless randomized, and set `substeps` to its control step.

The G1 class (either engine): a floating base with one hinge per body, 7 <= nv <= 64, at most 63 bodies and 8
children per body; one position servo per hinge, damping, armature, frictionloss, joint limits; explicit contact pairs
among planes, spheres, capsules, boxes, cylinders and convex meshes (with exact adjoints); Newton solver, pyramidal
cone, implicitfast. The forest engine also takes several trees, hinge and slide roots, motors and Euler; see
`docs/engines.md` for its scope, including these experimental features:

- models with 65-128 dofs or more than 64 bodies (experimental);
- contact supports beyond 32 dofs (experimental);
- convex-pair margins (experimental);
- spatial tendons and muscles / activation dynamics (experimental; `docs/equality.md`).

Validated robots: Unitree G1, Go1, Go2, A1, H1; ANYmal C; Spot; Booster T1; ROBOTIS OP3; Berkeley Humanoid; Apptronik
Apollo; PNDbotics Adam Lite; Fourier N1. Measured qvel gradient median relative error against float64 MuJoCo C:
5.7e-7 to 1.0e-5. Exporting a robot and step times: `docs/robots.md`. Terrain: `docs/terrain.md`.

## Per-geom contact inverse weights

When jointless bodies are fused into their parents (`fusestatic`), their contacts take the parent's inverse weight and
become softer or stiffer than in the source model. A model can carry the source's values per geom:

```xml
<custom>
  <numeric name="nablaengine_geom_invweight" size="NGEOM" data="..."/>  <!-- per geom; -1: the body's -->
</custom>
```

Both engines then use each geom's value for every contact quantity, forward and adjoint.
`nablaengine.envs.convert.to_engine(xml, assets, invweight="geom")` writes the field where bodies were fused. MuJoCo
ignores it (the unfused model is its equivalent). Without the field results are unchanged.

## Dynamics queries (`dyn_query`, forest engine)

```python
from nablaengine import dyn_query
info = dyn_query.init(model=m, foot_pairs=[], com_body=None, params=None)
M, bias = dyn_query.dynamics(qpos, qvel, com=None)     # (B, nv, nv), (B, nv)
jacp, jacr = dyn_query.jacobian(qpos, bodies, points)  # (B, P, 3, nv) each; bodies (P,) / (B, P), points (P, 3) / (B, P, 3)
qpos_n, xpos, xquat, cvel, subtree_com = dyn_query.kinematics(qpos, qvel, com)
# PyTorch: dyn_query.torch_init(device, model=m, ...); torch_dynamics; torch_jacobian; torch_kinematics
```

| output | MuJoCo equivalent |
|---|---|
| `M` | `mj_fullM` of `qM` (armature included, zeros between trees) |
| `bias` | `qfrc_bias` (Coriolis, centrifugal, gravity; no passive or applied forces) |
| `jacp`, `jacr` | `mj_jac(point, body)` |
| `kinematics` | `jax_ffi.kinematics`'s outputs, for any forest model (several trees, slides, free bodies) |

`com` is the step's com row; `dof_armature`, `torso_mass`, `torso_inertia` and `gravity` change `M` / `bias`, other
fields do not (`xfrc` is an applied force: use `jacp' f + jacr' t`). For a terrain build with object slots pass
`init(com_layout=step_info["com_layout"])`. Gradients: `jacobian()` and `kinematics()` have analytic VJPs;
`dynamics()` is forward only. Separate library from the step's (no `jax_ffi.init` needed); one per process; no
per-world fixture placements; refuses `body_inertial`. Measured against float64 MuJoCo: `M` <= 1.6e-7, `bias` <=
6.3e-7 (worst world, relative).

## Distance queries (forest engine; experimental)

`nablaengine.dist_query` measures signed distances between chosen geom or body pairs, batched over worlds, with a
VJP with respect to `qpos`. Pairs add no constraint rows to the step.

```python
from nablaengine import dist_query

pairs = [("geom", "floor", "box_g"), ("body", "finger", "box"), (12, 17)]   # by geom name, body name or geom id
info = dist_query.init(model, pairs, max_distance=0.3)                     # also initialises jax_ffi
dist, points, normal, far = dist_query.query(qpos, com)                    # jax.custom_vjp
```

PyTorch: `dist_query.torch_init(model, pairs, max_distance=0.3, device=0)`, `dist_query.torch_query(qpos, com)`.

Outputs (`S` pairs): `dist (B, S)` (negative when overlapping, `max_distance` beyond it), `points (B, S, 2, 3)` (the
closest point on each geom, world frame), `normal (B, S, 3)` (geom 1 to geom 2), `far (B, S)`; `faces=True` adds the
box face ids, `elements=True` the terrain element. On a terrain build `dist_query.TERRAIN` names the terrain as the
second geom of a pair. Gradients flow from `dist` and `points` to `qpos`.

Limits: forest engine only; plane, sphere, capsule, box, convex mesh, cylinder, ellipsoid (not two planes); up to 128
pairs; not with an actuator model, and not with both a terrain and per-world parameters. Check:
`scripts/forest/dist_query_check.py`.

## PyTorch

```python
from nablaengine import torch_ffi
info = torch_ffi.init(device=None, model=m, foot_pairs=fp)   # same build options as jax_ffi.init, plus artifact=
q1, v1, w1 = torch_ffi.step(qpos, qvel, warm, ctrl, friction, com)   # torch.autograd through qpos, qvel, ctrl
```

Inputs are contiguous float32 CUDA tensors on one device; nothing is cast or copied implicitly (wrong dtypes, devices
or shapes raise `TypeError`). `step`, `forward`, `backward`, `kinematics`, `scale_grad`, `contact_table`,
`capacity_counts` and `read_diagnostics` mirror the JAX functions, and outputs and gradients are bitwise those of
`jax_ffi` on the same GPU. Calls run on torch's current stream without host synchronization after the first upload,
so they can be captured in a CUDA graph (call `init(device)` first), and are `torch.library` custom ops under
`torch.compile`. Build an artifact ahead of time with `python -m nablaengine.torch_ffi build [--arch sm_XY ...]`;
loading it needs only torch, NumPy and the NVIDIA driver (`DK_TORCH_ARTIFACT` selects one).

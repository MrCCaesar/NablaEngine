# Engines

NablaEngine generates CUDA code from the MuJoCo model. Two code generators share one interface:

- the **G1 engine** (also called the tree engine, `nablaengine.g1_engine`): one kinematic tree whose root (body 1) has a
  free joint and whose other bodies each have one hinge, driven by one position servo per hinge (legged robots);
- the **forest engine** (`nablaengine.forest_engine`): every supported model, including several trees under the world
  (an arm and free objects, a hand and a cube), hinge and slide roots (the DeepMind Control Suite tasks) and free trees
  with motors, Euler or joint springs (humanoid, ant).

`jax_ffi.init`, `torch_ffi.init` / `torch_ffi.build` and `jax_ffi.EnginePhysics` pick the engine from the model. The
functions, shapes and options are the same for both. `info["engine"]` (`"g1"` or `"forest"`) says which one a library
was built for.

## Which model goes where

The forest engine is the default for every model. A model goes to the G1 engine only when all of these hold:

1. body 1 has exactly one joint, and it is free;
2. every body descends from body 1 (one tree);
3. you asked for it: `engines.prefer_tree(m)`, or `DK_ENGINE=tree`.

```python
from nablaengine import engines

engines.name(m)            # "g1" or "forest", without building anything
engines.prefer_tree(m)     # route this G1-class model to the G1 engine (ValueError for other models)
engines.prefer_forest(m)   # route it to the forest engine (the last call for a model wins)
```

`DK_ENGINE`: `forest` or `auto` / unset (the default: forest), `tree` or `g1` (G1-class models to the G1 engine). A
per-model `prefer_*` call wins over the variable. A G1-class model on the forest engine by default or through
`DK_ENGINE` keeps the G1 engine's COM-offset body (`torso_link`); with `prefer_forest` it has none unless
`init(com_body=...)` names one.

The G1 engine is built around the structure of one free tree and is faster on it. It gets bug fixes only: new features
go to the forest engine.

So the mjlab G1 and the robots of [robots.md](robots.md) (Go1, Go2, A1, ANYmal C, Spot, H1, Apollo, ...) run on the
forest engine unless you ask for the G1 engine; everything else (DMC and Brax tasks, arms, hands, scenes with free
objects) always does.

## What is the same on both engines

- `step` / `forward` / `backward` / `make_step` (JAX), `step` / `forward` / `backward` (PyTorch), with qpos `(B, nq)`,
  qvel and warm `(B, nv)`, ctrl `(B, nu)`, friction `(B,)`, com `(B, info["com_words"])`.
- The adjoint options (`clamp_kv`, `normal`, `clamp_soft`, `grad_decay`), `tape_bytes`, diagnostics and overflow
  counters, `EnginePhysics`, the contact output (`contacts=True`, `contact_table`).
- Per-world physics parameters (`init(params=)`, [api.md](api.md)) and actuator models (`init(actuator=)`,
  [actuators.md](actuators.md)).
- Model validation: an unsupported model raises `nablaengine.model.capability.UnsupportedModel` (a `ValueError`) before
  anything is built, listing the failing checks by code (`capability.check` for the G1 engine, `capability.check_forest`
  for the forest engine). `DK_ALLOW_UNSUPPORTED` lets named codes through with a warning.

Read `info["com_layout"]` rather than assuming 3 com words: each engine appends its own per-world extras (fixtures,
terrain object slots, parameters) after the COM offset.

## What each engine supports

| | G1 engine | forest engine |
|---|---|---|
| Structure | one free tree, one hinge per body, at most 63 bodies, 7 <= nv <= 64 | any number of trees; free roots, hinges, slides; up to 128 bodies and 128 dofs (beyond 64: experimental) |
| Actuators | one position servo per hinge, gear 1, no forcerange | motors, position and velocity servos (affine bias), at most one per dof; tendon actuators; muscles and activation dynamics (experimental) |
| ctrl | not clamped to ctrlrange | clamped to ctrlrange (zero gradient where clamped) |
| Integrator | implicitfast | implicitfast, Euler (with or without eulerdamp) |
| Passive forces | damping, armature, frictionloss (one solref / solimp for all dofs) | also joint springs, gravity compensation, tendon stiffness / damping / armature, per-dof frictionloss impedance |
| Joint limits | hinge limits, one solref / solimp | hinge and slide limits, each joint's own solref / solimp |
| Contact geoms | plane, sphere, capsule, box, cylinder, convex mesh | the same plus ellipsoids; every pair of these types |
| Contact margins | 0 only | MuJoCo's margin and gap (convex pairs: experimental) |
| Friction cone, condim | pyramidal, elliptic; condim 1, 3, 4 | pyramidal, elliptic; condim 1, 3, 4 |
| Equalities, tendons | none | joint, connect and tendon equalities; fixed tendons; spatial tendons (experimental). See [equality.md](equality.md) |
| Noslip | no | yes (below) |
| Solref | standard form | standard and direct forms (a mixed solref is refused, T13) |
| Contacts per world | default 32 or 64, at most 64 | default 32 / 64 / 255 (by pair count), at most 255 |
| Contact table | 8 words; 12 with `init(contact_forces=True)` | always 12 words (with the forces) |
| `com_body` default | `torso_link` | none |
| Kinematics outputs | yes | models with one free tree |
| BAM servo model | yes | yes, without deferred (mesh) pairs, at most 64 dofs |
| Per-world `foot_torsional` / `object_size`, `gravity`, `xfrc` | `foot_torsional` only | all but `foot_torsional` |

Neither engine takes ball joints, more than 128 bodies or dofs, condim 6 (rolling friction), or height fields as contact
pair geoms. For condim 6, `envs.convert.to_engine(condim6_to=3)` converts the pairs to condim 3 explicitly (with a
warning). Height fields go through a terrain build ([terrain.md](terrain.md)), or
`envs.convert.to_engine(drop_types=(mjGEOM_HFIELD,))` drops their pairs where they are decoration. `to_engine` freezes
mocap bodies at their model pose.

The interface defaults for `substeps` / `iterations` (4 / 10) are mjlab's. For other models pass
`iterations=m.opt.iterations` and the model's own control step.

## Speed of the two engines

The same single-tree robots on both engines (RTX PRO 6000, 4096 worlds, 4 substeps, 10 iterations; ms per control
step, forward / forward + backward). Both give the same physics up to float32 rounding (qvel max relative difference
4e-7 to 2.4e-5).

| robot | G1 engine | forest engine |
|---|---|---|
| G1 (mjlab) | 1.28 / 1.89 | 1.44 / 2.13 |
| Go1 | 0.65 / 0.92 | 0.78 / 1.06 |
| Go2 | 0.79 / 1.05 | 0.88 / 1.15 |
| H1 | 0.66 / 0.98 | 0.76 / 1.10 |
| Spot | 0.72 / 0.98 | 0.83 / 1.09 |
| T1 | 1.49 / 2.41 | 1.68 / 3.60 |

On V100 the forest engine is the faster of the two on robots with at most 32 dofs.

## Forest engine features

Nothing has to be switched on for these: the model's fields are what the engine simulates.

### Velocity servos

MuJoCo's `<velocity kv>`, with or without forcerange. Gradients at the forcerange follow `clamp_kv`
(`make_step(clamp_kv=)`): the default 0 keeps the kv term on a clamped actuator (a surrogate), 1 gives the exact
derivative of the clamped force. Use `clamp_kv=1` when the gradient must match the simulated step. Measured with
`clamp_kv=1`: gradients against float64 MuJoCo C, relative error at most 5.7e-5.

### Joint dry friction

`dof_frictionloss` rows with each dof's own `solreffriction` / `solimpfriction`, forward and gradients. Set it in the
model (`m.dof_frictionloss[i]`, `m.dof_solref[i]`, `m.dof_solimp[i]`, or the joint's MJCF attributes). The gradient is
exact for the stick / slip pattern the forward took; it jumps when a row changes between sticking and sliding. Not with
noslip or `invweight0="per_world"`. Measured on the G1 with leg frictionloss: qvel against MuJoCo Warp, median relative
error 1.6e-6.

### Per-world frictionloss (experimental)

`PhysicsParams` fields `dof_frictionloss` (nv), `frictionloss_solref` (2) and `frictionloss_solimp` (4), with gradients
([api.md](api.md), "Per-world physics parameters").

### Standalone free bodies

A free body without jointed children (a manipulated object) gets MuJoCo's implicit gyroscopic term under implicitfast,
forward and gradients. Per-world inertials on such a body are not used in this term. Not enabled on sm_90 by default
(`DK_FOREST_FREE_GYRO`).

### Contact margins

MuJoCo's pair margin and gap: a contact is a constraint while dist < margin. Supported on the analytic pairs (plane
against sphere, capsule, ellipsoid, cylinder or box; sphere-sphere, sphere-capsule, capsule-capsule, sphere-cylinder).
Refused (T10): margins with a terrain build, the mid-phase, noslip or `contact_surrogate`, and two pairs of the same
geoms with different margins.

### Convex-pair margins (experimental)

Margins on sphere-box, capsule-box, box-box, plane-mesh and the other convex pairs. A box-box margin on a G1-class model
is refused.

### Contact-gradient surrogates (experimental)

Backward-only relaxations for contact-rich scenes: `init(contact_surrogate=True)`, then `cone_mu` and `impulse_clip` on
`make_step` / `step` / `backward`. The defaults give the exact adjoint. Pyramidal cone only. See [api.md](api.md).

## Large models (experimental)

The forest engine takes models with up to 128 dofs and 128 bodies (world included) with no switch. Models with 65 to 128
dofs or more than 64 bodies are experimental. What changes:

- beyond 64 dofs, M and the Newton Hessian are factored on their sparsity patterns or by a blocked dense factor
  (`DK_NV128_*`, [configuration.md](configuration.md));
- a model with many resting contacts may need `init(max_contacts=...)`; the constraint rows stop at 1023 (a warning says
  when they do not fit);
- not available beyond 64 bodies: kinematics outputs, `invweight0="per_world"`, noslip.

Measured: exact-mode gradients against float64 MuJoCo C, median relative error 1e-4 to 3e-4 on 65- and 113-dof scenes.

### Contact supports beyond 32 dofs (experimental)

A contact's support (the dofs on one of its two bodies' chains) may have up to 64 dofs (T11). Models with supports of at
most 32 build as before.

### `DK_FOREST_CHOL_BLK=1` (experimental)

For models with 44 <= nv <= 64, factor M and the Newton Hessian with the register-blocked dense factor used beyond 64
dofs. Faster on V100 from 44 dofs (forward 0.85x at 48 dofs, 0.21x at 64); results equal to float rounding.

### `DK_FOREST_LATENCY=1` (experimental)

Two warps per world for small batches (a few worlds: search, evaluation of one robot). Faster at 1 to 64 worlds, slower
from a few hundred. Not for models with mesh pairs. Results are bitwise those of the default build.

### Capacity diagnostics (experimental)

`init(..., capacity_diag=True)`, then `contacts_found, rows_needed = jax_ffi.capacity_counts(info)` gives per world the
contacts and constraint rows a scene needs, to size `max_contacts` / `max_rows`. Results are bitwise those of a build
without it.

## Noslip

A model with `opt.noslip_iterations > 0` gets MuJoCo's noslip post-pass on the forest engine, forward and gradients. The
model options are the API (`noslip_iterations`, `noslip_tolerance`). `init(noslip_pairs=...)` restricts the pass to
chosen pairs (pair indices or names, (body, body) or (geom, geom) name pairs, or `"objects"`: the pairs between a free,
unactuated object and another non-world body).

Limits: pyramidal cone; no equality constraints, tendons, terrain, motor limits, BAM, margins or contact-gradient
surrogates; at most 31 bodies and 64 dofs. The backward costs about twice the plain one. Measured on a standing G1: qvel
against float64 MuJoCo C, median relative error 1.1e-5.

## Terrain on the forest engine

Terrain builds (`init(terrain=)`, `set_terrain`, `raycast`, object slots, `nablaengine.sensors`) run on both engines. On
the forest engine they take single-tree models (route a G1-class model with `engines.prefer_forest(m)` or
`DK_ENGINE=forest`), with fixtures, the mid-phase and the model's own sphere-box / capsule-box pairs. Moving object
slots, slot and terrain materials and sand are forest-only ([terrain.md](terrain.md), [sensors.md](sensors.md)).
Measured against the G1 engine on the same inputs: the same contact sets, qvel median relative difference 1.5e-7.

## Collision mid-phase (experimental)

`DK_MIDPHASE` adds a mid-phase in front of the per-pair broad phase: bounding spheres per body (and per body group), so
that only pairs between nearby bodies are tested. It helps scenes with many pairs per body pair (object clutter, objects
made of many convex pieces), not typical robots.

| `DK_MIDPHASE` | effect |
|---|---|
| `auto` / unset | only models that name explicit groups (a text element `nablaengine_midphase` holding a JSON list of lists of body names) |
| `1` | body-level spheres for every forest model with contact pairs |
| `2` | body and group levels |
| `0` | never |

The contact set is unchanged; mid-phase builds sort contacts by (pair, slot), so results can differ from the default
build by float32 rounding. Refused with contact margins.

## Environment variables

`DK_DEFER_BOX` and `DK_MIDPHASE` act on the forest engine only. `DK_MAXCON` overrides the default contact capacity on
both (the defaults differ). `DK_FIXTURES` sets the G1 engine's default model; the forest engine always needs
`init(model=...)`. All variables: [configuration.md](configuration.md).

## Checks

- `scripts/verify.sh --engine forest`, `engine_step_check.py --engine forest`, `gradient_check.py --engine forest`: the
  forward and gradient checks on either engine.
- `scripts/identity_gate.py`: the G1 engine's default builds are unchanged.
- `scripts/velocity_servo_check.py`: velocity servos against float64 MuJoCo C.
- `scripts/forest/`: the forest engine's feature checks (frictionloss, solref forms, margins, free bodies, large models,
  contact supports).
- `scripts/collision/`: cylinder and ellipsoid pairs, the mid-phase.
- `scripts/forest_terrain_check.py`: terrain builds on the forest engine against the G1 engine.

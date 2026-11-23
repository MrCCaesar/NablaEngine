# Terrain

A terrain build adds static terrain (boxes and heightfields shared by every world) to a single-tree robot: the G1 or
another robot from [robots.md](robots.md). Typical uses are stairs, steps, obstacles and rough heightfields such as
mjlab's sub-terrain grids, with every world on its own sub-terrain.

The terrain is data, not part of the compiled model. It lives in GPU memory, is uploaded once per device, can be
replaced at any time without recompiling, and can be large (thousands of boxes, millions of heightfield samples).

## Usage

```python
import mujoco
from nablaengine import terrain as T, torch_ffi

scene = mujoco.MjModel.from_xml_path("scene_with_terrain.xml")  # box / hfield geoms on the world body
ter = T.Terrain.from_model(scene)            # or Terrain.load("terrain.npz"); Terrain.save() writes one
cfg = T.TerrainConfig(geoms="feet")          # which robot geoms collide with it (compile time)

torch_ffi.init(terrain=cfg)                  # the G1; or init(model=m, foot_pairs=..., terrain=cfg)
torch_ffi.set_terrain(ter)                   # upload; call again to replace the terrain
qpos1, qvel1, warm1 = torch_ffi.step(qpos, qvel, warm, ctrl, friction, com)   # as without terrain
```

`jax_ffi` has the same entries (`init(terrain=cfg)`, `set_terrain`, `raycast`, `grid_pattern`); both interfaces give
bitwise the same steps and gradients.

`Terrain.from_model(m)` reads the box and hfield geoms of the world body and of bodies welded to it, in world
coordinates. `Terrain.add_to_spec(spec)` writes a terrain into an `MjSpec`, e.g. to build a MuJoCo reference scene.

Both engines build terrain. On the forest engine a G1-class model is routed there by default or with
`engines.prefer_forest(m)`; it can also have fixtures, the collision mid-phase and its own sphere-box / capsule-box
pairs ([engines.md](engines.md)). Materials, sand and moving object slots are forest-only.

## Options (`TerrainConfig`)

| option | default | meaning |
|---|---|---|
| `geoms` | `"feet"` | `"feet"`: the geoms of the foot-floor pairs; `"body"`: every robot geom with a floor pair; or a list of geom names / ids, each with a floor pair. Spheres, capsules and boxes (boxes in the analytic mode only) |
| `floor` | `False` | keep the model's floor plane pairs as well |
| `mode` | `"analytic"` | heightfield contacts: `"analytic"` or `"native"` |
| `cap_points` | `3` | heightfield sample points per capsule (analytic mode) |
| `maxcon` | the model's | contact capacity per world; worlds beyond it are flagged and counted |
| `objects` | `0` | per-world kinematic box slots, posed through `com` ([sensors.md](sensors.md)) |
| `object_velocity` | `False` | forest engine: slot velocities (moving platforms, belts) |
| `object_material` | `False` | forest engine: a contact material per slot and world |
| `terrain_material` | `False` | forest engine: a material per terrain element |
| `terrain_sand` | `False` | forest engine: a sand fraction per terrain element (experimental) |
| `sand_other` | `(1.0, 0.0, 0.0, 0.0)` | sand fields for foot contacts off the terrain (floor, object slots) |

Each colliding robot geom gets one terrain contact pair with the parameters of its floor pair; foot pairs keep the
per-world `friction` input. `T.robot_terrain_geoms(m, foot_pairs, cfg.geoms)` lists them.

Constraint rows have their own capacity (`init(max_rows=)`, 192 for the mjlab G1). A capsule foot across the seam of
two boxes touches both, so feet on seams can exceed it; rows beyond it are dropped and counted in
`read_diagnostics()['fwd_row_overflow_worlds']`. Raise `max_rows` for such courses.

The terrain broad phase handles 32 robot geoms per pass. The G1's `"body"` has 33; leaving out one geom that rarely
touches the terrain avoids a second pass:

```python
body = [m.geom(g).name for g, _, _ in T.robot_terrain_geoms(m, foot_pairs, "body")]
cfg = T.TerrainConfig(geoms=[n for n in body if n != "head_collision"])   # 32 geoms: one pass
```

`set_terrain(ter, cell=1.0)` builds a uniform 2-D grid of `cell`-metre cells for the broad phase. Contacts are sorted
by pair, terrain element and contact index, so results depend only on the contact set.

## Contacts

- **Boxes:** MuJoCo's closed-form sphere-box and capsule-box collisions; robot boxes use box-box collision.
- **Heightfields, `analytic`:** fixed sample points (a sphere's centre, `cap_points` points along a capsule) against
  the plane of the triangle below each point. A smooth approximation of MuJoCo's prism contacts.
- **Heightfields, `native`:** MuJoCo's rule: two triangular prisms per cell, each measured exactly against the sphere
  or capsule, at most 4 contacts kept per geom and heightfield. Contact points are differentiated as fixed to the geom.

MuJoCo Warp collides heightfields with its general convex collision, so both modes differ from it on heightfields by
construction. Measured against MuJoCo Warp on stairs and box obstacles: qvel median relative error 8e-7 to 1.2e-6,
contact counts equal in every world.

## Height scan rays

```python
offsets, directions = jax_ffi.grid_pattern(size=(1.6, 1.0), resolution=0.1, direction=(0.0, 0.0, -1.0))
hits = jax_ffi.raycast(pos, rot, offsets, directions, alignment="base", max_distance=10.0)
hits["distances"]   # (B, N), -1 where nothing is hit; also hits["normals_w"], hits["hit_pos_w"] (B, N, 3)
```

`grid_pattern` makes the rays of mjlab's `GridPatternCfg`; `raycast` casts them from a frame (`pos (B, 3)`,
`rot (B, 3, 3)`) with `alignment` `"base"`, `"yaw"` or `"world"`, as mjlab's `RayCastSensor`. Only the terrain is hit, not the robot; no gradient. For depth cameras, LiDAR and differentiable rays
see [sensors.md](sensors.md).

## Terrain element materials (forest engine)

```python
cfg = T.TerrainConfig(geoms="feet", terrain_material=True)
jax_ffi.init(terrain=cfg)
ter = T.Terrain.from_model(m_scene)
ter.material = np.stack([T.terrain_material(1.0, "none")] * (len(ter.boxes) + len(ter.hf_head)))
ter.material[3] = T.terrain_material(0.2, "override")      # box 3: a low-friction patch (ice)
ter.material[5] = T.terrain_material(1.2, "max")           # box 5: rubber, mixed as MuJoCo mixes geom frictions
jax_ffi.set_terrain(ter)                                    # no recompilation
```

`Terrain.material` holds one row per element (the boxes, then the heightfields; `terrain.TER_MAT_WORDS` = 8): `mu`,
`solref` (2), `solimp` (4) and a mode. The terrain is shared, so materials are per element, not per world.

| mode | friction | solref / solimp | MuJoCo equivalent |
|---|---|---|---|
| `"none"` (also elements without a row) | the robot geom's floor pair's, or the per-world `friction` input for foot pairs | the pair's | element friction 0 |
| `"max"` | max(that friction, `mu`) | the pair's | element friction `mu` (MuJoCo's max mixing) |
| `"override"` | `mu` | the element's | the element's geom has the higher `priority` |

With randomized foot friction, an `"override"` patch of 0.2 is slippery in every world, while `"max"` with 0.2 changes
nothing. Materials get no gradient. Without `terrain_material` a mode other than `"none"` is refused by
`set_terrain`. The G1 engine refuses `terrain_material`. Measured: qvel against MuJoCo Warp (override as geom
priority), median relative error 1.1e-6 in all three modes.

## Sand elements (forest engine, experimental)

Sand next to hard ground on one terrain: each element gets a sand fraction k that scales the per-world foot offset
(`foot_offset`, experimental, [api.md](api.md) "Foot offsets") on the foot contacts with that element.

```python
from nablaengine.params import PhysicsParams

P = PhysicsParams(("foot_offset",))
cfg = T.TerrainConfig(geoms="feet", terrain_sand=True)            # needs the foot_offset parameter
info = jax_ffi.init(model=m, foot_pairs=fp, params=P, terrain=cfg, foot_normal=True, contact_grad=True)
n = len(ter.boxes) + len(ter.hf_head)
mat = np.stack([T.terrain_material(1.0, "none")] * n)               # 8-word rows: hard ground (k = 0)
mat = np.concatenate([mat, np.zeros((n, T.TER_SAND_WORDS), np.float32)], 1)   # (n, 12)
mat[3, 8:] = T.sand_fields(k=1.0, s_max=0.025, yield_force=0.65, rate=0.5)    # box 3: sand
ter.material = mat
jax_ffi.set_terrain(ter)
com = np.concatenate([com_offset, P.pack(m.nv, foot_offset=s)], 1)  # s (B, 4): the sinkage per world
q1, v1, w1, con = jax_ffi.step(qpos, qvel, warm, ctrl, friction, com, contacts=True)
tab = jax_ffi.contact_table(con)
tab["foot_normal_split"]    # (B, 2, 4) N: [sand, hard] x (left heel, left toe, right heel, right toe)
tab["foot_sand_presence"]   # (B, 4): > 0 where the foot touches sand
tab["foot_sand_params"]     # (B, 2, 2, 3): per foot, heel / toe: (s_max, F_y, rate) of the ground under it
```

The 12-word row (`terrain.TER_MAT_SAND_WORDS`) is the 8 material words, then k (0 hard, 1 full sand), s_max, the yield
force F_y and the sinkage rate. The engine uses only k: a foot contact on element e takes the distance d + k_e s(x).
s_max, F_y and rate are stored for your sinkage model and returned per foot in `foot_sand_params` (no gradient).
`foot_normal_split` is differentiable with `contact_grad=True`.

Limits: the limits of `foot_offset` and `contact_grad` (pyramidal cone, condim 1 and 3); rows with k > 0 in a build
without `terrain_sand` are refused; the G1 engine refuses it. At a sand / hard boundary the support steps by k s (no
blending). Measured: qvel against MuJoCo Warp (each foot contact with margin -k s), median relative error 9.7e-7.

## Checks

- `scripts/terrain_check.py`: terrain builds against MuJoCo Warp (forward, rays) and float64 MuJoCo C (gradients).
- `scripts/forest_terrain_check.py`: terrain builds on the forest engine against the G1 engine.
- `scripts/forest/contact_vjp_check.py`: slot and terrain materials.
- `scripts/forest/sand_elem_check.py`: sand elements.

# Microduck: the sphere-foot engine model

Pollen Robotics' Microduck (20 dofs, 14 Dynamixel XL330 servos) as an engine model, `fixtures/robots/microduck/`. Its
sole meshes (`sole_left` / `sole_right`, ~7900 vertices, collided by MuJoCo through their convex hulls) are replaced by
4 spheres per foot with explicit floor-sphere pairs, and the scripts here check that the contact behaviour stays that
of the meshes.

Source model: `model/` (git ignores it) is a copy of `pollen-robotics/microduck_rl`'s
`src/mjlab_microduck/robot/microduck/` at commit bae39f5 (the MJCF: Apache-2.0; the meshes under `assets/`:
CC BY-NC-SA 4.0, so the copy stays local):

```bash
cp -r <microduck_rl>/src/mjlab_microduck/robot/microduck model
python fit_feet.py && python build_scenes.py      # python with mujoco 3.10.0 (mjlab 1.3.0's pin), numpy, scipy
python make_fixtures.py <microduck_rl> --check    # python with mujoco 3.17.0: the fixture, compared with the committed one
python compare.py <microduck_rl> <alpha_walking.onnx>   # the dynamic comparison below (mujoco 3.10.0)
```

From a fresh checkout these reproduce the committed `fit_results.json` and fixture bitwise, and `scene_spheres4.xml`
with the sha256 recorded in the fixture's `provenance`. The scene must be written by MuJoCo 3.10.0: 3.17 writes its
inertial frames with a difference in the fifth digit.

All scenes use the Velocity task's mjlab 1.3.0 settings: dt 0.005, implicitfast, Newton, pyramidal, iterations 10,
ls_iterations 20, ccd_iterations 50, feet condim 3 / friction 1.0 / priority 1. Simulator here: MuJoCo C 3.10.0 on
CPU.

## Files

| File | What |
|---|---|
| `fit_feet.py`, `fit_results.json` | fits K = 4 / 6 / 8 spheres per foot to the hull's support function (tilts up to 60 deg); `fit_results.json` is the fit the fixture's spheres come from (sphere centres and radii in the ankle frames, no mesh data) |
| `build_scenes.py` | writes `model/scene_mesh.xml`, `model/scene_spheres{K}.xml` (contype contacts), `model/scene_spheres{K}_pairs.xml` (explicit floor-sphere pairs, no contype contacts) |
| `make_fixtures.py` | `fixtures/robots/microduck/microduck_spheres4.{xml,json}` from `scene_spheres4.xml` (its docstring has the steps) |
| `compare.py` | test A: 200 random single-foot drops; test C: whole robot, PD (kp 2.0) at the policies' default pose, stand / squat + sway / pushes, 3 seeds -> `compare_results.json`, `drops_results.json`. Test B (the official walking policy through `infer_policy.py`) is written but was not run |

## Results

Geometry (lowest point of the foot vs the hull, over foot tilts):

| K | max err <= 10 deg | <= 30 deg | <= 60 deg | radii |
|---|---|---|---|---|
| 4 | 0.42 mm | 0.52 mm | 0.88 mm | 12.5-13.5 mm |
| 6 | 0.24 mm | 0.38 mm | 0.76 mm | 7.2-13.2 mm |
| 8 | 0.18-0.21 mm | 0.18-0.38 mm | 0.68 mm | 7.2-13.2 mm |

In the whole robot at random poses, mesh and spheres reach the same lowest point within 0.2 mm.

Single-foot drops (0.37 kg, from up to 30 deg tilt, 1 s): end position differs from the mesh by
1.6 mm median (3.9-4.2 mm p90), orientation 3.1-3.4 deg median. The mesh foot never comes to rest
(0 / 200 still after 1 s; dropped flat it still rocks at 0.3-1.7 rad/s after 3 s, 1-3 flickering
contacts, 0.74 mm penetration); every sphere foot settles (98-100 %, 4 steady contacts, 0.13 mm).

Whole robot (mean of 3 seeds):

| scene | trunk diff vs mesh at 2 / 5 / 8 s | quiet-stance ang. vel. rms | contacts per foot (stand) | slip p95 stand / sway | us per step (CPU) |
|---|---|---|---|---|---|
| mesh | - | 0.095 rad/s | 1.9 | 3.2 / 9.5 mm/s | 83 |
| spheres 4 | 0.47 / 0.37 / 0.40 mm | 0.0024 rad/s | 4 | 1.0 / 5.2 mm/s | 124 |
| spheres 6 | 0.67 / 0.54 / 0.31 mm | 0.0016 rad/s | 5.7 | 0.7 / 3.8 mm/s | 160 |
| spheres 8 | 0.55 / 0.48 / 0.46 mm | 0.0016 rad/s | 5.5 | 0.8 / 4.2 mm/s | 160 |

The same seed (2) falls under the 1.5 N pushes in every scene. `_pairs` scenes give results identical
to the contype scenes they come from.

## MuJoCo Warp check (CPU device, this Mac)

mujoco-warp 3.8.1 + warp-lang 1.12.0 (mjlab's pins) run on Warp's CPU device (no CUDA on Apple
Silicon). 2 s of PD stance from the same state:

| scene | Warp CPU | MuJoCo C |
|---|---|---|
| spheres 4 pairs | 4 contacts / foot, ang. vel. rms 0.003 rad/s | identical trunk pose (0.00 mm), 4 / foot |
| mesh | 4 contacts / foot, ang. vel. rms 0.004 rad/s, steady | 2 / foot at t = 2 s, chatters (0.095 rad/s rms above) |

So the mesh-foot chatter is a MuJoCo C artifact: in Warp, what mjlab trains with, the mesh foot
stands as quietly as the spheres. Warp CPU speed: ~2-3 ms per step for 1 world, 0.4-0.75 ms per
world-step at 16-64 worlds (MuJoCo C: 18 us per step), fine for checks only.

## Not covered

- The servo model: `docs/actuators.md` (`init(actuator="bam")`), `scripts/bam_check.py --model microduck`.
- Walking: no policy is part of these checks. With the description's servo gain (0.55 N m/rad) the robot does not
  stand from the STAND keyframe (`fixtures/robots/README.md`); `compare.py` holds the pose with kp 2.0.

# Per-world objects and sensors

A terrain build ([terrain.md](terrain.md)) can add kinematic objects per world, and `nablaengine.sensors` renders the
G1's sensors on the same scene: depth and segmentation, an RGB camera, the Livox MID-360 LiDAR, a height scan and an
IMU, with optional gradients with respect to the poses. The examples use JAX; the PyTorch interface has the same
pieces (`torch_ffi`, `nablaengine.sensors.torch_rays`).

## Per-world objects

```python
from nablaengine import jax_ffi, terrain as T

cfg = T.TerrainConfig(geoms="body", objects=8)   # 8 object slots per world (kinematic boxes)
jax_ffi.init(terrain=cfg)
jax_ffi.set_terrain(T.Terrain.from_model(m_scene))

# (B, K, 15) = position (3), rotation matrix row major (9), half sizes (3); half sizes 0 = absent
objects = ...                                    # set at reset or at any step, inside jit; no recompilation
com = T.pack_com(com_offset, objects)            # com becomes (B, 3 + 15 K)
q1, v1, w1 = jax_ffi.step(qpos, qvel, warm, ctrl, friction, com, iterations=10)
```

Objects are kinematic: they push the robot but nothing pushes them; move them by changing their pose. Their contacts
use the same robot geoms and contact parameters as the terrain. The step is differentiable as before; the slots'
poses get no gradient. Measured: qvel against MuJoCo Warp (the boxes as mocap bodies), median relative error 1.6e-6 on
worlds with object contacts.

### Moving objects and conveyor belts (forest engine)

`TerrainConfig(objects=K, object_velocity=True)` gives each slot 18 motion words that its contacts see: moving floors,
elevators, turntables, pushers, conveyor belts and escalators, also while they accelerate.

```python
cfg = T.TerrainConfig(geoms="body", objects=8, object_velocity=True)
jax_ffi.init(terrain=cfg)
vel = np.stack([T.slot_velocity(v=[0.8, 0.0, 0.0], a=[0.5, 0.0, 0.0]),   # a platform speeding up
                T.slot_velocity(s=[1.0, 0.0, 0.0]), ...])               # a belt: fixed pose, surface along its x axis
com = T.pack_com(com_offset, objects, vel)       # (B, 3 + 15 K + 18 K)
```

Per slot, at the start of the control step: `v`, `w` (rigid velocity, world frame, `v` at the slot's position), `s`
(surface velocity in the slot's frame; the pose does not move), `a`, `al` (linear and angular acceleration) and `sd`
(the rate of `s`). Within the control step the engine moves the slot through its substeps with these accelerations;
you set the next control step's pose and velocities. Friction drives the robot towards the surface's velocity.

The motion words get gradients (com's `object_velocity` columns, `info["com_layout"]`). The G1 engine raises
`ValueError`. Measured: qvel against MuJoCo Warp (the slot as a heavy driven body), median relative error 1.5e-6.

### Per-slot contact material (forest engine)

`TerrainConfig(objects=K, object_material=True)` gives each slot a contact material per world: friction `mu`,
`solref` (standard or direct form) and `solimp` (dmin, dmax, width, midpoint). The slot's contacts use them in place of
the terrain pair's parameters, like a MuJoCo `<pair>`.

```python
cfg = T.TerrainConfig(geoms="body", objects=2, object_velocity=True, object_material=True)
jax_ffi.init(terrain=cfg)
mat = np.stack([T.slot_material(friction=0.3, solref=(0.02, 1.0), solimp=(0.9, 0.95, 0.001, 0.5)),  # ice
                T.slot_material(friction=1.2)])                                                      # rubber mat
com = T.pack_com(com_offset, objects, vel, np.broadcast_to(mat, (B, 2, 7)))  # (B, 3 + (15 + 18 + 7) K)
```

The 7 material words per slot (`info["com_layout"]["object_material"]`) get gradients. Limits: pyramidal cone, condim
1 or 3. Measured: gradients of `mu`, `solref`, `solimp` against float64 MuJoCo C, median relative error 3e-6 to 1.4e-5.

## Sensors

```python
from nablaengine.sensors import cameras, g1

sens = g1.G1Sensors(m,                            # the engine's MjModel (body names)
                    terrain=ter,                  # the same Terrain as set_terrain
                    robot_mjcf=G1_XML,            # mjlab's g1.xml: the visual meshes
                    depth=cameras.D435I_DEPTH.with_resolution(64, 48),
                    lidar=cameras.MID360.with_rays(4096),        # or None
                    height_scan=cameras.HeightScan(size=(1.6, 1.0), resolution=0.1))
_, xpos, xquat, _, _ = jax_ffi.kinematics(q1, v1, com)          # body poses of the state
out = sens(xpos, xquat, objects, lidar_frame=t)   # jit-able; t: the LiDAR frame (step count at 10 Hz)
out["depth"]        # (B, H, W) z-depth in metres, 0 = invalid (no hit, < min_depth, > 10 m)
out["seg"]          # (B, H, W) int32 hit ids (-1: none)
out["lidar"]        # (B, n, 3) points in the LiDAR frame (zeros: no return); lidar_valid, lidar_seg
out["height_scan"]  # (B, ny, nx) body z minus hit z (terrain and objects, not the robot)
depth = g1.G1Sensors.depth_noise(key, out["depth"])             # z^2 Gaussian, dropout, edge dropout
```

- **Depth.** `cameras.D435I_DEPTH`: the D435i's field of view and minimum depth at its 1280x720 stream (87 x 58
  degrees, 0.28 m). For other streams use `cameras.d435i_depth("848x480", 64, 48)` (`cameras.D435_STREAMS`).
- **RGB.** `G1Sensors(..., rgb=cameras.D435I_COLOR.with_resolution(w, h))` adds `out["rgb"]` (B, H, W, 3) in [0, 1]:
  flat colours (MJCF rgba for the robot, `g1.Shading` for the scene), one directional light plus ambient. No textures,
  no shadows.
- **LiDAR.** `cameras.MID360` fires Livox's recorded MID-360 scan pattern (shipped in `nablaengine/sensors/data` with
  its MIT license): 200,000 directions per second, elevation -7 to 52 degrees, different every frame. `with_rays(n)`
  thins each 0.1 s frame to n rays; `lidar_frame` picks the frame (the pattern repeats every 40 frames).
  `Lidar(..., pattern="spiral")` uses a fixed golden-angle spiral instead. The LiDAR ignores its own body's meshes.
- **Height scan.** `cameras.HeightScan(size, resolution)`, as mjlab's `height_scan`.
- **IMU.** `g1.Imu(m, "imu_in_torso", dt=0.02).read(kin_now, kin_prev)` returns gyro and accelerometer in the site
  frame from the kinematics outputs `(xpos, xquat, cvel, subtree_com)` of two consecutive control steps. With per-world
  gravity (`PhysicsParams` field `gravity`) pass `read(..., gravity=g)`, `g` (B, 3) or (3,).
- **Latency.** `delay = g1.Delay(frames=2)`; `state = delay.init(x)`; `state, y = delay.push(state, x)`.
- Joint encoders are `qpos[7:]` / `qvel[6:]`.

Mounts are Unitree's URDF poses (`d435_joint`, `mid360_joint` on `torso_link`); rays start 3 cm in front of each
sensor (its housing).

Hit ids: `id >> 24` is the category (0 terrain box, 1 terrain heightfield, 2 floor plane, 3 object slot, 4 robot mesh),
`id & 0xFFFFFF` the index. `g1.label_map(ids, {raycore.ID_OBJECT: slot_labels, ...})` maps them to labels.

Measured against MuJoCo C's `mj_ray` (float64) on mjlab's rough terrain with the G1's meshes and box objects: hit
agreement 1.0, distance error median 1.3e-7.

## Object geometry

`nablaengine.object_geometry.ObjectGeometry` returns the scene's object and terrain geometry per world from the
engine's state, batched over B and JAX-traceable:

```python
from nablaengine import jax_ffi, object_geometry as og

info = jax_ffi.init(model=m, foot_pairs=fp, terrain=cfg, params=P)
geo = og.ObjectGeometry(m, geoms=("box_collision",), info=info, params=P, slot_faces="exposed",
                        terrain=static_terrain, tiles=(origins, 8.0))
table, src, report = geo.exposed_slot_faces(scene_slots)          # host, once per scene set
g = geo.from_state(qpos, qvel, com, slot_faces=(table[scene], src[scene]), tile=tile)
```

The result holds each world's `gravity` and `up`, the floor plane (`floor_valid`, `floor_height`, `floor_normal`), the
objects' boxes (`obj_pos`, `obj_rot`, `obj_half`, `obj_vel`, `obj_angvel`, `obj_valid`, `obj_kind`, `obj_movable`) and
face records (`faces` (B, F, 16): centre, outward normal, in-face axes, half widths, class top / side / bottom, on
flag). Terrain boxes contribute only their exposed faces; heightfields give none. Model objects are differentiable
with respect to `qpos` and `qvel`, slots with respect to their `com` words. `geo.box_slots(g)` turns the objects into
`G1Sensors` object slots. Measured: poses, velocities and faces within 5.9e-7 (relative) of float64 MuJoCo C.

## Sensor gradients

```python
sens = g1.G1Sensors(m, ..., grad="first_order")   # or per sensor: grad={"depth": ("smoothed", 1.0), "height_scan": "first_order"}
loss = lambda xpos, xquat, objects: f(sens(xpos, xquat, objects)["depth"])
jax.grad(loss, argnums=(0, 1, 2))(xpos, xquat, objects)      # then the kinematics VJP to qpos
```

- `grad="none"` (default): no gradient. The forward outputs are the same with every option.
- `grad="first_order"`: depth, LiDAR points and height scan are differentiable with respect to the body poses (the
  sensor's mount, the hit robot body) and the object slots' poses. It is the gradient of the hit plane: zero for
  misses, for hits jumping to another surface and for grazing hits (within 0.57 degrees of the surface). For `rgb` it
  is the shading change of rotating surfaces.
- `grad=("smoothed", sigma)` (depth, height scan, rgb; sigma in pixels or grid cells): the gradient of the image
  smoothed by a Gaussian, so edges and silhouettes give a bounded, nonzero gradient. The forward image is still exact.
  The LiDAR has no smoothed mode (it falls back to `"first_order"`).
- `grad_impl="fused"` (default) is a fused CUDA backward; `grad_impl="xla"` gives the same gradients in XLA, with
  forward mode (`jax.jvp`) too.
- PyTorch: `SceneRays.cast(..., grad="first_order")` and `SceneRays.cast_image(spec, value_fn, sigma, ...)`.

Measured (first order, against float64 central differences of `mj_ray`): median relative error 3e-8 to 2.3e-7 per
surface category.

## OptiX backend (RT cores)

Rays use CUDA kernels by default (`backend="software"`). On GPUs with ray-tracing cores, install NVIDIA's OptiX SDK
(8.0 or later), set `$OPTIX_ROOT` and pass `G1Sensors(..., backend="optix")` (or `"auto"`: OptiX where `$OPTIX_ROOT` is
set). Distances, ids and normals are recomputed with the software kernels' exact tests (hit agreement 0.999999). If a
cast fails with OptiX's out-of-memory error, lower `XLA_PYTHON_CLIENT_MEM_FRACTION`. The OptiX headers are not part of
this repository; nothing of the backend is built unless it is chosen. Gradients with the OptiX backend have not been
checked.

On an RTX PRO 6000 at 4096 worlds, depth 64 x 48 with the robot's meshes takes 4.58 ms per call with the software
kernels and 2.01 ms with OptiX.

## Checks

- `scripts/sensors/sensor_grad_check.py`: first-order gradients against float64 central differences of `mj_ray`.
- `scripts/sensors/sensor_grad_smooth_check.py`: smoothed gradients against a supersampled reference.

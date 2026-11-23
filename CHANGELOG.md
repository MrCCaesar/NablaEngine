# Changelog

## 1.0.0

- **Forest engine**, now the default for every model: several kinematic trees, hinge and slide roots, motors and
  affine actuators, Euler, joint springs, gravity compensation, up to 64 bodies. `DK_ENGINE=tree` or
  `engines.prefer_tree(m)` keeps the G1 engine for G1-class models (`docs/engines.md`).
- Contact geoms: cylinders, ellipsoids and convex meshes; contact margins; noslip (pyramidal cone).
- Per-joint limit and per-dof frictionloss solref / solimp; MuJoCo's direct solref form.
- Joint and connect equality constraints, fixed tendons (`docs/equality.md`).
- Contact forces, contact materials, gradients of the contact output (`contact_grad=True`).
- Per-world parameters: masses, inertias, COM offsets, gravity, external wrenches, foot and physics parameters,
  inverse weights (`docs/api.md`).
- Dynamics queries: mass matrix, bias force, point Jacobians with their VJP (`nablaengine.dyn_query`).
- Actuator models: torque-speed limits (`init(actuator="unitree_g1")` or a `motor.MotorLimits`) and the BAM servo
  model for Dynamixel XL330 servos (`init(actuator="bam")`, parameters from Rhoban's BAM; `docs/actuators.md`).
- Sensors with gradients: depth camera, LiDAR, height scan and IMU, per-world objects and their geometry
  (`nablaengine.sensors`, `nablaengine.object_geometry`, `docs/sensors.md`).
- Capacity: `init(max_contacts=, max_rows=)` raise the per-world contact and constraint-row limits.
- DeepMind Control Suite and Brax tasks and a MuJoCo model converter (`nablaengine.envs`, `docs/envs.md`).
- Capability codes that name what a refused model needs (T0-T14, H codes).
- Fix: implicit integration of standalone free bodies includes MuJoCo's gyroscopic term (off by default on sm_90
  until verified there).
- Experimental: models with 65-128 dofs or more than 64 bodies; contact supports beyond 32 dofs; margins on convex
  pairs; spatial tendons; muscles and activation dynamics; distance queries (`nablaengine.dist_query`);
  contact-gradient surrogates (`cone_mu`, `impulse_clip`); per-world frictionloss; foot offsets and sand on terrain;
  `DK_FOREST_CHOL_BLK`, `DK_FOREST_LATENCY`, `DK_MIDPHASE`, `capacity_diag`; Jetson Orin build notes
  (`docs/orin.md`); JAX 0.4.38.

## 0.2.0

- **Other robots.** Models of the engine's class beyond the G1:
  - 7 to 64 dofs (the G1 has 35) and up to 8 children per body (quadrupeds);
  - joint damping and frictionloss;
  - box-box contacts following MuJoCo Warp's general convex collision (GJK, EPA, multi-contact clipping), with an
    exact adjoint;
  - `init(com_body=...)` names the body of the per-world COM offset.

  Validated against MuJoCo Warp and float64 MuJoCo on the Unitree Go1, Go2, A1 and H1, ANYbotics ANYmal C, Boston
  Dynamics Spot, Booster T1, ROBOTIS OP3 and Berkeley Humanoid (`docs/robots.md`, `docs/api.md` Other robots).
  `scripts/robots/` exports and checks a model. The G1's builds are unchanged (bitwise).
  The Unitree H1 was first brought to the engine by @ShaoxiongYao in #1; 0.2.0's implementation supersedes it.
- **Terrain.** Static boxes and heightfields shared by every world, uploaded at run time without recompiling:
  - MuJoCo's closed-form box contacts;
  - two heightfield contact modes (analytic sample points, or MuJoCo's prisms measured exactly);
  - the adjoint;
  - a height scan (`raycast`, as mjlab's `RayCastSensor`).

  Available in PyTorch and JAX (`init(terrain=TerrainConfig(...))`, `set_terrain`; `docs/terrain.md`). Checked against
  MuJoCo Warp with `scripts/terrain_check.py` on the committed scenes `fixtures/mjlab/g1_terrain_*.mjb`.

## 0.1.0

First release.

- The fused CUDA engine for the Unitree G1 (mjlab model): one control step for thousands of worlds per call, with
  MuJoCo's physics and an analytic adjoint of the step.
- JAX interface (`nablaengine.jax_ffi`, `docs/api.md`, extra `nablaengine[jax]`) and PyTorch interface
  (`nablaengine.torch_ffi`, `docs/api_torch.md`, extra `nablaengine[torch]`), with bitwise the same results; the
  PyTorch one needs no JAX and runs a library built once per GPU architecture.
- Diagnostics on the device, including worlds beyond the contact (64) or constraint-row (192) capacity, whose extra
  contacts / rows are dropped: counted and flagged, with optional warnings (`set_overflow_warnings`).
- Verification against MuJoCo Warp, float64 MuJoCo and MJX (`scripts/verify.sh`, `docs/verification.md`) and the
  benchmark harness (`benchmarks/`).

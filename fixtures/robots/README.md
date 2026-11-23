# Robot fixtures (`fixtures/robots`)

The robots of `docs/robots.md`, as engine models with reference data, so that `scripts/robots/regress.py` (and
`scripts/verify.sh`) checks them without the robots' sources or MuJoCo Warp installed.

Per robot `<r>/`:

| File | Content |
|---|---|
| `<r>_flat.xml`, `<r>_flat.json` | The engine model written by `scripts/robots/export_robot.py` (visual meshes dropped, explicit contact pairs, actuator force limits on the joints) and its metadata (foot pairs, COM body, initial state, substeps; `source` names the scene it was exported from). |
| `ref.npz` | 512 states of the robot's `robot_check.py corpus` (rollouts of random PD targets in MuJoCo Warp) and native MuJoCo Warp 3.17.0's control step from each (`robot_check.py ref`, 10 iterations): the inputs, `out_qpos` / `out_qvel`, the contact counts; drawn at random (seed 0) from the 4096-state reference of `docs/results/robots_check_v100.json` (`source_rows`: their rows there; `scripts/robots/make_fixtures.py`). |
| `<r>_flat_reference.xml` | Only for a model with per-geom contact inverse weights (`n1_full`): the same export without `--fusestatic`, MuJoCo's equivalent of the model (`reference_model` in the JSON). `ref.npz`'s outputs are MuJoCo Warp's on this model (for `n1_full` recomputed on 2026-11-12 from the same 512 inputs), and `grad64`'s MuJoCo C runs it. |
| `LICENSE` | The licence of the robot's description, copied verbatim from its MuJoCo Menagerie directory. |
| `meshes/` | Collision meshes (N1's feet), the files of the robot's MuJoCo Menagerie directory. |

`results.json`: `robot_check.py check` and `grad64` (64 states) on exactly these inputs through the JAX interface,
V100 (sm_70); `regress.py` gates on it (its docstring). The PyTorch interface gives bitwise the same numbers.

| Robot | Exported from | Description licence |
|---|---|---|
| Unitree Go1 (`go1`) | mjlab's `unitree_go1_flat_env_cfg` | BSD 3-Clause, Unitree Robotics (Menagerie `unitree_go1`) |
| Unitree Go2 (`go2`) | MuJoCo Menagerie `unitree_go2/scene.xml` | BSD 3-Clause, Unitree Robotics |
| Unitree A1 (`a1`) | MuJoCo Menagerie `unitree_a1/scene.xml` | BSD 3-Clause, Unitree Robotics |
| ANYbotics ANYmal C (`anymal_c`) | MuJoCo Menagerie `anybotics_anymal_c/scene.xml` | BSD 3-Clause, ANYbotics AG |
| Boston Dynamics Spot (`spot`) | MuJoCo Playground `spot/xmls/scene_mjx_feetonly_flat_terrain.xml` | BSD 3-Clause, Clearpath Robotics (Menagerie `boston_dynamics_spot`) |
| Unitree H1 (`h1`) | MuJoCo Playground `h1/xmls/scene_mjx_feetonly.xml` | BSD 3-Clause, Unitree Robotics (Menagerie `unitree_h1`) |
| Booster T1 (`t1`) | MuJoCo Playground `t1/xmls/scene_mjx_feetonly_flat_terrain.xml` | Apache 2.0 (Menagerie `booster_t1`) |
| ROBOTIS OP3 (`op3`) | MuJoCo Playground `op3/xmls/scene_mjx_feetonly.xml` | Apache 2.0 (Menagerie `robotis_op3`) |
| Berkeley Humanoid (`berkeley_humanoid`) | MuJoCo Playground `berkeley_humanoid/xmls/scene_mjx_feetonly_flat_terrain.xml` | BSD 3-Clause, Hybrid Robotics (Menagerie `berkeley_humanoid`) |
| Apptronik Apollo (`apollo`) | MuJoCo Playground `apollo/xmls/scene_mjx_feetonly_flat_terrain.xml` (`--fusestatic`) | Apache 2.0 (Menagerie `apptronik_apollo`) |
| PNDbotics Adam Lite (`adam_lite`) | MuJoCo Menagerie `pndbotics_adam_lite/scene.xml` | MIT, pndrobotics (Menagerie `pndbotics_adam_lite`) |
| PNDbotics Adam Lite, full collision (`adam_lite_full`) | the same scene, every collision geom's pairs (cylinder-cylinder, box-cylinder, sphere-cylinder, sphere-box, box-box) | MIT, pndrobotics |
| Fourier N1 (`n1`) | MuJoCo Menagerie `fourier_n1/scene.xml` (`--fusestatic`; mesh feet: `meshes/`, the two collision meshes as in Menagerie) | Apache 2.0 (Menagerie `fourier_n1`) |
| Fourier N1, full collision (`n1_full`) | the same scene, every collision geom's pairs (mesh-mesh, cylinder-mesh, sphere-mesh, cylinder-cylinder, ...) | Apache 2.0 (Menagerie `fourier_n1`) |
| ANYbotics ANYmal B (`anymal_b`) | MuJoCo Menagerie `anybotics_anymal_b/scene.xml` | BSD 3-Clause, ANYbotics AG |
| Google Barkour v0 (`barkour_v0`) | MuJoCo Menagerie `google_barkour_v0/scene_mjx.xml` | Apache 2.0, DeepMind Technologies (Menagerie `google_barkour_v0`) |
| Google Barkour vB (`barkour_vb`) | MuJoCo Menagerie `google_barkour_vb/scene_mjx.xml` with MuJoCo Playground's Barkour joystick changes (`--fusestatic`) | Apache 2.0, DeepMind Technologies (Menagerie `google_barkour_vb`) |
| Boston Dynamics Spot with arm (`spot_arm`) | MuJoCo Menagerie `boston_dynamics_spot/scene_arm.xml` (`--fusestatic`) | BSD 3-Clause, Clearpath Robotics (Menagerie `boston_dynamics_spot`) |
| Unitree G1 with hands (`g1_with_hands`) | MuJoCo Menagerie `unitree_g1/scene_with_hands.xml` (45 bodies, nv 49) | BSD 3-Clause, Unitree Robotics (Menagerie `unitree_g1`) |

mjlab and MuJoCo Playground, whose scene files and settings the exports follow, are Apache 2.0
([mjlab](https://github.com/mujocolab/mjlab), [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground));
the robot descriptions come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

### Actuator gains of the later fixtures

The last five robots (added 2026-11-12, MuJoCo Menagerie commit f3b6013) keep the actuator gains of the stack that
trains them, or, where no MuJoCo Playground or mjlab environment trains the variant, Menagerie's own position
actuators as shipped; no gain is chosen here. All are feet-only exports (`--feet-only`, explicit foot-floor pairs).

| Robot | Gains from | kp | kv | Force limit | Armature | Joint damping / frictionloss | Timestep, substeps | Export options |
|---|---|---|---|---|---|---|---|---|
| `anymal_b` | Menagerie `anymal_b.xml` (class `affine`) | 100 | 0 | 40 | 0 | 1 / 0.1 | 2 ms, 10 | `--foot-class foot --feet-only --locomotion --com-body base --substeps 10` |
| `barkour_v0` | Menagerie `barkour_v0_mjx.xml` (class `multi_mode_controlled_actuator`) | 35 | 0.65 | 18 | 0.0109 | 0.5239 / 0 | 2 ms, 10 | `--key home --foot-class lowerLegFootLeft,lowerLegFootRight --feet-only --com-body chassis --substeps 10` |
| `barkour_vb` | MuJoCo Playground 0.2.0 `locomotion/barkour/joystick.py` (it sets kp, damping and `sim_dt` on Menagerie's scene; `ctrl_dt` 0.02) | 35 | 0.5 | 18 | 0.011 | 0.5239 / 0 | 4 ms, 5 | `--key home --foot-geoms foot_front_left,foot_hind_left,foot_front_right,foot_hind_right --feet-only --fusestatic --kp 35 --damping 0.5239 --timestep 0.004 --com-body torso --substeps 5` |
| `spot_arm` | Menagerie `spot_arm.xml` (class `spot`, legs and arm) | 500 | 40 | 1000 (joint `actuatorfrcrange`) | 0 | 0 / 0 | 2 ms, 10 | `--key home --foot-geoms FL,FR,HL,HR --feet-only --locomotion --fusestatic --invweight converted --drop-unpaired-meshes --com-body body --substeps 10` |
| `g1_with_hands` | Menagerie `g1_with_hands.xml` (class `g1`: `kp="500" dampratio="1"`) | 500 | 4.48-16.2 (dampratio 1) | 1.4-139 per joint (`actuatorfrcrange`) | 0.01 | 0 / 0.3 | 2 ms, 10 | `--key stand --foot-class foot --feet-only --drop-unpaired-meshes --com-body torso_link --substeps 10` |

`barkour_vb` and `spot_arm` need `--fusestatic` (welded jointless bodies: barkour_vb's camera mount `oak/` and an
unnamed one, spot_arm's `arm_link_hr0`); no geom's contact inverse
weight changes (`geom_invweight` false), so `spot_arm` takes `--invweight converted` with the same values.
`--drop-unpaired-meshes` drops the collision meshes in no contact pair (no body's inertia changes).

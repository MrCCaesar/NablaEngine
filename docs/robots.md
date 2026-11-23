# Other robots

Besides the Unitree G1, the engine runs other legged robots of its model class (`docs/api.md`, Other robots): a
floating base with one hinge per body, up to 63 bodies and 64 dofs, and contact pairs between planes, spheres,
capsules, boxes and cylinders, and convex meshes against planes. This page shows how to bring a robot in, check it,
and what was measured.

## Exporting a model

The engine simulates explicit contact pairs only. `scripts/robots/export_robot.py` turns a scene into an engine model:
- it drops the visual meshes;
- it turns MuJoCo's contype / conaffinity contacts into explicit pairs with MuJoCo's mixed contact parameters;
- it orders the actuators by joint and moves their force limits onto the joints.

It writes `OUT.xml` and `OUT.json`. The JSON holds the foot pairs, the body that takes the COM offset, the initial
state and the control step's substeps.

```bash
# an mjlab robot (run in an mjlab environment)
python scripts/robots/export_robot.py --robot go1 --out models/go1/go1_flat
# a MuJoCo Playground feet-only scene
python scripts/robots/export_robot.py --mjcf <playground>/locomotion/h1/xmls/scene_mjx_feetonly.xml --name h1 \
    --foot-geoms left_foot1,left_foot2,left_foot3,right_foot1,right_foot2,right_foot3 --com-body torso_link --substeps 5 --key home --out models/h1/h1_flat
# a MuJoCo Menagerie scene: feet-only collision, pyramidal cone and condim 3, torque motors as position servos
python scripts/robots/export_robot.py --mjcf <menagerie>/unitree_go2/scene.xml --name go2 --foot-class foot \
    --feet-only --locomotion --servo 35,0.5 --com-body base --substeps 10 --key home --out models/go2/go2_flat
# named foot geoms (Adam Lite's two cylinders per foot)
python scripts/robots/export_robot.py --mjcf <menagerie>/pndbotics_adam_lite/scene.xml --name adam_lite \
    --foot-geoms toe1_left,toe2_left,toe1_right,toe2_right --feet-only --locomotion --servo 100,2 --com-body pelvis \
    --substeps 10 --out models/adam_lite/adam_lite_flat
# mesh feet (Fourier N1; the collision meshes are copied next to the model, meshes/)
python scripts/robots/export_robot.py --mjcf <menagerie>/fourier_n1/scene.xml --name n1 \
    --foot-geoms left_foot_pitch_link_collision0,right_foot_pitch_link_collision0 --feet-only --locomotion \
    --servo 80,2 --com-body base_link --substeps 10 --key home --fusestatic --out models/n1/n1_flat
# jointless bodies (Apollo's camera frames) merged into their parents: one hinge per body after the root
python scripts/robots/export_robot.py --mjcf <playground>/locomotion/apollo/xmls/scene_mjx_feetonly_flat_terrain.xml \
    --name apollo --foot-geoms collision_l_sole,collision_r_sole --com-body base_link --substeps 4 --key knees_bent \
    --fusestatic --out models/apollo/apollo_flat
# model changes an environment makes in code (MuJoCo Playground's Barkour joystick: kp, damping, timestep)
python scripts/robots/export_robot.py --mjcf <menagerie>/google_barkour_vb/scene_mjx.xml --name barkour_vb --key home \
    --foot-geoms foot_front_left,foot_hind_left,foot_front_right,foot_hind_right --feet-only --fusestatic \
    --kp 35 --damping 0.5239 --timestep 0.004 --com-body torso --substeps 5 --out models/barkour_vb/barkour_vb_flat
# feet-only with mesh collision geoms elsewhere (G1 with hands): the meshes in no pair dropped
python scripts/robots/export_robot.py --mjcf <menagerie>/unitree_g1/scene_with_hands.xml --name g1_with_hands \
    --key stand --foot-class foot --feet-only --drop-unpaired-meshes --com-body torso_link --substeps 10 \
    --out models/g1_with_hands/g1_with_hands_flat
```

With `--fusestatic`, a fused body changes the contact softness of its own geoms and its parent's, and MuJoCo can
change other bodies' too (`docs/api.md`, Per-geom contact inverse weights). Where that reaches a geom of a contact pair, the export writes the unfused model's
values per geom (the custom numeric `nablaengine_geom_invweight`; needs `nablaengine` importable, e.g.
`PYTHONPATH=src`) and also writes `OUT_reference.xml`, the same export without `--fusestatic`. That file is MuJoCo's
equivalent of the model (MuJoCo C and MuJoCo Warp ignore the field) and `OUT.json` names it (`reference_model`).
`--invweight converted` keeps MuJoCo's fused values instead: no field, no reference model. Of the models below,
only the full-collision N1 gets the field (its torso, waist, base and hand geoms; the feet-only N1 and Apollo have no
pair on a touched geom).

Then use the model in either interface:

```python
m = mujoco.MjModel.from_xml_path("models/go2/go2_flat.xml")
meta = json.load(open("models/go2/go2_flat.json"))
torch_ffi.init(model=m, foot_pairs=meta["foot_pairs"], com_body=meta["com_body"])
torch_ffi.step(qpos, qvel, warm, ctrl, friction, com, substeps=meta["substeps_per_control_step"])
```

A model outside the class raises `UnsupportedModel` with the failing checks before anything is built.

## Checking a model

`scripts/robots/robot_check.py` runs four steps:
- `corpus`: rollouts of random PD targets in MuJoCo Warp;
- `ref`: one control step from those states in MuJoCo Warp;
- `check`: the engine on the same inputs;
- `grad64`: the engine's gradients against float64 central differences of MuJoCo C;
- `adjdebug`: the debug adjoint's built-in residual check (and its gradients bitwise the production build's).

For a model with a `reference_model`, `corpus`, `ref` and `grad64` run MuJoCo Warp and MuJoCo C on that model and the
engine on the exported one.

```bash
python scripts/robots/robot_check.py corpus --model models/go2/go2_flat --out models/go2/corpus.npz
python scripts/robots/robot_check.py ref    --model models/go2/go2_flat --corpus models/go2/corpus.npz --out models/go2/ref.npz
python scripts/robots/robot_check.py check  --model models/go2/go2_flat --ref models/go2/ref.npz
python scripts/robots/robot_check.py grad64 --model models/go2/go2_flat --ref models/go2/ref.npz
python scripts/robots/robot_check.py time   --model models/go2/go2_flat --ref models/go2/ref.npz
```

`scripts/robots/regress.py` checks the nineteen models below in one command on the committed `fixtures/robots`
(512 reference states each, gated on the results recorded there; `scripts/verify.sh` runs it):

```bash
python scripts/robots/regress.py --grad 64           # forward vs MuJoCo Warp, gradients vs float64 MuJoCo C
python scripts/robots/regress.py --robots t1,apollo  # some of them, forward only
```

## Results

Accuracy: the table in `docs/api.md` (Other robots), and `docs/results/robots_check_v100.json`.

Time per 20 ms control step at 4096 worlds on one RTX PRO 6000 (10 iterations, medians of 30;
`docs/results/robots_time_rtx6000.json`):

| robot | substeps | forward ms | forward + backward ms |
|---|---|---|---|
| Unitree Go1 | 4 | 0.684 | 1.014 |
| Unitree Go2 | 10 | 1.956 | 2.661 |
| Unitree A1 | 10 | 1.898 | 2.598 |
| ANYbotics ANYmal C | 10 | 1.861 | 2.563 |
| Boston Dynamics Spot | 5 | 0.946 | 1.304 |
| Unitree H1 | 5 | 0.852 | 1.316 |
| Booster T1 | 10 | 3.931 | 6.285 |
| ROBOTIS OP3 | 5 | 1.871 | 3.480 |
| Berkeley Humanoid | 10 | 2.867 | 4.830 |

The G1's own timings (its benchmark states) are in the README.

Cost per substep: spheres and capsules about 0.2 ms, box-box contacts about 0.3-0.4 ms.

The substeps follow each source's timestep: Menagerie's 2 ms, Playground's 2-4 ms, mjlab's 5 ms.

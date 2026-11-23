# NablaEngine

**Thousands of worlds per call on one GPU, with analytic gradients for learning, control and system identification, aligned with MuJoCo.**

*Pronounced "Nabla Engine".*

## News

- **2026-11-23, 1.0.2:** on PyPI: `pip install nablaengine`.
- **2026-11-23, 1.0.1:** Pollen Robotics' Microduck with BAM servo checks (#3), and an overview of the supported
  scenes.
- **2026-11-22, 1.0.0:** the forest engine (several trees, motors, meshes, equality constraints, tendons, noslip),
  per-world parameters, dynamics queries, the BAM servo model, and DeepMind Control Suite / Brax tasks.
- **2026-11-07, 0.2.0:** other legged robots (Unitree Go1, Go2, A1 and H1 (#1), ANYbotics ANYmal C, Boston Dynamics
  Spot, Booster T1, ROBOTIS OP3, Berkeley Humanoid; `docs/robots.md`) and terrain: boxes, heightfields and a height
  scan (`docs/terrain.md`). The G1's results are unchanged.
- **2026-10-31, 0.1.0:** first release: the Unitree G1.

All release notes: [Releases](https://github.com/sailab-ai/NablaEngine/releases), [`CHANGELOG.md`](https://github.com/sailab-ai/NablaEngine/blob/main/CHANGELOG.md).

<p align="center"><img src="https://raw.githubusercontent.com/sailab-ai/NablaEngine/main/docs/assets/speed_fwd.png" alt="Forward step: NablaEngine 1.21 ms per control step, against MuJoCo Warp, Genesis, Isaac Lab, MJX and MuJoCo C" width="820"></p>
<p align="center"><img src="https://raw.githubusercontent.com/sailab-ai/NablaEngine/main/docs/assets/speed_bwd.png" alt="A step and its gradient: NablaEngine 1.78 ms, MuJoCo Warp with its adjoint pull request 20.5 ms" width="820"></p>
<p align="center"><img src="https://raw.githubusercontent.com/sailab-ai/NablaEngine/main/docs/assets/speed_ppo.png" alt="Time per PPO iteration: NablaEngine 0.123 s, MuJoCo Warp 0.355 s" width="820"></p>

<sub>One 20 ms control step for 4096 Unitree G1 worlds on one NVIDIA RTX PRO 6000 (forward, and forward with the
gradient from the input state), and one PPO iteration (4096 environments x 24 steps, plus 20 gradient steps) with the
same configuration, on the G1 engine (`DK_ENGINE=tree`; the forest engine, the default since 1.0.0, is measured
against it in `docs/engines.md`). MuJoCo Warp and MJX simulate the same model (medians over 3 processes,
`docs/results/paper_rtx6000/paper_bench.md`); Isaac Lab runs its own G1 asset with its task defaults (median over
rounds, `docs/results/physx_rtx6000/physx_bench_results.md`). Other GPUs (H100, RTX 3090, V100): the
[project page](https://sailab.ai/nablamimic/).</sub>

NablaEngine is a differentiable GPU physics engine, built first for the Unitree G1 humanoid and since 1.0.0 for a
wider class of MuJoCo models. It simulates thousands of worlds per call with MuJoCo's physics and returns analytic gradients of every step with respect to the state and the actions,
for first-order policy learning, trajectory optimization and system identification. It plugs into JAX
(`jax.jit`, `jax.grad`, `lax.scan`) and PyTorch (`torch.autograd`, CUDA graphs, `torch.compile`) as a differentiable
function, with bitwise the same results in both.

- **Fast.** One control step of 4096 worlds in 1.21 ms, its gradient in 1.78 ms, and the full 71x99 Jacobian of the
  step in 9.7 µs per world (RTX PRO 6000, the G1 engine).
- **Exact.** Against float64 MuJoCo on 16,384 held-out states: forward median relative error 8.5e-7, gradient median
  relative errors 3.6e-5 / 1.0e-5 / 1.9e-6 (qpos / qvel / ctrl) against float64 finite differences
  (`docs/results/paper_accuracy_heldout_v100.md`).
- **Deterministic.** Outputs and gradients are bitwise identical from run to run.

**Used by.** [∇Mimic (NablaMimic)](https://github.com/sailab-ai/NablaMimic), whole-body motion tracking learned with
the engine's simulation gradients.

**Paper.** [∇Mimic: Ultra-Efficient Humanoid Control with Differentiable Simulation](https://sailab.ai/nablamimic/assets/nablamimic.pdf) ([project page](https://sailab.ai/nablamimic/))

## Scope

Two code generators share one interface; the engine is picked from the model (`docs/engines.md`):

- **Forest engine** (default): several kinematic trees, hinge and slide roots, motors and affine actuators, Euler and
  implicitfast, up to 64 bodies; planes, spheres, capsules, boxes, cylinders, ellipsoids and convex meshes; equality
  constraints and fixed tendons; noslip; per-world parameters; dynamics queries; the DeepMind Control Suite and Brax
  tasks (`nablaengine.envs`).
- **G1 engine**: one free tree with position servos (the mjlab G1 and the robots of `docs/robots.md`); select it with
  `DK_ENGINE=tree` or `engines.prefer_tree(m)`.

Both run MuJoCo's Newton solver in float32 with analytic gradients, terrain builds (`docs/terrain.md`) and the BAM
servo model (`docs/actuators.md`). Experimental features are listed in `CHANGELOG.md`. Unsupported models are
rejected with an error that names the missing features.

Requirements: Linux x86_64, an NVIDIA GPU of compute capability 7.0 or newer (tested: V100, RTX 3090, H100, RTX PRO
6000), Python 3.12, and a CUDA 12.x toolkit (nvcc) for the one-time build of the CUDA library (about 15 s, cached).

## What you can simulate

- **Humanoids on mixed terrain**: the Unitree G1 and other humanoids on stairs, steps, boxes and rough heightfields,
  with a height scan and moving platforms (`docs/terrain.md`, `docs/robots.md`). Motion tracking built on the engine:
  [∇Mimic](https://github.com/sailab-ai/NablaMimic).
- **Humanoid-object interaction and loco-manipulation**: humanoids and arms with free objects in one scene (forest
  engine): contacts with boxes and convex meshes, per-world objects, contact forces and distance queries with gradients
  (`docs/engines.md`, `docs/api.md`; example scene `scripts/forest/models/g1box.xml`).
- **Legged robots**: Unitree Go1, Go2, A1 and H1, ANYmal C, Spot, Booster T1, OP3, Microduck and others
  (`docs/robots.md`, `scripts/robots/`).
- **Control benchmarks**: DeepMind Control Suite and Brax tasks (`nablaengine.envs`, `docs/envs.md`).
- **Actuator models**: the BAM servo model and torque-speed limits (`docs/actuators.md`).
- **Experimental**: models with up to 128 dofs, spatial tendons and muscles (`docs/engines.md`, `docs/equality.md`).

## Installation

From PyPI (Linux, Python 3.12, an NVIDIA GPU):

```bash
pip install "nablaengine[jax,cuda]"   # the JAX interface, with JAX's CUDA 12 plugin
pip install "nablaengine[torch]"      # the PyTorch interface (on a V100 the cu126 wheels of torch)
```

The CUDA library is compiled on first use, so a CUDA 12.x toolkit is needed as well (`nvcc` on `PATH`, or
`NVCC=/path/to/cuda-12.x/bin/nvcc`). The package includes the G1 model and the corpus of the quick start; the checks,
the scripts and the other robot models need a checkout:

```bash
git clone https://github.com/sailab-ai/NablaEngine.git NablaEngine && cd NablaEngine
scripts/setup_env.sh               # a Python 3.12 environment in .dk_env/ with the tested versions
source .dk_env/activate.sh
scripts/verify.sh --quick          # checks the installation in a few minutes
```

For PyTorch, add it to that environment and build the engine's library once (`docs/api_torch.md`):

```bash
pip install torch                  # a CUDA 12 build; on a V100 the cu126 wheels
python -m nablaengine.torch_ffi build
```

Options, a PyTorch-only environment, a manual installation and the packaged files: `docs/install.md`.

## Quick start

JAX:

```python
import jax
import jax.numpy as jnp

from nablaengine import fixtures, jax_ffi

jax_ffi.init()                         # builds the CUDA library on first use, then loads it from the cache
z = fixtures.load_corpus("mf_s0")      # 4096 states of the G1
B = 256
qpos, qvel, warm, ctrl, friction, com = (jnp.asarray(z[k][:B]) for k in
                                         ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com"))

qpos1, qvel1, warm1 = jax_ffi.step(qpos, qvel, warm, ctrl, friction, com)  # one control step: 4 substeps of 5 ms


def loss(ctrl):
    q, v, _ = jax_ffi.step(qpos, qvel, warm, ctrl, friction, com)
    return jnp.sum(v ** 2)


g = jax.jit(jax.grad(loss))(ctrl)      # through the analytic adjoint
```

PyTorch (no JAX needed):

```python
import torch

from nablaengine import fixtures, torch_ffi

torch_ffi.init()                       # loads the library built by `python -m nablaengine.torch_ffi build`
z = fixtures.load_corpus("mf_s0")
qpos, qvel, warm, ctrl, friction, com = (torch.from_numpy(z[k][:256]).cuda() for k in
                                         ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com"))
ctrl.requires_grad_()
qpos1, qvel1, warm1 = torch_ffi.step(qpos, qvel, warm, ctrl, friction, com)
(qvel1 ** 2).sum().backward()          # ctrl.grad through the analytic adjoint
```

Inputs are float32 arrays with the worlds as the leading dimension: `qpos (B, 36)`, `qvel (B, 35)`, the solver's warm
start `warm (B, 35)`, the position targets `ctrl (B, 29)`, and per-world foot `friction (B,)` and torso
centre-of-mass offset `com (B, 3)` for domain randomization. By default each world holds up to 64 contacts and 192
constraint rows (`init(max_contacts=, max_rows=)` raises them); a world beyond them (e.g. deep interpenetration after a
fall) drops the extra ones and is flagged and counted (`docs/api.md`, Diagnostics, Capacity).

## Documentation

- `docs/api.md`: the JAX interface: options, gradient modes, kinematics outputs, rollouts with `lax.scan` and
  `jax.checkpoint`, diagnostics.
- `docs/api_torch.md`: the PyTorch interface: the build artifact, CUDA graphs, `torch.compile`, several GPUs.
- `docs/engines.md`: the two engines: which model goes where, what each supports, measurements.
- `docs/equality.md`: equality constraints, fixed and spatial tendons, muscles (forest engine).
- `docs/envs.md`: the DeepMind Control Suite and Brax tasks and the MuJoCo model converter.
- `docs/robots.md`: other robots: exporting and checking a model, accuracy and timings.
- `docs/terrain.md`: terrain builds: boxes, heightfields, the height scan.
- `docs/sensors.md`: per-world objects, their geometry per world (`object_geometry`), and the G1's sensors (depth,
  segmentation, RGB, LiDAR, height scan, IMU) with their gradients.
- `docs/verification.md`: what `scripts/verify.sh` checks, against which references.
- `docs/actuators.md`: the BAM servo model.
- `docs/configuration.md`: environment variables and build switches.
- `docs/orin.md`: build notes for Jetson Orin (sm_87).
- `benchmarks/README.md`, `scripts/paper_bench/README.md`: reproduce the accuracy tables and the timings.

## Licence

Apache License 2.0 (`LICENSE`), Copyright 2026 The NablaEngine Authors. `NOTICE` and `THIRD_PARTY_NOTICES.md` list the
code derived from MuJoCo Warp and MuJoCo (Apache-2.0) and the model, which is derived from MuJoCo Menagerie's Unitree
G1 (BSD-3-Clause, `fixtures/mjlab/LICENSE`) with mjlab's configuration (Apache-2.0).

## Citation

A paper reference will be added here. Until then, please cite the software (`CITATION.cff`):

```bibtex
@software{nablaengine,
  title  = {NablaEngine: fast and exact differentiable physics simulation for embodied AI},
  author = {{The NablaEngine Authors}},
  year   = {2026},
  url    = {https://github.com/sailab-ai/NablaEngine}
}
```

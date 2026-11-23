# Task environments (`nablaengine.envs`)

Batched, differentiable versions of the DeepMind Control Suite tasks and Brax's locomotion tasks, for JAX and
PyTorch. Each task uses the original's model, observation, reward, initial-state randomization and control step, so a
controller made here runs unchanged in the original environment (dm_control's `suite.load`, Brax's
`envs.get_environment(name, backend="mjx")`).

## Usage

```python
from nablaengine import envs

task = envs.make("cheetah-run", backend="jax")      # or backend="torch", device="cuda:0"
state = task.reset(key, 4096)                       # jax.random.PRNGKey / torch.Generator on the device
state2 = task.step(state, action)                   # (B, nu); differentiable in state.qpos, state.qvel and action
obs = task.obs(state2)                              # (B, obs_dim), the original's layout
r, terms = task.reward(state2, action, state)       # (B,), per-term dict
done = task.terminated(state2)                      # DMC: never; Brax: the healthy checks
trunc = task.truncated(state2)                      # state2.t >= task.time_limit
state = task.where(done | trunc, task.reset(key2, 4096), state2)   # nothing resets by itself
```

One task per process: the engine library is built for one model. The first `make` compiles it (about a minute);
later processes load it from the cache.

`make(name, backend="jax", integrator=None, refine=None, iterations=None, reward=None, device=None)`:

- `integrator`: `None` (the original's; RK4 becomes Euler), `"euler"` or `"implicitfast"`;
- `refine`: engine substeps per original physics step (`None`: 4 for the RK4 tasks cartpole and acrobot, else 1);
- `iterations`: Newton iterations per substep (`None`: the original's);
- `reward`: `fn(task, state2, action, state) -> (r, terms)` replaces the original reward.

## Interface

| member | meaning |
|---|---|
| `obs_dim`, `critic_obs_dim`, `obs_keys` | observation size; `obs_keys` lists the original's entries in order, `[(key, shape)]` |
| `nu`, `action_low`, `action_high` | the ctrlrange (actions are not rescaled) |
| `control_dt`, `time_limit` | the original's control step and episode length in control steps (1000 for every task) |
| `max_reward` | per step (DMC: 1; Brax: None) |
| `reset(key, B)` | `TaskState(qpos, qvel, warm, t, aux)` with the original's initial-state distributions |
| `step(state, action)` | clamps the action to the ctrlrange (zero gradient outside), runs one control step, `t + 1` |
| `clamp(action)` | the applied action |
| `obs`, `obs_dict`, `critic_obs` | differentiable, no normalization |
| `reward(state2, action, state)` | the original reward of the transition |
| `terminated`, `truncated` | separate, for bootstrapping at truncations |
| `where(mask, a, b)` | per-world selection between two states |
| `frames(state)` | the original model's kinematics of the state (`envs.kin.Frames`) |

`step` and `reset` attach the state's kinematics to the returned `TaskState`, and `obs`, `reward`, `terminated` and
`frames` reuse them. A state built another way (`_replace`, `where`, a `jit` / `scan` boundary) recomputes them on
use; results are the same. With PyTorch, two separate backward passes that both stop at the same state need
`retain_graph=True` on the first. `DK_ENVS_REUSE_FRAMES=0` (set before import) recomputes the kinematics on every call.

`DK_ENVS_FK` picks the PyTorch kinematics form: `auto` (default: batched on CUDA, body by body on CPU), `batched` or
`bodies`.

## Tasks

DMC (needs `dm_control`): `cartpole-{balance,balance_sparse,swingup,swingup_sparse}`, `pendulum-swingup`,
`acrobot-{swingup,swingup_sparse}`, `cheetah-run`, `walker-{stand,walk,run}`, `hopper-{stand,hop}`,
`reacher-{easy,hard}`, `finger-{spin,turn_easy,turn_hard}`, `humanoid-{stand,walk,run}`.

Brax (needs `brax`): `brax-halfcheetah`, `brax-hopper`, `brax-walker2d`, `brax-ant`.

## How the models are converted

`envs.convert.to_engine` gives the same physics in the forest engine's structure: a body with several joints becomes
a chain of one-joint bodies, gears fold into the actuator gains, contype / conaffinity become explicit pairs, jointless
bodies are fused into their parents, disable flags become model fields. Observations and rewards are computed on the
original model.

- Fused bodies keep their source contact softness per geom (`invweight="geom"`, the default);
  `invweight="converted"` uses the fused model's values.
- Joint limits keep each joint's solref / solimp; their margins are set to 0. Contact margins and gaps are kept on
  every pair type the forest engine collides.
- Mocap bodies are frozen at their model pose. Frame sensors on a fused body move to an added site; other sensors that
  pin the body are dropped. The engine computes no MuJoCo sensors.
- condim-6 pairs are refused (H17); `to_engine(..., condim6_to=3)` converts them with a warning. No task here has any.
- Everything removed or changed is listed in `Converted.notes`.

## Differences from the originals

- cartpole, acrobot: Euler at a quarter step (`refine=4`) in place of RK4.
- humanoid: the head and hands are fused into their parents; the reset resamples up to 32 times to avoid an initial
  contact.
- finger: the suite's elliptic cone is kept (`convert.to_engine(..., cone="pyramidal")` gives the pyramidal variant);
  the decoration cylinders' contact pairs are left out.
- hopper, finger: the touch observation comes from the contacts of the control step's last physics step; it is 0
  right after a reset.
- brax-ant: one Newton iteration with an exact line search (MJX: 4 line-search iterations).
- float32 throughout (dm_control: float64).

Measured: observations and rewards on the originals' own transitions agree to 2e-6 or better (float32 against
float64).

## Checks

- `scripts/envs/dmc_check.py --task NAME`, `scripts/envs/brax_check.py --task NAME`: observations, rewards,
  open-loop rollouts and reset distributions against the originals.
- `scripts/envs/kin_check.py`: the kinematics against MuJoCo C (values and gradients); `scripts/envs/kin_bench.py`
  times them.

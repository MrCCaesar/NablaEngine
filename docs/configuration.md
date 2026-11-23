# Configuration

Physics comes from the model, the `init()` options and the per-world parameters (`docs/api.md`), not from these
knobs. There are two kinds of `DK_*` knobs:

- **Environment variables**, read by Python at build or run time.
- **Compile-time switches**, C preprocessor macros (`-DNAME=VALUE`) of the CUDA source. They act only in a build that
  passes them; exporting one as an environment variable does nothing. The FFI libraries use the defaults.

"Changes results?" compares with the default build: **no** means bitwise identical outputs, tape and gradients;
**rounding** means float32 rounding differences only; **yes** means different results.

## Environment variables

### Build and toolchain

| variable | default | effect | changes results? |
|---|---|---|---|
| `NVCC` | `$NVCC`, `nvcc` on `PATH`, `$CUDA_HOME/bin/nvcc`, pip `nvidia.cuda_nvcc`, `/usr/local/cuda/bin/nvcc` | nvcc for every build; use CUDA 12.x (sm_120 needs 12.8+) | yes (another nvcc gives other binaries) |
| `CUDA_HOME` | unset | `$CUDA_HOME/bin/nvcc` is an nvcc candidate | as `NVCC` |
| `DK_ARCH` | first visible GPU's | build target (`sm_70`, `sm_86`, `sm_90`, `sm_120`, ...); set it to build without a GPU | yes if not the GPU's own architecture |
| `DK_BUILD_JOBS` | CPU count | parallel nvcc / ptxas units; lower it on machines with little RAM per core | no |
| `DK_CACHE_DIR` | `$XDG_CACHE_HOME/nablaengine`, else `~/.cache/nablaengine` | where built libraries are kept (never pruned) | no |
| `XDG_CACHE_HOME` | `~/.cache` | see `DK_CACHE_DIR` | no |
| `DK_TORCH_ARTIFACT` | the cache's `torch_<arch>_current.json` | PyTorch artifact to load | no |
| `CUDA_VISIBLE_DEVICES` | all GPUs | GPUs a process uses; `DK_ARCH` follows the first | no (one GPU type) |
| `DK_FIXTURES` | `<checkout>/fixtures/mjlab`, else the packaged copy (model only) | fixtures directory (default model and references) | selects data |
| `DK_BENCH_DATA` | `<checkout>/benchmarks/data`, else the packaged copy (`mf_s0` only) | where benchmark state sets are read from | selects data |
| `DK_KEEP_WORK` | unset | `1` keeps the scripts' temporary work directories | no |
| `TMPDIR` | `/tmp` | where work directories are made | no |
| `JAX_PLATFORMS` | JAX's | some scripts set `cpu` | no |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `true` | set `false` when JAX shares a GPU | no |

### Engine and physics

| variable | default | effect | changes results? |
|---|---|---|---|
| `DK_ENGINE` | `auto` (forest) | engine for G1-class models: `forest`, `tree` (or `g1`), `auto`; `engines.prefer_forest/prefer_tree(m)` override per model | `tree` vs `forest`: rounding |
| `DK_CLAMP_IMPLICIT` | `0` | implicitfast damping on a dof clamped at its actuator force range: 0 keeps kv, 1 drops it, 2 fades it (differentiated); one value per process | 1, 2: yes |
| `DK_CLAMP_IMPLICIT_DELTA` | `0.2` | width of mode 2 (only with `DK_CLAMP_IMPLICIT=2`) | yes (mode 2) |
| `DK_MAXCON` | 64 (32 for models with at most 64 pairs) | contact capacity per world; experiment switch, warns when set (prefer `init(max_contacts=)`) | yes |
| `DK_OVERFLOW_WARN` | `0` | `1`: diagnostics readers warn on contact / row capacity overflow | no |
| `DK_ALLOW_UNSUPPORTED` | unset | comma-separated capability codes (or `all`) turned into warnings; the result is no longer MuJoCo's physics | yes, for the models it lets through |
| `DK_DEFER_BOX` | `0` | forest: `1` computes box-box and capsule-box pairs in the deferred collision kernel | yes (contact order) |
| `DK_ENVS_FK` | `auto` | PyTorch envs kinematics form: `batched`, `bodies`, `auto` | rounding |
| `DK_GJK_MJ4` | `0` | `1`: MuJoCo Warp's newer GJK termination rule | only in degenerate GJK cases |
| `DK_CCD_NT` | `4` | tangents per pass in the native-CCD contact VJP: 1, 2, 3, 4, 6 or 12; larger is faster and uses more local memory | n > 1: rounding |
| `DK_MIDPHASE` (experimental) | `auto` | forest collision mid-phase: `auto` (models naming groups), `0`, `1` (node level), `2` (node and group level) | rounding (contact order) |
| `DK_FOREST_LATENCY` (experimental) | `0` | `1`: two warps per world for small batches (faster below a few hundred worlds, slower above) | no |
| `DK_FOREST_CHOL_BLK` (experimental) | `0` | `1`: register-blocked Cholesky for forest models with 44-64 dofs (faster from 44 dofs on V100) | 49-64 dofs on V100: no; otherwise rounding |

### Per-architecture code generation

| variable | default | effect | changes results? |
|---|---|---|---|
| `DK_SM120_FRONTEND` | `n7` | sm_120 front end: `n7` (all units from compute_90 PTX), `split`, `native` | yes on sm_120 |
| `DK_SM120_FORWARD_FRONTEND` | unset | deprecated alias of `DK_SM120_FRONTEND` | as `DK_SM120_FRONTEND` |
| `DK_SM120_PRE_CAP` | `1` | sm_120 NVVM 7 forward: register cap on `k_pre` | no |
| `DK_SM103_FRONTEND` | `adjoint` | sm_103: `adjoint` (adjoint units from compute_90 PTX), `n7`, `native` | yes on sm_103 (gradients) |
| `DK_SM103_REG_CAPS` | `1` | sm_103: register caps on the forward and the light adjoints | no |
| `DK_SM103_FOREST_ADJ_CAP` | `1` | sm_103 forest adjoint cap: `1` automatic, `0` none, `N` (64-255) | no |
| `DK_SM90_REG_CAPS` | `1` | sm_90: register caps on the forward kernels (needs nvcc 12.4+) | no |
| `DK_SM90_CAPBOX_COOP` | `0` | sm_90: cooperative capsule-box contact VJP pass | no |
| `DK_SM90_ADJ_CAP` | `1` | sm_90: G1-engine adjoint cap for terrain builds: `1`, `0`, `N` (64-255) | no |
| `DK_SM90_FOREST_ADJ_CAP` | `0` | sm_90: apply the terrain adjoint cap to forest terrain builds | no |
| `DK_FOREST_POST_CAP` | target default | forest forward `k_post` register cap: unset, `0`, `1` | no |
| `DK_FOREST_HESS_UNROLL` | unset (unrolled except sm_90) | forest Hessian mask loop: `1` unrolled everywhere, `0` loops everywhere | no |
| `DK_FOREST_CHOL_REG` | `0` | sm_90: `chol_solve_reg` for forest models with nv <= 32 | rounding |
| `DK_FOREST_CHOL_WIDE` | `0` | sm_90: `chol_solve_bcast_wide` for forest models with 36-48 dofs | rounding |
| `DK_FOREST_CCD_COOP` | `0` | sm_90: cooperative box-box CCD VJP pass | rounding (gradients) |
| `DK_FOREST_FREE_GYRO` | `0` | sm_90: gyroscopic implicit block for standalone free bodies | yes (those bodies) |
| `DK_NV128_MCHOL` | `loop` | M factor beyond 64 dofs: `loop`, `straight`, `dense` | rounding |
| `DK_NV128_HCHOL` | `auto` | Newton Hessian factor beyond 64 dofs: `auto`, `sparse`, `dense` | rounding |
| `DK_NV128_REUSE` | `0` | `1`: reuse the Hessian factor when no row changed zone (beyond 64 dofs) | rounding |
| `DK_NV128_DCHOL` | `blk` | dense factor form beyond 64 dofs: `blk`, `crout` | no |
| `DK_NV128_DCHOL_W` | `8` (<= 72 dofs), else `12` | block width (1-32) of the `blk` factor | no |
| `DK_NV128_PRE_WPSM`, `DK_NV128_POST_WPSM` | `0` | cap on resident worlds per SM of `k_pre` / `k_post` beyond 64 dofs | no |

### Testing and tooling

| variable | default | effect | changes results? |
|---|---|---|---|
| `DK_TEST_VIRTUAL_ARCH` | unset | testing: every unit from this virtual architecture's PTX | as a front-end change |
| `DK_TEST_FORWARD_VIRTUAL_ARCH` | unset | testing: the forward unit from this virtual architecture's PTX | as a front-end change |
| `DK_DIGESTS` | unset | extra manifest of forward digests to compare with | no |
| `DK_SPLIT_ONLY` | unset | timing binary: run only `k_pre` (`1`) or `k_post` (`2`) | timing only |
| `DK_TAPE_IN` | unset | internal: the adjoint binary runs on another binary's tape | n/a |
| `CUOBJDUMP` | next to nvcc, then `PATH`, then conda | tool for SASS digests | no |
| `PTXAS` | pip `nvidia.cuda_nvcc`, then next to nvcc | ptxas for targets the local nvcc cannot compile for (SASS digests only) | no |
| `ROBOT_DROP_BOX` | unset | robot comparison script: drop box-geom pairs | yes |

Benchmark and timing kits document their own `*_BENCH_*` variables; none changes results.

## Build options

`init()` options are arguments, documented in `docs/api.md`. Only `actuator=` (alias `motor_limits=`) selects
physics: `None` leaves the generated source unchanged; torque-speed limits change results where a force reaches a
curve; `"bam"` changes the step signature (state `act`) and every result. For velocity servos with a force range,
`clamp_kv=1` is the exact derivative and the default 0 is MuJoCo Warp's surrogate.

## Compile-time switches

| macro | default | effect | changes results? |
|---|---|---|---|
| `DK_ADJ_STAGE` | `3` | adjoint coverage: 1 smooth dynamics; 2 + limits and world-geom contacts; 3 + body-body contacts (`jax_ffi` passes `$DK_ADJ_STAGE`) | yes (gradients) |
| `DK_ADJ_CLAMP_KV` | `1` | clamped actuators in the adjoint (the FFI's `clamp_kv`) | yes (gradients) |
| `DK_ADJ_NORMAL`, `DK_ADJ_NORMAL_EPS` | `3`, `1e-3f` | self-contact normal derivative (the FFI's `normal`) and its eps | yes (gradients) |
| `DK_ADJ_PARALLEL`, `DK_ADJ_PARALLEL_EPS` | `DK_ADJ_NORMAL != 0`, `1.2e-3f` | hold near-parallel capsule-capsule parameters constant | yes (near-parallel states) |
| `DK_ADJ_WORLDS_PER_SM` | `-1` | adjoint worlds per SM: -1 GPU's choice, 0 max, N cap | no |
| `DK_ADJ_DEBUG` | undefined | write the adjoint's per-substep intermediates | no |
| `DK_CAPBOX_COOP` | `1` (G1 engine on sm_90: `0`) | cooperative capsule-box / sphere-box contact VJP pass | no |
| `DK_CCD_COOP` | `1` (sm_90: `0`) | forest: cooperative box-box CCD VJP pass | rounding (gradients) |
| `DK_FREE_GYRO` | `1` (sm_90: `0`) | forest: gyroscopic implicit block for standalone free bodies | yes (those bodies) |
| `CY_REC` | `1` | CCD contact VJP replays only the recorded final features | no |
| `DK_SPLIT` | `1` | forward: split `k_pre` / `k_post` (1) or fused `k_substep` (0) | no |
| `DK_SPLIT_ORDER`, `SPLIT_NBUCKET`, `SPLIT_ROW_SHIFT` | `1`, `64`, `1` | `k_post` dispatch by row count | no |
| `SPLIT_PRE_MINB`, `SPLIT_POST_MINB`, `SPLIT_POST_MAXNREG` | `12`, `12`, undefined | forward register caps | no at the defaults and 16 / 128 |
| `SPLIT_NCON`, `SPLIT_NEFC`, `DK_SPLIT_ALL_BIG` | `min(MAXCON, 32)`, `min(NEFC, 128)`, undefined | `k_post` shared-memory capacity; debug: all worlds in global memory | no |
| `DK_UNIFORM`, `DK_JTF_GATHER`, `DK_JV_VEC`, `DK_HESS_DOFPAR`, `DK_GRAD_COST` | `1` | optimization toggles | no |
| `NEWTON_REUSE` | on for 33-35 dofs | reuse the Cholesky factor when no row changed zone | rounding |
| `CHOL_BCAST`, `CHOL_COMPACT`, `CHOL_INLINE` | model / architecture dependent | Cholesky variant and inlining | rounding |
| `CHOL_SMALL`, `CHOL_TREES`, `CHOL_COMPS`, `CB` | model dependent | forest Cholesky variants (small, per tree, per component) | rounding |
| `DK_INL_GRAD`, `DK_INL_MATVEC`, `DK_INL_COLLIDE`, `DK_INL_ROWSJV`, `DK_INL_COST` | see source | helper inlining | can (FMA contraction) |
| `DKC` | `__device__` | memory space of the model arrays | performance experiment |
| `BOX_BOX_CCD` | `1` | box-box contacts: native CCD (1) or separating-axis primitive (0) | yes |
| `CV_DEFER`, `CVQ_PER_WORLD`, `DEFER_BOX` | model dependent, `64`, `$DK_DEFER_BOX` | forest deferred-pair collision kernel and its queue | no (`DEFER_BOX`: yes) |
| `CV_EXHAUSTIVE`, `CV_NO_CCD` | `1`, undefined | mesh support by exhaustive search; timing: skip CCD | can; `CV_NO_CCD`: yes |
| `TERRAIN_SORT` | `1` (terrain builds) | sort terrain contacts by (pair, element, slot) | yes (contact order) |
| `DK_PROFILE`, `DK_TAPE_VERIFY`, `KIN_WPB` | undefined, undefined, `4` | phase timers; debug Hessian check; kinematics worlds per block | timing / debug / no |

Generated from the model, not meant to be overridden: `MAXCON` (`init(max_contacts=)`), `NEFC` (192,
`init(max_rows=)`), `KDOF` (largest contact support: up to 64 on the forest engine, beyond 32 experimental), model
sizes (`NQ`, `NV`, `NU`, `NB`, `NPAIR`, ...), the tape layout, and feature macros written only into builds that use
the feature (equality rows, foot offsets, sand).

# Jetson Orin (sm_87) build notes (experimental)

Notes for running the engine on NVIDIA Jetson Orin (aarch64, integrated Ampere GPU, compute capability 8.7, JetPack
6.x). The device code, the JAX interface and the smoke test were checked on x86_64 GPUs (V100, RTX 3090) and by
compiling for `sm_87`; the engine has not yet been run on an Orin device.

## Requirements on the device

| | requirement | notes |
|---|---|---|
| CUDA toolkit | nvcc with `sm_87` (CUDA 11.4 or newer; JetPack 6 ships 12.2 or 12.6) | `/usr/local/cuda/bin/nvcc`, usually not on `PATH` |
| host compiler | the g++ nvcc uses (Ubuntu 22.04's g++ 11) | |
| Python | 3.10 (JetPack 6's system Python) or 3.12 | run the scripts with `PYTHONPATH=src` |
| JAX | a jaxlib + CUDA plugin built for `sm_87`, jax 0.5.0 or newer (`jax.ffi`), or jax 0.4.38 | see below |
| MuJoCo | `mujoco==3.17.0` | PyPI has aarch64 wheels |

The engine's CUDA library is generated from the model and built by nvcc on first use. The target comes from JAX's
device (`sm_87`); `$DK_ARCH` overrides it. The library is cached in `$DK_CACHE_DIR` (default `~/.cache/nablaengine`)
under a key that includes the nvcc version, so build on the device.

## sm_87 build defaults

`sm_87` gets the native build (`-arch=sm_87` for every unit) with no register caps, as `sm_86` and `sm_70`. Shared
memory is sized at run time from the device's attributes. Compiled with nvcc 12.4 for the forest engine's G1 library,
every kernel has the same register count as on `sm_86` (within one register) and none spills.

The engine runs one 32-thread block per world, and an SM holds at most 16 resident blocks: 16 worlds per SM. AGX Orin
64 GB has 16 SMs; Orin NX and Orin Nano 8 GB have 8.

## JAX on JetPack 6

- JAX's PyPI CUDA 12 plugin has no `sm_87` code, so its operations fail on Orin with
  `cudaErrorNoKernelImageForDevice` even though `jax.devices()` lists the GPU.
- Use a JAX built for `sm_87`: the jetson-containers `jax` package (jax 0.6.2 for L4T 36), the Jetson AI Lab pip
  index, or a source build (`python build/build.py build --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt
  --cuda_version=12.6.0 --cudnn_version=9.3.0 --cuda_compute_capabilities=sm_87`).
- Do not mix the pip `nvidia-*-cu12` wheels with JetPack's CUDA libraries.
- The engine passes its smoke test with jax 0.6.2 on Python 3.10 and 3.12 and with jax 0.12.0.1 on Python 3.12
  (x86_64). With 0.6.2, XLA may need `XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda`.

## JetPack 5

JetPack 5 has CUDA 11.4, cuDNN 8.6 and Python 3.8. The engine side works there in principle:

- jax 0.4.38 has no `jax.ffi`; `jax_ffi.init()` exposes `jax.extend.ffi` as `jax.ffi` automatically.
- nvcc 11.4 builds every unit for `sm_87`. On an Ubuntu 22.04 host, nvcc 11.4 with g++ 11 fails in glibc's headers;
  use g++ 9 (`NVCC_PREPEND_FLAGS="-ccbin <g++ 9>"`).

The constraint is JAX: jaxlib 0.4.38 needs CUDA 12.1+ and cuDNN 9.1+ libraries and Python 3.10+ (as do
`mujoco==3.17.0` and `numpy>=2.2`). A JetPack 5 device therefore needs CUDA 12 libraries for Tegra and a newer Python
next to the system ones (e.g. in a container). JetPack 6 avoids this.

## Build and smoke test

```bash
export NVCC=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH
export DK_BUILD_JOBS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false
PYTHONPATH=src python3 scripts/orin/smoke.py [--engine forest|tree] [--json out.json]
PYTHONPATH=src python3 scripts/orin/build_sm87.py [--arch sm_87] [--compare sm_86] [--cache DIR]   # no GPU needed
```

`scripts/orin/smoke.py` builds the G1 library (forest engine by default), runs 64 worlds from stored inputs (one control step, a
15-step rollout, a one-step VJP) and compares them with a reference in `scripts/orin/data/smoke_forest_g1.npz`. It
passes at max abs error 1e-4 on qpos and 1e-2 on qvel after one step and 1e-2 relative norm on the VJP; the rollout is
only reported. It also checks that repeated calls are bitwise identical and that no world overflows. On an RTX 3090
with nvcc 12.6 every output was bitwise equal to the reference.

`scripts/orin/build_sm87.py` compiles the library for a target and prints registers and spills per kernel, optionally
next to another architecture.

Notes:

- The build compiles 13 nvcc units. The CPU and GPU share memory, so keep `DK_BUILD_JOBS` low (4). The build took
  about 40 s on an x86 workstation.
- The scripts use JAX's default device and no GPU index; they run unchanged on any CUDA GPU.
- Set the power mode and clocks (`nvpmodel`, `jetson_clocks`) before timing, and record them with the results.
- Not verified yet: anything on an Orin device (JAX on `sm_87`, the build with JetPack's nvcc, the Tegra driver,
  results, timing, unified-memory behaviour) and JetPack 7.2 / CUDA 13.

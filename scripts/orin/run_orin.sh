#!/usr/bin/env bash
# Jetson Orin feasibility run (docs/orin.md): environment check, engine build for the GPU, smoke test.
#
#   bash scripts/orin/run_orin.sh [python]        # from the repository root; python defaults to python3
#
# Writes orin_device/{env.txt,build.txt,smoke.json,smoke.txt} and stops at
# the first failing step. Environment it honours: NVCC (default /usr/local/cuda/bin/nvcc where it exists),
# DK_CACHE_DIR (the library cache, default ~/.cache/nablaengine), DK_BUILD_JOBS (parallel nvcc jobs; default 4 here:
# Orin's memory is shared with the GPU), XLA_FLAGS (passed through).
set -euo pipefail
PY=${1:-python3}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT=orin_device
mkdir -p "$OUT"
if [ -z "${NVCC:-}" ] && [ -x /usr/local/cuda/bin/nvcc ]; then export NVCC=/usr/local/cuda/bin/nvcc; fi
export PATH="$(dirname "${NVCC:-/usr/local/cuda/bin/nvcc}"):$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DK_FIXTURES="${DK_FIXTURES:-$ROOT/fixtures/mjlab}"
export DK_BUILD_JOBS="${DK_BUILD_JOBS:-4}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DK_ENGINE=forest

echo "== 1. environment" | tee "$OUT/env.txt"
{
  uname -a
  [ -f /etc/nv_tegra_release ] && head -1 /etc/nv_tegra_release
  echo "nvcc: ${NVCC:-<none set>}"; "${NVCC:-nvcc}" --version | tail -2
  "$PY" --version
  "$PY" - <<'EOF'
import importlib.metadata as md
for p in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda13-plugin", "jax-cuda13-pjrt", "mujoco",
          "numpy"):
    try:
        print(f"{p} {md.version(p)}")
    except md.PackageNotFoundError:
        pass
import jax, jax.numpy as jnp
d = jax.devices()[0]
print("device", d, d.platform, "compute capability", getattr(d, "compute_capability", None))
x = jnp.ones((256, 256))
print("jax matmul + solve on the GPU:", float((x @ x).sum()), float(jnp.linalg.solve(x + 256 * jnp.eye(256), x[:, 0]).sum()))
from nablaengine.jax_ffi import ensure_jax_ffi
ensure_jax_ffi()  # jax < 0.5: jax.extend.ffi
print("jax.ffi.include_dir:", jax.ffi.include_dir())
EOF
} 2>&1 | tee -a "$OUT/env.txt"

echo "== 2. engine library build (first use, forest engine, mjlab G1)" | tee "$OUT/build.txt"
"$PY" - 2>&1 <<'EOF' | tee -a "$OUT/build.txt"
import time
from nablaengine import jax_ffi
t = time.time()
info = jax_ffi.init()
print({k: info[k] for k in ("engine", "arch", "arch_flags", "built", "path") if k in info}, f"{time.time() - t:.1f} s")
EOF

echo "== 3. smoke test (against the stored reference)"
"$PY" scripts/orin/smoke.py --json "$OUT/smoke.json" 2>&1 | tee "$OUT/smoke.txt"

echo "done: $OUT"

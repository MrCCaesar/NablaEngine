#!/usr/bin/env bash
# One-command verification of the G1 engine on the public corpora (benchmarks/data/mf_s0, mf_s1):
#   scripts/verify.sh [--engine tree|forest] [--quick] [OUTDIR]          (default OUTDIR: runs/verify)
#
# --engine forest: the mjlab G1 through the forest engine ($DK_ENGINE=forest for every check: nablaengine.engines; the
# tree engine's default COM-offset body) against the same MuJoCo Warp / truth gates: env, the control-step checks
# (engine_step_check.py --engine forest, without --tape), the gradients, the robots (forest builds of the G1-class
# robots), capability. The tree engine's binaries and terrain builds (ffi_bitwise / ffi, kinematics, terrain_*) are
# left out. Default: the forest engine ($DK_ENGINE unset, auto or forest; --engine tree or DK_ENGINE=tree: the tree engine).
#
# --quick (a few minutes on one GPU):
#   env              scripts/env_check.py (versions, nvcc, the fixture model, a test kernel, JAX's devices)
#   step_mf_s0       one control step of 256 worlds of mf_s0, the truth file's sampled states first, against the
#                    MuJoCo Warp reference (native mujoco-warp 3.17.0) and the MuJoCo C float64 truth, and the
#                    outputs' digests (scripts/engine_step_check.py --select truth --gate)
#   gradient_G0      the G0 group of mf_s0: the JAX interface's VJPs against the finite-difference truth
#                    (scripts/gradient_check.py --groups G0)
#   ffi_bitwise      the JAX interface bitwise against the check binaries for the interface default (clamp_kv 0,
#                    normal 3) at 50 iterations, and its forward bitwise against this tree's engine_step_check.py
#                    (scripts/jax_ffi_check.py --sections bitwise)
#   robots_apollo    the Apptronik Apollo of fixtures/robots (34 bodies, box-box feet): its forward against the
#                    MuJoCo Warp reference and its gradients on 64 states against float64 MuJoCo C
#                    (scripts/robots/regress.py --robots apollo --grad 64)
#   capability       the model checks (scripts/capability_check.py, CPU)
# Without --quick (about 25 minutes on one V100, 15 with the libraries of the other robots built): env; per corpus
# mf_s0 and mf_s1 the control step on all 4096 states with the forward-tape checks, on the truth's states, and the
# contact sets of one substep; every gradient group of both corpora; every section of scripts/jax_ffi_check.py and
# scripts/kinematics_check.py; a terrain build per heightfield mode on the test terrain against its MuJoCo Warp
# reference (scripts/terrain_check.py check --gate: the box regions; fixtures/mjlab/mjw317_terrain_feet.npz) and the native
# heightfield contacts' replay bitwise against the forward (terrain_check.py replay, also from a -fmad=false build); the nineteen
# robots of fixtures/robots against their MuJoCo Warp references and, on 64 states each, float64 MuJoCo C
# (scripts/robots/regress.py --grad 64; one library build per robot); capability.
#
# Each check prints one line, PASS or FAIL, with its log (and JSON) in OUTDIR; the exit status is 1 when any check
# failed, else 0. The pass criteria are the scripts' own (their GATES / THRESHOLDS, documented in each script).
# Needs: the packages of requirements.txt, nvcc of CUDA 12 or newer ($NVCC, CUDA_HOME or on PATH), one GPU (the first
# visible one), PYTHONPATH=src (the default here) or an installed nablaengine; $PYTHON names the interpreter.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
QUICK=0
ENGINE=${DK_ENGINE:-forest}
if [ "${1:-}" = "--engine" ]; then ENGINE=${2:-}; shift 2; fi
case "$ENGINE" in tree|g1) ENGINE=tree ;; forest|auto|"") ENGINE=forest ;; *) echo "--engine: tree or forest" >&2; exit 2 ;; esac
export DK_ENGINE=$ENGINE
if [ "${1:-}" = "--quick" ]; then QUICK=1; shift; fi
O=${1:-runs/verify}
mkdir -p "$O" || exit 1
export PYTHONPATH=${PYTHONPATH:-src}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
PY=${PYTHON:-python}
FAILED=()

check() {  # name command...: run it, log to OUTDIR/name.log, print PASS / FAIL
  local name=$1 t0=$SECONDS
  shift
  rm -f "$O/$name.json"  # a result left by an earlier run must not stand in for this one
  if "$@" > "$O/$name.log" 2>&1; then
    echo "PASS  $name ($((SECONDS - t0)) s)"
  else
    echo "FAIL  $name ($((SECONDS - t0)) s; $O/$name.log)"
    FAILED+=("$name")
  fi
}

check env "$PY" scripts/env_check.py
if [ "$ENGINE" = forest ]; then
  if [ "$QUICK" = 1 ]; then
    check step_mf_s0 "$PY" scripts/engine_step_check.py --engine forest --corpus mf_s0 --substeps 4 --nworld 256 \
      --check 256 --select truth --reps 0 --gate --out "$O/step_mf_s0.json"
    check gradient_G0 "$PY" scripts/gradient_check.py --engine forest --corpora mf_s0 --groups G0 \
      --out "$O/gradient_G0.json"
    check robots_apollo "$PY" scripts/robots/regress.py --robots apollo --grad 64 --python "$PY" \
      --out "$O/robots_apollo.json"
  else
    for c in mf_s0 mf_s1; do
      check "step_$c" "$PY" scripts/engine_step_check.py --engine forest --corpus "$c" --substeps 4 --check 4096 \
        --reps 0 --gate --out "$O/step_$c.json"
      check "step_truth_$c" "$PY" scripts/engine_step_check.py --engine forest --corpus "$c" --substeps 4 \
        --nworld 256 --check 256 --select truth --reps 0 --gate --out "$O/step_truth_$c.json"
      check "contacts_$c" "$PY" scripts/engine_step_check.py --engine forest --corpus "$c" --substeps 1 --check 4096 \
        --reps 0 --gate --out "$O/contacts_$c.json"
    done
    check gradient "$PY" scripts/gradient_check.py --engine forest --corpora mf_s0,mf_s1 --out "$O/gradient.json"
    check robots "$PY" scripts/robots/regress.py --grad 64 --python "$PY" --out "$O/robots.json"
  fi
elif [ "$QUICK" = 1 ]; then
  check step_mf_s0 "$PY" scripts/engine_step_check.py --corpus mf_s0 --substeps 4 --nworld 256 --check 256 \
    --select truth --reps 0 --gate --out "$O/step_mf_s0.json"
  check gradient_G0 "$PY" scripts/gradient_check.py --corpora mf_s0 --groups G0 --out "$O/gradient_G0.json"
  check ffi_bitwise "$PY" scripts/jax_ffi_check.py --sections bitwise --corpora mf_s0 --variants 0,3 \
    --bitwise-iterations 50 --out "$O/ffi_bitwise.json"
  check robots_apollo "$PY" scripts/robots/regress.py --robots apollo --grad 64 --python "$PY" \
    --out "$O/robots_apollo.json"
else
  for c in mf_s0 mf_s1; do
    check "step_$c" "$PY" scripts/engine_step_check.py --corpus "$c" --substeps 4 --check 4096 --reps 0 --tape \
      --gate --out "$O/step_$c.json"
    check "step_truth_$c" "$PY" scripts/engine_step_check.py --corpus "$c" --substeps 4 --nworld 256 --check 256 \
      --select truth --reps 0 --gate --out "$O/step_truth_$c.json"
    check "contacts_$c" "$PY" scripts/engine_step_check.py --corpus "$c" --substeps 1 --check 4096 --reps 0 --gate \
      --out "$O/contacts_$c.json"
  done
  check gradient "$PY" scripts/gradient_check.py --corpora mf_s0,mf_s1 --out "$O/gradient.json"
  check ffi "$PY" scripts/jax_ffi_check.py --corpora mf_s0,mf_s1 --out "$O/ffi.json"
  check kinematics "$PY" scripts/kinematics_check.py --corpora mf_s0,mf_s1 --out "$O/kinematics.json"
  for mode in analytic native; do
    check "terrain_$mode" "$PY" scripts/terrain_check.py check --ref fixtures/mjlab/mjw317_terrain_feet.npz \
      --scene fixtures/mjlab/g1_terrain_feet.mjb --api jax --mode "$mode" --gate --out "$O/terrain_$mode.json"
  done
  check terrain_replay "$PY" scripts/terrain_check.py replay --scene fixtures/mjlab/g1_terrain_feet.mjb \
    --out "$O/terrain_replay.json"
  check robots "$PY" scripts/robots/regress.py --grad 64 --python "$PY" --out "$O/robots.json"
fi
check capability env JAX_PLATFORMS=cpu "$PY" scripts/capability_check.py --out "$O/capability.json"

if [ ${#FAILED[@]} -eq 0 ]; then
  echo "verify: PASS"
  exit 0
fi
echo "verify: FAIL (${FAILED[*]})"
exit 1

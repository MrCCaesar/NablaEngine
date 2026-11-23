"""nvcc and target architecture for the hand-written CUDA kernels."""

import functools
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

FRONTEND_VAR = "DK_SM120_FRONTEND"
LEGACY_FRONTEND_VAR = "DK_SM120_FORWARD_FRONTEND"
FRONTENDS = ("n7", "split", "native")
TEST_FORWARD_VAR = "DK_TEST_FORWARD_VIRTUAL_ARCH"
TEST_VAR = "DK_TEST_VIRTUAL_ARCH"
N7_VIRTUAL = {"sm_120": "compute_90", "sm_103": "compute_90"}
SM103_FRONTEND_VAR = "DK_SM103_FRONTEND"
SM103_FRONTENDS = ("adjoint", "n7", "native")
SM90_CAPS = ("SPLIT_PRE_MINB=16", "SPLIT_POST_MAXNREG=128")
REG_CAPS = {"sm_90": ("DK_SM90_REG_CAPS", SM90_CAPS, False),
            "sm_90a": ("DK_SM90_REG_CAPS", SM90_CAPS, False),
            "sm_120": ("DK_SM120_PRE_CAP", ("SPLIT_PRE_MINB=16",), True),
            "sm_103": ("DK_SM103_REG_CAPS", SM90_CAPS, False)}
ADJ_CAPS = {"sm_103": ("DK_SM103_REG_CAPS", ("ADJ_MAXNREG=120",))}
FOREST_ADJ_CAPS = {"sm_103": ("DK_SM103_REG_CAPS", ("ADJ_MAXNREG=128",))}
FOREST_POST_VAR = "DK_FOREST_POST_CAP"
FOREST_ADJ_VAR = "DK_SM103_FOREST_ADJ_CAP"
CAPBOX_COOP_OFF = ("sm_90", "sm_90a")
CAPBOX_COOP_VAR = "DK_SM90_CAPBOX_COOP"
TREE_ADJ_CAPS = {"sm_90": ("ADJ_MAXNREG=168",), "sm_90a": ("ADJ_MAXNREG=168",)}
TREE_ADJ_VAR = "DK_SM90_ADJ_CAP"
FOREST_TER_CAP_VAR = "DK_SM90_FOREST_ADJ_CAP"
FOREST_HOLD_TARGETS = ("sm_90", "sm_90a")
FOREST_HESS_VAR = "DK_FOREST_HESS_UNROLL"
FOREST_CHOL_VAR = "DK_FOREST_CHOL_REG"
FOREST_WIDE_VAR = "DK_FOREST_CHOL_WIDE"
FOREST_CCD_COOP_VAR = "DK_FOREST_CCD_COOP"
FOREST_FREE_GYRO_VAR = "DK_FOREST_FREE_GYRO"
CAP_BOUNDS = {"SPLIT_PRE_MINB": ("SPLIT_PRE_MINB",), "SPLIT_POST_MAXNREG": ("SPLIT_POST_MAXNREG", "SPLIT_POST_MINB"),
              "ADJ_MAXNREG": ("ADJ_MAXNREG",)}
MAXNREG_NVCC = (12, 4)


def nvcc():
    """The nvcc to build with: $NVCC, nvcc on PATH, $CUDA_HOME/bin/nvcc, a pip nvidia.cuda_nvcc package's nvcc, then
    /usr/local/cuda/bin/nvcc (the first that exists); FileNotFoundError if none does."""
    cands = [os.environ.get("NVCC"), shutil.which("nvcc")]
    if os.environ.get("CUDA_HOME"):
        cands.append(str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc"))
    try:
        import nvidia.cuda_nvcc
        cands.append(str(Path(list(nvidia.cuda_nvcc.__path__)[0]) / "bin" / "nvcc"))
    except ImportError:
        pass
    cands.append("/usr/local/cuda/bin/nvcc")
    for c in cands:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError(
        "no nvcc found: set NVCC to a CUDA 12.x nvcc (or CUDA_HOME, or put nvcc on PATH). The pip wheel "
        "nvidia-cuda-nvcc-cu12 has no nvcc (only ptxas); install a toolkit, e.g. conda's cuda-nvcc "
        "(scripts/setup_env.sh, README.md)")


def visible_gpu():
    """The first visible GPU as nvidia-smi names it (index or UUID): the first entry of $CUDA_VISIBLE_DEVICES, else
    None (nvidia-smi's GPU 0). nvidia-smi itself ignores CUDA_VISIBLE_DEVICES and numbers the GPUs in PCI order, CUDA
    fastest first unless CUDA_DEVICE_ORDER=PCI_BUS_ID: the same GPU on a machine with one GPU type."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return v if v and not v.startswith("-") else None


def device_arch():
    """sm_XY of the first visible GPU (nvidia-smi), None without nvidia-smi or a GPU."""
    q = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    for cmd in ([*q, "-i", visible_gpu()] if visible_gpu() else None, q):
        if cmd is None:
            continue
        try:
            out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 universal_newlines=True).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            continue
        if out:
            return "sm_" + out[0].replace(".", "")
    return None


def arch():
    """$DK_ARCH, else the first visible GPU's architecture (device_arch). A $DK_ARCH that is not the GPU's warns (once
    per process): code built for an older target runs that target's code paths (CHOL_COMPACT follows __CUDA_ARCH__)
    from its SASS within the same major version, else from the driver's compilation of its PTX, so the results are not
    bitwise those of a build for the GPU; code built for a newer target does not load. No warning without a GPU (a
    build for another machine)."""
    if os.environ.get("DK_ARCH"):
        a = os.environ["DK_ARCH"]
        dev = device_arch()
        if dev is not None and dev != a:
            warnings.warn(f"DK_ARCH={a} but the GPU is {dev}: the CUDA code is built for {a} (unset DK_ARCH to build "
                          f"for the GPU)", RuntimeWarning, stacklevel=2)
        return a
    a = device_arch()
    if a is None:
        raise RuntimeError("no GPU found by nvidia-smi: set DK_ARCH (e.g. sm_70, sm_86, sm_87, sm_90, sm_120) to build "
                           "without one")
    return a


def frontend():
    """$DK_SM120_FRONTEND: 'n7' (the default), 'split' or 'native'. The deprecated $DK_SM120_FORWARD_FRONTEND (n7 / native)
    is read where $DK_SM120_FRONTEND is unset, with a DeprecationWarning (n7 -> n7, native -> native)."""
    v = (os.environ.get(FRONTEND_VAR) or "").strip().lower()
    old = (os.environ.get(LEGACY_FRONTEND_VAR) or "").strip().lower()
    if old:
        _warn_legacy(old, bool(v))
        if not v:
            if old not in ("n7", "native"):
                raise ValueError(f"${LEGACY_FRONTEND_VAR}={old!r}: must be n7 or native (deprecated: use ${FRONTEND_VAR})")
            v = old
    v = v or "n7"
    if v not in FRONTENDS:
        raise ValueError(f"${FRONTEND_VAR}={v!r}: must be one of {', '.join(FRONTENDS)}")
    return v


def sm103_frontend():
    """$DK_SM103_FRONTEND: 'adjoint' (the default), 'n7' or 'native' (module docstring)."""
    v = (os.environ.get(SM103_FRONTEND_VAR) or "adjoint").strip().lower()
    if v not in SM103_FRONTENDS:
        raise ValueError(f"${SM103_FRONTEND_VAR}={v!r}: must be one of {', '.join(SM103_FRONTENDS)}")
    return v


def frontend_of(target):
    """The front-end switch value of a build target with NVVM 7 units (N7_VIRTUAL): $DK_SM103_FRONTEND on sm_103,
    $DK_SM120_FRONTEND (frontend()) on sm_120."""
    return sm103_frontend() if target == "sm_103" else frontend()


@functools.lru_cache(maxsize=None)
def _warn_legacy(old, shadowed):
    warnings.warn(f"${LEGACY_FRONTEND_VAR} is deprecated: use ${FRONTEND_VAR} (n7 / split / native)"
                  + (f"; ${FRONTEND_VAR} is set and wins" if shadowed else f"; {old} is taken as ${FRONTEND_VAR}={old}"),
                  DeprecationWarning, stacklevel=4)


def nvvm_version(virt, nvcc_path=None):
    """The front end nvcc uses for a virtual architecture (compute_XY), from the header of the PTX it makes for a tiny
    kernel ('NVVM 7.0.1', 'NVVM 20.0.0'), or None if nvcc cannot make it."""
    return _nvvm_version(virt, nvcc_path or nvcc())


@functools.lru_cache(maxsize=None)
def _nvvm_version(virt, nvcc_path):
    with tempfile.TemporaryDirectory(prefix="dk_nvvm_") as d:
        src, ptx = Path(d) / "k.cu", Path(d) / "k.ptx"
        src.write_text("__global__ void k(float* a) { a[threadIdx.x] *= 2.f; }\n")
        r = subprocess.run([nvcc_path, f"-arch={virt}", "-ptx", "-o", str(ptx), str(src)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode or not ptx.exists():
            return None
        m = re.search(r"Based on NVVM ([\d.]+)", ptx.read_text())
        return f"NVVM {m.group(1)}" if m else None


@functools.lru_cache(maxsize=None)
def _n7_applies(target, nvcc_path):
    virt = N7_VIRTUAL[target]
    native, n7 = nvvm_version("compute_" + target[3:], nvcc_path), nvvm_version(virt, nvcc_path)
    if n7 and n7.startswith("NVVM 7.") and not (native or "").startswith("NVVM 7."):
        return virt
    var = SM103_FRONTEND_VAR if target == "sm_103" else FRONTEND_VAR
    print(f"nablaengine.cuda: {var}={frontend_of(target)} not applied on {target} (front ends: compute_{target[3:]} "
          f"{native}, {virt} {n7}): every unit keeps the native build", file=sys.stderr)
    return None


@functools.lru_cache(maxsize=None)
def _warn_test(var, virt):
    warnings.warn(f"${var}={virt}: {'the forward / kinematics unit is' if var == TEST_FORWARD_VAR else 'every unit is'} "
                  f"built from {virt} PTX (a testing switch, not the default build)", RuntimeWarning, stacklevel=4)


def virtual_arch(unit=None, target=None, nvcc_path=None):
    """The virtual architecture whose PTX a compilation unit is built from, or None for the native build
    (-arch=<target>). unit 'forward': the forward / kinematics unit; anything else: the adjoint units, the FFI unit and
    test programs. sm_120 (see the module docstring): compute_90 for every unit with $DK_SM120_FRONTEND n7 (the
    default), for the forward unit only with split, none with native (n7 / split only where the PTX headers confirm
    it). sm_103: compute_90 for every unit but the forward one with $DK_SM103_FRONTEND adjoint (the default), every
    unit with n7, none with native. The testing switches take precedence on any target: $DK_TEST_FORWARD_VIRTUAL_ARCH for the forward unit,
    $DK_TEST_VIRTUAL_ARCH for every unit."""
    target = target or arch()
    fwd = unit == "forward"
    for var, applies in ((TEST_FORWARD_VAR, fwd), (TEST_VAR, True)):
        test = (os.environ.get(var) or "").strip()
        if test and applies:
            _warn_test(var, test)
            return test
    if target not in N7_VIRTUAL:
        return None
    mode = frontend_of(target)
    if mode == "native" or (mode == "split" and not fwd) or (mode == "adjoint" and fwd):
        return None
    return _n7_applies(target, nvcc_path or nvcc())


def forward_virtual_arch(target=None, nvcc_path=None):
    """virtual_arch of the forward / kinematics unit (compute_90 on sm_120 with n7 or split)."""
    return virtual_arch("forward", target, nvcc_path)


def arch_flags(unit=None, target=None, nvcc_path=None):
    """nvcc's architecture flags for a compilation unit (unit 'forward': the forward / kinematics kernels; anything
    else: the adjoint units, the FFI unit, test programs): ['-arch=<virtual>', '-code=<target>'] where virtual_arch gives
    one (sm_120: every unit by default, see the module docstring), else ['-arch=<target>']."""
    target = target or arch()
    virt = virtual_arch(unit, target, nvcc_path)
    return [f"-arch={virt}", f"-code={target}"] if virt else [f"-arch={target}"]


def forward_tag(target=None, nvcc_path=None):
    """'' where the forward unit has the native flags, else 'fwd-<virtual architecture>' (names forward references:
    the forward's bits depend only on the forward unit's flags, so sm_120's n7 and split share 'fwd-compute_90')."""
    virt = forward_virtual_arch(target, nvcc_path)
    return f"fwd-{virt}" if virt else ""


def reg_caps(target=None):
    """The register-cap switch of the build target (REG_CAPS: $DK_SM90_REG_CAPS on sm_90 / sm_90a, $DK_SM120_PRE_CAP on
    sm_120, $DK_SM103_REG_CAPS on sm_103): True for 1 (the default), False for 0; None for a target without caps."""
    target = target or arch()
    if target not in REG_CAPS:
        return None
    var = REG_CAPS[target][0]
    v = (os.environ.get(var) or "1").strip()
    if v not in ("0", "1"):
        raise ValueError(f"${var}={v!r}: must be 1 or 0")
    return v == "1"


def defined_macros(options):
    """The macro names an nvcc option list defines (-DX, -DX=Y, -D X, --define-macro X, --define-macro=X)."""
    out, it = set(), iter(str(a) for a in options)
    for a in it:
        if a in ("-D", "--define-macro"):
            a = "-D" + next(it, "")
        elif a.startswith("--define-macro="):
            a = "-D" + a.split("=", 1)[1]
        if a.startswith("-D") and len(a) > 2:
            out.add(a[2:].split("=", 1)[0])
    return out


def reg_caps_defines(options=(), caps=SM90_CAPS):
    """The caps (e.g. ('SPLIT_PRE_MINB=16', ...)) as -D options, less each cap whose kernel the caller's nvcc options
    already bound (defined_macros; CAP_BOUNDS)."""
    have = defined_macros(options)
    return [f"-D{d}" for d in caps if not have.intersection(CAP_BOUNDS[d.split("=", 1)[0]])]


@functools.lru_cache(maxsize=None)
def _maxnreg_ok(nvcc_path, var):
    r = subprocess.run([nvcc_path, "--version"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       universal_newlines=True)
    m = re.search(r"release (\d+)\.(\d+)", r.stdout)
    rel = (int(m.group(1)), int(m.group(2))) if m else None
    if rel and rel >= MAXNREG_NVCC:
        return True
    print(f"nablaengine.cuda: {var}=1 not applied (nvcc release {'.'.join(map(str, rel)) if rel else '?'}; "
          "__maxnreg__ needs 12.4 or newer): the forward unit keeps the launch bounds of the other targets",
          file=sys.stderr)
    return False


FOREST_POST_DROP = ("sm_90", "sm_90a")


def forest_post_cap(target=None):
    """$DK_FOREST_POST_CAP: True for 1 (the forest engine keeps k_post's __maxnreg__ cap), False for 0; unset: the
    target's default, False on sm_90 / sm_90a (ptxas's lane rematerialization, module docstring), True elsewhere
    (sm_103: bitwise the uncapped forward, B300 G1 4096 worlds k_post 1.260 -> 1.167 ms)."""
    v = (os.environ.get(FOREST_POST_VAR) or "").strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"${FOREST_POST_VAR}={v!r}: must be 0 or 1")
    if v == "":
        return (target or arch()) not in FOREST_POST_DROP
    return v == "1"


def unit_defines(unit=None, target=None, options=(), nvcc_path=None, forest=False):
    """The -D options a compilation unit gets for its build target besides the caller's nvcc options: unit 'forward'
    (the split forward kernels' unit; see the module docstring) gets the target's register caps (REG_CAPS: sm_90 /
    sm_90a and sm_103 ['-DSPLIT_PRE_MINB=16', '-DSPLIT_POST_MAXNREG=128'], sm_120 with the NVVM 7 forward
    ['-DSPLIT_PRE_MINB=16'])
    where its switch is 1 (the default), each unless `options` already bound that kernel; every other unit and target:
    none. forest: a forest-engine unit, which gets no SPLIT_POST_MAXNREG where forest_post_cap(target) is False (by
    default on sm_90 / sm_90a: ptxas's miscompile, module docstring)."""
    if unit != "forward":
        return []
    target = target or arch()
    if not reg_caps(target):
        return []
    var, caps, needs_n7 = REG_CAPS[target]
    if forest and not forest_post_cap(target):
        caps = tuple(c for c in caps if not c.startswith("SPLIT_POST_MAXNREG="))
    if needs_n7 and not forward_virtual_arch(target, nvcc_path):
        return []
    if any(c.startswith("SPLIT_POST_MAXNREG=") for c in caps) and not _maxnreg_ok(nvcc_path or nvcc(), var):
        return []
    return reg_caps_defines(options, caps)


def forest_hold_defines(target=None, text="", options=()):
    """The -D options of a forest-engine unit (its text) on a build target of FOREST_HOLD_TARGETS (sm_90 / sm_90a),
    decided by the target (not __CUDA_ARCH__: sm_120's and sm_103's units from compute_90 PTX see 900 too):
    ['-DDK_FOREST_HESS_UNROLL=0'] for a text with the forest hessian()'s unrolled contact-mask loops (DK_HESS_UNROLLED)
    unless $DK_FOREST_HESS_UNROLL is set (forest_engine.step then defines it in the text, every target); and
    ['-DDK_FOREST_SM90_SMALL=1'] (chol_solve_small for NV <= 32 as before 2026-11-16 instead of chol_solve_reg) for a
    text with CHOL_SMALL's choice unless $DK_FOREST_CHOL_REG is 1; ['-DCHOL_BCAST_WIDE=0'] (36 <= NV <= 48: the
    previous CHOL_SOLVE instead of chol_solve_bcast_wide, and with it no NEWTON_REUSE for those sizes) for a text with
    chol_solve_bcast_wide unless $DK_FOREST_CHOL_WIDE is 1; ['-DDK_CCD_COOP=0'] (native-CCD box-box contacts: their VJP
    passes on the contact's lane as before, forest_engine.ccd_coop) for a text with that pass unless $DK_FOREST_CCD_COOP
    is 1; ['-DDK_FREE_GYRO=0'] (standalone free bodies in implicitfast without MuJoCo 3.17's gyroscopic block, as
    before, forest_engine.free_gyro) for a text with that block unless $DK_FOREST_FREE_GYRO is 1. A caller's own option
    of the same macro wins. [] on every other target."""
    target = target or arch()
    if target not in FOREST_HOLD_TARGETS:
        return []
    out = []
    own = lambda name: any(o == f"-D{name}" or o.startswith(f"-D{name}=") for o in options)
    if "DK_HESS_UNROLLED" in text and not (os.environ.get(FOREST_HESS_VAR) or "").strip() \
            and not own("DK_FOREST_HESS_UNROLL"):
        out.append("-DDK_FOREST_HESS_UNROLL=0")
    v = (os.environ.get(FOREST_CHOL_VAR) or "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"${FOREST_CHOL_VAR}={v!r}: must be 0 or 1")
    if "DK_FOREST_SM90_SMALL" in text and v == "0" and not own("DK_FOREST_SM90_SMALL") and not own("CHOL_SMALL"):
        out.append("-DDK_FOREST_SM90_SMALL=1")
    w = (os.environ.get(FOREST_WIDE_VAR) or "0").strip()
    if w not in ("0", "1"):
        raise ValueError(f"${FOREST_WIDE_VAR}={w!r}: must be 0 or 1")
    nv = re.search(r"^#define NV (\d+)", text, re.M)
    if "chol_solve_bcast_wide" in text and w == "0" and not own("CHOL_BCAST_WIDE") and nv \
            and 36 <= int(nv.group(1)) <= 48:
        out.append("-DCHOL_BCAST_WIDE=0")
    c = (os.environ.get(FOREST_CCD_COOP_VAR) or "0").strip()
    if c not in ("0", "1"):
        raise ValueError(f"${FOREST_CCD_COOP_VAR}={c!r}: must be 0 or 1")
    if "DK_CCD_COOP" in text and c == "0" and not own("DK_CCD_COOP"):
        out.append("-DDK_CCD_COOP=0")
    g = (os.environ.get(FOREST_FREE_GYRO_VAR) or "0").strip()
    if g not in ("0", "1"):
        raise ValueError(f"${FOREST_FREE_GYRO_VAR}={g!r}: must be 0 or 1")
    if "DK_FREE_GYRO" in text and g == "0" and not own("DK_FREE_GYRO"):
        out.append("-DDK_FREE_GYRO=0")
    return out


def capbox_coop_defines(target=None, text="", options=(), forest=False):
    """['-DDK_CAPBOX_COOP=0'] for a tree-engine unit whose text has the cooperative box contact VJP pass (the macro
    DK_CAPBOX_COOP in text), or a forest-engine unit of a terrain build with it (cbter_dir_vjp in text: the terrain
    capsule-box contacts' pass), on a build target of CAPBOX_COOP_OFF (sm_90 / sm_90a) where $DK_SM90_CAPBOX_COOP is 0
    (the default), unless `options` already define DK_CAPBOX_COOP; else [] (forest units without terrain, other
    targets, texts without the pass, the switch at 1). Bitwise unchanged: both paths give the same gradients (module
    docstring)."""
    target = target or arch()
    if target not in CAPBOX_COOP_OFF or "DK_CAPBOX_COOP" not in text or (forest and "cbter_dir_vjp(" not in text):
        return []
    v = (os.environ.get(CAPBOX_COOP_VAR) or "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"${CAPBOX_COOP_VAR}={v!r}: must be 0 or 1")
    if v == "1" or any(o == "-DDK_CAPBOX_COOP" or o.startswith("-DDK_CAPBOX_COOP=") for o in options):
        return []
    return ["-DDK_CAPBOX_COOP=0"]


def tree_adj_cap():
    """$DK_SM90_ADJ_CAP: None for 1 (the default: TREE_ADJ_CAPS on terrain builds), 0 for no cap, N for ADJ_MAXNREG=N on
    every tree-engine unit that launches k_adjoint (64..255; experiments, e.g. a bitwise check of other models)."""
    v = (os.environ.get(TREE_ADJ_VAR) or "1").strip()
    if v == "1":
        return None
    if v == "0":
        return 0
    if v.isdigit() and 64 <= int(v) <= 255:
        return int(v)
    raise ValueError(f"${TREE_ADJ_VAR}={v!r}: must be 1, 0 or a register count 64..255")


def tree_adj_cap_defines(target=None, options=(), nvcc_path=None, forest=False, terrain=False):
    """The tree engine's k_adjoint register cap on sm_90 / sm_90a (TREE_ADJ_CAPS: ['-DADJ_MAXNREG=168']) for a unit that
    launches k_adjoint of a terrain build (terrain) where $DK_SM90_ADJ_CAP is 1 (the default); ['-DADJ_MAXNREG=N'] on
    every tree-engine such unit with N. Forest units (forest): the same for a terrain build's units (the tree engine's
    terrain VJP device code; since 2026-11-15) only where $DK_SM90_FOREST_ADJ_CAP is 1 (default 0: no cap until
    verified on sm_90), none without terrain. None with 0, for other targets, a caller's own
    -DADJ_MAXNREG, or an nvcc older than 12.4. The unit's text needs with_adj_bounds (g1_engine.adjoint). Register
    allocation only (bitwise unchanged; module docstring)."""
    target = target or arch()
    if target not in TREE_ADJ_CAPS or (forest and not terrain):
        return []
    if forest:
        v = (os.environ.get(FOREST_TER_CAP_VAR) or "0").strip()
        if v not in ("0", "1"):
            raise ValueError(f"${FOREST_TER_CAP_VAR}={v!r}: must be 0 or 1")
        if v == "0":
            return []
    n = tree_adj_cap()
    if n == 0 or (n is None and not terrain):
        return []
    caps = TREE_ADJ_CAPS[target] if n is None else (f"ADJ_MAXNREG={n}",)
    if not _maxnreg_ok(nvcc_path or nvcc(), TREE_ADJ_VAR):
        return []
    return reg_caps_defines(options, caps)


def forest_adj_cap():
    """$DK_SM103_FOREST_ADJ_CAP: None for 1 (the default: FOREST_ADJ_CAPS' value where forest_adj_light), 0 for no
    cap, N for a register cap of N on every forest model (64..255; e.g. for a timing sweep). Read only where FOREST_ADJ_CAPS has the build target and its register-cap
    switch is 1."""
    v = (os.environ.get(FOREST_ADJ_VAR) or "1").strip()
    if v == "1":
        return None
    if v == "0":
        return 0
    if v.isdigit() and 64 <= int(v) <= 255:
        return int(v)
    raise ValueError(f"${FOREST_ADJ_VAR}={v!r}: must be 1, 0 or a register count 64..255")


def forest_adj_light(m):
    """Whether the forest engine's sm_103 adjoint cap applies to model m by default (FOREST_ADJ_CAPS under
    $DK_SM103_FOREST_ADJ_CAP=1): every contact pair between planes, spheres and capsules (no box, mesh, cylinder,
    ellipsoid or height-field pair, whose contact VJPs and replays take k_adjoint to 236-255 registers) and the
    pyramidal cone. ptxas 12.9 (NVVM 7 adjoint on sm_103): such models' uncapped k_adjoint stays at about 120-150
    registers, so the cap at 128 costs a few spilled bytes for up to 16 worlds per SM (B300: G1 on the forest engine
    backward -12.5%); with box pairs it costs 1.3-2.4 KB of spills and was slower (B300: +7.3% backward on a
    hand scene)."""
    import mujoco
    light = {int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE)}
    if int(m.opt.cone) != int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        return False
    return all(int(m.geom_type[g]) in light for g in list(m.pair_geom1) + list(m.pair_geom2))


def adjoint_defines(target=None, options=(), nvcc_path=None, forest=False, light=True):
    """The -D options of a unit that launches k_adjoint (the FFI library's adjoint variant units, the adjoint check
    binary) for its build target (ADJ_CAPS: sm_103 ['-DADJ_MAXNREG=120'] where $DK_SM103_REG_CAPS is 1, the default),
    unless `options` already bound k_adjoint; every other target: none. forest: a forest-engine unit (FOREST_ADJ_CAPS:
    sm_103 ['-DADJ_MAXNREG=128'] where `light` (forest_adj_light of the model) and $DK_SM103_FOREST_ADJ_CAP is 1, the
    default; none with 0; ['-DADJ_MAXNREG=N'] for every forest model with N). Register allocation only (bitwise
    unchanged)."""
    target = target or arch()
    table = FOREST_ADJ_CAPS if forest else ADJ_CAPS
    if target not in table or not reg_caps(target):
        return []
    var, caps = table[target]
    if forest:
        n = forest_adj_cap()
        if n == 0 or (n is None and not light):
            return []
        if n is not None:
            caps = (f"ADJ_MAXNREG={n}",)
    if not _maxnreg_ok(nvcc_path or nvcc(), var):
        return []
    return reg_caps_defines(options, caps)


def compile_command(src, out, extra=(), unit=None, defines=True, target=None, nvcc_path=None):
    """compile_cuda's command line for build target target (default arch()): [nvcc, -O3, <arch_flags(unit)>,
    -std=c++17, [extra], [unit_defines(unit)], -o, out, src] (also scripts/identity_gate.py's)."""
    extra = list(extra)
    return [nvcc_path or nvcc(), "-O3", *arch_flags(unit, target, nvcc_path), "-std=c++17", *extra,
            *(unit_defines(unit, target, extra, nvcc_path) if defines else []), "-o", str(out), str(src)]


def compile_cuda(src, out, extra=(), unit=None, defines=True):
    """nvcc -O3 <arch_flags(unit)> -std=c++17 [extra] [unit_defines(unit)] -o out src (unit 'forward': the forward /
    kinematics kernels' unit, which gets the target's register caps; on sm_120 every unit is built from compute_90 PTX
    by default; default unit: the adjoint units' / test programs' flags). defines=False: without unit_defines (the
    caller passes every -D option itself). The command is compile_command's."""
    subprocess.run(compile_command(src, out, extra, unit, defines), check=True)

"""Build the engine's JAX FFI library for a target architecture without that GPU, and report every kernel's resources.

For Jetson Orin (sm_87) from any machine with nvcc: the library units come from the production build plan
(jax_ffi.build_plan, the same units, -D options and architecture flags that jax_ffi.init builds on the target), are
compiled and linked by jax_ffi._build into a separate cache directory, and ptxas -v gives the registers, stack frame,
spill stores / loads and static shared memory of every kernel of every unit (cuobjdump -res-usage of the linked
library is recorded too). --compare adds another
target (default sm_86, the closest one with a known-good build) for a side-by-side table.

The library this produces holds sm_87 device code and a host part for THIS machine's CPU: on an x86 host it proves that
the device code compiles and links for sm_87, not that it runs. On the Orin itself the same command (or simply the
first jax_ffi.init / replay_bench.py run) builds the aarch64 library that the engine loads.

  PYTHONPATH=src python scripts/orin/build_sm87.py [--arch sm_87] [--compare sm_86] [--engine forest|tree]
      [--cache DIR] [--out result.json] [--cross-nvcc NVCC --cross-ccbin G++ --cross-target-dir DIR]

--cross-ccbin adds a cross build of --arch's library for another host CPU (e.g. aarch64 from x86, with a cross g++
and a CUDA cross target directory): every unit compiled and linked with the build's own flags. It checks that the
host code compiles for that CPU; the library is not loaded, and the engine's own cache keys it differently (the nvcc
version is part of the key), so it is not a substitute for the build on the device.

Environment: $NVCC (else nvcc on PATH, $CUDA_HOME/bin/nvcc, ... /usr/local/cuda/bin/nvcc: nablaengine.cuda.nvcc),
$DK_FIXTURES (the model, default fixtures/mjlab). No GPU is needed. Exit status 1 if a build fails, or if a kernel spills on --arch where it does
not on --compare.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def kernel_name(mangled):
    """The last identifier of an Itanium-mangled (nested) name, e.g. k_adjoint of _ZN<namespace>9k_adjointE...;
    the name itself where it is not mangled."""
    if not mangled.startswith("_Z"):
        return mangled
    i = 3 if mangled.startswith("_ZN") else 2
    last = None
    while i < len(mangled) and mangled[i].isdigit():
        j = i
        while mangled[j].isdigit():
            j += 1
        n = int(mangled[i:j])
        last, i = mangled[j:j + n], j + n
    return last or mangled


def kernel_resources(lib, cuobjdump):
    """{kernel (demangled-ish short name): dict(reg, stack, shared, spill_st, spill_ld)} from cuobjdump -res-usage."""
    out = subprocess.run([cuobjdump, "-res-usage", str(lib)], check=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, universal_newlines=True).stdout
    res, name = {}, None
    for line in out.splitlines():
        m = re.search(r"Function (\S+):", line)
        if m:
            name = m.group(1)
            continue
        m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", line)
        if m and name:
            key = kernel_name(name)
            while key in res:
                key += "'"
            res[key] = dict(reg=int(m.group(1)), stack=int(m.group(2)), shared=int(m.group(3)),
                            local=int(m.group(4)), mangled=name)
            name = None
    return res


def ptxas_report(plan, arch, include, work):
    """Compile each unit to an object with -Xptxas -v (the build's own flags): {unit: {kernel: dict(reg, spill_st,
    spill_ld, stack, smem)}} parsed from ptxas's verbose output."""
    from nablaengine import cuda, jax_ffi
    from concurrent.futures import ThreadPoolExecutor
    nvcc = cuda.nvcc()
    rep = {}

    def one(name):
        src = work / name
        src.write_text(plan["units"][name])
        cmd = [nvcc, *jax_ffi.unit_command(name, arch, plan["defines"], plan["unit_arch"], include),
               "-Xptxas", "-v", "-c", "-o", str(work / (name + ".o")), str(src)]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        if r.returncode:
            raise RuntimeError(f"nvcc failed on {name}:\n{r.stdout[-4000:]}")
        return r.stdout

    jobs = int(os.environ.get("DK_BUILD_JOBS") or 0) or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(plan["units"])))) as ex:
        logs = dict(zip(sorted(plan["units"]), ex.map(one, sorted(plan["units"]))))
    for name, log in logs.items():
        kernels, cur = {}, None
        for line in log.splitlines():
            m = re.search(r"Compiling entry function '(\S+)' for '(sm_\w+)'", line)
            if m:
                cur = kernel_name(m.group(1))
                kernels[cur] = {}
                continue
            m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
            if m and cur:
                kernels[cur].update(stack=int(m.group(1)), spill_st=int(m.group(2)), spill_ld=int(m.group(3)))
                continue
            m = re.search(r"Used (\d+) registers", line)
            if m and cur:
                kernels[cur]["reg"] = int(m.group(1))
                s = re.search(r"(\d+) bytes smem", line)
                kernels[cur]["smem"] = int(s.group(1)) if s else 0
        rep[name] = kernels
    return rep


def cross_build(plan, arch, include, nvcc, ccbin, target_dir, work):
    """The library's units compiled and linked for another host CPU (nvcc -ccbin <cross g++> -target-dir <dir>), with
    the build's own per-unit flags; returns the .so path. A compile check of the host code for that CPU: the result is
    not loaded here."""
    from concurrent.futures import ThreadPoolExecutor
    from nablaengine import jax_ffi
    cross = ["-ccbin", ccbin, *(["-target-dir", target_dir] if target_dir else [])]
    work.mkdir(parents=True, exist_ok=True)

    def one(name):
        (work / name).write_text(plan["units"][name])
        r = subprocess.run([nvcc, *cross, *jax_ffi.unit_command(name, arch, plan["defines"], plan["unit_arch"],
                                                                 include), "-c", "-o", str(work / (name + ".o")),
                            str(work / name)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
        if r.returncode:
            raise RuntimeError(f"cross nvcc failed on {name}:\n{r.stdout[-4000:]}")
    jobs = int(os.environ.get("DK_BUILD_JOBS") or 0) or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(plan["units"])))) as ex:
        list(ex.map(one, sorted(plan["units"])))
    out = work / "lib.so"
    r = subprocess.run([nvcc, *cross, "-shared", f"-arch={arch}", "-o", str(out),
                        *(str(work / (n + ".o")) for n in sorted(plan["units"]))], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode:
        raise RuntimeError(f"cross link failed:\n{r.stdout[-4000:]}")
    return out


def model_and_pairs(engine):
    from nablaengine import engines, jax_ffi
    m, fp, com = jax_ffi._model()
    (engines.prefer_forest if engine == "forest" else engines.prefer_tree)(m)
    return m, fp, com


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arch", default="sm_87")
    ap.add_argument("--compare", default="sm_86", help="another target for the table ('' for none)")
    ap.add_argument("--engine", default="forest", choices=("forest", "tree"))
    ap.add_argument("--cache", default=None, help="cache directory of the built libraries (default a temporary one)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-work", action="store_true", help="keep the temporary work directory")
    ap.add_argument("--cross-nvcc", default=None, help="also a cross build of --arch's library with this nvcc ...")
    ap.add_argument("--cross-ccbin", default=None, help="... and this host compiler (e.g. aarch64-linux-gnu-g++)")
    ap.add_argument("--cross-target-dir", default=None, help="nvcc -target-dir of the cross build (e.g. sbsa-linux, "
                    "aarch64-linux)")
    args = ap.parse_args()

    import jax
    from nablaengine import cuda, identity, jax_ffi
    nvcc = cuda.nvcc()
    ver = subprocess.run([nvcc, "--version"], check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout
    jax_ffi.ensure_jax_ffi()
    include = jax.ffi.include_dir()
    cuobjdump = identity.find_tool("cuobjdump")
    tmp = Path(tempfile.mkdtemp(prefix="orin_build_"))
    os.environ["DK_CACHE_DIR"] = args.cache or str(tmp / "cache")
    m, fp, com = model_and_pairs(args.engine)
    rel = re.search(r"release [\d.]+", ver).group(0)
    print(f"nvcc {nvcc} ({rel}), jax {jax.__version__}, engine {args.engine}, host {os.uname().machine}")
    result = dict(engine=args.engine, nvcc=ver.strip().splitlines()[-1], jax=jax.__version__,
                  host=os.uname().machine, targets={})
    bad = False
    for arch in [a for a in (args.arch, args.compare) if a]:
        plan = jax_ffi.build_plan(m, fp, arch, com_body=com)
        t = time.time()
        path, built, key = jax_ffi._build(plan["units"], plan["defines"], arch, include, plan["consts"],
                                          plan["unit_arch"] or None)
        dt = time.time() - t
        work = tmp / arch
        work.mkdir()
        rep = ptxas_report(plan, arch, include, work)
        res = kernel_resources(path, cuobjdump) if cuobjdump else {}
        size = path.stat().st_size
        print(f"\n{arch}: library {path.name} ({size / 1e6:.1f} MB, built {built}, {dt:.0f} s), units "
              f"{len(plan['units'])}, arch flags forward {plan['arch_flags']['forward']} adjoint "
              f"{plan['arch_flags']['adjoint']}, forward caps {[d for d in plan['forward_defines'] if 'MAXNREG' in d or 'MINB' in d]}")
        print(f"  {'unit':<16} {'kernel':<22} {'reg':>4} {'stack':>6} {'spill st/ld':>12} {'smem':>6}")
        for u, ks in rep.items():
            for k, v in ks.items():
                sp = v.get("spill_st", 0) + v.get("spill_ld", 0)
                print(f"  {u:<16} {k:<22} {v.get('reg', 0):>4} {v.get('stack', 0):>6} "
                      f"{v.get('spill_st', 0):>5}/{v.get('spill_ld', 0):<6} {v.get('smem', 0):>6}{'  SPILL' if sp else ''}")
        result["targets"][arch] = dict(library=path.name, bytes=size, key=key, build_seconds=round(dt, 1),
                                       arch_flags=plan["arch_flags"], forward_defines=plan["forward_defines"],
                                       ptxas=rep, cuobjdump=res)
    if args.cross_ccbin:
        plan = jax_ffi.build_plan(m, fp, args.arch, com_body=com)
        t = time.time()
        lib = cross_build(plan, args.arch, include, args.cross_nvcc or nvcc, args.cross_ccbin, args.cross_target_dir,
                          tmp / "cross")
        desc = subprocess.run(["file", "-b", str(lib)], stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
        res = kernel_resources(lib, cuobjdump) if cuobjdump else {}
        print(f"\ncross build ({args.cross_ccbin}, target dir {args.cross_target_dir}): {lib.stat().st_size / 1e6:.1f} "
              f"MB in {time.time() - t:.0f} s: {desc}; {len(res)} kernels")
        result["cross"] = dict(ccbin=Path(args.cross_ccbin).name, target_dir=args.cross_target_dir, file=desc,
                               bytes=lib.stat().st_size, kernels=sorted(res))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))
    if args.compare and args.compare in result["targets"]:
        a, b = (result["targets"][x]["ptxas"] for x in (args.arch, args.compare))
        diff = [(u, k, a[u][k].get("reg"), b.get(u, {}).get(k, {}).get("reg")) for u in a for k in a[u]
                if a[u][k].get("reg") != b.get(u, {}).get(k, {}).get("reg")]
        more = [(u, k) for u in a for k in a[u] if a[u][k].get("spill_st", 0) > b.get(u, {}).get(k, {}).get("spill_st", 0)]
        bad |= bool(more)
        print(f"\nspill stores above {args.compare}'s: {more or 'none'}")
        print(f"registers {args.arch} vs {args.compare}: {len(diff)} kernels differ" +
              "".join(f"\n  {u} {k}: {x} vs {y}" for u, k, x, y in diff[:20]))
    if not args.keep_work:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nRESULT", "FAIL (more spills than the comparison target)" if bad else "PASS")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

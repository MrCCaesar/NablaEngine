"""Checks of chol_solve_bcast_wide, the forest engine's Cholesky for 36 <= NV <= 48.

  unit    the factor + solve alone on random SPD matrices (graded mass-matrix-like and M + J' D J Hessian-like), one
          warp per matrix, for each NV of --nv against chol_solve_compact and chol_solve_reg (same source text):
          the factor bitwise both of theirs, the solution bitwise chol_solve_reg's, errors against float64, cycles and
          time per call (requires every bitwise statement)
  engine  the forest engine's forward (scripts/engine_step_check.py's forest binary) on an nv 43 model with contacts
          (scripts/forest/models/s3.xml, a G1 with hands and a free box, with the finger joints of both hands other
          than the thumb yaws removed: wide_scene) from MuJoCo C states (scripts/eq/connect_check.corpus), built three
          ways: default (chol_solve_bcast_wide with NEWTON_REUSE_WIDE), -DCHOL_BCAST_WIDE=0 -DCHOL_COMPACT=0
          (chol_solve_reg, without the reuse) and -DCHOL_BCAST_WIDE=0 (the previous default); outputs compared,
          -DNEWTON_REUSE_WIDE=0 required bitwise equal to the chol_solve_reg build, ms per control step of each
  reuse   the JAX FFI forward (production build, forward variant) on an nv 48 robot (s3.xml without its box, the left
          hand's finger joints but its thumb yaw removed: wide_scene) from a fixed corpus (16 worlds, constant random
          controls, --calls calls of 10 substeps), default build and -DNEWTON_REUSE_WIDE=0, each in a process of its
          own: both must finish with finite states, close to each other. An earlier default build failed there
          (CUDA_ERROR_ILLEGAL_ADDRESS at call 3, sm_70): the re-use in k_post's Hand::big path (a world beyond
          k_post's shared-memory capacity) with chol_resolve as a call

  python scripts/forest/chol_wide_check.py unit [--nv 36,40,41,48] [--out unit.json]
  python scripts/forest/chol_wide_check.py engine [--nworld 4096] [--out engine.json]
  python scripts/forest/chol_wide_check.py reuse [--calls 10] [--out reuse.json]
Exit status 0 when every requirement holds, 1 otherwise.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "eq"))

KER = r"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#define NTRI (NV * (NV + 1) / 2)
#define TRI(i, j) ((i) * ((i) + 1) / 2 + (j))
@@FUNCS@@
template <int V>
__global__ void k(const float* A, const float* B, float* L, float* X, long long* cyc, int nw, int reps) {
  extern __shared__ float sh[];
  const int lane = threadIdx.x & 31, w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (w >= nw) return;
  float* sA = sh + (threadIdx.x >> 5) * (2 * NTRI + NV + 128);
  float* sL = sA + NTRI; float* sb = sL + NTRI; float* cb = (float*)(((size_t)(sb + NV) + 15) & ~(size_t)15);
  long long t = 0;
  for (int rep = 0; rep < reps; rep++) {
    for (int e = lane; e < NTRI; e += 32) sA[e] = A[(size_t)w * NTRI + e];
    for (int e = lane; e < NV; e += 32) sb[e] = B[(size_t)w * NV + e];
    __syncwarp();
    long long t0 = clock64();
    if (V == 0) chol_solve_compact(sA, sL, sb, lane);
    else if (V == 1) chol_solve_reg(sA, sL, sb, lane);
    else chol_solve_bcast_wide(sA, sL, sb, lane, cb);
    __syncwarp();
    t += clock64() - t0;
  }
  for (int e = lane; e < NTRI; e += 32) L[(size_t)w * NTRI + e] = sL[e];
  for (int e = lane; e < NV; e += 32) X[(size_t)w * NV + e] = sb[e];
  if (lane == 0) cyc[w] = t / reps;
}
int main(int argc, char** argv) {
  const char* dir = argv[1]; int nw = atoi(argv[2]), var = atoi(argv[3]), wpb = atoi(argv[4]), reps = atoi(argv[5]);
  std::vector<float> hA((size_t)nw * NTRI), hB((size_t)nw * NV);
  char p[512]; snprintf(p, sizeof p, "%s/A.bin", dir); FILE* f = fopen(p, "rb"); fread(hA.data(), 4, hA.size(), f); fclose(f);
  snprintf(p, sizeof p, "%s/B.bin", dir); f = fopen(p, "rb"); fread(hB.data(), 4, hB.size(), f); fclose(f);
  float *A, *B, *L, *X; long long* C;
  cudaMalloc(&A, hA.size() * 4); cudaMalloc(&B, hB.size() * 4); cudaMalloc(&L, hA.size() * 4); cudaMalloc(&X, hB.size() * 4);
  cudaMalloc(&C, nw * 8);
  cudaMemcpy(A, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(B, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice);
  size_t sm = wpb * (2 * NTRI + NV + 128) * 4;
  auto kf = var == 0 ? k<0> : var == 1 ? k<1> : k<2>;
  cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  kf<<<(nw + wpb - 1) / wpb, 32 * wpb, sm>>>(A, B, L, X, C, nw, reps);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  kf<<<(nw + wpb - 1) / wpb, 32 * wpb, sm>>>(A, B, L, X, C, nw, reps);
  cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1);
  cudaError_t err = cudaGetLastError(); if (err) { printf("err %s\n", cudaGetErrorString(err)); return 1; }
  std::vector<float> hL(hA.size()), hX(hB.size()); std::vector<long long> hC(nw);
  cudaMemcpy(hL.data(), L, hL.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(hX.data(), X, hX.size() * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hC.data(), C, nw * 8, cudaMemcpyDeviceToHost);
  snprintf(p, sizeof p, "%s/L%d.bin", dir, var); f = fopen(p, "wb"); fwrite(hL.data(), 4, hL.size(), f); fclose(f);
  snprintf(p, sizeof p, "%s/X%d.bin", dir, var); f = fopen(p, "wb"); fwrite(hX.data(), 4, hX.size(), f); fclose(f);
  double cs = 0; for (auto c : hC) cs += c;
#ifdef WPROFH
  if (var == 2) { unsigned long long hp[8]; cudaMemcpyFromSymbol(hp, g_prof, sizeof hp); printf("prof"); for (int i = 0; i < 5; i++) printf(" %.1f", hp[i] / (2.0 * reps * nw)); printf("\n"); }
#endif
  printf("{\"var\": %d, \"cycles\": %.1f, \"ms\": %.4f}\n", var, cs / nw, ms / reps);
}
"""


def spd(rng, nw, nv):
    """mass-matrix-like (graded diagonal, coupled) and Hessian-like (M + J' D J) SPD matrices"""
    out = np.empty((nw, nv, nv))
    for w in range(nw):
        kind = w % 3
        d = 10.0 ** rng.uniform(-2.5, 1.0, nv)
        Bm = rng.standard_normal((nv, nv)) * np.sqrt(d)[:, None] * 0.3
        A = Bm @ Bm.T + np.diag(d)
        if kind >= 1:
            J = rng.standard_normal((rng.integers(5, 60), nv)) * (rng.random((1, nv)) < 0.6)
            A += J.T @ np.diag(10.0 ** rng.uniform(-1, 3 if kind == 2 else 1.5, len(J))) @ J
        out[w] = A
    return out




def extract():
    from nablaengine.forest_engine import source
    s = source.DEVICE
    i, j = s.index("#ifndef CHOL_INLINE\n"), s.index("\n// Broadcast variant (same contract")
    k, l = s.index("\n// Wide broadcast variant"), s.index("#ifndef CHOL_CB\n")
    return s[i:j] + "#define CHOL_BCAST_WIDE (NV >= 36 && NV <= 48)\n" + s[k:l]


def cmd_unit(a, work):
    from nablaengine.cuda import nvcc, arch
    funcs = extract()
    target = a.arch or arch()
    res, ok = [], True
    for nv in map(int, a.nv.split(",")):
        cu = work / f"unit{nv}.cu"
        cu.write_text(f"#define NV {nv}\n" + KER.replace("@@FUNCS@@", funcs))
        exe = work / f"unit{nv}"
        subprocess.run([nvcc(), "-O3", f"-arch={target}", "-std=c++17", "-w", "-o", str(exe), str(cu)], check=True)
        rng = np.random.default_rng(nv)
        A32 = spd(rng, a.nmat, nv).astype(np.float32)
        b32 = rng.standard_normal((a.nmat, nv)).astype(np.float32)
        il = np.tril_indices(nv)
        np.ascontiguousarray(A32[:, il[0], il[1]]).tofile(work / "A.bin")
        b32.tofile(work / "B.bin")
        A64 = A32.astype(np.float64)
        x64 = np.linalg.solve(A64, b32.astype(np.float64)[..., None])[..., 0]
        L64 = np.linalg.cholesky(A64)[:, il[0], il[1]]
        row, out = dict(nv=nv, arch=target), {}
        for v, name in enumerate(("compact", "reg", "wide")):
            o = subprocess.run([str(exe), str(work), str(a.nmat), str(v), "3", str(a.reps)], check=True,
                               stdout=subprocess.PIPE, text=True).stdout
            js = json.loads(o.strip().splitlines()[-1])
            X = np.fromfile(work / f"X{v}.bin", np.float32).reshape(a.nmat, nv)
            Lp = np.fromfile(work / f"L{v}.bin", np.float32).reshape(a.nmat, -1)
            ex = np.linalg.norm(X - x64, axis=1) / np.linalg.norm(x64, axis=1)
            el = np.abs(Lp - L64).max(1) / np.abs(L64).max(1)
            row[name] = dict(cycles=js["cycles"], ms=js["ms"], x_rel_median=float(np.median(ex)),
                             x_rel_p99=float(np.quantile(ex, .99)), L_rel_median=float(np.median(el)))
            out[name] = (X, Lp)
        row["L_wide_eq_compact"] = bool(np.array_equal(out["wide"][1], out["compact"][1]))
        row["L_wide_eq_reg"] = bool(np.array_equal(out["wide"][1], out["reg"][1]))
        row["x_wide_eq_reg"] = bool(np.array_equal(out["wide"][0], out["reg"][0]))
        ok &= row["L_wide_eq_compact"] and row["L_wide_eq_reg"] and row["x_wide_eq_reg"]
        print(json.dumps(row), flush=True)
        res.append(row)
    return dict(unit=res, pass_=bool(ok))


def cmd_engine(a, work):
    import mujoco
    import connect_check as cc
    from engine_step_check import forest_binary_source
    from nablaengine.cuda import compile_cuda
    from nablaengine.forest_engine.step import com_words
    m = mujoco.MjModel.from_xml_string(wide_scene(("left", "right"), keep_box=True))
    nw = a.nworld
    q, v, u = cc.corpus(m, min(nw, 256), 0)
    src, consts = forest_binary_source(m, [], None)
    (work / "fwd.cu").write_text(src)
    consts.tofile(work / "consts.bin")
    rep = lambda x: np.resize(np.asarray(x), (nw,) + np.asarray(x).shape[1:])
    arr = dict(qpos=rep(q), qvel=rep(v), warm=np.zeros((nw, m.nv)), ctrl=rep(u), fric=np.ones(nw),
               com=np.zeros((nw, com_words(m))))
    for k, x in arr.items():
        np.ascontiguousarray(x, np.float32).tofile(work / f"{k}.bin")
    builds = dict(wide=[], wide_noreuse=["-DNEWTON_REUSE_WIDE=0"], reg=["-DCHOL_BCAST_WIDE=0", "-DCHOL_COMPACT=0"],
                  previous=["-DCHOL_BCAST_WIDE=0"])
    res, outs = dict(nv=m.nv, nworld=nw), {}
    for name, defs in builds.items():
        exe = work / f"fwd_{name}"
        compile_cuda(work / "fwd.cu", exe, defs, unit="forward")

        def run(reps):
            o = subprocess.run([str(exe), str(work), str(nw), str(a.iterations), str(reps), "4", "0", "0"],
                               check=True, stdout=subprocess.PIPE, text=True).stdout
            return json.loads(next(x for x in o.splitlines() if x.startswith('{"nworld"')))
        run(0)
        outs[name] = (np.fromfile(work / "out_qvel.bin", np.float32).reshape(nw, m.nv),
                      np.fromfile(work / "out_info.bin", np.int32).reshape(nw, 4))
        res[name] = dict(ms=min(run(a.reps)["ms"] for _ in range(3)), defines=defs)
    for name in ("wide", "reg", "previous"):
        v, i = outs[name]
        assert np.isfinite(v).all(), name
    base = outs["previous"][0]
    for name in ("wide", "wide_noreuse", "reg"):
        v = outs[name][0]
        ev = np.linalg.norm(v - base, axis=1) / np.maximum(np.linalg.norm(base, axis=1), 1e-6)
        res[name]["vs_previous"] = dict(bitwise=bool(np.array_equal(v, base)), qvel_rel_median=float(np.median(ev)),
                                        qvel_rel_max=float(ev.max()))
    res["wide_noreuse_eq_reg"] = bool(np.array_equal(outs["wide_noreuse"][0], outs["reg"][0]))
    return dict(engine=res, pass_=res["wide_noreuse_eq_reg"])


def wide_scene(sides=("left",), keep_box=False):
    """MJCF text of S3 (scripts/forest/models/s3.xml, without its inactive weld) with the finger joints of the hands
    `sides` other than <side>_thumb_yaw_joint removed (the fingers rigid, fused into their parents; 11 per hand) and
    its box unless keep_box: nv 48 for the left hand without the box (reuse), 43 for both hands with it (engine)"""
    import re
    import mujoco
    src = REPO / "scripts" / "forest" / "models" / "s3.xml"
    x = re.sub(r'\s*<weld name="base_hold"[^>]*/>', "", src.read_text())
    x = x.replace('meshdir="assets"', f'meshdir="{(src.parent / "assets").resolve()}"')
    if not keep_box:
        x = re.sub(r'\s*<body name="box0".*?</body>', "", x, flags=re.S)
        x = re.sub(r'\s*<pair [^>]*\bbox0_[^>]*/>', "", x)
        x = re.sub(r'(<key [^>]*qpos=")([^"]*)"', lambda mt: mt.group(1) + " ".join(mt.group(2).split()[:-7]) + '"', x)
    m = mujoco.MjModel.from_xml_string(x)
    jn = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
    names = [jn(j) for j in range(m.njnt) for sd in sides if jn(j).startswith(f"{sd}_")
             and re.search(r"_(index|middle|ring|little|thumb)_", jn(j)) and jn(j) != f"{sd}_thumb_yaw_joint"]
    assert len(names) == 11 * len(sides), names
    ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names}
    qadr = {int(m.jnt_qposadr[j]) for j in ids}
    acts = {a for a in range(m.nu) if int(m.actuator_trnid[a, 0]) in ids}
    for n in names:
        x, c = re.subn(rf'\s*<joint name="{n}"[^>]*/>', "", x)
        assert c == 1, n
        x = re.sub(rf'\s*<general [^>]*joint="{n}"[^>]*/>', "", x)
        x = re.sub(r'\s*<joint name="[^"]*" joint1="[^"]*" joint2="[^"]*"[^>]*/>',
                   lambda mt: "" if f'"{n}"' in mt.group(0) else mt.group(0), x)

    def keep(drop):
        return lambda mt: mt.group(1) + " ".join(e for i, e in enumerate(mt.group(2).split()) if i not in drop) + '"'
    x = re.sub(r'(<key [^>]*?qpos=")([^"]*)"', keep(qadr), x)
    x = re.sub(r'(<key [^>]*?ctrl=")([^"]*)"', keep(acts), x)
    x, c = re.subn(r"<compiler ", '<compiler fusestatic="true" ', x, count=1)
    assert c == 1
    return x


def reuse_run(scene, defines, out, calls):
    """(a process of its own) the rollout of cmd_reuse with -D defines in every unit; writes out (.npz: qvel, calls
    done, the error)"""
    sys.path.insert(0, str(REPO / "scripts" / "forest"))
    from nablaengine import jax_ffi
    jax_ffi.VARIANTS = (jax_ffi.FWD_VARIANT,)
    defs = [f"-D{d}" for d in defines.split(",") if d]
    orig = jax_ffi.library_sources

    def sources(*args, **kw):
        units, consts, d = orig(*args, **kw)
        return units, consts, {u: list(d[u]) + defs for u in d}
    jax_ffi.library_sources = sources
    import jax.numpy as jnp
    import connect_check as cc
    import nv128_check as nc
    m = nc.load(scene)
    q, v, _ = cc.corpus(m, 256, 0)
    info = nc.init(m)
    nw, it = 16, int(m.opt.iterations)
    lo, hi = cc.ctrl_range(m)
    u = jnp.asarray(np.random.default_rng(1).uniform(lo, hi, (nw, m.nu)).astype(np.float32))
    qq, vv, ww = jnp.asarray(q[:nw]), jnp.asarray(v[:nw]), jnp.zeros((nw, m.nv), jnp.float32)
    fr, com = jnp.ones((nw,), jnp.float32), jnp.zeros((nw, int(info.get("com_words", 3))), jnp.float32)
    done, err = 0, ""
    try:
        for _ in range(calls):
            qq, vv, ww = jax_ffi.step(qq, vv, ww, u, fr, com, substeps=10, iterations=it)
            np.asarray(vv)
            done += 1
    except Exception as e:
        err = str(e)[:300]
    np.savez(out, qvel=np.asarray(vv) if not err else np.zeros((nw, m.nv), np.float32), done=done, err=err, nv=m.nv)


def cmd_reuse(a, work):
    scene = work / "reuse_nv48.xml"
    scene.write_text(wide_scene())
    res, outs, ok = dict(calls=a.calls), {}, True
    for name, defs in (("default", ""), ("noreuse", "NEWTON_REUSE_WIDE=0")):
        out = work / f"reuse_{name}.npz"
        env = dict(os.environ, DK_CACHE_DIR=os.environ.get("DK_CACHE_DIR") or str(work / "cache"))
        subprocess.run([sys.executable, __file__, "_reuse_run", "--scene", str(scene), "--defines", defs,
                        "--npz", str(out), "--calls", str(a.calls)], env=env, check=True)
        z = np.load(out)
        outs[name] = z["qvel"]
        fin = bool(np.isfinite(z["qvel"]).all())
        res[name] = dict(defines=defs, nv=int(z["nv"]), calls_done=int(z["done"]), error=str(z["err"]), finite=fin)
        ok &= int(z["done"]) == a.calls and fin and int(z["nv"]) == 48
        print(json.dumps({name: res[name]}), flush=True)
    if ok:
        b, v = outs["noreuse"], outs["default"]
        ev = np.linalg.norm(v - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-6)
        res["default_vs_noreuse"] = dict(qvel_rel_median=float(np.median(ev)), qvel_rel_max=float(ev.max()))
        ok &= float(np.median(ev)) < 1e-2
    return dict(reuse=res, pass_=bool(ok))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("unit", "engine", "reuse", "_reuse_run"))
    ap.add_argument("--nv", default="36,37,38,39,40,41,42,43,44,45,46,47,48")
    ap.add_argument("--nmat", type=int, default=4096)
    ap.add_argument("--arch", default=None, help="unit: the build target (default: this GPU's)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--calls", type=int, default=10, help="reuse: control steps (10 substeps each)")
    ap.add_argument("--scene", default="", help=argparse.SUPPRESS)
    ap.add_argument("--defines", default="", help=argparse.SUPPRESS)
    ap.add_argument("--npz", default="", help=argparse.SUPPRESS)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "_reuse_run":
        return reuse_run(a.scene, a.defines, a.npz, a.calls)
    with tempfile.TemporaryDirectory() as d:
        r = dict(unit=cmd_unit, engine=cmd_engine, reuse=cmd_reuse)[a.cmd](a, Path(d))
    ok = r.pop("pass_")
    r["pass"] = ok
    print(json.dumps(r, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

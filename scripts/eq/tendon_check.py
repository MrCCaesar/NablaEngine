"""Fixed tendons in the forest engine's forward (forest_engine.tendon), against float64 MuJoCo C (the arbiter) and
MuJoCo Warp (dense Jacobian):"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import connect_check as cc

ROWS_KERNEL = r"""
#include <cstdio>
#include <cstdlib>
#include <vector>
// the tendon rows of each state (substep_pre on struct Sim): nrows, then per row kind, a, b, pos, D, aref
__global__ void k_rows(int nw, const float* qpos, const float* qvel, float* out, int nout) {
  extern __shared__ __align__(16) unsigned char dk_smem[];
  Sim& s = *reinterpret_cast<Sim*>(dk_smem);
  const int lane = threadIdx.x & 31, w = blockIdx.x;
  if (w >= nw) return;
  for (int i = lane; i < NQ; i += 32) s.qpos[i] = qpos[(size_t)w * NQ + i];
  for (int i = lane; i < NV; i += 32) { s.qvel[i] = qvel[(size_t)w * NV + i]; s.qacc[i] = 0.f; }
  if (lane < 3) s.com[lane] = 0.f;
  if (lane == 0) { s.fric = 1.f; s.unsupported = 0; s.oforce = nullptr; }
  __syncwarp();
  const float ctrl[NU > 0 ? NU : 1] = {};
  substep_pre(s, ctrl, lane, nullptr);
  __syncwarp();
  float* o = out + (size_t)w * nout;
  if (lane == 0) {
    int n = 0;
    for (int e = 0; e < NEQT; e++) {
      const int r = s.nefc_dof + NEQJ + 3 * NEQC + e;
      float* q = o + 1 + 6 * n++;
      q[0] = 8; q[1] = e; q[2] = 0; q[3] = et_pos(s.qpos, e); q[4] = s.efc_D[r]; q[5] = s.efc_aref[r];
    }
    for (int r = s.nefc_dof + NEQR; r < s.nefc && s.efc_kind[r] == 7; r++) {
      const int l = s.efc_a[r];
      float* q = o + 1 + 6 * n++;
      q[0] = 7; q[1] = l; q[2] = s.efc_b[r];
      q[3] = RSUB(tl_dist(s.qpos, l, s.efc_b[r] < 0), c_tlmargin[l]); q[4] = s.efc_D[r]; q[5] = s.efc_aref[r];
    }
    o[0] = n;
  }
}
int main(int argc, char** argv) {
  const char* dir = argv[1];
  const int nw = atoi(argv[2]), nout = atoi(argv[3]), ncst = atoi(argv[4]);
  char path[1024];
  auto rd = [&](const char* nm, size_t n) {
    std::vector<float> v(n);
    snprintf(path, sizeof path, "%s/%s", dir, nm);
    FILE* f = fopen(path, "rb");
    if (!f || fread(v.data(), 4, n, f) != n) { printf("read %s\n", nm); exit(1); }
    fclose(f);
    return v;
  };
  std::vector<float> cf = rd("consts.bin", ncst);
  const float* p = cf.data();
LOAD_CONSTS
  std::vector<float> hq = rd("qpos.bin", (size_t)nw * NQ), hv = rd("qvel.bin", (size_t)nw * NV);
  float *dq, *dv, *dout;
  cudaMalloc(&dq, hq.size() * 4); cudaMalloc(&dv, hv.size() * 4); cudaMalloc(&dout, (size_t)nw * nout * 4);
  cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice);
  cudaFuncSetAttribute(k_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Sim));
  k_rows<<<nw, 32, sizeof(Sim)>>>(nw, dq, dv, dout, nout);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<float> ho((size_t)nw * nout);
  cudaMemcpy(ho.data(), dout, ho.size() * 4, cudaMemcpyDeviceToHost);
  snprintf(path, sizeof path, "%s/out.bin", dir);
  FILE* f = fopen(path, "wb");
  fwrite(ho.data(), 4, ho.size(), f);
  fclose(f);
  printf("%s\n", cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}
"""


def engine_rows(m, q, v, work):
    """The engine's tendon rows per state: [(kind, index, side, pos, D, aref)] (kind 8: tendon equality index; 7: the
    limited tendon's index and side +1 lower / -1 upper)."""
    from nablaengine import cuda
    from nablaengine.forest_engine import adjoint, host, tendon
    nmax = tendon.nrows_max(m)
    consts, lays, sizes = adjoint.engine_constants(m, [])
    work = Path(work)
    (work / "rows.cu").write_text(adjoint.cuda_source(m) + ROWS_KERNEL.replace("LOAD_CONSTS", host.loader(lays, sizes)))
    consts.astype(np.float32).tofile(work / "consts.bin")
    cuda.compile_cuda(str(work / "rows.cu"), str(work / "rows"), extra=["-w"])
    nout = 1 + 6 * max(nmax, 1)
    np.asarray(q, np.float32).tofile(work / "qpos.bin")
    np.asarray(v, np.float32).tofile(work / "qvel.bin")
    r = subprocess.run([str(work / "rows"), str(work), str(len(q)), str(nout), str(len(consts))], capture_output=True,
                       text=True)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    o = np.fromfile(work / "out.bin", np.float32).reshape(len(q), nout)
    return [[tuple(o[w, 1 + 6 * k:7 + 6 * k]) for k in range(int(o[w, 0]))] for w in range(len(q))]


def c_rows(m, q, v):
    """MuJoCo C's tendon rows per state: [(kind, index, side, pos - margin, D, aref)] in the engine's order (tendon
    equalities in equality order, then the limit rows by (limited tendon, lower before upper))."""
    from nablaengine.forest_engine import tendon
    jac0 = int(m.opt.jacobian)
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    d = mujoco.MjData(m)
    eqs = [r["e"] for r in tendon.tendon_eqs(m)]
    lim = tendon.limited(m)
    out = []
    try:
        for w in range(len(q)):
            d.qpos[:] = q[w]; d.qvel[:] = v[w]; d.ctrl[:] = 0
            mujoco.mj_forward(m, d)
            J = d.efc_J.reshape(d.nefc, m.nv)
            rows = []
            for k, e in enumerate(eqs):
                r = np.nonzero((d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == e))[0]
                assert len(r) == 1
                r = r[0]
                rows.append((8, k, 0, d.efc_pos[r], d.efc_D[r], d.efc_aref[r]))
            lr = np.nonzero(d.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)[0]
            tl = []
            for r in lr:
                t = int(d.efc_id[r])
                c = tendon.coefs(m, t)
                d0 = max(c, key=lambda i: abs(c[i]))
                side = 1 if J[r, d0] * c[d0] > 0 else -1
                tl.append((7, lim.index(t), side, d.efc_pos[r] - d.efc_margin[r], d.efc_D[r], d.efc_aref[r]))
            rows += sorted(tl, key=lambda x: (x[1], -x[2]))
            out.append(rows)
    finally:
        m.opt.jacobian = jac0
    return out


def run_mjw(m, q, v, u, py, work, substeps):
    work = Path(work)
    mujoco.mj_saveModel(m, str(work / "model.mjb"), None)
    np.savez(work / "states.npz", qpos=q, qvel=v, ctrl=u)
    script = Path(__file__).resolve().parent / "mjw_step.py"
    r = subprocess.run([py, str(script), str(work / "model.mjb"), str(work / "states.npz"), str(work / "mjw.npz"),
                        "--substeps", str(substeps), "--iterations", str(int(m.opt.iterations)), "--dense"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ""})
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return dict(np.load(work / "mjw.npz")), json.loads(r.stdout.strip().splitlines()[-1])


def c_step(m, q, v, u, S):
    d = mujoco.MjData(m)
    Q, V, A = [], [], []
    for w in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q[w]; d.qvel[:] = v[w]; d.ctrl[:] = u[w]
        d.qacc_warmstart[:] = 0
        for _ in range(S):
            mujoco.mj_step(m, d)
        Q.append(d.qpos.copy()); V.append(d.qvel.copy()); A.append(d.qacc.copy())
    return np.array(Q), np.array(V), np.array(A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mjw-python", default=None)
    ap.add_argument("--states", type=int, default=256)
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--uniform-limits", action="store_true",
                    help="every joint limit on the first set (connect_check.py; earlier runs only: the engine reads per-joint sets)")
    ap.add_argument("--no-fluid", action="store_true", help="connect_check.py's option")
    ap.add_argument("--drop-spatial", action="store_true", help="connect_check.py's option")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cv = cc.load(a.model, None, a.no_fluid, a.uniform_limits, a.drop_spatial)
    m, mo = cv.model, cv.original
    from nablaengine.forest_engine import equality, tendon
    S = a.substeps
    td = sorted({d for t in tendon.used(m) for d in tendon.coefs(m, t)})
    rep = dict(model=os.path.relpath(Path(a.model).resolve(), Path(__file__).resolve().parents[2]), nv=int(m.nv),
               nbody=int(m.nbody), ntendon=int(m.ntendon), tendon_eqs=len(tendon.tendon_eqs(m)),
               limited_tendons=len(tendon.limited(m)), passive_tendons=len(tendon.passive(m)),
               armature_tendons=len(tendon.armature(m)), joint_eqs=len(equality.rows(m)),
               connects=len(equality.connects(m)), spans_trees=tendon.spans_trees(m), notes=cv.notes,
               integrator=mujoco.mjtIntegrator(int(m.opt.integrator)).name, substeps=S, mujoco=mujoco.__version__,
               seed=a.seed, tendon_dofs=td, uniform_limits=a.uniform_limits, no_fluid=a.no_fluid,
               tendon_actuators=len(tendon.actuators(m)))
    q, v, u = cc.corpus(mo, a.states, a.seed)
    info = cc.engine_init(m)
    qe, ve, ae = (np.asarray(x, np.float64) for x in cc.engine_step(m, info, q, v, u, substeps=S))
    qc, vc, ac = c_step(mo, q, v, u, S)
    f = lambda x, y, sl=slice(None): cc.stats(cc.rel(x[:, sl], y[:, sl]))
    rep["forward_vs_C"] = dict(qpos=f(qe, qc), qvel=f(ve, vc), qacc=f(ae, ac), qvel_tendon_dofs=f(ve, vc, td),
                               qacc_tendon_dofs=f(ae, ac, td))
    dd = mujoco.MjData(mo)
    ncon, nlim = [], []
    for w in range(len(q)):
        dd.qpos[:] = q[w]
        mujoco.mj_forward(mo, dd)
        ncon.append(dd.ncon)
        nlim.append(int((dd.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON).sum()))
    rep["contacts_mean"] = float(np.mean(ncon))
    rep["tendon_limit_rows_mean"] = float(np.mean(nlim))
    with tempfile.TemporaryDirectory(prefix="tendon_check_") as work:
        er = engine_rows(m, q, v, work)
        cr = c_rows(mo, q.astype(np.float64), v.astype(np.float64))
        same, err = 0, dict(pos=[], D=[], aref=[], aref_abs=[])
        both = 0
        for w in range(len(q)):
            ke = [(int(x[0]), int(x[1]), int(x[2])) for x in er[w]]
            kc = [(int(x[0]), int(x[1]), int(x[2])) for x in cr[w]]
            both += sum(1 for l in set(k[1] for k in kc if k[0] == 7) if (7, l, 1) in kc and (7, l, -1) in kc)
            if ke != kc:
                continue
            same += 1
            for E, C in zip(er[w], cr[w]):
                err["pos"].append(abs(E[3] - C[3]))
                err["D"].append(abs(E[4] - C[4]) / C[4])
                err["aref"].append(abs(E[5] - C[5]) / max(abs(C[5]), 1.0))
                err["aref_abs"].append(abs(E[5] - C[5]))
        rep["rows"] = dict(same_rows_fraction=same / len(q), rows_per_state=float(np.mean([len(r) for r in cr])),
                           limit_rows_both_sides=both, engine_vs_C={k: cc.stats(x) for k, x in err.items()})
        if a.mjw_python:
            z, ver = run_mjw(mo, q, v, u, a.mjw_python, work, S)
            rep["mjw"] = ver
            rep["mjw_dense_vs_C"] = dict(qvel=f(z["qvel"], vc), qacc=f(z["qacc"], ac), qvel_tendon_dofs=f(z["qvel"], vc, td))
            rep["engine_vs_mjw_dense"] = dict(qvel=f(ve, z["qvel"]), qacc=f(ae, z["qacc"]),
                                              qvel_tendon_dofs=f(ve, z["qvel"], td))
    txt = json.dumps(rep, indent=1)
    print(txt)
    if a.out:
        out = Path(a.out)
        if out.exists():
            sys.exit(f"{out} exists (not overwritten)")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt)


if __name__ == "__main__":
    main()

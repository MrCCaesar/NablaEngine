"""Spatial tendons in the forest engine (forest_engine.spatial): unit scenes against MuJoCo Warp 3.17 and float64 MuJoCo
C, exact gradients against float64 MuJoCo C central differences.

Scenes (SCENES; the arm of scripts/forest/models/spatial_arm.xml with a tendon / actuator block, through
nablaengine.envs.convert.to_engine): site (site-only paths, stiffness with a deadband, damping), pulley (two pulleys,
Euler), sphere (wrap without and with a side site), cylinder (without, with an outside and with an inside side site),
limit (limited spatial tendons, ranges from the corpus' length percentiles, a margin), actuator (motor with gear 2,
position actuator with kv and force range, general affine actuator on spatial tendons), free (a free root: tendons
from world sites onto the floating arm), bodies (the elbow with two hinges, one off its origin, which the converter
splits; a jointless plate on the upper arm with a site and a wrap sphere that frame / subtree sensors name: the converter
moves the frame sensors to sites and drops the subtree one, so the plate fuses; tendons over both and a motor), packed (1250 short spring tendons and one over all four dofs, a motor and a
position actuator: over forest_engine.spatial.MAX_TS_WORDS in the KTEN-strided storage, so the build takes the packed
layout). States: scripts/eq/connect_check.py's corpus (MuJoCo C rollouts with random
velocities and controls). One engine library per process: one scene per run (--scene), or --all (subprocesses).

  kin   the engine's L and J (substep_pre's storage, a small CUDA program) against MuJoCo Warp (mjw_tendon.py run by
        --mjw-python) and MuJoCo C: absolute errors (m; J in m per unit dof)
  fwd   one control step of 1 and 4 substeps from a zero warm start: qvel relative error per world (|dv| / max(|v|, 1))
        of the engine against MuJoCo Warp (scripts/eq/mjw_step.py) and MuJoCo C, and the tape's forward bitwise the
        plain step's
  fd    the VJP of one control step (--substeps) in the exact mode (clamp_kv 1, normal 0) against central differences
        of MuJoCo C (scripts/eq/connect_grad.py's fd_grad: tight solver, eps 1e-6), relative error per world

  PYTHONPATH=src python scripts/forest/spatial_check.py kin --scene sphere --mjw-python <venv>/bin/python
  PYTHONPATH=src python scripts/forest/spatial_check.py fd --all --out runs/spatial/fd.json

Exit status 1 when a median exceeds its bound (kin: J vs MuJoCo C 1e-5 m; fwd: engine vs MuJoCo C 1e-4; fd: 1e-3).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eq"))
import connect_check as cc
import connect_grad as cg

ARM = REPO / "scripts" / "forest" / "models" / "spatial_arm.xml"

SCENES = {
    "site": """<tendon>
  <spatial name="t1" stiffness="40" springlength="0.55 0.6" damping="0.5">
    <site site="w0"/><site site="a1"/><site site="b1"/><site site="c1"/><site site="d1"/></spatial>
  <spatial name="t2" stiffness="25" damping="0.2"><site site="w1"/><site site="a3"/><site site="b2"/><site site="d2"/></spatial>
</tendon>""",
    "pulley": """<tendon>
  <spatial name="p1" stiffness="30" damping="0.3">
    <site site="w0"/><site site="a1"/><pulley divisor="2"/><site site="a2"/><site site="b1"/><pulley divisor="3"/>
    <site site="b2"/><site site="d2"/></spatial>
</tendon>""",
    "sphere": """<tendon>
  <spatial name="s1" stiffness="60" damping="0.4"><site site="a1"/><geom geom="wsph"/><site site="b1"/><site site="d1"/></spatial>
  <spatial name="s2" stiffness="40" damping="0.3"><site site="a3"/><geom geom="wsph" sidesite="side_out2"/><site site="b2"/></spatial>
</tendon>""",
    "cylinder": """<tendon>
  <spatial name="c1" stiffness="60" damping="0.4"><site site="a1"/><geom geom="wcyl"/><site site="b1"/></spatial>
  <spatial name="c2" stiffness="40" damping="0.3"><site site="a2"/><geom geom="wcyl" sidesite="side_out"/><site site="b1"/><site site="c1"/></spatial>
  <spatial name="c3" stiffness="50" damping="0.3"><site site="a3"/><geom geom="wcyl" sidesite="side_in"/><site site="b2"/></spatial>
</tendon>""",
    "limit": """<tendon>
  <spatial name="l1" limited="true" range="RANGE0" margin="0.002" solreflimit="0.01 1" damping="0.2">
    <site site="a1"/><geom geom="wsph"/><site site="b1"/><site site="d1"/></spatial>
  <spatial name="l2" limited="true" range="RANGE1" solreflimit="0.02 0.8" solimplimit="0.8 0.95 0.001 0.5 2">
    <site site="w0"/><site site="a1"/><site site="b1"/><site site="c1"/></spatial>
</tendon>""",
    "actuator": """<tendon>
  <spatial name="m1"><site site="w0"/><site site="a1"/><site site="b1"/><site site="c1"/><site site="d1"/></spatial>
  <spatial name="m2"><site site="a2"/><geom geom="wcyl" sidesite="side_out"/><site site="b1"/><site site="c1"/></spatial>
  <spatial name="m3"><site site="a1"/><geom geom="wsph"/><site site="b1"/><site site="d1"/></spatial>
</tendon>
<actuator>
  <motor tendon="m1" gear="2" ctrlrange="-1 1" ctrllimited="true"/>
  <position tendon="m2" kp="30" kv="1" ctrlrange="0.30 0.42" forcerange="-4 4"/>
  <general tendon="m3" gainprm="5" biasprm="0 -10 -0.5" biastype="affine" ctrlrange="-1 1"/>
</actuator>""",
    "free": """<tendon>
  <spatial name="f1" stiffness="80" damping="0.5"><site site="w0"/><site site="a1"/><site site="b1"/></spatial>
  <spatial name="f2" stiffness="50" damping="0.3"><site site="a3"/><geom geom="wsph" sidesite="side_out2"/><site site="b2"/><site site="d2"/></spatial>
</tendon>""",
}


def _packed_block(n=1250):
    pairs = [("w0", "a1"), ("w1", "a2"), ("a1", "b1"), ("a3", "b2"), ("b1", "c1"), ("c1", "d1"), ("b2", "d2")]
    ts = ['  <spatial name="long"><site site="w0"/><site site="a1"/><site site="b1"/><site site="c1"/><site site="d1"/>'
          '</spatial>']
    for k in range(n):
        s0, s1 = pairs[k % len(pairs)]
        ts.append(f'  <spatial name="k{k}" stiffness="{0.02 + 0.00003 * k:.5f}" damping="0.001"><site site="{s0}"/>'
                  f'<site site="{s1}"/></spatial>')
    return ("<tendon>\n" + "\n".join(ts) + "\n</tendon>\n<actuator>\n"
            '  <motor tendon="long" gear="2" ctrlrange="-1 1" ctrllimited="true"/>\n'
            '  <position tendon="k4" kp="30" kv="1" ctrlrange="0.10 0.20" forcerange="-4 4"/>\n</actuator>')


SCENES["bodies"] = """<tendon>
  <spatial name="g1" stiffness="40" damping="0.3"><site site="w0"/><site site="p1"/><site site="b1"/><site site="d1"/></spatial>
  <spatial name="g2" stiffness="50" damping="0.3"><site site="a3"/><geom geom="psph" sidesite="p_side"/><site site="b2"/></spatial>
  <spatial name="g3" stiffness="60" damping="0.4"><site site="p1"/><geom geom="wsph"/><site site="b1"/><site site="c1"/></spatial>
</tendon>
<actuator><motor tendon="g1" gear="2" ctrlrange="-1 1" ctrllimited="true"/></actuator>
<sensor>
  <framelinvel name="plate_v" objtype="body" objname="plate"/><frameangvel name="plate_w" objtype="body" objname="plate"/>
  <framepos name="plate_p" objtype="xbody" objname="plate" reftype="body" refname="l2"/>
  <framequat name="tip_q" objtype="site" objname="d1" reftype="xbody" refname="plate"/>
  <subtreecom name="plate_com" body="plate"/>
</sensor>"""
SCENES["packed"] = _packed_block()
EULER = {"pulley"}


def scene_xml(name):
    base = ARM.read_text()
    blk = SCENES[name]
    if name in EULER:
        base = base.replace('integrator="implicitfast"', 'integrator="Euler"')
    if name == "bodies":
        base = base.replace('<joint name="h2" axis="0 1 0" range="-2 2"/>',
                            '<joint name="h2x" axis="1 0 0" range="-0.6 0.6"/>'
                            '<joint name="h2" axis="0 1 0" pos="0.02 0 0.01" range="-2 2"/>')
        base = base.replace('<body name="l2" pos="0.3 0 0">',
                            '<body name="plate" pos="0.15 0.0 0.06" euler="10 -20 30"><geom name="plate" type="box" '
                            'size="0.03 0.02 0.005" pos="0.01 0.005 0"/><geom name="psph" type="sphere" size="0.025" '
                            'pos="0.06 0 -0.01"/><site name="p1" pos="-0.01 0.01 0.01"/><site name="p_side" '
                            'pos="0.06 0 0.05"/></body>\n      <body name="l2" pos="0.3 0 0">')
    if name == "free":
        base = base.replace('<joint name="h1" axis="0 1 0" range="-1.5 1.5"/>', '<freejoint name="root"/>')
    if name == "limit":
        free = blk.replace('limited="true"', 'limited="false"').replace('range="RANGE0"', "").replace(
            'range="RANGE1"', "")
        m0 = mujoco.MjModel.from_xml_string(base.replace("</mujoco>", free + "\n</mujoco>"))
        q, v, u = cc.corpus(m0, 64, 0)
        d = mujoco.MjData(m0)
        L = []
        for w in range(len(q)):
            d.qpos[:] = q[w]
            mujoco.mj_kinematics(m0, d)
            mujoco.mj_comPos(m0, d)
            mujoco.mj_tendon(m0, d)
            L.append(d.ten_length.copy())
        L = np.array(L)
        for t in range(2):
            lo, hi = np.percentile(L[:, t], [30, 70])
            blk = blk.replace(f"RANGE{t}", f"{lo:.6f} {hi:.6f}")
    return base.replace("</mujoco>", blk + "\n</mujoco>")


def load_scene(name):
    from nablaengine.envs import convert
    return convert.to_engine(scene_xml(name))


KIN_KERNEL = r"""
#include <cstdio>
#include <cstdlib>
#include <vector>
// the tendons' L and J of each state (substep_pre on struct Sim; forest_engine.spatial's storage)
__global__ void k_kin(int nw, const float* qpos, const float* qvel, float* out) {
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
  for (int i = lane; i < NTEN + NTEN * KTEN; i += 32) {  // L, then J in KTEN-strided rows (either storage layout)
    const int t = (i - NTEN) / KTEN, k = (i - NTEN) % KTEN;
    out[(size_t)w * (NTEN + NTEN * KTEN) + i] = i < NTEN ? TS_L(i) : k < c_tnd[t] ? TS_J(t)[k] : 0.f;
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
  cudaFuncSetAttribute(k_kin, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Sim));
  k_kin<<<nw, 32, sizeof(Sim)>>>(nw, dq, dv, dout);
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


def engine_kin(m, q, v, work):
    """The engine's (L (B, ntendon), J (B, ntendon, nv)) at the states (the used tendons; others NaN)."""
    from nablaengine import cuda
    from nablaengine.forest_engine import adjoint, host, tendon
    ts = tendon.used(m)
    sup = [[d for d, _ in tendon.support_list(m, t)] for t in ts]
    kt = max(len(s) for s in sup)
    consts, lays, sizes = adjoint.engine_constants(m, [])
    work = Path(work)
    (work / "kin.cu").write_text(adjoint.cuda_source(m) + KIN_KERNEL.replace("LOAD_CONSTS", host.loader(lays, sizes)))
    consts.astype(np.float32).tofile(work / "consts.bin")
    cuda.compile_cuda(str(work / "kin.cu"), str(work / "kin"), extra=["-w"])
    nt = len(ts)
    nout = nt + nt * kt
    np.asarray(q, np.float32).tofile(work / "qpos.bin")
    np.asarray(v, np.float32).tofile(work / "qvel.bin")
    r = subprocess.run([str(work / "kin"), str(work), str(len(q)), str(nout), str(len(consts))], capture_output=True,
                       text=True)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    o = np.fromfile(work / "out.bin", np.float32).reshape(len(q), nout)
    L = np.full((len(q), m.ntendon), np.nan)
    J = np.full((len(q), m.ntendon, m.nv), np.nan)
    for k, t in enumerate(ts):
        L[:, t] = o[:, k]
        J[:, t] = 0.0
        for j, d in enumerate(sup[k]):
            J[:, t, d] = o[:, nt + kt * k + j]
    return L, J


def c_kin(m, q):
    d = mujoco.MjData(m)
    L, J = [], []
    for w in range(len(q)):
        d.qpos[:] = q[w]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_tendon(m, d)
        L.append(d.ten_length.copy())
        Jd = np.zeros((m.ntendon, m.nv))
        for t in range(m.ntendon):
            for k in range(int(m.ten_J_rownnz[t])):
                Jd[t, int(m.ten_J_colind[m.ten_J_rowadr[t] + k])] = d.ten_J[m.ten_J_rowadr[t] + k]
        J.append(Jd)
    return np.array(L), np.array(J)


def wrapnum(m, q):
    """Per tendon: {path points (MuJoCo C ten_wrapnum): states} over the corpus (a wrap that engages adds its two
    tangent points)."""
    d = mujoco.MjData(m)
    out = [dict() for _ in range(m.ntendon)]
    for w in range(len(q)):
        d.qpos[:] = q[w]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_tendon(m, d)
        for t in range(m.ntendon):
            k = int(d.ten_wrapnum[t])
            out[t][k] = out[t].get(k, 0) + 1
    return out


def mjw_kin(m, q, v, py, work):
    work = Path(work)
    mujoco.mj_saveModel(m, str(work / "mt.mjb"), None)
    np.savez(work / "st.npz", qpos=q, qvel=v)
    r = subprocess.run([py, "-I", str(REPO / "scripts" / "forest" / "mjw_tendon.py"), str(work / "mt.mjb"),
                        str(work / "st.npz"), str(work / "mt.npz")], capture_output=True, text=True,
                       env={k: x for k, x in os.environ.items() if k != "PYTHONPATH"})
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    z = np.load(work / "mt.npz")
    return z["ten_length"], z["ten_J"], json.loads(r.stdout.strip().splitlines()[-1])


def abs_stats(x):
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    return dict(median=float(np.median(x)), p90=float(np.percentile(x, 90)), max=float(x.max())) if x.size else {}


def cmd_kin(a, cv):
    m = cv.model
    q, v, u = cc.corpus(cv.original, a.states, a.seed)
    with tempfile.TemporaryDirectory() as tmp:
        Le, Je = engine_kin(m, q, v, tmp)
        Lc, Jc = c_kin(m, q)
        Ls, Js = c_kin(cv.original, q)
        used = ~np.isnan(Le[0])
        out = dict(tendons=int(used.sum()), L_vs_c=abs_stats(np.abs(Le[:, used] - Lc[:, used])),
                   J_vs_c=abs_stats(np.abs(Je[:, used] - Jc[:, used]).max(-1)),
                   c_converted_vs_source_L=abs_stats(np.abs(Lc - Ls)), wrapnum=wrapnum(m, q), c_converted_vs_source_J=abs_stats(np.abs(Jc - Js).max(-1)),
                   L_range=[float(np.nanmin(Lc)), float(np.nanmax(Lc))], J_scale=float(np.abs(Jc).max()))
        if a.mjw_python:
            Lw, Jw, ver = mjw_kin(m, q, v, a.mjw_python, tmp)
            out.update(L_vs_mjw=abs_stats(np.abs(Le[:, used] - Lw[:, used])),
                       J_vs_mjw=abs_stats(np.abs(Je[:, used] - Jw[:, used]).max(-1)),
                       mjw_vs_c_L=abs_stats(np.abs(Lw - Lc)), mjw_vs_c_J=abs_stats(np.abs(Jw - Jc).max(-1)), versions=ver)
    return out, out["J_vs_c"]["median"] < 1e-5


def c_steps(m, q, v, u, n):
    d = mujoco.MjData(m)
    V = []
    for w in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = q[w], v[w], u[w]
        for _ in range(n):
            mujoco.mj_step(m, d)
        V.append(d.qvel.copy())
    return np.array(V)


def mjw_steps(m, q, v, u, n, py):
    with tempfile.TemporaryDirectory() as tmp:
        mujoco.mj_saveModel(m, os.path.join(tmp, "m.mjb"), None)
        np.savez(os.path.join(tmp, "s.npz"), qpos=q, qvel=v, ctrl=u)
        r = subprocess.run([py, "-I", str(REPO / "scripts" / "eq" / "mjw_step.py"), os.path.join(tmp, "m.mjb"),
                            os.path.join(tmp, "s.npz"), os.path.join(tmp, "o.npz"), "--substeps", str(n), "--iterations",
                            str(int(m.opt.iterations))], capture_output=True, text=True,
                           env={k: x for k, x in os.environ.items() if k != "PYTHONPATH"})
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
        return np.load(os.path.join(tmp, "o.npz"))["qvel"]


def cmd_fwd(a, cv):
    m = cv.model
    q, v, u = cc.corpus(cv.original, a.states, a.seed)
    info = cc.engine_init(m)
    out, ok = {}, True
    for n in (1, 4):
        _, ve, _ = cc.engine_step(m, info, q, v, u, substeps=n)
        ve = np.asarray(ve, np.float64)
        vc = c_steps(m, q, v, u, n)
        vs = c_steps(cv.original, q, v, u, n)
        r = dict(engine_vs_c=cc.stats(cc.rel(ve, vc)), c_converted_vs_source=cc.stats(cc.rel(vc, vs)),
                 finite=bool(np.isfinite(ve).all()))
        if a.mjw_python:
            vw = mjw_steps(m, q, v, u, n, a.mjw_python)
            r.update(engine_vs_mjw=cc.stats(cc.rel(ve, vw)), mjw_vs_c=cc.stats(cc.rel(vw, vc)))
        ok &= r["engine_vs_c"]["median"] < 1e-4
        out[f"substeps_{n}"] = r
    out["tape_forward_bitwise"] = bool(cg.tape_bitwise(m, info, q, v, u, 4))
    return out, ok


def cmd_fd(a, cv):
    m = cv.model
    q, v, u = cc.corpus(cv.original, a.states, a.seed)
    info = cc.engine_init(m)
    mm = cg.tight(m)
    out, ok = {}, True
    for S in a.substeps:
        rng = np.random.default_rng(a.seed + 7)
        gq = rng.normal(size=(len(q), m.nq))
        gq[:, cg.free_quats(m)] = 0.0
        gv = rng.normal(size=(len(q), m.nv))
        _, _, (bq, bv, bu) = cg.engine_vjp(m, info, q, v, u, gq, gv, S, clamp_kv=1, normal=0)
        e, eq_, ev, eu = [], [], [], []
        nv, nu = m.nv, m.nu
        for w in range(len(q)):
            q64, v64, u64 = (x[w].astype(np.float64) for x in (q, v, u))
            ge = np.concatenate([cg.q_to_tangent(m, q64, bq[w]), bv[w], bu[w]])
            gf = cg.fd_grad(mm, q64, v64, u64, gq[w], gv[w], S)
            e.append(cg.err(ge, gf))
            eq_.append(cg.err(ge, gf, slice(0, nv)))
            ev.append(cg.err(ge, gf, slice(nv, 2 * nv)))
            if nu:
                eu.append(cg.err(ge, gf, slice(2 * nv, 2 * nv + nu)))
        r = dict(engine_vs_fd=cc.stats(e), q=cc.stats(eq_), v=cc.stats(ev))
        if eu:
            r["ctrl"] = cc.stats(eu)
        ok &= r["engine_vs_fd"]["median"] < 1e-3
        out[f"substeps_{S}"] = r
    return out, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("kin", "fwd", "fd"))
    ap.add_argument("--scene", default="site", choices=sorted(SCENES))
    ap.add_argument("--all", action="store_true", help="every scene, one subprocess each")
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--substeps", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mjw-python", default="", help="a Python with mujoco_warp (3.17) for the kin / fwd reference")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.all:
        res, ok = {}, True
        for sc in SCENES:
            argv = [sys.executable, __file__, a.cmd, "--scene", sc, "--states", str(a.states), "--seed", str(a.seed),
                    "--substeps", *map(str, a.substeps)] + (["--mjw-python", a.mjw_python] if a.mjw_python else [])
            r = subprocess.run(argv, capture_output=True, text=True)
            try:
                res[sc] = json.loads(r.stdout)
            except json.JSONDecodeError:
                res[sc] = dict(error=(r.stdout + r.stderr)[-3000:], **{"pass": False})
            ok &= bool(res[sc].get("pass"))
            print(sc, json.dumps(res[sc].get("result", res[sc]))[:600], file=sys.stderr, flush=True)
        rep = dict(command=a.cmd, scenes=res)
    else:
        cv = load_scene(a.scene)
        res, ok = dict(kin=cmd_kin, fwd=cmd_fwd, fd=cmd_fd)[a.cmd](a, cv)
        rep = dict(command=a.cmd, scene=a.scene, states=a.states, seed=a.seed, result=res, notes=cv.notes)
    from nablaengine import provenance
    rep["provenance"] = provenance.provenance()
    rep["pass"] = bool(ok)
    txt = json.dumps(rep, indent=1, default=str)
    print(txt)
    if a.out:
        o = Path(a.out)
        if o.exists():
            sys.exit(f"{o} exists (not overwritten)")
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(txt)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

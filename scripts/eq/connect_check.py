"""Connect equalities in the forest engine's forward (forest_engine.equality), against float64 MuJoCo C (the arbiter)
and MuJoCo Warp (dense Jacobian, primed: its Jdot v at the current velocities):"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def load(path, tendon_to_joint=None, no_fluid=False, uniform_limits=False, drop_spatial=False, condim6_to=None):
    from nablaengine.envs import convert
    p = Path(path).resolve()
    xml = p.read_text()
    assets = {str(q.relative_to(p.parent)): q.read_bytes() for q in p.parent.rglob("*.xml")}
    cwd = os.getcwd()
    os.chdir(p.parent)
    try:
        if tendon_to_joint:
            xml, assets = drop_tendons(xml, assets, tendon_to_joint)
        if drop_spatial:
            xml, assets = drop_spatial_tendons(xml, assets)
        if no_fluid:
            xml, assets = drop_fluid(xml, assets)
        if uniform_limits:
            xml = same_limits(xml, assets)
        cv = convert.to_engine(xml, assets, condim6_to=condim6_to)
    finally:
        os.chdir(cwd)
    return cv


def drop_tendons(xml, assets, joint):
    """xml (and its included assets) without <tendon> sections, actuators on a tendon moved to joint."""
    def edit(text):
        root = ET.fromstring(text)
        for parent in list(root.iter()):
            for el in list(parent):
                if el.tag == "tendon":
                    parent.remove(el)
                elif el.get("tendon") is not None and parent.tag == "actuator":
                    del el.attrib["tendon"]
                    el.set("joint", joint)
                    el.set("forcelimited", "false")
        return ET.tostring(root, encoding="unicode")
    return edit(xml), {k: (edit(v.decode()).encode() if k.endswith(".xml") else v) for k, v in assets.items()}


def drop_spatial_tendons(xml, assets):
    """xml (and its included assets) without spatial tendons (capability H25); the fixed tendons stay."""
    def edit(text):
        root = ET.fromstring(text)
        for parent in list(root.iter("tendon")):
            for el in list(parent):
                if el.tag == "spatial":
                    parent.remove(el)
        return ET.tostring(root, encoding="unicode")
    return edit(xml), {k: (edit(v.decode()).encode() if k.endswith(".xml") else v) for k, v in assets.items()}


def drop_fluid(xml, assets):
    """xml (and its included assets) with the options' fluid viscosity / density removed."""
    def edit(text):
        root = ET.fromstring(text)
        for el in root.iter("option"):
            for at in ("viscosity", "density"):
                el.attrib.pop(at, None)
        return ET.tostring(root, encoding="unicode")
    return edit(xml), {k: (edit(v.decode()).encode() if k.endswith(".xml") else v) for k, v in assets.items()}


def same_limits(xml, assets):
    """xml (MjSpec's text) with every limited hinge / slide taking the first one's solreflimit / solimplimit (--uniform-
    limits: earlier runs, when the forest engine had one set for all joint limits)."""
    m0 = mujoco.MjModel.from_xml_string(xml, assets)
    lim = [j for j in range(m0.njnt) if m0.jnt_limited[j] and int(m0.jnt_type[j]) in (2, 3)]
    spec = mujoco.MjSpec.from_string(xml, assets)
    for j in lim:
        js = spec.joint(m0.joint(j).name)
        js.solref_limit = list(m0.jnt_solref[lim[0]])
        js.solimp_limit = list(m0.jnt_solimp[lim[0]])
    return spec.to_xml()


def home(m):
    k = next((i for i in range(m.nkey) if m.key(i).name == "home"), 0 if m.nkey else None)
    return (m.key_qpos[k].copy(), m.key_ctrl[k].copy()) if k is not None else (m.qpos0.copy(), np.zeros(m.nu))


def ctrl_range(m):
    lo, hi = np.full(m.nu, -1.0), np.full(m.nu, 1.0)
    for a in range(m.nu):
        if m.actuator_ctrllimited[a]:
            lo[a], hi[a] = m.actuator_ctrlrange[a]
    return lo, hi


def corpus(m, n, seed, t_min=0.1, every=10):
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    q0, _ = home(m)
    lo, hi = ctrl_range(m)
    Q, V, U = [], [], []
    while len(Q) < n:
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q0
        d.qvel[:] = rng.normal(0, 0.5, m.nv)
        t, k = 0.0, 0
        while t < t_min + 0.4 and len(Q) < n:
            if k % 50 == 0:
                d.ctrl[:] = rng.uniform(lo, hi)
            mujoco.mj_step(m, d)
            t, k = t + m.opt.timestep, k + 1
            if t >= t_min and k % every == 0:
                Q.append(d.qpos.copy()); V.append(d.qvel.copy()); U.append(d.ctrl.copy())
    f = lambda x: np.asarray(x, np.float32)
    return f(Q), f(V), f(U)


def connect_eqs(m):
    return [e for e in range(m.neq) if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_CONNECT) and m.eq_active0[e]]


def points(m, d, e):
    """(p1, b1, p2, b2) of connect e (MuJoCo C kinematics in d)."""
    if int(m.eq_objtype[e]) == int(mujoco.mjtObj.mjOBJ_SITE):
        s1, s2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        return d.site_xpos[s1].copy(), int(m.site_bodyid[s1]), d.site_xpos[s2].copy(), int(m.site_bodyid[s2])
    b1, b2 = int(m.eq_obj1id[e]), max(int(m.eq_obj2id[e]), 0)
    p1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[e, 0:3]
    p2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ m.eq_data[e, 3:6]
    return p1, b1, p2, b2


def residual(m, d, q):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    return [float(np.linalg.norm(np.subtract(*points(m, d, e)[::2]))) for e in connect_eqs(m)]


def c_rows(m, q, v):
    """MuJoCo C's connect rows at each state: pos, D, aref (3 each), J (3, nv) and Jdot v (3), per connect."""
    jac0 = int(m.opt.jacobian)
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    d = mujoco.MjData(m)
    out = []
    try:
        for w in range(len(q)):
            d.qpos[:] = q[w]; d.qvel[:] = v[w]; d.ctrl[:] = 0
            mujoco.mj_forward(m, d)
            J = d.efc_J.reshape(d.nefc, m.nv)
            rows = []
            for e in connect_eqs(m):
                r = np.nonzero((d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == e))[0]
                assert len(r) == 3
                p1, b1, p2, b2 = points(m, d, e)
                jd = np.zeros(3)
                for p, b, sg in ((p1, b1, 1.0), (p2, b2, -1.0)):
                    jp = np.zeros((3, m.nv))
                    mujoco.mj_jacDot(m, d, jp, None, p, b)
                    jd += sg * jp @ d.qvel
                rows.append(dict(pos=d.efc_pos[r].copy(), D=d.efc_D[r].copy(), aref=d.efc_aref[r].copy(),
                                 J=J[r].copy(), vel=J[r] @ d.qvel, jdv=jd))
            out.append(rows)
    finally:
        m.opt.jacobian = jac0
    return out


def c_step(m, q, v, u):
    d = mujoco.MjData(m)
    Q, V, A = [], [], []
    for w in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q[w]; d.qvel[:] = v[w]; d.ctrl[:] = u[w]
        d.qacc_warmstart[:] = 0
        mujoco.mj_step(m, d)
        Q.append(d.qpos.copy()); V.append(d.qvel.copy()); A.append(d.qacc.copy())
    return np.array(Q), np.array(V), np.array(A)


def rel(x, ref):
    return (np.abs(np.asarray(x, np.float64) - ref) / np.maximum(np.abs(ref), 1.0)).max(axis=-1)


def stats(x):
    x = np.asarray(x, np.float64).ravel()
    if not len(x):
        return None
    return dict(median=float(np.median(x)), p90=float(np.quantile(x, 0.9)), max=float(x.max()), n=int(len(x)))


def engine_init(m):
    from nablaengine import engines, jax_ffi
    engines.prefer_forest(m)
    return jax_ffi.init(model=m, foot_pairs=[])


def engine_step(m, info, q, v, u, substeps=1, warm=None):
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    B = len(q)
    cw = int(info.get("com_words", 3))
    out = jax_ffi.step(jnp.asarray(q), jnp.asarray(v), jnp.zeros((B, m.nv), jnp.float32) if warm is None else warm,
                       jnp.asarray(u), jnp.ones((B,), jnp.float32), jnp.zeros((B, cw), jnp.float32),
                       substeps=substeps, iterations=int(m.opt.iterations))
    return out[0], out[1], out[2]


ROWS_KERNEL = r"""
#include <cstdio>
#include <cstdlib>
#include <vector>
// the connect rows of each state (substep_pre on struct Sim): nefc, nefc_dof, per connect row D, aref, its J over the
// support, and ec_eval's pos, J v, Jdot v
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
  const int per = 3 * (2 + KEQC + 3);
  for (int c = lane; c < NEQC; c += 32) {
    float* oc = o + 2 + per * c;
    float p[3], v[3], jd[3], J[3 * KEQC];
    ec_eval<float>(c, s.Q, s.X, s.qvel, p, v, jd, J, -1);
    for (int r = 0; r < 3; r++) {
      const int row = s.nefc_dof + NEQJ + 3 * c + r;
      float* orow = oc + r * (2 + KEQC + 3);
      orow[0] = s.efc_D[row]; orow[1] = s.efc_aref[row];
      for (int k = 0; k < KEQC; k++) orow[2 + k] = ec_J(s)[(3 * c + r) * KEQC + k];
      orow[2 + KEQC] = p[r]; orow[3 + KEQC] = v[r]; orow[4 + KEQC] = jd[r];
    }
  }
  if (lane == 0) { o[0] = s.nefc; o[1] = s.nefc_dof; }
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
    """The engine's connect rows at the states: (B, nconnect, 3, [D, aref, J (KEQC), pos, vel, jdv]), support lists."""
    from nablaengine import cuda
    from nablaengine.forest_engine import adjoint, equality, host
    cs = equality.connects(m)
    kc = max(len(c["dofs"]) for c in cs)
    consts, lays, sizes = adjoint.engine_constants(m, [])
    work = Path(work)
    (work / "rows.cu").write_text(adjoint.cuda_source(m) + ROWS_KERNEL.replace("LOAD_CONSTS", host.loader(lays, sizes)))
    consts.astype(np.float32).tofile(work / "consts.bin")
    cuda.compile_cuda(str(work / "rows.cu"), str(work / "rows"), extra=["-w"])
    per = 3 * (2 + kc + 3)
    nout = 2 + per * len(cs)
    np.asarray(q, np.float32).tofile(work / "qpos.bin")
    np.asarray(v, np.float32).tofile(work / "qvel.bin")
    r = subprocess.run([str(work / "rows"), str(work), str(len(q)), str(nout), str(len(consts))], capture_output=True,
                       text=True)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    o = np.fromfile(work / "out.bin", np.float32).reshape(len(q), nout)
    return o[:, 2:].reshape(len(q), len(cs), 3, 2 + kc + 3), [c["dofs"] for c in cs], kc


def run_mjw(m, q, v, u, py, work):
    work = Path(work)
    mujoco.mj_saveModel(m, str(work / "model.mjb"), None)
    np.savez(work / "states.npz", qpos=q, qvel=v, ctrl=u)
    script = Path(__file__).resolve().parent / "mjw_step.py"
    r = subprocess.run([py, str(script), str(work / "model.mjb"), str(work / "states.npz"), str(work / "mjw.npz"),
                        "--substeps", "1", "--iterations", str(int(m.opt.iterations)), "--dense", "--prime"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ""})
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return dict(np.load(work / "mjw.npz")), json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mjw-python", default=None)
    ap.add_argument("--states", type=int, default=256)
    ap.add_argument("--drift-worlds", type=int, default=16)
    ap.add_argument("--drift-steps", type=int, default=1000)
    ap.add_argument("--tendon-to-joint", default=None)
    ap.add_argument("--no-fluid", action="store_true")
    ap.add_argument("--uniform-limits", action="store_true",
                    help="every joint limit on the first set (connect_check.py; earlier runs only: the engine reads per-joint sets)")
    ap.add_argument("--drop-spatial", action="store_true", help="remove the spatial tendons (the fixed ones stay)")
    ap.add_argument("--condim6-to", type=int, default=None, choices=(3,),
                    help="convert condim-6 pairs to condim 3 (convert.to_engine(condim6_to=3); default: kept, and the "
                         "forest engine refuses them, capability H17)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cv = load(a.model, a.tendon_to_joint, a.no_fluid, a.uniform_limits, a.drop_spatial, a.condim6_to)
    m, mo = cv.model, cv.original
    from nablaengine.forest_engine import equality
    rep = dict(model=os.path.relpath(Path(a.model).resolve(), Path(__file__).resolve().parents[2]), nv=int(m.nv),
               nbody=int(m.nbody), neq=int(m.neq), connects=len(equality.connects(m)), joint_eqs=len(equality.rows(m)),
               notes=cv.notes, tendon_to_joint=a.tendon_to_joint, no_fluid=a.no_fluid, uniform_limits=a.uniform_limits, drop_spatial=a.drop_spatial, mujoco=mujoco.__version__, seed=a.seed)
    q, v, u = corpus(mo, a.states, a.seed)
    info = engine_init(m)
    qe, ve, ae = (np.asarray(x, np.float64) for x in engine_step(m, info, q, v, u))
    qc, vc, ac = c_step(mo, q, v, u)
    cd = sorted({d for c in equality.connects(m) for d in c["dofs"]})
    rep["forward_vs_C"] = dict(qvel=stats(rel(ve, vc)), qacc=stats(rel(ae, ac)), qvel_chains=stats(rel(ve[:, cd], vc[:, cd])),
                               qacc_chains=stats(rel(ae[:, cd], ac[:, cd])))
    ncon = []
    dd = mujoco.MjData(mo)
    for w in range(len(q)):
        dd.qpos[:] = q[w]
        mujoco.mj_forward(mo, dd)
        ncon.append(dd.ncon)
    rep["contacts_mean"] = float(np.mean(ncon))
    with tempfile.TemporaryDirectory(prefix="connect_check_") as work:
        er, sup, kc = engine_rows(m, q, v, work)
        cr = c_rows(mo, q.astype(np.float64), v.astype(np.float64))
        err = dict(pos=[], D=[], aref=[], aref_abs=[], J=[], vel=[], jdv=[], jdv_abs=[])
        for w in range(len(q)):
            for c, rows in enumerate(cr[w]):
                E = er[w, c]
                nd = len(sup[c])
                Jc = rows["J"][:, sup[c]]
                err["pos"].append(np.abs(E[:, 2 + kc] - rows["pos"]).max())
                err["D"].append((np.abs(E[:, 0] - rows["D"]) / rows["D"]).max())
                err["aref"].append(np.abs(E[:, 1] - rows["aref"]).max() / max(np.abs(rows["aref"]).max(), 1.0))
                err["aref_abs"].append(np.abs(E[:, 1] - rows["aref"]).max())
                err["J"].append(np.abs(E[:, 2:2 + nd] - Jc).max())
                err["vel"].append(np.abs(E[:, 3 + kc] - rows["vel"]).max() / max(np.abs(rows["vel"]).max(), 1.0))
                err["jdv"].append(np.abs(E[:, 4 + kc] - rows["jdv"]).max() / max(np.abs(rows["jdv"]).max(), 1.0))
                err["jdv_abs"].append(np.abs(E[:, 4 + kc] - rows["jdv"]).max())
        rep["rows_engine_vs_C"] = {k: stats(x) for k, x in err.items()}
        rep["rows_C_scale"] = dict(pos=stats([np.abs(r["pos"]).max() for rr in cr for r in rr]),
                                   aref=stats([np.abs(r["aref"]).max() for rr in cr for r in rr]),
                                   jdv=stats([np.abs(r["jdv"]).max() for rr in cr for r in rr]))
        if a.mjw_python:
            z, ver = run_mjw(mo, q, v, u, a.mjw_python, work)
            rep["mjw"] = ver
            rep["mjw_dense_primed_vs_C"] = dict(qvel=stats(rel(z["qvel"], vc)), qacc=stats(rel(z["qacc"], ac)),
                                                qvel_chains=stats(rel(z["qvel"][:, cd], vc[:, cd])),
                                                qacc_chains=stats(rel(z["qacc"][:, cd], ac[:, cd])))
            rep["engine_vs_mjw_dense_primed"] = dict(qvel=stats(rel(ve, z["qvel"])), qacc=stats(rel(ae, z["qacc"])),
                                                     qvel_chains=stats(rel(ve[:, cd], z["qvel"][:, cd])),
                                                     qacc_chains=stats(rel(ae[:, cd], z["qacc"][:, cd])))
            mr = dict(pos=[], D=[], aref=[])
            for w in range(len(q)):
                for c, e in enumerate(connect_eqs(mo)):
                    R = z["eq_rows3"][w, e]
                    rows = cr[w][c]
                    mr["pos"].append(np.abs(R[:, 0] - rows["pos"]).max())
                    mr["D"].append((np.abs(R[:, 1] - rows["D"]) / rows["D"]).max())
                    mr["aref"].append(np.abs(R[:, 2] - rows["aref"]).max() / max(np.abs(rows["aref"]).max(), 1.0))
            rep["rows_mjw_vs_C"] = {k: stats(x) for k, x in mr.items()}
    nd_ = min(a.drift_worlds, len(q))
    rng = np.random.default_rng(a.seed + 1)
    lo, hi = ctrl_range(mo)
    ud = rng.uniform(lo, hi, (nd_, mo.nu)).astype(np.float32)
    qd, vd = q[:nd_].copy(), v[:nd_].copy()
    import jax.numpy as jnp
    warm = jnp.zeros((nd_, m.nv), jnp.float32)
    qq, vv = jnp.asarray(qd), jnp.asarray(vd)
    de = mujoco.MjData(mo)
    res_e = np.zeros((nd_, len(connect_eqs(mo))))
    finite = True
    for k in range(a.drift_steps // 10):
        qq, vv, warm = engine_step(m, info, qq, vv, jnp.asarray(ud), substeps=10, warm=warm)
        qn = np.asarray(qq, np.float64)
        finite &= bool(np.isfinite(qn).all())
        for w in range(nd_):
            res_e[w] = np.maximum(res_e[w], residual(mo, de, qn[w]))
    res_c = np.zeros_like(res_e)
    res_c_end = []
    dc = mujoco.MjData(mo)
    dk = mujoco.MjData(mo)
    for w in range(nd_):
        mujoco.mj_resetData(mo, dc)
        dc.qpos[:] = qd[w]; dc.qvel[:] = vd[w]; dc.ctrl[:] = ud[w]
        for k in range(a.drift_steps):
            mujoco.mj_step(mo, dc)
            if (k + 1) % 10 == 0:
                res_c[w] = np.maximum(res_c[w], residual(mo, dk, dc.qpos))
        res_c_end.append(max(residual(mo, dk, dc.qpos) or [0]))
    rep["drift"] = dict(steps=a.drift_steps, worlds=nd_, engine_finite=finite,
                        engine_max_residual=stats(res_e.max(axis=1)), C_max_residual=stats(res_c.max(axis=1)),
                        start_residual=stats([max(residual(mo, dk, qd[w]) or [0]) for w in range(nd_)]),
                        engine_end_residual=stats([max(residual(mo, dk, qn[w]) or [0]) for w in range(nd_)]),
                        C_end_residual=stats(res_c_end))
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

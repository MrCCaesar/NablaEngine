"""The noslip pass's Gauss-Seidel order with per-world parameters (noslip_check.py params' worlds: --n worlds, box size x
mass factors, states settled by MuJoCo C with noslip 10; --scene arm: the arm holding its box, or grip)."""
import argparse
import copy
import json
import sys
import types
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="arm", help="a noslip_check.py params scene (arm, grip)")
ap.add_argument("--noslip", type=int, default=10)
ap.add_argument("--n", type=int, default=64, help="worlds")
ap.add_argument("--sizes", default="0.95,1.0,1.05", help="box size factors")
ap.add_argument("--masses", default="0.5,1.0,2.0", help="box mass / moment factors")
ap.add_argument("--newton", type=int, default=100, help="the engine's Newton iterations")
ap.add_argument("--worlds", default="all")
ap.add_argument("--out", default="")
A = ap.parse_args()
import noslip_check as N
from nablaengine import jax_ffi
from nablaengine.params import PhysicsParams
from nablaengine.forest_engine import tape as T
NS = A.noslip
src = (HERE / "noslip_ref.py").read_text()
anchor = "    niter = [0] * nisl\n"
assert src.count(anchor) == 1
src = src.replace(anchor, "    blocks = PERM(blocks)\n" + anchor)
ns = {"__name__": "noslip_ref_perm"}
exec(compile(src, "noslip_ref_perm", "exec"), ns)
CUR = {}


def perm(blocks):
    if CUR.get("rank") is None:
        return blocks
    out = []
    for bl in blocks:
        fr = [b for b in bl if b[0] == "f"]
        ps = [b for b in bl if b[0] == "p"]
        ps = sorted(ps, key=lambda b: (CUR["rank"][CUR["efc_id"][b[1]]], b[1]))
        out.append(fr + ps)
    return out


ns["PERM"] = perm
a = types.SimpleNamespace(scene=A.scene, noslip=NS, n=8, seed=0)
m0, g, st0, _ = N.params_model(a)
m0.opt.noslip_tolerance = 0.0
facs, mfacs = [float(x) for x in A.sizes.split(",")], [float(x) for x in A.masses.split(",")]
n, settle = A.n, 5
W = [(facs[i % len(facs)], mfacs[(i // len(facs)) % len(mfacs)]) for i in range(n)]
models = {w: N.world_model(m0, g, *w) for w in set(W)}
P = PhysicsParams(("object_size", "torso_mass", "torso_inertia"), object_size_max=tuple(m0.geom_size[g] * max(facs) * 1.001))
info = jax_ffi.init(model=m0, foot_pairs=[], com_body="box", params=P)
b = int(m0.geom_bodyid[g])
cw0 = int(info["com_words"]) - int(info.get("param_words", 0))
com = np.concatenate([np.zeros((n, cw0), np.float32), P.pack(m0.nv, object_size=np.array([m0.geom_size[g] * f for f, _ in W]),
      torso_mass=np.array([[m0.body_mass[b] * mf] for _, mf in W]), torso_inertia=np.array([m0.body_inertia[b] * mf for _, mf in W]))], 1)
st = {k: [] for k in ("qpos", "qvel", "ctrl", "warm")}
for i, w in enumerate(W):
    mm = models[w]; d = mujoco.MjData(mm); j = i % len(st0["qpos"])
    mm.opt.noslip_iterations = 10
    d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = st0["qpos"][j], st0["qvel"][j], st0["ctrl"][j], st0["warm"][j]
    for _ in range(settle + 5 * (i // len(set(W)))):
        mujoco.mj_step(mm, d)
    mm.opt.noslip_iterations = NS
    for k, v in (("qpos", d.qpos), ("qvel", d.qvel), ("ctrl", d.ctrl), ("warm", d.qacc_warmstart)):
        st[k].append(v.copy())
st = {k: np.array(v) for k, v in st.items()}
f32 = N.f32
r = jax_ffi.forward(f32(st["qpos"]), f32(st["qvel"]), f32(st["warm"]), f32(st["ctrl"]), f32(np.ones(n)), f32(com),
                    iterations=A.newton, substeps=1, tape=True)
v = np.asarray(r[1], np.float64)
dec = T.decode(np.asarray(r[-1]), m0)
pos_e = dec["ns_con"][..., :3]
inf = np.asarray(r[3])
res = {}
worlds = [int(x) for x in A.worlds.split(",")] if A.worlds != "all" else list(range(n))
for i in worlds:
    w = W[i]; mm = copy.deepcopy(models[w]); mm.opt.tolerance, mm.opt.iterations = 1e-12, 100

    def ref(rank):
        d = mujoco.MjData(mm)
        d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = st["qpos"][i], st["qvel"][i], st["ctrl"][i], st["warm"][i]
        mm.opt.noslip_iterations = 0; mujoco.mj_forward(mm, d); mm.opt.noslip_iterations = NS
        CUR["efc_id"] = np.asarray(d.efc_id).copy()
        CUR["rank"] = rank
        ns["step"](mm, d, NS, 0.0)
        return d

    d0 = ref(None)
    nc = int(d0.ncon)
    pc = np.array([c.pos for c in d0.contact[:nc]])
    ne = int(inf[i, 1])
    pe = pos_e[i, :ne]
    dist = np.linalg.norm(pe[:, None] - pc[None], axis=2)
    match = dist.argmin(1)
    ok = ne == nc and len(set(match.tolist())) == nc and float(dist.min(1).max()) < 1e-4
    d1 = None
    if ok:
        rank = np.empty(nc, int)
        rank[match] = np.arange(nc)
        d1 = ref(rank)
    den = max(np.linalg.norm(d0.qvel), 1e-2)
    res[i] = dict(size=w[0], mass=w[1], ncon=nc, matched=bool(ok), max_pos_dist=float(dist.min(1).max()),
                  same_order=bool(ok and (match == np.arange(nc)).all()),
                  rel_c_order=float(np.linalg.norm(v[i] - d0.qvel) / den),
                  rel_engine_order=float(np.linalg.norm(v[i] - d1.qvel) / den) if ok else None)
    print(i, json.dumps(res[i]), flush=True)
x = [r_["rel_engine_order"] for r_ in res.values() if r_["rel_engine_order"] is not None]
y = [r_["rel_c_order"] for r_ in res.values()]
summ = dict(noslip_iterations=NS, newton_engine=A.newton, n=len(res), matched=len(x),
            engine_order_median=float(np.median(x)), engine_order_max=max(x), c_order_median=float(np.median(y)),
            c_order_max=max(y), same_order=sum(r_["same_order"] for r_ in res.values()))
print("SUMMARY", json.dumps(summ))
if A.out:
    Path(A.out).write_text(json.dumps(dict(section="order", scene=A.scene, **summ, provenance=N.provenance(),
                                           worlds=res), indent=1))

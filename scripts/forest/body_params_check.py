"""Per-world body inertials, sole thickness and per-world invweight0 (PhysicsParams fields body_inertial,
sole_thickness, PhysicsParams(invweight0="per_world"); forest_engine.body_params) on a G1 + box scene:
scripts/forest/models/g1box.xml (the G1 of fixtures/mjlab: capsule feet, 7 per foot; a free box with every robot geom
and the floor paired with it; com_body the box); with --box2 before the command, a second free box "box2" (5 cm
half-extents, 0.3 kg) paired with the box and the floor (box-box contacts; 33 bodies: the forest engine's >32-body
code, which does not take these parameters yet).

Variants: each world takes one of VARIANTS (box size and mass with the box's moments as a uniform box's, the masses /
moments / COMs of the feet, wrists and torso link, the sole thickness). Per variant a MuJoCo model with those values
and mj_setConst (variant_model: invweight0 etc. recomputed as a per-model build has them; --nominal-meaninertia
keeps the base model's stat.meaninertia, the solver's tolerance scale, which a parameter build keeps), and one
engine build with every field whose batch mixes the variants. One engine library per process: the builds run in
separate processes and leave dumps.

  iwtab    (host) the iw0_world algorithm in numpy on the build's tables (body_params.iw_tables) against
           mj_setConst's body_invweight0 / dof_invweight0, on the variants' models (float32 rounding equality)
  corpus   (MuJoCo C) the states: upright G1 states of a G1 corpus (--g1-corpus, default nablaengine.fixtures'
           default, mf_s0: root height >= 0.65 m), the box of world i at size SIZES[i % 3] placed in front of the
           robot's feet / shins (on the floor), pushed towards the robot, box2 on its top face, then 0-6 control steps
           of each world's MuJoCo model (the corpus's position targets) to settle into contact; each world a variant
           of its box size (--cross K: every world variant K)
  run      (JAX) --mode params (the fields, invweight0 per world unless --iw model) or --mode plain --variant K (the
           variant's model, no parameters); forward + backward of fixed cotangents; --dump NPZ
  compare  a params dump against the plain dumps (--plain K=NPZ ...) on each variant's worlds: bitwise / max diff
  ref      (MuJoCo Warp) one control step of each world with its variant's model
  fwd      a dump against REF: qvel relative error per variant
  pgrad    (torch, MuJoCo C float64) the parameters' gradients (g_params of the new fields) against central
           differences of each world's model with the field changed (no mj_setConst: the gradients treat invweight0
           as a constant); --iw model builds (see the module docstring of body_params)
  parity   (torch) a torch step's outputs and gradients vs a JAX dump of the same build (bitwise)
  time     (JAX) forward + backward per call at --nworld worlds, --mode params / plain

  python scripts/forest/body_params_check.py corpus --out W/corpus.npz --nworld 64
  python scripts/forest/body_params_check.py run --corpus W/corpus.npz --dump W/params.npz
  python scripts/forest/body_params_check.py run --corpus W/corpus.npz --mode plain --variant 0 --dump W/plain0.npz
  python scripts/forest/body_params_check.py compare --corpus W/corpus.npz --dump W/params.npz --plain 0=W/plain0.npz
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
SCENE = REPO / "scripts" / "forest" / "models" / "g1box.xml"
BOX = "box_collision"
ITER, SUB = 50, 4
BOX2 = False
INERTIAL_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
                   "torso_link")
SOLE_GEOMS = tuple(f"{s}_foot{k}_collision" for s in ("left", "right") for k in range(1, 8))
VARIANTS = (
    (1.0, 1.0, 1.0, 1.0, 1.0, 0.0, (0.0, 0.0, 0.0)),
    (0.6, 0.5, 1.0, 1.0, 1.0, 0.0, (0.0, 0.0, 0.0)),
    (1.4, 2.0, 1.0, 1.0, 1.0, 0.0, (0.0, 0.0, 0.0)),
    (1.0, 1.0, 1.6, 1.0, 1.0, 0.0, (0.0, 0.0, 0.0)),
    (0.6, 1.0, 0.7, 2.5, 1.3, 0.0, (0.0, 0.0, 0.0)),
    (1.4, 0.7, 1.3, 0.5, 0.8, 0.0, (0.0, 0.0, 0.0)),
    (1.0, 1.0, 1.0, 1.0, 1.0, 0.004, (0.0, 0.0, 0.0)),
    (0.6, 1.0, 1.2, 1.0, 1.0, -0.002, (0.01, -0.005, 0.003)),
    (1.4, 1.5, 1.0, 1.0, 1.4, 0.008, (-0.004, 0.006, -0.01)),
)
CROSS = {9: (0.519, 0.436, 1.037), 10: (0.3, 0.3, 0.3), 11: (2.0, 2.0, 2.0)}
SIZES = tuple(sorted({v[0] for v in VARIANTS}))


class _Scene:
    """The G1 + box scene's helpers (the box: body "box", geom box_collision; box2 on top of it)."""

    @staticmethod
    def base_model():
        """g1box.xml plus a second free box "box2" (geom box2g, 5 cm half-extents, 0.3 kg) paired with the box
        (box-box) and the floor (BOX2)."""
        import mujoco
        from nablaengine.envs import convert
        if not BOX2:
            return convert.to_engine(SCENE.read_text()).model
        s = mujoco.MjSpec.from_string(SCENE.read_text())
        b = s.worldbody.add_body(name="box2", pos=[2.0, 0.0, 0.05])
        b.add_freejoint(name="box2_free")
        b.add_geom(name="box2g", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], mass=0.3, contype=0,
                   conaffinity=0)
        s.add_pair(name="box_box2", geomname1=BOX, geomname2="box2g")
        s.add_pair(name="floor_box2", geomname1="floor", geomname2="box2g")
        for k in s.keys:
            s.delete(k)
        s.compile()
        return convert.to_engine(s.to_xml()).model

    @staticmethod
    def box_geom(m):
        import mujoco
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, BOX)

    @staticmethod
    def box_body(m):
        import mujoco
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")

    @staticmethod
    def box_qadr(m, body="box"):
        import mujoco
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        return int(m.jnt_qposadr[m.body_jntadr[b]]), int(m.jnt_dofadr[m.body_jntadr[b]])

    @staticmethod
    def scaled(m, f):
        """m with the box geom at f times its half-extents (geom_rbound, geom_aabb with it)."""
        g, mb = O.box_geom(m), copy.deepcopy(m)
        mb.geom_size[g] = m.geom_size[g] * f
        mb.geom_rbound[g] = float(np.linalg.norm(mb.geom_size[g]))
        mb.geom_aabb[g] = np.concatenate([np.zeros(3), mb.geom_size[g]])
        return mb

    @staticmethod
    def set_mass(m, mass, inertia):
        """m (MuJoCo C, in place) with the box's mass and principal moments: body_subtreemass and a simple free
        body's dof_M0 as mj_setConst would give them (invweight0 unchanged)."""
        import mujoco
        b = O.box_body(m)
        m.body_mass[b], m.body_inertia[b] = float(mass), inertia
        m.body_subtreemass[b] = float(mass)
        j = m.body_jntadr[b]
        if m.body_simple[b] and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            d = m.jnt_dofadr[j]
            m.dof_M0[d:d + 6] = np.r_[np.full(3, mass), inertia] + m.dof_armature[d:d + 6]


O = _Scene


def base_model():
    return O.base_model()


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def g1_states(n, corpus=None, min_height=0.65):
    """(qpos, qvel, ctrl) of n upright G1 states (root height >= min_height) of a G1 corpus
    (nablaengine.fixtures.load_corpus), cycled."""
    from nablaengine import fixtures
    z = fixtures.load_corpus(corpus)
    idx = np.nonzero(z["qpos"][:, 2] >= min_height)[0]
    idx = idx[np.arange(n) % len(idx)]
    return z["qpos"][idx].astype(np.float64), z["qvel"][idx].astype(np.float64), z["ctrl"][idx].astype(np.float64)


def place_corpus(seed, nworld, sizes, g1_corpus=None):
    """The states (dict of arrays): world i's box at sizes[i % 3] in front of the robot's feet / shins (its near
    face 0.06-0.13 m in front of the pelvis, on the floor), pushed towards the robot, box2 on its top face; 0-6
    control steps of the world's MuJoCo model to settle; the box's MuJoCo C contacts at the state."""
    import mujoco
    m0 = base_model()
    rq, rv, rc = g1_states(nworld, g1_corpus)
    rng = np.random.default_rng(seed)
    g, (qa, da) = O.box_geom(m0), O.box_qadr(m0)
    s0 = m0.geom_size[g].copy()
    out = {k: [] for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "size", "factor", "mass", "inertia")}
    models = {f: O.scaled(m0, f) for f in sorted(set(sizes))}
    bb = O.box_body(m0)
    for i in range(nworld):
        f = sizes[i % 3]
        m = models[f]
        mass, inertia = m0.body_mass[bb], m0.body_inertia[bb].copy()
        O.set_mass(m, mass, inertia)
        d = mujoco.MjData(m)
        d.qpos[:rq.shape[1]] = rq[i]
        d.qvel[:rv.shape[1]] = rv[i]
        sz = s0 * f
        mujoco.mj_kinematics(m, d)
        pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        fw = d.xmat[pb].reshape(3, 3)[:, 0].copy()
        fw[2] = 0.0
        fw /= np.linalg.norm(fw)
        head = np.arctan2(fw[1], fw[0])
        yaw = head + rng.uniform(-0.3, 0.3)
        qb = _qmul(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), m0.qpos0[qa + 3:qa + 7])
        qb /= np.linalg.norm(qb)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, qb)
        c = d.xpos[pb] + fw * (0.10 + sz[0] + rng.uniform(-0.04, 0.03))
        c[2] = sz[2] + rng.uniform(-0.003, 0.004)
        d.qpos[qa:qa + 3] = c - R.reshape(3, 3) @ m0.geom_pos[g]
        d.qpos[qa + 3:qa + 7] = qb
        d.qvel[da:da + 6] = 0.0
        d.qvel[da:da + 3] = -fw * rng.uniform(0.0, 0.6)
        p2, y2 = rng.uniform(-0.6, 0.6, 2), rng.uniform(-np.pi, np.pi)
        if BOX2:
            qa2, da2 = O.box_qadr(m0, "box2")
            d.qpos[qa2:qa2 + 3] = c + R.reshape(3, 3)[:, :2] @ (p2 * sz[:2])
            d.qpos[qa2 + 2] = 2 * sz[2] + 0.05 + rng.uniform(-0.003, 0.004)
            d.qpos[qa2 + 3:qa2 + 7] = [np.cos(y2 / 2), 0, 0, np.sin(y2 / 2)]
            d.qvel[da2:da2 + 6] = 0.0
        d.ctrl[:] = rc[i]
        for _ in range(int(rng.integers(0, 7)) * SUB):
            mujoco.mj_step(m, d)
        u = rc[i] + rng.normal(0, 0.05, m.nu)
        for k, v in (("qpos", d.qpos), ("qvel", d.qvel), ("qacc_warmstart", d.qacc_warmstart), ("ctrl", u),
                     ("size", sz), ("factor", f), ("mass", mass), ("inertia", inertia)):
            out[k].append(np.array(v, np.float64))
    out = {k: np.asarray(v) for k, v in out.items()}
    for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
        out[k] = out[k].astype(np.float32)
    nbox, nrob = [], []
    floor = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for i in range(nworld):
        m = models[float(out["factor"][i])]
        O.set_mass(m, out["mass"][i], out["inertia"][i])
        d = mujoco.MjData(m)
        d.qpos[:] = out["qpos"][i]
        mujoco.mj_forward(m, d)
        nbox.append(sum(1 for c in d.contact[:d.ncon] if g in (c.geom1, c.geom2)))
        nrob.append(sum(1 for c in d.contact[:d.ncon] if g in (c.geom1, c.geom2) and floor not in (c.geom1, c.geom2)))
    out["box_contacts"], out["box_robot_contacts"] = np.asarray(nbox), np.asarray(nrob)
    return out


def ids(m):
    import mujoco
    b = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in INERTIAL_BODIES]
    g = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in SOLE_GEOMS]
    assert min(b) >= 1 and min(g) >= 0
    return b, g


def variant_values(m0, v):
    """{field: values} of variant v (one world), the box's size included."""
    bb, g = O.box_body(m0), O.box_geom(m0)
    if v in CROSS:
        sz, ms = m0.geom_size[g] * np.asarray(CROSS[v]), m0.body_mass[bb]
        f, rb, rf, rw, rt, th, co = VARIANTS[0]
        vals = dict(object_size=sz, torso_mass=np.array([ms]),
                    torso_inertia=ms / 3 * np.array([sz[1] ** 2 + sz[2] ** 2, sz[0] ** 2 + sz[2] ** 2,
                                                     sz[0] ** 2 + sz[1] ** 2]))
    else:
        f, rb, rf, rw, rt, th, co = VARIANTS[v]
        vals = dict(object_size=m0.geom_size[g] * f, torso_mass=np.array([m0.body_mass[bb] * f ** 3 * rb]),
                    torso_inertia=m0.body_inertia[bb] * f ** 5 * rb)
    bi, _ = ids(m0)
    x = np.zeros((len(bi), 7))
    for k, b in enumerate(bi):
        r = (rf, rf, rw, rw, rt)[k]
        x[k, 0], x[k, 1:4] = m0.body_mass[b] * r, m0.body_inertia[b] * r
        if k in (0, 1, 4):
            x[k, 4:] = co
    vals["body_inertial"] = x
    vals["sole_thickness"] = np.array([th])
    return vals


def variant_model(m0, v, nominal_meaninertia=False, setconst=True):
    """MuJoCo model of variant v: the values in place (float32-rounded as the engine's per-world values) and
    mj_setConst (invweight0, ... as a per-model build)."""
    import mujoco
    vals = variant_values(m0, v)
    m = copy.deepcopy(m0)
    f32 = lambda x: np.asarray(x, np.float32).astype(np.float64)
    bb, g = O.box_body(m0), O.box_geom(m0)
    m.geom_size[g] = f32(vals["object_size"])
    m.geom_rbound[g] = float(np.linalg.norm(m.geom_size[g]))
    m.geom_aabb[g] = np.concatenate([np.zeros(3), m.geom_size[g]])
    m.body_mass[bb], m.body_inertia[bb] = f32(vals["torso_mass"])[0], f32(vals["torso_inertia"])
    bi, sg = ids(m0)
    x = vals["body_inertial"]
    for k, b in enumerate(bi):
        m.body_mass[b], m.body_inertia[b] = f32(x[k, 0]), f32(x[k, 1:4])
        m.body_ipos[b] = m0.body_ipos[b] + f32(x[k, 4:])
    for gg in sg:
        m.geom_pos[gg] = m0.geom_pos[gg] + f32(vals["sole_thickness"][0]) * np.array([0.0, 0.0, -1.0])
    if not setconst:
        O.set_mass(m, m.body_mass[bb], m.body_inertia[bb].copy())
    if setconst:
        mi = m.stat.meaninertia
        mujoco.mj_setConst(m, mujoco.MjData(m))
        if nominal_meaninertia:
            m.stat.meaninertia = mi
    return m


def params(m0, iw="per_world", osize=None, osize_variant=None, basic=False):
    """The parameter build: osize None (no object_size: the box at the model's size in every world) or the factor
    of object_size_max (a world is bitwise its size's plain build only at that size: object_size's bounds)."""
    from nablaengine.params import PhysicsParams
    f = ("torso_mass", "torso_inertia", "body_inertial", "sole_thickness")
    if basic:
        return PhysicsParams(("object_size", "torso_mass", "torso_inertia"), invweight0=iw,
                             object_size_max=tuple(m0.geom_size[O.box_geom(m0)] * osize))
    if osize_variant is not None:
        return PhysicsParams(("object_size",) + f, invweight0=iw, inertial_bodies=INERTIAL_BODIES, sole_geoms=SOLE_GEOMS,
                             object_size_max=tuple(float(x) for x in np.float32(
                                 variant_values(m0, osize_variant)["object_size"])))
    if osize is None:
        return PhysicsParams(f, inertial_bodies=INERTIAL_BODIES, sole_geoms=SOLE_GEOMS, invweight0=iw)
    return PhysicsParams(("object_size",) + f, object_size_max=tuple(m0.geom_size[O.box_geom(m0)] * osize),
                         inertial_bodies=INERTIAL_BODIES, sole_geoms=SOLE_GEOMS, invweight0=iw)


def com_rows(P, m0, z, info):
    B = len(z["qpos"])
    cw0 = int(info["com_words"]) - P.width(m0.nv)
    per = [variant_values(m0, int(v)) for v in z["variant"]]
    vals = {f: np.stack([p[f] for p in per]).reshape(B, -1) for f in P.fields}
    return np.concatenate([np.zeros((B, cw0), np.float32), P.pack(m0.nv, **vals)], 1)


def iw_numpy(m, t, ip7=None):
    """The iw0_world algorithm in numpy (double) on tables t (body_params.iw_tables of the base model) with m's
    inertials and armature: (body_invweight0[:, 0], dof_invweight0)."""
    nb, nv, D = m.nbody, m.nv, t["maxdep"]
    ci = np.zeros((nb, 10))
    cp = np.zeros((nb, 3))
    for b in range(1, nb):
        R, Ri = t["xmat"][b].reshape(3, 3), t["ximat"][b].reshape(3, 3)
        c = t["xpos"][b] + R @ m.body_ipos[b]
        I = Ri @ np.diag(m.body_inertia[b]) @ Ri.T
        Io = I + m.body_mass[b] * (c @ c * np.eye(3) - np.outer(c, c))
        ci[b] = [m.body_mass[b], *(m.body_mass[b] * c), Io[0, 0], Io[1, 1], Io[2, 2], Io[0, 1], Io[0, 2], Io[1, 2]]
        cp[b] = c
    crb = np.zeros((nb, 10))
    for b in range(1, nb):
        for c in range(1, nb):
            if (int(t["sub"][b]) >> c) & 1:
                crb[b] += ci[c]
    row = np.zeros((nv, D))
    for i in range(nv):
        cr = crb[t["dbody"][i]]
        s = t["cdof"][i]
        Io = np.array([[cr[4], cr[7], cr[8]], [cr[7], cr[5], cr[9]], [cr[8], cr[9], cr[6]]])
        h = cr[1:4]
        L = Io @ s[:3] + np.cross(h, s[3:])
        P = cr[0] * s[3:] - np.cross(h, s[:3])
        for p in range(t["dd"][i] + 1):
            tt = t["cdof"][t["chain"][i, p]]
            row[i, p] = tt[:3] @ L + tt[3:] @ P
        row[i, 0] += m.dof_armature[i]
    for k in range(nv - 1, -1, -1):
        rk = row[k].copy()
        for i in range(nv):
            p = t["dd"][k] - t["dd"][i]
            if i == k or p < 1 or t["chain"][k, p] != i:
                continue
            tt = rk[p] / rk[0]
            for q in range(t["dd"][i] + 1):
                row[i, q] -= tt * rk[p + q]
        row[k, 1:t["dd"][k] + 1] = rk[1:t["dd"][k] + 1] / rk[0]

    def solve(st, ys):
        acc = 0.0
        for y in ys:
            y = y.copy()
            for k in range(nv - 1, -1, -1):
                p = t["dd"][st] - t["dd"][k]
                if p < 0 or t["chain"][st, p] != k:
                    continue
                acc += y[p] ** 2 / row[k, 0]
                for q in range(1, t["dd"][k] + 1):
                    y[p + q] -= row[k, q] * y[p]
        return acc
    biw = np.zeros(nb)
    for b in range(1, nb):
        st = t["blast"][b]
        ys = np.zeros((3, D))
        for p in range(t["dd"][st] + 1):
            s = t["cdof"][t["chain"][st, p]]
            ys[:, p] = s[3:] + np.cross(s[:3], cp[b])
        biw[b] = solve(st, ys) / 3
    diw = np.zeros(nv)
    for st, r0, nr, od, no in t["tasks"]:
        ys = np.zeros((nr, D))
        for r in range(nr):
            ys[r, t["dd"][st] - t["dd"][r0 + r]] = 1.0
        diw[od:od + no] = solve(st, ys) / nr
    return biw, diw


def cmd_iwunit(a):
    """iw0_world compiled alone (nvcc) on each variant model's tables (the model's double values, as a world at those
    values reads them) against float32(mj_setConst)."""
    import subprocess
    import tempfile
    from nablaengine.forest_engine import body_params
    m0 = base_model()
    res, d = {}, tempfile.mkdtemp()
    for v in list(range(len(VARIANTS))) + sorted(CROSS):
        m = variant_model(m0, v)
        src = (f"#include <cstdio>\n#define NB {m.nbody}\n#define NV {m.nv}\n#define PW_ARM -1\n"
               "__device__ float c_binvw[NB];\n__device__ float c_misc[2] = {0.f, -1.f};\n"
               + body_params.iw_header(m) + body_params.IW_DEVICE + r"""
__global__ void k(float* ob, float* od) {
  __shared__ float sb[NB], sd[NV], sb2[NB], sd2[NV];
  __shared__ double sh[NV * IW_MAXDEP];
  iw0_world(nullptr, nullptr, nullptr, threadIdx.x, sb, sd, nullptr);  // (the shuffle path)
  iw0_world(nullptr, nullptr, nullptr, threadIdx.x, sb2, sd2, sh);  // (k_iw's shared-row path: the same bits)
  __syncwarp();
  if (threadIdx.x == 0) {
    for (int i = 0; i < NB; i++) if (__float_as_uint(sb[i]) != __float_as_uint(sb2[i])) sb[i] = -1.f;
    for (int i = 0; i < NV; i++) if (__float_as_uint(sd[i]) != __float_as_uint(sd2[i])) sd[i] = -1.f;
  }
  __syncwarp();
  for (int i = threadIdx.x; i < NB; i += 32) ob[i] = sb[i];
  for (int i = threadIdx.x; i < NV; i += 32) od[i] = sd[i];
}
int main() {
  float *ob, *od;
  cudaMallocManaged(&ob, NB * 4);
  cudaMallocManaged(&od, NV * 4);
  k<<<1, 32>>>(ob, od);
  cudaDeviceSynchronize();
  fwrite(ob, 4, NB, stdout);
  fwrite(od, 4, NV, stdout);
  return 0;
}
""")
        Path(d, "iw.cu").write_text(src)
        subprocess.run(["nvcc", "-O3", f"-arch={a.arch}", "-o", f"{d}/iw", f"{d}/iw.cu"], check=True)
        o = np.frombuffer(subprocess.run([f"{d}/iw"], capture_output=True, check=True).stdout, np.float32)
        b, dd = o[:m.nbody], o[m.nbody:]
        rb, rd = m.body_invweight0[:, 0].astype(np.float32), m.dof_invweight0.astype(np.float32)
        res[v] = dict(body_equal=f"{int((b[1:] == rb[1:]).sum())}/{m.nbody - 1}", dof_equal=f"{int((dd == rd).sum())}/{m.nv}")
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


ADJ_UNIT = ("struct Adj { float com[14]; float pw[NPARAM]; float giw[NB], gdiw[NV]; float gpw[NPARAM + 1];"
            " double iwsh[NV * IW_MAXDEP]; };\n")


def cmd_iwgrad(a):
    """iw0_grad compiled alone: random dL / d invweight0 (per body, per dof) -> the device's cotangents of one
    inertial body's mass, moments, COM offset and of the armature against numpy (dense M^-1 at qpos0)."""
    import subprocess
    import tempfile
    import mujoco
    from nablaengine.forest_engine import body_params
    m0 = base_model()
    k = mujoco.mj_name2id(m0, 1, a.body)
    nb, nv = m0.nbody, m0.nv
    rng = np.random.default_rng(0)
    G = rng.standard_normal(nb).astype(np.float32); G[0] = 0
    H = rng.standard_normal(nv).astype(np.float32)
    NP = 7 + nv
    src = f"""#include <cstdio>
    #define NB {nb}
    #define NV {nv}
    #define NIB 1
    #define PW_INERTIAL 0
    #define PW_ARM 7
    #define NPARAM {NP}
    #define IB_SLOT(b) ((b) == {k} ? 0 : -1)
    __device__ float c_binvw[NB];
    __device__ float c_misc[2] = {{0.f, -1.f}};
    """ + body_params.iw_header(m0) + body_params.IW_DEVICE + "\n#define IW_NTG 1\n" + ADJ_UNIT \
        + body_params._itab("c_iw_tg", [k, 0, 1, 4]) + body_params.IW_GRAD_DEVICE + r"""
    __global__ void kk(const float* in, float* out) {
      __shared__ Adj s;
      int lane = threadIdx.x;
      for (int i = lane; i < 14; i += 32) s.com[i] = in[i];
      for (int i = lane; i < NPARAM; i += 32) s.pw[i] = in[14 + i];
      for (int i = lane; i < NB; i += 32) s.giw[i] = in[14 + NPARAM + i];
      for (int i = lane; i < NV; i += 32) s.gdiw[i] = in[14 + NPARAM + NB + i];
      for (int i = lane; i <= NPARAM; i += 32) s.gpw[i] = 0.f;
      __syncwarp();
      iw0_grad(s, lane);
      for (int i = lane; i < NPARAM; i += 32) out[i] = s.gpw[i];
    }
    int main() {
      static float in[14 + NPARAM + NB + NV];
      fread(in, 4, 14 + NPARAM + NB + NV, stdin);
      float *din, *dout; cudaMalloc(&din, sizeof(in)); cudaMalloc(&dout, NPARAM * 4);
      cudaMemcpy(din, in, sizeof(in), cudaMemcpyHostToDevice);
      kk<<<1, 32>>>(din, dout);
      static float out[NPARAM]; cudaMemcpy(out, dout, NPARAM * 4, cudaMemcpyDeviceToHost);
      fwrite(out, 4, NPARAM, stdout);
      return 0;
    }
    """
    d = tempfile.mkdtemp()
    open(f"{d}/g.cu", "w").write(src)
    subprocess.run(["nvcc", "-O3", f"-arch={a.arch}", "-o", f"{d}/g", f"{d}/g.cu"], check=True)
    com = np.zeros(14, np.float32); com[7] = m0.body_mass[k]; com[8:11] = m0.body_inertia[k]
    pw = np.concatenate([com[7:14], m0.dof_armature]).astype(np.float32)
    inp = np.concatenate([com, pw, G, H]).astype(np.float32)
    out = np.frombuffer(subprocess.run([f"{d}/g"], input=inp.tobytes(), capture_output=True, check=True).stdout, np.float32)
    t = body_params.iw_tables(m0)
    dd = mujoco.MjData(m0); mujoco.mj_forward(m0, dd)
    M = np.zeros((nv, nv)); mujoco.mj_fullM(m0, dd, M); Mi = np.linalg.inv(M)
    def jac(b):
        jp = np.zeros((3, nv)); jr = np.zeros((3, nv)); mujoco.mj_jacBodyCom(m0, dd, jp, jr, b); return jp, jr
    B = np.zeros_like(M); own = {}
    for b in range(1, nb):
        jp, _ = jac(b); X = Mi @ jp.T; B += G[b] / 3 * X @ X.T; own[b] = X
    for st, r0, nr, od, no in t["tasks"]:
        w = H[od:od + no].astype(np.float64).sum() / nr
        for r in range(nr):
            e = np.zeros(nv); e[r0 + r] = 1; x = Mi @ e; B += w * np.outer(x, x)
    jp, jr = jac(k); Ri = dd.ximat[k].reshape(3, 3); R = dd.xmat[k].reshape(3, 3)
    ref = np.zeros(NP)
    ref[0] = -np.trace(jp @ B @ jp.T)
    ref[1:4] = [-(Ri[:, a] @ jr @ B @ jr.T @ Ri[:, a]) for a in range(3)]
    gc = np.zeros(3)
    for l in range(3):
        e = np.zeros(3); e[l] = 1
        dJ = np.stack([np.cross(jr[:, i], e) for i in range(nv)], 1)
        gc[l] = -m0.body_mass[k] * np.trace(dJ @ B @ jp.T + jp @ B @ dJ.T) + 2 * G[k] / 3 * np.trace(dJ @ own[k])
    ref[4:7] = R.T @ gc
    ref[7:] = -np.diag(B)
    rel = np.abs(out - ref) / np.maximum(np.abs(ref), 1e-9)
    res = dict(body=a.body, mass_moments_com_rel=[float(x) for x in rel[:7]], armature_rel_max=float(rel[7:].max()))
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_iwtab(a):
    from nablaengine.forest_engine import body_params
    m0 = base_model()
    t = body_params.iw_tables(m0)
    res = {}
    for v in range(len(VARIANTS)):
        m = variant_model(m0, v)
        b, d = iw_numpy(m, t)
        rb = np.abs(b - m.body_invweight0[:, 0]) / np.maximum(m.body_invweight0[:, 0], 1e-300)
        rd = np.abs(d - m.dof_invweight0) / m.dof_invweight0
        res[v] = dict(body_rel_max=float(rb[1:].max()), dof_rel_max=float(rd.max()),
                      body_f32_equal=float((b.astype(np.float32) == m.body_invweight0[:, 0].astype(np.float32)).mean()),
                      dof_f32_equal=float((d.astype(np.float32) == m.dof_invweight0.astype(np.float32)).mean()),
                      box_invweight0=[float(x) for x in m.body_invweight0[O.box_body(m0)]])
    print(json.dumps(res, indent=1))


def cmd_corpus(a):
    sizes = SIZES if a.factor is None else (a.factor,) * 3
    z = place_corpus(a.seed, a.nworld, sizes, a.g1_corpus)
    B = len(z["qpos"])
    rng = np.random.default_rng(a.seed + 1)
    var = np.zeros(B, int)
    for i in range(B):
        ok = [v for v in range(len(VARIANTS)) if np.isclose(VARIANTS[v][0], z["factor"][i])] or [0]
        var[i] = a.cross if a.cross is not None else ok[rng.integers(len(ok))]
    z["variant"] = var
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **z)
    print(json.dumps(dict(nworld=B, per_variant=np.bincount(var, minlength=12).tolist(),
                          box_contacts_mean=float(z["box_contacts"].mean()),
                          worlds_with_box_robot_contacts=int((z["box_robot_contacts"] > 0).sum()), out=a.out)))


def build(a, m0, z):
    from nablaengine import jax_ffi
    B = len(z["qpos"])
    if a.mode == "plain":
        info = jax_ffi.init(model=variant_model(m0, a.variant, a.nominal_meaninertia, not a.no_setconst),
                            foot_pairs=[])
        return info, np.zeros((B, int(info["com_words"])), np.float32)
    P = params(m0, a.iw, a.osize, a.osize_variant, basic=getattr(a, "basic", False))
    info = jax_ffi.init(model=m0, foot_pairs=[], com_body="box", params=P,
                        **({"contact_surrogate": True} if getattr(a, "surrogate", False) else {}))
    return info, com_rows(P, m0, z, info)


def cmd_run(a):
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    z = np.load(a.corpus)
    m0 = base_model()
    info, com = build(a, m0, z)
    B = len(z["qpos"])
    j = lambda x: jnp.asarray(np.ascontiguousarray(x, np.float32))
    fr = j(np.ones(B))
    args = (j(z["qpos"]), j(z["qvel"]), j(z["qacc_warmstart"]), j(z["ctrl"]), fr, j(com))
    r = jax_ffi.forward(*args, iterations=ITER, substeps=SUB, tape=True, contacts=True)
    q1, v1, w1, inf, con, tape = (np.asarray(x) for x in r[:6])
    rng = np.random.default_rng(7)
    yq, yv = rng.standard_normal((B, m0.nq)).astype(np.float32), rng.standard_normal((B, m0.nv)).astype(np.float32)
    g = jax_ffi.backward(j(tape), fr, j(com), j(yq), j(yv), substeps=SUB, clamp_kv=1, normal=0)
    out = dict(q1=q1, v1=v1, w1=w1, info=inf, con=con, yq=yq, yv=yv, com=com, g_qpos=np.asarray(g[0]),
               g_qvel=np.asarray(g[1]), g_ctrl=np.asarray(g[2]), g_info=np.asarray(g[3]))
    if len(g) > 4:
        out.update(g_params=np.asarray(g[4]), g_friction=np.asarray(g[5]))
    np.savez(a.dump, **out)
    print(json.dumps(dict(mode=a.mode, variant=a.variant, iw=a.iw, nworld=B, engine=info["engine"],
                          flags=int((inf[:, 3] != 0).sum()), adj_flags=int((out["g_info"] != 0).sum()),
                          contacts_mean=float(inf[:, 1].mean()), dump=a.dump)))


def _rel(u, v):
    u, v = u.astype(np.float64), v.astype(np.float64)
    return np.linalg.norm(u - v, axis=1) / np.maximum(np.linalg.norm(v, axis=1), 1e-6)


def cmd_compare(a):
    z, x = np.load(a.corpus), np.load(a.dump)
    res = {}
    for spec in a.plain:
        k, path = spec.split("=")
        k = int(k)
        y = np.load(path)
        sel = z["variant"] == k
        r = dict(nworld=int(sel.sum()))
        for f in ("q1", "v1", "w1", "info", "g_qpos", "g_qvel", "g_ctrl"):
            u, v = x[f][sel], y[f][sel]
            same = np.all(u.view(np.uint32) == v.view(np.uint32), axis=tuple(range(1, u.ndim))) \
                if u.dtype == np.float32 else np.all(u == v, axis=tuple(range(1, u.ndim)))
            e = dict(bitwise_worlds=int(same.sum()))
            if u.dtype == np.float32 and u.ndim == 2:
                rr = _rel(u, v)
                e.update(rel_median=float(np.median(rr)), rel_max=float(rr.max()))
            r[f] = e
        res[k] = r
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_ref(a):
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    z = np.load(a.corpus)
    m0 = base_model()
    B = len(z["qpos"])
    oq, ov, nc = np.zeros((B, m0.nq), np.float32), np.zeros((B, m0.nv), np.float32), np.zeros(B, int)
    for v in range(12):
        idx = np.nonzero(z["variant"] == v)[0]
        if not len(idx):
            continue
        m = variant_model(m0, v, a.nominal_meaninertia)
        m.opt.iterations = ITER
        mw = mjw.put_model(m)
        dw = mjw.put_data(m, mujoco.MjData(m), nworld=len(idx), nconmax=64 * len(idx), njmax=256 * len(idx))
        for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
            wp.copy(getattr(dw, k), wp.array(np.ascontiguousarray(z[k][idx], np.float32)))
        for _ in range(SUB):
            mjw.step(mw, dw)
        oq[idx], ov[idx] = dw.qpos.numpy(), dw.qvel.numpy()
        nacon = int(dw.nacon.numpy()[0])
        nc[idx] = np.bincount(dw.contact.worldid.numpy()[:nacon], minlength=len(idx))
    np.savez(a.out, out_qpos=oq, out_qvel=ov, ncon=nc)
    print(json.dumps(dict(nworld=B, contacts_mean=float(nc.mean()), out=a.out)))


def cmd_fwd(a):
    z, r, d = np.load(a.corpus), np.load(a.ref), np.load(a.dump)
    res = {}
    for v in range(12):
        sel = z["variant"] == v
        if not sel.any():
            continue
        rel = _rel(d["v1"][sel], r["out_qvel"][sel])
        res[v] = dict(nworld=int(sel.sum()), qvel_rel_median=float(np.median(rel)),
                      qvel_rel_p90=float(np.quantile(rel, 0.9)), qvel_rel_max=float(rel.max()),
                      qpos_abs_max=float(np.abs(d["q1"][sel] - r["out_qpos"][sel]).max()),
                      ncon_equal=float((d["info"][sel, 1] == r["ncon"][sel]).mean()))
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def _summary(x):
    return dict(rel_median=float(np.median(x)), rel_p90=float(np.quantile(x, 0.9)), rel_max=float(x.max()))


def cmd_pgrad(a):
    import mujoco
    import torch
    from nablaengine import torch_ffi
    z = np.load(a.corpus)
    m0 = base_model()
    P = params(m0, a.iw, a.osize, a.osize_variant, a.basic)
    info = torch_ffi.init(model=m0, foot_pairs=[], com_body="box", params=P)
    B = min(a.nworld, len(z["qpos"]))
    sel = np.nonzero(np.isin(z["variant"], (0, 1, 2)))[0][:B] if a.basic else np.arange(B)
    B = len(sel)
    zz = {k: z[k][sel] for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "variant")}
    com = com_rows(P, m0, zz, info)
    dev = "cuda"
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x, np.float32), device=dev)
    comt = t(com).requires_grad_()
    rng = np.random.default_rng(3)
    yq, yv = rng.standard_normal((B, m0.nq)), rng.standard_normal((B, m0.nv))
    q1, v1, _ = torch_ffi.step(t(zz["qpos"]), t(zz["qvel"]), t(zz["qacc_warmstart"]), t(zz["ctrl"]),
                               t(np.ones(B)), comt, iterations=ITER, substeps=SUB, clamp_kv=1)
    ((t(yq) * q1).sum() + (t(yv) * v1).sum()).backward()
    cw0 = com.shape[1] - P.width(m0.nv)
    gp = comt.grad.cpu().numpy().astype(np.float64)[:, cw0:]
    lay = P.layout(m0.nv)
    models = {v: variant_model(m0, v, setconst=a.setconst_fd) for v in sorted(set(int(x) for x in zz['variant']))}
    for m in models.values():
        m.opt.iterations, m.opt.tolerance, m.opt.ls_tolerance = 100, 1e-12, 1e-12
    bi, sg = ids(m0)

    def run(i, field, x):
        """C float64 one control step of world i with field's values x (its variant's model otherwise)."""
        v = int(zz["variant"][i])
        m = copy.deepcopy(models[v])
        if field == "body_inertial":
            x = x.reshape(-1, 7)
            for k, b in enumerate(bi):
                m.body_mass[b], m.body_inertia[b] = x[k, 0], x[k, 1:4]
                m.body_ipos[b] = m0.body_ipos[b] + x[k, 4:]
        elif field in ("torso_mass", "torso_inertia"):
            bb = O.box_body(m0)
            if field == "torso_mass":
                m.body_mass[bb] = x[0]
            else:
                m.body_inertia[bb] = x
            if not a.setconst_fd:
                O.set_mass(m, m.body_mass[bb], m.body_inertia[bb].copy())
        elif field == "sole_thickness":
            for gg in sg:
                m.geom_pos[gg] = m0.geom_pos[gg] + x[0] * np.array([0.0, 0.0, -1.0])
        d = mujoco.MjData(m)
        if a.setconst_fd and field is not None:
            mujoco.mj_setConst(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = (np.asarray(zz[k][i], np.float64) for k in ("qpos", "qvel", "ctrl"))
        d.qacc_warmstart[:] = zz["qacc_warmstart"][i]
        for _ in range(SUB):
            mujoco.mj_step(m, d)
        return d.qpos.copy(), d.qvel.copy()
    v1n = v1.detach().cpu().numpy().astype(np.float64)
    res = dict(nworld=B, iw=a.iw, setconst_fd=a.setconst_fd)
    vals = [variant_values(m0, int(v)) for v in zz["variant"]]
    fv = np.array([np.linalg.norm(v1n[i] - run(i, None, None)[1]) / max(np.linalg.norm(run(i, None, None)[1]), 1e-6)
                   for i in range(B)])
    res["forward_qvel_rel_median_vs_C"] = float(np.median(fv))
    for f in ("body_inertial", "sole_thickness", "torso_mass", "torso_inertia"):
        if f not in lay:
            continue
        o, w = lay[f]
        groups = {"mass": [0], "inertia": [1, 2, 3], "com": [4, 5, 6]} if f == "body_inertial" else {"all": None}
        for gname, cols in groups.items():
            rel, mag = [], []
            for i in range(B):
                x0 = np.asarray(vals[i][f], np.float64).ravel()
                u = np.zeros(w)
                if cols is None:
                    u[:] = 1.0
                else:
                    for k in range(len(bi)):
                        u[[7 * k + c for c in cols]] = rng.standard_normal(len(cols))
                u /= np.linalg.norm(u)
                if f == "sole_thickness":
                    eps = a.eps_len
                elif cols is None:
                    eps = a.eps * float(np.min(np.abs(x0)))
                elif gname == "com":
                    eps = a.eps_len
                else:
                    eps = a.eps * float(np.min(np.abs(x0[np.nonzero(u)[0]])))
                (qp, vp), (qm, vm) = run(i, f, x0 + eps * u), run(i, f, x0 - eps * u)
                fd = (yq[i] @ (qp - qm) + yv[i] @ (vp - vm)) / (2 * eps)
                g = gp[i, o:o + w] @ u
                rel.append(abs(fd - g) / max(abs(fd), abs(g), 1e-12))
                mag.append((abs(fd), abs(g)))
            x, mag = np.asarray(rel), np.asarray(mag)
            on = mag.max(1) >= 1e-7
            res[f"{f}:{gname}"] = dict(_summary(x[on]) if on.any() else {}, acting_share=float(on.mean()),
                                       fd_abs_median=float(np.median(mag[:, 0])))
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_parity(a):
    import torch
    from nablaengine import torch_ffi
    z = np.load(a.corpus)
    x = np.load(a.dump)
    m0 = base_model()
    P = params(m0, a.iw, a.osize, a.osize_variant)
    torch_ffi.init(model=m0, foot_pairs=[], com_body="box", params=P)
    t = lambda v: torch.as_tensor(np.ascontiguousarray(v, np.float32), device="cuda")
    B = len(z["qpos"])
    xs = [t(z["qpos"]).requires_grad_(), t(z["qvel"]).requires_grad_(), t(z["ctrl"]).requires_grad_(),
          t(x["com"]).requires_grad_()]
    q1, v1, w1 = torch_ffi.step(xs[0], xs[1], t(z["qacc_warmstart"]), xs[2], t(np.ones(B)), xs[3], iterations=ITER,
                                substeps=SUB, clamp_kv=1)
    ((t(x["yq"]) * q1).sum() + (t(x["yv"]) * v1).sum()).backward()
    cw0 = x["com"].shape[1] - P.width(m0.nv)
    pairs = dict(q1=(q1, x["q1"]), v1=(v1, x["v1"]), w1=(w1, x["w1"]), g_qpos=(xs[0].grad, x["g_qpos"]),
                 g_qvel=(xs[1].grad, x["g_qvel"]), g_ctrl=(xs[2].grad, x["g_ctrl"]),
                 g_params=(xs[3].grad[:, cw0:], x["g_params"]))
    res = {k: bool(np.array_equal(u.detach().cpu().numpy().view(np.uint32), np.asarray(v, np.float32).view(np.uint32)))
           for k, (u, v) in pairs.items()}
    print(json.dumps(res, indent=1))


def cmd_time(a):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    z = np.load(a.corpus)
    m0 = base_model()
    n = a.nworld
    rep = lambda x: np.resize(np.asarray(x), (n,) + np.shape(x)[1:])
    zz = {k: rep(z[k]) for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "variant")}
    a.variant = 0
    info, com = build(a, m0, zz)
    j = lambda x: jnp.asarray(np.ascontiguousarray(x, np.float32))
    args = [j(zz["qpos"]), j(zz["qvel"]), j(zz["qacc_warmstart"]), j(zz["ctrl"]), j(np.ones(n)), j(com)]
    step = jax_ffi.make_step(iterations=a.iterations, substeps=SUB) if hasattr(jax_ffi, "make_step") else None

    def loss(q, v, w, u, f, c):
        q1, v1, _ = step(q, v, w, u, f, c)
        return jnp.sum(q1) + jnp.sum(v1)
    fb = jax.jit(jax.grad(loss, argnums=(0, 1, 3, 5)))
    fw = jax.jit(lambda *x: step(*x)[1])
    out = {}
    for name, fn in (("forward", fw), ("forward_backward", fb)):
        jax.block_until_ready(fn(*args))
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            ts.append(time.perf_counter() - t0)
        out[name + "_ms"] = float(np.median(ts) * 1e3)
    print(json.dumps(dict(mode=a.mode, iw=a.iw, nworld=n, **out)))


def main():
    global BOX2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box2", action="store_true", help="the scene with box2 (a box-box pair); every command "
                    "of one run takes the same choice")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("iwtab")
    p = sp.add_parser("iwgrad")
    p.add_argument("--arch", default="sm_70")
    p.add_argument("--body", default="left_ankle_roll_link")
    p.add_argument("--out", default=None)
    p = sp.add_parser("iwunit")
    p.add_argument("--arch", default="sm_70")
    p.add_argument("--out", default=None)
    p = sp.add_parser("corpus")
    p.add_argument("--out", required=True)
    p.add_argument("--nworld", type=int, default=288)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cross", type=int, default=None)
    p.add_argument("--factor", type=float, default=None)
    p.add_argument("--g1-corpus", default=None, help="the G1 corpus of the robot's states (default: "
                   "nablaengine.fixtures' default, mf_s0)")
    for name in ("run", "time"):
        p = sp.add_parser(name)
        p.add_argument("--corpus", required=True)
        p.add_argument("--mode", choices=("params", "plain"), default="params")
        p.add_argument("--variant", type=int, default=0)
        p.add_argument("--iw", choices=("model", "per_world", "per_world_mass"), default="per_world")
        p.add_argument("--nominal-meaninertia", action="store_true")
        p.add_argument("--osize", type=float, default=None)
        p.add_argument("--osize-variant", type=int, default=None)
        p.add_argument("--no-setconst", action="store_true", help="plain: the variant's model without mj_setConst")
        p.add_argument("--surrogate", action="store_true", help="params: a contact_surrogate build (surrogates off)")
        p.add_argument("--basic", action="store_true", help="params: object_size and the box's torso fields only "
                       "(needs --osize; invweight0 per_world_mass needs it: the box is its only target)")
        if name == "run":
            p.add_argument("--dump", required=True)
        else:
            p.add_argument("--nworld", type=int, default=4096)
            p.add_argument("--reps", type=int, default=20)
            p.add_argument("--iterations", type=int, default=5)
    p = sp.add_parser("compare")
    p.add_argument("--corpus", required=True)
    p.add_argument("--dump", required=True)
    p.add_argument("--plain", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p = sp.add_parser("ref")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nominal-meaninertia", action="store_true")
    p = sp.add_parser("fwd")
    p.add_argument("--corpus", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--dump", required=True)
    p.add_argument("--out", default=None)
    p = sp.add_parser("pgrad")
    p.add_argument("--corpus", required=True)
    p.add_argument("--nworld", type=int, default=64)
    p.add_argument("--iw", choices=("model", "per_world"), default="model")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--eps-len", type=float, default=1e-7)
    p.add_argument("--basic", action="store_true", help="main's fields only (object_size, torso_mass / inertia)")
    p.add_argument("--setconst-fd", action="store_true", help="the references' models through mj_setConst (each "
                   "perturbed one too): the gap is the invweight0 derivative the gradients leave out")
    p.add_argument("--osize", type=float, default=None)
    p.add_argument("--osize-variant", type=int, default=None)
    p.add_argument("--out", default=None)
    p = sp.add_parser("parity")
    p.add_argument("--corpus", required=True)
    p.add_argument("--dump", required=True)
    p.add_argument("--iw", choices=("model", "per_world", "per_world_mass"), default="per_world")
    p.add_argument("--osize", type=float, default=None)
    p.add_argument("--osize-variant", type=int, default=None)
    a = ap.parse_args()
    BOX2 = a.box2
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()

"""Gravity compensation of fused bodies (nablaengine.envs.convert.to_engine): where a jointless body is fused into a
parent with a different body gravcomp, one body's gravcomp at its COM cannot reproduce the source's forces."""
import mujoco
import numpy as np

NUMERIC = "nablaengine_gravcomp_offset"


def offsets(m):
    """(nbody, 3) float64 gravcomp application offsets from the custom numeric NUMERIC, or None without it or when
    every offset is 0."""
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, NUMERIC)
    if i < 0:
        return None
    adr, n = int(m.numeric_adr[i]), int(m.numeric_size[i])
    if n != 3 * m.nbody:
        raise ValueError(f"custom numeric {NUMERIC}: size {n}, the model has {m.nbody} bodies (3 values per body)")
    v = np.array(m.numeric_data[adr:adr + n], np.float64).reshape(m.nbody, 3)
    if not np.isfinite(v).all():
        raise ValueError(f"custom numeric {NUMERIC}: the values must be finite")
    return v if v.any() else None


def groups(m0, static):
    """{root body id: [member body ids]} of model m0 for the jointless bodies named in static: each fuses into its
    nearest ancestor not in static (the root; the world included)."""
    out = {}
    ids = {mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, n) for n in static}
    for b in sorted(ids):
        r = int(m0.body_parentid[b])
        while r in ids:
            r = int(m0.body_parentid[r])
        out.setdefault(r, [r]).append(b)
    return out


def mixed(m0, static):
    """{root: (members, gc_eff)} of groups(m0, static) whose members' gravcomp differ (rooted below the world), with
    the group's mass-weighted gravcomp."""
    out = {}
    for r, mem in groups(m0, static).items():
        gc = m0.body_gravcomp[mem]
        if r == 0 or np.all(gc == gc[0]):
            continue
        mass = m0.body_mass[mem]
        out[r] = (mem, float(np.dot(gc, mass) / mass.sum()) if mass.sum() > 0 else float(gc[0]))
    return out


def offset_field(m0, m, mix):
    """(nbody of m, 3) offsets for the groups mix (mixed's) of source m0 in the fused model m (bodies by name)."""
    d0, d = mujoco.MjData(m0), mujoco.MjData(m)
    d.qpos[:] = d0.qpos
    mujoco.mj_kinematics(m0, d0)
    mujoco.mj_kinematics(m, d)
    out = np.zeros((m.nbody, 3))
    for r, (mem, _) in mix.items():
        w = m0.body_gravcomp[mem] * m0.body_mass[mem]
        if w.sum() == 0:
            continue
        p = (w[:, None] * d0.xipos[mem]).sum(0) / w.sum()
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, m0.body(r).name)
        out[b] = d.xmat[b].reshape(3, 3).T @ (p - d.xipos[b])
    return out

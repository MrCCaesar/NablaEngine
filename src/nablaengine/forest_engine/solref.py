"""MuJoCo's two solref forms in the forest engine's constraint rows (capability T13)."""
import mujoco
import numpy as np

CODE = "T13"


def _flag(m, bit):
    return bool(int(m.opt.disableflags) & int(bit))


def rows(m):
    """[(kind, name, solref (2,))] of the solrefs the forest engine's rows read: every contact pair's, the limited
    hinge / slide joints', the frictionloss dofs', the active equalities' and the limited tendons' (none of a kind the
    model's flags disable)."""
    D, O = mujoco.mjtDisableBit, mujoco.mjtObj
    name = lambda o, i: mujoco.mj_id2name(m, o, i) or str(i)
    out = []
    if not _flag(m, D.mjDSBL_CONSTRAINT):
        if not _flag(m, D.mjDSBL_CONTACT):
            out += [("pair", name(O.mjOBJ_PAIR, p), np.array(m.pair_solref[p])) for p in range(m.npair)]
        if not _flag(m, D.mjDSBL_LIMIT):
            out += [("joint limit", name(O.mjOBJ_JOINT, j), np.array(m.jnt_solref[j])) for j in range(m.njnt)
                    if m.jnt_limited[j] and m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
            out += [("tendon limit", name(O.mjOBJ_TENDON, t), np.array(m.tendon_solref_lim[t]))
                    for t in range(m.ntendon) if m.tendon_limited[t]]
        if not _flag(m, D.mjDSBL_FRICTIONLOSS):
            out += [("frictionloss dof", str(d), np.array(m.dof_solref[d])) for d in range(m.nv)
                    if m.dof_frictionloss[d] > 0]
        if not _flag(m, D.mjDSBL_EQUALITY):
            out += [("equality", name(O.mjOBJ_EQUALITY, e), np.array(m.eq_solref[e])) for e in range(m.neq)
                    if m.eq_active0[e]]
    return out


def form(sr):
    """'standard' (both entries > 0), 'direct' (both <= 0) or 'mixed'."""
    a, b = float(sr[0]) > 0, float(sr[1]) > 0
    return "standard" if a and b else "direct" if not a and not b else "mixed"


def needed(m):
    """Whether a row of the model has a direct solref (apply() edits kbimp)."""
    return any(form(sr) == "direct" for _, _, sr in rows(m))


def problems(m):
    """[reason] of the model's solref uses the forest engine does not read as MuJoCo (module docstring; empty: none)."""
    out = [f"{k} {n}: mixed solref ({sr[0]:g}, {sr[1]:g})" for k, n, sr in rows(m) if form(sr) == "mixed"]
    if _flag(m, mujoco.mjtDisableBit.mjDSBL_REFSAFE):
        h2 = 2.0 * float(m.opt.timestep)
        out += [f"{k} {n}: refsafe disabled, time constant {sr[0]:g} < 2 h" for k, n, sr in rows(m)
                if form(sr) == "standard" and float(np.float32(sr[0])) < float(np.float32(h2))]
    if int(m.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE):
        out.append("mjENBL_OVERRIDE (the contact override o_solref / o_solimp / o_margin / o_friction)")
    return out


def apply(src, m):
    """src (adjoint.cuda_source's unit) with kbimp's direct form for a model that needs it (needed(m)); else src
    unchanged. A text that already has the edit (a foot_solref build, a terrain contact material) is returned as is.
    Raises ValueError for a model with problems() (capability T13 refuses it before a build)."""
    bad = problems(m)
    if bad:
        raise ValueError(f"forest engine solref ({CODE}): {bad[:4]}")
    if not needed(m):
        return src
    from nablaengine.g1_engine import world_params as G
    (old, new, _), = G.DIRECT_SOLREF
    if src.count(new) == 1 and old not in src:
        return src
    if src.count(old) != 1:
        raise RuntimeError(f"forest_engine.solref: kbimp's k / b lines found {src.count(old)} times, expected 1")
    return src.replace(old, new.replace("(world_params: foot_solref)", "(forest_engine.solref)"))

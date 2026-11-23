"""The forest engine beyond 64 dofs (forest_engine.nv128, design the dense factor: the dense baseline) on the nv128 test scenes
(scripts/forest/models/s1.xml nv 113 / 64 bodies, s3.xml nv 65 / 56 bodies, s3nobox nv 59 the control below 64 dofs; their
inactive weld `base_hold` removed,
which the forest engine does not model; MuJoCo C and MuJoCo Warp run the same model):"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eq"))
import connect_check as cc
import connect_grad as cg


def load(scene):
    """The scene (s1 / s2 / s3, an MJCF path, or a compiled .mjb as given) without the inactive weld base_hold. s3nobox: S3 without its box (nv 59,
    55 bodies: the same robot below 64 dofs, a control that main's engine runs too); s1_b<k>: S1 with boxes 0 .. k-1 only."""
    if scene.endswith(".mjb"):
        return mujoco.MjModel.from_binary_path(scene)
    drop = []
    if scene == "s3noself":
        from nablaengine.forest_engine import nv128
        m3 = load("s3")
        info = nv128.hpattern(m3)
        x = re.sub(r'\s*<weld name="base_hold"[^>]*/>', "", (ROOT / "scripts/forest/models/s3.xml").read_text())
        for k, dyn in enumerate(info["pdyn"]):
            if dyn:
                nm = mujoco.mj_id2name(m3, mujoco.mjtObj.mjOBJ_PAIR, k)
                x, n = re.subn(rf'\s*<pair [^>]*name="{re.escape(nm)}"[^>]*/>', "", x)
                assert n == 1, nm
        return mujoco.MjModel.from_xml_string(x)
    if scene == "s3nobox":
        scene, drop = "s3", [0]
    elif re.fullmatch(r"s1_b\d", scene):
        scene, drop = "s1", list(range(int(scene[-1]), 9))
    p = Path(scene) if scene.endswith(".xml") else ROOT / "scripts" / "forest" / "models" / f"{scene}.xml"
    x = re.sub(r'\s*<weld name="base_hold"[^>]*/>', "", p.read_text())
    for k in drop:
        x = re.sub(rf'\s*<body name="box{k}".*?</body>', "", x, flags=re.S)
        x = re.sub(rf'\s*<pair [^>]*\bbox{k}_[^>]*/>', "", x)
    if drop:
        x = re.sub(r'(<key [^>]*qpos=")([^"]*)"',
                   lambda mt: mt.group(1) + " ".join(mt.group(2).split()[:-7 * len(drop)]) + '"', x)
    return mujoco.MjModel.from_xml_string(x)


MAX_CONTACTS = None


def init(m):
    from nablaengine import engines, jax_ffi
    engines.prefer_forest(m)
    return jax_ffi.init(model=m, foot_pairs=[], max_contacts=MAX_CONTACTS)


def c_limits(m, q, v):
    d = mujoco.MjData(m)
    out = []
    for w in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:] = q[w], v[w]
        mujoco.mj_forward(m, d)
        out.append(int((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT).sum()))
    return np.array(out)


def c_contacts(m, q):
    d = mujoco.MjData(m)
    out = []
    for w in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q[w]
        mujoco.mj_forward(m, d)
        out.append(int(d.ncon))
    return np.array(out)


def lq_check(m, q, rec):
    """T4-style: the tape's L_Q against MuJoCo C's M (mj_fullM at the input state, float64) + h diag(kv + damping):
    Q = R' R where the record's flag says the tree form (the tree factor), else L L'. Off the diagonal Q must equal M
    (relative to max |M| of the world); on the diagonal Q - M is h damping, or h (kv + damping) where the actuator's
    kv is on (the closer of the two, relative to M_ii). Also: the share of L_Q's entries off M's tree pattern that are
    nonzero (the tree form keeps them zero)."""
    from nablaengine.forest_engine import tape as ftape
    nv, h = m.nv, float(m.opt.timestep)
    kv = np.zeros(nv)
    for a_ in range(m.nu):
        if int(m.actuator_trntype[a_]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            kv[int(m.jnt_dofadr[int(m.actuator_trnid[a_, 0])])] += -float(m.actuator_biasprm[a_, 2])
    damp = np.asarray(m.dof_damping, np.float64)
    pat = np.eye(nv, dtype=bool)
    for i in range(nv):
        j = int(m.dof_parentid[i])
        while j >= 0:
            pat[i, j] = True
            j = int(m.dof_parentid[j])
    tree = rec.get("lq_tree", np.zeros(len(q), bool))
    d = mujoco.MjData(m)
    off, dia, offpat = [], [], 0
    for w in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q[w]
        mujoco.mj_forward(m, d)
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(m, d, M)
        Lw = ftape.tri_unpack(rec["L_Q"][w].astype(np.float64), nv)
        offpat += int(np.count_nonzero(Lw[~pat & np.tri(nv, dtype=bool)]))
        Q = Lw.T @ Lw if tree[w] else Lw @ Lw.T
        o = ~np.eye(nv, dtype=bool)
        off.append(float(np.abs(Q - M)[o].max() / np.abs(M).max()))
        r = np.diag(Q) - np.diag(M)
        e1, e2 = np.abs(r - h * damp), np.abs(r - h * (damp + kv))
        dia.append(float((np.minimum(e1, e2) / np.diag(M)).max()))
    return dict(tree_form=int(tree.sum()), states=len(q), offdiag_rel=cc.stats(np.array(off)),
                diag_rel=cc.stats(np.array(dia)), nonzero_off_pattern=offpat)


def lh_form_H(m, L, sparse, info=None):
    """H from a record's L_H (packed, float64 result): L L' (dense form) or, for the sparse-Newton form (record flag
    lh_sparse), R' R in the kernel rows of forest_engine.nv128.hpattern (R[r][c] at TRI of the two rows' dofs),
    mapped back to dof order."""
    from nablaengine.forest_engine import nv128, tape as ftape
    nv = m.nv
    if not sparse:
        Lw = ftape.tri_unpack(np.asarray(L, np.float64), nv)
        return Lw @ Lw.T
    info = info or nv128.hpattern(m)
    perm, kpar = info["perm"], info["kpar"]
    T = nv128.ktree(kpar)
    R = np.zeros((nv, nv))
    for r in range(nv):
        for c in T["anc"][r] + [r]:
            a_, b_ = perm[r], perm[c]
            R[r, c] = L[nv128._ptri(a_, b_)]
    Hk = R.T @ R
    H = np.zeros((nv, nv))
    H[np.ix_(perm, perm)] = Hk
    return H


def lh_check(m, rec):
    """The records' L_H forms (the sparse Newton factor: lh_sparse = H factored on P_H, else the dense fallback / dense build) and, for
    the sparse form, the entries of L_H outside the elimination tree's closure (must stay 0: H on P_H)."""
    from nablaengine.forest_engine import nv128
    out = dict(sparse=int(np.sum(rec["lh_sparse"])), states=int(np.size(rec["lh_sparse"])))
    if out["sparse"] and m.nv > 64:
        info = nv128.hpattern(m)
        perm, T = info["perm"], nv128.ktree(info["kpar"])
        inside = np.zeros(m.nv * (m.nv + 1) // 2, bool)
        for r in range(m.nv):
            for c in T["anc"][r] + [r]:
                inside[nv128._ptri(perm[r], perm[c])] = True
        L = np.asarray(rec["L_H"]).reshape(-1, inside.size)[np.asarray(rec["lh_sparse"]).ravel()]
        out["nonzero_outside_closure"] = int(np.count_nonzero(L[:, ~inside]))
        out["closure_nnz"] = int(inside.sum())
    return out


def class_states(m, k, seed=0, t_min=0.1):
    """the sparse Newton factor: k states per contact class (MuJoCo C contacts at the state, classes of
    forest_engine.sparse_pattern.Pattern with the model's pairs): 'none' (corpus states with every free body lifted
    by 1 m: no contact), 'foot' (contacts with the world only), 'hand_box' (a robot - object contact, no dynamic pair),
    'dynamic' (a contact of a pair outside P_H: a robot self contact across branches; the dense fallback). From a
    corpus of 1024 states. Returns q, v, u (float32) and the labels."""
    from nablaengine.forest_engine import sparse_pattern as sp
    q, v, u = cc.corpus(m, 1024, seed, t_min=t_min)
    P = sp.Pattern(m, pairs=[(int(x), int(y)) for x, y in zip(m.pair_geom1, m.pair_geom2)])
    cls = {frozenset((int(p.geoms[0]), int(p.geoms[1]))): ("dynamic" if not p.in_pattern else p.cls) for p in P.pairs}
    d = mujoco.MjData(m)
    pick = {c: [] for c in ("foot", "hand_box", "dynamic")}
    for w in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:] = q[w], v[w]
        mujoco.mj_forward(m, d)
        c = {cls.get(frozenset((int(d.contact.geom1[i]), int(d.contact.geom2[i]))), "?") for i in range(d.ncon)}
        lab = ("dynamic" if "dynamic" in c else "hand_box" if "robot_object" in c
               else "foot" if c and c <= {"world", "chain"} else None)
        if lab and len(pick[lab]) < k:
            pick[lab].append(w)
    idx, labels = [], []
    for lab in ("foot", "hand_box", "dynamic"):
        idx += pick[lab]
        labels += [lab] * len(pick[lab])
    qn = q[:k].copy()
    for j in range(m.njnt):
        if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            qn[:, int(m.jnt_qposadr[j]) + 2] += 1.0
    Q = np.concatenate([qn, q[idx]]).astype(np.float32)
    V = np.concatenate([v[:k], v[idx]]).astype(np.float32)
    U = np.concatenate([u[:k], u[idx]]).astype(np.float32)
    return Q, V, U, ["none"] * k + labels


def states(a, m):
    """The corpus (connect_check.corpus), or with --classes K class_states' selection; (q, v, u, labels or None)."""
    if a.classes:
        return class_states(m, a.classes, a.seed, a.t_min)
    q, v, u = cc.corpus(m, a.states, a.seed, t_min=a.t_min)
    return q, v, u, None


def record_forms(m, info, q, v, u, S):
    """Per world the share of its S substeps whose record has the sparse-Newton sparse L_H (tape.decode lh_sparse)."""
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    from nablaengine.forest_engine import tape as ftape
    B = len(q)
    o = jax_ffi.forward(jnp.asarray(q), jnp.asarray(v), jnp.zeros((B, m.nv), jnp.float32), jnp.asarray(u),
                        jnp.ones((B,), jnp.float32), jnp.zeros((B, int(info.get("com_words", 3))), jnp.float32),
                        substeps=S, iterations=int(m.opt.iterations), tape=True)
    rec = ftape.decode(np.asarray(o[-1]).reshape(B, S, -1), m, info.get("max_contacts") or info.get("maxcon"), info.get("max_rows") or info.get("nefc"))
    return [float(x) for x in rec["lh_sparse"].mean(1)]


def by_label(res, labels, forms):
    """Stats of per-world results by class label and by record form (all sparse / some dense)."""
    out = {}
    if labels:
        for lab in sorted(set(labels)):
            out[lab] = cc.stats(np.array([r for r, l_ in zip(res, labels) if l_ == lab]))
    f = np.array(forms)
    for name, sel in (("form_sparse", f == 1.0), ("form_dense_some", f < 1.0)):
        if sel.any():
            out[name] = cc.stats(np.array(res)[sel])
    return out


def cmd_forward(a, m, rep):
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    from nablaengine.forest_engine import tape as ftape
    q, v, u = cc.corpus(m, a.states, a.seed, t_min=a.t_min)
    if a.iterations:
        m.opt.iterations = a.iterations
    info = init(m)
    rep["engine"] = {k: info.get(k) for k in ("engine", "max_contacts", "max_rows", "arch")}
    B = len(q)
    args = (jnp.asarray(q), jnp.asarray(v), jnp.zeros((B, m.nv), jnp.float32), jnp.asarray(u),
            jnp.ones((B,), jnp.float32), jnp.zeros((B, int(info.get("com_words", 3))), jnp.float32))
    it = int(m.opt.iterations)
    o = jax_ffi.forward(*args, substeps=1, iterations=it, tape=True, contacts=True)
    qe, ve = np.asarray(o[0], np.float64), np.asarray(o[1], np.float64)
    con, tp = o[4], np.asarray(o[5])
    ne = np.asarray(jax_ffi.contact_table(con)["valid"]).sum(axis=1)
    st = jax_ffi.step(*args, substeps=1, iterations=it)
    rep["tape_bitwise_step"] = bool(np.array_equal(np.asarray(st[0]), np.asarray(o[0]))
                                    and np.array_equal(np.asarray(st[1]), np.asarray(o[1])))
    rec = ftape.decode(tp.reshape(B, -1), m, info.get("max_contacts") or info.get("maxcon"), info.get("max_rows") or info.get("nefc"))
    nlim_c = c_limits(m, q.astype(np.float64), v.astype(np.float64))
    rep["tape_nlim_vs_C"] = dict(equal=int((rec["nlim"].ravel() == nlim_c).sum()), states=B,
                                 mean=float(nlim_c.mean()), max=int(nlim_c.max()),
                                 nefc_max=int(rec["nefc"].max()), ncon_max=int(rec["ncon"].max()),
                                 nonconv=int(rec["nonconv"].sum()), unsupported=int((rec["unsupported"] != 0).sum()))
    rep["tape_LQ_vs_C"] = lq_check(m, q, rec)
    rep["tape_LH"] = lh_check(m, rec)
    if a.dump:
        np.savez(a.dump, qpos=qe, qvel=ve, L_H=rec["L_H"], lh_sparse=rec["lh_sparse"], niter=rec["niter"],
                 ncon=rec["ncon"], refact=rec["refact"])
    qc, vc, _ = cc.c_step(m, q, v, u)
    nc = c_contacts(m, q)
    rep["forward_vs_C"] = dict(qvel=cc.stats(cc.rel(ve, vc)), qpos=cc.stats(cc.rel(qe, qc)),
                               contacts_equal=int((ne == nc).sum()), contacts_C_mean=float(nc.mean()),
                               contacts_C_max=int(nc.max()), contacts_engine_mean=float(ne.mean()))
    if a.mjw_python:
        with tempfile.TemporaryDirectory(prefix="nv128_mjw_") as work:
            work = Path(work)
            mujoco.mj_saveModel(m, str(work / "model.mjb"), None)
            np.savez(work / "states.npz", qpos=q, qvel=v, ctrl=u)
            r = subprocess.run([a.mjw_python, str(ROOT / "scripts" / "eq" / "mjw_step.py"), str(work / "model.mjb"),
                                str(work / "states.npz"), str(work / "mjw.npz"), "--substeps", "1", "--iterations",
                                str(it), "--nconmax", "128", "--njmax", "1024"] + (["--dense"] if m.nv <= 60 else []),
                               capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ""})
            if r.returncode:
                raise RuntimeError(r.stderr[-3000:])
            z = dict(np.load(work / "mjw.npz"))
            rep["mjw"] = json.loads(r.stdout.strip().splitlines()[-1])
        rep["forward_vs_mjw"] = dict(qvel=cc.stats(cc.rel(ve, z["qvel"])), qpos=cc.stats(cc.rel(qe, z["qpos"])),
                                     contacts_equal=int((ne == z["ncon"]).sum()))
        rep["mjw_vs_C"] = dict(qvel=cc.stats(cc.rel(z["qvel"], vc)), qpos=cc.stats(cc.rel(z["qpos"], qc)),
                               contacts_equal=int((z["ncon"] == nc).sum()))
    nw = min(a.rollout_worlds, B)
    rng = np.random.default_rng(a.seed + 1)
    lo, hi = cc.ctrl_range(m)
    ur = jnp.asarray(rng.uniform(lo, hi, (nw, m.nu)).astype(np.float32))
    qq, vv, ww = jnp.asarray(q[:nw]), jnp.asarray(v[:nw]), jnp.zeros((nw, m.nv), jnp.float32)
    fr, com = jnp.ones((nw,), jnp.float32), jnp.zeros((nw, int(info.get("com_words", 3))), jnp.float32)
    finite, vmax = True, 0.0
    for _ in range(a.rollout_steps // 10):
        qq, vv, ww = jax_ffi.step(qq, vv, ww, ur, fr, com, substeps=10, iterations=it)
        vn = np.asarray(vv)
        finite &= bool(np.isfinite(vn).all() and np.isfinite(np.asarray(qq)).all())
        vmax = max(vmax, float(np.abs(vn).max()))
    rep["rollout"] = dict(worlds=nw, steps=a.rollout_steps, finite=finite, qvel_absmax=vmax,
                          box_z_min=float(np.asarray(qq)[:, [int(m.jnt_qposadr[j]) + 2 for j in range(1, m.njnt)
                                                            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]].min())
                          if any(m.jnt_type[1:] == mujoco.mjtJoint.mjJNT_FREE) else None)


def cmd_forms(a, m, rep):
    """the sparse Newton factor: on rollouts, the share of world-substeps whose record has the dense L_H (dense_share: a contact of a
    dynamic pair, one outside P_H, in the final Hessian), whose Newton iterations used the low-rank correction
    (lowrank_share) or the dense factor (dense_in_iterations_share: more correction columns than nv128.HW_COLS): --rollout-worlds worlds from the corpus states, --rollout-steps substeps
    as calls of 10 with the tape, random constant controls held for 100 substeps; per call the records' lh_sparse."""
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    from nablaengine.forest_engine import tape as ftape
    q, v, u = cc.corpus(m, max(a.states or 0, a.rollout_worlds), a.seed, t_min=a.t_min)
    if a.iterations:
        m.opt.iterations = a.iterations
    info = init(m)
    it, nw = int(m.opt.iterations), a.rollout_worlds
    rng = np.random.default_rng(a.seed + 3)
    lo, hi = cc.ctrl_range(m)
    qq, vv, ww = jnp.asarray(q[:nw]), jnp.asarray(v[:nw]), jnp.zeros((nw, m.nv), jnp.float32)
    fr, com = jnp.ones((nw,), jnp.float32), jnp.zeros((nw, int(info.get("com_words", 3))), jnp.float32)
    dense, ncon, steps, finite, lowr, dnit = [], [], 0, True, [], []
    from nablaengine.forest_engine import nv128
    pdyn, hits = nv128.hpattern(m)["pdyn"], {}
    for k in range(a.rollout_steps // 10):
        if k % 10 == 0:
            ur = jnp.asarray(rng.uniform(lo, hi, (nw, m.nu)).astype(np.float32))
        o = jax_ffi.forward(qq, vv, ww, ur, fr, com, substeps=10, iterations=it, tape=True)
        qq, vv, ww = o[0], o[1], o[2]
        rec = ftape.decode(np.asarray(o[-1]).reshape(nw, 10, -1), m, info.get("max_contacts") or info.get("maxcon"), info.get("max_rows") or info.get("nefc"))
        dense.append(~rec["lh_sparse"])
        lowr.append(rec["h_lowrank"])
        dnit.append(rec["h_dense"])
        ncon.append(rec["ncon"])
        pr = rec["pair"]
        for p_ in np.unique(pr[(pr >= 0)]):
            if pdyn[int(p_)]:
                hits[int(p_)] = hits.get(int(p_), 0) + int(np.any(pr == p_, axis=-1).sum())
        finite &= bool(np.isfinite(np.asarray(vv)).all())
        steps += 10
    d = np.concatenate(dense, 1)
    nc_ = np.concatenate(ncon, 1)
    rep["forms"] = dict(worlds=nw, substeps=steps, finite=finite, dense_share=float(d.mean()),
                        lowrank_share=float(np.concatenate(lowr, 1).mean()),
                        dense_in_iterations_share=float(np.concatenate(dnit, 1).mean()),
                        dense_share_per_world=cc.stats(d.mean(1)), worlds_ever_dense=int(d.any(1).sum()),
                        dense_share_with_contacts=float(d[nc_ > 0].mean()) if (nc_ > 0).any() else None,
                        ncon_mean=float(nc_.mean()),
                        by_100_substeps=[float(x) for x in d.reshape(nw, -1, 100).mean((0, 2))] if steps % 100 == 0 else None)
    bn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or str(int(m.geom_bodyid[g]))
    top = sorted(hits.items(), key=lambda kv: -kv[1])[:40]
    rep["forms"]["dynamic_pairs_top"] = [dict(pair=p_, bodies=[bn(m.pair_geom1[p_]), bn(m.pair_geom2[p_])],
                                              share=n_ / (nw * steps)) for p_, n_ in top]


def cmd_tapeverify(a, m, rep):
    """A -DDK_TAPE_VERIFY build (every unit; jax_ffi.library_sources wrapped): the forward re-assembles and re-factors H
    at the final zones in every substep and flags records whose L_H differs bitwise from the last Newton factor
    (tape.FLAG_VERIFY_DIFF). Expected: only records whose zones changed after the last factorization (refact) or whose
    last factor was the sparse-Newton low-rank form (h_lowrank: the tape's factor is the full H's, dense). Corpus states
    (--states), --substeps[0] substeps."""
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    from nablaengine.forest_engine import tape as ftape
    orig = jax_ffi.library_sources

    def wrapped(*args, **kw):
        units, consts, defines = orig(*args, **kw)
        return units, consts, {k: list(v) + ["-DDK_TAPE_VERIFY"] for k, v in defines.items()}

    jax_ffi.library_sources = wrapped
    q, v, u, labels = states(a, m)
    if a.iterations:
        m.opt.iterations = a.iterations
    info = init(m)
    S, B = a.substeps[0], len(q)
    o = jax_ffi.forward(jnp.asarray(q), jnp.asarray(v), jnp.zeros((B, m.nv), jnp.float32), jnp.asarray(u),
                        jnp.ones((B,), jnp.float32), jnp.zeros((B, int(info.get("com_words", 3))), jnp.float32),
                        substeps=S, iterations=int(m.opt.iterations), tape=True)
    rec = ftape.decode(np.asarray(o[-1]).reshape(B, S, -1), m, info.get("max_contacts") or info.get("maxcon"), info.get("max_rows") or info.get("nefc"))
    vd, rf = rec["verify_diff"], rec["refact"]
    lr = rec.get("h_lowrank", np.zeros_like(vd))
    rep["tapeverify"] = dict(records=int(vd.size), verify_diff=int(vd.sum()), refact=int(rf.sum()),
                             lowrank_records=int(lr.sum()), sparse_form=int(rec["lh_sparse"].sum()),
                             verify_diff_with_refact=int((vd & rf).sum()),
                             verify_diff_lowrank_no_refact=int((vd & lr & ~rf).sum()),
                             verify_diff_unexplained=int((vd & ~rf & ~lr).sum()))


def cmd_grad(a, m, rep):
    q, v, u, labels = states(a, m)
    rep["labels"] = labels
    if a.iterations:
        m.opt.iterations = a.iterations
    from nablaengine import jax_ffi
    info = init(m)
    mm = cg.tight(m)
    rng = np.random.default_rng(a.seed + 7)
    nv, nu = m.nv, m.nu
    blocks = dict(q=slice(0, nv), v=slice(nv, 2 * nv), u=slice(2 * nv, 2 * nv + nu))
    rep["by_substeps"] = {}
    for S in a.substeps:
        gq = rng.normal(size=(len(q), m.nq))
        gq[:, cg.free_quats(m)] = 0.0
        gv = rng.normal(size=(len(q), nv))
        jax_ffi.read_diagnostics(reset=True)
        _, _, (bq, bv, bu) = cg.engine_vjp(m, info, q, v, u, gq, gv, S, **a.adjoint)
        diag = jax_ffi.read_diagnostics(reset=True)
        res, bl, ncon = [], {k: [] for k in blocks}, []
        dd = mujoco.MjData(mm)
        for w in range(len(q)):
            q64, v64, u64 = (x[w].astype(np.float64) for x in (q, v, u))
            ge = np.concatenate([cg.q_to_tangent(m, q64, bq[w]), bv[w], bu[w]])
            gf = cg.fd_grad(mm, q64, v64, u64, gq[w], gv[w], S)
            res.append(cg.err(ge, gf))
            for k, sl in blocks.items():
                bl[k].append(cg.err(ge, gf, sl))
            ncon.append(cg.rollout(mm, dd, q64, v64, u64, S)[2])
        forms = record_forms(m, info, q, v, u, S)
        rep["by_substeps"][str(S)] = dict(engine_vs_fd=cc.stats(res), blocks={k: cc.stats(x) for k, x in bl.items()},
                                          per_world=[float(x) for x in res], contacts_max=[int(x) for x in ncon],
                                          forms=forms, by_class=by_label(res, labels, forms),
                                          tape_bitwise=cg.tape_bitwise(m, info, q, v, u, S),
                                          replay_mismatch_worlds=int(diag.get("replay_mismatch_worlds", 0)),
                                          nonfinite_rows=int(diag.get("nonfinite_rows", 0)))


def engine_fd(m, info, q, v, u, gq, gv, S, eps):
    """The engine's own float32 central differences of L along every input coordinate of one world (q tangent, v, u):
    all 2 (2 nv + nu) perturbed states in one batched step."""
    import jax.numpy as jnp
    from nablaengine import jax_ffi
    nv, nu = m.nv, m.nu
    Q, V, U = [], [], []
    for i in range(2 * nv + nu):
        for sg in (1.0, -1.0):
            qq, vv, uu = q.astype(np.float64).copy(), v.astype(np.float64).copy(), u.astype(np.float64).copy()
            if i < nv:
                dv = np.zeros(nv); dv[i] = 1.0
                mujoco.mj_integratePos(m, qq, dv, sg * eps)
            elif i < 2 * nv:
                vv[i - nv] += sg * eps
            else:
                uu[i - 2 * nv] += sg * eps
            Q.append(qq); V.append(vv); U.append(uu)
    B = len(Q)
    f = lambda x: jnp.asarray(np.asarray(x, np.float32))
    o = jax_ffi.step(f(Q), f(V), jnp.zeros((B, nv), jnp.float32), f(U), jnp.ones((B,), jnp.float32),
                     jnp.zeros((B, int(info.get("com_words", 3))), jnp.float32), substeps=S, iterations=int(m.opt.iterations))
    L = np.asarray(o[0], np.float64) @ gq + np.asarray(o[1], np.float64) @ gv
    return (L[0::2] - L[1::2]) / (2 * eps)


def cmd_selffd(a, m, rep):
    """The adjoint against the engine's own float32 central differences (eps 1e-4 and 3e-4; the columns where the two
    agree to 1e-2 relative are compared): whether the VJP is the derivative of the engine's forward, independent of
    MuJoCo C's contact differences. The adjoint is the implicit derivative at the solver's optimum: where the Newton
    iterations stop early (5 in these scenes) it is not the derivative of the truncated forward (--iterations)."""
    q, v, u, labels = states(a, m)
    rep["labels"] = labels
    if a.iterations:
        m.opt.iterations = a.iterations
    info = init(m)
    rng = np.random.default_rng(a.seed + 7)
    rep["by_substeps"] = {}
    for S in a.substeps:
        gq = rng.normal(size=(len(q), m.nq))
        gq[:, cg.free_quats(m)] = 0.0
        gv = rng.normal(size=(len(q), m.nv))
        _, _, (bq, bv, bu) = cg.engine_vjp(m, info, q, v, u, gq, gv, S, **a.adjoint)
        res, hi, valid, ncon = [], [], [], []
        for w in range(len(q)):
            ge = np.concatenate([cg.q_to_tangent(m, q[w].astype(np.float64), bq[w]), bv[w], bu[w]])
            f1 = engine_fd(m, info, q[w], v[w], u[w], gq[w], gv[w], S, 1e-4)
            f2 = engine_fd(m, info, q[w], v[w], u[w], gq[w], gv[w], S, 3e-4)
            ok = np.abs(f1 - f2) <= 1e-2 * np.maximum(np.abs(f1), np.abs(f2)) + 1e-3 * max(np.abs(f1).max(), 1e-12)
            sc = max(np.abs(f1[ok]).max(), 1e-12) if ok.any() else 1.0
            res.append(float(np.abs(ge[ok] - f1[ok]).max() / sc) if ok.any() else 0.0)
            k = np.arange(64, m.nv)
            k = k[ok[k]]
            hi.append(float(np.abs(ge[k] - f1[k]).max() / sc) if len(k) else 0.0)
            valid.append(float(ok.mean()))
            ncon.append(int(cg.rollout(m, mujoco.MjData(m), q[w].astype(np.float64), v[w].astype(np.float64),
                                       u[w].astype(np.float64), 1)[2]))
        forms = record_forms(m, info, q, v, u, S)
        rep["by_substeps"][str(S)] = dict(engine_vs_selffd=cc.stats(res), dofs_64up_q=cc.stats(hi),
                                          forms=forms, by_class=by_label(res, labels, forms),
                                          valid_fraction=cc.stats(valid), per_world=[float(x) for x in res],
                                          contacts_C=ncon)


def cmd_torch(a, m, rep):
    import jax
    import jax.numpy as jnp
    import torch
    from nablaengine import engines, jax_ffi, torch_ffi
    engines.prefer_forest(m)
    info = jax_ffi.init(model=m, foot_pairs=[])
    torch_ffi.init(model=m, foot_pairs=[])
    q, v, u = cc.corpus(m, a.states, a.seed, t_min=a.t_min)
    B, it, S = len(q), int(m.opt.iterations), 4
    cw = int(info.get("com_words", 3))
    rng = np.random.default_rng(a.seed + 3)
    gq, gv = rng.normal(size=(B, m.nq)).astype(np.float32), rng.normal(size=(B, m.nv)).astype(np.float32)
    jargs = (jnp.asarray(q), jnp.asarray(v), jnp.zeros((B, m.nv), jnp.float32), jnp.asarray(u),
             jnp.ones((B,), jnp.float32), jnp.zeros((B, cw), jnp.float32))

    def f(qq, vv, uu):
        o = jax_ffi.step(qq, vv, jargs[2], uu, jargs[4], jargs[5], substeps=S, iterations=it)
        return o[0], o[1], o[2]

    (jq, jv, jw), pull = jax.vjp(f, jargs[0], jargs[1], jargs[3])
    jg = pull((jnp.asarray(gq), jnp.asarray(gv), jnp.zeros_like(jw)))
    dev = torch.device("cuda")
    t = [torch.tensor(np.asarray(x), device=dev) for x in jargs]
    for k in (0, 1, 3):
        t[k].requires_grad_(True)
    tq, tv, tw = torch_ffi.step(*t, substeps=S, iterations=it)[:3]
    torch.autograd.backward([tq, tv], [torch.tensor(gq, device=dev), torch.tensor(gv, device=dev)])
    same = lambda x, y: bool(np.array_equal(np.asarray(x), y.detach().cpu().numpy()))
    rep["torch_vs_jax"] = dict(states=B, substeps=S, qpos=same(jq, tq), qvel=same(jv, tv), warm=same(jw, tw),
                               g_qpos=same(jg[0], t[0].grad), g_qvel=same(jg[1], t[1].grad),
                               g_ctrl=same(jg[2], t[3].grad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("mode", choices=("forward", "grad", "selffd", "torch", "forms", "tapeverify"))
    ap.add_argument("--states", type=int, default=None)
    ap.add_argument("--substeps", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--mjw-python", default=None)
    ap.add_argument("--rollout-worlds", type=int, default=16)
    ap.add_argument("--rollout-steps", type=int, default=1000)
    ap.add_argument("--iterations", type=int, default=None, help="the engine's Newton iterations (default: the "
                    "model's, 5 in these scenes; the corpus is made with the model's)")
    ap.add_argument("--max-contacts", type=int, default=None, help="the build's contact capacity (default: step.maxcon)")
    ap.add_argument("--t-min", type=float, default=0.1, help="corpus: states from t_min .. t_min + 0.4 s of the MuJoCo C "
                    "rollouts (connect_check.corpus; later: the robot has fallen, many contacts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clamp-kv", type=int, default=None, help="grad / selffd: jax_ffi.step's clamp_kv (default: its "
                    "own, 0; 1 is the exact derivative where an actuator force is clamped)")
    ap.add_argument("--normal", type=int, default=None, help="grad / selffd: jax_ffi.step's normal (default: its own, 3; "
                    "0 exact)")
    ap.add_argument("--classes", type=int, default=0, help="grad / selffd: K states per contact class (class_states: "
                    "none, foot, hand_box, dynamic) instead of the corpus")
    ap.add_argument("--dump", default=None, help="forward: also save the outputs and the records' L_H (npz)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.adjoint = {k: v for k, v in (("clamp_kv", a.clamp_kv), ("normal", a.normal)) if v is not None}
    a.states = a.states or dict(forward=256, grad=16, selffd=16, torch=64, forms=256, tapeverify=256)[a.mode]
    global MAX_CONTACTS
    MAX_CONTACTS = a.max_contacts
    m = load(a.scene)
    rep = dict(scene=a.scene, mode=a.mode, iterations=a.iterations or int(m.opt.iterations), nv=int(m.nv), nbody=int(m.nbody), npair=int(m.npair), states=a.states,
               mujoco=mujoco.__version__, seed=a.seed, adjoint=a.adjoint)
    dict(forward=cmd_forward, grad=cmd_grad, selffd=cmd_selffd, torch=cmd_torch, forms=cmd_forms, tapeverify=cmd_tapeverify)[a.mode](a, m, rep)
    txt = json.dumps(rep, indent=1)
    print(txt)
    if a.out:
        o = Path(a.out)
        if o.exists():
            sys.exit(f"{o} exists (not overwritten)")
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(txt)


if __name__ == "__main__":
    main()

"""envs.kin's batched forward kinematics against the per-body form it replaced (the reference) and
against MuJoCo C (float64 mj_kinematics / mj_comPos / mj_comVel), per model of the envs tasks, on random states:"""
import argparse
import json
import sys

import mujoco
import numpy as np

from nablaengine.envs import kin as kin_mod

J = mujoco.mjtJoint


def fk_reference(self, o, qpos, qvel=None):
    """The per-body forward kinematics envs.kin had before (the reference)."""
    mv = lambda R, x: o.einsum("bij,bj->bi", R, x)
    mm = lambda A, Bm: o.einsum("bij,bjk->bik", A, Bm)
    eye = o.const(np.eye(3))[None] * o.ones_like(qpos[:, :1])[:, :, None]
    zero = o.zeros_like(qpos[:, :1]) * o.const(np.zeros(3))[None]
    xpos, xmat, w, v0 = [zero], [eye], [zero], [zero]
    for b in range(1, self.nbody):
        p = self.parent[b]
        x = xpos[p] + mv(xmat[p], o.const(self.pos[b])[None] * o.ones_like(qpos[:, :1]))
        R = mm(xmat[p], o.const(self.rot[b])[None] * o.ones_like(qpos[:, :1])[:, :, None])
        wb, vb = w[p], v0[p]
        for jt, axis, jpos, qa, da, q0 in self.joints[b]:
            if jt == J.mjJNT_FREE:
                x = qpos[:, qa:qa + 3]
                q = qpos[:, qa + 3:qa + 7]
                q = q / o.norm(q)[:, None]
                R = kin_mod._quat_mat(o, q)
                if qvel is not None:
                    wb = mv(R, qvel[:, da + 3:da + 6])
                    vb = qvel[:, da:da + 3] - o.cross(wb, x)
                continue
            u = mv(R, o.const(axis)[None] * o.ones_like(qpos[:, :1]))
            if jt == J.mjJNT_SLIDE:
                x = x + u * (qpos[:, qa:qa + 1] - q0)
                if qvel is not None:
                    vb = vb + u * qvel[:, da:da + 1]
                continue
            anchor = x + mv(R, o.const(jpos)[None] * o.ones_like(qpos[:, :1]))
            R = mm(R, kin_mod._axis_angle(o, axis, qpos[:, qa] - q0))
            x = anchor - mv(R, o.const(jpos)[None] * o.ones_like(qpos[:, :1]))
            if qvel is not None:
                wj = u * qvel[:, da:da + 1]
                wb = wb + wj
                vb = vb - o.cross(wj, anchor)
        xpos.append(x)
        xmat.append(R)
        w.append(wb)
        v0.append(vb)
    return kin_mod.Frames(self, o, o.stack(xpos, 1), o.stack(xmat, 1), o.stack(w, 1) if qvel is not None else None,
                          o.stack(v0, 1) if qvel is not None else None)


def outputs(F, k):
    """The compared quantities of Frames F (dict of arrays, worlds first)."""
    o = F.o
    d = {"xpos": F.xpos, "xmat": F.xmat, "w": F.w, "v0": F.v0, "xipos": F.xipos}
    if k.m.nsite:
        d["site_xpos"] = o.stack([F.site_xpos(i) for i in range(k.m.nsite)], 1)
        d["site_xmat"] = o.stack([F.site_xmat(i) for i in range(k.m.nsite)], 1)
    d["geom_xpos"] = o.stack([F.geom_xpos(i) for i in range(k.m.ngeom)], 1)
    d["subtree_com"] = o.stack([F.subtree_com(b) for b in range(1, k.nbody)], 1)
    d["subtree_linvel"] = o.stack([F.subtree_linvel(b) for b in range(1, k.nbody)], 1)
    return d


def random_states(m, B, rng):
    qpos = np.tile(m.qpos0, (B, 1))
    qvel = rng.normal(0, 2.0, (B, m.nv))
    for j in range(m.njnt):
        a, t = m.jnt_qposadr[j], m.jnt_type[j]
        if t == J.mjJNT_FREE:
            qpos[:, a:a + 3] = rng.normal(0, 1.0, (B, 3))
            qpos[:, a + 3:a + 7] = rng.normal(0, 1.0, (B, 4)) * rng.uniform(0.5, 2.0, (B, 1))
        elif t == J.mjJNT_HINGE:
            qpos[:, a] += rng.uniform(-np.pi, np.pi, B)
        elif t == J.mjJNT_SLIDE:
            qpos[:, a] += rng.normal(0, 0.5, B)
    return qpos, qvel


def mujoco_outputs(m, qpos, qvel):
    """xpos, xmat, xipos, site_xpos, w, v0 of MuJoCo C (float64)."""
    d = mujoco.MjData(m)
    out = {"xpos": [], "xmat": [], "xipos": [], "w": [], "v0": []}
    if m.nsite:
        out["site_xpos"] = []
    for q, v in zip(qpos, qvel):
        d.qpos[:], d.qvel[:] = q, v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        out["xpos"].append(d.xpos.copy())
        out["xmat"].append(d.xmat.reshape(-1, 3, 3).copy())
        out["xipos"].append(d.xipos.copy())
        if m.nsite:
            out["site_xpos"].append(d.site_xpos.copy())
        ang, lin = d.cvel[:, :3], d.cvel[:, 3:]
        c = d.subtree_com[m.body_rootid]
        out["w"].append(ang.copy())
        out["v0"].append(lin + np.cross(c, ang))
    return {k: np.stack(v) for k, v in out.items()}


def fr_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-jax", action="store_true")
    ap.add_argument("--tol", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    from nablaengine import envs
    from nablaengine.envs.ops import TorchOps
    names = a.tasks.split(",") if a.tasks else envs.tasks()
    ot = TorchOps(a.device)

    def fr_torch(loss, q, v):
        q, v = q.clone().requires_grad_(True), v.clone().requires_grad_(True)
        L, d = loss(q, v)
        gq, gv = torch.autograd.grad(L, (q, v))
        return ({n: x.detach().double().cpu().numpy() for n, x in d.items()}, gq.double().cpu().numpy(),
                gv.double().cpu().numpy())
    backends = {"torch": (ot, lambda x: torch.as_tensor(x, dtype=torch.float32, device=ot.device), fr_torch)}
    if not a.no_jax:
        import jax
        import jax.numpy as jnp
        from nablaengine.envs.ops import JaxOps

        def fr_jax(loss, q, v):
            (L, d), (gq, gv) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(q, v)
            return ({n: np.asarray(x, np.float64) for n, x in d.items()}, np.asarray(gq, np.float64),
                    np.asarray(gv, np.float64))
        backends["jax"] = (JaxOps(), lambda x: jnp.asarray(x, jnp.float32), fr_jax)

    report, worst, seen = {}, 0.0, {}
    for name in names:
        xml, assets = envs.spec(name).source()
        key = (xml, tuple(sorted(assets)))
        if key in seen:
            report[name] = {"same_model_as": seen[key]}
            continue
        seen[key] = name
        m = mujoco.MjModel.from_xml_string(xml, assets)
        k = kin_mod.Kinematics(m)
        rng = np.random.default_rng(a.seed)
        qpos, qvel = random_states(m, a.batch, rng)
        mj = mujoco_outputs(m, qpos.astype(np.float32).astype(np.float64), qvel.astype(np.float32).astype(np.float64))
        r = {"nbody": int(m.nbody), "nq": int(m.nq)}
        for bname, (o, to, fr) in backends.items():
            d0 = outputs(k.fk(o, to(qpos[:2]), to(qvel[:2])), k)
            wts = {n: to(rng.normal(0, 1, (1,) + tuple(x.shape[1:]))) for n, x in d0.items()}
            res = {}
            forms = (("new", lambda q, v: k.fk(o, q, v)), ("ref", lambda q, v: fk_reference(k, o, q, v)))
            if bname == "jax":
                forms += (("batched", lambda q, v: k._fk_batched(o, q, v)),)
            for tag, fk in forms:
                def loss(q, v, fk=fk):
                    d = outputs(fk(q, v), k)
                    return sum((d[n] * wts[n]).sum() for n in d), d
                res[tag] = fr(loss, to(qpos), to(qvel))
            (vn, gqn, gvn), (vr, gqr, gvr) = res["new"], res["ref"]
            rb = {}
            if "batched" in res:
                vb, gqb, gvb = res["batched"]
                e = max([float(np.abs(vb[n] - vr[n]).max() / max(1.0, np.abs(vr[n]).max())) for n in vb] +
                        [float(np.abs(gb - gr).max() / max(1.0, np.abs(gr).max()))
                         for gb, gr in ((gqb, gqr), (gvb, gvr))])
                rb["batched_form_relative"] = {"new_vs_ref": e, "scale": 1.0}
                worst = max(worst, e)
            for n in vn:
                scale = max(1.0, float(np.abs(vr[n]).max()))
                e = float(np.abs(vn[n] - vr[n]).max())
                rb[n] = {"new_vs_ref": e, "scale": scale}
                if n in mj:
                    rb[n]["new_vs_mujoco"] = float(np.abs(vn[n] - mj[n]).max())
                    rb[n]["ref_vs_mujoco"] = float(np.abs(vr[n] - mj[n]).max())
                worst = max(worst, e / scale)
            for gname, gn, gr in (("grad_qpos", gqn, gqr), ("grad_qvel", gvn, gvr)):
                scale = max(1.0, float(np.abs(gr).max()))
                e = float(np.abs(gn - gr).max())
                rb[gname] = {"new_vs_ref": e, "scale": scale}
                worst = max(worst, e / scale)
            Fn, Fr = k.fk(o, to(qpos), None), fk_reference(k, o, to(qpos), None)
            e = max(float(np.abs(np.asarray(fr_np(Fn.xpos)) - np.asarray(fr_np(Fr.xpos))).max()),
                    float(np.abs(np.asarray(fr_np(Fn.xmat)) - np.asarray(fr_np(Fr.xmat))).max()))
            rb["no_qvel_frames"] = {"new_vs_ref": e, "scale": rb["xpos"]["scale"]}
            worst = max(worst, e / rb["xpos"]["scale"])
            r[bname] = rb
        report[name] = r
        summary = {b: max(v["new_vs_ref"] / v["scale"] for v in r[b].values()) for b in backends}
        print(name, json.dumps(summary), file=sys.stderr, flush=True)
    out = {"worst_relative": worst, "tol": a.tol, "batch": a.batch, "device": a.device, "tasks": report}
    s = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    print(s)
    sys.exit(0 if worst <= a.tol else 1)


if __name__ == "__main__":
    main()

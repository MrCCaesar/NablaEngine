"""Forward kinematics of an MJCF model's original structure (nablaengine.envs): the body, site and geom frames, the
COMs and the velocities MuJoCo's mj_kinematics / mj_comPos / subtree sensors give, as differentiable array code (JAX or
PyTorch, envs.ops) for the task observations and rewards. The physics runs on the converted model (envs.convert);
the qpos / qvel layouts are the same, so these quantities refer to the original bodies, sites and geoms by name."""
from dataclasses import dataclass
from functools import cached_property

import os

import mujoco
import numpy as np

_FK = os.environ.get("DK_ENVS_FK", "auto")
assert _FK in ("auto", "batched", "bodies"), f"DK_ENVS_FK={_FK!r}: use auto, batched or bodies"

J = mujoco.mjtJoint


def _quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


class Kinematics:
    def __init__(self, m):
        self.m = m
        self.nbody = m.nbody
        self.parent = [int(x) for x in m.body_parentid]
        self.pos = np.array(m.body_pos)
        self.rot = np.stack([_quat2mat(q) for q in m.body_quat])
        self.ipos = np.array(m.body_ipos)
        self.irot = np.stack([_quat2mat(q) for q in m.body_iquat])
        self.mass = np.array(m.body_mass)
        self.joints = [[(int(m.jnt_type[j]), np.array(m.jnt_axis[j]), np.array(m.jnt_pos[j]), int(m.jnt_qposadr[j]),
                         int(m.jnt_dofadr[j]), float(m.qpos0[m.jnt_qposadr[j]]))
                        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])]
                       for b in range(m.nbody)]
        self.site_body, self.site_pos = np.array(m.site_bodyid), np.array(m.site_pos)
        self.site_rot = np.stack([_quat2mat(q) for q in m.site_quat]) if m.nsite else np.zeros((0, 3, 3))
        self.geom_body, self.geom_pos = np.array(m.geom_bodyid), np.array(m.geom_pos)
        self.geom_rot = np.stack([_quat2mat(q) for q in m.geom_quat]) if m.ngeom else np.zeros((0, 3, 3))
        self.sub = [[c for c in range(m.nbody) if self._descends(c, b)] for b in range(m.nbody)]

    def _descends(self, c, b):
        while c != b and c != 0:
            c = self.parent[c]
        return c == b

    def body(self, name):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)

    def site(self, name):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name)

    def geom(self, name):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, name)

    def _fk_bodies(self, o, qpos, qvel=None):
        mv = lambda R, x: o.einsum("bij,bj->bi", R, x)
        mm = lambda A, Bm: o.einsum("bij,bjk->bik", A, Bm)
        one = o.ones_like(qpos[:, :1])
        eye = o.const(np.eye(3))[None] * one[:, :, None]
        zero = o.zeros_like(qpos[:, :1]) * o.const(np.zeros(3))[None]
        xpos, xmat, w, v0 = [zero], [eye], [zero], [zero]
        for b in range(1, self.nbody):
            p = self.parent[b]
            x = xpos[p] + mv(xmat[p], o.const(self.pos[b])[None] * one)
            R = mm(xmat[p], o.const(self.rot[b])[None] * one[:, :, None])
            wb, vb = w[p], v0[p]
            for jt, axis, jpos, qa, da, q0 in self.joints[b]:
                if jt == J.mjJNT_FREE:
                    x = qpos[:, qa:qa + 3]
                    q = qpos[:, qa + 3:qa + 7]
                    q = q / o.norm(q)[:, None]
                    R = _quat_mat(o, q)
                    if qvel is not None:
                        wb = mv(R, qvel[:, da + 3:da + 6])
                        vb = qvel[:, da:da + 3] - o.cross(wb, x)
                    continue
                u = mv(R, o.const(axis)[None] * one)
                if jt == J.mjJNT_SLIDE:
                    x = x + u * (qpos[:, qa:qa + 1] - q0)
                    if qvel is not None:
                        vb = vb + u * qvel[:, da:da + 1]
                    continue
                assert jt == J.mjJNT_HINGE, "ball joints: not supported"
                anchor = x + mv(R, o.const(jpos)[None] * one)
                R = mm(R, _axis_angle(o, axis, qpos[:, qa] - q0))
                x = anchor - mv(R, o.const(jpos)[None] * one)
                if qvel is not None:
                    wj = u * qvel[:, da:da + 1]
                    wb = wb + wj
                    vb = vb - o.cross(wj, anchor)
            xpos.append(x)
            xmat.append(R)
            w.append(wb)
            v0.append(vb)
        return Frames(self, o, o.stack(xpos, 1), o.stack(xmat, 1), o.stack(w, 1) if qvel is not None else None,
                      o.stack(v0, 1) if qvel is not None else None)

    def _plan(self):
        m, nb = self.m, self.nbody
        depth = [0] * nb
        for b in range(1, nb):
            depth[b] = depth[self.parent[b]] + 1
        free = [b for b in range(1, nb) if any(j[0] == J.mjJNT_FREE for j in self.joints[b])]
        for b in free:
            assert len(self.joints[b]) == 1 and self.parent[b] == 0, "free joints: alone, on bodies of the world"
        assert m.body_jntnum[0] == 0
        rest = sorted((b for b in range(1, nb) if b not in free), key=lambda b: depth[b])
        order = free + rest
        P = {"free": free, "rest": rest, "order": order}
        pos_of = {b: i for i, b in enumerate(order)}
        levels, start = [], 0
        while start < len(order):
            d = depth[order[start]]
            stop = start
            while stop < len(order) and depth[order[stop]] == d:
                stop += 1
            levels.append((start, stop))
            start = stop
        P["levels"] = levels
        P["lpar"] = [np.array([pos_of[self.parent[order[i]]] - levels[k - 1][0] for i in range(s0, s1)])
                     for k, (s0, s1) in enumerate(levels) if k > 0]
        P["perm"] = np.array([0] + [1 + pos_of[b] for b in range(1, nb)])
        nr, K = len(rest), max([len(self.joints[b]) for b in rest] + [0])
        qi, di = np.zeros((K, nr), int), np.zeros((K, nr), int)
        q0, hm, sm = np.zeros((K, nr)), np.zeros((K, nr)), np.zeros((K, nr))
        ax, jp = np.zeros((K, nr, 3)), np.zeros((K, nr, 3))
        for i, b in enumerate(rest):
            for k, (jt, axis, jpos, qa, da, qz) in enumerate(self.joints[b]):
                assert jt in (J.mjJNT_HINGE, J.mjJNT_SLIDE), "ball joints: not supported"
                qi[k, i], di[k, i], q0[k, i], ax[k, i], jp[k, i] = qa, da, qz, axis, jpos
                (hm if jt == J.mjJNT_HINGE else sm)[k, i] = 1
        Ks = np.zeros((K, nr, 3, 3))
        Ks[..., 0, 1], Ks[..., 0, 2], Ks[..., 1, 2] = -ax[..., 2], ax[..., 1], -ax[..., 0]
        Ks[..., 1, 0], Ks[..., 2, 0], Ks[..., 2, 1] = ax[..., 2], -ax[..., 1], ax[..., 0]
        KK = Ks @ Ks
        rot, pos = self.rot[rest], self.pos[rest]
        T = lambda a: np.moveaxis(a, 0, -1)[..., None]
        P.update(K=K, qi=qi, di=di, q0=q0[..., None], hm=[h[:, None] for h in hm], sm=[h[:, None] for h in sm],
                 present=(hm + sm)[..., None], padded=bool((hm + sm).min(initial=1) == 0), mixed=hm.any(1) & sm.any(1),
                 sm_any=sm.any(1), hm_any=hm.any(1), rot=T(rot), pos=T(pos))
        P["slots"] = list(range(1, K))
        if K:
            ra = np.einsum("nij,nj->ni", rot, ax[0])
            anchor = pos + np.einsum("nij,nj->ni", rot, jp[0])
            P.update(rK=T(rot @ Ks[0]), rKK=T(rot @ KK[0]), rKj=T(np.einsum("nij,nj->ni", rot @ Ks[0], jp[0])),
                     rKKj=T(np.einsum("nij,nj->ni", rot @ KK[0], jp[0])), ra=T(ra), cw=T(ra * hm[0][:, None]),
                     cv=T(ra * sm[0][:, None] + np.cross(anchor, ra) * hm[0][:, None]))
            P["H"] = [T(np.concatenate([Ks[k], KK[k], (Ks[k] @ jp[k][..., None]), (KK[k] @ jp[k][..., None]),
                                        ax[k][..., None], jp[k][..., None]], -1)) for k in range(1, K)]
        P["fq"] = np.array([self.joints[b][0][3] for b in free], int).reshape(-1)
        P["fd"] = np.array([self.joints[b][0][4] for b in free], int).reshape(-1)
        world = np.zeros((3, 6, 1, 1))
        world[:, :3, 0, 0] = np.eye(3)
        P["world"] = world
        return P

    def fk(self, o, qpos, qvel=None):
        """Frames of the batch qpos (B, nq), qvel (B, nv) (None: no velocities)."""
        if o.name == "jax" or _FK == "bodies" or (_FK == "auto" and getattr(qpos, "device", None) is not None
                                                  and qpos.device.type != "cuda"):
            return self._fk_bodies(o, qpos, qvel)
        return self._fk_batched(o, qpos, qvel)

    def _fk_batched(self, o, qpos, qvel=None):
        if not hasattr(self, "_P"):
            self._P = self._plan()
        P, c = self._P, o.const
        vel = qvel is not None
        qT, vT = _t(o, qpos), (_t(o, qvel) if vel else None)
        loc = []
        if len(P["free"]):
            fq = P["fq"][None]
            x = _take(o, qT, fq + np.arange(3)[:, None], 0)
            q = _take(o, qT, fq + np.arange(3, 7)[:, None], 0)
            q = q / o.sqrt(o.sum(q * q, 0))[None]
            R = _quat_mat_t(o, q)
            cols = [R, x[:, None]]
            if vel:
                fd = P["fd"][None]
                w = o.sum(R * _take(o, vT, fd + np.arange(3, 6)[:, None], 0)[None], 1)
                cols += [w[:, None], (_take(o, vT, fd + np.arange(3)[:, None], 0) - _cross(o, w, x))[:, None]]
            loc.append(o.concat(cols, 1))
        if len(P["rest"]):
            K = P["K"]
            if K:
                qq = _take(o, qT, P["qi"], 0) - c(P["q0"])
                th = qq * c(np.stack(P["hm"]))
                s, c1 = _unbind(o, o.sin(th), 0), _unbind(o, 1 - o.cos(th), 0)
                d = _unbind(o, qq * c(np.stack(P["sm"])), 0)
                R = c(P["rot"]) + s[0] * c(P["rK"]) + c1[0] * c(P["rKK"])
                x = c(P["pos"]) - s[0] * c(P["rKj"]) - c1[0] * c(P["rKKj"])
                if P["sm_any"][0]:
                    x = x + d[0] * c(P["ra"])
                if vel:
                    qd = _take(o, vT, P["di"], 0)
                    qd = _unbind(o, qd * c(P["present"]) if P["padded"] else qd, 0)
                    w = qd[0] * c(P["cw"])
                    v = qd[0] * c(P["cv"])
                for k in P["slots"]:
                    H = o.sum(R[:, :, None] * c(P["H"][k - 1])[None], 1)
                    hK, hKK, hKj, hKKj, u, hj = (_sq1(h) if h.shape[1] == 1 else h
                                                 for h in _split(o, H, [3, 3, 1, 1, 1, 1], 1))
                    xk = x - s[k] * hKj - c1[k] * hKKj
                    if P["sm_any"][k]:
                        xk = xk + d[k] * u
                    if vel:
                        a = qd[k] * u
                        if P["mixed"][k]:
                            ah, as_ = a * c(P["hm"][k]), a * c(P["sm"][k])
                            w, v = w + ah, v + _cross(o, x + hj, ah) + as_
                        elif P["hm_any"][k]:
                            w, v = w + a, v + _cross(o, x + hj, a)
                        else:
                            v = v + a
                    R = R + s[k] * hK + c1[k] * hKK
                    x = xk
            else:
                one = o.ones_like(qT[:1])
                R = c(P["rot"]) * one
                x = c(P["pos"]) * one
                w = v = x * 0
            loc.append(o.concat([R, x[:, None]] + ([w[:, None], v[:, None]] if vel else []), 1))
        L = loc[0] if len(loc) == 1 else o.concat(loc, 2)
        nc = 6 if vel else 4
        lv = P["levels"]
        Ls = _split(o, L, [e - b for b, e in lv], 2) if len(lv) > 1 else [L]
        out = [c(P["world"][:, :nc]) + o.zeros_like(qT[:1])[None, None], Ls[0]]
        prev = Ls[0]
        for Ll, lp in zip(Ls[1:], P["lpar"]):
            Sp = _take(o, prev, lp, 2)
            Rp, *rest = _split(o, Sp, [3, 1, 1, 1] if vel else [3, 1], 1)
            G = o.sum(Rp[:, :, None] * Ll[None], 1)
            if vel:
                xp, wp, vp = rest
                GR, Gx, Gw, Gv = _split(o, G, [3, 1, 1, 1], 1)
                prev = o.concat([GR, Gx + xp, Gw + wp, Gv + vp + _cross(o, xp, Gw)], 1)
            else:
                GR, Gx = _split(o, G, [3, 1], 1)
                prev = o.concat([GR, Gx + rest[0]], 1)
            out.append(prev)
        S = _contig(o, _transpose(o, _take(o, o.concat(out, 2), P["perm"], 2), (3, 2, 0, 1)))
        f = _split(o, S, [3, 1, 1, 1] if vel else [3, 1])
        return Frames(self, o, _sq(f[1]), f[0], _sq(f[2]) if vel else None, _sq(f[3]) if vel else None)


def _t(o, x):
    return x.T


def _transpose(o, x, axes):
    return x.permute(*axes) if o.name == "torch" else o.np.transpose(x, axes)


def _contig(o, x):
    return x.contiguous() if o.name == "torch" else x


def _cross(o, a, b):
    if o.name == "torch":
        return o.torch.linalg.cross(a, b, dim=0)
    return o.np.cross(a, b, axis=0)


def _sq1(x):
    return x.reshape(tuple(x.shape[:1]) + tuple(x.shape[2:]))


def _quat_mat_t(o, q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    r = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    return o.stack(r, 0).reshape((3, 3) + tuple(q.shape[1:]))


def _sq(x):
    return x.reshape(x.shape[:-1])


def _take(o, x, idx, axis):
    idx = np.asarray(idx)
    if o.name == "torch":
        t = o.torch.index_select(x, axis, o.index(idx.reshape(-1)))
        return t.reshape(tuple(x.shape[:axis]) + idx.shape + tuple(x.shape[axis + 1:]))
    return o.np.take(x, idx, axis=axis)


def _split(o, x, sizes, axis=-1):
    if o.name == "torch":
        return o.torch.split(x, sizes, axis)
    return o.np.split(x, np.cumsum(sizes)[:-1], axis=axis)


def _unbind(o, x, axis):
    if o.name == "torch":
        return x.unbind(axis)
    return [o.np.take(x, i, axis=axis) for i in range(x.shape[axis])]


def _quat_mat(o, q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    return o.stack(r, -1).reshape(q.shape[:-1] + (3, 3))


def _axis_angle(o, axis, ang):
    a = np.asarray(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], float)
    c, s = o.cos(ang)[:, None, None], o.sin(ang)[:, None, None]
    return o.const(np.eye(3))[None] + s * o.const(K)[None] + (1 - c) * o.const(K @ K)[None]


@dataclass
class Frames:
    kin: Kinematics
    o: object
    xpos: object
    xmat: object
    w: object
    v0: object

    @cached_property
    def xipos(self):
        if self.o.name == "jax":
            return self.xpos + self.o.einsum("bnij,nj->bni", self.xmat, self.o.const(self.kin.ipos))
        return self.xpos + (self.xmat * self.o.const(self.kin.ipos)[None, :, None]).sum(-1)

    def site_xpos(self, i):
        k = self.kin
        b = int(k.site_body[i])
        return self.xpos[:, b] + (self.xmat[:, b] * self.o.const(k.site_pos[i])).sum(-1)

    def site_xmat(self, i):
        k = self.kin
        b = int(k.site_body[i])
        return self.xmat[:, b] @ self.o.const(k.site_rot[i])

    def geom_xpos(self, i, local_pos=None):
        """geom i's world position (local_pos: its position in its body frame instead of the model's, e.g. a target
        moved per world: (B, 3))."""
        k = self.kin
        b = int(k.geom_body[i])
        lp = self.o.const(k.geom_pos[i]) if local_pos is None else local_pos[:, None]
        return self.xpos[:, b] + (self.xmat[:, b] * lp).sum(-1)

    def point_vel(self, b, p):
        """Velocity (B, 3) of the point p (B, 3) moving with body b."""
        return self.v0[:, b] + self.o.cross(self.w[:, b], p)

    @cached_property
    def _comvel(self):
        return self.v0 + self.o.cross(self.w, self.xipos)

    def _subtree_mean(self, x, b):
        k = self.kin
        wt = np.zeros(k.nbody)
        wt[k.sub[b]] = k.mass[k.sub[b]] / k.mass[k.sub[b]].sum()
        return (x * self.o.const(wt)[None, :, None]).sum(1)

    def subtree_com(self, b):
        return self._subtree_mean(self.xipos, b)

    def subtree_linvel(self, b):
        """MuJoCo's subtreelinvel sensor of body b: the subtree's linear momentum / mass."""
        return self._subtree_mean(self._comvel, b)

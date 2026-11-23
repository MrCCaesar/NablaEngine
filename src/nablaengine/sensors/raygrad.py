"""First-order gradients of ray hit distances (shared by jax_rays and torch_rays; written for jax.numpy and torch)."""
MIN_COS = 1e-2
ID_SHIFT, ID_MASK = 24, (1 << 24) - 1
CAT_OBJECT, CAT_ROBOT = 3, 4


def qmat(xp, q):
    """Rotation matrices (..., 3, 3) of unit quaternions q (..., 4) (w, x, y, z), as G1Sensors._qmat."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return xp.stack([xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     xp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     xp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)



def _ein(xp, *args):
    if xp.__name__ == "jax.numpy":
        import jax
        return xp.einsum(*args, precision=jax.lax.Precision.HIGHEST)
    return xp.einsum(*args)

def _gather(xp, a, idx):
    B, N = idx.shape
    tail = tuple(a.shape[2:])
    flat = a.reshape(B, a.shape[1], -1)
    C = flat.shape[2]
    if xp.__name__ == "torch":
        out = xp.gather(flat, 1, idx.long()[..., None].expand(B, N, C))
    else:
        out = xp.take_along_axis(flat, xp.broadcast_to(idx[..., None], (B, N, C)), axis=1, mode="fill", fill_value=0)
    return out.reshape((B, N) + tail)


def _none(xp, M):
    return 0 if xp.__name__ == "torch" else M


def _rays(xp, pos, rot, off, dirs):
    o = pos[:, None, :] + _ein(xp, "bij,nj->bni", rot, off)
    r = _ein(xp, "bij,nj->bni", rot, dirs)
    return o, r


def _arange(xp, n, like):
    return xp.arange(n, device=like.device) if xp.__name__ == "torch" else xp.arange(n)


def residuals(xp, pos, rot, off, dirs, xpos, xquat, objects, dist, gid, nrm, mbody, min_cos=MIN_COS):
    """The constants of the tangent map at the primal point (no gradient through them): w = n / (n . r) (0 where not
    valid: misses, grazing), t, the hit point and normal, the object slot one-hot of each hit (B, N, K) with the hit
    point and normal in the slot's frame, the robot body of each hit with the hit point and normal in the body's
    frame. nrm: the normal the tangent maps use (the RGB: facing the camera)."""
    o, r = _rays(xp, pos, rot, off, dirs)
    hit = o + dist[..., None] * r
    den = (nrm * r).sum(-1)
    valid = (dist >= 0) & (abs(den) >= min_cos)
    inv = xp.where(valid, 1.0 / xp.where(valid, den, xp.ones_like(den)), xp.zeros_like(den))
    cat, idx = gid >> ID_SHIFT, gid & ID_MASK
    hitm = dist >= 0
    res = dict(w=nrm * inv[..., None], dist=dist, hit=hit, nrm=nrm)
    K = objects.shape[1]
    if K:
        oh = (hitm & (cat == CAT_OBJECT))[..., None] & (idx[..., None] == _arange(xp, K, idx))
        res["oh"] = oh.to(dist.dtype) if xp.__name__ == "torch" else oh.astype(dist.dtype)
        P, Ro = objects[..., 0:3], objects[..., 3:12].reshape(objects.shape[:2] + (3, 3))
        loc = _ein(xp, "bkji,bnj->bnki", Ro, hit) - _ein(xp, "bkji,bkj->bki", Ro, P)[:, None]
        res["oloc"] = _ein(xp, "bnk,bnki->bni", res["oh"], loc)
        res["onloc"] = _ein(xp, "bnk,bkji,bnj->bni", res["oh"], Ro, nrm)
    nm = mbody.shape[0]
    res["rob"] = hitm & (cat == CAT_ROBOT)
    gi = idx.clip(0, nm - 1)
    bidx = mbody[gi.long() if xp.__name__ == "torch" else gi]
    res["bidx"] = xp.where(res["rob"], bidx.to(idx.dtype) if xp.__name__ == "torch" else bidx,
                           xp.full_like(idx, _none(xp, xpos.shape[1])))
    Rb = qmat(xp, _gather(xp, xquat, res["bidx"]))
    res["rloc"] = _ein(xp, "bnji,bnj->bni", Rb, hit - _gather(xp, xpos, res["bidx"]))
    res["rnloc"] = _ein(xp, "bnji,bnj->bni", Rb, nrm)
    return res


def tangent_dist(xp, pos, rot, off, dirs, xpos, xquat, objects, res):
    """A function of the inputs whose derivative at the primal point is the first-order dt (its value has no
    meaning): -w . (o + t r - x), x the hit's material point (moving with its object slot or robot body; static
    surfaces: no term)."""
    w, t = res["w"], res["dist"]
    g = -(w * pos[:, None]).sum(-1) - _ein(xp, "bni,bij,nj->bn", w, rot, off) \
        - t * _ein(xp, "bni,bij,nj->bn", w, rot, dirs)
    if "oh" in res:
        P, Ro = objects[..., 0:3], objects[..., 3:12].reshape(objects.shape[:2] + (3, 3))
        g = g + _ein(xp, "bnk,bni,bki->bn", res["oh"], w, P) + _ein(xp, "bnk,bni,bkij,bnj->bn", res["oh"], w, Ro,
                                                                           res["oloc"])
    Rb = qmat(xp, _gather(xp, xquat, res["bidx"]))
    xr = _gather(xp, xpos, res["bidx"]) + _ein(xp, "bnij,bnj->bni", Rb, res["rloc"])
    return g + xp.where(res["rob"], (w * xr).sum(-1), xp.zeros_like(g))



def vjp_torch(u, pos, rot, off, dirs, xpos, xquat, objects, dist, gid, nrm, mbody, min_cos, need):
    """The transpose of tangent_dist's derivative in closed form for PyTorch (autograd through residuals /
    tangent_dist is several times slower): cotangents of (pos, rot, off, dirs, xpos, xquat, objects) for dist's
    cotangent u (B, N), None where need[i] is False. Moving surfaces are handled on their rays only (index_add)."""
    import torch
    B, N = dist.shape
    o, r = _rays(torch, pos, rot, off, dirs)
    den = (nrm * r).sum(-1)
    valid = (dist >= 0) & (den.abs() >= min_cos)
    W = torch.where(valid, u / torch.where(valid, den, torch.ones_like(den)), torch.zeros_like(den))[..., None] * nrm
    Wt = W * dist[..., None]
    out = [None] * 7
    if need[0]:
        out[0] = -W.sum(1)
    if need[1]:
        out[1] = -(torch.einsum("bni,nj->bij", W, off) + torch.einsum("bni,nj->bij", Wt, dirs))
    if need[2]:
        out[2] = -torch.einsum("bni,bij->nj", W, rot)
    if need[3]:
        out[3] = -torch.einsum("bni,bij->nj", Wt, rot)
    cat, idx = gid >> ID_SHIFT, (gid & ID_MASK).long()

    def subset(c):
        sel = (valid & (cat == c)).reshape(-1).nonzero().squeeze(1)
        b = sel // N
        hit = o.reshape(-1, 3)[sel] + dist.reshape(-1)[sel, None] * r.reshape(-1, 3)[sel]
        return sel, b, hit, W.reshape(-1, 3)[sel]

    K = objects.shape[1]
    if need[6]:
        g = torch.zeros(B * K, 12, device=u.device, dtype=u.dtype)
        if K:
            sel, b, hit, Ws = subset(CAT_OBJECT)
            k = idx.reshape(-1)[sel].clamp(0, K - 1)
            ob = objects[b, k]
            Ro = ob[:, 3:12].reshape(-1, 3, 3)
            loc = torch.einsum("sji,sj->si", Ro, hit - ob[:, 0:3])
            g.index_add_(0, b * K + k, torch.cat([Ws, (Ws[:, :, None] * loc[:, None, :]).flatten(1)], 1))
        out[6] = torch.cat([g.reshape(B, K, 12), torch.zeros(B, K, objects.shape[2] - 12, device=u.device,
                                                              dtype=u.dtype)], 2)
    if need[4] or need[5]:
        nb = xpos.shape[1]
        sel, b, hit, Ws = subset(CAT_ROBOT)
        body = mbody[idx.reshape(-1)[sel].clamp(0, mbody.shape[0] - 1)]
        Rb = qmat(torch, xquat[b, body])
        loc = torch.einsum("sji,sj->si", Rb, hit - xpos[b, body])
        key = b * nb + body
        if need[4]:
            out[4] = torch.zeros(B * nb, 3, device=u.device, dtype=u.dtype).index_add_(0, key, Ws).reshape(B, nb, 3)
        if need[5]:
            gR = torch.zeros(B * nb, 9, device=u.device, dtype=u.dtype).index_add_(
                0, key, (Ws[:, :, None] * loc[:, None, :]).flatten(1)).reshape(B, nb, 3, 3)
            with torch.enable_grad():
                q = xquat.detach().requires_grad_(True)
                out[5], = torch.autograd.grad(qmat(torch, q), q, gR)
    return out



from dataclasses import dataclass


@dataclass(frozen=True)
class ImageSpec:
    """An image sensor's layout for the smoothed gradient and the fused backward: kind "depth" | "height" | "rgb",
    H x W pixels; projection ptype 0 pinhole (col = cx - p0 Y / X, row = cy - p1 Z / X of the hit in the sensor frame)
    or 1 grid (col = X p0, row = Y p0); the value V = (x - c) . (R a_cam + a_world)."""
    kind: str
    H: int
    W: int
    ptype: int
    p0: float
    p1: float
    a_cam: tuple = (0.0, 0.0, 0.0)
    a_world: tuple = (0.0, 0.0, 0.0)


def image_spec(sensor, kind):
    """ImageSpec of a cameras.DepthCamera (kind "depth" or "rgb") or cameras.HeightScan (kind "height")."""
    import numpy as np
    if kind == "height":
        nx = len(np.arange(-sensor.size[0] / 2, sensor.size[0] / 2 + sensor.resolution * 0.5, sensor.resolution))
        ny = len(np.arange(-sensor.size[1] / 2, sensor.size[1] / 2 + sensor.resolution * 0.5, sensor.resolution))
        return ImageSpec("height", ny, nx, 1, 1.0 / sensor.resolution, 0.0, a_world=(0.0, 0.0, -1.0))
    fx = sensor.width / 2 / np.tan(np.radians(sensor.hfov) / 2)
    fy = sensor.height / 2 / np.tan(np.radians(sensor.vfov) / 2)
    return ImageSpec(kind, sensor.height, sensor.width, 0, float(fx), float(fy),
                     a_cam=(1.0, 0.0, 0.0) if kind == "depth" else (0.0, 0.0, 0.0))


_MATS = {}


def smoothing_mats(n, sigma):
    """(G, D) (n, n) float32 NumPy: the 1-D Gaussian of sigma (cells; truncated at 3 sigma, edge-replicated border:
    the padded weights fold onto the border cell; sigma 0: identity) and the central difference (one-sided at the
    border)."""
    import numpy as np
    key = (n, float(sigma))
    if key not in _MATS:
        G = np.zeros((n, n))
        if sigma > 0:
            r = int(np.ceil(3 * sigma))
            k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
            k /= k.sum()
            for i in range(n):
                for m, km in zip(range(-r, r + 1), k):
                    G[i, min(max(i + m, 0), n - 1)] += km
        else:
            G = np.eye(n)
        D = np.zeros((n, n))
        if n > 1:
            for i in range(n):
                lo, hi = max(i - 1, 0), min(i + 1, n - 1)
                D[i, hi] += 1.0 / (hi - lo)
                D[i, lo] -= 1.0 / (hi - lo)
        _MATS[key] = (G.astype(np.float32), D.astype(np.float32))
    return _MATS[key]


class Smoother:
    """The separable smoothing of (B, H, W[, C]) images: S = Gv I Gu^T, its image gradient (d/dcol, d/drow) and the
    transposes (xp: jax.numpy or torch; the matrices on like's device)."""

    def __init__(self, xp, H, W, sigma, like=None):
        Gv, Dv = smoothing_mats(H, sigma)
        Gu, Du = smoothing_mats(W, sigma)
        if xp.__name__ == "torch":
            t = lambda a: xp.as_tensor(a, device=like.device)
        else:
            t = xp.asarray
        self.xp, self.Gv, self.Gu, self.Dv, self.Du = xp, t(Gv), t(Gu), t(Dv), t(Du)

    def g(self, x):
        return _ein(self.xp, "ij,bjk...,lk->bil...", self.Gv, x, self.Gu)

    def gt(self, x):
        return _ein(self.xp, "ji,bjk...,kl->bil...", self.Gv, x, self.Gu)

    def grad(self, S):
        return _ein(self.xp, "bik...,lk->bil...", S, self.Du), _ein(self.xp, "ij,bjk...->bik...", self.Dv, S)


def _sl(axis, a, b, nd):
    return tuple(slice(a, b) if i == axis else slice(None) for i in range(nd))


def _pad_axis(xp, x, axis, before):
    shp = list(x.shape)
    shp[axis] = 1
    z = xp.zeros(tuple(shp), dtype=x.dtype, **({"device": x.device} if xp.__name__ == "torch" else {}))
    parts = [z, x] if before else [x, z]
    return xp.cat(parts, axis) if xp.__name__ == "torch" else xp.concatenate(parts, axis)


def _interfaces(xp, img, m, t, axis):
    nd = m.ndim
    A, B = _sl(axis, None, -1, nd), _sl(axis, 1, None, nd)
    ma, mb = m[A] > 0, m[B] > 0
    pick_a = ma & (~mb | (t[A] <= t[B]))
    J = img[_sl(axis, 1, None, img.ndim)] - img[_sl(axis, None, -1, img.ndim)]
    return J, pick_a


def _half_weights(xp, n, like, axis, nd):
    import numpy as np
    w = np.full(n, 0.5, np.float32)
    if n > 1:
        w[0] = w[-1] = 1.0
    shp = [1] * nd
    shp[axis] = n
    w = w.reshape(shp)
    return xp.as_tensor(w, device=like.device) if xp.__name__ == "torch" else xp.asarray(w)


def _flux(xp, img, m, t, vel, axis):
    J, pick_a = _interfaces(xp, img, m, t, axis)
    nd = m.ndim
    v = xp.where(pick_a, vel[_sl(axis, None, -1, nd)], vel[_sl(axis, 1, None, nd)])
    if img.ndim == 4:
        v = v[..., None]
    f = J * v
    w = _half_weights(xp, img.shape[axis], img, axis, img.ndim)
    return (_pad_axis(xp, f, axis, False) + _pad_axis(xp, f, axis, True)) * w


def _flux_t(xp, img, m, t, c, axis):
    J, pick_a = _interfaces(xp, img, m, t, axis)
    w = _half_weights(xp, img.shape[axis], img, axis, img.ndim)
    cw = c * w
    g = (cw[_sl(axis, None, -1, img.ndim)] + cw[_sl(axis, 1, None, img.ndim)]) * J
    if img.ndim == 4:
        g = g.sum(-1)
    z = xp.zeros_like(g)
    return _pad_axis(xp, xp.where(pick_a, g, z), axis, False) + _pad_axis(xp, xp.where(pick_a, z, g), axis, True)


def image_tangent(sm, img, m, t, dV, dcol, drow):
    """dS (B, H, W[, C]) of the smoothed image for per-pixel tangents dV (B, H, W[, C]) of the value and dcol, drow
    (B, H, W) of the image coordinates; m (B, H, W) the moving-value mask (float), t (B, H, W) the ray distances
    (sm: a Smoother; sigma 0: the unsmoothed image's)."""
    xp = sm.xp
    mm = m[..., None] if img.ndim == 4 else m
    return sm.g(mm * dV - _flux(xp, img, m, t, dcol, 2) - _flux(xp, img, m, t, drow, 1))


def image_transpose(sm, img, m, t, c):
    """The transpose of image_tangent for the cotangent c of S: (per-pixel cotangent of dV (shape of img), of dcol,
    of drow (B, H, W))."""
    xp = sm.xp
    ct = sm.gt(c)
    mm = m[..., None] if img.ndim == 4 else m
    return mm * ct, -_flux_t(xp, img, m, t, ct, 2), -_flux_t(xp, img, m, t, ct, 1)


def functionals(xp, pos, rot, off, dirs, xpos, xquat, objects, res, spec, light=None):
    """Per-ray functions of the inputs whose derivatives at the primal point are the per-ray tangents the fused
    backward transposes (their values have no meaning): dict(t: the hit distance's first order (tangent_dist), V: the
    hit's value, col / row: its image coordinates, ndl: (-light) . n of its (turning) normal)."""
    out = dict(t=tangent_dist(xp, pos, rot, off, dirs, xpos, xquat, objects, res))
    x = res["hit"]
    n = res["nrm"]
    if "oh" in res:
        P, Ro = objects[..., 0:3], objects[..., 3:12].reshape(objects.shape[:2] + (3, 3))
        sel = res["oh"].sum(-1)[..., None] > 0
        xo = _ein(xp, "bnk,bki->bni", res["oh"], P) + _ein(xp, "bnk,bkij,bnj->bni", res["oh"], Ro, res["oloc"])
        no = _ein(xp, "bnk,bkij,bnj->bni", res["oh"], Ro, res["onloc"])
        x, n = xp.where(sel, xo, x), xp.where(sel, no, n)
    Rb = qmat(xp, _gather(xp, xquat, res["bidx"]))
    xr = _gather(xp, xpos, res["bidx"]) + _ein(xp, "bnij,bnj->bni", Rb, res["rloc"])
    nr = _ein(xp, "bnij,bnj->bni", Rb, res["rnloc"])
    rob = res["rob"][..., None]
    x, n = xp.where(rob, xr, x), xp.where(rob, nr, n)
    xc = x - pos[:, None]
    ac = _ein(xp, "bij,j->bi", rot, xp.asarray(spec.a_cam, dtype=rot.dtype) if xp.__name__ != "torch"
                   else xp.tensor(spec.a_cam, dtype=rot.dtype, device=rot.device))
    aw = (xp.asarray(spec.a_world, dtype=rot.dtype) if xp.__name__ != "torch"
          else xp.tensor(spec.a_world, dtype=rot.dtype, device=rot.device))
    out["V"] = (xc * ac[:, None]).sum(-1) + (xc * aw).sum(-1)
    xcam = _ein(xp, "bji,bnj->bni", rot, xc)
    if spec.ptype == 0:
        X = xp.where(res["dist"] >= 0, xcam[..., 0], xp.ones_like(xcam[..., 0]))
        out["col"] = -spec.p0 * xcam[..., 1] / X
        out["row"] = -spec.p1 * xcam[..., 2] / X
    else:
        out["col"], out["row"] = spec.p0 * xcam[..., 0], spec.p0 * xcam[..., 1]
    if light is not None:
        lt = xp.asarray(light, dtype=rot.dtype) if xp.__name__ != "torch" else xp.tensor(light, dtype=rot.dtype,
                                                                                        device=rot.device)
        out["ndl"] = -(n * lt).sum(-1)
    return out


def depth_value(xp, camera, housing, dirs):
    """value_fn for cast_image of a depth camera (cameras.DepthCamera; the ops of G1Sensors' depth): image (B, H, W)
    z-depth (0 invalid), aux the seg ids."""
    H, W = camera.height, camera.width

    def fn(d, gid, nrm, pos, rot):
        d = xp.where(d >= 0, d + housing, -xp.ones_like(d))
        z = d * dirs[:, 0]
        ok = (d >= 0) & (z >= camera.min_depth) & (z <= camera.max_depth)
        return (xp.where(ok, z, xp.zeros_like(z)).reshape(-1, H, W),
                xp.where(d >= 0, gid, -xp.ones_like(gid)).reshape(-1, H, W), ok.reshape(-1, H, W), None, nrm)
    return fn


def height_value(xp, scan):
    """value_fn for cast_image of a height scan (cameras.HeightScan; the ops of G1Sensors' height scan): image (B, ny,
    nx), aux the hit mask."""
    spec = image_spec(scan, "height")

    def fn(d, gid, nrm, pos, rot):
        hit_z = pos[:, None, 2] + scan.offset_z - xp.where(d >= 0, d, xp.full_like(d, scan.max_distance))
        img = (pos[:, None, 2] - hit_z).reshape(pos.shape[0], spec.H, spec.W)
        return img, (d >= 0).reshape(img.shape), (d >= 0).reshape(img.shape), None, nrm
    return fn

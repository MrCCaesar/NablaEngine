"""Per-world object geometry: the poses, sizes, velocities and face tables of a scene's objects and terrain, read from
the engine's state, in the world frame (docs/sensors.md, "Object geometry")."""
import math

import numpy as np

FACE_WORDS = 16
CLASS_TOP, CLASS_SIDE, CLASS_BOTTOM = 0, 1, 2
CLASS_NAMES = ("top", "side", "bottom")
COS_CLASS = math.cos(math.radians(45.0))
SRC_TERRAIN = -2
KIND_BODY, KIND_STATIC, KIND_SLOT = 0, 1, 2
ROW_WORDS = 7
_UPRIGHT_TOL = 1e-6


def _hi():
    import jax
    return jax.lax.Precision.HIGHEST


def _einsum(*args):
    import jax.numpy as jnp
    return jnp.einsum(*args, precision=_hi())


def geom_world_pose(xpos, xquat, body, pos, rot, geom_offset=None):
    """World pose of geoms fixed to bodies: centres (B, K, 3) and rotations (B, K, 3, 3) from the body poses xpos (B,
    nbody, 3), xquat (B, nbody, 4) (jax_ffi.kinematics'), the geoms' bodies body (K,), local positions pos (K, 3) and
    rotations rot (K, 3, 3) (the model's geom_pos / geom_quat); geom_offset: None, or a per-world shift of the centre in
    the body frame, (B, 3) or (B, K, 3), added to pos."""
    import jax.numpy as jnp
    from nablaengine.sensors.g1 import G1Sensors
    B, K = xpos.shape[0], len(body)
    Rb = G1Sensors._qmat(xquat[:, body])
    gp = jnp.broadcast_to(jnp.asarray(pos), (B, K, 3))
    if geom_offset is not None:
        gp = gp + jnp.broadcast_to(jnp.asarray(geom_offset, jnp.float32).reshape(B, -1, 3), (B, K, 3))
    p = xpos[:, body] + _einsum("bkij,bkj->bki", Rb, gp)
    R = _einsum("bkij,kjl->bkil", Rb, jnp.asarray(rot))
    return p, R


def _cross(a, b):
    import jax.numpy as jnp
    return jnp.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1], a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                      a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


def up_vector(gravity):
    """Unit up vectors (B, 3) = -g / |g| of gravity vectors (B, 3) (+z where g = 0)."""
    import jax.numpy as jnp
    n2 = jnp.sum(gravity * gravity, -1, keepdims=True)
    safe = jnp.where(n2 > 0, n2, 1.0)
    return jnp.where(n2 > 0, -gravity / jnp.sqrt(safe), jnp.asarray([0.0, 0.0, 1.0], gravity.dtype))


def surface_class(normal, up):
    """Surface class (B, ...) int32 of unit normals (B, ..., 3) for up vectors (B, 3): CLASS_TOP / CLASS_SIDE /
    CLASS_BOTTOM (module docstring)."""
    import jax.numpy as jnp
    c = jnp.sum(normal * up.reshape(up.shape[:1] + (1,) * (normal.ndim - 2) + (3,)), -1)
    return jnp.where(c > COS_CLASS, CLASS_TOP, jnp.where(c < -COS_CLASS, CLASS_BOTTOM, CLASS_SIDE)).astype(jnp.int32)


def box_face_records(centre, rot, half, valid=None):
    """The six faces (..., 6, FACE_WORDS) of boxes (centre (..., 3), rotation (..., 3, 3), half sizes (..., 3)), in
    the order -x, +x, -y, +y, -z, +z of the box frame (the scene-tracking work's box_faces): centre c + s R e_a h_a,
    normal s R e_a, u, v the box's other two axes in order (R e_b, R e_c), half widths h_b, h_c; class 0 (set by the
    caller), on 1 (valid (...,) bool: 0 and the whole record zero where False). JAX, differentiable."""
    import jax.numpy as jnp
    recs = []
    for ax in range(3):
        o1, o2 = [k for k in range(3) if k != ax]
        for sg in (-1.0, 1.0):
            n = sg * rot[..., :, ax]
            c = centre + n * half[..., ax:ax + 1]
            one = jnp.ones_like(half[..., :1])
            recs.append(jnp.concatenate([c, n, rot[..., :, o1], rot[..., :, o2], half[..., o1:o1 + 1],
                                         half[..., o2:o2 + 1], 0 * one, one], -1))
    out = jnp.stack(recs, -2)
    if valid is not None:
        out = jnp.where(valid[..., None, None], out, 0.0)
    return out


def classify_faces(faces, up):
    """faces (B, F, FACE_WORDS) with the class column set from the normals and up (B, 3); padded records (on = 0) stay
    zero."""
    import jax.numpy as jnp
    cls = surface_class(faces[..., 3:6], up).astype(faces.dtype)
    cls = jnp.where(faces[..., 15] > 0, cls, 0.0)
    return jnp.concatenate([faces[..., :14], cls[..., None], faces[..., 15:]], -1)


def box_rows(boxes):
    """(rows (n, 7) float64, upright (n,) bool) of boxes given as rows (n, 7) (centre, half sizes, yaw; hz <= 0: absent;
    the scene-tracking work's scene rows) or as words (n, 15) (pos, rotation row major, half sizes:
    terrain.Terrain.boxes, object slots; half sizes 0: absent). A word box is upright when its rotation keeps the z
    axis vertical (|R_zz| >= 1 - 1e-6); its yaw is atan2(R_yx, R_xx). Rows are upright."""
    b = np.asarray(boxes, np.float64)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    if b.shape[-1] == ROW_WORDS:
        return b.reshape(-1, ROW_WORDS).copy(), np.ones(len(b.reshape(-1, ROW_WORDS)), bool)
    if b.shape[-1] != 15:
        raise ValueError(f"object_geometry: boxes as rows (n, 7) or words (n, 15), got {b.shape}")
    b = b.reshape(-1, 15)
    R = b[:, 3:12].reshape(-1, 3, 3)
    rows = np.zeros((len(b), ROW_WORDS))
    rows[:, :3], rows[:, 3:6] = b[:, :3], b[:, 12:15]
    rows[:, 6] = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    present = np.all(b[:, 12:15] > 0, 1)
    rows[~present] = 0.0
    upright = np.abs(R[:, 2, 2]) >= 1 - _UPRIGHT_TOL
    return rows, upright | ~present


def _rects_touch(lo, hi, margin):
    return np.all((lo[:, None, :] <= hi[None, :, :] + margin) & (lo[None, :, :] <= hi[:, None, :] + margin), -1)


def _make_disjoint(b, tol, max_pieces, ground=0.0):
    k = len(b)
    on = b[:, 5] > 0
    if on.sum() <= 1:
        return b.astype(np.float32), np.arange(k), "unchanged"
    on_idx = np.nonzero(on)[0]
    yaw = b[on, 6]
    if np.ptp(np.mod(yaw - yaw[0] + np.pi, 2 * np.pi)) > 1e-3:
        return b.astype(np.float32), np.arange(k), "skipped"
    c, s_ = np.cos(yaw[0]), np.sin(yaw[0])
    rects = []
    for r in b[on]:
        x, y = c * r[0] + s_ * r[1], -s_ * r[0] + c * r[1]
        rects.append([x - r[3], x + r[3], y - r[4], y + r[4], r[2] + r[5]])
    ra = np.asarray(rects)
    near = _rects_touch(ra[:, [0, 2]], ra[:, [1, 3]], 2 * tol)
    order = sorted(range(len(rects)), key=lambda i: -rects[i][4])
    placed, owner, src, changed = [], [], [], False
    for i in order:
        pieces = [rects[i][:4]]
        for h, hi in zip(placed, owner):
            if not near[i, hi]:
                continue
            nxt = []
            for (x0, x1, y0, y1) in pieces:
                ox = min(x1, h[1]) - max(x0, h[0])
                oy = min(y1, h[3]) - max(y0, h[2])
                if ox <= tol or oy <= tol:
                    nxt.append((x0, x1, y0, y1))
                    continue
                changed = True
                if h[2] <= y0 + tol and h[3] >= y1 - tol:
                    nxt += [p for p in ((x0, h[0], y0, y1), (h[1], x1, y0, y1)) if p[1] - p[0] > tol]
                elif h[0] <= x0 + tol and h[1] >= x1 - tol:
                    nxt += [p for p in ((x0, x1, y0, h[2]), (x0, x1, h[3], y1)) if p[3] - p[2] > tol]
                else:
                    return b.astype(np.float32), np.arange(k), "skipped"
            pieces = nxt
        placed += [list(p) + [rects[i][4]] for p in pieces]
        owner += [i] * len(pieces)
        src += [int(on_idx[i])] * len(pieces)
    if not changed:
        return b.astype(np.float32), np.arange(k), "unchanged"
    if max_pieces is not None and len(placed) > max(k, int(max_pieces)):
        return b.astype(np.float32), np.arange(k), "skipped"
    out = np.zeros((len(placed), ROW_WORDS))
    for j, (x0, x1, y0, y1, t) in enumerate(placed):
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        out[j] = [c * xm - s_ * ym, s_ * xm + c * ym, (t + ground) / 2, (x1 - x0) / 2, (y1 - y0) / 2, (t - ground) / 2,
                  yaw[0]]
    return out.astype(np.float32), np.asarray(src, np.int64), "converted"


def _rect_faces(x0, x1, y0, y1, zlo, top, yaw, side_cov=None, tol=1e-4):
    c, s = math.cos(yaw), math.sin(yaw)
    ex, ey, ez = np.array([c, s, 0.0]), np.array([-s, c, 0.0]), np.array([0.0, 0.0, 1.0])
    w = lambda x, y, z: x * ex + y * ey + z * ez
    out = []
    if top - zlo > tol:
        out.append(np.concatenate([w((x0 + x1) / 2, (y0 + y1) / 2, top), ez, ex, ey,
                                   [(x1 - x0) / 2, (y1 - y0) / 2, 0.0, 1.0]]))
    sides = (("x", -1, x0, y0, y1), ("x", 1, x1, y0, y1), ("y", -1, y0, x0, x1), ("y", 1, y1, x0, x1))
    for si, (ax, sg, pos, a0, a1) in enumerate(sides):
        strips = side_cov[si] if side_cov is not None else []
        cuts = sorted({a0, a1} | {min(max(v, a0), a1) for st in strips for v in st[:2]})
        segs = []
        for p, q in zip(cuts[:-1], cuts[1:]):
            if q - p <= tol:
                continue
            m = 0.5 * (p + q)
            h = max([st[2] for st in strips if st[0] - tol <= m <= st[1] + tol] + [zlo])
            if segs and abs(segs[-1][2] - h) < tol and abs(segs[-1][1] - p) < tol:
                segs[-1][1] = q
            else:
                segs.append([p, q, h])
        for p, q, h in segs:
            if top - h <= tol:
                continue
            am, zm = 0.5 * (p + q), 0.5 * (h + top)
            if ax == "x":
                ctr, n, u = w(pos, am, zm), sg * ex, ey
            else:
                ctr, n, u = w(am, pos, zm), sg * ey, ex
            out.append(np.concatenate([ctr, n, u, ez, [(q - p) / 2, (top - h) / 2, 1.0, 1.0]]))
    return out


def _inside(p, b, tol=1e-4):
    c, s = math.cos(b[6]), math.sin(b[6])
    d = p - b[:3]
    lx, ly = c * d[0] + s * d[1], -s * d[0] + c * d[1]
    return abs(lx) < b[3] - tol and abs(ly) < b[4] - tol and abs(d[2]) < b[5] - tol


def _row_aabb(r):
    c, s = abs(math.cos(r[6])), abs(math.sin(r[6]))
    ex, ey = c * r[3] + s * r[4], s * r[3] + c * r[4]
    return (r[0] - ex, r[1] - ey), (r[0] + ex, r[1] + ey)


def _upright_faces(rows, tol, max_pieces, ground):
    b = rows
    if not (b[:, 5] > 0).any():
        return np.zeros((0, FACE_WORDS)), np.zeros(0, np.int64), "empty"
    d, dsrc, st = _make_disjoint(b, tol, max_pieces, ground)
    d = np.asarray(d, np.float64)
    on = d[:, 5] > 0
    if st != "skipped" and len(np.unique(np.round(np.mod(d[on, 6], 2 * np.pi), 6))) <= 1:
        yaw = float(d[on][0, 6])
        c, s = math.cos(yaw), math.sin(yaw)
        cols = []
        for r in d[on]:
            x, y = c * r[0] + s * r[1], -s * r[0] + c * r[1]
            cols.append((x - r[3], x + r[3], y - r[4], y + r[4], max(r[2] - r[5], ground), r[2] + r[5]))
        ca = np.asarray(cols).reshape(-1, 6)
        near = _rects_touch(ca[:, [0, 2]], ca[:, [1, 3]], 2 * tol)
        csrc = dsrc[on]
        faces, fsrc = [], []
        for j, (x0, x1, y0, y1, zlo, top) in enumerate(cols):
            cov = [[], [], [], []]
            for i in np.nonzero(near[j])[0]:
                if i == j:
                    continue
                nx0, nx1, ny0, ny1, _, ntop = cols[i]
                oy, ox = (max(y0, ny0), min(y1, ny1)), (max(x0, nx0), min(x1, nx1))
                if oy[1] - oy[0] > tol and abs(nx1 - x0) < tol:
                    cov[0].append((oy[0], oy[1], ntop))
                if oy[1] - oy[0] > tol and abs(nx0 - x1) < tol:
                    cov[1].append((oy[0], oy[1], ntop))
                if ox[1] - ox[0] > tol and abs(ny1 - y0) < tol:
                    cov[2].append((ox[0], ox[1], ntop))
                if ox[1] - ox[0] > tol and abs(ny0 - y1) < tol:
                    cov[3].append((ox[0], ox[1], ntop))
            f = _rect_faces(x0, x1, y0, y1, zlo, top, yaw, cov, tol)
            faces += f
            fsrc += [int(csrc[j])] * len(f)
        return np.asarray(faces, np.float64).reshape(-1, FACE_WORDS), np.asarray(fsrc, np.int64), "disjoint"
    sel = np.nonzero(b[:, 5] > 0)[0]
    boxes = b[sel]
    bb = [_row_aabb(r) for r in boxes]
    lo, hi = np.asarray([x[0] for x in bb]), np.asarray([x[1] for x in bb])
    near = _rects_touch(lo, hi, 2 * tol)
    faces, fsrc = [], []
    for k, r in enumerate(boxes):
        zlo, top = max(r[2] - r[5], ground), r[2] + r[5]
        c, s = math.cos(r[6]), math.sin(r[6])
        x, y = c * r[0] + s * r[1], -s * r[0] + c * r[1]
        others = [boxes[i] for i in np.nonzero(near[k])[0] if i != k]
        for f in _rect_faces(x - r[3], x + r[3], y - r[4], y + r[4], zlo, top, float(r[6])):
            if not any(_inside(f[0:3], o, tol) for o in others):
                faces.append(f)
                fsrc.append(int(sel[k]))
    return np.asarray(faces, np.float64).reshape(-1, FACE_WORDS), np.asarray(fsrc, np.int64), "fallback"


def _box_faces_np(centre, R, half):
    out = []
    for ax in range(3):
        o1, o2 = [k for k in range(3) if k != ax]
        for sg in (-1.0, 1.0):
            n = sg * R[:, ax]
            out.append(np.concatenate([centre + n * half[ax], n, R[:, o1], R[:, o2], [half[o1], half[o2], 0.0, 1.0]]))
    return np.asarray(out)


def exposed_faces(boxes, tol=1e-4, max_pieces=None, ground=0.0, with_source=False):
    """The exposed faces of one box scene (module docstring): boxes as rows (n, 7) or words (n, 15) (box_rows). Returns
    (faces (m, FACE_WORDS) float64, mode "empty" / "disjoint" / "fallback"), with with_source also src (m,) int, each
    face's box index. Classes as the reference's: 0 for the treads (normal +z), 1 for the sides (the device recomputes
    them from each world's gravity). Rows give the scene-tracking work's exposed_faces bitwise (with max_pieces=8, its
    slot limit: a scene cut into more pieces then falls back); None (the default) puts no limit on the pieces. Boxes
    that are not upright (words only) emit their six faces (classes 0 for normals within 45 deg of +z, else 1).
    ground: the height the columns stand on, below which no face is emitted (the scene-tracking scenes': 0, the
    default; a terrain with pits below z = 0, e.g. mjlab's inverted stairs, needs its lowest level, or None: the
    lowest box bottom of the scene)."""
    rows, upright = box_rows(boxes)
    cut = rows.copy()
    cut[~upright] = 0.0
    if ground is None:
        on = cut[:, 5] > 0
        ground = float(np.min(cut[on, 2] - cut[on, 5])) if on.any() else 0.0
    faces, src, mode = _upright_faces(cut, tol, max_pieces, float(ground))
    tilt = np.nonzero(~upright & np.all(np.asarray(boxes, np.float64).reshape(len(rows), -1)[:, -3:] > 0, 1))[0]
    if len(tilt):
        w = np.asarray(boxes, np.float64).reshape(len(rows), -1)
        extra = []
        for i in tilt:
            f = _box_faces_np(w[i, :3], w[i, 3:12].reshape(3, 3), w[i, 12:15])
            f[:, 14] = np.where(f[:, 5] > COS_CLASS, 0.0, 1.0)
            extra.append(f)
        faces = np.concatenate([faces] + extra, 0)
        src = np.concatenate([src, np.repeat(tilt, 6)])
        if mode == "empty":
            mode = "fallback"
    return (faces, mode, src) if with_source else (faces, mode)


def face_table(scenes, tol=1e-4, max_pieces=None, min_faces=1, ground=0.0):
    """Exposed faces of S scenes (S, n, 7 or 15) -> (faces (S, F, FACE_WORDS) float32 padded with zero records, src (S,
    F) int32 (each face's box index, -1 padding), report {faces_max, faces_mean, empty, disjoint, fallback}); F >=
    min_faces. Identical scenes are computed once."""
    sc = np.asarray(scenes)
    per, modes, memo = [], dict(empty=0, disjoint=0, fallback=0), {}
    for s in sc:
        key = s.tobytes()
        if key not in memo:
            memo[key] = exposed_faces(s, tol, max_pieces, ground, with_source=True)
        f, m, src = memo[key]
        per.append((f, src))
        modes[m] += 1
    F = max(int(min_faces), max([len(f) for f, _ in per] + [0]))
    out = np.zeros((len(per), F, FACE_WORDS), np.float32)
    srcs = np.full((len(per), F), -1, np.int32)
    for i, (f, s) in enumerate(per):
        out[i, :len(f)] = f
        srcs[i, :len(f)] = s
    return out, srcs, dict(faces_max=F, faces_mean=float(np.mean([len(f) for f, _ in per])) if per else 0.0, **modes)


def tile_scenes(boxes, origins, size):
    """Split static terrain boxes (n, 15) into tiles: the boxes whose centre lies in the tile [origin - size / 2, origin
    + size / 2) (xy; origins (T, 2 or 3), size (2,) or a scalar), e.g. mjlab's sub-terrains (TerrainGenerator's
    terrain_origins, the sub-terrain size). Returns (T, n_max, 15) padded with absent boxes (zeros) and the box
    indices (T, n_max) (-1 padding). Boxes outside every tile are left out."""
    b = np.asarray(boxes, np.float64).reshape(-1, 15)
    o = np.asarray(origins, np.float64).reshape(-1, np.shape(origins)[-1])[:, :2]
    sz = np.broadcast_to(np.asarray(size, np.float64), (2,))
    lo, hi = o - sz / 2, o + sz / 2
    inside = np.all((b[None, :, :2] >= lo[:, None]) & (b[None, :, :2] < hi[:, None]), -1)
    n = max(1, int(inside.sum(1).max()) if len(o) else 1)
    out = np.zeros((len(o), n, 15))
    idx = np.full((len(o), n), -1, np.int64)
    for t in range(len(o)):
        sel = np.nonzero(inside[t])[0]
        out[t, :len(sel)] = b[sel]
        idx[t, :len(sel)] = sel
    return out, idx


class ObjectGeometry:
    """Per-world geometry of a build's objects, faces, floor and gravity (module docstring). Construction (host): the
    model, the box geoms that are objects and the build (info, jax_ffi.init's: its com layout, terrain config and
    per-world parameters); a call (device, jit-able): the kinematics outputs and com of B worlds."""

    def __init__(self, model, geoms=None, bodies=None, *, info=None, params=None, com_layout=None, terrain=None,
                 tiles=None, slot_faces="box", floor="auto", faces=True, tol=1e-4, max_pieces=None, ground=0.0):
        """model: the physics model (MjModel, the engine's). geoms: box geom names or ids (one object each, in this
        order); else bodies: names whose single box geom is taken; neither: no model objects. A geom on a moving body
        is movable (KIND_BODY), one on the world body or welded to it static (KIND_STATIC).
        info: jax_ffi.init's (or torch_ffi's) info, for com_layout, the terrain build's slots and floor flag; or pass
        com_layout (info["com_layout"]) directly. params: the build's PhysicsParams (object_size: its geom takes each
        world's half sizes from com; gravity: each world's gravity from com); None reads jax_ffi's current build when
        its com layout has parameter words.
        terrain: a static terrain.Terrain: the exposed faces of its boxes (heightfields: none), one table for all
        worlds, or with tiles=(origins (T, 2|3), size) one table per tile (tile_scenes), chosen per world at the call.
        slot_faces: the terrain slots' faces, "box" (each slot's 6 faces, on the device), "exposed" (the call takes a
        table of each world's exposed slot faces, exposed_slot_faces) or None (no slot faces). floor: "auto" (the
        model's first plane geom of the world body, unless the terrain build drops the floor: TerrainConfig.floor
        False), None (no floor) or a plane geom name / id. faces: False skips every face table. tol, max_pieces,
        ground: exposed_faces' (the terrain's and exposed_slot_faces')."""
        import mujoco
        box = int(mujoco.mjtGeom.mjGEOM_BOX)
        ids = []
        if geoms is not None:
            for g in geoms:
                gid = int(g) if not isinstance(g, str) else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                if not 0 <= gid < model.ngeom or int(model.geom_type[gid]) != box:
                    raise ValueError(f"ObjectGeometry: {g!r} is not a box geom of the model")
                ids.append(gid)
        elif bodies is not None:
            for name in bodies:
                b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if b < 1:
                    raise ValueError(f"ObjectGeometry: no body {name!r}")
                gs = [g for g in range(model.ngeom)
                      if int(model.geom_bodyid[g]) == b and int(model.geom_type[g]) == box]
                if len(gs) != 1:
                    raise ValueError(f"ObjectGeometry: body {name!r} has {len(gs)} box geoms (name it: geoms=...)")
                ids.append(gs[0])
        from nablaengine.sensors import cameras
        self.geoms = np.asarray(ids, np.int32)
        self.body = np.asarray([model.geom_bodyid[g] for g in ids], np.int32)
        self.root = np.asarray([model.body_rootid[b] for b in self.body], np.int32)
        self.pos = np.asarray([model.geom_pos[g] for g in ids], np.float32).reshape(-1, 3)
        self.rot = np.asarray([cameras.quat_matrix(model.geom_quat[g]) for g in ids], np.float32).reshape(-1, 3, 3)
        self.size = np.asarray([model.geom_size[g] for g in ids], np.float32).reshape(-1, 3)
        self.movable_body = np.asarray([model.body_weldid[b] != 0 for b in self.body], bool)
        self.kb = len(ids)
        if info is None and com_layout is None:
            com_layout = {"com_offset": (0, 3)}
        lay = dict(com_layout if com_layout is not None else info["com_layout"])
        self.com_layout = {k: (int(v[0]), int(v[1])) for k, v in lay.items()}
        self.com_words = max(o + w for o, w in self.com_layout.values())
        tkey = (info or {}).get("terrain") or {}
        o, w = self.com_layout.get("objects", (0, 0))
        from nablaengine import terrain as T
        self.ks = w // T.OBJ_WORDS
        self.slot_off = o
        self.slot_vel = self.com_layout.get("object_velocity")
        self.K = self.kb + self.ks
        self.size_obj = None
        self.gravity_off = self.com_layout.get("params.gravity", (None,))[0]
        if "params.object_size" in self.com_layout:
            if params is None:
                from nablaengine import jax_ffi
                params = jax_ffi._STATE.get("params")
            if params is None:
                raise ValueError("ObjectGeometry: the com layout has object_size words: pass the build's params=")
            from nablaengine._optional import FOREST, load
            object_size = load("nablaengine.forest_engine", FOREST, "object_size")
            sg = object_size.size_geom(model, params, (info or {}).get("com_body"))
            hit = np.nonzero(self.geoms == sg)[0]
            self.size_obj = int(hit[0]) if len(hit) else None
            self.size_off = self.com_layout["params.object_size"][0]
        self.gravity0 = np.asarray(model.opt.gravity, np.float32)
        self.floor_geom = None
        plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
        if floor == "auto":
            if not tkey or tkey.get("floor", False):
                ps = [g for g in range(model.ngeom) if int(model.geom_type[g]) == plane
                      and int(model.body_weldid[model.geom_bodyid[g]]) == 0]
                self.floor_geom = ps[0] if ps else None
        elif floor is not None:
            gid = (int(floor) if not isinstance(floor, str)
                   else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor))
            if not 0 <= gid < model.ngeom or int(model.geom_type[gid]) != plane:
                raise ValueError(f"ObjectGeometry: floor {floor!r} is not a plane geom")
            self.floor_geom = gid
        if self.floor_geom is not None:
            d = mujoco.MjData(model)
            mujoco.mj_kinematics(model, d)
            n = d.geom_xmat[self.floor_geom].reshape(3, 3)[:, 2]
            self.floor_normal = np.asarray(n, np.float32)
            self.floor_height = np.float32(np.dot(d.geom_xpos[self.floor_geom], n))
        self.faces = bool(faces)
        if slot_faces not in ("box", "exposed", None):
            raise ValueError(f"ObjectGeometry: slot_faces {slot_faces!r}: 'box', 'exposed' or None")
        self.slot_faces = slot_faces if (self.faces and self.ks) else None
        self.tol, self.max_pieces, self.ground = tol, max_pieces, ground
        self.terrain_faces = None
        self.tiles = None
        self.terrain_report = None
        if self.faces and terrain is not None and len(terrain.boxes):
            if tiles is None:
                sc, idx = np.asarray(terrain.boxes, np.float64)[None], np.arange(len(terrain.boxes))[None]
            else:
                sc, idx = tile_scenes(terrain.boxes, *tiles)
                self.tiles = len(sc)
            f, src, rep = face_table(sc, tol, max_pieces, ground=ground)
            self.terrain_faces = f
            self.terrain_src = np.where(src >= 0, np.take_along_axis(idx, np.maximum(src, 0), 1), -1)
            self.terrain_report = rep

    def exposed_slot_faces(self, slots, min_faces=1):
        """The exposed faces of each world's (or scene's) slot boxes, slots (S, K_slots, 15) (com's object words;
        NumPy): (faces (S, F, 16) float32, src (S, F) int32 slot index, report) for the call's slot_faces=(faces,
        src) (gather the rows of each world's scene). Constant per scene: compute once per scene set."""
        return face_table(np.asarray(slots, np.float64).reshape(-1, self.ks, 15), self.tol, self.max_pieces, min_faces,
                          self.ground)

    def slot_words(self, com):
        """The terrain slots' words (B, K_slots, 15) of com rows."""
        return com[:, self.slot_off:self.slot_off + 15 * self.ks].reshape(com.shape[0], self.ks, 15)

    def from_state(self, qpos, qvel, com, **kw):
        """__call__ on jax_ffi.kinematics(qpos, qvel, com) (differentiable with respect to qpos and qvel)."""
        from nablaengine import jax_ffi
        return self(jax_ffi.kinematics(qpos, qvel, com), com, **kw)

    def __call__(self, kin, com, *, half_sizes=None, geom_offset=None, slot_faces=None, tile=None):
        """The geometry of B worlds (dict of arrays; module docstring) from kin = jax_ffi.kinematics' outputs (qpos_n,
        xpos, xquat, cvel, subtree_com) and the com rows (B, com_words).
        half_sizes: None (the model objects' geom_size, and the object_size geom's each world's from com), or (B, 3) /
        (B, K_model, 3) in place of them (as BoxSlots.objects); geom_offset: None, or a per-world shift (B, 3) / (B,
        K_model, 3) of the model objects' centres in their body frames. slot_faces: with
        ObjectGeometry(slot_faces="exposed"), (faces (B, F_s, 16), src (B, F_s)) each world's rows of
        exposed_slot_faces. tile: (B,) int, each world's tile with ObjectGeometry(tiles=...)."""
        import jax.numpy as jnp
        _, xpos, xquat, cvel, subtree_com = kin
        B = xpos.shape[0]
        f32 = jnp.float32
        out = {}
        if self.gravity_off is not None:
            g = com[:, self.gravity_off:self.gravity_off + 3].astype(f32)
        else:
            g = jnp.broadcast_to(jnp.asarray(self.gravity0), (B, 3))
        up = up_vector(g)
        out["gravity"], out["up"] = g, up
        if self.floor_geom is not None:
            out["floor_valid"] = jnp.ones((B,), bool)
            out["floor_height"] = jnp.full((B,), self.floor_height, f32)
            out["floor_normal"] = jnp.broadcast_to(jnp.asarray(self.floor_normal), (B, 3))
        else:
            out["floor_valid"] = jnp.zeros((B,), bool)
            out["floor_height"] = jnp.zeros((B,), f32)
            out["floor_normal"] = jnp.zeros((B, 3), f32)
        pos, rot, half, vel, ang, valid = [], [], [], [], [], []
        if self.kb:
            p, R = geom_world_pose(xpos, xquat, self.body, self.pos, self.rot, geom_offset)
            if half_sizes is None:
                hs = jnp.broadcast_to(jnp.asarray(self.size), (B, self.kb, 3))
                if self.size_obj is not None:
                    hs = hs.at[:, self.size_obj].set(com[:, self.size_off:self.size_off + 3].astype(f32))
            else:
                hs = jnp.broadcast_to(jnp.asarray(half_sizes, f32).reshape(B, -1, 3), (B, self.kb, 3))
            cv = cvel[:, self.body]
            w = cv[..., :3]
            v = cv[..., 3:] + _cross(w, p - subtree_com[:, self.root])
            pos.append(p), rot.append(R), half.append(hs), vel.append(v), ang.append(w)
            valid.append(jnp.ones((B, self.kb), bool))
        if self.ks:
            sw = self.slot_words(com).astype(f32)
            pos.append(sw[..., :3]), rot.append(sw[..., 3:12].reshape(B, self.ks, 3, 3)), half.append(sw[..., 12:15])
            if self.slot_vel is not None:
                mv = com[:, self.slot_vel[0]:self.slot_vel[0] + self.slot_vel[1]].astype(f32).reshape(B, self.ks, -1)
                vel.append(mv[..., 0:3]), ang.append(mv[..., 3:6])
            else:
                vel.append(jnp.zeros((B, self.ks, 3), f32)), ang.append(jnp.zeros((B, self.ks, 3), f32))
            valid.append(sw[..., 14] > 0)
        cat = lambda xs, shape: jnp.concatenate(xs, 1) if xs else jnp.zeros((B, 0) + shape, f32)
        out["obj_pos"], out["obj_rot"], out["obj_half"] = cat(pos, (3,)), cat(rot, (3, 3)), cat(half, (3,))
        out["obj_vel"], out["obj_angvel"] = cat(vel, (3,)), cat(ang, (3,))
        out["obj_valid"] = jnp.concatenate(valid, 1) if valid else jnp.zeros((B, 0), bool)
        kind = [KIND_BODY if m else KIND_STATIC for m in self.movable_body] + [KIND_SLOT] * self.ks
        out["obj_kind"] = jnp.asarray(np.asarray(kind, np.int32).reshape(-1))
        out["obj_movable"] = jnp.asarray(np.asarray(list(self.movable_body) + [False] * self.ks, bool).reshape(-1))
        if not self.faces:
            return out
        fs, fo, fm = [], [], []
        if self.kb:
            fs.append(box_face_records(pos[0], rot[0], half[0]).reshape(B, 6 * self.kb, FACE_WORDS))
            fo.append(jnp.broadcast_to(jnp.repeat(jnp.arange(self.kb, dtype=jnp.int32), 6), (B, 6 * self.kb)))
            fm.append(jnp.broadcast_to(jnp.asarray(np.repeat(self.movable_body, 6)), (B, 6 * self.kb)))
        if self.slot_faces == "box":
            sv = valid[-1]
            fs.append(box_face_records(pos[-1], rot[-1], half[-1], sv).reshape(B, 6 * self.ks, FACE_WORDS))
            ids = jnp.repeat(jnp.arange(self.kb, self.K, dtype=jnp.int32), 6)
            fo.append(jnp.where(jnp.repeat(sv, 6, 1), ids, -1))
            fm.append(jnp.zeros((B, 6 * self.ks), bool))
        elif self.slot_faces == "exposed":
            if slot_faces is None:
                raise ValueError("ObjectGeometry(slot_faces='exposed'): pass slot_faces=(faces, src) per world "
                                 "(exposed_slot_faces)")
            sf, ss = slot_faces
            sf = jnp.asarray(sf, f32).reshape(B, -1, FACE_WORDS)
            ss = jnp.asarray(ss, jnp.int32).reshape(B, -1)
            fs.append(sf)
            fo.append(jnp.where(ss >= 0, ss + self.kb, -1))
            fm.append(jnp.zeros(ss.shape, bool))
        tb = None
        if self.terrain_faces is not None:
            if self.tiles is None:
                tf = jnp.broadcast_to(jnp.asarray(self.terrain_faces[0]), (B,) + self.terrain_faces.shape[1:])
                tsrc = jnp.broadcast_to(jnp.asarray(self.terrain_src[0], jnp.int32), (B, self.terrain_faces.shape[1]))
            else:
                if tile is None:
                    raise ValueError("ObjectGeometry(tiles=...): pass tile=(B,) each world's tile index")
                t = jnp.asarray(tile, jnp.int32).reshape(B)
                tf = jnp.take(jnp.asarray(self.terrain_faces), t, axis=0)
                tsrc = jnp.take(jnp.asarray(self.terrain_src, jnp.int32), t, axis=0)
            fs.append(tf)
            fo.append(jnp.where(tsrc >= 0, SRC_TERRAIN, -1).astype(jnp.int32))
            fm.append(jnp.zeros(tsrc.shape, bool))
            nb = sum(x.shape[1] for x in fs[:-1])
            tb = jnp.concatenate([jnp.full((B, nb), -1, jnp.int32), tsrc.astype(jnp.int32)], 1)
        if fs:
            faces = classify_faces(jnp.concatenate(fs, 1), up)
            out["faces"] = faces
            out["face_valid"] = faces[..., 15] > 0
            out["face_object"] = jnp.concatenate(fo, 1).astype(jnp.int32)
            out["face_movable"] = jnp.concatenate(fm, 1)
            out["face_class"] = faces[..., 14].astype(jnp.int32)
        else:
            out["faces"] = jnp.zeros((B, 0, FACE_WORDS), f32)
            out["face_valid"] = jnp.zeros((B, 0), bool)
            out["face_object"] = jnp.zeros((B, 0), jnp.int32)
            out["face_movable"] = jnp.zeros((B, 0), bool)
            out["face_class"] = jnp.zeros((B, 0), jnp.int32)
        if tb is not None:
            out["terrain_box"] = tb
        return out

    def box_slots(self, g):
        """G1Sensors object slots (B, K, 15) (pos, rotation row major, half sizes) of a call's output g."""
        import jax.numpy as jnp
        B, K = g["obj_pos"].shape[:2]
        return jnp.concatenate([g["obj_pos"], g["obj_rot"].reshape(B, K, 9), g["obj_half"]], -1).astype(jnp.float32)

"""The robot's visual meshes for the rays: triangles in each mesh geom's frame, a BVH per mesh (binned SAH split,
leaves of at most LEAF triangles), and the geoms' bodies mapped by name onto the engine's model
(the mjlab G1 fixture has no meshes; mjlab's g1.xml has the visual meshes of the same bodies)."""
import hashlib
from pathlib import Path

import mujoco
import numpy as np

LEAF = 4


SAH_BINS = 16


def _area(lo, hi):
    e = np.maximum(hi - lo, 0.0)
    return 2.0 * (e[..., 0] * e[..., 1] + e[..., 1] * e[..., 2] + e[..., 2] * e[..., 0])


def bvh(tris, sah=True):
    """(node bounds (n, 6) lo / hi, node ints (n, 2), triangle order): ints (left, right) for an inner node, (-(first) - 1,
    count) for a leaf over tris[order[first:first + count]]; node 0 is the root. sah: a binned surface-area heuristic
    split (SAH_BINS bins on the longest centroid axis), else the median."""
    t3 = tris.reshape(-1, 3, 3)
    cen, lo_t, hi_t = t3.mean(1), t3.min(1), t3.max(1)
    order = np.arange(len(tris))
    bounds, ints = [], []
    stack = [(0, len(tris), -1, 0)]
    while stack:
        s, e, parent, side = stack.pop()
        idx = order[s:e]
        node = len(bounds)
        lo, hi = lo_t[idx].min(0), hi_t[idx].max(0)
        bounds.append(np.concatenate([lo, hi]))
        ints.append([-(s) - 1, e - s])
        if parent >= 0:
            ints[parent][side] = node
        if e - s <= LEAF:
            continue
        c = cen[idx]
        cmin, cmax = c.min(0), c.max(0)
        ax = int(np.argmax(cmax - cmin))
        mid = (e - s) // 2
        if sah and cmax[ax] > cmin[ax]:
            b = np.minimum(((c[:, ax] - cmin[ax]) / (cmax[ax] - cmin[ax]) * SAH_BINS).astype(int), SAH_BINS - 1)
            cnt = np.bincount(b, minlength=SAH_BINS)
            blo = np.full((SAH_BINS, 3), np.inf)
            bhi = np.full((SAH_BINS, 3), -np.inf)
            np.minimum.at(blo, b, lo_t[idx])
            np.maximum.at(bhi, b, hi_t[idx])
            llo, lhi = np.minimum.accumulate(blo, 0), np.maximum.accumulate(bhi, 0)
            rlo, rhi = np.minimum.accumulate(blo[::-1], 0)[::-1], np.maximum.accumulate(bhi[::-1], 0)[::-1]
            ln, rn = np.cumsum(cnt), np.cumsum(cnt[::-1])[::-1]
            cost = np.full(SAH_BINS - 1, np.inf)
            for k in range(SAH_BINS - 1):
                if ln[k] and rn[k + 1]:
                    cost[k] = _area(llo[k], lhi[k]) * ln[k] + _area(rlo[k + 1], rhi[k + 1]) * rn[k + 1]
            k = int(np.argmin(cost))
            if np.isfinite(cost[k]):
                left = b <= k
                order[s:e] = np.concatenate([idx[left], idx[~left]])
                mid = int(left.sum())
            else:
                order[s:e] = idx[np.argpartition(c[:, ax], mid)]
        else:
            order[s:e] = idx[np.argpartition(c[:, ax], mid)]
        ints[node] = [0, 0]
        stack.append((s + mid, e, node, 1))
        stack.append((s, s + mid, node, 0))
    return np.asarray(bounds, np.float32), np.asarray(ints, np.int32), order


def cluster(vert, face, cell):
    """Vertex-clustering simplification: vertices snapped to the mean of their cell (a cell x cell x cell grid), faces
    whose corners merge dropped (shape kept within about a cell)."""
    key = np.floor(vert / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    nv = inv.max() + 1
    cnt = np.bincount(inv, minlength=nv)
    v2 = np.stack([np.bincount(inv, weights=vert[:, k], minlength=nv) for k in range(3)], 1) / cnt[:, None]
    f2 = inv[face]
    keep = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    f2 = np.unique(np.sort(f2[keep], 1), axis=0, return_index=True)[1]
    return v2, inv[face][keep][np.sort(f2)]


def load(mjcf, model, group=2, cache=None, simplify=0.0):
    """dict(mgeom (G, 12), mroot (G,), nodeb, nodei, tri (T, 9), names [geom names], source_geom (G,): the geom index
    in mjcf, rgba (G, 4): the geom's colour, its material's where it has one) for the mesh geoms of geom group
    `group` in mjcf, posed on model's bodies of the same names (a body missing from model, e.g. fused into its parent
    by a locked joint: the nearest ancestor that model has, at the pose in mjcf's qpos0; asserted to exist). Cached in `cache` (a directory).
    simplify > 0: the meshes simplified by vertex clustering at that cell size (metres; cluster())."""
    src = Path(mjcf)
    key = hashlib.sha256(src.read_bytes() + str(group).encode() + b"sah16rgba" + str(float(simplify)).encode()
                         + ",".join(model.body(b).name for b in range(model.nbody)).encode()).hexdigest()[:16]
    path = Path(cache) / f"robot_mesh_{key}.npz" if cache else None
    if path is not None and path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    v = mujoco.MjModel.from_xml_path(str(src))
    geoms = [g for g in range(v.ngeom) if v.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and v.geom_group[g] == group]
    mgeom, mroot, nb_all, ni_all, tri_all, names, src_geom, rgba = [], [], [], [], [], [], [], []
    nnode = ntri = 0
    vd = None
    for g in geoms:
        name = v.body(v.geom_bodyid[g]).name
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        gpos, gquat = v.geom_pos[g], v.geom_quat[g]
        if b < 0:
            a = int(v.body_parentid[v.geom_bodyid[g]])
            while a > 0 and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, v.body(a).name) < 0:
                a = int(v.body_parentid[a])
            assert a > 0, f"body {name!r} of mesh geom {g}: neither it nor an ancestor is in the model"
            if vd is None:
                vd = mujoco.MjData(v)
                mujoco.mj_kinematics(v, vd)
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, v.body(a).name)
            Ra = vd.xmat[a].reshape(3, 3)
            gpos = Ra.T @ (vd.geom_xpos[g] - vd.xpos[a])
            gquat = np.zeros(4)
            mujoco.mju_mat2Quat(gquat, (Ra.T @ vd.geom_xmat[g].reshape(3, 3)).ravel())
        mid = v.geom_dataid[g]
        va, vn = v.mesh_vertadr[mid], v.mesh_vertnum[mid]
        fa, fn = v.mesh_faceadr[mid], v.mesh_facenum[mid]
        vert, face = v.mesh_vert[va:va + vn], v.mesh_face[fa:fa + fn]
        if simplify > 0:
            vert, face = cluster(vert.astype(np.float64), face, simplify)
        tri = vert[face].reshape(-1, 9).astype(np.float32)
        nb, ni, order = bvh(tri)
        tri = tri[order]
        leaf = ni[:, 0] < 0
        ni = ni.copy()
        ni[leaf, 0] -= ntri
        ni[~leaf] += nnode
        c = vert.mean(0)
        r = float(np.linalg.norm(vert - c, axis=1).max())
        mgeom.append(np.concatenate([[b], gpos, gquat, c, [r]]))
        mroot.append(nnode)
        nb_all.append(nb), ni_all.append(ni), tri_all.append(tri)
        names.append(v.geom(g).name or f"{name}_mesh{g}")
        src_geom.append(g)
        rgba.append(v.mat_rgba[v.geom_matid[g]] if v.geom_matid[g] >= 0 else v.geom_rgba[g])
        nnode += len(nb)
        ntri += len(tri)
    out = dict(mgeom=np.asarray(mgeom, np.float32), mroot=np.asarray(mroot, np.int32),
               nodeb=np.concatenate(nb_all).astype(np.float32), nodei=np.concatenate(ni_all).astype(np.int32),
               tri=np.concatenate(tri_all).astype(np.float32), names=np.asarray(names),
               source_geom=np.asarray(src_geom, np.int32), rgba=np.asarray(rgba, np.float32))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **out)
    return out


def _sphere_cap(r, segments, rings, z0, sign):
    out = []
    for i in range(rings + 1):
        phi = 0.5 * np.pi * i / rings
        rr, zz = r * np.cos(phi), z0 + sign * r * np.sin(phi)
        th = 2 * np.pi * np.arange(segments) / segments
        out.append(np.stack([rr * np.cos(th), rr * np.sin(th), np.full(segments, zz)], -1))
    return out


def _band_faces(nring, segments):
    f = []
    for i in range(nring - 1):
        for j in range(segments):
            a, b = i * segments + j, i * segments + (j + 1) % segments
            c, d = a + segments, b + segments
            f += [[a, b, d], [a, d, c]]
    return f


def primitive_mesh(gtype, size, segments=12):
    """(vertices (V, 3), faces (F, 3)) of a geom primitive in its own frame: sphere, capsule, cylinder, ellipsoid, box
    (vertices on the surface: a polygonal approximation inside the curved surfaces)."""
    T = mujoco.mjtGeom
    rings = max(segments // 4, 2)
    if gtype == T.mjGEOM_BOX:
        sx, sy, sz = size[:3]
        v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
        f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
             [0, 6, 4], [1, 5, 7], [1, 7, 3]]
        return v, np.asarray(f)
    if gtype in (T.mjGEOM_SPHERE, T.mjGEOM_ELLIPSOID, T.mjGEOM_CAPSULE):
        r = size[0]
        h = size[1] if gtype == T.mjGEOM_CAPSULE else 0.0
        top = _sphere_cap(r, segments, rings, h, 1.0)
        bot = _sphere_cap(r, segments, rings, -h, -1.0)
        ringsv = bot[::-1] + top
        v = np.concatenate(ringsv)
        f = np.asarray(_band_faces(len(ringsv), segments))
        if gtype == T.mjGEOM_ELLIPSOID:
            v = v / r * np.asarray(size[:3])
        return v, f
    if gtype == T.mjGEOM_CYLINDER:
        r, h = size[0], size[1]
        th = 2 * np.pi * np.arange(segments) / segments
        ring = lambda z: np.stack([r * np.cos(th), r * np.sin(th), np.full(segments, z)], -1)
        v = np.concatenate([ring(-h), ring(h), [[0, 0, -h], [0, 0, h]]])
        f = _band_faces(2, segments)
        n = 2 * segments
        f += [[n, (j + 1) % segments, j] for j in range(segments)]
        f += [[n + 1, segments + j, segments + (j + 1) % segments] for j in range(segments)]
        return v, np.asarray(f)
    raise ValueError(f"primitive_mesh: geom type {int(gtype)} not supported")


def from_collision(model, body=None, segments=12, rgba=(0.55, 0.55, 0.57, 1.0)):
    """The robot as its physics model's collision geoms (a cheaper stand-in for the visual meshes; robot_mesh.load's
    dict): every collision geom of the bodies in body's kinematic tree (body: a name, default the
    first moving body's tree), tessellated (primitive_mesh: segments around each curved geom; mesh geoms as their
    mesh), merged into one mesh per body in the body frame, all of colour rgba. Collision geoms: contype or
    conaffinity nonzero, or in an explicit contact pair."""
    if body is None:
        b0 = 1
    else:
        b0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if b0 < 1:
            raise ValueError(f"from_collision: no body {body!r}")
    root = int(model.body_rootid[b0])
    per = {}
    paired = set(np.asarray(model.pair_geom1).tolist()) | set(np.asarray(model.pair_geom2).tolist())
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        coll = bool(model.geom_contype[g] or model.geom_conaffinity[g]) or g in paired
        if b == 0 or int(model.body_rootid[b]) != root or not coll:
            continue
        t = int(model.geom_type[g])
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            v, f = np.asarray(model.mesh_vert[va:va + vn], np.float64), np.asarray(model.mesh_face[fa:fa + fn])
        elif t == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        else:
            v, f = primitive_mesh(t, np.asarray(model.geom_size[g], np.float64), segments)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        vb = v @ R.reshape(3, 3).T + model.geom_pos[g]
        per.setdefault(b, []).append(vb[f].reshape(-1, 9))
    if not per:
        raise ValueError("from_collision: no collision geoms on the robot's bodies")
    if len(per) > 64:
        raise ValueError(f"from_collision: {len(per)} bodies with collision geoms (the ray kernels take 64 meshes)")
    mgeom, mroot, nb_all, ni_all, tri_all, names = [], [], [], [], [], []
    nnode = ntri = 0
    for b, tris in sorted(per.items()):
        tri = np.concatenate(tris).astype(np.float32)
        nb, ni, order = bvh(tri)
        tri = tri[order]
        leaf = ni[:, 0] < 0
        ni = ni.copy()
        ni[leaf, 0] -= ntri
        ni[~leaf] += nnode
        vert = tri.reshape(-1, 3).astype(np.float64)
        c = vert.mean(0)
        r = float(np.linalg.norm(vert - c, axis=1).max())
        mgeom.append(np.concatenate([[b], np.zeros(3), [1.0, 0.0, 0.0, 0.0], c, [r]]))
        mroot.append(nnode)
        nb_all.append(nb), ni_all.append(ni), tri_all.append(tri)
        names.append(f"{model.body(b).name}_collision")
        nnode += len(nb)
        ntri += len(tri)
    G = len(mgeom)
    return dict(mgeom=np.asarray(mgeom, np.float32), mroot=np.asarray(mroot, np.int32),
                nodeb=np.concatenate(nb_all).astype(np.float32), nodei=np.concatenate(ni_all).astype(np.int32),
                tri=np.concatenate(tri_all).astype(np.float32), names=np.asarray(names),
                source_geom=np.full(G, -1, np.int32), rgba=np.tile(np.asarray(rgba, np.float32), (G, 1)))

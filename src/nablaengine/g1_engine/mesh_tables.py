"""The mesh tables of a model's collision meshes, shared by both engines (g1_engine.mesh, forest_engine.convex): the
collision meshes' vertices, hull graphs (MuJoCo's mesh_graph: the hill-climbing support) and polygons (mesh_poly*: the
multicontact faces) in compact arrays (MESH_LAYOUT), addresses rebased."""
import mujoco
import numpy as np

MESH_LAYOUT = (
    ("c_gmesh", "i", "ng"), ("c_mvadr", "i", "nm"), ("c_mvnum", "i", "nm"), ("c_mgadr", "i", "nm"),
    ("c_mpadr", "i", "nm"), ("c_mpnum", "i", "nm"), ("c_mvert", "f", "nv3"), ("c_mgraph", "i", "ngr"),
    ("c_mpnorm", "f", "np3"), ("c_mpvadr", "i", "np"), ("c_mpvnum", "i", "np"), ("c_mpvert", "i", "npv"),
    ("c_mpmadr", "i", "nv"), ("c_mpmnum", "i", "nv"), ("c_mpmap", "i", "npm"),
)


def mesh_tables(m, geoms):
    """dict of the MESH_LAYOUT arrays for the collision geoms geoms (step.collision_geoms), and the sizes; None when
    none is a mesh."""
    gm = [int(m.geom_dataid[g]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else -1 for g in geoms]
    used = sorted(set(x for x in gm if x >= 0))
    if not used:
        return None
    cid = {k: i for i, k in enumerate(used)}
    t = {k: [] for k, _, _ in MESH_LAYOUT}
    t["c_gmesh"] = [cid[x] if x >= 0 else -1 for x in gm]
    nv = ng = npoly = npv = npm = 0
    for k in used:
        va, vn = int(m.mesh_vertadr[k]), int(m.mesh_vertnum[k])
        pa, pn = int(m.mesh_polyadr[k]), int(m.mesh_polynum[k])
        t["c_mvadr"].append(nv)
        t["c_mvnum"].append(vn)
        t["c_mpadr"].append(npoly)
        t["c_mpnum"].append(pn)
        t["c_mvert"] += list(m.mesh_vert[va:va + vn].ravel())
        ga = int(m.mesh_graphadr[k])
        if ga >= 0:
            gn = 2 + 3 * int(m.mesh_graph[ga]) + 3 * int(m.mesh_graph[ga + 1])
            t["c_mgadr"].append(ng)
            t["c_mgraph"] += list(m.mesh_graph[ga:ga + gn])
            ng += gn
        else:
            t["c_mgadr"].append(-1)
        t["c_mpnorm"] += list(m.mesh_polynormal[pa:pa + pn].ravel())
        for i in range(pa, pa + pn):
            a, n = int(m.mesh_polyvertadr[i]), int(m.mesh_polyvertnum[i])
            t["c_mpvadr"].append(npv)
            t["c_mpvnum"].append(n)
            t["c_mpvert"] += list(m.mesh_polyvert[a:a + n])
            npv += n
        for i in range(va, va + vn):
            a, n = int(m.mesh_polymapadr[i]), int(m.mesh_polymapnum[i])
            t["c_mpmadr"].append(npm)
            t["c_mpmnum"].append(n)
            t["c_mpmap"] += list(m.mesh_polymap[a:a + n])
            npm += n
        nv += vn
        npoly += pn
    sizes = dict(ng=len(geoms), nm=len(used), nv3=3 * nv, ngr=max(ng, 1), np3=3 * npoly, np=npoly, npv=npv, nv=nv,
                 npm=npm)
    if not t["c_mgraph"]:
        t["c_mgraph"] = [0]
    poly_max = max(t["c_mpvnum"]) if t["c_mpvnum"] else 0
    deg_max = max(t["c_mpmnum"]) if t["c_mpmnum"] else 0
    return dict(arrays=t, sizes=sizes, poly_max=poly_max, deg_max=deg_max)


def layout(m, geoms):
    """[(symbol, kind, length)] of the mesh constants (empty without mesh collision geoms)."""
    mt = mesh_tables(m, geoms)
    if mt is None:
        return []
    return [(s, k, mt["sizes"][n]) for s, k, n in MESH_LAYOUT]


def constants(m, geoms):
    """The mesh constants as float64 arrays in layout() order."""
    mt = mesh_tables(m, geoms)
    if mt is None:
        return []
    return [np.asarray(mt["arrays"][s], np.float64) for s, _, _ in MESH_LAYOUT]

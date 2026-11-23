"""Cylinder contacts of the forest engine (model_hooks): every pair MuJoCo 3.17 collides with a cylinder among the
forest engine's geom types, through the tree engine's code for them."""
import mujoco

from nablaengine.g1_engine import cylinder as g1_cylinder

GEOM = 5

CCD_COND = "((t1 == 2 && t2 == 5) || (t1 == 5 && (t2 == 5 || t2 == 3)))"


def _types(m):
    return [tuple(sorted((int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]]))))
            for p in range(m.npair)]


def has_cylinder(m):
    """A cylinder in a contact pair."""
    cyl = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    return any(cyl in t for t in _types(m))


def ccd_pairs(m):
    """A capsule-cylinder, cylinder-cylinder or cylinder-box pair (MuJoCo's convex collision, CCD_COND)."""
    G = mujoco.mjtGeom
    cyl = int(G.mjGEOM_CYLINDER)
    want = {tuple(sorted((cyl, int(t)))) for t in (G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER, G.mjGEOM_BOX)}
    return any(t in want for t in _types(m))


def ccd_with_cylinder(m):
    """A convex (CCD) pair with a cylinder, a mesh included: the multicontact's polygon buffers hold its 16-gon caps."""
    G = mujoco.mjtGeom
    return ccd_pairs(m) or tuple(sorted((int(G.mjGEOM_CYLINDER), int(G.mjGEOM_MESH)))) in _types(m)


def check(m):
    """The cylinder pairs of model m the engine has none of (empty: all are supported): a cylinder against a geom type
    other than a plane, sphere, capsule, ellipsoid (forest_engine.ellipsoid), cylinder, box or mesh."""
    G = mujoco.mjtGeom
    cyl = int(G.mjGEOM_CYLINDER)
    ok = {tuple(sorted((int(t), cyl))) for t in (G.mjGEOM_PLANE, G.mjGEOM_SPHERE, G.mjGEOM_CAPSULE, G.mjGEOM_ELLIPSOID,
                                                  G.mjGEOM_CYLINDER, G.mjGEOM_BOX, G.mjGEOM_MESH)}
    return sorted({t for t in _types(m) if cyl in t and t not in ok})


def fwd_ccd(fn="cyc_pair_r"):
    """The forward's branch for CCD_COND (collide_pair): past the OBB test (forest_engine.convex), the contacts of
    cylccd's forward (fn: its RF wrapper, or model_hooks' DK_CCD_FLOAT experiment's)."""
    return ("  if " + CCD_COND + " {  // capsule-, cylinder-, box-cylinder (forest_engine.cylinder: g1_engine.cylccd's "
            "MuJoCo Warp 3.17\n    // CCD), past the OBB test\n"
            "    if (obb_separated(g1, p1, q1, g2, p2, q2)) return;\n"
            "    CSink o;\n"
            f"    {fn}(t1, g1, p1, q1, sz1, t2, g2, p2, q2, sz2, o);\n"
            "    sink_flush(s, pair, o);\n"
            "    return;\n  }\n")


def hooks(m, part, fn="cyc_pair_r"):
    """The (anchor, replacement, count) edits of the forest text of model m (part "engine" or "forward") for its
    cylinder pairs (none without a cylinder in a pair): the primitives' forward and adjoint (g1_engine.cylinder), the
    CCD pairs' forward branch (fwd_ccd; its device code and adjoint: model_hooks.convex_mj_hooks)."""
    if part not in ("engine", "forward") or not has_cylinder(m):
        return ()
    bad = check(m)
    if bad:
        raise ValueError(f"cylinder pairs the forest engine does not have (MuJoCo geom types): {bad}")
    prim = g1_cylinder.hooks()
    out = prim[:3] + (prim[3:] if part == "engine" else ())
    if ccd_pairs(m):
        anchor = "  atomicOr(&s.unsupported, 1);  // other type combinations do not occur in the G1 model\n"
        out += ((anchor, fwd_ccd(fn) + anchor, 1),)
    return out

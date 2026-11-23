"""Per-geom contact inverse weights (both engines)."""
import mujoco
import numpy as np

NUMERIC = "nablaengine_geom_invweight"

HOOKS = (
    ("c_binvw[c_gbody[PG1(pr)]]", "c_ginvw[PG1(pr)]"),
    ("c_binvw[c_gbody[PG2(pr)]]", "c_ginvw[PG2(pr)]"),
    ("RADD(c_binvw[b1], c_binvw[b2])", "RADD(c_ginvw[g1], c_ginvw[g2])"),
)


def geom_values(m):
    """The model's per-geom inverse weights (ngeom,) float64 from the custom numeric NUMERIC (negative entries: the
    geom's body's value), or None without it."""
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, NUMERIC)
    if i < 0:
        return None
    adr, n = int(m.numeric_adr[i]), int(m.numeric_size[i])
    if n != m.ngeom:
        raise ValueError(f"custom numeric {NUMERIC}: size {n}, the model has {m.ngeom} geoms (one value per geom)")
    v = np.array(m.numeric_data[adr:adr + n], np.float64)
    if not np.isfinite(v).all():
        raise ValueError(f"custom numeric {NUMERIC}: the values must be finite")
    body = m.body_invweight0[m.geom_bodyid, 0]
    return np.where(v < 0, body, v)


def table(m, geoms, extra=0):
    """The engine's table c_ginvw (float64, one per collision-geom table row: geoms, then `extra` world rows of value 0,
    e.g. the G1 class's terrain pseudo geom), or None when the model needs none: no NUMERIC field, or every collision
    geom's value equal in float32 to its body's (the text and constants then stay as without the field)."""
    v = geom_values(m)
    if v is None or not len(geoms):
        return None
    geoms = np.asarray(geoms, int)
    v = v[geoms]
    if (v.astype(np.float32) == m.body_invweight0[m.geom_bodyid[geoms], 0].astype(np.float32)).all():
        return None
    return np.concatenate([v, np.zeros(extra)])


def header(m, geoms):
    """The table's declaration for step_header (\"\" without a table)."""
    return "DKC float c_ginvw[NGEOMC];  // per-geom contact inverse weights (nablaengine.model.invweight)\n" \
        if table(m, geoms) is not None else ""


def apply(src, m, geoms):
    """src (a unit's full text, after the elliptic edits) reading c_ginvw for the contacts' inverse weight where the
    model has a table, else src unchanged."""
    if table(m, geoms) is None:
        return src
    hit = 0
    for old, new in HOOKS:
        hit += src.count(old)
        src = src.replace(old, new)
    assert hit >= 2, "nablaengine.model.invweight: the contact inverse weight's expressions were not found"
    rest = [ln for ln in src.splitlines() if "c_binvw[" in ln and not ln.startswith("DKC float c_binvw[")]
    assert not rest, f"nablaengine.model.invweight: c_binvw is still read: {rest}"
    return src


def fill(src, m):
    """The field for a model m made from the model src by fusing jointless bodies into their parents (fusestatic; m's
    bodies and geoms carry src's names, the fused bodies are absent from m): (field (m.ngeom,), values (m.ngeom,)) or
    None when no geom's value would differ from its body's in float32 (also when nothing was fused). field: src's
    body_invweight0[:, 0] for"""
    present = {m.body(b).name for b in range(m.nbody)}
    touched = set()
    for b0 in range(1, src.nbody):
        if src.body(b0).name in present:
            continue
        if not src.body(b0).name:
            raise ValueError(f"invweight.fill: source body {b0} is unnamed and absent from the fused model")
        touched.add(b0)
        par = int(src.body_parentid[b0])
        while par and src.body(par).name not in present:
            par = int(src.body_parentid[par])
        touched.add(par)
    touched.discard(0)
    if not touched:
        return None
    body = m.body_invweight0[m.geom_bodyid, 0]
    values = body.copy()
    field = np.full(m.ngeom, -1.0)
    for g in range(m.ngeom):
        g0 = mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_GEOM, m.geom(g).name) if m.geom(g).name else -1
        if g0 < 0:
            raise ValueError(f"invweight.fill: geom {g} ({m.geom(g).name!r}) has no namesake in the source model")
        b0 = int(src.geom_bodyid[g0])
        v0 = src.body_invweight0[b0, 0]
        if b0 in touched or np.float32(v0) != np.float32(body[g]):
            values[g] = field[g] = v0
    if (values.astype(np.float32) == body.astype(np.float32)).all():
        return None
    return field, values


def add(spec, field):
    """spec with the field (fill's) as the custom numeric NUMERIC."""
    spec.add_numeric(name=NUMERIC, size=len(field), data=list(map(float, field)))
    return spec


def fill_spec(spec):
    """fill for spec (fusestatic set) against the same spec compiled without fusestatic: (fill's result or None, the
    fused model, in whose geom order the field is). Call it before spec's first compile: MuJoCo's fusestatic removes
    the fused bodies from the spec itself when it compiles. Works on copies (unnamed bodies and geoms named there
    only, to match them); spec is unchanged. Only the masses, inertias and kinematic tree matter (body_invweight0):
    later edits of actuators, pairs, damping or contact parameters do not change the field."""
    named = spec.copy()
    for i, b in enumerate(named.bodies[1:]):
        b.name = b.name or f"_body{i}"
    for k, g in enumerate(named.geoms):
        g.name = g.name or f"_geom{k}"
    unfused = named.copy()
    unfused.compiler.fusestatic = False
    m0 = unfused.compile()
    mf = named.compile()
    return fill(m0, mf), mf


def attach(spec, filled, pairs_only=True):
    """spec (compiled since fill_spec, its geoms unchanged) with fill_spec's field as the custom numeric NUMERIC:
    (spec's compiled model, the values) or (spec's compiled model, None) without a field (spec unchanged).
    pairs_only: the field only where it changes a contact pair's geom (the engines' table, table()); otherwise spec
    stays as it was."""
    m = spec.compile()
    res, mf = filled
    if res is not None and pairs_only:
        g = sorted({int(x) for x in m.pair_geom1} | {int(x) for x in m.pair_geom2})
        body = m.body_invweight0[m.geom_bodyid[g], 0]
        if (res[1][g].astype(np.float32) == body.astype(np.float32)).all():
            res = None
    if res is None:
        return m, None
    assert m.ngeom == mf.ngeom and all(m.geom(g).name in ("", mf.geom(g).name) for g in range(m.ngeom)), \
        "invweight.attach: the geoms changed since fill_spec"
    assert np.array_equal(m.geom_bodyid, mf.geom_bodyid) and np.array_equal(m.geom_type, mf.geom_type)
    add(spec, res[0])
    m = spec.compile()
    assert np.array_equal(geom_values(m), res[1])
    return m, res[1]

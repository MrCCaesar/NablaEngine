"""The forward tape of the G1 engine: one record per world and substep, the residuals of the analytic adjoint."""

import numpy as np

from nablaengine.g1_engine import step

ALIGN = 32
FLAG_UNSUPPORTED = 31
FLAG_REFACT = 1 << 8
FLAG_NONCONV = 1 << 9
FLAG_SINGULAR = 1 << 10
FLAG_VERIFY_DIFF = 1 << 11
SINGULAR_NU = 5e-3
CON_NONE = 0xFFFF
CON_NONE_WIDE = 0xFFFFFFFF

_FIELDS = (
    ("hdr", "OFF_HDR", 1),
    ("flags", "OFF_FLAGS", 1),
    ("min_nu", "OFF_MIN_NU", 1),
    ("reserved", "OFF_RESERVED", 1),
    ("clamp", "OFF_CLAMP", lambda nq, nv, ntri, nc: -(-nv // 32)),
    ("zone", "OFF_ZONE", lambda nq, nv, ntri, nc: -(-step.NEFC // 32)),
    ("con", "OFF_CON", lambda nq, nv, ntri, nc: -(-nc // 2)),
    ("qpos", "OFF_QPOS", lambda nq, nv, ntri, nc: nq),
    ("qvel", "OFF_QVEL", lambda nq, nv, ntri, nc: nv),
    ("qacc", "OFF_QACC", lambda nq, nv, ntri, nc: nv),
    ("qacc_adv", "OFF_QADV", lambda nq, nv, ntri, nc: nv),
    ("kvw", "OFF_KVW", lambda nq, nv, ntri, nc: nv),
    ("mslope", "OFF_MSLOPE", lambda nq, nv, ntri, nc: nv),
    (None, None, ALIGN),
    ("L_H", "OFF_LH", lambda nq, nv, ntri, nc: ntri),
    (None, None, ALIGN),
    ("L_Q", "OFF_LQ", lambda nq, nv, ntri, nc: ntri),
    (None, None, ALIGN),
)


_TERRAIN_FIELDS = (("telem", "OFF_TELEM", lambda nq, nv, ntri, nc: -(-nc // 2)),)
_NATIVE_FIELDS = (("taux", "OFF_TAUX", lambda nq, nv, ntri, nc: nc),)


def kind(terrain):
    """The record kind of a build: None (no terrain), "analytic" or "native" (terrain: (TerrainConfig, geoms), a
    TerrainConfig, or None)."""
    if terrain is None:
        return None
    cfg = terrain[0] if isinstance(terrain, tuple) else terrain
    return cfg.mode


def _wide(m, terrain):
    return not terrain and m.npair >= step.WIDE_PAIRS


def _table(m, max_contacts=None, terrain=False, max_rows=None, motor=False):
    nc = max_contacts or step.maxcon(m)
    nv, ntri, nefc, wide = m.nv, m.nv * (m.nv + 1) // 2, step.nefc(max_rows), _wide(m, terrain)
    npmax = 1024 if terrain else (65536 if wide else 512)
    assert nv <= 64 and m.npair < npmax and nc <= 255 and nefc <= step.MAX_ROWS, \
        "field widths (clamp u64, pair bits, ncon u8, zone words)"
    fields = _FIELDS
    if step.clamp_implicit()[0] != 2:
        fields = tuple(f for f in fields if f[0] != "kvw")
    if not motor:
        fields = tuple(f for f in fields if f[0] != "mslope")
    if terrain:
        i = [f[0] for f in _FIELDS].index("con") + 1
        base = tuple(f for f in _FIELDS if motor or f[0] != "mslope")
        fields = base[:i] + _TERRAIN_FIELDS + (_NATIVE_FIELDS if terrain == "native" else ()) + base[i:]
    out, off = [], 0
    for name, define, n in fields:
        if name is None:
            off = -(-off // n) * n
            continue
        n = n(m.nq, nv, ntri, nc) if callable(n) else n
        if name == "zone":
            n = -(-nefc // 32)
        elif name == "con" and wide:
            n = nc
        out.append((name, define, off, n))
        off += n
    return out, off


def layout(m, max_contacts=None, terrain=False, max_rows=None, motor=False):
    """{name: (offset, length)} of a record in 32-bit words (mjlab model: L_H at 192, L_Q at 832, 1472 words).
    terrain: a terrain build's record kind (kind(): "analytic" or "native"; _TERRAIN_FIELDS, _NATIVE_FIELDS). motor:
    a motor-limit build (nablaengine.motor: the mslope field)."""
    return {name: (off, n) for name, _, off, n in _table(m, max_contacts, terrain, max_rows, motor)[0]}


def rec_words(m, max_contacts=None, terrain=False, max_rows=None, motor=False):
    """Record length in words (a multiple of ALIGN)."""
    return _table(m, max_contacts, terrain, max_rows, motor)[1]


def header(m, max_contacts=None, terrain=False, max_rows=None, motor=False):
    """#define OFF_* / REC_WORDS / TAPE_* for the engine's CUDA code (step.py)."""
    tab, total = _table(m, max_contacts, terrain, max_rows, motor)
    lines = [f"#define REC_WORDS {total}  // tape record (tape.py): words per world and substep"]
    lines += [f"#define {define} {off}  // {name}, {n} words" for name, define, off, n in tab]
    lines += [f"#define TAPE_REFACT {FLAG_REFACT}u", f"#define TAPE_NONCONV {FLAG_NONCONV}u",
              f"#define TAPE_SINGULAR {FLAG_SINGULAR}u", f"#define TAPE_VERIFY_DIFF {FLAG_VERIFY_DIFF}u",
              f"#define TAPE_SINGULAR_NU {SINGULAR_NU}f",
              f"#define TAPE_CON_NONE {CON_NONE_WIDE if _wide(m, terrain) else CON_NONE}u"]
    return "\n" + "\n".join(lines) + "\n"


def tri_unpack(packed, n):
    """(..., n (n + 1) / 2) packed lower triangles TRI(i, j) -> (..., n, n) lower-triangular matrices."""
    packed = np.asarray(packed)
    i, j = np.tril_indices(n)
    out = np.zeros(packed.shape[:-1] + (n, n), packed.dtype)
    out[..., i, j] = packed[..., i * (i + 1) // 2 + j]
    return out


def factor_forms(m):
    """{"L_Q": form, "L_H": form} of model m's records under $DK_TREE_CHOL (step.tree_chol_stages): "dense" (A = L L',
    the default) or "tree" (A = R' R, R on the tree pattern; the same packed storage)."""
    st = step.tree_chol_stages(m)
    return {"L_Q": "tree" if st else "dense", "L_H": "tree" if st and st["newton"] else "dense"}


def factor_product(L, form):
    """The matrix of factors L (..., n, n) (tri_unpack) of form factor_forms(): L L' (dense) or L' L (tree)."""
    Lt = np.swapaxes(L, -1, -2)
    return L @ Lt if form == "dense" else Lt @ L


def decode(arr, m, max_contacts=None, terrain=False, max_rows=None, motor=False):
    """Records (..., REC_WORDS) float32 -> dict of arrays (leading shape kept): ncon, nefc, nlim, niter, flags (and
    unsupported / refact / nonconv / singular / verify_diff), min_nu, clamp (u64 dof mask) and clamp_dof (bool, nv),
    zone (bool, the row capacity), pair / slot (int, maxcon; pair -1 past ncon), qpos, qvel, qacc, qacc_adv, L_H, L_Q
    (packed); terrain: a terrain build's record, with elem (the contact's terrain element); max_contacts, max_rows: the
    build's capacities (None: the defaults); motor: a motor-limit build's record, with mslope (-df/dqdot per dof)."""
    a = np.ascontiguousarray(arr, np.float32)
    nw = rec_words(m, max_contacts, terrain, max_rows, motor)
    assert a.shape[-1] == nw, (a.shape, nw)
    u = a.view(np.uint32)
    lay = layout(m, max_contacts, terrain, max_rows, motor)
    nefc = step.nefc(max_rows)
    sl = lambda name: slice(lay[name][0], lay[name][0] + lay[name][1])
    hdr, flags = u[..., lay["hdr"][0]].astype(np.int64), u[..., lay["flags"][0]].astype(np.int64)
    out = dict(ncon=hdr & 255, nefc=(hdr >> 8) & 255, nlim=(hdr >> 16) & 255, niter=hdr >> 24, flags=flags,
               unsupported=flags & FLAG_UNSUPPORTED, refact=(flags & FLAG_REFACT) != 0,
               nonconv=(flags & FLAG_NONCONV) != 0, singular=(flags & FLAG_SINGULAR) != 0,
               verify_diff=(flags & FLAG_VERIFY_DIFF) != 0, min_nu=a[..., lay["min_nu"][0]])
    if nefc > 255:
        out["nefc"] = u[..., lay["reserved"][0]].astype(np.int64)
    cw = u[..., sl("clamp")].astype(np.uint64)
    out["clamp"] = sum(cw[..., k] << np.uint64(32 * k) for k in range(cw.shape[-1]))
    out["clamp_dof"] = ((cw[..., :, None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).astype(bool).reshape(
        cw.shape[:-1] + (-1,))[..., :m.nv]
    zw = u[..., sl("zone")].astype(np.uint64)
    out["zone"] = ((zw[..., :, None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).astype(bool).reshape(
        zw.shape[:-1] + (-1,))[..., :nefc]
    nc = max_contacts or step.maxcon(m)
    if _wide(m, terrain):
        e = u[..., sl("con")].astype(np.int64)
        pb = 16
    else:
        e = np.ascontiguousarray(u[..., sl("con")]).view(np.uint16)[..., :nc].astype(np.int64)
        pb = 10 if terrain else 9
    past = np.arange(nc) >= out["ncon"][..., None]
    out["pair"], out["slot"] = np.where(past, -1, e & ((1 << pb) - 1)), np.where(past, -1, e >> pb)
    if terrain:
        x = np.ascontiguousarray(u[..., sl("telem")]).view(np.uint16)[..., :nc].astype(np.int64)
        out["elem"] = np.where(past, -1, x)
    for name in ("qpos", "qvel", "qacc", "qacc_adv", "L_H", "L_Q") + (("mslope",) if motor else ()):
        out[name] = a[..., sl(name)]
    return out

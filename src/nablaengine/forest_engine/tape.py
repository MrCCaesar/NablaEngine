"""The forward tape of the G1 engine: one record per world and substep, the residuals of the analytic adjoint."""

import numpy as np

from nablaengine.forest_engine import step

ALIGN = 32
FLAG_UNSUPPORTED = 31
FLAG_REFACT = 1 << 8
FLAG_NONCONV = 1 << 9
FLAG_SINGULAR = 1 << 10
FLAG_VERIFY_DIFF = 1 << 11
FLAG_LQ_TREE = 1 << 12
FLAG_LH_SPARSE = 1 << 13
FLAG_H_LOWRANK = 1 << 14
FLAG_H_DENSE = 1 << 15
SINGULAR_NU = 5e-3
CON_NONE = 0xFFFFFFFF

_FIELDS = (
    ("hdr", "OFF_HDR", 1),
    ("flags", "OFF_FLAGS", 1),
    ("min_nu", "OFF_MIN_NU", 1),
    ("reserved", "OFF_RESERVED", 1),
    ("clamp", "OFF_CLAMP", lambda nq, nv, ntri, nc, nef: -(-nv // 32)),
    ("kvoff", "OFF_KVOFF", lambda nq, nv, ntri, nc, nef: -(-nv // 32)),
    ("uclamp", "OFF_UCLAMP", lambda nq, nv, ntri, nc, nef: -(-nv // 32)),
    ("zone", "OFF_ZONE", lambda nq, nv, ntri, nc, nef: -(-nef // 32)),
    ("con", "OFF_CON", lambda nq, nv, ntri, nc, nef: nc),
    ("qpos", "OFF_QPOS", lambda nq, nv, ntri, nc, nef: nq),
    ("qvel", "OFF_QVEL", lambda nq, nv, ntri, nc, nef: nv),
    ("qacc", "OFF_QACC", lambda nq, nv, ntri, nc, nef: nv),
    ("qacc_adv", "OFF_QADV", lambda nq, nv, ntri, nc, nef: nv),
    ("kvw", "OFF_KVW", lambda nq, nv, ntri, nc, nef: nv),
    ("mslope", "OFF_MSLOPE", lambda nq, nv, ntri, nc, nef: nv),
    (None, None, ALIGN),
    ("L_H", "OFF_LH", lambda nq, nv, ntri, nc, nef: ntri),
    (None, None, ALIGN),
    ("L_Q", "OFF_LQ", lambda nq, nv, ntri, nc, nef: ntri),
    (None, None, ALIGN),
)


_TERRAIN_FIELDS = (("telem", "OFF_TELEM", lambda nq, nv, ntri, nc, nef: -(-nc // 2)),)
_NATIVE_FIELDS = (("taux", "OFF_TAUX", lambda nq, nv, ntri, nc, nef: nc),)
_NOSLIP_FIELDS = (
    ("ns_qn", "OFF_NSQN", lambda nq, nv, ntri, nc, nef: nv),
    ("ns_hdr", "OFF_NSHDR", 1),
    ("ns_st", "OFF_NSST", lambda nq, nv, ntri, nc, nef: nc),
    ("ns_con", "OFF_NSCON", lambda nq, nv, ntri, nc, nef: 13 * nc),
    ("ns_f", "OFF_NSF", lambda nq, nv, ntri, nc, nef: nef),
    ("ns_df", "OFF_NSDF", lambda nq, nv, ntri, nc, nef: nef),
    ("ns_jar", "OFF_NSJAR", lambda nq, nv, ntri, nc, nef: nef),
    (None, None, ALIGN),
    ("L_M", "OFF_LM", lambda nq, nv, ntri, nc, nef: ntri),
    (None, None, ALIGN),
)


def _table(m, max_contacts=None, max_rows=None, motor=False, terrain=False):
    nc = max_contacts or step.maxcon(m)
    nv, ntri = m.nv, m.nv * (m.nv + 1) // 2
    nlim = int(sum(m.jnt_limited[j] for j in range(m.njnt) if m.jnt_type[j] in (2, 3)))
    assert nv <= 128 and m.npair < 65536 and nc <= 255 and step.nefc(nc, m, max_rows) < 1024 and (nlim < 64 or nv > 64), \
        "field widths (pair 16 bits, ncon 8, nefc 10, nlim 6 bits)"
    fields = _FIELDS
    if step.clamp_implicit()[0] != 2:
        fields = tuple(f for f in fields if f[0] != "kvw")
    if not motor:
        fields = tuple(f for f in fields if f[0] != "mslope")
    from nablaengine.forest_engine import tendon as _ten
    if _ten.applies(m) and _ten.spatial_build(m):
        i = [f[0] for f in fields].index("qvel") + 1
        from nablaengine.forest_engine import muscle
        nta = max(len(_ten.actuators(m)), 1)
        fields = fields[:i] + (("tact", "OFF_TACT", (2 + (muscle.NDER if muscle.applies(m) else 0)) * nta),) + fields[i:]
    if terrain:
        i = [f[0] for f in fields].index("con") + 1
        fields = fields[:i] + _TERRAIN_FIELDS + (_NATIVE_FIELDS if terrain == "native" else ()) + fields[i:]
    if int(m.opt.noslip_iterations) > 0:
        fields = fields + _NOSLIP_FIELDS
    st = _storage(m)
    out, off = [], 0
    for name, define, n in fields:
        if name is None:
            off = -(-off // n) * n
            continue
        n = n(m.nq, nv, ntri, nc, step.nefc(nc, m, max_rows)) if callable(n) else n
        if st is not None and name in ("L_H", "L_Q"):
            n = st["nnz_lh" if name == "L_H" else "nnz_lq"]
        out.append((name, define, off, n))
        off += n
    return out, off


def _storage(m):
    if m.nv <= 64:
        return None
    from nablaengine.forest_engine import nv128
    return nv128.storage(m)


def kind(terrain):
    """The record kind of a build: False (no terrain), "analytic" or "native" (terrain: (TerrainConfig, geoms), a
    TerrainConfig or None)."""
    if terrain is None:
        return False
    cfg = terrain[0] if isinstance(terrain, tuple) else terrain
    return "native" if cfg.mode == "native" else "analytic"


def layout(m, max_contacts=None, terrain=False, max_rows=None, motor=False):
    """{name: (offset, length)} of a record in 32-bit words (mjlab model: L_H at 192, L_Q at 832, 1472 words). motor:
    a motor-limit build (nablaengine.motor: the mslope field). terrain: a terrain build's record kind (kind())."""
    return {name: (off, n) for name, _, off, n in _table(m, max_contacts, max_rows, motor, terrain)[0]}


def rec_words(m, max_contacts=None, terrain=False, max_rows=None, motor=False):
    """Record length in words (a multiple of ALIGN)."""
    return _table(m, max_contacts, max_rows, motor, terrain)[1]


def header(m, max_contacts=None, terrain=False, max_rows=None, motor=False):
    """#define OFF_* / REC_WORDS / TAPE_* for the engine's CUDA code (step.py)."""
    tab, total = _table(m, max_contacts, max_rows, motor, terrain)
    lines = [f"#define REC_WORDS {total}  // tape record (tape.py): words per world and substep"]
    lines += [f"#define {define} {off}  // {name}, {n} words" for name, define, off, n in tab]
    lines += [f"#define TAPE_REFACT {FLAG_REFACT}u", f"#define TAPE_NONCONV {FLAG_NONCONV}u",
              f"#define TAPE_SINGULAR {FLAG_SINGULAR}u", f"#define TAPE_VERIFY_DIFF {FLAG_VERIFY_DIFF}u",
              f"#define TAPE_SINGULAR_NU {SINGULAR_NU}f", f"#define TAPE_CON_NONE {CON_NONE}u"]
    return "\n" + "\n".join(lines) + "\n"


def tri_unpack(packed, n):
    """(..., n (n + 1) / 2) packed lower triangles TRI(i, j) -> (..., n, n) lower-triangular matrices."""
    packed = np.asarray(packed)
    i, j = np.tril_indices(n)
    out = np.zeros(packed.shape[:-1] + (n, n), packed.dtype)
    out[..., i, j] = packed[..., i * (i + 1) // 2 + j]
    return out


def decode(arr, m, max_contacts=None, max_rows=None, motor=False, terrain=False):
    """Records (..., REC_WORDS) float32 -> dict of arrays (leading shape kept): ncon, nefc, nlim, niter, flags (and
    unsupported / refact / nonconv / singular / verify_diff / lq_tree: L_Q = R, Q = R' R, the tree factor / lh_sparse: L_H = R on P_H, nv128
    the sparse Newton factor), min_nu,
    clamp (u64 dof mask) and clamp_dof (bool, nv), zone (bool, step.nefc), pair / slot (int, maxcon; pair -1 past
    ncon), qpos, qvel, qacc, qacc_adv, L_H, L_Q (packed triangles; the pattern storage records, whose L_Q / L_H hold only
    their patterns' entries, expanded with zeros elsewhere);
    motor: a motor-limit build's record, with mslope (-df/dqdot per dof); terrain: a terrain build's record kind
    (kind()), with elem (the contact's terrain element) and, native, taux."""
    a = np.ascontiguousarray(arr, np.float32)
    nw = rec_words(m, max_contacts, terrain, max_rows=max_rows, motor=motor)
    assert a.shape[-1] == nw, (a.shape, nw)
    u = a.view(np.uint32)
    lay = layout(m, max_contacts, terrain, max_rows=max_rows, motor=motor)
    sl = lambda name: slice(lay[name][0], lay[name][0] + lay[name][1])
    hdr, flags = u[..., lay["hdr"][0]].astype(np.int64), u[..., lay["flags"][0]].astype(np.int64)
    nlim = u[..., lay["reserved"][0]].astype(np.int64) if m.nv > 64 else (hdr >> 18) & 63
    out = dict(ncon=hdr & 255, nefc=(hdr >> 8) & 1023, nlim=nlim, niter=hdr >> 24, flags=flags,
               unsupported=flags & FLAG_UNSUPPORTED, refact=(flags & FLAG_REFACT) != 0,
               nonconv=(flags & FLAG_NONCONV) != 0, singular=(flags & FLAG_SINGULAR) != 0,
               verify_diff=(flags & FLAG_VERIFY_DIFF) != 0, lq_tree=(flags & FLAG_LQ_TREE) != 0,
               lh_sparse=(flags & FLAG_LH_SPARSE) != 0, h_lowrank=(flags & FLAG_H_LOWRANK) != 0,
               h_dense=(flags & FLAG_H_DENSE) != 0,
               min_nu=a[..., lay["min_nu"][0]])
    cw = u[..., sl("clamp")].astype(np.uint64)
    out["clamp"] = sum(cw[..., k] << np.uint64(32 * k) for k in range(cw.shape[-1]))
    out["clamp_dof"] = ((cw[..., :, None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).astype(bool).reshape(
        cw.shape[:-1] + (-1,))[..., :m.nv]
    zw = u[..., sl("zone")].astype(np.uint64)
    out["zone"] = ((zw[..., :, None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).astype(bool).reshape(
        zw.shape[:-1] + (-1,))
    nc = max_contacts or step.maxcon(m)
    out["zone"] = out["zone"][..., :step.nefc(nc, m, max_rows)]
    e = u[..., sl("con")].astype(np.int64)
    past = np.arange(nc) >= out["ncon"][..., None]
    out["pair"], out["slot"] = np.where(past, -1, e & 0xFFFF), np.where(past, -1, e >> 16)
    if terrain:
        x = np.ascontiguousarray(u[..., sl("telem")]).view(np.uint16)[..., :nc].astype(np.int64)
        out["elem"] = np.where(past, -1, x)
        if terrain == "native":
            out["taux"] = np.where(past, 0, u[..., sl("taux")].astype(np.int64))
    for name in ("qpos", "qvel", "qacc", "qacc_adv", "L_H", "L_Q") + (("mslope",) if motor else ()):
        out[name] = a[..., sl(name)]
    st = _storage(m)
    if st is not None:
        ntri = m.nv * (m.nv + 1) // 2
        for name, lay_ in (("L_Q", st["tl"]), ("L_H", st["cl"])):
            if lay_ is not None:
                full = np.zeros(out[name].shape[:-1] + (ntri,), np.float32)
                full[..., np.asarray(lay_["tri"])] = out[name]
                out[name] = full
    if "ns_qn" in lay:
        nh = u[..., lay["ns_hdr"][0]].astype(np.int64)
        out.update(ns_qacc_newton=a[..., sl("ns_qn")], ns_iter=nh & 255, ns_flags=nh & ~255,
                   ns_status=u[..., sl("ns_st")].astype(np.int64), ns_force=a[..., sl("ns_f")], L_M=a[..., sl("L_M")],
                   ns_con=a[..., sl("ns_con")].reshape(a.shape[:-1] + (nc, 13)))
    return out

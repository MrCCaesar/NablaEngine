"""Opt-in: the register-blocked dense factor of forest_engine.nv128 (dense_blk) for forest-engine models with
44 <= nv <= 64 ($DK_FOREST_CHOL_BLK=1; default 0: the text unchanged)."""
import os

VAR = "DK_FOREST_CHOL_BLK"
W = 8
NV_MIN, NV_MAX = 44, 64


def on():
    """$DK_FOREST_CHOL_BLK: 0 (default) or 1."""
    v = (os.environ.get(VAR) or "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"${VAR}={v!r}: must be 0 or 1")
    return v == "1"


def enabled(m):
    return on() and NV_MIN <= m.nv <= NV_MAX


def functions(nv, W):
    """chol_solve_blk(A, L, b, lane) (nv128.dense_blk's text, renamed) and chol_resolve_blk(L, b, lane) (its solves)."""
    from nablaengine.forest_engine import nv128
    txt = nv128.dense_blk(nv, W)
    assert txt.count(nv128.DENSE_FN) == 1
    txt = txt.replace("// (nv128 stage 6) chol_solve_dense, register-blocked: see forest_engine.nv128.dense_blk",
                      f"// (DK_FOREST_CHOL_BLK) chol_solve_blk: nv128.dense_blk for NV = {nv}, blocks of {W} dofs "
                      "(forest_engine.chol_blk)")
    txt = txt.replace(nv128.DENSE_FN, nv128.DENSE_FN.replace("chol_solve_dense(", "chol_solve_blk("))
    head, body = txt.split("  float y[", 1)
    tail = "#undef BFMS\n#undef BMUL\n"
    assert head.count("__device__") == 1 and body.endswith(tail)
    hdr = head[:head.index("__device__")]
    res = (hdr.replace("chol_solve_blk: nv128.dense_blk", "chol_resolve_blk: the solves of nv128.dense_blk")
           + "// inlined: k_post calls it from its SimN and its Hand::big path; as a call (as chol_resolve was in an\n"
           + "// earlier revision) ptxas 12.9 left registers live at one site holding values recomputed for the other (sm_120)\n"
           + "__device__ __forceinline__ void chol_resolve_blk(const float* L, float* b, int lane) {\n"
           + "  __syncwarp();\n  float y[" + body)
    return txt + res


ANCHOR_SEL = "#if CHOL_BCAST\n#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_bcast(A, L, b, lane, cb)\n"
ANCHOR_WIDE = "#ifndef CHOL_BCAST_WIDE\n#define CHOL_BCAST_WIDE (NV >= 36 && NV <= 48)\n#endif\n"
ANCHOR_COMPS = "  if (__any_sync(0xffffffffu, n > CB)) {  // a large component: the dense factorization\n" \
               "    CHOL_SOLVE(A, L, b, lane, cb);\n"
ANCHOR_M = ("#if CHOL_TREES\n#define CHOL_M_SOLVE(A, L, b, lane, cb) chol_solve_trees(A, L, b, lane)\n#else\n"
            "#define CHOL_M_SOLVE(A, L, b, lane, cb) CHOL_SOLVE(A, L, b, lane, cb)\n#endif\n")
ANCHOR_REUSE = ("#if NEWTON_REUSE || NEWTON_REUSE_WIDE\n    if (zsame && !refact) {  // no row changed its zone in the last "
                "step: H and its factor in s.L are the same\n      for (int i = lane; i < NV; i += 32) s.p[i] = -s.grad[i];\n"
                "      __syncwarp();\n      chol_resolve(s.L, s.p, lane, s.cbuf);\n")


def _rep(src, a, b):
    if src.count(a) != 1:
        raise RuntimeError(f"forest_engine.chol_blk: anchor found {src.count(a)} times: {a[:60]!r}")
    return src.replace(a, b)


def apply(src, m):
    """adjoint.cuda_source's text with chol_solve_blk for M and the Newton Hessian where enabled(m); else src."""
    if not enabled(m):
        return src
    nv = m.nv
    if nv <= 48:
        src = _rep(src, ANCHOR_WIDE, "#ifndef CHOL_BCAST_WIDE\n#define CHOL_BCAST_WIDE 0  // (DK_FOREST_CHOL_BLK)\n#endif\n")
    src = _rep(src, ANCHOR_SEL, functions(nv, W) + ANCHOR_SEL)
    src = _rep(src, ANCHOR_COMPS, ANCHOR_COMPS.replace("CHOL_SOLVE(A, L, b, lane, cb);",
                                                       "chol_solve_blk(A, L, b, lane);  // (DK_FOREST_CHOL_BLK)"))
    src = _rep(src, ANCHOR_M, ANCHOR_M
               + "#if !CHOL_COMPS  // (DK_FOREST_CHOL_BLK)\n#undef CHOL_H_SOLVE\n"
               "#define CHOL_H_SOLVE(A, L, b, lane, cb) chol_solve_blk(A, L, b, lane)\n#endif\n"
               "#if !CHOL_TREES  // (DK_FOREST_CHOL_BLK)\n#undef CHOL_M_SOLVE\n"
               "#define CHOL_M_SOLVE(A, L, b, lane, cb) chol_solve_blk(A, L, b, lane)\n#endif\n")
    src = _rep(src, ANCHOR_REUSE, ANCHOR_REUSE.replace(
        "#if NEWTON_REUSE || NEWTON_REUSE_WIDE\n",
        "#ifndef NEWTON_REUSE_BLK  // (DK_FOREST_CHOL_BLK)\n#define NEWTON_REUSE_BLK 1\n#endif\n"
        "#if NEWTON_REUSE_BLK && !CHOL_COMPS\n").replace(
        "chol_resolve(s.L, s.p, lane, s.cbuf);", "chol_resolve_blk(s.L, s.p, lane);"))
    return src

"""More than 64 bodies in the forest engine (capability T1: at most 128, the world included; nv128 design, nb128)."""
import numpy as np

NOTE = "  // (nb128: NB > 64)"

HEADER = (
    "typedef unsigned __int128 BMask;  // (nb128: NB > 64) a body set, bit b = body b\n"
    "__device__ __forceinline__ BMask bm_shfl(BMask x, int src) {  // (nb128) a warp shuffle of a body set\n"
    "  const unsigned long long lo = __shfl_sync(0xffffffffu, (unsigned long long)x, src);\n"
    "  const unsigned long long hi = __shfl_sync(0xffffffffu, (unsigned long long)(x >> 64), src);\n"
    "  return ((BMask)hi << 64) | lo;\n"
    "}\n"
    "__device__ __forceinline__ int bm_ffs(BMask x) {  // (nb128) the lowest set bit, 1-based; 0: none (as __ffsll)\n"
    "  const unsigned long long lo = (unsigned long long)x, hi = (unsigned long long)(x >> 64);\n"
    "  return lo ? __ffsll(lo) : (hi ? 64 + __ffsll(hi) : 0);\n"
    "}\n"
)

DECL = "DKC unsigned long long c_ancmask[NB];  // (nb64: NB > 32)\n"


def needed(m):
    return m.nbody > 64


FORWARD = (
    ("    unsigned long long am1[CPL], am2[CPL];  // (nb64: NB > 32)\n", "    BMask am1[CPL], am2[CPL];" + NOTE + "\n", 1),
    ("      unsigned long long a1 = am1[0], a2 = am2[0];  // (nb64: NB > 32)\n",
     "      BMask a1 = am1[0], a2 = am2[0];" + NOTE + "\n", 1),
    ("      unsigned long long m1 = __shfl_sync(0xffffffffu, a1, c & 31);  // (nb64: NB > 32)\n"
     "      unsigned long long m2 = __shfl_sync(0xffffffffu, a2, c & 31);\n",
     "      BMask m1 = bm_shfl(a1, c & 31);" + NOTE + "\n      BMask m2 = bm_shfl(a2, c & 31);\n", 1),
)

ADJOINT = (
    ("  const unsigned long long m1 = c_ancmask[b1], m2 = c_ancmask[b2], sup = m1 ^ m2;  // (nb64: NB > 32)",
     "  const BMask m1 = c_ancmask[b1], m2 = c_ancmask[b2], sup = m1 ^ m2;" + NOTE, 1),
    ("  unsigned long long hb = sup & ~1ull;", "  BMask hb = sup & ~(BMask)1;", 1),
)
WALK = ("    const int bi = __ffsll(hb) - 1;\n    hb &= hb - 1ull;\n", "    const int bi = bm_ffs(hb) - 1;\n    hb &= hb - 1;\n")
ADJ_JTF = ("      const unsigned long long m1 = c_ancmask[c_gbody[PG1(pair)]], m2 = c_ancmask[c_gbody[PG2(pair)]];"
           "  // (nb64: NB > 32)",
           "      const BMask m1 = c_ancmask[c_gbody[PG1(pair)]], m2 = c_ancmask[c_gbody[PG2(pair)]];" + NOTE)
SURROGATE = (
    ("  const unsigned long long m1 = c_ancmask[b1], m2 = c_ancmask[b2];  // (nb64: NB > 32)\n"
     "  unsigned long long hb = (m1 ^ m2) & ~1ull;\n",
     "  const BMask m1 = c_ancmask[b1], m2 = c_ancmask[b2];" + NOTE + "\n  BMask hb = (m1 ^ m2) & ~(BMask)1;\n", 1),
    ("      const unsigned long long m1 = c_ancmask[c1], m2 = c_ancmask[c2];  // (nb64: NB > 32)\n",
     "      const BMask m1 = c_ancmask[c1], m2 = c_ancmask[c2];" + NOTE + "\n", 1),
)
RED = ("    for (int h = 0; h < 2; h++) {  // (nb64: NB > 32) pass h: the bodies lane + 32 h (the whole warp: shuffles)\n",
       "    for (int h = 0; h < (NB + 31) / 32; h++) {  // (nb64, nb128: NB > 64) pass h: the bodies lane + 32 h (the "
       "whole warp: shuffles)\n")
ASSERT = ('static_assert(NB <= 64, "the contact reduction: two passes, the bodies lane and lane + 32 (nb64)");',
          'static_assert(NB <= 128, "the contact reduction: (NB + 31) / 32 passes, the bodies lane + 32 h (nb128)");')

KIN = (
    ("  float o[2][6] = {};", "  float o[(NB + 31) / 32][6] = {};  /* (nb128: NB > 64) */", 1),
    ("  for (int h = 0; h < 2; h++) {\n    const int b = lane + 32 * h;\n",
     "  for (int h = 0; h < (NB + 31) / 32; h++) {  // (nb128)\n    const int b = lane + 32 * h;\n", 1),
    ("  for (int h = 0; h < 2; h++)\n    if (lane + 32 * h < NB)\n",
     "  for (int h = 0; h < (NB + 31) / 32; h++)  // (nb128)\n    if (lane + 32 * h < NB)\n", 1),
)


def ancmask_packed(m, mask):
    """c_ancmask packed for host.loader's "u" kind (float32 16-bit halves) with four 32-bit words per body (low word
    first: the device's unsigned __int128), and its layout entry."""
    words = np.asarray([(int(x) >> (32 * k)) & 0xFFFFFFFF for x in mask for k in range(4)], np.uint64)
    halves = np.stack([words & np.uint64(0xFFFF), words >> np.uint64(16)], 1).ravel()
    return halves.astype(np.float32), ("c_ancmask", "u", "nb", 4)


def _apply(src, hooks):
    for a, b, n in hooks:
        if src.count(a) != n:
            raise RuntimeError(f"nb128 hook anchor found {src.count(a)} times, expected {n}: {a[:70]!r}")
        src = src.replace(a, b)
    return src


def _one(src, a, b, what):
    if src.count(a) != 1:
        raise RuntimeError(f"nb128: {what} found {src.count(a)} times, expected 1")
    return src.replace(a, b)


def apply(src, m, part="engine"):
    """src (part "engine": adjoint.cuda_source's text after nb64 and nv128; "kinematics": the kinematics text after
    nb64's) with the edits for more than 64 bodies; unchanged for at most 64."""
    if not needed(m):
        return src
    assert m.nbody <= 128, "nb128: at most 128 bodies (four per lane)"
    if part == "kinematics":
        return _apply(src, KIN)
    src = _one(src, DECL, HEADER + "DKC BMask c_ancmask[NB];" + NOTE + "\n", "c_ancmask's declaration")
    n = src.count("support_mask(unsigned long long x) {")
    if n != 1:
        raise RuntimeError(f"nb128: support_mask found {n} times, expected 1")
    src = src.replace("support_mask(unsigned long long x) {", "support_mask(BMask x) {")
    if "#if DOF_BODY_ROOT6\n  return (unsigned long long)(x >> 2) << 6" in src:
        src = src.replace("#if DOF_BODY_ROOT6\n  return (unsigned long long)(x >> 2) << 6",
                          "#if 0  // (nb128: the closed form needs NV <= 64)\n  return (unsigned long long)(x >> 2) << 6", 1)
    src = _apply(src, FORWARD)
    if "__global__ void k_adjoint(" in src:
        if "adj_sur_hessian(" in src:
            src = _apply(src, SURROGATE)
        src = _apply(src, ADJOINT)
        if src.count(WALK[0]) != (2 if "adj_sur_hessian(" in src else 1):
            raise RuntimeError(f"nb128: the support walk found {src.count(WALK[0])} times")
        src = src.replace(WALK[0], WALK[1])
        if ADJ_JTF[0] in src:
            src = _one(src, ADJ_JTF[0], ADJ_JTF[1], "adj_jtf's masks")
        src = _one(src, ASSERT[0], ASSERT[1], "the reduction's assert")
        src = _one(src, RED[0], RED[1], "the reduction's pass loop")
    if "unsigned long long m1 = c_ancmask" in src or "__ffsll(hb)" in src:
        raise RuntimeError("nb128: a 64-bit body mask left in the text")
    return src

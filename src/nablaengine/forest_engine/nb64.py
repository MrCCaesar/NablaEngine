"""More than 32 bodies in the forest engine (capability T1: at most 64, the world included)."""
import numpy as np

STALE_ASSERT = 'static_assert(NB <= 31, "the contact reduction: lanes 0 and 31 take two of the root rotation dofs");'
NOTE = "  // (nb64: NB > 32)"


def needed(m):
    return m.nbody > 32


def ancmask_packed(m, mask):
    """(c_ancmask packed for host.loader's "u" kind as float32, its layout entry): one 32-bit word per body, two
    (low, high: the device's unsigned long long) for more than 32 bodies."""
    if m.nbody <= 32:
        mask = np.asarray(mask, np.int64)
        return np.stack([mask & 0xFFFF, mask >> 16], 1).ravel().astype(np.float32), ("c_ancmask", "u", "nb", 1)
    mask = np.asarray([int(x) & 0xFFFFFFFFFFFFFFFF for x in mask], np.uint64)
    words = np.stack([mask & np.uint64(0xFFFFFFFF), mask >> np.uint64(32)], 1).ravel()
    halves = np.stack([words & np.uint64(0xFFFF), words >> np.uint64(16)], 1).ravel()
    return halves.astype(np.float32), ("c_ancmask", "u", "nb", 2)


FORWARD = (
    ("DKC unsigned int c_ancmask[NB];\n", "DKC unsigned long long c_ancmask[NB];" + NOTE + "\n", 1),
    ("    if (d == L) {\n      float Qp[4], Xp[3], Wp[3], Vp[3], ALp[3], Ap[3], Qb[4], Xb[3], Wb[3], Vb[3], ALb[3], Ab[3];\n",
     "    for (int b = lane; b < NB; b += 32) if (c_depth[b] == L) {  // (nb64: NB > 32, bodies lane and lane + 32)\n"
     "      const int p = c_parent[b];\n"
     "      float Qp[4], Xp[3], Wp[3], Vp[3], ALp[3], Ap[3], Qb[4], Xb[3], Wb[3], Vb[3], ALb[3], Ab[3];\n", 1),
    ("  const int rt = (b >= 1 && b < NB) ? c_root[b] : 1;  // the body's tree root, whose origin is the tree's point O\n"
     "  const float O[3] = {s.X[rt][0], s.X[rt][1], s.X[rt][2]};\n"
     "  if (b >= 1 && b < NB) {\n",
     "  for (int b = lane; b < NB; b += 32) if (b >= 1) {" + NOTE + "\n"
     "    const int rt = c_root[b];  // the body's tree root, whose origin is the tree's point O\n"
     "    const float O[3] = {s.X[rt][0], s.X[rt][1], s.X[rt][2]};\n", 1),
    ("    if (d == L && c_nchild[b] > 0) {\n      int nch = c_nchild[b];\n",
     "    for (int b = lane; b < NB; b += 32) if (c_depth[b] == L && c_nchild[b] > 0) {" + NOTE + "\n"
     "      int nch = c_nchild[b];\n", 1),
    ("  if (b >= 1 && b < NB) torques(b, s.Q[b], s.F[b], s.N[b], s.qsm);  // qsm holds the bias for now\n"
     "  float sw[3], sv[3], kw[3], kv[3], n[3], f[3];\n"
     "  const float q1[4] = {s.Q[rt][0], s.Q[rt][1], s.Q[rt][2], s.Q[rt][3]};  // the root's rotation (its dofs' subspace)\n"
     "  const bool rfree = b >= 1 && b < NB && JT_OF(rt) == JFREE;\n"
     "  if (b >= 1 && b < NB && JT_OF(b) != JFREE) {\n",
     "  for (int b = lane; b < NB; b += 32) if (b >= 1) torques(b, s.Q[b], s.F[b], s.N[b], s.qsm);  // (nb64)\n"
     "  for (int b = lane; b < NB; b += 32) if (b >= 1) {  // (nb64: NB > 32, the rows of bodies lane and lane + 32)\n"
     "  const int rt = c_root[b];\n"
     "  const float O[3] = {s.X[rt][0], s.X[rt][1], s.X[rt][2]};\n"
     "  float sw[3], sv[3], kw[3], kv[3], n[3], f[3];\n"
     "  const float q1[4] = {s.Q[rt][0], s.Q[rt][1], s.Q[rt][2], s.Q[rt][3]};  // the root's rotation (its dofs' subspace)\n"
     "  const bool rfree = JT_OF(rt) == JFREE;\n"
     "  if (JT_OF(b) != JFREE) {\n", 1),
    ("  if (b >= 1 && b < NB && JT_OF(b) == JFREE) {\n    float J1[10];\n",
     "  if (JT_OF(b) == JFREE) {\n    float J1[10];\n", 1),
    ("        s.M[TRI(r0 + j, r0 + i)] = dot3(kw, n) + dot3(kv, f) + (i == j ? c_armature[r0 + j] : 0.f);\n"
     "      }\n    }\n  }\n  __syncwarp();\n",
     "        s.M[TRI(r0 + j, r0 + i)] = dot3(kw, n) + dot3(kv, f) + (i == j ? c_armature[r0 + j] : 0.f);\n"
     "      }\n    }\n  }\n  }  // (nb64)\n  __syncwarp();\n", 1),
    ("__device__ __forceinline__ unsigned long long support_mask(unsigned x) {",
     "__device__ __forceinline__ unsigned long long support_mask(unsigned long long x) {" + NOTE, 1),
    ("    unsigned am1[CPL], am2[CPL];\n", "    unsigned long long am1[CPL], am2[CPL];" + NOTE + "\n", 1),
    ("      unsigned a1 = am1[0], a2 = am2[0];\n", "      unsigned long long a1 = am1[0], a2 = am2[0];" + NOTE + "\n", 1),
    ("      unsigned m1 = __shfl_sync(0xffffffffu, a1, c & 31);\n      unsigned m2 = __shfl_sync(0xffffffffu, a2, c & 31);\n",
     "      unsigned long long m1 = __shfl_sync(0xffffffffu, a1, c & 31);" + NOTE + "\n"
     "      unsigned long long m2 = __shfl_sync(0xffffffffu, a2, c & 31);\n", 1),
)

ADJOINT = (
    ("    if (d == L) {\n      const float *Qp = s.Q[p], *Xp = s.X[p];\n",
     "    for (int b = lane; b < NB; b += 32) if (c_depth[b] == L) {" + NOTE + "\n"
     "      const int p = c_parent[b];\n      const float *Qp = s.Q[p], *Xp = s.X[p];\n", 1),
    ("  float r[3];\n  for (int k = 0; k < 3; k++) r[k] = b < NB ? s.X[b][k] - s.X[p][k] : 0.f;\n"
     "  const float sa = s.at[i], sl = lam[i], sy = ny[i];\n  for (int L = 1; L <= MAXD; L++) {\n    if (d == L) {\n",
     "  for (int L = 1; L <= MAXD; L++) {\n"
     "    for (int b = lane; b < NB; b += 32) if (c_depth[b] == L) {" + NOTE + "\n"
     "      const int p = c_parent[b], i = c_dadr[b];\n"
     "      float r[3];\n      for (int k = 0; k < 3; k++) r[k] = s.X[b][k] - s.X[p][k];\n"
     "      const float sa = s.at[i], sl = lam[i], sy = ny[i];\n", 1),
    ("  if (b >= 1 && b < NB) {\n    float ip[3];\n    body_ipos(b, s.com, ip);\n    body_local_vjp(",
     "  for (int b = lane; b < NB; b += 32) if (b >= 1) {" + NOTE + "\n    float ip[3];\n    body_ipos(b, s.com, ip);\n"
     "    body_local_vjp(", 1),
    ("    if (d == L) {\n      float bW[3], bAL[3], bA[3], bAL2[3], bA2[3], bWl[3], bVl[3], bWy[3], bVy[3], F[3], N[3], t[3], u[3];\n",
     "    for (int b = lane; b < NB; b += 32) if (c_depth[b] == L) {" + NOTE + "\n      const int p = c_parent[b];\n"
     "      float bW[3], bAL[3], bA[3], bAL2[3], bA2[3], bWl[3], bVl[3], bWy[3], bVy[3], F[3], N[3], t[3], u[3];\n", 1),
    ("  const unsigned m1 = c_ancmask[b1], m2 = c_ancmask[b2], sup = m1 ^ m2;",
     "  const unsigned long long m1 = c_ancmask[b1], m2 = c_ancmask[b2], sup = m1 ^ m2;" + NOTE, 1),
    ("  unsigned hb = sup & ~1u;", "  unsigned long long hb = sup & ~1ull;", 1),
    ("    const int bi = __ffs(hb) - 1;\n    hb &= hb - 1u;\n", "    const int bi = __ffsll(hb) - 1;\n    hb &= hb - 1ull;\n", 1),
    ("      const unsigned m1 = c_ancmask[c_gbody[PG1(pair)]], m2 = c_ancmask[c_gbody[PG2(pair)]];",
     "      const unsigned long long m1 = c_ancmask[c_gbody[PG1(pair)]], m2 = c_ancmask[c_gbody[PG2(pair)]];" + NOTE, 1),
)

SURROGATE = (
    ("  const unsigned m1 = c_ancmask[b1], m2 = c_ancmask[b2];\n  unsigned hb = (m1 ^ m2) & ~1u;\n#pragma unroll 1\n"
     "  while (hb) {\n    const int bi = __ffs(hb) - 1;\n    hb &= hb - 1u;\n",
     "  const unsigned long long m1 = c_ancmask[b1], m2 = c_ancmask[b2];  // (nb64: NB > 32)\n"
     "  unsigned long long hb = (m1 ^ m2) & ~1ull;\n#pragma unroll 1\n"
     "  while (hb) {\n    const int bi = __ffsll(hb) - 1;\n    hb &= hb - 1ull;\n", 1),
    ("      const unsigned m1 = c_ancmask[c1], m2 = c_ancmask[c2];\n",
     "      const unsigned long long m1 = c_ancmask[c1], m2 = c_ancmask[c2];  // (nb64: NB > 32)\n", 1),
)

from nablaengine.g1_engine.model_hooks import ADJ_DEBUG_NB64 as ADJ_DEBUG

RED_SETUP = "  const bool body = lane >= 1 && lane < NB;\n"
RED_SETUP_END = "  int ntri = 0, nquad = 0;"
RED_START = "    float f[3] = {0.f, 0.f, 0.f}, t[3] = {0.f, 0.f, 0.f}, pt[3] = {0.f, 0.f, 0.f}, pv[3] = {0.f, 0.f, 0.f};\n"
RED_END = "    __syncwarp();\n    APROF(APROF_CONTACT_REDUCE);\n"


def _indent(s, n=2):
    return "".join((" " * n + ln if ln.strip() else ln) for ln in s.splitlines(True))


def reduction(src):
    """adj_contacts' body reduction as two passes over the batch (pass h: the bodies lane + 32 h): the lane's body
    setup (body, rb, jt, di, zl) moves from before the batch loop into each pass."""
    for a, n in ((RED_SETUP, 1), (RED_START, 1), (RED_END, 1)):
        if src.count(a) != n:
            raise RuntimeError(f"nb64 reduction anchor found {src.count(a)} times, expected {n}: {a[:70]!r}")
    i = src.index(RED_SETUP)
    j = src.index(RED_SETUP_END, i)
    setup = src[i:j]
    k = src.index(RED_START, j)
    e = src.index(RED_END, k)
    body = src[k:e]
    s2 = setup.replace(RED_SETUP, "  const int bl = lane + 32 * h;\n  const bool body = bl >= 1 && bl < NB;\n", 1)
    old = "  const int rb = body ? lane : 1,"
    if s2.count(old) != 1:
        raise RuntimeError("nb64 reduction: the lane's body line not found")
    s2 = s2.replace(old, "  const int rb = body ? bl : 1,", 1)
    new = ("    for (int h = 0; h < 2; h++) {  // (nb64: NB > 32) pass h: the bodies lane + 32 h (the whole warp: shuffles)\n"
           + _indent(s2, 4) + _indent(body, 2) + "    }\n")
    return src[:i] + src[j:k] + new + src[e:]


def _apply(src, hooks):
    for a, b, n in hooks:
        if src.count(a) != n:
            raise RuntimeError(f"nb64 hook anchor found {src.count(a)} times, expected {n}: {a[:70]!r}")
        src = src.replace(a, b)
    return src


def apply(src, m, part="engine"):
    """src (part "engine": adjoint.cuda_source's text after every other edit; "kinematics": the kinematics text before
    the others) with the edits for more than 32 bodies; unchanged for at most 31 bodies (32: the adjoint's assert)."""
    if part == "kinematics":
        if not needed(m):
            return src
        from nablaengine.g1_engine import model_hooks as g1_hooks
        return _apply(src, g1_hooks.kin_nb64_hooks())
    if m.nbody < 32:
        return src
    if m.nbody == 32:
        return _apply(src, ((STALE_ASSERT, 'static_assert(NB <= 32, "the contact reduction: one body per lane '
                                           '(nb64: lane 0 the world)");', 1),))
    if "adj_sur_hessian(" in src:
        src = _apply(src, SURROGATE)
    src = _apply(src, ((STALE_ASSERT, 'static_assert(NB <= 64, "the contact reduction: two passes, the bodies lane and '
                                      'lane + 32 (nb64)");', 1),) + FORWARD + ADJOINT)
    if "#ifdef DK_ADJ_DEBUG" in src and "__device__ void adj_residual(" in src:
        src = _apply(src, ADJ_DEBUG)
    return reduction(src)

"""More than 64 dofs in the forest engine (capability H4: at most 128; the dense factor: a dense correctness
baseline, at most 64 bodies)."""
import re

NOTE = "  // (nv128: NV > 64)"


def needed(m):
    return m.nv > 64


DEVICE = r"""
#define NDL ((NV + 31) / 32)  // (nv128: NV > 64) dofs per lane: lane l holds the dofs l, l + 32, ..
// (nv128: NV > 64) chol_solve_dense: CHOL_SOLVE's contract (A, L packed lower triangles TRI(i, j) in shared memory, L
// may alias A; b, a shared NV vector, solved in place; L keeps the true diagonal) for any NV. A left-looking (Crout)
// factorization in place in L: column k's pivot on every lane from row k (broadcast loads), lane l the entries of rows
// l, l + 32, .. below it from the finished columns (dot products over shared memory), two __syncwarp() per column;
// then the forward and backward solves with NDL entries per lane in registers (column k broadcast by __shfl_sync).
// Explicit roundings: k_substep, k_pre and k_post agree bitwise. The design's stage-1 baseline (~NV^3 / 192 FMAs per
// lane, NV serial steps): correct, not fast
#define DFMS(a, b, c) __fmaf_rn(-(a), (b), (c))
#define DMUL(a, b) __fmul_rn((a), (b))
__device__ __noinline__ void chol_solve_dense(const float* A, float* L, float* b, int lane) {
  if (A != L) {
    __syncwarp();
    for (int e = lane; e < NTRI; e += 32) L[e] = A[e];
  }
  __syncwarp();
  for (int k = 0; k < NV; k++) {
    const float* Lk = L + TRI(k, 0);
    float s = Lk[k];
    for (int j = 0; j < k; j++) s = DFMS(Lk[j], Lk[j], s);
    const float dkk = fmaxf(s, 1e-30f);
    float inv;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(dkk));
    inv = DMUL(inv, DFMS(DMUL(DMUL(0.5f, dkk), inv), inv, 1.5f));  // one Newton step
    for (int i = k + 1 + lane; i < NV; i += 32) {
      float* Li = L + TRI(i, 0);
      float e = Li[k];
      for (int j = 0; j < k; j++) e = DFMS(Li[j], Lk[j], e);
      Li[k] = DMUL(e, inv);
    }
    __syncwarp();  // (every lane has read L_kk)
    if (lane == (k & 31)) L[TRI(k, k)] = DMUL(dkk, inv);
    __syncwarp();
  }
  float y[NDL];
#pragma unroll
  for (int t = 0; t < NDL; t++) {
    const int r = lane + 32 * t;
    y[t] = r < NV ? b[r] : 0.f;
  }
  for (int k = 0; k < NV; k++) {  // forward: L y = b, column-wise
    float v = y[0];
#pragma unroll
    for (int t = 1; t < NDL; t++)
      if ((k >> 5) == t) v = y[t];
    const float yk = __fdiv_rn(__shfl_sync(0xffffffffu, v, k & 31), L[TRI(k, k)]);
#pragma unroll
    for (int t = 0; t < NDL; t++) {
      const int r = lane + 32 * t;
      if (r == k) y[t] = yk;
      else if (r > k && r < NV) y[t] = DFMS(L[TRI(r, k)], yk, y[t]);
    }
  }
  for (int k = NV - 1; k >= 0; k--) {  // backward: L' x = y (row k of L, contiguous)
    float v = y[0];
#pragma unroll
    for (int t = 1; t < NDL; t++)
      if ((k >> 5) == t) v = y[t];
    const float xk = __fdiv_rn(__shfl_sync(0xffffffffu, v, k & 31), L[TRI(k, k)]);
#pragma unroll
    for (int t = 0; t < NDL; t++) {
      const int r = lane + 32 * t;
      if (r == k) y[t] = xk;
      else if (r < k) y[t] = DFMS(L[TRI(k, r)], xk, y[t]);
    }
  }
  __syncwarp();
#pragma unroll
  for (int t = 0; t < NDL; t++) {
    const int r = lane + 32 * t;
    if (r < NV) b[r] = y[t];
  }
  __syncwarp();
}
#undef DFMS
#undef DMUL
"""

DENSE_FN = "__device__ __noinline__ void chol_solve_dense(const float* A, float* L, float* b, int lane) {\n"


def dchol(nv):
    """$DK_NV128_DCHOL at source generation (more than 64 dofs): the form of chol_solve_dense, the dense Newton
    factor (and the dense CHOL_SOLVE / surrogate H_mu uses): blk (default: dense_blk, block columns of
    $DK_NV128_DCHOL_W dofs, default 8 up to 72 dofs and 12 beyond) or crout (the dense factor's left-looking loop). Both do the
    same operations in the same order: the factor and the solution are bitwise equal. Microbenchmark (V100, cycles per
    factor + solve, W = 8 / 12 / 16): nv 65 42k / 41k / 48k, 77 46k / 45k / 43k, 89 57k / 53k / 51k, 101 71k / 67k /
    65k, 113 87k / 81k / 84k, 128 106k / 100k / 110k; in the engine (k_post, forward call) S1 107 / 97 ms (W 8 / 12),
    S3 19.1 / 21.2 ms (W 8 / 16)."""
    import os
    raw = os.environ.get("DK_NV128_DCHOL", "") or "blk"
    if raw not in ("blk", "crout"):
        raise ValueError(f"DK_NV128_DCHOL={raw!r}: blk or crout")
    w = int(os.environ.get("DK_NV128_DCHOL_W", "") or (8 if nv <= 72 else 12))
    if not 1 <= w <= 32:
        raise ValueError(f"DK_NV128_DCHOL_W={w}: 1 .. 32")
    return raw, w


def dense_blk(nv, W):
    """chol_solve_dense as a register-blocked factor (the register-blocked factor) with chol_solve_dense's contract and its operations in
    the same order (bitwise the same L and x). Block columns of W dofs, left-looking: the rows j0 .. NV - 1 of block
    column [j0, j0 + W) in registers (lane l: rows j0 + l + 32 t, a<t>_<c>: column j0 + c), first the update by the
    finished columns 0 .. j0 - 1 (a loop over p: the lane's L[i][p], the broadcast L[j0 + c][p], one FMA per register,
    p ascending as the Crout dot products), then right-looking inside the block with shuffles (the pivot from lane c,
    the scaled column entry of row j0 + c' from lane c'), then the block stored. Every entry (i, k) gets
    e = A_ik - sum_j L_ij L_kj with j ascending and the same FMA operands as the Crout form, the pivot formula is
    chol_solve_dense's. The two triangular solves form y / L_kk as RN(1 / L_kk) refined by one residual step
    (Markstein: the correctly rounded quotient, = __fdiv_rn barring over- / underflow), the reciprocals before the
    chains, the pivots by 32-row segments (the pivot's register a compile-time index). Microbenchmark (V100, dense SPD,
    4096 worlds; cycles per factor + solve): nv 113 392k -> 81k (W = 12), nv 65 153k -> 42k (W = 8)."""
    ndl = (nv + 31) // 32
    F = "0xffffffffu"
    o = ["// (nv128 stage 6) chol_solve_dense, register-blocked: see forest_engine.nv128.dense_blk",
         "#define BFMS(a, b, c) __fmaf_rn(-(a), (b), (c))", "#define BMUL(a, b) __fmul_rn((a), (b))",
         DENSE_FN.rstrip("\n"),
         "  if (A != L) {\n    __syncwarp();\n    for (int e = lane; e < NTRI; e += 32) L[e] = A[e];\n  }",
         "  __syncwarp();"]
    for J in range((nv + W - 1) // W):
        j0, nj = W * J, min(W, nv - W * J)
        NT = (nv - j0 + 31) // 32
        o.append(f"  {{  // block column {J}: columns {j0} .. {j0 + nj - 1}, rows {j0} .. NV - 1")
        for t in range(NT):
            o.append(f"    const int i{t} = {j0 + 32 * t} + lane, t{t} = TRI(min(i{t}, NV - 1), 0);")
        regs = [f"a{t}_{c} = {f'i{t} < NV' if t else f'lane >= {c}'} ? L[t{t} + {j0 + c}] : 0.f"
                for t in range(NT) for c in range(nj)]
        o.append("    float " + ", ".join(regs) + ";")
        if j0:
            o.append("#pragma unroll 2")
            o.append(f"    for (int p = 0; p < {j0}; p++) {{  // the finished columns")
            o.append("      " + " ".join(f"const float x{t} = L[t{t} + p];" for t in range(NT)))
            for c in range(nj):
                o.append(f"      {{ const float l = L[{(j0 + c) * (j0 + c + 1) // 2} + p]; "
                         + " ".join(f"a{t}_{c} = BFMS(x{t}, l, a{t}_{c});" for t in range(NT)) + " }")
            o.append("    }")
        for c in range(nj):
            o.append(f"    {{ const float dkk = fmaxf(__shfl_sync({F}, a0_{c}, {c}), 1e-30f);  // pivot {j0 + c}")
            o.append("      float inv;")
            o.append('      asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(dkk));')
            o.append("      inv = BMUL(inv, BFMS(BMUL(BMUL(0.5f, dkk), inv), inv, 1.5f));  // one Newton step")
            o.append("      " + " ".join(f"const float x{t} = BMUL(a{t}_{c}, inv);" for t in range(NT)))
            o.append(f"      a0_{c} = lane == {c} ? BMUL(dkk, inv) : x0;"
                     + "".join(f" a{t}_{c} = x{t};" for t in range(1, NT)))
            for c2 in range(c + 1, nj):
                o.append(f"      {{ const float l = __shfl_sync({F}, x0, {c2}); "
                         + " ".join(f"a{t}_{c2} = BFMS(x{t}, l, a{t}_{c2});" for t in range(NT)) + " }")
            o.append("    }")
        for t in range(NT):
            for c in range(nj):
                o.append(f"    if ({f'i{t} < NV' if t else f'lane >= {c} && i0 < NV'}) L[t{t} + {j0 + c}] = a{t}_{c};")
        o.append("    __syncwarp();")
        o.append("  }")
    o += [f"  float y[{ndl}], rc[{ndl}];", f"  int rb[{ndl}];  // the lane's rows' starts TRI(r, 0) (clamped)",
          "#pragma unroll", f"  for (int t = 0; t < {ndl}; t++) {{",
          "    const int r = lane + 32 * t, rc_ = min(r, NV - 1);", "    y[t] = r < NV ? b[r] : 0.f;",
          "    rb[t] = TRI(rc_, 0);", "    rc[t] = __frcp_rn(L[rb[t] + rc_]);", "  }"]
    for s in range(ndl):
        o.append(f"  for (int k = {32 * s}; k < {min(32 * s + 32, nv)}; k++) {{  // forward: L y = b, pivots {32 * s} ..")
        o.append(f"    const float a = __shfl_sync({F}, y[{s}], k & 31), r = __shfl_sync({F}, rc[{s}], k & 31), "
                 "d = L[TRI(k, k)];")
        o.append("    const float q = BMUL(a, r), yk = __fmaf_rn(r, __fmaf_rn(-d, q, a), q);  // (= a / d)")
        for t in range(s, ndl):
            if t == s:
                o.append(f"    {{ const float u = BFMS(L[rb[{t}] + min(k, lane + {32 * t})], yk, y[{t}]);\n"
                         f"      y[{t}] = lane + {32 * t} == k ? yk : (lane + {32 * t} > k && lane + {32 * t} < NV ? u : y[{t}]); }}")
            else:
                o.append(f"    {{ const float u = BFMS(L[rb[{t}] + k], yk, y[{t}]);\n"
                         f"      y[{t}] = lane + {32 * t} < NV ? u : y[{t}]; }}")
        o.append("  }")
    for s in reversed(range(ndl)):
        o.append(f"  for (int k = {min(32 * s + 31, nv - 1)}; k >= {32 * s}; k--) {{  // backward: L' x = y, pivots .. {32 * s}")
        o.append(f"    const float a = __shfl_sync({F}, y[{s}], k & 31), r = __shfl_sync({F}, rc[{s}], k & 31), "
                 "d = L[TRI(k, k)];")
        o.append("    const float q = BMUL(a, r), xk = __fmaf_rn(r, __fmaf_rn(-d, q, a), q);")
        o.append("    const int tk = TRI(k, 0);")
        for t in range(s + 1):
            if t == s:
                o.append(f"    {{ const float u = BFMS(L[tk + min(k, lane + {32 * t})], xk, y[{t}]);\n"
                         f"      y[{t}] = lane + {32 * t} == k ? xk : (lane + {32 * t} < k ? u : y[{t}]); }}")
            else:
                o.append(f"    y[{t}] = BFMS(L[tk + lane + {32 * t}], xk, y[{t}]);")
        o.append("  }")
    o += ["  __syncwarp();", "#pragma unroll", f"  for (int t = 0; t < {ndl}; t++) {{",
          "    const int r = lane + 32 * t;", "    if (r < NV) b[r] = y[t];", "  }", "  __syncwarp();", "}",
          "#undef BFMS", "#undef BMUL"]
    return "\n".join(o) + "\n"


def _dense_swap(src, nv):
    form, w = dchol(nv)
    if form != "blk":
        return src
    if src.count(DENSE_FN) != 1:
        raise RuntimeError("nv128: chol_solve_dense not found once")
    i = src.index(DENSE_FN)
    end = "\n#undef DFMS\n#undef DMUL\n"
    j = src.index(end, i)
    assert src[j - 1] == "}"
    return (src[:i] + "#undef DFMS\n#undef DMUL\n" + dense_blk(nv, w) + "#define DFMS(a, b, c) __fmaf_rn(-(a), (b), (c))\n"
            "#define DMUL(a, b) __fmul_rn((a), (b))" + src[j:])


COPY_DEV = r"""// (nv128 stage 6) L = M (NTRI entries; entry e by lane e mod 32, as the plain loop) with 8 loads in flight per lane:
// M is in Hand (global memory) in the split forward, and the loop's one load at a time made the copy 0.5M cycles per
// world-substep of S1 (nv 113)
__device__ __forceinline__ void nv128_copy(float* L, const float* M, int lane) {
  for (int e0 = 0; e0 < NTRI; e0 += 256) {
    float v[8];
#pragma unroll
    for (int u = 0; u < 8; u++) v[u] = e0 + 32 * u + lane < NTRI ? M[e0 + 32 * u + lane] : 0.f;
#pragma unroll
    for (int u = 0; u < 8; u++)
      if (e0 + 32 * u + lane < NTRI) L[e0 + 32 * u + lane] = v[u];
  }
}
"""
COPY_SITES = (
    ("  for (int e = lane; e < NTRI; e += 32) s.L[e] = s.M[e];  // (nv128: L = M, entry e by lane e mod 32, unstaged)\n",
     "  nv128_copy(s.L, s.M, lane);  // (nv128: L = M, entry e by lane e mod 32; stage 6: loads in flight)\n"),
    ("  for (int e = lane; e < NTRI; e += 32) s.L[e] = s.M[e];\n  __syncwarp();\n",
     "  nv128_copy(s.L, s.M, lane);  // (nv128 stage 6)\n  __syncwarp();\n"),
    ("  for (int q = lane; q < NNZ_LH; q += 32) s.L.v[q] = s.M[nvc_tri[q]];  // (nv128 stage 5: L = M on the closure)\n",
     "  for (int q0 = 0; q0 < NNZ_LH; q0 += 256) {  // (nv128 stage 5: L = M on the closure; stage 6: loads in flight)\n"
     "    float v_[8];\n#pragma unroll\n"
     "    for (int u = 0; u < 8; u++) v_[u] = q0 + 32 * u + lane < NNZ_LH ? s.M[nvc_tri[q0 + 32 * u + lane]] : 0.f;\n"
     "#pragma unroll\n    for (int u = 0; u < 8; u++)\n"
     "      if (q0 + 32 * u + lane < NNZ_LH) s.L.v[q0 + 32 * u + lane] = v_[u];\n  }\n"),
)
NDL_LINE = "#define NDL ((NV + 31) / 32)  // (nv128: NV > 64) dofs per lane: lane l holds the dofs l, l + 32, ..\n"


def _copy_swap(src):
    if src.count(NDL_LINE) != 1:
        raise RuntimeError("nv128: NDL's definition not found once")
    src = src.replace(NDL_LINE, NDL_LINE + COPY_DEV)
    for a, b in COPY_SITES:
        src = src.replace(a, b)
    return src


def tri_solve(ndl):
    """tri_solve_packed with ndl entries per lane, written as the two-entry form (for its two entries the same text:
    scalar registers y0 .., rows clamped to a valid one, selects, #pragma unroll 5), the entries 2, 3 likewise. (A form
    with arrays y[NDL] matched it bitwise in a standalone test, but forced onto the nv 59 control scene it gave wrong
    position gradients in 3 of 256 worlds inside k_adjoint; not understood.)"""
    T = range(ndl)
    rows = "\n".join(f"  const int r{t} = min(lane + {32 * t}, NV - 1);  // (clamped to a valid row)" if t else
                     "  const int r0 = lane;" for t in T)
    has = "\n".join(f"  const bool has{t} = lane + {32 * t} < NV;" for t in T if t)
    starts = "  const int " + ", ".join(f"t{t} = TRI(r{t}, 0)" for t in T) + ";  // the rows' starts"
    init = "  float " + ", ".join("y0 = b[r0]" if t == 0 else f"y{t} = has{t} ? b[r{t}] : 0.f" for t in T) + ";"
    sel = " : ".join([f"k < {32 * (u + 1)} ? y{u}" for u in range(ndl - 1)] + [f"y{ndl - 1}"])

    def step(var, cmp):
        return "\n".join(f"    y{t} = r{t} == k ? {var} : ({'' if t == 0 else f'has{t} && '}r{t} {cmp} k ? y{t} - l{t} * {var} : "
                         f"y{t});" for t in T)
    lf = ", ".join(f"l{t} = L[t{t} + min(k, r{t})]" for t in T)
    lb = ", ".join(f"l{t} = L[tk + min(r{t}, k)]" for t in T)
    store = "\n".join("  b[r0] = y0;" if t == 0 else f"  if (has{t}) b[r{t}] = y{t};" for t in T)
    return f"""__device__ __noinline__ void tri_solve_packed(const float* L, float* b, int lane) {{
  static_assert(NV <= {32 * ndl}, "{ndl} entries per lane (nv128)");
{rows}
{has}
{starts}
  __syncwarp();
{init}
#pragma unroll 5
  for (int k = 0; k < NV; k++) {{  // forward: L y = b
    const float d = L[TRI(k, k)], {lf};
    const float yk = div_fast(__shfl_sync(0xffffffffu, {sel}, k & 31), d, rcp_step(d));
{step("yk", ">")}
  }}
#pragma unroll 5
  for (int k = NV - 1; k >= 0; k--) {{  // backward: L' x = y
    const int tk = TRI(k, 0);
    const float d = L[tk + k], {lb};
    const float xk = div_fast(__shfl_sync(0xffffffffu, {sel}, k & 31), d, rcp_step(d));
{step("xk", "<")}
  }}
  __syncwarp();
{store}
  __syncwarp();
}}
#endif
"""


TRI_START = ('__device__ __noinline__ void tri_solve_packed(const float* L, float* b, int lane) {\n'
             '  static_assert(NV <= 64, "two entries per lane");\n')
TRI_END = "  if (has1) b[r1] = y1;\n  __syncwarp();\n}\n#endif\n"

SUPPORT_RE = re.compile(r"__device__ __forceinline__ unsigned long long support_mask\((unsigned(?: long long)? x)\) \{"
                        r"[^\n]*\n#if DOF_BODY_ROOT6\n.*?\n#endif\n\}\n", re.S)

FORWARD = (
    ("#ifndef NEWTON_REUSE\n#define NEWTON_REUSE (NV > 32 && NV <= 35)\n#endif\n",
     "#ifndef NEWTON_REUSE\n#define NEWTON_REUSE (NV > 32 && NV <= 35)\n#endif\n"
     "struct DofMask { unsigned w[NDL]; };  // (nv128) dof bits: bit i of word w is dof 32 w + i\n", 1),
    ("  unsigned long long con_mask[MAXCON];  // the support as dof bits (bit i: dof i; con_dof[c][k] is the k-th set "
     "bit)\n",
     "  DofMask con_mask[MAXCON];  // the support as dof bits (nv128: NDL words; con_dof[c][k] is the k-th set bit)\n", 1),
    ("  unsigned long long con_mask[MAXCON];\n", "  DofMask con_mask[MAXCON];" + NOTE + "\n", 1),
    ("  const unsigned long long* con_mask;\n", "  const DofMask* con_mask;" + NOTE + "\n", 1),
    ("  unsigned long long* con_mask;\n", "  DofMask* con_mask;" + NOTE + "\n", 1),
    ("  float M[NTRI], L[NTRI];  // overlap share storage (unions), the read-only ones the loop seldom reads stay in Hand\n",
     "  const float* M;  // (nv128: in Hand; SimN within the 48 KB of dynamic shared memory k_post gets without an opt-in)\n"
     "  float L[NTRI];  // overlap share storage (unions), the read-only ones the loop seldom reads stay in Hand\n", 1),
    ('    static_assert(NV <= 64 && MAXCON <= 255, "two dofs per lane; contact counts in 8 bits");\n',
     '    static_assert(NV <= 128 && MAXCON <= 255, "NDL dofs per lane (nv128); contact counts in 8 bits");\n', 1),
    ("    float dw[2][3], dv[2][3], dO[2][3];\n    int dbody[2];\n",
     "    float dw[NDL][3], dv[NDL][3], dO[NDL][3];" + NOTE + "\n    int dbody[NDL];\n", 1),
    ("    for (int h = 0; h < 2; h++) {\n      int i = lane + 32 * h;\n      dbody[h] = ",
     "    for (int h = 0; h < NDL; h++) {" + NOTE + "\n      int i = lane + 32 * h;\n      dbody[h] = ", 1),
    ("      for (int h = 0; h < 2; h++) {\n        int i = lane + 32 * h, bi = dbody[h];\n",
     "      for (int h = 0; h < NDL; h++) {" + NOTE + "\n        int i = lane + 32 * h, bi = dbody[h];\n", 1),
    ("  float acc[2];\n#pragma unroll\n  for (int t = 0; t < 2; t++) {\n    const int i = lane + 32 * t;\n",
     "  float acc[NDL];" + NOTE + "\n#pragma unroll\n  for (int t = 0; t < NDL; t++) {\n    const int i = lane + 32 * t;\n",
     1),
    ("    unsigned glo = 0u, ghi = 0u;\n    if (c0 + lane < ncon) {\n      con_force(s, f, c0 + lane, g);\n"
     "      const unsigned long long m = s.con_mask[c0 + lane];\n      glo = (unsigned)m; ghi = (unsigned)(m >> 32);\n"
     "    }\n",
     "    unsigned glo = 0u, gw[NDL] = {};  // (nv128: the support's words; glo = word 0)\n"
     "    if (c0 + lane < ncon) {\n      con_force(s, f, c0 + lane, g);\n"
     "      const DofMask m = s.con_mask[c0 + lane];\n#pragma unroll\n      for (int t = 0; t < NDL; t++) gw[t] = m.w[t];\n"
     "      glo = gw[0];\n    }\n", 1),
    ("    if (NV > 32)\n      for (unsigned hm = __ballot_sync(0xffffffffu, ghi != 0u); hm; hm &= hm - 1u) {\n"
     "        const int i = __ffs(hm) - 1;\n"
     "        const unsigned lo = __shfl_sync(0xffffffffu, glo, i), hi = __shfl_sync(0xffffffffu, ghi, i);\n",
     "#pragma unroll\n    for (int t = 1; t < NDL; t++)  // (nv128: the dofs 32 t .. from word t, the words below it give "
     "the offset)\n      for (unsigned hm = __ballot_sync(0xffffffffu, gw[t] != 0u); hm; hm &= hm - 1u) {\n"
     "        const int i = __ffs(hm) - 1;\n        int kb = 0;\n#pragma unroll\n"
     "        for (int u = 0; u < t; u++) kb += __popc(__shfl_sync(0xffffffffu, gw[u], i));\n"
     "        const unsigned hi = __shfl_sync(0xffffffffu, gw[t], i);\n", 1),
    ("          const int k = __popc(lo) + __popc(hi & below);\n          acc[1] += ",
     "          const int k = kb + __popc(hi & below);\n          acc[t] += ", 1),
    ("  float mv[(NTRI + 31) / 32];  // L = M: all loads first (entry e by lane e mod 32)\n#pragma unroll\n"
     "  for (int k = 0; k < (NTRI + 31) / 32; k++) if (lane + 32 * k < NTRI) mv[k] = s.M[lane + 32 * k];\n", "", 1),
    ("#pragma unroll\n  for (int k = 0; k < (NTRI + 31) / 32; k++) if (lane + 32 * k < NTRI) s.L[lane + 32 * k] = mv[k];\n",
     "  for (int e = lane; e < NTRI; e += 32) s.L[e] = s.M[e];  // (nv128: L = M, entry e by lane e mod 32, unstaged)\n",
     1),
    ("  const int ndof = uniform_int(s.nefc_dof, 8);\n",
     "  const int ndof = uniform_int(s.nefc_dof, 9);  // (nv128: up to 2 NV dof rows)\n", 1),
    ("#ifndef CHOL_INLINE\n", "#if NV <= 64  // (nv128: two rows per lane; NV > 64: chol_solve_dense)\n#ifndef CHOL_INLINE\n", 1),
    ("\n// Broadcast variant (same contract", "#endif  // (nv128)\n" + DEVICE + "\n// Broadcast variant (same contract", 1),
    ("#elif CHOL_COMPACT\n#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_compact(A, L, b, lane)\n#else\n"
     "#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_reg(A, L, b, lane)\n#endif\n",
     "#else\n#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_dense(A, L, b, lane)  // (nv128: NV > 64)\n#endif\n", 1),
    ("      unsigned hd = (unsigned)ncon | (unsigned)s.nefc << 8 | (unsigned)(s.nefc_dof - NFLOSS) << 18\n"
     "                    | (unsigned)min(it, 255) << 24;\n",
     "      unsigned hd = (unsigned)ncon | (unsigned)s.nefc << 8 | (unsigned)min(it, 255) << 24;  // (nv128: nlim in the "
     "reserved word)\n", 1),
    ("      __stcs(rec + OFF_RESERVED, 0.f);\n",
     "      __stcs(rec + OFF_RESERVED, __uint_as_float((unsigned)(s.nefc_dof - NFLOSS)));  // (nv128: nlim)\n", 1),
)

ADJOINT = (
    ("  int hdr;  // the record's header: ncon | nefc << 8 | nlim << 18 | niter << 24\n",
     "  int hdr;  // the record's header: ncon | nefc << 8 | niter << 24 (nv128: nlim 0)\n"
     "  int nlimw;  // (nv128) the record's active limit rows (its reserved word)\n", 1),
    ("    s.hdr = (int)hd;\n",
     "    s.hdr = (int)hd;\n    s.nlimw = (int)__float_as_uint(__ldcs(rec + OFF_RESERVED));  // (nv128)\n", 1),
)
NLIM_RE = re.compile(r"\(\(unsigned\)s\.hdr >> 18\) & 63u?")

SURROGATE = (
    ("    chol_solve_compact(s.L, s.L, s.lam, lane);\n", "    chol_solve_dense(s.L, s.L, s.lam, lane);" + NOTE + "\n", 1),
    ("    float h[2] = {0.f, 0.f};\n    for (int t = 0; t < 2; t++) {\n      const int j = lane + 32 * t;\n",
     "    float h[NDL] = {};" + NOTE + "\n    for (int t = 0; t < NDL; t++) {\n      const int j = lane + 32 * t;\n", 1),
    ("    for (int t = 0; t < 2; t++) {\n      const int j = lane + 32 * t;\n      if (j <= i) s.L[TRI(i, j)] = h[t];\n",
     "    for (int t = 0; t < NDL; t++) {\n      const int j = lane + 32 * t;\n      if (j <= i) s.L[TRI(i, j)] = h[t];\n", 1),
)



HOST = (
    ("  for (int i = lane; i < NTRI; i += 32) s.M[i] = g.M[i];\n",
     "  if (lane == 0) s.M = g.M;  // (nv128: M stays in Hand)\n", 1),
)


WPSM_SPLIT = ("                                           size_t smem_pre = sizeof(SimP), size_t smem_post = sizeof(SimN)) {\n",
              "                                           size_t smem_pre = nv128_smem(0), size_t smem_post = nv128_smem(1)) {"
              "  // (nv128 stage 6)\n")
WPSM_DEV = r"""// (nv128 stage 6) k_pre's / k_post's dynamic shared memory: sizeof(SimP) / sizeof(SimN), or, with a cap of
// NV128_PRE_WPSM / NV128_POST_WPSM worlds per SM (> 0; nv128.wpsm), more, so that at most that many are resident
// (the SM's shared memory / cap - the 1 KB the runtime reserves per block; at most 48 KB: no opt-in)
#ifndef NV128_PRE_WPSM
#define NV128_PRE_WPSM 0
#endif
#ifndef NV128_POST_WPSM
#define NV128_POST_WPSM 0
#endif
// (nv128) a per-world state above the 48 KB launch default (SimN at about 120 dofs with 255 contacts) needs the
// kernels' opt-in to more dynamic shared memory (cudaFuncAttributeMaxDynamicSharedMemorySize; at most the device's
// per-block limit, e.g. 96 KB on V100); set once per process and device, before the first launch
static void nv128_optin(int k, size_t bytes) {
  static bool done[2][64] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= 64 || done[k][dev]) return;
  done[k][dev] = true;
  if (k) {
    cudaFuncSetAttribute(k_post, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
  } else {
    cudaFuncSetAttribute(k_pre, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
#if CV_DEFER
    cudaFuncSetAttribute(k_pre_a, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
#endif
  }
}
static size_t nv128_smem(int k) {
  const size_t base = k ? sizeof(SimN) : sizeof(SimP);
  const int cap = k ? NV128_POST_WPSM : NV128_PRE_WPSM;
  if (base > 48 * 1024) nv128_optin(k, base);  // (a failed opt-in leaves the launch's error to report)
  if (cap <= 0) return base;
  int dev = 0, sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
  size_t s = (size_t)sm / (size_t)cap;
  s = s > 1024 ? (s - 1024) & ~(size_t)127 : 0;
  if (s > 48 * 1024) s = 48 * 1024;
  return s > base ? s : base;
}
"""


def wpsm():
    """$DK_NV128_PRE_WPSM / $DK_NV128_POST_WPSM at source generation (more than 64 dofs): caps on the resident worlds per
    SM of k_pre / k_post through their dynamic shared memory (0 / unset: none, the default)."""
    import os
    out = []
    for k in ("PRE", "POST"):
        v = int(os.environ.get(f"DK_NV128_{k}_WPSM", "") or 0)
        if v < 0:
            raise ValueError(f"DK_NV128_{k}_WPSM={v}: >= 0")
        out.append(v)
    return tuple(out)


def host(src, m):
    """host.KERNEL (the split forward's k_pre / k_post) with the edits for more than 64 dofs (SimN's M as a pointer into
    Hand; the launches' dynamic shared memory through nv128_smem, wpsm); unchanged for at most 64."""
    if not needed(m):
        return src
    src = _apply(src, HOST + (WPSM_SPLIT + (1,),))
    pre, post = wpsm()
    dev = (f"#define NV128_PRE_WPSM {pre}\n#define NV128_POST_WPSM {post}\n" if pre or post else "") + WPSM_DEV
    a = "static inline cudaError_t dk_forward_split("
    if src.count(a) != 1:
        raise RuntimeError("nv128: dk_forward_split not found once")
    return src.replace(a, dev + a)


def _apply(src, hooks):
    for a, b, n in hooks:
        if src.count(a) != n:
            raise RuntimeError(f"nv128 hook anchor found {src.count(a)} times, expected {n}: {a[:70]!r}")
        src = src.replace(a, b)
    return src


def apply(src, m):
    """src (adjoint.cuda_source's text after every other edit, nb64's included) with the edits for more than 64
    dofs; unchanged for at most 64."""
    if not needed(m):
        return src
    assert m.nv <= 128, "nv128: at most 128 dofs (four per lane)"
    if len(SUPPORT_RE.findall(src)) != 1:
        raise RuntimeError("nv128: support_mask not found once")
    src = SUPPORT_RE.sub(lambda mt: (
        f"__device__ __forceinline__ DofMask support_mask({mt.group(1)}) {{{NOTE}\n"
        "  DofMask m = {};\n  for (int i = 0; i < NV; i++)\n"
        "    if ((x >> c_dofbody[i]) & 1u) m.w[i >> 5] |= 1u << (i & 31);\n  return m;\n}\n"), src)
    src = _apply(src, FORWARD)
    src = _dense_swap(src, m.nv)
    a = "  for (int t = 0; t < 2; t++)\n    if (lane + 32 * t < NV) "
    want = 2 if "eq_jtf(s, f, lane + 32 * t, acc[t])" in src else 1
    if src.count(a) != want:
        raise RuntimeError(f"nv128: rows_jtf's per-dof loops found {src.count(a)} times, expected {want}")
    src = src.replace(a, "  for (int t = 0; t < NDL; t++)  // (nv128)\n    if (lane + 32 * t < NV) ")
    if "__global__ void k_adjoint(" in src:
        if src.count(TRI_START) != 1:
            raise RuntimeError("nv128: tri_solve_packed (two entries per lane) not found once")
        i = src.index(TRI_START)
        j = src.index(TRI_END, i) + len(TRI_END)
        src = src[:i] + tri_solve((m.nv + 31) // 32) + src[j:]
        src = _apply(src, ADJOINT)
        n = len(NLIM_RE.findall(src))
        if n < 2:
            raise RuntimeError(f"nv128: the adjoint's reads of the header's nlim found {n} times, expected >= 2")
        src = NLIM_RE.sub("s.nlimw", src)
        if "adj_sur_hessian(" in src:
            src = _apply(src, SURROGATE)
    if re.search(r"\bunsigned long long con_mask\b|\bglo\b.*\bghi\b|acc\[2\]|mv\[k\]", src):
        raise RuntimeError("nv128: a two-dofs-per-lane site left in the text")
    if not sparse(m):
        return _copy_swap(src)
    st = storage(m)
    if st["lh"] and "adj_sur_hessian(" in src:
        raise ValueError("nv128: the contact surrogate rebuilds H from a dense L_H; this model's records hold L_H on the "
                         "closure of P_H: build it with DK_NV128_HCHOL=dense")
    src = apply_sparse(src, m, st)
    done = False
    if hchol() != "dense" and "adj_sur_hessian(" not in src:
        info = hpattern(m)
        if hsparse(m, src, info):
            src, done = apply_hsparse(src, m, info, st), True
    assert done or not st["lh"], "nv128: L_H on the closure without the sparse Newton factor"
    return _copy_swap(src)


def _tri(i, j):
    return i * (i + 1) // 2 + j


def tree(par):
    """The dof tree of a parent array (MuJoCo's dof_parentid: a parent has a smaller index): dict of anc (nearest
    first), depth, height, desc (sorted), the leaves-first pivot order (height, then index) and the root-first order
    (depth, then index)."""
    nv = len(par)
    anc = []
    for i in range(nv):
        a, j = [], int(par[i])
        while j >= 0:
            assert j < i, "nv128 tree: a dof's parent must come first"
            a.append(j)
            j = int(par[j])
        anc.append(a)
    children = [[] for _ in range(nv)]
    for i in range(nv):
        if par[i] >= 0:
            children[int(par[i])].append(i)
    height = [0] * nv
    for i in reversed(range(nv)):
        height[i] = 1 + max((height[c] for c in children[i]), default=-1)
    depth = [len(a) for a in anc]
    desc = [[] for _ in range(nv)]
    for j in range(nv):
        for i in anc[j]:
            desc[i].append(j)
    return dict(nv=nv, anc=anc, depth=depth, height=height, desc=desc,
                leaves_first=sorted(range(nv), key=lambda k: (height[k], k)),
                root_first=sorted(range(nv), key=lambda k: (depth[k], k)))


def tree_chol(par, name="chol_solve_tree"):
    """chol_solve_tree(A, L, b, lane) for the dof tree par as straight-line code: A = R' R with R lower triangular on
    the tree pattern (row i: the columns of i's ancestors and i), eliminated leaves first, so there is no fill (MuJoCo's
    mj_factorM order, as the tree engine's DK_TREE_CHOL); b := A^-1 b in place. Contract as CHOL_SOLVE (A, L packed
    lower triangles in shared or global memory, L may alias A; b a shared NV vector) except that L receives R, and only
    on the pattern: entries off the pattern are left as they are (zero when L aliases A and A is on the pattern)."""
    T = tree(par)
    nv, anc, depth, desc = T["nv"], T["anc"], T["depth"], T["desc"]
    ndl = (nv + 31) // 32
    rows = lambda t: [r for r in range(32 * t, min(32 * t + 32, nv))]
    D = [max(depth[r] for r in rows(t)) for t in range(ndl)]
    base = [sum(D[u] + 1 for u in range(t)) for t in range(ndl)]
    ntab = base[-1] + D[-1] + 1
    off = [0] * (ntab * 32)
    for t in range(ndl):
        for r in rows(t):
            ra = list(reversed(anc[r])) + [r]
            for e, c in enumerate(ra):
                off[(base[t] + e) * 32 + (r & 31)] = _tri(r, c)

    def mask(rs, t):
        return sum(1 << (r & 31) for r in rs if r >> 5 == t)

    def has(t, e):
        return mask([r for r in rows(t) if depth[r] >= e], t)

    F = "0xffffffffu"
    o = [f"// (nv128 stage 3) {name}: see forest_engine.nv128.tree_chol",
         f"__device__ const unsigned short nvt_off[{ntab * 32}] = {{{', '.join(map(str, off))}}};",
         f"__device__ __noinline__ void {name}(const float* A, float* L, float* b, int lane) {{",
         "  const unsigned bit = 1u << lane;",
         "  const unsigned short* off = nvt_off + lane;",
         "  __syncwarp();"]
    for t in range(ndl):
        regs = []
        for e in range(D[t] + 1):
            mk = has(t, e)
            regs.append(f"a{t}_{e} = (0x{mk:08x}u & bit) ? A[off[{(base[t] + e) * 32}]] : 0.f" if mk else f"a{t}_{e} = 0.f")
        o.append("  float " + ", ".join(regs) + ";")
        o.append(f"  float y{t} = lane + {32 * t} < NV ? b[lane + {32 * t}] : 0.f, v{t} = 0.f;  // v: 1 / R_rr")
    for k in T["leaves_first"]:
        sk, sl, dk = k >> 5, k & 31, depth[k]
        a = anc[k]
        o.append(f"  {{  // pivot {k}")
        o.append(f"    const float dkk = fmaxf(__shfl_sync({F}, a{sk}_{dk}, {sl}), 1e-30f);")
        o.append("    float inv;")
        o.append('    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(dkk));')
        o.append("    inv = TMUL(inv, TFMS(TMUL(TMUL(0.5f, dkk), inv), inv, 1.5f));")
        for e in range(dk):
            o.append(f"    const float r{e} = TMUL(__shfl_sync({F}, a{sk}_{e}, {sl}), inv);")
        o.append(f"    const float z = TMUL(__shfl_sync({F}, y{sk}, {sl}), inv);")
        st = "".join(f" a{sk}_{e} = r{e};" for e in range(dk))
        o.append(f"    if (lane == {sl}) {{{st} a{sk}_{dk} = TMUL(dkk, inv); y{sk} = z; v{sk} = inv; }}")
        for t in range(ndl):
            at = [i for i in a if i >> 5 == t]
            if not at:
                continue
            o.append(f"    if (0x{mask(at, t):08x}u & bit) {{")
            o.append("      float c = 0.f;" + "".join(f" if (lane == {i & 31}) c = r{depth[i]};" for i in at))
            for e in range(min(dk, D[t] + 1)):
                o.append(f"      a{t}_{e} = TFMS(c, r{e}, a{t}_{e});")
            o.append(f"      y{t} = TFMS(c, z, y{t});")
            o.append("    }")
        o.append("  }")
    for i in T["root_first"]:
        si, li = i >> 5, i & 31
        o.append(f"  {{ const float x = __shfl_sync({F}, TMUL(y{si}, v{si}), {li});  // x_{i}")
        o.append(f"    if (lane == {li}) y{si} = x;")
        for t in range(ndl):
            mk = mask(desc[i], t)
            if mk:
                o.append(f"    if (0x{mk:08x}u & bit) y{t} = TFMS(a{t}_{depth[i]}, x, y{t});")
        o.append("  }")
    o.append("  __syncwarp();  // (every lane has read A and b)")
    for t in range(ndl):
        o.append(f"  if (lane + {32 * t} < NV) b[lane + {32 * t}] = y{t};")
        for e in range(D[t] + 1):
            mk = has(t, e)
            if mk:
                o.append(f"  if (0x{mk:08x}u & bit) L[off[{(base[t] + e) * 32}]] = a{t}_{e};")
    o.append("  __syncwarp();")
    o.append("}")
    return ("#define TFMS(a, b, c) __fmaf_rn(-(a), (b), (c))\n#define TMUL(a, b) __fmul_rn((a), (b))\n"
            + "\n".join(o) + "\n#undef TFMS\n#undef TMUL\n")


def tree_solve(par, name="tree_solve_packed"):
    """tree_solve_packed(L, b, lane): b := (R' R)^-1 b for a factor R of tree_chol in the packed triangle L (shared
    memory; the tape's L_Q), straight-line: lane c holds b_c, b_{c + 32}, .. in registers; R' z = b leaves first (z_k
    by one shuffle and a division, the ancestors' lanes subtract R[k][c] z_k from their own entry of row k), then
    R x = z root first (the descendants' lanes subtract R[r][i] x_i). Divisions as tri_solve_packed (div_fast)."""
    T = tree(par)
    nv, anc, desc = T["nv"], T["anc"], T["desc"]
    ndl = (nv + 31) // 32

    def mask(rs, t):
        return sum(1 << (r & 31) for r in rs if r >> 5 == t)

    F = "0xffffffffu"
    o = [f"// (nv128 stage 3) {name}: see forest_engine.nv128.tree_solve",
         f"__device__ __noinline__ void {name}(const float* L, float* b, int lane) {{",
         "  const unsigned bit = 1u << lane;",
         "  __syncwarp();"]
    for t in range(ndl):
        o.append(f"  float y{t} = lane + {32 * t} < NV ? b[lane + {32 * t}] : 0.f;")
    for k in T["leaves_first"]:
        sk, sl = k >> 5, k & 31
        o.append(f"  {{ const float d = L[{_tri(k, k)}];")
        o.append(f"    const float z = div_fast(__shfl_sync({F}, y{sk}, {sl}), d, rcp_step(d));")
        o.append(f"    if (lane == {sl}) y{sk} = z;")
        for t in range(ndl):
            mk = mask(anc[k], t)
            if mk:
                o.append(f"    if (0x{mk:08x}u & bit) y{t} -= L[{_tri(k, 0)} + lane + {32 * t}] * z;")
        o.append("  }")
    for i in T["root_first"]:
        si, li = i >> 5, i & 31
        o.append(f"  {{ const float d = L[{_tri(i, i)}];")
        o.append(f"    const float x = div_fast(__shfl_sync({F}, y{si}, {li}), d, rcp_step(d));")
        o.append(f"    if (lane == {li}) y{si} = x;")
        for t in range(ndl):
            mk = mask(desc[i], t)
            if mk:
                o.append(f"    if (0x{mk:08x}u & bit) y{t} -= L[TRI(lane + {32 * t}, {i})] * x;")
        o.append("  }")
    o.append("  __syncwarp();")
    for t in range(ndl):
        o.append(f"  if (lane + {32 * t} < NV) b[lane + {32 * t}] = y{t};")
    o.append("  __syncwarp();")
    o.append("}")
    return "\n".join(o) + "\n"


def tree_ok(m):
    """Whether M (and M + h kv) of model m stays on the tree pattern, so the tree factor's tree factor applies: no noslip
    (its L_M and its adjoint's solves keep the dense form), and every fixed tendon that adds to M or to M + h kv (armature,
    passive damping, tendon actuators' kv) runs along one root-to-leaf chain (forest_engine.tendon's outer products
    c c' then stay on the pattern), and no dof has more than 32 ancestor dofs (tree_chol_loop: one lane each; e.g.
    MyoChallenge's table tennis arm, 38). Otherwise the dense factor of the dense factor (CHOL_SOLVE) stays."""
    from nablaengine.forest_engine import noslip, tendon
    if noslip.enabled(m):
        return False
    if max(tree([int(p) for p in m.dof_parentid])["depth"], default=0) > 32:
        return False
    if m.ntendon and tendon.applies(m):
        ts = set(tendon.armature(m)) | set(tendon.passive(m)) | {t for _, t in tendon.actuators(m)}
        for t in ts:
            d = [x for x, _ in tendon.support_list(m, t)]
            if any(not tendon._ancestor(m, j, i) for a, i in enumerate(d) for j in d[a + 1:]):
                return False
    return True


def mchol():
    """$DK_NV128_MCHOL at source generation (more than 64 dofs only): loop (default: tree_chol_loop), straight
    (tree_chol: the same factor as straight-line code, other pivot order) or dense (the dense factor's chol_solve_dense,
    L_Q dense). V100, 4096 worlds, one control step: S3 (nv 65) forward 48.0 / 52.0 / 49.0-52.0 ms, forward +
    backward 105.8 / 109.3 / 105.4; S1 (nv 113) forward 283 / 284 / 344, forward + backward 455 / 455 / 518
    (the tree factor): the straight-line code is the fastest alone
    but its size (~100 KB of SASS per copy at nv 65) slows the rest of k_post."""
    import os
    raw = os.environ.get("DK_NV128_MCHOL", "") or "loop"
    if raw not in ("straight", "loop", "dense"):
        raise ValueError(f"DK_NV128_MCHOL={raw!r}: straight, loop or dense")
    return raw


def sparse(m):
    """the tree factor's tree-sparse factor of M and M + h kv (CHOL_M_SOLVE = chol_solve_tree, L_Q in tree form): more than
    64 dofs, tree_ok and $DK_NV128_MCHOL not dense."""
    return needed(m) and tree_ok(m) and mchol() != "dense"



M_SOLVE = ("#if CHOL_TREES\n#define CHOL_M_SOLVE(A, L, b, lane, cb) chol_solve_trees(A, L, b, lane)\n#else\n"
           "#define CHOL_M_SOLVE(A, L, b, lane, cb) CHOL_SOLVE(A, L, b, lane, cb)\n#endif\n")

TRI_MUL_RTR = r"""// (nv128 stage 3) out = R' (R x) for L_Q = R (-DDK_ADJ_DEBUG's |M a*|); tmp: NV scratch; all shared
__device__ void tri_mul_rtr(const float* L, const float* x, float* tmp, float* out, int lane) {
  for (int i = lane; i < NV; i += 32) {  // tmp = R x, row i
    float acc = 0.f;
    for (int j = 0; j <= i; j++) acc += L[TRI(i, j)] * x[j];
    tmp[i] = acc;
  }
  __syncwarp();
  for (int j = lane; j < NV; j += 32) {  // out = R' tmp, column j
    float acc = 0.f;
    for (int i = j; i < NV; i++) acc += L[TRI(i, j)] * tmp[i];
    out[j] = acc;
  }
  __syncwarp();
}
"""


def apply_sparse(src, m, st=None):
    """the tree factor's edits (sparse(m)): chol_solve_tree for CHOL_M_SOLVE (k_pre's smooth acceleration, k_post's
    implicitfast M + h kv: the tape's L_Q becomes R, Q = R' R, entries off the tree pattern zero), the record's flag
    FLAG_LQ_TREE, and the adjoint's A1 solve with L_Q by tree_solve_packed (DK_ADJ_DEBUG: tri_mul_rtr).
    the pattern storage (st["lq"], storage): k_pre's factor of M into L on M's tree pattern (chol_solve_tree_m, SimP's L of
    NNZ_LQ words), the record's L_Q gathered onto it (tape_put_idx), the adjoint's copy and solve on it (tree_solve_t);
    with st["lh"] k_post's factor of M + h kv on the closure of P_H (chol_solve_tree_c; apply_hsparse)."""
    from nablaengine.forest_engine.tape import FLAG_LQ_TREE
    st = st or storage(m)
    par = [int(p) for p in m.dof_parentid]
    emit = tree_chol if mchol() == "straight" else tree_chol_loop
    dev = emit(par)
    if st["lq"]:
        dev += tree_chol_loop(par, "chol_solve_tree_m", lidx=st["tl"]["idx"], pfx="nvm")
        if st["lh"]:
            dev += tree_chol_loop(par, "chol_solve_tree_c", aidx=st["cl"]["idx"], lidx=st["cl"]["idx"], pfx="nvc")
    src = _apply(src, ((M_SOLVE, dev + "#define CHOL_M_SOLVE(A, L, b, lane, cb) chol_solve_tree(A, L, b, "
                                 "lane)  // (nv128 stage 3: M on its tree pattern)\n", 1),
                       ("                    | (nu < TAPE_SINGULAR_NU ? TAPE_SINGULAR : 0u);\n",
                        "                    | (nu < TAPE_SINGULAR_NU ? TAPE_SINGULAR : 0u)\n"
                        f"                    | {FLAG_LQ_TREE}u;  // (nv128 stage 3: L_Q in tree form, Q = R' R)\n", 1)))
    if "__global__ void k_adjoint(" in src:
        from nablaengine.forest_engine.tape import FLAG_H_DENSE, FLAG_H_LOWRANK, FLAG_LH_SPARSE
        src = _apply(src, (("    s.flags |= (int)__float_as_uint(__ldcs(rec + OFF_FLAGS));\n",
                            f"    s.flags |= (int)(__float_as_uint(__ldcs(rec + OFF_FLAGS)) & ~{FLAG_LQ_TREE | FLAG_LH_SPARSE | FLAG_H_LOWRANK | FLAG_H_DENSE}u);"
                            "  // (nv128: not the factor forms)\n", 1),))
        anchor = "__device__ __noinline__ void adj_solves(Adj& s, const float* rec, int lane) {\n"
        solve = tree_solve_t(par) + tri_mul_rtr_t(par) if st["lq"] else tree_solve(par) + TRI_MUL_RTR
        src = _apply(src, ((anchor, solve + anchor, 1),
                           ("  tri_solve_packed(s.L, s.y, lane);\n",
                            "  tree_solve_packed(s.L, s.y, lane);  // (nv128 stage 3: L_Q = R, Q = R' R)\n", 1),
                           ("    tri_mul_llt(s.L, s.a, s.tmp, s.did, lane);\n",
                            "    tri_mul_rtr(s.L, s.a, s.tmp, s.did, lane);  // (nv128 stage 3: L_Q = R)\n", 1)))
    if st["lq"]:
        src = apply_lq(src, m, st)
    return src


S5_LQ = (
    ("    struct { float L[NTRI]; alignas(16) float cbuf[CHOL_CB]; };  // CHOL_SOLVE's scratch\n",
     "    struct { float L[NNZ_LQ]; alignas(16) float cbuf[CHOL_CB]; };  // CHOL_SOLVE's scratch (nv128 stage 5: L of "
     "M on its tree pattern)\n", 1),
    ("  CHOL_M_SOLVE(s.M, s.L, s.qacc0, lane, s.cbuf);\n",
     "  chol_solve_tree_m(s.M, LRAW(s.L), s.qacc0, lane);  // (nv128 stage 5: R into L on M's tree pattern)\n", 1),
    ("    tape_put(rec + OFF_LQ, s.L, NTRI, REC_WORDS - OFF_LQ, lane);\n",
     "    tape_put_idx(rec + OFF_LQ, LRAW(s.L), nvq_src, NNZ_LQ, REC_WORDS - OFF_LQ, lane);  // (nv128 stage 5: L_Q on M's "
     "tree pattern)\n", 1),
)
S5_LQ_ADJ = (
    ("      float L[NTRI];  // a factor from the record: L_Q (A1), then L_H (A2)\n",
     "      float L[NNZ_ADJ];  // a factor from the record: L_Q (A1), then L_H (A2) (nv128 stage 5: as stored there)\n", 1),
    ("  for (int e = lane; e < NTRI; e += 32) s.L[e] = __ldcs(rec + OFF_LQ + e);\n",
     "  for (int e = lane; e < NNZ_LQ; e += 32) s.L[e] = __ldcs(rec + OFF_LQ + e);  // (nv128 stage 5: tlayout)\n", 1),
)
S5_LH_LOAD = "  for (int e = lane; e < NTRI; e += 32) s.L[e] = __ldcs(rec + OFF_LH + e);\n"


def apply_lq(src, m, st):
    """the pattern storage's edits of every lq_compact build (apply_sparse): the definitions (s5_header), SimP's L, k_pre's factor of
    M into tlayout, the record's L_Q gathered into tlayout, the adjoint's copies of L_Q and L_H as long as they are."""
    if src.count("struct Sim {\n") != 1:
        raise RuntimeError("nv128: struct Sim not found once")
    src = src.replace("struct Sim {\n", s5_header(m, st) + "struct Sim {\n")
    src = _apply(src, S5_LQ)
    if "__global__ void k_adjoint(" in src:
        src = _apply(src, S5_LQ_ADJ)
        if src.count(S5_LH_LOAD) < 1:
            raise RuntimeError("nv128: the adjoint's L_H load not found")
        src = src.replace(S5_LH_LOAD, "  for (int e = lane; e < NNZ_LH; e += 32) s.L[e] = __ldcs(rec + OFF_LH + e);"
                                      "  // (nv128 stage 5: as stored)\n")
    return src


S5_LH = (
    ("  float L[NTRI];  // overlap share storage (unions), the read-only ones the loop seldom reads stay in Hand\n",
     "  HL L;  // (nv128 stage 5: on the closure of P_H) overlap share storage (unions), the read-only ones the loop seldom "
     "reads stay in Hand\n", 1),
    ("  float M[NTRI], L[NTRI];  // mass matrix and Cholesky scratch, packed lower triangles TRI(i, j)\n",
     "  float M[NTRI];  // mass matrix, packed lower triangle TRI(i, j)\n"
     "  HL L;  // Cholesky scratch (nv128 stage 5: on the closure of P_H)\n", 1),
    ("  for (int e = lane; e < NTRI; e += 32) s.L[e] = s.M[e];  // (nv128: L = M, entry e by lane e mod 32, unstaged)\n",
     "  for (int q = lane; q < NNZ_LH; q += 32) s.L.v[q] = s.M[nvc_tri[q]];  // (nv128 stage 5: L = M on the closure)\n"
     "  __syncwarp();  // (the dof rows below add by the lane of the packed index)\n", 1),
    ("  for (int e = lane; e < NTRI; e += 32) s.L[e] = s.M[e];\n  __syncwarp();\n",
     "  for (int q = lane; q < NNZ_LH; q += 32) s.L.v[q] = s.M[nvc_tri[q]];  // (nv128 stage 5: L = M on the closure)\n"
     "  __syncwarp();\n", 1),
    ("  CHOL_M_SOLVE(s.L, s.L, s.tmp, lane, s.cbuf);\n",
     "  chol_solve_tree_c(s.L.v, s.L.v, s.tmp, lane);  // (nv128 stage 5: M + h kv on the closure of P_H)\n", 1),
    ("    tape_put(rec + OFF_LH, s.L, NTRI, OFF_LQ - OFF_LH, lane);\n",
     "    tape_put(rec + OFF_LH, s.L.v, NNZ_LH, OFF_LQ - OFF_LH, lane);  // (nv128 stage 5: L_H on the closure of P_H)\n", 1),
)


def tree_chol_loop(par, name="chol_solve_tree", aidx=None, lidx=None, pfx="nvt"):
    """chol_solve_tree as loops over the pivots (compact code: the straight-line form's ~5 instructions per factor
    entry come to ~100 KB of SASS per copy at nv 65, which evicts the Newton loop's code from the instruction caches).
    Same contract and factor as tree_chol, other order: pivots by decreasing index (a valid leaves-first order, MuJoCo's
    mj_factorM order), one loop per register slot t (rows 32 t .. 32 t + 31; their ancestors are in slots <= t), so
    the pivot row's slot is a compile-time constant. Rows in registers by depth (a<t>_<e>, e < the row's depth; the
    diagonal in g<t>). Pivot k (lane k & 31): its row broadcast, unrolled to the slot's deepest row without branches
    (registers past a row's depth hold 0, so r_e = 0 there), every lane forms r_e = A[k][e] inv; lane e keeps r_e (w)
    and stores it to L (its column from nvt_pa); the lanes of k's ancestors (nvt_anc, per slot u <= t) take
    c = r_{depth row} by one shuffle of w and update A[i][e] -= c r_e for every e of the slot's depth range (entries
    past the row's own depth are scratch, never read), A[i][i] -= c c, y_i -= c z_k (the R' z = b solve). Then R x = z
    root first from L (the descendants' lanes, nvt_desc). Explicit roundings; the pivot formula is chol_solve_dense's.
    aidx, lidx (the pattern storage): the storage of A and of L as functions (i, j) -> index for a dof i and j = i or an ancestor
    of i (None: the packed triangle TRI(i, j)); with either given, the same factor through index tables (prefix pfx):
    _tree_chol_loop_map."""
    if aidx is not None or lidx is not None:
        return _tree_chol_loop_map(par, name, aidx or _tri, lidx or _tri, pfx)
    T = tree(par)
    nv, anc, depth, desc = T["nv"], T["anc"], T["depth"], T["desc"]
    ndl = (nv + 31) // 32
    rows = lambda t: [r for r in range(32 * t, min(32 * t + 32, nv))]
    D = [max(depth[r] for r in rows(t)) for t in range(ndl)]
    base = [sum(D[u] for u in range(t)) for t in range(ndl)]
    ntab = max(base[-1] + D[-1], 1)
    off = [0] * (ntab * 32)
    for t in range(ndl):
        for r in rows(t):
            for e, c in enumerate(reversed(anc[r])):
                off[(base[t] + e) * 32 + (r & 31)] = _tri(r, c)
    dep = [depth[32 * t + l] if 32 * t + l < nv else -1 for t in range(ndl) for l in range(32)]
    assert max(depth) <= 32, "nv128 tree_chol_loop: at most 32 ancestors (one lane each)"
    pa = [_tri(k, list(reversed(anc[k]))[l]) if l < depth[k] else 0 for k in range(nv) for l in range(32)]
    ancm = [sum(1 << (i & 31) for i in anc[k] if i >> 5 == u) for k in range(nv) for u in range(ndl)]
    rf = T["root_first"]
    dm = [sum(1 << (j & 31) for j in desc[i] if j >> 5 == t) for i in rf for t in range(ndl)]
    lst = lambda v: ", ".join(map(str, v))
    F = "0xffffffffu"
    o = [f"// (nv128 stage 3) {name}: see forest_engine.nv128.tree_chol_loop",
         f"__device__ const unsigned short nvt_off[{ntab * 32}] = {{{lst(off)}}};",
         f"__device__ const signed char nvt_dep[{ndl * 32}] = {{{lst(dep)}}};",
         f"__device__ const unsigned short nvt_pa[{nv * 32}] = {{{lst(pa)}}};",
         f"__constant__ unsigned char nvt_dk[{nv}] = {{{lst(depth)}}};",
         f"__constant__ unsigned nvt_anc[{nv * ndl}] = {{{lst(ancm)}}};",
         f"__constant__ unsigned char nvt_rf[{nv}] = {{{lst(rf)}}};",
         f"__constant__ unsigned nvt_desc[{nv * ndl}] = {{{lst(dm)}}};",
         f"__device__ __noinline__ void {name}(const float* A, float* L, float* b, int lane) {{",
         "  const unsigned bit = 1u << lane;",
         "  const unsigned short* off = nvt_off + lane;",
         "  __syncwarp();"]
    for t in range(ndl):
        r = f"lane + {32 * t}"
        o.append(f"  const int d{t} = nvt_dep[{r}], o{t} = ({r}) * ({r} + 1) / 2;  // the row's depth, TRI(row, 0)")
        regs = [f"a{t}_{e} = d{t} > {e} ? A[off[{(base[t] + e) * 32}]] : 0.f" for e in range(D[t])]
        o.append("  float " + ", ".join(regs + [f"g{t} = d{t} >= 0 ? A[o{t} + {r}] : 0.f"]) + ";")
        o.append(f"  float y{t} = d{t} >= 0 ? b[{r}] : 0.f;")
    o.append("  __syncwarp();  // (A and b read: L may alias A)")
    for t in reversed(range(ndl)):
        o.append(f"  for (int k = {min(32 * t + 31, nv - 1)}; k >= {32 * t}; k--) {{  // the pivots of slot {t}")
        o.append("    const int sl = k & 31;")
        o.append(f"    const float dkk = fmaxf(__shfl_sync({F}, g{t}, sl), 1e-30f);")
        o.append("    float inv;")
        o.append('    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(dkk));')
        o.append("    inv = TMUL(inv, TFMS(TMUL(TMUL(0.5f, dkk), inv), inv, 1.5f));")
        o.append(f"    const float z = TMUL(__shfl_sync({F}, y{t}, sl), inv);")
        o.append("    float w = 0.f;  // r_lane")
        for e in range(D[t]):
            o.append(f"    const float r{e} = TMUL(__shfl_sync({F}, a{t}_{e}, sl), inv);  // (0 from depth k on)")
            o.append(f"    if (lane == {e}) w = r{e};")
        o.append("    if (lane < nvt_dk[k]) L[nvt_pa[k * 32 + lane]] = w;")
        o.append(f"    if (lane == sl) {{ L[k * (k + 1) / 2 + k] = g{t} = TMUL(dkk, inv); y{t} = z; }}")
        for u in range(t + 1):
            o.append(f"    {{ const unsigned am = nvt_anc[k * {ndl} + {u}];")
            o.append(f"      if (am) {{ const float c = __shfl_sync({F}, w, d{u} & 31);")
            o.append("        if (am & bit) {")
            for e in range(min(D[u], D[t])):
                o.append(f"          a{u}_{e} = TFMS(c, r{e}, a{u}_{e});")
            o.append(f"          g{u} = TFMS(c, c, g{u}); y{u} = TFMS(c, z, y{u});")
            o.append("        } } }")
        o.append("  }")
    o.append("  __syncwarp();")
    o.append("  const float " + ", ".join(f"rq{t} = __frcp_rn(d{t} >= 0 ? g{t} : 1.f)" for t in range(ndl))
             + ";  // (stage 6: the reciprocals of the diagonal before the chain)")
    o.append("  for (int q = 0; q < NV; q++) {  // R x = z, root first")
    o.append("    const int i = nvt_rf[q], li = i & 31, si = i >> 5;")
    ysel = " : ".join([f"si == {t} ? y{t}" for t in range(ndl - 1)] + [f"y{ndl - 1}"])
    csel = " : ".join([f"si == {t} ? rq{t}" for t in range(ndl - 1)] + [f"rq{ndl - 1}"])
    o.append(f"    const float a_ = __shfl_sync({F}, {ysel}, li), r_ = __shfl_sync({F}, {csel}, li);")
    o.append(f"    const float x = {_mk_div('a_', 'r_', 'L[i * (i + 1) / 2 + i]')};")
    for t in range(ndl):
        o.append(f"    if (lane == li && si == {t}) y{t} = x;")
        o.append(f"    if (nvt_desc[q * {ndl} + {t}] & bit) y{t} = TFMS(L[o{t} + i], x, y{t});")
    o.append("  }")
    o.append("  __syncwarp();")
    for t in range(ndl):
        o.append(f"  if (d{t} >= 0) b[lane + {32 * t}] = y{t};")
    o.append("  __syncwarp();")
    o.append("}")
    return ("#define TFMS(a, b, c) __fmaf_rn(-(a), (b), (c))\n#define TMUL(a, b) __fmul_rn((a), (b))\n"
            + "\n".join(o) + "\n#undef TFMS\n#undef TMUL\n")


def _mk_div(a, r, d):
    return f"__fmaf_rn({r}, __fmaf_rn(-({d}), __fmul_rn({a}, {r}), {a}), __fmul_rn({a}, {r}))"


def _ptri(i, j):
    return _tri(max(i, j), min(i, j))


def ktree(kpar):
    """The tree of kernel rows kpar (the sparse Newton factor: a parent has a LARGER index than its children, rows leaves first): dict of
    anc (nearest first), depth, desc and root_first (depth, then index)."""
    nv = len(kpar)
    anc = []
    for i in range(nv):
        a, j = [], int(kpar[i])
        while j >= 0:
            assert j > i, "nv128 ktree: a row's parent must come after it"
            a.append(j)
            j = int(kpar[j])
        anc.append(a)
    depth = [len(a) for a in anc]
    desc = [[] for _ in range(nv)]
    for j in range(nv):
        for i in anc[j]:
            desc[i].append(j)
    return dict(nv=nv, anc=anc, depth=depth, desc=desc, root_first=sorted(range(nv), key=lambda k: (depth[k], k)))


def hpattern(m, extra=()):
    """The static Hessian pattern of model m for the sparse Newton factor (forest_engine.sparse_pattern.Pattern with the model's
    explicit contact pairs, the engine's pair table: tree(M), the constraint cliques, the world / chain / object pairs
    and the cheap self-pair groups; the other self pairs are dynamic), its symbolic factor's elimination tree and the
    kernel layout of chol_solve_hp. Returns a dict:"""
    if not extra:
        key = _fingerprint(m)
        if key not in _HP_CACHE:
            _HP_CACHE[key] = _hpattern(m, ())
        return _HP_CACHE[key]
    return _hpattern(m, extra)


_HP_CACHE = {}


def _fingerprint(m):
    import hashlib
    import numpy as np
    h = hashlib.sha256()
    for f in ("dof_parentid", "dof_bodyid", "body_parentid", "body_dofnum", "body_dofadr", "jnt_dofadr", "jnt_type",
              "geom_bodyid", "site_bodyid", "pair_geom1", "pair_geom2", "eq_type", "eq_objtype", "eq_obj1id",
              "eq_obj2id", "tendon_adr", "tendon_num", "wrap_type", "wrap_objid"):
        a = np.ascontiguousarray(getattr(m, f, np.zeros(0)))
        h.update(f.encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def _hpattern(m, extra):
    from nablaengine.forest_engine import sparse_pattern as sp
    P = sp.Pattern(m, pairs=[(int(a), int(b)) for a, b in zip(m.pair_geom1, m.pair_geom2)])
    F = P.factor
    if extra:
        adj = list(P.adj)
        for k in extra:
            P._add(adj, P.pairs[k].support)
        F = sp.symbolic(F.n, adj, P.order)
        for p in P.pairs:
            p.in_pattern = F.is_clique(p.support)
    n = F.n
    dep = [0] * n
    for k in reversed(range(n)):
        dep[k] = dep[F.etree[k]] + 1 if F.etree[k] >= 0 else 0
    ndl = (n + 31) // 32

    def slots(order):
        return [max(dep[order[i]] for i in range(32 * t, min(32 * t + 32, n))) for t in range(ndl)]

    cands = [list(range(n)), sorted(range(n), key=lambda k: (-dep[k], k))]
    order = min(cands, key=lambda o: (sum(slots(o)), cands.index(o)))
    kidx = {p: i for i, p in enumerate(order)}
    kpar = [kidx[F.etree[p]] if F.etree[p] >= 0 else -1 for p in order]
    assert all(kpar[i] < 0 or kpar[i] > i for i in range(n)), "nv128 hpattern: kernel order not children first"
    return dict(perm=[int(F.perm[p]) for p in order], kpar=kpar, pdyn=[not p.in_pattern for p in P.pairs],
                nnz_ph=F.nnz, nnz_closure=sum(d + 1 for d in dep), depth=max(dep), slots=slots(order),
                dense=n * (n + 1) // 2, npair_dyn=sum(not p.in_pattern for p in P.pairs))


HREG_BUDGET = 96
HW_COLS = 12


def hchol():
    """$DK_NV128_HCHOL at source generation (more than 64 dofs only): auto (default: the Newton Hessian factored on its
    static pattern P_H by chol_solve_hp when every contact pair of the model is on P_H, else dense), sparse (P_H with
    hp_lowrank's correction in the worlds with contacts off the pattern, the dense factor beyond HW_COLS columns) or
    dense (chol_solve_dense for every world). V100, S3 (929 of 1544 pairs off P_H; 36 % of world-substeps of random-
    control rollouts with such a contact in H): dense 48.4-51.8 ms forward / 105.5-107.0 forward + backward, sparse
    54.1-56.9 / 115.1-115.4 (a per-world choice between two factor codes costs more than either alone: measured with
    forced forms in the same code); every world on P_H (s3noself: S3 without its off-pattern
    pairs) auto 40.5-41.2 / 94.5-95.5 against dense 45.4-48.4 / 102.5-103.0."""
    import os
    raw = os.environ.get("DK_NV128_HCHOL", "") or "auto"
    if raw not in ("auto", "sparse", "dense"):
        raise ValueError(f"DK_NV128_HCHOL={raw!r}: auto, sparse or dense")
    return raw


def hsparse(m, src=None, info=None):
    """the sparse Newton factor's sparse Newton factor applies: the tree factor on (sparse(m)), $DK_NV128_HCHOL sparse, or auto and no contact
    pair off P_H, the pyramidal cone (the elliptic cone's Hessian blocks are not hp_lowrank's C), no contact surrogate
    in the text (its H_mu rebuilds H = L L' from a dense L_H), the elimination tree at most 64 deep and its rows within
    HREG_BUDGET registers (S1 / S2: P_H is 70-77 % of dense, the tree 72-81 deep: the dense factor stays)."""
    import mujoco
    if not sparse(m) or hchol() == "dense" or int(m.opt.cone) != int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        return False
    if src is not None and "adj_sur_hessian(" in src:
        return False
    info = info or hpattern(m)
    if hchol() == "auto" and info["npair_dyn"]:
        return False
    return info["depth"] <= 64 and sum(info["slots"]) <= HREG_BUDGET


def perm_chol_loop(kpar, perm, name="chol_solve_hp", pfx="nvh", cl=False):
    """chol_solve_hp(A, L, b, lane): A = R' R on the dof tree kpar of kernel rows (leaves first: kpar[r] > r) whose dofs
    are perm[r], b := A^-1 b in place (A, L the packed triangles of the original dof order, L may alias A: entry
    (r, c) at TRI(max(perm r, perm c), min(..)); b in dof order). For P_H's elimination tree (hpattern): R on the
    tree's ancestor closure, no fill. As tree_chol_loop (the tree factor) with the slot order mirrored: lane l holds the rows
    l, l + 32, .. (slot t = row >> 5), each row's entries by depth in registers (a<t>_<e>: the ancestor at depth e,
    the diagonal in g<t>), one loop per slot over its pivots by increasing index (ancestors are in slots >= t), the
    entries of the slot's depth range unrolled without branches (entries past a row's depth are scratch, never read
    as R), the pivot's entries r_e kept by lane e (w; e >= 32: lane e - 32, w1) for the ancestors' c = r_{depth}. The
    pivot's lane keeps its finished row in its registers; R goes to L in one pass after the loop (through the table of
    the loads), R x = z (root first) computes its addresses. Entries of L outside the closure are left as they are
    (zero for a Hessian on P_H). Explicit roundings; the pivot formula is chol_solve_dense's. cl (the pattern storage): A and L
    are stored on the closure itself (clayout: kernel row r's entries by depth from <pfx>_cb[r], the diagonal last)
    instead of the packed triangle: the same factor, addresses c<t> + e."""
    T = ktree(kpar)
    nv, anc, depth, desc = T["nv"], T["anc"], T["depth"], T["desc"]
    ndl = (nv + 31) // 32
    assert max(depth) <= 64, "nv128 perm_chol_loop: at most 64 ancestors (two registers on 32 lanes)"
    rows = lambda t: [r for r in range(32 * t, min(32 * t + 32, nv))]
    D = [max(depth[r] for r in rows(t)) for t in range(ndl)]
    base = [sum(D[u] for u in range(t)) for t in range(ndl)]
    ntab = max(base[-1] + D[-1], 1)
    off = [0] * (ntab * 32)
    for t in range(ndl):
        for r in rows(t):
            for e, c in enumerate(reversed(anc[r])):
                off[(base[t] + e) * 32 + (r & 31)] = _ptri(perm[r], perm[c])
    lane_of = lambda t, l: 32 * t + l
    dep = [depth[lane_of(t, l)] if lane_of(t, l) < nv else -1 for t in range(ndl) for l in range(32)]
    pv = [perm[lane_of(t, l)] if lane_of(t, l) < nv else 0 for t in range(ndl) for l in range(32)]
    ancm = [sum(1 << (i & 31) for i in anc[k] if i >> 5 == u) for k in range(nv) for u in range(ndl)]
    rf = T["root_first"]
    dm = [sum(1 << (j & 31) for j in desc[i] if j >> 5 == t) for i in rf for t in range(ndl)]
    lst = lambda v: ", ".join(map(str, v))
    F = "0xffffffffu"
    cb = _cbase(depth)
    o = [f"// (nv128 stage 4) {name}: see forest_engine.nv128.perm_chol_loop"]
    if cl:
        o += [f"__device__ const unsigned short {pfx}_cb[{ndl * 32}] = "
              f"{{{lst([cb[lane_of(t, l)] if lane_of(t, l) < nv else 0 for t in range(ndl) for l in range(32)])}}};",
              f"__constant__ unsigned short {pfx}_cbk[{nv}] = {{{lst(cb[:nv])}}};",
              f"__constant__ unsigned char {pfx}_kd[{nv}] = {{{lst(depth)}}};"]
    else:
        o.append(f"__device__ const unsigned short {pfx}_off[{ntab * 32}] = {{{lst(off)}}};")
    o += [f"__device__ const signed char {pfx}_dep[{ndl * 32}] = {{{lst(dep)}}};",
         f"__device__ const unsigned char {pfx}_pv[{ndl * 32}] = {{{lst(pv)}}};",
         f"__constant__ unsigned {pfx}_anc[{nv * ndl}] = {{{lst(ancm)}}};",
         f"__constant__ unsigned char {pfx}_rf[{nv}] = {{{lst(rf)}}};",
         f"__constant__ unsigned char {pfx}_pk[{nv}] = {{{lst(perm)}}};",
         f"__constant__ unsigned {pfx}_desc[{nv * ndl}] = {{{lst(dm)}}};",
         f"__device__ __noinline__ void {name}(const float* A, float* L, float* b, int lane, bool back = true) {{",
         "  const unsigned bit = 1u << lane;"]
    if not cl:
        o.append(f"  const unsigned short* off = {pfx}_off + lane;")
    o.append("  __syncwarp();")
    for t in range(ndl):
        r = f"lane + {32 * t}"
        if cl:
            o.append(f"  const int d{t} = {pfx}_dep[{r}], p{t} = {pfx}_pv[{r}], c{t} = {pfx}_cb[{r}];"
                     "  // depth, dof, the row's start on the closure")
            regs = [f"a{t}_{e} = d{t} > {e} ? A[c{t} + {e}] : 0.f" for e in range(D[t])]
            o.append("  float " + ", ".join(regs + [f"g{t} = d{t} >= 0 ? A[c{t} + d{t}] : 0.f"]) + ";")
        else:
            o.append(f"  const int d{t} = {pfx}_dep[{r}], p{t} = {pfx}_pv[{r}], o{t} = p{t} * (p{t} + 1) / 2;"
                     "  // depth, dof, TRI(dof, 0)")
            regs = [f"a{t}_{e} = d{t} > {e} ? A[off[{(base[t] + e) * 32}]] : 0.f" for e in range(D[t])]
            o.append("  float " + ", ".join(regs + [f"g{t} = d{t} >= 0 ? A[o{t} + p{t}] : 0.f"]) + ";")
        o.append(f"  float y{t} = d{t} >= 0 ? b[p{t}] : 0.f;")
    o.append("  __syncwarp();  // (A and b read: L may alias A)")
    for t in range(ndl):
        wide = D[t] > 32
        o.append(f"  for (int k = {32 * t}; k <= {min(32 * t + 31, nv - 1)}; k++) {{  // the pivots of slot {t}")
        o.append("    const int sl = k & 31;")
        o.append(f"    const float dkk = fmaxf(__shfl_sync({F}, g{t}, sl), 1e-30f);")
        o.append("    float inv;")
        o.append('    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(dkk));')
        o.append("    inv = TMUL(inv, TFMS(TMUL(TMUL(0.5f, dkk), inv), inv, 1.5f));")
        o.append(f"    const float z = TMUL(__shfl_sync({F}, y{t}, sl), inv);")
        if D[t]:
            o.append("    float w = 0.f" + (", w1 = 0.f;" if wide else ";") + "  // r_lane (, r_lane+32)")
        for e in range(D[t]):
            o.append(f"    const float r{e} = TMUL(__shfl_sync({F}, a{t}_{e}, sl), inv);  // (0 from depth k on)")
            o.append(f"    if (lane == {e}) w = r{e};" if e < 32 else f"    if (lane == {e - 32}) w1 = r{e};")
        keep = " ".join(f"a{t}_{e} = r{e};" for e in range(D[t]))
        o.append(f"    if (lane == sl) {{ {keep} g{t} = TMUL(dkk, inv); y{t} = z; }}  // (the finished row of R)")
        for u in range(t, ndl):
            if not D[t]:
                break
            o.append(f"    {{ const unsigned am = {pfx}_anc[k * {ndl} + {u}];")
            if wide and D[u] > 32:
                o.append(f"      if (am) {{ const float c0 = __shfl_sync({F}, w, d{u} & 31), "
                         f"c1 = __shfl_sync({F}, w1, d{u} & 31), c = d{u} >= 32 ? c1 : c0;")
            else:
                o.append(f"      if (am) {{ const float c = __shfl_sync({F}, w, d{u} & 31);")
            o.append("        if (am & bit) {")
            for e in range(min(D[u], D[t])):
                o.append(f"          a{u}_{e} = TFMS(c, r{e}, a{u}_{e});")
            o.append(f"          g{u} = TFMS(c, c, g{u}); y{u} = TFMS(c, z, y{u});")
            o.append("        } } }")
        o.append("  }")
    for t in range(ndl):
        o.append(f"  if (d{t} >= 0) L[c{t} + d{t}] = g{t};" if cl else f"  if (d{t} >= 0) L[o{t} + p{t}] = g{t};")
        for e in range(D[t]):
            o.append(f"  if (d{t} > {e}) L[c{t} + {e}] = a{t}_{e};" if cl else
                     f"  if (d{t} > {e}) L[off[{(base[t] + e) * 32}]] = a{t}_{e};")
    o.append("  __syncwarp();")
    o.append("  const float " + ", ".join(f"rq{t} = __frcp_rn(d{t} >= 0 ? g{t} : 1.f)" for t in range(ndl))
             + ";  // (stage 6: the reciprocals of the diagonal before the chain)")
    o.append("  if (back)  // (false: z = R'^-1 b to b, for hp_lowrank)")
    o.append("  for (int q = 0; q < NV; q++) {  // R x = z, root first")
    ysel = " : ".join([f"si == {t} ? y{t}" for t in range(ndl - 1)] + [f"y{ndl - 1}"])
    csel = " : ".join([f"si == {t} ? rq{t}" for t in range(ndl - 1)] + [f"rq{ndl - 1}"])
    if cl:
        o.append(f"    const int i = {pfx}_rf[q], li = i & 31, si = i >> 5, ki = {pfx}_kd[i];")
        o.append(f"    const float a_ = __shfl_sync({F}, {ysel}, li), r_ = __shfl_sync({F}, {csel}, li);")
        o.append(f"    const float x = {_mk_div('a_', 'r_', f'L[{pfx}_cbk[i] + ki]')};")
    else:
        o.append(f"    const int i = {pfx}_rf[q], li = i & 31, si = i >> 5, pi = {pfx}_pk[i], ti = pi * (pi + 1) / 2;")
        o.append(f"    const float a_ = __shfl_sync({F}, {ysel}, li), r_ = __shfl_sync({F}, {csel}, li);")
        o.append(f"    const float x = {_mk_div('a_', 'r_', 'L[ti + pi]')};")
    for t in range(ndl):
        o.append(f"    if (lane == li && si == {t}) y{t} = x;")
        ent = f"L[c{t} + ki]" if cl else f"L[p{t} > pi ? o{t} + pi : ti + p{t}]"
        o.append(f"    if ({pfx}_desc[q * {ndl} + {t}] & bit) y{t} = TFMS({ent}, x, y{t});")
    o.append("  }")
    o.append("  __syncwarp();")
    for t in range(ndl):
        o.append(f"  if (d{t} >= 0) b[p{t}] = y{t};")
    o.append("  __syncwarp();")
    o.append("}")
    return ("#define TFMS(a, b, c) __fmaf_rn(-(a), (b), (c))\n#define TMUL(a, b) __fmul_rn((a), (b))\n"
            + "\n".join(o) + "\n#undef TFMS\n#undef TMUL\n")


def perm_solve_loop(kpar, perm, name, div="div_fast(y_, d, rcp_step(d))", pfx=None, multi=False, cl=False):
    """name(L, b, lane): b := (R' R)^-1 b for a factor R of perm_chol_loop(kpar, perm) in the packed triangle L
    (shared memory; the tape's L_H), as loops (a straight-line form made the adjoint ~2 ms slower on S3): the vector in
    registers, lane l holding the kernel rows l, l + 32, .. (dofs perm[row]); R' z = b over the pivots by increasing
    index, one loop per slot (z_k from lane k & 31, the lanes of k's ancestors subtract R[k][their row] z_k), then
    R x = z root first (the descendants' lanes). Addresses computed (TRI of the two dofs), masks and dofs from constant
    tables. Also <pfx>_vsolve(L, y, lane): the same on a vector already in that register layout (y[NDL]: lane l's
    rows). div: the division (y_ / d; the adjoint's div_fast, the forward's __fdiv_rn). cl (the pattern storage): L stored on the
    closure (clayout) instead of the packed triangle (as perm_chol_loop's cl)."""
    T = ktree(kpar)
    nv, anc, desc = T["nv"], T["anc"], T["desc"]
    assert not (cl and multi), "nv128 perm_solve_loop: the closure storage has no low-rank form"
    cb = _cbase(T["depth"])
    ndl = (nv + 31) // 32
    pfx = pfx or name
    pv = [perm[32 * t + l] if 32 * t + l < nv else 0 for t in range(ndl) for l in range(32)]
    ancm = [sum(1 << (i & 31) for i in anc[k] if i >> 5 == u) for k in range(nv) for u in range(ndl)]
    rf = T["root_first"]
    dm = [sum(1 << (j & 31) for j in desc[i] if j >> 5 == t) for i in rf for t in range(ndl)]
    lst = lambda v: ", ".join(map(str, v))
    F = "0xffffffffu"
    o = [f"// (nv128 stage 4) {name}: see forest_engine.nv128.perm_solve_loop",
         f"__device__ const unsigned char {pfx}_pv[{ndl * 32}] = {{{lst(pv)}}};",
         f"__constant__ unsigned {pfx}_anc[{nv * ndl}] = {{{lst(ancm)}}};",
         f"__constant__ unsigned char {pfx}_rf[{nv}] = {{{lst(rf)}}};",
         f"__constant__ unsigned char {pfx}_pk[{nv}] = {{{lst(perm)}}};",
         f"__constant__ unsigned {pfx}_desc[{nv * ndl}] = {{{lst(dm)}}};"]
    if cl:
        o += [f"__device__ const unsigned short {pfx}_cb[{ndl * 32}] = "
              f"{{{lst([cb[32 * t + l] if 32 * t + l < nv else 0 for t in range(ndl) for l in range(32)])}}};",
              f"__device__ const unsigned char {pfx}_kdl[{ndl * 32}] = "
              f"{{{lst([T['depth'][32 * t + l] if 32 * t + l < nv else 0 for t in range(ndl) for l in range(32)])}}};",
              f"__constant__ unsigned short {pfx}_cbk[{nv}] = {{{lst(cb[:nv])}}};",
              f"__constant__ unsigned char {pfx}_kd[{nv}] = {{{lst(T['depth'])}}};"]
    o += [f"__device__ __forceinline__ void {pfx}_vsolve(const float* L, float (&y)[{ndl}], int lane) {{",
          "  const unsigned bit = 1u << lane;"]
    for t in range(ndl):
        if cl:
            o.append(f"  const int c{t} = {pfx}_cb[lane + {32 * t}], q{t} = {pfx}_kdl[lane + {32 * t}];"
                     "  // the row's start on the closure, its depth")
        else:
            o.append(f"  const int p{t} = {pfx}_pv[lane + {32 * t}], o{t} = p{t} * (p{t} + 1) / 2;  // the row's dof, TRI(dof, 0)")
    for t in range(ndl):
        o.append(f"  for (int k = {32 * t}; k <= {min(32 * t + 31, nv - 1)}; k++) {{  // R' z = b: the pivots of slot {t}")
        if cl:
            o.append(f"    const int ck = {pfx}_cbk[k];")
            o.append(f"    const float d = L[ck + {pfx}_kd[k]], y_ = __shfl_sync({F}, y[{t}], k & 31);")
        else:
            o.append(f"    const int pk = {pfx}_pk[k], tk = pk * (pk + 1) / 2;")
            o.append(f"    const float d = L[tk + pk], y_ = __shfl_sync({F}, y[{t}], k & 31);")
        o.append(f"    const float z = {div};")
        o.append(f"    if (lane == (k & 31)) y[{t}] = z;")
        for u in range(t, ndl):
            ent = f"L[ck + q{u}]" if cl else f"L[pk > p{u} ? tk + p{u} : o{u} + pk]"
            o.append(f"    if ({pfx}_anc[k * {ndl} + {u}] & bit) y[{u}] -= {ent} * z;")
        o.append("  }")
    o.append("  for (int q = 0; q < NV; q++) {  // R x = z, root first")
    ysel = " : ".join([f"si == {t} ? y[{t}]" for t in range(ndl - 1)] + [f"y[{ndl - 1}]"])
    if cl:
        o.append(f"    const int i = {pfx}_rf[q], li = i & 31, si = i >> 5, ki = {pfx}_kd[i];")
        o.append(f"    const float d = L[{pfx}_cbk[i] + ki], y_ = __shfl_sync({F}, {ysel}, li);")
    else:
        o.append(f"    const int i = {pfx}_rf[q], li = i & 31, si = i >> 5, pi = {pfx}_pk[i], ti = pi * (pi + 1) / 2;")
        o.append(f"    const float d = L[ti + pi], y_ = __shfl_sync({F}, {ysel}, li);")
    o.append(f"    const float x = {div};")
    for t in range(ndl):
        o.append(f"    if (lane == li && si == {t}) y[{t}] = x;")
        ent = f"L[c{t} + ki]" if cl else f"L[p{t} > pi ? o{t} + pi : ti + p{t}]"
        o.append(f"    if ({pfx}_desc[q * {ndl} + {t}] & bit) y[{t}] -= {ent} * x;")
    o.append("  }")
    o.append("}")
    if multi:
        o += _fsolve_back(nv, ndl, pfx, div, F, ysel)
    if name == "chol_resolve_hp":
        o += ["#if NV128_REUSE", f"__device__ __forceinline__ void {name}(const float* L, float* b, int lane) {{", "#else"]
    o.append(f"__device__ __noinline__ void {name}(const float* L, float* b, int lane) {{")
    if name == "chol_resolve_hp":
        o.append("#endif")
    o.append(f"  float y[{ndl}];")
    o.append("  __syncwarp();")
    for t in range(ndl):
        o.append(f"  y[{t}] = lane + {32 * t} < NV ? b[{pfx}_pv[lane + {32 * t}]] : 0.f;")
    o.append(f"  {pfx}_vsolve(L, y, lane);")
    o.append("  __syncwarp();")
    for t in range(ndl):
        o.append(f"  if (lane + {32 * t} < NV) b[{pfx}_pv[lane + {32 * t}]] = y[{t}];")
    o.append("  __syncwarp();")
    o.append("}")
    return "\n".join(o) + "\n"


def _fsolve_back(nv, ndl, pfx, div, F, ysel):
    o = []
    o.append(f"__device__ __forceinline__ void {pfx}_fsolve_m(const float* L, float (&V)[HW_COLS][{ndl}], int nc, int lane) {{")
    o.append("  const unsigned bit = 1u << lane;")
    for t in range(ndl):
        o.append(f"  const int p{t} = {pfx}_pv[lane + {32 * t}], o{t} = p{t} * (p{t} + 1) / 2;")
    for t in range(ndl):
        o.append(f"  for (int k = {32 * t}; k <= {min(32 * t + 31, nv - 1)}; k++) {{  // the pivots of slot {t}")
        o.append(f"    const int pk = {pfx}_pk[k], tk = pk * (pk + 1) / 2;")
        o.append("    const float inv = __frcp_rn(L[tk + pk]);")
        o.append("    float zz[HW_COLS];")
        o.append("#pragma unroll")
        o.append("    for (int a = 0; a < HW_COLS; a++) {")
        o.append(f"      zz[a] = a < nc ? __shfl_sync({F}, V[a][{t}], k & 31) * inv : 0.f;")
        o.append(f"      if (lane == (k & 31)) V[a][{t}] = zz[a];")
        o.append("    }")
        for u in range(t, ndl):
            o.append(f"    if ({pfx}_anc[k * {ndl} + {u}] & bit) {{")
            o.append(f"      const float l = L[pk > p{u} ? tk + p{u} : o{u} + pk];")
            o.append("#pragma unroll")
            o.append(f"      for (int a = 0; a < HW_COLS; a++) if (a < nc) V[a][{u}] -= l * zz[a];")
            o.append("    }")
        o.append("  }")
    o.append("}")
    o.append(f"__device__ __noinline__ void {pfx}_back(const float* L, float* b, int lane) {{")
    o.append("  const unsigned bit = 1u << lane;")
    o.append(f"  float y[{ndl}];")
    o.append("  __syncwarp();")
    for t in range(ndl):
        o.append(f"  const int p{t} = {pfx}_pv[lane + {32 * t}], o{t} = p{t} * (p{t} + 1) / 2;")
        o.append(f"  y[{t}] = lane + {32 * t} < NV ? b[p{t}] : 0.f;")
    o.append("  for (int q = 0; q < NV; q++) {")
    o.append(f"    const int i = {pfx}_rf[q], li = i & 31, si = i >> 5, pi = {pfx}_pk[i], ti = pi * (pi + 1) / 2;")
    o.append(f"    const float d = L[ti + pi], y_ = __shfl_sync({F}, {ysel}, li);")
    o.append(f"    const float x = {div};")
    for t in range(ndl):
        o.append(f"    if (lane == li && si == {t}) y[{t}] = x;")
        o.append(f"    if ({pfx}_desc[q * {ndl} + {t}] & bit) y[{t}] -= L[p{t} > pi ? o{t} + pi : ti + p{t}] * x;")
    o.append("  }")
    o.append("  __syncwarp();")
    for t in range(ndl):
        o.append(f"  if (lane + {32 * t} < NV) b[p{t}] = y[{t}];")
    o.append("  __syncwarp();")
    o.append("}")
    return o


CHOL_RESOLVE_DENSE = r"""// (nv128 stage 4) chol_resolve_dense(L, b, lane): b := (L L')^-1 b with chol_solve_dense's factor in L (its two
// triangular solves, the same text: NEWTON_REUSE's analog for the dense worlds)
// NV128_REUSE: chol_resolve_dense and chol_resolve_hp inlined. k_post calls them from its SimN and its Hand::big path;
// as calls, the pattern ptxas miscompiled in chol_resolve could not be ruled out for every
// target (forest_engine.chol_blk: chol_resolve_blk)
#define DFMS(a, b, c) __fmaf_rn(-(a), (b), (c))
#if NV128_REUSE
__device__ __forceinline__ void chol_resolve_dense(const float* L, float* b, int lane) {
#else
__device__ __noinline__ void chol_resolve_dense(const float* L, float* b, int lane) {
#endif
  __syncwarp();
""" + DEVICE[DEVICE.index("  float y[NDL];\n"):DEVICE.index("#undef DFMS")] + "#undef DFMS\n"

WOODBURY = r"""
// (nv128 stage 4) the Newton Hessian of a world with contacts of dynamic pairs (pairs whose support is not a clique of
// P_H, nvh_pdyn): H = R' R + sum_c U_c C_c U_c', R' R = H_P assembled without them (hessian(.., hskip)) and factored by
// chol_solve_hp, U_c = the contact's frame rows (normal, tangents[, spin]) as dof vectors, C_c its CD x CD block of
// con_C (positive semidefinite). hp_lowrank (between chol_solve_hp's two triangular solves): H = R' (I + V C V') R,
// V = R'^-1 U, so H^-1 g = R^-1 (w - V (I + C W)^-1 C V' w) with w = R'^-1 g, W = V' V (the Sherman-Morrison-Woodbury
// identity; I + C W is nonsingular). At most HW_COLS columns (hp_count * CD); beyond, the dense factor of the full H.
template <class S>
__device__ __forceinline__ bool hp_dyn_on(const S& s, int c) {  // a contact of a dynamic pair with a quadratic row
  const int pr = s.con_pair[c];
  if (pr >= NPAIR || !((nvh_pdyn[pr >> 5] >> (pr & 31)) & 1u)) return false;
  const int r0 = s.con_row[c], nr = s.con_nrow[c];
  bool on = false;
  for (int k = 0; k < nr; k++) on |= s.efc_jar[r0 + k] < 0.f && s.efc_D[r0 + k] != 0.f;
  return on;
}
template <class S>
__device__ __forceinline__ int hp_count(const S& s, int ncon, int lane) {  // (uniform)
  int n = 0;
  for (int c0 = 0; c0 < ncon; c0 += 32) n += __popc(__ballot_sync(0xffffffffu, c0 + lane < ncon && hp_dyn_on(s, c0 + lane)));
  return n;
}
template <class S>
__device__ __forceinline__ void hp_ucol(const S& s, int c, int j, float (&u)[NDL], int lane) {
  // column j of contact c's frame rows in the vsolve layout (lane l: the dofs of kernel rows l, l + 32, ..)
  const DofMask mk = s.con_mask[c];
#pragma unroll
  for (int t = 0; t < NDL; t++) {
    const int d = chol_resolve_hp_pv[lane + 32 * t];
    u[t] = 0.f;
    if (lane + 32 * t < NV && ((mk.w[d >> 5] >> (d & 31)) & 1u)) {
      int k = __popc(mk.w[d >> 5] & ((1u << (d & 31)) - 1u));
#pragma unroll
      for (int w = 0; w < NDL; w++) if (w < (d >> 5)) k += __popc(mk.w[w]);
      u[t] = s.con_J[c][j][k];
    }
  }
}
template <class S>
__device__ __noinline__ void hp_lowrank(S& s, int ncon, int lane) {
  // H = R' R + U C U' = R' (I + V C V') R with V = R'^-1 U: on entry s.p = w = R'^-1 g (chol_solve_hp without its back
  // solve); y = (I + V C V')^-1 w = w - V (I + C W)^-1 C V' w with W = V' V (Woodbury), to s.p; the caller's R x = y
  // finishes p = H^-1 g. One forward pass for all columns (fsolve_m), W symmetric by construction
  constexpr int NW = HW_COLS / CD;  // contacts
  int cid[NW];
  int n = 0;
  for (int c0 = 0; c0 < ncon; c0 += 32)
    for (unsigned b = __ballot_sync(0xffffffffu, c0 + lane < ncon && hp_dyn_on(s, c0 + lane)); b; b &= b - 1u)
      if (n < NW) cid[n++] = c0 + __ffs(b) - 1;
  const int nc = n * CD;
  float V[HW_COLS][NDL], w[NDL];
#pragma unroll
  for (int a = 0; a < HW_COLS; a++) {
    if (a < nc) hp_ucol(s, cid[a / CD], a % CD, V[a], lane);
    else {
#pragma unroll
      for (int t = 0; t < NDL; t++) V[a][t] = 0.f;
    }
  }
  chol_resolve_hp_fsolve_m(s.L, V, nc, lane);
  __syncwarp();
#pragma unroll
  for (int t = 0; t < NDL; t++) w[t] = lane + 32 * t < NV ? s.p[chol_resolve_hp_pv[lane + 32 * t]] : 0.f;
  // this lane's row r of (I + C W | C V' w) (lanes < nc; C = the row's contact block: [C0 C1 C3 (C5); C1 C2 0 (0);
  // C3 0 C4 (0); (C5 0 0 C6)]), accumulated as the entries of W and V' w arrive (warp sums)
  float cr[CD];
  {
    const int q = lane % CD;
    const float* C = s.con_C[nc ? cid[min(lane, nc - 1) / CD] : 0];
    cr[0] = q == 0 ? C[0] : q == 1 ? C[1] : q == 2 ? C[3] : C[CC - 2];
    cr[1] = q == 0 ? C[1] : q == 1 ? C[2] : 0.f;
    cr[2] = q == 0 ? C[3] : q == 2 ? C[4] : 0.f;
#if HAS_CONDIM4
    cr[3] = q == 0 ? C[5] : q == 3 ? C[6] : 0.f;
#endif
  }
  const int blk = lane / CD;
  float Ar[HW_COLS + 1];
#pragma unroll
  for (int b = 0; b <= HW_COLS; b++) Ar[b] = b == lane ? 1.f : 0.f;
#pragma unroll
  for (int a = 0; a < HW_COLS; a++) {
    if (a >= nc) break;
    float acc = 0.f;
#pragma unroll
    for (int t = 0; t < NDL; t++) acc += V[a][t] * w[t];
    acc = warp_sum(acc);
    if (blk == a / CD) Ar[HW_COLS] += cr[a % CD] * acc;  // (C V' w)_r
#pragma unroll
    for (int b = 0; b <= a; b++) {  // W_ab = W_ba
      float wab = 0.f;
#pragma unroll
      for (int t = 0; t < NDL; t++) wab += V[a][t] * V[b][t];
      wab = warp_sum(wab);
      if (blk == a / CD) Ar[b] += cr[a % CD] * wab;
      if (b != a && blk == b / CD) Ar[a] += cr[b % CD] * wab;
    }
  }
#pragma unroll
  for (int b = 0; b <= HW_COLS; b++) Ar[b] = lane < nc ? Ar[b] : 0.f;
  // Gaussian elimination with partial pivoting, row r on lane r
#pragma unroll
  for (int k = 0; k < HW_COLS; k++) {
    if (k >= nc) break;
    float v = lane >= k && lane < nc ? fabsf(Ar[k]) : -1.f;
    int piv = lane;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {  // argmax (the smaller lane on ties: the same on every lane)
      const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
      const int p2 = __shfl_xor_sync(0xffffffffu, piv, o);
      if (v2 > v || (v2 == v && p2 < piv)) { v = v2; piv = p2; }
    }
    const int src = lane == k ? piv : lane == piv ? k : lane;
#pragma unroll
    for (int j = k; j <= HW_COLS; j++) Ar[j] = __shfl_sync(0xffffffffu, Ar[j], src);
    const float pv = __shfl_sync(0xffffffffu, Ar[k], k);  // (every lane: a shuffle inside the select hung the warp)
    const float f = lane > k && lane < nc ? Ar[k] / pv : 0.f;
#pragma unroll
    for (int j = k; j <= HW_COLS; j++) Ar[j] -= f * __shfl_sync(0xffffffffu, Ar[j], k);
  }
  float x[HW_COLS];
#pragma unroll
  for (int k = HW_COLS - 1; k >= 0; k--) {
    x[k] = 0.f;
    if (k >= nc) continue;
    float acc = __shfl_sync(0xffffffffu, Ar[HW_COLS], k);
#pragma unroll
    for (int j = k + 1; j < HW_COLS; j++) acc -= __shfl_sync(0xffffffffu, Ar[j], k) * x[j];
    x[k] = acc / __shfl_sync(0xffffffffu, Ar[k], k);
  }
#pragma unroll
  for (int t = 0; t < NDL; t++) {
    float v = w[t];
#pragma unroll
    for (int a = 0; a < HW_COLS; a++) v -= V[a][t] * x[a];
    if (lane + 32 * t < NV) s.p[chol_resolve_hp_pv[lane + 32 * t]] = v;
  }
  __syncwarp();
}
// the per-world state of the Newton loop in one int (k_post sits at its register cap: no further live values there):
// bits 0-1 the form of the last factor (0 P_H, 1 P_H + the low-rank correction, 2 dense), bit 2 a low-rank iteration
// with columns happened, bit 3 a dense one (outside the tape's re-factorization), bit 4 the last factor has correction
// columns (then the tape's final factor is re-assembled: the full H, dense). The form: dense for the tape's
// re-factorization of a world with contacts of dynamic pairs (refact) and beyond HW_COLS correction columns
template <class S>
__device__ __noinline__ int hp_mode(const S& s, int ncon, bool refact, int hst, int lane) {
  const int nd = hp_count(s, ncon, lane);
  int md = nd == 0 ? 0 : (nd * CD <= HW_COLS && !refact && NV128_HFORCE != 1) ? 1 : 2;
  if (NV128_HFORCE == 2) md = 0;  // (diagnosis: every world on P_H, H wrong where contacts are off it)
  if (NV128_HFORCE == 3) md = 2;  // (diagnosis: every world dense)
  if (NV128_HFORCE == 4 && md == 0) md = 1;  // (every world through hp_lowrank, no columns where none is needed)
  if (NV128_HFORCE == 5 && nd > 0 && !refact) md = 3;  // (H_P's direction, the contacts off P_H left out: a
                                                        // preconditioned step; the tape's factor stays the full H's)
  const bool cols = (md == 1 && nd > 0) || md == 3;  // (this factor is not the full H's: the tape needs it)
  return (hst & 12) | md | (cols ? 4 | 16 : 0) | (md == 2 && !refact ? 8 : 0);
}
// the Newton direction p = -H^-1 grad in the form md (one call site in the loop, as CHOL_H_SOLVE)
template <class S>
__device__ __noinline__ void hp_newton_dir(S& s, int ncon, int md, int lane) {
  md = uniform_int(md, 2);  // (a vote: the branches below are uniform; md 0 .. 3)
  for (int i = lane; i < NV; i += 32) s.p[i] = -s.grad[i];
  __syncwarp();
  if (md == 2) chol_solve_dense(s.L, s.L, s.p, lane);
  else if (md != 1) chol_solve_hp(s.L, s.L, s.p, lane);  // (H (form 0) or H_P (3) = R' R on P_H's elimination tree)
  else {  // H = R' R + U C U': R'^-1 g, the low-rank step, R^-1
    chol_solve_hp(s.L, s.L, s.p, lane, false);
    hp_lowrank(s, ncon, lane);
    chol_resolve_hp_back(s.L, s.p, lane);
  }
}
"""

H_CALL = ("      hessian(s, ncon, world, grp, lane);\n      PROF(7);\n"
          "      for (int i = lane; i < NV; i += 32) s.p[i] = -s.grad[i];\n      __syncwarp();\n"
          "      CHOL_H_SOLVE(s.L, s.L, s.p, lane, s.cbuf);\n")
H_CALL_NEW = ("      hst = hp_mode(s, ncon, refact, hst, lane);  // (nv128 stage 4: the factor's form, hp_mode)\n"
              "      hessian(s, ncon, world, grp, lane, (hst & 1) != 0);  // (forms 1, 3: H_P)\n      PROF(7);\n"
              "      hp_newton_dir(s, ncon, hst & 3, lane);\n")
H_REUSE = ("#if NEWTON_REUSE || NEWTON_REUSE_WIDE\n"
           "    if (zsame && !refact) {  // no row changed its zone in the last step: H and its factor in s.L are the same\n"
           "      for (int i = lane; i < NV; i += 32) s.p[i] = -s.grad[i];\n      __syncwarp();\n"
           "      chol_resolve(s.L, s.p, lane, s.cbuf);\n    } else\n#endif\n")
H_REUSE_NEW = ("#if NV128_REUSE\n"
               "    if (zsame && !refact && UNIFORM((hst & 3) != 1)) {  // (nv128 stage 4) no row changed its zone: the same H, its\n"
               "      for (int i = lane; i < NV; i += 32) s.p[i] = -s.grad[i];  // factor in s.L (not with the low-rank\n"
               "      __syncwarp();                                             // correction: con_C is gone)\n"
               "      if (UNIFORM((hst & 3) == 2)) chol_resolve_dense(s.L, s.p, lane);\n      else chol_resolve_hp(s.L, s.p, lane);\n"
               "    } else\n#endif\n")
H_LOOP = "  bool conv = false, zchanged = rec && iterations <= 0, refact = zchanged, zsame = false;\n"
H_LOOP_NEW = H_LOOP + "  int hst = 2;  // (nv128 stage 4) the factor's form and the forms used (hp_mode)\n"
H_REFACT = "        refact = zchanged;\n"
H_REFACT_NEW = ("        refact = zchanged || UNIFORM((hst & 16) != 0);  // (nv128 stage 4: L_H of the full H for the tape: the low-rank\n"
                "        // correction's world is re-assembled and factored densely)\n")
H_SIG = ("__device__ __forceinline__ void hessian(S& s, int ncon, const unsigned (&world)[CPL], const unsigned (&grp)[CPL],\n"
         "                                        int lane) {\n")
H_SIG_NEW = ("__device__ __forceinline__ void hessian(S& s, int ncon, const unsigned (&world)[CPL], const unsigned (&grp)[CPL],\n"
             "                                        int lane, bool hskip = false) {  // (nv128 stage 4: hskip, the\n"
             "  // contacts of dynamic pairs left out: H_P for hp_lowrank)\n")
H_ON = "      on = s.con_C[c][0] != 0.f;  // C_nn is the sum of the curvatures\n"
H_ON_NEW = ("      on = s.con_C[c][0] != 0.f && !(hskip && s.con_pair[c] < NPAIR\n"
            "                                     && ((nvh_pdyn[s.con_pair[c] >> 5] >> (s.con_pair[c] & 31)) & 1u));\n")


def hreuse():
    """$DK_NV128_REUSE at source generation (the sparse Newton factor builds): 1 re-uses the last Newton factor when no row changed its
    zone in the last step (NEWTON_REUSE's analog: chol_resolve_hp / chol_resolve_dense; not after the low-rank
    correction), 0 (default) refactors."""
    import os
    raw = os.environ.get("DK_NV128_REUSE", "") or "0"
    if raw not in ("0", "1"):
        raise ValueError(f"DK_NV128_REUSE={raw!r}: 0 or 1")
    return int(raw)


def hw_cols():
    """hp_lowrank's column capacity: HW_COLS, or $DK_NV128_HWCOLS (a multiple of CD; diagnosis)."""
    import os
    return int(os.environ.get("DK_NV128_HWCOLS", "") or HW_COLS)


HFORCE = {"": 0, "dense": 1, "sparse_all": 2, "dense_all": 3, "lowrank_all": 4, "hp_only": 5}


def hforce():
    """$DK_NV128_HFORCE (timing diagnosis of DK_NV128_HCHOL=sparse builds only; the sparse Newton factor):
    dense (a world with a contact of a dynamic pair takes the dense factor instead of the low-rank correction),
    sparse_all (every world on P_H: H wrong where contacts are off it), dense_all (every world dense), lowrank_all
    (every world through hp_lowrank), hp_only (H_P's direction without the correction: inexact); unset: hp_mode's
    choice."""
    import os
    return HFORCE[os.environ.get("DK_NV128_HFORCE", "")]


def pdyn_table(pdyn, npair):
    words = [sum(1 << (b & 31) for b in range(32 * w, min(32 * w + 32, npair)) if pdyn[b]) for w in range((npair + 31) // 32)]
    return (f"// (nv128 stage 4) the pairs whose support is not a clique of P_H (forest_engine.nv128.hpattern)\n"
            f"__constant__ unsigned nvh_pdyn[{max(len(words), 1)}] = {{{', '.join(map(str, words or [0]))}}};\n"
            "#ifndef NV128_REUSE  // (nv128 stage 4) re-use the last Newton factor when no row changed its zone\n"
            f"#define NV128_REUSE {hreuse()}\n#endif\n"
            f"#define NV128_HFORCE {hforce()}  // (diagnosis: 1 the dense factor instead of the low-rank correction)\n"
            f"#define HW_COLS {hw_cols()}  // hp_lowrank's columns at most (nv128.hw_cols)\n")


def apply_hsparse(src, m, info, st=None):
    """the sparse Newton factor's edits (hsparse(m)): the Newton Hessian factored by chol_solve_hp (perm_chol_loop on P_H's elimination
    tree); a world with contacts of dynamic pairs in H gets H_P (hessian's hskip) and hp_lowrank's low-rank correction
    (up to HW_COLS columns; beyond, chol_solve_dense of the full H); the tape's final factor is on P_H or (such a world)
    dense, its form in the record (tape.FLAG_LH_SPARSE), the forms the iterations used in FLAG_H_LOWRANK / FLAG_H_DENSE;
    the adjoint's lam solve by the record's form (hp_solve_packed or tri_solve_packed); NV128_REUSE (NEWTON_REUSE's
    analog, $DK_NV128_REUSE). the pattern storage (st["lh"]: no dynamic pair): the factor stored on the closure (clayout) in
    SimN / Sim (struct HL), k_post's factor of M + h kv there too (chol_solve_tree_c), the record's L_H and the adjoint's
    copy as stored (S5_LH)."""
    from nablaengine.forest_engine.tape import FLAG_H_DENSE, FLAG_H_LOWRANK, FLAG_LH_SPARSE, FLAG_LQ_TREE
    st = st or storage(m)
    cl = st["lh"]
    v = ".v" if cl else ""
    perm, kpar = info["perm"], info["kpar"]
    anchor = "#define CHOL_M_SOLVE(A, L, b, lane, cb) chol_solve_tree(A, L, b, lane)"
    if src.count(anchor) != 1:
        raise RuntimeError("nv128: CHOL_M_SOLVE's tree-factor definition not found once")
    if not info["npair_dyn"]:
        flag = f"                    | {FLAG_LQ_TREE}u;  // (nv128 stage 3: L_Q in tree form, Q = R' R)\n"
        dev = (f"#ifndef NV128_REUSE  // (nv128 stage 4) re-use the last Newton factor when no row changed its zone\n"
               f"#define NV128_REUSE {hreuse()}\n#endif\n" + perm_chol_loop(kpar, perm, cl=cl)
               + perm_solve_loop(kpar, perm, "chol_resolve_hp", "__fdiv_rn(y_, d)", cl=cl))
        reuse = ("#if NV128_REUSE\n"
                 "    if (zsame && !refact) {  // (nv128 stage 4) no row changed its zone: the same H, its factor in s.L\n"
                 "      for (int i = lane; i < NV; i += 32) s.p[i] = -s.grad[i];\n      __syncwarp();\n"
                 f"      chol_resolve_hp(s.L{v}, s.p, lane);\n    }} else\n#endif\n")
        src = _apply(src, ((anchor, dev + anchor, 1), (H_REUSE, reuse, 1),
                           ("      CHOL_H_SOLVE(s.L, s.L, s.p, lane, s.cbuf);\n",
                            f"      chol_solve_hp(s.L{v}, s.L{v}, s.p, lane);  // (nv128 stage 4: H = R' R on P_H's elimination tree)\n",
                            1),
                           (flag, f"                    | {FLAG_LQ_TREE}u | {FLAG_LH_SPARSE}u;  // (nv128 stages 3, 4: L_Q, L_H "
                            "in tree form)\n", 1)))
        if "__global__ void k_adjoint(" in src:
            anchor = "__device__ __noinline__ void adj_solves(Adj& s, const float* rec, int lane) {\n"
            src = _apply(src, ((anchor, perm_solve_loop(kpar, perm, "hp_solve_packed", cl=cl) + anchor, 1),
                               ("  tri_solve_packed(s.L, s.lam, lane);\n}\n",
                                "  hp_solve_packed(s.L, s.lam, lane);  // (nv128 stage 4: L_H = R, H = R' R)\n}\n", 1)))
        if cl:
            src = _apply(src, S5_LH)
            ten = "__device__ __forceinline__ void ten_outer(float* A, "
            if ten in src:
                src = _apply(src, ((ten, "template <class LA>  // (nv128 stage 5: float* or HL&)\n"
                                         "__device__ __forceinline__ void ten_outer(LA&& A, ", 1),))
        return src
    dev = (pdyn_table(info["pdyn"], int(m.npair)) + perm_chol_loop(kpar, perm)
           + perm_solve_loop(kpar, perm, "chol_resolve_hp", "__fdiv_rn(y_, d)", multi=True) + CHOL_RESOLVE_DENSE
           + WOODBURY)
    hanchor = "// the Newton Hessian into s.L (see below); world, grp:"
    flag = f"                    | {FLAG_LQ_TREE}u;  // (nv128 stage 3: L_Q in tree form, Q = R' R)\n"
    src = _apply(src, ((hanchor, dev + hanchor, 1), (H_SIG, H_SIG_NEW, 1), (H_ON, H_ON_NEW, 1),
                       (H_CALL, H_CALL_NEW, 1),
                       (H_REUSE, H_REUSE_NEW, 1), (H_LOOP, H_LOOP_NEW, 1), (H_REFACT, H_REFACT_NEW, 1),
                       (flag, f"                    | {FLAG_LQ_TREE}u  // (nv128 stage 3: L_Q in tree form, Q = R' R)\n"
                        f"                    | ((hst & 3) == 2 ? 0u : {FLAG_LH_SPARSE}u) | ((hst & 4) ? {FLAG_H_LOWRANK}u : 0u)\n"
                        f"                    | ((hst & 8) ? {FLAG_H_DENSE}u : 0u);  // (nv128 stage 4: L_H's form, the forms used)\n",
                        1)))
    if "__global__ void k_adjoint(" in src:
        anchor = "__device__ __noinline__ void adj_solves(Adj& s, const float* rec, int lane) {\n"
        call = "  tri_solve_packed(s.L, s.lam, lane);\n}\n"
        src = _apply(src, ((anchor, perm_solve_loop(kpar, perm, "hp_solve_packed") + anchor, 1),
                           (call, f"  if (__any_sync(0xffffffffu, (__float_as_uint(__ldcs(rec + OFF_FLAGS)) & {FLAG_LH_SPARSE}u) "
                            "!= 0u))  // (nv128 stage 4: the record's form of L_H)\n"
                            "    hp_solve_packed(s.L, s.lam, lane);\n  else\n    tri_solve_packed(s.L, s.lam, lane);\n}\n",
                            1)))
    return src


def _cbase(depth):
    out, k = [], 0
    for d in depth:
        out.append(k)
        k += d + 1
    return out + [k]


def tlayout(par):
    """M's tree pattern stored row by row (the pattern storage; the tape's L_Q, k_pre's factor of M): dof i's row holds its
    ancestors root first, then the diagonal, from tbase[i]. Returns dict(nnz, tbase, depth, idx(i, j) for j = i or an
    ancestor of i, tri: the packed-triangle index TRI(i, j) of every stored entry in storage order)."""
    T = tree(par)
    depth, anc = T["depth"], T["anc"]
    tb = _cbase(depth)
    tri = [_tri(i, a) for i in range(T["nv"]) for a in list(reversed(anc[i])) + [i]]
    return dict(nnz=tb[-1], tbase=tb[:-1], depth=depth, idx=lambda i, j: tb[i] + depth[j], tri=tri)


def clayout(info):
    """The closure of P_H's elimination tree stored row by row (the pattern storage; the Newton factor in k_post's / k_substep's
    shared memory, the adjoint's, the tape's L_H for models whose every pair is on P_H): kernel row r (dof perm[r],
    hpattern) holds its elimination-tree ancestors root first, then the diagonal, from cbase[r]. Returns dict(nnz,
    cbase, depth, idx(i, j) for two dofs one of which is the other's elimination-tree ancestor, tri: TRI of every
    stored entry in storage order, slot: per packed-triangle index the storage index, nnz (a spare slot that is
    never read) for the entries outside the closure)."""
    perm, kpar = info["perm"], info["kpar"]
    T = ktree(kpar)
    depth, anc = T["depth"], T["anc"]
    cb = _cbase(depth)
    kidx = {d: r for r, d in enumerate(perm)}
    ancs = [set(a) for a in anc]

    def idx(i, j):
        ri, rj = kidx[i], kidx[j]
        if ri > rj:
            ri, rj = rj, ri
        assert ri == rj or rj in ancs[ri], f"nv128 clayout: dofs {i}, {j} not on the closure"
        return cb[ri] + depth[rj]

    tri = [_ptri(perm[r], perm[a]) for r in range(T["nv"]) for a in list(reversed(anc[r])) + [r]]
    nv = len(perm)
    slot = [cb[-1]] * (nv * (nv + 1) // 2)
    for q, e in enumerate(tri):
        slot[e] = q
    return dict(nnz=cb[-1], cbase=cb[:-1], depth=depth, idx=idx, tri=tri, slot=slot)


def lq_compact(m):
    """the pattern storage: L_Q (the record's, k_pre's factor of M, the adjoint's copy) stored on M's tree pattern (tlayout):
    the tree factor's loop factor (sparse(m), $DK_NV128_MCHOL loop). The straight-line form (diagnosis) keeps the triangle."""
    return sparse(m) and mchol() == "loop"


def lh_compact(m):
    """the pattern storage: the Newton factor stored on the closure of P_H (clayout) in SimN / Sim / Adj and the record's L_H sized
    to it, for the builds where no world can leave P_H: the sparse Newton factor's P_H factor (hsparse) on a model whose every contact
    pair is on P_H (no dynamic pair: one factor code for every world, $DK_NV128_HCHOL auto or sparse). Builds with a
    dense fallback ($DK_NV128_HCHOL=sparse with dynamic pairs) or the dense factor keep the packed triangle: their
    records hold a world's dense L_H. (The contact surrogate's builds refuse such models: apply.)"""
    if not lq_compact(m) or hchol() == "dense":
        return False
    return hsparse(m) and not hpattern(m)["npair_dyn"]


def storage(m):
    """The pattern storage of model m (more than 64 dofs): dict(lq, lh: the forms above, nnz_lq, nnz_lh: the record's
    L_Q / L_H words, ntri, tl / cl: tlayout / clayout or None)."""
    ntri = m.nv * (m.nv + 1) // 2
    lq, lh = lq_compact(m), lh_compact(m)
    tl = tlayout([int(p) for p in m.dof_parentid]) if lq else None
    cl = clayout(hpattern(m)) if lh else None
    return dict(lq=lq, lh=lh, ntri=ntri, nnz_lq=tl["nnz"] if lq else ntri, nnz_lh=cl["nnz"] if lh else ntri,
                tl=tl, cl=cl)


def _tree_chol_loop_map(par, name, aidx, lidx, pfx):
    T = tree(par)
    nv, anc, depth, desc = T["nv"], T["anc"], T["depth"], T["desc"]
    ndl = (nv + 31) // 32
    rows = lambda t: [r for r in range(32 * t, min(32 * t + 32, nv))]
    D = [max(depth[r] for r in rows(t)) for t in range(ndl)]
    base = [sum(D[u] for u in range(t)) for t in range(ndl)]
    ntab = max(base[-1] + D[-1], 1)
    aoff, loff = [0] * (ntab * 32), [0] * (ntab * 32)
    for t in range(ndl):
        for r in rows(t):
            for e, c in enumerate(reversed(anc[r])):
                aoff[(base[t] + e) * 32 + (r & 31)] = aidx(r, c)
                loff[(base[t] + e) * 32 + (r & 31)] = lidx(r, c)
    same = aoff == loff
    assert max(depth) <= 32, "nv128 tree_chol_loop: at most 32 ancestors (one lane each)"
    dep = [depth[32 * t + l] if 32 * t + l < nv else -1 for t in range(ndl) for l in range(32)]
    ad = [aidx(32 * t + l, 32 * t + l) if 32 * t + l < nv else 0 for t in range(ndl) for l in range(32)]
    ld = [lidx(k, k) for k in range(nv)]
    ancm = [sum(1 << (i & 31) for i in anc[k] if i >> 5 == u) for k in range(nv) for u in range(ndl)]
    rf = T["root_first"]
    dm = [sum(1 << (j & 31) for j in desc[i] if j >> 5 == t) for i in rf for t in range(ndl)]
    lst = lambda v: ", ".join(map(str, v))
    F = "0xffffffffu"
    lo = f"{pfx}_aoff" if same else f"{pfx}_loff"
    o = [f"// (nv128 stage 5) {name}: forest_engine.nv128.tree_chol_loop with A and L stored on patterns",
         f"__device__ const unsigned short {pfx}_aoff[{ntab * 32}] = {{{lst(aoff)}}};"]
    if not same:
        o.append(f"__device__ const unsigned short {pfx}_loff[{ntab * 32}] = {{{lst(loff)}}};")
    o += [f"__device__ const signed char {pfx}_dep[{ndl * 32}] = {{{lst(dep)}}};",
          f"__device__ const unsigned short {pfx}_ad[{ndl * 32}] = {{{lst(ad)}}};",
          f"__constant__ unsigned short {pfx}_ld[{nv}] = {{{lst(ld)}}};",
          f"__constant__ unsigned char {pfx}_dk[{nv}] = {{{lst(depth)}}};",
          f"__constant__ unsigned {pfx}_anc[{nv * ndl}] = {{{lst(ancm)}}};",
          f"__constant__ unsigned char {pfx}_rf[{nv}] = {{{lst(rf)}}};",
          f"__constant__ unsigned {pfx}_desc[{nv * ndl}] = {{{lst(dm)}}};",
          f"__device__ __noinline__ void {name}(const float* A, float* L, float* b, int lane) {{",
          "  const unsigned bit = 1u << lane;",
          f"  const unsigned short* off = {pfx}_aoff + lane;",
          "  __syncwarp();"]
    for t in range(ndl):
        r = f"lane + {32 * t}"
        o.append(f"  const int d{t} = {pfx}_dep[{r}];  // the row's depth")
        regs = [f"a{t}_{e} = d{t} > {e} ? A[off[{(base[t] + e) * 32}]] : 0.f" for e in range(D[t])]
        o.append("  float " + ", ".join(regs + [f"g{t} = d{t} >= 0 ? A[{pfx}_ad[{r}]] : 0.f"]) + ";")
        o.append(f"  float y{t} = d{t} >= 0 ? b[{r}] : 0.f;")
    o.append("  __syncwarp();  // (A and b read: L may alias A)")
    for t in reversed(range(ndl)):
        o.append(f"  for (int k = {min(32 * t + 31, nv - 1)}; k >= {32 * t}; k--) {{  // the pivots of slot {t}")
        o.append("    const int sl = k & 31;")
        o.append(f"    const float dkk = fmaxf(__shfl_sync({F}, g{t}, sl), 1e-30f);")
        o.append("    float inv;")
        o.append('    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(dkk));')
        o.append("    inv = TMUL(inv, TFMS(TMUL(TMUL(0.5f, dkk), inv), inv, 1.5f));")
        o.append(f"    const float z = TMUL(__shfl_sync({F}, y{t}, sl), inv);")
        o.append("    float w = 0.f;  // r_lane")
        for e in range(D[t]):
            o.append(f"    const float r{e} = TMUL(__shfl_sync({F}, a{t}_{e}, sl), inv);  // (0 from depth k on)")
            o.append(f"    if (lane == {e}) w = r{e};")
        o.append(f"    if (lane < {pfx}_dk[k]) L[{lo}[({base[t]} + lane) * 32 + sl]] = w;")
        o.append(f"    if (lane == sl) {{ L[{pfx}_ld[k]] = g{t} = TMUL(dkk, inv); y{t} = z; }}")
        for u in range(t + 1):
            o.append(f"    {{ const unsigned am = {pfx}_anc[k * {ndl} + {u}];")
            o.append(f"      if (am) {{ const float c = __shfl_sync({F}, w, d{u} & 31);")
            o.append("        if (am & bit) {")
            for e in range(min(D[u], D[t])):
                o.append(f"          a{u}_{e} = TFMS(c, r{e}, a{u}_{e});")
            o.append(f"          g{u} = TFMS(c, c, g{u}); y{u} = TFMS(c, z, y{u});")
            o.append("        } } }")
        o.append("  }")
    o.append("  __syncwarp();")
    o.append("  const float " + ", ".join(f"rq{t} = __frcp_rn(d{t} >= 0 ? g{t} : 1.f)" for t in range(ndl))
             + ";  // (stage 6: the reciprocals of the diagonal before the chain)")
    o.append("  for (int q = 0; q < NV; q++) {  // R x = z, root first")
    o.append(f"    const int i = {pfx}_rf[q], li = i & 31, si = i >> 5, di = {pfx}_dk[i];")
    ysel = " : ".join([f"si == {t} ? y{t}" for t in range(ndl - 1)] + [f"y{ndl - 1}"])
    csel = " : ".join([f"si == {t} ? rq{t}" for t in range(ndl - 1)] + [f"rq{ndl - 1}"])
    o.append(f"    const float a_ = __shfl_sync({F}, {ysel}, li), r_ = __shfl_sync({F}, {csel}, li);")
    o.append(f"    const float x = {_mk_div('a_', 'r_', f'L[{pfx}_ld[i]]')};")
    for t in range(ndl):
        o.append(f"    if (lane == li && si == {t}) y{t} = x;")
        o.append(f"    if ({pfx}_desc[q * {ndl} + {t}] & bit) y{t} = TFMS(L[{lo}[({base[t]} + di) * 32 + lane]], x, y{t});")
    o.append("  }")
    o.append("  __syncwarp();")
    for t in range(ndl):
        o.append(f"  if (d{t} >= 0) b[lane + {32 * t}] = y{t};")
    o.append("  __syncwarp();")
    o.append("}")
    return ("#define TFMS(a, b, c) __fmaf_rn(-(a), (b), (c))\n#define TMUL(a, b) __fmul_rn((a), (b))\n"
            + "\n".join(o) + "\n#undef TFMS\n#undef TMUL\n")


def tree_solve_t(par, name="tree_solve_packed"):
    """tree_solve with L stored on M's tree pattern (the pattern storage: the record's L_Q in tlayout): the same straight-line
    solve, the same operations; the entry (k, the lane's row c) of R' z = b at tbase[k] + depth[c] (dl<t>: the lane's
    row's depth), the entry (the lane's row r, i) of R x = z at tbase[r] + depth[i] (tb<t>: the lane's row start)."""
    T = tree(par)
    nv, anc, desc, depth = T["nv"], T["anc"], T["desc"], T["depth"]
    L = tlayout(par)
    tb = L["tbase"]
    ndl = (nv + 31) // 32

    def mask(rs, t):
        return sum(1 << (r & 31) for r in rs if r >> 5 == t)

    lst = lambda v: ", ".join(map(str, v))
    F = "0xffffffffu"
    o = [f"// (nv128 stages 3, 5) {name}: see forest_engine.nv128.tree_solve_t",
         f"__device__ const unsigned short nvqa_tb[{ndl * 32}] = "
         f"{{{lst([tb[32 * t + l] if 32 * t + l < nv else 0 for t in range(ndl) for l in range(32)])}}};",
         f"__device__ const unsigned char nvqa_dl[{ndl * 32}] = "
         f"{{{lst([depth[32 * t + l] if 32 * t + l < nv else 0 for t in range(ndl) for l in range(32)])}}};",
         f"__device__ __noinline__ void {name}(const float* L, float* b, int lane) {{",
         "  const unsigned bit = 1u << lane;",
         "  __syncwarp();"]
    for t in range(ndl):
        o.append(f"  float y{t} = lane + {32 * t} < NV ? b[lane + {32 * t}] : 0.f;")
        o.append(f"  const int tb{t} = nvqa_tb[lane + {32 * t}], dl{t} = nvqa_dl[lane + {32 * t}];")
    for k in T["leaves_first"]:
        sk, sl = k >> 5, k & 31
        o.append(f"  {{ const float d = L[{L['idx'](k, k)}];")
        o.append(f"    const float z = div_fast(__shfl_sync({F}, y{sk}, {sl}), d, rcp_step(d));")
        o.append(f"    if (lane == {sl}) y{sk} = z;")
        for t in range(ndl):
            mk = mask(anc[k], t)
            if mk:
                o.append(f"    if (0x{mk:08x}u & bit) y{t} -= L[{tb[k]} + dl{t}] * z;")
        o.append("  }")
    for i in T["root_first"]:
        si, li = i >> 5, i & 31
        o.append(f"  {{ const float d = L[{L['idx'](i, i)}];")
        o.append(f"    const float x = div_fast(__shfl_sync({F}, y{si}, {li}), d, rcp_step(d));")
        o.append(f"    if (lane == {li}) y{si} = x;")
        for t in range(ndl):
            mk = mask(desc[i], t)
            if mk:
                o.append(f"    if (0x{mk:08x}u & bit) y{t} -= L[tb{t} + {depth[i]}] * x;")
        o.append("  }")
    o.append("  __syncwarp();")
    for t in range(ndl):
        o.append(f"  if (lane + {32 * t} < NV) b[lane + {32 * t}] = y{t};")
    o.append("  __syncwarp();")
    o.append("}")
    return "\n".join(o) + "\n"


def tri_mul_rtr_t(par):
    """tri_mul_rtr (-DDK_ADJ_DEBUG's |M a*|) for L_Q = R stored on M's tree pattern (tlayout): lane 0 alone."""
    L = tlayout(par)
    T = tree(par)
    rows = [i for i in range(T["nv"]) for _ in range(T["depth"][i] + 1)]
    cols = [a for i in range(T["nv"]) for a in list(reversed(T["anc"][i])) + [i]]
    lst = lambda v: ", ".join(map(str, v))
    return (f"// (nv128 stage 5) out = R' (R x) for L_Q = R on M's tree pattern (-DDK_ADJ_DEBUG's |M a*|); tmp: NV scratch\n"
            f"__device__ const unsigned char nvqd_r[{L['nnz']}] = {{{lst(rows)}}};\n"
            f"__device__ const unsigned char nvqd_c[{L['nnz']}] = {{{lst(cols)}}};\n"
            "__device__ void tri_mul_rtr(const float* L, const float* x, float* tmp, float* out, int lane) {\n"
            "  __syncwarp();\n"
            "  if (lane == 0) {\n"
            "    for (int i = 0; i < NV; i++) tmp[i] = out[i] = 0.f;\n"
            f"    for (int q = 0; q < {L['nnz']}; q++) tmp[nvqd_r[q]] += L[q] * x[nvqd_c[q]];\n"
            f"    for (int q = 0; q < {L['nnz']}; q++) out[nvqd_c[q]] += L[q] * tmp[nvqd_r[q]];\n"
            "  }\n"
            "  __syncwarp();\n"
            "}\n")


def s5_header(m, st):
    """The pattern-storage definitions ahead of struct Sim: the words of the stored factors (NNZ_LQ, NNZ_LH, NNZ_ADJ), the
    table of L_Q's entries in k_post's storage of L (nvq_src: the tape's L_Q is gathered from it), tape_put_idx, LRAW,
    and with the Newton factor on the closure struct HL (its storage, entries addressed by the packed triangle's
    index through nvs_slot, so the Hessian assembly's code stays the same)."""
    tl, cl = st["tl"], st["cl"]
    lidx = cl["idx"] if st["lh"] else _tri
    par = [int(p) for p in m.dof_parentid]
    T = tree(par)
    src = [lidx(i, a) for i in range(m.nv) for a in list(reversed(T["anc"][i])) + [i]]
    lst = lambda v: ", ".join(map(str, v))
    o = ["// (nv128 stage 5) the factors on their patterns (forest_engine.nv128.storage): words of L_Q (M's tree pattern),",
         "// of the Newton factor (the closure of P_H, or the packed triangle) and of the adjoint's copy of either",
         f"#define NNZ_LQ {st['nnz_lq']}",
         f"#define NNZ_LH {st['nnz_lh']}",
         f"#define NNZ_ADJ {max(st['nnz_lq'], st['nnz_lh'])}",
         "// the stored entries of L_Q in substep_post's storage of L, in the record's order (tlayout)",
         f"__device__ const unsigned short nvq_src[{tl['nnz']}] = {{{lst(src)}}};",
         "__device__ __noinline__ void tape_put_idx(float* dst, const float* src, const unsigned short* idx, int n, int end,",
         "                                          int lane) {",
         "  for (int i = lane; i < end; i += 32) __stcs(dst + i, i < n ? src[idx[i]] : 0.f);",
         "}",
         "__device__ __forceinline__ float* LRAW(float* p) { return p; }"]
    if st["lh"]:
        o += ["// (nv128 stage 5) the Newton factor's storage on the closure of P_H (clayout): v[nvs_slot[e]] is the entry of",
              "// packed-triangle index e (the assembly's addressing; entries outside the closure go to the spare slot",
              "// NNZ_LH, never read); the factor's own code (chol_solve_hp, chol_solve_tree_c) addresses v directly",
              f"__device__ const unsigned short nvs_slot[NTRI] = {{{lst(cl['slot'])}}};",
              "// the packed-triangle index of every stored entry (the copies L = M)",
              f"__device__ const unsigned short nvc_tri[NNZ_LH] = {{{lst(cl['tri'])}}};",
              "struct HL {",
              "  float v[NNZ_LH + 1];",
              "  __device__ __forceinline__ float& operator[](int e) { return v[nvs_slot[e]]; }",
              "  __device__ __forceinline__ const float& operator[](int e) const { return v[nvs_slot[e]]; }",
              "};",
              "__device__ __forceinline__ float* LRAW(HL& h) { return h.v; }"]
    return "\n".join(o) + "\n"

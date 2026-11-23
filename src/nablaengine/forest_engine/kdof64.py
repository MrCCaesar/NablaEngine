"""Contact supports of more than 32 dofs in the forest engine (capability T11: at most MAX_KDOF)."""
import math

MAX_KDOF = 64
NOTE = "  // (kdof64: KDOF > 32)"


def needed(m):
    from nablaengine.forest_engine import step
    return step.max_support(m) > 32


def _sets_of_block(lo, hi, passes):
    out, cols, used = [], [], 0
    widths = [math.ceil((hi - max(lo, b)) / passes) for b in range(hi)]
    assert max(widths) <= 32
    chunks = []
    for b in range(hi):
        if used + widths[b] > 32:
            chunks.append(cols)
            cols, used = [], 0
        cols.append(b)
        used += widths[b]
    chunks.append(cols)
    for cols in chunks:
        tab = [[(0, 0, 255)] * 32 for _ in range(passes)]
        lane = 0
        for b in cols:
            rows = widths[b]
            for j in range(rows):
                for t in range(passes):
                    a = max(lo, b) + j + rows * t
                    tab[t][lane] = (a, b, a) if a < hi else (b, b, 255)
                lane += 1
        out.append((lo, hi, tab))
    return out


def hess_columns_wide(kdof, lo, passes=3):
    """step.hess_columns' sets from row lo on, where a row block has more than 32 columns: per block the row count
    (1 .. 3 passes' worth) with the most valid pairs per set (ties: the longer block)."""
    sets = []
    while lo < kdof:
        best = None
        for r in range(1, min(kdof - lo, 4 * passes) + 1):
            s = _sets_of_block(lo, lo + r, passes)
            pairs = sum(1 for _, _, tab in s for row in tab for _, _, thr in row if thr != 255)
            key = (pairs / len(s), r)
            if best is None or key > best[0]:
                best = (key, s, r)
        sets += best[1]
        lo += best[2]
    return sets


def check_sets(sets, kdof):
    """Every pair (a, b), b <= a < kdof, on exactly one (set, pass, lane); HSET_LO nondecreasing; rows of a set within
    [lo, hi); per set and lane one column."""
    seen = {}
    los = [lo for lo, _, _ in sets]
    assert los == sorted(los)
    for k, (lo, hi, tab) in enumerate(sets):
        for lane in range(32):
            bs = {tab[t][lane][1] for t in range(len(tab)) if tab[t][lane][2] != 255}
            assert len(bs) <= 1
            for t in range(len(tab)):
                a, b, thr = tab[t][lane]
                if thr == 255:
                    continue
                assert thr == a and lo <= a < hi and b <= a
                assert (a, b) not in seen, (a, b)
                seen[(a, b)] = (k, t, lane)
    assert set(seen) == {(a, b) for a in range(kdof) for b in range(a + 1)}
    return True


ENGINE = (
    ('  static_assert(KDOF <= 32, "hess_columns fits 32 lanes");\n',
     '  static_assert(KDOF <= 255, "hess_columns: 8-bit support positions");' + NOTE + '\n', 1),
    ("  for (int c = 0; c < ncon; c++) {  // scatter each contact over its support (one dof per lane)\n"
     "    if (lane < s.con_nd[c])\n"
     "      out[s.con_dof[c][lane]] += s.con_f[c][0] * s.con_J[c][0][lane] + s.con_f[c][1] * s.con_J[c][1][lane]\n"
     "                                 + s.con_f[c][2] * s.con_J[c][2][lane]\n"
     "#if HAS_CONDIM4\n"
     "                                 + s.con_f[c][3] * s.con_J[c][3][lane]\n"
     "#endif\n"
     "          ;\n",
     "  for (int c = 0; c < ncon; c++) {  // scatter each contact over its support (lane l: entries l, "
     "l + 32, ..)" + NOTE + "\n"
     "    for (int k = lane; k < s.con_nd[c]; k += 32)\n"
     "      out[s.con_dof[c][k]] += s.con_f[c][0] * s.con_J[c][0][k] + s.con_f[c][1] * s.con_J[c][1][k]\n"
     "                              + s.con_f[c][2] * s.con_J[c][2][k]\n"
     "#if HAS_CONDIM4\n"
     "                              + s.con_f[c][3] * s.con_J[c][3][k]\n"
     "#endif\n"
     "          ;\n", 1),
)


PREFETCH = ("  int ha[HSETS][3], hb[HSETS], hthr[HSETS][3];  // this lane's support pairs (ha, hb) of one column hb "
            "per set, the\n"
            "#pragma unroll                                   // same for every contact, valid when nd > hthr "
            "(step.hess_columns)\n"
            "  for (int k = 0; k < HSETS; k++)\n#pragma unroll\n    for (int t = 0; t < 3; t++) {\n"
            "      unsigned e = c_hcol[(3 * k + t) * 32 + lane];\n"
            "      ha[k][t] = e & 255; hb[k] = (e >> 8) & 255; hthr[k][t] = e >> 16;\n    }\n")
SET_AT = "      if (nd <= hlo[k]) break;  // no rows of this set in the support\n"
SET_LOAD = ("      int ha_k[3], hb_k = 0, hthr_k[3];  // this lane's pairs of set k (kdof64: loaded per set, not "
            "HSETS x 7 registers)\n#pragma unroll\n      for (int t = 0; t < 3; t++) {\n"
            "        const unsigned e = c_hcol[(3 * k + t) * 32 + lane];\n"
            "        ha_k[t] = e & 255; hb_k = (e >> 8) & 255; hthr_k[t] = e >> 16;\n      }\n")
HSET_EDITS = ((PREFETCH, "  // (kdof64: the support pairs (ha, hb, valid when nd > hthr) are loaded per set in the "
                         "assembly below)\n", 1),
              (SET_AT, SET_AT + SET_LOAD, 1), ("hb[k]", "hb_k", 2), ("ha[k][t]", "ha_k[t]", 2),
              ("hthr[k][t]", "hthr_k[t]", 2))


def apply(src, m):
    """src (adjoint.cuda_source's text) with the edits for a contact support of more than 32 dofs; unchanged
    otherwise."""
    if not needed(m):
        return src
    for a, b, n in ENGINE + HSET_EDITS:
        if src.count(a) != n:
            raise RuntimeError(f"kdof64 anchor found {src.count(a)} times, expected {n}: {a[:70]!r}")
        src = src.replace(a, b)
    return src

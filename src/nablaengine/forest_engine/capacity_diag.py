"""Per-world capacity diagnostics of the forest engine's forward (init(capacity_diag=True); off: the text unchanged)."""

INFO_WORDS = 6

_DECL = "  int cap_con, cap_rows, cap_xrows;  // (capacity_diag) contacts found, rows needed, dropped contacts' rows\n"

_PYRAMID_NEED = "    int need = ndof + ncon + 3 * ntri + 2 * nquad;\n"
_ELLIPTIC_NEED = "    int need = ndof + ncon + 2 * ntri + nquad;\n"
_ROWS = {
    "pyramid": "#define CAP_ROWS(d) ((d) >= 3 ? ((d) == 4 ? 6 : 4) : 1)",
    "elliptic": "#define CAP_ROWS(d) ((d) >= 3 ? ((d) == 4 ? 4 : 3) : 1)",
}

ADD_CONTACT = ("  int c = atomicAdd(&s.ncon, 1);\n  if (c >= MAXCON) return false;\n",
               "  int c = atomicAdd(&s.ncon, 1);\n"
               "  if (c >= MAXCON) {  // (capacity_diag) the dropped contact's rows\n"
               "    atomicAdd(&s.cap_xrows, CAP_ROWS(PDIM(pair)));\n    return false;\n  }\n")
ADD_CONTACT_HEAD = "template <class S>\n__device__ __forceinline__ bool add_contact(S& s, int pair, int slot, float dist,"
RESET = ("      s.ncon = 0;\n      s.min_nu_bits = 0x7f800000;  // +inf\n",
         "      s.ncon = 0;\n      s.cap_xrows = 0;  // (capacity_diag)\n      s.min_nu_bits = 0x7f800000;  // +inf\n")
NEED = ("    if (need > NEFC) s.unsupported |= 16;  // constraint rows beyond capacity are dropped\n",
        "    if (need > NEFC) s.unsupported |= 16;  // constraint rows beyond capacity are dropped\n"
        "    s.cap_con = max(s.cap_con, s.ncon);  // (capacity_diag) the substep's counts before truncation\n"
        "    s.cap_rows = max(s.cap_rows, need + s.cap_xrows);\n")
SIM_DECL = "  int ncon, nefc, unsupported;\n  int min_nu_bits;"
SIMP_DECL = ("  int ncon, nefc, unsupported, min_nu_bits, nefc_dof;\n"
             "  int con_pair[MAXCON], con_dim[MAXCON], con_row[MAXCON], con_nrow[MAXCON], con_nd[MAXCON];\n"
             "  float con_mu[MAXCON];\n  unsigned char con_slot[MAXCON];\n")
HAND_DECL = "  int ncon, nefc, nefc_dof, unsupported, min_nu_bits;\n"

HOST = (
    ("  if (lane == 0) { s.fric = fric[w]; s.unsupported = 0; }\n",
     "  if (lane == 0) { s.fric = fric[w]; s.unsupported = 0; s.cap_con = s.cap_rows = 0; }\n"),
    ("  if (lane == 0) { info[w * 4] = nit; info[w * 4 + 1] = s.ncon; info[w * 4 + 2] = s.nefc; info[w * 4 + 3] = s.unsupported; }\n",
     "  if (lane == 0) {\n"
     "    info[w * 6] = nit; info[w * 6 + 1] = s.ncon; info[w * 6 + 2] = s.nefc; info[w * 6 + 3] = s.unsupported;\n"
     "    info[w * 6 + 4] = s.cap_con; info[w * 6 + 5] = s.cap_rows;  // (capacity_diag)\n  }\n"),
    ("    s.unsupported = k ? g.unsupported : 0;  // accumulated over the substeps (the post half does not change it)\n",
     "    s.unsupported = k ? g.unsupported : 0;  // accumulated over the substeps (the post half does not change it)\n"
     "    s.cap_con = k ? g.cap_con : 0; s.cap_rows = k ? g.cap_rows : 0;  // (capacity_diag)\n"),
    ("    g.ncon = s.ncon; g.nefc = s.nefc; g.nefc_dof = s.nefc_dof; g.unsupported = s.unsupported;\n",
     "    g.ncon = s.ncon; g.nefc = s.nefc; g.nefc_dof = s.nefc_dof; g.unsupported = s.unsupported;\n"
     "    g.cap_con = s.cap_con; g.cap_rows = s.cap_rows;  // (capacity_diag)\n"),
    ("    info[w * 4] = nit; info[w * 4 + 1] = s.ncon; info[w * 4 + 2] = s.nefc; info[w * 4 + 3] = s.unsupported;\n",
     "    info[w * 6] = nit; info[w * 6 + 1] = s.ncon; info[w * 6 + 2] = s.nefc; info[w * 6 + 3] = s.unsupported;\n"
     "    info[w * 6 + 4] = g.cap_con; info[w * 6 + 5] = g.cap_rows;  // (capacity_diag)\n"),
)
HOST_DEFER = (
    ("    if (lane == 0) { s.ncon = g.ncon; s.min_nu_bits = g.min_nu_bits; s.unsupported = g.unsupported; }\n",
     "    if (lane == 0) {\n      s.ncon = g.ncon; s.min_nu_bits = g.min_nu_bits; s.unsupported = g.unsupported;\n"
     "      s.cap_xrows = g.cap_xrows;  // (capacity_diag)\n    }\n"),
    ("  if (lane == 0) { g.ncon = s.ncon; g.min_nu_bits = s.min_nu_bits; g.unsupported = s.unsupported; }\n",
     "  if (lane == 0) {\n    g.ncon = s.ncon; g.min_nu_bits = s.min_nu_bits; g.unsupported = s.unsupported;\n"
     "    g.cap_xrows = s.cap_xrows;  // (capacity_diag)\n  }\n"),
    ("    if (c >= MAXCON) { atomicOr(&g.unsupported, 4); continue; }\n",
     "    if (c >= MAXCON) {  // (capacity_diag: the dropped contact's rows)\n"
     "      atomicOr(&g.unsupported, 4);\n      atomicAdd(&g.cap_xrows, CAP_ROWS(PDIM(pair)));\n      continue;\n"
     "    }\n"),
)


def _sub(src, a, b, n=1, what=""):
    k = src.count(a)
    if k != n:
        raise RuntimeError(f"capacity_diag: {what or a[:60]!r} found {k} times (expected {n})")
    return src.replace(a, b)


def cone(src):
    """"pyramid" or "elliptic": the row count of the engine text's assembly (forest_engine.elliptic's edit)."""
    p, e = src.count(_PYRAMID_NEED), src.count(_ELLIPTIC_NEED)
    if (p, e) == (1, 0):
        return "pyramid"
    if (p, e) == (0, 1):
        return "elliptic"
    raise RuntimeError(f"capacity_diag: substep_pre's row count found {p} (pyramidal) / {e} (elliptic) times")


def engine(src):
    """The engine text (adjoint.cuda_source's, after every other edit) with the counters."""
    rows = _ROWS[cone(src)]
    src = _sub(src, SIM_DECL, _DECL + SIM_DECL, what="struct Sim's counts")
    src = _sub(src, SIMP_DECL, SIMP_DECL + _DECL, what="struct SimP's counts")
    src = _sub(src, HAND_DECL, HAND_DECL + _DECL, what="struct Hand's counts")
    src = _sub(src, ADD_CONTACT_HEAD, rows + "  // (capacity_diag)\n" + ADD_CONTACT_HEAD, what="add_contact")
    src = _sub(src, *ADD_CONTACT, what="add_contact's capacity check")
    src = _sub(src, *RESET, what="substep_pre's contact reset")
    src = _sub(src, *NEED, what="substep_pre's row count")
    return src


def host(src):
    """The host kernels' text (forest_engine.host.KERNEL after the other host edits) with the counts in info."""
    for a, b in HOST:
        src = _sub(src, a, b)
    if "k_ccd" in src:
        for a, b in HOST_DEFER:
            src = _sub(src, a, b)
    return src


def ffi(src):
    """The XLA FFI unit (jax_ffi FFI_UNIT, dk_ffi.cu) with info (B, 6): the forward handlers' shape check, the
    diagnostics' and the last-call record's stride."""
    n = src.count("dims_are(*info, {B, 4})")
    if n < 1:
        raise RuntimeError("capacity_diag: the forward handlers' info check not found")
    src = src.replace("dims_are(*info, {B, 4})", "dims_are(*info, {B, 6})").replace("info (B, 4) int32",
                                                                                   "info (B, 6) int32")
    return _sub(src, "info[4 * w", "info[6 * w", 6, "k_diag_fwd / k_last_fwd's info stride")


def torch_unit(src):
    """The torch library's own unit (torch_ffi TORCH_UNIT, dk_torch.cu) with info (B, 6): k_tdiag_fwd's stride."""
    return _sub(src, "info[4 * w", "info[6 * w", 4, "k_tdiag_fwd's info stride")


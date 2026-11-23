"""The contact forces in the G1 engine's contact table (an opt-in: jax_ffi.init / torch_ffi.build(contact_forces=...)),
and the outputs derived from them (per-substep tables, per-foot wrenches, the contact frame's tangent, the terrain
element). Applied to the generated text only when asked for, so the default build is unchanged
(scripts/identity_gate.py); the state, the tape and the gradients are bitwise those of the build without the options."""
from dataclasses import dataclass

CON_WORDS = 12
MODES = ("last", "substeps")
MAX_FEET = 8


@dataclass(frozen=True)
class Spec:
    """A build's contact-force options: mode ("last" / "substeps"), feet (body ids), frame (t1 words), terrain,
    foot_split (the foot and toe-weight words: the forest engine with foot_offset), foot_sand (the sand-fraction word:
    foot_split with TerrainConfig.terrain_sand)."""
    mode: str = "last"
    feet: tuple = ()
    frame: bool = False
    terrain: bool = False
    foot_split: bool = False
    foot_sand: bool = False

    @property
    def con_words(self):
        return (CON_WORDS + (3 if self.frame else 0) + (1 if self.terrain else 0) + (2 if self.foot_split else 0)
                + (1 if self.foot_sand else 0))

    @property
    def plain(self):
        """The original opt-in (the last substep's 12-word table only): the FFI unit keeps its MAXCON * 12 check."""
        return (self.mode == "last" and not self.feet and not self.frame and not self.terrain
                and not self.foot_split)

    def layout(self):
        """{name: (offset, width)} of a contact's words."""
        lay = dict(pair=(0, 1), dist=(1, 1), pos=(2, 3), normal=(5, 3), force=(8, 3), force_normal=(11, 1))
        o = CON_WORDS
        if self.frame:
            lay["tangent1"] = (o, 3)
            o += 3
        if self.terrain:
            lay["element"] = (o, 1)
            o += 1
        if self.foot_split:
            lay["foot_side"] = (o, 1)
            lay["toe_weight"] = (o + 1, 1)
            o += 2
        if self.foot_sand:
            lay["sand_k"] = (o, 1)
        return lay


def width(spec, maxcon, nsub):
    """Words per world of the contact output for nsub substeps."""
    t = (nsub if spec.mode == "substeps" else 1) * maxcon * spec.con_words
    return t + (nsub * len(spec.feet) * 6 + len(spec.feet) if spec.feet else 0)


def as_spec(contact_forces=False, foot_wrench=None, contact_frame=False, terrain=False, m=None, foot_split=False,
            foot_sand=False):
    """The Spec of init()'s arguments, None for the build without the forces. contact_forces: False / None, True or
    "last", "substeps"; foot_wrench: None / False, True (the G1's left_ankle_roll_link, right_ankle_roll_link) or body
    names / ids (needs m for names); contact_frame: the t1 words; terrain: a terrain build; foot_split: the foot and
    toe-weight words (the forest engine's foot_normal). foot_wrench, contact_frame and foot_split turn the forces on
    ("last") when contact_forces is off."""
    if contact_forces is True:
        mode = "last"
    elif not contact_forces:
        mode = None
    elif contact_forces in MODES:
        mode = contact_forces
    else:
        raise ValueError(f"contact_forces={contact_forces!r}: False, True, 'last' or 'substeps'")
    feet = ()
    if foot_wrench is not None and foot_wrench is not False:
        names = ("left_ankle_roll_link", "right_ankle_roll_link") if foot_wrench is True else tuple(foot_wrench)
        if not names or len(names) > MAX_FEET:
            raise ValueError(f"foot_wrench: 1 .. {MAX_FEET} bodies")
        ids = []
        for b in names:
            if isinstance(b, str):
                import mujoco
                assert m is not None
                i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                if i < 0:
                    raise ValueError(f"foot_wrench: no body {b!r} in the model")
            else:
                i = int(b)
                if m is not None and not 0 < i < m.nbody:
                    raise ValueError(f"foot_wrench: body id {i} out of range")
            ids.append(int(i))
        if len(set(ids)) != len(ids) or 0 in ids:
            raise ValueError("foot_wrench: distinct bodies other than the world body")
        feet = tuple(ids)
    if mode is None and (feet or contact_frame or foot_split):
        mode = "last"
    if mode is None:
        return None
    if foot_sand and not (foot_split and terrain):
        raise ValueError("foot_sand: needs foot_split (foot_normal) in a terrain build")
    return Spec(mode, feet, bool(contact_frame), bool(terrain), bool(foot_split), bool(foot_sand))


_STRUCT_SIM = "struct Sim {\n"
_STRUCT_SIMN = "struct SimNT {"
_HAND_BIG = "  SimNfull big;  // k_post's state"
_SOLVER = "  if (niter_out && lane == 0) *niter_out = it;\n"
_FORCES = r"""  if (s.ocf) {  // contact_forces: the contacts' forces at the solution (contact_forces.py)
    for (int r = lane; r < s.nefc; r += 32) {
      float ds, dd, cst;
      row_eval(s, r, s.efc_jar[r], &ds, &dd, &cst);
      s.efc_jv[r] = -ds;
    }
    __syncwarp();
    const int nc = min(s.ncon, MAXCON);
    for (int c = lane; c < nc; c += 32) {
      float g[4] = {0.f, 0.f, 0.f, 0.f};
      con_force(s, s.efc_jv, c, g);
      float* o = s.ocf + (size_t)c * CON_WORDS;
      o[8] = g[0]; o[9] = g[1]; o[10] = g[2]; o[11] = g[3];  // frame forces, torsion (cf_finish rotates them)
    }
    __syncwarp();
  }
"""

_HOST_HEAD = "// tape: null, or nw * nsub records of REC_WORDS (tape.py), record w * nsub + k for world w and substep k\n"
_HOST_FUNCS = r"""// ---- contact_forces (contact_forces.py): the contact output of a world: CF_T tables of MAXCON * CON_WORDS words
// (CF_SUB: one per substep, else the last substep's), then with CF_NFOOT feet nsub * CF_NFOOT * 6 wrench words and
// CF_NFOOT maxima of the feet's normal force
#define CF_T(nsub) (CF_SUB ? (size_t)(nsub) : (size_t)1)
__device__ __forceinline__ size_t cf_stride(int nsub) {
  return CF_T(nsub) * MAXCON * CON_WORDS + (CF_NFOOT ? (size_t)nsub * CF_NFOOT * 6 + CF_NFOOT : 0);
}
__device__ __forceinline__ float* cf_tab(float* ocon, int w, int k, int nsub) {  // substep k's table
  return ocon + (size_t)w * cf_stride(nsub) + (CF_SUB ? (size_t)k * MAXCON * CON_WORDS : 0);
}
__device__ __forceinline__ bool cf_on(int k, int nsub) { return CF_SUB || CF_NFOOT || k == nsub - 1; }  // forces
__device__ __forceinline__ bool cf_wr(int k, int nsub) { return CF_SUB || k == nsub - 1; }  // the other words
#if CF_NFOOT
__device__ __forceinline__ int cf_foot(int i) {
  constexpr int fb[CF_NFOOT] = CF_FEET;
  return fb[i];
}
#endif
// a substep's table words other than the forces (pair, dist, pos, normal, t1, element) from its collision; the force
// words of the slots past the world's contacts zero
template <class S>
__device__ __forceinline__ void cf_write(float* tab, const S& s, int ncon, int lane) {
  for (int c = lane; c < MAXCON; c += 32) {
    float* o = tab + (size_t)c * CON_WORDS;
    const bool v = c < ncon;
    o[0] = v ? (float)s.con_pair[c] : -1.f; o[1] = v ? s.con_dist[c] : 0.f;
    for (int t = 0; t < 3; t++) { o[2 + t] = v ? s.con_pos[c][t] : 0.f; o[5 + t] = v ? s.con_frame[c][t] : 0.f; }
#if CF_FRAME
    for (int t = 0; t < 3; t++) o[CF_T1 + t] = v ? s.con_frame[c][3 + t] : 0.f;
#endif
#if CF_TERRAIN
    o[CF_ELEM] = v && s.con_pair[c] >= NPAIR ? (float)s.con_elem[c] : -1.f;
#endif
    if (!v) for (int t = 8; t < 12; t++) o[t] = 0.f;
  }
}
// after a substep whose forces were wanted: the frame forces (words 8 .. 10) into world coordinates, f_n into word 11,
// and with feet the substep's wrench of each foot (xf: the feet's body origins, CF_NFOOT * 3) about its body origin;
// tab: the substep's table, k: the substep
__device__ __forceinline__ void cf_finish(float* tab, int ncon, const int* pair, const float (*pos)[3],
                                          const float (*fr)[9], const float* xf, int k, int nsub, int lane) {
#if CF_NFOOT
  float acc[CF_NFOOT][7];
#pragma unroll
  for (int i = 0; i < CF_NFOOT; i++)
#pragma unroll
    for (int j = 0; j < 7; j++) acc[i][j] = 0.f;
#endif
  for (int c = lane; c < ncon; c += 32) {
    float* o = tab + (size_t)c * CON_WORDS;
    const float f0 = o[8], f1 = o[9], f2 = o[10], tq = o[11];
    float fw[3];
    for (int j = 0; j < 3; j++) fw[j] = f0 * fr[c][j] + f1 * fr[c][3 + j] + f2 * fr[c][6 + j];
    for (int j = 0; j < 3; j++) o[8 + j] = fw[j];
    o[11] = f0;
#if CF_NFOOT
    const int pr = pair[c], b1 = c_gbody[PG1(pr)], b2 = c_gbody[PG2(pr)];
#pragma unroll
    for (int i = 0; i < CF_NFOOT; i++) {
      const float sg = b2 == cf_foot(i) ? 1.f : (b1 == cf_foot(i) ? -1.f : 0.f);  // on the foot: + on geom 2
      if (sg == 0.f) continue;
      float f[3], r[3];
      for (int j = 0; j < 3; j++) { f[j] = sg * fw[j]; r[j] = pos[c][j] - xf[3 * i + j]; }
      acc[i][0] += f[0]; acc[i][1] += f[1]; acc[i][2] += f[2];
      acc[i][3] += r[1] * f[2] - r[2] * f[1] + sg * tq * fr[c][0];
      acc[i][4] += r[2] * f[0] - r[0] * f[2] + sg * tq * fr[c][1];
      acc[i][5] += r[0] * f[1] - r[1] * f[0] + sg * tq * fr[c][2];
      acc[i][6] += f0;
    }
#endif
  }
#if CF_NFOOT
#pragma unroll
  for (int i = 0; i < CF_NFOOT; i++)
#pragma unroll
    for (int j = 0; j < 7; j++)
#pragma unroll
      for (int sh = 16; sh > 0; sh >>= 1) acc[i][j] += __shfl_xor_sync(0xffffffffu, acc[i][j], sh);
  if (lane == 0) {
    float* wr = tab - (CF_SUB ? (size_t)k * MAXCON * CON_WORDS : 0) + CF_T(nsub) * MAXCON * CON_WORDS;
#pragma unroll
    for (int i = 0; i < CF_NFOOT; i++) {
      for (int j = 0; j < 6; j++) wr[((size_t)k * CF_NFOOT + i) * 6 + j] = acc[i][j];
      float* mx = wr + (size_t)nsub * CF_NFOOT * 6 + i;
      *mx = k ? fmaxf(*mx, acc[i][6]) : acc[i][6];
    }
  }
#endif
  __syncwarp();
}

"""

_KSUB_LOOP = """  for (int k = 0; k < nsub; k++)
    substep(s, ctrl + (size_t)w * (NV - 6), iterations, lane, &nit,
            tape ? tape + ((size_t)w * nsub + k) * REC_WORDS : nullptr);
"""
_KSUB_LOOP_NEW = """  for (int k = 0; k < nsub; k++) {
    if (lane == 0) s.ocf = cf_on(k, nsub) ? cf_tab(ocon, w, k, nsub) : nullptr;  // (contact_forces)
    __syncwarp();
    substep(s, ctrl + (size_t)w * (NV - 6), iterations, lane, &nit,
            tape ? tape + ((size_t)w * nsub + k) * REC_WORDS : nullptr);
    __syncwarp();
    if (s.ocf) {
      float xf[CF_NFOOT ? 3 * CF_NFOOT : 1];
#if CF_NFOOT
      for (int i = 0; i < 3 * CF_NFOOT; i++) xf[i] = s.X[cf_foot(i / 3)][i % 3];
#endif
      cf_finish(s.ocf, min(s.ncon, MAXCON), s.con_pair, s.con_pos, s.con_frame, xf, k, nsub, lane);
      if (k < nsub - 1 && cf_wr(k, nsub)) cf_write(s.ocf, s, min(s.ncon, MAXCON), lane);
    }
  }
"""
_KSUB_WRITE = """  for (int c = lane; c < MAXCON; c += 32) {  // contacts of the last substep's collision: pair, dist, pos, normal
    float* o = ocon + ((size_t)w * MAXCON + c) * 8;
    bool v = c < min(s.ncon, MAXCON);
    o[0] = v ? (float)s.con_pair[c] : -1.f; o[1] = v ? s.con_dist[c] : 0.f;
    for (int k = 0; k < 3; k++) { o[2 + k] = v ? s.con_pos[c][k] : 0.f; o[5 + k] = v ? s.con_frame[c][k] : 0.f; }
  }
"""
_KSUB_WRITE_NEW = """  cf_write(cf_tab(ocon, w, nsub - 1, nsub), s, min(s.ncon, MAXCON), lane);  // (contact_forces: the forces are in)
"""
_PRE_PTR = "  substep_pre(s, ctrl + (size_t)w * (NV - 6), lane, tape ? tape + ((size_t)w * nsub + k) * REC_WORDS : nullptr);\n"
_PRE_PTR_NEW = ("  if (lane == 0) g.ocf = cf_on(k, nsub) ? cf_tab(ocon, w, k, nsub) : nullptr;  // (contact_forces)\n"
                + _PRE_PTR
                + "#if CF_NFOOT\n"
                  "  if (lane < 3 * CF_NFOOT) g.cf_x[lane] = s.X[cf_foot(lane / 3)][lane % 3];  // (the feet's origins)\n"
                  "#endif\n")
_PRE_WRITE = """  if (k == nsub - 1)  // contacts of the last substep's collision: pair, dist, pos, normal
    for (int c = lane; c < MAXCON; c += 32) {
      float* o = ocon + ((size_t)w * MAXCON + c) * 8;
      bool v = c < ncon;
      o[0] = v ? (float)s.con_pair[c] : -1.f; o[1] = v ? s.con_dist[c] : 0.f;
      for (int t = 0; t < 3; t++) { o[2 + t] = v ? s.con_pos[c][t] : 0.f; o[5 + t] = v ? s.con_frame[c][t] : 0.f; }
    }
"""
_PRE_WRITE_NEW = """  if (cf_wr(k, nsub)) cf_write(cf_tab(ocon, w, k, nsub), s, ncon, lane);  // (contact_forces: forces in k_post)
"""
_POST_PTR = "    s.min_nu_bits = g.min_nu_bits;\n    s.qpos = q;"
_POST_PTR_NEW = "    s.min_nu_bits = g.min_nu_bits;\n    s.ocf = g.ocf;  // (contact_forces)\n    s.qpos = q;"
_POST_WRITE = "  substep_post(s, iterations, lane, &nit, tape ? tape + ((size_t)w * nsub + k) * REC_WORDS : nullptr);\n"
_POST_WRITE_NEW = _POST_WRITE + """  __syncwarp();
  if (g.ocf)  // (contact_forces) the forces into world coordinates, the feet's wrenches
#if CF_NFOOT
    cf_finish(g.ocf, ncon, g.con_pair, g.con_pos, g.con_frame, g.cf_x, k, nsub, lane);
#else
    cf_finish(g.ocf, ncon, g.con_pair, g.con_pos, g.con_frame, nullptr, k, nsub, lane);
#endif
"""


def _sub(src, old, new, count, what):
    n = src.count(old)
    assert n == count, f"contact_forces: {what}: {n} anchors, expected {count}"
    return src.replace(old, new)


def engine(src, spec=None):
    """The engine text (adjoint.cuda_source) with the contact forces: the defines, the pointers, the solver tail."""
    spec = spec or Spec()
    if spec.foot_split:
        raise ValueError("foot_normal (the foot_split words): the forest engine only (this model goes to the tree "
                         "engine; engines.prefer_forest or DK_ENGINE=forest)")
    src = _sub(src, _STRUCT_SIM, _STRUCT_SIM + "  float* ocf;  // contact_forces: the world's contact table of the "
               "substep, else null\n", 1, "struct Sim")
    src = _sub(src, _STRUCT_SIMN, _STRUCT_SIMN + "\n  float* ocf;  // contact_forces (Hand::ocf)\n ", 1, "struct SimNT")
    hand = "  float* ocf;  // contact_forces: k_pre's table pointer for k_post (null: not wanted)\n"
    if spec.feet:
        hand += "  float cf_x[3 * CF_NFOOT];  // contact_forces: the feet's body origins of the substep (k_pre)\n"
    src = _sub(src, _HAND_BIG, hand + _HAND_BIG, 1, "Hand")
    src = _sub(src, _SOLVER, _SOLVER + _FORCES, 1, "the solver's tail")
    defs = [f"#define CON_WORDS {spec.con_words}  // the contact table per contact, with the forces (contact_forces.py)",
            f"#define CF_SUB {int(spec.mode == 'substeps')}  // a table per substep",
            f"#define CF_NFOOT {len(spec.feet)}  // feet with wrenches",
            f"#define CF_FRAME {int(spec.frame)}  // the tangent t1's words",
            f"#define CF_TERRAIN {int(spec.terrain)}  // the terrain element's word"]
    lay = spec.layout()
    if spec.frame:
        defs.append(f"#define CF_T1 {lay['tangent1'][0]}")
    if spec.terrain:
        defs.append(f"#define CF_ELEM {lay['element'][0]}")
    if spec.feet:
        defs.append("#define CF_FEET {" + ", ".join(str(b) for b in spec.feet) + "}  // the feet's body ids")
    return "\n".join(defs) + "\n" + src


def host(src, spec=None):
    """The host kernels' text (host.KERNEL) with the contact table's forces and the options of spec."""
    src = _sub(src, _HOST_HEAD, _HOST_FUNCS + _HOST_HEAD, 1, "the host functions")
    src = _sub(src, _KSUB_LOOP, _KSUB_LOOP_NEW, 1, "k_substep's loop")
    src = _sub(src, _KSUB_WRITE, _KSUB_WRITE_NEW, 1, "k_substep's table")
    src = _sub(src, _PRE_PTR, _PRE_PTR_NEW, 1, "pre_world's pointer")
    src = _sub(src, _PRE_WRITE, _PRE_WRITE_NEW, 1, "pre_world's table")
    src = _sub(src, _POST_PTR, _POST_PTR_NEW, 1, "post_world_on's pointer")
    src = _sub(src, _POST_WRITE, _POST_WRITE_NEW, 1, "post_world_on's forces")
    return src


def ffi(src, spec):
    """The FFI unit's contact output check: (B, MAXCON * W) for the plain opt-in, else (B, CF_WIDTH(substeps))."""
    if spec.plain:
        assert src.count("MAXCON * 8") >= 2
        return src.replace("MAXCON * 8", f"MAXCON * {spec.con_words}")
    nf = len(spec.feet)
    t = "(int64_t)substeps * " if spec.mode == "substeps" else ""
    w = f"({t}MAXCON * {spec.con_words}" + (f" + (int64_t)substeps * {6 * nf} + {nf}" if nf else "") + ")"
    old = 'if (!dims_are(*con, {B, MAXCON * 8})) return bad("contact output must be (B, MAXCON * 8) float32");'
    new = (f'if (!dims_are(*con, {{B, {w}}})) return bad("contact output must be (B, contact_forces width) float32 '
           f'(g1_engine.contact_forces.width)");')
    return _sub(src, old, new, 1, "the FFI's contact output")


def substeps_of(info, ncol):
    """The substep count of a contact output with ncol words per world (None where the layout does not depend on it)."""
    cf, mc, cw = info.get("contact_forces"), int(info["maxcon"]), int(info.get("con_words", 8))
    nf = len(cf["feet"]) if cf else 0
    if cf and cf["mode"] == "substeps":
        n, r = divmod(ncol - nf, mc * cw + 6 * nf)
    elif nf:
        n, r = divmod(ncol - mc * cw - nf, 6 * nf)
    else:
        return None
    if r or n < 1:
        raise ValueError(f"a contact output of {ncol} words per world does not fit this build's layout")
    return n


SAND_EPS = 1e-3


def decode(con, info, substeps=None, sand=None):
    """The contact output (B, width) as a dict of arrays (NumPy, JAX or torch: reshape and slicing only); pair stays
    a float array (the interfaces convert it). See jax_ffi.contact_table. sand: a foot_sand build's elements' sand
    fields (n elements, 4: k, s_max, F_y, rate; Terrain.sand_array) for foot_sand_params (None: not returned)."""
    B, ncol = con.shape[0], con.shape[1]
    cf, mc, cw = info.get("contact_forces"), int(info["maxcon"]), int(info.get("con_words", 8))
    n = substeps if substeps is not None else substeps_of(info, ncol)
    sub = bool(cf) and cf["mode"] == "substeps"
    nt = n if sub else 1
    if cf and n is not None and ncol != width(Spec(cf["mode"], tuple(cf["feet"]), cf["frame"], cf["terrain"],
                                                   bool(cf.get("foot_split", False)),
                                                   bool(cf.get("foot_sand", False))), mc, n):
        raise ValueError(f"a contact output of {ncol} words per world, not this build's for {n} substeps")
    c = con[:, :nt * mc * cw].reshape((B, nt, mc, cw) if sub else (B, mc, cw))
    out = dict(pair=c[..., 0], dist=c[..., 1], pos=c[..., 2:5], normal=c[..., 5:8])
    if cw >= 12:
        out["force"], out["force_normal"] = c[..., 8:11], c[..., 11]
    lay = (cf or {}).get("con_layout", {})
    if "tangent1" in lay:
        o = lay["tangent1"][0]
        t1, nn = c[..., o:o + 3], out["normal"]
        n0, n1, n2, a0, a1, a2 = nn[..., 0], nn[..., 1], nn[..., 2], t1[..., 0], t1[..., 1], t1[..., 2]
        t2 = [n1 * a2 - n2 * a1, n2 * a0 - n0 * a2, n0 * a1 - n1 * a0]
        f = out["force"]
        out["tangent1"] = t1
        out["tangent2"] = _stack(t2, t1)
        out["force_t1"] = f[..., 0] * a0 + f[..., 1] * a1 + f[..., 2] * a2
        out["force_t2"] = f[..., 0] * t2[0] + f[..., 1] * t2[1] + f[..., 2] * t2[2]
    if "element" in lay:
        out["element"] = c[..., lay["element"][0]]
    if "foot_side" in lay:
        side, w, fn = c[..., lay["foot_side"][0]], c[..., lay["toe_weight"][0]], out["force_normal"]
        out["foot_side"], out["toe_weight"] = side, w
        cols = []
        for k in (0, 1):
            on = fn * (side == k)
            cols += [(on * (1 - w)).sum(-1), (on * w).sum(-1)]
        out["foot_normal"] = _stack(cols, fn)
        if "sand_k" in lay:
            _sand_outputs(out, c, lay, (info.get("terrain") or {}).get("sand_other"), sand)
    if cf and cf["feet"]:
        nf = len(cf["feet"])
        wr = con[:, nt * mc * cw:]
        w6 = wr[:, :n * nf * 6].reshape((B, n, nf, 6))
        out["foot_force"], out["foot_moment"] = w6[..., 0:3], w6[..., 3:6]
        out["foot_fn_max"] = wr[:, n * nf * 6:]
        f = out["foot_force"]
        out["foot_fnet_max"] = _amax1((f[..., 0] ** 2 + f[..., 1] ** 2 + f[..., 2] ** 2) ** 0.5)
        out["foot_fup_max"] = _amax1(f[..., 2])
    return out


def _sand_outputs(out, c, lay, other, sand):
    side, w, fn = out["foot_side"], out["toe_weight"], out["force_normal"]
    k = c[..., lay["sand_k"][0]]
    out["sand_k"] = k
    sp, hp, pr, wts = [], [], [], []
    for f in (0, 1):
        on = side == f
        for part in (1 - w, w):
            a = on * part
            sp.append((fn * k * a).sum(-1))
            hp.append((fn * (1 - k) * a).sum(-1))
            pr.append((k * a).sum(-1))
            wts.append((fn * k * a, k * a))
    sand_part, hard_part = _stack(sp, fn), _stack(hp, fn)
    sh = _stack([sand_part, hard_part], fn)
    out["foot_normal_split"] = sh.transpose(-1, -2) if hasattr(sh, "new_zeros") else sh.swapaxes(-1, -2)
    out["foot_sand_presence"] = _stack(pr, fn)
    if sand is None:
        return
    other = tuple(other) if other is not None else (1.0, 0.0, 0.0, 0.0)
    torch = hasattr(fn, "new_zeros")
    if torch:
        import torch as _t
        tab = _t.as_tensor(sand, dtype=fn.dtype, device=fn.device).reshape(-1, 4)
        tab = _t.cat([tab, _t.as_tensor([other], dtype=fn.dtype, device=fn.device)], 0)
        stop = lambda x: x.detach()
    elif type(fn).__module__.startswith("numpy"):
        import numpy as _np
        tab = _np.concatenate([_np.asarray(sand, fn.dtype).reshape(-1, 4), _np.asarray([other], fn.dtype)], 0)
        stop = lambda x: x
    else:
        import jax
        import jax.numpy as jnp
        tab = jnp.concatenate([jnp.asarray(sand, fn.dtype).reshape(-1, 4), jnp.asarray([other], fn.dtype)], 0)
        stop = jax.lax.stop_gradient
    n = tab.shape[0] - 1
    e = c[..., lay["element"][0]]
    ok = (e >= 0) & (e < n)
    idx = ok * e + (1 - ok * 1) * n
    idx = idx.long() if torch else idx.astype("int32")
    par = tab[idx][..., 1:]
    cols = []
    for wf, wk in wts:
        wf, wk = stop(wf), stop(wk)
        sf, sk = wf.sum(-1), wk.sum(-1)
        use = sf >= SAND_EPS
        wt = use[..., None] * wf + (1 - use[..., None] * 1) * wk
        tot = use * sf + (1 - use * 1) * sk
        num = (wt[..., None] * par).sum(-2)
        cols.append(num / (tot + (tot <= 0))[..., None] * (tot > 0)[..., None])
    p = _stack(cols, fn)
    if torch:
        p = p.transpose(-1, -2)
    else:
        p = p.swapaxes(-1, -2)
    out["foot_sand_params"] = stop(p.reshape(p.shape[:-2] + (2, 2, 3)))


def _amax1(x):
    if hasattr(x, "new_zeros"):
        return x.amax(1)
    return x.max(1)


def _stack(cols, like):
    if hasattr(like, "new_zeros"):
        import torch
        return torch.stack(cols, -1)
    if type(like).__module__.startswith("numpy"):
        import numpy as np
        return np.stack(cols, -1)
    import jax.numpy as jnp
    return jnp.stack(cols, -1)


def pair_geoms(m, foot_pairs=(), terrain=None):
    """(npair + terrain pairs, 2) model geom ids of each contact pair id of the table, in the engine's geom order (the
    force words are the force on column 1's geom, the normal points from column 0's to column 1's): a model pair p
    is (pair_geom1, pair_geom2) swapped where geom 1's type comes after geom 2's in MuJoCo's type order (a plane
    first, then sphere, capsule, ..., as MuJoCo's collision table; g1_engine.step.pair_constants); a terrain build's
    virtual pair npair + t is (-1: the terrain or an object slot, the robot geom of terrain.robot_terrain_geoms' t-th
    entry). terrain: the build's TerrainConfig."""
    import numpy as np
    out = []
    for p in range(m.npair):
        g1, g2 = int(m.pair_geom1[p]), int(m.pair_geom2[p])
        out.append((g2, g1) if int(m.geom_type[g1]) > int(m.geom_type[g2]) else (g1, g2))
    if terrain is not None:
        from nablaengine import terrain as _terrain
        out += [(-1, int(g)) for g, _, _ in _terrain.robot_terrain_geoms(m, foot_pairs, terrain.geoms)]
    return np.asarray(out, np.int64).reshape(-1, 2)

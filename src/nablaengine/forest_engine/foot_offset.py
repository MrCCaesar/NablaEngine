"""The per-world foot support offset (nablaengine.params field foot_offset) and the foot contacts' heel / toe split of
the contact output (jax_ffi.init(foot_normal=True): contact_forces Spec.foot_split) on the forest engine. Applied to
the generated text only for a build with them; every other build is unchanged."""
import mujoco
import numpy as np

FIELD = "foot_offset"


def active(params):
    return bool(params) and FIELD in params.fields


def body_ids(m, params):
    """The two foot bodies' model ids."""
    out = []
    for b in params.foot_bodies:
        i = b if isinstance(b, int) else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if not 1 <= i < m.nbody:
            raise ValueError(f"PhysicsParams foot_bodies: no (moving) body {b!r}")
        out.append(int(i))
    return out


def resolve(m, params):
    """params with foot_bodies as model ids (unchanged without foot_offset)."""
    if not active(params):
        return params
    import dataclasses
    return dataclasses.replace(params, foot_bodies=tuple(body_ids(m, params)))


def check(m, params, foot_pairs, terrain=None, surrogate=False):
    """ValueError where foot_offset does not fit the forest build of m (terrain: (TerrainConfig, robot geoms))."""
    if not active(params):
        return
    fb = body_ids(m, params)
    fp = list(foot_pairs or ())
    if not fp:
        raise ValueError("PhysicsParams foot_offset: init(foot_pairs=) must name the foot-ground pairs it applies to")
    if surrogate:
        raise ValueError("PhysicsParams foot_offset: not with contact_surrogate")
    if int(m.opt.noslip_iterations) > 0:
        raise ValueError("PhysicsParams foot_offset: not with the noslip pass (opt.noslip_iterations > 0)")
    ok = (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE))
    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    on_foot = 0
    for p in fp:
        g1, g2 = int(m.pair_geom1[p]), int(m.pair_geom2[p])
        t1, t2 = int(m.geom_type[g1]), int(m.geom_type[g2])
        if t2 == plane:
            g1, g2, t1, t2 = g2, g1, t2, t1
        b2 = int(m.geom_bodyid[g2])
        if b2 not in fb:
            continue
        if t1 != plane or int(m.geom_bodyid[g1]) != 0:
            raise ValueError(f"PhysicsParams foot_offset: foot pair {p} ({m.geom(g1).name}, {m.geom(g2).name}) is not "
                             "a foot geom against the static floor plane")
        if t2 not in ok:
            raise ValueError(f"PhysicsParams foot_offset: foot geom {m.geom(g2).name} is not a sphere or capsule")
        on_foot += 1
    if terrain is not None:
        for g, _, ft in terrain[1]:
            if ft and int(m.geom_bodyid[g]) in fb:
                if int(m.geom_type[g]) not in ok:
                    raise ValueError(f"PhysicsParams foot_offset: terrain foot geom {m.geom(g).name} is not a sphere "
                                     "or capsule")
                on_foot += 1
    if not on_foot:
        raise ValueError(f"PhysicsParams foot_offset: no foot pair has a geom on the foot bodies {params.foot_bodies}")


def _f(x):
    return f"{float(np.float32(x))!r}f"


def defines(m, params):
    lay = params.layout(m.nv)
    fb = body_ids(m, params)
    h, t = params.foot_line
    return (f"#define PW_FOFF {lay[FIELD][0]}  // the foot offsets in the world's parameters (forest_engine.foot_offset)\n"
            f"#define FOFF_BL {fb[0]}  // the left / right foot bodies\n#define FOFF_BR {fb[1]}\n"
            f"#define FOFF_XH {_f(h)}  // the heel / toe line along the foot axis (m)\n#define FOFF_XT {_f(t)}\n"
            f"#define FOFF_LEN {_f(np.float32(t) - np.float32(h))}\n"
            "#define FOFF_AXIS {" + ", ".join(_f(x) for x in params.foot_axis) + "}  // the foot axis, foot frame\n")


DEVICE = r"""// ---- foot_offset (forest_engine.foot_offset): a contact's foot (0 left, 1 right, -1 none: a foot pair whose
// second geom is on a foot body), its point along the foot (a_w . (p - X_b), a_w = R_b FOFF_AXIS), the toe weight and
// the offset s = s_h + (s_t - s_h) w; explicit roundings (the adjoint's replay computes the same bits)
__device__ __forceinline__ int foff_side(int pair) {
  const int b = c_gbody[PG2(pair)];
  return PFOOT(pair) ? (b == FOFF_BL ? 0 : (b == FOFF_BR ? 1 : -1)) : -1;
}
__device__ __forceinline__ float foff_x(const float* Q, const float* X, const float* p, float* aw) {
  const float ax[3] = FOFF_AXIS;
  qrot_rn(Q, ax, aw);
  const float r[3] = {RSUB(p[0], X[0]), RSUB(p[1], X[1]), RSUB(p[2], X[2])};
  return dot3_rn(aw, r);
}
__device__ __forceinline__ float foff_w(float x) {
  return fminf(fmaxf(__fdiv_rn(RSUB(x, FOFF_XH), FOFF_LEN), 0.f), 1.f);
}
__device__ __forceinline__ float foff_dw(float x) {  // d w / d x (0 where clipped)
  return x > FOFF_XH && x < FOFF_XT ? 1.f / FOFF_LEN : 0.f;
}
__device__ __forceinline__ float foff_s(const float* sp, float w) { return RFMA(RSUB(sp[1], sp[0]), w, sp[0]); }

"""
COMPACT = r"""// (foot_offset) the foot contacts' distance d + s(x), then the contacts with d + s >= 0 out (stable)
template <class S>
__device__ __forceinline__ void foff_contacts(S& s, int lane) {
  const int n = uniform_int(min(s.ncon, MAXCON), CON_BITS);
  int out = 0;
  for (int c0 = 0; c0 < n; c0 += 32) {
    const int c = c0 + lane;
    int pr = 0, sl = 0, dm = 0, fs = -1;
    float di = 0.f, mu = 0.f, po[3] = {0.f, 0.f, 0.f}, fr[9];
#if defined(HAS_TERRAIN) && HAS_TERRAIN
    unsigned short el = 0;
#if TERRAIN_HF_NATIVE
    unsigned ax = 0u;
#endif
#endif
    bool keep = c < n;
    if (keep) {
      pr = s.con_pair[c]; sl = s.con_slot[c]; dm = s.con_dim[c]; di = s.con_dist[c]; mu = s.con_mu[c];
      for (int k = 0; k < 3; k++) po[k] = s.con_pos[c][k];
      for (int k = 0; k < 9; k++) fr[k] = s.con_frame[c][k];
#if defined(HAS_TERRAIN) && HAS_TERRAIN
      el = s.con_elem[c];
#if TERRAIN_HF_NATIVE
      ax = s.con_aux[c];
#endif
#endif
      fs = foff_side(pr);
      if (fs >= 0) {
        const int fb = c_gbody[PG2(pr)];
        float aw[3];
        di = RADD(di, foff_s(s.pw + PW_FOFF + 2 * fs, foff_w(foff_x(s.Q[fb], s.X[fb], po, aw))));
        keep = di < 0.f;
      }
    }
    const unsigned bal = __ballot_sync(0xffffffffu, keep);
    const int dst = out + __popc(bal & ((1u << lane) - 1u));
    __syncwarp();
    if (keep && dst != c) {
      s.con_pair[dst] = pr; s.con_slot[dst] = (unsigned char)sl; s.con_dim[dst] = dm; s.con_dist[dst] = di;
      s.con_mu[dst] = mu;
      for (int k = 0; k < 3; k++) s.con_pos[dst][k] = po[k];
      for (int k = 0; k < 9; k++) s.con_frame[dst][k] = fr[k];
#if defined(HAS_TERRAIN) && HAS_TERRAIN
      s.con_elem[dst] = el;
#if TERRAIN_HF_NATIVE
      s.con_aux[dst] = ax;
#endif
#endif
    } else if (keep && fs >= 0) {
      s.con_dist[c] = di;
    }
    out += __popc(bal);
    __syncwarp();
  }
  if (lane == 0) s.ncon -= n - out;
  __syncwarp();
}

"""
_COMPACT_AT = "  if constexpr (PHASE == 1) return;  // (k_pre_a: the caller hands the contacts and geom poses to k_ccd)\n"
_COMPACT_CALL = "  foff_contacts(s, lane);  // (foot_offset)\n"
_HEAD = "__device__ __forceinline__ float adj_contact_one(Adj& s, int c, int pair, int slot, unsigned zone,"
_REPLAY_TAIL = """  const int fo_s = foff_side(pair);  // (foot_offset) the replayed distance d + s(x)
  float fo_aw[3] = {0.f, 0.f, 0.f}, fo_x = 0.f, fo_w = 0.f;
  if (fo_s >= 0) {
    fo_x = foff_x(s.Q[b2], s.X[b2], p, fo_aw);
    fo_w = foff_w(fo_x);
    d = RADD(d, foff_s(s.pw + PW_FOFF + 2 * fo_s, fo_w));
  }
"""
_PRIM = "  if (zone) {  // the primitive's VJP"
_XTERM = """  float fo_C = 0.f;  // (foot_offset) the distance cotangent onto the offsets, the x term's P coefficient
  if (fo_s >= 0) {
    const float* sp = s.pw + PW_FOFF + 2 * fo_s;
    const float dw = foff_dw(fo_x);
    fo_C = cot[0] * (sp[1] - sp[0]) * dw;
    FOFF_SPLIT_TERM
    fop[2 * fo_s] += cot[0] * (1.f - fo_w);
    fop[2 * fo_s + 1] += cot[0] * fo_w;
    fop[4] = fo_C;  // (a capsule-box terrain contact of the cooperative pass: its foot-pose term after cb_coop)
    for (int t = 0; t < 3; t++) cot[1 + t] += fo_C * fo_aw[t];
  }
"""
_B2VJP = "    geom_to_body_vjp(s.gx[g2], s.X[b2], gb + 6, gb + 9, fg + 6, fg + 9);\n"
_B2VJP_NEW = _B2VJP + """    if (fo_C != 0.f) {  // (foot_offset) the x term on the foot's pose
      const float r[3] = {p[0] - s.X[b2][0], p[1] - s.X[b2][1], p[2] - s.X[b2][2]};
      float u[3];
      cross(fo_aw, r, u);
      for (int t = 0; t < 3; t++) { fg[6 + t] -= fo_C * fo_aw[t]; fg[9 + t] += fo_C * u[t]; }
    }
"""
_COOP_AT = ("    cb_coop(s, lane, cb >= 0 && CB_COOP_PAIR(pair, slot), pair, slot, cw);  // (model_hooks.capbox_coop_hooks)\n"
            "#endif\n")
_COOP_FIX = """#if CB_COOP_ON
    if (cb >= 0 && CB_COOP_PAIR(pair, slot) && fop[4] != 0.f) {  // (foot_offset) the x term on the foot's pose
      const int fb = c_gbody[PG2(pair)];
      const float ax[3] = FOFF_AXIS;
      float aw[3], u[3];
      qrot_rn(s.Q[fb], ax, aw);
      const float r[3] = {cw[9] - s.X[fb][0], cw[10] - s.X[fb][1], cw[11] - s.X[fb][2]};
      cross(aw, r, u);
      for (int t = 0; t < 3; t++) { cw[18 + t] -= fop[4] * aw[t]; cw[21 + t] += fop[4] * u[t]; }
    }
#endif
"""
_PP_DECL = "    float pp[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};"
_SUM_AT = "    const unsigned skip = __ballot_sync(0xffffffffu, on && !use), mis = __ballot_sync(0xffffffffu, bad);\n"
_SUM = """    for (int q = 0; q < 4; q++) {  // (foot_offset) the batch's offset terms, a fixed order; the gradient is -P
      const float v = -warp_sum(fop[q]);
      if (lane == 0) s.gpw[PW_FOFF + q] += v;
    }
"""


def _sub(src, a, b, n, what):
    k = src.count(a)
    if k != n:
        raise RuntimeError(f"foot_offset: {what}: anchor found {k} times, expected {n}: {a[:80]!r}")
    return src.replace(a, b)


SAND_DEVICE = r"""// ---- (TerrainConfig.terrain_sand) a foot contact's sand fraction k: its terrain element's (g_tmat word 8),
// FOFF_KOTHER off the terrain (the floor plane's pairs, the object slots)
__device__ __forceinline__ float foff_k(int pair, int e) {
  if (pair < NPAIR || e >= OBJ_E0 || !g_tmat) return FOFF_KOTHER;
  return g_tmat[TER_MW * e + 8];
}

"""


def _sand_cfg(terrain):
    if terrain is None:
        return False, 1.0
    cfg = terrain[0] if isinstance(terrain, tuple) else terrain
    if not getattr(cfg, "terrain_sand", False):
        return False, 1.0
    return True, float(cfg.sand_other[0])


def sand_words(terrain):
    """The device's words per terrain element (TER_MW) of a terrain_sand build: 12 (the material and sand fields)."""
    from nablaengine import terrain as T
    return T.TER_MAT_SAND_WORDS


_SAND_COMPACT = ("        di = RADD(di, foff_s(s.pw + PW_FOFF + 2 * fs, foff_w(foff_x(s.Q[fb], s.X[fb], po, aw))));\n",
                 "        di = RADD(di, RMUL(foff_k(pr, el), foff_s(s.pw + PW_FOFF + 2 * fs, foff_w(foff_x(s.Q[fb], s.X[fb], "
                 "po, aw)))));  // (terrain_sand) d + k s(x)\n")
_SAND_REPLAY = ("    d = RADD(d, foff_s(s.pw + PW_FOFF + 2 * fo_s, fo_w));\n",
                "    fo_sv = foff_s(s.pw + PW_FOFF + 2 * fo_s, fo_w);  // (terrain_sand) d + k s(x)\n"
                "    fo_k = foff_k(pair, slot >> 8);\n"
                "    d = RADD(d, RMUL(fo_k, fo_sv));\n")
_SAND_REPLAY_DECL = ("  float fo_aw[3] = {0.f, 0.f, 0.f}, fo_x = 0.f, fo_w = 0.f;\n",
                     "  float fo_aw[3] = {0.f, 0.f, 0.f}, fo_x = 0.f, fo_w = 0.f;\n"
                     "  float fo_k = 1.f, fo_sv = 0.f;  // (terrain_sand)\n")
_SAND_XTERM = [("    fo_C = cot[0] * (sp[1] - sp[0]) * dw;\n",
                "    const float fo_ck = cot[0] * fo_k;  // (terrain_sand) the distance cotangent times k\n"
                "    fo_C = fo_ck * (sp[1] - sp[0]) * dw;\n"),
               ("    fop[2 * fo_s] += cot[0] * (1.f - fo_w);\n    fop[2 * fo_s + 1] += cot[0] * fo_w;\n",
                "    fop[2 * fo_s] += fo_ck * (1.f - fo_w);\n    fop[2 * fo_s + 1] += fo_ck * fo_w;\n"),
]


def engine(src, m, params, terrain=None):
    """The final engine text (adjoint.cuda_source, after body_params) with foot_offset; unchanged without it. terrain:
    the build's ((TerrainConfig, robot geoms)); a terrain_sand build takes the elements' sand fractions."""
    if not active(params):
        return src
    sand, kother = _sand_cfg(terrain)
    add = """template <class S>
__device__ __forceinline__ bool add_contact(S& s, int pair, int slot, float dist, const float* pos,
                                            const float* frame) {
"""
    compact = COMPACT
    if sand:
        compact = _sub(compact, *_SAND_COMPACT, 1, "foff_contacts (terrain_sand)")
        compact = f"#define FOFF_KOTHER {_f(kother)}  // (terrain_sand) k off the terrain\n" + SAND_DEVICE + compact
    src = _sub(src, add, DEVICE + compact + add, 1, "add_contact")
    src = _sub(src, _COMPACT_AT, _COMPACT_AT + _COMPACT_CALL, 1, "substep_pre's collision end")
    i = src.index(_HEAD) if src.count(_HEAD) == 1 else -1
    if i < 0:
        raise RuntimeError("foot_offset: adj_contact_one's head")
    k = src.index(") {", i)
    if k - i > 400 or "float* pp" not in src[i:k]:
        raise RuntimeError("foot_offset: adj_contact_one's parameters (world_params' pp)")
    src = src[:k] + ", float* fop" + src[k:]
    calls = src.count("adj_contact_one(s, c, pair, slot, zone, cw, dbg, pp")
    if calls != 1:
        raise RuntimeError(f"foot_offset: adj_contact_one's calls ({calls})")
    j = src.index("adj_contact_one(s, c, pair, slot, zone, cw, dbg, pp")
    e = src.index(")", j + len("adj_contact_one(s, c, pair, slot, zone, cw, dbg, pp"))
    src = src[:e] + ", fop" + src[e:]
    body = src.index(") {", src.index(_HEAD)) + 3
    end = src.index("\n}\n", body)
    fn = src[body:end]
    import re
    rp = re.findall(r"\n  const int br = adj_pair_geom[^\n]*;\n", fn)
    if len(rp) != 1:
        raise RuntimeError(f"foot_offset: adj_contact_one's replay found {len(rp)} times")
    tail, xterm = _REPLAY_TAIL, _XTERM.replace("    FOFF_SPLIT_TERM\n", "")
    if sand:
        tail = _sub(_sub(tail, *_SAND_REPLAY_DECL, 1, "replay (terrain_sand)"), *_SAND_REPLAY, 1, "replay (terrain_sand)")
        for a, b in _SAND_XTERM:
            xterm = _sub(xterm, a, b, 1, "x term (terrain_sand)")
    fn = fn.replace(rp[0], rp[0] + tail)
    fn = _sub(fn, _PRIM, xterm + _PRIM, 1, "adj_contact_one's primitive VJP")
    fn = _sub(fn, _B2VJP, _B2VJP_NEW, 1, "adj_contact_one's body-2 pose cotangent")
    src = src[:body] + fn + src[end:]
    src = _sub(src, _PP_DECL, _PP_DECL + "\n    float fop[5] = {0.f, 0.f, 0.f, 0.f, 0.f};  // (foot_offset)", 1,
               "adj_contacts' pp")
    if "HAS_TERRAIN 1" in src and "#define CB_COOP_ON" in src:
        src = _sub(src, _COOP_AT, _COOP_AT + _COOP_FIX, 1, "adj_contacts' cb_coop call")
    src = _sub(src, _SUM_AT, _SUM + _SUM_AT, 1, "adj_contacts' parameter sums")
    return defines(m, params) + src


_CFW_AT = "    if (!v) for (int t = 8; t < 12; t++) o[t] = 0.f;\n  }\n}\n"
_CFW = """#if CF_FSPLIT
    {  // (foot_offset: foot_split) the contact's foot and toe weight
      const int fs = v ? foff_side(s.con_pair[c]) : -1;
      float fw = 0.f;
      if (fs >= 0) {
        const int fb = c_gbody[PG2(s.con_pair[c])];
        float aw[3];
        fw = foff_w(foff_x(s.Q[fb], s.X[fb], s.con_pos[c], aw));
      }
      o[CF_FS] = (float)fs;
      o[CF_FS + 1] = fw;
    }
#endif
"""


_CFW_SAND = ("      o[CF_FS + 1] = fw;\n",
             "      o[CF_FS + 1] = fw;\n"
             "      o[CF_FS + 2] = fs >= 0 ? foff_k(s.con_pair[c], s.con_elem[c]) : 0.f;  // (terrain_sand: CF_FSAND) the "
             "contact's sand fraction\n")


def host_split(src, sand=False):
    """The host text's cf_write with the foot_split words (after forest_engine.contact_forces.host); sand (Spec.foot_sand,
    TerrainConfig.terrain_sand): the sand-fraction word too."""
    i = src.index("__device__ __forceinline__ void cf_write(")
    j = src.index(_CFW_AT, i)
    return src[:j] + (_sub(_CFW, *_CFW_SAND, 1, "cf_write (foot_sand)") if sand else _CFW) + src[j:]


def engine_split(src):
    """adj_contact_one's toe-weight word cotangent in the x term (a contact_grad build with foot_split; after
    forest_engine.contact_vjp), before fop[4] takes the x term's C for the cooperative pass' contacts (a terrain
    capsule-box contact's foot-pose term after cb_coop, _COOP_FIX: the full C, the toe weight's term included). (A
    foot_sand build's sand word is a constant per element: no cotangent term.)"""
    term = ("    if (fo_s >= 0 && s.cgt) fo_C -= s.cgt[(size_t)c * CON_WORDS + CF_FS + 1] * foff_dw(fo_x);"
            "  // (foot_split: the toe weight word)\n")
    fop4 = ("    fop[4] = fo_C;  // (a capsule-box terrain contact of the cooperative pass: its foot-pose term after "
            "cb_coop)\n")
    cot = "    for (int t = 0; t < 3; t++) cot[1 + t] += fo_C * fo_aw[t];\n"
    return _sub(src, fop4 + cot, term + fop4 + cot, 1, "the x term (foot_split)")

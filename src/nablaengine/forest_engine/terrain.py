"""Terrain builds on the forest engine (nablaengine.terrain; the tree engine's g1_engine.step.TERRAIN_DEVICE text)."""
import re

import mujoco
import numpy as np

from nablaengine.forest_engine import step

TERRAIN_GTYPE = 6


def _g1_step():
    from nablaengine.g1_engine import step as g1_step
    return g1_step


def _cfg(terrain):
    return terrain[0] if isinstance(terrain, tuple) else terrain


def check(m, terrain):
    """ValueError where a terrain build ((TerrainConfig, robot geoms)) does not fit the forest build of m."""
    cfg, tg = terrain
    geoms = step.collision_geoms(m)
    if not tg:
        raise ValueError("terrain build: no robot geom collides with the terrain")
    if m.npair + len(tg) >= 65536:
        raise ValueError(f"{m.npair} pairs and {len(tg)} terrain geoms: a forest terrain build holds fewer than 65536")
    if len(geoms) + 1 > 255:
        raise ValueError(f"{len(geoms)} collision geoms: a forest terrain build holds at most 254 (8-bit geom index)")
    if step.broad_groups(m) is not None:
        raise ValueError(f"terrain builds: at most {step.GROUP_PAIRS} model pairs (the broad phase's pair groups)")
    ok = {int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_BOX)}
    for g, _, _ in tg:
        if int(m.geom_type[g]) not in ok:
            raise ValueError(f"terrain geom {m.geom(g).name}: spheres, capsules and boxes collide with the terrain")
    if has_box_geoms(m, terrain) and cfg.mode == "native":
        raise ValueError("box robot geoms against the terrain: the analytic heightfield mode only")


def has_box_geoms(m, terrain):
    """Whether a box robot geom collides with the terrain (box_geom_hooks)."""
    box = int(mujoco.mjtGeom.mjGEOM_BOX)
    return any(int(m.geom_type[g]) == box for g, _, _ in terrain[1])


def com_words(m, terrain):
    """Per-world com words of a terrain build: the forest's (COM offset, fixtures) and the object slots."""
    from nablaengine import terrain as T
    cfg = _cfg(terrain)
    return (step.com_words(m) + (T.OBJ_WORDS + (T.OBJ_VEL_WORDS if cfg.object_velocity else 0)) * int(cfg.objects)
            + (T.OBJ_MAT_WORDS * int(cfg.objects) if getattr(cfg, "object_material", False) else 0))


def const_layout(m, terrain):
    """step.const_layout(m) with NTG more pairs (c_pword) and one more collision-geom row (the pseudo geom)."""
    ntg, ng = len(terrain[1]), len(step.collision_geoms(m))
    geom_syms = {s for s, _, _ in step.GEOM_LAYOUT} | {"c_ginvw"}
    out = []
    for s, k, n in step.const_layout(m):
        if s == "c_pword":
            n = n + ntg
        elif s in geom_syms:
            assert n % ng == 0, (s, n, ng)
            n = n + n // ng
        out.append((s, k, n))
    return out


def pair_constants(m, foot_pairs, terrain):
    """step.pair_constants(m, foot_pairs) in const_layout(m, terrain) order: after the model's pairs one virtual pair per
    terrain geom (g1 = the terrain's pseudo geom, the last geom-table row; the geom's floor pair's parameter set,
    condim and foot flag), and the pseudo geom's row (world body, type TERRAIN_GTYPE, at the origin, zero sizes and
    bounds, inverse weight 0)."""
    base = step.pair_constants(m, foot_pairs)
    geoms = step.collision_geoms(m)
    gid = {g: i for i, g in enumerate(geoms)}
    sets = step.pair_param_sets(m)[1]
    tg = len(geoms)
    words = np.asarray([step.pair_word(tg, gid[g], sets[p], int(m.pair_dim[p]), int(ft), 0, 0)
                        for g, p, ft in terrain[1]], np.int64)
    extra = {"c_pword": np.stack([words & 0xFFFF, words >> 16], 1).ravel(),
             "c_gbody": [0], "c_gtype": [TERRAIN_GTYPE], "c_gsize": [0.0] * 3, "c_gpos": [0.0] * 3,
             "c_gquat": [1.0, 0.0, 0.0, 0.0], "c_grbound": [0.0], "c_gaabb": [0.0] * 6, "c_ginvw": [0.0]}
    parts, off = [], 0
    for s, k, n in step.const_layout(m):
        nf = 2 * n if k == "u" else n
        parts.append(base[off:off + nf])
        if s in extra:
            parts.append(np.asarray(extra[s], np.float64).astype(np.float32))
        off += nf
    assert off == len(base), (off, len(base))
    return np.concatenate(parts).astype(np.float32)


def defines(m, terrain):
    """The terrain's #defines (the tree engine's step_header terrain block; COM_WORDS: the forest's com_words)."""
    from nablaengine import terrain as T
    cfg, tg = terrain
    return (f"#define HAS_TERRAIN 1\n#define NTG {len(tg)}  // robot geoms that collide with the terrain\n"
            f"#define TERRAIN_FLOOR {int(cfg.floor)}  // keep the model's floor plane pairs\n"
            f"#define TERRAIN_CAP_PTS {int(cfg.cap_points)}  // heightfield samples per capsule\n"
            f"#define TERRAIN_HF_NATIVE {int(cfg.mode == 'native')}  // heightfield contacts per prism\n"
            f"#define TERRAIN_NOBJ {int(cfg.objects)}  // per-world object slots (in com after the COM offset)\n"
            f"#define OBJ_E0 {T.OBJ_E0}  // the element index of object slot 0\n"
            "#ifndef TERRAIN_SORT\n#define TERRAIN_SORT 1  // contacts sorted by (pair, element, slot)\n#endif\n"
            + ("#define OBJ_VEL 1  // the object slots' velocities (com after the slots) enter their contacts\n"
               if cfg.object_velocity else ""))


def _sub1(text, a, b, what):
    if text.count(a) != 1:
        raise RuntimeError(f"forest terrain {what}: anchor found {text.count(a)} times: {a[:70]!r}")
    return text.replace(a, b)


def header(text, m, terrain):
    """step.step_header(m, ...)'s text for a terrain build: NGEOMC + 1, c_pword with the virtual pairs, the geom
    arrays with the pseudo geom's row, COM_WORDS with the object slots; the terrain's #defines first."""
    p, g, ntg = m.npair, len(step.collision_geoms(m)), len(terrain[1])
    text = _sub1(text, f"#define NGEOMC {g}\n", f"#define NGEOMC {g + 1}  // (the terrain's pseudo geom last)\n",
                 "header")
    text = _sub1(text, f"DKC unsigned int c_pword[{max(p, 1)}];",
                 f"DKC unsigned int c_pword[{p + ntg}];  // (terrain: NTG virtual pairs from NPAIR on)\n//", "header")
    text = _sub1(text, f"DKC int c_gbody[{max(g, 1)}], c_gtype[{max(g, 1)}];",
                 "DKC int c_gbody[NGEOMC], c_gtype[NGEOMC];", "header")
    text = _sub1(text, f"DKC float c_gsize[{max(g, 1) * 3}], c_gpos[{max(g, 1) * 3}], c_gquat[{max(g, 1) * 4}], "
                       f"c_grbound[{max(g, 1)}],\n    c_gaabb[{max(g, 1) * 6}];",
                 "DKC float c_gsize[NGEOMC * 3], c_gpos[NGEOMC * 3], c_gquat[NGEOMC * 4], c_grbound[NGEOMC],\n"
                 "    c_gaabb[NGEOMC * 6];", "header")
    if step.fixture_groups(m):
        a = f"__device__ const int c_fixg[{g}] = {{"
        i = text.index(a)
        j = text.index("};", i)
        text = _sub1(text, text[i:j + 2], f"__device__ const int c_fixg[{g + 1}] = {{" + text[i + len(a):j] + ", -1};",
                     "header (fixtures)")
    text = _sub1(text, f"#define COM_WORDS {step.com_words(m)}\n",f"#define COM_WORDS {com_words(m, terrain)}  "
                 "// (the COM offset, then the terrain's object slots)\n", "header")
    return defines(m, terrain) + text


SORT_DEVICE = r"""// A terrain build's contact order (forest_engine.terrain; the tree engine's terrain_sort_contacts with CPL contacts per
// lane): sorted by (pair, element, slot), ties in their order of arrival, so that the order, and hence the constraint
// rows and the solver's bits, depend only on the contact set; nothing moves when the contacts arrived sorted
template <class S>
__device__ void terrain_sort_contacts(S& s, int ncon, int lane) {
#if TERRAIN_SORT
  __syncwarp();
  unsigned long long key[CPL];
  int rk[CPL];
  bool moved = false;
#pragma unroll
  for (int h = 0; h < CPL; h++) {
    const int c = lane + 32 * h;
    key[h] = c < ncon ? (unsigned long long)s.con_pair[c] << 32 | (unsigned long long)s.con_elem[c] << 8 |
                            (unsigned long long)s.con_slot[c]
                      : ~0ull;
    rk[h] = 0;
  }
  for (int j = 0; j < ncon; j++) {
    const unsigned long long kj = (unsigned long long)s.con_pair[j] << 32 | (unsigned long long)s.con_elem[j] << 8 |
                                  (unsigned long long)s.con_slot[j];
#pragma unroll
    for (int h = 0; h < CPL; h++) rk[h] += kj < key[h] || (kj == key[h] && j < lane + 32 * h);
  }
#pragma unroll
  for (int h = 0; h < CPL; h++) moved |= lane + 32 * h < ncon && rk[h] != lane + 32 * h;
  if (!__any_sync(0xffffffffu, moved)) return;
  int pr[CPL], dm[CPL];
  unsigned char sl[CPL];
  unsigned short el[CPL];
  float di[CPL], mu[CPL], ps[CPL][3], fr[CPL][9];
#if TERRAIN_HF_NATIVE
  unsigned aux[CPL];
#endif
#pragma unroll
  for (int h = 0; h < CPL; h++) {
    const int c = lane + 32 * h;
    if (c < ncon) {
      pr[h] = s.con_pair[c]; sl[h] = s.con_slot[c]; el[h] = s.con_elem[c]; dm[h] = s.con_dim[c];
      di[h] = s.con_dist[c]; mu[h] = s.con_mu[c];
      for (int k = 0; k < 3; k++) ps[h][k] = s.con_pos[c][k];
      for (int k = 0; k < 9; k++) fr[h][k] = s.con_frame[c][k];
#if TERRAIN_HF_NATIVE
      aux[h] = s.con_aux[c];
#endif
    }
  }
  __syncwarp();
#pragma unroll
  for (int h = 0; h < CPL; h++) {
    if (lane + 32 * h < ncon) {
      const int c = rk[h];
      s.con_pair[c] = pr[h]; s.con_slot[c] = sl[h]; s.con_elem[c] = el[h]; s.con_dim[c] = dm[h];
      s.con_dist[c] = di[h]; s.con_mu[c] = mu[h];
      for (int k = 0; k < 3; k++) s.con_pos[c][k] = ps[h][k];
      for (int k = 0; k < 9; k++) s.con_frame[c][k] = fr[h][k];
#if TERRAIN_HF_NATIVE
      s.con_aux[c] = aux[h];
#endif
    }
  }
  __syncwarp();
#endif
}

"""


def device_text(src):
    """The tree engine's TERRAIN_DEVICE with capbox_segs generated from the forest text src and the forest's
    terrain_sort_contacts."""
    g1 = _g1_step()
    dev = g1.TERRAIN_DEVICE
    a = dev.index("// permutation of a per-contact array: contact c (lanes c, c - 32) moves to rk[c / 32]")
    b = dev.index("// the terrain's contacts of the robot geoms (virtual pairs NPAIR .. NPAIR + NTG - 1)")
    dev = dev[:a] + SORT_DEVICE + dev[b:]
    return g1.capbox_segs_source(src) + "\n" + dev.lstrip("\n") + "\n"


STEP_HOOKS = (
    ("""template <class S>
__device__ DK_INL_COLLIDE void collide_pair(S& s, int pair) {""", None),
    ("""  unsigned char con_slot[MAXCON];  // which contact of its pair (tape): plane-capsule end, parallel capsule-capsule candidate
""",
     """  unsigned char con_slot[MAXCON];  // which contact of its pair (tape): plane-capsule end, parallel capsule-capsule candidate
  unsigned short con_elem[MAXCON];  // terrain builds: the contact's terrain element (add_contact's slot >> 8)
"""),
    ("""  float con_mu[MAXCON];
  unsigned char con_slot[MAXCON];
  float *con_dist""",
     """  float con_mu[MAXCON];
  unsigned char con_slot[MAXCON];
  unsigned short con_elem[MAXCON];
  float *con_dist"""),
    ("""  s.con_slot[c] = (unsigned char)slot;
""",
     """  s.con_slot[c] = (unsigned char)slot;
  s.con_elem[c] = (unsigned short)(slot >> 8);
"""),
    ("""      __stcs(rec + OFF_CON + i, __uint_as_float(e));
    }
""",
     """      __stcs(rec + OFF_CON + i, __uint_as_float(e));
    }
  if (rec)  // tape (terrain): the contacts' terrain elements, 16 bits each
    for (int i = lane; i < (MAXCON + 1) / 2; i += 32) {
      const unsigned x0 = 2 * i < ncon ? (unsigned)s.con_elem[2 * i] : 0u;
      const unsigned x1 = 2 * i + 1 < ncon ? (unsigned)s.con_elem[2 * i + 1] : 0u;
      __stcs(rec + OFF_TELEM + i, __uint_as_float(x0 | x1 << 16));
    }
"""),
    ("""  if (PPLANE(pr)) {
#if FLOOR_G >= 0
    const float* n = fn;""",
     """  if (PPLANE(pr)) {
#if !TERRAIN_FLOOR
    return false;  // (terrain build: the terrain replaces the floor)
#endif
#if FLOOR_G >= 0
    const float* n = fn;"""),
    ("""    for (int i = lane; i < ncand; i += 32) collide_pair(s, s.cand[i]);
""",
     """    for (int i = lane; i < ncand; i += 32) collide_pair(s, s.cand[i]);
    collide_terrain(s, lane);
"""),
    ("""  int ncon = uniform_int(min(s.ncon, MAXCON), CON_BITS);
  {  // contact Jacobians:""",
     """  int ncon = uniform_int(min(s.ncon, MAXCON), CON_BITS);
  terrain_sort_contacts(s, ncon, lane);
  {  // contact Jacobians:"""),
)

STEP_NATIVE_HOOKS = (
    ("""  unsigned short con_elem[MAXCON];  // terrain builds: the contact's terrain element (add_contact's slot >> 8)
""",
     """  unsigned short con_elem[MAXCON];  // terrain builds: the contact's terrain element (add_contact's slot >> 8)
  unsigned con_aux[MAXCON];  // native heightfield contacts: the candidate id (HfCand.pid), else 0
"""),
    ("""  unsigned short con_elem[MAXCON];
  float *con_dist""",
     """  unsigned short con_elem[MAXCON];
  unsigned con_aux[MAXCON];
  float *con_dist"""),
    ("""  s.con_elem[c] = (unsigned short)(slot >> 8);
  s.con_dim[c] = PDIM(pair);
  s.con_dist[c] = dist;
  for (int k = 0; k < 3; k++) s.con_pos[c][k] = pos[k];
  for (int k = 0; k < 9; k++) s.con_frame[c][k] = frame[k];
  s.con_mu[c] = PFOOT(pair) ? s.fric : PMU(pair);
  return true;""",
     """  s.con_elem[c] = (unsigned short)(slot >> 8);
  s.con_aux[c] = 0u;
  s.con_dim[c] = PDIM(pair);
  s.con_dist[c] = dist;
  for (int k = 0; k < 3; k++) s.con_pos[c][k] = pos[k];
  for (int k = 0; k < 9; k++) s.con_frame[c][k] = frame[k];
  s.con_mu[c] = PFOOT(pair) ? s.fric : PMU(pair);
  return true;"""),
    ("""      __stcs(rec + OFF_TELEM + i, __uint_as_float(x0 | x1 << 16));
    }
""",
     """      __stcs(rec + OFF_TELEM + i, __uint_as_float(x0 | x1 << 16));
    }
  if (rec)  // tape (native heightfields): the contacts' candidate ids
    for (int i = lane; i < MAXCON; i += 32) __stcs(rec + OFF_TAUX + i, __uint_as_float(i < ncon ? s.con_aux[i] : 0u));
"""),
)


OBJ_VEL_SLOT = r"""// (forest_engine.terrain, OBJ_VEL) the slots moved to the substep's time (obj_advance), their motion words after
// them as in com; dk_objc: the world's slots in com
#define OBJ_VW 18  // motion words per slot: v, w, s, a, al, sd
__shared__ float dk_objk[TERRAIN_NOBJ * (15 + OBJ_VW)];
__shared__ const float* dk_objc;
__shared__ float dk_objt;  // the substep's time t = k h (obj_advance)
// object slot e's motion words at the substep: v (3) at the slot's position, w (3), world; s (3) slot frame; a, al
// (world), sd (slot frame)
__device__ __forceinline__ const float* obj_vel(int e) { return dk_obj + 15 * TERRAIN_NOBJ + OBJ_VW * e; }
// R x for a slot's row-major rotation ob + 3
__device__ __forceinline__ void obj_rot(const float* ob, const float* x, float* y) {
  for (int t = 0; t < 3; t++) y[t] = ob[3 + 3 * t] * x[0] + ob[4 + 3 * t] * x[1] + ob[5 + 3 * t] * x[2];
}
// the slot's surface velocity u and acceleration A at point p of a contact with normal n
__device__ __forceinline__ void obj_point_motion(int e, const float* p, const float* n, float* u, float* A) {
  const float *ov = obj_vel(e), *ob = dk_obj + 15 * e;
  const float r[3] = {p[0] - ob[0], p[1] - ob[1], p[2] - ob[2]};
  float rs[3], rd[3];
  cross(ov + 3, r, u);
  obj_rot(ob, ov + 6, rs);
  obj_rot(ob, ov + 15, rd);
  const float rn = dot3(n, rs), dn = dot3(n, rd);
  for (int t = 0; t < 3; t++) u[t] += ov[t] + rs[t] - n[t] * rn;
  cross(ov + 12, r, A);
  for (int t = 0; t < 3; t++) A[t] += ov[9 + t] + rd[t] - n[t] * dn;
}
// R' = exp([th]) R (Rodrigues; th = 0: R copied)
__device__ __forceinline__ void obj_rotate(const float* th3, const float* o, float* d) {
  const float th = sqrtf(dot3(th3, th3));
  if (!(th > 0.f)) {
    for (int i = 0; i < 9; i++) d[i] = o[i];
    return;
  }
  const float k[3] = {th3[0] / th, th3[1] / th, th3[2] / th}, sn = sinf(th), cs = 1.f - cosf(th);
  const float Q[9] = {1.f - cs * (k[1] * k[1] + k[2] * k[2]), -sn * k[2] + cs * k[0] * k[1], sn * k[1] + cs * k[0] * k[2],
                      sn * k[2] + cs * k[0] * k[1], 1.f - cs * (k[0] * k[0] + k[2] * k[2]), -sn * k[0] + cs * k[1] * k[2],
                      -sn * k[1] + cs * k[0] * k[2], sn * k[0] + cs * k[1] * k[2], 1.f - cs * (k[0] * k[0] + k[1] * k[1])};
  float R[9];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) R[3 * i + j] = Q[3 * i] * o[j] + Q[3 * i + 1] * o[3 + j] + Q[3 * i + 2] * o[6 + j];
  for (int i = 0; i < 9; i++) d[i] = R[i];
}
// the world's slots at substep k (t = k h) into dk_objk, dk_obj pointed at it
__device__ __forceinline__ void obj_advance(int k, float h, int lane) {
  __syncwarp();
  const float t = k * h;
  for (int e = lane; e < TERRAIN_NOBJ; e += 32) {
    const float *o = dk_objc + 15 * e, *ov = dk_objc + 15 * TERRAIN_NOBJ + OBJ_VW * e;
    float* d = dk_objk + 15 * e;
    float* dv = dk_objk + 15 * TERRAIN_NOBJ + OBJ_VW * e;
    const float kk = 0.5f * (float)(k * (k + 1));  // sum_{j<=k} j
    for (int i = 0; i < 3; i++) d[i] = o[i] + (t * ov[i] + (h * h * kk) * ov[9 + i]);
    const bool alz = ov[12] == 0.f && ov[13] == 0.f && ov[14] == 0.f;
    if (alz) {  // constant w: one rotation exp(t [w])
      const float th3[3] = {t * ov[3], t * ov[4], t * ov[5]};
      obj_rotate(th3, o + 3, d + 3);
    } else {  // R_k = exp(h [w_k]) ... exp(h [w_1]) R
      for (int i = 0; i < 9; i++) d[3 + i] = o[3 + i];
      for (int j = 1; j <= k; j++) {
        const float th3[3] = {h * (ov[3] + (j * h) * ov[12]), h * (ov[4] + (j * h) * ov[13]), h * (ov[5] + (j * h) * ov[14])};
        obj_rotate(th3, d + 3, d + 3);
      }
    }
    for (int i = 12; i < 15; i++) d[i] = o[i];
    for (int i = 0; i < 3; i++) {
      dv[i] = ov[i] + t * ov[9 + i];
      dv[3 + i] = ov[3 + i] + t * ov[12 + i];
      dv[6 + i] = ov[6 + i] + t * ov[15 + i];
    }
    for (int i = 9; i < OBJ_VW; i++) dv[i] = ov[i];
  }
  if (lane == 0) { dk_obj = dk_objk; dk_objt = t; }
  __syncwarp();
}
"""
OBJ_VEL_FORWARD = (
    ("  con_jv(s, s.qvel, lane);  // contact J qvel for the contact rows' velocities (its __syncwarp also publishes the "
     "above)\n",
     "  con_jv(s, s.qvel, lane);  // contact J qvel for the contact rows' velocities (its __syncwarp also publishes the "
     "above)\n"
     "  for (int c = lane; c < ncon; c += 32)  // (forest_engine.terrain, OBJ_VEL) relative to the object slot's surface\n"
     "    if (s.con_pair[c] >= NPAIR && s.con_elem[c] >= OBJ_E0) {\n"
     "      const int e = s.con_elem[c] - OBJ_E0;\n"
     "      float p[3], fr[9], u[3], A[3];\n"
     "      for (int t = 0; t < 3; t++) p[t] = s.con_pos[c][t];\n"
     "      for (int t = 0; t < 9; t++) fr[t] = s.con_frame[c][t];\n"
     "      obj_point_motion(e, p, fr, u, A);\n"
     "      for (int d = 0; d < 3; d++) s.con_dot[c][d] -= dot3(fr + 3 * d, u);\n"
     "#if HAS_CONDIM4\n"
     "      s.con_dot[c][3] -= dot3(fr, obj_vel(e) + 3);\n"
     "#endif\n"
     "    }\n"
     "  __syncwarp();\n"),
    ("      float D = efc_D(iw, imp);\n",
     "      float D = efc_D(iw, imp);\n"
     "      if (con && s.con_pair[c] >= NPAIR && s.con_elem[c] >= OBJ_E0 && bb != 0.f) {  // (OBJ_VEL) the slot's acceleration\n"
     "        const int e = s.con_elem[c] - OBJ_E0;\n"
     "        float p[3], fr[9], u[3], A[3];\n"
     "        for (int t = 0; t < 3; t++) p[t] = s.con_pos[c][t];\n"
     "        for (int t = 0; t < 9; t++) fr[t] = s.con_frame[c][t];\n"
     "        obj_point_motion(e, p, fr, u, A);\n"
     "        for (int d = 0; d < 3; d++) s.con_dot[c][d] -= dot3(fr + 3 * d, A) / bb;\n"
     "#if HAS_CONDIM4\n"
     "        s.con_dot[c][3] -= dot3(fr, obj_vel(e) + 12) / bb;\n"
     "#endif\n"
     "      }\n"),
)
OBJ_VEL_HOST = (
    ("    if (lane == 0) s.oforce = k == nsub - 1 ? s.con_fo : nullptr;  // the contact forces of the last substep\n",
     "    obj_advance(k, c_misc[0], lane);  // (forest_engine.terrain, OBJ_VEL) the slots at the substep\n"
     "    if (lane == 0) s.oforce = k == nsub - 1 ? s.con_fo : nullptr;  // the contact forces of the last substep\n",
     1),
    ("  load_fixtures(s, com, w, lane);\n  if (lane == 0) {\n    s.fric = fric[w];\n    s.unsupported = k ? g.unsupported : 0;",
     "  load_fixtures(s, com, w, lane);\n"
     "  obj_advance(k, c_misc[0], lane);  // (forest_engine.terrain, OBJ_VEL) the slots at the substep\n"
     "  if (lane == 0) {\n    s.fric = fric[w];\n    s.unsupported = k ? g.unsupported : 0;", 2),
)
OBJ_VEL_ADJ_KERNEL = (
    ("    const size_t r = (size_t)w * nsub + k;\n    adj_substep(",
     "    const size_t r = (size_t)w * nsub + k;\n"
     "    obj_advance(k, c_misc[0], lane);  // (forest_engine.terrain, OBJ_VEL) the slots at the substep\n"
     "    adj_substep(", 1),
)
OBJ_VEL_ADJOINT = (
    ("#define CS_N (HAS_CONDIM4 ? 33 : 24)",
     "#define CS_OV (HAS_CONDIM4 ? 33 : 24)  // (forest_engine.terrain, OBJ_VEL) the slot's motion cotangents (18 + 6)\n"
     "#define CS_N (CS_OV + 24)"),
    ("  float Wv[3], Wa[3], Wl[3], cot[13], fg[12];  // cot = (d bar, p bar, frame bar): adj_contact_geom's cotangent\n",
     "  const int oe = pair >= NPAIR && (slot >> 8) >= OBJ_E0 ? (slot >> 8) - OBJ_E0 : -1;  // (OBJ_VEL) its slot\n"
     "  if (oe >= 0) {  // (forest_engine.terrain, OBJ_VEL) relative to the slot's surface\n"
     "    float u[3], A[3];\n"
     "    obj_point_motion(oe, p, fr, u, A);\n"
     "    const float* ov = obj_vel(oe);\n"
     "    for (int t = 0; t < 3; t++) { Vv[t] -= u[t]; Ov[t] -= ov[3 + t]; Va[t] -= A[t]; Oa[t] -= ov[12 + t]; }\n"
     "  }\n"
     "  float Wv[3], Wa[3], Wl[3], cot[13], fg[12];  // cot = (d bar, p bar, frame bar): adj_contact_geom's cotangent\n"),
    ("  cot[1] = cot[2] = cot[3] = 0.f;\n  contact_pbar_motion(Wv, Wa, Wl, Ov, Oa, Ol, cot + 1);\n",
     "  float og[24];  // (OBJ_VEL) the slot's motion cotangents (18), then the pose partials' contact parts (6)\n"
     "#pragma unroll\n"
     "  for (int t = 0; t < 24; t++) og[t] = 0.f;\n"

     "  if (oe >= 0) {\n"
     "    const float *ov = obj_vel(oe), *ob = dk_obj + 15 * oe;\n"
     "    float rs[3], rd[3], r[3], x[3];\n"
     "    obj_rot(ob, ov + 6, rs);\n"
     "    obj_rot(ob, ov + 15, rd);\n"
     "    for (int t = 0; t < 3; t++) r[t] = p[t] - ob[t];\n"
     "    const float rn = dot3(fr, rs), dn = dot3(fr, rd), wn = dot3(fr, Wv), an = dot3(fr, Wa);\n"
     "    for (int t = 0; t < 3; t++) cot[4 + t] += rn * Wv[t] + wn * rs[t] + dn * Wa[t] + an * rd[t];  // n bar\n"

     "    float wt[3], at[3];\n"
     "    for (int t = 0; t < 3; t++) { og[t] = Wv[t]; og[9 + t] = Wa[t]; wt[t] = Wv[t] - fr[t] * wn; at[t] = Wa[t] - fr[t] * an; }\n"
     "    cross(r, Wv, og + 3);\n"

     "    cross(r, Wa, og + 12);\n"
     "#if HAS_CONDIM4\n"
     "    for (int t = 0; t < 3; t++) { og[3 + t] += AWv[t]; og[12 + t] += AWa[t]; }\n"
     "#endif\n"
     "    for (int j = 0; j < 3; j++) {\n"
     "      og[6 + j] = ob[3 + j] * wt[0] + ob[6 + j] * wt[1] + ob[9 + j] * wt[2];\n"
     "      og[15 + j] = ob[3 + j] * at[0] + ob[6 + j] * at[1] + ob[9 + j] * at[2];\n"
     "    }\n"
     "    cross(rs, wt, x);  // the rotation partial's surface terms (og[21 ..] completed below)\n"
     "    for (int t = 0; t < 3; t++) og[21 + t] = -x[t];\n"
     "    cross(rd, at, x);\n"
     "    for (int t = 0; t < 3; t++) og[21 + t] -= x[t];\n"
     "  }\n"
     "  cot[1] = cot[2] = cot[3] = 0.f;\n  contact_pbar_motion(Wv, Wa, Wl, Ov, Oa, Ol, cot + 1);\n"
),
    ("  for (int t = 0; t < 12; t++) cs[12 + t] = fg[t];\n",
     "  for (int t = 0; t < 12; t++) cs[12 + t] = fg[t];\n"
     "  if (oe >= 0) {  // (OBJ_VEL) the slot pose's partials without the robot geom's terms\n"
     "    const float *ov = obj_vel(oe), *ob = dk_obj + 15 * oe;\n"
     "    float x[3], y[3], nb[3], r[3];\n"
     "    cross(Wv, ov + 3, x);\n"
     "    cross(Wa, ov + 12, y);\n"
     "    for (int t = 0; t < 3; t++) { og[18 + t] = cot[1 + t] + x[t] + y[t]; r[t] = p[t] - ob[t]; }\n"
     "    make_frame_vjp(fr, cot + 4, nb);\n"
     "    cross(fr, nb, x);\n"
     "    cross(r, cot + 1, y);\n"
     "    for (int t = 0; t < 3; t++) og[21 + t] += x[t] + y[t];\n"
     "  }\n"
     "#pragma unroll\n"
     "  for (int t = 0; t < 24; t++) cs[CS_OV + t] = og[t];  // (OBJ_VEL)\n"),
)
OBJ_VEL_GRAD = (
    ("  float qbar[NQ], vbar[NV], ubar[NU];  // cotangents of the substep's output, then of its input; ctrl (summed)\n",
     "  float qbar[NQ], vbar[NV], ubar[NU];  // cotangents of the substep's output, then of its input; ctrl (summed)\n"
     "  float ovbar[OBJ_VW * TERRAIN_NOBJ];  // (forest_engine.terrain, OBJ_VEL) the slots' motion cotangents (summed)\n"),
    ("  for (int i = lane; i < NU; i += 32) s.ubar[i] = 0.f;\n",
     "  for (int i = lane; i < NU; i += 32) s.ubar[i] = 0.f;\n"
     "  for (int i = lane; i < OBJ_VW * TERRAIN_NOBJ; i += 32) s.ovbar[i] = 0.f;  // (OBJ_VEL)\n"),
    ("    APROF(APROF_CONTACT_ROWS);\n",
     "    APROF(APROF_CONTACT_ROWS);\n"
     "    {  // (forest_engine.terrain, OBJ_VEL) the batch's object contacts' slot motion cotangents (after cb_coop)\n"
     "      const int oe = cb >= 0 && pair >= NPAIR && (slot >> 8) >= OBJ_E0 ? (slot >> 8) - OBJ_E0 : -1;\n"
     "      float g[18];\n"
     "      const float t = dk_objt, tq = 0.5f * t * (t + c_misc[0]);  // d v_k / d a, d pos_k / d a\n"
     "#pragma unroll\n"
     "      for (int q = 0; q < 9; q++) { g[q] = cw[CS_OV + q]; g[9 + q] = cw[CS_OV + 9 + q] + t * cw[CS_OV + q]; }\n"
     "      if (oe >= 0 && t > 0.f) {  // the slot's motion within the step through the contact geometry\n"
     "        const float *ov = obj_vel(oe), *ob = dk_obj + 15 * oe, *X = s.X[c_gbody[PG2(pair)]];\n"
     "        float xs[3], ts[3], r[3], u[3];\n"
     "        for (int i = 0; i < 3; i++) { xs[i] = cw[CS_OV + 18 + i] - cw[18 + i]; r[i] = X[i] - ob[i]; }\n"
     "        cross(r, cw + 18, u);\n"
     "        for (int i = 0; i < 3; i++) ts[i] = cw[CS_OV + 21 + i] - cw[21 + i] - u[i];\n"
     "        const float w0[3] = {ov[3] - t * ov[12], ov[4] - t * ov[13], ov[5] - t * ov[14]};  // the step's w\n"
     "        const float ph[3] = {t * w0[0], t * w0[1], t * w0[2]}, th2 = dot3(ph, ph), th = sqrtf(th2);\n"
     "        const float ca = th < 1e-3f ? 0.5f - th2 / 24.f : (1.f - cosf(th)) / th2;\n"
     "        const float cb3 = th < 1e-3f ? 1.f / 6.f - th2 / 120.f : (th - sinf(th)) / (th2 * th);\n"
     "        float a1[3], a2[3];\n"
     "        cross(ph, ts, a1);\n"
     "        cross(ph, a1, a2);\n"
     "        for (int i = 0; i < 3; i++) {\n"
     "          const float jt = ts[i] - ca * a1[i] + cb3 * a2[i];\n"
     "          g[i] -= t * xs[i]; g[9 + i] -= tq * xs[i]; g[3 + i] -= t * jt; g[12 + i] -= tq * jt;\n"
     "        }\n"
     "      }\n"
     "      unsigned om = __ballot_sync(0xffffffffu, oe >= 0);\n"
     "      while (om) {  // slot by slot (uniform: om is the warp's)\n"
     "        const int k = __shfl_sync(0xffffffffu, oe, __ffs(om) - 1);\n"
     "        const bool mine = oe == k;\n"
     "        om &= ~__ballot_sync(0xffffffffu, mine);\n"
     "#pragma unroll\n"
     "        for (int q = 0; q < 18; q++) {\n"
     "          float sv = mine ? g[q] : 0.f;\n"
     "          for (int o = 16; o > 0; o >>= 1) sv += __shfl_xor_sync(0xffffffffu, sv, o);\n"
     "          if (lane == 0) s.ovbar[OBJ_VW * k + q] += sv;\n"
     "        }\n"
     "      }\n"
     "    }\n"),
)


MAT_DECL = r"""// (forest_engine.terrain, OBJ_MAT / TER_MAT) contact materials of the object slots (OBJ_MW words each in com after
// the slots and their motion words; dk_omat: the world's, set with dk_obj) and of the static terrain's elements (g_tmat:
// TER_MW words each, the boxes then the heightfields, set with the terrain; null: none)
#define OBJ_MAT_ON @OBJ@
#define TER_MAT_ON @TER@
#define OBJ_MW 7
#define TER_MW 8
#if OBJ_MAT_ON
__shared__ const float* dk_omat;
__device__ __forceinline__ const float* obj_mat(int e) { return dk_omat + OBJ_MW * e; }
#endif
#if TER_MAT_ON
__device__ const float* g_tmat;
#endif
// the material of a contact of pair with terrain element e into *p: 0 none (the pair's parameters), 1 its friction
// mixed by max, 2 its friction, solref and solimp (an object slot's: 2)
__device__ __forceinline__ int el_mat(int pair, int e, const float** p) {
  *p = nullptr;
  if (pair < NPAIR) return 0;
#if OBJ_MAT_ON
  if (e >= OBJ_E0) { *p = obj_mat(e - OBJ_E0); return 2; }
#endif
#if TER_MAT_ON
  if (e < OBJ_E0 && g_tmat) {
    const float* q = g_tmat + TER_MW * e;
    const int md = (int)q[7];
    if (md == 1 || md == 2) { *p = q; return md; }
  }
#endif
  return 0;
}

"""
_MAT_DECL_AT = ("// slot: which contact of the pair this is (the tape's contact table, tape.py): the end e of a plane-capsule "
                "pair\n")
_MAT_MU = "  s.con_mu[c] = PFOOT(pair) ? s.fric : PMU(pair);\n"
_MAT_MU_NEW = ("  s.con_mu[c] = el_mu(pair, slot >> 8, PFOOT(pair) ? s.fric : PMU(pair));  // (forest_engine.terrain, "
               "OBJ_MAT / TER_MAT)\n")
_EL_MU_AT = "  return 0;\n}\n\n"
_EL_MU = """  return 0;
}
// (forest_engine.terrain, OBJ_MAT / TER_MAT) the friction of a new contact of pair with element e, mu the pair's (or
// the foot pair's per-world friction): an override's mu, max(mu, the element's) for mode 1, else mu
__device__ __noinline__ float el_mu(int pair, int e, float mu) {
  const float* em;
  const int md = el_mat(pair, e, &em);
  return md == 2 ? em[0] : (md == 1 ? fmaxf(mu, em[0]) : mu);
}

"""
_MAT_KB = "      float k, bb, imp;\n      kbimp(sref, simp, pos, h, &k, &bb, &imp);\n"
_MAT_KB_NEW = """      float omsi[5];  // (forest_engine.terrain, OBJ_MAT / TER_MAT) an overriding material's impedance
      if (con) {
        const float* em;
        if (el_mat(s.con_pair[c], s.con_elem[c], &em) == 2) {
          for (int t = 0; t < 4; t++) omsi[t] = em[3 + t];
          omsi[4] = simp[4];
          sref = em + 1;
          simp = omsi;
        }
      }
"""
_MAT_ADJ_MU = "  const float mu = PFOOT(pair) ? s.fric : PMU(pair);\n"
_MAT_ADJ_MU_NEW = """  const int om_e = pair >= NPAIR && (slot >> 8) >= OBJ_E0 ? (slot >> 8) - OBJ_E0 : -1;  // (OBJ_MAT) its slot
  const float* em;  // (OBJ_MAT / TER_MAT) the element's material and mode (el_mat)
  const int em_md = el_mat(pair, slot >> 8, &em);
  const float mu_b = PFOOT(pair) ? s.fric : PMU(pair);
  const float mu = em_md == 2 ? em[0] : (em_md == 1 ? fmaxf(mu_b, em[0]) : mu_b);
  const bool em_mu = em_md == 2 || (em_md == 1 && em[0] > mu_b);  // (the friction is the element's)
  (void)om_e; (void)em_mu;
"""
_MAT_ADJ_SEL = """  float omsi[5];  // (forest_engine.terrain, OBJ_MAT / TER_MAT) an overriding material's impedance
  if (em_md == 2) {
    for (int t = 0; t < 4; t++) omsi[t] = em[3 + t];
    omsi[4] = simp[4];
    sref = em + 1;
    simp = omsi;
  }
"""
_MAT_ADJ_IMP_P = "  adj_row_impedance(sref, simp, d, contact_invw(pair, mu), ri);\n"
_MAT_ADJ_IMP = "  adj_row_impedance(PSOLREF(pair), PSOLIMP(pair), d, contact_invw(pair, mu), ri);\n"
_MAT_PB = "  cot[1] = cot[2] = cot[3] = 0.f;\n  contact_pbar_motion(Wv, Wa, Wl, Ov, Oa, Ol, cot + 1);\n"
_MAT_PARTIALS = ("  if (om_e >= 0 && zone)  // (OBJ_MAT) the slot material's partials (P sign)\n"
                 "    contact_param_vjp(Vv, Va, Vl, fr, mu, dim, zone, d, D, k, b, imp, contact_invw(pair, mu), sref, simp, mg);\n")
_MAT_REDUCE = """    {  // (forest_engine.terrain, OBJ_MAT) the batch's object contacts' material cotangents, slot by slot (a warp sum)
      const int oe = cb >= 0 && pair >= NPAIR && (slot >> 8) >= OBJ_E0 ? (slot >> 8) - OBJ_E0 : -1;
      unsigned om = __ballot_sync(0xffffffffu, oe >= 0);
      while (om) {  // (uniform: om is the warp's)
        const int k = __shfl_sync(0xffffffffu, oe, __ffs(om) - 1);
        const bool mine = oe == k;
        om &= ~__ballot_sync(0xffffffffu, mine);
#pragma unroll
        for (int q = 0; q < 7; q++) {  // contact_param_vjp's (tc, dr, dmin, dmax, width, mid, mu) -> the words' order
          float sv = mine ? mg[q] : 0.f;
          for (int o = 16; o > 0; o >>= 1) sv += __shfl_xor_sync(0xffffffffu, sv, o);
          if (lane == 0) s.omg[OBJ_MW * k + (q == 6 ? 0 : q + 1)] -= sv;  // (the gradient is -P)
        }
      }
    }
"""


def material_engine(src, m, terrain):
    """The final engine text (adjoint.cuda_source, after world_params / world_forces / velocity_grad_out) with the
    contact materials of the object slots (TerrainConfig.object_material, OBJ_MAT) and of the static terrain's elements
    (TerrainConfig.terrain_material, TER_MAT); else unchanged."""
    cfg = _cfg(terrain) if terrain is not None else None
    objm, term = (bool(getattr(cfg, "object_material", False)), bool(getattr(cfg, "terrain_material", False))) \
        if cfg is not None else (False, False)
    sand = bool(getattr(cfg, "terrain_sand", False)) if cfg is not None else False
    if not (objm or term or sand):
        return src
    what = "TerrainConfig." + ("object_material" if objm else "terrain_material")
    if (objm or term) and "#define HAS_CONDIM4 1" in src:
        raise ValueError(f"{what}: not with condim-4 (torsional) contact pairs")
    if (objm or term) and int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        raise ValueError(f"{what}: the pyramidal cone only")
    from nablaengine.g1_engine import world_params as G
    decl = MAT_DECL.replace("@OBJ@", str(int(objm))).replace("@TER@", str(int(term or sand)))
    if sand:
        from nablaengine.forest_engine import foot_offset as _fo
        decl = _sub1(decl, "#define TER_MW 8\n", f"#define TER_MW {_fo.sand_words(cfg)}  // (with the sand fields: "
                     "TerrainConfig.terrain_sand)\n", "TER_MW (terrain_sand)")
    src = _sub1(src, _MAT_DECL_AT, decl + _MAT_DECL_AT, "forward (contact material)")
    if not (objm or term):
        return src
    if src.count(_MAT_MU) != 2:
        raise RuntimeError(f"forest terrain (contact material): contact friction sites {src.count(_MAT_MU)}, expected 2")
    src = src.replace(_MAT_MU, _MAT_MU_NEW)
    src = _sub1(src, decl, decl.replace(_EL_MU_AT, _EL_MU), "forward (contact material: el_mu)")
    src = _sub1(src, _MAT_KB, _MAT_KB_NEW + _MAT_KB, "forward (contact material impedance)")
    if G.DIRECT_SOLREF[0][0] in src:
        src = _sub1(src, G.DIRECT_SOLREF[0][0], G.DIRECT_SOLREF[0][1], "kbimp (contact material)")
    src = _sub1(src, _MAT_ADJ_MU, _MAT_ADJ_MU_NEW, "adjoint (contact material mu)")
    if _MAT_ADJ_IMP_P in src:
        src = _sub1(src, _MAT_ADJ_IMP_P, _MAT_ADJ_SEL + _MAT_ADJ_IMP_P, "adjoint (contact material impedance)")
    else:
        src = _sub1(src, _MAT_ADJ_IMP, "  const float *sref = PSOLREF(pair), *simp = PSOLIMP(pair);\n" + _MAT_ADJ_SEL
                    + _MAT_ADJ_IMP_P, "adjoint (contact material impedance)")
    pf = "  if (PFOOT(pair) && zone) {  // the foot pair's parameters (world_params)\n"
    if pf in src:
        src = _sub1(src, pf, pf.replace("zone) {", "zone && em_md != 2) {"), "adjoint (contact material, foot "
                    "parameters)")
        src = _sub1(src, _MAT_PB, "  if (PFOOT(pair) && zone && em_mu) pp[6] = 0.f;  // (OBJ_MAT / TER_MAT)\n" + _MAT_PB,
                    "adjoint (contact material, foot friction)")
    if not objm:
        return src
    head = "__device__ __forceinline__ float adj_contact_one(Adj& s, int c, int pair, int slot, unsigned zone,"
    if "void contact_param_vjp(" not in src:
        src = _sub1(src, head, G.PARTIALS_DEVICE.lstrip("\n") + head, "adjoint (object material partials)")
    sig = "                                                 float* dbg"
    i = src.index(head)
    j = src.index(sig, i)
    if j - i > 400:
        raise RuntimeError("forest terrain (object material): adj_contact_one's dbg parameter")
    src = src[:j] + sig + ", float* mg" + src[j + len(sig):]
    src = _sub1(src, "adj_contact_one(s, c, pair, slot, zone, cw, dbg", "adj_contact_one(s, c, pair, slot, zone, cw, dbg, mg",
                "adjoint (object material call)")
    src = _sub1(src, _MAT_PB, _MAT_PARTIALS + _MAT_PB, "adjoint (object material partials)")
    cw = "    float cw[CS_N];  // the lane's contact (adj_contact_one), read by the body lanes by shuffles\n"
    src = _sub1(src, cw, cw + "    float mg[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};  // (OBJ_MAT) its slot material's "
                "partials\n", "adjoint (object material words)")
    src = _sub1(src, "    APROF(APROF_CONTACT_ROWS);\n", "    APROF(APROF_CONTACT_ROWS);\n" + _MAT_REDUCE,
                "adjoint (object material reduction)")
    ub = "  float qbar[NQ], vbar[NV], ubar[NU];  // cotangents of the substep's output, then of its input; ctrl (summed)\n"
    src = _sub1(src, ub, ub + "  float omg[OBJ_MW * TERRAIN_NOBJ];  // (forest_engine.terrain, OBJ_MAT) the slots' material "
                "cotangents (summed)\n", "adjoint (object material state)")
    z = "  for (int i = lane; i < NU; i += 32) s.ubar[i] = 0.f;\n"
    src = _sub1(src, z, z + "  for (int i = lane; i < OBJ_MW * TERRAIN_NOBJ; i += 32) s.omg[i] = 0.f;  // (OBJ_MAT)\n",
                "adjoint (object material zero)")
    return material_grad_out(src, terrain)


def material_grad_out(src, terrain):
    """k_adjoint's gctrl rows with the slots' material cotangents last (after ctrl's, the parameters' and the motion
    words')."""
    vel = _cfg(terrain).object_velocity
    pw = "NPARAM" in src and "gctrl[(size_t)w * (NU + NPARAM + 1" in src
    x = "NU" + (" + NPARAM + 1" if pw else "") + (" + OBJ_VW * TERRAIN_NOBJ" if vel else "")
    old = "gctrl[(size_t)w * NU + " if x == "NU" else f"gctrl[(size_t)w * ({x}) + "
    new = f"gctrl[(size_t)w * ({x} + OBJ_MW * TERRAIN_NOBJ) + "
    n = src.count(old)
    if n < 1:
        raise RuntimeError(f"forest terrain (object material): k_adjoint's gctrl rows not found ({old!r})")
    src = src.replace(old, new)
    a = "  if (lane == 0) info[w] = s.flags;\n"
    return _sub1(src, a, "  __syncwarp();\n  for (int i = lane; i < OBJ_MW * TERRAIN_NOBJ; i += 32) "
                 f"gctrl[(size_t)w * ({x} + OBJ_MW * TERRAIN_NOBJ) + {x} + i] = s.omg[i];  // (OBJ_MAT)\n" + a,
                 "adjoint (object material cotangents)")


def velocity_grad_out(src, terrain):
    """k_adjoint's gctrl rows with the slots' motion cotangents last (TerrainConfig.object_velocity; on the final
    text: after world_params' row width NU + NPARAM + 1); else unchanged."""
    if terrain is None or not _cfg(terrain).object_velocity:
        return src
    a = "  for (int i = lane; i < NU; i += 32) gctrl[(size_t)w * NU + i] = s.ubar[i];\n"
    if a in src:
        return _sub1(src, a, "  for (int i = lane; i < NU; i += 32) gctrl[(size_t)w * (NU + OBJ_VW * TERRAIN_NOBJ) + i] = "
                     "s.ubar[i];\n  __syncwarp();\n  for (int i = lane; i < OBJ_VW * TERRAIN_NOBJ; i += 32) "
                     "gctrl[(size_t)w * (NU + OBJ_VW * TERRAIN_NOBJ) + NU + i] = s.ovbar[i];  // (OBJ_VEL)\n",
                     "adjoint (object velocity cotangents)")
    p = "gctrl[(size_t)w * (NU + NPARAM + 1) + "
    if src.count(p) != 2:
        raise RuntimeError(f"forest terrain: k_adjoint's gctrl rows not found ({src.count(p)} parameter writes)")
    src = src.replace(p, "gctrl[(size_t)w * (NU + NPARAM + 1 + OBJ_VW * TERRAIN_NOBJ) + ")
    a = "NU + i] = s.gpw[i];\n"
    return _sub1(src, a, a + "  __syncwarp();\n  for (int i = lane; i < OBJ_VW * TERRAIN_NOBJ; i += 32) gctrl[(size_t)w * "
                 "(NU + NPARAM + 1 + OBJ_VW * TERRAIN_NOBJ) + NU + NPARAM + 1 + i] = s.ovbar[i];  // (OBJ_VEL)\n",
                 "adjoint (object velocity cotangents)")


def _apply(src, hooks, what):
    for a, b in hooks:
        src = _sub1(src, a, b, what)
    return src


def step_source(src, terrain, m=None):
    """The forest's STEP_DEVICE + STEP_KERNEL (src) with the terrain's device code and hooks (m: the model, for box
    robot geoms' CCD text, box_ccd_device)."""
    dev = device_text(src)
    if m is not None and has_box_geoms(m, terrain):
        dev = box_ccd_device(m) + dev
    if _cfg(terrain).object_velocity:
        dev = dev + OBJ_VEL_SLOT
    src = _apply(src, [(a, dev + a if b is None else b) for a, b in STEP_HOOKS], "forward")
    if _cfg(terrain).object_velocity:
        src = _apply(src, OBJ_VEL_FORWARD, "forward (object velocity)")
    if _cfg(terrain).mode == "native":
        src = _apply(src, STEP_NATIVE_HOOKS, "forward (native)")
    if m is not None and step.has_defer(m):
        src = _apply(src, DEFER_ENGINE_HOOKS, "forward (Hand)")
    if m is not None and sorted_build(m):
        src = _apply(src, SORT_ELEM_HOOKS, "forward (sort_contacts)")
        if _cfg(terrain).mode == "native":
            src = _apply(src, SORT_AUX_HOOKS, "forward (sort_contacts, native)")
    return src


def sorted_build(m):
    """Whether the forest build of m sorts its contacts by (pair, slot) after the collision (step.sort_contacts: the
    split forward's deferred pairs, step.has_defer; the collision mid-phase, midphase.enabled)."""
    from nablaengine.forest_engine import midphase
    return step.has_defer(m) or midphase.enabled(m)


SORT_ELEM_HOOKS = (
    ("""  unsigned char sl[CPL];
  float mu[CPL], ds[CPL], ps[CPL][3], fr[CPL][9];
""", """  unsigned char sl[CPL];
  float mu[CPL], ds[CPL], ps[CPL][3], fr[CPL][9];
  unsigned short el[CPL];  // (forest_engine.terrain: the terrain element)
"""),
    ("""      pr[h] = s.con_pair[c]; sl[h] = s.con_slot[c]; dm[h] = s.con_dim[c]; mu[h] = s.con_mu[c]; ds[h] = s.con_dist[c];
""", """      pr[h] = s.con_pair[c]; sl[h] = s.con_slot[c]; dm[h] = s.con_dim[c]; mu[h] = s.con_mu[c]; ds[h] = s.con_dist[c];
      el[h] = s.con_elem[c];
"""),
    ("""      s.con_pair[r] = pr[h]; s.con_slot[r] = sl[h]; s.con_dim[r] = dm[h]; s.con_mu[r] = mu[h]; s.con_dist[r] = ds[h];
""", """      s.con_pair[r] = pr[h]; s.con_slot[r] = sl[h]; s.con_dim[r] = dm[h]; s.con_mu[r] = mu[h]; s.con_dist[r] = ds[h];
      s.con_elem[r] = el[h];
"""),
)
SORT_AUX_HOOKS = (
    ("""  unsigned short el[CPL];  // (forest_engine.terrain: the terrain element)
""", """  unsigned short el[CPL];  // (forest_engine.terrain: the terrain element)
  unsigned ax[CPL];  // (the native candidate id)
"""),
    ("""      el[h] = s.con_elem[c];
""", """      el[h] = s.con_elem[c];
      ax[h] = s.con_aux[c];
"""),
    ("""      s.con_elem[r] = el[h];
""", """      s.con_elem[r] = el[h];
      s.con_aux[r] = ax[h];
"""),
)

DEFER_ENGINE_HOOKS = (
    ("""  float gx[NGEOMC][3], gq[NGEOMC][4];  // k_pre_a's geom poses, for k_ccd
  unsigned char con_slot[MAXCON];
""", """  float gx[NGEOMC][3], gq[NGEOMC][4];  // k_pre_a's geom poses, for k_ccd
  unsigned char con_slot[MAXCON];
  unsigned short con_elem[MAXCON];  // (forest_engine.terrain: the contacts' terrain elements)
#if TERRAIN_HF_NATIVE
  unsigned con_aux[MAXCON];
#endif
"""),
)
DEFER_HOST_HOOKS = (
    ("""      s.con_pair[c] = g.con_pair[c]; s.con_slot[c] = g.con_slot[c]; s.con_dim[c] = g.con_dim[c];
      s.con_mu[c] = g.con_mu[c];
""", """      s.con_pair[c] = g.con_pair[c]; s.con_slot[c] = g.con_slot[c]; s.con_dim[c] = g.con_dim[c];
      s.con_mu[c] = g.con_mu[c];
      s.con_elem[c] = g.con_elem[c];  // (forest_engine.terrain)
#if TERRAIN_HF_NATIVE
      s.con_aux[c] = g.con_aux[c];
#endif
"""),
    ("""    g.con_pair[c] = s.con_pair[c]; g.con_slot[c] = s.con_slot[c]; g.con_dim[c] = s.con_dim[c];
    g.con_mu[c] = s.con_mu[c];
""", """    g.con_pair[c] = s.con_pair[c]; g.con_slot[c] = s.con_slot[c]; g.con_dim[c] = s.con_dim[c];
    g.con_mu[c] = s.con_mu[c];
    g.con_elem[c] = s.con_elem[c];  // (forest_engine.terrain)
#if TERRAIN_HF_NATIVE
    g.con_aux[c] = s.con_aux[c];
#endif
"""),
    ("""    g.con_pair[c] = pair; g.con_slot[c] = (unsigned char)o.slot[j]; g.con_dim[c] = PDIM(pair);
""", """    g.con_pair[c] = pair; g.con_slot[c] = (unsigned char)o.slot[j]; g.con_dim[c] = PDIM(pair);
    g.con_elem[c] = 0;  // (forest_engine.terrain: a model pair)
#if TERRAIN_HF_NATIVE
    g.con_aux[c] = 0u;
#endif
"""),
)


_COM_LOAD = "  if (lane < 3) s.com[lane] = com[(size_t)w * COM_WORDS + lane];\n"
_OBJ_PTR = ("  dk_obj = com + (size_t)w * COM_WORDS + 3 + 7 * NFIX;  // the world's object slots (terrain objects, "
            "forest_engine.terrain)\n")
OBJECT_HOOKS = {"host": ((_COM_LOAD, _COM_LOAD + _OBJ_PTR, 3),),
                "adjoint": ((_COM_LOAD, _COM_LOAD + _OBJ_PTR, 1),),
                "kinematics": ()}


_CF_KSUB = "    if (lane == 0) {  // (contact_forces: the table carries the forces, oforce stays null)\n"


def object_hooks(src, hooks, terrain, m=None):
    """src with OBJECT_HOOKS[hooks] (step.OBJECT_*_HOOKS) for a terrain build with object slots, else unchanged; m (the
    model; hooks "host"): with deferred pairs (step.has_defer) also DEFER_HOST_HOOKS."""
    if m is not None and hooks == "host" and step.has_defer(m):
        src = _apply(src, DEFER_HOST_HOOKS, "host (deferred pairs)")
    if not _cfg(terrain).objects:
        return src
    vel, mat = _cfg(terrain).object_velocity, getattr(_cfg(terrain), "object_material", False)
    ptr = _OBJ_PTR + ("  dk_objc = dk_obj;  // (OBJ_VEL: obj_advance's source)\n" if vel else "")
    if mat:
        from nablaengine import terrain as T
        off = (T.OBJ_WORDS + (T.OBJ_VEL_WORDS if vel else 0)) * int(_cfg(terrain).objects)
        ptr += f"  dk_omat = dk_obj + {off};  // the world's slot materials (forest_engine.terrain, OBJ_MAT)\n"
    extra = {"host": OBJ_VEL_HOST, "adjoint": OBJ_VEL_ADJ_KERNEL, "kinematics": ()}[hooks] if vel else ()
    for a, b, n in tuple((a, b.replace(_OBJ_PTR, ptr), n) for a, b, n in OBJECT_HOOKS[hooks]) + extra:
        if src.count(a) == 0 and a == OBJ_VEL_HOST[0][0] and src.count(_CF_KSUB) == 1:
            a, b = _CF_KSUB, b.replace(OBJ_VEL_HOST[0][0], _CF_KSUB)
        if src.count(a) != n:
            raise RuntimeError(f"forest object hook anchor found {src.count(a)} times, expected {n}: {a[:70]!r}")
        src = src.replace(a, b)
    return src


def _adj_hooks():
    from nablaengine.dual import DUAL_DEVICE
    from nablaengine.g1_engine import adjoint as g1_adj, adjoint_contact as g1_ac
    return (
        ("// adj_contact_geom for contact (pair, slot) of the engine's pair table",
         DUAL_DEVICE + g1_adj.capbox_dual_source() + g1_adj.CBTER_DIR_VJP_TREE + g1_ac.TERRAIN_DEVICE.lstrip("\n")
         + "\n// adj_contact_geom for contact (pair, slot) of the engine's pair table"),
        ("  if (!cot) return br;\n  const float* pb = cot + 1;\n",
         "  if (!cot) return br;\n" + g1_adj.ADJ_CAPBOX_DUAL_TREE + "  const float* pb = cot + 1;\n"),
        ("""  int g1 = PG1(pair), g2 = PG2(pair);
  return adj_contact_geom(""",
         """  if (pair >= NPAIR)  // a terrain contact (forest_engine.terrain)
    return adj_terrain_geom(slot, 0u, c_gtype[PG2(pair)], c_gsize + 3 * PG2(pair), gx[PG2(pair)], gq[PG2(pair)], dist,
                            pos, fr, nu, cot, gb);
  int g1 = PG1(pair), g2 = PG2(pair);
  return adj_contact_geom("""),
        ("""  unsigned con[MAXCON];  // the record's contact table: pair | slot << 16
""",
         """  unsigned con[MAXCON];  // the record's contact table: pair | slot << 16
  unsigned telem[(MAXCON + 1) / 2];  // the contacts' terrain elements, 16 bits each (terrain build)
"""),
        ("""  for (int i = lane; i < MAXCON; i += 32) s.con[i] = __float_as_uint(__ldcs(rec + OFF_CON + i));
""",
         """  for (int i = lane; i < MAXCON; i += 32) s.con[i] = __float_as_uint(__ldcs(rec + OFF_CON + i));
  for (int i = lane; i < (MAXCON + 1) / 2; i += 32) s.telem[i] = __float_as_uint(__ldcs(rec + OFF_TELEM + i));
"""),
        ("""    const int pair = e & 0xFFFFu, slot = e >> 16, dim = on ? PDIM(pair) : 0;""",
         """    const unsigned x = on ? (s.telem[c >> 1] >> (16 * (c & 1))) & 0xFFFFu : 0u;  // (terrain: the element)
    const int pair = e & 0xFFFFu, slot = (int)(e >> 16 | x << 8), dim = on ? PDIM(pair) : 0;"""),
    )


ADJ_NATIVE_HOOKS = (
    ("""  unsigned telem[(MAXCON + 1) / 2];  // the contacts' terrain elements, 16 bits each (terrain build)
""",
     """  unsigned telem[(MAXCON + 1) / 2];  // the contacts' terrain elements, 16 bits each (terrain build)
  unsigned taux[MAXCON];  // native heightfield contacts: the candidate ids
"""),
    ("""  for (int i = lane; i < (MAXCON + 1) / 2; i += 32) s.telem[i] = __float_as_uint(__ldcs(rec + OFF_TELEM + i));
""",
     """  for (int i = lane; i < (MAXCON + 1) / 2; i += 32) s.telem[i] = __float_as_uint(__ldcs(rec + OFF_TELEM + i));
  for (int i = lane; i < MAXCON; i += 32) s.taux[i] = __float_as_uint(__ldcs(rec + OFF_TAUX + i));
"""),
    ("""// a geom's pose cotangent (xbar, tbar = gb + 6 j of adj_contact_geom) onto its body's:""",
     """// adj_pair_geom with a terrain contact's candidate id
__device__ __forceinline__ int adj_pair_geom_aux(int pair, int slot, unsigned aux, const float (*gx)[3],
                                                 const float (*gq)[4], float* dist, float* pos, float* fr, float* nu,
                                                 const float* cot, float* gb) {
  if (pair >= NPAIR)
    return adj_terrain_geom(slot, aux, c_gtype[PG2(pair)], c_gsize + 3 * PG2(pair), gx[PG2(pair)], gq[PG2(pair)], dist,
                            pos, fr, nu, cot, gb);
  return adj_pair_geom(pair, slot, gx, gq, dist, pos, fr, nu, cot, gb);
}

// a geom's pose cotangent (xbar, tbar = gb + 6 j of adj_contact_geom) onto its body's:"""),
    ("""adj_pair_geom(pair, slot, s.gx, s.gq, &d, p, fr, &nu, nullptr, nullptr)""",
     """adj_pair_geom_aux(pair, slot, s.taux[c], s.gx, s.gq, &d, p, fr, &nu, nullptr, nullptr)"""),
    ("""adj_pair_geom(pair, slot, s.gx, s.gq, &d2, p2, fr2, &nu2, cot, gb)""",
     """adj_pair_geom_aux(pair, slot, s.taux[c], s.gx, s.gq, &d2, p2, fr2, &nu2, cot, gb)"""),
)


def adjoint_source(src, terrain):
    """The forest's contact adjoint (adjoint_contact.source(m) + ADJ_DEVICE + ADJ_KERNEL) with the terrain contacts and
    the object pointer."""
    src = _apply(src, _adj_hooks(), "adjoint")
    if _cfg(terrain).object_velocity:
        src = _apply(src, OBJ_VEL_ADJOINT + OBJ_VEL_GRAD, "adjoint (object velocity)")
    if _cfg(terrain).mode == "native":
        src = _apply(src, ADJ_NATIVE_HOOKS, "adjoint (native)")
    return object_hooks(src, "adjoint", terrain)


def coop_dispatch(src):
    """capbox_coop_hooks' dispatch for a terrain build: the terrain capsule-box contacts (g1_engine.model_hooks.
    tree_coop_dispatch's terrain part: pair >= NPAIR, the capsule's 6 directions) and, if the model has its own
    sphere-box / capsule-box pairs (forest_engine.model_hooks.CB_COOP_DISPATCH in src), theirs (pair < NPAIR, 12
    directions) in one cb_coop_dir."""
    from nablaengine.g1_engine import model_hooks as g1_hooks
    from nablaengine.forest_engine import model_hooks as fh
    ter = g1_hooks.tree_coop_dispatch(False, True)
    if fh.CB_COOP_DISPATCH not in src:
        return ter
    sig = "                                             const float* rec) {\n"
    p0, p1 = "#define CB_COOP_PAIR(pair, slot) (", ")  // (model_hooks.tree_coop_dispatch)\n"
    head, tcond = ter[:ter.index(p0)], ter[ter.index(p0) + len(p0):ter.index(p1)]
    tbody = ter[ter.index(sig) + len(sig):ter.rindex("  return 0.f;\n}\n")]
    f = fh.CB_COOP_DISPATCH
    f0, f1 = "#define CB_COOP_PAIR(pair, slot) ", "\n#define CB_COOP_WIDE"
    fcond = f[len(f0):f.index(f1)]
    fsig = "const float* rec) {\n"
    fbody = f[f.index(fsig) + len(fsig):f.rindex("}\n")]
    assert f.startswith(f0) and tbody.startswith("  if (pair >= NPAIR) {"), (f[:40], tbody[:40])
    return (head + f"#define CB_COOP_PAIR(pair, slot) (((pair) < NPAIR && {fcond}) || {tcond})  "
            "// (forest_engine.terrain.coop_dispatch)\n"
            "#define CB_COOP_WIDE(pair, slot) ((pair) < NPAIR)  // (the model's pairs: both geoms' 12 directions; "
            "terrain: the capsule's 6)\n"
            "// (forest_engine.terrain.coop_dispatch) direction j's gb[j] of contact (pair, slot), rec: cot[0..3], nbar\n"
            "__device__ __forceinline__ float cb_coop_dir(const float (*gx)[3], const float (*gq)[4], int pair, int slot,"
            " int j,\n" + sig + tbody + fbody + "}\n\n")


def coop_hooks(src):
    """The cooperative pass over the terrain capsule-box contacts' pose directions (DK_CAPBOX_COOP;
    g1_engine.model_hooks.capbox_coop_hooks with coop_dispatch), on the full text after model_hooks: the tree engine's
    cb_coop (g1_engine.model_hooks.TREE_COOP_DEV, tree_coop_dev: the batch's directions spread evenly over the warp,
    ceil(6 n / 32) dual replays for n terrain contacts, geom_to_body_vjp once per contact; each direction the same
    cb_coop_dir on the same inputs as the earlier packing CAPBOX_COOP_DEV, so the same gradients bit for bit). A model
    whose own pairs already have the pass (sphere-box / capsule-box pairs: forest_engine.model_hooks' capbox_coop_hooks
    with CAPBOX_COOP_DEV) gets its dispatch replaced by coop_dispatch's (both kinds in one cb_coop_dir) and its cb_coop
    by tree_coop_dev(True) (contacts with 12 directions); the forest's builds without terrain keep CAPBOX_COOP_DEV."""
    from nablaengine.g1_engine import model_hooks as g1_hooks
    from nablaengine.forest_engine import model_hooks as fh
    disp = coop_dispatch(src)
    if fh.CB_COOP_DISPATCH in src:
        for a, b in ((fh.CB_COOP_DISPATCH, disp), (g1_hooks.CAPBOX_COOP_DEV, g1_hooks.tree_coop_dev(True))):
            if src.count(a) != 1:
                raise RuntimeError(f"forest terrain coop: the model's pass found {src.count(a)} times: {a[:70]!r}")
            src = src.replace(a, b)
        return src
    if "CB_COOP_PAIR" in src:
        raise RuntimeError("forest terrain coop: a cooperative pass with an unknown dispatch")
    for a, b, n in g1_hooks.capbox_coop_hooks(disp, g1_hooks.tree_coop_dev(False)):
        if src.count(a) != n:
            raise RuntimeError(f"forest terrain coop hook anchor found {src.count(a)} times, expected {n}: {a[:70]!r}")
        src = src.replace(a, b)
    return src


def box_ccd_device(m):
    """The forward's text for box robot geoms against terrain boxes (the tree engine's, model_hooks.box_box_hook):
    nablaengine.ccd's native box-box CCD (ccd_box_box; its CSink struct is the forest's, step.py) with its #defines
    (ccd.header) and boxes_separated (the separating-axis prefilter)."""
    import re as _re
    from nablaengine import ccd
    from nablaengine.g1_engine import model_hooks as g1_hooks
    dev, n = _re.subn(r"#define SINK_MAX 8\nstruct CSink \{.*?\};\n", "", ccd.DEVICE, count=1, flags=_re.S)
    if n != 1:
        raise RuntimeError("forest terrain: no CSink struct in nablaengine.ccd's text")
    wrap = g1_hooks.box_box_hook()[1][len(ccd.DEVICE):]
    sep = wrap[:wrap.index("template <class S>\n__device__ void box_box(")]
    assert "__device__ bool boxes_separated(" in sep
    return ("// box robot geoms against the terrain (forest_engine.terrain.box_ccd_device: nablaengine.ccd's box-box CCD, in\n"
            "// namespace tccd: its vector helpers are named as the forest's own)\n"
            + ccd.header(m) + "namespace tccd {\n" + dev + "}  // namespace tccd\nusing tccd::ccd_box_box;\n" + sep)


def box_geom_hooks(src, m, terrain):
    """Box robot geoms against the terrain (the tree engine's g1_engine.model_hooks.TERRAIN_BOX_GEOMS on the full text:
    against a terrain box or object slot MuJoCo Warp's native CCD, ccd_box_box, and its VJP ccd_box_box_vjp; against a
    heightfield the box's 8 corners as sample points), with ccd.ADJ_DEVICE before adj_contact_geom. Unchanged unless a
    box robot geom collides with the terrain."""
    if not has_box_geoms(m, terrain):
        return src
    from nablaengine import ccd
    from nablaengine.g1_engine import model_hooks as g1_hooks
    head = "__device__ __noinline__ int adj_contact_geom("
    adj = "namespace tccd {\n" + ccd.ADJ_DEVICE + "}  // namespace tccd\nusing tccd::ccd_box_box_vjp;\n"
    for a, b, n in ((head, adj + head, 1),) + g1_hooks.TERRAIN_BOX_GEOMS:
        if src.count(a) != n:
            raise RuntimeError(f"forest terrain box hook anchor found {src.count(a)} times, expected {n}: {a[:70]!r}")
        src = src.replace(a, b)
    return src

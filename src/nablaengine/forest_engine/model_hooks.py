"""Edits of the kernels' text for models outside the G1's structure (nv < 32, more than 3 children per body)."""

ENGINE_CHILD = (
    ("DKC int c_parent[NB], c_depth[NB], c_qadr[NB], c_dadr[NB], c_nchild[NB], c_child[NB * 3], c_nanc[NB],",
     "DKC int c_parent[NB], c_depth[NB], c_qadr[NB], c_dadr[NB], c_nchild[NB], c_child[NB * NCH], c_nanc[NB],", 1),
    ("c_child[3 * b + i]", "c_child[NCH * b + i]", 3),
)
KIN_CHILD = (
    ("  for (int i = 0; i < 3; i++) ch[i] = i < n ? c_child[3 * b + i] : 0;",
     "  for (int i = 0; i < NCH; i++) ch[i] = i < n ? c_child[NCH * b + i] : 0;", 1),
    ("  int ch[3];", "  int ch[NCH];", 2),
)
ENGINE_SMALL_NV = (
    ("  const int r0 = lane, r1 = lane + 32;\n  const bool has1 = r1 < NV;",
     "  const int r0 = min(lane, NV - 1), r1 = lane + 32;  // lanes >= NV repeat row NV - 1 (model_hooks)\n"
     "  const bool has1 = r1 < NV;", 2),
    ("  const int r0 = lane, r1 = min(lane + 32, NV - 1);",
     "  const int r0 = min(lane, NV - 1), r1 = min(lane + 32, NV - 1);  // (model_hooks)", 1),
)

ENGINE_ADJ_SCOPE = (
    ('static_assert(!HAS_BOX && NFLOSS == 0, "the contact adjoint is for the mjlab model: no boxes, no frictionloss rows");',
     '// (model_hooks: plane-box contacts and frictionloss rows in the contact adjoint)', 1),
    ('static_assert(HAS_BOX == 0 && NFLOSS == 0, "the adjoint is for the mjlab model (no boxes, no frictionloss rows)");',
     '// (model_hooks: plane-box contacts and frictionloss rows in the adjoint)', 1),
)
ENGINE_PLANE_BOX = (
    ("    if (t2 != 2) return -1;\n",
     """    if (t2 == 3) {  // plane-box corner `slot` (model_hooks): the forward's collide_pair arithmetic, then its VJP
      float cl[3] = {(slot & 1) ? sz2[0] : -sz2[0], (slot & 2) ? sz2[1] : -sz2[1], (slot & 4) ? sz2[2] : -sz2[2]}, cw[3];
      qrot(q2, cl, cw);
      float cd = dot3(p2, n) - dot3(p1, n);
      *dist = cd + dot3(n, cw);
      for (int k = 0; k < 3; k++) pos[k] = cw[k] + p2[k] - 0.5f * n[k] * (*dist);
      make_frame(n, fr);
      if (!(-0.5f < n[1] && n[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
      if (cot) {  // c = x + cw: xbar = cbar, tbar = cw x cbar; cbar = pbar - n (n . pbar) / 2 + n dbar
        float pn = dot3(n, cot + 1);
        for (int k = 0; k < 3; k++) gb[6 + k] = cot[1 + k] + n[k] * (cot[0] - 0.5f * pn);
        cross(cw, gb + 6, gb + 9);
      }
      return br;
    }
    if (t2 != 2) return -1;\n""", 1),
)

ENGINE_DAMPING = (
    ("    c_inert[NB * 3], c_grav[3], c_armature[NV], c_kp[NV], c_kv[NV], c_frclim[NV], c_misc[2];\n",
     "    c_inert[NB * 3], c_grav[3], c_armature[NV], c_kp[NV], c_kv[NV], c_frclim[NV], c_misc[2];\n"
     "DKC float c_damp[NV];  // joint damping (model_hooks)\n", 1),
    ("      s.qsm[i] = act - s.qsm[i];",
     "      s.qsm[i] = act - c_damp[i] * s.qvel[i] - s.qsm[i];  // the passive damping force (model_hooks)", 1),
    ("{ s.L[TRI(i, i)] += h * c_kv[i]; s.tmp[i] = s.Ma[i]; }",
     "{ s.L[TRI(i, i)] += h * (c_kv[i] + c_damp[i]); s.tmp[i] = s.Ma[i]; }", 1),
    ("{ s.abar[i] -= h * c_kv[i] * s.y[i]; s.ny[i] = -s.y[i]; }",
     "{ s.abar[i] -= h * (c_kv[i] + c_damp[i]) * s.y[i]; s.ny[i] = -s.y[i]; }", 1),
    ("{ float x = s.did[i] - h * c_kv[i] * s.a[i]; acc += x * x; }",
     "{ float x = s.did[i] - h * (c_kv[i] + c_damp[i]) * s.a[i]; acc += x * x; }", 1),
    ("    s.ubar[i - 6] += kl;\n  }\n  __syncwarp();\n",
     "    s.ubar[i - 6] += kl;\n  }\n  __syncwarp();\n"
     "  for (int i = lane; i < NV; i += 32) s.Pv[i] += c_damp[i] * s.lam[i];  // joint damping (model_hooks)\n"
     "  __syncwarp();\n", 1),
    ("    float r = s.did[i] + c_armature[i] * s.a[i] - tau - s.jtf[i];",
     "    float r = s.did[i] + c_armature[i] * s.a[i] - tau + c_damp[i] * s.v[i] - s.jtf[i];", 1),
)

ENGINE_FLOSS = (
    ("      const int r = nl + __popc(bal & below);",
     "      const int r = NFLOSS + nl + __popc(bal & below);  // after the frictionloss rows (model_hooks)", 1),
    ("    const int r0 = nlim + c + 3 * (ntri + __popc(b3 & below));  // the contact's first row",
     "    const int r0 = NFLOSS + nlim + c + 3 * (ntri + __popc(b3 & below));  // the contact's first row (model_hooks)",
     1),
    ("""  const unsigned below = (1u << lane) - 1u;
  int nl = 0;
#pragma unroll 1
  for (int i0 = 0; i0 < NV; i0 += 32) {
    const int i = min(i0 + lane, NV - 1);  // clamped so that the loads need no branch (as the forward)""",
     """  const unsigned below = (1u << lane) - 1u;
  // the frictionloss rows (model_hooks): row r = the number of frictionloss dofs before (dof order, the first rows),
  // J = e_i, pos 0, aref = -b v_i (substep_pre), constant D; on its quadratic zone (the tape's bit) the row force
  // -D (a_i + b v_i) gives P_v += D b lam (limit_row_vjp without the position terms); its linear pieces are constant
  int nf = 0;
#pragma unroll 1
  for (int i0 = 0; i0 < NV; i0 += 32) {
    const int i = i0 + lane;
    const bool on = i < NV && c_floss[i] > 0.f;
    const unsigned bal = __ballot_sync(0xffffffffu, on);
    if (on) {
      const int r = nf + __popc(bal & below);
      if ((s.zone[r >> 5] >> (r & 31)) & 1u) {
        float ri[6];
        adj_row_impedance(c_dsolref, c_dsolimp, 0.f, c_dinvw[i], ri);
        s.Pv[i] += ri[4] * ri[1] * s.lam[i];
      }
    }
    nf += __popc(bal);
  }
  int nl = 0;
#pragma unroll 1
  for (int i0 = 0; i0 < NV; i0 += 32) {
    const int i = min(i0 + lane, NV - 1);  // clamped so that the loads need no branch (as the forward)""", 1),
)


from nablaengine.dual import DUAL_DEVICE


def has_sphere_box(m):
    """A sphere-box contact pair (the forward's collide_pair needs ENGINE_SPHERE_BOX)."""
    return _has_pair(m, "SPHERE", "BOX")


def has_capsule_box(m):
    """A capsule-box contact pair."""
    return _has_pair(m, "CAPSULE", "BOX")


def _has_pair(m, a, b):
    import mujoco
    want = sorted((int(getattr(mujoco.mjtGeom, "mjGEOM_" + a)), int(getattr(mujoco.mjtGeom, "mjGEOM_" + b))))
    return any(sorted((int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]]))) == want
               for p in range(m.npair))


def capbox_source():
    """step.py's sphere_box and capsule_box's candidate search (g1_engine.step.capbox_segs_source on the forest
    engine's text) as templates over the scalar type, generated from their text, with the candidate's sphere centre and
    a contact slot's replay: cb_sphere_box_t, cb_segs_t, cb_point_t, cb_slot_t. Over RF the forward computes the bits
    that the adjoint's D1 replay (ADJ_CAPBOX) takes, its branches included (nablaengine.dual)."""
    import re
    from nablaengine.dual import RF_DEVICE
    from nablaengine.forest_engine import step
    from nablaengine.g1_engine import step as g1_step
    src = step.STEP_DEVICE
    a = re.search(r"__device__ void sphere_box\(", src).start()
    body = src[a:src.index("\n}\n", a) + 3] + g1_step.capbox_segs_source(src)
    body = re.sub(r"\bfloat\b", "T", body)
    body = re.sub(r"\b(fabs|sqrt|fmin|fmax)f\(", r"t_\1(", body)
    body = re.sub(r"\b(mv3|mtv3|mm3|mmt3|dot3|safe_div)\(", r"t_\1(", body)
    body = re.sub(r"\bsphere_box\(", "cb_sphere_box_t(", body)
    body = re.sub(r"\bcapbox_segs\(", "cb_segs_t(", body)
    body = re.sub(r"__device__ (__noinline__ )?(void|int) ", r"template <class T>\n__device__ \1\2 ", body)
    if "add_contact" in body or "atomic" in body:
        raise RuntimeError("capbox_source: unexpected side effects in the generated text")
    rf = "#ifndef DK_RF_DEVICE\n#define DK_RF_DEVICE\n" + RF_DEVICE + "#endif\n"
    return ("// sphere-box and capsule-box contacts over a scalar type (model_hooks.capbox_source: step.py's sphere_box"
            " and\n// capsule_box's candidate search, generated from their text)\n" + DUAL_DEVICE + rf + body + r"""
// the centre of capsule_box's candidate sphere at segment parameter seg (cb_segs_t), capsule_box's arithmetic
template <class T>
__device__ void cb_point_t(const T* cp, const T* axisw, T hl, const T* bp, const T* R, T seg, T* ce) {
  T d[3] = {cp[0] - bp[0], cp[1] - bp[1], cp[2] - bp[2]}, pos[3], axis[3], lc[3], t[3];
  t_mtv3(R, d, pos);
  t_mtv3(R, axisw, axis);
  for (int k = 0; k < 3; k++) lc[k] = pos[k] + axis[k] * hl * seg;
  t_mv3(R, lc, t);
  for (int k = 0; k < 3; k++) ce[k] = bp[k] + t[k];
}

// contact `slot` of a sphere (t1 1: slot 0) or capsule (t1 2: candidate slot; a1 its axis) against a box (R2 row
// major, sz2 its half sizes): 1 and its distance, position, normal (from geom 1 towards the box) when it exists
template <class T>
__device__ int cb_slot_t(int t1, int slot, const T* p1, const T* a1, const float* sz1, const T* p2, const T* R2,
                         const float* sz2, T* dist, T* pos, T* nrm) {
  T Z[3] = {T(sz2[0]), T(sz2[1]), T(sz2[2])};
  if (t1 == 1) {
    if (slot != 0) return 0;
    cb_sphere_box_t<T>(p1, T(sz1[0]), p2, R2, Z, dist, pos, nrm);
    return 1;
  }
  T sg[2], C[3];
  if (slot >= cb_segs_t<T>(p1, a1, T(sz1[1]), p2, R2, Z, sg)) return 0;
  cb_point_t<T>(p1, a1, T(sz1[1]), p2, R2, sg[slot], C);
  cb_sphere_box_t<T>(C, T(sz1[0]), p2, R2, Z, dist, pos, nrm);
  return 1;
}

// the forward's sphere-box (t1 1) and capsule-box (t1 2) contacts over RF (the adjoint's replay takes its branches)
template <class S>
__device__ void cb_contacts(S& s, int pair, int t1, const float* p1, const float* a1, const float* sz1, const float* p2,
                            const float* R2, const float* sz2) {
  RF P1[3], A1[3], P2[3], Q2[9], Z[3], sg[2] = {RF(0.f), RF(0.f)};
  for (int k = 0; k < 3; k++) { P1[k] = RF(p1[k]); A1[k] = RF(a1[k]); P2[k] = RF(p2[k]); Z[k] = RF(sz2[k]); }
  for (int k = 0; k < 9; k++) Q2[k] = RF(R2[k]);
  const int ns = t1 == 1 ? 1 : cb_segs_t<RF>(P1, A1, RF(sz1[1]), P2, Q2, Z, sg);
  for (int e = 0; e < ns; e++) {
    RF C[3], dd, pp[3], nn[3];
    if (t1 == 1) for (int k = 0; k < 3; k++) C[k] = P1[k];
    else cb_point_t<RF>(P1, A1, RF(sz1[1]), P2, Q2, sg[e], C);
    cb_sphere_box_t<RF>(C, RF(sz1[0]), P2, Q2, Z, &dd, pp, nn);
    if (dd.v < 0.f) {
      float pos[3] = {pp[0].v, pp[1].v, pp[2].v}, n[3] = {nn[0].v, nn[1].v, nn[2].v}, fr[9];
      make_frame(n, fr);
      add_contact(s, pair, e, dd.v, pos, fr);
    }
  }
}

""")


def capbox_hooks(m):
    """Sphere-box and capsule-box pairs (MuJoCo Warp's sphere_box, capsule_box_witness + capsule_box): the forward's
    contacts over RF (cb_contacts; ahead of box_pair, outside HAS_BOX), and the contact adjoint's replay and VJP
    (ADJ_CAPBOX, part "engine")."""
    hooks = (("#if HAS_BOX\n// capsule-box (t1 2) or box-box (t1 3) into o:",
              capbox_source() + "#if HAS_BOX\n// capsule-box (t1 2) or box-box (t1 3) into o:", 1),
             ("    capsule_box(o, 0, p1, a1, c_gsize[3 * g1], c_gsize[3 * g1 + 1], p2, R2, c_gsize + 3 * g2);\n",
              "    cb_contacts(o, 0, 2, p1, a1, c_gsize + 3 * g1, p2, R2, c_gsize + 3 * g2);  // (model_hooks.capbox_hooks)\n",
              1))
    return hooks + (ENGINE_SPHERE_BOX if has_sphere_box(m) else ())


ENGINE_SPHERE_BOX = (
    ("  if ((t1 == 2 || t1 == 3) && t2 == 3) {  // capsule-box, box-box: past the OBB test, in k_ccd (the split forward)\n",
     """  if (t1 == 1 && t2 == 3) {  // sphere-box (MuJoCo Warp's sphere_box; model_hooks.capbox_hooks: over RF)
    float R2[9];
    quat2mat(q2, R2);
    cb_contacts(s, pair, 1, p1, p1, sz1, p2, R2, sz2);
    return;
  }
  if ((t1 == 2 || t1 == 3) && t2 == 3) {  // capsule-box, box-box: past the OBB test, in k_ccd (the split forward)\n""", 1),
)

ADJ_CAPBOX = """  if ((t1 == 1 || t1 == 2) && t2 == 3) {  // sphere-box / capsule-box contact `slot` (model_hooks.capbox_hooks): the
    // forward's RF arithmetic replayed (the forward's bits), then over duals (D1, the same values) once per pose
    // direction of each geom (cb_dir_vjp): exact distance, position and normal tangents
#if CB_COOP_ON
    if (cot) return 0;  // (the VJP: adj_contacts' cooperative pass, cb_coop)
#endif
    float R2[9], a1[3] = {0.f, 0.f, 1.f}, nrm[3];
    quat2mat(q2, R2);
    if (t1 == 2) zaxis_rn(q1, a1);
    {  // (over RF: the forward's arithmetic, the values of the D1 replays without their tangents)
      RF P1[3], A1[3], P2[3], Q2[9], dd, pp[3], nn[3];
      for (int k = 0; k < 3; k++) { P1[k] = RF(p1[k]); A1[k] = RF(a1[k]); P2[k] = RF(p2[k]); }
      for (int k = 0; k < 9; k++) Q2[k] = RF(R2[k]);
      if (!cb_slot_t<RF>(t1, slot, P1, A1, sz1, P2, Q2, sz2, &dd, pp, nn)) return -1;
      *dist = dd.v;
      for (int k = 0; k < 3; k++) { pos[k] = pp[k].v; nrm[k] = nn[k].v; }
    }
    make_frame(nrm, fr);
    if (!(-0.5f < nrm[1] && nrm[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
    if (cot) {
      float nbar[3];
      make_frame_vjp(fr, cot + 4, nbar);
      for (int j = 0; j < 12; j++) gb[j] = cb_dir_vjp(t1, slot, j, p1, a1, sz1, p2, R2, sz2, cot, nbar);
    }
    return br;
  }
"""

CB_DIR_VJP = """// (model_hooks.CB_DIR_VJP) pose direction j of a sphere-box / capsule-box contact's VJP, gb[j]
__device__ __noinline__ float cb_dir_vjp(int t1, int slot, int j, const float* p1, const float* a1, const float* sz1,
                                         const float* p2, const float* R2, const float* sz2, const float* cot,
                                         const float* nbar) {
  const int b = j / 6, a = j % 3;
  const bool rot = (j % 6) >= 3;
  if (t1 == 1 && b == 0 && rot) return 0.f;  // (a sphere's contact does not turn with it: gb stays 0)
  float e[3] = {0.f, 0.f, 0.f}, ta[3];
  e[a] = 1.f;
  cross(e, a1, ta);
  D1 P1[3], A1[3], P2[3], Q2[9], dd, pp[3], nn[3];
  for (int k = 0; k < 3; k++) {
    P1[k] = D1(p1[k], !rot && b == 0 && k == a ? 1.f : 0.f);
    A1[k] = D1(a1[k], rot && b == 0 ? ta[k] : 0.f);
    P2[k] = D1(p2[k], !rot && b == 1 && k == a ? 1.f : 0.f);
  }
  for (int c = 0; c < 3; c++) {  // ([e_a] R)_rc = (e_a x R_col c)_r
    float col[3] = {R2[c], R2[3 + c], R2[6 + c]}, x[3];
    cross(e, col, x);
    for (int r = 0; r < 3; r++) Q2[3 * r + c] = D1(R2[3 * r + c], rot && b == 1 ? x[r] : 0.f);
  }
  cb_slot_t<D1>(t1, slot, P1, A1, sz1, P2, Q2, sz2, &dd, pp, nn);
  float g = cot[0] * dd.d;
  for (int k = 0; k < 3; k++) g += cot[1 + k] * pp[k].d + nbar[k] * nn[k].d;
  return g;
}

"""

CB_COOP_DISPATCH = """#define CB_COOP_PAIR(pair, slot) ((c_gtype[PG1(pair)] == 1 || c_gtype[PG1(pair)] == 2) && c_gtype[PG2(pair)] == 3)
#define CB_COOP_WIDE(pair, slot) 1  // (12 directions: both geoms move)
// (model_hooks.CB_DIR_VJP) cb_dir_vjp of contact (pair, slot) from the geoms' world poses, rec: cot[0..3], nbar
__device__ __forceinline__ float cb_coop_dir(const float (*gx)[3], const float (*gq)[4], int pair, int slot, int j,
                                             const float* rec) {
  const int g1 = PG1(pair), g2 = PG2(pair), t1 = c_gtype[g1];
  float R2[9], a1[3] = {0.f, 0.f, 1.f};
  quat2mat(gq[g2], R2);
  if (t1 == 2) zaxis_rn(gq[g1], a1);
  return cb_dir_vjp(t1, slot, j, gx[g1], a1, c_gsize + 3 * g1, gx[g2], R2, c_gsize + 3 * g2, rec, rec + 4);
}

"""


def plane_mesh_hooks(part="engine"):
    """Plane-mesh contacts as MuJoCo Warp 3.17's plane_convex (g1_engine.mesh: the deepest vertex, the incident
    polygon pruned to 4 by hull4, the mesh-centre filter), in place of the older plane_mesh: CV_PLANE (the fused
    forward's collide_pair and the split forward's k_ccd) becomes plane_mesh_mj over RF (msh_plane), and (part
    "engine") the adjoint replays msh_plane_t over D1 in adj_mesh_geom once (the replay, which records the slot's
    vertex) and evaluates the slot's expression msh_slot_t per pose direction (g1_engine.mesh's SEARCH_ONCE: bitwise
    the same tangents; the vertex handed from the replay call to the VJP call, MESH_MVID). forest_engine.convex's RF
    instance of the older mesh collision (CV_ROUNDED) is left out: the mesh pairs are MuJoCo Warp 3.17's ports
    (convex_mj_hooks)."""
    from nablaengine.dual import RF_DEVICE
    from nablaengine.g1_engine import mesh
    head = ("// box-box (MuJoCo Warp's box_box, margin 0): separating-axis test, then face-vertex or edge-edge clipping\n"
            "#define BB_MAXPT 24\n")
    rf = "#ifndef DK_RF_DEVICE\n#define DK_RF_DEVICE\n" + RF_DEVICE + "#endif\n"
    wrap = """// plane-mesh (model_hooks.plane_mesh_hooks): CV_PLANE's signature, MuJoCo Warp 3.17's plane_convex
__device__ __noinline__ void plane_mesh_mj(const float* pp, const float* n, int g2i, const float* p2, const float* q2,
                                           CSink& o) {
  o.n = 0; o.unsupported = 0;
  float d4[4], p4[12], fr[9];
  int vid_[4];
  msh_plane(pp, n, c_gmesh[g2i], p2, q2, d4, p4, vid_);
  make_frame(n, fr);
  for (int i = 0; i < 4; i++)
    if (d4[i] < 0.f) add_contact(o, 0, i, d4[i], p4 + 3 * i, fr);
}
"""
    adj_old = "      cvd::plane_mesh(P1, nn, g2i, P2, Q2, o);\n"
    adj_new = ("      D1 d4[4], p4[12];  // (model_hooks.plane_mesh_hooks: the forward's msh_plane over D1)\n"
               "      if (j < 0) {\n"
               "        int vid[4] = {-1, -1, -1, -1};\n"
               "        msh_plane_t<D1>(P1, nn, c_gmesh[g2i], P2, Q2, d4, p4, vid);\n"
               "        *mvid = vid[slot & 3];\n"
               "        cvd::real f9[9];\n        cvd::make_frame(nn, f9);\n        o.n = 0; o.unsupported = 0;\n"
               "        for (int i = 0; i < 4; i++)\n          if (d4[i] < 0.f) cvd::add_contact(o, 0, i, d4[i], p4 + 3 * i, f9);\n"
               "      } else {  // (the slot only, from the replay's vertex)\n"
               "        msh_slot_t<D1>(P1, nn, c_mvadr[c_gmesh[g2i]], *mvid, P2, Q2, d4 + (slot & 3), p4 + 3 * (slot & 3));\n"
               "        cvd::real f9[9];\n        cvd::make_frame(nn, f9);\n        o.n = 0; o.unsupported = 0;\n"
               "        cvd::add_contact(o, 0, slot, d4[slot & 3], p4 + 3 * (slot & 3), f9);\n"
               "      }\n")
    rounded = "#ifndef CV_ROUNDED\n#define CV_ROUNDED 1  // (convex.rounded_source; $DK_CV_ROUNDED=0: the float forward)\n#endif\n"
    mdev = mesh.DEVICE
    for a, b in mesh.SEARCH_ONCE[:5] + mesh.SEARCH_ONCE[6:7]:
        if mdev.count(a) != 1:
            raise RuntimeError(f"plane_mesh_hooks: no {a!r} in g1_engine.mesh's text")
        mdev = mdev.replace(a, b)
    return ((head, "#if HAS_MESH\n" + DUAL_DEVICE + rf + mdev + wrap + "#endif\n" + head, 1),
            (rounded, "#define CV_ROUNDED 0  // (model_hooks.plane_mesh_hooks: convex.rounded_source's RF instance not needed)\n",
             1),
            ("#define CV_PLANE plane_mesh_r\n", "#define CV_PLANE plane_mesh_mj  // (model_hooks.plane_mesh_hooks)\n", 1),
            ("#define CV_PLANE plane_mesh\n", "#define CV_PLANE plane_mesh_mj  // (model_hooks.plane_mesh_hooks)\n", 1)
            ) + (((adj_old, adj_new, 1),) if part == "engine" else ())


def boxbox_fwd_hooks():
    """The forward's box-box primitive (BOX_BOX_CCD 0): MuJoCo Warp's box_box (nablaengine.boxbox, up to 8 contacts)
    over RF, so that the adjoint's D1 replay (ADJ_BOX_BOX_PRIM: bb_slot_t) takes its branches bit for bit; the
    templates go ahead of step.py's box_box, whose primitive half it replaces."""
    from nablaengine import boxbox
    from nablaengine.dual import RF_DEVICE
    head = ("// box-box (MuJoCo Warp's box_box, margin 0): separating-axis test, then face-vertex or edge-edge clipping\n"
            "#define BB_MAXPT 24\n")
    start = ("                        const float* R2, const float* s2) {\n  const float margin = 0.f;\n")
    body = """#if !BOX_BOX_CCD
  {  // MuJoCo Warp's box_box (nablaengine.boxbox; model_hooks.boxbox_fwd_hooks) over RF
    RF P1[3], Q1[9], P2[3], Q2[9], cd[BB_MAXVERT], cp[BB_MAXVERT][3], cn[3];
    for (int k = 0; k < 3; k++) { P1[k] = RF(p1[k]); P2[k] = RF(p2[k]); }
    for (int k = 0; k < 9; k++) { Q1[k] = RF(R1[k]); Q2[k] = RF(R2[k]); }
    const int nb = bb_box_box_t<RF>(P1, Q1, s1, P2, Q2, s2, 0.f, cd, cp, cn);
    if (nb > 0) {
      float nrm[3] = {cn[0].v, cn[1].v, cn[2].v}, fr[9];
      make_frame(nrm, fr);
      for (int i = 0; i < nb; i++)
        if (cd[i].v < 0.f) {
          float pos[3] = {cp[i][0].v, cp[i][1].v, cp[i][2].v};
          add_contact(s, pair, i, cd[i].v, pos, fr);
        }
    }
    return;
  }
#endif
"""
    rf = "#ifndef DK_RF_DEVICE\n#define DK_RF_DEVICE\n" + RF_DEVICE + "#endif\n"
    return ((head, rf + boxbox.DEVICE + head, 1), (start, start + body, 1))


def box_box_prim_source():
    """step.py's box_box with the separating-axis primitive (BOX_BOX_CCD 0, NATIVECCD disabled) as a template over the
    scalar type, generated from its text: box_box_prim_t<T>(p1, R1, s1, p2, R2, s2, slot, &dist, pos, nrm) -> 1 when
    contact slot (the forward's add_contact slot) exists, its distance, position and normal (the frame is
    make_frame(nrm)). With T = D1 the adjoint replays it once per pose direction (ADJ_BOX_BOX_PRIM)."""
    from nablaengine.forest_engine import step
    src = step.STEP_DEVICE
    head = ("template <class S>\n__device__ void box_box(S& s, int pair, const float* p1, const float* R1, "
            "const float* s1, const float* p2,\n                        const float* R2, const float* s2) {\n")
    i = src.index(head)
    j = src.index("  float fr[9];\n  make_frame(nrm, fr);\n#if BOX_BOX_CCD", i)
    body = src[i + len(head):j]
    import re
    for a, b in (("flag_unsupported(s, 2);", "(void)0;"), ("return;", "return 0;")):
        body = body.replace(a, b)
    body = re.sub(r"\bfloat\b", "T", body)
    body = re.sub(r"\b(mv3|mtv3|mm3|mmt3|dot3|safe_div|rotmore_of)\(", r"t_\1(", body)
    body = re.sub(r"\b(fabs|sqrt|fmin|fmax)f\(", r"t_\1(", body)
    if "atomic" in body or "add_contact" in body:
        raise RuntimeError("box_box_prim_source: unexpected side effects in the generated text")
    sig = ("// step.py's box_box (the separating-axis primitive) over a scalar type (model_hooks.box_box_prim_source)\n"
           "template <class T>\n"
           "__device__ __noinline__ int box_box_prim_t(const T* p1, const T* R1, const float* s1, const T* p2,\n"
           "                                           const T* R2, const float* s2, int slot, T* out_dist, T* out_pos,\n"
           "                                           T* out_nrm) {\n")
    tail = """  if (slot < 0 || slot >= n || !(depth[slot] < 0.f)) return 0;
  T pl[3] = {pts[slot][0], pts[slot][1], pts[slot][2] + hz}, pc[3];
  t_mv3(rw, pl, pc);
  for (int k = 0; k < 3; k++) { out_pos[k] = pc[k] + pw[k]; out_nrm[k] = nrm[k]; }
  *out_dist = depth[slot];
  return 1;
}
"""
    return sig + body + tail


ADJ_BOX_BOX_PRIM = """  if (t1 == 3 && t2 == 3) {  // box-box, the separating-axis primitive (nablaengine.boxbox): the contact's distance,
    // position and normal differentiated in forward mode (D1) along the 12 pose directions (positions; rotations
    // about the world axes, dR = [e] R), with the collision's branches fixed by the values
    float R1[9], R2[9], nrm[3];
    quat2mat(q1, R1);
    quat2mat(q2, R2);
    {
      D1 P1[3], Q1[9], P2[3], Q2[9], dd, pp[3], nn[3];
      for (int k = 0; k < 3; k++) { P1[k] = D1(p1[k]); P2[k] = D1(p2[k]); }
      for (int k = 0; k < 9; k++) { Q1[k] = D1(R1[k]); Q2[k] = D1(R2[k]); }
      if (!bb_slot_t<D1>(P1, Q1, sz1, P2, Q2, sz2, slot, &dd, pp, nn)) return -1;
      *dist = dd.v;
      for (int k = 0; k < 3; k++) { pos[k] = pp[k].v; nrm[k] = nn[k].v; }
    }
    make_frame(nrm, fr);
    if (!(-0.5f < nrm[1] && nrm[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
    if (cot) {
      float nbar[3];
      make_frame_vjp(fr, cot + 4, nbar);
      for (int j = 0; j < 12; j++) {
        D1 P1[3], Q1[9], P2[3], Q2[9], dd, pp[3], nn[3];
        const int b = j / 6, a = j % 3;
        const bool rot = (j % 6) >= 3;
        for (int k = 0; k < 3; k++) {
          P1[k] = D1(p1[k], !rot && b == 0 && k == a ? 1.f : 0.f);
          P2[k] = D1(p2[k], !rot && b == 1 && k == a ? 1.f : 0.f);
        }
        for (int r = 0; r < 3; r++)
          for (int c = 0; c < 3; c++) {  // ([e_a] R)_rc = (e_a x R_col c)_r
            const float* R = b ? R2 : R1;
            float e[3] = {0.f, 0.f, 0.f}, col[3] = {R[c], R[3 + c], R[6 + c]}, x[3];
            e[a] = 1.f;
            cross(e, col, x);
            Q1[3 * r + c] = D1(R1[3 * r + c], rot && b == 0 ? x[r] : 0.f);
            Q2[3 * r + c] = D1(R2[3 * r + c], rot && b == 1 ? x[r] : 0.f);
          }
        bb_slot_t<D1>(P1, Q1, sz1, P2, Q2, sz2, slot, &dd, pp, nn);
        gb[j] = cot[0] * dd.d;
        for (int k = 0; k < 3; k++) gb[j] += cot[1 + k] * pp[k].d + nbar[k] * nn[k].d;
      }
    }
    return br;
  }
"""


def convex_dual_source(ns="cvd", real="D1"):
    """forest_engine.convex.instantiate (kept here for the adjoint's callers)."""
    from nablaengine.forest_engine import convex
    return convex.instantiate(ns, real)


ADJ_MESH = """#if HAS_MESH
// a mesh pair's contact (model_hooks): plane_mesh or convex_pair replayed over duals (cvd) once per pose direction
// (positions; rotations about the world axes, dR = [e] R) with the branches fixed by the values: the contact slot's
// distance, position and frame (j = -1) and their exact tangents (gb, adj_contact_geom's layout). Plane-mesh: the
// replay pass records the slot's vertex in *mvid (adj_contact_one's lane variable, -1 before its replay call) and the
// pose directions' passes evaluate only the slot's expression from it (plane_mesh_hooks); the VJP's call (cot) after
// the replay skips its own replay pass (dist, pos, fr are not read there: zeros)
__device__ __noinline__ int adj_mesh_geom(int g1i, int t1, int g2i, int t2, int slot, const float* p1, const float* q1,
                                          const float* p2, const float* q2, float* dist, float* pos, float* fr,
                                          float* nu, const float* cot, float* gb, int* mvid) {
  float R1[9], R2[9];
  quat2mat(q1, R1);
  quat2mat(q2, R2);
  *nu = __int_as_float(0x7f800000);
  const bool once = t1 == 0 && cot && *mvid >= 0;  // (the replay call's vertex)
  if (once) {
    *dist = 0.f;
    for (int k = 0; k < 3; k++) pos[k] = 0.f;
    for (int k = 0; k < 9; k++) fr[k] = 0.f;
  }
  for (int j = once ? 0 : -1; j < (cot ? 12 : 0); j++) {
    cvd::real P1[3], Q1[9], P2[3], Q2[9];
    const int b = j < 0 ? -1 : j / 6, a = j < 0 ? 0 : j % 3;
    const bool rot = j >= 0 && (j % 6) >= 3;
    for (int k = 0; k < 3; k++) {
      P1[k] = D1(p1[k], !rot && b == 0 && k == a ? 1.f : 0.f);
      P2[k] = D1(p2[k], !rot && b == 1 && k == a ? 1.f : 0.f);
    }
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++) {
        const float* R = b == 1 ? R2 : R1;
        float e[3] = {0.f, 0.f, 0.f}, col[3] = {R[c], R[3 + c], R[6 + c]}, x[3];
        e[a] = 1.f;
        cross(e, col, x);
        Q1[3 * r + c] = D1(R1[3 * r + c], rot && b == 0 ? x[r] : 0.f);
        Q2[3 * r + c] = D1(R2[3 * r + c], rot && b == 1 ? x[r] : 0.f);
      }
    cvd::DSink o;
    if (t1 == 0) {
      float n0[3];
      zaxis_rn(q1, n0);  // the plane's normal: the forward's value (collide_pair), the z axis's tangent
      cvd::real nn[3] = {D1(n0[0], Q1[2].d), D1(n0[1], Q1[5].d), D1(n0[2], Q1[8].d)};
      cvd::plane_mesh(P1, nn, g2i, P2, Q2, o);
    } else {
      cvd::convex_pair(g1i, t1, P1, Q1, g2i, t2, P2, Q2, o);
    }
    int i = 0;
    while (i < o.n && o.slot[i] != slot) i++;
    if (i == o.n) return -1;
    if (j < 0) {
      *dist = o.dist[i].v;
      for (int k = 0; k < 3; k++) pos[k] = o.pos[i][k].v;
      for (int k = 0; k < 9; k++) fr[k] = o.fr[i][k].v;
    } else {
      gb[j] = cot[0] * o.dist[i].d;
      for (int k = 0; k < 3; k++) gb[j] += cot[1 + k] * o.pos[i][k].d;
      for (int k = 0; k < 9; k++) gb[j] += cot[4 + k] * o.fr[i][k].d;
    }
  }
  return (-0.5f < fr[1] && fr[1] < 0.5f) ? 0 : ADJ_BR_FRAME_Z;
}
#endif
"""


MESH_MVID = (
    ("__device__ __forceinline__ int adj_pair_geom(int pair, int slot, const float (*gx)[3], const float (*gq)[4],\n"
     "                                             float* dist, float* pos, float* fr, float* nu, const float* cot,\n"
     "                                             float* gb) {\n",
     "__device__ __forceinline__ int adj_pair_geom(int pair, int slot, const float (*gx)[3], const float (*gq)[4],\n"
     "                                             float* dist, float* pos, float* fr, float* nu, const float* cot,\n"
     "                                             float* gb, int* mvid) {  // mvid: model_hooks.ADJ_MESH\n"),
    ("  const int br = adj_pair_geom(pair, slot, s.gx, s.gq, &d, p, fr, &nu, nullptr, nullptr);\n",
     "  int mvid = -1;  // (model_hooks.ADJ_MESH: the replay's plane-mesh vertex for the VJP's call)\n"
     "  const int br = adj_pair_geom(pair, slot, s.gx, s.gq, &d, p, fr, &nu, nullptr, nullptr, &mvid);\n"),
    ("    adj_pair_geom(pair, slot, s.gx, s.gq, &d2, p2, fr2, &nu2, cot, gb);\n",
     "    adj_pair_geom(pair, slot, s.gx, s.gq, &d2, p2, fr2, &nu2, cot, gb, &mvid);\n"),
)


def has_floss(m):
    return bool((m.dof_frictionloss > 0).any())


def box_pairs(m):
    """(there is a pair with a box, every such pair is plane-box or box-box: the adjoint has them, there is a box-box
    pair)."""
    import mujoco
    box, plane = int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_PLANE)
    pairs = [(int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]])) for p in range(m.npair)]
    with_box = [t for t in pairs if box in t]
    return bool(with_box), all(plane in t or t == (box, box) for t in with_box), (box, box) in with_box


def integrator(m):
    """The forest engine's integrator for model m: "implicitfast", "euler" (MuJoCo's Euler with eulerdamp: the joint
    damping implicit) or "explicit" (Euler with eulerdamp disabled, or without joint damping: qvel' = qvel + h qacc,
    as MuJoCo skips the damped solve then)."""
    import mujoco
    I = mujoco.mjtIntegrator
    if m.opt.integrator == I.mjINT_IMPLICITFAST:
        return "implicitfast"
    assert m.opt.integrator == I.mjINT_EULER, (
        f"integrator {I(int(m.opt.integrator)).name}: the forest engine has implicitfast and Euler")
    damp = not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP) and bool((m.dof_damping > 0).any())
    return "euler" if damp else "explicit"


_L_IMPLICIT = """#if DK_CLAMP_IMPLICIT == 2
    s.L[TRI(i, i)] += h * ((off ? 0.f : c_kv[i] * fabsf(s.kvw[i])) + c_damp[i]);
#else
    s.L[TRI(i, i)] += h * ((off ? 0.f : c_kv[i]) + c_damp[i]);
#endif
"""
_Q_IMPLICIT = """#if DK_CLAMP_IMPLICIT == 2
    const float kv = ((s.kvoff[i >> 5] >> (i & 31)) & 1u) ? 0.f : c_kv[i] * fabsf(s.kvw[i]);
#else
    const float kv = ((s.kvoff[i >> 5] >> (i & 31)) & 1u) ? 0.f : c_kv[i];
#endif
    s.abar[i] -= h * (kv + c_damp[i]) * s.y[i];
"""
_NO_SOFT_KV = """#if DK_CLAMP_IMPLICIT == 2
#error "DK_CLAMP_IMPLICIT=2 (the implicit kv's fading weight) needs implicitfast"
#endif
"""
ENGINE_EULER = (
    (_L_IMPLICIT, _NO_SOFT_KV + "    (void)off;\n    s.L[TRI(i, i)] += h * c_damp[i];  // Euler, eulerdamp (model_hooks)\n", 1),
    (_Q_IMPLICIT, "    s.abar[i] -= h * c_damp[i] * s.y[i];  // Euler, eulerdamp: Q = M + h damping (model_hooks)\n", 1),
)
ENGINE_EXPLICIT = (
    (_L_IMPLICIT, _NO_SOFT_KV + "    (void)off;  // explicit Euler (model_hooks): qacc' = qacc after the solve\n", 1),
    ("  CHOL_M_SOLVE(s.L, s.L, s.tmp, lane, s.cbuf);\n",
     "  CHOL_M_SOLVE(s.L, s.L, s.tmp, lane, s.cbuf);\n"
     "  for (int i = lane; i < NV; i += 32) s.tmp[i] = s.qacc[i];  // explicit Euler (model_hooks)\n  __syncwarp();\n", 1),
    (_Q_IMPLICIT, "    // explicit Euler (model_hooks): Q = M, a* bar = a_adv bar\n", 1),
)

ENGINE_SPRING = (
    ("DKC float c_kq[NV], c_afrc[NV], c_ctrlr[NV * 2], c_damp[NV], c_gcomp[NB];\n",
     "DKC float c_kq[NV], c_afrc[NV], c_ctrlr[NV * 2], c_damp[NV], c_gcomp[NB];\n"
     "DKC float c_stiff[NV], c_sref[NV];  // joint stiffness, spring reference (model_hooks)\n", 1),
    ("      s.qsm[i] = act - c_damp[i] * s.qvel[i] - s.qsm[i];\n",
     "      s.qsm[i] = act - c_damp[i] * s.qvel[i] - c_stiff[i] * (s.qpos[c_dqadr[i]] - c_sref[i]) - s.qsm[i];"
     "  // spring (model_hooks)\n", 1),
    ("    s.Pv[i] += c_damp[i] * s.lam[i];\n    if (a < 0) continue;\n",
     "    s.Pv[i] += c_damp[i] * s.lam[i];\n    s.Pq[i] += c_stiff[i] * s.lam[i];  // joint spring (model_hooks)\n"
     "    if (a < 0) continue;\n", 1),
)


def has_spring(m):
    """A hinge / slide joint with stiffness (free joints: none, source.constants asserts)."""
    return bool((m.jnt_stiffness > 0).any())


def spring_constants(m):
    """(c_stiff, c_sref) per dof for ENGINE_SPRING."""
    import numpy as np
    k, ref = np.zeros(m.nv), np.zeros(m.nv)
    for j in range(m.njnt):
        if m.jnt_stiffness[j] > 0:
            k[m.jnt_dofadr[j]], ref[m.jnt_dofadr[j]] = m.jnt_stiffness[j], m.qpos_spring[m.jnt_qposadr[j]]
    return k, ref


def needed(m):
    """(the forest engine: always applied)"""
    return True


def _apply(src, hooks):
    for a, b, n in hooks:
        if src.count(a) != n:
            raise RuntimeError(f"model hook anchor found {src.count(a)} times, expected {n}: {a[:70]!r}")
        src = src.replace(a, b)
    return src


def apply(src, m, part="engine", damped=False):
    """src with the hooks model m needs (the forest engine: NCH and joint damping are in its text; the contact pairs
    beyond the mjlab primitives get their forward and adjoint hooks here: plane-box, box-box, sphere-box, capsule-box,
    meshes); part: "engine" (adjoint.cuda_source's text), "forward" (the forward alone) or "kinematics". damped: an
    Euler model without damping takes the damped update ("euler") instead of "explicit" (a build with per-world
    dof_damping: world_params.damped)."""
    hooks = ()
    if part == "kinematics":
        from nablaengine.forest_engine import nb64, nb128
        src = nb128.apply(nb64.apply(src, m, "kinematics"), m, "kinematics")
    if m.nv < 32 and part in ("engine", "forward"):
        hooks += ENGINE_SMALL_NV if part == "engine" else ENGINE_SMALL_NV[:1]
    if part == "engine":
        from nablaengine.forest_engine import source
        hooks += source.anc_hooks(m)
        hooks += source.gcoff_hooks(m)
        from nablaengine.forest_engine import step
        hooks += step.limit_hooks(m)
    if has_floss(m) and part == "engine":
        hooks += ENGINE_FLOSS + (('static_assert(NFLOSS == 0, "the forest adjoint: no frictionloss rows yet");',
                                  '// (model_hooks: the frictionloss rows in the adjoint)', 1),
                                 ('static_assert(NFLOSS == 0, "(the forest engine: the contact adjoint is not used, '
                                  'DK_ADJ_CONTACTS 0)");', '// (model_hooks: the contact rows after the frictionloss rows)',
                                  1))
        from nablaengine.forest_engine import step
        hooks += step.floss_hooks(m)
    if part == "engine" and integrator(m) != "implicitfast":
        hooks += ENGINE_EULER if integrator(m) == "euler" or damped else ENGINE_EXPLICIT
    if part == "engine" and has_spring(m):
        hooks += ENGINE_SPRING
    if box_pairs(m)[0] and part == "engine":
        hooks += ENGINE_PLANE_BOX
    mesh = has_mesh(m)
    if (box_prim(m) or mesh) and part == "engine":
        extra = DUAL_DEVICE + (convex_dual_source() + ADJ_MESH if mesh else "")
        hooks += (("__device__ __noinline__ int adj_contact_geom(", extra + "__device__ __noinline__ int adj_contact_geom(",
                   1),)
    if mesh and part in ("engine", "forward"):
        hooks += plane_mesh_hooks(part)
    capbox = has_sphere_box(m) or has_capsule_box(m)
    from nablaengine.g1_engine import model_hooks as g1_hooks
    if capbox and part in ("engine", "forward"):
        hooks += capbox_hooks(m)
    if capbox and part == "engine":
        hooks += (("  if (t1 != 2 || t2 != 2) return -1;\n", ADJ_CAPBOX + "  if (t1 != 2 || t2 != 2) return -1;\n", 1),
                  ("__device__ __noinline__ int adj_contact_geom(", CB_DIR_VJP + "__device__ __noinline__ int adj_contact_geom(",
                   1)) + g1_hooks.capbox_coop_hooks(CB_COOP_DISPATCH)
    if box_prim(m) and part in ("engine", "forward"):
        hooks += boxbox_fwd_hooks()
    if box_prim(m) and part == "engine":
        hooks += (("  float n[3], c1b[3] = {0.f, 0.f, 0.f}, c2b[3] = {0.f, 0.f, 0.f}, nb[3];\n",
                   ADJ_BOX_BOX_PRIM + "  float n[3], c1b[3] = {0.f, 0.f, 0.f}, c2b[3] = {0.f, 0.f, 0.f}, nb[3];\n", 1),)
    if mesh and part == "engine":
        hooks += (("  int g1 = PG1(pair), g2 = PG2(pair);\n  return adj_contact_geom(",
                   "  int g1 = PG1(pair), g2 = PG2(pair);\n#if HAS_MESH\n  if (c_gtype[g2] == 4 && c_gtype[g1] == 0)  // (model_hooks: "
                   "plane-mesh)\n"
                   "    return adj_mesh_geom(g1, c_gtype[g1], g2, c_gtype[g2], slot, gx[g1], gq[g1], gx[g2], gq[g2], dist, "
                   "pos, fr, nu, cot, gb, mvid);\n#endif\n  return adj_contact_geom(", 1),)
        hooks += tuple((a, b, 1) for a, b in MESH_MVID)
    from nablaengine.forest_engine import cylinder
    hooks += cylinder.hooks(m, part, "cyc_pair_f" if _ccd_float() else "cyc_pair_r")
    from nablaengine.forest_engine import ellipsoid
    hooks += ellipsoid.hooks(m, part, "cyc_pair_f" if _ccd_float() else "cyc_pair_r")
    if part in ("engine", "forward"):
        hooks += convex_mj_hooks(m, part)
    from nablaengine import ccd
    src = _apply(src, hooks)
    if part == "engine" and (any(ccd_pairs(m)) or cylinder.ccd_pairs(m) or ellipsoid.ccd_pairs(m)):
        from nablaengine.g1_engine import cylccd
        src = cylccd.apply_rec(src)
        if ccd_pairs(m)[1] and not sbb_boxbox(m):
            from nablaengine.forest_engine import ccd_coop
            src = ccd_coop.apply(src)
    if part == "forward":
        from nablaengine.g1_engine import exact_replay
        src = exact_replay.apply(src, m)
    return ccd.apply_gjk_mj4(src)


def ccd_pairs(m):
    """(the model has a mesh pair other than plane-mesh, it has a box-box pair collided by native CCD): the pairs
    MuJoCo Warp collides by its convex collision (GJK, EPA, multicontact), which the forest engine runs as the G1
    engine's port of MuJoCo Warp 3.17's (convex_mj_hooks)."""
    import mujoco
    G = mujoco.mjtGeom
    mesh, plane = int(G.mjGEOM_MESH), int(G.mjGEOM_PLANE)
    types = [(int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]])) for p in range(m.npair)]
    return (any(mesh in t and plane not in t for t in types),
            box_pairs(m)[2] and not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD))


def convex_mj_header(m):
    """The #defines of g1_engine.cylccd's code for the forest model m: MuJoCo Warp's EPA iterations (16 when every
    convex pair is box-box, else ccd_iterations), the multicontact buffers (npolygonmax, nmeshdegmax: at least a box's
    4 and 3), the mesh tables (forest_engine.convex's, the same layout), multiccd (box-box always, as MuJoCo Warp)."""
    import mujoco
    from nablaengine.forest_engine import convex, cylinder, step
    meshes, boxbox = ccd_pairs(m)
    mt = convex.mesh_tables(m, step.collision_geoms(m))
    it = int(m.opt.ccd_iterations)
    multi = int(not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD))
    cyl = cylinder.ccd_with_cylinder(m)
    from nablaengine.forest_engine import ellipsoid
    ell = ellipsoid.ccd_pairs(m)
    return (f"#define CY_GJK_ITERATIONS {it}\n#define CY_EPA_ITERATIONS {it if meshes or cyl or ell else 16}\n"
            f"#define CY_CCD_TOLERANCE {float(m.opt.ccd_tolerance):.9g}f\n"
            f"#define CY_NPOLY {max(16 if cyl else 4, mt['poly_max'] if mt else 0)}\n#define CY_NDEG {max(3, mt['deg_max'] if mt else 0)}\n"
            f"#define CY_HAS_MESH {int(mt is not None)}\n#define CY_MULTICCD {multi}\n")


_CYT = "#define CYT(t) ((t) == 4 ? 6 : (t))  // (model_hooks.convex_mj_hooks: the forest's mesh type 4 as cylccd's 6)\n"

CONVEX_MJ_WRAP = r"""// convex pairs (model_hooks.convex_mj_hooks): CV_PAIR's signature, g1_engine.cylccd's MuJoCo Warp 3.17 CCD over RF
__device__ __noinline__ void convex_pair_mj(int g1i, int t1, const float* p1, const float* q1, int g2i, int t2,
                                            const float* p2, const float* q2, CSink& o) {
  cyc_pair_r(CYT(t1), g1i, p1, q1, c_gsize + 3 * g1i, CYT(t2), g2i, p2, q2, c_gsize + 3 * g2i, o);
}
"""

ADJ_CCD = """  if ((t2 == 4 && t1 != 0) || (BOX_BOX_CCD && t1 == 3 && t2 == 3)) {  // native CCD pairs (model_hooks.convex_mj_hooks:
    // g1_engine.cylccd): the forward's contact `slot` replayed, its VJP by D1 along the 12 pose directions
    if (cyc_adj(CYT(t1), p1, q1, sz1, CYT(t2), p2, q2, sz2, slot, cot, gb, dist, pos, fr, crec) < 0) return -1;
    if (!(-0.5f < fr[1] && fr[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
    return br;
  }
"""


def _ccd_float():
    import os
    return os.environ.get("DK_CCD_FLOAT", "0") not in ("", "0")


def convex_mj_hooks(m, part):
    """The forest engine's native CCD pairs (ccd_pairs: a mesh with a sphere, capsule, box or mesh; box-box unless
    NATIVECCD is disabled) through the G1 engine's port of MuJoCo Warp 3.17's convex collision (g1_engine.cylccd: gjk
    phase, EPA, multicontact; the forward over RF, the adjoint's replay over D1, exact tangents), in place of
    forest_engine.convex (CV_PAIR) and step.py's native-CCD box_box: the device code ahead of box_pair, CV_PAIR,
    box_pair's box-box, and (part "engine") the contact adjoint's branch ADJ_CCD with cyc_adj."""
    import re
    from nablaengine.g1_engine import cylccd
    from nablaengine.forest_engine import cylinder
    meshes, boxbox = ccd_pairs(m)
    cyl = cylinder.ccd_pairs(m)
    from nablaengine.forest_engine import ellipsoid
    ell = ellipsoid.ccd_pairs(m)
    if not (meshes or boxbox or cyl or ell):
        return ()
    dev = cylccd.device()
    if ellipsoid.has_ellipsoid(m):
        dev = ellipsoid.support_text(dev)
    dev = re.sub(r"#ifndef CSINK_DEFINED\n.*?#endif\n", "", dev, count=1, flags=re.S)
    old = "  if (!((f1 && f2) || (g1.type == 2 && f2) || (f1 && g2.type == 2))) midx = -1;  // (_supports_multiccd)"
    if dev.count(old) != 3:
        raise RuntimeError("convex_mj_hooks: no _supports_multiccd test in cylccd's text")
    dev = dev.replace(old, "  if (!((g1.type == 3 && g2.type == 3) || (CY_MULTICCD && ((f1 && f2) || (g1.type == 2 && f2) || "
                           "(f1 && g2.type == 2))))) midx = -1;\n  // (_supports_multiccd; model_hooks.convex_mj_hooks: "
                           "box-box always)")
    wrap, adj_fn = CONVEX_MJ_WRAP, cylccd.ADJ_FN
    if _ccd_float():
        fdev = cylccd.instantiate("cycf", "float")
        i = cylccd.GLOBAL.index("// the forward's contacts of the pair")
        dev = "#define CY_REC 0\n" + dev
        fwd = cylccd.GLOBAL[i:].replace("cyc_pair_r", "cyc_pair_f").replace("cyc_pair_x", "cyc_pair_xf")
        fwd = fwd.replace("RF P1[3]", "float P1[3]")
        fwd = fwd.replace("cyc_geom<RF>", "cyc_geom<float>").replace("cycr::", "cycf::").replace(", RF>(", ", float>(")
        fwd = fwd.replace("d.dist[i].v", "d.dist[i]").replace("d.pos[i][k].v", "d.pos[i][k]").replace("d.fr[i][k].v", "d.fr[i][k]")
        dev += fdev + fwd
        wrap, adj_fn = wrap.replace("cyc_pair_r(", "cyc_pair_f("), adj_fn.replace("cyc_pair_r(", "cyc_pair_f(")
    dev = convex_mj_header(m) + _CYT + dev + wrap
    sbb = boxbox and sbb_boxbox(m)
    if sbb:
        dev += sbb_device(m)
    anchor = "#if HAS_BOX\n// capsule-box (t1 2) or box-box (t1 3) into o:"
    hooks = ((anchor, dev + anchor, 1),)
    if meshes:
        hooks += (("#define CV_PAIR convex_pair_r\n", "#define CV_PAIR convex_pair_mj  // (model_hooks.convex_mj_hooks)\n", 1),
                  ("#define CV_PAIR convex_pair\n", "#define CV_PAIR convex_pair_mj  // (model_hooks.convex_mj_hooks)\n", 1))
    if sbb:
        old = "    box_box(o, 0, p1, R1, c_gsize + 3 * g1, p2, R2, c_gsize + 3 * g2);\n"
        hooks += ((old, "#if BOX_BOX_CCD  // (model_hooks.sbb_device: nablaengine.ccd's box-box CCD, the tree engine's)\n"
                        "    if (!sbb::boxes_separated(p1, R1, c_gsize + 3 * g1, p2, R2, c_gsize + 3 * g2))\n"
                        "      sbb::ccd_box_box(p1, R1, c_gsize + 3 * g1, p2, R2, c_gsize + 3 * g2, o);\n#else\n"
                   + old + "#endif\n", 1),)
    elif boxbox:
        old = "    box_box(o, 0, p1, R1, c_gsize + 3 * g1, p2, R2, c_gsize + 3 * g2);\n"
        hooks += ((old, "#if BOX_BOX_CCD  // (model_hooks.convex_mj_hooks: MuJoCo Warp 3.17's CCD)\n"
                        f"    {'cyc_pair_f' if _ccd_float() else 'cyc_pair_r'}(3, g1, p1, q1, c_gsize + 3 * g1, 3, g2, p2, q2, "
                        "c_gsize + 3 * g2, o);\n#else\n"
                   + old + "#endif\n", 1),)
    if part == "engine":
        head = "__device__ __noinline__ int adj_contact_geom("
        adj = ADJ_CCD
        if cyl:
            old = "(BOX_BOX_CCD && t1 == 3 && t2 == 3)) {"
            adj = adj.replace(old, "(BOX_BOX_CCD && t1 == 3 && t2 == 3)\n      || " + cylinder.CCD_COND + ") {")
        if ell:
            old = "(BOX_BOX_CCD && t1 == 3 && t2 == 3)"
            adj = adj.replace(old, old + "\n      || " + ellipsoid.CCD_COND, 1)
        if sbb:
            adj_fn = "namespace sbb {\n" + _sbb_adj_device() + "}  // namespace sbb\n" + adj_fn
            adj = ADJ_SBB + adj
        hooks += ((head, adj_fn + head, 1),
                  ("  if (t1 != 2 || t2 != 2) return -1;\n", adj + "  if (t1 != 2 || t2 != 2) return -1;\n", 1))
        if meshes:
            hooks += (("      cvd::convex_pair(g1i, t1, P1, Q1, g2i, t2, P2, Q2, o);\n",
                       "      return -1;  // (model_hooks.convex_mj_hooks: the convex pairs in cyc_adj)\n", 1),)
    return hooks


def sbb_boxbox(m):
    """A G1-class model (engines.is_g1_class: one free tree, e.g. the Booster T1's foot-foot pair) collides its box-box
    pairs under native CCD with the tree engine's code, nablaengine.ccd's ccd_box_box, and differentiates them with its
    exact VJP ccd_box_box_vjp (ADJ_SBB), as g1_engine.model_hooks does, in place of g1_engine.cylccd's CCD and the
    replay of all of it over D1 along 12 pose directions per contact (cyc_adj): that replay took a world with a
    Booster T1 foot-foot contact 10x the adjoint time of the others (V100: 38.3 against 3.8 ms for 4096 such worlds),
    and the slowest worlds set the kernel's time. Multi-tree models (objects) keep cyc_adj (exact tangents also for the EPA
    features ccd_box_box_vjp takes as points of their boxes)."""
    from nablaengine import engines
    return engines.is_g1_class(m)


def sbb_device(m):
    """nablaengine.ccd's box-box CCD in namespace sbb (its CSink struct is the forest's, step.py), its #defines
    (ccd.header) and the separating-axis prefilter boxes_separated (g1_engine.model_hooks.box_box_hook)."""
    import re as _re
    from nablaengine import ccd
    from nablaengine.g1_engine import model_hooks as g1_hooks
    dev, n = _re.subn(r"#define SINK_MAX 8\nstruct CSink \{.*?\};\n", "", ccd.DEVICE, count=1, flags=_re.S)
    if n != 1:
        raise RuntimeError("sbb_device: no CSink struct in nablaengine.ccd's text")
    wrap = g1_hooks.box_box_hook()[1][len(ccd.DEVICE):]
    sep = wrap[:wrap.index("template <class S>\n__device__ void box_box(")]
    assert "__device__ bool boxes_separated(" in sep
    hdr = "".join(f"#ifndef {l.split()[1]}\n{l}\n#endif\n" for l in ccd.header(m).splitlines())
    return ("// a G1-class model's box-box pairs (forest_engine.model_hooks.sbb_device: nablaengine.ccd's CCD, the tree\n"
            "// engine's)\n" + hdr + "namespace sbb {\n" + dev + sep + "}  // namespace sbb\n")


def _sbb_adj_device():
    from nablaengine import ccd
    return ccd.ADJ_DEVICE


ADJ_SBB = """  if (BOX_BOX_CCD && t1 == 3 && t2 == 3) {  // box-box of a G1-class model (model_hooks.sbb_boxbox): the forward's
    // CCD contact `slot` and its exact VJP (nablaengine.ccd ccd_box_box_vjp), as the tree engine's ADJ_BOX_BOX
    float R1[9], R2[9];
    quat2mat(q1, R1);
    quat2mat(q2, R2);
    if (!sbb::ccd_box_box_vjp(p1, R1, sz1, p2, R2, sz2, slot, cot, gb, dist, pos, fr)) return -1;
    if (!(-0.5f < fr[1] && fr[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
    return br;
  }
"""


def has_mesh(m):
    """A mesh collision geom in a pair: its contacts get the forward-mode VJP (adj_mesh_geom)."""
    import mujoco
    mesh = int(mujoco.mjtGeom.mjGEOM_MESH)
    return any(int(m.geom_type[g]) == mesh for p in range(m.npair) for g in (m.pair_geom1[p], m.pair_geom2[p]))


def box_prim(m):
    """A box-box pair collided by the separating-axis primitive (NATIVECCD disabled): its contacts get the adjoint."""
    import mujoco
    return box_pairs(m)[2] and bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD)

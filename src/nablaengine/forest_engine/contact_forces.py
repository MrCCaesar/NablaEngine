"""The tree engine's contact-force options (g1_engine.contact_forces) on the forest engine: contact_forces="substeps"
(a table per substep), foot_wrench (per substep the summed force and moment on given bodies, any bodies of any tree),
contact_frame (the frame's first tangent t1). The Spec, the output layout (width), decode and pair_geoms are the tree
engine's (g1_engine.contact_forces), so init() / contact_table() are the same on both engines."""
from nablaengine.g1_engine import contact_forces as _g1

Spec, width, decode, pair_geoms, as_spec = _g1.Spec, _g1.width, _g1.decode, _g1.pair_geoms, _g1.as_spec

_CON_WORDS = ("#define CON_WORDS 12  // the forward's contact table per contact: pair, dist, pos 3, normal 3, force 3 "
              "(world), normal force\n")
_STRUCT_SIM = "struct Sim {\n"
_STRUCT_SIMN = "struct SimNT {  // substep_post: Newton, implicitfast; capacity NC contacts, NR rows. Members whose lifetimes do not\n"
_HAND_BIG = "  SimNfull big;  // k_post's state of a world whose contacts or rows exceed SimN's capacity\n"
_SOLVER = "  if (niter_out && lane == 0) *niter_out = it;\n"
_FORCES = _g1._FORCES

_KSUB_LOOP = """    if (lane == 0) s.oforce = k == nsub - 1 ? s.con_fo : nullptr;  // the contact forces of the last substep
    __syncwarp();
    substep(s, ctrl + (size_t)w * NU, iterations, lane, &nit,
            tape ? tape + ((size_t)w * nsub + k) * REC_WORDS : nullptr);
  }
"""
_KSUB_LOOP_NEW = """    if (lane == 0) {  // (contact_forces: the table carries the forces, oforce stays null)
      s.oforce = nullptr;
      s.ocf = ocon && cf_on(k, nsub) ? cf_tab(ocon, w, k, nsub) : nullptr;
    }
    __syncwarp();
    substep(s, ctrl + (size_t)w * NU, iterations, lane, &nit,
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
_KSUB_WRITE = """  for (int c = lane; c < MAXCON; c += 32) {  // contacts of the last substep: pair, dist, pos, normal, force, normal force
    float* o = ocon + ((size_t)w * MAXCON + c) * CON_WORDS;
    bool v = c < min(s.ncon, MAXCON);
    o[0] = v ? (float)s.con_pair[c] : -1.f; o[1] = v ? s.con_dist[c] : 0.f;
    for (int k = 0; k < 3; k++) { o[2 + k] = v ? s.con_pos[c][k] : 0.f; o[5 + k] = v ? s.con_frame[c][k] : 0.f; }
    for (int k = 0; k < 3; k++)
      o[8 + k] = v ? s.con_fo[c][0] * s.con_frame[c][k] + s.con_fo[c][1] * s.con_frame[c][3 + k]
                         + s.con_fo[c][2] * s.con_frame[c][6 + k] : 0.f;
    o[11] = v ? s.con_fo[c][0] : 0.f;
  }
"""
_KSUB_WRITE_NEW = """  if (ocon) cf_write(cf_tab(ocon, w, nsub - 1, nsub), s, min(s.ncon, MAXCON), lane);  // (contact_forces: forces in)
"""
_PRE_WRITE = """  if (k == nsub - 1)  // contacts of the last substep's collision: pair, dist, pos, normal
    for (int c = lane; c < MAXCON; c += 32) {
      float* o = ocon + ((size_t)w * MAXCON + c) * CON_WORDS;  // (words 8 .. 11: the forces, k_post)
      bool v = c < ncon;
      o[0] = v ? (float)s.con_pair[c] : -1.f; o[1] = v ? s.con_dist[c] : 0.f;
      for (int t = 0; t < 3; t++) { o[2 + t] = v ? s.con_pos[c][t] : 0.f; o[5 + t] = v ? s.con_frame[c][t] : 0.f; }
      if (!v) for (int t = 8; t < 12; t++) o[t] = 0.f;  // (past the world's contacts: k_post writes none)
    }
"""
_PRE_WRITE_NEW = """  if (lane == 0) g.ocf = ocon && cf_on(k, nsub) ? cf_tab(ocon, w, k, nsub) : nullptr;  // (contact_forces)
#if CF_NFOOT
  if (lane < 3 * CF_NFOOT) g.cf_x[lane] = s.X[cf_foot(lane / 3)][lane % 3];  // (the feet's body origins)
#endif
  if (ocon && cf_wr(k, nsub)) cf_write(cf_tab(ocon, w, k, nsub), s, ncon, lane);  // (contact_forces: forces in k_post)
"""
_POST_PTR = "    s.oforce = k == nsub - 1 ? fo : nullptr;  // the contact forces of the last substep (Hand::con_force)\n"
_POST_PTR_NEW = ("    s.oforce = nullptr;  // (contact_forces: the table carries the forces)\n"
                 "    s.ocf = g.ocf;\n")
_POST_WRITE = """  if (k == nsub - 1 && ocon)  // the contact table's forces (k_pre wrote the rest): world force, normal force
    for (int c = lane; c < ncon; c += 32) {
      float* o = ocon + ((size_t)w * MAXCON + c) * CON_WORDS;
      for (int j = 0; j < 3; j++)
        o[8 + j] = fo[c][0] * g.con_frame[c][j] + fo[c][1] * g.con_frame[c][3 + j] + fo[c][2] * g.con_frame[c][6 + j];
      o[11] = fo[c][0];
    }
"""
_POST_WRITE_NEW = """  __syncwarp();
  if (g.ocf)  // (contact_forces) the forces into world coordinates, the feet's wrenches
#if CF_NFOOT
    cf_finish(g.ocf, ncon, g.con_pair, g.con_pos, g.con_frame, g.cf_x, k, nsub, lane);
#else
    cf_finish(g.ocf, ncon, g.con_pair, g.con_pos, g.con_frame, nullptr, k, nsub, lane);
#endif
"""


def check(m, spec):
    """ValueError where spec does not fit the forest build of m (the terrain flag: jax_ffi checks it against the build;
    its word is cf_write's, from s.con_elem of the forest terrain build, forest_engine.terrain)."""
    for b in spec.feet:
        if not 0 < int(b) < m.nbody:
            raise ValueError(f"foot_wrench: body id {b} out of range")


def engine(src, spec):
    """The forest engine text (adjoint.cuda_source's unit) with the contact forces of spec: the defines, the pointers,
    the solver tail."""
    _sub = _g1._sub
    defs = [f"#define CON_WORDS {spec.con_words}  // the contact table per contact, with the forces "
            "(forest_engine.contact_forces)",
            f"#define CF_SUB {int(spec.mode == 'substeps')}  // a table per substep",
            f"#define CF_NFOOT {len(spec.feet)}  // bodies with wrenches",
            f"#define CF_FRAME {int(spec.frame)}  // the tangent t1's words",
            f"#define CF_TERRAIN {int(spec.terrain)}  // the terrain element's word (forest_engine.terrain)"]
    lay = spec.layout()
    if spec.frame:
        defs.append(f"#define CF_T1 {lay['tangent1'][0]}")
    if spec.terrain:
        defs.append(f"#define CF_ELEM {lay['element'][0]}")
    if spec.feet:
        defs.append("#define CF_FEET {" + ", ".join(str(b) for b in spec.feet) + "}  // the bodies' ids")
    if spec.foot_split:
        defs += ["#define CF_FSPLIT 1  // the foot / toe-weight words (forest_engine.foot_offset)",
                 f"#define CF_FS {lay['foot_side'][0]}"]
    if spec.foot_sand:
        defs.append("#define CF_FSAND 1  // the foot contact's sand fraction word (forest_engine.foot_offset)")
    src = _sub(src, _CON_WORDS, "\n".join(defs) + "\n", 1, "forest CON_WORDS")
    src = _sub(src, _STRUCT_SIM, _STRUCT_SIM + "  float* ocf;  // contact_forces: the world's contact table of the "
               "substep, else null\n", 1, "forest struct Sim")
    src = _sub(src, _STRUCT_SIMN, _STRUCT_SIMN + "  float* ocf;  // contact_forces (Hand::ocf)\n", 1,
               "forest struct SimNT")
    hand = "  float* ocf;  // contact_forces: k_pre's table pointer for k_post (null: not wanted)\n"
    if spec.feet:
        hand += "  float cf_x[3 * CF_NFOOT];  // contact_forces: the bodies' frame origins of the substep (k_pre)\n"
    src = _sub(src, _HAND_BIG, hand + _HAND_BIG, 1, "forest Hand")
    src = _sub(src, _SOLVER, _SOLVER + _FORCES, 1, "forest solver tail")
    return src


def host(src, spec):
    """The forest host kernels' text (host.KERNEL) with the contact table of spec."""
    _sub = _g1._sub
    src = _sub(src, _g1._HOST_HEAD, _g1._HOST_FUNCS + _g1._HOST_HEAD, 1, "forest host functions")
    src = _sub(src, _KSUB_LOOP, _KSUB_LOOP_NEW, 1, "forest k_substep's loop")
    src = _sub(src, _KSUB_WRITE, _KSUB_WRITE_NEW, 1, "forest k_substep's table")
    src = _sub(src, _PRE_WRITE, _PRE_WRITE_NEW, 1, "forest pre_world's table")
    src = _sub(src, _POST_PTR, _POST_PTR_NEW, 1, "forest post_world_on's pointer")
    src = _sub(src, _POST_WRITE, _POST_WRITE_NEW, 1, "forest post_world_on's forces")
    return src

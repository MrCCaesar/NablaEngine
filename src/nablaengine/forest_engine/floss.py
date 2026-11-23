"""Per-world joint dry friction (PhysicsParams fields dof_frictionloss, frictionloss_solref, frictionloss_solimp) in the
forest engine's text, applied by world_params.apply_params on a parameter build that has one of them (other builds:
the text unchanged)."""

FIELDS = ("dof_frictionloss", "frictionloss_solref", "frictionloss_solimp")
ROW_FLOSS_READ = "c_floss[s.efc_a[r]]"
STATE_PTR = "  const float* dampw;\n"
STATE_PTR_NEW = STATE_PTR + "  const float* flossw;  // the world's frictionloss bounds (forest_engine.floss)\n"
NT_PTR = "  const float* dampw;  // in Hand (world_params)\n"
NT_PTR_NEW = NT_PTR + "  const float* flossw;  // in Hand (forest_engine.floss)\n"
HAND = "  float dampw[NV];  // the world's damping (world_params)\n"
HAND_NEW = HAND + "  float flossw[NV];  // the world's frictionloss bounds (forest_engine.floss)\n"
LOAD = "  if (lane == 0) s.dampw = s.pw + (PW_DAMP >= 0 ? PW_DAMP : 0);\n"
LOAD_NEW = LOAD + "  if (lane == 0) s.flossw = s.pw + (PW_FLOSS >= 0 ? PW_FLOSS : 0);  // (forest_engine.floss)\n"
SPLIT_COPY = "g.dampw[i] = s.dampw[i]; }"
SPLIT_COPY_NEW = "g.dampw[i] = s.dampw[i]; g.flossw[i] = s.flossw[i]; }"
SPLIT_PTR = "s.dampw = g.dampw;"
SPLIT_PTR_NEW = "s.dampw = g.dampw; s.flossw = g.flossw;"
FWD_DECL = "      float iw, pos = 0.f, mu = 0.f, fsi[5];\n"
FWD_DECL_NEW = "      float iw, pos = 0.f, mu = 0.f, fsi[5], fli[5];\n"
FWD_KB = "      float k, bb, imp;\n      kbimp(sref, simp, pos, h, &k, &bb, &imp);\n"
FWD_KB_NEW = """      if (!con && s.efc_kind[j] == 0) {  // a frictionloss row: the world's impedance (forest_engine.floss)
        if (PW_FSREF >= 0) sref = s.pw + (PW_FSREF >= 0 ? PW_FSREF : 0);
        if (PW_FSIMP >= 0) {
          for (int q = 0; q < 4; q++) fli[q] = s.pw[(PW_FSIMP >= 0 ? PW_FSIMP : 0) + q];
          fli[4] = simp[4];
          simp = fli;
        }
      }
""" + FWD_KB
LOOP_OLD = """    const unsigned bal = __ballot_sync(0xffffffffu, on);
    if (on) {
      const int r = nf + __popc(bal & below);
"""
LOOP_NEW = """    const unsigned bal = __ballot_sync(0xffffffffu, on);
    float fg[3] = {0.f, 0.f, 0.f};  // the lane's d/d (tc, dmin, dmax) (forest_engine.floss)
    if (on) {
      const int r = nf + __popc(bal & below);
"""
SUM_OLD = "    nf += __popc(bal);\n"
SUM_NEW = """    if (PW_FSREF >= 0 || PW_FSIMP >= 0) {  // (forest_engine.floss: the warp's sum, lane 0 adds it)
      for (int o = 16; o > 0; o >>= 1)
        for (int q = 0; q < 3; q++) fg[q] += __shfl_xor_sync(0xffffffffu, fg[q], o);
      if (lane == 0) {
        if (PW_FSREF >= 0) s.gpw[PW_FSREF >= 0 ? PW_FSREF : 0] += fg[0];
        if (PW_FSIMP >= 0) { s.gpw[PW_FSIMP >= 0 ? PW_FSIMP : 0] += fg[1]; s.gpw[(PW_FSIMP >= 0 ? PW_FSIMP : 0) + 1] += fg[2]; }
      }
    }
    nf += __popc(bal);
"""
ADJ_OLD = """      if ((s.zone[r >> 5] >> (r & 31)) & 1u) {
        float ri[6];
        adj_row_impedance(@SREF@, @SIMP@, 0.f, c_dinvw[i], ri);
        s.Pv[i] += ri[4] * ri[1] * s.lam[i];
      }
"""
ADJ_NEW = """      const float* sref = @SREF@;  // (forest_engine.floss: the world's parameters)
      const float* simp = @SIMP@;
      float fli[5];
      if (PW_FSREF >= 0) sref = s.pw + (PW_FSREF >= 0 ? PW_FSREF : 0);
      if (PW_FSIMP >= 0) {
        for (int q = 0; q < 4; q++) fli[q] = s.pw[(PW_FSIMP >= 0 ? PW_FSIMP : 0) + q];
        fli[4] = simp[4];
        simp = fli;
      }
      float ri[6];
      adj_row_impedance(sref, simp, 0.f, c_dinvw[i], ri);
      const float b = ri[1], imp = ri[2], D = ri[4], x = s.a[i] + b * s.v[i], lam = s.lam[i];
      if ((s.zone[r >> 5] >> (r & 31)) & 1u) {
        s.Pv[i] += D * b * lam;
        if (PW_FSREF >= 0 && sref[0] > RMUL(2.f, c_misc[0]))  // d/d tc (the time constant above its 2 h floor)
          fg[0] = D * b * s.v[i] / sref[0] * lam;
        if (PW_FSIMP >= 0) {  // d/d dmin, d/d dmax (width, midpoint: 0)
          const bool mean = simp[0] == simp[1] || simp[2] <= 1e-15f;  // (kbimp: imp the mean of dmin and dmax)
          const float dDi = D / (imp * (1.f - imp));
          const float in0 = simp[0] > 1e-4f && simp[0] < 0.9999f ? 1.f : 0.f;
          const float in1 = simp[1] > 1e-4f && simp[1] < 0.9999f ? 1.f : 0.f;
          fg[1] = in0 * (-x * dDi * (mean ? 0.5f : 1.f)) * lam;
          fg[2] = in1 * (-x * dDi * (mean ? 0.5f : 0.f) + D * b * s.v[i] / fminf(fmaxf(simp[1], 1e-4f), 0.9999f)) * lam;
        }
      } else if (PW_FLOSS >= 0) {  // a linear piece: the force -sign(x) F
        s.gpw[(PW_FLOSS >= 0 ? PW_FLOSS : 0) + i] += (x > 0.f ? -1.f : 1.f) * lam;
      }
"""


def _count(src, a, n, what):
    k = src.count(a)
    if (k != n) if n is not None else k < 1:
        raise RuntimeError(f"forest_engine.floss: {what} found {k} times, expected {n if n is not None else '>= 1'}")


def active(params):
    return bool(params) and any(f in params.fields for f in FIELDS)


def apply(src, m, params, part):
    """src with the per-world frictionloss edits (params has one of FIELDS; part "engine" or "host": world_params'
    parts, after its own edits). The kinematics are unchanged."""
    if not active(params) or part == "kinematics":
        return src
    bound = "dof_frictionloss" in params.fields
    if part == "host":
        if not bound:
            return src
        _count(src, LOAD, None, "the parameters' load")
        _count(src, SPLIT_COPY, 1, "k_pre's Hand copy")
        _count(src, SPLIT_PTR, 1, "k_post's Hand pointers")
        return src.replace(LOAD, LOAD_NEW).replace(SPLIT_COPY, SPLIT_COPY_NEW).replace(SPLIT_PTR, SPLIT_PTR_NEW)
    from nablaengine.forest_engine import step
    lay = params.layout(m.nv)
    off = lambda f: lay[f][0] if f in lay else -1
    sets = step.floss_sets(m)
    sr, si = ("c_dsolrefv + 2 * i", "c_dsolimpv + 5 * i") if sets else ("c_dsolref", "c_dsolimp")
    adj_old = ADJ_OLD.replace("@SREF@", sr).replace("@SIMP@", si)
    if sets:
        adj_old = adj_old.replace("c_dinvw[i], ri);\n", "c_dinvw[i], ri);  // (step.floss_hooks)\n")
    _count(src, adj_old, 1, "the frictionloss rows' adjoint (model_hooks.ENGINE_FLOSS)")
    src = src.replace(adj_old, ADJ_NEW.replace("@SREF@", sr).replace("@SIMP@", si))
    _count(src, LOOP_OLD, 1, "the frictionloss rows' adjoint loop")
    _count(src, SUM_OLD, 1, "the frictionloss rows' adjoint count")
    src = src.replace(LOOP_OLD, LOOP_NEW).replace(SUM_OLD, SUM_NEW)
    if bound:
        _count(src, STATE_PTR, 3, "the states' dampw")
        _count(src, NT_PTR, 1, "SimNT's dampw")
        _count(src, HAND, 1, "Hand's dampw")
        _count(src, LOAD, 1, "the adjoint's parameter load")
        _count(src, ROW_FLOSS_READ, None, "row_floss's bound")
        src = (src.replace(STATE_PTR, STATE_PTR_NEW).replace(NT_PTR, NT_PTR_NEW).replace(HAND, HAND_NEW)
               .replace(LOAD, LOAD_NEW).replace(ROW_FLOSS_READ, "s.flossw[s.efc_a[r]]"))
    if "frictionloss_solref" in params.fields or "frictionloss_solimp" in params.fields:
        _count(src, FWD_DECL, 1, "substep_pre's row impedance declarations")
        _count(src, FWD_KB, 1, "substep_pre's kbimp call")
        src = src.replace(FWD_DECL, FWD_DECL_NEW).replace(FWD_KB, FWD_KB_NEW)
    return (f"#define PW_FLOSS {off('dof_frictionloss')}  // the world's dof_frictionloss (forest_engine.floss)\n"
            f"#define PW_FSREF {off('frictionloss_solref')}\n#define PW_FSIMP {off('frictionloss_solimp')}\n" + src)


def check(m, params):
    """ValueError where the fields do not apply to model m: they need dofs with frictionloss rows (model
    dof_frictionloss > 0: the rows the worlds' values act on)."""
    import numpy as np
    if not active(params):
        return
    if not (np.asarray(m.dof_frictionloss) > 0).any():
        raise ValueError(f"PhysicsParams {[f for f in params.fields if f in FIELDS]}: the model has no frictionloss "
                         "rows; set dof_frictionloss > 0 on the dofs whose friction the worlds set (the model's value "
                         "marks the rows)")

"""Sphere-box contacts of the G1 engine (model_hooks): MuJoCo Warp's sphere_box primitive, the engine's sphere_box
(step.STEP_DEVICE, the core capsule-box uses; mujoco_warp/_src/collision_primitive_core.py, Copyright 2025 The Newton
Developers, Apache License 2.0). The forward's collide_pair and the adjoint's replay call one __noinline__ copy
(sb_sphere_box); the VJP replays the contact over duals (nablaengine.dual D1) once per pose direction of each geom,
with sphere_box as a template over the scalar type (sb_sphere_box_t, generated from its float text). One contact per
pair (slot 0)."""
import re

import mujoco

FWD = """  if (t1 == 1 && t2 == 3) {  // sphere-box (g1_engine.sphbox)
    float R2[9], dist, pos[3], n[3], fr[9];
    quat2mat(q2, R2);
    sb_sphere_box(p1, sz1[0], p2, R2, sz2, &dist, pos, n);
    if (dist < 0.f) {
      make_frame(n, fr);
      add_contact(s, pair, 0, dist, pos, fr);
    }
    return;
  }
"""

ADJ = """  if (t1 == 1 && t2 == 3) {  // sphere-box (g1_engine.sphbox): the forward's contact, then its VJP over duals (the
    // sphere's 3 and the box's 6 pose directions; the box's rotation columns turn with it)
    float R2[9], bn[3];
    quat2mat(q2, R2);
    sb_sphere_box(p1, sz1[0], p2, R2, sz2, dist, pos, bn);
    make_frame(bn, fr);
    if (!(-0.5f < bn[1] && bn[1] < 0.5f)) br |= ADJ_BR_FRAME_Z;
    if (cot) {
      float nbar[3];
      make_frame_vjp(fr, cot + 4, nbar);
      for (int j = 0; j < 12; j++) {
        const int g = j / 6, jj = j % 6;  // geom g (0: the sphere, 1: the box), direction jj
        if (g == 0 && jj >= 3) continue;  // (a sphere's rotation moves nothing)
        float ea[3] = {0.f, 0.f, 0.f};
        ea[jj % 3] = 1.f;
        D1 S[3], B[3], Rd[9], Z[3], dd, pp[3], nn[3];
        for (int t = 0; t < 3; t++) {
          S[t] = D1(p1[t], g == 0 && t == jj ? 1.f : 0.f);
          B[t] = D1(p2[t], g == 1 && jj < 3 && t == jj ? 1.f : 0.f);
          Z[t] = D1(sz2[t]);
        }
        for (int c = 0; c < 3; c++) {  // d col = e x col
          float col[3] = {R2[c], R2[3 + c], R2[6 + c]}, dc[3];
          cross(ea, col, dc);
          for (int r = 0; r < 3; r++) Rd[3 * r + c] = D1(R2[3 * r + c], g == 1 && jj >= 3 ? dc[r] : 0.f);
        }
        sb_sphere_box_t<D1>(S, D1(sz1[0]), B, Rd, Z, &dd, pp, nn);
        float gj = cot[0] * dd.d;
        for (int t = 0; t < 3; t++) gj += cot[1 + t] * pp[t].d + nbar[t] * nn[t].d;
        gb[j] = gj;
      }
    }
    return br;
  }
"""


def device():
    """The noinline float copy and the template (from step.STEP_DEVICE's sphere_box text)."""
    from nablaengine.dual import DUAL_DEVICE
    from nablaengine.g1_engine import step
    m = re.search(r"__device__ void sphere_box\(", step.STEP_DEVICE)
    if m is None:
        raise RuntimeError("sphbox: no sphere_box in step.STEP_DEVICE")
    body = step.STEP_DEVICE[m.start():step.STEP_DEVICE.index("\n}\n", m.start()) + 3]
    body = re.sub(r"\bfloat\b", "T", body)
    body = re.sub(r"\b(fabs|sqrt|fmin|fmax)f\(", r"t_\1(", body)
    body = re.sub(r"\b(mv3|mtv3|dot3)\(", r"t_\1(", body)
    body = body.replace("__device__ void sphere_box(", "template <class T>\n__device__ void sb_sphere_box_t(", 1)
    return (DUAL_DEVICE + "// sphere-box contacts (g1_engine.sphbox): sphere_box over a scalar type, and the forward's / "
            "replay's single float copy\n" + body
            + "__device__ __noinline__ void sb_sphere_box(const float* sp, float r, const float* bp, const float* R,\n"
            "                                           const float* bs, float* dist, float* pos, float* n) {\n"
            "  sphere_box(sp, r, bp, R, bs, dist, pos, n);\n}\n\n")


def pairs(m):
    """Whether model m has a sphere-box pair."""
    G = mujoco.mjtGeom
    want = sorted((int(G.mjGEOM_SPHERE), int(G.mjGEOM_BOX)))
    return any(sorted((int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]]))) == want
               for p in range(m.npair))


def hooks():
    """The (anchor, replacement, count) edits of the engine text for sphere-box pairs."""
    sphere_anchor = "  atomicOr(&s.unsupported, 1);  // other type combinations do not occur in the G1 model\n"
    dev_anchor = "// box-box (MuJoCo Warp's box_box, margin 0): separating-axis test, then face-vertex or edge-edge clipping\n"
    return ((sphere_anchor, FWD + sphere_anchor, 1),
            (dev_anchor, device() + dev_anchor, 1),
            ("  if (t1 != 2 || t2 != 2) return -1;\n", ADJ + "  if (t1 != 2 || t2 != 2) return -1;\n", 1))

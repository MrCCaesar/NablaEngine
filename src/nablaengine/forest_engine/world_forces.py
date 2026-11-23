"""Per-world gravity and external wrenches (nablaengine.params fields gravity and xfrc) in the forest engine's text. A
build without these fields is unchanged."""
import numpy as np

FIELDS = ("gravity", "xfrc")


def active(params):
    return bool(params) and any(f in params.fields for f in FIELDS)


def body_ids(m, params):
    """The model ids of PhysicsParams.xfrc_bodies (ValueError where one is not a moving body)."""
    import mujoco
    out = []
    for b in params.xfrc_bodies:
        bid = b if isinstance(b, int) else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if not 1 <= bid < m.nbody:
            raise ValueError(f"PhysicsParams xfrc_bodies: {b!r} is not a moving body of the model")
        out.append(int(bid))
    if len(set(out)) != len(out):
        raise ValueError(f"PhysicsParams xfrc_bodies {params.xfrc_bodies}: a body more than once")
    return out


def gcoff(m):
    from nablaengine.model import gravcomp
    return gravcomp.offsets(m) is not None


def check(m, params):
    """ValueError where gravity / xfrc do not apply to forest model m."""
    if not active(params):
        return
    if "xfrc" in params.fields:
        body_ids(m, params)
    if "gravity" in params.fields and gcoff(m):
        raise ValueError("PhysicsParams gravity: not with gravcomp applied off the COM (fused bodies of different "
                         "gravcomp, nablaengine.model.gravcomp)")


def defines(m, params):
    lay = params.layout(m.nv)
    gr = lay["gravity"][0] if "gravity" in lay else -1
    xo = lay["xfrc"][0] if "xfrc" in lay else -1
    ids = body_ids(m, params) if "xfrc" in lay else []
    slot = "".join(f"(b) == {b} ? {k} : " for k, b in enumerate(ids)) + "-1"
    return (f"#define PW_GRAV {gr}  // the world's gravity in its parameters (forest_engine.world_forces)\n"
            f"#define PW_XFRC {xo}  // the external wrenches (forest_engine.world_forces)\n"
            f"#define XFRC_SLOT(b) ({slot})  // body -> its wrench's slot, -1 none\n")


XFRC_DEVICE = r"""
// world_forces: the external wrench of body b (its xfrc slot; none: unchanged) on the body's RNE force about its origin:
// F -= f, N -= t + (R p) x f (f, t world frame, p body frame)
__device__ __forceinline__ void xfrc_apply(const float* pw, int b, const float* Q, float* F, float* N) {
  const int k = XFRC_SLOT(b);
  if (k < 0) return;
  const float* x = pw + (PW_XFRC >= 0 ? PW_XFRC : 0) + 9 * k;
  float e[3], u[3];
  qrot(Q, x + 6, e);
  cross(e, x, u);
  for (int i = 0; i < 3; i++) { F[i] -= x[i]; N[i] -= x[3 + i] + u[i]; }
}
"""
SUBSTEP_ANCHOR = "template <int PHASE = 0, class S>\n__device__ void substep_pre(S& s, const float* ctrl, int lane, float* rec) {\n"
FWD_CALL = "    body_force(b, Qb, Wb, ALb, Ab, ip, Fb, Nb);\n"
FWD_XFRC = FWD_CALL + "    xfrc_apply(s.pw, b, Qb, Fb, Nb);  // (world_forces)\n"
FWD_GRAV = ("s.A[0][k] = -c_grav[k]; }  // gravity\n", "s.A[0][k] = -s.pw[(PW_GRAV >= 0 ? PW_GRAV : 0) + k]; }  // gravity "
            "(world_forces: the world's)\n")
ADJ_GRAV = ("s.A[0][k] = -c_grav[k]; }\n  }\n  __syncwarp();\n",
            "s.A[0][k] = -s.pw[(PW_GRAV >= 0 ? PW_GRAV : 0) + k]; }  // (world_forces)\n  }\n  __syncwarp();\n")
ROOT_GRAV = ("A[k] = RSUB(qa[k], c_grav[k]); }", "A[k] = RSUB(qa[k], -Ap[k]); }  /* (world_forces: Ap = -g) */")
TREE_DOWN_END = ("  __syncwarp();\n  for (int i = lane; i < NV; i += 32) { s.Pq[i] = 0.f; s.Pv[i] = 0.f; }\n  __syncwarp();\n}\n"
                 "\n// A7")
ADJ_BLOCK = r"""  __syncwarp();
  {  // world_forces: the external wrenches' VJP and the cotangents of the wrenches and of gravity (module docstring)
#if PW_GRAV >= 0
    float g0 = 0.f, g1 = 0.f, g2 = 0.f;
#endif
    for (int b = lane; b < NB; b += 32) if (b >= 1) {
#if PW_GRAV >= 0
      const float wg = 1.f - c_gcomp[b];
      g0 += wg * s.bA[b][0]; g1 += wg * s.bA[b][1]; g2 += wg * s.bA[b][2];
#endif
#if PW_XFRC >= 0
      const int k = XFRC_SLOT(b);
      if (k >= 0) {
        const float* x = s.pw + (PW_XFRC >= 0 ? PW_XFRC : 0) + 9 * k;
        float* gx = s.gpw + (PW_XFRC >= 0 ? PW_XFRC : 0) + 9 * k;
        const float *Wl = s.Wl[b], *Vl = s.Vl[b];
        float e[3], u[3], fw[3];
        qrot(s.Q[b], x + 6, e);
        cross(Wl, e, u);
        for (int i = 0; i < 3; i++) { gx[i] += Vl[i] + u[i]; gx[3 + i] += Wl[i]; }
        cross(x, Wl, fw);  // f x Wl
        qrot_inv(s.Q[b], fw, u);
        for (int i = 0; i < 3; i++) gx[6 + i] += u[i];
        cross(e, x, u);
        for (int i = 0; i < 3; i++) { s.bVl[b][i] -= x[i]; s.bWl[b][i] -= x[3 + i] + u[i]; }
        cross(e, fw, u);  // e rotates with the body: e x (f x Wl)
        for (int i = 0; i < 3; i++) s.tb[b][i] -= u[i];
      }
#endif
    }
#if PW_GRAV >= 0
    g0 = warp_sum(g0); g1 = warp_sum(g1); g2 = warp_sum(g2);
    if (lane == 0) {
      s.gpw[(PW_GRAV >= 0 ? PW_GRAV : 0)] += g0;
      s.gpw[(PW_GRAV >= 0 ? PW_GRAV : 0) + 1] += g1;
      s.gpw[(PW_GRAV >= 0 ? PW_GRAV : 0) + 2] += g2;
    }
#endif
  }
"""


def _sub(src, a, b, n=1, what=""):
    if src.count(a) != n:
        raise RuntimeError(f"world_forces: {what or a[:60]!r} found {src.count(a)} times, expected {n}")
    return src.replace(a, b)


def _function_span(src, head):
    i = src.find(head)
    if i < 0 or src.find(head, i + 1) >= 0:
        raise RuntimeError(f"world_forces: function {head[:60]!r} not found once")
    j = src.find("\n}\n", i)
    return i, j + 3


def _gravcomp_gw(src, head, call_old, call_new):
    i, j = _function_span(src, head)
    body = src[i:j]
    k = body.find(") {\n")
    body = body[:k] + ", const float* gwld" + body[k:]
    body = body.replace("c_grav[", "gwld[")
    src = src[:i] + body + src[j:]
    return _sub(src, call_old, call_new)


def engine(src, m, params):
    """The final engine text (adjoint.cuda_source) of a build with gravity / xfrc (unchanged without)."""
    if not active(params):
        return src
    lay = params.layout(m.nv)
    if "xfrc" in lay:
        src = _sub(src, SUBSTEP_ANCHOR, XFRC_DEVICE + SUBSTEP_ANCHOR)
        src = _sub(src, FWD_CALL, FWD_XFRC)
    if "gravity" in lay:
        src = _sub(src, *FWD_GRAV)
        src = _sub(src, *ADJ_GRAV)
        src = _sub(src, *ROOT_GRAV)
        if np.any(np.asarray(m.body_gravcomp) != 0):
            gw = "s.pw + (PW_GRAV >= 0 ? PW_GRAV : 0)"
            src = _gravcomp_gw(src, "__device__ __forceinline__ void body_force(int b,", FWD_CALL,
                               FWD_CALL.replace("Fb, Nb);", f"Fb, Nb, {gw});"))
            src = _gravcomp_gw(src, "__device__ void body_local_vjp(int b,", "s.bWy[b], s.bVy[b], s.tb[b]);\n",
                               f"s.bWy[b], s.bVy[b], s.tb[b], {gw});\n")
        left = [ln for ln in src.splitlines() if "c_grav[" in ln and "c_grav[3]" not in ln
                and not ("gc * c_grav[k]" in ln or "c_gcomp[b] * c_grav[k]" in ln)]
        if left:
            raise RuntimeError(f"world_forces: c_grav read left in the engine text: {left[:3]}")
    src = _sub(src, TREE_DOWN_END, ADJ_BLOCK + TREE_DOWN_END, what="adj_tree_down's end")
    return defines(m, params) + src


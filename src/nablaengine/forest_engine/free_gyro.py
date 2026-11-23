"""The gyroscopic velocity derivative of standalone free bodies in implicitfast (MuJoCo 3.17)."""
import mujoco
import numpy as np

FWD_ANCHOR = "  if (rec) {  // tape: L_Q = chol(M + h kv), a* and a_adv"
DEF_ANCHOR = "template <bool NARROW> struct ZWord { typedef unsigned T; };"
ADJ_Y_ANCHOR = "  for (int i = lane; i < NV; i += 32) s.y[i] = s.abar[i];\n"
ADJ_COMPOSE_ANCHOR = "__device__ void adj_compose(Adj& s, int lane) {\n"
ADJ_COMPOSE_MOTOR = "__device__ void adj_compose(Adj& s, int lane, const float* rec) {\n"
ADJ_DEF_ANCHOR = "// A1 / A2, from abar = a_adv bar: y = Q^-1 a_adv bar (L_Q)"


def bodies(m):
    """The standalone free bodies (free joint, tree of 6 dofs) of an implicitfast model, ascending; [] otherwise."""
    if int(m.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
        return []
    out = []
    for b in range(1, m.nbody):
        if int(m.body_jntnum[b]) != 1:
            continue
        j = int(m.body_jntadr[b])
        if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        if int(m.tree_dofnum[int(m.dof_treeid[int(m.jnt_dofadr[j])])]) == 6:
            out.append(b)
    return out


def applies(m):
    return bool(bodies(m))


def check(m):
    """ValueError for a tendon with damping, or with an actuator whose kv is not 0, whose support includes a standalone
    free body's dofs (their implicit terms are added outside M; the block would miss them)."""
    if not applies(m):
        return
    for b in bodies(m):
        kids = [k for k in range(1, m.nbody) if int(m.body_parentid[k]) == b]
        if kids:
            raise ValueError(f"body {b}: a standalone free body with jointless child bodies {kids} (forest_engine."
                             "free_gyro takes the body's own inertials; fuse them, nablaengine.envs.convert)")
    from nablaengine.forest_engine import tendon
    dofs = set()
    for b in bodies(m):
        d0 = int(m.body_dofadr[b])
        dofs |= set(range(d0, d0 + 6))
    from nablaengine.forest_engine import muscle
    acts = set()
    for a, t in tendon.actuators(m):
        if muscle.is_general(m, a):
            kv = muscle.plain_constants(m, a)[2]
        else:
            kv = -m.actuator_biasprm[a, 2] if int(m.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE) else 0.0
        if kv != 0:
            acts.add(t)
    for t in range(m.ntendon):
        if not (tendon.damping(m, t) > 0 or t in acts):
            continue
        sup = {d for d, _ in tendon.support_list(m, t)}
        if sup & dofs:
            raise ValueError(f"tendon {t}: damping or an actuator on a standalone free body's dofs under implicitfast "
                             "(forest_engine.free_gyro: the body's implicit block does not include it)")


def _f(x):
    return f"{float(np.float32(x))!r}f"


def constants(m):
    """Per body: mass, COM offset (3, body frame), inertia about the COM in the body frame (xx yy zz xy xz yz)."""
    out = []
    for b in bodies(m):
        q = m.body_iquat[b]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        I = R @ np.diag(m.body_inertia[b]) @ R.T
        out.append([m.body_mass[b], *m.body_ipos[b], I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])
    return np.asarray(out, np.float64)


def _defs(m, damp):
    bs = bodies(m)
    c = constants(m)
    dof = ", ".join(str(int(m.body_dofadr[b])) for b in bs)
    qad = ", ".join(str(int(m.jnt_qposadr[m.body_jntadr[b]])) for b in bs)
    vals = ",\n  ".join(", ".join(_f(x) for x in row) for row in c)
    return rf"""#ifndef DK_FREE_GYRO
#define DK_FREE_GYRO 1  // (forest_engine.free_gyro; sm_90: nablaengine.cuda.forest_hold_defines)
#endif
#if DK_FREE_GYRO
// forest_engine.free_gyro: the standalone free bodies (MuJoCo 3.17's implicit free-body block): dof and qpos
// addresses, mass, COM offset, inertia about the COM (body frame: xx yy zz xy xz yz)
#define NFG {len(bs)}
__device__ const int c_fg_dof[NFG] = {{{dof}}}, c_fg_qadr[NFG] = {{{qad}}};
__device__ const float c_fg[NFG * 10] = {{
  {vals}}};
#define FG_DAMP(s, d) ({damp})

// R (row-major) of the normalized quaternion q
__device__ __forceinline__ void fg_rot(const float* q, float* R) {{
  const float n = rsqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const float Q[4] = {{q[0] * n, q[1] * n, q[2] * n, q[3] * n}};
  for (int k = 0; k < 3; k++) {{
    float e[3] = {{0.f, 0.f, 0.f}}, c[3];
    e[k] = 1.f;
    qrot(Q, e, c);
    R[k] = c[0]; R[3 + k] = c[1]; R[6 + k] = c[2];
  }}
}}

// body f's K(w) = s w' - (w . s) I + [w x s]x and C(w) = -m [s]x K + [w]x I_c - [I_c w]x (body frame, row-major)
__device__ __forceinline__ void fg_kc(int f, const float* w, float* K, float* C) {{
  const float* p = c_fg + 10 * f;
  const float m = p[0], s[3] = {{p[1], p[2], p[3]}};
  const float I[9] = {{p[4], p[7], p[8], p[7], p[5], p[9], p[8], p[9], p[6]}};
  const float ws = w[0] * s[0] + w[1] * s[1] + w[2] * s[2];
  const float x[3] = {{w[1] * s[2] - w[2] * s[1], w[2] * s[0] - w[0] * s[2], w[0] * s[1] - w[1] * s[0]}};  // w x s
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) K[3 * i + j] = s[i] * w[j] - (i == j ? ws : 0.f);
  K[1] -= x[2]; K[2] += x[1]; K[3] += x[2]; K[5] -= x[0]; K[6] -= x[1]; K[7] += x[0];
  float Iw[3];
  for (int i = 0; i < 3; i++) Iw[i] = I[3 * i] * w[0] + I[3 * i + 1] * w[1] + I[3 * i + 2] * w[2];
  for (int j = 0; j < 3; j++) {{
    const float k0 = K[j], k1 = K[3 + j], k2 = K[6 + j];
    const float i0 = I[j], i1 = I[3 + j], i2 = I[6 + j];
    C[j] = -m * (s[1] * k2 - s[2] * k1) + (w[1] * i2 - w[2] * i1);
    C[3 + j] = -m * (s[2] * k0 - s[0] * k2) + (w[2] * i0 - w[0] * i2);
    C[6 + j] = -m * (s[0] * k1 - s[1] * k0) + (w[0] * i1 - w[1] * i0);
  }}
  C[1] += Iw[2]; C[2] -= Iw[1]; C[3] -= Iw[2]; C[5] += Iw[0]; C[6] += Iw[1]; C[7] -= Iw[0];  // - [I_c w]x
}}

// body f's h G columns 3..5 (6 x 3, row-major) at the local angular velocity w and rotation R
__device__ __forceinline__ void fg_hg(int f, const float* w, const float* R, float h, float* G) {{
  float K[9], C[9];
  fg_kc(f, w, K, C);
  const float hm = -h * c_fg[10 * f];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {{
      G[3 * i + j] = hm * (R[3 * i] * K[j] + R[3 * i + 1] * K[3 + j] + R[3 * i + 2] * K[6 + j]);
      G[9 + 3 * i + j] = h * C[3 * i + j];
    }}
}}

// forward: body f's 6 entries of a_adv = A^-1 (M a*), A = M6 + h damping + h G (LU with partial pivoting)
template <class S>
__device__ __forceinline__ void fg_solve(S& s, int f, float h) {{
  const int d0 = c_fg_dof[f];
  float A[6][7], R[9], G[18];
  for (int i = 0; i < 6; i++) {{
    for (int j = 0; j < 6; j++) A[i][j] = s.M[i >= j ? TRI(d0 + i, d0 + j) : TRI(d0 + j, d0 + i)];
    A[i][i] += h * FG_DAMP(s, d0 + i);
    A[i][6] = s.Ma[d0 + i];
  }}
  fg_rot(s.qpos + c_fg_qadr[f] + 3, R);
  const float w[3] = {{s.qvel[d0 + 3], s.qvel[d0 + 4], s.qvel[d0 + 5]}};
  fg_hg(f, w, R, h, G);
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 3; j++) A[i][3 + j] += G[3 * i + j];
  for (int k = 0; k < 6; k++) {{
    int p = k;
    for (int i = k + 1; i < 6; i++)
      if (fabsf(A[i][k]) > fabsf(A[p][k])) p = i;
    if (p != k)
      for (int j = k; j < 7; j++) {{ const float t = A[k][j]; A[k][j] = A[p][j]; A[p][j] = t; }}
    const float inv = 1.f / A[k][k];
    for (int i = k + 1; i < 6; i++) {{
      const float l = A[i][k] * inv;
      for (int j = k + 1; j < 7; j++) A[i][j] -= l * A[k][j];
    }}
  }}
  for (int i = 5; i >= 0; i--) {{
    float x = A[i][6];
    for (int j = i + 1; j < 6; j++) x -= A[i][j] * A[j][6];
    A[i][6] = x / A[i][i];
  }}
  for (int i = 0; i < 6; i++) s.tmp[d0 + i] = A[i][6];
}}
#endif
"""


ADJ_DEVICE = r"""#if DK_FREE_GYRO
// adjoint (forest_engine.free_gyro), A1: columns k of Z = Q0^-1 (h G_c) of every body in zk (k = 0, 1, 2), then
// u = Q0 y from (I + h Z)' u = a_adv bar, written over a_adv bar's angular entries (lane f: body f)
__device__ __forceinline__ void fg_adj_cols(Adj& s, float* z0, float* z1, float* z2, int lane) {
  const float h = c_misc[0];
  for (int i = lane; i < NV; i += 32) { z0[i] = 0.f; z1[i] = 0.f; z2[i] = 0.f; }
  __syncwarp();
  for (int f = lane; f < NFG; f += 32) {
    const int d0 = c_fg_dof[f];
    float R[9], G[18];
    fg_rot(s.q + c_fg_qadr[f] + 3, R);
    const float w[3] = {s.v[d0 + 3], s.v[d0 + 4], s.v[d0 + 5]};
    fg_hg(f, w, R, h, G);
    for (int i = 0; i < 6; i++) { z0[d0 + i] = G[3 * i]; z1[d0 + i] = G[3 * i + 1]; z2[d0 + i] = G[3 * i + 2]; }
  }
  __syncwarp();
}
__device__ __forceinline__ void fg_adj_u(Adj& s, const float* z0, const float* z1, const float* z2, int lane) {
  for (int f = lane; f < NFG; f += 32) {
    const int d0 = c_fg_dof[f];
    const float* z[3] = {z0 + d0, z1 + d0, z2 + d0};
    float r[3], B[9];  // B = I + Z_r' (3 x 3), r = a_adv bar_r - Z_l' a_adv bar_l
    for (int k = 0; k < 3; k++) {
      r[k] = s.abar[d0 + 3 + k] - (z[k][0] * s.abar[d0] + z[k][1] * s.abar[d0 + 1] + z[k][2] * s.abar[d0 + 2]);
      for (int j = 0; j < 3; j++) B[3 * k + j] = (k == j ? 1.f : 0.f) + z[k][3 + j];
    }
    const float c0 = B[4] * B[8] - B[5] * B[7], c1 = B[5] * B[6] - B[3] * B[8], c2 = B[3] * B[7] - B[4] * B[6];
    const float idet = 1.f / (B[0] * c0 + B[1] * c1 + B[2] * c2);
    const float u0 = (r[0] * c0 + r[1] * (B[2] * B[7] - B[1] * B[8]) + r[2] * (B[1] * B[5] - B[2] * B[4])) * idet;
    const float u1 = (r[0] * c1 + r[1] * (B[0] * B[8] - B[2] * B[6]) + r[2] * (B[2] * B[3] - B[0] * B[5])) * idet;
    const float u2 = (r[0] * c2 + r[1] * (B[1] * B[6] - B[0] * B[7]) + r[2] * (B[0] * B[4] - B[1] * B[3])) * idet;
    s.abar[d0 + 3] = u0; s.abar[d0 + 4] = u1; s.abar[d0 + 5] = u2;
  }
  __syncwarp();
}
// adjoint, A8: P_v += h d(y' G a_adv)/dw, P_q += h d(y' G a_adv)/d(theta) of body f (y = -ny, a_adv = a* - a~)
__device__ __forceinline__ void fg_adj_compose(Adj& s, int lane) {
  const float h = c_misc[0];
  for (int f = lane; f < NFG; f += 32) {
    const int d0 = c_fg_dof[f];
    const float m = c_fg[10 * f];
    float R[9], K[9], C[9], y[6], a[3], u[3];
    fg_rot(s.q + c_fg_qadr[f] + 3, R);
    for (int i = 0; i < 6; i++) y[i] = -s.ny[d0 + i];
    for (int k = 0; k < 3; k++) a[k] = s.a[d0 + 3 + k] - s.at[d0 + 3 + k];
    for (int k = 0; k < 3; k++) u[k] = R[k] * y[0] + R[3 + k] * y[1] + R[6 + k] * y[2];  // R' y_l
    for (int k = 0; k < 3; k++) {  // d/dw_k: G is linear in w
      float e[3] = {0.f, 0.f, 0.f};
      e[k] = 1.f;
      fg_kc(f, e, K, C);
      float g = 0.f;
      for (int i = 0; i < 3; i++) {
        const float Ka = K[3 * i] * a[0] + K[3 * i + 1] * a[1] + K[3 * i + 2] * a[2];
        const float Ca = C[3 * i] * a[0] + C[3 * i + 1] * a[1] + C[3 * i + 2] * a[2];
        g += -m * u[i] * Ka + y[3 + i] * Ca;
      }
      s.Pv[d0 + 3 + k] += h * g;
    }
    const float w[3] = {s.v[d0 + 3], s.v[d0 + 4], s.v[d0 + 5]};
    fg_kc(f, w, K, C);
    float Ka[3];
    for (int i = 0; i < 3; i++) Ka[i] = K[3 * i] * a[0] + K[3 * i + 1] * a[1] + K[3 * i + 2] * a[2];
    const float hm = -h * m;  // d/d(theta) of -m y_l' R [theta]x K a: -m (K a) x (R' y_l)
    s.Pq[d0 + 3] += hm * (Ka[1] * u[2] - Ka[2] * u[1]);
    s.Pq[d0 + 4] += hm * (Ka[2] * u[0] - Ka[0] * u[2]);
    s.Pq[d0 + 5] += hm * (Ka[0] * u[1] - Ka[1] * u[0]);
  }
  __syncwarp();
}
#endif
"""


def _once(src, a, b):
    n = src.count(a)
    if n != 1:
        raise RuntimeError(f"free_gyro: anchor found {n} times: {a[:70]!r}")
    return src.replace(a, b)


def apply(src, m):
    """The edits for a model with standalone free bodies under implicitfast (bodies()); the text unchanged otherwise.
    On the final text (after world_params: a per-world damping build reads s.dampw)."""
    if not applies(m):
        return src
    check(m)
    damp = "s.dampw[d]" if "s.dampw[i]" in src else "c_damp[d]"
    src = _once(src, DEF_ANCHOR, _defs(m, damp.replace("[d]", "[(d)]")) + DEF_ANCHOR)
    src = _once(src, ADJ_DEF_ANCHOR, ADJ_DEVICE + ADJ_DEF_ANCHOR)
    src = _once(src, FWD_ANCHOR, "#if DK_FREE_GYRO  // (forest_engine.free_gyro: a_adv of the standalone free bodies)\n"
                "  __syncwarp();\n  for (int f = lane; f < NFG; f += 32) fg_solve(s, f, h);\n  __syncwarp();\n#endif\n"
                + FWD_ANCHOR)
    i = src.index(ADJ_Y_ANCHOR) + len(ADJ_Y_ANCHOR)
    line = src[i:src.index("\n", i) + 1]
    if "s.L" not in line or "s.y" not in line:
        raise RuntimeError(f"free_gyro: adj_solves' solve line: {line!r}")
    solves = "".join(line.replace("s.y", z) for z in ("s.y", "s.lam", "s.ny"))
    src = _once(src, ADJ_Y_ANCHOR, "#if DK_FREE_GYRO  // (forest_engine.free_gyro: a_adv bar -> u = Q0 y)\n"
                "  fg_adj_cols(s, s.y, s.lam, s.ny, lane);\n" + solves + "  __syncwarp();\n"
                "  fg_adj_u(s, s.y, s.lam, s.ny, lane);\n#endif\n" + ADJ_Y_ANCHOR)
    head = ADJ_COMPOSE_ANCHOR if ADJ_COMPOSE_ANCHOR in src else ADJ_COMPOSE_MOTOR
    src = _once(src, head, head + "#if DK_FREE_GYRO  // (forest_engine.free_gyro)\n"
                "  fg_adj_compose(s, lane);\n#endif\n")
    return src

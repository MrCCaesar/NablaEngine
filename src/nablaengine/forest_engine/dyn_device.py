"""CUDA source of the forest engine's dynamics queries (nablaengine.dyn_query): the joint-space mass matrix M(q)
(MuJoCo's mj_fullM of qM), the bias force qfrc_bias (Coriolis, centrifugal and gravity: MuJoCo's mj_rne at qacc = 0,
no passive forces) and the Jacobians of world points on bodies (MuJoCo's mj_jac: jacp, jacr, 3 x nv each), with the
VJP of the Jacobians with respect to qpos and the points."""
import mujoco
import numpy as np

KQ_WPB = 4

_DEVICE_END = "\n// in-place Cholesky"


def device_text():
    """source.DEVICE up to body_ipos (the dynamics' device functions; the solves after it are not needed)."""
    from nablaengine.forest_engine import source
    i = source.DEVICE.find(_DEVICE_END)
    assert i > 0 and "body_ipos(int b" in source.DEVICE[:i], "forest source.DEVICE: body_ipos before the solves"
    return source.DEVICE[:i] + "\n"


def tendon_armature(m):
    """The fixed tendons' armature as a dense (nv, nv) matrix sum_t a_t c_t c_t' (None without any): MuJoCo adds it to
    qM (mj_tendonArmature)."""
    if not m.ntendon or not np.any(m.tendon_armature > 0):
        return None
    out = np.zeros((m.nv, m.nv))
    for t in range(m.ntendon):
        a = float(m.tendon_armature[t])
        if a <= 0:
            continue
        if int(m.wrap_type[m.tendon_adr[t]]) != int(mujoco.mjtWrap.mjWRAP_JOINT):
            raise ValueError(f"dyn_query: tendon {t} has armature and is not a fixed tendon")
        c = np.zeros(m.nv)
        for k in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t]):
            j = int(m.wrap_objid[k])
            c[m.jnt_dofadr[j]] += m.wrap_prm[k]
        out += a * np.outer(c, c)
    anc = np.zeros((m.nv, m.nv), bool)
    for i in range(m.nv):
        b = int(m.dof_bodyid[i])
        while b > 0:
            for j in range(m.nv):
                if int(m.dof_bodyid[j]) == b:
                    anc[i, j] = anc[j, i] = True
            b = int(m.body_parentid[b])
    return np.where(anc, out, 0.0)


def constants(m, com_body=None):
    """(packed constants float32, loader layout, sizes): source.constants with gravcomp 0 (and gravity 0 under
    mjDSBL_GRAVITY), then c_dofbody, c_doft (the dof's index in its joint) and, with tendon armature, c_tarm."""
    from nablaengine.forest_engine import source
    c, maxd = source.constants(m, com_body)
    c = c.copy()
    lay = list(source.const_layout(m))
    sizes = {"nb": m.nbody, "nv": m.nv, "one": 1}
    off = {}
    o = 0
    for sym, _, cnt, width in lay:
        n = sizes[cnt] * width
        off[sym] = (o, n)
        o += n
    assert o == len(c), (o, len(c))
    o, n = off["c_gcomp"]
    c[o:o + n] = 0.0
    if m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY):
        o, n = off["c_grav"]
        c[o:o + n] = 0.0
    dofbody = np.asarray(m.dof_bodyid, np.float32)
    doft = np.asarray([i - m.jnt_dofadr[m.dof_jntid[i]] for i in range(m.nv)], np.float32)
    parts = [c, dofbody, doft]
    lay += [("c_dofbody", "i", "nv", 1), ("c_doft", "i", "nv", 1)]
    ta = tendon_armature(m)
    if ta is not None:
        parts.append(np.asarray(ta, np.float32).ravel())
        lay.append(("c_tarm", "f", "nv2", 1))
        sizes["nv2"] = m.nv * m.nv
    return np.concatenate(parts).astype(np.float32), lay, sizes, maxd


def header(m, maxd, params=None, pw_off=3):
    """The model header (source.header with this unit's extra tables) and the parameters' offsets (-1: absent)."""
    from nablaengine.forest_engine import source
    h = source.header(m, maxd, KQ_WPB)
    w = source.anc_width(m)
    if w != 12:
        a = "    c_anc[NB * 12];\n"
        assert h.count(a) == 1
        h = h.replace(a, f"    c_anc[NB * {w}];\n")
    lay = params.layout(m.nv) if params else {}
    off = lambda f: lay[f][0] if f in lay else -1
    npw = params.width(m.nv) if params else 0
    h += ("DKC int c_dofbody[NV], c_doft[NV];  // (dyn_query) dof -> body, dof -> index in its joint\n"
          + ("DKC float c_tarm[NV * NV];  // the fixed tendons' armature\n#define TARM(i) (c_tarm[i])\n"
             if tendon_armature(m) is not None else "#define TARM(i) 0.f\n")
          + f"#define KQ_WPB {KQ_WPB}\n#define PW_OFF {pw_off}\n#define NPARAM {npw}\n"
          f"#define COMW (PW_OFF + NPARAM)  // floats per world of the com input\n"
          f"#define PW_ARM {off('dof_armature')}\n#define PW_TMASS {off('torso_mass')}\n"
          f"#define PW_TINERT {off('torso_inertia')}\n#define PW_GRAV {off('gravity')}  // the world's gravity\n"
          f"#define KQ_GRAV_ON {int(not m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY))}\n"
          f"#define KQ_TORSO {int('torso_mass' in lay or 'torso_inertia' in lay)}\n")
    return h


def device(params=None):
    """device_text() with the torso fields' edits of body_ipos, body_force and body_inertia (forest_engine.world_params'
    TORSO_ENGINE entries whose anchors are in this text; ip[7]: offset, mass, moments) and, with the gravity field, a
    free root's acceleration from the world body's (fwd_body: A = qacc - g read as qacc - (-Ap), Ap = -g the world's;
    forest_engine.world_forces' ROOT_GRAV edit, the same bits), else unchanged."""
    src = device_text()
    if params and "gravity" in params.fields:
        from nablaengine.forest_engine import world_forces as WF
        a, b = WF.ROOT_GRAV
        if src.count(a) != 1:
            raise RuntimeError(f"dyn_query: the free root's gravity read found {src.count(a)} times, expected 1")
        src = src.replace(a, b)
    if not (params and ("torso_mass" in params.fields or "torso_inertia" in params.fields)):
        return src
    from nablaengine.forest_engine import world_params as W
    used = 0
    for a, b, n in W.TORSO_ENGINE:
        if a in src:
            if src.count(a) != n:
                raise RuntimeError(f"dyn_query: torso edit anchor found {src.count(a)} times, expected {n}: {a[:60]!r}")
            src = src.replace(a, b)
            used += 1
    if used != 5:
        raise RuntimeError(f"dyn_query: the torso edits of the device functions did not apply ({used} of 5)")
    return src


KERNELS = r"""
#ifndef KQ_RENORM
#define KQ_RENORM 1
#endif
struct KQ {  // one world's tree state (shared memory)
  float Q[NB][4], X[NB][3], W[NB][3], V[NB][3], AL[NB][3], A[NB][3], F[NB][3], N[NB][3], J[NB][10];
  float qpos[NQ], qvel[NV], zero[NV], bias[NV], gtan[NV];
  float com[7];  // the COM offset, the COM-offset body's mass and principal moments (KQ_TORSO)
};

template <class S>
__device__ __forceinline__ void kq_load(S& s, int w, const float* qpos, const float* qvel, const float* com, int lane) {
  for (int i = lane; i < NQ; i += 32) s.qpos[i] = qpos[(size_t)w * NQ + i];
  for (int i = lane; i < NV; i += 32) { s.qvel[i] = qvel ? qvel[(size_t)w * NV + i] : 0.f; s.zero[i] = 0.f; }
  if (lane < 3) s.com[lane] = com ? com[(size_t)w * COMW + lane] : 0.f;
#if KQ_TORSO
  if (lane < 4 && com) s.com[3 + lane] = torso_param(com + (size_t)w * COMW + PW_OFF, lane);  // (the Jacobians: none)
#endif
  if (lane == 0) {
    s.Q[0][0] = 1.f; s.Q[0][1] = s.Q[0][2] = s.Q[0][3] = 0.f;
#if PW_GRAV >= 0  // the world's gravity (PhysicsParams gravity; zero under mjDSBL_GRAVITY as opt.gravity)
    for (int k = 0; k < 3; k++) {
      s.X[0][k] = s.W[0][k] = s.V[0][k] = s.AL[0][k] = 0.f;
      s.A[0][k] = (com && KQ_GRAV_ON) ? -com[(size_t)w * COMW + PW_OFF + (PW_GRAV >= 0 ? PW_GRAV : 0) + k] : -c_grav[k];
    }
#else
    for (int k = 0; k < 3; k++) { s.X[0][k] = s.W[0][k] = s.V[0][k] = s.AL[0][k] = 0.f; s.A[0][k] = -c_grav[k]; }
#endif
  }
  __syncwarp();
}

// poses, twists and the bias accelerations (qacc = 0) root to leaf (the forward's fwd_body)
template <class S>
__device__ __forceinline__ void kq_tree_down(S& s, int lane) {
  for (int L = 1; L <= MAXD; L++) {
    for (int b = lane; b < NB; b += 32)
      if (c_depth[b] == L) {
        const int p = c_parent[b];
        float Qp[4], Xp[3], Wp[3], Vp[3], ALp[3], Ap[3], Qb[4], Xb[3], Wb[3], Vb[3], ALb[3], Ab[3];
        for (int k = 0; k < 4; k++) Qp[k] = s.Q[p][k];
        for (int k = 0; k < 3; k++) {
          Xp[k] = s.X[p][k]; Wp[k] = s.W[p][k]; Vp[k] = s.V[p][k]; ALp[k] = s.AL[p][k]; Ap[k] = s.A[p][k];
        }
        fwd_body(b, Qp, Xp, Wp, Vp, ALp, Ap, s.qpos, s.qvel, s.zero, Qb, Xb, Wb, Vb, ALb, Ab);
#if KQ_RENORM  // the body's quaternion renormalized, as MuJoCo's mj_kinematics does (the step's forward does not)
        const float rn = 1.f / sqrtf(Qb[0] * Qb[0] + Qb[1] * Qb[1] + Qb[2] * Qb[2] + Qb[3] * Qb[3]);
        for (int k = 0; k < 4; k++) Qb[k] *= rn;
#endif
        for (int k = 0; k < 4; k++) s.Q[b][k] = Qb[k];
        for (int k = 0; k < 3; k++) {
          s.X[b][k] = Xb[k]; s.W[b][k] = Wb[k]; s.V[b][k] = Vb[k]; s.AL[b][k] = ALb[k]; s.A[b][k] = Ab[k];
        }
      }
    __syncwarp();
  }
}

__global__ void __launch_bounds__(32 * KQ_WPB) k_dq_dyn(int nw, const float* qpos, const float* qvel,
                                                        const float* com, float* oM, float* obias) {
  __shared__ KQ sm[KQ_WPB];
  const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5, w = blockIdx.x * KQ_WPB + wi;
  if (w >= nw) return;  // (the whole warp)
  KQ& s = sm[wi];
  kq_load(s, w, qpos, qvel, com, lane);
  kq_tree_down(s, lane);
  for (int b = lane; b < NB; b += 32)  // each body's force and its spatial inertia about its tree root's origin
    if (b >= 1) {
      const int rt = c_root[b];
#if KQ_TORSO
      float ip[7];
#else
      float ip[3];
#endif
      float Qb[4], Xb[3], Wb[3], ALb[3], Ab[3], Fb[3], Nb[3], Jb[10], O[3];
      for (int k = 0; k < 4; k++) Qb[k] = s.Q[b][k];
      for (int k = 0; k < 3; k++) {
        Xb[k] = s.X[b][k]; Wb[k] = s.W[b][k]; ALb[k] = s.AL[b][k]; Ab[k] = s.A[b][k]; O[k] = s.X[rt][k];
      }
      body_ipos(b, s.com, ip);
      body_force(b, Qb, Wb, ALb, Ab, ip, Fb, Nb);
      body_inertia(b, Qb, Xb, O, ip, Jb);
      for (int k = 0; k < 3; k++) { s.F[b][k] = Fb[k]; s.N[b][k] = Nb[k]; }
      for (int k = 0; k < 10; k++) s.J[b][k] = Jb[k];
    }
  __syncwarp();
  for (int L = MAXD - 1; L >= 1; L--) {  // leaf to root: forces, moments about the body origins, composite inertias
    for (int b = lane; b < NB; b += 32)
      if (c_depth[b] == L && c_nchild[b] > 0) {
        float Xb[3], Fb[3], Nb[3], Jb[10];
        for (int k = 0; k < 3; k++) { Xb[k] = s.X[b][k]; Fb[k] = s.F[b][k]; Nb[k] = s.N[b][k]; }
        for (int k = 0; k < 10; k++) Jb[k] = s.J[b][k];
        for (int i = 0; i < c_nchild[b]; i++) {
          const int c = c_child[NCH * b + i];
          float Xc[3], Fc[3], Nc[3];
          for (int k = 0; k < 3; k++) { Xc[k] = s.X[c][k]; Fc[k] = s.F[c][k]; Nc[k] = s.N[c][k]; }
          accumulate(Xb, Xc, Fc, Nc, Fb, Nb);
          for (int k = 0; k < 10; k++) Jb[k] += s.J[c][k];
        }
        for (int k = 0; k < 3; k++) { s.F[b][k] = Fb[k]; s.N[b][k] = Nb[k]; }
        for (int k = 0; k < 10; k++) s.J[b][k] = Jb[k];
      }
    __syncwarp();
  }
  for (int b = lane; b < NB; b += 32)
    if (b >= 1) torques(b, s.Q[b], s.F[b], s.N[b], s.bias);
  float* Mw = oM + (size_t)w * NV * NV;
  for (int i = lane; i < NV * NV; i += 32) Mw[i] = TARM(i);
  __syncwarp();
  for (int i = lane; i < NV; i += 32) {  // row i: the dofs of its body (up to i) and of the body's ancestors
    const int b = c_dofbody[i], rt = c_root[b];
    const float* O = s.X[rt];
    float sw[3], sv[3], kw[3], kv[3], n[3], f[3];
    dof_motion(b, c_doft[i], s.Q[b], s.X[b], O, sw, sv);
    apply_inertia(s.J[b], sw, sv, n, f);
#if PW_ARM >= 0
    const float arm = com[(size_t)w * COMW + PW_OFF + PW_ARM + i];
#else
    const float arm = c_armature[i];
#endif
    for (int k = c_dadr[b]; k <= i; k++) {
      dof_motion(b, c_doft[k], s.Q[b], s.X[b], O, kw, kv);
      const float v = dot3(kw, n) + dot3(kv, f) + (k == i ? arm : 0.f) + TARM(i * NV + k);
      Mw[i * NV + k] = v;
      Mw[k * NV + i] = v;
    }
    for (int a = c_parent[b]; a > 0; a = c_parent[a]) {
      const int k0 = c_dadr[a], nk = JT_OF(a) == JFREE ? 6 : 1;
      for (int t = 0; t < nk; t++) {
        dof_motion(a, t, s.Q[a], s.X[a], O, kw, kv);
        const float v = dot3(kw, n) + dot3(kv, f) + TARM(i * NV + k0 + t);
        Mw[i * NV + k0 + t] = v;
        Mw[(k0 + t) * NV + i] = v;
      }
    }
  }
  __syncwarp();
  for (int i = lane; i < NV; i += 32) obias[(size_t)w * NV + i] = s.bias[i];
}

// the column of dof i (body b) at the world point p: jacr w, jacp v (MuJoCo's mj_jac)
__device__ __forceinline__ void kq_col(const KQ& s, int i, int b, const float* p, float* w, float* v) {
  const int t = c_doft[i];
  dof_motion(b, t, s.Q[b], s.X[b], p, w, v);
  if (JT_OF(b) == JFREE && t >= 3) {  // (dof_motion takes a free body's rotations about its origin)
    float r[3] = {p[0] - s.X[b][0], p[1] - s.X[b][1], p[2] - s.X[b][2]};
    cross(w, r, v);
  }
}
// whether body b is body a or one of its ancestors
__device__ __forceinline__ bool kq_in_chain(int b, int a) {
  while (a > 0 && a != b) a = c_parent[a];
  return a == b && b > 0;
}

__global__ void __launch_bounds__(32 * KQ_WPB) k_dq_jac(int nw, int np, const float* qpos, const int* bodies,
                                                        const float* pts, float* jacp, float* jacr) {
  __shared__ KQ sm[KQ_WPB];
  const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5, w = blockIdx.x * KQ_WPB + wi;
  if (w >= nw) return;
  KQ& s = sm[wi];
  kq_load(s, w, qpos, nullptr, nullptr, lane);
  kq_tree_down(s, lane);
  for (int j = 0; j < np; j++) {
    const size_t q = (size_t)w * np + j;
    const int bp = bodies[q];
    const float p[3] = {pts[3 * q], pts[3 * q + 1], pts[3 * q + 2]};
    const bool ok = bp >= 0 && bp < NB;
    float* op = jacp + q * 3 * NV;
    float* orr = jacr + q * 3 * NV;
    for (int i = lane; i < NV; i += 32) {
      float cw[3] = {0.f, 0.f, 0.f}, cv[3] = {0.f, 0.f, 0.f};
      const int b = c_dofbody[i];
      if (!ok) { cw[0] = cw[1] = cw[2] = cv[0] = cv[1] = cv[2] = __int_as_float(0x7fc00000); }
      else if (kq_in_chain(b, bp)) kq_col(s, i, b, p, cw, cv);
      for (int r = 0; r < 3; r++) { op[r * NV + i] = cv[r]; orr[r * NV + i] = cw[r]; }
    }
  }
}

// the twist of dof k (body b): rotation om about the point c, translation tr
__device__ __forceinline__ void kq_twist(const KQ& s, int k, int b, float* om, float* c, float* tr) {
  const int t = c_doft[k], jt = JT_OF(b);
  for (int r = 0; r < 3; r++) { om[r] = 0.f; tr[r] = 0.f; c[r] = s.X[b][r]; }
  if (jt == JHINGE) qrot(s.Q[b], c_axis + 3 * b, om);
  else if (jt == JSLIDE) qrot(s.Q[b], c_axis + 3 * b, tr);
  else if (t < 3) tr[t] = 1.f;
  else { float e[3] = {0.f, 0.f, 0.f}; e[t - 3] = 1.f; qrot(s.Q[b], e, om); }
}

__global__ void __launch_bounds__(32 * KQ_WPB) k_dq_jac_vjp(int nw, int np, const float* qpos, const int* bodies,
                                                            const float* pts, const float* gjacp, const float* gjacr,
                                                            float* gq, float* gpts) {
  __shared__ KQ sm[KQ_WPB];
  const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5, w = blockIdx.x * KQ_WPB + wi;
  if (w >= nw) return;
  KQ& s = sm[wi];
  kq_load(s, w, qpos, nullptr, nullptr, lane);
  kq_tree_down(s, lane);
  for (int i = lane; i < NV; i += 32) s.gtan[i] = 0.f;
  __syncwarp();
  for (int j = 0; j < np; j++) {
    const size_t q = (size_t)w * np + j;
    const int bp = bodies[q];
    const float p[3] = {pts[3 * q], pts[3 * q + 1], pts[3 * q + 2]};
    const bool ok = bp > 0 && bp < NB;
    const float* Gp = gjacp + q * 3 * NV;
    const float* Gr = gjacr + q * 3 * NV;
    float gp[3] = {0.f, 0.f, 0.f};
    for (int k = lane; k < NV; k += 32) {
      const int bk = c_dofbody[k];
      if (!ok || !kq_in_chain(bk, bp)) continue;
      float om[3], c[3], tr[3], gt = 0.f;
      kq_twist(s, k, bk, om, c, tr);
      const bool kfree = JT_OF(bk) == JFREE;
      for (int i = 0; i < NV; i++) {  // the columns distal to dof k
        const int bi = c_dofbody[i];
        if (!kq_in_chain(bi, bp)) continue;
        const int ti = c_doft[i], jti = JT_OF(bi);
        bool distal = bi != bk && kq_in_chain(bk, bi);
        if (bi == bk && kfree && ti >= 3) distal = true;  // a free joint's rotation columns move with its own dofs
        if (!distal || (jti == JFREE && ti < 3)) continue;
        const float g_p[3] = {Gp[i], Gp[NV + i], Gp[2 * NV + i]}, g_r[3] = {Gr[i], Gr[NV + i], Gr[2 * NV + i]};
        if (jti == JSLIDE) {
          float a[3], d[3];
          qrot(s.Q[bi], c_axis + 3 * bi, a);
          cross(om, a, d);
          gt += dot3(g_p, d);
          continue;
        }
        float wv[3], cv[3], dw[3], r[3], dX[3], u[3], u2[3];
        kq_col(s, i, bi, p, wv, cv);
        cross(om, wv, dw);
        for (int e = 0; e < 3; e++) r[e] = s.X[bi][e] - c[e];
        cross(om, r, dX);
        for (int e = 0; e < 3; e++) { dX[e] += tr[e]; r[e] = p[e] - s.X[bi][e]; }
        cross(dw, r, u);
        cross(wv, dX, u2);
        for (int e = 0; e < 3; e++) u[e] -= u2[e];
        gt += dot3(g_r, dw) + dot3(g_p, u);
      }
      s.gtan[k] += gt;
      if (!(kfree && c_doft[k] < 3) && JT_OF(bk) != JSLIDE) {  // the point's: Gp_k x w_k
        float wv[3], cv[3], g_p[3] = {Gp[k], Gp[NV + k], Gp[2 * NV + k]}, u[3];
        kq_col(s, k, bk, p, wv, cv);
        cross(g_p, wv, u);
        for (int e = 0; e < 3; e++) gp[e] += u[e];
      }
    }
    for (int e = 0; e < 3; e++) gp[e] = warp_sum(gp[e]);
    if (lane < 3) gpts[3 * q + lane] = ok ? gp[lane] : 0.f;
    __syncwarp();
  }
  __syncwarp();
  // tangent -> qpos
  for (int b = lane; b < NB; b += 32) {
    if (b < 1) continue;
    const int qa = c_qadr[b], da = c_dadr[b];
    float* g = gq + (size_t)w * NQ;
    if (JT_OF(b) != JFREE) { g[qa] = s.gtan[da]; continue; }
    for (int e = 0; e < 3; e++) g[qa + e] = s.gtan[da + e];
    const float* qp = s.qpos + qa + 3;
    const float nrm = sqrtf(qp[0] * qp[0] + qp[1] * qp[1] + qp[2] * qp[2] + qp[3] * qp[3]);
    const float v[4] = {0.f, s.gtan[da + 3], s.gtan[da + 4], s.gtan[da + 5]};
    float a[4];
    qmul(s.Q[b], v, a);
    for (int e = 0; e < 4; e++) g[qa + 3 + e] = 2.f * a[e] / nrm;
  }
}

// ---- kinematics outputs (jax_ffi.kinematics on the forest engine): qpos_n, xpos, xquat, cvel, subtree_com with the
// conventions of nablaengine.g1_engine.kinematics generalized to the forest (MuJoCo's mj_kinematics + mj_comPos +
// mj_comVel): every free joint's quaternion normalized into qpos_n; subtree_com of every body (body 0: all trees;
// a subtree lighter than mjMINVAL: its body's xipos); cvel_b = [W_b; V_b + W_b x (subtree_com[root(b)] - X_b)] at
// the subtree COM of b's tree root (body 0: zeros)
#define KQ_MINVAL 1e-15f
struct KK {
  float Q[NB][4], X[NB][3], W[NB][3], V[NB][3];
  union {
    struct { float AL[NB][3], A[NB][3]; };  // (fwd_body's accelerations: unused)
    struct { float bW[NB][3], bV[NB][3]; };  // the VJP's twist cotangents
  };
  float XI[NB][3], C[NB][3], Mt[NB];  // xipos, subtree sums -> subtree COMs, subtree masses
  float F[NB][3], N[NB][3], rb[NB][3], G[NB][3];  // the VJP's pose sweep, r bar, subtree-sum path cotangents
  float qpos[NQ], qvel[NV], zero[NV], gq[NQ], gv[NV];
  float com[7];
};

__device__ __forceinline__ float kq_mass(int b, const float* ip) {
#if KQ_TORSO
  return ip[3];
#else
  (void)ip;
  return c_mass[b];
#endif
}

__device__ __forceinline__ void kq_subtree(KK& s, int lane) {
  for (int b = lane; b < NB; b += 32) {
    if (b == 0) { for (int k = 0; k < 3; k++) s.XI[0][k] = s.C[0][k] = 0.f; s.Mt[0] = 0.f; continue; }
#if KQ_TORSO
    float ip[7];
#else
    float ip[3];
#endif
    float c[3];
    body_ipos(b, s.com, ip);
    qrot(s.Q[b], ip, c);
    const float m = kq_mass(b, ip);
    for (int k = 0; k < 3; k++) { s.XI[b][k] = s.X[b][k] + c[k]; s.C[b][k] = m * s.XI[b][k]; }
    s.Mt[b] = m;
  }
  __syncwarp();
  for (int L = MAXD - 1; L >= 1; L--) {
    for (int b = lane; b < NB; b += 32)
      if (c_depth[b] == L)
        for (int i = 0; i < c_nchild[b]; i++) {
          const int c = c_child[NCH * b + i];
          for (int k = 0; k < 3; k++) s.C[b][k] += s.C[c][k];
          s.Mt[b] += s.Mt[c];
        }
    __syncwarp();
  }
  if (lane == 0)  // the world body: every tree, in body order
    for (int b = 1; b < NB; b++)
      if (c_parent[b] == 0) { for (int k = 0; k < 3; k++) s.C[0][k] += s.C[b][k]; s.Mt[0] += s.Mt[b]; }
  __syncwarp();
  for (int b = lane; b < NB; b += 32) {
    const float M = s.Mt[b];
    for (int k = 0; k < 3; k++) s.C[b][k] = M >= KQ_MINVAL ? s.C[b][k] / M : s.XI[b][k];
  }
  __syncwarp();
}

__global__ void __launch_bounds__(32 * KQ_WPB) k_dq_kin(int nw, const float* qpos, const float* qvel, const float* com,
                                                        float* qn, float* xpos, float* xquat, float* cvel, float* sc) {
  __shared__ KK sm[KQ_WPB];
  const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5, w = blockIdx.x * KQ_WPB + wi;
  if (w >= nw) return;
  KK& s = sm[wi];
  kq_load(s, w, qpos, qvel, com, lane);
  kq_tree_down(s, lane);
  kq_subtree(s, lane);
  for (int i = lane; i < NQ; i += 32) qn[(size_t)w * NQ + i] = s.qpos[i];
  __syncwarp();
  for (int b = lane; b < NB; b += 32) {
    float* xp = xpos + ((size_t)w * NB + b) * 3;
    float* xq = xquat + ((size_t)w * NB + b) * 4;
    float* cv = cvel + ((size_t)w * NB + b) * 6;
    float* o = sc + ((size_t)w * NB + b) * 3;
    for (int k = 0; k < 3; k++) { xp[k] = s.X[b][k]; o[k] = s.C[b][k]; }
    for (int k = 0; k < 4; k++) xq[k] = s.Q[b][k];
    if (b == 0) { for (int k = 0; k < 6; k++) cv[k] = 0.f; continue; }
    if (JT_OF(b) == JFREE)
      for (int k = 0; k < 4; k++) qn[(size_t)w * NQ + c_qadr[b] + 3 + k] = s.Q[b][k];
    const int rt = c_root[b];
    float r[3], u[3];
    for (int k = 0; k < 3; k++) r[k] = s.C[rt][k] - s.X[b][k];
    cross(s.W[b], r, u);
    for (int k = 0; k < 3; k++) { cv[k] = s.W[b][k]; cv[3 + k] = s.V[b][k] + u[k]; }
  }
}

// the VJP: cotangents of k_dq_kin's outputs -> gq (nw, NQ), gv (nw, NV) of (qpos, qvel); gq_in / gv_in (null: zero)
// added last
__global__ void __launch_bounds__(32 * KQ_WPB) k_dq_kin_vjp(int nw, const float* qpos, const float* qvel,
                                                            const float* com, const float* gqn, const float* gxpos,
                                                            const float* gxquat, const float* gcvel, const float* gsc,
                                                            const float* gq_in, const float* gv_in, float* gq,
                                                            float* gv) {
  __shared__ KK sm[KQ_WPB];
  const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5, w = blockIdx.x * KQ_WPB + wi;
  if (w >= nw) return;
  KK& s = sm[wi];
  kq_load(s, w, qpos, qvel, com, lane);
  kq_tree_down(s, lane);
  kq_subtree(s, lane);
  // cvel: W bar, V bar; r = C_root - X_b: r bar = V bar x W_b (into rb: the root's C bar, -X_b's f bar)
  for (int b = lane; b < NB; b += 32) {
    const size_t row = (size_t)w * NB + b;
    float wb[3], vb[3], r[3], t[3], rr[3] = {0.f, 0.f, 0.f};
    for (int k = 0; k < 3; k++) { wb[k] = gcvel[row * 6 + k]; vb[k] = gcvel[row * 6 + 3 + k]; }
    if (b == 0) for (int k = 0; k < 3; k++) wb[k] = vb[k] = 0.f;
    else {
      for (int k = 0; k < 3; k++) r[k] = s.C[c_root[b]][k] - s.X[b][k];
      cross(r, vb, t);
      for (int k = 0; k < 3; k++) wb[k] += t[k];
      cross(vb, s.W[b], rr);
    }
    for (int k = 0; k < 3; k++) { s.bW[b][k] = wb[k]; s.bV[b][k] = vb[k]; s.rb[b][k] = rr[k]; }
  }
  __syncwarp();
  // subtree COM cotangents: direct + the cvel reference points (per tree root, its bodies in order)
  for (int b = lane; b < NB; b += 32) {
    float g[3];
    for (int k = 0; k < 3; k++) g[k] = gsc[((size_t)w * NB + b) * 3 + k];
    if (b >= 1 && c_parent[b] == 0)
      for (int d = 1; d < NB; d++)
        if (c_root[d] == b) for (int k = 0; k < 3; k++) g[k] += s.rb[d][k];
    for (int k = 0; k < 3; k++) s.G[b][k] = g[k];
  }
  __syncwarp();
  // C_b = S_b / M_b (or xipos_b): S bar_b = C bar_b / M_b; G_d = sum over b in {0, ancestors, d} of S bar_b; xipos bar
  // _d = m_d G_d (+ C bar_d where M_d < MINVAL); f bar_d = xipos bar_d - r bar_d (cvel), tau bar_d = c_d x xipos bar_d
  // + 1/2 vec(g_Q Q*) (xquat, and qpos_n's quaternion of a free body)
  float g0[3];
  for (int k = 0; k < 3; k++) g0[k] = s.Mt[0] >= KQ_MINVAL ? s.G[0][k] / s.Mt[0] : 0.f;
  __syncwarp();
  for (int b = lane; b < NB; b += 32) {  // S bar in place (G), the direct xipos part into F
    const bool heavy = s.Mt[b] >= KQ_MINVAL;
    for (int k = 0; k < 3; k++) {
      s.F[b][k] = heavy ? 0.f : s.G[b][k];
      s.G[b][k] = heavy ? s.G[b][k] / s.Mt[b] : 0.f;
    }
  }
  __syncwarp();
  if (lane == 0) for (int k = 0; k < 3; k++) s.G[0][k] = g0[k];
  __syncwarp();
  for (int L = 1; L <= MAXD; L++) {
    for (int b = lane; b < NB; b += 32)
      if (c_depth[b] == L) for (int k = 0; k < 3; k++) s.G[b][k] += s.G[c_parent[b]][k];
    __syncwarp();
  }
  for (int b = lane; b < NB; b += 32) {
    if (b == 0) continue;
    const size_t row = (size_t)w * NB + b;
#if KQ_TORSO
    float ip[7];
#else
    float ip[3];
#endif
    float c[3], g[3], t[3], tb[3];
    body_ipos(b, s.com, ip);
    qrot(s.Q[b], ip, c);
    const float m = kq_mass(b, ip);
    for (int k = 0; k < 3; k++) g[k] = s.F[b][k] + m * s.G[b][k];
    cross(c, g, t);
    float gQ[4];
    for (int k = 0; k < 4; k++) gQ[k] = gxquat[row * 4 + k];
    if (JT_OF(b) == JFREE)
      for (int k = 0; k < 4; k++) gQ[k] += gqn[(size_t)w * NQ + c_qadr[b] + 3 + k];
    const float* q = s.Q[b];  // vec(g_Q q*)
    tb[0] = 0.5f * (-gQ[0] * q[1] + gQ[1] * q[0] - gQ[2] * q[3] + gQ[3] * q[2]);
    tb[1] = 0.5f * (-gQ[0] * q[2] + gQ[1] * q[3] + gQ[2] * q[0] - gQ[3] * q[1]);
    tb[2] = 0.5f * (-gQ[0] * q[3] - gQ[1] * q[2] + gQ[2] * q[1] + gQ[3] * q[0]);
    for (int k = 0; k < 3; k++) {
      s.F[b][k] = g[k] + gxpos[row * 3 + k] - s.rb[b][k];
      s.N[b][k] = t[k] + tb[k];
    }
  }
  __syncwarp();
  // leaf to root: the twist recursion reversed with the pose sweep (a parent pulls its children's finished cotangents)
  for (int L = MAXD; L >= 1; L--) {
    for (int b = lane; b < NB; b += 32) {
      if (c_depth[b] != L) continue;
      const int p = c_parent[b], jt = JT_OF(b);
      float bW[3], bV[3], F[3], N[3], t[3], u[3];
      for (int k = 0; k < 3; k++) { bW[k] = s.bW[b][k]; bV[k] = s.bV[b][k]; F[k] = s.F[b][k]; N[k] = s.N[b][k]; }
      for (int i = 0; i < c_nchild[b]; i++) {
        const int c = c_child[NCH * b + i];
        float r[3];
        for (int k = 0; k < 3; k++) r[k] = s.X[c][k] - s.X[b][k];
        cross(r, s.bV[c], t);  // V_c = V_b + W_b x r_c
        cross(r, s.F[c], u);
        for (int k = 0; k < 3; k++) {
          bW[k] += s.bW[c][k] + t[k]; bV[k] += s.bV[c][k];
          F[k] += s.F[c][k] - s.rb[c][k]; N[k] += s.N[c][k] + u[k];
        }
      }
      const int iv = c_dadr[b], iq = c_qadr[b];
      float rb[3] = {0.f, 0.f, 0.f};
      if (jt == JFREE) {  // V = qvel[0:3], W = R qvel[3:6]; X = qpos[0:3], Q = normalize(qpos[3:7])
        float l[3];
        qrot_inv(s.Q[b], bW, l);
        for (int k = 0; k < 3; k++) { s.gv[iv + k] = bV[k]; s.gv[iv + 3 + k] = l[k]; s.gq[iq + k] = F[k]; }
        cross(s.W[b], bW, t);
        for (int k = 0; k < 3; k++) N[k] += t[k];
        const float n0[4] = {0.f, N[0], N[1], N[2]}, *u4 = s.Q[b], *q = s.qpos + iq + 3;
        float a[4];
        qmul(n0, u4, a);  // dL/dQ (tangential) = 2 [0; N] Q
        const float nq = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        const float ua = u4[0] * a[0] + u4[1] * a[1] + u4[2] * a[2] + u4[3] * a[3];
        for (int k = 0; k < 4; k++) s.gq[iq + 3 + k] = 2.f * (a[k] - u4[k] * ua) / nq;
      } else {
        float z[3];
        qrot(s.Q[b], c_axis + 3 * b, z);
        const float qd = s.qvel[iv];
        if (jt == JHINGE) {  // W = W_p + z qd
          s.gv[iv] = dot3(z, bW);
          for (int k = 0; k < 3; k++) u[k] = qd * bW[k];
        } else {  // slide: V = V_p + W_p x r + z qd
          s.gv[iv] = dot3(z, bV);
          for (int k = 0; k < 3; k++) u[k] = qd * bV[k];
        }
        cross(z, u, t);  // z = R_b axis rotates with the body
        cross(bV, s.W[p], rb);  // V_b = V_p + W_p x r_b
        for (int k = 0; k < 3; k++) { N[k] += t[k]; F[k] += rb[k]; }
        s.gq[iq] = jt == JHINGE ? dot3(z, N) : dot3(z, F);
      }
      for (int k = 0; k < 3; k++) {
        s.bW[b][k] = bW[k]; s.bV[b][k] = bV[k]; s.F[b][k] = F[k]; s.N[b][k] = N[k]; s.rb[b][k] = rb[k];
      }
    }
    __syncwarp();
  }
  for (int i = lane; i < NQ; i += 32) {  // qpos_n passes the entries other than the free quaternions through
    float g = s.gq[i];
    bool fq = false;
    for (int b = 1; b < NB; b++)
      if (JT_OF(b) == JFREE && i >= c_qadr[b] + 3 && i < c_qadr[b] + 7) fq = true;
    if (!fq) g += gqn[(size_t)w * NQ + i];
    gq[(size_t)w * NQ + i] = gq_in ? gq_in[(size_t)w * NQ + i] + g : g;
  }
  for (int i = lane; i < NV; i += 32) gv[(size_t)w * NV + i] = gv_in ? gv_in[(size_t)w * NV + i] + s.gv[i] : s.gv[i];
}

extern "C" int dq_kinematics(cudaStream_t st, int nw, const float* qpos, const float* qvel, const float* com,
                             float* qn, float* xpos, float* xquat, float* cvel, float* sc) {
  if (nw > 0) k_dq_kin<<<(nw + KQ_WPB - 1) / KQ_WPB, 32 * KQ_WPB, 0, st>>>(nw, qpos, qvel, com, qn, xpos, xquat, cvel, sc);
  return (int)cudaGetLastError();
}
extern "C" int dq_kinematics_vjp(cudaStream_t st, int nw, const float* qpos, const float* qvel, const float* com,
                                 const float* gqn, const float* gxpos, const float* gxquat, const float* gcvel,
                                 const float* gsc, const float* gq_in, const float* gv_in, float* gq, float* gv) {
  if (nw > 0) k_dq_kin_vjp<<<(nw + KQ_WPB - 1) / KQ_WPB, 32 * KQ_WPB, 0, st>>>(nw, qpos, qvel, com, gqn, gxpos, gxquat,
                                                                              gcvel, gsc, gq_in, gv_in, gq, gv);
  return (int)cudaGetLastError();
}
extern "C" int dq_upload(const float* p, cudaStream_t st) {
@LOAD_CONSTS@
  return (int)cudaGetLastError();
}
extern "C" int dq_dynamics(cudaStream_t st, int nw, const float* qpos, const float* qvel, const float* com, float* M,
                           float* bias) {
  if (nw > 0) k_dq_dyn<<<(nw + KQ_WPB - 1) / KQ_WPB, 32 * KQ_WPB, 0, st>>>(nw, qpos, qvel, com, M, bias);
  return (int)cudaGetLastError();
}
extern "C" int dq_jacobian(cudaStream_t st, int nw, int np, const float* qpos, const int* bodies, const float* pts,
                           float* jacp, float* jacr) {
  if (nw > 0 && np > 0) k_dq_jac<<<(nw + KQ_WPB - 1) / KQ_WPB, 32 * KQ_WPB, 0, st>>>(nw, np, qpos, bodies, pts, jacp, jacr);
  return (int)cudaGetLastError();
}
extern "C" int dq_jacobian_vjp(cudaStream_t st, int nw, int np, const float* qpos, const int* bodies, const float* pts,
                               const float* gjacp, const float* gjacr, float* gq, float* gpts) {
  if (nw > 0) k_dq_jac_vjp<<<(nw + KQ_WPB - 1) / KQ_WPB, 32 * KQ_WPB, 0, st>>>(nw, np, qpos, bodies, pts, gjacp, gjacr,
                                                                              gq, gpts);
  return (int)cudaGetLastError();
}
extern "C" void dq_sizes(long long* o) {
  o[0] = NQ; o[1] = NV; o[2] = NB; o[3] = COMW; o[4] = (long long)sizeof(KQ); o[5] = (long long)sizeof(KK);
}
"""


def cuda_source(m, com_body=None, params=None, pw_off=3):
    """(the unit's text, packed constants): header, the device functions, the kernels and their C entries."""
    from nablaengine.forest_engine import host
    import re
    consts, lay, sizes, maxd = constants(m, com_body)
    load = host.loader(lay, sizes)
    load = re.sub(r"cudaMemcpyToSymbol\(([^;]*)\);", r"cudaMemcpyToSymbolAsync(\1, 0, cudaMemcpyHostToDevice, st);",
                  load)
    assert "cudaMemcpyToSymbol(" not in load
    src = header(m, maxd, params, pw_off) + device(params) + KERNELS.replace("@LOAD_CONSTS@", load)
    return src, consts

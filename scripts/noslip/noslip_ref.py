"""float64 NumPy reference of MuJoCo's noslip post-pass (MuJoCo 3.17 engine_solver.c solNoSlip and its use in
mj_fwdConstraint, reimplemented from the algorithm; no code copied), on MuJoCo C's own constraint data after the main
solve, with an optional restriction to chosen rows (the forest engine's noslip_pairs mode, which MuJoCo does not have)."""
import mujoco
import numpy as np

MINVAL = 1e-15


def dense_J(m, d):
    nv, ne = m.nv, d.nefc
    if mujoco.mj_isSparse(m):
        J = np.zeros((ne, nv))
        for i in range(ne):
            a, n = d.efc_J_rowadr[i], d.efc_J_rownnz[i]
            J[i, d.efc_J_colind[a:a + n]] = d.efc_J[a:a + n]
        return J
    return d.efc_J[:ne * nv].reshape(ne, nv).copy()


def full_M(m, d):
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    return M


def contact_pair(m, d, con):
    """The explicit pair index of a contact (MuJoCo: the pair's geoms), -1 for a contype contact."""
    g = {int(con.geom1), int(con.geom2)}
    for p in range(m.npair):
        if {int(m.pair_geom1[p]), int(m.pair_geom2[p])} == g:
            return p
    return -1


def noslip(m, d, iterations, tolerance, pairs=None, force=None):
    """(qacc, force, info) of the noslip pass from d's main solve (force: start forces, default d.efc_force)."""
    if int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        raise NotImplementedError("pyramidal cone only")
    nv, ne = m.nv, d.nefc
    J, M = dense_J(m, d), full_M(m, d)
    Minv_JT = np.linalg.solve(M, J.T)
    A = J @ Minv_JT
    R = np.asarray(d.efc_R[:ne], float)
    b = J @ d.qacc_smooth - d.efc_aref[:ne]
    f = np.array(d.efc_force[:ne] if force is None else force, float)
    typ = np.asarray(d.efc_type[:ne])
    T = mujoco.mjtConstraint
    floss = np.asarray(d.efc_frictionloss[:ne], float)
    if d.nisland > 0:
        isl = np.asarray(d.efc_island[:ne])
        nisl = d.nisland
        Mdiag = np.diag(M)
        scale = [1.0 / max(MINVAL, Mdiag[np.asarray(d.dof_island[:nv]) == k].sum()) for k in range(nisl)]
    else:
        isl, nisl, scale = np.zeros(ne, int), 1, [1.0 / max(MINVAL, np.trace(M))]
    blocks = [[] for _ in range(nisl)]
    i = 0
    while i < ne:
        if typ[i] in (T.mjCNSTR_FRICTION_DOF, T.mjCNSTR_FRICTION_TENDON):
            blocks[isl[i]].append(("f", i))
            i += 1
        elif typ[i] == T.mjCNSTR_CONTACT_PYRAMIDAL:
            con = d.contact[d.efc_id[i]]
            dim = int(con.dim)
            sel = pairs is None or contact_pair(m, d, con) in pairs
            for k in range(dim - 1):
                if sel:
                    blocks[isl[i]].append(("p", i + 2 * k))
            i += 2 * (dim - 1)
        else:
            i += 1
    niter = [0] * nisl
    done = [False] * nisl
    for k in range(nisl):
        rows = np.flatnonzero(isl == k)
        it = 0
        while it < iterations:
            imp = 0.0
            if it == 0:
                imp += float(0.5 * np.sum(f[rows] ** 2 * R[rows]))
            for kind, j in blocks[k]:
                if kind == "f":
                    res = b[j] + A[j] @ f
                    ainv = 1.0 / max(MINVAL, A[j, j])
                    old = f[j]
                    f[j] = min(max(f[j] - res * ainv, -floss[j]), floss[j])
                    dl = f[j] - old
                    imp -= 0.5 * dl * dl / ainv + dl * res
                    continue
                res = b[j:j + 2] + A[j:j + 2] @ f
                old = f[j:j + 2].copy()
                Ac = A[j:j + 2, j:j + 2].copy()
                Ac[0, 0] = max(1e-10, Ac[0, 0])
                Ac[1, 1] = max(1e-10, Ac[1, 1])
                bc = res - Ac @ old
                mid = 0.5 * (f[j] + f[j + 1])
                K1 = Ac[0, 0] + Ac[1, 1] - Ac[0, 1] - Ac[1, 0]
                K0 = mid * (Ac[0, 0] - Ac[1, 1]) + bc[0] - bc[1]
                if K1 < MINVAL:
                    f[j] = f[j + 1] = mid
                else:
                    y = -K0 / K1
                    if y < -mid:
                        f[j], f[j + 1] = 0.0, 2 * mid
                    elif y > mid:
                        f[j], f[j + 1] = 2 * mid, 0.0
                    else:
                        f[j], f[j + 1] = mid + y, mid - y
                dl = f[j:j + 2] - old
                chg = 0.5 * dl @ Ac @ dl + dl @ res
                if chg > 1e-10:
                    f[j:j + 2] = old
                    chg = 0.0
                imp -= chg
            imp *= scale[k]
            it += 1
            if imp < tolerance:
                done[k] = True
                break
        niter[k] = it
    qacc = d.qacc_smooth + Minv_JT @ f
    return qacc, f, dict(niter=niter, converged=done, nisland=nisl)


def check(m, d0, iterations, tolerance):
    """MuJoCo C's own noslip at d0's state against the reference on C's main solve: (|dqacc| / |qacc|, max |df|)."""
    d = mujoco.MjData(m)
    for k in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
        getattr(d, k)[:] = getattr(d0, k)
    ns, nt = m.opt.noslip_iterations, m.opt.noslip_tolerance
    m.opt.noslip_iterations = 0
    mujoco.mj_forward(m, d)
    qa, f, info = noslip(m, d, iterations, tolerance)
    m.opt.noslip_iterations, m.opt.noslip_tolerance = iterations, tolerance
    d2 = mujoco.MjData(m)
    for k in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
        getattr(d2, k)[:] = getattr(d0, k)
    mujoco.mj_forward(m, d2)
    m.opt.noslip_iterations, m.opt.noslip_tolerance = ns, nt
    return (float(np.linalg.norm(qa - d2.qacc) / max(np.linalg.norm(d2.qacc), 1e-12)),
            float(np.abs(f - d2.efc_force[:d2.nefc]).max()) if d2.nefc == d.nefc else None, info)


def step(m, d, iterations, tolerance, pairs=None):
    """One mj_step of m with the reference noslip (pairs: noslip()'s) in place of MuJoCo's: the main solve
    (noslip_iterations 0), the reference's qacc, then MuJoCo's integrator on it (implicitfast / Euler); d is advanced."""
    ns = m.opt.noslip_iterations
    m.opt.noslip_iterations = 0
    try:
        mujoco.mj_forward(m, d)
        qacc, f, info = noslip(m, d, iterations, tolerance, pairs)
        d.qacc[:] = qacc
        d.efc_force[:d.nefc] = f
        d.qfrc_constraint[:] = dense_J(m, d).T @ f
        d.qacc_warmstart[:] = qacc
        if int(m.opt.integrator) in (int(mujoco.mjtIntegrator.mjINT_IMPLICIT),
                                     int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)):
            mujoco.mj_implicit(m, d)
        else:
            mujoco.mj_Euler(m, d)
    finally:
        m.opt.noslip_iterations = ns
    return info

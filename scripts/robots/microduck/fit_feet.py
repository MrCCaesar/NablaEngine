"""Fit K spheres per foot to the convex hull of Microduck's sole mesh.

MuJoCo collides a mesh through its convex hull, and against a plane only the hull's
support function matters: for a floor whose upward normal is u (in the foot body's
frame), the deepest point of the foot along d = -u is h(d) = max_v v.d. A union of
spheres has h_s(d) = max_i (c_i.d + r_i). We fit centers c_i and radii r_i so that
h_s matches h over every foot tilt up to THETA_MAX, weighting the tilts seen in
walking (<= 30 deg) more.

Output: fit_results.json with, per foot and per K, the spheres in the ankle body's
frame and the fit errors.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
SCENE = HERE / "model" / "scene_walk.xml"
THETA_MAX = np.radians(60.0)
KS = (4, 6, 8)


def quat2mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def foot_hull(m, name):
    """Hull vertices of the foot mesh in the ankle body's frame + the sole's axes."""
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    mid = m.geom_dataid[g]
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = m.mesh_vert[a:a + n].astype(float)
    R = quat2mat(m.geom_quat[g])
    vb = v @ R.T + m.geom_pos[g]
    hv = vb[ConvexHull(vb).vertices]
    down = R @ np.array([-1.0, 0, 0])      # sole normal, out of the sole (measured: geom -x)
    along = R @ np.array([0, 0, 1.0])      # 54 mm axis
    across = R @ np.array([0, 1.0, 0])     # 41 mm axis
    return dict(geom=g, body=int(m.geom_bodyid[g]), hull=hv, down=down, along=along, across=across)


def directions(f, theta_max=THETA_MAX, nt=25, nphi=36):
    th = np.linspace(0, theta_max, nt)
    ph = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    D, T = [f["down"]], [0.0]
    for t in th[1:]:
        for p in ph:
            D.append(np.cos(t) * f["down"] + np.sin(t) * (np.cos(p) * f["along"] + np.sin(p) * f["across"]))
            T.append(t)
    return np.array(D), np.array(T)


def support_hull(hv, D):
    return (D @ hv.T).max(1)


def support_spheres(C, r, D, tau=None):
    s = D @ C.T + r[None, :]
    if tau is None:
        return s.max(1)
    mx = s.max(1, keepdims=True)                       # smooth max for the optimizer
    return (mx + tau * np.log(np.exp((s - mx) / tau).sum(1, keepdims=True)))[:, 0]


def init_layout(f, K, r0=0.010):
    """Spheres on the sole's footprint corners/edges, centers r0 inside the sole."""
    hv, dn, al, ac = f["hull"], f["down"], f["along"], f["across"]
    h = hv @ dn
    bot = hv[h > h.max() - 0.002]                      # near-bottom hull points
    pa, pc = bot @ al, bot @ ac
    a0, a1, c0, c1 = pa.min(), pa.max(), pc.min(), pc.max()
    ia = {4: [0, 1], 6: [0, 0.5, 1], 8: [0, 1 / 3, 2 / 3, 1]}[K]
    pts = [(a0 + s * (a1 - a0), c) for s in ia for c in (c0, c1)]
    C = []
    for pa_, pc_ in pts:
        base = h.max() * dn + pa_ * al + pc_ * ac      # point on the lowest plane
        C.append(base - r0 * dn)
    return np.array(C), np.full(K, r0)


def fit(f, K):
    D, T = directions(f)
    w = np.where(T <= np.radians(30), 1.0, 0.4)
    target = support_hull(f["hull"], D)
    C0, r0 = init_layout(f, K)
    x0 = np.concatenate([C0.ravel(), r0])
    lo = np.concatenate([np.full(3 * K, -np.inf), np.full(K, 0.002)])
    hi = np.concatenate([np.full(3 * K, np.inf), np.full(K, 0.040)])

    def unpack(x):
        return x[:3 * K].reshape(K, 3), x[3 * K:]

    x = x0
    for tau in (1e-3, 3e-4, 1e-4, 3e-5):               # anneal the smooth max
        res = least_squares(lambda x: w * (support_spheres(*unpack(x), D, tau) - target),
                            x, bounds=(lo, hi), x_scale=1e-3, max_nfev=4000)
        x = res.x
    C, r = unpack(x)
    err = support_spheres(C, r, D) - target            # > 0: spheres reach lower than the sole
    out = dict(K=K, centers=C.tolist(), radii=r.tolist())
    for lim in (10, 20, 30, 45, 60):
        s = T <= np.radians(lim) + 1e-9
        out[f"maxabs_mm_le{lim}"] = float(np.abs(err[s]).max() * 1e3)
        out[f"rms_mm_le{lim}"] = float(np.sqrt((err[s] ** 2).mean()) * 1e3)
    # flat stance: which spheres touch within 0.3 mm, and the area they span vs the hull's flat patch
    hs = (C @ f["down"]) + r
    touching = hs > hs.max() - 3e-4
    out["flat_touching"] = int(touching.sum())
    out["flat_lowest_err_mm"] = float((hs.max() - target[0]) * 1e3)

    def area(P):
        if len(P) < 3:
            return 0.0
        return float(ConvexHull(P).volume)
    P2 = np.stack([C @ f["along"], C @ f["across"]], 1)
    hv = f["hull"]; hh = hv @ f["down"]
    patch = hv[hh > hh.max() - 5e-4]
    out["flat_sphere_polygon_mm2"] = area(P2[touching]) * 1e6
    out["hull_patch_0p5mm_mm2"] = area(np.stack([patch @ f["along"], patch @ f["across"]], 1)) * 1e6
    out["footprint_mm"] = [float(np.ptp(hv @ f["along"]) * 1e3), float(np.ptp(hv @ f["across"]) * 1e3)]
    return out


def main():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    results = {}
    for side in ("left", "right"):
        f = foot_hull(m, f"{side}_foot_collision")
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, f["body"])
        results[side] = dict(body=body, down=f["down"].tolist(), along=f["along"].tolist(),
                             across=f["across"].tolist(), hull_vertices=len(f["hull"]), fits={})
        for K in KS:
            r = fit(f, K)
            results[side]["fits"][str(K)] = r
            print(f"{side:5s} K={K}: max|err| <=10deg {r['maxabs_mm_le10']:.2f} mm, <=30deg {r['maxabs_mm_le30']:.2f} mm, "
                  f"<=60deg {r['maxabs_mm_le60']:.2f} mm; rms<=30 {r['rms_mm_le30']:.2f} mm; flat: {r['flat_touching']} touching, "
                  f"polygon {r['flat_sphere_polygon_mm2']:.0f} mm2 (hull patch {r['hull_patch_0p5mm_mm2']:.0f}); "
                  f"radii mm {np.round(np.array(r['radii'])*1e3, 1).tolist()}")
    (HERE / "fit_results.json").write_text(json.dumps(results, indent=1))
    print("wrote", HERE / "fit_results.json")


if __name__ == "__main__":
    sys.exit(main())

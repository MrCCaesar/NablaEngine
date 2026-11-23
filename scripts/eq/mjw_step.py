"""MuJoCo Warp's step on a state corpus (run with a Python that has mujoco_warp; scripts/eq/eq_check.py calls it):"""
import argparse
import json

import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("states")
    ap.add_argument("out")
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--nconmax", type=int, default=64)
    ap.add_argument("--njmax", type=int, default=1024)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--prime", action="store_true")
    a = ap.parse_args()
    import warp as wp
    import mujoco_warp as mjw
    m = mujoco.MjModel.from_binary_path(a.model)
    if a.iterations:
        m.opt.iterations = a.iterations
    if a.dense:
        m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    z = np.load(a.states)
    B = len(z["qpos"])
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=a.nconmax, njmax=a.njmax)
    for k in ("qpos", "qvel", "ctrl"):
        wp.copy(getattr(dw, k), wp.array(np.ascontiguousarray(z[k], np.float32)))
    wp.copy(dw.qacc_warmstart, wp.zeros((B, m.nv), dtype=float))
    if a.prime:
        mjw.forward(mw, dw)
        wp.copy(dw.qacc_warmstart, wp.zeros((B, m.nv), dtype=float))
    rows = {}
    for s in range(a.substeps):
        mjw.step(mw, dw)
        if s == 0:
            ne = dw.ne.numpy()
            typ, eid = dw.efc.type.numpy(), dw.efc.id.numpy()
            eqt = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            out = np.full((B, max(m.neq, 1), 3), np.nan, np.float32)
            out3 = np.full((B, max(m.neq, 1), 3, 3), np.nan, np.float32)
            seen = np.zeros((B, max(m.neq, 1)), int)
            pos, D, aref = dw.efc.pos.numpy(), dw.efc.D.numpy(), dw.efc.aref.numpy()
            for w in range(B):
                for r in range(min(int(dw.nefc.numpy()[w]), a.njmax)):
                    if typ[w, r] == eqt:
                        e = eid[w, r]
                        out[w, e] = (pos[w, r], D[w, r], aref[w, r])
                        if seen[w, e] < 3:
                            out3[w, e, seen[w, e]] = (pos[w, r], D[w, r], aref[w, r])
                        seen[w, e] += 1
            rows = dict(eq_rows=out, eq_rows3=out3, ne=ne)
    nacon = int(dw.nacon.numpy()[0])
    ncon = np.bincount(dw.contact.worldid.numpy()[:nacon], minlength=B)
    np.savez(a.out, qpos=dw.qpos.numpy(), qvel=dw.qvel.numpy(), qacc=dw.qacc.numpy(), ncon=ncon, **rows)
    print(json.dumps(dict(mujoco=mujoco.__version__, mujoco_warp=getattr(mjw, "__version__", "?"), nworld=B,
                          contacts_mean=float(ncon.mean()), contacts_max=int(ncon.max()))))


if __name__ == "__main__":
    main()

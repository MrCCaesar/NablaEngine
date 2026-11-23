"""MuJoCo Warp's step with the activation state (run with a Python that has mujoco_warp; scripts/forest/muscle_check.py
calls it):"""
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
    a = ap.parse_args()
    import warp as wp
    import mujoco_warp as mjw
    m = mujoco.MjModel.from_binary_path(a.model)
    if a.iterations:
        m.opt.iterations = a.iterations
    z = np.load(a.states)
    B = len(z["qpos"])
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=a.nconmax, njmax=a.njmax)
    keys = ("qpos", "qvel", "ctrl") + (("act",) if m.na else ())
    for k in keys:
        wp.copy(getattr(dw, k), wp.array(np.ascontiguousarray(z[k], np.float32)))
    wp.copy(dw.qacc_warmstart, wp.zeros((B, m.nv), dtype=float))
    f1 = None
    for s in range(a.substeps):
        mjw.step(mw, dw)
        if s == 0:
            f1 = dw.actuator_force.numpy().copy()
    nacon = int(dw.nacon.numpy()[0])
    ncon = np.bincount(dw.contact.worldid.numpy()[:nacon], minlength=B)
    np.savez(a.out, qpos=dw.qpos.numpy(), qvel=dw.qvel.numpy(), act=dw.act.numpy() if m.na else np.zeros((B, 0)),
             force1=f1, ncon=ncon)
    print(json.dumps(dict(mujoco=mujoco.__version__, mujoco_warp=getattr(mjw, "__version__", "?"), nworld=B,
                          contacts_mean=float(ncon.mean()), contacts_max=int(ncon.max()))))


if __name__ == "__main__":
    main()

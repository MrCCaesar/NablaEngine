"""MuJoCo Warp's tendon lengths and Jacobians on a state corpus (run with a Python that has mujoco_warp 3.17; called by
scripts/forest/spatial_check.py):"""
import json
import sys

import mujoco
import numpy as np


def main():
    model, states, out = sys.argv[1:4]
    import warp as wp
    import mujoco_warp as mjw
    m = mujoco.MjModel.from_binary_path(model)
    z = np.load(states)
    B = len(z["qpos"])
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=B, nconmax=64, njmax=256)
    wp.copy(dw.qpos, wp.array(np.ascontiguousarray(z["qpos"], np.float32)))
    wp.copy(dw.qvel, wp.array(np.ascontiguousarray(z["qvel"], np.float32)))
    mjw.forward(mw, dw)
    L = dw.ten_length.numpy()
    Js = dw.ten_J.numpy()
    nnz, adr, col = (getattr(mw, k).numpy() for k in ("ten_J_rownnz", "ten_J_rowadr", "ten_J_colind"))
    J = np.zeros((B, m.ntendon, m.nv), np.float32)
    for t in range(m.ntendon):
        for k in range(int(nnz[t])):
            J[:, t, int(col[adr[t] + k])] = Js[:, adr[t] + k]
    np.savez(out, ten_length=L, ten_J=J, ten_velocity=dw.ten_velocity.numpy())
    print(json.dumps(dict(mujoco=mujoco.__version__, mujoco_warp=getattr(mjw, "__version__", "?"), nworld=B)))


if __name__ == "__main__":
    main()

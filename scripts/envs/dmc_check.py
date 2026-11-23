"""Agreement of nablaengine.envs' DMC tasks with dm_control (MuJoCo C, float64), per task:"""
import argparse
import json
import sys

import numpy as np


def dm_env(name, seed):
    from dm_control import suite
    domain, task = name.split("-", 1)
    return suite.load(domain, task, task_kwargs={"random": seed})


def flat(obs):
    return np.concatenate([np.asarray(v, np.float64).ravel() for v in obs.values()])


def aux_of(name, physics):
    """The engine task's aux for dm_control's current episode."""
    if name.startswith("reacher"):
        return {"target": np.array(physics.named.model.geom_pos["target", :2], np.float32)}
    if name.startswith("finger"):
        out = {"touch": np.array(physics.named.data.sensordata[["touchtop", "touchbottom"]], np.float32)}
        if "turn" in name:
            out["target"] = np.array(physics.named.model.site_pos["target", ["x", "z"]], np.float32)
        return out
    if name.startswith("hopper"):
        return {"touch": np.array(physics.named.data.sensordata[["touch_toe", "touch_heel"]], np.float32)}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--backend", default="jax")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--refine", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from nablaengine import envs
    task = envs.make(a.task, backend=a.backend, refine=a.refine)
    o = task.o
    as_np = (lambda x: np.asarray(x)) if a.backend == "jax" else (lambda x: x.detach().cpu().numpy())

    def batch(rows, dtype=np.float32):
        return o.const(np.asarray(rows, dtype))

    def state(qpos, qvel, aux, t=None):
        from nablaengine.envs.env import TaskState
        B = len(qpos)
        aux = {k: batch(np.stack([x[k] for x in aux])) for k in (aux[0] if aux else {})}
        return TaskState(batch(qpos), batch(qvel), o.zeros((B, task.nv)), o.zeros((B,), dtype=np.int32), aux)

    rng = np.random.default_rng(0)
    lo, hi = task.action_low, task.action_high
    rec = []
    for ep in range(a.episodes):
        env = dm_env(a.task, ep)
        ts = env.reset()
        p = env.physics
        for t in range(a.steps):
            q0, v0 = p.data.qpos.copy(), p.data.qvel.copy()
            u = rng.uniform(lo, hi)
            ts = env.step(u)
            rec.append((q0, v0, u, p.data.qpos.copy(), p.data.qvel.copy(), flat(ts.observation), ts.reward,
                        aux_of(a.task, p)))
    q0, v0, u, q1, v1, ob, rw, ax = zip(*rec)
    s0, s1 = state(q0, v0, ax), state(q1, v1, ax)
    ours_ob = as_np(task.obs(s1)).astype(np.float64)
    ours_rw = as_np(task.reward(s1, batch(u), s0)[0]).astype(np.float64)
    ob, rw = np.array(ob), np.array(rw, np.float64)
    keys = task.obs_keys
    widths = [int(np.prod(s)) for _, s in keys]
    per_key, c = {}, 0
    for (k, _), wdt in zip(keys, widths):
        per_key[k] = float(np.abs(ours_ob[:, c:c + wdt] - ob[:, c:c + wdt]).max())
        c += wdt
    out = {"task": a.task, "backend": a.backend, "refine": task.refine, "notes": task.notes, "obs_dim": task.obs_dim,
           "obs_keys": [[k, list(s)] for k, s in keys], "dm_obs_dim": int(ob.shape[1]),
           "matched": {"transitions": len(rec), "obs_max_abs": per_key,
                       "reward_max_abs": float(np.abs(ours_rw - rw).max()),
                       "reward_mean_abs": float(np.abs(ours_rw - rw).mean())}}

    E = a.episodes
    envs_dm = [dm_env(a.task, 100 + ep) for ep in range(E)]
    for e in envs_dm:
        e.reset()
    st = state([e.physics.data.qpos.copy() for e in envs_dm], [e.physics.data.qvel.copy() for e in envs_dm],
               [aux_of(a.task, e.physics) for e in envs_dm])
    st = st._replace(warm=batch([e.physics.data.qacc_warmstart.copy() for e in envs_dm]))
    envs_pt = [dm_env(a.task, 100 + ep) for ep in range(E)]
    for e in envs_pt:
        e.reset()
        e.physics.data.qpos[:] += 1e-6 * np.random.default_rng(5).standard_normal(task.nq)
        e.physics.forward()
    err_q, err_v, dr, dob, err_pt, R_dm, R_ours = [], [], [], [], [], np.zeros(E), np.zeros(E)
    for t in range(a.steps):
        U = rng.uniform(lo, hi, (E, task.nu))
        tss = [e.step(U[i]) for i, e in enumerate(envs_dm)]
        for i, e in enumerate(envs_pt):
            e.step(U[i])
        err_pt.append(np.abs(np.array([e.physics.data.qpos for e in envs_pt])
                             - np.array([e.physics.data.qpos for e in envs_dm])).max(1))
        r_dm = np.array([x.reward for x in tss], np.float64)
        st2 = task.step(st, batch(U))
        r_ours = as_np(task.reward(st2, batch(U), st)[0]).astype(np.float64)
        st = st2
        qd = np.array([e.physics.data.qpos for e in envs_dm])
        vd = np.array([e.physics.data.qvel for e in envs_dm])
        err_q.append(np.abs(as_np(st.qpos) - qd).max(1))
        err_v.append(np.abs(as_np(st.qvel) - vd).max(1) / (1 + np.abs(vd).max(1)))
        dr.append(np.abs(r_ours - r_dm))
        dob.append(np.abs(as_np(task.obs(st)) - np.array([flat(x.observation) for x in tss])).max(1))
        R_dm += r_dm
        R_ours += r_ours
    err_q, err_v, dr = np.array(err_q), np.array(err_v), np.array(dr)
    mark = [t for t in (0, 9, 24, 49, 99, 199, 499, 999) if t < a.steps]
    out["openloop"] = {"episodes": E, "steps": a.steps,
                       "qpos_max_abs_median_over_episodes": {str(t + 1): float(np.median(err_q[t])) for t in mark},
                       "qvel_rel_median_over_episodes": {str(t + 1): float(np.median(err_v[t])) for t in mark},
                       "obs_max_abs_median_over_episodes": {str(t + 1): float(np.median(dob[t])) for t in mark},
                       "baseline_dm_perturbed_1e-6_qpos_max_abs_median": {str(t + 1): float(np.median(err_pt[t]))
                                                                          for t in mark},
                       "reward_abs_diff_mean": float(dr.mean()), "reward_abs_diff_max": float(dr.max()),
                       "return_dm": R_dm.tolist(), "return_engine": R_ours.tolist()}

    N = 1024
    dmq, dmv = [], []
    for i in range(N):
        e = dm_env(a.task, 1000 + i)
        e.reset()
        dmq.append(e.physics.data.qpos.copy())
        dmv.append(e.physics.data.qvel.copy())
    key = task._rand_probe()
    s = task.reset(key, 4096)
    oq, ov = as_np(s.qpos), as_np(s.qvel)
    dmq, dmv = np.array(dmq), np.array(dmv)
    out["reset"] = {"dm_resets": N, "engine_resets": 4096,
                    "qpos_mean": [dmq.mean(0).round(4).tolist(), oq.mean(0).round(4).tolist()],
                    "qpos_std": [dmq.std(0).round(4).tolist(), oq.std(0).round(4).tolist()],
                    "qvel_std": [dmv.std(0).round(4).tolist(), ov.std(0).round(4).tolist()]}
    txt = json.dumps(out, indent=1)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

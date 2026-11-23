"""Agreement of nablaengine.envs' Brax tasks with brax.envs (backend "mjx", float32), per task:"""
import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--refine", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from brax import envs as benvs
    from nablaengine import envs
    from nablaengine.envs.env import TaskState

    name = {"brax-halfcheetah": "halfcheetah", "brax-hopper": "hopper", "brax-walker2d": "walker2d",
            "brax-ant": "ant"}[a.task]
    benv = benvs.get_environment(name, backend="mjx")
    reset, step = jax.jit(jax.vmap(benv.reset)), jax.jit(jax.vmap(benv.step))
    task = envs.make(a.task, refine=a.refine)
    E, T = a.episodes, a.steps
    key = jax.random.PRNGKey(0)
    acts = jax.random.uniform(jax.random.PRNGKey(1), (T, E, task.nu), minval=-1, maxval=1)

    def ours(ps):
        return TaskState(ps.q, ps.qd, jnp.zeros_like(ps.qd), jnp.zeros(ps.q.shape[0], jnp.int32),
                         {"x": ps.x.pos[:, 0]})

    bs = reset(jax.random.split(key, E))
    st = ours(bs.pipeline_state)
    st = st._replace(warm=bs.pipeline_state.qacc_warmstart)
    m_ob, m_rw, m_dn, eq, ev, dr = [], [], [], [], [], []
    alive = np.ones(E, bool)
    for t in range(T):
        s_prev = ours(bs.pipeline_state)
        bs = step(bs, acts[t])
        s_now = ours(bs.pipeline_state)
        ob = np.asarray(task.obs(s_now))
        rw, _ = task.reward(s_now, acts[t], s_prev)
        dn = np.asarray(task.terminated(s_now))
        m_ob.append(np.abs(ob - np.asarray(bs.obs)).max(1))
        m_rw.append(np.abs(np.asarray(rw) - np.asarray(bs.reward)))
        m_dn.append(dn == (np.asarray(bs.done) > 0.5))
        st2 = task.step(st, acts[t])
        r2, _ = task.reward(st2, acts[t], st)
        st = st2
        live = alive.copy()
        eq.append(np.where(live, np.abs(np.asarray(st.qpos) - np.asarray(bs.pipeline_state.q)).max(1), np.nan))
        vd = np.asarray(bs.pipeline_state.qd)
        ev.append(np.where(live, np.abs(np.asarray(st.qvel) - vd).max(1) / (1 + np.abs(vd).max(1)), np.nan))
        dr.append(np.where(live, np.abs(np.asarray(r2) - np.asarray(bs.reward)), np.nan))
        alive &= np.asarray(bs.done) < 0.5
    eq, ev, dr = np.array(eq), np.array(ev), np.array(dr)
    mark = [t for t in (0, 9, 24, 49, 99, 199, 499, 999) if t < T]
    med = lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else None
    out = {"task": a.task, "refine": task.refine, "notes": task.notes, "obs_dim": task.obs_dim,
           "brax_obs_dim": int(bs.obs.shape[1]), "control_dt": task.control_dt,
           "matched": {"transitions": E * T, "obs_max_abs": float(np.max(m_ob)),
                       "reward_max_abs": float(np.max(m_rw)), "reward_mean_abs": float(np.mean(m_rw)),
                       "termination_agree": float(np.mean(m_dn))},
           "openloop": {"episodes": E, "steps": T, "alive_at_end": int(alive.sum()),
                        "qpos_max_abs_median": {str(t + 1): med(eq[t]) for t in mark},
                        "qvel_rel_median": {str(t + 1): med(ev[t]) for t in mark},
                        "reward_abs_diff_mean": float(np.nanmean(dr)), "reward_abs_diff_max": float(np.nanmax(dr))}}
    N = 4096
    b0 = reset(jax.random.split(jax.random.PRNGKey(7), N)).pipeline_state
    s0 = task.reset(jax.random.PRNGKey(8), N)
    out["reset"] = {"n": N,
                    "qpos_mean": [np.asarray(b0.q).mean(0).round(4).tolist(), np.asarray(s0.qpos).mean(0).round(4).tolist()],
                    "qpos_std": [np.asarray(b0.q).std(0).round(4).tolist(), np.asarray(s0.qpos).std(0).round(4).tolist()],
                    "qvel_std": [np.asarray(b0.qd).std(0).round(4).tolist(), np.asarray(s0.qvel).std(0).round(4).tolist()],
                    "x_max_abs": float(np.abs(np.asarray(b0.x.pos[:, 0]).mean(0) - np.asarray(s0.aux["x"]).mean(0)).max())}
    txt = json.dumps(out, indent=1)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

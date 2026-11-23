"""Brax's locomotion tasks (brax.envs v2, Copyright 2021 The Brax Authors, Apache License 2.0) on the engine, as
brax.envs.get_environment(name, backend="mjx") defines them at its defaults: the same MJCF assets (loaded from an
installed brax) and MJX options, observation (exclude_current_positions_from_observation=True; ant without contact
forces), reward, termination, reset noise and action repeat (n_frames)."""
import importlib.resources
import math

import mujoco
import numpy as np

from nablaengine.envs.dmc import Spec


class BraxSpec(Spec):
    asset = None
    n_frames = None
    max_reward = None
    lagged = True
    reset_noise = None
    qvel_noise = "uniform"
    mjx_options = {}

    @property
    def name(self):
        return self.task

    def source(self):
        f = importlib.resources.files("brax") / "envs" / "assets" / f"{self.asset}.xml"
        return f.read_text(), {}

    @property
    def control_dt(self):
        return self.n_frames * self._timestep

    @property
    def time_limit(self):
        return 1000 * self.control_dt

    def __init__(self):
        m = mujoco.MjModel.from_xml_string(self.source()[0])
        self._timestep = float(m.opt.timestep)

    def adjust_model(self, m):
        for k, v in self.mjx_options.items():
            if k == "disableflags":
                m.opt.disableflags |= int(v)
            else:
                setattr(m.opt, k, v)

    def notes(self, task):
        n = []
        if self.mjx_options.get("iterations") is not None:
            n.append(f"MJX options {self.mjx_options} (ls_iterations 4: the engine's line search is exact)")
        return n

    @staticmethod
    def _init_q(m):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "init_qpos")
        if i < 0:
            return np.array(m.qpos0)
        a, n = int(m.numeric_adr[i]), int(m.numeric_size[i])
        assert n == m.nq, f"init_qpos has {n} values, nq {m.nq}"
        return np.array(m.numeric_data[a:a + n])

    def _torso(self, env, qpos):
        return env.kin.fk(env.o, qpos).xpos[:, 1]

    def reset(self, env, rnd, B):
        o, m, s = env.o, env.original, self.reset_noise
        init = np.asarray(self._init_q(m), np.float32)
        q = o.const(init)[None] + rnd.uniform((B, m.nq), -s, s)
        v = s * rnd.normal((B, m.nv)) if self.qvel_noise == "normal" else rnd.uniform((B, m.nv), -s, s)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                a = int(m.jnt_qposadr[j])
                quat = q[:, a + 3:a + 7] / o.norm(q[:, a + 3:a + 7])[:, None]
                q = o.concat([q[:, :a + 3], quat, q[:, a + 7:]], -1)
        return q, v, {"x": self._torso(env, q)}

    def after_step(self, env, s2, q_prev, con=None):
        return {**s2.aux, "x": self._torso(env, q_prev)}

    def _xvel(self, env, s2, s):
        return (s2.aux["x"] - s.aux["x"]) / env.control_dt

    def _ctrl_cost(self, env, action, w):
        return w * env.o.sum(action * action, -1)


class Halfcheetah(BraxSpec):
    task, asset, n_frames, reset_noise, qvel_noise = "brax-halfcheetah", "half_cheetah", 5, 0.1, "normal"

    def obs(self, env, s, F):
        return {"obs": env.o.concat([s.qpos[:, 1:], s.qvel], -1)}

    def reward(self, env, s2, action, s, F):
        xv = self._xvel(env, s2, s)[:, 0]
        ctrl = self._ctrl_cost(env, action, 0.1)
        return xv - ctrl, {"reward_run": xv, "reward_ctrl": -ctrl, "x_velocity": xv}


class _Planar(BraxSpec):
    """hopper / walker2d: the observed height is the torso's z, the velocities clipped to +-10."""
    reset_noise, qvel_noise = 5e-3, "uniform"
    healthy_z = healthy_angle = None
    healthy_state = None

    def obs(self, env, s, F):
        o = env.o
        pos = o.concat([s.aux["x"][:, 2:3], s.qpos[:, 2:]], -1)
        return {"obs": o.concat([pos, o.clip(s.qvel, -10.0, 10.0)], -1)}

    def healthy(self, env, s2):
        o = env.o
        z, angle = s2.aux["x"][:, 2], s2.qpos[:, 2]
        ok = (z > self.healthy_z[0]) & (z < self.healthy_z[1])
        ok = ok & (angle > self.healthy_angle[0]) & (angle < self.healthy_angle[1])
        if self.healthy_state is not None:
            sv = o.concat([s2.qpos[:, 2:], s2.qvel], -1)
            lo, hi = self.healthy_state
            ok = ok & o.logical_and(o.min(sv, -1) > lo, o.max(sv, -1) < hi)
        return ok

    def reward(self, env, s2, action, s, F):
        xv = self._xvel(env, s2, s)[:, 0]
        ctrl = self._ctrl_cost(env, action, 1e-3)
        healthy = env.o.ones_like(xv)
        return xv + healthy - ctrl, {"reward_forward": xv, "reward_ctrl": -ctrl, "reward_healthy": healthy,
                                     "x_velocity": xv}

    def terminated(self, env, s2, F):
        return ~self.healthy(env, s2)


class Hopper(_Planar):
    task, asset, n_frames = "brax-hopper", "hopper", 4
    healthy_z, healthy_angle, healthy_state = (0.7, math.inf), (-0.2, 0.2), (-100.0, 100.0)


class Walker2d(_Planar):
    task, asset, n_frames = "brax-walker2d", "walker2d", 4
    healthy_z, healthy_angle = (0.8, 2.0), (-1.0, 1.0)


class Ant(BraxSpec):
    task, asset, n_frames, reset_noise, qvel_noise = "brax-ant", "ant", 5, 0.1, "normal"
    mjx_options = {"solver": mujoco.mjtSolver.mjSOL_NEWTON, "disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                   "iterations": 1, "ls_iterations": 4}

    def obs(self, env, s, F):
        return {"obs": env.o.concat([s.qpos[:, 2:], s.qvel], -1)}

    def reward(self, env, s2, action, s, F):
        v = self._xvel(env, s2, s)
        ctrl = self._ctrl_cost(env, action, 0.5)
        healthy = env.o.ones_like(ctrl)
        return v[:, 0] + healthy - ctrl, {"reward_forward": v[:, 0], "reward_survive": healthy,
                                          "reward_ctrl": -ctrl, "x_velocity": v[:, 0], "y_velocity": v[:, 1]}

    def terminated(self, env, s2, F):
        z = s2.aux["x"][:, 2]
        return (z < 0.2) | (z > 1.0)


TASKS = {"brax-halfcheetah": Halfcheetah, "brax-hopper": Hopper, "brax-walker2d": Walker2d, "brax-ant": Ant}

"""Batched, differentiable task environments on the engine (nablaengine.envs): Task wraps a task definition (envs.dmc,
envs.brax) around the engine's step (jax_ffi or torch_ffi) and the original model's kinematics (envs.kin)."""
import math
import os as _os
_REUSE_FRAMES = _os.environ.get("DK_ENVS_REUSE_FRAMES", "1") != "0"
from typing import NamedTuple

import mujoco
import numpy as np

from nablaengine.envs import convert, kin, ops as ops_mod


MAX_SUBSTEPS = 32


class _Fields(NamedTuple):
    qpos: object
    qvel: object
    warm: object
    t: object
    aux: dict


class TaskState(_Fields):
    """The five fields (a NamedTuple: a JAX pytree, unpacks as before). An instance also has a __dict__ where step and
    reset keep the frames of exactly this object (_frames, see Task.frames); a new object (_replace, TaskState(...),
    pytree unflatten) never inherits them, copies and pickles drop them."""

    def __getstate__(self):
        return None


class JaxRand:
    def __init__(self, key):
        import jax
        self.jax, self.key = jax, key

    def _k(self):
        self.key, k = self.jax.random.split(self.key)
        return k

    def uniform(self, shape, lo=0.0, hi=1.0):
        return self.jax.random.uniform(self._k(), shape, minval=lo, maxval=hi)

    def normal(self, shape):
        return self.jax.random.normal(self._k(), shape)


class TorchRand:
    def __init__(self, gen, device):
        import torch
        self.torch, self.gen, self.device = torch, gen, device

    def uniform(self, shape, lo=0.0, hi=1.0):
        return lo + (hi - lo) * self.torch.rand(shape, generator=self.gen, device=self.device)

    def normal(self, shape):
        return self.torch.randn(shape, generator=self.gen, device=self.device)


class Task:
    """A task (spec: an envs.dmc / envs.brax definition) on the engine."""

    def __init__(self, spec, backend="jax", integrator=None, refine=None, iterations=None, reward=None, device=None):
        self.spec, self.backend = spec, backend
        xml, assets = spec.source()
        conv = convert.to_engine(xml, assets, **spec.convert_options())
        m = conv.model
        spec.adjust_model(m)
        if integrator is not None:
            I = mujoco.mjtIntegrator
            m.opt.integrator = {"euler": I.mjINT_EULER, "implicitfast": I.mjINT_IMPLICITFAST}[integrator]
        self.physics_dt = float(conv.original.opt.timestep)
        self.n_sub = int(round(spec.control_dt / self.physics_dt))
        assert abs(self.n_sub * self.physics_dt - spec.control_dt) < 1e-9, (spec.control_dt, self.physics_dt)
        self.refine = int(refine if refine is not None else spec.refine)
        m.opt.timestep = self.physics_dt / self.refine
        self.model, self.original, self.conversion = m, conv.original, conv
        self.notes = list(conv.notes) + spec.notes(self)
        self.iterations = int(iterations if iterations is not None else spec.iterations(m))
        self.kin = kin.Kinematics(conv.original)
        self.o = ops_mod.JaxOps() if backend == "jax" else ops_mod.TorchOps(device)
        o = self.o
        lo, hi = np.array(m.actuator_ctrlrange, np.float32).T
        lim = m.actuator_ctrllimited.astype(bool)
        lo, hi = np.where(lim, lo, -np.inf).astype(np.float32), np.where(lim, hi, np.inf).astype(np.float32)
        self.action_low, self.action_high = lo, hi
        self._lo, self._hi = o.const(lo), o.const(hi)
        self.nu, self.nq, self.nv = m.nu, m.nq, m.nv
        self.control_dt = spec.control_dt
        self.time_limit = int(round(spec.time_limit / spec.control_dt))
        self.max_reward = spec.max_reward
        self._reward = reward
        from nablaengine import engines
        engines.prefer_forest(m)
        if backend == "jax":
            from nablaengine import jax_ffi
            self.info = jax_ffi.init(model=m, foot_pairs=[])
            self._ffi = jax_ffi
        else:
            from nablaengine import torch_ffi
            self.info = torch_ffi.init(device, model=m, foot_pairs=[])
            self._ffi = torch_ffi
        self._com_words = int(self.info.get("com_words", 3)) if isinstance(self.info, dict) else 3
        probe = self.reset(self._rand_probe(), 2)
        self.obs_keys = [(k, tuple(v.shape[1:])) for k, v in self.obs_dict(probe).items()]
        self.obs_dim = int(sum(math.prod(s) for _, s in self.obs_keys))
        self.critic_obs_dim = self.obs_dim

    def _rand_probe(self):
        if self.backend == "jax":
            import jax
            return jax.random.PRNGKey(0)
        import torch
        return torch.Generator(device=self.o.device).manual_seed(0)

    def rand(self, key):
        return JaxRand(key) if self.backend == "jax" else TorchRand(key, self.o.device)

    def clamp(self, action):
        """The applied action: action clamped to the ctrlrange (gradient zero outside, as MJX)."""
        return self.o.clip(action, self._lo, self._hi)

    def reset(self, key, B):
        """B initial states with the original's randomization (key: a jax PRNGKey / a torch.Generator)."""
        out = self.spec.reset(self, self.rand(key), B)
        qpos, qvel, aux = out[:3]
        warm = out[3] if len(out) > 3 else self.o.zeros((B, self.nv))
        return self._attach(TaskState(qpos, qvel, warm, self.o.zeros((B,), dtype=np.int32), aux))

    def physics(self, qpos, qvel, warm, ctrl, substeps, contacts=False):
        """The engine's step: substeps of the engine timestep from (qpos, qvel, warm) under ctrl (B, nu); contacts:
        also the last substep's contacts (a dict of (B, maxcon) arrays, jax_ffi.contact_table; not differentiated)."""
        o = self.o
        B = qpos.shape[0]
        fr = o.full((B,), 1.0)
        com = o.zeros((B, self._com_words))
        left, con = int(substeps), None
        while left:
            k = min(left, MAX_SUBSTEPS)
            last = k == left
            if contacts and last and self.backend == "jax":
                qpos, qvel, warm, ctab = self._ffi.step(qpos, qvel, warm, ctrl, fr, com, iterations=self.iterations,
                                                        substeps=k, contacts=True)
                con = {key: self.o.jax.lax.stop_gradient(v) for key, v in self._ffi.contact_table(ctab).items()}
            else:
                qpos, qvel, warm = self._ffi.step(qpos, qvel, warm, ctrl, fr, com, iterations=self.iterations,
                                                  substeps=k)
                if contacts and last:
                    con = {key: v.detach().clone() for key, v in self._ffi.last_contacts(self.o.device).items()}
            left -= k
        return (qpos, qvel, warm, con) if contacts else (qpos, qvel, warm)

    def step(self, state, action):
        """The next state: action clamped to the ctrlrange, the original's control step (n_sub physics steps)."""
        u = self.clamp(action)
        sp = self.spec
        if not (sp.lagged or sp.contacts):
            q, v, w = self.physics(state.qpos, state.qvel, state.warm, u, self.n_sub * self.refine)
            return self._attach(TaskState(q, v, w, state.t + 1, state.aux))
        q, v, w, q_prev = state.qpos, state.qvel, state.warm, None
        n = self.n_sub * self.refine
        if sp.lagged:
            if self.n_sub > 1:
                q, v, w = self.physics(q, v, w, u, (self.n_sub - 1) * self.refine)
            q_prev, n = q, self.refine
        out = self.physics(q, v, w, u, n, contacts=sp.contacts)
        s2 = self._attach(TaskState(*out[:3], state.t + 1, state.aux))
        return self._attach(s2._replace(aux=sp.after_step(self, s2, q_prev, out[3] if sp.contacts else None)), s2)

    def _key(self, state):
        if self.backend == "jax":
            return ()
        q, v = state.qpos, state.qvel
        if q.is_inference() or v.is_inference():
            return None
        return q._version, v._version, q.requires_grad, v.requires_grad

    def _attach(self, state, like=None):
        c = None if like is None else like.__dict__.get("_frames")
        if c is None or c[0] is not self or like.qpos is not state.qpos or like.qvel is not state.qvel:
            c = (self, self._key(state), self.kin.fk(self.o, state.qpos, state.qvel))
        state.__dict__["_frames"] = c
        return state

    def frames(self, state):
        """envs.kin.Frames of state (qpos, qvel): the ones step / reset attached to this state object if they still
        match it, else computed (the same arrays either way). The attached frames are one autograd graph shared by every
        reader of the state (torch): two separate backward passes that both stop at the state need retain_graph=True
        for the first; $DK_ENVS_REUSE_FRAMES=0 recomputes per call (the behaviour before 2026-11-12)."""
        d = getattr(state, "__dict__", None) if _REUSE_FRAMES else None
        c = d.get("_frames") if d else None
        if c is not None and c[0] is self and c[1] is not None and c[1] == self._key(state):
            F = c[2]
            if self.backend == "torch" and F.xpos.requires_grad and not self.o.torch.is_grad_enabled():
                F = kin.Frames(F.kin, F.o, F.xpos.detach(), F.xmat.detach(), F.w.detach(), F.v0.detach())
            return F
        return self.kin.fk(self.o, state.qpos, state.qvel)

    def obs_dict(self, state):
        return self.spec.obs(self, state, self.frames(state))

    def obs(self, state):
        """The original's observation vector (DMC: the observation dict flattened in its order)."""
        d = self.obs_dict(state)
        B = state.qpos.shape[0]
        return self.o.concat([v.reshape(B, -1) for v in d.values()], -1)

    def critic_obs(self, state):
        return self.obs(state)

    def reward(self, state2, action, state):
        """(r (B,), terms): the original reward of the transition state -> state2 under action (as passed, before the
        clamp, as the originals read the control), or the reward given to make()."""
        if self._reward is not None:
            return self._reward(self, state2, action, state)
        return self.spec.reward(self, state2, action, state, self.frames(state2))

    def terminated(self, state2):
        return self.spec.terminated(self, state2, self.frames(state2))

    def truncated(self, state2):
        return state2.t >= self.time_limit

    def where(self, mask, a, b):
        """Per world: a where mask (B,) else b (states of the same task)."""
        o = self.o

        def sel(x, y):
            mk = mask.reshape(mask.shape + (1,) * (len(x.shape) - 1))
            return o.where(mk, x, y)
        return TaskState(*(sel(x, y) for x, y in zip(a[:4], b[:4])), {k: sel(a.aux[k], b.aux[k]) for k in a.aux})

"""DeepMind Control Suite tasks (dm_control.suite, Copyright 2017 The dm_control Authors, Apache License 2.0) on the
engine: each task's observation, reward and initial-state randomization ported from dm_control's task code, on the
suite's own MJCF models (loaded from an installed dm_control: dm_control.suite.<domain>.get_model_and_assets)."""
import importlib
import math

import numpy as np

from nablaengine.envs.rewards import tolerance


class Spec:
    domain = task = None
    control_dt = None
    time_limit = None
    max_reward = 1.0
    refine = 1
    lagged = False
    contacts = False

    def after_step(self, env, s2, q_prev, con):
        return s2.aux

    @property
    def name(self):
        return f"{self.domain}-{self.task}"

    def source(self):
        mod = importlib.import_module(f"dm_control.suite.{self.domain}")
        return mod.get_model_and_assets()

    def adjust_model(self, m):
        pass

    def notes(self, task):
        return []

    def convert_options(self):
        return {}

    def iterations(self, m):
        return int(m.opt.iterations)

    def terminated(self, env, s, F):
        return env.o.zeros((s.qpos.shape[0],), dtype=np.bool_)

    def reset(self, env, rnd, B):
        raise NotImplementedError


def penetrating(env, q):
    """Whether pose q (B, nq) has a contact: the engine's collision in a one-substep call at rest (its contact table
    is the input state's), as dm_control's ncon > 0 after mj_forward."""
    o = env.o
    B = q.shape[0]
    z = o.zeros((B, env.nv))
    con = env.physics(q, z, z, o.zeros((B, env.nu)), 1, contacts=True)[3]
    hit = con["valid"] & (con["dist"] < 0)
    return o.sum(o.where(hit, o.ones_like(con["dist"]), o.zeros_like(con["dist"])), -1) > 0


def resample_until_free(env, rnd, B, draw, rounds):
    """dm_control's loop: draw poses until they have no contact; here a fixed number of rounds, each redrawing the
    worlds still in contact."""
    q = draw()
    for _ in range(rounds):
        q = env.o.where(penetrating(env, q)[:, None], draw(), q)
    return q


def _pair_bodies(env):
    if getattr(env, "_pair_bodies", None) is None:
        import mujoco
        from nablaengine.forest_engine.step import pair_order
        m, m0 = env.model, env.original
        first, second = [], []
        for p in range(m.npair):
            g1, g2 = pair_order(m, int(m.pair_geom1[p]), int(m.pair_geom2[p]))
            b = [int(m0.geom_bodyid[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, m.geom(g).name)]) for g in (g1, g2)]
            first.append(b[0])
            second.append(b[1])
        env._pair_bodies = (np.array(first + [-1], np.int32), np.array(second + [-1], np.int32))
    return env._pair_bodies


def touch(env, s2, con, sites):
    """MuJoCo's touch sensors at the named sites (sphere or ellipsoid zones), (B, len(sites)): the normal forces of
    the contacts of the site's body whose force ray (from the contact point along the normal force on that body)
    meets the zone (mju_rayGeom)."""
    import mujoco
    o, m0 = env.o, env.original
    first, second = _pair_bodies(env)
    F = env.frames(s2)
    pair = con["pair"]
    idx = o.where(con["valid"], pair, o.zeros_like(pair) - 1)
    b1, b2 = o.const(first, np.int32)[idx], o.const(second, np.int32)[idx]
    fn, pos, nrm = con["force_normal"], con["pos"], con["normal"]
    out = []
    for name in sites:
        i = env.kin.site(name)
        assert int(m0.site_type[i]) in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID))
        sz = np.array(m0.site_size[i], np.float32)
        if int(m0.site_type[i]) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            sz = np.full(3, sz[0], np.float32)
        sb = int(m0.site_bodyid[i])
        on1, on2 = b1 == sb, b2 == sb
        ray = o.where(on2[..., None], -nrm, nrm)
        R = F.site_xmat(i)
        pl = o.einsum("bji,bcj->bci", R, pos - F.site_xpos(i)[:, None, :]) / o.const(sz)
        dl = o.einsum("bji,bcj->bci", R, ray) / o.const(sz)
        a, b, c = o.sum(dl * dl, -1), o.sum(pl * dl, -1), o.sum(pl * pl, -1) - 1
        disc = b * b - a * c
        hit = (disc >= 0) & (-b + o.sqrt(o.maximum(disc, o.zeros_like(disc))) >= 0)
        use = con["valid"] & (on1 | on2) & (fn > 0) & hit
        out.append(o.sum(o.where(use, fn, o.zeros_like(fn)), -1))
    return o.stack(out, 1)


def _xmat(F, b, ij):
    i, j = "xyz".index(ij[0]), "xyz".index(ij[1])
    return F.xmat[:, b, i, j]


def _randomize_joints(env, rnd, B, qpos):
    import mujoco
    m = env.original
    cols = [qpos[:, i] for i in range(m.nq)]
    for j in range(m.njnt):
        t, a = int(m.jnt_type[j]), int(m.jnt_qposadr[j])
        if m.jnt_limited[j] and t in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            lo, hi = (float(x) for x in m.jnt_range[j])
            cols[a] = rnd.uniform((B,), lo, hi)
        elif not m.jnt_limited[j] and t == int(mujoco.mjtJoint.mjJNT_HINGE):
            cols[a] = rnd.uniform((B,), -math.pi, math.pi)
        elif t == int(mujoco.mjtJoint.mjJNT_FREE):
            q = rnd.uniform((B, 4))
            q = q / env.o.norm(q)[:, None]
            for k in range(4):
                cols[a + 3 + k] = q[:, k]
        else:
            assert not m.jnt_limited[j], f"joint {j}: limited ball joint"
    return cols


def _qpos0(env, B):
    return env.o.const(np.asarray(env.original.qpos0, np.float32))[None] * env.o.full((B, 1), 1.0)


def _small_control(o, action):
    return tolerance(o, action, margin=1, value_at_margin=0, sigmoid="quadratic")


class Cartpole(Spec):
    domain, control_dt, time_limit = "cartpole", 0.01, 10.0
    refine = 4
    _CART_RANGE, _ANGLE_COSINE_RANGE = (-.25, .25), (.995, 1)

    def __init__(self, task, swing_up, sparse):
        self.task, self.swing_up, self.sparse = task, swing_up, sparse

    def notes(self, task):
        return [f"the suite's cartpole integrates with RK4; the engine with Euler, {task.refine} substeps per step "
                "(Task(refine=k))"]

    def reset(self, env, rnd, B):
        o, nv = env.o, env.nv
        if self.swing_up:
            cols = [.01 * rnd.normal((B,)), math.pi + .01 * rnd.normal((B,))]
            cols += [.1 * rnd.normal((B,)) for _ in range(nv - 2)]
        else:
            cols = [rnd.uniform((B,), -.1, .1)] + [rnd.uniform((B,), -.034, .034) for _ in range(nv - 1)]
        return o.stack(cols, 1), .01 * rnd.normal((B, nv)), {}

    def obs(self, env, s, F):
        pos = [s.qpos[:, :1]]
        for b in range(2, env.original.nbody):
            pos += [_xmat(F, b, "zz")[:, None], _xmat(F, b, "xz")[:, None]]
        return {"position": env.o.concat(pos, -1), "velocity": s.qvel}

    def reward(self, env, s2, action, s, F):
        o = env.o
        cart = s2.qpos[:, 0]
        cos = o.stack([_xmat(F, b, "zz") for b in range(2, env.original.nbody)], 1)
        if self.sparse:
            c = tolerance(o, cart, self._CART_RANGE)
            a = o.prod(tolerance(o, cos, self._ANGLE_COSINE_RANGE), -1)
            return c * a, {"cart_in_bounds": c, "angle_in_bounds": a}
        upright = o.mean((cos + 1) / 2, -1)
        centered = (1 + tolerance(o, cart, margin=2)) / 2
        small_control = (4 + _small_control(o, action)[:, 0]) / 5
        small_velocity = (1 + o.min(tolerance(o, s2.qvel[:, 1:], margin=5), -1)) / 2
        r = upright * small_control * small_velocity * centered
        return r, {"upright": upright, "centered": centered, "small_control": small_control,
                   "small_velocity": small_velocity}


class Pendulum(Spec):
    domain, task, control_dt, time_limit = "pendulum", "swingup", 0.02, 20.0
    _COSINE_BOUND = math.cos(math.radians(8))

    def reset(self, env, rnd, B):
        return rnd.uniform((B, 1), -math.pi, math.pi), env.o.zeros((B, 1)), {}

    def obs(self, env, s, F):
        b = env.kin.body("pole")
        return {"orientation": env.o.stack([_xmat(F, b, "zz"), _xmat(F, b, "xz")], 1), "velocity": s.qvel[:, :1]}

    def reward(self, env, s2, action, s, F):
        r = tolerance(env.o, _xmat(F, env.kin.body("pole"), "zz"), (self._COSINE_BOUND, 1))
        return r, {"upright": r}


class Acrobot(Spec):
    domain, control_dt, time_limit = "acrobot", 0.01, 10.0
    refine = 4

    def __init__(self, task, sparse):
        self.task, self.sparse = task, sparse

    def notes(self, task):
        return [f"the suite's acrobot integrates with RK4; the engine with Euler, {task.refine} substeps per step "
                "(Task(refine=k))"]

    def reset(self, env, rnd, B):
        return rnd.uniform((B, 2), -math.pi, math.pi), env.o.zeros((B, 2)), {}

    def obs(self, env, s, F):
        u, l = env.kin.body("upper_arm"), env.kin.body("lower_arm")
        o = env.o
        return {"orientations": o.stack([_xmat(F, u, "xz"), _xmat(F, l, "xz"), _xmat(F, u, "zz"), _xmat(F, l, "zz")], 1),
                "velocity": s.qvel}

    def reward(self, env, s2, action, s, F):
        k, o = env.kin, env.o
        d = o.norm(F.site_xpos(k.site("target")) - F.site_xpos(k.site("tip")))
        radius = float(env.original.site_size[k.site("target"), 0])
        r = tolerance(o, d, bounds=(0, radius), margin=0 if self.sparse else 1)
        return r, {"to_target": d}


class Cheetah(Spec):
    domain, task, control_dt, time_limit = "cheetah", "run", 0.01, 10.0
    _RUN_SPEED = 10

    def reset(self, env, rnd, B):
        m, o = env.original, env.o
        cols = [o.zeros((B,)) for _ in range(m.nq)]
        for j in range(m.njnt):
            if m.jnt_limited[j]:
                cols[int(m.jnt_qposadr[j])] = rnd.uniform((B,), *(float(x) for x in m.jnt_range[j]))
        q = o.stack(cols, 1)
        z, u = o.zeros((B, m.nv)), o.zeros((B, m.nu))
        _, v, w = env.physics(_qpos0(env, B), z, z, u, env.refine)
        q = q + env.physics_dt * v
        q, v, w = env.physics(q, v, w, u, 199 * env.refine)
        return q, v, {}, w

    def obs(self, env, s, F):
        return {"position": s.qpos[:, 1:], "velocity": s.qvel}

    def reward(self, env, s2, action, s, F):
        speed = F.subtree_linvel(env.kin.body("torso"))[:, 0]
        r = tolerance(env.o, speed, bounds=(self._RUN_SPEED, float("inf")), margin=self._RUN_SPEED,
                      value_at_margin=0, sigmoid="linear")
        return r, {"speed": speed}


class Walker(Spec):
    domain, control_dt, time_limit = "walker", 0.025, 25.0
    _STAND_HEIGHT = 1.2

    def __init__(self, task, move_speed):
        self.task, self.move_speed = task, move_speed

    def reset(self, env, rnd, B):
        q = env.o.stack(_randomize_joints(env, rnd, B, _qpos0(env, B)), 1)
        return q, env.o.zeros((B, env.nv)), {}

    def obs(self, env, s, F):
        o = env.o
        ori = o.stack([F.xmat[:, 1:, 0, 0], F.xmat[:, 1:, 0, 2]], -1).reshape(s.qpos.shape[0], -1)
        return {"orientations": ori, "height": F.xpos[:, env.kin.body("torso"), 2], "velocity": s.qvel}

    def reward(self, env, s2, action, s, F):
        o, torso = env.o, env.kin.body("torso")
        height = F.xpos[:, torso, 2]
        standing = tolerance(o, height, bounds=(self._STAND_HEIGHT, float("inf")), margin=self._STAND_HEIGHT / 2)
        upright = (1 + _xmat(F, torso, "zz")) / 2
        stand = (3 * standing + upright) / 4
        if self.move_speed == 0:
            return stand, {"standing": standing, "upright": upright}
        speed = F.subtree_linvel(torso)[:, 0]
        move = tolerance(o, speed, bounds=(self.move_speed, float("inf")), margin=self.move_speed / 2,
                         value_at_margin=0.5, sigmoid="linear")
        return stand * (5 * move + 1) / 6, {"standing": standing, "upright": upright, "move": move, "speed": speed}


class Reacher(Spec):
    domain, control_dt, time_limit = "reacher", 0.02, 20.0

    def __init__(self, task, target_size):
        self.task, self.target_size = task, target_size

    def reset(self, env, rnd, B):
        o = env.o
        q = o.stack(_randomize_joints(env, rnd, B, _qpos0(env, B)), 1)
        angle = rnd.uniform((B,), 0.0, 2 * math.pi)
        radius = rnd.uniform((B,), .05, .20)
        return q, o.zeros((B, env.nv)), {"target": o.stack([radius * o.sin(angle), radius * o.cos(angle)], 1)}

    def _to_target(self, env, s, F):
        k, o = env.kin, env.o
        g = k.geom("target")
        z = o.const(np.float32(env.original.geom_pos[g, 2])) * o.ones_like(s.aux["target"][:, :1])
        tgt = F.geom_xpos(g, o.concat([s.aux["target"], z], -1))
        return tgt[:, :2] - F.geom_xpos(k.geom("finger"))[:, :2]

    def obs(self, env, s, F):
        return {"position": s.qpos, "to_target": self._to_target(env, s, F), "velocity": s.qvel}

    def reward(self, env, s2, action, s, F):
        o = env.o
        d = o.norm(self._to_target(env, s2, F))
        radii = self.target_size + float(env.original.geom_size[env.kin.geom("finger"), 0])
        return tolerance(o, d, (0, radii)), {"distance": d}


class Hopper(Spec):
    domain, control_dt, time_limit = "hopper", 0.02, 20.0
    _STAND_HEIGHT, _HOP_SPEED = 0.6, 2
    contacts = True

    def __init__(self, task, hopping):
        self.task, self.hopping = task, hopping

    def notes(self, task):
        return ["touch (MuJoCo's touch sensors) from the engine's last-substep contacts; at a reset it is 0 (the "
                "suite computes the reset state's contact forces with mj_forward)"]

    def after_step(self, env, s2, q_prev, con):
        return {**s2.aux, "touch": touch(env, s2, con, ("touch_toe", "touch_heel"))}

    def reset(self, env, rnd, B):
        q = env.o.stack(_randomize_joints(env, rnd, B, _qpos0(env, B)), 1)
        return q, env.o.zeros((B, env.nv)), {"touch": env.o.zeros((B, 2))}

    def obs(self, env, s, F):
        return {"position": s.qpos[:, 1:], "velocity": s.qvel, "touch": env.o.log1p(s.aux["touch"])}

    def reward(self, env, s2, action, s, F):
        o, k = env.o, env.kin
        xi = F.xipos
        height = xi[:, k.body("torso"), 2] - xi[:, k.body("foot"), 2]
        standing = tolerance(o, height, (self._STAND_HEIGHT, 2))
        if self.hopping:
            speed = F.subtree_linvel(k.body("torso"))[:, 0]
            hopping = tolerance(o, speed, bounds=(self._HOP_SPEED, float("inf")), margin=self._HOP_SPEED / 2,
                                value_at_margin=0.5, sigmoid="linear")
            return standing * hopping, {"standing": standing, "hopping": hopping, "speed": speed}
        small_control = (o.mean(_small_control(o, action), -1) + 4) / 5
        return standing * small_control, {"standing": standing, "small_control": small_control}


class Humanoid(Spec):
    domain, control_dt, time_limit = "humanoid", 0.025, 25.0
    _STAND_HEIGHT = 1.4
    RESET_ROUNDS = 32

    def __init__(self, task, move_speed, pure_state=False):
        self.task, self.move_speed, self.pure_state = task, move_speed, pure_state

    def notes(self, task):
        return [f"reset: dm_control resamples until the pose has no contact; here up to {self.RESET_ROUNDS} rounds "
                "(envs.dmc.penetrating(task, qpos) tells the worlds still in contact)"]

    def reset(self, env, rnd, B):
        o = env.o
        q = resample_until_free(env, rnd, B, lambda: o.stack(_randomize_joints(env, rnd, B, _qpos0(env, B)), 1),
                                self.RESET_ROUNDS)
        return q, o.zeros((B, env.nv)), {}

    def _extremities(self, env, F):
        o, k = env.o, env.kin
        t = k.body("torso")
        R, x = F.xmat[:, t], F.xpos[:, t]
        out = []
        for side in ("left_", "right_"):
            for limb in ("hand", "foot"):
                d = F.xpos[:, k.body(side + limb)] - x
                out.append(o.einsum("bi,bij->bj", d, R))
        return o.concat(out, -1)

    def obs(self, env, s, F):
        if self.pure_state:
            return {"position": s.qpos, "velocity": s.qvel}
        k, o = env.kin, env.o
        t = k.body("torso")
        return {"joint_angles": s.qpos[:, 7:], "head_height": F.xpos[:, k.body("head"), 2],
                "extremities": self._extremities(env, F),
                "torso_vertical": o.stack([F.xmat[:, t, 2, 0], F.xmat[:, t, 2, 1], F.xmat[:, t, 2, 2]], 1),
                "com_velocity": F.subtree_linvel(t), "velocity": s.qvel}

    def reward(self, env, s2, action, s, F):
        o, k = env.o, env.kin
        t = k.body("torso")
        standing = tolerance(o, F.xpos[:, k.body("head"), 2], bounds=(self._STAND_HEIGHT, float("inf")),
                             margin=self._STAND_HEIGHT / 4)
        upright = tolerance(o, F.xmat[:, t, 2, 2], bounds=(0.9, float("inf")), sigmoid="linear", margin=1.9,
                            value_at_margin=0)
        stand = standing * upright
        small_control = (4 + o.mean(_small_control(o, action), -1)) / 5
        comv = F.subtree_linvel(t)
        terms = {"standing": standing, "upright": upright, "small_control": small_control}
        if self.move_speed == 0:
            dont_move = o.mean(tolerance(o, comv[:, :2], margin=2), -1)
            return small_control * stand * dont_move, {**terms, "dont_move": dont_move}
        speed = o.norm(comv[:, :2])
        move = tolerance(o, speed, bounds=(self.move_speed, float("inf")), margin=self.move_speed, value_at_margin=0,
                         sigmoid="linear")
        move = (5 * move + 1) / 6
        return small_control * stand * move, {**terms, "move": move, "speed": speed}


class Finger(Spec):
    domain, control_dt, time_limit = "finger", 0.02, 20.0
    contacts = True
    RESET_ROUNDS = 32
    _SPIN_VELOCITY = 15.0

    def __init__(self, task, target_radius=None):
        self.task, self.target_radius = task, target_radius

    def convert_options(self):
        import mujoco
        return {"drop_types": (int(mujoco.mjtGeom.mjGEOM_CYLINDER),)}

    def notes(self, task):
        return ["the decoration cylinders' contact pairs are left out (they never touch: the proximal hub only turns "
                "about its own axis, 0.4 m from the spinner; the spinner hub lies inside its caps; 0 of 30,000 "
                "random-action steps in dm_control had one; leaving them out keeps the build small)",
                "touch from the engine's last-substep contacts; at a reset it is 0"]

    def adjust_model(self, m):
        if self.target_radius is None:
            import mujoco
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
            m.dof_damping[m.jnt_dofadr[j]] = 0.03

    def reset(self, env, rnd, B):
        o = env.o
        q = resample_until_free(env, rnd, B, lambda: o.stack(_randomize_joints(env, rnd, B, _qpos0(env, B)), 1),
                                self.RESET_ROUNDS)
        aux = {"touch": o.zeros((B, 2))}
        if self.target_radius is not None:
            m0, k = env.original, env.kin
            ang = rnd.uniform((B,), -math.pi, math.pi)
            j = mujoco_id(m0, "joint", "hinge")
            hb = int(m0.jnt_bodyid[j])
            anchor = np.array(m0.body_pos[hb], np.float32) + np.array(m0.jnt_pos[j], np.float32)
            radius = float(m0.geom_size[k.geom("cap1")].sum())
            aux["target"] = o.stack([anchor[0] + radius * o.sin(ang), anchor[2] + radius * o.cos(ang)], 1)
        return q, o.zeros((B, env.nv)), aux

    def after_step(self, env, s2, q_prev, con):
        return {**s2.aux, "touch": touch(env, s2, con, ("touchtop", "touchbottom"))}

    def _xz(self, F, env, site):
        x = F.site_xpos(env.kin.site(site))
        return env.o.stack([x[:, 0], x[:, 2]], 1)

    def obs(self, env, s, F):
        o, k = env.o, env.kin
        sp = o.stack([F.xpos[:, k.body("spinner"), 0], F.xpos[:, k.body("spinner"), 2]], 1)
        tip = self._xz(F, env, "tip") - sp
        qa = lambda n: s.qpos[:, env.original.jnt_qposadr[mujoco_id(env.original, "joint", n)]]
        va = lambda n: s.qvel[:, env.original.jnt_dofadr[mujoco_id(env.original, "joint", n)]]
        d = {"position": o.stack([qa("proximal"), qa("distal"), tip[:, 0], tip[:, 1]], 1),
             "velocity": o.stack([va("proximal"), va("distal"), va("hinge")], 1),
             "touch": o.log1p(s.aux["touch"])}
        if self.target_radius is not None:
            tgt = s.aux["target"] - sp
            d["target_position"] = tgt
            d["dist_to_target"] = o.norm(tgt - tip) - self.target_radius
        return d

    def reward(self, env, s2, action, s, F):
        o = env.o
        if self.target_radius is None:
            hv = s2.qvel[:, env.original.jnt_dofadr[mujoco_id(env.original, "joint", "hinge")]]
            r = o.where(hv <= -self._SPIN_VELOCITY, o.ones_like(hv), o.zeros_like(hv))
            return r, {"hinge_velocity": hv}
        dist = self.obs(env, s2, F)["dist_to_target"]
        return o.where(dist <= 0, o.ones_like(dist), o.zeros_like(dist)), {"dist_to_target": dist}


def mujoco_id(m, kind, name):
    import mujoco
    return mujoco.mj_name2id(m, getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}"), name)


TASKS = {
    "cartpole-balance": lambda: Cartpole("balance", swing_up=False, sparse=False),
    "cartpole-balance_sparse": lambda: Cartpole("balance_sparse", swing_up=False, sparse=True),
    "cartpole-swingup": lambda: Cartpole("swingup", swing_up=True, sparse=False),
    "cartpole-swingup_sparse": lambda: Cartpole("swingup_sparse", swing_up=True, sparse=True),
    "pendulum-swingup": Pendulum,
    "acrobot-swingup": lambda: Acrobot("swingup", sparse=False),
    "acrobot-swingup_sparse": lambda: Acrobot("swingup_sparse", sparse=True),
    "cheetah-run": Cheetah,
    "walker-stand": lambda: Walker("stand", 0),
    "walker-walk": lambda: Walker("walk", 1),
    "walker-run": lambda: Walker("run", 8),
    "hopper-stand": lambda: Hopper("stand", False),
    "hopper-hop": lambda: Hopper("hop", True),
    "finger-spin": lambda: Finger("spin"),
    "finger-turn_easy": lambda: Finger("turn_easy", 0.07),
    "finger-turn_hard": lambda: Finger("turn_hard", 0.03),
    "humanoid-stand": lambda: Humanoid("stand", 0),
    "humanoid-walk": lambda: Humanoid("walk", 1),
    "humanoid-run": lambda: Humanoid("run", 10),
    "reacher-easy": lambda: Reacher("easy", .05),
    "reacher-hard": lambda: Reacher("hard", .015),
}

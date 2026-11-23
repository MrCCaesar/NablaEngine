"""Per-world physics parameters: model values that each world (environment) sets in its step's input, for domain
randomization. Chosen at build time: PhysicsParams(fields) passed as init(..., params=) of jax_ffi / torch_ffi; a build
without them is unchanged."""
from dataclasses import dataclass

import numpy as np

WIDTHS = {"foot_solref": 2, "foot_solimp": 4, "dof_damping": "nv", "dof_armature": "nv", "dof_frictionloss": "nv",
          "frictionloss_solref": 2, "frictionloss_solimp": 4, "torso_mass": 1,
          "torso_inertia": 3, "foot_torsional": 1, "object_size": 3, "gravity": 3, "xfrc": "xfrc",
          "body_inertial": "inertial", "sole_thickness": 1, "foot_offset": 4}
XFRC_WORDS = 9
INERTIAL_WORDS = 7
INVWEIGHT0 = ("model", "per_world", "per_world_mass")
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_LINE = (-0.064, 0.142)


@dataclass(frozen=True)
class PhysicsParams:
    fields: tuple = ()
    size_geom: object = None
    object_size_max: tuple = None
    xfrc_bodies: tuple = ()
    inertial_bodies: tuple = ()
    sole_geoms: tuple = ()
    sole_normal: tuple = None
    invweight0: str = "model"
    foot_bodies: tuple = ()
    foot_line: tuple = None
    foot_axis: tuple = None

    def __post_init__(self):
        f = tuple(self.fields)
        bad = [x for x in f if x not in WIDTHS]
        if bad or len(set(f)) != len(f):
            raise ValueError(f"PhysicsParams fields {f}: each of {tuple(WIDTHS)} at most once")
        object.__setattr__(self, "fields", tuple(x for x in WIDTHS if x in f))
        if "object_size" in f:
            if self.object_size_max is None:
                raise ValueError("PhysicsParams object_size: object_size_max=(x, y, z), the largest half-extents of "
                                 "any world, is required (the collision bounds are built for it)")
            mx = tuple(float(x) for x in np.broadcast_to(np.asarray(self.object_size_max, np.float64), (3,)))
            if not all(np.isfinite(x) and x > 0 for x in mx):
                raise ValueError(f"PhysicsParams object_size_max={self.object_size_max}: three positive numbers")
            object.__setattr__(self, "object_size_max", mx)
            if isinstance(self.size_geom, (np.integer, int)) and not isinstance(self.size_geom, bool):
                object.__setattr__(self, "size_geom", int(self.size_geom))
            elif self.size_geom is not None and not isinstance(self.size_geom, str):
                raise ValueError(f"PhysicsParams size_geom={self.size_geom!r}: a geom name or id")
        elif self.size_geom is not None or self.object_size_max is not None:
            raise ValueError("PhysicsParams size_geom / object_size_max belong to the object_size field")
        xb = (self.xfrc_bodies,) if isinstance(self.xfrc_bodies, (str, int, np.integer)) else tuple(self.xfrc_bodies)
        xb = tuple(int(x) if isinstance(x, (int, np.integer)) and not isinstance(x, bool) else x for x in xb)
        if "xfrc" in f:
            if not xb or not all(isinstance(x, (str, int)) and not isinstance(x, bool) for x in xb):
                raise ValueError(f"PhysicsParams xfrc: xfrc_bodies={self.xfrc_bodies!r} must name one or more bodies "
                                 "(names or model ids)")
            if len(set(xb)) != len(xb):
                raise ValueError(f"PhysicsParams xfrc_bodies={xb}: a body more than once")
        elif xb:
            raise ValueError("PhysicsParams xfrc_bodies belongs to the xfrc field")
        object.__setattr__(self, "xfrc_bodies", xb)
        ib = _ids(self.inertial_bodies)
        if "body_inertial" in f:
            if not ib or not all(isinstance(x, (str, int)) and not isinstance(x, bool) for x in ib):
                raise ValueError(f"PhysicsParams body_inertial: inertial_bodies={self.inertial_bodies!r} must name one "
                                 "or more bodies (names or model ids)")
            if len(set(ib)) != len(ib):
                raise ValueError(f"PhysicsParams inertial_bodies={ib}: a body more than once")
        elif ib:
            raise ValueError("PhysicsParams inertial_bodies belongs to the body_inertial field")
        object.__setattr__(self, "inertial_bodies", ib)
        sg = _ids(self.sole_geoms)
        if "sole_thickness" in f:
            if not all(isinstance(x, (str, int)) and not isinstance(x, bool) for x in sg) or len(set(sg)) != len(sg):
                raise ValueError(f"PhysicsParams sole_geoms={self.sole_geoms!r}: distinct geom names or ids")
            n = (0.0, 0.0, -1.0) if self.sole_normal is None else tuple(
                float(x) for x in np.broadcast_to(np.asarray(self.sole_normal, np.float64), (3,)))
            if not (np.all(np.isfinite(n)) and abs(np.linalg.norm(n) - 1.0) < 1e-6):
                raise ValueError(f"PhysicsParams sole_normal={self.sole_normal}: a unit vector")
            object.__setattr__(self, "sole_normal", n)
        elif sg or self.sole_normal is not None:
            raise ValueError("PhysicsParams sole_geoms / sole_normal belong to the sole_thickness field")
        object.__setattr__(self, "sole_geoms", sg)
        fb = _ids(self.foot_bodies)
        if "foot_offset" in f:
            fb = fb or FOOT_BODIES
            if len(fb) != 2 or len(set(fb)) != 2 or not all(isinstance(x, (str, int)) and not isinstance(x, bool)
                                                            for x in fb):
                raise ValueError(f"PhysicsParams foot_bodies={self.foot_bodies!r}: two distinct bodies (left, right), "
                                 "names or ids")
            ln = FOOT_LINE if self.foot_line is None else tuple(float(x) for x in np.asarray(self.foot_line).ravel())
            if len(ln) != 2 or not (np.all(np.isfinite(ln)) and ln[1] > ln[0]):
                raise ValueError(f"PhysicsParams foot_line={self.foot_line}: (heel, toe) with heel < toe")
            ax = (1.0, 0.0, 0.0) if self.foot_axis is None else tuple(
                float(x) for x in np.broadcast_to(np.asarray(self.foot_axis, np.float64), (3,)))
            if not (np.all(np.isfinite(ax)) and abs(np.linalg.norm(ax) - 1.0) < 1e-6):
                raise ValueError(f"PhysicsParams foot_axis={self.foot_axis}: a unit vector")
            object.__setattr__(self, "foot_line", tuple(float(x) for x in ln))
            object.__setattr__(self, "foot_axis", ax)
        elif fb or self.foot_line is not None or self.foot_axis is not None:
            raise ValueError("PhysicsParams foot_bodies / foot_line / foot_axis belong to the foot_offset field")
        object.__setattr__(self, "foot_bodies", fb)
        if self.invweight0 not in INVWEIGHT0:
            raise ValueError(f"PhysicsParams invweight0={self.invweight0!r}: one of {INVWEIGHT0}")
        if self.invweight0 != "model" and not f:
            raise ValueError(f"PhysicsParams invweight0={self.invweight0!r} needs a parameter build (one or more fields)")

    def __bool__(self):
        return bool(self.fields)

    def layout(self, nv):
        """{field: (offset, width)} within the per-world parameter block."""
        out, o = {}, 0
        for f in self.fields:
            w = (nv if WIDTHS[f] == "nv" else XFRC_WORDS * len(self.xfrc_bodies) if WIDTHS[f] == "xfrc"
                 else INERTIAL_WORDS * len(self.inertial_bodies) if WIDTHS[f] == "inertial" else WIDTHS[f])
            out[f] = (o, w)
            o += w
        return out

    def width(self, nv):
        return sum(w for _, w in self.layout(nv).values())

    def key(self):
        """The build's key: the field names, and with object_size its geom and maximum, with xfrc its bodies, with
        body_inertial its bodies, with sole_thickness its geoms and normal, with foot_offset its feet, line and axis,
        and a per-world invweight0 (as the last entry)."""
        extra = {}
        if "object_size" in self.fields:
            extra.update(size_geom=self.size_geom, object_size_max=list(self.object_size_max))
        if "xfrc" in self.fields:
            extra.update(xfrc_bodies=list(self.xfrc_bodies))
        if "body_inertial" in self.fields:
            extra.update(inertial_bodies=list(self.inertial_bodies))
        if "sole_thickness" in self.fields:
            extra.update(sole_geoms=list(self.sole_geoms), sole_normal=list(self.sole_normal))
        if "foot_offset" in self.fields:
            extra.update(foot_bodies=list(self.foot_bodies), foot_line=list(self.foot_line),
                         foot_axis=list(self.foot_axis))
        if self.invweight0 != "model":
            extra.update(invweight0=self.invweight0)
        return list(self.fields) + ([extra] if extra else [])

    def pack(self, nv, **values):
        """The per-world parameter block (B, width(nv)) float32 from arrays by field name ((B, width) or a broadcastable
        shape); every field of the build is required."""
        lay = self.layout(nv)
        missing = [f for f in lay if f not in values]
        extra = [f for f in values if f not in lay]
        if missing or extra:
            raise ValueError(f"PhysicsParams.pack: missing {missing}, not in this build {extra}")
        B = max(np.shape(np.atleast_2d(v))[0] for v in values.values())
        out = np.zeros((B, self.width(nv)), np.float32)
        for f, (o, w) in lay.items():
            out[:, o:o + w] = np.broadcast_to(np.asarray(values[f], np.float32).reshape(-1, w), (B, w))
        if "dof_frictionloss" in lay:
            o, w = lay["dof_frictionloss"]
            if not np.all(out[:, o:o + w] >= 0):
                raise ValueError("PhysicsParams.pack: dof_frictionloss below 0")
        if "frictionloss_solref" in lay:
            o, w = lay["frictionloss_solref"]
            if not np.all(out[:, o:o + w] > 0):
                raise ValueError("PhysicsParams.pack: frictionloss_solref takes the standard form (both entries > 0)")
        if "object_size" in lay:
            o, w = lay["object_size"]
            sz = out[:, o:o + w]
            if not (np.all(sz > 0) and np.all(sz <= np.float32(self.object_size_max) * np.float32(1 + 1e-6))):
                raise ValueError(f"PhysicsParams.pack: object_size outside (0, object_size_max={self.object_size_max}]")
        if "foot_offset" in lay:
            o, w = lay["foot_offset"]
            if not np.all(out[:, o:o + w] >= 0):
                raise ValueError("PhysicsParams.pack: foot_offset must be >= 0 (a support lowered under the foot; the "
                                 "forest engine's contacts have margin 0)")
        return out

    def defaults(self, m, foot_pairs, com_body=None):
        """The model's own values per field (one world), e.g. as pack(**defaults) for an unchanged world (com_body: the
        torso fields' body, default the G1's, and object_size's default body; the foot fields take the first foot
        pair's; gravity: opt.gravity; xfrc: zero wrenches, each point at its body's COM, body_ipos; body_inertial: each
        body's body_mass, body_inertia and a zero COM offset, (k, 7); sole_thickness: 0; foot_offset: zeros)."""
        import mujoco
        from nablaengine.g1_engine import source
        fp = list(foot_pairs)
        if not fp and {"foot_solref", "foot_solimp", "foot_torsional"} & set(self.fields):
            raise ValueError("PhysicsParams.defaults: the foot fields need foot pairs")
        out = {}
        if "torso_mass" in self.fields or "torso_inertia" in self.fields:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, com_body or source.COM_BODY)
            if b < 1:
                raise ValueError(f"PhysicsParams.defaults: no body {com_body or source.COM_BODY!r}")
            if "torso_mass" in self.fields:
                out["torso_mass"] = np.asarray([m.body_mass[b]], np.float32)
            if "torso_inertia" in self.fields:
                out["torso_inertia"] = np.asarray(m.body_inertia[b], np.float32)
        if "foot_solref" in self.fields:
            out["foot_solref"] = np.asarray(m.pair_solref[fp[0]], np.float32)
        if "foot_solimp" in self.fields:
            out["foot_solimp"] = np.asarray(m.pair_solimp[fp[0], :4], np.float32)
        if "foot_torsional" in self.fields:
            out["foot_torsional"] = np.asarray([m.pair_friction[fp[0], 2]], np.float32)
        if "dof_damping" in self.fields:
            out["dof_damping"] = np.asarray(m.dof_damping, np.float32)
        if "dof_armature" in self.fields:
            out["dof_armature"] = np.asarray(m.dof_armature, np.float32)
        if "dof_frictionloss" in self.fields:
            out["dof_frictionloss"] = np.asarray(m.dof_frictionloss, np.float32)
        if "frictionloss_solref" in self.fields or "frictionloss_solimp" in self.fields:
            fl = np.flatnonzero(m.dof_frictionloss > 0)
            d0 = int(fl[0]) if len(fl) else 0
            if "frictionloss_solref" in self.fields:
                out["frictionloss_solref"] = np.asarray(m.dof_solref[d0], np.float32)
            if "frictionloss_solimp" in self.fields:
                out["frictionloss_solimp"] = np.asarray(m.dof_solimp[d0, :4], np.float32)
        if "gravity" in self.fields:
            out["gravity"] = np.asarray(m.opt.gravity, np.float32)
        if "xfrc" in self.fields:
            x = np.zeros((len(self.xfrc_bodies), XFRC_WORDS), np.float32)
            for i, b in enumerate(self.xfrc_bodies):
                bid = b if isinstance(b, int) else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                if not 1 <= bid < m.nbody:
                    raise ValueError(f"PhysicsParams.defaults: no (moving) body {b!r}")
                x[i, 6:] = m.body_ipos[bid]
            out["xfrc"] = x.reshape(-1)
        if "body_inertial" in self.fields:
            x = np.zeros((len(self.inertial_bodies), INERTIAL_WORDS), np.float32)
            for i, b in enumerate(self.inertial_bodies):
                bid = b if isinstance(b, int) else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                if not 1 <= bid < m.nbody:
                    raise ValueError(f"PhysicsParams.defaults: no (moving) body {b!r}")
                x[i, 0], x[i, 1:4] = m.body_mass[bid], m.body_inertia[bid]
            out["body_inertial"] = x
        if "sole_thickness" in self.fields:
            out["sole_thickness"] = np.zeros(1, np.float32)
        if "foot_offset" in self.fields:
            out["foot_offset"] = np.zeros(4, np.float32)
        if "object_size" in self.fields:
            from nablaengine._optional import FOREST, load
            object_size = load("nablaengine.forest_engine", FOREST, "object_size")
            out["object_size"] = np.asarray(m.geom_size[object_size.size_geom(m, self, com_body)], np.float32)
        return out


def _ids(x):
    t = (x,) if isinstance(x, (str, int, np.integer)) else tuple(x)
    return tuple(int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else v for v in t)


def as_params(params):
    """PhysicsParams from None, a PhysicsParams or a sequence of field names."""
    if params is None:
        return PhysicsParams()
    if isinstance(params, PhysicsParams):
        return params
    if isinstance(params, str):
        return PhysicsParams((params,))
    return PhysicsParams(tuple(params))

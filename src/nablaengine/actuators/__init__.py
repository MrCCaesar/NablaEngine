"""Actuator models on top of the model's MJCF actuators (opt-in): init(actuator=...) of nablaengine.jax_ffi,
nablaengine.torch_ffi (init / build) and jax_ffi.EnginePhysics."""

from dataclasses import dataclass

from nablaengine.motor import PRESETS, MotorCurve, MotorLimits

__all__ = ["Actuators", "Bam", "Spec", "TorqueSpeed", "resolve"]


@dataclass(frozen=True)
class TorqueSpeed:
    """Torque-speed limits of one motor type (nablaengine.motor.MotorCurve) on the joints its pattern matches."""
    curve: MotorCurve


@dataclass(frozen=True)
class Bam:
    """Rhoban's BAM servo model (Better Actuator Models, model M6) with a parameter set ("xl330/m6", bundled, or a path
    to a BAM parameters json of an XL330 M6 set), the servo firmware's proportional gain kp_fw, the nominal supply
    voltage vin_nom (V), the supply's floor vin_min (V) and the firmware current limit max_current ("firmware": the
    servo's own, 1.75 A for the XL330; a current in A; None: off). Both engines (e.g. a single-tree robot with a free root and
    one position servo per hinge). nablaengine.actuators.bam has the model and its engine form."""
    params: str = "xl330/m6"
    kp_fw: float = 200.0
    vin_nom: float = 7.4
    vin_min: float = 6.0
    max_current: object = "firmware"

    def key(self):
        mc = self.max_current if self.max_current in (None, "firmware") else f"{float(self.max_current):g}"
        return (f"bam:{self.params}:kp_fw={float(self.kp_fw):g}:vin_nom={float(self.vin_nom):g}"
                f":vin_min={float(self.vin_min):g}:max_current={mc}")


class Actuators:
    """Actuator models by joint: Actuators((joint-name regular expression, model), ...), model a TorqueSpeed or a Bam
    (a bare MotorCurve is taken as TorqueSpeed); each actuated joint takes the first pattern that matches its name in
    full. name: the build key's label (as MotorLimits.name)."""

    def __init__(self, *entries, name="custom"):
        out = []
        for e in entries:
            if not (isinstance(e, tuple) and len(e) == 2 and isinstance(e[0], str)):
                raise TypeError(f"Actuators: entries are (joint regex, model) pairs, not {e!r}")
            p, mdl = e
            if isinstance(mdl, MotorCurve):
                mdl = TorqueSpeed(mdl)
            if not isinstance(mdl, (TorqueSpeed, Bam)):
                raise TypeError(f"Actuators: the model of {p!r} must be a TorqueSpeed, a Bam or a MotorCurve, not "
                                f"{type(mdl).__name__}")
            out.append((p, mdl))
        if not out:
            raise ValueError("Actuators: no entries")
        self.entries, self.name = tuple(out), str(name)

    def __repr__(self):
        return f"Actuators({', '.join(repr(e) for e in self.entries)}, name={self.name!r})"


@dataclass(frozen=True)
class Spec:
    """A resolved actuator option: motor (a MotorLimits: torque-speed limits, the existing motor hooks) or bam (a Bam
    on every actuated joint), never both (yet)."""
    motor: MotorLimits = None
    bam: Bam = None

    def key(self, m):
        """The build's info["actuator"]: MotorLimits.key(m) for torque-speed limits (= info["motor_limits"])."""
        return self.motor.key(m) if self.motor is not None else self.bam.key()


def _from_kind(d, where):
    d = dict(d)
    kind = d.pop("kind")
    if kind == "bam":
        try:
            return Spec(bam=Bam(**d))
        except TypeError as e:
            raise TypeError(f"{where}={{'kind': 'bam', ...}}: {e} (keys: params, kp_fw, vin_nom, vin_min, max_current)") from None
    if kind in ("torque_speed", "motor_limits"):
        if set(d) - {"curves", "name", "require_all"}:
            raise TypeError(f"{where}={{'kind': {kind!r}, ...}}: keys curves, name, require_all")
        c = d.get("curves")
        return Spec(motor=MotorLimits(curves=tuple(c.items()) if isinstance(c, dict) else tuple(c),
                                      name=d.get("name", "custom"), require_all=d.get("require_all", True)))
    raise ValueError(f"{where}: unknown kind {kind!r} (one of 'bam', 'torque_speed')")


def _from_actuators(a):
    ts = [(p, x.curve) for p, x in a.entries if isinstance(x, TorqueSpeed)]
    bam = [x for _, x in a.entries if isinstance(x, Bam)]
    if not bam:
        return Spec(motor=MotorLimits(curves=tuple(ts), name=a.name))
    if ts:
        raise ValueError("actuator: BAM together with torque-speed limits in one build is not supported")
    if len(set(bam)) != 1 or a.entries[-1][0] != ".*":
        raise ValueError("actuator: BAM takes one Bam model on every actuated joint (Actuators((r'.*', Bam(...)),))")
    return Spec(bam=bam[0])


def resolve(actuator=None, motor_limits=None, where="actuator"):
    """The Spec of init's actuator= / motor_limits= (aliases; both given raises ValueError), None for none. Accepts
    None / False, a preset name ("unitree_g1", "mjlab_g1_dc") or "bam", a MotorLimits, a TorqueSpeed (every actuated
    joint), a Bam (every actuated joint), a {joint regex: MotorCurve | TorqueSpeed | Bam} dict, a {"kind": "bam" |
    "torque_speed", ...} dict, an Actuators or a Spec."""
    a_off, m_off = actuator is None or actuator is False, motor_limits is None or motor_limits is False
    if not a_off and not m_off:
        raise ValueError(f"{where}: actuator= and motor_limits= are aliases; give one of them, not both")
    x, name = (motor_limits, "motor_limits") if a_off else (actuator, "actuator")
    if x is None or x is False:
        return None
    if isinstance(x, Spec):
        return x
    if isinstance(x, MotorLimits):
        return Spec(motor=x)
    if isinstance(x, str):
        if x == "bam":
            return Spec(bam=Bam())
        if x not in PRESETS:
            raise ValueError(f"{name}={x!r}: one of {sorted(PRESETS) + ['bam']}, a MotorLimits, an Actuators or a "
                             "dict")
        return Spec(motor=PRESETS[x])
    if isinstance(x, TorqueSpeed):
        return Spec(motor=MotorLimits(curves=((r".*", x.curve),)))
    if isinstance(x, Bam):
        return Spec(bam=x)
    if isinstance(x, Actuators):
        return _from_actuators(x)
    if isinstance(x, dict):
        if "kind" in x:
            return _from_kind(x, name)
        if all(isinstance(v, MotorCurve) for v in x.values()):
            return Spec(motor=MotorLimits(curves=tuple(x.items())))
        return _from_actuators(Actuators(*x.items()))
    raise TypeError(f"{name}: None, a preset name, 'bam', a MotorLimits, a TorqueSpeed, a Bam, an Actuators or a dict, "
                    f"not {type(x).__name__}")


def check_engine(spec, forest, where):
    """Refuse what the model's engine does not simulate (both engines have the BAM servo model: g1_engine /
    forest_engine.bam_hooks; nothing is refused today)."""
    return


def check_bam_build(spec, where, terrain=None, params=None, contact_forces=False):
    """Refuse the build options BAM does not support yet (ValueError): a terrain build, per-world parameters, the
    contact-force table, DK_CLAMP_IMPLICIT=2."""
    if spec is None or spec.bam is None:
        return
    bad = [what for on, what in ((terrain is not None, "a terrain build"), (bool(params), "per-world parameters "
                                 "(params=)"), (bool(contact_forces), "contact_forces=True")) if on]
    from nablaengine.g1_engine import step
    if step.clamp_implicit()[0] == 2:
        bad.append("DK_CLAMP_IMPLICIT=2")
    if bad:
        raise ValueError(f"{where}: actuator='bam' is not available with {', '.join(bad)} yet")


def prepare_model(spec, m):
    """The model the build simulates: bam.prepare(m, spec.bam) for BAM (the servo model's engine form, checked), else
    m itself."""
    if spec is None or spec.bam is None:
        return m
    from nablaengine._optional import BAM, load
    bam = load("nablaengine.actuators", BAM, "bam")
    out = bam.prepare(m, spec.bam)
    bam.check_model(out, spec.bam)
    return out

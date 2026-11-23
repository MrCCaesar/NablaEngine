"""dm_control's reward shaping (dm_control.utils.rewards.tolerance, Copyright 2017 The dm_control Authors, Apache
License 2.0) as differentiable array code (envs.ops): the same values; where an out-of-bounds branch is not taken,
its argument is replaced before the sigmoid, so the gradients stay finite with infinite bounds."""
import math

DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoid(o, x, value_at_1, sigmoid):
    if sigmoid in ("cosine", "linear", "quadratic"):
        assert 0 <= value_at_1 < 1, value_at_1
    else:
        assert 0 < value_at_1 < 1, value_at_1
    if sigmoid == "gaussian":
        scale = math.sqrt(-2 * math.log(value_at_1))
        return o.exp(-0.5 * (x * scale) ** 2)
    if sigmoid == "hyperbolic":
        scale = math.acosh(1 / value_at_1)
        return 1 / o.cosh(x * scale)
    if sigmoid == "long_tail":
        scale = math.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)
    if sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (o.abs(x) * scale + 1)
    if sigmoid == "cosine":
        sx = x * (math.acos(2 * value_at_1 - 1) / math.pi)
        inside = o.abs(sx) < 1
        sx = o.where(inside, sx, o.zeros_like(sx))
        return o.where(inside, (1 + o.cos(math.pi * sx)) / 2, o.zeros_like(sx))
    if sigmoid == "linear":
        sx = x * (1 - value_at_1)
        return o.where(o.abs(sx) < 1, 1 - sx, o.zeros_like(sx))
    if sigmoid == "quadratic":
        sx = x * math.sqrt(1 - value_at_1)
        return o.where(o.abs(sx) < 1, 1 - sx ** 2, o.zeros_like(sx))
    if sigmoid == "tanh_squared":
        scale = math.atanh(math.sqrt(1 - value_at_1))
        return 1 - o.tanh(x * scale) ** 2
    raise ValueError(f"unknown sigmoid {sigmoid!r}")


def tolerance(o, x, bounds=(0.0, 0.0), margin=0.0, sigmoid="gaussian", value_at_margin=DEFAULT_VALUE_AT_MARGIN):
    """1 where lower <= x <= upper, else the sigmoid of the distance to the interval in units of margin (margin 0:
    an indicator)."""
    lower, upper = bounds
    assert lower <= upper and margin >= 0
    in_bounds = (x >= lower) & (x <= upper)
    one, zero = o.ones_like(x), o.zeros_like(x)
    if margin == 0:
        return o.where(in_bounds, one, zero)
    below = x < lower
    lo = lower if math.isfinite(lower) else 0.0
    hi = upper if math.isfinite(upper) else 0.0
    d = o.where(below, lo - x, x - hi) / margin
    d = o.where(in_bounds, zero, d)
    return o.where(in_bounds, one, _sigmoid(o, d, value_at_margin, sigmoid))

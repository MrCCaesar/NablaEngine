"""The engine generator for a model: nablaengine.g1_engine (the tree engine) for the G1 class (body 1 a free root and
every body in its tree: legged robots), nablaengine.forest_engine for the others (several trees under the world: a
fixed-base arm and free objects, the DeepMind Control Suite tasks, a dexterous hand and a cube). The interfaces
(jax_ffi, torch_ffi) build from the modules modules(m) returns and report the engine they picked as info["engine"]
("g1" or "forest"); the G1 class's text is g1_engine's, unchanged
(scripts/identity_gate.py). docs/engines.md: one interface, two engines."""
import os
import weakref
from types import SimpleNamespace

import mujoco

G1, FOREST = "g1", "forest"
ENV = "DK_ENGINE"
CHOICES = {"": "auto", "auto": "auto", "tree": "tree", "g1": "tree", "forest": "forest"}

_PREFER_FOREST = weakref.WeakSet()
_PREFER_TREE = weakref.WeakSet()


def prefer_forest(m):
    """Route model m to the forest engine even if it is of the G1 class (nablaengine.envs: legged models driven by
    torque motors or integrated with Euler, which the G1 engine does not simulate). Without it such a model is
    rejected by the G1 engine's capability check, as before. Such a model keeps the forest engine's own defaults (no
    COM-offset body unless init(com_body=) names one)."""
    _PREFER_TREE.discard(m)
    _PREFER_FOREST.add(m)


def prefer_tree(m):
    """Route the G1-class model m to the tree engine (nablaengine.g1_engine) whatever $DK_ENGINE says (the escape hatch
    for when the forest engine becomes the default). ValueError for a model that is not of the G1 class (only the
    forest engine simulates it)."""
    if not is_g1_class(m):
        raise ValueError("engines.prefer_tree: the model is not of the G1 class (body 1 a free root with every body in "
                         "its tree); only the forest engine simulates it")
    _PREFER_FOREST.discard(m)
    _PREFER_TREE.add(m)


def choice():
    """$DK_ENGINE as "auto", "tree" or "forest" (ValueError for another value)."""
    v = (os.environ.get(ENV) or "").strip().lower()
    if v not in CHOICES:
        raise ValueError(f"${ENV}={os.environ.get(ENV)!r}: one of tree, forest, auto")
    return CHOICES[v]


def is_g1_class(m):
    """Body 1 has a free joint and every other body descends from it (the tree engine's class of models)."""
    if m.nbody < 2 or m.body_jntnum[1] != 1 or m.jnt_type[m.body_jntadr[1]] != mujoco.mjtJoint.mjJNT_FREE:
        return False
    return all(int(m.body_rootid[b]) == 1 for b in range(1, m.nbody))


def default():
    """The engine $DK_ENGINE gives a G1-class model without prefer_forest / prefer_tree: "forest" (auto, unset) or
    "tree"."""
    return "tree" if choice() == "tree" else "forest"


def is_forest(m):
    """True unless m is of the G1 class (is_g1_class) and goes to the tree engine (module docstring: prefer_forest /
    prefer_tree, then $DK_ENGINE; auto: the forest engine)."""
    if not is_g1_class(m):
        return True
    if m in _PREFER_FOREST:
        return True
    if m in _PREFER_TREE:
        return False
    return default() == "forest"


def tree_defaults(m):
    """Whether model m gets the tree engine's defaults (the COM-offset body torso_link): it goes to the tree engine, or
    it is of the G1 class and goes to the forest engine through $DK_ENGINE or the default rather than prefer_forest."""
    return is_g1_class(m) and m not in _PREFER_FOREST


def name(m):
    """The name of model m's engine: "g1" or "forest" (info["engine"])."""
    return FOREST if is_forest(m) else G1


def modules(m):
    """SimpleNamespace(adjoint, host, kinematics, model_hooks, source, step, tape, forest, name) of the model's
    engine."""
    forest = is_forest(m)
    if forest:
        from nablaengine._optional import FOREST as FOREST_FEATURE, load
        adjoint, host, kinematics, model_hooks, source, step, tape = load(
            "nablaengine.forest_engine", FOREST_FEATURE,
            "adjoint", "host", "kinematics", "model_hooks", "source", "step", "tape")
    else:
        from nablaengine.g1_engine import adjoint, host, kinematics, model_hooks, source, step, tape
    return SimpleNamespace(adjoint=adjoint, host=host, kinematics=kinematics, model_hooks=model_hooks, source=source,
                           step=step, tape=tape, forest=forest, name=FOREST if forest else G1)


def require_forest(m, com_body, where, foot_pairs=None):
    """The forest engine's model checks (nablaengine.model.capability.check_forest, scope "forest"): raises
    capability.UnsupportedModel (a ValueError, as before) listing the failing checks by code; $DK_ALLOW_UNSUPPORTED
    applies as for the G1 engine. Returns the report."""
    from nablaengine.model import capability
    return capability.require(m, foot_pairs, com_body, "forest", where)


def require(m, foot_pairs, com_body, where):
    """The capability check of model m for its engine (the G1 engine's "engine" scope, or the forest engine's):
    raises capability.UnsupportedModel (a ValueError). Returns the report."""
    from nablaengine.model import capability
    if is_forest(m):
        return require_forest(m, com_body, where, foot_pairs)
    return capability.require(m, foot_pairs, com_body, "engine", where)

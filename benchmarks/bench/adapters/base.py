"""The interface every engine adapter implements, and the shared tangent-space helpers. Standard library and numpy
only: this module is imported in every engine environment."""

import re

import numpy as np

NATIVE_KINDS = ("native", "ours", "unmerged-upstream", "labelled")
DEVICES = ("gpu", "cpu")
OPS = ("fwd", "fwd_train", "bwd", "vjp", "jac")
GRAD_BLOCKS = ("qpos", "qvel", "ctrl")
DIAG_KEYS = ("ncon_mean", "ncon_max", "nefc_mean", "nefc_max", "niter_mean", "niter_max", "niter_hist", "overflow",
             "unsupported", "nonfinite_worlds", "extra")
SETUP_KEYS = ("build_s", "compile_s", "capture_s", "load_s")
HINT_KEYS = ("reps", "rounds", "warmup_calls", "warmup_s")
OOM_RE = re.compile(r"out of memory|out_of_memory|resource_exhausted|resource exhausted|cudaerrormemoryallocation|"
                    r"cuda_error_out_of_memory|failed to allocate|\boom\b")


class NotSupported(Exception):
    """The engine has no native way to do this; str(exc) is the recorded message."""


def is_oom(exc):
    """True for an out-of-memory error (MemoryError, or an engine error whose text says so)."""
    if isinstance(exc, MemoryError):
        return True
    return bool(OOM_RE.search(f"{type(exc).__name__} {exc}".lower()))


class Adapter:
    """Base class: the defaults of the optional hooks (see the module docstring for the contract)."""

    id = ""
    label = ""
    native = "labelled"
    device = "gpu"
    tracks = ("A",)
    variants = {"default": {}}

    def setup(self, problem, states, nworld, iterations, **opts):
        raise NotImplementedError

    def ops(self):
        raise NotImplementedError

    def not_supported(self):
        return {}

    def prepare(self, op):
        return {}

    def sync(self):
        pass

    def timing_hints(self, op):
        return {}

    def outputs(self):
        raise NotImplementedError

    def gradient(self, cot_qpos, cot_qvel):
        raise NotSupported(f"{self.id}: no native gradient")

    def jacobian(self):
        raise NotSupported(f"{self.id}: no native Jacobian")

    def gradient_not_supported(self):
        return {}

    def gradient_report(self):
        return {}

    def diagnostics(self):
        return {}

    def memory(self):
        return {}

    def provenance(self):
        return {}

    def semantics(self):
        return {}

    def close(self):
        pass

    @classmethod
    def check_env(cls):
        return {}

    @classmethod
    def describe(cls):
        """Class attributes as recorded with every cell."""
        return dict(id=cls.id, label=cls.label, native=cls.native, device=cls.device, tracks=list(cls.tracks),
                    variants=sorted(cls.variants))


def validate_class(cls, engine_id=None):
    """Problems of an adapter class's attributes (empty list: fine)."""
    out = []
    if not (isinstance(cls, type) and issubclass(cls, Adapter)):
        return [f"{cls!r} is not an Adapter subclass"]
    if engine_id is not None and cls.id != engine_id:
        out.append(f"{cls.__name__}.id = {cls.id!r}, registered as {engine_id!r}")
    if cls.native not in NATIVE_KINDS:
        out.append(f"{cls.__name__}.native = {cls.native!r} not in {NATIVE_KINDS}")
    if cls.device not in DEVICES:
        out.append(f"{cls.__name__}.device = {cls.device!r} not in {DEVICES}")
    if "default" not in cls.variants:
        out.append(f"{cls.__name__}.variants has no 'default'")
    if not cls.label:
        out.append(f"{cls.__name__}.label is empty")
    return out



def check_layout(nq, nv):
    if nq != nv + 1:
        raise NotImplementedError(f"the tangent helpers assume a free joint followed by hinges (nq = nv + 1), got "
                                  f"nq {nq}, nv {nv}: use mj_differentiatePos / mj_integratePos for other models")


def quat_mul(a, b):
    """Hamilton product of (..., 4) quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def qpos_tangent(g, q):
    """qpos-form gradients g (B, nq) at the states q (B, nq) -> the tangent form (B, nv): the quaternion part as
    1/2 Im(q* (x) g) / |q|^2, the rest unchanged (nablaengine.reference.adjoint_groups.qpos_tangent: the left
    inverse of MuJoCo Warp's dof-to-qpos map; for |q| = 1 exactly g . tangent_to_qpos(q))."""
    g = np.asarray(g, np.float64)
    q = np.asarray(q, np.float64)
    check_layout(g.shape[1], g.shape[1] - 1)
    xi, gq = q[:, 3:7], g[:, 3:7]
    w, x, y, z = xi.T
    a, b, c, d = gq.T
    v = np.stack([w * b - x * a - y * d + z * c, w * c + x * d - y * a - z * b, w * d - x * c + y * b - z * a], 1)
    return np.concatenate([g[:, :3], 0.5 * v / np.sum(xi * xi, 1, keepdims=True), g[:, 7:]], 1)


def tangent_to_qpos(q):
    """(B, nq) -> (B, nq, nv): d qpos / d tangent at q (position: identity; quaternion: 1/2 q (x) (0, e_i); hinges:
    identity), the map mj_integratePos linearises."""
    q = np.asarray(q, np.float64)
    B, nq = q.shape
    nv = nq - 1
    M = np.zeros((B, nq, nv))
    M[:, :3, :3] = np.eye(3)
    for i in range(3):
        e = np.zeros(4)
        e[1 + i] = 0.5
        M[:, 3:7, 3 + i] = quat_mul(q[:, 3:7], e)
    M[:, 7:, 6:] = np.eye(nq - 7)
    return M


def perturbed_qpos(q, h):
    """(nq,) -> (nv, nq): mj_integratePos(q, h e_j, 1) for every tangent direction j (the quaternion renormalised, as
    mj_integratePos does)."""
    q = np.asarray(q, np.float64)
    nq, nv = len(q), len(q) - 1
    out = np.tile(q, (nv, 1))
    out[np.arange(3), np.arange(3)] += h
    for i in range(3):
        r = np.zeros(4)
        r[0], r[1 + i] = np.cos(h / 2), np.sin(h / 2)
        out[3 + i, 3:7] = quat_mul(q[3:7], r)
    out[np.arange(6, nv), np.arange(7, nq)] += h
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def vjp_from_jacobian(J, cot_qpos, cot_qvel):
    """The VJP {qpos (tangent), qvel, ctrl} of Jacobians J (B, nq + nv, 2 nv + nu) for cotangents (B, nq), (B, nv)."""
    J = np.asarray(J, np.float64)
    nq, nv = cot_qpos.shape[1], cot_qvel.shape[1]
    c = np.concatenate([np.asarray(cot_qpos, np.float64), np.asarray(cot_qvel, np.float64)], 1)
    g = np.einsum("bi,bij->bj", c, J)
    if J.shape[1] != nq + nv:
        raise ValueError(f"J has {J.shape[1]} rows, cotangents {nq} + {nv}")
    return dict(qpos=g[:, :nv], qvel=g[:, nv:2 * nv], ctrl=g[:, 2 * nv:])


def jacobian_row_cotangents(nq, nv):
    """The nq + nv unit output cotangents (row r of the Jacobian = the VJP of e_r): (cot_qpos (R, nq), cot_qvel
    (R, nv)), R = nq + nv (71 for the G1: dk_jac's 71 VJPs)."""
    E = np.eye(nq + nv)
    return E[:, :nq], E[:, nq:]


def split_gradient(g, nv, nu):
    """(B, 2 nv + nu) in the Jacobian's column order -> {qpos (tangent), qvel, ctrl}."""
    g = np.asarray(g, np.float64)
    return dict(qpos=g[:, :nv], qvel=g[:, nv:2 * nv], ctrl=g[:, 2 * nv:2 * nv + nu])

"""The array operations the task definitions use, for JAX and PyTorch (nablaengine.envs): one task text, two
frameworks. Every operation takes and returns the framework's arrays (float32); constants come from NumPy via
const() (PyTorch: cached by value on the device, so repeated calls copy nothing from the host and a captured CUDA
graph replays them; JAX: jit embeds them in the compiled computation), and gathers take index(list)."""
import numpy as np


def _key(a):
    return a.dtype.str, a.shape, a.tobytes()


class JaxOps:
    name = "jax"

    def __init__(self):
        import jax
        import jax.numpy as jnp
        self.jax, self.np = jax, jnp
        for f in ("cos", "sin", "sqrt", "exp", "abs", "tanh", "cosh", "log1p", "maximum", "minimum", "where",
                  "stack", "zeros_like", "ones_like", "arctan2", "einsum", "square", "logical_and", "logical_or"):
            setattr(self, f, getattr(jnp, f))

    def const(self, x, dtype=np.float32):
        return self.np.asarray(np.asarray(x, dtype))

    def index(self, idx):
        """An index array for gathering along an axis (x[:, ops.index(list)])."""
        return np.asarray(idx, np.int32)

    def concat(self, xs, axis=-1):
        return self.np.concatenate(xs, axis=axis)

    def sum(self, x, axis=-1):
        return self.np.sum(x, axis=axis)

    def mean(self, x, axis=-1):
        return self.np.mean(x, axis=axis)

    def prod(self, x, axis=-1):
        return self.np.prod(x, axis=axis)

    def min(self, x, axis=-1):
        return self.np.min(x, axis=axis)

    def max(self, x, axis=-1):
        return self.np.max(x, axis=axis)

    def norm(self, x, axis=-1):
        return self.np.sqrt(self.np.sum(x * x, axis=axis))

    def clip(self, x, lo, hi):
        return self.np.clip(x, lo, hi)

    def cross(self, a, b):
        return self.np.cross(a, b)

    def full(self, shape, v, like=None):
        return self.np.full(shape, v, self.np.float32)

    def zeros(self, shape, like=None, dtype=np.float32):
        return self.np.zeros(shape, dtype)

    def at_set(self, x, idx, v):
        return x.at[idx].set(v)

    def clamp_ste(self, x, lo, hi):
        """x clamped to [lo, hi], gradient zero outside (as MJX's ctrl clamp)."""
        return self.np.clip(x, lo, hi)


class TorchOps:
    name = "torch"

    def __init__(self, device=None):
        import torch
        self.torch = torch
        self.device = torch.device(device or "cuda")
        self._cache = {}
        for f in ("cos", "sin", "sqrt", "exp", "abs", "tanh", "cosh", "log1p", "maximum", "minimum", "where",
                  "stack", "zeros_like", "ones_like", "einsum", "square", "logical_and", "logical_or"):
            setattr(self, f, getattr(torch, f))
        self.arctan2 = torch.atan2

    def const(self, x, dtype=np.float32):
        a = np.asarray(x, dtype)
        k = _key(a)
        if k not in self._cache:
            self._cache[k] = self.torch.as_tensor(a, device=self.device)
        return self._cache[k]

    def index(self, idx):
        """A device index tensor for gathering along an axis (x[:, ops.index(list)])."""
        return self.const(idx, np.int64)

    def concat(self, xs, axis=-1):
        return self.torch.cat(xs, dim=axis)

    def sum(self, x, axis=-1):
        return self.torch.sum(x, dim=axis)

    def mean(self, x, axis=-1):
        return self.torch.mean(x, dim=axis)

    def prod(self, x, axis=-1):
        return self.torch.prod(x, dim=axis)

    def min(self, x, axis=-1):
        return self.torch.amin(x, dim=axis)

    def max(self, x, axis=-1):
        return self.torch.amax(x, dim=axis)

    def norm(self, x, axis=-1):
        return self.torch.sqrt(self.torch.sum(x * x, dim=axis))

    def clip(self, x, lo, hi):
        return self.torch.clamp(x, lo, hi)

    def cross(self, a, b):
        return self.torch.linalg.cross(a, b)

    def full(self, shape, v, like=None):
        return self.torch.full(shape, v, dtype=self.torch.float32, device=self.device)

    def zeros(self, shape, like=None, dtype=np.float32):
        return self.torch.zeros(shape, dtype=getattr(self.torch, np.dtype(dtype).name), device=self.device)

    def at_set(self, x, idx, v):
        y = x.clone()
        y[idx] = v
        return y

    def clamp_ste(self, x, lo, hi):
        return self.torch.clamp(x, lo, hi)

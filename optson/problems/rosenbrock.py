from __future__ import annotations

from typing import List, Optional

from ..call_counter import class_method_call_counter
from ..model import ModelProxy
from ..problem import Problem
from ..vector import Scalar, Vec, vec_sum, zeros_like

try:
    import jax  # type: ignore[import]
    import jax.numpy as jnp  # type: ignore[import]

    JAX = True
except ImportError:
    JAX = False


@class_method_call_counter
class RosenBrockND(Problem):
    """A test funtion to solve the rosenbrock function in arbitrary shapes."""

    def _f(self, x: Vec) -> Scalar:
        return vec_sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    def f(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Scalar:
        assert indices is None
        return self._f(model.x)

    def g(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Vec:
        assert indices is None
        x = model.x

        if JAX and isinstance(x, jnp.ndarray):
            return jax.grad(self._f)(x)
        derivative = zeros_like(x)
        derivative[:-1] += -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])  # type: ignore
        derivative[1:] += 200 * (x[1:] - x[:-1] ** 2)  # type: ignore
        return derivative

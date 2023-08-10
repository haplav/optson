from __future__ import annotations
from typing import List, Optional

from ..call_counter import class_method_call_counter
from ..model import ModelProxy
from ..problem import Problem
from ..vector import Scalar, Vec, as_vec, vec_sum


@class_method_call_counter
class SimpleQuadratic2D(Problem):
    def f(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Scalar:
        assert indices is None
        x = model.x
        return 100 * x[0] ** 2 + 200 * x[1] ** 2

    def g(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Vec:
        assert indices is None
        x = model.x
        return as_vec([200 * x[0], 400 * x[1]])


@class_method_call_counter
class SimpleQuadraticND(Problem):
    def f(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Scalar:
        assert indices is None
        x = model.x
        return vec_sum(x * x)

    def g(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Vec:
        assert indices is None
        x = model.x
        return 2 * x

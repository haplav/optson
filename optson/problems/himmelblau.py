from __future__ import annotations
from typing import List, Optional

import numpy as np

from ..call_counter import class_method_call_counter
from ..model import ModelProxy
from ..problem import Problem
from ..vector import Scalar, Vec, as_vec


@class_method_call_counter
class Himmelblau(Problem):
    """Himmelblau's 2-dimensional function.

    Himmelblau's function is defined as:

    .. math::

        f(x,y)=(x^{2}+y-11)^{2}+(x+y^{2}-7)^{2}

    """

    def f(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Scalar:
        """
        Returns the value of Himmelblau's function at the given coordinates.

        Args:
            model (ModelProxy): The model.
            indices (Optional[List[int]]): The sample indices.

        Returns:
            Scalar: The function value.
        """

        assert indices is None
        x = model.x
        assert isinstance(x, np.ndarray)
        X = x[0]
        Y = x[1]
        return (X**2 + Y - 11) ** 2 + (X + Y**2 - 7) ** 2

    def g(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Vec:
        """
        Returns a numpy.ndarray containing the gradient of Himmelblau's function at the given coordinates.

        Args:
            model (ModelProxy): The model.
            indices (Optional[List[int]]): The sample indices.

        Returns:
            Vec: The gradient vector.
        """

        assert indices is None
        x = model.x
        X = x[0]
        Y = x[1]
        return as_vec(
            [
                2 * (2 * X * (X**2 + Y - 11) + X + Y**2 - 7),
                2 * (X**2 + 2 * Y * (X + Y**2 - 7) + Y - 11),
            ]
        )

from __future__ import annotations
from numpy import dot
from sklearn.datasets import make_regression  # type: ignore
from typing import List, Optional

import numpy as np

from ..call_counter import class_method_call_counter
from ..model import ModelProxy
from ..problem import Problem
from ..vector import Scalar, Vec


@class_method_call_counter
class RegressionProblem(Problem):
    """

    A simple regression problem for testing stochastic algorithms.

    Args:
        n_samples (int, optional): The number of samples. Defaults to 500.
        random_seed (int, optional): The random seed. Defaults to 42.
        n_features (int, optional): The number of features. Defaults to 1.
        n_informative (int, optional): The number of informative features. Defaults to 1.
        noise (float, optional): The noise level. Defaults to 35.0.
    """

    def __init__(
        self,
        n_samples: int = 500,
        random_seed: int = 42,
        n_features: int = 1,
        n_informative: int = 1,
        noise: float = 35.0,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.x, self.y, self.coeff = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            random_state=0,
            noise=noise,
            coef=True,
        )
        self.x = np.c_[np.ones(len(self.x)), self.x]

    def f(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Scalar:
        """
        Misfit for either the mini-batch or the control group for a given model and iteration.
        Iteration refers to which mini-batch or control group should be used and not to the
        vector x necessarily.

        Args:
            model (ModelProxy): The model.
            indices (Optional[List[int]]): The sample indices.

        Returns:
            Scalar: The function value.
        """
        x = model.x
        (x_batch, y_batch) = (
            (self.x, self.y) if indices is None else (self.x[indices], self.y[indices])
        )
        batch_size = len(x_batch)
        hypothesis = dot(x_batch, x)
        loss = hypothesis - y_batch
        return np.sum(loss**2) / (2 * batch_size)

    def g(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Vec:
        """
        Gradient for either the mini-batch or the control group for a given model and iteration.
        Iteration refers to which mini-batch or control group should be used and not to the
        vector x necessarily.

        Args:
            model (ModelProxy): The model.
            indices (Optional[List[int]]): The sample indices.

        Returns:
            Vec: The gradient vector.
        """
        x = model.x
        (x_batch, y_batch) = (
            (self.x, self.y) if indices is None else (self.x[indices], self.y[indices])
        )
        batch_size = len(x_batch)
        hypothesis = dot(x_batch, x)
        loss = hypothesis - y_batch
        return dot(x_batch.T, loss) / batch_size

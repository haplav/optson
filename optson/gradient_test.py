from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from .batch_manager import BatchManager, EmptyBatchManager
from .model import Model
from .problem import Problem
from .utils import InstanceOrType, get_instance
from .vector import InVec, as_vec, dot


class GradientTest:
    """A class that facilitates gradient testing of a `Problem`.

    Args:
        x0 (InVec): The initial model vector.
        h (InVec): A list of step sizes for which to perform the gradient test.
        problem (Problem): The `Problem`.
        batch_manager (InstanceOrType[BatchManager], optional): The batch manager.
            Defaults to :class:`~optson.batch_manager.EmptyBatchManager`.
        verbose (bool, optional): Verbosity. Defaults to False.
    """

    def __init__(
        self,
        x0: InVec,
        h: InVec,
        problem: Problem,
        batch_manager: InstanceOrType[BatchManager] = EmptyBatchManager,
        verbose: bool = False,
    ):
        batch_manager = get_instance(batch_manager)
        self.m = Model.create_initial(
            problem=problem, batch_manager=batch_manager, x0=as_vec(x0)
        )
        self.relative_errors: Deque[float] = deque()
        self.h = as_vec(h)
        self.verbose = verbose
        self.test_completed = False

    def __call__(self) -> None:
        dm = np.ones_like(self.m.x)
        for idx, h_val in enumerate(self.h):
            m_test = self.m.new(x=self.m.x + h_val * dm, radius=h_val)
            lhs = (m_test.fx_cg_prev - self.m.fx_cg) / h_val
            rhs = dot(self.m._problem.preconditioner(self.m.gx_cg), dm)
            rel_error = abs(rhs - lhs) / abs(rhs)
            self.relative_errors.append(rel_error)

            if self.verbose:  # pragma: no cover
                print(f"Test value {idx} out of {len(self.h)}")
                print("Current h value:", h_val)
                print("Predicted change in misfit", h_val * rhs)
                print("Actual change in misfit", m_test.fx_cg_prev - self.m.fx_cg)
                print("All h values", self.h[: idx + 1])
                print("Relative errors", self.relative_errors)
        self.test_completed = True

    def plot(self) -> None:
        """
        Plot the results of the gradient test. Performs the test if not yet performed.
        """
        import matplotlib.pyplot as plt  # type: ignore

        if not self.test_completed:
            self()

        plt.figure()
        plt.loglog(self.h, self.relative_errors, "-*")
        plt.title("Gradient test")
        plt.xlabel("h")
        plt.ylabel("Relative error")

        if plt.get_backend() != "agg":
            plt.show()
        elif self.verbose:
            print("Detected backend `agg`, not calling plt.show().")

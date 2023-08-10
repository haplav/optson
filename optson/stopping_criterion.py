from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Tuple

from .model import Model
from .vector import norm


class StoppingCriterion(ABC):
    """Abstract base class for stopping criteria.

    Args:
        max_iterations (int, optional): The maximum number of parameter updates. Defaults to 100.
        tolerance (float, optional): Tolerance for the minimum gradient norm. Defaults to 1e-4.
        divergence_tolerance (float, optional): Tolerance for the maximum gradient norm. Defaults to 1e8.
        verbose (bool, optional): Verbosity. Defaults to True.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        divergence_tolerance: float = 1e8,
        verbose: bool = True,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.divergence_tolerance = divergence_tolerance
        self.verbose = verbose
        self.initial_misfit = None

    @abstractmethod
    def get_conditions(self, m: Model) -> Iterator[Tuple[bool, str]]:
        """
        An iterator over tuples, each with a stopping criterium and a human-readable message.

        Args:
            m (Model): The model.

        Yields:
            Iterator[Tuple[bool, str]]: A bool and a message.
        """
        raise NotImplementedError

    def _condition_satisfied(self, cond: bool, msg: str) -> bool:
        if cond:
            if self.verbose:
                print(f"{msg}. Stopping now")
            return True
        return False

    def __call__(self, m: Model) -> bool:
        """Evaluate all stopping criteria.

        Args:
            m (Model): The model.

        Returns:
            bool: True if any condition is satisfied, otherwise False.
        """
        return any(
            self._condition_satisfied(condition, msg)
            for condition, msg in self.get_conditions(m)
        )


class BasicStoppingCriterion(StoppingCriterion):
    """A basic stopping criterion.

    Args:
        max_iterations (int, optional): The maximum number of parameter updates. Defaults to 100.
        tolerance (float, optional): Tolerance for the minimum gradient norm. Defaults to 1e-4.
        divergence_tolerance (float, optional): Tolerance for the maximum gradient norm. Defaults to 1e8.
        verbose (bool, optional): Verbosity. Defaults to True.
    """

    def get_conditions(self, m: Model) -> Iterator[Tuple[bool, str]]:
        yield (m.iteration >= self.max_iterations, "Max number of iterations reached")
        norm_gx = float(norm(m.gx))
        yield (
            norm_gx < self.tolerance,
            f"Gradient norm below tolerance {self.tolerance}",
        )
        yield (
            norm_gx > self.divergence_tolerance,
            f"Gradient norm above divergence tolerance {self.divergence_tolerance}",
        )

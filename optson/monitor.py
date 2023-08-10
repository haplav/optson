from __future__ import annotations

from abc import ABC, abstractmethod

from .model import Model
from .vector import norm


class Monitor(ABC):
    """Abstract base class for implementations of a `Monitor`.

    Args:
        step (int, optional): The number of iterations
            between prints. Defaults to 1.
    """

    def __init__(self, step: int = 1):
        self.step = step

    @abstractmethod
    def __call__(self, m: Model) -> None:
        raise NotImplementedError()


class EmptyMonitor(Monitor):
    """An Empty monitor that does nothing."""

    def __call__(self, m: Model) -> None:
        pass


class BasicMonitor(Monitor):
    """A basic monitor that prints the gradient norm at each step.

    Args:
        step (int, optional): The number of iterations
            between prints. Defaults to 1.
    """

    def __call__(self, m: Model) -> None:
        if m.iteration % self.step == 0:
            print(f"{m.iteration:3d} {norm(m.gx):.8e}")

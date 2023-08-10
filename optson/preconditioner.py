from __future__ import annotations

from abc import ABC, abstractmethod

from .vector import Vec


class Preconditioner(ABC):
    @abstractmethod
    def __call__(self, x: Vec) -> Vec:
        """Apply a preconditioner to x and return the preconditioned result.

        Args:
            x (Vec): The input vector

        Returns:
            Vec: The preconditioned result.
        """
        raise NotImplementedError


class IdentityPreconditioner(Preconditioner):
    """
    This is essentially an empty preconditioner that returns the same output as the input.

    """

    def __call__(self, x: Vec) -> Vec:
        """Returns x

        Args:
            x (Vec): the input vector

        Returns:
            Vec: the output vector, identical to the input
        """
        return x

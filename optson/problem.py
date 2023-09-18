from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from optson.model import ModelProxy

from .hessian import Hessian, LBFGSHessian
from .preconditioner import Preconditioner, IdentityPreconditioner
from .utils import InstanceOrType, get_instance
from .vector import Scalar, Vec


class Problem(ABC):
    """
    Abstract base class for user-defined implementations of `Problem`.
    """

    def __init__(
        self,
        H: InstanceOrType[Hessian] = LBFGSHessian,
        preconditioner: InstanceOrType[Preconditioner] = IdentityPreconditioner,
    ):
        super().__init__()
        self.H = get_instance(H)
        self.preconditioner = get_instance(preconditioner)

        self.call_counters: Dict[str, int] = {}

    @abstractmethod
    def f(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Scalar:
        """Compute the function of the objective function. In seismology, this is typically referred to as the misfit function,
        in machine learning this is the loss function.

        Args:
            model (ModelProxy): The ModelProxy object.
            indices (typing.Optional[typing.List[int]]): A list of indices in the case of stochastic optimization,
                otherwise None.

        Returns:
            Scalar: The function value.
        """
        raise NotImplementedError

    @abstractmethod
    def g(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Vec:
        """Compute gradient of the model vector with respect to the objective function.
        Implementor can rely on the fact that f() is evaluated prior to g() - see `Model.gx`.

        Args:
            model (ModelProxy): The ModelProxy object.
            indices (typing.Optional[typing.List[int]]): A list of indices in the case of stochastic optimization,
                otherwise None.

        Returns:
            Vec: The gradient vector.
        """

        raise NotImplementedError


class DummyProblem(Problem):
    """An empty dummy problem. Not relevant for users of **Optson**."""

    def f(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Scalar:
        raise NotImplementedError

    def g(
        self,
        model: ModelProxy,
        indices: Optional[List[int]],
    ) -> Vec:
        raise NotImplementedError

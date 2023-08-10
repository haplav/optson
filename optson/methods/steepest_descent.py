from __future__ import annotations

from ..ls import LSStepsize, BacktrackingLSStepsize, SteepestDescentLSDirection
from ..update import LSUpdate
from ..utils import InstanceOrType, get_instance


class SteepestDescentUpdate(LSUpdate):
    """
    A standard steepest descent algorithm. This method can be used
    with both non-stochastic methods and overlapping mini-batches.

    Args:
        initial_step_size (float, optional): The initial step size. Defaults to 1.0.
        max_step_size (float, optional): The maximum step size. Defaults to float("inf").
        max_iterations (int, optional): The maximum number of iterations in the line-search. Defaults to 100.
        initial_step_as_percentage (bool, optional): Initial step as a percentage of the model. Defaults to False.
        stepsize (InstanceOrType[LSStepsize], optional): A step size algorithm.
            Defaults to :class:`~optson.ls.BacktrackingLSStepsize`.
        verbose (bool, optional): Verbosity. Defaults to False.
    """

    def __init__(
        self,
        initial_step_size: float = 1.0,
        max_step_size: float = float("inf"),
        max_iterations: int = 100,
        initial_step_as_percentage: bool = False,
        stepsize: InstanceOrType[LSStepsize] = BacktrackingLSStepsize,
        verbose: bool = False,
    ):
        super().__init__(
            direction=SteepestDescentLSDirection(verbose=verbose),
            stepsize=get_instance(
                stepsize,
                max_iterations=max_iterations,
                initial=initial_step_size,
                max_step_size=max_step_size,
                initial_step_as_percentage=initial_step_as_percentage,
                verbose=verbose,
            ),
            verbose=verbose,
        )

    @property
    def needs_H(self) -> bool:
        """Evaluates to False, since no Hessian is needed for steepest descent.

        Returns:
            bool: False
        """
        return False

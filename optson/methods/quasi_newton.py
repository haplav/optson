from __future__ import annotations

from ..ls import LSStepSize, BacktrackingLSStepSize, QuasiNewtonLSDirection
from ..update import LSUpdate
from ..utils import InstanceOrType, get_instance


class QuasiNewtonUpdate(LSUpdate):
    """
    Quasi-Newton update. If all default settings are used and no custom Hessian is passed,
    it will use L-BFGS with backtracking line-search.

    Args:
        initial_step_size (float, optional): The initial step size. Defaults to 1.0.
        max_step_size (float, optional): The maximum step size. Defaults to 1.0.
        max_iterations (int, optional): The maximum number of line-search iterations. Defaults to 100.
        initial_step_as_percentage (bool, optional): Define the initial step size as a percentage. Defaults to False.
        verbose (bool, optional): Verbosity. Defaults to False.
        step_size (InstanceOrType[LSStepSize], optional): The step size algorithm to use.
            Defaults to :class:`~optson.ls.BacktrackingLSStepSize`.
    """

    def __init__(
        self,
        initial_step_size: float = 1.0,
        max_step_size: float = 1.0,
        max_iterations: int = 100,
        initial_step_as_percentage: bool = False,
        verbose: bool = False,
        step_size: InstanceOrType[LSStepSize] = BacktrackingLSStepSize,
    ):
        # TODO: This method may also need a fallback.
        # A max step size of 1.0 only makes sense in the context where p is H_inv g,
        # but not when the search direction is the steepest descent direction.
        super().__init__(
            direction=QuasiNewtonLSDirection(verbose=verbose),
            step_size=get_instance(
                step_size,
                max_iterations=max_iterations,
                initial=initial_step_size,
                max_step_size=max_step_size,
                initial_step_as_percentage=initial_step_as_percentage,
                verbose=verbose,
            ),
            verbose=verbose,
        )

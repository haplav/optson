from __future__ import annotations

from ..methods.steepest_descent import SteepestDescentUpdate
from ..tr import TRRadius, TRStep, BasicTRRadius, DogLegTRStep
from ..update import TRUpdate, Update
from ..utils import InstanceOrType, get_instance


class BasicTRUpdate(TRUpdate):
    """
    A trust-region update algorithm. If all default options are used and the default `Hessian` is used,
    this will use Trust-Region L-BFGS, using the Dogleg algorithm to solve the trust-region subproblem.

    Args:
        max_rejected (int, optional): Maximum number of rejected model updates before using the fallback. Defaults to 5.
        tr_step (InstanceOrType[TRStep], optional): The class/object that computes
            the trust-region step. Defaults to :class:`~optson.tr.DogLegTRStep`.
        tr_radius (InstanceOrType[TRRadius], optional): The algorithm that controls
            the trust-region radius. Defaults to :class:`~optson.tr.BasicTRRadius`.
        fallback (InstanceOrType[Update], optional): The fallback algorithm
            if e.g. no curvature information is available.
            Defaults to :class:`~optson.methods.steepest_descent.SteepestDescentUpdate`.
        verbose (bool, optional): Verbosity. Defaults to False.
    """

    def __init__(
        self,
        max_rejected: int = 5,
        tr_step: InstanceOrType[TRStep] = DogLegTRStep,
        tr_radius: InstanceOrType[TRRadius] = BasicTRRadius,
        fallback: InstanceOrType[Update] = SteepestDescentUpdate,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            tr_step=get_instance(tr_step, verbose=verbose),
            tr_radius=get_instance(tr_radius, verbose=verbose),
            fallback=get_instance(fallback, verbose=verbose),
            max_rejected=max_rejected,
            verbose=verbose,
        )

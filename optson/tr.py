from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .model import Model
from .vector import Scalar, dot, norm


class TRRadius(ABC):
    """
    Abstract base class for classes that update the trust-region radius.

    Args:
        initial (float, optional): The initial trust-region radius. Defaults to 1.0.
        maximum (float, optional): The maximum trust-region radius. Defaults to float("inf").
        verbose (bool, optional): Verbosity. Defaults to False.
    """

    def __init__(
        self,
        initial: float = 1.0,
        maximum: float = float("inf"),
        verbose: bool = False,
    ):
        self.initial = initial
        self.maximum = maximum
        self.verbose = verbose
        self.current: Optional[float] = None

    def __call__(self) -> float:
        if self.current is None:
            self.current = self.initial
        return self.current

    @abstractmethod
    def update(self, m: Model, m_trial: Model) -> bool:
        raise NotImplementedError


class TRStep(ABC):
    """
    Abstract base class for derived classes that solve the trust-region subproblem.
    """

    @abstractmethod
    def __call__(self, m: Model, tr_radius: float) -> Model:
        raise NotImplementedError()


# Basic TR Radius manager.
class BasicTRRadius(TRRadius):
    def __init__(
        self,
        initial: float = 1.0,
        maximum: float = 1e200,
        verbose: bool = False,
        eta: float = 1e-4,
    ):
        super().__init__(initial=initial, maximum=maximum, verbose=verbose)
        self.eta = eta

    # TODO better name? check_model_and_update_radius?
    def update(self, m: Model, m_trial: Model) -> bool:
        """
        Algorithm 4.1 from Nocedal.

        This function decides whether a model gets accepted or not. In addition,
        it either shrinks or grows the trust-region radius.
        """
        self.__call__()
        P = m._problem.preconditioner
        p = m_trial.p

        assert self.current is not None
        assert p is not None

        B_p = m_trial.H_p
        pred = -dot(P(m.gx_cg), p) - 0.5 * dot(P(p), B_p)
        ared = m.fx_cg - m_trial.fx_cg_prev

        if self.verbose:
            print("PRED:", pred, "ARED:", ared)

        if pred <= 0.0:
            # NOTE: This should generally not occur, at least in the stochastic case.
            # since we already check for a positive pred in Model._find_gx_cg.
            if self.verbose:
                print("Predicted a negative misfit change.")
            m._problem.H.reset()
            return False

        rho = ared / pred
        norm_p = norm(p)
        epsilon = 0.1 * norm_p  # TODO: check if this is an okay choice
        if rho < 0.25:
            self.current *= 0.25
        elif rho > 0.75 and abs(norm_p - self.current) < epsilon:
            self.current = min(2 * self.current, self.maximum)
        return bool(rho > self.eta)


class DogLegTRStep(TRStep):
    """
    This class solves the trust-region subproblem using the Dogleg method to compute the
    step for a given trust-region radius.

    Args:
        verbose (bool, optional): Verbosity. Defaults to False.
    """

    def __init__(self, verbose: bool = False):
        self.step_type: str = ""
        self.verbose = verbose

    def __call__(self, m: Model, tr_radius: float) -> Model:
        """
        Returns a trial-model, given a model and a trust-region radius.

        Args:
            m (Model): The current model.
            tr_radius (float): The trust-region radius.

        Returns:
            Model: The trial model.
        """
        self.step_type = ""
        tau: Optional[Scalar] = None

        p_B = -m.Hinv_gx
        norm_p_B = norm(p_B)
        if norm_p_B <= tr_radius:
            self.step_type = "full"
            p = p_B
        else:
            p_U = -dot(m.gx, m.gx) / dot(m.gx, m.H_gx) * m.gx
            # TODO: improve typing of dot to avoid having to ignore.
            dot_p_U = dot(p_U, p_U)  # type: ignore[arg-type]
            norm_p_U = np.sqrt(dot_p_U)

            if norm_p_U >= tr_radius:  # tau between 0 and 1
                self.step_type = "gradient"
                tau = tr_radius / norm_p_U
                p = tau * p_U
            else:
                # Find the solution to the scalar quadratic equation.
                # Compute the intersection of the trust region boundary
                # and the line segment connecting the Cauchy and Newton points.
                # This requires solving a quadratic equation.
                # ||p_u + tau*(p_b - p_u)||**2 == trust_radius**2
                # Solve this for positive time t using the quadratic formula.
                # From https://sudonull.com/post/68061-Optimization-method-Trust-Region-DOGLEG-Python-implementation-example
                self.step_type = "dogleg"
                pB_pU = p_B - p_U
                dot_pB_pU = dot(pB_pU, pB_pU)
                dot_pU_pB_pU = dot(p_U, pB_pU)  # type: ignore[arg-type]
                fact = dot_pU_pB_pU**2 - dot_pB_pU * (dot_p_U - tr_radius**2)
                tau = (-dot_pU_pB_pU + np.sqrt(fact)) / dot_pB_pU
                p = p_U + tau * (p_B - p_U)  # type: ignore
                tau += 1.0  # print proper tau value according to Nocedal's algorithm

        assert self.step_type
        if self.verbose:
            print(f"Proposed step type: {self.step_type}", end="")
            if tau is None:
                print()
            else:
                print(f"; tau = {tau:.4f}")

        m_trial = m.new(m.x, p, radius=tr_radius)
        if p is p_B:
            # p = -(Hinv * gx)  =>  H * p = -gx
            m_trial._set_H_p(-m.gx)

        return m_trial

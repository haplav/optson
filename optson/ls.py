from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .model import Model
from .scalar_optimizer import ScalarOptimizer, GoldenRatioScalarOptimizer
from .utils import InstanceOrType, get_instance
from .vector import Scalar, Vec, dot


class LSDirection(ABC):
    """
    Abstract base class for search directions.

    Args:
        verbose (bool, optional): Verbosity. Defaults to False.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @abstractmethod
    def __call__(self, m: Model) -> Vec:
        """Returns the search direction for model m.

        Args:
            m (Model): The model.

        Returns:
            Vec: The search direction.
        """
        raise NotImplementedError()


class LSStepsize(ABC):
    """Abstract base class for line-search algorithms.

    Args:
        max_iterations (int, optional): The maximum number of line search iterations. Defaults to 100.
        initial (float, optional): The initial step size. Defaults to 1.0.
        max_step_size (float, optional): The max step size. Defaults to float("inf").
        initial_step_as_percentage (bool, optional): Initial step as percentage of the model. Defaults to False.
        verbose (bool, optional): Verbosity. Defaults to False.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        initial: float = 1.0,
        max_step_size: float = float("inf"),
        initial_step_as_percentage: bool = False,
        verbose: bool = False,
    ):
        self.max_iterations = max_iterations
        self._initial = initial
        self.max_step_size = max_step_size
        self.initial_step_as_percentage = initial_step_as_percentage
        self.verbose = verbose

        self.current: Optional[float] = None

    @abstractmethod
    def __call__(self, m: Model, p: Vec) -> Model:
        """Perform the line-search.

        Args:
            m (Model): The model.
            p (Vec): The search direction.

        Returns:
            Model: The updated `Model`.
        """
        raise NotImplementedError

    def get_initial(self, m: Model) -> float:
        """Get the initial step.

        Args:
            m (Model): The model.

        Returns:
            float: The step length.
        """
        if self.current is None:
            if self.initial_step_as_percentage:
                self.current = self._initial / max(abs(m.gx.flatten() / m.x.flatten()))  # type: ignore [call-overload]
            else:
                self.current = self._initial
        return self.current

    @property
    def nIterations(self) -> int:
        """
        Placeholder for the number of line-search iterations.

        Returns:
            int: The number of iterations.
        """
        raise NotImplementedError

    @property
    def nAllIterations(self) -> int:
        """
        Placeholder for the total number of line-search iterations among multiple successful model updates.

        Returns:
            int: The number of iterations.
        """
        raise NotImplementedError

    @property
    def nFuncEval(self) -> int:
        """
        Placeholder for the number of function evaluations.

        Returns:
            int: The number of function evaluations.
        """
        raise NotImplementedError

    @property
    def nAllFuncEval(self) -> int:
        """
        Placeholder for the total number of function evaluations among many model updates.

        Returns:
            int: The total number of function evaluations.
        """
        raise NotImplementedError


class SteepestDescentLSDirection(LSDirection):
    """Steepest descent direction."""

    def __call__(self, m: Model) -> Vec:
        """Get the steepest descent direction.

        Args:
            m (Model): The model.

        Returns:
            Vec: The direction.
        """
        return -m.gx


class QuasiNewtonLSDirection(LSDirection):
    def __call__(self, m: Model) -> Vec:
        """
        Get the Quasi-Newton direction. This direction falls back to steepest descent if
        no curvature information is available.

        Args:
            m (Model): The model.

        Returns:
            Vec: The Quasi-Newton direction.
        """
        if m._problem.H.ready():
            p = -m.Hinv_gx
            if dot(p, m.gx) > 0.0:  # Ensure descent direction.
                m._problem.H.reset()
                if self.verbose:
                    print(
                        "Quasi-Newton direction is not a descent direction. Reverting to steepest descent."
                    )
                p = -m.gx
        else:
            if self.verbose:
                print("Hessian not ready. Reverting to steepest descent.")
            p = -m.gx
        return p


class ConstantLSStepsize(LSStepsize):
    def __init__(
        self,
        initial: float = 1e-3,
        initial_step_as_percentage: bool = False,
    ):
        """
        A constant step size, used in combination with e.g. Adam.

        Args:
            initial (float, optional): The initial step size. Defaults to 1e-3.
            initial_step_as_percentage (bool, optional): Initial step size as a max percentage of the model. Defaults to False.
        """
        super().__init__(
            max_iterations=1,
            initial=initial,
            max_step_size=initial,
            initial_step_as_percentage=initial_step_as_percentage,
        )

    def __call__(self, m: Model, p: Vec) -> Model:
        """Get the updated model.

        Args:
            m (Model): The current `Model`.
            p (Vec): The search `Direction`.

        Returns:
            Model: The updated `Model`.
        """
        alpha = self.get_initial(m)
        return m.new(x=m.x + alpha * p).accept()


class BacktrackingLSStepsize(LSStepsize):
    def __init__(
        self,
        max_iterations: int = 100,
        initial: float = 1.0,
        max_step_size: float = float("inf"),
        initial_step_as_percentage: bool = False,
        verbose: bool = False,
        rho: float = 0.5,
        c1: float = 1e-4,
    ):
        """Backtracking line-search algorithm based on the algorithm provided in the Numerical Optimization by Nocedal.

        Args:
            max_iterations (int, optional): The maximum number of line-search iterations. Defaults to 100.
            initial (float, optional): The initial step length. Defaults to 1.0.
            max_step_size (float, optional): The maximum step size. Defaults to float("inf").
            initial_step_as_percentage (bool, optional): Initial step as a max percentage change in the model.
                Defaults to False.
            verbose (bool, optional): Verbosity. Defaults to False.
            rho (float, optional): Determines how quickly back-tracking occurs. Defaults to 0.5.
            c1 (float, optional): The constant for the sufficient decrease condtion or Wolfe I. Defaults to 1e-4.
        """
        super().__init__(
            max_iterations=max_iterations,
            initial=initial,
            max_step_size=max_step_size,
            initial_step_as_percentage=initial_step_as_percentage,
            verbose=verbose,
        )
        self.rho = rho
        self.c1 = c1

        self._nIterations = 0
        self._nAllIterations = 0
        self._nFuncEval = 0
        self._nAllFuncEval = 0

    def __call__(self, m: Model, p: Vec) -> Model:
        alpha = self.get_initial(m)
        c1, rho = self.c1, self.rho
        nEval = 1
        gTp = dot(m.gx_cg, p)
        l = None
        for i in range(self.max_iterations):
            m_trial = m.new(x=m.x + alpha * p, radius=alpha)
            l = m.fx_cg + c1 * alpha * gTp
            phi = m_trial.fx_cg_prev
            self._monitor(i, nEval, phi, l, alpha)
            if phi <= l:
                self.current = min(alpha * 2.0, self.max_step_size)
                m_trial.accept()
                break
            alpha *= rho
            nEval += 1
        self._nFuncEval = nEval
        self._nIterations = i
        self._nAllFuncEval += self._nFuncEval
        self._nAllIterations += i
        return m_trial

    @property
    def nIterations(self) -> int:
        """The number of line-search iterations for the current model.

        Returns:
            int: The number of iterations.
        """
        return self._nIterations

    @property
    def nAllIterations(self) -> int:
        """The total number of line-search iterations among potentially multiple successful model updates.

        Returns:
            int: The number of iterations.
        """
        return self._nAllIterations

    @property
    def nFuncEval(self) -> int:
        """
        The number of function evaluations among potentially multiple model updates.

        Returns:
            int: The total number of function evaluations.
        """
        return self._nFuncEval

    @property
    def nAllFuncEval(self) -> int:
        """
        The total number of function evaluations among potentially multiple model updates.

        Returns:
            int: The total number of function evaluations.
        """
        return self._nAllFuncEval

    def _monitor(
        self, i: int, nEval: int, phi: Scalar, l: Scalar, alpha: Scalar
    ) -> None:
        if not self.verbose:
            return
        print(
            f"    {i:3d} {nEval:3d} L phi(x) = {float(phi):+.8e} {' >' if phi > l else '<='} "
            f"l(x) = {float(l):+.8e} alpha = {float(alpha):+.8e}"
        )


class ScalarOptimizerLSStepsize(LSStepsize):
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        initial: float = 1.0,
        max_step_size: float = 2.0,
        initial_step_as_percentage: bool = False,
        verbose: bool = False,
        scalar_optimizer: InstanceOrType[ScalarOptimizer] = GoldenRatioScalarOptimizer,
    ):
        super().__init__(
            initial=initial,
            max_step_size=max_step_size,
            max_iterations=max_iterations,
            initial_step_as_percentage=initial_step_as_percentage,
            verbose=verbose,
        )
        self.scalar_optimizer = get_instance(
            scalar_optimizer,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=verbose,
        )
        self.m_trial: Optional[Model] = None

    def __call__(self, m: Model, p: Vec) -> Model:
        def f(alpha: float) -> float:
            self.m_trial = m.new(x=m.x, p=alpha * p, radius=alpha)
            f_alpha = self.m_trial.fx_cg_prev
            assert isinstance(f_alpha, float)
            return f_alpha

        self.m_trial = None
        a = 0.0
        b = min(2.0 * self.get_initial(m), self.max_step_size)
        _ = self.scalar_optimizer(f, (a, b))
        m_new = self.m_trial
        assert isinstance(m_new, Model)
        if m_new.fx_cg_prev < m.fx_cg:
            m_new.accept()
        return m_new

    @property
    def nIterations(self) -> int:
        return self.scalar_optimizer.nIterations

    @property
    def nAllIterations(self) -> int:
        return self.scalar_optimizer.nAllIterations

    @property
    def nFuncEval(self) -> int:
        return self.scalar_optimizer.nFuncEval

    @property
    def nAllFuncEval(self) -> int:
        return self.scalar_optimizer.nAllFuncEval

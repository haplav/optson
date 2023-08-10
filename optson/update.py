from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod
from typing import Union

import h5py

from .ls import LSDirection, LSStepsize
from .model import Model
from .tr import TRRadius, TRStep
from .vector import Vec, dot, norm


class Update(ABC):
    """
    An abstract base class for implementations of `Update`.
    """

    @abstractmethod
    def __call__(self, m: Model) -> Model:
        """Update the Model m.

        Args:
            m (Model): The model to be updated.

        Returns:
            Model: The updated model.
        """
        raise NotImplementedError

    def store_attributes(
        self, state_file: Union[pathlib.Path, str, None] = None
    ) -> None:
        """Store attributes of the update instance to a state file.

        Args:
            state_file (Union[pathlib.Path, str, None], optional): The state file. Defaults to None.
        """
        pass

    def get_attributes(
        self,
        target_vec: Vec,
        state_file: Union[pathlib.Path, str, None] = None,
    ) -> None:
        """Get attributes of the update instance from a state file.

        Args:
            state_file (Union[pathlib.Path, str, None], optional): The state file. Defaults to None.
        """
        pass

    @property
    def needs_H(self) -> bool:
        """Property that informs `Model` if the Hessian needs to be updated, as is for example the case in L-BFGS.
        This defaults to true if not defined by the derived class.

        Returns:
            bool: True in this case.
        """
        return True


class LSUpdate(Update):
    """An abstract base class for line-search methods.

    Args:
        direction (LSDirection): The algorithm that defines the search direction.
        stepsize (LSStepsize): The algorithm that defines the step sizes.
        verbose (bool, optional): Verbosity. Defaults to False.
    """

    def __init__(
        self,
        direction: LSDirection,
        stepsize: LSStepsize,
        verbose: bool = False,
    ):
        self.direction = direction
        self.stepsize = stepsize
        self.verbose = verbose

    def __call__(self, m: Model) -> Model:
        """Update the Model m.

        Args:
            m (Model): The model to be updated.

        Returns:
            Model: The updated model.
        """
        p = self.direction(m)
        return self.stepsize(m, p)

    def store_attributes(
        self, state_file: Union[pathlib.Path, str, None] = None
    ) -> None:
        if not state_file:
            return

        with h5py.File(state_file, "a") as f:
            if "LineSearch" not in f.keys():
                f.create_group("LineSearch")
            if hasattr(self.stepsize, "current") and self.stepsize.current is not None:
                f["LineSearch"].attrs["current"] = self.stepsize.current

    def get_attributes(
        self, target_vec: Vec, state_file: Union[pathlib.Path, str, None] = None
    ) -> None:
        if not state_file or not os.path.exists(state_file):
            return

        with h5py.File(state_file, "r") as f:
            if "LineSearch" in f.keys() and "current" in f["LineSearch"].attrs:
                self.stepsize.current = f["LineSearch"].attrs["current"]


class TRUpdate(Update):
    """
    Abstract base class for trust-region updating.

    Args:
        tr_step (TRStep): The TRStep.
        tr_radius (TRRadius): The TRRadius.
        fallback (Update): The fallback method.
        max_rejected (int, optional): The maximum number of rejected models. Defaults to 5.
        verbose (bool, optional): Verbosity. Defaults to False.
    """

    def __init__(
        self,
        tr_step: TRStep,
        tr_radius: TRRadius,
        fallback: Update,
        max_rejected: int = 5,
        verbose: bool = False,
    ) -> None:
        self.tr_step = tr_step
        self.tr_radius = tr_radius
        self.fallback = fallback
        self.max_rejected = max_rejected
        self.verbose = verbose

    def _tr_step_is_descent_direction(self, m: Model, m_trial: Model) -> bool:
        p = m_trial.p
        assert p is not None
        if dot(p, m.gx) > 0.0:
            if self.verbose:
                print(
                    "The proposed trust_region step is unlikely to be a descent direction. Reverting to fallback."
                )
            return False
        return True

    def __call__(self, m: Model) -> Model:
        n_rejected = 0
        while True:
            if not m._problem.H.ready() or n_rejected >= self.max_rejected:
                m_trial = self.fallback(m)
                if self.tr_radius.current is None:
                    self.tr_radius.current = 2.0 * float(norm(m_trial.x - m.x))
                break
            else:
                assert isinstance(self.tr_radius(), float)
                m_trial = self.tr_step(m, self.tr_radius())
                if not self._tr_step_is_descent_direction(m, m_trial):
                    m._problem.H.reset()
                    continue
                if self.tr_radius.update(m, m_trial):
                    m_trial.accept()
                    break
                else:
                    n_rejected += 1

        return m_trial

    def store_attributes(
        self, state_file: Union[pathlib.Path, str, None] = None
    ) -> None:
        if not state_file:
            return

        with h5py.File(state_file, "a") as f:
            if "TrustRegion" not in f.keys():
                f.create_group("TrustRegion")
            f["TrustRegion"].attrs["TrRadius"] = self.tr_radius()

    def get_attributes(
        self, target_vec: Vec, state_file: Union[pathlib.Path, str, None] = None
    ) -> None:
        if not state_file or not os.path.exists(state_file):
            return

        with h5py.File(state_file, "r") as f:
            if "TrustRegion" in f.keys():
                self.tr_radius.initial = f["TrustRegion"].attrs["TrRadius"]

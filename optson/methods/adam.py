from __future__ import annotations

import os
import pathlib
from typing import Optional, Union

import h5py

from ..ls import LSDirection, ConstantLSStepsize
from ..model import Model
from ..preconditioner import Preconditioner, IdentityPreconditioner
from ..update import LSUpdate
from ..utils import InstanceOrType, get_instance
from ..vector import Vec, as_target_vec, as_vec, median, zeros_like


class AdamLSDirection(LSDirection):
    """Computes the Adam search direction as detailed in the original Adam paper: https://arxiv.org/abs/1412.6980

    Args:
        beta_1 (float, optional): The first momentum strength. Defaults to 0.9.
        beta_2 (float, optional): The second momentum strength. Defaults to 0.999.
        epsilon (float, optional): epsilon. Defaults to 1e-8.
        relative_epsilon (bool, optional): Compute epsilon as a fraction of the median `v_t`. Defaults to False.
        verbose (bool, optional): Verbosity. Defaults to False.
    """

    def __init__(
        self,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        relative_epsilon: bool = False,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.relative_epsilon = relative_epsilon

        self.t: Optional[int] = None
        self.vt: Optional[Vec] = None
        self.mt: Optional[Vec] = None

    def __call__(self, m: Model) -> Vec:
        """Compute the direction for Model m.

        Args:
            m (Model): The model.

        Returns:
            Vec: The direction.
        """
        gt = as_vec(m.gx)

        if self.t is None or self.mt is None or self.vt is None:
            self.mt = zeros_like(gt)
            self.vt = zeros_like(gt)
            self.t = 0

        self.t += 1
        assert self.mt is not None
        assert self.vt is not None
        self.mt = self.beta_1 * self.mt + (1 - self.beta_1) * gt
        self.vt = self.beta_2 * self.vt + (1 - self.beta_2) * gt**2

        self.mt_hat = self.mt / (1 - self.beta_1**self.t)
        self.vt_hat = self.vt / (1 - self.beta_2**self.t)
        if self.relative_epsilon:
            epsilon = median(self.vt_hat) * self.epsilon
        else:
            epsilon = self.epsilon
        return -self.mt_hat / ((self.vt_hat**0.5) + epsilon)


class AdamUpdate(LSUpdate):
    """The Adam update algorithm.

    Args:
        beta_1 (float, optional): The first momentum strength. Defaults to 0.9.
        beta_2 (float, optional): The second momentum strength. Defaults to 0.999.
        epsilon (float, optional): Parameter epsilon. Defaults to 1e-8.
        alpha (float, optional): Parameter alpha. Defaults to 1.0.
        relative_epsilon (bool, optional): Compute epsilon as a fraction of the median `v_t`. Defaults to False.
        preconditioner (InstanceOrType[Preconditioner], optional): The preconditioner
            applied to the adam update. Defaults to :class:`~optson.preconditioner.IdentityPreconditioner`.
        verbose (bool, optional): Verbosity. Defaults to False.
    """

    _MNAME = "Adam"

    def __init__(
        self,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        alpha: float = 1.0,
        relative_epsilon: bool = False,
        preconditioner: InstanceOrType[Preconditioner] = IdentityPreconditioner,
        verbose: bool = False,
    ):
        super().__init__(
            direction=AdamLSDirection(
                beta_1, beta_2, epsilon, relative_epsilon, verbose=verbose
            ),
            stepsize=ConstantLSStepsize(alpha),
            verbose=verbose,
        )
        self.preconditioner = get_instance(preconditioner)

    def __call__(self, m: Model) -> Model:
        """Return the updated model

        Args:
            m (Model): The model to be updated.

        Returns:
            Model: The next (updated) model
        """
        p = self.preconditioner(self.direction(m))
        return self.stepsize(m, p)

    def store_attributes(
        self, state_file: Union[pathlib.Path, str, None] = None
    ) -> None:
        """Store the attributes to a state file.

        Args:
            state_file (Union[pathlib.Path, str, None], optional): The state file. Defaults to None.
        """
        if not state_file:
            return
        assert isinstance(self.direction, AdamLSDirection)  # make mypy happy
        with h5py.File(state_file, "a") as f:
            if self._MNAME not in f.keys():
                f.create_group(self._MNAME)
            f[self._MNAME].attrs["t"] = self.direction.t

            if "mt" not in f[self._MNAME]:
                f[self._MNAME].create_dataset("mt", data=self.direction.mt)
            else:
                f[self._MNAME]["mt"][:] = self.direction.mt

            if "vt" not in f[self._MNAME]:
                f[self._MNAME].create_dataset("vt", data=self.direction.vt)
            else:
                f[self._MNAME]["vt"][:] = self.direction.vt

    def get_attributes(
        self, target_vec: Vec, state_file: Union[pathlib.Path, str, None] = None
    ) -> None:
        """Get the attributes from a state file.

        Args:
            target_vec (Vec): The target vec, used for matching the datatype.
            state_file (Union[pathlib.Path, str, None], optional): The state file. Defaults to None.
        """
        if not state_file or not os.path.exists(state_file):
            return

        assert isinstance(self.direction, AdamLSDirection)  # make mypy happy
        with h5py.File(state_file, "r") as f:
            if self._MNAME in f.keys():
                self.direction.t = f[self._MNAME].attrs["t"]
                self.direction.mt = as_target_vec(f[self._MNAME]["mt"][:], target_vec)
                self.direction.vt = as_target_vec(f[self._MNAME]["vt"][:], target_vec)

    @property
    def needs_H(self) -> bool:
        """Evaluates to False. Since no Hessian is needed for Adam.

        Returns:
            bool: False
        """
        return False

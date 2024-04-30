from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Union

import h5py
import numpy as np

from .call_counter import class_method_call_counter
from .vector import (
    Scalar,
    Vec,
    array_to_deque,
    as_target_vec,
    as_vec,
    copy_vec,
    deque_to_numpy_array,
    dot,
    zeros_like,
)


class Hessian(ABC):
    """Abstract base class for implementations of the `Hessian`"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def apply(self, v: Vec) -> Vec:
        """Apply the Hessian to vector v.

        Args:
            v (Vec): The vector to which the Hessian is applied.


        Returns:
            Vec: The result.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_inverse(self, v: Vec) -> Vec:
        """Apply the inverse hessian to vector v.

        Args:
            v (Vec): The vector.

        Returns:
            Vec: The result.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the curvature information."""
        pass

    def ready(self) -> bool:
        """Is the Hessian ready for use?

        Returns:
            bool: True if the Hessian is ready for use. Otherwise False.
        """
        return True

    def update(self, s_update: Vec, y_update: Vec) -> bool:
        """Update the curvature information.

        Args:
            s_update (Vec): The model difference.
            y_update (Vec): The gradient difference.

        Returns:
            bool: True if successful, otherwise False.
        """
        return True

    def save(self, state_file: Union[pathlib.Path, str, None] = None) -> None:
        """Save the state of the curvature information.

        Args:
            state_file (Union[pathlib.Path, str, None], optional): The state file.
                Defaults to None.
        """
        pass

    def read(
        self,
        target_vec: Vec,
        state_file: Union[pathlib.Path, str, None] = None,
    ) -> None:
        """Read the curvature information from a state file.

        Args:
            target_vec (Vec): An example Vec to infer the datatype needed.
            state_file (Union[pathlib.Path, str, None], optional): The state file.
                Defaults to None.
        """
        pass


@class_method_call_counter
class LBFGSHessian(Hessian):
    """
    Implementation of the LBFGS algorithm.
    (see e.g.: https://en.wikipedia.org/wiki/Limited-memory_BFGS)

    Args:
        max_history (int, optional): The maximum history. Defaults to 15.
        verbose (bool, optional): Verbosity. Defaults to False.
    """

    def __init__(self, max_history: int = 15, verbose: bool = False):
        super().__init__()
        self.m = max_history
        self.verbose = verbose

        self.currently_stored_vectors: int = 0
        self.S: Deque[Vec]
        self.Y: Deque[Vec]
        self.rho: Deque[Scalar]
        self.Hk_sk: List[Vec]
        self.Hk_sk_exists: List[bool]

    def save(self, state_file: Union[pathlib.Path, str, None] = None) -> None:
        """Saves the state of the Hessian to disk.

        Args:
            state_file (Union[pathlib.Path, str, None]], optional): The state file.
                Defaults to None.
        """
        if state_file is None or self.currently_stored_vectors < 1:
            return
        if self.verbose:
            print("Writing LBFGS cache to disk")

        with h5py.File(state_file, "a") as f:
            if "LBFGS" not in f.keys():
                f.create_group("LBFGS")
            f["LBFGS"].attrs["currently_stored_vectors"] = self.currently_stored_vectors
            if "S" not in f["LBFGS"]:
                f["LBFGS"].create_dataset("S", data=deque_to_numpy_array(self.S))
            else:
                f["LBFGS"]["S"][:] = deque_to_numpy_array(self.S)
            if "Y" not in f["LBFGS"]:
                f["LBFGS"].create_dataset("Y", data=deque_to_numpy_array(self.Y))
            else:
                f["LBFGS"]["Y"][:] = deque_to_numpy_array(self.Y)
            if "rho" not in f["LBFGS"]:
                f["LBFGS"].create_dataset("rho", data=self.rho)
            else:
                f["LBFGS"]["rho"][:] = self.rho

    def read(
        self,
        target_vec: Vec,
        state_file: Union[pathlib.Path, str, None] = None,
    ) -> None:
        """Read the state from disk.

        Args:
            target_vec (Vec): An example Vec to inform the correct type.
            state_file (Union[pathlib.Path, str, None], optional): The state file.
                Defaults to None.
        """
        if state_file is None or not os.path.exists(state_file):
            return

        if self.verbose:
            print("Reading LBFGS cache from disk")

        with h5py.File(state_file, "r") as f:
            if "LBFGS" in f:
                self.currently_stored_vectors = f["LBFGS"].attrs[
                    "currently_stored_vectors"
                ]
                self.S = array_to_deque(
                    as_target_vec(f["LBFGS"]["S"][:], target_vec), self.m
                )
                self.Y = array_to_deque(
                    as_target_vec(f["LBFGS"]["Y"][:], target_vec), self.m
                )
                self.rho = deque(f["LBFGS"]["rho"][:])

    def update(self, s_update: Vec, y_update: Vec) -> bool:
        """Update the LBFGS history

        Args:
            s_update (Vec): Model difference
            y_update (Vec): Gradient difference

        Returns:
            bool: Update successful.
        """
        if len(s_update) != len(y_update):
            raise ValueError("Model and gradient vector should have the same length.")

        d_s_y = dot(s_update, y_update)

        # We need to ensure a positive rho to ensure
        # positive-definiteness of the Hessian (eq 6.7 in Nocedal 2nd edition)
        if d_s_y <= 0.0:
            if self.verbose:
                print("Curvature condition not satisfied")
                print("No vectors were added, and the LBFGS history has been reset.")
            return False
        rho = 1.0 / d_s_y

        if self.currently_stored_vectors == 0:
            self.S = deque(zeros_like(s_update) for _ in range(self.m))
            self.Y = deque(zeros_like(s_update) for _ in range(self.m))
            self.rho = deque(0.0 for _ in range(self.m))
        else:
            self.S.rotate(1)
            self.Y.rotate(1)
            self.rho.rotate(1)

        self.currently_stored_vectors = min(self.currently_stored_vectors + 1, self.m)

        self.S[0] = s_update
        self.Y[0] = y_update
        self.rho[0] = rho
        return True

    def apply_inverse(self, v: Vec) -> Vec:
        """Apply the Inverse Hessian approximation to vector v:

        Args:
            v (Vec): vector to which the Hessian is applied.

        Returns:
            Vec: The result.
        """
        if self.currently_stored_vectors < 1:
            raise ValueError("Please update the history first.")
        assert len(self.S) and len(self.Y) and len(self.rho)

        q = copy_vec(v)
        alpha = np.zeros(self.currently_stored_vectors)
        for i in range(self.currently_stored_vectors):
            alpha[i] = self.rho[i] * dot(self.S[i], q)
            q = q - alpha[i] * self.Y[i]

        gamma_k = dot(self.S[0], self.Y[0]) / dot(self.Y[0], self.Y[0])
        H_0 = (
            1 * gamma_k
        )  # Note that this is the inverse (maybe we want to rename this?)
        z = H_0 * q

        for i in range(self.currently_stored_vectors - 1, -1, -1):
            beta_i = self.rho[i] * dot(self.Y[i], z)
            z = z + self.S[i] * (alpha[i] - beta_i)
        return as_vec(z)

    def apply(self, v: Vec) -> Vec:
        """Apply the Hessian approximation to vector v:

        Args:
            v (Vec): vector to which the Hessian is applied.

        Returns:
            Vec: The result.
        """
        self.Hk_sk = [np.zeros(len(v)) for _ in range(self.currently_stored_vectors)]
        self.Hk_sk_exists = [False] * self.currently_stored_vectors
        return self._apply_hessian_recursive(self.currently_stored_vectors, v)

    def _get_hessian_k_sk(self, k: int, s_k: Vec) -> Vec:
        """
        Lookup H_k(s_k) from cache or compute it.
        This massively reduces the number of recursions
        E.g. for history of 20, it reduces function calls by a factor of ~9000
        """
        if not self.Hk_sk_exists[k]:
            self.Hk_sk[k] = self._apply_hessian_recursive(k, s_k)
            self.Hk_sk_exists[k] = True
        return self.Hk_sk[k]

    def _apply_hessian_recursive(self, k: int, v: Vec) -> Vec:
        i = self.currently_stored_vectors - k
        assert len(self.Y) and len(self.S)
        if k == 0:
            # Inverse as in the regular Hessian..
            gamma_k = dot(self.Y[0], self.Y[0]) / dot(self.S[0], self.Y[0])
            H_0 = 1 * gamma_k
            return H_0 * v

        Hk_sk = self._get_hessian_k_sk(k - 1, self.S[i])
        Hk_v = self._apply_hessian_recursive(k - 1, v)

        gam_1 = dot(self.S[i], Hk_v) / dot(self.S[i], Hk_sk)
        gam_2 = dot(self.Y[i], v) / dot(self.Y[i], self.S[i])
        return as_vec(Hk_v - gam_1 * Hk_sk + gam_2 * self.Y[i])

    def reset(self) -> None:
        """Reset the LBFGS history."""
        self.currently_stored_vectors = 0
        self.S = deque()
        self.Y = deque()
        self.rho = deque()
        self.Hk_sk = []
        self.Hk_sk_exists = []

    def ready(self) -> bool:
        """Returns true if more than one curvature pair is available.

        Returns:
            bool: True or False.
        """
        return self.currently_stored_vectors >= 1

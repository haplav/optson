from __future__ import annotations

import pathlib
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Union

import h5py
import numpy as np

from .batch_manager import BatchManager, EmptyBatchManager
from .problem import Problem, DummyProblem
from .vector import InVec, Scalar, Vec, as_target_vec, as_vec, dot


@dataclass(frozen=True)
class ModelProxy:
    """The ModelProxy object that gives problem access to the most important information of `Model`.

    Args:
        x (Vec): The model vector.
        name (str): The name of the vector.
        iteration (int): The iteration number.
        _batch_manager (BatchManager): The `BatchManager` instance.
        _problem (Problem): The `Problem` instance.
        previous (Optional[ModelProxy], optional): The previous `ModelProxy`. Defaults to None.
        radius (Optional[float], optional): The trust-region radius or step_size. Defaults to None.
    """

    #: The `Vec` with the model parameters.
    x: Vec
    #: The name of the `Model`.
    name: str
    #: The iteration number.
    iteration: int
    _batch_manager: BatchManager
    _problem: Problem
    #: The previous `ModelProxy`.
    previous: Optional[ModelProxy] = None
    #: The step size or trust-region radius.
    radius: Optional[float] = None

    @property
    def accepted(self) -> bool:
        """Is the model accepted?

        Returns:
            bool: Model accepted or not.
        """
        raise NotImplementedError

    @cached_property
    def batch(self) -> Optional[List[int]]:
        """The mini-batch. If no mini-batches are used, this will be None.
        Returns:
            Optional[List[int]]: The sample indices or None.
        """
        raise NotImplementedError

    @cached_property
    def control_group(self) -> List[int]:
        """The current control group, this may be an empty list if no control group is used.
        Returns:
            List[int]: The sample indices.
        """
        raise NotImplementedError

    @cached_property
    def control_group_previous(self) -> List[int]:
        """The previous control group, this may be an empty list if no control group is used.
        Returns:
            List[int]: The sample indices.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def descriptor(self) -> str:
        """Provides a descriptor of the current model.
        Returns:
            str: The description.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Model(ModelProxy):
    """The model class. This frozen data class holds the current model vector
    and metainformation. In addition, it caches important things such as the gradient
    and function values.

    Args:
        x (Vec): The model vector.
        name (str): The name of the vector.
        iteration (int): The iteration number.
        _batch_manager (BatchManager): The `BatchManager` instance.
        _problem (Problem): The `Problem` instance.
        previous (Optional[ModelProxy], optional): The previous `ModelProxy`. Defaults to None.
        radius (Optional[float], optional): The trust-region radius or step_size. Defaults to None.
        p (Vec, optional): The update step.
        store_history (bool, optional): Store the history of models. Defaults to False.
        update_H (bool, optional): Update the Hessian. Defaults to True.
    """

    #: The update step.
    p: Optional[Vec] = None
    #: Is the history stored?
    store_history: bool = False
    #: Is the Hessian updated?
    update_H: bool = True

    @cached_property
    def batch(self) -> Optional[List[int]]:
        """The mini-batch if running **Optson** in a stochastic mode, otherwise None.

        Returns:
            Optional[List[int]]: The sample indices or None.
        """
        return self._batch_manager.get_batch(self.iteration)

    @cached_property
    def control_group(self) -> List[int]:
        """The control group indices, this may be an empty list.

        Returns:
            List[int]: The sample indices.
        """
        return self._batch_manager.get_control_group(self.iteration)

    @cached_property
    def control_group_previous(self) -> List[int]:
        """The previous group indices, this may be an empty list.

        Returns:
            List[int]: The sample indices.
        """
        return self._batch_manager.get_control_group(self.iteration - 1)

    @cached_property
    def fx(self) -> Scalar:
        """The (mini-batch) function value of the current model.

        Returns:
            Scalar: The function value.
        """
        return self._problem.f(self, self.batch)

    @cached_property
    def gx(self) -> Vec:
        """The (mini-batch) gradient of the current model.

        Returns:
            Vec: The gradient.
        """
        return self._problem.g(self, self.batch)

    @cached_property
    def H_gx(self) -> Vec:
        """The Hessian applied to `self.gx`.

        Returns:
            Vec: The result.
        """

        return self._problem.H.apply(self.gx)

    @cached_property
    def Hinv_gx(self) -> Vec:
        """The inverse Hessian applied to `self.gx`.

        Returns:
            Vec: The result.
        """
        return self._problem.H.apply_inverse(self.gx)

    @cached_property
    def H_p(self) -> Vec:
        """The Hessian applied to the update p.

        Returns:
            Vec: The result.
        """
        assert self.p is not None
        return self._problem.H.apply(self.p)

    # set H_p to save computation in special cases
    def _set_H_p(self, H_p: Vec) -> None:
        self.__dict__["H_p"] = H_p

    @cached_property
    def Hinv_p(self) -> Vec:
        """The inverse Hessian applied to the update p.

        Returns:
            Vec: The result.
        """
        assert self.p is not None
        return self._problem.H.apply_inverse(self.p)

    @cached_property
    def fx_cg(self) -> Scalar:
        """The function value of the control group.

        Returns:
            Scalar: The function value.
        """
        if not self._batch_manager.stochastic:
            return self.fx
        # make sure gx_cg is computed first so that the control group is tuned
        _ = self.gx_cg
        return self._problem.f(self, self.control_group)

    @cached_property
    def gx_cg(self) -> Vec:
        """
        The control group gradient.

        Returns:
            Vec: The gradient.
        """
        return self._find_gx_cg() if self._batch_manager.stochastic else as_vec(self.gx)

    @cached_property
    def fx_cg_prev(self) -> Scalar:
        """
        The function value of the previous control group for this model.

        Returns:
            Scalar: The function value.
        """
        if self._batch_manager.stochastic:
            return self._problem.f(self, self.control_group_previous)
        return self.fx

    @cached_property
    def gx_cg_prev(self) -> Vec:
        """
        The gradient of the previous control group for this model.

        Returns:
            Vec: The gradient.
        """
        if self._batch_manager.stochastic:
            return self._problem.g(self, self.control_group_previous)
        else:
            return as_vec(self.gx)

    def clear_cg(self) -> None:
        """Clear the control group misfit and gradient."""
        self._clear_properties(
            "control_group",
            "fx_cg",
            "gx_cg",
        )

    def extend_control_group(self) -> bool:
        """Extend the control group.

        Returns:
            bool: True if successful.
        """
        ok = self._batch_manager.extend_control_group(self.iteration)
        if ok:
            self.clear_cg()
        return ok

    @classmethod
    def create_initial(
        cls,
        problem: Problem,
        batch_manager: BatchManager,
        x0: InVec,
        name: str = "x",
        iteration: int = 0,
        radius: Optional[float] = None,
        accepted: bool = True,
        store_history: bool = False,
        update_H: bool = True,
    ) -> Model:
        """Create an initial Model.

        Args:
            problem (Problem): The problem.
            batch_manager (BatchManager): The batch manager.
            x0 (InVec): The model vector.
            name (str, optional): The model name. Defaults to "x".
            iteration (int, optional): The iteration number. Defaults to 0.
            radius (Optional[float], optional): The trust-region radius or step size. Defaults to None.
            accepted (bool, optional): Set the model to accepted or not. Defaults to True.
            store_history (bool, optional): Store the history. Defaults to False.
            update_H (bool, optional): Update the Hessian. Defaults to True.

        Returns:
            Model: The Model.
        """
        m = cls(
            _problem=problem,
            _batch_manager=batch_manager,
            x=as_vec(x0),
            name=name,
            iteration=iteration,
            radius=radius,
            store_history=store_history,
            update_H=update_H,
        )
        if accepted:
            m.accept()
        return m

    @classmethod
    def wrap_vec(
        cls,
        x0: InVec,
        name: str = "x",
        iteration: int = 0,
        radius: Optional[float] = None,
    ) -> Model:
        """Convenience function that creates a Model from a Vec.

        Args:
            x0 (InVec): The vector to be used.
            name (str, optional): The name. Defaults to "x".
            iteration (int, optional): The iteration number. Defaults to 0.
            radius (Optional[float], optional): The trust-region radius or step size. Defaults to None.

        Returns:
            Model: The Model.
        """

        return cls(
            _problem=DummyProblem(),
            _batch_manager=EmptyBatchManager(),
            x=as_vec(x0),
            name=name,
            iteration=iteration,
            radius=radius,
        )

    def _update_hessian(self, m_prev: ModelProxy) -> None:
        """Update the Hessian.

        Args:
            m_prev (ModelProxy): The previous Model.
        """
        assert isinstance(m_prev, Model)
        s_update: Vec = self.x - m_prev.x
        y_update: Vec = self.gx_cg_prev - m_prev.gx_cg
        self._problem.H.update(s_update, y_update)

    def accept(self) -> Model:
        """Accept the model.

        Returns:
            Model: The accepted Model.
        """
        if getattr(self, "_accepted", False):
            return self
        if self.previous and self.update_H:
            self._update_hessian(self.previous)
        if not self.store_history and self.previous is not None:
            object.__setattr__(self.previous, "previous", None)
        self._clear_properties("p", "H_p", "Hinv_p")
        object.__setattr__(self, "_accepted", True)
        return self

    @property
    def accepted(self) -> bool:
        """Property that shows if a Model is accepted.

        Returns:
            bool: True or False.
        """
        return getattr(self, "_accepted", False)

    @cached_property
    def descriptor(self) -> str:
        """Get the description of a model.

        Returns:
            str: The description.
        """
        if not self.radius:
            return f"{self.name}_{self.iteration:05d}"
        else:
            return f"{self.name}_{self.iteration:05d}_Radius_{self.radius}"

    def new(
        self,
        x: Vec,
        p: Optional[Vec] = None,
        name: Optional[str] = None,
        iteration: Optional[int] = None,
        radius: Optional[float] = None,
    ) -> Model:
        """Create a new Model.

        Args:
            x (Vec): The new model vector.
            p (Optional[Vec], optional): The model update. Defaults to None.
            name (Optional[str], optional): The name. Defaults to None.
            iteration (Optional[int], optional): The iteration number. Defaults to None.
            radius (Optional[float], optional): The trust-region radius or step size. Defaults to None.

        Returns:
            Model: The new model.
        """

        if iteration is None:
            iteration = self.iteration + 1
        if name is None:
            name = self.name
        if radius is None:
            radius = self.radius
        return self.__class__(
            x=x,
            _problem=self._problem,
            _batch_manager=self._batch_manager,
            name=name,
            iteration=iteration,
            radius=radius,
            p=p,
            store_history=self.store_history,
            update_H=self.update_H,
            previous=self,
        )

    def __post_init__(self) -> None:
        if self.p is not None:
            # do x = x + p automatically
            object.__setattr__(self, "x", self.x + self.p)
            if isinstance(self.p, np.ndarray):
                self.p.flags.writeable = False
        if isinstance(self.x, np.ndarray):
            self.x.flags.writeable = False

    def _clear_properties(self, *properties: str) -> None:
        for attr in properties:
            self.__dict__.pop(attr, None)

    def _find_gx_cg(self) -> Vec:
        """
        Here, we ensure gx is also a descent direction for the control group.
        """
        Hinv_dot_gx = None
        gx = self.gx  # Call gx first for workflow reasons
        while True:
            gx_cg = self._problem.g(self, self.control_group)

            # Enforce positive predicted reductions for both gradients.
            if self._problem.H.ready():
                Hinv_dot_gx = (
                    dot(self.Hinv_gx, gx) if Hinv_dot_gx is None else Hinv_dot_gx
                )
                pred_cg = dot(self.Hinv_gx, gx_cg) - 0.5 * Hinv_dot_gx
                pred_mb = 0.5 * Hinv_dot_gx

                if pred_mb > 0.0 and pred_cg < 0.0:
                    self.extend_control_group()
                    continue
                elif pred_mb < 0.0:  # H not positive-definite
                    self._problem.H.reset()

            if dot(gx, gx_cg) > 0.0:
                break
            self.extend_control_group()
        return gx_cg

    def store(self, state_file: Union[pathlib.Path, str]) -> None:
        """Store the model to disk.

        Args:
            state_file (Union[pathlib.Path, str]): The state file.
        """
        with h5py.File(state_file, "a") as f:
            if "model" not in f:
                f.create_dataset("model", data=self.x)
            md = f["model"]
            md[:] = self.x
            md.attrs["name"] = self.name
            if self.radius:
                md.attrs["radius"] = self.radius
            md.attrs["iteration"] = self.iteration
        if self.update_H:
            self._problem.H.save(state_file)
        assert self._batch_manager is not None
        self._batch_manager.save(state_file)

    @classmethod
    def read(
        cls,
        state_file: Union[pathlib.Path, str],
        problem: Problem,
        batch_manager: BatchManager,
        target_vec: Vec,
        store_history: bool,
        update_H: bool,
        verbose: bool,
    ) -> Model:
        """Reads the model from disk.

        Args:
            state_file (Union[pathlib.Path, str]): The state file.
            problem (Problem): The problem.
            batch_manager (BatchManager): The batch manager.
            target_vec (Vec): The target vec.
            store_history (bool): Store history.
            update_H (bool): Update the Hessian.
            verbose (bool): Verbosity.

        Returns:
            Model: The model.
        """
        with h5py.File(state_file, "r") as f:
            md = f["model"]
            x = as_target_vec(md[:], target_vec)
            name = md.attrs["name"]
            radius = md.attrs["radius"] if "radius" in md.attrs else None
            iteration = md.attrs["iteration"]
        if verbose:
            print("Getting model from cache file...")
        m = cls.create_initial(
            problem=problem,
            x0=x,
            name=name,
            batch_manager=batch_manager,
            iteration=int(iteration),
            radius=radius,
            store_history=store_history,
            update_H=update_H,
        )
        if m.update_H:
            m._problem.H.read(target_vec, state_file)
        m._batch_manager.load(state_file)
        return m

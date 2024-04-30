from __future__ import annotations

import os
import pathlib
from typing import List, Union

from .batch_manager import BatchManager, EmptyBatchManager
from .methods.trust_region import BasicTRUpdate
from .model import Model
from .monitor import Monitor, EmptyMonitor
from .problem import Problem
from .stopping_criterion import StoppingCriterion, BasicStoppingCriterion
from .update import Update
from .utils import InstanceOrType, get_instance
from .vector import InVec, Vec, as_vec


class Optimizer:
    """The Optimizer class. This is the entry point for working with Optson.
    By providing all the arguments, you can more or less fully customize the type of optimization
    and the stopping criteria.

    Please have a look at the :ref:`tutorials` section for more information.

    Args:
        problem (InstanceOrType[Problem]): The problem, derived from :class:`~optson.problem.Problem`
        update (InstanceOrType[Update], optional): The update method. Defaults
            to :class:`~optson.methods.trust_region.BasicTRUpdate`.
        batch_manager (InstanceOrType[BatchManager], optional): The batch manager instance.
            Defaults to :class:`~optson.batch_manager.EmptyBatchManager`.
        monitor (InstanceOrType[Monitor], optional): The monitor. Defaults to :class:`~optson.monitor.EmptyMonitor`.
        stopping_criterion (InstanceOrType[StoppingCriterion], optional): The stopping criterion.
            Defaults to :class:`~optson.stopping_criterion.BasicStoppingCriterion`.
        state_file (typing.Union[pathlib.Path, str, None]], optional): An optional filepath were
            the state will be saved. Defaults to None.
        store_models (bool, optional): Keep models in memory. Defaults to False.
        verbose (bool, optional): Verbosity. Defaults to False.
    """

    def __init__(
        self,
        problem: InstanceOrType[Problem],
        update: InstanceOrType[Update] = BasicTRUpdate,
        batch_manager: InstanceOrType[BatchManager] = EmptyBatchManager,
        monitor: InstanceOrType[Monitor] = EmptyMonitor,
        stopping_criterion: InstanceOrType[StoppingCriterion] = BasicStoppingCriterion,
        state_file: Union[pathlib.Path, str, None] = None,
        store_models: bool = False,
        verbose: bool = False,
    ):
        self.problem = get_instance(problem)
        self.update = get_instance(update, verbose=verbose)
        self.batch_manager = get_instance(batch_manager)
        self.monitor = get_instance(monitor)
        self.stopping_criterion = get_instance(stopping_criterion, verbose=verbose)
        self.state_file = state_file
        self.store_models = store_models
        self.verbose = verbose

        self.models: List[Model] = []

    def iterate(self, x0: InVec) -> Model:
        """Start iterating

        Args:
            x0 (InVec): The input vector, this must be either a type that is convertible
                to :class:`~numpy.ndarray` of floats or something of type: :class:`~optson.vector.Vec`.

        Returns:
            Model: The latest :class:`~optson.model.Model` after iterating.
        """
        x0 = as_vec(x0)
        m = self._initialize(x0)
        while True:
            if self.store_models:
                # TODO use linked Models instead
                self.models.append(m)
            if self.stopping_criterion(m):
                break
            self.monitor(m)

            m_trial = self.update(m)
            if m_trial.accepted:
                m = m_trial
                self._store_state(m=m)
            else:  # pragma: no cover
                print("No suitable update could be found. Stopping now.")
                break
        return m

    def _store_state(self, m: Model) -> None:
        """
        Stores a model on disk. Always overwrites existing models.
        This allows the iterate function to continue iterating from where it previously stopped.
        """
        if not self.state_file:
            return
        if self.verbose:  # pragma: no cover
            print("Writing optimizer state file... Don't interrupt the process now.")
        m.store(self.state_file)
        self.update.store_attributes(self.state_file)

        if self.verbose:  # pragma: no cover
            print("Save state completed.")

    def _initialize(self, x0: Vec) -> Model:
        """
        Initialize a model from disk if possible, otherwise initializes with x0.
        """
        if not self.state_file or not os.path.exists(self.state_file):
            return Model.create_initial(
                problem=self.problem,
                batch_manager=self.batch_manager,
                x0=x0,
                store_history=self.store_models,
                update_H=self.update.needs_H,
            )

        self.update.get_attributes(x0, self.state_file)
        return Model.read(
            state_file=self.state_file,
            problem=self.problem,
            batch_manager=self.batch_manager,
            target_vec=x0,
            store_history=self.store_models,
            update_H=self.update.needs_H,
            verbose=self.verbose,
        )

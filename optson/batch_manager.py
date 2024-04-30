from __future__ import annotations

import pathlib
import pickle
from abc import ABC, abstractmethod
from functools import cached_property
from random import Random
from typing import List, Optional, Set, Tuple, Union

import h5py
import numpy as np

from .utils import h5_load_list, h5_save_list


class BatchManager(ABC):
    """An abstract base class for a `BatchManager`."""

    def __init__(self) -> None:
        self.iteration: int = -1
        self.control_group: List[int] = []
        self.control_group_previous: List[int] = []
        self.batch: Optional[List[int]] = None

    @abstractmethod
    def update(self, iteration: int) -> Tuple[Optional[List[int]], List[int]]:
        """Update the current mini-batch.

        Args:
            iteration (int): The current iteration.

        Returns:
            Tuple[Optional[List[int]], List[int]]: A list of indices or None.
        """
        raise NotImplementedError

    def _update(self, iteration: int) -> None:
        assert iteration == self.iteration + 1
        self.control_group_previous = self.control_group
        self.batch, self.control_group = self.update(iteration)
        self.iteration = iteration

    def get_batch(self, iteration: int) -> Optional[List[int]]:
        """Get the mini-batch of an iteration if applicable.

        Args:
            iteration (int): The current iteration.

        Returns:
            Optional[List[int]]: The mini-batch or None.
        """
        if iteration == self.iteration:
            return self.batch
        elif iteration == self.iteration + 1:
            self._update(iteration)
            return self.batch
        msg = (
            f"Can only get the current or the next iteration's batch."
            f"Current iteration is {self.iteration} while {iteration} was requested."
        )
        raise ValueError(msg)

    def get_control_group(self, iteration: int) -> List[int]:
        """Gets the control group for an iteration.

        Args:
            iteration (int): The iteration.

        Returns:
            List[int]: The control group of the iteration.
        """
        if iteration in [self.iteration - 1, -1]:
            return self.control_group_previous
        elif iteration == self.iteration:
            return self.control_group
        elif iteration == self.iteration + 1:
            self.control_group_previous = self.control_group
            self._update(iteration)
            return self.control_group
        msg = (
            f"Only the current or the previous iteration can be requested."
            f"Expected {self.iteration}, {self.iteration - 1} or -1, got {iteration}"
        )
        raise ValueError(msg)

    @abstractmethod
    def extend_control_group(self, iteration: int) -> bool:
        """Extend the control group.

        Args:
            iteration (int): The iteration.

        Returns:
            bool: Extended or not.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, file: Union[pathlib.Path, str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, file: Union[pathlib.Path, str]) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def stochastic(self) -> bool:
        raise NotImplementedError


class EmptyBatchManager(BatchManager):
    """An empty batch manager. Using this batch manager will instruct **Optson** to use non-stochastic optimization methods."""

    def update(self, iteration: int) -> Tuple[Optional[List[int]], List[int]]:
        """Update the batch manager.

        Args:
            iteration (int): The iteration number.

        Returns:
            Tuple[Optional[List[int]], List[int]]: Just `None, []` in this case.
        """
        return None, []

    def get_batch(self, iteration: int) -> Optional[List[int]]:
        """Returns None, which represents using all data.

        Args:
            iteration (int): The iteration.

        Returns:
            Optional[List[int]]: Just `None` in this case.
        """

        return None

    def get_control_group(self, iteration: int) -> List[int]:
        """Gets the control group, which is not applicable.

        Args:
            iteration (int): The iteration.

        Returns:
            List[int]: An empty list.
        """
        return []

    def extend_control_group(self, iteration: int) -> bool:
        """
        Extend the control. Since there is None, we return False.

        Args:
            iteration (int): The iteration.

        Returns:
            bool: False
        """
        return False

    def save(self, file: Union[pathlib.Path, str]) -> None:
        """Empty save.

        Args:
            file (Union[pathlib.Path, str]): The state file.
        """
        pass

    def load(self, file: Union[pathlib.Path, str]) -> None:
        """Empty load.

        Args:
            file (Union[pathlib.Path, str]): The state file.
        """
        pass

    @property
    def stochastic(self) -> bool:
        """Property that indicates whether the batch manager is used for stochastic optimization.

        Returns:
            bool: Just `False` in this case.
        """
        return False


class SimpleBatchManager(BatchManager):
    """A simple batch manager that selects non-overlapping mini-batches that sequentially go through the whole dataset.

    Args:
        batch_size (int): The batch size.
        n_samples (int): The total number of samples.
        samples (Optional[Union[List[int], Set[int]]], optional): A specific set of samples that should be sampled.
         Defaults to None.
    """

    def __init__(
        self,
        batch_size: int,
        n_samples: int,
        samples: Optional[Union[List[int], Set[int]]] = None,
    ):
        super().__init__()
        assert batch_size <= n_samples
        if samples:
            assert n_samples == len(samples)
        self.n_samples = n_samples

        # Needs to be a list to support slicing.
        self.samples = list(samples) if samples else list(range(self.n_samples))
        self.batch_size = batch_size
        self.current_index = 0

    def update(self, iteration: int) -> Tuple[Optional[List[int]], List[int]]:
        if iteration == self.iteration + 1:
            if self.current_index + self.batch_size > self.n_samples:
                self.current_index = 0
            end_index = self.current_index + self.batch_size
            batch = self.samples[self.current_index : end_index]
            self.current_index += self.batch_size
        return batch, []

    def extend_control_group(self, iteration: int) -> bool:
        raise ValueError(
            "Control groups are not defined in the context of the SimpleBatchManager. "
            "If you use methods that rely on trust-region or line-search techniques, "
            "please use a different BatchManager."
        )

    def save(self, file: Union[pathlib.Path, str]) -> None:
        assert self.batch is not None
        with h5py.File(file, "a") as f:
            grp = f.require_group("batch_manager")
            grp.attrs["iteration"] = self.iteration
            grp.attrs["current_index"] = self.current_index
            h5_save_list(grp, "batch", self.batch)

    def load(self, file: Union[pathlib.Path, str]) -> None:
        with h5py.File(file, "r") as f:
            grp = f["batch_manager"]
            self.iteration = grp.attrs["iteration"]
            self.current_index = grp.attrs["current_index"]
            self.batch = h5_load_list(grp, "batch", int)

    @property
    def stochastic(self) -> bool:
        return True


class ControlGroupBatchManager(BatchManager):
    """A batch manager that uses overlapping batches. In other words
    it uses a control group. This makes it possible to use line-search
    or trust-region based approaches with subsets of the data.

    It also enables using the overlapping samples to update curvature information
    in Quasi-Newton methods.

    Args:
        n_samples (int, optional): The number of samples. Defaults to 500.
        batch_size (int, optional): The initial batch size. Defaults to 20.
        random_seed (int, optional): The random seed. Defaults to 10.
        control_group_percentage (float, optional): The percentage of a batch that will overlap
            with the next batch. Defaults to 0.5.
        batch_grow_factor (float, optional): The factor at which the next batch should grow
            as a function of the latest control group. Defaults to 2.0.
        extend_control_group_factor (float, optional): This parameter controls how quickly the
            batch size grows when the control group is extended. Defaults to 0.5.
        samples (Optional[Union[List[int], Set[int]]], optional): A set of samples. Defaults to None.
    """

    def __init__(
        self,
        n_samples: int = 500,
        batch_size: int = 20,
        random_seed: int = 10,
        control_group_percentage: float = 0.5,
        batch_grow_factor: float = 2.0,
        extend_control_group_factor: float = 0.5,
        samples: Optional[Union[List[int], Set[int]]] = None,
    ):
        super().__init__()
        assert batch_size > 1, "This batch manager needs 2 samples per batch."
        assert batch_size <= n_samples, "Batch size cannot exceed the number of samples"
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.random = Random(random_seed)
        self.control_group_percentage = control_group_percentage
        # Will grow batch by at most this factor times the control group size
        self.batch_grow_factor = batch_grow_factor
        self.extend_control_group_factor = extend_control_group_factor
        self.samples = samples

    @cached_property
    def _all_samples_set(self) -> Set[int]:
        return set(self.samples) if self.samples else set(range(self.n_samples))

    def update(self, iteration: int) -> Tuple[Optional[List[int]], List[int]]:
        control_group = self.control_group_previous
        n_to_select = self.batch_size - len(control_group)
        all_samples_set: Set[int] = self._all_samples_set
        samples_to_select_from = list(all_samples_set - set(control_group))
        batch = (
            self.random.sample(samples_to_select_from, k=n_to_select) + control_group
        )
        assert len(batch) == self.batch_size

        k = max(int(self.control_group_percentage * self.batch_size), 1)
        control_group = self.random.sample(batch, k)
        return batch, control_group

    def extend_control_group(self, iteration: int) -> bool:
        """Extends the control group.

        Args:
            iteration (int): The iteration number.

        Raises:
            ValueError: When an iteration other than the current iteration is passed.

        Returns:
            bool: A boolean that indicates whether extension was possible.
        """
        if iteration != self.iteration:
            raise ValueError(
                f"Only the current iteration {self.iteration} can be extended. "
                f"Got iteration {iteration}."
            )

        current_control_group = self.control_group
        current_batch = self.batch
        assert current_batch is not None
        not_selected_yet = list(set(current_batch) - set(current_control_group))
        n_not_selected = len(not_selected_yet)

        if n_not_selected >= 1:
            amount_to_select = int(
                np.ceil(self.extend_control_group_factor * n_not_selected)
            )
        else:
            return False
        selected = self.random.sample(not_selected_yet, k=amount_to_select)
        self.control_group += selected

        control_size = len(self.control_group)
        self.batch_size = min(
            self.n_samples, int(np.ceil(self.batch_grow_factor * control_size))
        )
        return True

    def save(self, file: Union[pathlib.Path, str]) -> None:
        state_void = np.void(pickle.dumps(self.random.getstate()))
        with h5py.File(file, "a") as f:
            grp = f.require_group("batch_manager")
            grp.attrs["iteration"] = self.iteration
            grp.attrs["batch_size"] = self.batch_size
            grp.attrs["random_state"] = state_void
            assert self.batch is not None
            h5_save_list(grp, "batch", self.batch)
            h5_save_list(grp, "control_group", self.control_group)
            h5_save_list(grp, "control_group_previous", self.control_group_previous)

    def load(self, file: Union[pathlib.Path, str]) -> None:
        with h5py.File(file, "r") as f:
            grp = f["batch_manager"]
            self.iteration = grp.attrs["iteration"]
            self.batch_size = grp.attrs["batch_size"]
            state_void = grp.attrs["random_state"]
            self.batch = h5_load_list(grp, "batch", int)
            self.control_group = h5_load_list(grp, "control_group", int)
            self.control_group_previous = h5_load_list(
                grp, "control_group_previous", int
            )

        self.random.setstate(pickle.loads(state_void.tobytes()))

    @property
    def stochastic(self) -> bool:
        return True

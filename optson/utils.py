from __future__ import annotations

from typing import Any, List, Type, TypeVar, Union

import h5py
import numpy as np

#: TypeVar of Any Type
T = TypeVar("T")

#: An instance of a class or the class itself.
InstanceOrType = Union[T, Type[T]]


def get_instance(input: InstanceOrType[T], *args: Any, **kwargs: Any) -> T:
    """
    Returns an instance if input is already an instance,
    otherwise, we initialize an instance based on the class of input with args and kwargs.

    Args:
        input (InstanceOrType[T]): The instance or a class

    Returns:
        T: The input instance or an instance of the class of input.
    """
    inst: T = input(*args, **kwargs) if isinstance(input, type) else input
    return inst


def format_floats(*t: float, format: str = "%.3e") -> str:
    """Format a float or a list of floats.

    Args:
        format (str, optional): The format. Defaults to "%.3e".

    Returns:
        str: The formatted result.
    """
    return ", ".join(format % x for x in t)


def h5_save_list(
    target: Union[h5py.File, h5py.Group],
    name: str,
    lst: List[Any],
) -> None:
    shape = (len(lst),)
    ndarray = np.asarray(lst)
    try:
        ds = target.require_dataset(name, shape, dtype=ndarray.dtype)
    except TypeError:
        del target[name]
        ds = target.create_dataset(name, shape, dtype=ndarray.dtype)
    ds[:] = ndarray


def h5_load_list(
    source: Union[h5py.File, h5py.Group], name: str, type: Type[T]
) -> List[T]:
    """Load a list from an .h5 file

    Args:
        source (Union[h5py.File, h5py.Group]): The h5 file or group
        name (str): The name of the list
        type (Type[T]): The type

    Returns:
        List[T]: The list.
    """
    dtype = np.dtype(type)
    ds = source[name]
    assert isinstance(ds, h5py.Dataset)
    ds = ds.astype(dtype)
    lst = np.asarray(ds, dtype=dtype).tolist()
    assert isinstance(lst, list)
    return lst

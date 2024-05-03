from __future__ import annotations

from collections import deque
from enum import Enum
from functools import partial
from math import sqrt
from typing import Any, Callable, Deque, Sequence, Union

import numpy as np
import numpy.typing

_NDArray = numpy.typing.NDArray[np.floating]
_Tensor = Any
_Array = Any

try:
    import torch
    from torch import Tensor as _Tensor  # type: ignore[no-redef]

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False

try:
    import jax.numpy as jnp
    from jax import Array as _Array  # type: ignore[no-redef]

    JAX_AVAIL = True
except ImportError:
    JAX_AVAIL = False

#: Vec
Vec = Union[_NDArray, _Tensor, _Array]

#: A float or 1D scalar
Scalar = Union[Vec, float]

#: InVec
InVec = Union[Vec, Sequence[float]]


class EVecType(Enum):
    """An enumeration of VecTypes."""

    NONE = None
    NUMPY = "numpy.typing.NDArray[np.floating]"
    TORCH = "torch.Tensor"
    JAX = "jax.Array"


def get_type(arr: InVec, fail: bool = True) -> EVecType:
    """Get the Vec type of arr.

    Args:
        arr (InVec): The input array or Vec
        fail (bool, optional): Fail if unknown type is passed. Defaults to True.

    Returns:
        EVecType: The EVecType value corresponding to the input array.
    """
    if isinstance(arr, np.ndarray):
        t = EVecType.NUMPY
        is_float = np.issubdtype(arr.dtype, np.floating)
    elif TORCH_AVAIL and isinstance(arr, torch.Tensor):
        t = EVecType.TORCH
        is_float = torch.is_floating_point(arr)
    elif JAX_AVAIL and isinstance(arr, jnp.ndarray):
        t = EVecType.JAX
        is_float = jnp.issubdtype(arr.dtype, np.floating)
    else:
        t = EVecType.NONE
        is_float = False

    if t.value:
        if not is_float:
            if fail:
                dtype = getattr(arr, "dtype", "<unknown>")
                raise TypeError(f"Unsupported array dtype: {dtype}")
            else:
                t = EVecType.NONE
    elif fail:
        raise TypeError(f"Unsupported array type: {type(arr)}.")

    return t


def as_vec(xin: Any) -> Vec:
    """Converts the input to a numpy.ndarray if xin is not already a Vec.

    Args:
        xin (Any): The input array.

    Returns:
        Vec: The converted result.
    """
    return xin if get_type(xin, fail=False).value else np.asarray(xin, dtype=np.double)


def dot(a: Vec, b: Vec) -> Scalar:
    """Computes the dot product.

    In our case, this always computes the element-wise sum.

    Args:
        a (Vec): A Vec.
        b (Vec): The other Vec.

    Returns:
        Scalar: The resulting value.
    """
    assert a.shape == b.shape, f"{a.shape}, {b.shape}"
    nDim = len(a.shape)
    f: Callable[..., Scalar]
    a_type, b_type = get_type(a), get_type(b)
    if a_type != b_type:
        raise TypeError("Inputs must be of the same type")
    if nDim == 1:
        if a_type == EVecType.TORCH:
            f = torch.dot
        elif a_type == EVecType.JAX:
            f = jnp.dot
        else:
            f = np.dot
    else:
        if a_type == EVecType.TORCH:
            f = partial(torch.tensordot, dims=nDim)
        elif a_type == EVecType.JAX:
            f = partial(jnp.tensordot, axes=nDim)
        else:
            f = partial(np.tensordot, axes=nDim)
    assert f is not None
    return f(a, b)


def median(arr: Vec) -> Scalar:
    """Computes the median of a Vec.

    Args:
        arr (Vec): The Vec.

    Returns:
        Scalar: The median.
    """
    t = get_type(arr)
    f: Callable[..., Scalar]
    if t == EVecType.TORCH:
        f = torch.median
    elif t == EVecType.JAX:
        f = jnp.median
    else:
        f = np.median
    return f(arr)


def vec_sum(arr: Vec) -> Scalar:
    """Returns the sum of the Vec.

    Args:
        arr (Vec): the array.

    Returns:
        Scalar: The summed array.
    """
    t = get_type(arr)
    f: Callable[..., Scalar]
    if t == EVecType.TORCH:
        f = torch.sum
    elif t == EVecType.JAX:
        f = jnp.sum
    else:
        f = np.sum
    r = f(arr)
    # reveal_type(r)
    return r


def zeros_like(arr: Vec) -> Vec:
    """Returns an array of zeros of the same type as arr.

    Args:
        arr (Vec): The input array.

    Returns:
        Vec: An array of zeroes.
    """
    t = get_type(arr)
    f: Callable[..., Vec]
    if t == EVecType.TORCH:
        f = torch.zeros_like
    elif t == EVecType.JAX:
        f = jnp.zeros_like
    else:
        f = np.zeros_like
    return f(arr)


def norm(a: Vec) -> float:
    """Returns the Euclidean norm of an array.

    Args:
        a (Vec): The array.

    Returns:
        float: The norm.
    """
    return sqrt(dot(a, a))


def deque_to_numpy_array(d: Deque[Vec]) -> np.ndarray:
    """Convert a Deque to an array. This is useful because Deques of
    certain types cannot be stored directly to h5 with h5py.

    Args:
        d (Deque[Vec]): The deque of Vecs.

    Returns:
        The array.
    """

    return np.stack(d)


def as_target_vec(arr: Vec, target_vec: Vec) -> Vec:
    """Convert arr to the same type as target_vec

    Args:
        arr (Vec): The array to be converted
        target_vec (Vec): The other array of a potentially different type.

    Returns:
        Vec: The converted result.
    """
    t = get_type(target_vec)
    if t == EVecType.TORCH:
        return torch.from_numpy(arr)
    elif t == EVecType.JAX:
        return jnp.array(arr)
    else:
        return arr


def copy_vec(arr: Vec) -> Vec:
    """Returns a copy of Vec of the same type.

    Args:
        arr (Vec): The array to be copied.

    Returns:
        Vec: The copy.
    """
    t = get_type(arr)
    f: Callable[..., Vec]
    if t == EVecType.TORCH:
        f = torch.clone
    elif t == EVecType.JAX:
        f = jnp.copy
    else:
        f = np.copy
    return f(arr)


def array_to_deque(arr: Vec, length: int) -> Deque[Vec]:
    """Convert an array to a deque of Vec.

    Args:
        arr (Vec): The Vec.
        length (int): The length of the Deque.

    Returns:
        Deque[Vec]: The Deque of Vecs.
    """
    t = get_type(arr)
    f: Callable[..., Sequence[Vec]]
    if t == EVecType.TORCH:
        f = torch.chunk
    elif t == EVecType.JAX:
        f = jnp.split
    else:
        f = np.split
    return deque(t.squeeze() for t in f(arr, length))

from __future__ import annotations

import functools
from typing import Callable


def func_call_counter(func: Callable) -> Callable:
    """Decorator that counts the number of function calls.

    Args:
        func (Callable): The function.

    Returns:
        Callable: The decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Increase the call count for the instance and method or function
        if not hasattr(wrapper, "_call_count"):
            wrapper._call_count = {}
        if func.__name__ not in wrapper._call_count:
            wrapper._call_count[func.__name__] = 0
        wrapper._call_count[func.__name__] += 1
        return func(*args, **kwargs)

    return wrapper


def get_func_count(func: Callable) -> int:
    """
    Get the number of function calls for a function decorated with :func:`~optson.call_counter.func_call_counter`.

    Args:
        func (Callable): The decorated function.

    Returns:
        int: The number of function calls.
    """
    return (
        func._call_count[func.__name__]
        if hasattr(func, "_call_count") and func.__name__ in func._call_count
        else 0
    )


def method_call_counter(func: Callable) -> Callable:
    """Decorator that counts method calls.

    Args:
        func (Callable): The method.

    Returns:
        Callable: The decorated method.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Increase the call count for the instance and method or function
        if not hasattr(self, "_call_count"):
            setattr(self, "_call_count", {})
        if func.__name__ not in self._call_count:
            self._call_count[func.__name__] = 0
        self._call_count[func.__name__] += 1
        # print(wrapper._call_count)
        return func(self, *args, **kwargs)

    return wrapper


def get_method_count(func: Callable) -> int:
    """
    Get the number of function calls for a function decorated
    with :func:`~optson.call_counter.method_call_counter`.

    Args:
        func (Callable): The method. The object it is bound to matters.

    Returns:
        int: The number of calls to the method.
    """
    if not hasattr(func, "__self__"):
        raise ValueError("Given func is not bound to an object")
    obj = func.__self__
    return (
        obj._call_count[func.__name__]
        if hasattr(obj, "_call_count") and func.__name__ in obj._call_count
        else 0
    )


def class_method_call_counter(cls) -> None:
    """
    This is class decorator that applies :func:`~optson.call_counter.get_method_count`
    to all public methods of the class.
    """
    # Iterate over all attributes of the class
    for name, attr in cls.__dict__.items():
        # Check if the attribute is a public function
        if callable(attr) and not name.startswith("_"):
            # Apply the count_calls_decorator to the method
            setattr(cls, name, method_call_counter(attr))
    return cls

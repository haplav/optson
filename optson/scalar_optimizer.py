from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Tuple

from .utils import format_floats


class ScalarOptimizer(ABC):
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        verbose: bool = False,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

        self.nIterations = 0
        self.nAllIterations = 0
        self.nFuncEval = 0
        self.nAllFuncEval = 0

    @abstractmethod
    def __call__(
        self, f: Callable[[float], float], interval: Tuple[float, float]
    ) -> float:
        raise NotImplementedError()


class BisectionScalarOptimizer(ScalarOptimizer):
    def __call__(
        self, f: Callable[[float], float], interval: Tuple[float, float]
    ) -> float:
        # a <= c <= x <= d <= b
        (a, b) = interval
        x = (a + b) / 2
        nEval = 0
        for i in range(self.max_iterations):
            c, d = (a + x) / 2.0, (x + b) / 2.0
            fx = f(x)
            if self.verbose:
                self._monitor(i, nEval, fx, a, c, x, d, b)
            if f(c) < fx:
                b, x = x, c
                nEval += 2
            elif fx > f(d):
                a, x = x, d
                nEval += 3
            else:
                a, b = c, d
                nEval += 3
            if b - a < self.tolerance:
                break
        self.nFuncEval = nEval
        self.nIterations = i
        self.nAllFuncEval += self.nFuncEval
        self.nAllIterations += i
        return x

    def _monitor(
        self,
        i: int,
        nEval: int,
        fx: float,
        a: float,
        c: float,
        x: float,
        d: float,
        b: float,
    ) -> None:
        print(
            f"    {i:3d} {nEval:3d} f(x) = {fx:.3e} (a, c, x, d, b) = ({format_floats(a, c, x, d, b)})"
        )


class GoldenRatioScalarOptimizer(BisectionScalarOptimizer):
    def __call__(
        self, f: Callable[[float], float], interval: Tuple[float, float]
    ) -> float:
        from math import sqrt

        (a, b) = interval
        phi = (1 + sqrt(5.0)) / 2.0
        d = a + (b - a) / phi
        c = a + b - d
        fc, fd = f(c), f(d)
        nEval = 2
        for i in range(self.max_iterations):
            if self.verbose:
                x = (c + d) / 2.0
                fx = (fc + fd) / 2.0
                self._monitor(i, nEval, fx, a, c, x, d, b)
            if fc < fd:
                b, d = d, c
                c = a + b - d
                fd, fc = fc, f(c)
                nEval += 1
            elif fc > fd:
                a, c = c, d
                d = a + b - c
                fc, fd = fd, f(d)
                nEval += 1
            else:
                a, b = c, d
                d = a + (b - a) / phi
                c = a + b - d
                fc, fd = f(c), f(d)
                nEval += 2
            if b - a < self.tolerance:
                break
        x = (a + b) / 2.0
        self.nFuncEval = nEval
        self.nIterations = i
        self.nAllFuncEval += self.nFuncEval
        self.nAllIterations += i
        return x

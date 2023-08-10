# This simplifies imports of different problems
from .himmelblau import Himmelblau
from .regression_problem import RegressionProblem
from .simple_quadratic import SimpleQuadratic2D, SimpleQuadraticND
from .rosenbrock import RosenBrockND

__all__ = [
    "Himmelblau",
    "RegressionProblem",
    "SimpleQuadratic2D",
    "SimpleQuadraticND",
    "RosenBrockND",
]

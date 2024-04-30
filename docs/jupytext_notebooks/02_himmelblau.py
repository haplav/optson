# %% [markdown]
# # Himmelblau's Function
#
# In this example, we solve the slightly more complicated function, the Himmelblau's 2-dimensional function.
# This function is not quadratic and contains several local minima.
#
# ## Imports

# %%
from typing import List, Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from numpy.linalg import norm

from optson.call_counter import class_method_call_counter, get_method_count
from optson.gradient_test import GradientTest
from optson.ls import ScalarOptimizerLSStepSize
from optson.methods import AdamUpdate, SteepestDescentUpdate, BasicTRUpdate
from optson.model import Model, ModelProxy
from optson.monitor import BasicMonitor
from optson.optimizer import Optimizer
from optson.problem import Problem
from optson.update import Update
from optson.utils import InstanceOrType, get_instance
from optson.vector import InVec, Scalar, Vec, as_vec

# %% [markdown]
# ## Implementing Himmelblau's Problem
# We first implement the problem with out typical template. We need to implement the methods `f` and `g`
# and use the correct function signature.


# %%
@class_method_call_counter
class Himmelblau(Problem):
    """Himmelblau's 2-dimensional function."""

    def f(self, model: ModelProxy, indices: Optional[List[int]]) -> Scalar:
        X = model.x[0]
        Y = model.x[1]
        return (X**2 + Y - 11) ** 2 + (X + Y**2 - 7) ** 2

    def g(self, model: ModelProxy, indices: Optional[List[int]]) -> Vec:
        X = model.x[0]
        Y = model.x[1]
        return as_vec(
            [
                2 * (2 * X * (X**2 + Y - 11) + X + Y**2 - 7),
                2 * (X**2 + 2 * Y * (X + Y**2 - 7) + Y - 11),
            ]
        )


# %% [markdown]
# ## Gradient Testing the problem
# Here, we confirm that we implemented our problem correctly. If we did things correctly the gradient
# test should give us a curve that looks roughly like a hockey stick with small relative errors.
# %%
h = np.logspace(-15, 2, 50)
gt = GradientTest(x0=np.array([4, 4]), h=h, problem=Himmelblau())
gt.plot()


# %% [markdown]
# All looks well!
#
# We now implement a convenience function to help us plot our results.


# %% Solving the problem
def solve_and_plot(x0: InVec, update: InstanceOrType[Update], max_iterations: int = 20):
    # Make data.
    X = np.arange(-5, 5, 0.05)
    Y = np.arange(-5, 5, 0.05)
    problem = Himmelblau()
    X, Y = np.meshgrid(X.flatten(), Y.flatten())
    Z = problem.f(Model.wrap_vec([X, Y]), indices=None)  # type: ignore [list-item]

    # Plot misfit surface
    fig, axs = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(8, 8))
    axs.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    # Set the orthographic projection.
    axs.set_proj_type("ortho")  # FOV = 0 deg

    opt = Optimizer(problem=problem, update=get_instance(update), store_models=True)
    step = 1 if max_iterations <= 20 else 10
    opt.monitor = BasicMonitor(step=step)
    opt.stopping_criterion.max_iterations = max_iterations
    m = opt.iterate(x0=x0)
    print(f"x = {m.x}\nf(x) = {m.fx}\n||g(x)|| = {norm(m.gx)}")
    print(f"# function evaluations: {get_method_count(problem.f)}")
    print(f"# gradient evaluations: {get_method_count(problem.g)}")

    x = [model.x[0] for model in opt.models]
    y = [model.x[1] for model in opt.models]
    z = [model.fx for model in opt.models]

    axs.plot3D(x, y, z, "r-*")
    plt.show()


# %% [markdown]
# ## Solve using a trust-region method
# %%
solve_and_plot(x0=[4.0, 4.0], update=BasicTRUpdate)

# %% [markdown]
# ## Using a Steepest descent with simple backtracking line-search
# %%
solve_and_plot(x0=[4.0, 4.0], update=SteepestDescentUpdate)

# %% [markdown]
# ## Using Steepest descent with the default `GoldenRatioScalarOptimizer`

# %%
solve_and_plot(
    x0=[4.0, 4.0], update=SteepestDescentUpdate(step_size=ScalarOptimizerLSStepSize)
)

# %% [markdown]
# ## Using Adam

# %%
solve_and_plot(x0=[4.0, 4.0], update=AdamUpdate(), max_iterations=100)

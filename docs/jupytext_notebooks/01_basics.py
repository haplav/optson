# %% [markdown]
# # A Basic Problem

# ## Imports

# %%
from typing import List, Optional

import numpy as np

from optson.call_counter import class_method_call_counter, get_method_count
from optson.gradient_test import GradientTest
from optson.methods.steepest_descent import SteepestDescentUpdate
from optson.model import ModelProxy
from optson.optimizer import Optimizer
from optson.problem import Problem
from optson.stopping_criterion import BasicStoppingCriterion
from optson.vector import Scalar, Vec, as_vec

# %% [markdown]
# ## 1. Defining a Problem

# %% [markdown]
# The first step in using **Optson** is to define the problem that you would like to solve.
# Users can define their problem by defining a class that inherits from the abstract class
# [`Problem`](optson.problem.Problem)
# There are also a few sample problems available in [`Problems`](optson.problems).
#
# By deriving the class from the [`Problem`](optson.problem.Problem)
# class, **Optson** ensures the methods ``f`` and ``g`` are provided with the correct signature.
# The methods take a [`ModelProxy`](optson.model.ModelProxy) as an input.
# It is a frozen dataclass wrapping the current solution vector.
# Additionally, it provides iteration and other relevant metadata related to the current Model.
# This object is typically an instance of [`Model`](optson.model.Model) managed by Optson, not the user.
# Below is an example of how the definition of a problem looks like in practice:


# %%
@class_method_call_counter
class SimpleQuadratic2D(Problem):
    def f(self, model: ModelProxy, indices: Optional[List[int]]) -> Scalar:
        x = model.x
        return 100 * x[0] ** 2 + 200 * x[1] ** 2

    def g(self, model: ModelProxy, indices: Optional[List[int]]) -> Vec:
        x = model.x
        return as_vec([200 * x[0], 400 * x[1]])


# %% [markdown]
# The `indices` parameter refers to the data sample indices.
# By default, it is `None` which means *all* indices.
# Values other than `None` are only relevant in the context of stochastic optimization.
# We will see this later in [Mini-batch regression](03_mini_batch_regression).


# %% [markdown]
# ## 2. Gradient testing the problem
#
# Optson has the built-in option to perform a gradient test.
# This tests if ``g`` is
# the actual gradient of ``f`` by comparing changes in ``f`` by predictions based on ``g``.
# If everything is fine, relative errors should be small and the curve should have the shape of a hockeystick.

# %%
# a list of perturbations to compare predicted changes in f with actual changes in f.
h = np.logspace(-10, 1, 40)
gt = GradientTest(x0=[4, 4], h=h, problem=SimpleQuadratic2D())

gt.plot()

# %% [markdown]
# ## 3. Calling the Optimizer
# We can then proceed to perform the optimization. For this purpose, we import [`Optimizer`](optson.optimizer.Optimizer).
# The optimizer can be initalized with settings that define the update rules and stopping criteria.
# By default, it uses [`BasicTRUpdate`](optson.methods.trust_region.BasicTRUpdate)
# to perform model updates, which is a trust-region approach
# that uses L-BFGS to approximate the Hessian.

# %%
pb = SimpleQuadratic2D()
# We perform 5 parameter updates.
sc = BasicStoppingCriterion(max_iterations=5)
opt = Optimizer(problem=pb, stopping_criterion=sc)

solution = opt.iterate(x0=[1.0, 1.0])

print(f"Found solution {solution.x} in {solution.iteration} iterations")

print(f"Required {get_method_count(pb.f)} function calls to f.")

# Access the solution vector
print(f"Solution vector: {solution.x}")

# %% [markdown]
# ### 3.1 Let's try different settings
# We can now solve the same problem using a steepest descent algorithm. For this purpose, we use
# [`SteepestDescentUpdate`](optson.methods.steepest_descent.SteepestDescentUpdate).
# By passing a ``stopping_criterion``,
# such as: [`BasicStoppingCriterion`](optson.stopping_criterion.BasicStoppingCriterion),
# we can control how many updates are performed. We also set ``store_models=True``.
# This tells **Optson** to store all the intermediate solutions.

# %%
sc = BasicStoppingCriterion(max_iterations=5)
opt = Optimizer(
    problem=pb, update=SteepestDescentUpdate(), stopping_criterion=sc, store_models=True
)

solution = opt.iterate(x0=[1.0, 1.0])

# print all models
for m in opt.models:
    print(m.x)


# %% [markdown]
# This concludes the first tutorial but there are more if you like.

# %%

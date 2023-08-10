# %% [markdown]
# # Mini-batch Regression
#
# Here we solve a simple regression problem using stochastic optimization.

# ## Imports

# %%
from typing import List, Optional
import numpy as np
import pylab  # type: ignore
from sklearn.datasets import make_regression  # type: ignore
from optson.batch_manager import ControlGroupBatchManager, SimpleBatchManager
from optson.gradient_test import GradientTest
from optson.methods import AdamUpdate, SteepestDescentUpdate, BasicTRUpdate
from optson.model import ModelProxy
from optson.monitor import BasicMonitor
from optson.optimizer import Optimizer
from optson.problem import Problem
from optson.stopping_criterion import BasicStoppingCriterion
from optson.vector import Scalar, Vec

# %% [markdown]
# ## Implementing the problem
# We start off by implementing the problem. In this case, we will implement
# a problem that uses the optional argument `indices` to compute function values and
# gradients only for a subset of the samples.
# If `None` is passed, we use all samples.

# %%
n_samples = 500


class RegressionProblem(Problem):
    def __init__(
        self,
        n_samples: int,
        random_seed: int = 42,
        n_features: int = 1,
        n_informative: int = 1,
        noise: float = 35.0,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.x, self.y, self.coeff = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            random_state=0,
            noise=noise,
            coef=True,
        )
        self.x = np.c_[np.ones_like(self.x), self.x]

    def f(self, model: ModelProxy, indices: Optional[List[int]]) -> Scalar:
        """Compute the loss."""
        x = model.x
        (x_batch, y_batch) = (
            (self.x, self.y) if indices is None else (self.x[indices], self.y[indices])
        )
        batch_size = len(x_batch)
        hypothesis = np.dot(x_batch, x)
        loss = hypothesis - y_batch
        return np.sum(loss**2) / (2 * batch_size)

    def g(self, model: ModelProxy, indices: Optional[List[int]]) -> Vec:
        """Compute the gradient."""
        x = model.x
        (x_batch, y_batch) = (
            (self.x, self.y) if indices is None else (self.x[indices], self.y[indices])
        )
        batch_size = len(x_batch)
        hypothesis = np.dot(x_batch, x)
        loss = hypothesis - y_batch
        return np.dot(x_batch.T, loss) / batch_size


# %% [markdown]
# ## Gradient testing the problem
# We test if we correctly computed the gradient, if this is the case, we should see a hockerstick shaped graph
#
# %%
h = np.logspace(-10, 3, 15)
grdtest = GradientTest(
    x0=np.zeros(2),
    h=h,
    problem=RegressionProblem(n_samples=n_samples),
    batch_manager=ControlGroupBatchManager(n_samples=n_samples),
)
grdtest.plot()
print(f"Minimum error: {min(grdtest.relative_errors):.2e}")

# %% [markdown]
# ## Standard non-stochastic L-BFGS

# %%
sc = BasicStoppingCriterion(tolerance=1e-6, max_iterations=200)
update = BasicTRUpdate(verbose=True)
rp = RegressionProblem(n_samples=500, noise=30, random_seed=42)
monitor = BasicMonitor(step=1)
opt = Optimizer(
    problem=rp,
    update=update,
    stopping_criterion=sc,
    monitor=monitor,
)
m = opt.iterate(x0=np.array([0, 0]))

theta = m.x
for _ in range(rp.x.shape[1]):
    y_predict = theta[0] + theta[1] * rp.x
pylab.plot(rp.x[:, 1], rp.y, "o")
pylab.plot(rp.x, y_predict, "k-")
pylab.show()

print(f"Solution: {m.x}")

# %% [markdown]
# ## Using steepest descent.

# %%
rp = RegressionProblem(n_samples=500, noise=30, random_seed=42)
opt = Optimizer(
    problem=rp,
    update=SteepestDescentUpdate,
    stopping_criterion=sc,
    monitor=monitor,
)
m = opt.iterate(x0=np.array([0, 0]))


theta = m.x
for _ in range(rp.x.shape[1]):
    y_predict = theta[0] + theta[1] * rp.x
pylab.plot(rp.x[:, 1], rp.y, "o")
pylab.plot(rp.x, y_predict, "k-")
pylab.show()

print(f"Solution: {m.x}")

# %% [markdown]
# ## Solving the problem with stochastic optimization
# By passing a batch manager, we can solve the same problem stochastically.
# Optson will then pass a list of indices to `Problem.f` and `Problem.g`.
# Obviously, the `Problem` implementation must support that.
#
# Here, we pass the `ControlGroupBatchManager`. This uses overlapping batches,
# which enables Optson to use line-search or Trust-region approaches, even when using subsets of the data.
#
# The `ControlGroupBatchManager` may grow the batch size as needed.

# %%
sc = BasicStoppingCriterion(tolerance=1e-4, max_iterations=20)
update = BasicTRUpdate(verbose=False)
rp = RegressionProblem(n_samples=500, noise=30, random_seed=42)
monitor = BasicMonitor(step=3)
batch_manager = ControlGroupBatchManager(
    batch_size=4,
    n_samples=n_samples,
    batch_grow_factor=2.0,
    extend_control_group_factor=0.2,
)
opt = Optimizer(
    problem=rp,
    update=update,
    stopping_criterion=sc,
    monitor=monitor,
    batch_manager=batch_manager,
    store_models=True,
)
m = opt.iterate(x0=np.array([0, 0]))
theta = m.x
for _ in range(rp.x.shape[1]):
    y_predict = theta[0] + theta[1] * rp.x
pylab.plot(rp.x[:, 1], rp.y, "o")
pylab.plot(rp.x, y_predict, "k-")
pylab.show()

print(f"Solution: {m.x}")

total_samples_used = 0
for m in opt.models:
    assert m.batch is not None
    total_samples_used += len(m.batch)

print(total_samples_used)

# %% [markdown]
# ##  Using Adam
# Here, we use the `SimpleBatchManager`, which defines mini-batches without an overlap with the previous
# model. This batch manager is suitable when no line-search or trust-region is used,
# such as is the case with Adam

# %%
sc = BasicStoppingCriterion(tolerance=1e-6, max_iterations=100)
rp = RegressionProblem(noise=30, random_seed=42, n_samples=500)
monitor = BasicMonitor(step=20)
opt = Optimizer(
    problem=rp,
    update=AdamUpdate(beta_1=0.9, beta_2=0.999, alpha=1.0),
    stopping_criterion=sc,
    monitor=monitor,
    batch_manager=SimpleBatchManager(batch_size=2, n_samples=500),
)
m = opt.iterate(x0=np.array([0, 0]))

theta = m.x
for _ in range(rp.x.shape[1]):
    y_predict = theta[0] + theta[1] * rp.x
pylab.plot(rp.x[:, 1], rp.y, "o")
pylab.plot(rp.x, y_predict, "k-")
pylab.show()

print(f"Solution: {m.x}")

# %%

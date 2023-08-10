import numpy as np

from optson.batch_manager import SimpleBatchManager
from optson.methods.adam import AdamUpdate
from optson.optimizer import Optimizer
from optson.problems import RegressionProblem
from optson.stopping_criterion import BasicStoppingCriterion


def test_adam():
    sc = BasicStoppingCriterion(tolerance=1e-5, max_iterations=100)
    rp = RegressionProblem(noise=0, random_seed=42)
    update = AdamUpdate(alpha=2.0)
    opt = Optimizer(
        problem=rp,
        update=update,
        stopping_criterion=sc,
        batch_manager=SimpleBatchManager(n_samples=500, batch_size=4),
    )
    m = opt.iterate(x0=np.array([0, 0]))
    np.testing.assert_array_almost_equal(m.x, [0.0, 45.7], decimal=0)
    assert isinstance(m._batch_manager, SimpleBatchManager)

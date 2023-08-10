import os
from typing import Type

import numpy as np
import pytest

from optson.methods import (
    AdamUpdate,
    QuasiNewtonUpdate,
    SteepestDescentUpdate,
    BasicTRUpdate,
)
from optson.optimizer import Optimizer
from optson.problem import Problem
from optson.problems import RosenBrockND, SimpleQuadraticND
from optson.update import Update

try:
    import torch

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False

if TORCH_AVAIL:

    @pytest.mark.parametrize(
        "update_cls",
        [SteepestDescentUpdate, AdamUpdate, BasicTRUpdate, QuasiNewtonUpdate],
    )
    @pytest.mark.parametrize(
        "problem_cls",
        [SimpleQuadraticND, RosenBrockND],
    )
    @pytest.mark.parametrize(
        "x0",
        [
            torch.ones((2, 2, 2)) * 3,
            torch.ones(10, dtype=torch.float32) * 2,
            torch.ones(10, dtype=torch.float64) * 2,
        ],
    )
    def test_cache(
        update_cls: Type[Update],
        problem_cls: Type[Problem],
        x0: torch.Tensor,
    ) -> None:
        state_file = "test_cache.h5"
        if os.path.exists(state_file):
            os.remove(state_file)

        prob = problem_cls()
        # if array_type == "torch":

        opt = Optimizer(problem=prob, state_file=state_file, update=update_cls)
        opt.stopping_criterion.max_iterations = 2
        opt.stopping_criterion.tolerance = 1e-20
        m_cache = opt.iterate(x0=x0)

        prob = problem_cls()

        opt = Optimizer(problem=prob, state_file=state_file, update=update_cls)
        opt.stopping_criterion.max_iterations = 3
        opt.stopping_criterion.tolerance = 1e-20
        m_cache = opt.iterate(x0=x0)

        prob = problem_cls()

        # Get optimizer from cache
        opt = Optimizer(problem=prob, state_file=state_file, update=update_cls)
        opt.stopping_criterion.max_iterations = 4
        opt.stopping_criterion.tolerance = 1e-20
        m_cache = opt.iterate(x0=x0)

        # Compare to solution without cache
        opt = Optimizer(problem=problem_cls, update=update_cls)
        opt.stopping_criterion.max_iterations = 4
        opt.stopping_criterion.tolerance = 1e-20
        m_no_cache = opt.iterate(x0=x0)
        # assert isinstance(m_no_cache.x, torch.Tensor)
        np.testing.assert_array_almost_equal(m_no_cache.x, m_cache.x)

        assert m_no_cache.x.dtype == m_cache.x.dtype == x0.dtype
        assert isinstance(m_no_cache.x, torch.Tensor)
        assert isinstance(m_cache.x, torch.Tensor)

        os.remove(state_file)

    @pytest.mark.parametrize(
        "update_cls",
        [SteepestDescentUpdate, AdamUpdate, BasicTRUpdate, QuasiNewtonUpdate],
    )
    def test_shape_irrelevance(update_cls: Type[Update]) -> None:
        opt = Optimizer(problem=SimpleQuadraticND, update=update_cls)
        opt.stopping_criterion.max_iterations = 5
        x0 = torch.ones((2, 2, 2, 2)) * 2
        m1 = opt.iterate(x0=x0)

        opt = Optimizer(problem=SimpleQuadraticND, update=update_cls)
        opt.stopping_criterion.max_iterations = 5
        x0_flat = x0.flatten()
        m2 = opt.iterate(x0=x0_flat)
        np.testing.assert_array_almost_equal(
            m1.x, torch.reshape(m2.x, x0.shape), decimal=8
        )

    @pytest.mark.parametrize(
        "update_cls",
        [
            SteepestDescentUpdate,
            AdamUpdate(alpha=0.01),
            BasicTRUpdate,
            QuasiNewtonUpdate,
        ],
    )
    def test_2D_rosenbrock(update_cls: Type[Update]) -> None:
        opt = Optimizer(problem=RosenBrockND, update=update_cls)
        opt.stopping_criterion.max_iterations = 10000
        opt.stopping_criterion.tolerance = 5e-3
        x0 = torch.ones(2, dtype=torch.float32) * 1.5
        m = opt.iterate(x0=x0)
        solution = torch.ones_like(x0)
        print(m.x, m.iteration)
        np.testing.assert_array_almost_equal(m.x, solution, decimal=2)

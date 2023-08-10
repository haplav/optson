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
    import jax.numpy as jnp
    from jax import Array

    JAX_AVAIL = True
except ImportError:
    JAX_AVAIL = False

if JAX_AVAIL:

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
        [jnp.ones((2, 2, 2)) * 3, jnp.ones(10) * 2],
    )
    def test_cache(
        update_cls: Type[Update],
        problem_cls: Type[Problem],
        x0: Array,
    ) -> None:
        state_file = "test_cache.h5"
        if os.path.exists(state_file):
            os.remove(state_file)

        opt = Optimizer(problem=problem_cls(), state_file=state_file, update=update_cls)
        opt.stopping_criterion.max_iterations = 2
        opt.stopping_criterion.tolerance = 1e-20
        m_cache = opt.iterate(x0=x0)

        # Get optimizer from cache
        opt = Optimizer(problem=problem_cls(), state_file=state_file, update=update_cls)
        opt.stopping_criterion.max_iterations = 3
        opt.stopping_criterion.tolerance = 1e-20
        m_cache = opt.iterate(x0=x0)

        # Get optimizer from cache
        opt = Optimizer(problem=problem_cls(), state_file=state_file, update=update_cls)
        opt.stopping_criterion.max_iterations = 4
        opt.stopping_criterion.tolerance = 1e-20
        m_cache = opt.iterate(x0=x0)

        # Compare to solution without cache
        opt = Optimizer(problem=problem_cls(), update=update_cls)
        opt.stopping_criterion.max_iterations = 4
        opt.stopping_criterion.tolerance = 1e-20
        m_no_cache = opt.iterate(x0=x0)

        np.testing.assert_array_almost_equal(m_no_cache.x, m_cache.x)
        assert isinstance(m_no_cache.x, Array)
        assert isinstance(m_cache.x, Array)
        os.remove(state_file)

    @pytest.mark.parametrize(
        "update_cls",
        [SteepestDescentUpdate, AdamUpdate, BasicTRUpdate, QuasiNewtonUpdate],
    )
    def test_shape_irrelevance(update_cls: Type[Update]) -> None:
        opt = Optimizer(problem=SimpleQuadraticND, update=update_cls)
        opt.stopping_criterion.max_iterations = 5
        x0 = jnp.ones((2, 2, 2, 2)) * 2
        m1 = opt.iterate(x0=x0)

        opt = Optimizer(problem=SimpleQuadraticND, update=update_cls)
        opt.stopping_criterion.max_iterations = 5
        x0_flat = x0.flatten()
        m2 = opt.iterate(x0=x0_flat)
        np.testing.assert_array_almost_equal(
            m1.x, jnp.reshape(m2.x, x0.shape), decimal=8
        )
        assert isinstance(m1.x, Array)
        assert isinstance(m2.x, Array)

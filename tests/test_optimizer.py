import os
import pathlib
from typing import Type

import numpy as np
import pytest

from optson.batch_manager import (
    BatchManager,
    ControlGroupBatchManager,
    EmptyBatchManager,
)
from optson.call_counter import get_method_count
from optson.methods import (
    AdamUpdate,
    QuasiNewtonUpdate,
    SteepestDescentUpdate,
    BasicTRUpdate,
)
from optson.optimizer import Optimizer
from optson.problem import Problem
from optson.problems import (
    Himmelblau,
    RegressionProblem,
    RosenBrockND,
    SimpleQuadratic2D,
    SimpleQuadraticND,
)
from optson.update import Update
from optson.vector import Vec


@pytest.fixture
def RP_reference_result() -> Vec:
    opt = Optimizer(
        problem=RegressionProblem(noise=0.0),
        update=BasicTRUpdate(),
        batch_manager=ControlGroupBatchManager,
    )
    opt.stopping_criterion.max_iterations = 1000
    opt.stopping_criterion.tolerance = 1e-5
    m = opt.iterate(x0=np.array([0.0, 0.0]))
    return m.x


@pytest.mark.parametrize(
    "update_cls",
    [SteepestDescentUpdate, AdamUpdate, BasicTRUpdate, QuasiNewtonUpdate],
)
def test_default_optimizer_SQ(update_cls: Type[Update]) -> None:
    opt = Optimizer(problem=SimpleQuadratic2D, update=update_cls)
    opt.stopping_criterion.max_iterations = 300
    m = opt.iterate(x0=np.array([2.0, 2.0]))
    solution = np.zeros(2)
    np.testing.assert_array_almost_equal(m.x, solution)


@pytest.mark.parametrize(
    "update_cls",
    [SteepestDescentUpdate, AdamUpdate, BasicTRUpdate, QuasiNewtonUpdate],
)
def test_default_optimizer_ND(update_cls: Type[Update]) -> None:
    opt = Optimizer(problem=SimpleQuadraticND, update=update_cls)
    opt.stopping_criterion.max_iterations = 300
    x0 = np.ones((2, 2, 2, 2)) * 2
    m = opt.iterate(x0=x0)
    solution = np.zeros_like(x0)
    np.testing.assert_array_almost_equal(m.x, solution, decimal=5)


@pytest.mark.parametrize(
    "update_cls",
    [SteepestDescentUpdate, AdamUpdate, BasicTRUpdate, QuasiNewtonUpdate],
)
def test_shape_irrelevance(update_cls: Type[Update]) -> None:
    opt = Optimizer(problem=SimpleQuadraticND, update=update_cls)
    opt.stopping_criterion.max_iterations = 5
    x0 = np.ones((2, 2, 2, 2)) * 2
    m1 = opt.iterate(x0=x0)

    opt = Optimizer(problem=SimpleQuadraticND, update=update_cls)
    opt.stopping_criterion.max_iterations = 5
    x0_flat = x0.flatten()
    m2 = opt.iterate(x0=x0_flat)
    np.testing.assert_array_almost_equal(m1.x, np.reshape(m2.x, x0.shape), decimal=8)


@pytest.mark.parametrize(
    "update_cls",
    [SteepestDescentUpdate, AdamUpdate, BasicTRUpdate, QuasiNewtonUpdate],
)
def test_default_optimizer_RP(
    update_cls: Type[Update], RP_reference_result: Vec
) -> None:
    opt = Optimizer(problem=RegressionProblem(noise=0.0), update=update_cls)
    opt.stopping_criterion.max_iterations = 1000
    opt.stopping_criterion.tolerance = 1e-5
    m = opt.iterate(x0=np.array([0.0, 0.0]))
    np.testing.assert_array_almost_equal(m.x, RP_reference_result, decimal=4)


@pytest.mark.parametrize(
    "update_cls",
    [SteepestDescentUpdate, AdamUpdate, BasicTRUpdate, QuasiNewtonUpdate],
)
@pytest.mark.parametrize(
    "problem_cls",
    [SimpleQuadratic2D, RegressionProblem, Himmelblau, RosenBrockND],
)
def test_cache(update_cls: Type[Update], problem_cls: Type[Problem]) -> None:
    state_file = pathlib.Path("test_cache.h5")
    if os.path.exists(state_file):
        os.remove(state_file)

    prob = problem_cls()
    if isinstance(prob, RosenBrockND):
        x0 = np.ones((2)) * 5.0
    else:
        x0 = np.array([2.0, 2.0])

    batch_manager_cls: Type[BatchManager]
    if isinstance(prob, RegressionProblem):
        batch_manager_cls = ControlGroupBatchManager
    else:
        batch_manager_cls = EmptyBatchManager

    opt = Optimizer(
        problem=prob,
        state_file=state_file,
        update=update_cls,
        batch_manager=batch_manager_cls,
    )
    opt.stopping_criterion.max_iterations = 3
    m_cache = opt.iterate(x0=x0.copy())

    prob = problem_cls()

    # Get optimizer from cache
    opt = Optimizer(
        problem=prob,
        state_file=state_file,
        update=update_cls,
        batch_manager=batch_manager_cls,
    )
    opt.stopping_criterion.max_iterations = 4
    m_cache = opt.iterate(x0=x0.copy())

    # Compare to solution without cache
    opt = Optimizer(
        problem=problem_cls,
        update=update_cls,
        batch_manager=batch_manager_cls,
    )
    opt.stopping_criterion.max_iterations = 4
    m_no_cache = opt.iterate(x0=x0.copy())
    np.testing.assert_array_almost_equal(m_no_cache.x, m_cache.x)

    os.remove(state_file)


@pytest.mark.parametrize(
    "update_cls",
    [SteepestDescentUpdate, AdamUpdate, BasicTRUpdate, QuasiNewtonUpdate],
)
def test_immediate_convergence(update_cls: Type[Update]):
    opt = Optimizer(problem=SimpleQuadratic2D, update=update_cls)
    assert isinstance(opt.update, update_cls)

    opt.stopping_criterion.max_iterations = 3
    x = [0.0, 0.0]
    m = opt.iterate(x)
    assert m.accepted
    assert m.iteration == 0
    assert np.array_equal(x, m.x)


def test_shortcut_H_p():
    from optson.tr import DogLegTRStep
    from optson.update import TRUpdate

    opt = Optimizer(problem=SimpleQuadratic2D, update=BasicTRUpdate)
    H = opt.problem.H

    if hasattr(H.apply_inverse, "_call_count"):
        print(H.apply_inverse._call_count)
    assert get_method_count(H.apply) == 0

    if get_method_count(H.apply_inverse) != 0:
        print(H.apply_inverse._call_count)
    assert get_method_count(H.apply_inverse) == 0

    assert isinstance(opt.update, TRUpdate)
    opt.stopping_criterion.max_iterations = 3
    opt.update.tr_radius.initial = 1e3
    m = opt.iterate([1.0, 1.0])
    assert m.accepted

    assert get_method_count(H.apply) == 0
    assert get_method_count(H.apply_inverse) == 2

    m_trial = opt.update(m)
    assert m_trial.accepted
    H_ = m_trial._problem.H
    assert H_ is H
    assert get_method_count(H.apply) == 0
    assert get_method_count(H.apply_inverse) == 3

    # check shortcut Hinv * H * gx = gx
    tr_step = opt.update.tr_step
    m_trial = tr_step(m, tr_radius=1e12)

    assert not m_trial.accepted
    _ = m_trial.H_p
    assert get_method_count(H.apply) == 0
    assert get_method_count(H.apply_inverse) == 3

    assert isinstance(tr_step, DogLegTRStep)
    assert tr_step.step_type == "full"
    assert np.array_equal(m_trial.H_p, -m.gx)
    assert get_method_count(H.apply) == 0
    assert get_method_count(H.apply_inverse) == 3

    _ = m_trial.H_p
    assert get_method_count(H.apply) == 0
    assert get_method_count(H.apply_inverse) == 3

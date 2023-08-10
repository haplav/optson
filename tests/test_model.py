import dataclasses

import numpy as np
import pytest

from optson.batch_manager import EmptyBatchManager
from optson.call_counter import get_method_count
from optson.model import Model
from optson.problems import SimpleQuadratic2D


@pytest.fixture
def m() -> Model:
    sq = SimpleQuadratic2D()
    x = np.ones(2)
    return Model.create_initial(
        problem=sq,
        x0=x,
        batch_manager=EmptyBatchManager(),
        accepted=False,
        store_history=True,
    )


def test_model_gx(m: Model):
    np.testing.assert_array_equal(m.gx, m._problem.g(m, indices=None))
    np.testing.assert_array_almost_equal(m.gx, [200.0, 400.0])


def test_model_gx_equals_gx_cg(m: Model):
    assert m.gx is m.gx_cg


def test_model_fx(m: Model):
    assert m.fx == 300.0
    np.testing.assert_array_equal(m.fx, m._problem.f(m, indices=None))
    assert isinstance(m.fx, float)


def test_model_frozen_x(m: Model):
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.accepted = True  # type: ignore
    with pytest.raises(ValueError):
        m.x *= 2  # type: ignore


def test_array_read_only(m: Model):
    with pytest.raises(ValueError):
        a = m.x
        a *= 2  # type: ignore


def test_new_model(m: Model):
    x_new = m.x * 2.0
    m2 = m.new(x=x_new)
    np.testing.assert_array_almost_equal(m2.x, x_new)
    assert m is m2.previous
    assert m2._problem is m._problem


def test_model_accept(m: Model):
    assert not m.accepted
    m.accept()
    assert m.accepted


def test_call_counter(m: Model):
    prob = m._problem
    for _ in range(5):
        _ = m.fx
        _ = m.gx

    assert get_method_count(prob.f) == 1
    assert get_method_count(prob.g) == 1

    m = m.new(m.x * 2.0)
    assert get_method_count(prob.f) == 1
    assert get_method_count(prob.g) == 1

    for _ in range(5):
        _ = m.fx
        _ = m.gx
    assert get_method_count(prob.f) == 2
    assert get_method_count(prob.g) == 2

    for _ in range(5):
        _ = prob.f(m, indices=None)
        _ = prob.g(m, indices=None)
    assert get_method_count(prob.f) == 7
    assert get_method_count(prob.g) == 7

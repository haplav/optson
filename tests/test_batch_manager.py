from pathlib import Path

import pytest

from optson.batch_manager import ControlGroupBatchManager, SimpleBatchManager


def test_simple_batch_manager():
    bm = SimpleBatchManager(n_samples=40, batch_size=4)

    assert bm.get_batch(0) == [0, 1, 2, 3]
    assert bm.get_batch(1) == [4, 5, 6, 7]
    assert bm.get_control_group(1) == []

    with pytest.raises(ValueError):
        bm.extend_control_group(1)

    bm.get_batch(2)
    with pytest.raises(ValueError):
        bm.get_batch(4)

    for i in range(3, 10):
        bm.get_batch(i)

    with pytest.raises(ValueError):
        bm.get_batch(1)

    assert bm.get_batch(10) == [0, 1, 2, 3]

    bm_state_file = Path("_bm_state.h5")

    if bm_state_file.exists():
        bm_state_file.unlink()

    bm.save(bm_state_file)

    bm = SimpleBatchManager(n_samples=40, batch_size=4)
    bm.load(bm_state_file)

    assert bm.iteration == 10
    assert bm.get_batch(10) == [0, 1, 2, 3]
    bm_state_file.unlink()


def test_basic_batch_manager():
    bm = ControlGroupBatchManager(n_samples=2, batch_size=2)
    assert len(bm.get_batch(0)) == 2
    assert len(bm.get_control_group(0)) == 1

    cg_0 = bm.get_control_group(0)
    bm.get_batch(1)
    assert bm.control_group_previous == cg_0
    bm.extend_control_group(1)
    bm.extend_control_group(1)
    bm.extend_control_group(1)
    bm.get_batch(2)
    assert len(bm.get_control_group(1)) == 2

    bm = ControlGroupBatchManager(n_samples=10, batch_size=2)
    for i in range(10):
        bm.get_batch(i)
        bm.extend_control_group(i)
    bm.get_batch(10)
    assert bm.batch_size == 10
    bm.extend_control_group(10)
    bm.extend_control_group(10)
    bm.extend_control_group(10)
    assert bm.extend_control_group(10) is False


if __name__ == "__main__":
    test_simple_batch_manager()
    test_basic_batch_manager()

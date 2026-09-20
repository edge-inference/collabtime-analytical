import os

import pytest

from benchmark_environment import capture_environment, set_cpu_affinity


def test_environment_capture_records_reproducibility_fields():
    environment = capture_environment()
    assert environment["cpu_model"]
    assert environment["logical_cpu_count"] >= 1
    assert isinstance(environment["cpu_affinity"], list)
    assert environment["python"]


def test_negative_affinity_is_rejected():
    with pytest.raises(ValueError):
        set_cpu_affinity(-1)


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux affinity only")
def test_existing_affinity_can_be_reapplied():
    available = os.sched_getaffinity(0)
    selected = min(available)
    original = set(available)
    try:
        set_cpu_affinity(selected)
        assert os.sched_getaffinity(0) == {selected}
    finally:
        os.sched_setaffinity(0, original)

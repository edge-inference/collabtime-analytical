import math

import pytest

from scheduler_capacity_model import (
    WorkloadClass,
    allen_cunneen_mean_wait_s,
    background_poll_rate,
    scheduler_utilization,
    service_scv,
    task_capacity_after_background_load,
    task_capacity_limits_after_background_load,
    total_cpu_load,
    usl_relative_capacity,
    worker_utilization,
)


def test_operational_cpu_load_and_utilization_use_rates_not_only_tasks():
    workloads = [
        WorkloadClass("task_path", 100.0, 2.0),
        WorkloadClass("idle_poll", 1000.0, 0.1),
    ]
    assert math.isclose(total_cpu_load(workloads), 0.3)
    assert math.isclose(scheduler_utilization(workloads, 2.0), 0.15)


def test_polling_rate_is_robot_driven_background_work():
    assert background_poll_rate(320.0, 0.1) == 3200.0
    with pytest.raises(ValueError):
        background_poll_rate(1.0, 0.0)


def test_task_capacity_reduces_after_background_load():
    capacity = task_capacity_after_background_load(
        effective_cpu_cores=5.0,
        background_cpu_load_cores=1.0,
        cpu_ms_per_task=2.0,
        utilization_limit=0.8,
    )
    assert capacity == 1500.0
    assert (
        task_capacity_after_background_load(
            effective_cpu_cores=1.0,
            background_cpu_load_cores=1.0,
            cpu_ms_per_task=2.0,
        )
        == 0.0
    )


def test_cpu_and_worker_capacity_are_not_conflated():
    background = [
        WorkloadClass(
            "polling",
            arrival_rate_per_s=100.0,
            cpu_ms_per_visit=1.0,
            elapsed_ms_per_visit=4.0,
        )
    ]
    limits = task_capacity_limits_after_background_load(
        effective_cpu_cores=2.0,
        effective_workers=4.0,
        background_workloads=background,
        cpu_ms_per_task=2.0,
        elapsed_ms_per_task=10.0,
    )
    assert math.isclose(limits.cpu_tasks_per_s, 950.0)
    assert math.isclose(limits.worker_tasks_per_s, 360.0)
    assert limits.tasks_per_s == limits.worker_tasks_per_s
    assert limits.bottleneck == "workers"
    assert math.isclose(worker_utilization(background, 4.0), 0.1)


def test_usl_does_not_assume_linear_worker_scaling():
    assert usl_relative_capacity(1, 0.2, 0.01) == 1.0
    assert usl_relative_capacity(8, 0.2, 0.01) < 8.0


def test_allen_cunneen_reduces_to_mm1_for_exponential_single_server():
    arrival_rate = 4.0
    service = 0.2
    expected = service * (arrival_rate * service) / (1.0 - arrival_rate * service)
    actual = allen_cunneen_mean_wait_s(
        arrival_rate_per_s=arrival_rate,
        mean_service_s=service,
        servers=1,
        arrival_scv=1.0,
        service_scv=1.0,
    )
    assert math.isclose(actual, expected)


def test_queue_wait_is_infinite_at_or_above_capacity():
    assert math.isinf(
        allen_cunneen_mean_wait_s(
            arrival_rate_per_s=10.0,
            mean_service_s=0.2,
            servers=2,
            arrival_scv=1.0,
            service_scv=0.5,
        )
    )


def test_service_scv_is_reported_from_distribution():
    assert service_scv([1.0, 1.0, 1.0]) == 0.0
    assert math.isclose(service_scv([1.0, 3.0]), 0.25)

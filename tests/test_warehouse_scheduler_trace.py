import math

from warehouse_scheduler_trace import (
    OperationSample,
    SchedulerInstrumentation,
    _operation_succeeded,
)


class FakeScheduler:
    def __init__(self):
        self.assignments = 0

    def assign_task_to_agent(self, agent_id, position):
        self.assignments += 1
        return (1, 2) if self.assignments % 2 == 0 else None

    def request_path(self, agent_id, start, goal, current_step):
        return [start, goal]


def test_operation_sample_maintains_totals_and_bounded_reservoir():
    sample = OperationSample(reservoir_size=3, seed=11)
    for index in range(10):
        sample.observe(100 + index, 80 + index, success=index % 2 == 0, exception=False)

    summary = sample.summary()
    assert summary["calls"] == 10
    assert summary["successes"] == 5
    assert summary["misses"] == 5
    assert summary["sample_count"] == 3
    assert math.isclose(summary["mean_wall_ms"], 104.5 / 1_000_000.0)
    assert summary["success_mean_cpu_ms"] < summary["miss_mean_cpu_ms"]


def test_scheduler_instrumentation_counts_successes_and_restores_methods():
    scheduler = FakeScheduler()
    original_assign = scheduler.assign_task_to_agent
    instrumentation = SchedulerInstrumentation(
        scheduler,
        method_names=("assign_task_to_agent", "request_path"),
        reservoir_size=10,
        seed=7,
    )

    assert scheduler.assign_task_to_agent(0, 1) is None
    assert scheduler.assign_task_to_agent(0, 1) == (1, 2)
    assert scheduler.request_path(0, 1, 2, 0) == [1, 2]
    summaries = instrumentation.summaries()

    assert summaries["assign_task_to_agent"]["calls"] == 2
    assert summaries["assign_task_to_agent"]["successes"] == 1
    assert summaries["assign_task_to_agent"]["misses"] == 1
    assert summaries["request_path"]["successes"] == 1

    instrumentation.restore()
    assert scheduler.assign_task_to_agent.__func__ is original_assign.__func__


def test_operation_success_semantics_are_explicit():
    assert not _operation_succeeded("assign_task_to_agent", None)
    assert _operation_succeeded("assign_task_to_agent", (1, 2))
    assert not _operation_succeeded("request_path", [])
    assert _operation_succeeded("request_path", [1, 2])
    assert not _operation_succeeded("other", False)
    assert _operation_succeeded("other", None)

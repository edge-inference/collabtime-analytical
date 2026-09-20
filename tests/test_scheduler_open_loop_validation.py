import math

from scheduler_open_loop_validation import (
    TimedRequest,
    poisson_offsets_ns,
    summarize_requests,
    summarize_replications,
)


def test_poisson_arrivals_are_open_loop_and_bounded_by_duration():
    offsets = poisson_offsets_ns(100.0, 2.0, seed=4)
    assert offsets == sorted(offsets)
    assert offsets
    assert offsets[-1] < 2_000_000_000


def test_request_summary_separates_queue_service_and_response():
    requests = [
        TimedRequest(0, 0, 10, 20, 50, 25, 1),
        TimedRequest(1, 100, 110, 130, 180, 40, 1),
    ]
    row = summarize_requests(
        requests,
        execution_model="threads",
        workers=1,
        target_rate=1.0,
        saturation_tps=10.0,
        load_fraction=0.1,
        duration_s=1.0,
        backlog_at_end=0,
        achieved_tps=2.0,
    )
    assert math.isclose(row["mean_queue_ms"], 15e-6)
    assert math.isclose(row["mean_service_ms"], 40e-6)
    assert math.isclose(row["mean_response_ms"], 55e-6)
    assert row["success_rate"] == 1.0


def test_replication_summary_keeps_small_sample_uncertainty():
    saturation = [
        {
            "seed": 1,
            "execution_model": "threads",
            "workers": 1,
            "saturation_tps": 100.0,
            "success_rate": 1.0,
        },
        {
            "seed": 2,
            "execution_model": "threads",
            "workers": 1,
            "saturation_tps": 120.0,
            "success_rate": 1.0,
        },
    ]
    queue = []
    for seed, duration_ns in ((1, 1_000_000), (2, 2_000_000)):
        row = summarize_requests(
            [TimedRequest(0, 0, 0, 0, duration_ns, duration_ns, 1)],
            execution_model="threads",
            workers=1,
            target_rate=10.0,
            saturation_tps=100.0,
            load_fraction=0.1,
            duration_s=1.0,
            backlog_at_end=0,
            achieved_tps=1.0,
        )
        queue.append({"seed": seed, **row})

    saturation_summary, queue_summary = summarize_replications(
        saturation, queue
    )
    assert saturation_summary[0]["replicates"] == 2
    assert saturation_summary[0]["saturation_tps"] == 110.0
    assert saturation_summary[0]["saturation_tps_ci"] > 0
    assert queue_summary[0]["replicates"] == 2
    assert math.isclose(queue_summary[0]["mean_service_ms"], 1.5)

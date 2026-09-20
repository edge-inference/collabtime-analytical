import math

from analyze_scheduler_trace import (
    _mean_ci,
    enrich_rows,
    validate_operational_identities,
)


def test_mean_ci_uses_small_sample_student_t_interval():
    mean, half_width = _mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(mean, 3.0)
    assert math.isclose(half_width, 1.963243161477561, rel_tol=1e-12)


def test_trace_enrichment_reconstructs_rates_and_scenario_discrepancy():
    aggregates = [
        {
            "fleet_size": "800",
            "seed": "1",
            "task_rate_tps": "4.8",
            "observation_s": "100",
            "completion_throughput_tps": "4",
            "terminal_throughput_tps": "4",
            "created_tasks": "400",
            "terminal_failed_tasks": "0",
            "active_tasks_start": "10",
            "active_tasks_end": "10",
            "active_task_delta": "0",
            "mean_active_tasks": "10",
            "task_flow_balance_error": "0",
            "active_task_drift_fraction_of_created": "0",
            "completion_to_created_ratio": "1",
            "mean_idle_robots": "2",
            "scheduler_cpu_s": "2",
            "trace_process_cpu_s": "4",
            "scheduler_cpu_ms_per_completed_task": "5",
            "scheduler_elapsed_ms_per_completed_task": "5.1",
            "completed_tasks": "400",
            "claimed_tasks": "410",
            "failed_task_attempts": "10",
        }
    ]
    operations = [
        {
            "fleet_size": "800",
            "seed": "1",
            "operation": "assign_task_to_agent",
            "calls": "2000",
            "successes": "400",
            "misses": "1600",
            "mean_cpu_ms": "0.5",
            "success_mean_cpu_ms": "0.6",
            "miss_mean_cpu_ms": "0.475",
            "total_cpu_s": "1",
            "cpu_ms_per_completed_task": "2.5",
            "calls_per_completed_task": "5",
        },
        {
            "fleet_size": "800",
            "seed": "1",
            "operation": "request_path",
            "calls": "400",
            "successes": "400",
            "misses": "0",
            "mean_cpu_ms": "2.5",
            "success_mean_cpu_ms": "2.5",
            "miss_mean_cpu_ms": "nan",
            "total_cpu_s": "1",
            "cpu_ms_per_completed_task": "2.5",
            "calls_per_completed_task": "1",
        },
    ]
    enriched_aggregates, enriched_operations = enrich_rows(aggregates, operations)
    aggregate = enriched_aggregates[0]
    assert math.isclose(aggregate["scheduler_cpu_core_load"], 0.02)
    assert math.isclose(
        aggregate["analytical_to_trace_elapsed_ratio"], 2800.0 / 5.1
    )
    assert math.isclose(aggregate["failed_attempts_per_completed_task"], 0.025)
    assert enriched_operations[0]["calls_per_s"] == 20.0

    checks = validate_operational_identities(
        enriched_aggregates, enriched_operations
    )
    assert checks[0]["load_identity_relative_error"] == 0.0
    assert checks[0]["poll_rate_relative_error"] == 0.0
    assert checks[0]["task_flow_balance_error"] == 0

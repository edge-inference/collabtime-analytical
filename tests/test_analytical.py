import math
from pathlib import Path

from comparison import DSMCentralizedComparison
from models import AgeOfInformationModel, PerformanceModel, RoutingParams
from queue_models import QueueModel
from sweep_propagation_sensitivity import (
    first_propagation_crossover,
    run_sweep as run_propagation_sweep,
)
from sweep_scheduler_sensitivity import run_sweep as run_scheduler_sweep


ROOT = Path(__file__).resolve().parents[1]


def test_large_multiserver_queue_is_numerically_stable():
    queue = QueueModel(
        arrival_rate=0.006,
        service_time=45_000.0,
        servers=480,
        Ca2=1.0,
        Cs2=0.5,
    )

    delay = queue.waiting_time()

    assert math.isfinite(delay)
    assert delay >= 0.0


def test_thesis_strong_config_has_jointly_stable_latency_points():
    comparison = DSMCentralizedComparison(str(ROOT / "config_thesis_strong.yaml"))

    result = comparison.comprehensive_comparison()

    assert max(comparison.config["system"]["arrival_rates"]) == 0.006
    assert result.performance_advantage["latency_sample_count"] > 0
    assert math.isfinite(result.performance_advantage["avg_latency_improvement"])
    assert result.performance_advantage["propagation_crossover"] == 500
    assert result.performance_advantage["scheduler_bottleneck_threshold"] == 600


def test_propagation_uses_expected_period_waits_and_separate_rtts():
    comparison = DSMCentralizedComparison(str(ROOT / "config_thesis_strong.yaml"))
    model = comparison.performance_model.prop_model

    central = model.central_propagation_time(100)
    dsm = model.dsm_propagation_time(100)

    assert central["batch"] == 50.0
    assert dsm["pre"] == 100.0
    assert central["handshake"] == 0.0
    assert dsm["handshake"] == 0.0

    comparison.network_params.central_handshake_rtt = 7.0
    comparison.network_params.dsm_handshake_rtt = 11.0
    assert model.central_propagation_time(100)["handshake"] == 7.0
    assert model.dsm_propagation_time(100)["handshake"] == 11.0


def test_default_task_service_and_capacity_use_one_work_node_visit():
    comparison = DSMCentralizedComparison(str(ROOT / "config_thesis_strong.yaml"))
    performance = comparison.performance_model

    latency = performance.total_latency(50, 0.0, "central")
    assert latency["task"] == 125_000.0
    assert math.isclose(
        performance.stability.fleet_capacity(50),
        50.0 / 125_000.0,
    )


def test_multiple_work_visits_are_consistent_in_latency_and_capacity():
    comparison = DSMCentralizedComparison(str(ROOT / "config_thesis_strong.yaml"))
    performance = PerformanceModel(
        comparison.system_params,
        comparison.network_params,
        comparison.queue_params,
        routing_params=RoutingParams(pickup_fraction=1.0, delivery_fraction=1.0),
    )

    assert performance.total_latency(50, 0.0, "central")["task"] == 170_000.0
    assert math.isclose(
        performance.stability.fleet_capacity(50),
        50.0 / 170_000.0,
    )


def test_configured_claim_backoff_and_poisson_aoi_rate_are_wired():
    comparison = DSMCentralizedComparison(str(ROOT / "config_thesis_strong.yaml"))
    expected_claim_ms = (50.0 / 3.0 + 20.0) / 0.9
    assert math.isclose(
        comparison.performance_model.prop_model.claim_time_dsm(),
        expected_claim_ms,
    )

    aoi = AgeOfInformationModel(target_freshness=300.0)
    rate = aoi.required_update_rate(0.05, service_time=20.0)
    assert math.isclose(rate, -math.log(0.05) / 280.0)


def test_no_propagation_advantage_is_not_reported_as_last_fleet_size():
    comparison = DSMCentralizedComparison(str(ROOT / "config_thesis_strong.yaml"))
    assert comparison._find_crossover([50, 100], [1.0, 1.0], [2.0, 2.0]) is None


def test_propagation_sensitivity_exposes_network_assumptions():
    comparison = DSMCentralizedComparison(str(ROOT / "config_thesis_strong.yaml"))

    link_rows, period_rows = run_propagation_sweep(comparison)
    link_results = {
        row["combined_per_hop_delay_ms"]: row for row in link_rows
    }
    period_results = {
        (row["batch_period_ms"], row["gossip_period_ms"]): row
        for row in period_rows
    }

    assert link_results[12.0]["crossover_n"] == 317
    assert period_results[(100.0, 200.0)]["crossover_n"] == 317
    assert first_propagation_crossover(
        comparison.performance_model.prop_model, 50, 800
    ) == 317
    assert comparison.network_params.hop_delay == 10.0
    assert comparison.network_params.serialization_delay == 2.0


def test_scheduler_parameter_sensitivity_exposes_assumption_dependence():
    comparison = DSMCentralizedComparison(str(ROOT / "config_thesis_strong.yaml"))

    sensitivity_rows, threshold_rows = run_scheduler_sweep(comparison)
    results = {
        (
            row["scheduler_workers"],
            row["base_demand_ms_per_task"],
            row["demand_growth_ms_per_robot_task"],
        ): row
        for row in sensitivity_rows
    }
    thresholds = {
        (row["scheduler_workers"], row["fleet_size"]): row
        for row in threshold_rows
    }

    assert results[(10, 1200.0, 0.0)]["scheduler_bottleneck_threshold_n"] == ""
    assert results[(10, 1200.0, 0.5)]["scheduler_bottleneck_threshold_n"] == 800
    assert results[(10, 1200.0, 2.0)]["scheduler_bottleneck_threshold_n"] == 600
    assert math.isclose(
        thresholds[(10, 800)]["scheduler_demand_threshold_ms_per_task"],
        1562.5,
    )

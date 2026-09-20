import math
from collections import Counter
from itertools import combinations

import numpy as np

from scheduler_component_factorial import (
    ComponentScenario,
    _design_matrix,
    default_scenarios,
    fit_regression,
    orthogonal_array_scenarios,
)


def test_default_design_contains_thesis_baseline_and_factor_variation():
    scenarios = default_scenarios()
    assert ComponentScenario(fleet_size=50) in scenarios
    assert ComponentScenario(fleet_size=800) in scenarios
    assert ComponentScenario(fleet_size=800, task_backlog=800) in scenarios
    assert ComponentScenario(fleet_size=800, reservation_horizon=10) in scenarios
    assert ComponentScenario(fleet_size=800, warehouse_width=40, warehouse_height=30) in scenarios
    assert len(scenarios) == len(set(scenarios))


def test_assignment_design_uses_algorithmic_work_terms():
    rows = [{"active_robots": 100, "assignment_iterations": 200}]
    matrix, names = _design_matrix(rows, "assignment")
    assert names == ("intercept", "active robots / 100", "task iterations / 100")
    assert np.allclose(matrix, [[1.0, 1.0, 2.0]])


def test_combined_factor_block_is_pairwise_balanced():
    scenarios = orthogonal_array_scenarios()
    assert len(scenarios) == 27
    rows = [
        (
            (100, 300, 800).index(scenario.fleet_size),
            ((40, 30), (80, 65), (120, 100)).index(
                (scenario.warehouse_width, scenario.warehouse_height)
            ),
            (50, 200, 800).index(scenario.task_backlog),
            (1, 3, 10).index(scenario.reservation_horizon),
            (0.25, 0.5, 0.8).index(scenario.active_fraction),
        )
        for scenario in scenarios
    ]
    for left, right in combinations(range(5), 2):
        counts = Counter((row[left], row[right]) for row in rows)
        assert len(counts) == 9
        assert set(counts.values()) == {3}


def test_cluster_bootstrap_fit_recovers_mechanistic_linear_relation():
    rows = []
    for scenario_index, active in enumerate((100, 200, 400, 800)):
        for seed in range(3):
            iterations = 50 + 10 * seed
            response = 0.1 + 0.02 * (active / 100) + 0.03 * (iterations / 100)
            rows.append(
                {
                    "scenario_id": f"scenario-{scenario_index}",
                    "active_robots": active,
                    "assignment_iterations": iterations,
                    "mean_assignment_cpu_ms": response,
                }
            )
    result = fit_regression(
        rows,
        "assignment",
        "mean_assignment_cpu_ms",
        bootstrap_samples=100,
        seed=9,
    )
    assert math.isclose(result.coefficients[0], 0.1, abs_tol=1e-10)
    assert math.isclose(result.coefficients[1], 0.02, abs_tol=1e-10)
    assert math.isclose(result.coefficients[2], 0.03, abs_tol=1e-10)

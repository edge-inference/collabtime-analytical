import math

import numpy as np

from scheduler_demand_study import FitResult, fit_model, predict_fit, select_fit


def test_centered_linear_fit_recovers_d0_and_alpha():
    fleet_sizes = np.asarray([50, 100, 200, 300, 500, 600, 700, 800], dtype=float)
    expected_d0 = 1.25
    expected_alpha = 0.0025
    demand = expected_d0 + expected_alpha * fleet_sizes

    fit = fit_model(fleet_sizes, demand, "linear", n_ref=400.0)
    demand_at_ref, alpha = fit.coefficients
    recovered_d0 = demand_at_ref - alpha * fit.n_ref

    assert math.isclose(recovered_d0, expected_d0, rel_tol=1e-12)
    assert math.isclose(alpha, expected_alpha, rel_tol=1e-12)
    assert np.allclose(predict_fit(fit, fleet_sizes), demand)


def test_holdout_selection_prefers_constant_for_constant_data():
    fleet_sizes = [50, 100, 200, 300, 500, 600, 700, 800] * 3
    demand = [0.75] * len(fleet_sizes)
    fits = [
        fit_model(fleet_sizes, demand, model, series="demand")
        for model in ("constant", "linear", "n_log_n", "quadratic")
    ]

    selected = select_fit(fits, series="demand")

    assert selected.model == "constant"


def test_selection_uses_simpler_scaling_law_within_one_standard_error():
    fits = [
        FitResult(
            "demand",
            "linear",
            400.0,
            [1.5, 0.002],
            1.0,
            0.1,
            0.9,
            -10.0,
            0.104,
            0.010816,
            0.0,
        ),
        FitResult(
            "demand",
            "n_log_n",
            400.0,
            [1.5, 0.0002],
            0.9,
            0.09,
            0.91,
            -11.0,
            0.101,
            0.010201,
            0.0,
        ),
        FitResult(
            "demand",
            "quadratic",
            400.0,
            [1.5, 0.002, 1e-7],
            0.8,
            0.08,
            0.92,
            -12.0,
            0.100,
            0.010000,
            0.001000,
        ),
    ]

    assert select_fit(fits, series="demand").model == "linear"

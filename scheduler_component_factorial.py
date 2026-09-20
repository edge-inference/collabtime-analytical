#!/usr/bin/env python3
"""Structured characterization of centralized scheduler component costs.

The experiment varies the implementation factors that actually determine work
instead of fitting fleet size alone.  It records isolated elapsed and thread
CPU time for assignment and full-reservation A* operations.  Visit ratios and
queue behavior are measured by separate studies.  The design combines thesis
fleet baselines, controlled one-factor perturbations, and a three-level
strength-two orthogonal array; it is not described as a full factorial.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scheduler_demand_study import StudyConfig, build_workload
from benchmark_environment import capture_environment, set_cpu_affinity
from astar_work_profile import profile_workload_path


THESIS_FLEETS = (50, 100, 200, 300, 500, 600, 700, 800)
ANCHOR_FLEETS = (100, 300, 800)
MAPS = ((40, 30), (80, 65), (120, 100))
BACKLOGS = (50, 200, 800)
HORIZONS = (1, 3, 10)
ACTIVE_FRACTIONS = (0.25, 0.5, 0.8)


@dataclass(frozen=True, order=True)
class ComponentScenario:
    fleet_size: int
    warehouse_width: int = 120
    warehouse_height: int = 100
    active_fraction: float = 0.8
    task_backlog: int = 200
    assignment_scan_limit: int = 200
    reservation_horizon: int = 3

    @property
    def scenario_id(self) -> str:
        return (
            f"n{self.fleet_size}_m{self.warehouse_width}x{self.warehouse_height}"
            f"_a{self.active_fraction:g}_q{self.task_backlog}"
            f"_l{self.assignment_scan_limit}_h{self.reservation_horizon}"
        )


@dataclass(frozen=True)
class FactorialConfig:
    warehouse_repo: str
    seeds: tuple[int, ...] = (8101, 8102, 8103, 8104, 8105, 8106)
    warmup_operations: int = 20
    samples_per_scenario: int = 80
    bootstrap_samples: int = 2000
    cpu_affinity: int | None = None


@dataclass(frozen=True)
class RegressionResult:
    response: str
    component: str
    predictor_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    ci_low: tuple[float, ...]
    ci_high: tuple[float, ...]
    in_sample_r_squared: float
    leave_scenario_out_rmse_ms: float


def default_scenarios() -> list[ComponentScenario]:
    scenarios = {
        ComponentScenario(fleet_size=fleet_size) for fleet_size in THESIS_FLEETS
    }
    for fleet_size in ANCHOR_FLEETS:
        for width, height in MAPS:
            scenarios.add(
                ComponentScenario(
                    fleet_size=fleet_size,
                    warehouse_width=width,
                    warehouse_height=height,
                )
            )
        for backlog in BACKLOGS:
            scenarios.add(
                ComponentScenario(fleet_size=fleet_size, task_backlog=backlog)
            )
        for horizon in HORIZONS:
            scenarios.add(
                ComponentScenario(
                    fleet_size=fleet_size, reservation_horizon=horizon
                )
            )
        for active_fraction in ACTIVE_FRACTIONS:
            scenarios.add(
                ComponentScenario(
                    fleet_size=fleet_size, active_fraction=active_fraction
                )
            )
    scenarios.update(orthogonal_array_scenarios())
    return sorted(scenarios)


def orthogonal_array_scenarios() -> list[ComponentScenario]:
    """Return an OA(27, 3^5, strength 2) combined-factor design.

    Each pair of factor levels occurs three times.  The linear columns over
    GF(3) are x, y, z, x+y, and x+z.
    """
    scenarios: list[ComponentScenario] = []
    for fleet_level, map_level, backlog_level in product(range(3), repeat=3):
        width, height = MAPS[map_level]
        horizon_level = (fleet_level + map_level) % 3
        active_level = (fleet_level + backlog_level) % 3
        scenarios.append(
            ComponentScenario(
                fleet_size=ANCHOR_FLEETS[fleet_level],
                warehouse_width=width,
                warehouse_height=height,
                active_fraction=ACTIVE_FRACTIONS[active_level],
                task_backlog=BACKLOGS[backlog_level],
                reservation_horizon=HORIZONS[horizon_level],
            )
        )
    return scenarios


def _assignment_iterations(workload: Any, scan_limit: int) -> int:
    assigned = {int(location) for location in workload.active_locations}
    eligible = 0
    inspected = 0
    found = False
    for raw_location in workload.task_locations:
        inspected += 1
        if int(raw_location) in assigned:
            continue
        eligible += 1
        if eligible > scan_limit and found:
            break
        found = True
    return inspected


def _time_call(function, *args, **kwargs) -> tuple[Any, int, int]:
    wall_start = time.perf_counter_ns()
    cpu_start = time.thread_time_ns()
    result = function(*args, **kwargs)
    cpu_ns = time.thread_time_ns() - cpu_start
    wall_ns = time.perf_counter_ns() - wall_start
    return result, wall_ns, cpu_ns


def measure_scenario(
    factorial_config: FactorialConfig,
    scenario: ComponentScenario,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_count = factorial_config.warmup_operations + factorial_config.samples_per_scenario
    study_config = StudyConfig(
        warehouse_repo=factorial_config.warehouse_repo,
        fleet_sizes=(scenario.fleet_size,),
        warehouse_width=scenario.warehouse_width,
        warehouse_height=scenario.warehouse_height,
        active_fraction=scenario.active_fraction,
        task_backlog=scenario.task_backlog,
        assignment_scan_limit=scenario.assignment_scan_limit,
        reservation_horizon=scenario.reservation_horizon,
        replicates=1,
        warmup_transactions=factorial_config.warmup_operations,
        samples_per_replicate=factorial_config.samples_per_scenario,
    )
    workload = build_workload(study_config, scenario.fleet_size, seed, sample_count)
    for index in range(factorial_config.warmup_operations):
        start = int(workload.request_starts[index])
        goal = int(workload.request_goals[index])
        workload.assign(start)
        workload.plan(start, goal, global_reservations=True)

    graph_nodes = int(workload.coords.shape[0])
    directed_edges = int(workload.indices.shape[0])
    active_robots = int(len(workload.active_locations))
    reservation_entries = int(np.count_nonzero(workload.path_values >= 0))
    reservation_slots = int(workload.path_values.size)
    assignment_iterations = _assignment_iterations(
        workload, scenario.assignment_scan_limit
    )
    raw_rows: list[dict[str, Any]] = []
    profile_records: list[tuple[dict[str, Any], list[int], int, int]] = []

    for sample_index in range(factorial_config.samples_per_scenario):
        request_index = factorial_config.warmup_operations + sample_index
        start = int(workload.request_starts[request_index])
        goal = int(workload.request_goals[request_index])
        if sample_index % 2 == 0:
            _, assignment_wall_ns, assignment_cpu_ns = _time_call(
                workload.assign, start
            )
            path, path_wall_ns, path_cpu_ns = _time_call(
                workload.plan, start, goal, global_reservations=True
            )
        else:
            path, path_wall_ns, path_cpu_ns = _time_call(
                workload.plan, start, goal, global_reservations=True
            )
            _, assignment_wall_ns, assignment_cpu_ns = _time_call(
                workload.assign, start
            )

        manhattan_distance = int(
            abs(float(workload.coords[start, 0]) - float(workload.coords[goal, 0]))
            + abs(float(workload.coords[start, 1]) - float(workload.coords[goal, 1]))
        )
        row = {
                "scenario_id": scenario.scenario_id,
                "seed": seed,
                "sample": sample_index,
                **asdict(scenario),
                "active_robots": active_robots,
                "graph_nodes": graph_nodes,
                "directed_edges": directed_edges,
                "reservation_entries": reservation_entries,
                "reservation_slots": reservation_slots,
                "assignment_iterations": assignment_iterations,
                "route_manhattan_cells": manhattan_distance,
                "path_nodes": len(path),
                "path_success": int(bool(path)),
                "assignment_wall_ms": assignment_wall_ns / 1_000_000.0,
                "assignment_cpu_ms": assignment_cpu_ns / 1_000_000.0,
                "path_wall_ms": path_wall_ns / 1_000_000.0,
                "path_cpu_ms": path_cpu_ns / 1_000_000.0,
            }
        raw_rows.append(row)
        profile_records.append((row, path, start, goal))

    # Profile work only after all timing samples so the Python mirror cannot
    # perturb cache, frequency, or allocator state inside the timed sequence.
    for row, path, start, goal in profile_records:
        work = profile_workload_path(workload, start, goal)
        if path != work.path:
            raise RuntimeError(
                "Untimed A* work profiler diverged from the timed Cython path "
                f"for {scenario.scenario_id}, seed={seed}, sample={row['sample']}"
            )
        row.update(
            {
                "profile_path_matches": 1,
                "astar_nodes_expanded": work.nodes_expanded,
                "astar_edge_evaluations": work.edge_evaluations,
                "astar_heap_pushes": work.heap_pushes,
                "valid_reservations_in_corridor": (
                    work.valid_reservations_in_corridor
                ),
            }
        )

    aggregate: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "seed": seed,
        **asdict(scenario),
        "active_robots": active_robots,
        "graph_nodes": graph_nodes,
        "directed_edges": directed_edges,
        "reservation_entries": reservation_entries,
        "reservation_slots": reservation_slots,
        "assignment_iterations": assignment_iterations,
    }
    for name in (
        "route_manhattan_cells",
        "path_nodes",
        "assignment_wall_ms",
        "assignment_cpu_ms",
        "path_wall_ms",
        "path_cpu_ms",
        "astar_nodes_expanded",
        "astar_edge_evaluations",
        "astar_heap_pushes",
        "valid_reservations_in_corridor",
    ):
        values = np.asarray([float(row[name]) for row in raw_rows])
        aggregate[f"mean_{name}"] = float(np.mean(values))
        aggregate[f"p95_{name}"] = float(np.percentile(values, 95))
    aggregate["path_success_rate"] = float(
        np.mean([float(row["path_success"]) for row in raw_rows])
    )
    aggregate["profile_path_match_rate"] = float(
        np.mean([float(row["profile_path_matches"]) for row in raw_rows])
    )
    return aggregate, raw_rows


def _design_matrix(
    rows: Sequence[dict[str, Any]], component: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    if component == "assignment":
        names = ("intercept", "active robots / 100", "task iterations / 100")
        matrix = np.column_stack(
            [
                np.ones(len(rows)),
                [float(row["active_robots"]) / 100.0 for row in rows],
                [float(row["assignment_iterations"]) / 100.0 for row in rows],
            ]
        )
    elif component == "path":
        names = (
            "intercept",
            "graph nodes / 1000",
            "reservation slots / 1000",
            "A* edge evaluations / 1000",
        )
        matrix = np.column_stack(
            [
                np.ones(len(rows)),
                [float(row["graph_nodes"]) / 1000.0 for row in rows],
                [float(row["reservation_slots"]) / 1000.0 for row in rows],
                [float(row["mean_astar_edge_evaluations"]) / 1000.0 for row in rows],
            ]
        )
    else:
        raise ValueError(f"Unknown component: {component}")
    return np.asarray(matrix, dtype=np.float64), names


def _fit_coefficients(
    rows: Sequence[dict[str, Any]], component: str, response: str
) -> tuple[np.ndarray, np.ndarray, float]:
    matrix, _ = _design_matrix(rows, component)
    observed = np.asarray([float(row[response]) for row in rows], dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, observed, rcond=None)
    predicted = matrix @ coefficients
    residual = observed - predicted
    rss = float(np.sum(residual**2))
    tss = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = 1.0 - rss / tss if tss > 0 else 1.0
    return coefficients, predicted, r_squared


def _leave_scenario_out_rmse(
    rows: Sequence[dict[str, Any]], component: str, response: str
) -> float:
    errors: list[float] = []
    scenario_ids = sorted({str(row["scenario_id"]) for row in rows})
    for scenario_id in scenario_ids:
        training = [row for row in rows if row["scenario_id"] != scenario_id]
        testing = [row for row in rows if row["scenario_id"] == scenario_id]
        coefficients, _, _ = _fit_coefficients(training, component, response)
        matrix, _ = _design_matrix(testing, component)
        observed = np.asarray([float(row[response]) for row in testing])
        errors.extend((observed - matrix @ coefficients).tolist())
    return float(np.sqrt(np.mean(np.asarray(errors) ** 2)))


def fit_regression(
    rows: Sequence[dict[str, Any]],
    component: str,
    response: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> RegressionResult:
    coefficients, _, r_squared = _fit_coefficients(rows, component, response)
    _, names = _design_matrix(rows, component)
    scenario_ids = sorted({str(row["scenario_id"]) for row in rows})
    by_scenario = {
        scenario_id: [row for row in rows if row["scenario_id"] == scenario_id]
        for scenario_id in scenario_ids
    }
    rng = np.random.default_rng(seed)
    bootstrapped: list[np.ndarray] = []
    for _ in range(bootstrap_samples):
        sampled_ids = rng.choice(scenario_ids, size=len(scenario_ids), replace=True)
        sampled_rows: list[dict[str, Any]] = []
        for index, scenario_id in enumerate(sampled_ids):
            for row in by_scenario[str(scenario_id)]:
                sampled_rows.append({**row, "scenario_id": f"{scenario_id}:{index}"})
        sample_coefficients, _, _ = _fit_coefficients(
            sampled_rows, component, response
        )
        bootstrapped.append(sample_coefficients)
    bootstrap_array = np.asarray(bootstrapped)
    return RegressionResult(
        response=response,
        component=component,
        predictor_names=names,
        coefficients=tuple(float(value) for value in coefficients),
        ci_low=tuple(float(value) for value in np.percentile(bootstrap_array, 2.5, axis=0)),
        ci_high=tuple(float(value) for value in np.percentile(bootstrap_array, 97.5, axis=0)),
        in_sample_r_squared=r_squared,
        leave_scenario_out_rmse_ms=_leave_scenario_out_rmse(
            rows, component, response
        ),
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    output_dir: Path,
    aggregate_rows: Sequence[dict[str, Any]],
    regressions: Sequence[RegressionResult],
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    for axis, component, response, label in (
        (axes[0], "assignment", "mean_assignment_cpu_ms", "Assignment"),
        (axes[1], "path", "mean_path_cpu_ms", "Path planning"),
    ):
        regression = next(
            result
            for result in regressions
            if result.component == component and result.response == response
        )
        coefficients = np.asarray(regression.coefficients)
        matrix, _ = _design_matrix(aggregate_rows, component)
        predicted = matrix @ coefficients
        observed = np.asarray([float(row[response]) for row in aggregate_rows])
        colors = [float(row["fleet_size"]) for row in aggregate_rows]
        axis.scatter(observed, predicted, c=colors, cmap="viridis", s=20, alpha=0.8)
        lower = min(float(np.min(observed)), float(np.min(predicted)))
        upper = max(float(np.max(observed)), float(np.max(predicted)))
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        axis.set_xlabel("Measured thread CPU time (ms/operation)")
        axis.set_ylabel("Mechanistic-model prediction (ms/operation)")
        axis.set_title(
            f"{label}: held-scenario RMSE "
            f"{regression.leave_scenario_out_rmse_ms:.3f} ms",
            loc="left",
        )
    figure.tight_layout()
    pdf_path = output_dir / "component_factorial_fit.pdf"
    png_path = output_dir / "component_factorial_fit.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def write_report(
    output_dir: Path,
    config: FactorialConfig,
    scenarios: Sequence[ComponentScenario],
    aggregate_rows: Sequence[dict[str, Any]],
    regressions: Sequence[RegressionResult],
) -> Path:
    baseline = [
        row
        for row in aggregate_rows
        if int(row["warehouse_width"]) == 120
        and int(row["warehouse_height"]) == 100
        and math.isclose(float(row["active_fraction"]), 0.8)
        and int(row["task_backlog"]) == 200
        and int(row["reservation_horizon"]) == 3
    ]
    lines = [
        "# Centralized Scheduler Component Characterization",
        "",
        "## Scope",
        "",
        "This host-specific experiment characterizes isolated assignment and",
        "full-reservation A* operations. It does not assume an operation count per",
        "completed task and does not include queue or network delay.",
        "",
        "The regression structure follows the implementation: assignment work uses",
        "the active-robot set and bounded task scan; path work uses graph size,",
        "reservation slots, and measured A* search work. Fleet-only polynomial",
        "selection is not used.",
        "",
        "## Design",
        "",
        f"- Structural scenarios: {len(scenarios)}",
        "- Design: thesis baselines, controlled one-factor perturbations, and a",
        "  three-level strength-two orthogonal array for combined-factor coverage",
        f"- Independent workload seeds: {len(config.seeds)}",
        f"- Timed operations per scenario and seed: {config.samples_per_scenario}",
        f"- Fleet sizes: {', '.join(map(str, THESIS_FLEETS))}",
        f"- Map sizes: {', '.join(f'{w}x{h}' for w, h in MAPS)}",
        f"- Task backlogs: {', '.join(map(str, BACKLOGS))}",
        f"- Reservation horizons: {', '.join(map(str, HORIZONS))}",
        f"- Active fractions: {', '.join(map(str, ACTIVE_FRACTIONS))}",
        "- Profiled path must match the timed Cython path for every accepted sample",
        "",
        "## Fixed-Geometry Baseline",
        "",
        "| N | Assignment CPU (ms/op) | Path CPU (ms/op) | Path elapsed (ms/op) |",
        "|---:|---:|---:|---:|",
    ]
    for fleet_size in THESIS_FLEETS:
        rows = [row for row in baseline if int(row["fleet_size"]) == fleet_size]
        if not rows:
            continue
        lines.append(
            f"| {fleet_size} | "
            f"{np.mean([float(row['mean_assignment_cpu_ms']) for row in rows]):.6f} | "
            f"{np.mean([float(row['mean_path_cpu_ms']) for row in rows]):.6f} | "
            f"{np.mean([float(row['mean_path_wall_ms']) for row in rows]):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Mechanistic Regressions",
            "",
            "Coefficients use the scaled predictor units shown below. Confidence",
            "intervals use a scenario-cluster bootstrap. Prediction error leaves one",
            "complete structural scenario out at a time.",
            "",
        ]
    )
    for result in regressions:
        lines.extend(
            [
                f"### {result.component.title()} - {result.response}",
                "",
                f"In-sample R-squared: {result.in_sample_r_squared:.4f}; "
                f"held-scenario RMSE: {result.leave_scenario_out_rmse_ms:.6f} ms.",
                "",
                "| Predictor | Coefficient | 95% cluster-bootstrap CI |",
                "|---|---:|---:|",
            ]
        )
        for name, coefficient, low, high in zip(
            result.predictor_names,
            result.coefficients,
            result.ci_low,
            result.ci_high,
        ):
            lines.append(
                f"| {name} | {coefficient:.9f} | [{low:.9f}, {high:.9f}] |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "These coefficients are machine- and implementation-specific calibration",
            "constants. Their predictor structure is portable; their numerical values",
            "are not. The path model is an empirical approximation over the tested map",
            "and route regimes, not the worst-case complexity of A*. Per-task scheduler",
            "demand requires operation visit ratios from the warehouse trace, and queue",
            "response requires independent open-loop validation.",
            "",
        ]
    )
    report_path = output_dir / "component_factorial_study.md"
    report_path.write_text("\n".join(lines), encoding="ascii")
    return report_path


def _git_revision(repo: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warehouse-repo", default="/home/modfi/ivalab/extern/warehouse"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=(8101, 8102, 8103, 8104, 8105, 8106)
    )
    parser.add_argument("--warmup-operations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--cpu-affinity", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/scheduler_capacity_study/component_factorial"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FactorialConfig(
        warehouse_repo=args.warehouse_repo,
        seeds=tuple(args.seeds),
        warmup_operations=args.warmup_operations,
        samples_per_scenario=args.samples,
        bootstrap_samples=args.bootstrap_samples,
        cpu_affinity=args.cpu_affinity,
    )
    set_cpu_affinity(config.cpu_affinity)
    environment_before = capture_environment()
    scenarios = default_scenarios()
    cases = [(scenario, seed) for seed in config.seeds for scenario in scenarios]
    random.Random(424242).shuffle(cases)
    aggregate_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (scenario, seed) in enumerate(cases, start=1):
        print(
            f"Component case {index}/{len(cases)}: {scenario.scenario_id}, seed={seed}",
            flush=True,
        )
        aggregate, samples = measure_scenario(config, scenario, seed)
        aggregate_rows.append(aggregate)
        raw_rows.extend(samples)
        _write_csv(args.output_dir / "component_aggregates.csv", aggregate_rows)
        _write_csv(args.output_dir / "component_samples.csv", raw_rows)

    regressions = [
        fit_regression(
            aggregate_rows,
            component,
            response,
            bootstrap_samples=config.bootstrap_samples,
            seed=9300 + offset,
        )
        for offset, (component, response) in enumerate(
            (
                ("assignment", "mean_assignment_cpu_ms"),
                ("assignment", "mean_assignment_wall_ms"),
                ("path", "mean_path_cpu_ms"),
                ("path", "mean_path_wall_ms"),
            )
        )
    ]
    (args.output_dir / "regressions.json").write_text(
        json.dumps([asdict(result) for result in regressions], indent=2) + "\n",
        encoding="ascii",
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "host_before": environment_before,
        "host_after": capture_environment(),
        "git": {
            "analytical": _git_revision(str(Path(__file__).resolve().parent)),
            "warehouse": _git_revision(config.warehouse_repo),
        },
    }
    (args.output_dir / "component_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    plot_results(args.output_dir, aggregate_rows, regressions)
    report = write_report(
        args.output_dir, config, scenarios, aggregate_rows, regressions
    )
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()

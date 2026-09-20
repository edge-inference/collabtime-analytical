#!/usr/bin/env python3
"""Pilot centralized scheduler component timing against warehouse code.

The study isolates operation execution from network delay and queue waiting and
records both elapsed time and thread CPU time.
It uses an event-triggered workload: each completed simulator task contributes
one assignment and one or more centralized path-planning visits.  The warehouse
geometry, assignment scan, and Cython A* implementation come from the warehouse
repository, but idle-agent polling and historical-task scans are deliberately
excluded because they are implementation artifacts rather than per-task work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import platform
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FLEET_SIZES = (50, 100, 200, 300, 500, 600, 700, 800)
DEFAULT_WORKER_COUNTS = (1, 5, 10)
MODEL_NAMES = ("constant", "linear", "n_log_n", "quadratic")
MODEL_COMPLEXITY = {
    "constant": 0,
    "linear": 1,
    "n_log_n": 2,
    "quadratic": 3,
}


@dataclass(frozen=True)
class StudyConfig:
    warehouse_repo: str
    fleet_sizes: tuple[int, ...] = DEFAULT_FLEET_SIZES
    warehouse_width: int = 120
    warehouse_height: int = 100
    active_fraction: float = 0.8
    task_backlog: int = 200
    assignment_scan_limit: int = 200
    reservation_horizon: int = 3
    path_visits_per_task: float = 1.0
    replicates: int = 12
    warmup_transactions: int = 25
    samples_per_replicate: int = 150
    layout_seed: int = 42
    sample_seed: int = 7301
    bootstrap_samples: int = 2000
    validation_fleet_size: int = 800
    validation_transactions: int = 1000
    validation_workers: tuple[int, ...] = DEFAULT_WORKER_COUNTS
    utilization_limit: float = 0.8


@dataclass
class FitResult:
    series: str
    model: str
    n_ref: float
    coefficients: list[float]
    rss: float
    rmse_ms: float
    r_squared: float
    aicc: float
    leave_fleet_out_rmse_ms: float
    leave_fleet_out_mse_ms2: float
    leave_fleet_out_mse_se_ms2: float


class EventDrivenSchedulerWorkload:
    """Read-only scheduler transaction used for timing and concurrency tests."""

    def __init__(
        self,
        coords: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        task_locations: np.ndarray,
        active_locations: np.ndarray,
        path_values: np.ndarray,
        path_timestamps: np.ndarray,
        request_starts: np.ndarray,
        request_goals: np.ndarray,
        astar_fast: Callable,
        assignment_scan_limit: int,
    ) -> None:
        self.coords = coords
        self.indptr = indptr
        self.indices = indices
        self.task_locations = task_locations
        self.active_locations = active_locations
        self.path_values = path_values
        self.path_timestamps = path_timestamps
        self.dummy_path_values = np.full((1, 1), -1, dtype=np.int32)
        self.dummy_path_timestamps = np.zeros(1, dtype=np.int32)
        self.request_starts = request_starts
        self.request_goals = request_goals
        self.astar_fast = astar_fast
        self.assignment_scan_limit = assignment_scan_limit

        num_nodes = coords.shape[0]
        self.jam_values = np.zeros(num_nodes, dtype=np.float32)
        self.jam_timestamps = np.zeros(num_nodes, dtype=np.int32)
        self.flow_values = np.zeros(num_nodes, dtype=np.float32)
        self.flow_timestamps = np.zeros(num_nodes, dtype=np.int32)
        self.cost_params = {
            "alpha": 2.0,
            "beta": 0.5,
            "max_aoi_ms": 999_999_999,
            "proximity_radius": 25.0,
            "conflict_penalty": 100.0,
        }

    def assign(self, agent_position: int) -> int:
        """Run the central assignment scan once without polling or claim I/O."""
        assigned_locations = {int(location) for location in self.active_locations}
        agent_x = self.coords[agent_position, 0]
        agent_y = self.coords[agent_position, 1]
        best_location = -1
        best_distance = float("inf")
        scanned = 0

        for raw_location in self.task_locations:
            location = int(raw_location)
            if location in assigned_locations:
                continue
            scanned += 1
            if scanned > self.assignment_scan_limit and best_location >= 0:
                break
            distance = abs(agent_x - self.coords[location, 0]) + abs(
                agent_y - self.coords[location, 1]
            )
            if distance < best_distance:
                best_distance = float(distance)
                best_location = location

        return best_location

    def plan(
        self, start: int, goal: int, *, global_reservations: bool = True
    ) -> list[int]:
        """Run centralized A* with corrected global or current dummy paths."""
        path_values = (
            self.path_values if global_reservations else self.dummy_path_values
        )
        path_timestamps = (
            self.path_timestamps if global_reservations else self.dummy_path_timestamps
        )
        return self.astar_fast(
            indptr=self.indptr,
            indices=self.indices,
            coords=self.coords,
            jam_values=self.jam_values,
            jam_timestamps=self.jam_timestamps,
            flow_values=self.flow_values,
            flow_timestamps=self.flow_timestamps,
            path_vals=path_values,
            path_timestamps=path_timestamps,
            start=int(start),
            goal=int(goal),
            cost_params=self.cost_params,
            current_time_ms=0,
        )

    def transaction(self, request_index: int) -> int:
        index = request_index % len(self.request_starts)
        start = int(self.request_starts[index])
        goal = int(self.request_goals[index])
        assignment = self.assign(start)
        path = self.plan(start, goal, global_reservations=True)
        return int(assignment >= 0 and bool(path))


def _prepend_warehouse_repo(warehouse_repo: str) -> None:
    repo = str(Path(warehouse_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _load_warehouse_code(warehouse_repo: str):
    _prepend_warehouse_repo(warehouse_repo)
    try:
        from perf.astar_fast import astar_fast
        from world.graph import WarehouseGraph
    except ImportError as exc:
        raise RuntimeError(
            "Warehouse dependencies are unavailable. Run the study with the "
            "ivalab virtual environment and build the Cython extension with "
            "`python perf/setup_cython.py build_ext --inplace`."
        ) from exc
    return WarehouseGraph, astar_fast


def _random_walk_paths(
    graph,
    traversable_nodes: Sequence[int],
    fleet_size: int,
    horizon: int,
    active_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    paths = np.full((fleet_size, horizon), -1, dtype=np.int32)
    active_count = min(fleet_size, int(round(active_fraction * fleet_size)))
    starts = rng.choice(traversable_nodes, size=active_count, replace=True)

    for agent_index, raw_start in enumerate(starts):
        current = int(raw_start)
        paths[agent_index, 0] = current
        for step in range(1, horizon):
            neighbors = tuple(graph.neighbors(current))
            if not neighbors:
                break
            current = int(neighbors[int(rng.integers(0, len(neighbors)))])
            paths[agent_index, step] = current
    return paths


def _sample_requests(
    coords: np.ndarray,
    traversable_nodes: Sequence[int],
    work_nodes: Sequence[int],
    count: int,
    rng: np.random.Generator,
    minimum_manhattan_distance: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    starts: list[int] = []
    goals: list[int] = []
    max_attempts = count * 100
    attempts = 0

    while len(starts) < count and attempts < max_attempts:
        attempts += 1
        start = int(traversable_nodes[int(rng.integers(0, len(traversable_nodes)))])
        goal = int(work_nodes[int(rng.integers(0, len(work_nodes)))])
        distance = abs(coords[start, 0] - coords[goal, 0]) + abs(
            coords[start, 1] - coords[goal, 1]
        )
        if start != goal and distance >= minimum_manhattan_distance:
            starts.append(start)
            goals.append(goal)

    if len(starts) != count:
        raise RuntimeError("Could not sample enough valid warehouse routes")
    return np.asarray(starts, dtype=np.int32), np.asarray(goals, dtype=np.int32)


def build_workload(
    config: StudyConfig,
    fleet_size: int,
    seed: int,
    request_count: int,
) -> EventDrivenSchedulerWorkload:
    WarehouseGraph, astar_fast = _load_warehouse_code(config.warehouse_repo)

    random.seed(config.layout_seed)
    graph_wrapper = WarehouseGraph(config.warehouse_width, config.warehouse_height)
    graph = graph_wrapper.graph
    coords = graph_wrapper.get_node_coords_array()
    indptr, indices, _ = graph_wrapper.get_csr_graph()
    traversable_nodes = [node for node, degree in graph.degree if degree > 0]
    work_nodes = [
        node
        for node in traversable_nodes
        if graph_wrapper.node_types.get(node) in ("pick_location", "pack_station")
    ]
    if not work_nodes:
        raise RuntimeError("Warehouse graph contains no pick or pack work nodes")

    population_fleet_size = max(max(config.fleet_sizes), fleet_size)
    active_rng = np.random.default_rng(seed + 1)
    task_rng = np.random.default_rng(seed + 2)
    path_rng = np.random.default_rng(seed + 3)
    request_rng = np.random.default_rng(seed + 4)
    active_count = min(fleet_size, int(round(config.active_fraction * fleet_size)))
    population_active_count = min(
        population_fleet_size,
        int(round(config.active_fraction * population_fleet_size)),
    )
    population_active_locations = active_rng.choice(
        traversable_nodes, size=population_active_count, replace=False
    ).astype(np.int32)
    active_locations = population_active_locations[:active_count]
    task_locations = task_rng.choice(
        work_nodes,
        size=config.task_backlog,
        replace=config.task_backlog > len(work_nodes),
    ).astype(np.int32)
    population_path_values = _random_walk_paths(
        graph,
        traversable_nodes,
        population_fleet_size,
        config.reservation_horizon,
        1.0,
        path_rng,
    )
    path_values = population_path_values[:fleet_size].copy()
    path_values[active_count:] = -1
    path_timestamps = np.zeros(fleet_size, dtype=np.int32)
    request_starts, request_goals = _sample_requests(
        coords, traversable_nodes, work_nodes, request_count, request_rng
    )

    return EventDrivenSchedulerWorkload(
        coords=coords,
        indptr=indptr,
        indices=indices,
        task_locations=task_locations,
        active_locations=active_locations,
        path_values=path_values,
        path_timestamps=path_timestamps,
        request_starts=request_starts,
        request_goals=request_goals,
        astar_fast=astar_fast,
        assignment_scan_limit=config.assignment_scan_limit,
    )


def measure_workload(
    workload: EventDrivenSchedulerWorkload,
    warmup_transactions: int,
    sample_transactions: int,
) -> dict[str, float]:
    for index in range(warmup_transactions):
        workload.transaction(index)
        array_index = index % len(workload.request_starts)
        workload.plan(
            int(workload.request_starts[array_index]),
            int(workload.request_goals[array_index]),
            global_reservations=False,
        )

    assignment_times_ms = np.empty(sample_transactions, dtype=np.float64)
    assignment_cpu_times_ms = np.empty(sample_transactions, dtype=np.float64)
    path_times_ms = np.empty(sample_transactions, dtype=np.float64)
    path_cpu_times_ms = np.empty(sample_transactions, dtype=np.float64)
    current_path_times_ms = np.empty(sample_transactions, dtype=np.float64)
    current_path_cpu_times_ms = np.empty(sample_transactions, dtype=np.float64)
    successful_paths = 0
    successful_current_paths = 0

    for sample_index in range(sample_transactions):
        request_index = warmup_transactions + sample_index
        array_index = request_index % len(workload.request_starts)
        start = int(workload.request_starts[array_index])
        goal = int(workload.request_goals[array_index])

        begin = time.perf_counter_ns()
        begin_cpu = time.thread_time_ns()
        workload.assign(start)
        after_assignment = time.perf_counter_ns()
        after_assignment_cpu = time.thread_time_ns()
        if sample_index % 2 == 0:
            path = workload.plan(start, goal, global_reservations=True)
            after_global_path = time.perf_counter_ns()
            after_global_path_cpu = time.thread_time_ns()
            current_path = workload.plan(start, goal, global_reservations=False)
            end = time.perf_counter_ns()
            end_cpu = time.thread_time_ns()
            global_path_ns = after_global_path - after_assignment
            global_path_cpu_ns = after_global_path_cpu - after_assignment_cpu
            current_path_ns = end - after_global_path
            current_path_cpu_ns = end_cpu - after_global_path_cpu
        else:
            current_path = workload.plan(start, goal, global_reservations=False)
            after_current_path = time.perf_counter_ns()
            after_current_path_cpu = time.thread_time_ns()
            path = workload.plan(start, goal, global_reservations=True)
            end = time.perf_counter_ns()
            end_cpu = time.thread_time_ns()
            current_path_ns = after_current_path - after_assignment
            current_path_cpu_ns = after_current_path_cpu - after_assignment_cpu
            global_path_ns = end - after_current_path
            global_path_cpu_ns = end_cpu - after_current_path_cpu

        assignment_times_ms[sample_index] = (after_assignment - begin) / 1_000_000.0
        assignment_cpu_times_ms[sample_index] = (
            after_assignment_cpu - begin_cpu
        ) / 1_000_000.0
        path_times_ms[sample_index] = global_path_ns / 1_000_000.0
        path_cpu_times_ms[sample_index] = global_path_cpu_ns / 1_000_000.0
        current_path_times_ms[sample_index] = current_path_ns / 1_000_000.0
        current_path_cpu_times_ms[sample_index] = (
            current_path_cpu_ns / 1_000_000.0
        )
        successful_paths += int(bool(path))
        successful_current_paths += int(bool(current_path))

    return {
        "assignment_mean_ms": float(np.mean(assignment_times_ms)),
        "assignment_cpu_mean_ms": float(np.mean(assignment_cpu_times_ms)),
        "assignment_median_ms": float(np.median(assignment_times_ms)),
        "assignment_p95_ms": float(np.percentile(assignment_times_ms, 95)),
        "path_mean_ms": float(np.mean(path_times_ms)),
        "path_cpu_mean_ms": float(np.mean(path_cpu_times_ms)),
        "path_median_ms": float(np.median(path_times_ms)),
        "path_p95_ms": float(np.percentile(path_times_ms, 95)),
        "path_success_rate": successful_paths / sample_transactions,
        "current_path_mean_ms": float(np.mean(current_path_times_ms)),
        "current_path_cpu_mean_ms": float(np.mean(current_path_cpu_times_ms)),
        "current_path_median_ms": float(np.median(current_path_times_ms)),
        "current_path_p95_ms": float(np.percentile(current_path_times_ms, 95)),
        "current_path_success_rate": successful_current_paths / sample_transactions,
    }


def run_calibration(config: StudyConfig) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    request_count = config.warmup_transactions + config.samples_per_replicate

    for replicate in range(config.replicates):
        fleet_order = list(config.fleet_sizes)
        random.Random(config.sample_seed + replicate).shuffle(fleet_order)
        print(f"Calibration replicate {replicate + 1}/{config.replicates}", flush=True)

        replicate_rows: dict[int, dict[str, float | int]] = {}
        seed = config.sample_seed + 10_000 * replicate
        for fleet_size in fleet_order:
            workload = build_workload(config, fleet_size, seed, request_count)
            measured = measure_workload(
                workload,
                config.warmup_transactions,
                config.samples_per_replicate,
            )
            demand_ms = (
                measured["assignment_mean_ms"]
                + config.path_visits_per_task * measured["path_mean_ms"]
            )
            cpu_demand_ms = (
                measured["assignment_cpu_mean_ms"]
                + config.path_visits_per_task * measured["path_cpu_mean_ms"]
            )
            current_wrapper_demand_ms = (
                measured["assignment_mean_ms"]
                + config.path_visits_per_task * measured["current_path_mean_ms"]
            )
            current_wrapper_cpu_demand_ms = (
                measured["assignment_cpu_mean_ms"]
                + config.path_visits_per_task
                * measured["current_path_cpu_mean_ms"]
            )
            replicate_rows[fleet_size] = {
                "replicate": replicate,
                "seed": seed,
                "fleet_size": fleet_size,
                "active_robots": int(round(config.active_fraction * fleet_size)),
                "task_backlog": config.task_backlog,
                "reservation_horizon": config.reservation_horizon,
                "path_visits_per_task": config.path_visits_per_task,
                **measured,
                "demand_ms_per_task": demand_ms,
                "cpu_demand_ms_per_task": cpu_demand_ms,
                "current_wrapper_demand_ms_per_task": current_wrapper_demand_ms,
                "current_wrapper_cpu_demand_ms_per_task": (
                    current_wrapper_cpu_demand_ms
                ),
            }
        rows.extend(replicate_rows[fleet_size] for fleet_size in config.fleet_sizes)
    return rows


def _design_matrix(model: str, fleet_sizes: np.ndarray, n_ref: float) -> np.ndarray:
    n = np.asarray(fleet_sizes, dtype=np.float64)
    centered = n - n_ref
    if model == "constant":
        return np.ones((len(n), 1), dtype=np.float64)
    if model == "linear":
        return np.column_stack((np.ones(len(n)), centered))
    if model == "n_log_n":
        transformed = n * np.log2(np.maximum(n, 1.0))
        ref_value = n_ref * math.log2(max(n_ref, 1.0))
        return np.column_stack((np.ones(len(n)), transformed - ref_value))
    if model == "quadratic":
        return np.column_stack((np.ones(len(n)), centered, centered**2))
    raise ValueError(f"Unknown model: {model}")


def fit_model(
    fleet_sizes: Sequence[float],
    values_ms: Sequence[float],
    model: str,
    series: str = "demand",
    n_ref: float | None = None,
) -> FitResult:
    n = np.asarray(fleet_sizes, dtype=np.float64)
    y = np.asarray(values_ms, dtype=np.float64)
    reference = float(np.median(np.unique(n))) if n_ref is None else float(n_ref)
    design = _design_matrix(model, n, reference)
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    predictions = design @ coefficients
    residuals = y - predictions
    rss = float(np.sum(residuals**2))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - rss / total_sum_squares if total_sum_squares > 0 else 1.0

    sample_count = len(y)
    parameter_count = design.shape[1]
    safe_rss = max(rss, np.finfo(float).tiny)
    aic = sample_count * math.log(safe_rss / sample_count) + 2 * parameter_count
    if sample_count > parameter_count + 1:
        aicc = aic + (
            2
            * parameter_count
            * (parameter_count + 1)
            / (sample_count - parameter_count - 1)
        )
    else:
        aicc = float("inf")

    holdout_errors: list[float] = []
    holdout_fleet_mse: list[float] = []
    for fleet_size in np.unique(n):
        training = n != fleet_size
        testing = ~training
        training_design = _design_matrix(model, n[training], reference)
        if np.linalg.matrix_rank(training_design) < training_design.shape[1]:
            continue
        held_coefficients, _, _, _ = np.linalg.lstsq(
            training_design, y[training], rcond=None
        )
        held_predictions = (
            _design_matrix(model, n[testing], reference) @ held_coefficients
        )
        fleet_errors = y[testing] - held_predictions
        holdout_errors.extend(fleet_errors.tolist())
        holdout_fleet_mse.append(float(np.mean(fleet_errors**2)))
    leave_fleet_out_rmse = (
        float(np.sqrt(np.mean(np.square(holdout_errors))))
        if holdout_errors
        else float("inf")
    )
    leave_fleet_out_mse = (
        float(np.mean(holdout_fleet_mse)) if holdout_fleet_mse else float("inf")
    )
    leave_fleet_out_mse_se = (
        float(np.std(holdout_fleet_mse, ddof=1) / math.sqrt(len(holdout_fleet_mse)))
        if len(holdout_fleet_mse) > 1
        else float("inf")
    )

    return FitResult(
        series=series,
        model=model,
        n_ref=reference,
        coefficients=[float(value) for value in coefficients],
        rss=rss,
        rmse_ms=rmse,
        r_squared=r_squared,
        aicc=aicc,
        leave_fleet_out_rmse_ms=leave_fleet_out_rmse,
        leave_fleet_out_mse_ms2=leave_fleet_out_mse,
        leave_fleet_out_mse_se_ms2=leave_fleet_out_mse_se,
    )


def fit_all_models(
    rows: Sequence[dict[str, float | int]],
) -> list[FitResult]:
    series_columns = {
        "assignment": "assignment_mean_ms",
        "corrected_global_path": "path_mean_ms",
        "current_wrapper_path": "current_path_mean_ms",
        "corrected_global_demand": "demand_ms_per_task",
        "current_wrapper_demand": "current_wrapper_demand_ms_per_task",
    }
    fleet_sizes = [float(row["fleet_size"]) for row in rows]
    fits: list[FitResult] = []
    for series, column in series_columns.items():
        values = [float(row[column]) for row in rows]
        for model in MODEL_NAMES:
            fits.append(fit_model(fleet_sizes, values, model, series=series))
    return fits


def select_fit(
    fits: Sequence[FitResult], series: str = "corrected_global_demand"
) -> FitResult:
    candidates = [fit for fit in fits if fit.series == series]
    if not candidates:
        raise ValueError(f"No fits found for series {series}")
    best_predictive_fit = min(candidates, key=lambda fit: fit.leave_fleet_out_rmse_ms)
    maximum_competitive_mse = (
        best_predictive_fit.leave_fleet_out_mse_ms2
        + best_predictive_fit.leave_fleet_out_mse_se_ms2
    )
    competitive = [
        fit
        for fit in candidates
        if fit.leave_fleet_out_mse_ms2 <= maximum_competitive_mse + 1e-15
    ]
    return min(
        competitive,
        key=lambda fit: (
            MODEL_COMPLEXITY[fit.model],
            fit.aicc,
            fit.leave_fleet_out_rmse_ms,
        ),
    )


def predict_fit(fit: FitResult, fleet_sizes: Sequence[float]) -> np.ndarray:
    design = _design_matrix(
        fit.model, np.asarray(fleet_sizes, dtype=np.float64), fit.n_ref
    )
    return design @ np.asarray(fit.coefficients, dtype=np.float64)


def bootstrap_linear_parameters(
    rows: Sequence[dict[str, float | int]],
    samples: int,
    seed: int,
) -> dict[str, float]:
    replicate_ids = sorted({int(row["replicate"]) for row in rows})
    grouped = {
        replicate: [row for row in rows if int(row["replicate"]) == replicate]
        for replicate in replicate_ids
    }
    n_ref = float(np.median(sorted({float(row["fleet_size"]) for row in rows})))
    rng = np.random.default_rng(seed)
    estimates = np.empty((samples, 3), dtype=np.float64)

    for index in range(samples):
        sampled_ids = rng.choice(replicate_ids, size=len(replicate_ids), replace=True)
        sampled_rows = [
            row for replicate in sampled_ids for row in grouped[int(replicate)]
        ]
        fit = fit_model(
            [float(row["fleet_size"]) for row in sampled_rows],
            [float(row["demand_ms_per_task"]) for row in sampled_rows],
            "linear",
            n_ref=n_ref,
        )
        demand_at_ref, alpha = fit.coefficients
        d0 = demand_at_ref - alpha * n_ref
        estimates[index] = (d0, alpha, demand_at_ref)

    point_fit = fit_model(
        [float(row["fleet_size"]) for row in rows],
        [float(row["demand_ms_per_task"]) for row in rows],
        "linear",
        n_ref=n_ref,
    )
    demand_at_ref, alpha = point_fit.coefficients
    d0 = demand_at_ref - alpha * n_ref
    lower = np.percentile(estimates, 2.5, axis=0)
    upper = np.percentile(estimates, 97.5, axis=0)
    return {
        "n_ref": n_ref,
        "d0_ms_per_task": d0,
        "d0_ci_low": float(lower[0]),
        "d0_ci_high": float(upper[0]),
        "alpha_ms_per_robot_task": alpha,
        "alpha_ci_low": float(lower[1]),
        "alpha_ci_high": float(upper[1]),
        "demand_at_n_ref_ms_per_task": demand_at_ref,
        "demand_at_n_ref_ci_low": float(lower[2]),
        "demand_at_n_ref_ci_high": float(upper[2]),
    }


_PROCESS_WORKLOAD: EventDrivenSchedulerWorkload | None = None


def _initialize_process_workload(
    config_dict: dict,
    fleet_size: int,
    seed: int,
    request_count: int,
) -> None:
    global _PROCESS_WORKLOAD
    config_dict["fleet_sizes"] = tuple(config_dict["fleet_sizes"])
    config_dict["validation_workers"] = tuple(config_dict["validation_workers"])
    config = StudyConfig(**config_dict)
    _PROCESS_WORKLOAD = build_workload(config, fleet_size, seed, request_count)


def _process_transaction(request_index: int) -> int:
    if _PROCESS_WORKLOAD is None:
        raise RuntimeError("Process workload was not initialized")
    return _PROCESS_WORKLOAD.transaction(request_index)


def _measure_executor_throughput(
    executor,
    transaction: Callable[[int], int],
    workers: int,
    transactions: int,
) -> tuple[float, float]:
    warmup_count = max(20, workers * 4)
    list(executor.map(transaction, range(warmup_count)))
    begin = time.perf_counter()
    successes = sum(
        executor.map(transaction, range(warmup_count, warmup_count + transactions))
    )
    elapsed = time.perf_counter() - begin
    return transactions / elapsed, successes / transactions


def run_capacity_validation(
    config: StudyConfig,
    fitted_demand_ms: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    request_count = (
        config.validation_transactions + max(config.validation_workers) * 4 + 20
    )
    validation_seed = config.sample_seed + 9_000_000
    thread_workload = build_workload(
        config, config.validation_fleet_size, validation_seed, request_count
    )

    for workers in config.validation_workers:
        predicted_tps = workers * 1000.0 / fitted_demand_ms
        with ThreadPoolExecutor(max_workers=workers) as executor:
            observed_tps, success_rate = _measure_executor_throughput(
                executor,
                thread_workload.transaction,
                workers,
                config.validation_transactions,
            )
        rows.append(
            {
                "execution_model": "threads",
                "workers": workers,
                "fleet_size": config.validation_fleet_size,
                "predicted_tps": predicted_tps,
                "observed_tps": observed_tps,
                "scaling_efficiency": observed_tps / predicted_tps,
                "success_rate": success_rate,
            }
        )

        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_process_workload,
            initargs=(
                asdict(config),
                config.validation_fleet_size,
                validation_seed,
                request_count,
            ),
        ) as executor:
            observed_tps, success_rate = _measure_executor_throughput(
                executor,
                _process_transaction,
                workers,
                config.validation_transactions,
            )
        rows.append(
            {
                "execution_model": "processes",
                "workers": workers,
                "fleet_size": config.validation_fleet_size,
                "predicted_tps": predicted_tps,
                "observed_tps": observed_tps,
                "scaling_efficiency": observed_tps / predicted_tps,
                "success_rate": success_rate,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="ascii") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fit_rows(fits: Sequence[FitResult]) -> list[dict]:
    return [
        {
            "series": fit.series,
            "model": fit.model,
            "n_ref": fit.n_ref,
            "coefficients": json.dumps(fit.coefficients),
            "rss": fit.rss,
            "rmse_ms": fit.rmse_ms,
            "r_squared": fit.r_squared,
            "aicc": fit.aicc,
            "leave_fleet_out_rmse_ms": fit.leave_fleet_out_rmse_ms,
            "leave_fleet_out_mse_ms2": fit.leave_fleet_out_mse_ms2,
            "leave_fleet_out_mse_se_ms2": fit.leave_fleet_out_mse_se_ms2,
        }
        for fit in fits
    ]


def _mean_and_ci(
    rows: Sequence[dict[str, float | int]], column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fleet_sizes = np.asarray(sorted({int(row["fleet_size"]) for row in rows}))
    means = np.empty(len(fleet_sizes))
    half_widths = np.empty(len(fleet_sizes))
    for index, fleet_size in enumerate(fleet_sizes):
        values = np.asarray(
            [float(row[column]) for row in rows if int(row["fleet_size"]) == fleet_size]
        )
        means[index] = np.mean(values)
        half_widths[index] = 1.96 * np.std(values, ddof=1) / math.sqrt(len(values))
    return fleet_sizes, means, half_widths


def plot_results(
    output_dir: Path,
    rows: Sequence[dict[str, float | int]],
    selected_fit: FitResult,
    capacity_rows: Sequence[dict[str, float | int | str]],
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

    fleet_sizes, assignment, assignment_ci = _mean_and_ci(rows, "assignment_mean_ms")
    _, path, path_ci = _mean_and_ci(rows, "path_mean_ms")
    _, demand, demand_ci = _mean_and_ci(rows, "demand_ms_per_task")
    _, current_demand, current_demand_ci = _mean_and_ci(
        rows, "current_wrapper_demand_ms_per_task"
    )
    axes[0].errorbar(
        fleet_sizes, assignment, yerr=assignment_ci, marker="o", label="Assignment"
    )
    axes[0].errorbar(
        fleet_sizes,
        path,
        yerr=path_ci,
        marker="s",
        label="Global-state path planning",
    )
    axes[0].errorbar(
        fleet_sizes,
        demand,
        yerr=demand_ci,
        marker="^",
        linewidth=1.8,
        label="Corrected global demand",
    )
    axes[0].errorbar(
        fleet_sizes,
        current_demand,
        yerr=current_demand_ci,
        marker="d",
        linewidth=1.2,
        label="Current-wrapper demand",
    )
    dense_n = np.linspace(min(fleet_sizes), max(fleet_sizes), 300)
    axes[0].plot(
        dense_n,
        predict_fit(selected_fit, dense_n),
        linestyle="--",
        color="black",
        label=f"Selected: {selected_fit.model}",
    )
    axes[0].set_xlabel("Fleet size N")
    axes[0].set_ylabel("Isolated elapsed time (ms/task)")
    axes[0].set_title("(a) Component timing pilot", loc="left")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")

    global_state_overhead = demand / current_demand
    axes[1].plot(fleet_sizes, global_state_overhead, marker="o")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Fleet size N")
    axes[1].set_ylabel("Corrected/current demand ratio")
    axes[1].set_title("(b) Global-state overhead", loc="left")

    worker_counts = sorted({int(row["workers"]) for row in capacity_rows})
    predicted = [
        next(
            float(row["predicted_tps"])
            for row in capacity_rows
            if int(row["workers"]) == workers
        )
        for workers in worker_counts
    ]
    axes[2].plot(worker_counts, predicted, marker="o", label="R / measured demand")
    for execution_model, marker in (("threads", "s"), ("processes", "^")):
        observed = [
            next(
                float(row["observed_tps"])
                for row in capacity_rows
                if int(row["workers"]) == workers
                and row["execution_model"] == execution_model
            )
            for workers in worker_counts
        ]
        axes[2].plot(
            worker_counts, observed, marker=marker, label=execution_model.title()
        )
    axes[2].set_xlabel("Scheduler workers R")
    axes[2].set_ylabel("Saturated throughput (tasks/s)")
    axes[2].set_title("(c) Independent capacity check", loc="left")
    axes[2].legend(frameon=False, fontsize=8)

    figure.tight_layout()
    pdf_path = output_dir / "scheduler_demand_calibration.pdf"
    png_path = output_dir / "scheduler_demand_calibration.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def _git_revision(repo: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_report(
    output_dir: Path,
    config: StudyConfig,
    rows: Sequence[dict[str, float | int]],
    fits: Sequence[FitResult],
    selected_fit: FitResult,
    linear_parameters: dict[str, float],
    capacity_rows: Sequence[dict[str, float | int | str]],
) -> Path:
    fleet_sizes, demand, demand_ci = _mean_and_ci(rows, "demand_ms_per_task")
    _, cpu_demand, _ = _mean_and_ci(rows, "cpu_demand_ms_per_task")
    _, current_demand, _ = _mean_and_ci(rows, "current_wrapper_demand_ms_per_task")
    demand_by_n = dict(zip(fleet_sizes.tolist(), demand.tolist()))
    cpu_demand_by_n = dict(zip(fleet_sizes.tolist(), cpu_demand.tolist()))
    ci_by_n = dict(zip(fleet_sizes.tolist(), demand_ci.tolist()))
    current_demand_by_n = dict(zip(fleet_sizes.tolist(), current_demand.tolist()))
    alpha_supported = not (
        linear_parameters["alpha_ci_low"] <= 0.0 <= linear_parameters["alpha_ci_high"]
    )

    demand_fits = sorted(
        (fit for fit in fits if fit.series == "corrected_global_demand"),
        key=lambda fit: fit.leave_fleet_out_rmse_ms,
    )
    component_linear_fits = {
        fit.series: fit
        for fit in fits
        if fit.model == "linear"
        and fit.series
        in {
            "assignment",
            "corrected_global_path",
            "current_wrapper_demand",
        }
    }
    assignment_fit = component_linear_fits["assignment"]
    path_fit = component_linear_fits["corrected_global_path"]
    current_fit = component_linear_fits["current_wrapper_demand"]
    assignment_d0 = (
        assignment_fit.coefficients[0]
        - assignment_fit.coefficients[1] * assignment_fit.n_ref
    )
    path_d0 = path_fit.coefficients[0] - path_fit.coefficients[1] * path_fit.n_ref
    current_d0 = (
        current_fit.coefficients[0] - current_fit.coefficients[1] * current_fit.n_ref
    )
    lines = [
        "# Centralized Scheduler Component-Timing Pilot",
        "",
        "## Scope",
        "",
        "This is a host-specific component microbenchmark, not a complete scheduler",
        "calibration, warehouse simulation, or network benchmark. It measures one",
        "event-triggered assignment plus one centralized path-planning visit per task.",
        "Queue waiting,",
        "idle-agent polling, network delay, and historical-task scans are excluded.",
        "",
        "Two path-planning cases are measured. The current central wrapper passes a 1x1",
        "dummy reservation array to A*. The corrected global-state case supplies the",
        "N x 3 reservation snapshot implied by the scheduler's three-node reservation",
        "horizon and the stated centralized architecture. The corrected case is a",
        "counterfactual component measurement; the current-wrapper case is a code audit.",
        "",
        "The operational decomposition is:",
        "",
        "`D_pilot(N) = S_assign(N) + V_path/task * S_path(N)`.",
        "Here S_assign and S_path are recorded as both isolated elapsed time and",
        "thread CPU time in ms/operation. The pilot fixes V_path/task at one rather",
        "than measuring the implemented visit ratio. D_pilot is therefore not the",
        "complete scheduler demand per completed task.",
        "",
        "The simulator exposes tasks directly, so all demand and throughput units in",
        "this study remain task-level.",
        "",
        "## Configuration",
        "",
        f"- Warehouse: {config.warehouse_width} x {config.warehouse_height}",
        f"- Fleet sizes: {', '.join(map(str, config.fleet_sizes))}",
        f"- Active fleet fraction: {config.active_fraction:.2f}",
        f"- Available-task backlog: {config.task_backlog}",
        f"- Reservation horizon: {config.reservation_horizon} nodes/robot",
        f"- Path-planning visits per task: {config.path_visits_per_task:g}",
        f"- Independent workload replicates: {config.replicates}",
        f"- Timed transactions per replicate and fleet size: {config.samples_per_replicate}",
        "",
        "## Measurements",
        "",
        "| N | Corrected elapsed (ms/task) | Corrected CPU (ms/task) | Current-wrapper elapsed (ms/task) | Corrected elapsed 95% CI half-width |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fleet_size in fleet_sizes:
        lines.append(
            f"| {int(fleet_size)} | {demand_by_n[int(fleet_size)]:.6f} "
            f"| {cpu_demand_by_n[int(fleet_size)]:.6f} "
            f"| {current_demand_by_n[int(fleet_size)]:.6f} "
            f"| {ci_by_n[int(fleet_size)]:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Scaling-Law Selection",
            "",
            "Models are ranked by leave-one-fleet-size-out prediction error. The",
            "one-standard-error rule defines the predictively competitive set, after",
            "which the structurally simpler form and AICc break the tie.",
            "",
            "| Model | Holdout RMSE (ms) | AICc | R-squared |",
            "|---|---:|---:|---:|",
        ]
    )
    for fit in demand_fits:
        lines.append(
            f"| {fit.model} | {fit.leave_fleet_out_rmse_ms:.6f} "
            f"| {fit.aicc:.3f} | {fit.r_squared:.5f} |"
        )

    lines.extend(
        [
            "",
            f"Selected model: **{selected_fit.model}**.",
            "",
            "The linear parameterization below is fitted to isolated elapsed time:",
            "",
            f"- `D0 = {linear_parameters['d0_ms_per_task']:.6f}` ms/task "
            f"(95% bootstrap CI {linear_parameters['d0_ci_low']:.6f} to "
            f"{linear_parameters['d0_ci_high']:.6f})",
            f"- `alpha = {linear_parameters['alpha_ms_per_robot_task']:.9f}` "
            "ms/(robot task) "
            f"(95% bootstrap CI {linear_parameters['alpha_ci_low']:.9f} to "
            f"{linear_parameters['alpha_ci_high']:.9f})",
            f"- Linear fleet-size effect excludes zero: {'yes' if alpha_supported else 'no'}.",
            "",
            "The component-level linear fits show where the fleet-size term originates:",
            "",
            "| Component | D0 (ms/task) | alpha (ms/robot/task) |",
            "|---|---:|---:|",
            f"| Assignment | {assignment_d0:.6f} | {assignment_fit.coefficients[1]:.9f} |",
            f"| Corrected global-state A* | {path_d0:.6f} | {path_fit.coefficients[1]:.9f} |",
            f"| Current-wrapper total | {current_d0:.6f} | {current_fit.coefficients[1]:.9f} |",
            "",
            "Neither this fitted pilot nor the existing analytical scenario is a complete",
            "implementation calibration. Visit ratios and background operation rates are",
            "measured separately by `warehouse_scheduler_trace.py`.",
            "",
            "## Capacity Validation",
            "",
            "The validation uses different request samples. Threads reproduce the current",
            "Python connection-pool interpretation; processes approximate independent",
            "CPU-capable scheduler replicas on this host. The predicted line is the",
            "reciprocal of isolated elapsed pilot time, not a queueing prediction.",
            "",
            "| Execution | Workers | Predicted tasks/s | Observed tasks/s | Efficiency |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in capacity_rows:
        lines.append(
            f"| {row['execution_model']} | {int(row['workers'])} "
            f"| {float(row['predicted_tps']):.2f} "
            f"| {float(row['observed_tps']):.2f} "
            f"| {float(row['scaling_efficiency']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Why The Times Are Milliseconds",
            "",
            "The benchmark uses CPU/Cython code, not a GPU. The graph has 12,000 nodes",
            "and remains fixed as N grows. All state is already in memory; there is no",
            "network, database, serialization, solver, or storage I/O. Assignment scans",
            "at most 200 task candidates, and the corrected reservation input adds only",
            "three integer entries per robot. These properties make millisecond-scale",
            "service times plausible. The independent one-worker throughput check closely",
            "matches the reciprocal of isolated elapsed time, providing a timer check for",
            "this narrow component transaction.",
            "",
            "## Interpretation Boundary",
            "",
            "The fitted values characterize one assumed component transaction on the",
            "recorded host. They are not scheduler demand per completed task until actual",
            "operation visit ratios and background rates are incorporated.",
            "They do not include network RTT, database/consensus service, serialization,",
            "or a global optimization solver. Those costs must remain separate terms or be",
            "measured on a specified deployment. Changing backlog, active fraction, route",
            "distribution, reservation horizon, hardware, or scheduler algorithm requires",
            "recalibration.",
            "",
            "## Method References",
            "",
            "- P. J. Denning and J. P. Buzen, operational laws and the service-demand",
            "  identity B/C: https://denninginstitute.com/pjd/PUBS/ENC/qn08.pdf",
            "- A. Georges, D. Buytaert, and L. Eeckhout, repeated measurement and",
            "  confidence intervals: https://biblio.ugent.be/publication/417084",
        ]
    )

    report_path = output_dir / "scheduler_demand_study.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warehouse-repo",
        type=Path,
        default=Path("/home/modfi/ivalab/extern/warehouse"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/scheduler_demand_study"),
    )
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--validation-transactions", type=int, default=1000)
    parser.add_argument(
        "--skip-capacity-validation",
        action="store_true",
        help="Only run calibration and model fitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StudyConfig(
        warehouse_repo=str(args.warehouse_repo.resolve()),
        replicates=args.replicates,
        samples_per_replicate=args.samples,
        bootstrap_samples=args.bootstrap_samples,
        validation_transactions=args.validation_transactions,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_calibration(config)
    fits = fit_all_models(rows)
    selected_fit = select_fit(fits)
    linear_parameters = bootstrap_linear_parameters(
        rows, config.bootstrap_samples, config.sample_seed + 1
    )
    fitted_demand_ms = float(
        predict_fit(selected_fit, [config.validation_fleet_size])[0]
    )
    capacity_rows = (
        []
        if args.skip_capacity_validation
        else run_capacity_validation(config, fitted_demand_ms)
    )

    _write_csv(args.output_dir / "calibration_replicates.csv", rows)
    _write_csv(args.output_dir / "model_fits.csv", _fit_rows(fits))
    _write_csv(args.output_dir / "capacity_validation.csv", capacity_rows)
    (args.output_dir / "linear_parameters.json").write_text(
        json.dumps(linear_parameters, indent=2) + "\n", encoding="ascii"
    )
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": asdict(config),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": sys.version,
        },
        "git": {
            "analytical": _git_revision(str(Path(__file__).resolve().parent)),
            "warehouse": _git_revision(config.warehouse_repo),
        },
        "selected_model": asdict(selected_fit),
    }
    (args.output_dir / "study_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )

    if capacity_rows:
        plot_results(args.output_dir, rows, selected_fit, capacity_rows)
    report_path = write_report(
        args.output_dir,
        config,
        rows,
        fits,
        selected_fit,
        linear_parameters,
        capacity_rows,
    )
    print(f"Selected demand model: {selected_fit.model}")
    print(
        "Linear fit: "
        f"D0={linear_parameters['d0_ms_per_task']:.6f} ms/task, "
        f"alpha={linear_parameters['alpha_ms_per_robot_task']:.9f} "
        "ms/(robot task)"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

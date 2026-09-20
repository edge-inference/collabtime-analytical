#!/usr/bin/env python3
"""Analyze warehouse scheduler traces against the analytical demand scenario."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import t


OPERATION_ORDER = (
    "assign_task_to_agent",
    "request_path",
    "report_flow",
    "report_jam",
    "update_agent_position",
    "notify_task_released",
)
OPERATION_LABELS = {
    "assign_task_to_agent": "Assignment",
    "request_path": "Path planning",
    "report_flow": "Flow reports",
    "report_jam": "Jam reports",
    "update_agent_position": "Position updates",
    "notify_task_released": "Task release",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _number(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def enrich_rows(
    aggregate_rows: Sequence[dict[str, str]],
    operation_rows: Sequence[dict[str, str]],
    *,
    scheduler_demand_base_ms: float = 1200.0,
    scheduler_demand_per_robot_ms: float = 2.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregates: list[dict[str, Any]] = []
    observations: dict[tuple[int, int], float] = {}
    for raw in aggregate_rows:
        row: dict[str, Any] = dict(raw)
        row["fleet_size"] = int(float(raw["fleet_size"]))
        row["seed"] = int(float(raw["seed"]))
        for key in (
            "task_rate_tps",
            "observation_s",
            "completion_throughput_tps",
            "terminal_throughput_tps",
            "mean_idle_robots",
            "mean_active_tasks",
            "active_task_drift_fraction_of_created",
            "completion_to_created_ratio",
            "scheduler_cpu_s",
            "trace_process_cpu_s",
            "scheduler_cpu_ms_per_completed_task",
            "scheduler_elapsed_ms_per_completed_task",
        ):
            row[key] = _number(raw, key)
        row["completed_tasks"] = int(_number(raw, "completed_tasks", 0.0))
        row["created_tasks"] = int(_number(raw, "created_tasks", 0.0))
        row["terminal_failed_tasks"] = int(
            _number(raw, "terminal_failed_tasks", 0.0)
        )
        row["active_tasks_start"] = int(_number(raw, "active_tasks_start", 0.0))
        row["active_tasks_end"] = int(_number(raw, "active_tasks_end", 0.0))
        row["active_task_delta"] = int(_number(raw, "active_task_delta", 0.0))
        row["task_flow_balance_error"] = int(
            _number(raw, "task_flow_balance_error", 0.0)
        )
        row["claimed_tasks"] = int(_number(raw, "claimed_tasks", 0.0))
        row["failed_task_attempts"] = int(
            _number(raw, "failed_task_attempts", 0.0)
        )
        row["failed_attempts_per_completed_task"] = (
            row["failed_task_attempts"] / row["completed_tasks"]
            if row["completed_tasks"] > 0
            else math.nan
        )
        observation_s = row["observation_s"]
        row["scheduler_cpu_core_load"] = _number(
            raw,
            "scheduler_cpu_core_load",
            row["scheduler_cpu_s"] / observation_s,
        )
        row["scheduler_fraction_of_process_cpu"] = _number(
            raw,
            "scheduler_fraction_of_process_cpu",
            row["scheduler_cpu_s"] / row["trace_process_cpu_s"],
        )
        row["analytical_demand_ms_per_task"] = (
            scheduler_demand_base_ms
            + scheduler_demand_per_robot_ms * row["fleet_size"]
        )
        traced_elapsed_demand = row["scheduler_elapsed_ms_per_completed_task"]
        row["analytical_to_trace_elapsed_ratio"] = (
            row["analytical_demand_ms_per_task"] / traced_elapsed_demand
            if traced_elapsed_demand > 0
            else math.nan
        )
        row["analytical_worker_occupancy_at_observed_throughput"] = (
            row["completion_throughput_tps"]
            * row["analytical_demand_ms_per_task"]
            / 1000.0
        )
        aggregates.append(row)
        observations[(row["fleet_size"], row["seed"])] = observation_s

    operations: list[dict[str, Any]] = []
    for raw in operation_rows:
        row = dict(raw)
        row["fleet_size"] = int(float(raw["fleet_size"]))
        row["seed"] = int(float(raw["seed"]))
        row["operation"] = raw["operation"]
        for key in (
            "calls",
            "successes",
            "misses",
            "mean_cpu_ms",
            "success_mean_cpu_ms",
            "miss_mean_cpu_ms",
            "total_cpu_s",
            "cpu_ms_per_completed_task",
            "calls_per_completed_task",
        ):
            row[key] = _number(raw, key)
        observation_s = observations[(row["fleet_size"], row["seed"])]
        row["calls_per_s"] = _number(
            raw, "calls_per_s", row["calls"] / observation_s
        )
        row["cpu_core_load"] = row["total_cpu_s"] / observation_s
        operations.append(row)
    return aggregates, operations


def _mean_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, math.nan
    critical_value = float(t.ppf(0.975, df=len(array) - 1))
    half_width = (
        critical_value * float(np.std(array, ddof=1)) / math.sqrt(len(array))
    )
    return mean, half_width


def summarize_by_fleet(
    aggregates: Sequence[dict[str, Any]], operations: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for fleet_size in sorted({int(row["fleet_size"]) for row in aggregates}):
        fleet_rows = [row for row in aggregates if row["fleet_size"] == fleet_size]
        summary: dict[str, Any] = {"fleet_size": fleet_size, "replicates": len(fleet_rows)}
        for key in (
            "completion_throughput_tps",
            "completion_to_created_ratio",
            "active_task_drift_fraction_of_created",
            "active_task_delta",
            "mean_active_tasks",
            "mean_idle_robots",
            "scheduler_cpu_core_load",
            "scheduler_cpu_ms_per_completed_task",
            "scheduler_elapsed_ms_per_completed_task",
            "analytical_to_trace_elapsed_ratio",
            "analytical_worker_occupancy_at_observed_throughput",
            "failed_attempts_per_completed_task",
        ):
            mean, half_width = _mean_ci([float(row[key]) for row in fleet_rows])
            summary[key] = mean
            summary[f"{key}_ci"] = half_width
        for operation in OPERATION_ORDER:
            operation_rows = [
                row
                for row in operations
                if row["fleet_size"] == fleet_size and row["operation"] == operation
            ]
            if operation_rows:
                for metric, output_name in (
                    ("cpu_ms_per_completed_task", "cpu_ms_per_task"),
                    ("calls_per_s", "calls_per_s"),
                    ("calls_per_completed_task", "calls_per_task"),
                ):
                    mean, half_width = _mean_ci(
                        [float(row[metric]) for row in operation_rows]
                    )
                    summary[f"{operation}_{output_name}"] = mean
                    summary[f"{operation}_{output_name}_ci"] = half_width
        summaries.append(summary)
    return summaries


def validate_operational_identities(
    aggregates: Sequence[dict[str, Any]], operations: Sequence[dict[str, Any]]
) -> list[dict[str, float | int]]:
    results: list[dict[str, float | int]] = []
    by_case: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in operations:
        by_case[(row["fleet_size"], row["seed"])].append(row)
    for aggregate in aggregates:
        key = (aggregate["fleet_size"], aggregate["seed"])
        predicted_load = sum(float(row["cpu_core_load"]) for row in by_case[key])
        observed_load = float(aggregate["scheduler_cpu_core_load"])
        relative_error = (
            abs(predicted_load - observed_load) / observed_load
            if observed_load > 0
            else 0.0
        )
        assignment = next(
            row for row in by_case[key] if row["operation"] == "assign_task_to_agent"
        )
        expected_poll_rate = float(aggregate["mean_idle_robots"]) / 0.1
        poll_error = (
            abs(float(assignment["calls_per_s"]) - expected_poll_rate)
            / expected_poll_rate
            if expected_poll_rate > 0
            else 0.0
        )
        results.append(
            {
                "fleet_size": key[0],
                "seed": key[1],
                "load_identity_relative_error": relative_error,
                "poll_rate_relative_error": poll_error,
                "task_flow_balance_error": int(
                    aggregate["task_flow_balance_error"]
                ),
                "active_task_drift_fraction_of_created": float(
                    aggregate["active_task_drift_fraction_of_created"]
                ),
            }
        )
    return results


def plot_summary(output_dir: Path, summaries: Sequence[dict[str, Any]]) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fleet_sizes = np.asarray([row["fleet_size"] for row in summaries])
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))
    axes = axes.ravel()

    axes[0].errorbar(
        fleet_sizes,
        [row["scheduler_cpu_core_load"] for row in summaries],
        yerr=[row["scheduler_cpu_core_load_ci"] for row in summaries],
        marker="o",
        capsize=2,
        label="Traced scheduler load",
    )
    axes[0].axhline(1.0, color="black", linestyle="--", label="One CPU core")
    axes[0].set_xlabel("Fleet size N")
    axes[0].set_ylabel("CPU core-equivalents")
    axes[0].set_title("(a) Measured scheduler CPU load", loc="left")
    axes[0].legend(frameon=False, fontsize=8)

    bottoms = np.zeros(len(summaries))
    bar_width = 0.65 * float(np.min(np.diff(fleet_sizes))) if len(fleet_sizes) > 1 else 1.0
    for operation in OPERATION_ORDER:
        values = np.asarray(
            [row.get(f"{operation}_cpu_ms_per_task", 0.0) for row in summaries]
        )
        if np.any(values > 0):
            axes[1].bar(
                fleet_sizes,
                values,
                width=bar_width,
                bottom=bottoms,
                label=OPERATION_LABELS[operation],
            )
            bottoms += values
    axes[1].set_xlabel("Fleet size N")
    axes[1].set_ylabel("CPU demand (ms/completed task)")
    axes[1].set_title("(b) Operation-demand decomposition", loc="left")
    axes[1].legend(frameon=False, fontsize=6)

    axes[2].errorbar(
        fleet_sizes,
        [row["active_task_drift_fraction_of_created"] for row in summaries],
        yerr=[
            row["active_task_drift_fraction_of_created_ci"] for row in summaries
        ],
        marker="o",
        capsize=2,
    )
    axes[2].axhline(0.0, color="black", linestyle="--")
    axes[2].set_xlabel("Fleet size N")
    axes[2].set_ylabel("Active-task change / tasks created")
    axes[2].set_title("(c) Observation-window task drift", loc="left")

    axes[3].errorbar(
        fleet_sizes,
        [row["analytical_to_trace_elapsed_ratio"] for row in summaries],
        marker="o",
        yerr=[row["analytical_to_trace_elapsed_ratio_ci"] for row in summaries],
        capsize=2,
    )
    axes[3].axhline(1.0, color="black", linestyle="--")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("Fleet size N")
    axes[3].set_ylabel("Configured / traced elapsed demand")
    axes[3].set_title("(d) Configured-scenario discrepancy", loc="left")
    figure.tight_layout()

    pdf_path = output_dir / "warehouse_trace_analysis.pdf"
    png_path = output_dir / "warehouse_trace_analysis.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def write_report(
    output_dir: Path,
    summaries: Sequence[dict[str, Any]],
    identity_checks: Sequence[dict[str, float | int]],
    *,
    scheduler_demand_base_ms: float,
    scheduler_demand_per_robot_ms: float,
) -> Path:
    max_load_error = max(
        float(row["load_identity_relative_error"]) for row in identity_checks
    )
    max_poll_error = max(float(row["poll_rate_relative_error"]) for row in identity_checks)
    max_flow_error = max(
        abs(int(row["task_flow_balance_error"])) for row in identity_checks
    )
    positive_drift_fleets = [
        int(row["fleet_size"])
        for row in summaries
        if float(row["active_task_drift_fraction_of_created"])
        - float(row["active_task_drift_fraction_of_created_ci"])
        > 0
    ]
    drift_text = (
        ", ".join(str(fleet_size) for fleet_size in positive_drift_fleets)
        if positive_drift_fleets
        else "none"
    )
    lines = [
        "# Scheduler Trace Model Validation",
        "",
        "## Purpose",
        "",
        "This report tests the configured analytical scheduler-demand scenario",
        "against operation rates, CPU demand, and in-service elapsed demand traced",
        "from the implemented",
        "polling-based centralized warehouse scheduler. It does not infer a",
        "multi-worker queue from the sequential Mesa execution.",
        "",
        "## Operational Checks",
        "",
        f"- Maximum relative error reconstructing aggregate scheduler CPU load from operation rates: {max_load_error:.3e}",
        f"- Maximum relative error between assignment-call rate and mean-idle-robots / 0.1 s: {max_poll_error:.3e}",
        f"- Maximum absolute task-flow conservation error: {max_flow_error} tasks",
        f"- Fleet sizes with a positive active-task drift whose 95% interval excludes zero: {drift_text}",
        "",
        "## Fleet Summary",
        "",
        "Values after the fleet size are replicate means with 95% Student-t",
        "confidence half-widths.",
        "",
        "| N | Reps | Throughput (tasks/s) | Active-task drift/created | Failed attempts/completion | Path calls/completion | Scheduler CPU core load | CPU ms/completion | Elapsed ms/completion | Configured/traced elapsed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['fleet_size']} | {row['replicates']} | "
            f"{row['completion_throughput_tps']:.3f} +/- {row['completion_throughput_tps_ci']:.3f} | "
            f"{row['active_task_drift_fraction_of_created']:.3f} +/- {row['active_task_drift_fraction_of_created_ci']:.3f} | "
            f"{row['failed_attempts_per_completed_task']:.2f} +/- {row['failed_attempts_per_completed_task_ci']:.2f} | "
            f"{row.get('request_path_calls_per_task', math.nan):.2f} +/- {row.get('request_path_calls_per_task_ci', math.nan):.2f} | "
            f"{row['scheduler_cpu_core_load']:.3f} +/- {row['scheduler_cpu_core_load_ci']:.3f} | "
            f"{row['scheduler_cpu_ms_per_completed_task']:.3f} +/- {row['scheduler_cpu_ms_per_completed_task_ci']:.3f} | "
            f"{row['scheduler_elapsed_ms_per_completed_task']:.3f} +/- {row['scheduler_elapsed_ms_per_completed_task_ci']:.3f} | "
            f"{row['analytical_to_trace_elapsed_ratio']:.2f} +/- {row['analytical_to_trace_elapsed_ratio_ci']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The observation windows at the listed positive-drift fleet sizes are",
            "not stationary: tasks enter the active set faster than they reach a",
            "terminal state. Their CPU and elapsed milliseconds per completion are",
            "descriptive window ratios, not steady-state lifecycle service-demand",
            "estimates. Aggregate scheduler CPU load remains a direct measurement of",
            "the operation workload executed during the simulated-time window.",
            "",
            "The configured demand is "
            f"`{scheduler_demand_base_ms:g} + {scheduler_demand_per_robot_ms:g}N` "
            "ms/task and is defined as worker",
            "occupancy. It is therefore compared with the sum of in-service elapsed",
            "times divided by completions where flow is approximately stationary.",
            "CPU demand is reported separately for",
            "resource utilization. If configured demand greatly exceeds traced elapsed",
            "demand while scheduler CPU load remains below one core-equivalent,",
            "a scheduler-limited throughput plateau is not supported by this",
            "implementation trace.",
            "",
            "The trace can still reveal high operation multiplicity from polling, task",
            "failures, retries, and replanning. Those effects should be modeled through",
            "operation rates. They do not justify replacing measured service demands with",
            "a larger constant chosen to reproduce end-to-end warehouse throughput.",
            "",
            "The configured scenario may remain a conditional stress case, but it must",
            "not be labeled implementation-calibrated. A batched global MAPF solver is a",
            "different algorithmic baseline and requires its own cycle-time measurements.",
            "",
        ]
    )
    path = output_dir / "warehouse_trace_analysis.md"
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("results/scheduler_capacity_study/warehouse_trace_pilot"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--analytical-config",
        type=Path,
        default=Path("config_thesis_strong.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.trace_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with args.analytical_config.open(encoding="ascii") as handle:
        analytical_config = yaml.safe_load(handle)
    central_config = analytical_config["network"]["central"]
    scheduler_demand_base_ms = float(central_config["scheduler_demand_base_ms"])
    scheduler_demand_per_robot_ms = float(
        central_config["scheduler_demand_per_robot_ms"]
    )
    aggregate_rows, operation_rows = enrich_rows(
        _read_csv(args.trace_dir / "trace_aggregate.csv"),
        _read_csv(args.trace_dir / "trace_operations.csv"),
        scheduler_demand_base_ms=scheduler_demand_base_ms,
        scheduler_demand_per_robot_ms=scheduler_demand_per_robot_ms,
    )
    summaries = summarize_by_fleet(aggregate_rows, operation_rows)
    identity_checks = validate_operational_identities(aggregate_rows, operation_rows)
    _write_csv(output_dir / "warehouse_trace_summary.csv", summaries)
    _write_csv(output_dir / "operational_checks.csv", identity_checks)
    plot_summary(output_dir, summaries)
    report = write_report(
        output_dir,
        summaries,
        identity_checks,
        scheduler_demand_base_ms=scheduler_demand_base_ms,
        scheduler_demand_per_robot_ms=scheduler_demand_per_robot_ms,
    )
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()

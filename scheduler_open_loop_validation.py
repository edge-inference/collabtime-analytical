#!/usr/bin/env python3
"""Open-loop queue validation for the event-driven scheduler component.

Requests arrive independently of completions.  Threads characterize the shared
Python execution model; processes are an optimistic read-only replica bound on
one host.  Neither execution mode is presented as the stateful polling-based
warehouse scheduler.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import random
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t

from benchmark_environment import capture_environment
from scheduler_capacity_model import allen_cunneen_mean_wait_s, service_scv
from scheduler_demand_study import EventDrivenSchedulerWorkload, StudyConfig, build_workload


@dataclass(frozen=True)
class OpenLoopConfig:
    warehouse_repo: str
    fleet_size: int = 800
    workers: tuple[int, ...] = (1, 2, 4, 6)
    load_fractions: tuple[float, ...] = (0.50, 0.80, 0.95, 1.05)
    arrival_duration_s: float = 3.0
    warmup_requests: int = 100
    saturation_requests: int = 1500
    seed: int = 9901


@dataclass(frozen=True)
class TimedRequest:
    request_id: int
    scheduled_ns: int
    submitted_ns: int
    service_start_ns: int
    completed_ns: int
    cpu_ns: int
    success: int


_PROCESS_WORKLOAD: EventDrivenSchedulerWorkload | None = None


def _initialize_process(config_dict: dict[str, Any], request_count: int) -> None:
    global _PROCESS_WORKLOAD
    config_dict["fleet_sizes"] = tuple(config_dict["fleet_sizes"])
    config_dict["validation_workers"] = tuple(config_dict["validation_workers"])
    config = StudyConfig(**config_dict)
    _PROCESS_WORKLOAD = build_workload(
        config,
        config.validation_fleet_size,
        config.sample_seed + 5_000_000,
        request_count,
    )


def _run_transaction(workload: EventDrivenSchedulerWorkload, request_id: int) -> int:
    return workload.transaction(request_id)


def _thread_timed_request(
    workload: EventDrivenSchedulerWorkload,
    payload: tuple[int, int, int],
) -> TimedRequest:
    request_id, scheduled_ns, submitted_ns = payload
    remaining_ns = scheduled_ns - time.perf_counter_ns()
    if remaining_ns > 0:
        time.sleep(remaining_ns / 1_000_000_000.0)
    service_start_ns = time.perf_counter_ns()
    cpu_start_ns = time.thread_time_ns()
    success = _run_transaction(workload, request_id)
    cpu_ns = time.thread_time_ns() - cpu_start_ns
    completed_ns = time.perf_counter_ns()
    return TimedRequest(
        request_id,
        scheduled_ns,
        submitted_ns,
        service_start_ns,
        completed_ns,
        cpu_ns,
        success,
    )


def _process_timed_request(payload: tuple[int, int, int]) -> TimedRequest:
    if _PROCESS_WORKLOAD is None:
        raise RuntimeError("Process workload was not initialized")
    request_id, scheduled_ns, submitted_ns = payload
    remaining_ns = scheduled_ns - time.perf_counter_ns()
    if remaining_ns > 0:
        time.sleep(remaining_ns / 1_000_000_000.0)
    service_start_ns = time.perf_counter_ns()
    cpu_start_ns = time.process_time_ns()
    success = _run_transaction(_PROCESS_WORKLOAD, request_id)
    cpu_ns = time.process_time_ns() - cpu_start_ns
    completed_ns = time.perf_counter_ns()
    return TimedRequest(
        request_id,
        scheduled_ns,
        submitted_ns,
        service_start_ns,
        completed_ns,
        cpu_ns,
        success,
    )


def poisson_offsets_ns(rate_per_s: float, duration_s: float, seed: int) -> list[int]:
    if rate_per_s <= 0 or duration_s <= 0:
        raise ValueError("rate and duration must be positive")
    rng = random.Random(seed)
    offsets: list[int] = []
    offset_s = rng.expovariate(rate_per_s)
    while offset_s < duration_s:
        offsets.append(int(round(offset_s * 1_000_000_000.0)))
        offset_s += rng.expovariate(rate_per_s)
    return offsets


def _measure_saturation(
    executor,
    transaction: Callable[[int], int],
    warmup_requests: int,
    measured_requests: int,
) -> tuple[float, float]:
    list(executor.map(transaction, range(warmup_requests)))
    start = time.perf_counter_ns()
    successes = sum(
        executor.map(
            transaction,
            range(warmup_requests, warmup_requests + measured_requests),
        )
    )
    elapsed_s = (time.perf_counter_ns() - start) / 1_000_000_000.0
    return measured_requests / elapsed_s, successes / measured_requests


def _submit_open_loop(
    executor,
    submit_function: Callable[[tuple[int, int, int]], TimedRequest],
    offsets_ns: Sequence[int],
    duration_s: float,
) -> tuple[list[TimedRequest], int, float]:
    # Preload jobs in chronological order so a client-side submitter cannot be
    # starved by the same GIL or CPU pool under test. Workers wait until each
    # request's future arrival timestamp before beginning service.
    preload_lead_s = max(1.0, len(offsets_ns) * 0.0002)
    start_ns = time.perf_counter_ns() + int(preload_lead_s * 1_000_000_000.0)
    futures: list[Future] = []
    for request_id, offset_ns in enumerate(offsets_ns):
        scheduled_ns = start_ns + int(offset_ns)
        submitted_ns = scheduled_ns
        futures.append(
            executor.submit(
                submit_function,
                (request_id, scheduled_ns, submitted_ns),
            )
        )
    if time.perf_counter_ns() >= start_ns:
        raise RuntimeError(
            "Open-loop preload exceeded its lead time; increase preload margin"
        )
    arrival_end_ns = start_ns + int(duration_s * 1_000_000_000.0)
    results = [future.result() for future in futures]
    completed_by_end = sum(result.completed_ns <= arrival_end_ns for result in results)
    backlog_at_end = len(results) - completed_by_end
    achieved_tps = completed_by_end / duration_s
    return results, backlog_at_end, achieved_tps


def summarize_requests(
    requests: Sequence[TimedRequest],
    *,
    execution_model: str,
    workers: int,
    target_rate: float,
    saturation_tps: float,
    load_fraction: float,
    duration_s: float,
    backlog_at_end: int,
    achieved_tps: float,
) -> dict[str, Any]:
    if not requests:
        raise ValueError("open-loop run produced no requests")
    queue_s = np.asarray(
        [(row.service_start_ns - row.submitted_ns) / 1e9 for row in requests]
    )
    service_s = np.asarray(
        [(row.completed_ns - row.service_start_ns) / 1e9 for row in requests]
    )
    response_s = np.asarray(
        [(row.completed_ns - row.submitted_ns) / 1e9 for row in requests]
    )
    schedule_lag_s = np.asarray(
        [(row.submitted_ns - row.scheduled_ns) / 1e9 for row in requests]
    )
    mean_service = float(np.mean(service_s))
    service_variability = service_scv(service_s)
    predicted_wait = allen_cunneen_mean_wait_s(
        arrival_rate_per_s=target_rate,
        mean_service_s=mean_service,
        servers=workers,
        arrival_scv=1.0,
        service_scv=service_variability,
    )
    return {
        "execution_model": execution_model,
        "workers": workers,
        "load_fraction_of_measured_saturation": load_fraction,
        "target_arrival_tps": target_rate,
        "measured_saturation_tps": saturation_tps,
        "submitted_requests": len(requests),
        "achieved_tps_during_arrivals": achieved_tps,
        "backlog_at_arrival_end": backlog_at_end,
        "success_rate": float(np.mean([row.success for row in requests])),
        "mean_service_ms": mean_service * 1000.0,
        "service_scv": service_variability,
        "mean_cpu_ms": float(np.mean([row.cpu_ns for row in requests])) / 1e6,
        "mean_queue_ms": float(np.mean(queue_s)) * 1000.0,
        "p50_queue_ms": float(np.percentile(queue_s, 50)) * 1000.0,
        "p95_queue_ms": float(np.percentile(queue_s, 95)) * 1000.0,
        "p99_queue_ms": float(np.percentile(queue_s, 99)) * 1000.0,
        "mean_response_ms": float(np.mean(response_s)) * 1000.0,
        "p95_response_ms": float(np.percentile(response_s, 95)) * 1000.0,
        "p99_response_ms": float(np.percentile(response_s, 99)) * 1000.0,
        "p99_arrival_schedule_lag_ms": float(np.percentile(schedule_lag_s, 99))
        * 1000.0,
        "allen_cunneen_mean_queue_ms": predicted_wait * 1000.0,
        "arrival_duration_s": duration_s,
    }


def _make_study_config(config: OpenLoopConfig) -> StudyConfig:
    max_requests = int(
        max(config.saturation_requests, 2 * config.arrival_duration_s * 5000)
    )
    return StudyConfig(
        warehouse_repo=config.warehouse_repo,
        fleet_sizes=(config.fleet_size,),
        validation_fleet_size=config.fleet_size,
        validation_workers=config.workers,
        validation_transactions=config.saturation_requests,
        sample_seed=config.seed,
        warmup_transactions=config.warmup_requests,
        samples_per_replicate=max_requests,
    )


def run_validation(
    config: OpenLoopConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    study_config = _make_study_config(config)
    request_count = study_config.samples_per_replicate + config.warmup_requests + 100
    thread_workload = build_workload(
        study_config,
        config.fleet_size,
        config.seed + 5_000_000,
        request_count,
    )
    saturation_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    process_context = multiprocessing.get_context("spawn")

    for execution_model in ("threads", "processes"):
        for workers in config.workers:
            if execution_model == "threads":
                executor_factory = lambda: ThreadPoolExecutor(max_workers=workers)
                saturation_function = thread_workload.transaction
                timed_function = lambda payload: _thread_timed_request(
                    thread_workload, payload
                )
            else:
                executor_factory = lambda: ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=process_context,
                    initializer=_initialize_process,
                    initargs=(asdict(study_config), request_count),
                )
                saturation_function = _process_saturation_transaction
                timed_function = _process_timed_request

            with executor_factory() as executor:
                saturation_tps, success_rate = _measure_saturation(
                    executor,
                    saturation_function,
                    config.warmup_requests,
                    config.saturation_requests,
                )
            saturation_rows.append(
                {
                    "seed": config.seed,
                    "execution_model": execution_model,
                    "workers": workers,
                    "saturation_tps": saturation_tps,
                    "success_rate": success_rate,
                }
            )

            for fraction_index, load_fraction in enumerate(config.load_fractions):
                target_rate = saturation_tps * load_fraction
                offsets = poisson_offsets_ns(
                    target_rate,
                    config.arrival_duration_s,
                    config.seed + 10_000 * workers + 100 * fraction_index,
                )
                with executor_factory() as executor:
                    # Initialize workers before the open-loop observation.
                    list(
                        executor.map(
                            saturation_function,
                            range(config.warmup_requests),
                        )
                    )
                    results, backlog, achieved = _submit_open_loop(
                        executor,
                        timed_function,
                        offsets,
                        config.arrival_duration_s,
                    )
                queue_rows.append(
                    {
                        "seed": config.seed,
                        **summarize_requests(
                            results,
                            execution_model=execution_model,
                            workers=workers,
                            target_rate=target_rate,
                            saturation_tps=saturation_tps,
                            load_fraction=load_fraction,
                            duration_s=config.arrival_duration_s,
                            backlog_at_end=backlog,
                            achieved_tps=achieved,
                        ),
                    }
                )
    return saturation_rows, queue_rows


def _mean_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, math.nan
    critical = float(t.ppf(0.975, df=len(array) - 1))
    half_width = critical * float(np.std(array, ddof=1)) / math.sqrt(len(array))
    return mean, half_width


def summarize_replications(
    saturation_rows: Sequence[dict[str, Any]],
    queue_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    saturation_summaries: list[dict[str, Any]] = []
    for execution_model in sorted(
        {str(row["execution_model"]) for row in saturation_rows}
    ):
        for workers in sorted(
            {
                int(row["workers"])
                for row in saturation_rows
                if row["execution_model"] == execution_model
            }
        ):
            rows = [
                row
                for row in saturation_rows
                if row["execution_model"] == execution_model
                and int(row["workers"]) == workers
            ]
            summary: dict[str, Any] = {
                "execution_model": execution_model,
                "workers": workers,
                "replicates": len(rows),
            }
            for metric in ("saturation_tps", "success_rate"):
                mean, half_width = _mean_ci(
                    [float(row[metric]) for row in rows]
                )
                summary[metric] = mean
                summary[f"{metric}_ci"] = half_width
            saturation_summaries.append(summary)

    queue_summaries: list[dict[str, Any]] = []
    grouping = sorted(
        {
            (
                str(row["execution_model"]),
                int(row["workers"]),
                float(row["load_fraction_of_measured_saturation"]),
            )
            for row in queue_rows
        }
    )
    queue_metrics = (
        "target_arrival_tps",
        "measured_saturation_tps",
        "submitted_requests",
        "achieved_tps_during_arrivals",
        "backlog_at_arrival_end",
        "success_rate",
        "mean_service_ms",
        "service_scv",
        "mean_cpu_ms",
        "mean_queue_ms",
        "p50_queue_ms",
        "p95_queue_ms",
        "p99_queue_ms",
        "mean_response_ms",
        "p95_response_ms",
        "p99_response_ms",
        "p99_arrival_schedule_lag_ms",
        "allen_cunneen_mean_queue_ms",
    )
    for execution_model, workers, load_fraction in grouping:
        rows = [
            row
            for row in queue_rows
            if row["execution_model"] == execution_model
            and int(row["workers"]) == workers
            and math.isclose(
                float(row["load_fraction_of_measured_saturation"]), load_fraction
            )
        ]
        summary = {
            "execution_model": execution_model,
            "workers": workers,
            "load_fraction_of_measured_saturation": load_fraction,
            "replicates": len(rows),
            "arrival_duration_s": float(rows[0]["arrival_duration_s"]),
        }
        for metric in queue_metrics:
            values = [float(row[metric]) for row in rows]
            if any(math.isinf(value) for value in values):
                mean, half_width = math.inf, math.nan
            else:
                mean, half_width = _mean_ci(values)
            summary[metric] = mean
            summary[f"{metric}_ci"] = half_width
        queue_summaries.append(summary)
    return saturation_summaries, queue_summaries


def _process_saturation_transaction(request_id: int) -> int:
    if _PROCESS_WORKLOAD is None:
        raise RuntimeError("Process workload was not initialized")
    return _PROCESS_WORKLOAD.transaction(request_id)


def fit_usl(saturation_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    fits: list[dict[str, Any]] = []
    for execution_model in ("threads", "processes"):
        rows = [row for row in saturation_rows if row["execution_model"] == execution_model]
        workers = np.asarray([float(row["workers"]) for row in rows])
        throughput = np.asarray([float(row["saturation_tps"]) for row in rows])
        relative = throughput / throughput[workers == 1][0]

        def residual(parameters: np.ndarray) -> np.ndarray:
            contention, coherency = parameters
            predicted = workers / (
                1.0
                + contention * (workers - 1.0)
                + coherency * workers * (workers - 1.0)
            )
            return predicted - relative

        fit = least_squares(residual, x0=np.asarray([0.05, 0.005]), bounds=(0, np.inf))
        predicted = relative + residual(fit.x)
        fits.append(
            {
                "execution_model": execution_model,
                "contention": float(fit.x[0]),
                "coherency": float(fit.x[1]),
                "rmse_relative_capacity": float(
                    np.sqrt(np.mean((predicted - relative) ** 2))
                ),
            }
        )
    return fits


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    output_dir: Path,
    saturation_rows: Sequence[dict[str, Any]],
    queue_rows: Sequence[dict[str, Any]],
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
    for model, marker in (("threads", "s"), ("processes", "o")):
        rows = [row for row in saturation_rows if row["execution_model"] == model]
        axes[0].errorbar(
            [row["workers"] for row in rows],
            [row["saturation_tps"] for row in rows],
            yerr=[row.get("saturation_tps_ci", math.nan) for row in rows],
            marker=marker,
            capsize=2,
            label=model.title(),
        )
    axes[0].set_xlabel("Workers")
    axes[0].set_ylabel("Measured saturated throughput (tasks/s)")
    axes[0].set_title("(a) Host worker scaling", loc="left")
    axes[0].legend(frameon=False)

    for model, marker in (("threads", "s"), ("processes", "o")):
        rows = [
            row
            for row in queue_rows
            if row["execution_model"] == model and int(row["workers"]) == 1
        ]
        axes[1].errorbar(
            [row["load_fraction_of_measured_saturation"] for row in rows],
            [row["p95_response_ms"] for row in rows],
            yerr=[row.get("p95_response_ms_ci", math.nan) for row in rows],
            marker=marker,
            capsize=2,
            label=model.title(),
        )
    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Offered load / measured saturation throughput")
    axes[1].set_ylabel("p95 response time (ms)")
    axes[1].set_title("(b) Open-loop response", loc="left")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    pdf_path = output_dir / "open_loop_validation.pdf"
    png_path = output_dir / "open_loop_validation.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def write_report(
    output_dir: Path,
    config: OpenLoopConfig,
    saturation_rows: Sequence[dict[str, Any]],
    queue_rows: Sequence[dict[str, Any]],
    usl_fits: Sequence[dict[str, Any]],
) -> Path:
    lines = [
        "# Scheduler Open-Loop Queue Validation",
        "",
        "## Scope",
        "",
        "This experiment validates queue and worker-scaling behavior for the",
        "read-only event-driven component transaction. Threads represent the shared",
        "Python execution model. Processes are an optimistic upper bound with",
        "independent state copies; they are not a consistency-preserving replicated",
        "scheduler deployment.",
        "",
        "## Saturated Throughput",
        "",
        "Values are replicate means with 95% Student-t confidence half-widths.",
        "",
        "| Execution | Workers | Reps | Throughput (tasks/s) | Success rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in saturation_rows:
        lines.append(
            f"| {row['execution_model']} | {row['workers']} | "
            f"{row['replicates']} | {row['saturation_tps']:.2f} +/- "
            f"{row['saturation_tps_ci']:.2f} | {row['success_rate']:.3f} +/- "
            f"{row['success_rate_ci']:.3f} |"
        )
    lines.extend(["", "## Worker-Scaling Fit", ""])
    for fit in usl_fits:
        lines.append(
            f"- {fit['execution_model']}: contention={fit['contention']:.6f}, "
            f"coherency={fit['coherency']:.6f}, relative-capacity "
            f"RMSE={fit['rmse_relative_capacity']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Open-Loop Results",
            "",
            "| Execution | Workers | Reps | Load fraction | Target tasks/s | Backlog at end | Mean service ms | Service SCV | Mean queue ms | Predicted mean queue ms | p95 response ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in queue_rows:
        predicted = row["allen_cunneen_mean_queue_ms"]
        predicted_text = "inf" if not math.isfinite(predicted) else f"{predicted:.3f}"
        lines.append(
            f"| {row['execution_model']} | {row['workers']} | {row['replicates']} | "
            f"{row['load_fraction_of_measured_saturation']:.2f} | "
            f"{row['target_arrival_tps']:.1f} | "
            f"{row['backlog_at_arrival_end']:.1f} +/- {row['backlog_at_arrival_end_ci']:.1f} | "
            f"{row['mean_service_ms']:.3f} +/- {row['mean_service_ms_ci']:.3f} | "
            f"{row['service_scv']:.3f} +/- {row['service_scv_ci']:.3f} | "
            f"{row['mean_queue_ms']:.3f} +/- {row['mean_queue_ms_ci']:.3f} | "
            f"{predicted_text} | {row['p95_response_ms']:.3f} +/- "
            f"{row['p95_response_ms_ci']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "Arrival schedules are generated independently of service completions, so",
            "overload appears as sustained backlog rather than client-side throttling.",
            "The GI/G/c result is a two-moment mean-wait approximation; empirical p95",
            "and p99 values are not claimed to follow from that mean. Multi-process",
            "capacity cannot be inserted as R scheduler replicas without specifying how",
            "their task registry and reservation state remain consistent.",
            "",
        ]
    )
    path = output_dir / "open_loop_validation.md"
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warehouse-repo", default="/home/modfi/ivalab/extern/warehouse"
    )
    parser.add_argument("--fleet-size", type=int, default=800)
    parser.add_argument("--workers", nargs="+", type=int, default=(1, 2, 4, 6))
    parser.add_argument(
        "--load-fractions", nargs="+", type=float, default=(0.50, 0.80, 0.95, 1.05)
    )
    parser.add_argument("--arrival-duration", type=float, default=5.0)
    parser.add_argument("--warmup-requests", type=int, default=100)
    parser.add_argument("--saturation-requests", type=int, default=3000)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=(9901, 9902, 9903)
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/scheduler_capacity_study/open_loop"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OpenLoopConfig(
        warehouse_repo=args.warehouse_repo,
        fleet_size=args.fleet_size,
        workers=tuple(args.workers),
        load_fractions=tuple(args.load_fractions),
        arrival_duration_s=args.arrival_duration,
        warmup_requests=args.warmup_requests,
        saturation_requests=args.saturation_requests,
        seed=args.seeds[0],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment_before = capture_environment()
    saturation_raw: list[dict[str, Any]] = []
    queue_raw: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"Open-loop replicate seed={seed}", flush=True)
        saturation_rows, queue_rows = run_validation(replace(config, seed=seed))
        saturation_raw.extend(saturation_rows)
        queue_raw.extend(queue_rows)
    saturation_rows, queue_rows = summarize_replications(
        saturation_raw, queue_raw
    )
    usl_fits = fit_usl(saturation_rows)
    _write_csv(args.output_dir / "saturation_raw.csv", saturation_raw)
    _write_csv(args.output_dir / "open_loop_raw.csv", queue_raw)
    _write_csv(args.output_dir / "saturation.csv", saturation_rows)
    _write_csv(args.output_dir / "open_loop.csv", queue_rows)
    (args.output_dir / "open_loop_metadata.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": asdict(config),
                "replicate_seeds": list(args.seeds),
                "host_before": environment_before,
                "host_after": capture_environment(),
                "usl_fits": usl_fits,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    plot_results(args.output_dir, saturation_rows, queue_rows)
    report = write_report(
        args.output_dir, config, saturation_rows, queue_rows, usl_fits
    )
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()

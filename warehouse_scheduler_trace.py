#!/usr/bin/env python3
"""Measure centralized scheduler operation visits in the warehouse model.

This is an implementation trace, not a warehouse performance comparison.  It
instruments non-overlapping public scheduler operations while the existing
centralized model runs headlessly.  The output supplies visit ratios and
resource demands for the operational identity

    D_task = sum_k V_k * S_k.

CPU time and elapsed in-service time are recorded separately.  Queue waiting
cannot occur in this trace because Mesa activates agents sequentially; queue
capacity is validated by a separate open-loop study.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from benchmark_environment import capture_environment, set_cpu_affinity


DEFAULT_FLEET_SIZES = (50, 100, 200, 300, 500, 600, 700, 800)
DEFAULT_METHODS = (
    "assign_task_to_agent",
    "request_path",
    "report_flow",
    "report_jam",
    "update_agent_position",
    "notify_task_released",
)


@dataclass(frozen=True)
class TraceConfig:
    warehouse_repo: str
    fleet_sizes: tuple[int, ...] = DEFAULT_FLEET_SIZES
    warehouse_width: int = 120
    warehouse_height: int = 100
    step_duration_s: float = 0.1
    warmup_s: float = 300.0
    observation_s: float = 300.0
    task_rate_per_robot: float = 0.006
    fixed_task_rate: float | None = None
    scheduler_replicas: int = 10
    seeds: tuple[int, ...] = (7301, 7302, 7303)
    reservoir_size: int = 20_000
    cpu_affinity: int | None = None


class OperationSample:
    """Online operation statistics with a bounded deterministic reservoir."""

    def __init__(self, reservoir_size: int, seed: int) -> None:
        self.reservoir_size = reservoir_size
        self.rng = random.Random(seed)
        self.calls = 0
        self.successes = 0
        self.exceptions = 0
        self.total_wall_ns = 0
        self.total_cpu_ns = 0
        self.success_wall_ns = 0
        self.success_cpu_ns = 0
        self.miss_wall_ns = 0
        self.miss_cpu_ns = 0
        self.wall_samples_ns: list[int] = []
        self.cpu_samples_ns: list[int] = []

    def reset(self) -> None:
        self.calls = 0
        self.successes = 0
        self.exceptions = 0
        self.total_wall_ns = 0
        self.total_cpu_ns = 0
        self.success_wall_ns = 0
        self.success_cpu_ns = 0
        self.miss_wall_ns = 0
        self.miss_cpu_ns = 0
        self.wall_samples_ns.clear()
        self.cpu_samples_ns.clear()

    def observe(
        self, wall_ns: int, cpu_ns: int, *, success: bool, exception: bool
    ) -> None:
        self.calls += 1
        self.successes += int(success)
        self.exceptions += int(exception)
        self.total_wall_ns += wall_ns
        self.total_cpu_ns += cpu_ns
        if success:
            self.success_wall_ns += wall_ns
            self.success_cpu_ns += cpu_ns
        elif not exception:
            self.miss_wall_ns += wall_ns
            self.miss_cpu_ns += cpu_ns

        if len(self.wall_samples_ns) < self.reservoir_size:
            self.wall_samples_ns.append(wall_ns)
            self.cpu_samples_ns.append(cpu_ns)
            return

        replacement = self.rng.randrange(self.calls)
        if replacement < self.reservoir_size:
            self.wall_samples_ns[replacement] = wall_ns
            self.cpu_samples_ns[replacement] = cpu_ns

    @staticmethod
    def _percentile(samples: Sequence[int], percentile: float) -> float:
        if not samples:
            return math.nan
        return float(np.percentile(np.asarray(samples, dtype=np.float64), percentile))

    def summary(self) -> dict[str, float | int]:
        misses = self.calls - self.successes - self.exceptions
        if self.calls:
            mean_wall_ms = self.total_wall_ns / self.calls / 1_000_000.0
            mean_cpu_ms = self.total_cpu_ns / self.calls / 1_000_000.0
        else:
            mean_wall_ms = math.nan
            mean_cpu_ms = math.nan
        success_mean_wall_ms = (
            self.success_wall_ns / self.successes / 1_000_000.0
            if self.successes
            else math.nan
        )
        success_mean_cpu_ms = (
            self.success_cpu_ns / self.successes / 1_000_000.0
            if self.successes
            else math.nan
        )
        miss_mean_wall_ms = (
            self.miss_wall_ns / misses / 1_000_000.0 if misses else math.nan
        )
        miss_mean_cpu_ms = (
            self.miss_cpu_ns / misses / 1_000_000.0 if misses else math.nan
        )
        return {
            "calls": self.calls,
            "successes": self.successes,
            "misses": misses,
            "exceptions": self.exceptions,
            "mean_wall_ms": mean_wall_ms,
            "mean_cpu_ms": mean_cpu_ms,
            "success_mean_wall_ms": success_mean_wall_ms,
            "success_mean_cpu_ms": success_mean_cpu_ms,
            "miss_mean_wall_ms": miss_mean_wall_ms,
            "miss_mean_cpu_ms": miss_mean_cpu_ms,
            "p50_wall_ms": self._percentile(self.wall_samples_ns, 50) / 1_000_000.0,
            "p95_wall_ms": self._percentile(self.wall_samples_ns, 95) / 1_000_000.0,
            "p99_wall_ms": self._percentile(self.wall_samples_ns, 99) / 1_000_000.0,
            "total_wall_s": self.total_wall_ns / 1_000_000_000.0,
            "total_cpu_s": self.total_cpu_ns / 1_000_000_000.0,
            "sample_count": len(self.wall_samples_ns),
        }


def _operation_succeeded(method_name: str, result: Any) -> bool:
    if method_name == "assign_task_to_agent":
        return result is not None
    if method_name == "request_path":
        return bool(result)
    if isinstance(result, bool):
        return result
    return True


class SchedulerInstrumentation:
    """Instrument selected bound methods on one scheduler instance."""

    def __init__(
        self,
        scheduler: Any,
        method_names: Iterable[str] = DEFAULT_METHODS,
        *,
        reservoir_size: int = 20_000,
        seed: int = 0,
    ) -> None:
        self.scheduler = scheduler
        self.samples: dict[str, OperationSample] = {}
        self.originals: dict[str, Callable[..., Any]] = {}

        for offset, method_name in enumerate(method_names):
            original = getattr(scheduler, method_name)
            sample = OperationSample(reservoir_size, seed + offset)
            self.originals[method_name] = original
            self.samples[method_name] = sample
            setattr(scheduler, method_name, self._wrap(method_name, original, sample))

    @staticmethod
    def _wrap(
        method_name: str,
        original: Callable[..., Any],
        sample: OperationSample,
    ) -> Callable[..., Any]:
        def measured(*args: Any, **kwargs: Any) -> Any:
            wall_start = time.perf_counter_ns()
            cpu_start = time.thread_time_ns()
            exception = False
            result: Any = None
            try:
                result = original(*args, **kwargs)
                return result
            except BaseException:
                exception = True
                raise
            finally:
                cpu_ns = time.thread_time_ns() - cpu_start
                wall_ns = time.perf_counter_ns() - wall_start
                sample.observe(
                    wall_ns,
                    cpu_ns,
                    success=(not exception and _operation_succeeded(method_name, result)),
                    exception=exception,
                )

        measured.__name__ = getattr(original, "__name__", method_name)
        measured.__doc__ = getattr(original, "__doc__", None)
        return measured

    def reset(self) -> None:
        for sample in self.samples.values():
            sample.reset()

    def restore(self) -> None:
        for method_name, original in self.originals.items():
            setattr(self.scheduler, method_name, original)

    def summaries(self) -> dict[str, dict[str, float | int]]:
        return {name: sample.summary() for name, sample in self.samples.items()}


def _prepend_repo(repo: str) -> None:
    resolved = str(Path(repo).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _load_warehouse(repo: str):
    _prepend_repo(repo)
    try:
        from world.graph import WarehouseGraph
        from world.model import WarehouseDSMModel
    except ImportError as exc:
        raise RuntimeError(
            "Warehouse dependencies are unavailable. Use the ivalab virtual "
            "environment and build perf/astar_fast first."
        ) from exc
    return WarehouseGraph, WarehouseDSMModel


def _agent_counts(model: Any) -> dict[str, int]:
    counts = {"idle": 0, "navigating": 0, "working": 0}
    for agent in model.schedule.agents:
        state = getattr(getattr(agent, "state", None), "value", "")
        if state in counts:
            counts[state] += 1
    return counts


def _sum_agent_metric(model: Any, key: str) -> int:
    return int(
        sum(int(getattr(agent, "metrics", {}).get(key, 0)) for agent in model.schedule.agents)
    )


def _git_revision(repo: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run_trace_case(
    config: TraceConfig, fleet_size: int, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    WarehouseGraph, WarehouseDSMModel = _load_warehouse(config.warehouse_repo)
    task_rate = (
        config.fixed_task_rate
        if config.fixed_task_rate is not None
        else config.task_rate_per_robot * fleet_size
    )
    logger = logging.getLogger(f"scheduler-trace-{fleet_size}-{seed}")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False

    random.seed(seed)
    graph = WarehouseGraph(config.warehouse_width, config.warehouse_height)
    model = WarehouseDSMModel(
        n_agents=fleet_size,
        warehouse_graph=graph,
        task_arrival_rate=task_rate,
        warehouse_width=config.warehouse_width,
        warehouse_height=config.warehouse_height,
        seed=seed,
        step_duration_s=config.step_duration_s,
        mode="centralized",
        centralized_replicas=config.scheduler_replicas,
        parallel_agents=False,
        logger=logger,
    )
    model.datacollector.collect = lambda _model: None
    instrumentation = SchedulerInstrumentation(
        model.central_scheduler,
        reservoir_size=config.reservoir_size,
        seed=seed,
    )

    warmup_steps = int(round(config.warmup_s / config.step_duration_s))
    observation_steps = int(round(config.observation_s / config.step_duration_s))
    try:
        progress_stride = max((warmup_steps + observation_steps) // 10, 1)
        for step in range(warmup_steps):
            model.step()
            if (step + 1) % progress_stride == 0:
                print(
                    f"  N={fleet_size} warmup {step + 1}/{warmup_steps}",
                    flush=True,
                )

        baseline = {
            "created": int(model.task_counter),
            "completed": len(model.completed_tasks),
            "terminal_failed": len(model.failed_tasks),
            "active": len(model.active_tasks),
            "failed": _sum_agent_metric(model, "tasks_failed"),
            "claimed": _sum_agent_metric(model, "tasks_claimed"),
        }
        instrumentation.reset()
        state_sums = {
            "idle": 0,
            "navigating": 0,
            "working": 0,
            "active_tasks": 0,
        }
        wall_start = time.perf_counter_ns()
        cpu_start = time.process_time_ns()
        for step in range(observation_steps):
            model.step()
            state = _agent_counts(model)
            for name, value in state.items():
                state_sums[name] += value
            state_sums["active_tasks"] += len(model.active_tasks)
            if (step + 1) % progress_stride == 0:
                print(
                    f"  N={fleet_size} observation {step + 1}/{observation_steps}",
                    flush=True,
                )
        process_cpu_s = (time.process_time_ns() - cpu_start) / 1_000_000_000.0
        process_wall_s = (time.perf_counter_ns() - wall_start) / 1_000_000_000.0

        completed = len(model.completed_tasks) - baseline["completed"]
        created = int(model.task_counter) - baseline["created"]
        terminal_failed = len(model.failed_tasks) - baseline["terminal_failed"]
        active_start = baseline["active"]
        active_end = len(model.active_tasks)
        active_delta = active_end - active_start
        flow_balance_error = created - completed - terminal_failed - active_delta
        failed = _sum_agent_metric(model, "tasks_failed") - baseline["failed"]
        claimed = _sum_agent_metric(model, "tasks_claimed") - baseline["claimed"]
        operation_summaries = instrumentation.summaries()
        scheduler_cpu_s = sum(
            float(summary["total_cpu_s"]) for summary in operation_summaries.values()
        )
        scheduler_wall_s = sum(
            float(summary["total_wall_s"]) for summary in operation_summaries.values()
        )
        normalizer = completed if completed > 0 else math.nan
        aggregate = {
            "fleet_size": fleet_size,
            "seed": seed,
            "task_rate_tps": task_rate,
            "warmup_s": config.warmup_s,
            "observation_s": config.observation_s,
            "step_duration_s": config.step_duration_s,
            "created_tasks": created,
            "claimed_tasks": claimed,
            "completed_tasks": completed,
            "terminal_failed_tasks": terminal_failed,
            "active_tasks_start": active_start,
            "active_tasks_end": active_end,
            "active_task_delta": active_delta,
            "mean_active_tasks": state_sums["active_tasks"] / observation_steps,
            "task_flow_balance_error": flow_balance_error,
            "active_task_drift_fraction_of_created": (
                active_delta / created if created > 0 else math.nan
            ),
            "completion_to_created_ratio": (
                completed / created if created > 0 else math.nan
            ),
            "failed_task_attempts": failed,
            "completion_throughput_tps": completed / config.observation_s,
            "terminal_throughput_tps": (
                completed + terminal_failed
            )
            / config.observation_s,
            "mean_idle_robots": state_sums["idle"] / observation_steps,
            "mean_navigating_robots": state_sums["navigating"] / observation_steps,
            "mean_working_robots": state_sums["working"] / observation_steps,
            "trace_process_wall_s": process_wall_s,
            "trace_process_cpu_s": process_cpu_s,
            "scheduler_cpu_s": scheduler_cpu_s,
            "scheduler_elapsed_s": scheduler_wall_s,
            "scheduler_cpu_core_load": scheduler_cpu_s / config.observation_s,
            "scheduler_fraction_of_process_cpu": (
                scheduler_cpu_s / process_cpu_s if process_cpu_s > 0 else math.nan
            ),
            "scheduler_cpu_ms_per_completed_task": scheduler_cpu_s * 1000.0 / normalizer,
            "scheduler_elapsed_ms_per_completed_task": scheduler_wall_s * 1000.0 / normalizer,
            "sequential_execution": True,
        }

        operation_rows: list[dict[str, Any]] = []
        for operation, summary in operation_summaries.items():
            calls = int(summary["calls"])
            operation_rows.append(
                {
                    "fleet_size": fleet_size,
                    "seed": seed,
                    "task_rate_tps": task_rate,
                    "operation": operation,
                    **summary,
                    "calls_per_s": calls / config.observation_s,
                    "calls_per_completed_task": calls / normalizer,
                    "cpu_ms_per_completed_task": float(summary["total_cpu_s"])
                    * 1000.0
                    / normalizer,
                    "elapsed_ms_per_completed_task": float(summary["total_wall_s"])
                    * 1000.0
                    / normalizer,
                }
            )
        return aggregate, operation_rows
    finally:
        instrumentation.restore()
        model.cleanup_shm()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value: Any, digits: int = 3) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(
    output_dir: Path,
    config: TraceConfig,
    aggregate_rows: Sequence[dict[str, Any]],
    operation_rows: Sequence[dict[str, Any]],
) -> Path:
    lines = [
        "# Warehouse Centralized Scheduler Operation Trace",
        "",
        "## Scope",
        "",
        "This trace measures the implemented polling-based centralized scheduler.",
        "Mesa activates robots sequentially, so the trace characterizes operation",
        "visits and single-process resource demand; it does not validate a",
        "multi-replica scheduler queue.",
        "",
        "CPU time is process/thread resource consumption. Elapsed time is measured",
        "inside each scheduler method and excludes any external network delay.",
        "",
        "## Configuration",
        "",
        f"- Warehouse: {config.warehouse_width} x {config.warehouse_height}",
        f"- Warmup: {config.warmup_s:g} simulated seconds",
        f"- Observation: {config.observation_s:g} simulated seconds",
        f"- Step duration: {config.step_duration_s:g} seconds",
        f"- Seeds: {', '.join(str(seed) for seed in config.seeds)}",
        f"- Task rate: {'fixed ' + str(config.fixed_task_rate) if config.fixed_task_rate is not None else str(config.task_rate_per_robot) + ' N'} tasks/s",
        "",
        "## Aggregate Observations",
        "",
        "| N | Seed | Offered tasks/s | Created | Completed | Terminal failed | Active start | Active end | Failed attempts | Throughput | Mean idle | Scheduler core load | CPU ms/completed task |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {fleet_size} | {seed} | {task_rate_tps:.3f} | {created_tasks} | "
            "{completed_tasks} | {terminal_failed_tasks} | "
            "{active_tasks_start} | {active_tasks_end} | {failed_task_attempts} | "
            "{completion_throughput_tps:.3f} | {mean_idle_robots:.1f} | "
            "{scheduler_cpu_core_load:.3f} | {cpu} |".format(
                **row,
                cpu=_format_number(row["scheduler_cpu_ms_per_completed_task"]),
            )
        )

    lines.extend(
        [
            "",
            "## Operation Visit Ratios",
            "",
            "| N | Seed | Operation | Calls/s | Calls/completed task | CPU ms/completed task | Mean CPU ms | Success CPU ms | Miss CPU ms | p95 elapsed ms |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in operation_rows:
        lines.append(
            "| {fleet_size} | {seed} | {operation} | {rate} | "
            "{visits} | {task_cpu} | {cpu} | {success_cpu} | {miss_cpu} | "
            "{p95} |".format(
                **row,
                rate=_format_number(row["calls_per_s"]),
                visits=_format_number(row["calls_per_completed_task"]),
                task_cpu=_format_number(row["cpu_ms_per_completed_task"]),
                cpu=_format_number(row["mean_cpu_ms"], 6),
                success_cpu=_format_number(row["success_mean_cpu_ms"], 6),
                miss_cpu=_format_number(row["miss_mean_cpu_ms"], 6),
                p95=_format_number(row["p95_wall_ms"], 6),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The per-completed-task normalization is meaningful only after a warmup",
            "long enough for the task flow to approach steady operation. Created tasks",
            "must balance terminal tasks plus the change in active tasks, and active-task",
            "drift must be reported before interpreting the normalized demand. Polling visits",
            "are an implementation workload, not an intrinsic property of centralized",
            "coordination. Network, remote database, replication, and process-to-process",
            "RPC costs are absent. Queue waiting requires a separate open-loop service",
            "experiment because this simulator invokes agents sequentially.",
            "",
        ]
    )
    path = output_dir / "warehouse_scheduler_trace.md"
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warehouse-repo",
        default="/home/modfi/ivalab/extern/warehouse",
    )
    parser.add_argument("--fleet-sizes", nargs="+", type=int, default=DEFAULT_FLEET_SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=(7301, 7302, 7303))
    parser.add_argument("--warehouse-width", type=int, default=120)
    parser.add_argument("--warehouse-height", type=int, default=100)
    parser.add_argument("--step-duration", type=float, default=0.1)
    parser.add_argument("--warmup", type=float, default=300.0)
    parser.add_argument("--observation", type=float, default=300.0)
    parser.add_argument("--task-rate-per-robot", type=float, default=0.006)
    parser.add_argument("--fixed-task-rate", type=float)
    parser.add_argument("--scheduler-replicas", type=int, default=10)
    parser.add_argument("--reservoir-size", type=int, default=20_000)
    parser.add_argument("--cpu-affinity", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/scheduler_capacity_study/warehouse_trace"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TraceConfig(
        warehouse_repo=args.warehouse_repo,
        fleet_sizes=tuple(args.fleet_sizes),
        warehouse_width=args.warehouse_width,
        warehouse_height=args.warehouse_height,
        step_duration_s=args.step_duration,
        warmup_s=args.warmup,
        observation_s=args.observation,
        task_rate_per_robot=args.task_rate_per_robot,
        fixed_task_rate=args.fixed_task_rate,
        scheduler_replicas=args.scheduler_replicas,
        seeds=tuple(args.seeds),
        reservoir_size=args.reservoir_size,
        cpu_affinity=args.cpu_affinity,
    )
    set_cpu_affinity(config.cpu_affinity)
    environment_before = capture_environment()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, Any]] = []
    operation_rows: list[dict[str, Any]] = []
    cases = [(fleet_size, seed) for seed in config.seeds for fleet_size in config.fleet_sizes]
    random.Random(90210).shuffle(cases)
    for index, (fleet_size, seed) in enumerate(cases, start=1):
        print(
            f"Trace case {index}/{len(cases)}: N={fleet_size}, seed={seed}",
            flush=True,
        )
        aggregate, operations = run_trace_case(config, fleet_size, seed)
        aggregate_rows.append(aggregate)
        operation_rows.extend(operations)

        checkpoint_aggregate = sorted(
            aggregate_rows,
            key=lambda row: (int(row["fleet_size"]), int(row["seed"])),
        )
        checkpoint_operations = sorted(
            operation_rows,
            key=lambda row: (
                int(row["fleet_size"]),
                int(row["seed"]),
                str(row["operation"]),
            ),
        )
        _write_csv(args.output_dir / "trace_aggregate.csv", checkpoint_aggregate)
        _write_csv(args.output_dir / "trace_operations.csv", checkpoint_operations)
        write_report(
            args.output_dir,
            config,
            checkpoint_aggregate,
            checkpoint_operations,
        )

    aggregate_rows.sort(key=lambda row: (int(row["fleet_size"]), int(row["seed"])))
    operation_rows.sort(
        key=lambda row: (int(row["fleet_size"]), int(row["seed"]), str(row["operation"]))
    )
    _write_csv(args.output_dir / "trace_aggregate.csv", aggregate_rows)
    _write_csv(args.output_dir / "trace_operations.csv", operation_rows)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "host_before": environment_before,
        "host_after": capture_environment(),
        "git": {
            "analytical": _git_revision(str(Path(__file__).resolve().parent)),
            "warehouse": _git_revision(config.warehouse_repo),
        },
    }
    (args.output_dir / "trace_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    report = write_report(args.output_dir, config, aggregate_rows, operation_rows)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()

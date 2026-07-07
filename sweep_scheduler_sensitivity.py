#!/usr/bin/env python3
"""Run a purely parametric scheduler-capacity sensitivity analysis.

The centralized scheduler demand is D_sched(N) = D0 + alpha*N milliseconds of
aggregate worker time per completed order. No simulation output is consumed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from comparison import DSMCentralizedComparison


REPLICAS = [5, 10, 15, 20]
BASE_DEMANDS_MS = [0.0, 300.0, 600.0, 1200.0, 2400.0]
PER_ROBOT_DEMAND_MS = [0.0, 0.5, 1.0, 2.0, 4.0]


def first_scheduler_bottleneck_threshold(
    fleet_sizes: list[int], central: list[float], physical: list[float]
) -> int | None:
    for fleet_size, central_limit, physical_limit in zip(
        fleet_sizes, central, physical
    ):
        if physical_limit > central_limit + 1e-12:
            return fleet_size
    return None


def run_sweep(
    comparison: DSMCentralizedComparison,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fleet_sizes = comparison.config["system"]["fleet_sizes"]
    stability = comparison.stability_boundary_analysis(fleet_sizes)
    physical_limits = stability["dsm_limits"]
    utilization_limit = comparison.queue_params.max_utilization

    sensitivity_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []

    for replicas in REPLICAS:
        for fleet_size, physical_limit in zip(fleet_sizes, physical_limits):
            threshold_ms = utilization_limit * replicas / physical_limit
            threshold_rows.append(
                {
                    "fleet_size": fleet_size,
                    "scheduler_workers": replicas,
                    "scheduler_demand_threshold_ms_per_order": threshold_ms,
                }
            )

        for base_ms in BASE_DEMANDS_MS:
            for per_robot_ms in PER_ROBOT_DEMAND_MS:
                central_limits = []
                for fleet_size, physical_limit in zip(fleet_sizes, physical_limits):
                    demand_ms = base_ms + per_robot_ms * fleet_size
                    scheduler_limit = (
                        float("inf")
                        if demand_ms <= 0
                        else utilization_limit * replicas / demand_ms
                    )
                    central_limits.append(min(physical_limit, scheduler_limit))

                bottleneck_threshold = first_scheduler_bottleneck_threshold(
                    fleet_sizes, central_limits, physical_limits
                )
                central_at_800 = central_limits[-1] * 1000.0
                physical_limit_at_800 = physical_limits[-1] * 1000.0
                sensitivity_rows.append(
                    {
                        "scheduler_workers": replicas,
                        "base_demand_ms_per_order": base_ms,
                        "per_robot_demand_ms_per_order": per_robot_ms,
                        "demand_at_800_ms_per_order": (
                            base_ms + per_robot_ms * fleet_sizes[-1]
                        ),
                        "scheduler_bottleneck_threshold_n": (
                            bottleneck_threshold
                            if bottleneck_threshold is not None
                            else ""
                        ),
                        "central_at_800_tps": central_at_800,
                        "physical_limit_at_800_tps": physical_limit_at_800,
                        "physical_to_central_capacity_ratio_at_800": (
                            physical_limit_at_800 / central_at_800
                        ),
                    }
                )

    return sensitivity_rows, threshold_rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="ascii") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_analysis(
    comparison: DSMCentralizedComparison,
    sensitivity_rows: list[dict[str, object]],
    threshold_rows: list[dict[str, object]],
    output_dir: Path,
) -> tuple[Path, Path]:
    matplotlib.rcParams.update(
        {
            "font.family": "Nimbus Roman",
            "font.serif": ["Nimbus Roman", "Times", "Times New Roman"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )

    fig, (ax_threshold, ax_heatmap) = plt.subplots(
        1, 2, figsize=(7.25, 2.9), constrained_layout=True
    )
    fleet_sizes = comparison.config["system"]["fleet_sizes"]

    for replicas in REPLICAS:
        values = [
            float(row["scheduler_demand_threshold_ms_per_order"]) / 1000.0
            for row in threshold_rows
            if int(row["scheduler_workers"]) == replicas
        ]
        ax_threshold.plot(
            fleet_sizes,
            values,
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=f"R={replicas}",
        )

    ax_threshold.set_yscale("log")
    ax_threshold.set_xlabel("Fleet size (N)")
    ax_threshold.set_ylabel("Scheduler demand threshold (s/order)")
    ax_threshold.set_title("(a) Scheduler-demand boundary", loc="left")
    ax_threshold.legend(frameon=False, fontsize=8, ncol=2)

    selected_workers = int(comparison.network_params.scheduler_replicas)
    grid = np.full((len(BASE_DEMANDS_MS), len(PER_ROBOT_DEMAND_MS)), np.nan)
    for row in sensitivity_rows:
        if int(row["scheduler_workers"]) != selected_workers:
            continue
        row_index = BASE_DEMANDS_MS.index(float(row["base_demand_ms_per_order"]))
        column_index = PER_ROBOT_DEMAND_MS.index(
            float(row["per_robot_demand_ms_per_order"])
        )
        bottleneck_threshold = row["scheduler_bottleneck_threshold_n"]
        if bottleneck_threshold != "":
            grid[row_index, column_index] = float(bottleneck_threshold)

    color_map = plt.get_cmap("viridis_r").copy()
    color_map.set_bad("#d9d9d9")
    image = ax_heatmap.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        aspect="auto",
        cmap=color_map,
        vmin=min(fleet_sizes),
        vmax=max(fleet_sizes),
    )
    ax_heatmap.set_xticks(np.arange(len(PER_ROBOT_DEMAND_MS)))
    ax_heatmap.set_xticklabels(
        [f"{value:g}" for value in PER_ROBOT_DEMAND_MS]
    )
    ax_heatmap.set_yticks(np.arange(len(BASE_DEMANDS_MS)))
    ax_heatmap.set_yticklabels(
        [f"{value / 1000.0:g}" for value in BASE_DEMANDS_MS]
    )
    ax_heatmap.set_xlabel(r"Growth $\alpha$ (ms/robot/order)")
    ax_heatmap.set_ylabel(r"Base demand $D_0$ (s/order)")
    ax_heatmap.set_title(
        f"(b) Scheduler-bottleneck threshold, R={selected_workers}", loc="left"
    )

    for row_index in range(len(BASE_DEMANDS_MS)):
        for column_index in range(len(PER_ROBOT_DEMAND_MS)):
            bottleneck_threshold = grid[row_index, column_index]
            label = (
                "none"
                if np.isnan(bottleneck_threshold)
                else f"N={int(bottleneck_threshold)}"
            )
            color = "#333333" if np.isnan(bottleneck_threshold) else "white"
            ax_heatmap.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=7.5,
            )

    color_bar = fig.colorbar(image, ax=ax_heatmap, fraction=0.05, pad=0.03)
    color_bar.set_label(r"First sampled threshold $N_{\mathrm{SB}}$")

    pdf_path = output_dir / "scheduler_capacity_sensitivity.pdf"
    png_path = output_dir / "scheduler_capacity_sensitivity.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config_thesis_strong.yaml")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/thesis_strong/sensitivity"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison = DSMCentralizedComparison(str(args.config))
    sensitivity_rows, threshold_rows = run_sweep(comparison)

    sensitivity_path = args.output_dir / "scheduler_sensitivity.csv"
    threshold_path = args.output_dir / "scheduler_capacity_thresholds.csv"
    write_csv(sensitivity_rows, sensitivity_path)
    write_csv(threshold_rows, threshold_path)
    pdf_path, _ = plot_analysis(
        comparison, sensitivity_rows, threshold_rows, args.output_dir
    )

    print(f"Sensitivity: {sensitivity_path}")
    print(f"Capacity thresholds: {threshold_path}")
    print(f"Figure: {pdf_path}")


if __name__ == "__main__":
    main()

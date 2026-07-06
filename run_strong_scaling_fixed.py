#!/usr/bin/env python3
"""Run a fixed-load analytical strong-scaling study."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comparison import DSMCentralizedComparison


def finite_latency_seconds(value_ms: float) -> float | None:
    if not math.isfinite(value_ms):
        return None
    return value_ms / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config_thesis_strong.yaml"),
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=5.0,
        help="Fixed total arrival rate in tasks/s (default: 5.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/thesis_strong/fixed_load_5tps"),
    )
    args = parser.parse_args()

    if args.arrival_rate < 0:
        parser.error("--arrival-rate must be non-negative")

    comparison = DSMCentralizedComparison(str(args.config))
    fleet_sizes = comparison.config["system"]["fleet_sizes"]
    fixed_rate_ms = args.arrival_rate / 1000.0
    boundaries = comparison.stability_boundary_analysis(fleet_sizes)

    rows: list[dict[str, object]] = []
    for index, fleet_size in enumerate(fleet_sizes):
        central_capacity = boundaries["central_limits"][index] * 1000.0
        collabtime_capacity = boundaries["dsm_limits"][index] * 1000.0
        central_stable = args.arrival_rate <= central_capacity
        collabtime_stable = args.arrival_rate <= collabtime_capacity

        central_latency = None
        if central_stable:
            central_latency = finite_latency_seconds(
                comparison.performance_model.total_latency(
                    fleet_size, fixed_rate_ms, "central"
                )["total"]
            )

        collabtime_latency = None
        if collabtime_stable:
            collabtime_latency = finite_latency_seconds(
                comparison.performance_model.total_latency(
                    fleet_size, fixed_rate_ms, "dsm"
                )["total"]
            )

        rows.append(
            {
                "fleet_size": fleet_size,
                "arrival_rate_tps": args.arrival_rate,
                "central_capacity_tps": central_capacity,
                "collabtime_capacity_tps": collabtime_capacity,
                "central_stable": central_stable,
                "collabtime_stable": collabtime_stable,
                "central_latency_s": central_latency,
                "collabtime_latency_s": collabtime_latency,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "strong_scaling_fixed_load.csv"
    with csv_path.open("w", newline="", encoding="ascii") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    matplotlib.rcParams.update(
        {
            "font.family": "Nimbus Roman",
            "font.serif": ["Nimbus Roman", "Times", "Times New Roman"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, (ax_capacity, ax_latency) = plt.subplots(
        1, 2, figsize=(7.25, 2.8), constrained_layout=True
    )

    central_capacities = [float(row["central_capacity_tps"]) for row in rows]
    collabtime_capacities = [float(row["collabtime_capacity_tps"]) for row in rows]
    ax_capacity.plot(fleet_sizes, central_capacities, "o-", label="Centralized")
    ax_capacity.plot(fleet_sizes, collabtime_capacities, "s-", label="CollabTime")
    ax_capacity.axhline(
        args.arrival_rate,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=rf"Fixed $\lambda={args.arrival_rate:g}$ tasks/s",
    )
    ax_capacity.set_xlabel("Fleet size (N)")
    ax_capacity.set_ylabel(r"Stable capacity $\lambda_{\max}$ (tasks/s)")
    ax_capacity.set_title("(a) Fixed-load stability", loc="left")
    ax_capacity.legend(frameon=False, fontsize=8)

    plotted_latency = False
    for key, marker, label in (
        ("central_latency_s", "o", "Centralized"),
        ("collabtime_latency_s", "s", "CollabTime"),
    ):
        points = [
            (int(row["fleet_size"]), float(row[key]))
            for row in rows
            if row[key] is not None
        ]
        if points:
            x_values, y_values = zip(*points)
            ax_latency.plot(x_values, y_values, marker=marker, label=label)
            plotted_latency = True

    if plotted_latency:
        ax_latency.legend(frameon=False, fontsize=8)
    else:
        ax_latency.text(
            0.5,
            0.5,
            "No stable points at this load",
            ha="center",
            va="center",
            transform=ax_latency.transAxes,
        )
    ax_latency.set_xlabel("Fleet size (N)")
    ax_latency.set_ylabel("Predicted response time (s)")
    ax_latency.set_title("(b) Stable-point latency", loc="left")

    pdf_path = args.output_dir / "strong_scaling_fixed_load.pdf"
    png_path = args.output_dir / "strong_scaling_fixed_load.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Fixed arrival rate: {args.arrival_rate:g} tasks/s")
    print("N    Central cap  Central stable  CollabTime cap  CollabTime stable")
    for row in rows:
        print(
            f"{int(row['fleet_size']):<4} "
            f"{float(row['central_capacity_tps']):>11.3f}  "
            f"{str(row['central_stable']):>14}  "
            f"{float(row['collabtime_capacity_tps']):>14.3f}  "
            f"{str(row['collabtime_stable']):>17}"
        )
    print(f"CSV: {csv_path}")
    print(f"Figure: {pdf_path}")


if __name__ == "__main__":
    main()

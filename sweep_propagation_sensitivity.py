#!/usr/bin/env python3
"""Sweep uncertain propagation-delay and periodic-update assumptions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from comparison import DSMCentralizedComparison
from propagation_models import PropagationModel


LINK_DELAYS_MS = [1.0, 2.0, 5.0, 10.0, 12.0, 20.0, 30.0, 50.0]
BATCH_PERIODS_MS = [50.0, 100.0, 200.0, 300.0]
GOSSIP_PERIODS_MS = [50.0, 100.0, 200.0, 300.0, 500.0]


def first_propagation_crossover(
    model: PropagationModel, min_fleet: int, max_fleet: int
) -> int | None:
    """Return the first integer fleet size where DSM propagation is faster."""
    for fleet_size in range(min_fleet, max_fleet + 1):
        central = model.central_propagation_time(fleet_size)["total"]
        dsm = model.dsm_propagation_time(fleet_size)["total"]
        if dsm < central:
            return fleet_size
    return None


def run_sweep(
    comparison: DSMCentralizedComparison,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    params = comparison.network_params
    model = comparison.performance_model.prop_model
    fleet_sizes = comparison.config["system"]["fleet_sizes"]
    min_fleet, max_fleet = min(fleet_sizes), max(fleet_sizes)

    original = {
        "hop_delay": params.hop_delay,
        "serialization_delay": params.serialization_delay,
        "batch_period": params.batch_period,
        "gossip_period": params.gossip_period,
    }
    link_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []

    try:
        for link_delay in LINK_DELAYS_MS:
            # These terms have the same per-hop coefficient in both equations.
            params.hop_delay = link_delay
            params.serialization_delay = 0.0
            crossover = first_propagation_crossover(model, min_fleet, max_fleet)
            central_800 = model.central_propagation_time(max_fleet)["total"]
            dsm_800 = model.dsm_propagation_time(max_fleet)["total"]
            link_rows.append(
                {
                    "combined_per_hop_delay_ms": link_delay,
                    "crossover_n": crossover if crossover is not None else "",
                    "central_at_max_fleet_ms": central_800,
                    "dsm_at_max_fleet_ms": dsm_800,
                    "central_to_dsm_ratio_at_max_fleet": central_800 / dsm_800,
                }
            )

        params.hop_delay = original["hop_delay"]
        params.serialization_delay = original["serialization_delay"]
        for batch_period in BATCH_PERIODS_MS:
            params.batch_period = batch_period
            for gossip_period in GOSSIP_PERIODS_MS:
                params.gossip_period = gossip_period
                crossover = first_propagation_crossover(
                    model, min_fleet, max_fleet
                )
                period_rows.append(
                    {
                        "batch_period_ms": batch_period,
                        "gossip_period_ms": gossip_period,
                        "crossover_n": crossover if crossover is not None else "",
                    }
                )
    finally:
        params.hop_delay = original["hop_delay"]
        params.serialization_delay = original["serialization_delay"]
        params.batch_period = original["batch_period"]
        params.gossip_period = original["gossip_period"]

    return link_rows, period_rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="ascii") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_analysis(
    comparison: DSMCentralizedComparison,
    link_rows: list[dict[str, object]],
    period_rows: list[dict[str, object]],
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

    fig, (ax_link, ax_period) = plt.subplots(
        1, 2, figsize=(7.25, 2.9), constrained_layout=True
    )

    link_delays = [float(row["combined_per_hop_delay_ms"]) for row in link_rows]
    crossovers = [
        np.nan if row["crossover_n"] == "" else float(row["crossover_n"])
        for row in link_rows
    ]
    ax_link.plot(link_delays, crossovers, "o-", linewidth=1.7, markersize=4)
    base_link_delay = (
        comparison.network_params.hop_delay
        + comparison.network_params.serialization_delay
    )
    fleet_sizes = comparison.config["system"]["fleet_sizes"]
    base_crossover = first_propagation_crossover(
        comparison.performance_model.prop_model,
        min(fleet_sizes),
        max(fleet_sizes),
    )
    if base_crossover is not None:
        ax_link.scatter(
            [base_link_delay],
            [base_crossover],
            facecolors="none",
            edgecolors="#d62728",
            s=90,
            linewidths=2,
            zorder=3,
            label="Base assumption",
        )
    ax_link.set_xscale("log")
    display_ticks = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    ax_link.set_xticks(display_ticks)
    ax_link.set_xticklabels([f"{value:g}" for value in display_ticks])
    ax_link.set_xlabel("Combined per-hop delay (ms)")
    ax_link.set_ylabel("Propagation crossover fleet size N")
    ax_link.set_title("(a) Communication-delay sensitivity", loc="left")
    ax_link.legend(frameon=False, fontsize=8)

    grid = np.full((len(BATCH_PERIODS_MS), len(GOSSIP_PERIODS_MS)), np.nan)
    for row in period_rows:
        row_index = BATCH_PERIODS_MS.index(float(row["batch_period_ms"]))
        column_index = GOSSIP_PERIODS_MS.index(float(row["gossip_period_ms"]))
        if row["crossover_n"] != "":
            grid[row_index, column_index] = float(row["crossover_n"])

    color_map = plt.get_cmap("viridis_r").copy()
    color_map.set_bad("#d9d9d9")
    image = ax_period.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        aspect="auto",
        cmap=color_map,
        vmin=min(value for value in grid.flat if not np.isnan(value)),
        vmax=max(value for value in grid.flat if not np.isnan(value)),
    )
    ax_period.set_xticks(np.arange(len(GOSSIP_PERIODS_MS)))
    ax_period.set_xticklabels([f"{value:g}" for value in GOSSIP_PERIODS_MS])
    ax_period.set_yticks(np.arange(len(BATCH_PERIODS_MS)))
    ax_period.set_yticklabels([f"{value:g}" for value in BATCH_PERIODS_MS])
    ax_period.set_xlabel("Gossip period (ms)")
    ax_period.set_ylabel("Central batch period (ms)")
    ax_period.set_title("(b) Period sensitivity", loc="left")

    for row_index in range(len(BATCH_PERIODS_MS)):
        for column_index in range(len(GOSSIP_PERIODS_MS)):
            crossover = grid[row_index, column_index]
            label = "none" if np.isnan(crossover) else f"N={int(crossover)}"
            color = "#333333" if np.isnan(crossover) else "white"
            ax_period.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=7.5,
            )

    base_batch_index = BATCH_PERIODS_MS.index(
        comparison.network_params.batch_period
    )
    base_gossip_index = GOSSIP_PERIODS_MS.index(
        comparison.network_params.gossip_period
    )
    ax_period.add_patch(
        Rectangle(
            (base_gossip_index - 0.5, base_batch_index - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#d62728",
            linewidth=2.0,
        )
    )
    color_bar = fig.colorbar(image, ax=ax_period, fraction=0.05, pad=0.03)
    color_bar.set_label("Crossover fleet size N")

    pdf_path = output_dir / "propagation_parameter_sensitivity.pdf"
    png_path = output_dir / "propagation_parameter_sensitivity.png"
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
    link_rows, period_rows = run_sweep(comparison)
    link_path = args.output_dir / "network_delay_sensitivity.csv"
    period_path = args.output_dir / "coordination_period_sensitivity.csv"
    write_csv(link_rows, link_path)
    write_csv(period_rows, period_path)
    pdf_path, _ = plot_analysis(
        comparison, link_rows, period_rows, args.output_dir
    )

    print(f"Network delay: {link_path}")
    print(f"Coordination periods: {period_path}")
    print(f"Figure: {pdf_path}")


if __name__ == "__main__":
    main()

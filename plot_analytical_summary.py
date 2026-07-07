#!/usr/bin/env python3
"""Create a three-panel analytical summary figure for the thesis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from comparison import DSMCentralizedComparison


FANOUT_FLEETS = [100, 300, 800]
FANOUTS = [1, 2, 4, 8]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config_thesis_strong.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/thesis_strong/plots/analytical_summary.pdf"),
    )
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "font.family": "Nimbus Roman",
        "font.serif": ["Nimbus Roman", "Times", "Times New Roman"],
        "font.sans-serif": ["Nimbus Roman", "Times", "Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    comp = DSMCentralizedComparison(str(args.config))
    fleet_sizes = comp.config["system"]["fleet_sizes"]
    fig, (ax_cap, ax_aoi, ax_fanout) = plt.subplots(
        1, 3, figsize=(7.25, 2.6), constrained_layout=True
    )

    robot_service_ms = (
        comp.system_params.expected_path_cells * comp.system_params.t_traverse
        + comp.system_params.t_work
    )
    scheduler_workers = comp.network_params.scheduler_replicas
    scheduler_boundaries = [
        scheduler_workers * robot_service_ms / fleet_size / 1000.0
        for fleet_size in fleet_sizes
    ]
    ax_cap.plot(
        fleet_sizes,
        scheduler_boundaries,
        "o-",
        linewidth=1.7,
        markersize=3.5,
        color="#1f77b4",
    )
    ax_cap.text(
        0.95,
        0.88,
        f"$R_{{\\mathrm{{sched}}}}={scheduler_workers}$",
        transform=ax_cap.transAxes,
        fontsize=8,
        ha="right",
        va="top",
    )
    ax_cap.set_xlabel("Fleet size (N)")
    ax_cap.set_ylabel(r"$D_{\mathrm{sched}}^*$ (s/order)")
    ax_cap.set_title("(a) Scheduler-demand boundary", loc="left")
    ax_cap.grid(False)

    gossip_periods = np.linspace(50, 500, 20)
    aoi = comp.aoi_violation_analysis(gossip_periods)
    ax_aoi.plot(
        gossip_periods,
        aoi["violation_probabilities"],
        "o-",
        linewidth=1.7,
        markersize=3.5,
        color="#d62728",
    )
    target_rate = 0.05
    ax_aoi.axhline(y=target_rate, color="#2ca02c", linestyle="--", linewidth=1.3)
    transmission_delay = comp.network_params.hop_delay * comp.network_params.tile_hops
    zero_violation_period = max(aoi["target_freshness"] - transmission_delay, 0.0)
    ax_aoi.axvline(x=zero_violation_period, color="#4c4cff", linestyle="--", linewidth=1.3)
    ax_aoi.text(62, target_rate + 0.018, "5%", fontsize=8, color="#2ca02c")
    ax_aoi.text(zero_violation_period + 8, 0.025, "280 ms", fontsize=8, color="#4c4cff")
    ax_aoi.set_xlabel("Gossip period (ms)")
    ax_aoi.set_ylabel("AoI violation probability")
    ax_aoi.set_ylim(bottom=0.0)
    ax_aoi.set_title("(b) Data freshness", loc="left")
    ax_aoi.grid(False)

    original_fanout = comp.network_params.gossip_fanout
    for n in FANOUT_FLEETS:
        values = []
        for fanout in FANOUTS:
            comp.network_params.gossip_fanout = fanout
            comp.setup_models()
            values.append(comp.performance_model.prop_model.dsm_propagation_time(n)["total"])
        ax_fanout.plot(FANOUTS, values, marker="o", linewidth=1.7, markersize=3.5, label=f"N={n}")
    comp.network_params.gossip_fanout = original_fanout
    comp.setup_models()
    ax_fanout.set_xlabel("Gossip fanout (f)")
    ax_fanout.set_ylabel("State propagation (ms)")
    ax_fanout.set_xticks(FANOUTS)
    ax_fanout.set_yscale("log")
    ax_fanout.set_title("(c) Fanout sensitivity", loc="left")
    ax_fanout.legend(frameon=False, loc="upper right")
    ax_fanout.grid(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

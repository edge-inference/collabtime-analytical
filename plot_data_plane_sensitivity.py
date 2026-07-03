#!/usr/bin/env python3
"""Create the combined data-plane sensitivity figure for the thesis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from comparison import DSMCentralizedComparison


FLEET_SIZES = [100, 300, 800]
FANOUTS = [1, 2, 4, 8]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config_thesis_strong.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/thesis_strong/plots/data_plane_sensitivity.pdf"))
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    comp = DSMCentralizedComparison(str(args.config))

    fig, (ax_aoi, ax_fanout) = plt.subplots(
        1, 2, figsize=(7.1, 3.2), constrained_layout=True
    )

    # Panel (a): deterministic periodic AoI violation probability.
    gossip_periods = np.linspace(50, 500, 20)
    aoi = comp.aoi_violation_analysis(gossip_periods)
    ax_aoi.plot(
        gossip_periods,
        aoi["violation_probabilities"],
        "o-",
        linewidth=1.8,
        markersize=4,
        color="#d62728",
    )
    target_rate = 0.05
    ax_aoi.axhline(y=target_rate, color="#2ca02c", linestyle="--", linewidth=1.5)
    target_fresh = aoi["target_freshness"]
    transmission_delay = comp.network_params.hop_delay * comp.network_params.tile_hops
    zero_violation_period = max(target_fresh - transmission_delay, 0.0)
    ax_aoi.axvline(
        x=zero_violation_period,
        color="#4c4cff",
        linestyle="--",
        linewidth=1.5,
    )
    ax_aoi.set_xlabel("Gossip period (ms)")
    ax_aoi.set_ylabel("AoI violation probability")
    ax_aoi.set_ylim(bottom=0.0)
    ax_aoi.set_title("(a) Freshness budget", loc="left", fontsize=10)
    ax_aoi.grid(False)

    # Panel (b): fanout sensitivity at representative fleet sizes.
    original_fanout = comp.network_params.gossip_fanout
    for n in FLEET_SIZES:
        values = []
        for fanout in FANOUTS:
            comp.network_params.gossip_fanout = fanout
            comp.setup_models()
            values.append(comp.performance_model.prop_model.dsm_propagation_time(n)["total"])
        ax_fanout.plot(FANOUTS, values, marker="o", linewidth=1.8, label=f"N={n}")
    comp.network_params.gossip_fanout = original_fanout
    comp.setup_models()

    ax_fanout.set_xlabel("Gossip fanout (f)")
    ax_fanout.set_ylabel("State propagation (ms)")
    ax_fanout.set_xticks(FANOUTS)
    ax_fanout.set_yscale("log")
    ax_fanout.set_title("(b) Fanout sensitivity", loc="left", fontsize=10)
    ax_fanout.legend(frameon=False, loc="upper right")
    ax_fanout.grid(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

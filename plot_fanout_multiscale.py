#!/usr/bin/env python3
"""Plot DSM propagation sensitivity to gossip fanout at several fleet sizes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from comparison import DSMCentralizedComparison


FLEET_SIZES = [100, 300, 800]
FANOUTS = [1, 2, 4, 8]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config_thesis_strong.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/thesis_strong/plots/fanout_multiscale.png"))
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 20,
        "legend.fontsize": 17,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    comp = DSMCentralizedComparison(str(args.config))
    original_fanout = comp.network_params.gossip_fanout

    fig, ax = plt.subplots(figsize=(5, 6), constrained_layout=True)
    for n in FLEET_SIZES:
        values = []
        for fanout in FANOUTS:
            comp.network_params.gossip_fanout = fanout
            comp.setup_models()
            values.append(comp.performance_model.prop_model.dsm_propagation_time(n)["total"])
        ax.plot(FANOUTS, values, marker="o", linewidth=2, label=f"N={n}")

    comp.network_params.gossip_fanout = original_fanout

    ax.set_xlabel("Gossip fanout (f)")
    ax.set_ylabel("DSM state propagation time (ms)")
    ax.set_xticks(FANOUTS)
    ax.set_yscale("log")
    ax.legend(frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

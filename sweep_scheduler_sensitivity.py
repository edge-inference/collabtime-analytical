#!/usr/bin/env python3
"""Sweep centralized scheduler assumptions for the thesis analytical model.

The goal is not to fit the empirical simulator exactly. This sweep checks
whether the predicted crossover is robust to plausible centralized scheduler
capacity assumptions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from comparison import DSMCentralizedComparison


REPLICAS = [5, 10, 15, 20]
PER_ROBOT_MS = [0.5, 1.0, 2.0, 4.0, 8.0]


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_config(config: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def first_capacity_crossover(fleet_sizes: list[int], central: list[float],
                             dsm: list[float]) -> int | None:
    for n, c, d in zip(fleet_sizes, central, dsm):
        if d > c:
            return n
    return None


def run_sweep(base_config: Path, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for replicas in REPLICAS:
        for per_robot_ms in PER_ROBOT_MS:
            cfg = load_config(base_config)
            central_cfg = cfg["network"]["central"]
            central_cfg["scheduler_replicas"] = replicas
            central_cfg["scheduler_service_per_robot_ms"] = per_robot_ms

            tmp_cfg = output_dir / f"_tmp_rep{replicas}_per{per_robot_ms}.yaml"
            write_config(cfg, tmp_cfg)

            comp = DSMCentralizedComparison(str(tmp_cfg))
            fleet_sizes = cfg["system"]["fleet_sizes"]
            stability = comp.stability_boundary_analysis(fleet_sizes)
            crossover = first_capacity_crossover(
                fleet_sizes,
                stability["central_limits"],
                stability["dsm_limits"],
            )

            rows.append({
                "scheduler_replicas": replicas,
                "per_robot_ms": per_robot_ms,
                "crossover_n": crossover if crossover is not None else "",
                "central_at_800_tps": stability["central_limits"][-1] * 1000.0,
                "dsm_at_800_tps": stability["dsm_limits"][-1] * 1000.0,
                "advantage_at_800": stability["throughput_advantage"][-1],
            })

            tmp_cfg.unlink(missing_ok=True)

    return rows


def write_csv(rows: list[dict], output_dir: Path) -> Path:
    csv_path = output_dir / "scheduler_sensitivity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_heatmap(rows: list[dict], output_dir: Path) -> Path:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    grid = np.full((len(REPLICAS), len(PER_ROBOT_MS)), np.nan)
    for row in rows:
        i = REPLICAS.index(int(row["scheduler_replicas"]))
        j = PER_ROBOT_MS.index(float(row["per_robot_ms"]))
        crossover = row["crossover_n"]
        grid[i, j] = float(crossover) if crossover != "" else np.nan

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(PER_ROBOT_MS)))
    ax.set_xticklabels([str(v) for v in PER_ROBOT_MS])
    ax.set_yticks(np.arange(len(REPLICAS)))
    ax.set_yticklabels([str(v) for v in REPLICAS])
    ax.set_xlabel("Scheduler cost per robot (ms)")
    ax.set_ylabel("Central scheduler replicas")

    for i in range(len(REPLICAS)):
        for j in range(len(PER_ROBOT_MS)):
            val = grid[i, j]
            text = "none" if np.isnan(val) else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center", color="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("First DSM > centralized capacity (N)")

    out = output_dir / "scheduler_sensitivity.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config_thesis_strong.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/thesis_strong/sensitivity"))
    args = parser.parse_args()

    rows = run_sweep(args.config, args.output_dir)
    csv_path = write_csv(rows, args.output_dir)
    plot_path = plot_heatmap(rows, args.output_dir)
    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot Best-of-K proxy gain and directional guardrail budget trajectories."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SERIES = {
    "proxy_gain_pool_sd": ("Target proxy", "#c18401", "o"),
    "control_budget_coverage": ("Coverage", "#2563a6", "s"),
    "control_budget_musecritic_mean5": ("MuseCritic", "#b45309", "^"),
    "control_budget_silence_rate": ("Silence", "#6b7280", "D"),
    "control_budget_repetition_rate": ("Repetition", "#b14b7d", "v"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Best-of-K reward attack trajectories")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.summary.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    arms = sorted(grouped)
    columns = 3
    rows_count = math.ceil(len(arms) / columns)
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(12, 3.2 * rows_count),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, arm in zip(axes.flat, arms):
        arm_rows = sorted(grouped[arm], key=lambda row: float(row["bon_kl"]))
        x = [float(row["bon_kl"]) for row in arm_rows]
        for key, (label, color, marker) in SERIES.items():
            if key not in arm_rows[0] or arm_rows[0][key] == "":
                continue
            axis.plot(
                x,
                [float(row[key]) for row in arm_rows],
                color=color,
                marker=marker,
                linewidth=1.6,
                markersize=4,
                label=label,
            )
        failed_x = [
            float(row["bon_kl"])
            for row in arm_rows
            if row.get("failed", "").lower() == "true"
        ]
        if failed_x:
            axis.scatter(
                failed_x,
                [-1.0] * len(failed_x),
                marker="x",
                s=42,
                linewidths=1.5,
                color="#b91c1c",
                zorder=5,
            )
        axis.axhline(0.0, color="#1f2937", linewidth=0.8)
        axis.axhline(-1.0, color="#1f2937", linewidth=0.8, linestyle="--")
        axis.set_title(arm.replace("_", " "), fontsize=9)
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    for axis in axes.flat[len(arms):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        if axis.get_visible():
            axis.set_xlabel("Best-of-K analytic KL")
    for axis in axes[:, 0]:
        axis.set_ylabel("Proxy SD / guardrail budget units")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle(args.title, fontsize=13, y=0.995)
    fig.text(
        0.5,
        0.01,
        "Controls are signed so positive is better; -1 crosses the preregistered non-inferiority budget.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    if output.suffix.lower() != ".svg":
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

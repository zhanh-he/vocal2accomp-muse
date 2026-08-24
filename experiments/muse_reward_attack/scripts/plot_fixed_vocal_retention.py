#!/usr/bin/env python3
"""Plot normalized co-generated gain and fixed-vocal gain retention."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    k = args.k or max(int(row["k"]) for row in rows)
    selected = [
        row for row in rows
        if int(row["k"]) == k and row["counterfactual_gain_retention"] != ""
    ]
    selected.sort(key=lambda row: float(row["counterfactual_gain_retention"]))
    labels = [row["arm"].replace("_", " ") for row in selected]
    retention = np.asarray([
        float(row["counterfactual_gain_retention"]) for row in selected
    ])
    positions = np.arange(len(selected))
    fig, axis = plt.subplots(figsize=(10, max(4.8, 0.45 * len(selected))))
    height = 0.34
    axis.barh(
        positions - height / 2,
        np.ones(len(selected)),
        height=height,
        color="#9ca3af",
        label="Co-generated gain (normalized)",
    )
    colors = np.where(retention >= 0.0, "#2563a6", "#b91c1c")
    axis.barh(
        positions + height / 2,
        retention,
        height=height,
        color=colors,
        label="Gain retained with vocal fixed",
    )
    axis.axvline(0.0, color="#111827", linewidth=0.9)
    axis.axvline(1.0, color="#6b7280", linewidth=0.8, linestyle="--")
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Counterfactual gain retention")
    axis.grid(axis="x", color="#e5e7eb", linewidth=0.7)
    handles, legend_labels = axis.get_legend_handles_labels()
    fig.suptitle(f"Do Best-of-{k} reward gains survive a fixed vocal?", y=0.995)
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
    )
    fig.text(
        0.5,
        0.01,
        "Negative retention means the co-generated proxy improved while the same score worsened with vocal fixed.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    if output.suffix.lower() != ".svg":
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

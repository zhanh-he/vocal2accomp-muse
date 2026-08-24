#!/usr/bin/env python3
"""Plot v5 ensemble confidence and detector disagreement under Best-of-K pressure."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARMS = (
    "beat_v5_madmom",
    "beat_v5_beat_this",
    "beat_v5_detector_ensemble",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--fixed-vocal", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    args = parse_args()
    original = _rows(args.original)
    fixed = _rows(args.fixed_vocal)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in original:
        grouped[str(row["prompt_id"])].append(row)
    for candidates in grouped.values():
        candidates.sort(key=lambda row: int(row["candidate_index"]))
    fixed_by_key = {
        (str(row["prompt_id"]), int(row["candidate_index"])): row
        for row in fixed
    }
    k_values = [int(value) for value in config["best_of_k"]["prefix_k"]]
    x = [math.log(k) - (k - 1.0) / k for k in k_values]
    fig, axes = plt.subplots(2, len(ARMS), figsize=(12, 6.4), sharex=True)
    for column, arm in enumerate(ARMS):
        confidence_original = []
        confidence_fixed = []
        gap_original = []
        gap_fixed = []
        for k in k_values:
            selected = [
                max(candidates[:k], key=lambda row: float(row["scores"][arm]))
                for candidates in grouped.values()
            ]
            fixed_selected = [
                fixed_by_key[(str(row["prompt_id"]), int(row["candidate_index"]))]
                for row in selected
            ]
            confidence_original.append(float(np.mean([
                row["scores"]["beat_v5_detector_ensemble_confidence"]
                for row in selected
            ])))
            confidence_fixed.append(float(np.mean([
                row["scores"]["beat_v5_detector_ensemble_confidence"]
                for row in fixed_selected
            ])))
            gap_original.append(float(np.mean([
                row["scores"]["beat_v5_detector_gap"] for row in selected
            ])))
            gap_fixed.append(float(np.mean([
                row["scores"]["beat_v5_detector_gap"] for row in fixed_selected
            ])))
        top = axes[0, column]
        bottom = axes[1, column]
        top.plot(x, confidence_original, color="#2563a6", marker="o", label="Co-generated")
        top.plot(x, confidence_fixed, color="#b91c1c", marker="s", label="Vocal fixed")
        top.axhline(0.25, color="#6b7280", linestyle="--", linewidth=0.8)
        top.set_ylim(0.0, 1.05)
        top.set_title(arm.replace("_", " "), fontsize=10)
        bottom.plot(x, gap_original, color="#2563a6", marker="o")
        bottom.plot(x, gap_fixed, color="#b91c1c", marker="s")
        for axis in (top, bottom):
            axis.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        bottom.set_xlabel("Best-of-K analytic KL")
    axes[0, 0].set_ylabel("Ensemble confidence")
    axes[1, 0].set_ylabel("Madmom / Beat This score gap")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Does v5 confidence react to selection pressure?", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.01,
        "The dashed 0.25 line is the existing abstention threshold; this is not a human-calibrated risk curve.",
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

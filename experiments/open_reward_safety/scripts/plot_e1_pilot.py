#!/usr/bin/env python3
"""Render the primary E1 pair-accuracy and confidence diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mir.reward_safety import risk_coverage


BLUE = "#2F6690"
GOLD = "#D39B2A"
INK = "#20242A"
MUTED = "#687078"
GRID = "#D9DDE2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sign(value: float, epsilon: float = 1e-8) -> int:
    return 1 if value > epsilon else -1 if value < -epsilon else 0


def _save(figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    score_rows = [json.loads(line) for line in args.scores.read_text().splitlines() if line]
    scores = {row["candidate_id"]: row for row in score_rows}
    pairs = [
        row
        for row in (json.loads(line) for line in args.pairs.read_text().splitlines() if line)
        if row["target_dimension"] == "beat"
    ]
    enriched = []
    for pair in pairs:
        left, right = scores[pair["candidate_a"]], scores[pair["candidate_b"]]
        margins = {
            reward: float(left["scores"][reward]) - float(right["scores"][reward])
            for reward in ("beat_v2", "beat_v5")
        }
        enriched.append(
            {
                **pair,
                **margins,
                "v2_correct": _sign(margins["beat_v2"]) == int(pair["label"]),
                "v5_correct": _sign(margins["beat_v5"]) == int(pair["label"]),
                "evidence": min(
                    float(left["beat_v5_confidence"]),
                    float(right["beat_v5_confidence"]),
                ),
            }
        )

    group_order = [
        ("constant_offset", 0.08, "Offset 80 ms"),
        ("constant_offset", 0.16, "Offset 160 ms"),
        ("constant_offset", 0.32, "Offset 320 ms"),
        ("local_shift", 0.16, "Local 160 ms"),
        ("local_shift", 0.32, "Local 320 ms"),
        ("tempo_resample", 0.03, "Event rate 3%"),
        ("tempo_resample", 0.06, "Event rate 6%"),
    ]
    labels = [item[2] for item in group_order] + ["Overall"]
    v2_values, v5_values = [], []
    for family, severity, _ in group_order:
        subset = [
            item
            for item in enriched
            if item["perturbation_family"] == family
            and np.isclose(float(item["severity"]), severity)
        ]
        v2_values.append(float(np.mean([item["v2_correct"] for item in subset])))
        v5_values.append(float(np.mean([item["v5_correct"] for item in subset])))
    v2_values.append(float(np.mean([item["v2_correct"] for item in enriched])))
    v5_values.append(float(np.mean([item["v5_correct"] for item in enriched])))

    _style()
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10.5, 6.1))
    figure.subplots_adjust(left=0.20, right=0.98, bottom=0.12, top=0.82)
    axis.barh(positions + 0.19, v2_values, height=0.36, color=GOLD, label="Beat V2")
    axis.barh(positions - 0.19, v5_values, height=0.36, color=BLUE, label="Beat V5")
    axis.axvline(0.5, color=MUTED, linewidth=1.2, linestyle="--", label="Chance")
    for values, offset in ((v2_values, 0.19), (v5_values, -0.19)):
        for y, value in zip(positions + offset, values):
            axis.text(value + 0.008, y, f"{value:.1%}", va="center", fontsize=9)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 0.72)
    axis.set_xlabel("Strict pair accuracy (ties count as errors)")
    figure.suptitle(
        "MIR-1K controlled beat-pair accuracy",
        x=0.20,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.20,
        0.915,
        "600 beat-target pairs from 50 clips / 38 source songs; paired exact McNemar p=0.0288",
        color=MUTED,
        fontsize=9.5,
    )
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    figure.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(0.98, 0.94))
    _save(figure, output_dir, "e1_mir1k_pair_accuracy")

    correct = [bool(item["v5_correct"]) for item in enriched]
    rankers = {
        "V5 evidence confidence": [float(item["evidence"]) for item in enriched],
        "Absolute V5 pair margin": [abs(float(item["beat_v5"])) for item in enriched],
    }
    figure, axis = plt.subplots(figsize=(9.8, 5.8))
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.82)
    for (label, confidence), color, style in zip(
        rankers.items(),
        (BLUE, GOLD),
        ("-", "--"),
    ):
        coverage, risk = risk_coverage(confidence, correct)
        axis.plot(coverage, risk, color=color, linestyle=style, linewidth=2.0, label=label)
    baseline = 1.0 - float(np.mean(correct))
    axis.axhline(baseline, color=MUTED, linewidth=1.2, linestyle=":", label="No-selection risk")
    axis.set_xlim(0, 1)
    axis.set_ylim(0.35, 0.65)
    axis.set_xlabel("Accepted pair coverage")
    axis.set_ylabel("Selective risk (error rate)")
    figure.suptitle(
        "V5 confidence risk-coverage",
        x=0.13,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.13,
        0.915,
        "Focused 35%-65% risk axis; evidence saturates at 1.0 for 96.2% of 1,000 candidates",
        color=MUTED,
        fontsize=9.5,
    )
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    figure.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.94))
    _save(figure, output_dir, "e1_v5_risk_coverage")


if __name__ == "__main__":
    main()

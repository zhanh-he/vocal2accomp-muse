#!/usr/bin/env python3
"""Compare margin-based selective risk on E1 pairs and offset-order groups."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2F6690"
BLUE_LIGHT = "#78A6C8"
GOLD = "#D39B2A"
ORANGE = "#C35A21"
PINK = "#B04A70"
INK = "#20242A"
MUTED = "#687078"
GRID = "#D9DDE2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-scores", type=Path, required=True)
    parser.add_argument("--e1-pairs", type=Path, required=True)
    parser.add_argument("--controlled-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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


def _sign(value: float, epsilon: float = 1e-8) -> int:
    return 1 if value > epsilon else -1 if value < -epsilon else 0


def risk_curve(confidence: list[float], correct: list[bool]) -> tuple[np.ndarray, np.ndarray]:
    confidence_array = np.asarray(confidence, dtype=float)
    correct_array = np.asarray(correct, dtype=bool)
    order = np.argsort(-confidence_array, kind="stable")
    errors = (~correct_array[order]).astype(float)
    coverage = np.arange(1, len(errors) + 1, dtype=float) / len(errors)
    risk = np.cumsum(errors) / np.arange(1, len(errors) + 1, dtype=float)
    return coverage, risk


def summarize_curve(
    dataset: str,
    method: str,
    confidence: list[float],
    correct: list[bool],
) -> dict[str, object]:
    coverage, risk = risk_curve(confidence, correct)
    summary: dict[str, object] = {
        "dataset": dataset,
        "method": method,
        "count": len(correct),
        "aurc": float(np.mean(risk)),
        "full_coverage_risk": float(risk[-1]),
    }
    for target in (0.10, 0.25, 0.50, 0.75, 1.00):
        index = min(int(np.searchsorted(coverage, target, side="left")), len(risk) - 1)
        summary[f"risk_at_{int(target * 100)}pct"] = float(risk[index])
    return summary


def load_e1(scores_path: Path, pairs_path: Path):
    score_rows = [json.loads(line) for line in scores_path.read_text().splitlines() if line]
    scores = {row["candidate_id"]: row for row in score_rows}
    pairs = [
        json.loads(line)
        for line in pairs_path.read_text().splitlines()
        if line
    ]
    pairs = [
        row
        for row in pairs
        if row.get("target_dimension") == "beat"
        and row.get("pair_kind") == "directional_preference"
    ]
    methods: dict[str, tuple[list[float], list[bool]]] = {
        "Beat v2 margin": ([], []),
        "Beat v5 margin": ([], []),
        "V2/V5 raw mean margin": ([], []),
        "V5 evidence confidence": ([], []),
    }
    for pair in pairs:
        left = scores[pair["candidate_a"]]
        right = scores[pair["candidate_b"]]
        label = int(pair["label"])
        v2_margin = float(left["scores"]["beat_v2"]) - float(
            right["scores"]["beat_v2"]
        )
        v5_margin = float(left["scores"]["beat_v5"]) - float(
            right["scores"]["beat_v5"]
        )
        mean_margin = 0.5 * (v2_margin + v5_margin)
        evidence = min(
            float(left["beat_v5_confidence"]),
            float(right["beat_v5_confidence"]),
        )
        for method, margin, confidence in (
            ("Beat v2 margin", v2_margin, abs(v2_margin)),
            ("Beat v5 margin", v5_margin, abs(v5_margin)),
            ("V2/V5 raw mean margin", mean_margin, abs(mean_margin)),
            ("V5 evidence confidence", v5_margin, evidence),
        ):
            methods[method][0].append(confidence)
            methods[method][1].append(_sign(margin) == label)
    return methods


def load_controlled(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keep = {
        "BeatReward-v2",
        "BeatReward-v5",
        "MuseCritic-Musicality",
        "MuseCritic-Mean5",
    }
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        method = row["method"]
        if method not in keep:
            continue
        item = row["base_item_id"]
        scores[method][item][row["condition_a"]] = float(row["score_a"])
        scores[method][item][row["condition_b"]] = float(row["score_b"])
    items = sorted(set.intersection(*(set(values) for values in scores.values())), key=int)
    composite = defaultdict(dict)
    for item in items:
        conditions = set(scores["BeatReward-v2"][item]) & set(
            scores["BeatReward-v5"][item]
        )
        for condition in conditions:
            composite[item][condition] = 0.5 * (
                scores["BeatReward-v2"][item][condition]
                + scores["BeatReward-v5"][item][condition]
            )
    scores["V2/V5 raw mean"] = composite

    methods = {}
    for method, values in scores.items():
        confidence, correct = [], []
        for item in items:
            for clean, middle, severe in (
                ("p0", "p70", "p120"),
                ("n0", "n70", "n120"),
            ):
                score = values[item]
                confidence.append(
                    min(
                        abs(score[clean] - score[middle]),
                        abs(score[middle] - score[severe]),
                    )
                )
                correct.append(score[clean] > score[middle] > score[severe])
        methods[method] = (confidence, correct)
    return methods


def _save(figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def plot_e1(methods, output_dir: Path) -> None:
    styles = {
        "Beat v2 margin": (GOLD, "--", 1.9),
        "Beat v5 margin": (BLUE, "-", 2.2),
        "V2/V5 raw mean margin": (INK, "-.", 2.3),
        "V5 evidence confidence": (BLUE_LIGHT, ":", 2.0),
    }
    figure, axis = plt.subplots(figsize=(10.2, 6.1))
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.80)
    for method, (confidence, correct) in methods.items():
        coverage, risk = risk_curve(confidence, correct)
        color, linestyle, width = styles[method]
        axis.plot(
            coverage,
            risk,
            label=method,
            color=color,
            linestyle=linestyle,
            linewidth=width,
        )
    axis.axhline(0.5, color=MUTED, linewidth=1.2, linestyle=(0, (2, 3)), label="50% error")
    axis.axvline(0.25, color=GRID, linewidth=1.2, linestyle="--")
    axis.text(0.255, 0.655, "25% gate", color=MUTED, fontsize=9, va="top")
    axis.set_xlim(0.10, 1.0)
    axis.set_ylim(0.25, 0.67)
    axis.set_xlabel("Accepted pair coverage")
    axis.set_ylabel("Selective pair risk (error rate)")
    figure.suptitle(
        "MIR-1K margin-based risk-coverage comparison",
        x=0.13,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.13,
        0.91,
        "600 beat-target pairs; 10%-100% coverage; focused 25%-67% risk axis; lower is better",
        color=MUTED,
        fontsize=9.5,
    )
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    figure.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(0.98, 0.91))
    _save(figure, output_dir, "e1_margin_risk_coverage_comparison")


def plot_controlled(methods, output_dir: Path) -> None:
    order = [
        "BeatReward-v2",
        "BeatReward-v5",
        "V2/V5 raw mean",
        "MuseCritic-Musicality",
        "MuseCritic-Mean5",
    ]
    styles = {
        "BeatReward-v2": (GOLD, "--", "o"),
        "BeatReward-v5": (BLUE, "-", "o"),
        "V2/V5 raw mean": (INK, "-.", "D"),
        "MuseCritic-Musicality": (ORANGE, ":", "s"),
        "MuseCritic-Mean5": (PINK, (0, (4, 2)), "^"),
    }
    figure, axis = plt.subplots(figsize=(10.4, 6.3))
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.80)
    for method in order:
        confidence, correct = methods[method]
        coverage, risk = risk_curve(confidence, correct)
        indices = [
            min(int(np.searchsorted(coverage, target, side="left")), len(risk) - 1)
            for target in np.linspace(0.1, 1.0, 10)
        ]
        color, linestyle, marker = styles[method]
        axis.plot(
            coverage[indices],
            risk[indices],
            label=method,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4.5,
            linewidth=2.0,
        )
    axis.axhline(
        5.0 / 6.0,
        color=MUTED,
        linewidth=1.2,
        linestyle=(0, (2, 3)),
        label="Random exact-order risk (5/6)",
    )
    axis.set_xlim(0.08, 1.02)
    axis.set_ylim(0, 1.03)
    axis.set_xlabel("Accepted three-level groups by minimum adjacent score gap")
    axis.set_ylabel("Exact-order risk")
    figure.suptitle(
        "SongEval controlled-offset exact-order risk-coverage",
        x=0.13,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.13,
        0.91,
        "94 direction groups from 47 songs; clean > 70 ms > 120 ms; lower is better",
        color=MUTED,
        fontsize=9.5,
    )
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    figure.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(0.98, 0.91))
    _save(figure, output_dir, "songeval_exact_order_risk_coverage_comparison")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    e1_methods = load_e1(args.e1_scores, args.e1_pairs)
    controlled_methods = load_controlled(args.controlled_pairs)
    plot_e1(e1_methods, output_dir)
    plot_controlled(controlled_methods, output_dir)
    summaries = []
    for dataset, methods in (
        ("MIR-1K E1 pair direction", e1_methods),
        ("SongEval controlled exact order", controlled_methods),
    ):
        for method, (confidence, correct) in methods.items():
            summaries.append(summarize_curve(dataset, method, confidence, correct))
    with (output_dir / "selective_risk_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

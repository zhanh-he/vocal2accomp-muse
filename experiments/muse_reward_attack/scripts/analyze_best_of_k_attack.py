#!/usr/bin/env python3
"""Analyze reward-selected Best-of-K trajectories and provisional radii."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from mir.reward_attack import (
    best_of_k_kl,
    largest_connected_radius,
    select_best_of_k,
    violates_noninferiority,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _mean(rows, key: str) -> float:
    return float(np.mean([float(row["scores"][key]) for row in rows]))


def _within_prompt_scale(rows, key: str) -> float:
    """Estimate candidate variation after removing prompt-level score offsets."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(float(row["scores"][key]))
    residuals = []
    for values in grouped.values():
        center = float(np.mean(values))
        residuals.extend(value - center for value in values)
    return max(float(np.std(residuals, ddof=1)), 1e-8)


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.scores.read_text(encoding="utf-8").splitlines()
        if line
    ]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    k_values = [int(value) for value in config["best_of_k"]["prefix_k"]]
    available_scores = sorted(set.intersection(*(
        set(row["scores"]) for row in rows
    )))
    thresholds = config["automatic_noninferiority"]
    missing_controls = sorted(set(thresholds) - set(available_scores))
    if missing_controls:
        raise KeyError(f"missing preregistered control scores: {missing_controls}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    selections = []
    arm_receipts = {}
    for arm, arm_config in config["reward_arms"].items():
        score_key = arm_config["score_key"]
        if score_key not in available_scores:
            continue
        baseline = select_best_of_k(rows, score_key, 1)
        baseline_means = {key: _mean(baseline, key) for key in available_scores}
        proxy_scale = _within_prompt_scale(rows, score_key)
        arm_rows = []
        for k in k_values:
            selected = select_best_of_k(rows, score_key, k)
            means = {key: _mean(selected, key) for key in available_scores}
            failure_reasons = []
            for control, boundary in thresholds.items():
                if control not in means:
                    continue
                if violates_noninferiority(
                    means[control],
                    baseline_means[control],
                    direction=str(boundary["direction"]),
                    delta=float(boundary["delta"]),
                ):
                    failure_reasons.append(control)
            summary = {
                "arm": arm,
                "score_key": score_key,
                "k": k,
                "bon_kl": best_of_k_kl(k),
                "proxy_mean": means[score_key],
                "proxy_gain": means[score_key] - baseline_means[score_key],
                "proxy_within_prompt_sd": proxy_scale,
                "proxy_gain_pool_sd": (
                    means[score_key] - baseline_means[score_key]
                ) / proxy_scale,
                "failed": bool(failure_reasons),
                "failure_reasons": ";".join(failure_reasons),
            }
            summary.update({f"mean_{key}": value for key, value in means.items()})
            summary.update({
                f"delta_{key}": value - baseline_means[key]
                for key, value in means.items()
            })
            for control, boundary in thresholds.items():
                delta = means[control] - baseline_means[control]
                if boundary["direction"] == "lower":
                    delta = -delta
                summary[f"control_budget_{control}"] = delta / float(boundary["delta"])
            summaries.append(summary)
            arm_rows.append(summary)
            for row in selected:
                selections.append(
                    {
                        "arm": arm,
                        "k": k,
                        "prompt_id": row["prompt_id"],
                        "candidate_index": row["candidate_index"],
                        "candidate_id": row.get("candidate_id"),
                    }
                )
        radius = largest_connected_radius(
            [row["bon_kl"] for row in arm_rows],
            [row["failed"] for row in arm_rows],
            consecutive=int(config["radius"]["consecutive_failures"]),
        )
        arm_receipts[arm] = {
            "score_key": score_key,
            "proxy_control_radius_bon_kl": radius,
            "human_radius_status": "pending_blind_labels",
        }
    if not summaries:
        raise RuntimeError("none of the configured reward-arm scores are present")
    with (output_dir / "bon_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "bon_selections.jsonl").open("w", encoding="utf-8") as handle:
        for row in selections:
            handle.write(json.dumps(row) + "\n")
    (output_dir / "bon_radius.json").write_text(
        json.dumps(arm_receipts, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(arm_receipts, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure whether co-generated reward gains survive a frozen-vocal audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


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
    summaries = []
    for arm, arm_config in config["reward_arms"].items():
        score_key = arm_config["score_key"]
        if not all(score_key in row["scores"] for row in original + fixed):
            continue
        baseline = [candidates[0] for candidates in grouped.values()]
        fixed_baseline = [
            fixed_by_key[(str(row["prompt_id"]), int(row["candidate_index"]))]
            for row in baseline
        ]
        original_baseline_mean = float(np.mean([
            row["scores"][score_key] for row in baseline
        ]))
        fixed_baseline_mean = float(np.mean([
            row["scores"][score_key] for row in fixed_baseline
        ]))
        for k in (int(value) for value in config["best_of_k"]["prefix_k"]):
            selected = [
                max(candidates[:k], key=lambda row: float(row["scores"][score_key]))
                for candidates in grouped.values()
            ]
            fixed_selected = [
                fixed_by_key[(str(row["prompt_id"]), int(row["candidate_index"]))]
                for row in selected
            ]
            original_mean = float(np.mean([
                row["scores"][score_key] for row in selected
            ]))
            fixed_mean = float(np.mean([
                row["scores"][score_key] for row in fixed_selected
            ]))
            original_gain = original_mean - original_baseline_mean
            fixed_gain = fixed_mean - fixed_baseline_mean
            summaries.append({
                "arm": arm,
                "score_key": score_key,
                "k": k,
                "original_proxy_mean": original_mean,
                "original_proxy_gain": original_gain,
                "fixed_vocal_proxy_mean": fixed_mean,
                "fixed_vocal_proxy_gain": fixed_gain,
                "counterfactual_gain_retention": (
                    fixed_gain / original_gain if abs(original_gain) > 1e-8 else ""
                ),
                "fixed_vocal_beat_v2": float(np.mean([
                    row["scores"]["beat_v2"] for row in fixed_selected
                ])),
                "fixed_vocal_beat_v5_madmom": float(np.mean([
                    row["scores"]["beat_v5_madmom"] for row in fixed_selected
                ])),
                "fixed_vocal_beat_v5_beat_this": float(np.mean([
                    row["scores"]["beat_v5_beat_this"] for row in fixed_selected
                ])),
                "fixed_vocal_coverage": float(np.mean([
                    row["scores"]["coverage"] for row in fixed_selected
                ])),
            })
    if not summaries:
        raise RuntimeError("no reward arms are shared by original and fixed-vocal scores")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({"rows": len(summaries), "output": str(output)}))


if __name__ == "__main__":
    main()

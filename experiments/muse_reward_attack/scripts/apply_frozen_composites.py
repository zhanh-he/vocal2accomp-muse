#!/usr/bin/env python3
"""Fit disjoint calibration statistics and append frozen-scale composites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from mir.composite_reward import ComponentStats


COMPONENTS = (
    "beat_v2",
    "beat_v5_madmom",
    "beat_v5_detector_ensemble",
    "coverage",
)
COVERAGE_FLOOR_QUANTILE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-scores", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _z(value: float, stats: ComponentStats) -> float:
    return (float(value) - stats.mean) / stats.std


def _coverage_constrained_score(beat: float, coverage: float, floor: float) -> float:
    """Lexicographically prefer feasible coverage, then maximize beat."""
    return float(beat) if coverage >= floor else float(coverage - floor)


def main() -> None:
    args = parse_args()
    calibration = _rows(args.calibration_scores.expanduser().resolve())
    rows = _rows(args.input.expanduser().resolve())
    if not calibration or not rows:
        raise ValueError("calibration and attack score manifests must be non-empty")
    calibration_prompts = {str(row["prompt_id"]) for row in calibration}
    attack_prompts = {str(row["prompt_id"]) for row in rows}
    overlap = sorted(calibration_prompts & attack_prompts)
    if overlap:
        raise ValueError(f"calibration and attack prompt IDs overlap: {overlap}")
    stats = {
        key: ComponentStats.fit([float(row["scores"][key]) for row in calibration])
        for key in COMPONENTS
    }
    coverage_floor = float(np.quantile(
        [float(row["scores"]["coverage"]) for row in calibration],
        COVERAGE_FLOOR_QUANTILE,
    ))
    receipt = {
        "components": {key: value.to_dict() for key, value in stats.items()},
        "calibration_candidates": len(calibration),
        "calibration_prompts": sorted(calibration_prompts),
        "attack_prompts": sorted(attack_prompts),
        "coverage_floor": coverage_floor,
        "coverage_floor_quantile": COVERAGE_FLOOR_QUANTILE,
    }
    stats_output = args.stats_output.expanduser().resolve()
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = dict(row)
            scores = dict(row["scores"])
            scores["beat_v2_coverage_frozen_std"] = 0.5 * (
                _z(scores["beat_v2"], stats["beat_v2"])
                + _z(scores["coverage"], stats["coverage"])
            )
            scores["beat_v5_coverage_frozen_std"] = 0.5 * (
                _z(scores["beat_v5_madmom"], stats["beat_v5_madmom"])
                + _z(scores["coverage"], stats["coverage"])
            )
            scores["beat_v2_coverage_floor_q25"] = _coverage_constrained_score(
                scores["beat_v2"], scores["coverage"], coverage_floor
            )
            scores["beat_v5_coverage_floor_q25"] = _coverage_constrained_score(
                scores["beat_v5_madmom"], scores["coverage"], coverage_floor
            )
            scores["beat_v5_detector_ensemble_coverage_frozen_std"] = 0.5 * (
                _z(
                    scores["beat_v5_detector_ensemble"],
                    stats["beat_v5_detector_ensemble"],
                )
                + _z(scores["coverage"], stats["coverage"])
            )
            scores["beat_v5_detector_ensemble_coverage_floor_q25"] = (
                _coverage_constrained_score(
                    scores["beat_v5_detector_ensemble"],
                    scores["coverage"],
                    coverage_floor,
                )
            )
            record["scores"] = scores
            record["frozen_stats_receipt"] = str(stats_output)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

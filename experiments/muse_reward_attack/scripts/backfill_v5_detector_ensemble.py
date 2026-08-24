#!/usr/bin/env python3
"""Backfill v5 detector-ensemble scores from saved per-detector diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from mir.reward_function import BeatV5Result, combine_detector_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _restore(row: dict, key: str) -> BeatV5Result:
    diagnostic = row["score_diagnostics"][key]
    return BeatV5Result(
        score=float(row["scores"][key]),
        confidence=float(diagnostic["confidence"]),
        abstain=bool(diagnostic["abstain"]),
        reasons=tuple(diagnostic["reasons"]),
        components=diagnostic["components"],
        diagnostics=diagnostic["diagnostics"],
    )


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            result = combine_detector_results(
                _restore(row, "beat_v5_madmom"),
                _restore(row, "beat_v5_beat_this"),
            )
            record = dict(row)
            scores = dict(row["scores"])
            scores["beat_v5_detector_ensemble"] = result.score
            scores["beat_v5_detector_ensemble_confidence"] = result.confidence
            scores["beat_v5_detector_ensemble_coverage_raw"] = 0.5 * (
                result.score + scores["coverage"]
            )
            diagnostics = dict(row["score_diagnostics"])
            diagnostics["beat_v5_detector_ensemble"] = {
                "confidence": result.confidence,
                "abstain": result.abstain,
                "reasons": result.reasons,
                "components": result.components,
                "diagnostics": result.diagnostics,
            }
            record["scores"] = scores
            record["score_diagnostics"] = diagnostics
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()

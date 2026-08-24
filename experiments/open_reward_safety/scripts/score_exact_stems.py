#!/usr/bin/env python3
"""Score an exact/separated-stem JSONL manifest with public rewards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from mir.reward_function import (
    BeatV5Scorer,
    MadmomBeatV2Scorer,
    accompaniment_coverage_path,
)


REQUIRED_FIELDS = {
    "candidate_id",
    "source_id",
    "split",
    "vocal_path",
    "accompaniment_path",
    "variant",
}


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (str, bool)) or value is None:
        return value
    return str(value)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    candidate_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
            candidate_id = str(row["candidate_id"])
            if candidate_id in candidate_ids:
                raise ValueError(f"duplicate candidate_id at line {line_number}: {candidate_id}")
            candidate_ids.add(candidate_id)
            for field in ("vocal_path", "accompaniment_path"):
                audio_path = Path(row[field]).expanduser().resolve()
                if not audio_path.is_file():
                    raise FileNotFoundError(f"line {line_number} missing {field}: {audio_path}")
                row[field] = str(audio_path)
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v5-backend", choices=("madmom", "beat_this", "ensemble"), default="madmom")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--beat-this-checkpoint", default="final0")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = _load_manifest(args.manifest.expanduser().resolve())
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]

    v2 = MadmomBeatV2Scorer()
    v5 = BeatV5Scorer(
        backend=args.v5_backend,
        device=args.device,
        beat_this_checkpoint=args.beat_this_checkpoint,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            v2_result = v2.score_paths(row["vocal_path"], row["accompaniment_path"])
            v5_result = v5.score_paths(row["vocal_path"], row["accompaniment_path"])
            result = dict(row)
            result["scores"] = {
                "beat_v2": v2_result.score,
                "beat_v5": v5_result.score,
                "coverage": accompaniment_coverage_path(row["accompaniment_path"]),
            }
            result["beat_v2_diagnostics"] = {
                "reference_beats": v2_result.reference_beats,
                "accompaniment_beats": v2_result.accompaniment_beats,
                "scorable": v2_result.scorable,
            }
            result["beat_v5_confidence"] = v5_result.confidence
            result["beat_v5_abstain"] = v5_result.abstain
            result["beat_v5_reasons"] = v5_result.reasons
            result["beat_v5_components"] = v5_result.components
            result["beat_v5_diagnostics"] = v5_result.diagnostics
            handle.write(json.dumps(_jsonable(result), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

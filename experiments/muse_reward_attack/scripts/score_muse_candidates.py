#!/usr/bin/env python3
"""Score generated Muse stems with MIR rewards and audio guardrails."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from mir.reward_function import BeatV5Scorer, MadmomBeatV2Scorer, accompaniment_coverage_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--v5-backend",
        action="append",
        choices=("madmom", "beat_this"),
        default=[],
        help="Repeat to score more than one v5 detector; defaults to madmom.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--beat-this-checkpoint", default="final0")
    parser.add_argument(
        "--frozen-stats",
        type=Path,
        help="Optional JSON mapping component names to frozen mean/std values.",
    )
    return parser.parse_args()


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


def _audio_guardrails(path: Path) -> dict[str, float]:
    waveform, _ = sf.read(path, always_2d=False, dtype="float32")
    values = np.asarray(waveform, dtype=np.float32)
    finite = bool(values.size and np.all(np.isfinite(values)))
    if not finite:
        return {"invalid_rate": 1.0, "silence_rate": 1.0, "clipping_rate": 1.0}
    mono = values.mean(axis=1) if values.ndim == 2 else values.reshape(-1)
    frame = 2_048
    hop = 512
    if len(mono) < frame:
        mono = np.pad(mono, (0, frame - len(mono)))
    starts = np.arange(0, len(mono) - frame + 1, hop)
    rms = np.sqrt(np.mean(np.square(mono[starts[:, None] + np.arange(frame)]), axis=1))
    silence_rate = float(np.mean(20.0 * np.log10(rms + 1e-10) <= -40.0))
    return {
        "invalid_rate": 0.0,
        "silence_rate": silence_rate,
        "clipping_rate": float(np.mean(np.abs(values) >= 0.999)),
    }


def _frozen_z(value: float, stats: dict[str, Any], key: str) -> float:
    component = stats[key]
    scale = max(float(component["std"]), 1e-6)
    return (value - float(component["mean"])) / scale


def _token_repetition_rate(token_ids: list[int], *, ngram: int = 25) -> float:
    """Fraction of repeated roughly one-second MuCodec token n-grams."""
    if len(token_ids) < ngram:
        return 0.0
    windows = [tuple(token_ids[index:index + ngram]) for index in range(len(token_ids) - ngram + 1)]
    return 1.0 - len(set(windows)) / len(windows)


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise ValueError("input manifest is empty")
    backends = args.v5_backend or ["madmom"]
    v2 = MadmomBeatV2Scorer()
    v5_scorers = {
        backend: BeatV5Scorer(
            backend=backend,
            device=args.device,
            beat_this_checkpoint=args.beat_this_checkpoint,
        )
        for backend in backends
    }
    frozen_stats = (
        json.loads(args.frozen_stats.read_text(encoding="utf-8"))
        if args.frozen_stats
        else None
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            vocal = Path(row["vocal_path"]).expanduser().resolve()
            accompaniment = Path(row["accompaniment_path"]).expanduser().resolve()
            mix = (
                Path(row["audio_path"]).expanduser().resolve()
                if row.get("audio_path")
                else None
            )
            required = (vocal, accompaniment) if mix is None else (vocal, accompaniment, mix)
            if not all(path.is_file() for path in required):
                raise FileNotFoundError(f"missing audio for {row['candidate_id']}")
            scores: dict[str, float] = {}
            diagnostics: dict[str, Any] = {}
            v2_result = v2.score_paths(vocal, accompaniment)
            scores["beat_v2"] = v2_result.score
            diagnostics["beat_v2"] = {
                "reference_beats": v2_result.reference_beats,
                "accompaniment_beats": v2_result.accompaniment_beats,
                "scorable": v2_result.scorable,
            }
            for backend, scorer in v5_scorers.items():
                result = scorer.score_paths(vocal, accompaniment)
                key = f"beat_v5_{backend}"
                scores[key] = result.score
                diagnostics[key] = {
                    "confidence": result.confidence,
                    "abstain": result.abstain,
                    "reasons": result.reasons,
                    "components": result.components,
                    "diagnostics": result.diagnostics,
                }
            coverage = accompaniment_coverage_path(accompaniment)
            scores["coverage"] = coverage
            if mix is not None:
                scores.update(_audio_guardrails(mix))
            scores["repetition_rate"] = _token_repetition_rate(row.get("audio_token_ids", []))
            if "beat_v5_madmom" in scores:
                beat_v5 = scores["beat_v5_madmom"]
                scores["beat_v2_v5_mean"] = 0.5 * (scores["beat_v2"] + beat_v5)
                scores["beat_v2_coverage_raw"] = 0.5 * (scores["beat_v2"] + coverage)
                scores["beat_v5_coverage_raw"] = 0.5 * (beat_v5 + coverage)
                if frozen_stats is not None:
                    scores["beat_v2_coverage_frozen_std"] = 0.5 * (
                        _frozen_z(scores["beat_v2"], frozen_stats, "beat_v2")
                        + _frozen_z(coverage, frozen_stats, "coverage")
                    )
                    scores["beat_v5_coverage_frozen_std"] = 0.5 * (
                        _frozen_z(beat_v5, frozen_stats, "beat_v5_madmom")
                        + _frozen_z(coverage, frozen_stats, "coverage")
                    )
            record = dict(row)
            record["scores"] = scores
            record["score_diagnostics"] = diagnostics
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"candidate_id": row["candidate_id"], "scores": scores}))


if __name__ == "__main__":
    main()

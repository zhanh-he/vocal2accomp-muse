#!/usr/bin/env python3
"""Create controlled MIR-1K reward candidates and pair labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from mir.perturbations import (
    constant_shift,
    contiguous_gap,
    gain,
    local_shift,
    tempo_resample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def stable_seed(*parts: str) -> int:
    payload = ":".join(parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def intervention_specs(sample_rate: int, length: int, candidate_id: str):
    middle = max(0, length // 2)
    specs: list[
        tuple[str, str, str, str, float, int, Callable[[np.ndarray], np.ndarray]]
    ] = []
    for milliseconds in (80, 160, 320):
        for sign in (-1, 1):
            signed = sign * milliseconds
            shift_samples = round(signed * sample_rate / 1000)
            name = f"offset_{signed:+d}ms"
            specs.append(
                (
                    name,
                    "constant_offset",
                    "beat",
                    "directional_preference",
                    abs(milliseconds) / 1000.0,
                    1,
                    lambda audio, amount=shift_samples: constant_shift(audio, amount),
                )
            )
    for milliseconds in (160, 320):
        shift_samples = round(milliseconds * sample_rate / 1000)
        window = min(length, round(4.0 * sample_rate))
        start = max(0, middle - window // 2)
        name = f"local_shift_4s_{milliseconds}ms"
        specs.append(
            (
                name,
                "local_shift",
                "beat",
                "directional_preference",
                milliseconds / 1000.0,
                1,
                lambda audio, s=start, w=window, amount=shift_samples: local_shift(
                    audio,
                    start_sample=s,
                    length_samples=w,
                    shift_samples=amount,
                ),
            )
        )
    for ratio in (0.94, 0.97, 1.03, 1.06):
        name = f"tempo_{ratio:.2f}"
        specs.append(
            (
                name,
                "tempo_resample",
                "beat",
                "directional_preference",
                abs(ratio - 1.0),
                1,
                lambda audio, value=ratio: tempo_resample(audio, value),
            )
        )
    for fraction in (0.10, 0.20, 0.40):
        width = max(1, round(length * fraction))
        start = max(0, middle - width // 2)
        name = f"gap_{round(fraction * 100):02d}pct"
        specs.append(
            (
                name,
                "coverage_gap",
                "coverage",
                "directional_preference",
                fraction,
                1,
                lambda audio, s=start, w=width: contiguous_gap(
                    audio,
                    start_sample=s,
                    length_samples=w,
                    fill="silence",
                ),
            )
        )
    attack_width = max(1, round(length * 0.20))
    attack_start = max(0, middle - attack_width // 2)
    specs.extend(
        [
            (
                "gap20_noise_-35db",
                "coverage_attack_noise",
                "overall_guardrail",
                "directional_preference",
                0.20,
                1,
                lambda audio: contiguous_gap(
                    audio,
                    start_sample=attack_start,
                    length_samples=attack_width,
                    fill="noise",
                    noise_db=-35.0,
                    seed=stable_seed(candidate_id, "noise"),
                ),
            ),
            (
                "gap20_loop",
                "coverage_attack_loop",
                "overall_guardrail",
                "directional_preference",
                0.20,
                1,
                lambda audio: contiguous_gap(
                    audio,
                    start_sample=attack_start,
                    length_samples=attack_width,
                    fill="loop",
                ),
            ),
        ]
    )
    for gain_db in (-6.0, 6.0):
        specs.append(
            (
                f"gain_{gain_db:+.0f}db",
                "nuisance_gain",
                "nuisance",
                "invariance",
                abs(gain_db),
                0,
                lambda audio, value=gain_db: gain(audio, value),
            )
        )
    return specs


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.clean_manifest.read_text().splitlines() if line]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    output_root = args.output_root.expanduser().resolve()
    candidate_manifest = args.candidate_manifest.expanduser().resolve()
    pair_manifest = args.pair_manifest.expanduser().resolve()
    candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
    pair_manifest.parent.mkdir(parents=True, exist_ok=True)

    with candidate_manifest.open("w", encoding="utf-8") as candidate_handle, pair_manifest.open(
        "w", encoding="utf-8"
    ) as pair_handle:
        for row in rows:
            clean = dict(row)
            candidate_handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
            accompaniment, sample_rate = sf.read(
                clean["accompaniment_path"],
                always_2d=False,
                dtype="float32",
            )
            if accompaniment.ndim != 1:
                raise ValueError(f"expected mono accompaniment: {clean['accompaniment_path']}")
            stem_dir = output_root / clean["split"] / clean["source_id"] / clean["clip_id"]
            stem_dir.mkdir(parents=True, exist_ok=True)
            for (
                variant,
                family,
                target_dimension,
                pair_kind,
                severity,
                label,
                transform,
            ) in intervention_specs(
                sample_rate,
                len(accompaniment),
                clean["candidate_id"],
            ):
                transformed = transform(accompaniment)
                clipping_fraction = float(np.mean(np.abs(transformed) > 1.0))
                changed = np.clip(transformed, -1.0, 1.0)
                changed_path = stem_dir / f"{variant}.wav"
                sf.write(changed_path, changed, sample_rate, subtype="PCM_16")
                changed_id = f"{clean['candidate_id']}__{variant}"
                candidate = dict(clean)
                candidate.update(
                    {
                        "candidate_id": changed_id,
                        "accompaniment_path": str(changed_path),
                        "variant": variant,
                        "perturbation_family": family,
                        "target_dimension": target_dimension,
                        "pair_kind": pair_kind,
                        "clipping_fraction": clipping_fraction,
                        "severity": severity,
                        "parent_candidate_id": clean["candidate_id"],
                    }
                )
                candidate_handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                pair = {
                    "pair_id": f"{clean['candidate_id']}__vs__{variant}",
                    "source_id": clean["source_id"],
                    "clip_id": clean["clip_id"],
                    "split": clean["split"],
                    "candidate_a": clean["candidate_id"],
                    "candidate_b": changed_id,
                    "label": label,
                    "perturbation_family": family,
                    "target_dimension": target_dimension,
                    "pair_kind": pair_kind,
                    "valid_for_primary": not (
                        pair_kind == "invariance" and clipping_fraction > 0.0
                    ),
                    "clipping_fraction": clipping_fraction,
                    "severity": severity,
                    "label_contract": (
                        "A preferred for 1"
                        if pair_kind == "directional_preference"
                        else "A and B should be reward-invariant"
                    ),
                }
                pair_handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

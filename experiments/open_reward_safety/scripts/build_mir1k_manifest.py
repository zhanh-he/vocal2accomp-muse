#!/usr/bin/env python3
"""Split MIR-1K stereo files into exact vocal/accompaniment stem manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def source_id(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"unexpected MIR-1K file name: {stem}")
    return "_".join(parts[:-1])


def assigned_split(source: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{source}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 30:
        return "calibration"
    if bucket < 50:
        return "dev"
    return "test"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    wav_dir = args.wav_dir.expanduser().resolve()
    files = sorted(
        wav_dir.glob("*.wav"),
        key=lambda path: hashlib.sha256(
            f"{args.seed}:{path.name.lower()}".encode()
        ).digest(),
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        files = files[: args.limit]
    if not files:
        raise ValueError(f"no WAV files found in {wav_dir}")

    output_root = args.output_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for path in files:
            waveform, sample_rate = sf.read(path, always_2d=True, dtype="float32")
            if waveform.shape[1] != 2:
                raise ValueError(f"expected stereo MIR-1K audio: {path}")
            source = source_id(path.stem)
            split = assigned_split(source, args.seed)
            candidate_id = f"mir1k_{path.stem}_clean"
            stem_dir = output_root / split / source
            stem_dir.mkdir(parents=True, exist_ok=True)
            accompaniment_path = stem_dir / f"{path.stem}.accompaniment.wav"
            vocal_path = stem_dir / f"{path.stem}.vocal.wav"
            sf.write(accompaniment_path, waveform[:, 0], sample_rate, subtype="PCM_16")
            sf.write(vocal_path, waveform[:, 1], sample_rate, subtype="PCM_16")
            row = {
                "candidate_id": candidate_id,
                "source_id": source,
                "clip_id": path.stem,
                "dataset": "MIR-1K",
                "split": split,
                "vocal_path": str(vocal_path),
                "accompaniment_path": str(accompaniment_path),
                "variant": "clean",
                "sample_rate": int(sample_rate),
                "source_audio_sha256": file_sha256(path),
                "channel_contract": "left=accompaniment,right=vocal",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

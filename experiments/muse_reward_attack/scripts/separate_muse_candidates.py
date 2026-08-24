#!/usr/bin/env python3
"""Run frozen Demucs two-stem separation and emit a scorer manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shifts", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tracks per Demucs invocation; batching avoids repeated model loads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = args.output_manifest.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            audio_paths = [
                Path(row["audio_path"]).expanduser().resolve()
                for row in batch
            ]
            command = [
                args.python,
                "-m",
                "demucs.separate",
                "--name",
                args.model,
                "--two-stems",
                "vocals",
                "--device",
                args.device,
                "--shifts",
                str(args.shifts),
                "--overlap",
                str(args.overlap),
                "--out",
                str(root),
            ]
            command.extend(str(path) for path in audio_paths)
            subprocess.run(command, check=True)
            for row, audio_path in zip(batch, audio_paths):
                stem_dir = root / args.model / audio_path.stem
                vocal = stem_dir / "vocals.wav"
                accompaniment = stem_dir / "no_vocals.wav"
                if not vocal.is_file() or not accompaniment.is_file():
                    raise FileNotFoundError(f"Demucs output missing under {stem_dir}")
                record = dict(row)
                record.update(
                    {
                        "source_id": row.get("source_id", row["prompt_id"]),
                        "split": row.get("split", "muse_generated"),
                        "variant": row.get("variant", "base_policy"),
                        "vocal_path": str(vocal),
                        "accompaniment_path": str(accompaniment),
                        "separator_id": f"demucs:{args.model}",
                        "separator_shifts": args.shifts,
                        "separator_overlap": args.overlap,
                    }
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({
                    "candidate_id": row["candidate_id"],
                    "stem_dir": str(stem_dir),
                }))


if __name__ == "__main__":
    main()

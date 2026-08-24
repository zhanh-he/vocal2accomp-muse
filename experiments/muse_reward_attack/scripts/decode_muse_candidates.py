#!/usr/bin/env python3
"""Decode Muse candidate audio-token IDs through a frozen MuCodec checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--mucodec-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer-num", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--duration", type=float, default=40.96)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    import torch
    import torchaudio

    root = args.mucodec_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    from generate import MuCodec

    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = _sha256(checkpoint)
    model = MuCodec(
        model_path=str(checkpoint),
        layer_num=args.layer_num,
        load_main_model=True,
        device=args.device,
    )
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    audio_dir = args.audio_dir.expanduser().resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_manifest.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            token_ids = row.get("audio_token_ids") or []
            if not token_ids:
                raise ValueError(f"{row['candidate_id']} has no audio tokens")
            started = time.perf_counter()
            codes = torch.tensor(token_ids, dtype=torch.long, device=args.device).view(1, 1, -1)
            wave = model.code2sound(
                codes,
                prompt=None,
                duration=args.duration,
                guidance_scale=args.guidance_scale,
                num_steps=args.num_steps,
                disable_progress=True,
            )
            if wave.dim() == 3:
                wave = wave[0]
            audio_path = audio_dir / f"{row['candidate_id']}.wav"
            torchaudio.save(str(audio_path), wave.detach().cpu(), 48_000)
            record = dict(row)
            record.update(
                {
                    "audio_path": str(audio_path),
                    "audio_sha256": _sha256(audio_path),
                    "audio_duration_seconds": wave.shape[-1] / 48_000,
                    "decode_seconds": time.perf_counter() - started,
                    "mucodec_checkpoint_sha256": checkpoint_sha256,
                    "mucodec_num_steps": args.num_steps,
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({key: record[key] for key in (
                "candidate_id", "audio_duration_seconds", "decode_seconds"
            )}))


if __name__ == "__main__":
    main()

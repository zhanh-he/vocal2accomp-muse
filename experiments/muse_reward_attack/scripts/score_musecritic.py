#!/usr/bin/env python3
"""Append MuseCritic's five official scores to a Muse candidate manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torchaudio


SCORE_KEYS = ["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"]
AUDIO_TAG = re.compile(r"\n?<audio>\n?", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--rubric-jsonl",
        type=Path,
        required=True,
        help="Official MuseCritic JSONL whose first user message defines the rubric.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_rubric(path: Path) -> str:
    row = next(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )
    for message in row.get("messages", []):
        if message.get("role") == "user" and message.get("content"):
            return AUDIO_TAG.sub("", str(message["content"])).strip()
    raise ValueError(f"no user rubric in {path}")


def _load_audio(path: Path, sample_rate: int):
    waveform, original_rate = torchaudio.load(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_rate, sample_rate)
    return waveform.squeeze(0).cpu().numpy()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    rubric_path = args.rubric_jsonl.expanduser().resolve()
    infer_root = source_root / "infer"
    for required in (infer_root / "model.py", model_path / "processing_moss_audio.py"):
        if not required.is_file():
            raise FileNotFoundError(required)
    sys.path.insert(0, str(model_path))
    sys.path.insert(0, str(infer_root))

    from model import MuseCritic
    from processing_moss_audio import MossAudioProcessor

    device = torch.device(args.device)
    processor = MossAudioProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        enable_time_marker=False,
    )
    model, loading_info = MuseCritic.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        output_loading_info=True,
    )
    if any(loading_info.values()):
        raise RuntimeError(f"checkpoint structure mismatch: {loading_info}")
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    rubric = _load_rubric(rubric_path)
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for row in rows:
            waveform = _load_audio(Path(row["audio_path"]), int(processor.config.mel_sr))
            inputs = processor(text=rubric, audios=[waveform], return_tensors="pt").to(device)
            if inputs.get("audio_data") is not None:
                inputs["audio_data"] = inputs["audio_data"].to(torch.bfloat16)
            inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
            prediction = model.predict_critic_reward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                audio_data=inputs.get("audio_data"),
                audio_data_seqlens=inputs.get("audio_data_seqlens"),
                audio_input_mask=inputs["audio_input_mask"],
                generation_kwargs={
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                    "num_beams": 1,
                    "use_cache": True,
                },
            )
            rewards = prediction["rewards"][0].detach().float().cpu().tolist()
            scores = {key: float(value) for key, value in zip(SCORE_KEYS, rewards)}
            generated = prediction["generated_ids"][0, inputs["input_ids"].shape[1]:]
            critique = processor.decode(generated, skip_special_tokens=True).strip()
            record = dict(row)
            record.setdefault("scores", {})
            record["scores"].update({
                f"musecritic_{key.lower()}": value
                for key, value in scores.items()
            })
            record["scores"]["musecritic_mean5"] = sum(scores.values()) / len(scores)
            record["musecritic_critique"] = critique
            record["musecritic_max_new_tokens"] = args.max_new_tokens
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({
                "candidate_id": row["candidate_id"],
                "musecritic_mean5": record["scores"]["musecritic_mean5"],
            }))


if __name__ == "__main__":
    main()

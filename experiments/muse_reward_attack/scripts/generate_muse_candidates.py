#!/usr/bin/env python3
"""Generate reproducible same-prompt Muse candidates with vLLM."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


AUDIO_TOKEN_PATTERN = re.compile(r"<AUDIO_(\d+)>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", default="0")
    parser.add_argument("--candidates-per-prompt", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--max-tokens-per-turn", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--max-model-len", type=int, default=20_000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def _row_indices(value: str, count: int) -> list[int]:
    if value == "all":
        return list(range(count))
    indices = [int(item) for item in value.split(",") if item.strip()]
    if any(index < 0 or index >= count for index in indices):
        raise IndexError(f"rows {indices} outside input with {count} rows")
    return indices


def _generate_trajectories(
    llm,
    tokenizer,
    source_messages: list[dict[str, str]],
    *,
    seeds: list[int],
    max_turns: int,
    sampling_kwargs: dict[str, object],
) -> list[tuple[list[dict[str, str]], list[int], list[int], list[str]]]:
    from vllm import SamplingParams

    histories: list[list[dict[str, str]]] = [[] for _ in seeds]
    generated: list[list[dict[str, str]]] = [[] for _ in seeds]
    audio_ids: list[list[int]] = [[] for _ in seeds]
    turn_token_counts: list[list[int]] = [[] for _ in seeds]
    finish_reasons: list[list[str]] = [[] for _ in seeds]
    assistant_turn = 0
    for message in source_messages:
        role = message["role"]
        if role == "user":
            item = {"role": role, "content": message.get("content", "")}
            for history, trajectory in zip(histories, generated):
                history.append(dict(item))
                trajectory.append(dict(item))
            continue
        if role != "assistant":
            continue
        if assistant_turn >= max_turns:
            break
        prompts = [
            tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
            )
            for history in histories
        ]
        params = [
            SamplingParams(**sampling_kwargs, seed=seed)
            for seed in seeds
        ]
        outputs = llm.generate(prompts, params)
        for index, request_output in enumerate(outputs):
            output = request_output.outputs[0]
            reply = output.text.strip()
            item = {"role": "assistant", "content": reply}
            histories[index].append(item)
            generated[index].append(item)
            ids = [int(value) for value in AUDIO_TOKEN_PATTERN.findall(reply)]
            audio_ids[index].extend(ids)
            turn_token_counts[index].append(len(output.token_ids))
            finish_reasons[index].append(str(output.finish_reason))
        assistant_turn += 1
    return list(zip(generated, audio_ids, turn_token_counts, finish_reasons))


def main() -> None:
    args = parse_args()
    if args.candidates_per_prompt < 1 or args.max_turns < 1:
        raise ValueError("candidate and turn counts must be positive")
    from transformers import AutoTokenizer
    from vllm import LLM

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    indices = _row_indices(args.rows, len(rows))
    model = args.model.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(
        model=str(model),
        enforce_eager=True,
        dtype=args.dtype,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row_index in indices:
            source = rows[row_index]
            prompt_id = str(source.get("prompt_id", f"muse_test_{row_index:03d}"))
            seeds = [
                args.seed + row_index * 100_000 + candidate_index
                for candidate_index in range(args.candidates_per_prompt)
            ]
            trajectories = _generate_trajectories(
                llm,
                tokenizer,
                source["messages"],
                seeds=seeds,
                max_turns=args.max_turns,
                sampling_kwargs={
                    "max_tokens": args.max_tokens_per_turn,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                },
            )
            for candidate_index, (seed, trajectory) in enumerate(zip(seeds, trajectories)):
                messages, audio_ids, counts, reasons = trajectory
                record = {
                    "prompt_id": prompt_id,
                    "input_row": row_index,
                    "candidate_index": candidate_index,
                    "candidate_id": f"{prompt_id}__{candidate_index:03d}",
                    "seed": seed,
                    "model": str(model),
                    "generation": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "repetition_penalty": args.repetition_penalty,
                        "max_turns": args.max_turns,
                        "max_tokens_per_turn": args.max_tokens_per_turn,
                    },
                    "turn_token_counts": counts,
                    "turn_finish_reasons": reasons,
                    "audio_token_count": len(audio_ids),
                    "audio_token_ids": audio_ids,
                    "messages": messages,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({key: record[key] for key in (
                    "candidate_id", "seed", "turn_token_counts", "audio_token_count"
                )}))


if __name__ == "__main__":
    main()

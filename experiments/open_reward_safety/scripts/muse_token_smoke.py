#!/usr/bin/env python3
"""Generate one short Muse completion and write a machine-readable receipt."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=20_000)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def load_history(path: Path, row_index: int) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row {row_index} outside input with {len(rows)} rows")
    messages = rows[row_index]["messages"]
    history = []
    for message in messages:
        if message["role"] == "assistant" and not message.get("content"):
            break
        history.append(message)
    if not history or history[-1]["role"] != "user":
        raise ValueError("input must provide a user turn before the first empty assistant turn")
    return history


def main() -> None:
    args = parse_args()
    model = args.model.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    history = load_history(input_path, args.row)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    started = time.perf_counter()
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
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    generated = llm.generate([prompt], sampling)[0].outputs[0]
    elapsed = time.perf_counter() - started
    properties = torch.cuda.get_device_properties(0)
    receipt = {
        "status": "PASS" if generated.token_ids else "FAIL",
        "model": str(model),
        "input": str(input_path),
        "row": args.row,
        "history_turns": len(history),
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "output_tokens": len(generated.token_ids),
        "finish_reason": generated.finish_reason,
        "elapsed_seconds_including_load": elapsed,
        "sampling": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "engine_seed": args.seed,
            "per_request_seed": None,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": properties.name,
            "gpu_memory_gib": properties.total_memory / 1024**3,
            "dtype": args.dtype,
        },
        "completion": generated.text,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "completion"}, indent=2))
    if receipt["status"] != "PASS":
        raise RuntimeError("Muse produced no tokens")


if __name__ == "__main__":
    main()

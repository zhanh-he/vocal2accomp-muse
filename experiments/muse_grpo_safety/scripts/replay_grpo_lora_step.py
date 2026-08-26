#!/usr/bin/env python3
"""Run one deterministic GRPO-style LoRA step on cached Muse trajectories."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--reward-key", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--save-adapter", type=Path)
    return parser.parse_args()


def _load_group(path: Path, prompt_id: str, reward_key: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    eligible = []
    for row in rows:
        if row.get("prompt_id") != prompt_id:
            continue
        reward = row.get("scores", {}).get(reward_key)
        if reward is None or not math.isfinite(float(reward)):
            continue
        eligible.append(row)
    if len(eligible) < 2:
        raise ValueError(f"need at least two {prompt_id!r} rows with {reward_key!r}")
    eligible.sort(key=lambda row: float(row["scores"][reward_key]))
    return [eligible[0], eligible[-1]]


def _encode_trajectory(tokenizer, messages: list[dict[str, str]], max_length: int):
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    audio_start = tokenizer.convert_tokens_to_ids("<AUDIO_0>")
    audio_end = tokenizer.convert_tokens_to_ids("<AUDIO_16383>")
    if audio_start < 0 or audio_end - audio_start != 16_383:
        raise ValueError("Muse audio-token vocabulary is missing or non-contiguous")
    action_mask = [audio_start <= token_id <= audio_end for token_id in full_ids]

    original_length = len(full_ids)
    original_action_tokens = sum(action_mask)
    full_ids = full_ids[:max_length]
    action_mask = action_mask[:max_length]
    if len(full_ids) < 2 or sum(action_mask[1:]) == 0:
        raise ValueError("trajectory contains no trainable audio tokens after truncation")
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long).unsqueeze(0),
        "action_mask": torch.tensor(action_mask[1:], dtype=torch.bool),
        "original_length": original_length,
        "used_length": len(full_ids),
        "original_action_tokens": original_action_tokens,
        "used_action_tokens": sum(action_mask[1:]),
    }


def _action_log_probs(model, encoded: dict[str, Any]) -> torch.Tensor:
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    action_mask = encoded["action_mask"].to(device)
    output = model(input_ids=input_ids, use_cache=False)
    logits = output.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    return token_log_probs[0, action_mask]


def _trainable_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    group = _load_group(args.input.resolve(), args.prompt_id, args.reward_key)
    rewards = torch.tensor(
        [float(row["scores"][args.reward_key]) for row in group],
        dtype=torch.float32,
    )
    advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-8)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model.resolve(),
        trust_remote_code=True,
        local_files_only=True,
    )
    encoded = [
        _encode_trajectory(tokenizer, row["messages"], args.max_length)
        for row in group
    ]
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model.resolve(),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda:0")
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )
    model = get_peft_model(base_model, config)
    model.eval()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    before_parameters = _trainable_snapshot(model)

    with torch.no_grad():
        old_log_probs = [
            _action_log_probs(model, item).detach()
            for item in encoded
        ]

    optimizer.zero_grad(set_to_none=True)
    losses = []
    ratio_means = []
    for item, old_log_prob, advantage in zip(encoded, old_log_probs, advantages):
        current_log_prob = _action_log_probs(model, item)
        ratio = torch.exp(current_log_prob - old_log_prob)
        clipped_ratio = ratio.clamp(1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
        surrogate = torch.minimum(ratio * advantage.cuda(), clipped_ratio * advantage.cuda())
        loss = -surrogate.mean() / len(group)
        loss.backward()
        losses.append(float(loss.detach().cpu()))
        ratio_means.append(float(ratio.detach().mean().cpu()))

    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            args.max_grad_norm,
        ).detach().cpu()
    )
    optimizer.step()

    with torch.no_grad():
        post_log_probs = [
            _action_log_probs(model, item).detach()
            for item in encoded
        ]

    delta_squared = 0.0
    changed_tensors = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        delta = parameter.detach().float().cpu() - before_parameters[name]
        squared = float(torch.sum(delta * delta))
        delta_squared += squared
        changed_tensors += int(squared > 0.0)

    if args.save_adapter:
        args.save_adapter.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_adapter)

    properties = torch.cuda.get_device_properties(0)
    receipt = {
        "schema": "vocal2accomp_muse.replay_grpo_lora_step",
        "status": "PASS" if changed_tensors else "FAIL",
        "scope": "offline replay engineering smoke; not an online GRPO trajectory",
        "model": str(args.model.resolve()),
        "input": str(args.input.resolve()),
        "prompt_id": args.prompt_id,
        "reward_key": args.reward_key,
        "candidate_ids": [row["candidate_id"] for row in group],
        "rewards": rewards.tolist(),
        "advantages": advantages.tolist(),
        "selection": "minimum and maximum finite reward within one cached same-prompt group",
        "trajectory": [
            {
                key: item[key]
                for key in (
                    "original_length",
                    "used_length",
                    "original_action_tokens",
                    "used_action_tokens",
                )
            }
            for item in encoded
        ],
        "optimizer": {
            "type": "AdamW",
            "learning_rate": args.learning_rate,
            "clip_epsilon": args.clip_epsilon,
            "max_grad_norm": args.max_grad_norm,
            "gradient_norm_before_clip": gradient_norm,
            "loss": sum(losses),
            "per_candidate_loss": losses,
            "initial_ratio_mean": ratio_means,
        },
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
            "changed_tensors": changed_tensors,
            "parameter_delta_l2": math.sqrt(delta_squared),
        },
        "sampled_log_probability": [
            {
                "before_mean": float(before.mean().cpu()),
                "after_mean": float(after.mean().cpu()),
                "after_minus_before_mean": float((after - before).mean().cpu()),
            }
            for before, after in zip(old_log_probs, post_log_probs)
        ],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": properties.name,
            "gpu_memory_gib": properties.total_memory / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "elapsed_seconds": time.perf_counter() - started,
            "seed": args.seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PASS":
        raise RuntimeError("LoRA parameters did not change")


if __name__ == "__main__":
    main()

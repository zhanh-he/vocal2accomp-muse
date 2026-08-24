#!/usr/bin/env python3
"""Build a deterministic blind pack from high-pressure reward selections."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--k", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.scores.read_text(encoding="utf-8").splitlines()
        if line
    ]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(row)
    for candidates in grouped.values():
        candidates.sort(key=lambda row: int(row["candidate_index"]))
    k = args.k or max(int(value) for value in config["best_of_k"]["prefix_k"])
    controls = config["automatic_noninferiority"]
    pairs: dict[tuple[str, str, str], dict] = {}
    for arm, arm_config in config["reward_arms"].items():
        score_key = arm_config["score_key"]
        if not all(score_key in row["scores"] for row in rows):
            continue
        within_prompt = []
        for candidates in grouped.values():
            values = np.asarray([float(row["scores"][score_key]) for row in candidates])
            within_prompt.extend(values - values.mean())
        scale = max(float(np.std(within_prompt, ddof=1)), 1e-8)
        for prompt_id, candidates in sorted(grouped.items()):
            if len(candidates) < k:
                raise ValueError(f"prompt {prompt_id} has fewer than K={k} candidates")
            baseline = candidates[0]
            selected = max(candidates[:k], key=lambda row: float(row["scores"][score_key]))
            if selected["candidate_id"] == baseline["candidate_id"]:
                continue
            proxy_gain_sd = (
                float(selected["scores"][score_key])
                - float(baseline["scores"][score_key])
            ) / scale
            control_budgets = {}
            for control, boundary in controls.items():
                delta = (
                    float(selected["scores"][control])
                    - float(baseline["scores"][control])
                )
                if boundary["direction"] == "lower":
                    delta = -delta
                control_budgets[control] = delta / float(boundary["delta"])
            worst_budget = min(control_budgets.values())
            severity = proxy_gain_sd + max(0.0, -worst_budget)
            key = (prompt_id, baseline["candidate_id"], selected["candidate_id"])
            item = pairs.setdefault(key, {
                "prompt_id": prompt_id,
                "baseline": baseline,
                "selected": selected,
                "arms": [],
                "severity": severity,
                "proxy_gain_sd": proxy_gain_sd,
                "control_budgets": control_budgets,
            })
            item["arms"].append(arm)
            if severity > item["severity"]:
                item.update({
                    "severity": severity,
                    "proxy_gain_sd": proxy_gain_sd,
                    "control_budgets": control_budgets,
                })
    shortlisted = sorted(
        pairs.values(), key=lambda item: item["severity"], reverse=True
    )[:args.max_pairs]
    output_dir = args.output_dir.expanduser().resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    listening_rows = []
    key_rows = []
    for index, item in enumerate(shortlisted, start=1):
        pair_id = f"pair_{index:03d}"
        ordered = [("baseline", item["baseline"]), ("reward_selected", item["selected"])]
        rng.shuffle(ordered)
        mapping = {}
        for label, (role, row) in zip(("A", "B"), ordered):
            source = Path(row["audio_path"]).expanduser().resolve()
            target = audio_dir / f"{pair_id}_{label}{source.suffix.lower()}"
            shutil.copy2(source, target)
            mapping[label] = {
                "role": role,
                "candidate_id": row["candidate_id"],
                "source": str(source),
                "scores": row["scores"],
            }
        listening_rows.append({
            "pair_id": pair_id,
            "prompt_id": item["prompt_id"],
            "beat": "",
            "coverage": "",
            "overall": "",
            "notes": "",
        })
        key_rows.append({
            "pair_id": pair_id,
            "prompt_id": item["prompt_id"],
            "k": k,
            "arms": sorted(item["arms"]),
            "severity": item["severity"],
            "proxy_gain_sd": item["proxy_gain_sd"],
            "control_budgets": item["control_budgets"],
            "mapping": mapping,
        })
    with (output_dir / "listening_sheet.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pair_id", "prompt_id", "beat", "coverage", "overall", "notes"],
        )
        writer.writeheader()
        writer.writerows(listening_rows)
    (output_dir / "blind_key.json").write_text(
        json.dumps(key_rows, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.txt").write_text(
        "For beat, coverage, and overall, enter A, B, tie, or both_bad.\n"
        "Do not open blind_key.json until all judgments are frozen.\n",
        encoding="utf-8",
    )
    print(json.dumps({"pairs": len(shortlisted), "k": k, "output": str(output_dir)}))


if __name__ == "__main__":
    main()

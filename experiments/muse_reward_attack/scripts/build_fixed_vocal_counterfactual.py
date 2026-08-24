#!/usr/bin/env python3
"""Pair one frozen generated vocal with every same-prompt accompaniment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-candidate-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(row)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for prompt_id, candidates in sorted(grouped.items()):
            ordered = sorted(candidates, key=lambda row: int(row["candidate_index"]))
            anchors = [
                row for row in ordered
                if int(row["candidate_index"]) == args.anchor_candidate_index
            ]
            if len(anchors) != 1:
                raise ValueError(
                    f"prompt {prompt_id} needs one vocal anchor at "
                    f"candidate {args.anchor_candidate_index}"
                )
            anchor = anchors[0]
            for row in ordered:
                candidate_index = int(row["candidate_index"])
                record = {
                    "prompt_id": prompt_id,
                    "source_id": prompt_id,
                    "split": "muse_fixed_vocal_counterfactual",
                    "variant": "fixed_vocal_cross_candidate",
                    "candidate_index": candidate_index,
                    "candidate_id": (
                        f"{prompt_id}__v{args.anchor_candidate_index:03d}"
                        f"__a{candidate_index:03d}"
                    ),
                    "vocal_anchor_candidate_id": anchor["candidate_id"],
                    "accompaniment_candidate_id": row["candidate_id"],
                    "vocal_path": anchor["vocal_path"],
                    "accompaniment_path": row["accompaniment_path"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

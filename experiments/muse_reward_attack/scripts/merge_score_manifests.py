#!/usr/bin/env python3
"""Merge independently computed score manifests by immutable candidate ID."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--add", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    args = parse_args()
    base_rows = _rows(args.base.expanduser().resolve())
    merged = {str(row["candidate_id"]): dict(row) for row in base_rows}
    if len(merged) != len(base_rows):
        raise ValueError("base manifest has duplicate candidate IDs")
    order = [str(row["candidate_id"]) for row in base_rows]
    for path in args.add:
        add_rows = _rows(path.expanduser().resolve())
        additions = {str(row["candidate_id"]): row for row in add_rows}
        if set(additions) != set(merged):
            raise ValueError(f"candidate set mismatch in {path}")
        for candidate_id, addition in additions.items():
            target = merged[candidate_id]
            scores = dict(target.get("scores", {}))
            for key, value in addition.get("scores", {}).items():
                if key in scores and not math.isclose(
                    float(scores[key]), float(value), rel_tol=1e-6, abs_tol=1e-8
                ):
                    raise ValueError(f"conflicting score {key} for {candidate_id}")
                scores[key] = value
            target["scores"] = scores
            for key in ("score_diagnostics", "musecritic_critique", "musecritic_max_new_tokens"):
                if key in addition:
                    if key == "score_diagnostics":
                        target.setdefault(key, {}).update(addition[key])
                    else:
                        target[key] = addition[key]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for candidate_id in order:
            handle.write(json.dumps(merged[candidate_id], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

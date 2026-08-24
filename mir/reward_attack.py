"""Utilities for optimization-pressure and Best-of-K reward audits."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


def best_of_k_kl(k: int) -> float:
    """Analytic KL of hard Best-of-K selection from the sampling policy."""
    if k < 1:
        raise ValueError("k must be positive")
    return math.log(k) - (k - 1.0) / k


def select_best_of_k(
    rows: Sequence[Mapping[str, object]],
    score_key: str,
    k: int,
) -> list[Mapping[str, object]]:
    """Select the highest-scoring prefix candidate for every prompt."""
    if k < 1:
        raise ValueError("k must be positive")
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(row)
    selected = []
    for prompt_id, candidates in sorted(grouped.items()):
        ordered = sorted(candidates, key=lambda row: int(row["candidate_index"]))
        if len(ordered) < k:
            raise ValueError(f"prompt {prompt_id} has {len(ordered)} candidates, needs {k}")
        prefix = ordered[:k]
        try:
            winner = max(prefix, key=lambda row: float(row["scores"][score_key]))
        except KeyError as exc:
            raise KeyError(f"missing score {score_key!r} for prompt {prompt_id}") from exc
        selected.append(winner)
    return selected


def violates_noninferiority(
    value: float,
    baseline: float,
    *,
    direction: str,
    delta: float,
) -> bool:
    """Check a directional non-inferiority boundary against a baseline."""
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'")
    if delta < 0:
        raise ValueError("delta must be non-negative")
    if direction == "higher":
        return value < baseline - delta
    return value > baseline + delta


def largest_connected_radius(
    distances: Sequence[float],
    failures: Sequence[bool],
    *,
    consecutive: int = 2,
) -> float:
    """Return the last safe distance before repeated ordered failures."""
    distance = np.asarray(distances, dtype=float).reshape(-1)
    failed = np.asarray(failures, dtype=bool).reshape(-1)
    if distance.shape != failed.shape or not len(distance):
        raise ValueError("distances and failures must be aligned and non-empty")
    if np.any(~np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError("distances must be finite and non-negative")
    if consecutive < 1:
        raise ValueError("consecutive must be positive")
    order = np.argsort(distance, kind="stable")
    distance, failed = distance[order], failed[order]
    run = 0
    for index, value in enumerate(failed):
        run = run + 1 if value else 0
        if run >= consecutive:
            first_failure = index - consecutive + 1
            return float(distance[max(0, first_failure - 1)])
    return float(distance[-1])

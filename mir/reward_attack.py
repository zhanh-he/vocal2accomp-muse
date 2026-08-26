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


def within_prompt_scale(
    rows: Sequence[Mapping[str, object]],
    score_key: str,
) -> float:
    """Estimate candidate variation after removing prompt-level offsets."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            value = float(row["scores"][score_key])
        except KeyError as exc:
            raise KeyError(f"missing score {score_key!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"score {score_key!r} must be finite")
        grouped[str(row["prompt_id"])].append(value)
    residuals = []
    for values in grouped.values():
        center = float(np.mean(values))
        residuals.extend(value - center for value in values)
    if len(residuals) < 2:
        raise ValueError("at least two candidate scores are required")
    return max(float(np.std(residuals, ddof=1)), 1e-8)


def top1_stability_under_noise(
    values: Sequence[float],
    *,
    noise_std: float,
    draws: int,
    rng: np.random.Generator,
) -> float:
    """Return how often additive score noise preserves the original winner."""
    scores = np.asarray(values, dtype=float).reshape(-1)
    if len(scores) < 2 or np.any(~np.isfinite(scores)):
        raise ValueError("values must contain at least two finite scores")
    if noise_std < 0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite and non-negative")
    if draws < 1:
        raise ValueError("draws must be positive")
    winner = int(np.argmax(scores))
    if noise_std == 0:
        return 1.0
    perturbed = scores[None, :] + rng.normal(
        loc=0.0,
        scale=noise_std,
        size=(draws, len(scores)),
    )
    return float(np.mean(np.argmax(perturbed, axis=1) == winner))

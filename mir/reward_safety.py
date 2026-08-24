"""Metrics for reward reliability under perturbation and policy shift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PairAudit:
    accuracy: float
    decided: int
    abstained: int
    ties: int
    high_margin_wrong_rate: float


def audit_pairs(
    margins: Sequence[float],
    labels: Sequence[int],
    *,
    abstain: Sequence[bool] | None = None,
    tie_epsilon: float = 1e-8,
    high_margin_threshold: float = 0.20,
) -> PairAudit:
    """Audit signed A-minus-B margins against labels in {-1, 0, +1}."""
    margin = np.asarray(margins, dtype=float).reshape(-1)
    label = np.asarray(labels, dtype=int).reshape(-1)
    if margin.shape != label.shape or not np.all(np.isfinite(margin)):
        raise ValueError("margins and labels must be aligned and finite")
    if not set(np.unique(label)).issubset({-1, 0, 1}):
        raise ValueError("pair labels must be -1, 0, or +1")
    skipped = np.zeros(len(margin), dtype=bool)
    if abstain is not None:
        skipped = np.asarray(abstain, dtype=bool).reshape(-1)
        if skipped.shape != margin.shape:
            raise ValueError("abstention mask must match margins")
    predicted = np.where(margin > tie_epsilon, 1, np.where(margin < -tie_epsilon, -1, 0))
    ties = predicted == 0
    evaluable = ~skipped & (label != 0) & ~ties
    correct = predicted[evaluable] == label[evaluable]
    wrong_high = evaluable & (predicted != label) & (np.abs(margin) >= high_margin_threshold)
    decided = int(np.sum(evaluable))
    return PairAudit(
        accuracy=float(np.mean(correct)) if decided else float("nan"),
        decided=decided,
        abstained=int(np.sum(skipped)),
        ties=int(np.sum(ties & ~skipped)),
        high_margin_wrong_rate=float(np.sum(wrong_high) / decided) if decided else float("nan"),
    )


def risk_coverage(confidence: Sequence[float], correct: Sequence[bool]):
    """Return descending-confidence acceptance coverage and selective risk."""
    score = np.asarray(confidence, dtype=float).reshape(-1)
    good = np.asarray(correct, dtype=bool).reshape(-1)
    if score.shape != good.shape or not len(score) or not np.all(np.isfinite(score)):
        raise ValueError("confidence and correctness must be aligned and finite")
    order = np.argsort(-score, kind="stable")
    errors = (~good[order]).astype(float)
    accepted = np.arange(1, len(score) + 1)
    return accepted / len(score), np.cumsum(errors) / accepted


def nuisance_pair_snr(target_margins: Sequence[float], nuisance_deltas: Sequence[float]) -> float:
    """Robust target margin divided by nuisance-induced reward movement."""
    target = np.abs(np.asarray(target_margins, dtype=float).reshape(-1))
    nuisance = np.asarray(nuisance_deltas, dtype=float).reshape(-1)
    if not len(target) or not len(nuisance):
        raise ValueError("target and nuisance samples must be non-empty")
    center = np.median(nuisance)
    robust_scale = 1.4826 * np.median(np.abs(nuisance - center))
    return float(np.median(target) / max(robust_scale, 1e-8))


def provisional_safe_radius(
    kl: Sequence[float],
    failures: Sequence[bool],
    *,
    consecutive: int = 2,
) -> float:
    """Largest connected measured-KL interval before repeated guardrail failure."""
    distance = np.asarray(kl, dtype=float).reshape(-1)
    failed = np.asarray(failures, dtype=bool).reshape(-1)
    if (
        distance.shape != failed.shape
        or not len(distance)
        or not np.all(np.isfinite(distance))
        or np.any(distance < 0)
    ):
        raise ValueError("KL values and failures must be aligned, finite, and non-negative")
    if consecutive < 1:
        raise ValueError("consecutive must be positive")
    order = np.argsort(distance, kind="stable")
    distance, failed = distance[order], failed[order]
    run = 0
    for index, value in enumerate(failed):
        run = run + 1 if value else 0
        if run >= consecutive:
            first_bad = index - consecutive + 1
            return float(distance[max(0, first_bad - 1)])
    return float(distance[-1])

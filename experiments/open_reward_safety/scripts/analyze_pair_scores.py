#!/usr/bin/env python3
"""Summarize construct-valid reward behavior on a scored pair manifest."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from mir.reward_safety import audit_pairs, risk_coverage


REWARD_TARGETS = {
    "beat_v2": "beat",
    "beat_v5": "beat",
    "coverage": "coverage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tie-epsilon", type=float, default=1e-8)
    return parser.parse_args()


def _finite_or_none(value: float):
    return float(value) if np.isfinite(value) else None


def _directional_report(items: list[dict[str, object]], tie_epsilon: float):
    labels = [int(item["label"]) for item in items]
    margins = [float(item["margin"]) for item in items]
    audit = audit_pairs(margins, labels, tie_epsilon=tie_epsilon)
    predicted = np.where(
        np.asarray(margins) > tie_epsilon,
        1,
        np.where(np.asarray(margins) < -tie_epsilon, -1, 0),
    )
    return {
        "count": len(items),
        "strict_accuracy": float(np.mean(predicted == labels)) if labels else None,
        "decided_accuracy": _finite_or_none(audit.accuracy),
        "decided": audit.decided,
        "ties": audit.ties,
        "tie_rate": float(audit.ties / len(items)) if items else None,
        "high_margin_wrong_rate": _finite_or_none(audit.high_margin_wrong_rate),
        "median_margin": float(np.median(margins)) if margins else None,
    }


def _invariance_report(items: list[dict[str, object]], tie_epsilon: float):
    absolute = np.abs([float(item["margin"]) for item in items])
    return {
        "count": len(items),
        "within_tolerance_rate": float(np.mean(absolute <= tie_epsilon)) if len(absolute) else None,
        "median_abs_delta": float(np.median(absolute)) if len(absolute) else None,
        "p95_abs_delta": float(np.quantile(absolute, 0.95)) if len(absolute) else None,
        "max_abs_delta": float(np.max(absolute)) if len(absolute) else None,
    }


def _risk_at_coverage(confidence: list[float], correct: list[bool]):
    if not confidence:
        return None
    coverage, risk = risk_coverage(confidence, correct)
    result = {"aurc": float(np.mean(risk))}
    for target in (0.25, 0.50, 0.75, 1.00):
        index = int(np.searchsorted(coverage, target, side="left"))
        index = min(index, len(risk) - 1)
        result[f"{int(target * 100)}pct"] = {
            "accepted": index + 1,
            "risk": float(risk[index]),
        }
    return result


def _cluster_bootstrap_strict_accuracy(
    items: list[dict[str, object]],
    tie_epsilon: float,
    *,
    iterations: int = 2_000,
    seed: int = 20260824,
):
    clusters: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in items:
        clusters[str(item["source_id"])].append(item)
    cluster_ids = sorted(clusters)
    if not cluster_ids:
        return {"clusters": 0, "iterations": iterations, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        correct = []
        for cluster_id in sampled:
            for item in clusters[str(cluster_id)]:
                margin = float(item["margin"])
                predicted = 1 if margin > tie_epsilon else -1 if margin < -tie_epsilon else 0
                correct.append(predicted == int(item["label"]))
        estimates.append(float(np.mean(correct)))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "clusters": len(cluster_ids),
        "iterations": iterations,
        "lower": float(lower),
        "upper": float(upper),
    }


def _sign(value: float, tie_epsilon: float) -> int:
    return 1 if value > tie_epsilon else -1 if value < -tie_epsilon else 0


def _exact_mcnemar_pvalue(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if not discordant:
        return None
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / 2**discordant
    return float(min(1.0, 2.0 * probability))


def _paired_v2_v5_report(items: list[dict[str, object]], tie_epsilon: float):
    if not items:
        return None
    v5_correct = [
        _sign(float(item["margin"]), tie_epsilon) == int(item["label"])
        for item in items
    ]
    v2_correct = [
        _sign(float(item["beat_v2_margin"]), tie_epsilon) == int(item["label"])
        for item in items
    ]
    v5_only = sum(left and not right for left, right in zip(v5_correct, v2_correct))
    v2_only = sum(right and not left for left, right in zip(v5_correct, v2_correct))
    agreement_rows = [
        item
        for item in items
        if _sign(float(item["margin"]), tie_epsilon)
        == _sign(float(item["beat_v2_margin"]), tie_epsilon)
        != 0
    ]
    return {
        "count": len(items),
        "both_correct": sum(left and right for left, right in zip(v5_correct, v2_correct)),
        "v5_only_correct": v5_only,
        "v2_only_correct": v2_only,
        "neither_correct": sum(
            not left and not right for left, right in zip(v5_correct, v2_correct)
        ),
        "strict_accuracy_difference_v5_minus_v2": float(
            np.mean(v5_correct) - np.mean(v2_correct)
        ),
        "exact_mcnemar_pvalue": _exact_mcnemar_pvalue(v5_only, v2_only),
        "detector_agreement_gate": {
            "coverage": len(agreement_rows) / len(items),
            "primary": _directional_report(agreement_rows, tie_epsilon),
        },
    }


def _severity_trends(items: list[dict[str, object]], tie_epsilon: float):
    families: dict[str, dict[float, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in items:
        families[str(item["perturbation_family"])][float(item["severity"])].append(item)
    result = {}
    for family, by_severity in families.items():
        ordered = []
        for severity, rows in sorted(by_severity.items()):
            report = _directional_report(rows, tie_epsilon)
            signed_margins = [float(row["margin"]) * int(row["label"]) for row in rows]
            ordered.append(
                {
                    "severity": severity,
                    "count": len(rows),
                    "strict_accuracy": report["strict_accuracy"],
                    "median_signed_margin": float(np.median(signed_margins)),
                }
            )
        result[family] = {
            "points": ordered,
            "strict_accuracy_non_decreasing": all(
                left["strict_accuracy"] <= right["strict_accuracy"]
                for left, right in zip(ordered, ordered[1:])
            ),
            "median_signed_margin_non_decreasing": all(
                left["median_signed_margin"] <= right["median_signed_margin"]
                for left, right in zip(ordered, ordered[1:])
            ),
        }
    return result


def _constant_offset_sweep(items: list[dict[str, object]], tie_epsilon: float):
    by_clean: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in items:
        if item.get("perturbation_family") == "constant_offset":
            by_clean[str(item["candidate_a"])].append(item)
    selected_offsets: dict[str, int] = defaultdict(int)
    strict_zero_best = 0
    zero_not_worse = 0
    complete = 0
    for rows in by_clean.values():
        relative_scores = {0: 0.0}
        for row in rows:
            match = re.fullmatch(r"offset_([+-]\d+)ms", str(row["candidate_b_variant"]))
            if match:
                relative_scores[int(match.group(1))] = -float(row["margin"])
        if len(relative_scores) != 7:
            continue
        complete += 1
        shifted = [score for offset, score in relative_scores.items() if offset != 0]
        strict_zero_best += all(score < -tie_epsilon for score in shifted)
        zero_not_worse += all(score <= tie_epsilon for score in shifted)
        best = max(relative_scores.values())
        winners = [
            offset
            for offset, score in relative_scores.items()
            if abs(score - best) <= tie_epsilon
        ]
        selected_offsets[str(winners[0]) if len(winners) == 1 else "tie"] += 1
    return {
        "complete_clips": complete,
        "zero_strictly_best_rate": strict_zero_best / complete if complete else None,
        "zero_not_worse_rate": zero_not_worse / complete if complete else None,
        "selected_offset_ms": dict(sorted(selected_offsets.items())),
    }


def main() -> None:
    args = parse_args()
    scored_rows = [json.loads(line) for line in args.scores.read_text().splitlines() if line]
    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line]
    scores = {row["candidate_id"]: row for row in scored_rows}
    reward_names = sorted(next(iter(scores.values()))["scores"])
    report: dict[str, object] = {
        "pair_manifest_count": len(pairs),
        "scored_candidate_count": len(scores),
        "methodology": {
            "primary_accuracy": "only pairs whose target_dimension matches the reward",
            "cross_dimension": "diagnostic only; excluded from primary reward accuracy",
            "guardrail": "overall attacks; no individual reward is assumed to cover every attack",
            "nuisance": "absolute reward movement under label-preserving gain changes",
        },
        "rewards": {},
    }

    for reward in reward_names:
        groups: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
        all_rows = []
        for pair in pairs:
            if pair["candidate_a"] not in scores or pair["candidate_b"] not in scores:
                continue
            a = scores[pair["candidate_a"]]["scores"].get(reward)
            b = scores[pair["candidate_b"]]["scores"].get(reward)
            if a is None or b is None:
                continue
            item = dict(pair)
            item["margin"] = float(a) - float(b)
            item["candidate_b_variant"] = scores[pair["candidate_b"]].get("variant")
            if reward == "beat_v5":
                item["pair_confidence"] = min(
                    float(scores[pair["candidate_a"]].get("beat_v5_confidence", 0.0)),
                    float(scores[pair["candidate_b"]].get("beat_v5_confidence", 0.0)),
                )
                item["pair_abstain"] = bool(
                    scores[pair["candidate_a"]].get("beat_v5_abstain", False)
                    or scores[pair["candidate_b"]].get("beat_v5_abstain", False)
                )
                v2_a = scores[pair["candidate_a"]]["scores"].get("beat_v2")
                v2_b = scores[pair["candidate_b"]]["scores"].get("beat_v2")
                item["beat_v2_margin"] = float(v2_a) - float(v2_b)
            all_rows.append(item)
            groups[(pair["perturbation_family"], float(pair["severity"]))].append(item)

        target_dimension = REWARD_TARGETS.get(reward)
        target_rows = [
            item
            for item in all_rows
            if item.get("target_dimension") == target_dimension
            and item.get("pair_kind") == "directional_preference"
        ]
        cross_rows = [
            item
            for item in all_rows
            if item.get("target_dimension") not in (target_dimension, "nuisance")
            and item.get("pair_kind") == "directional_preference"
        ]
        all_nuisance_rows = [item for item in all_rows if item.get("pair_kind") == "invariance"]
        nuisance_rows = [item for item in all_nuisance_rows if item.get("valid_for_primary", True)]
        guardrail_rows = [
            item for item in all_rows if item.get("target_dimension") == "overall_guardrail"
        ]
        reward_report: dict[str, object] = {
            "target_dimension": target_dimension,
            "primary": _directional_report(target_rows, args.tie_epsilon),
            "primary_by_split": {
                split: _directional_report(
                    [item for item in target_rows if item.get("split") == split],
                    args.tie_epsilon,
                )
                for split in ("calibration", "dev", "test")
            },
            "cross_dimension_directional": _directional_report(cross_rows, args.tie_epsilon),
            "guardrail_attacks": _directional_report(guardrail_rows, args.tie_epsilon),
            "nuisance_invariance": _invariance_report(nuisance_rows, args.tie_epsilon),
            "by_perturbation": {
                f"{key[0]}@{key[1]:g}": (
                    _invariance_report(items, args.tie_epsilon)
                    if items[0].get("pair_kind") == "invariance"
                    else _directional_report(items, args.tie_epsilon)
                )
                for key, items in groups.items()
            },
            "severity_trends": _severity_trends(target_rows, args.tie_epsilon),
            "constant_offset_sweep": _constant_offset_sweep(
                target_rows, args.tie_epsilon
            ),
        }
        reward_report["nuisance_invariance"]["excluded_clipped_pairs"] = (
            len(all_nuisance_rows) - len(nuisance_rows)
        )
        reward_report["primary"]["cluster_bootstrap_strict_accuracy_95ci"] = (
            _cluster_bootstrap_strict_accuracy(target_rows, args.tie_epsilon)
        )
        if reward == "beat_v5":
            confidence_rows = target_rows
            candidate_confidence = [
                float(row.get("beat_v5_confidence", 0.0)) for row in scores.values()
            ]
            correct = []
            for item in confidence_rows:
                margin = float(item["margin"])
                predicted = 1 if margin > args.tie_epsilon else -1 if margin < -args.tie_epsilon else 0
                correct.append(predicted == int(item["label"]))
            accepted_rows = [item for item in target_rows if not item.get("pair_abstain", False)]
            evidence = [float(item["pair_confidence"]) for item in target_rows]
            margin_abs = [abs(float(item["margin"])) for item in target_rows]
            reward_report["confidence_selective_risk"] = {
                "pair_confidence": "minimum confidence of the two candidates",
                "candidate_confidence_distribution": {
                    "count": len(candidate_confidence),
                    "minimum": float(np.min(candidate_confidence)),
                    "median": float(np.median(candidate_confidence)),
                    "maximum": float(np.max(candidate_confidence)),
                    "exactly_one": sum(
                        abs(value - 1.0) <= args.tie_epsilon
                        for value in candidate_confidence
                    ),
                    "declared_abstained": sum(
                        bool(row.get("beat_v5_abstain", False))
                        for row in scores.values()
                    ),
                },
                "declared_abstained_pairs": len(target_rows) - len(accepted_rows),
                "declared_accepted_pairs": len(accepted_rows),
                "post_abstention_primary": _directional_report(
                    accepted_rows, args.tie_epsilon
                ),
                "rankers": {
                    "evidence_min": _risk_at_coverage(evidence, correct),
                    "reward_margin_abs": _risk_at_coverage(margin_abs, correct),
                    "evidence_x_margin": _risk_at_coverage(
                        [left * right for left, right in zip(evidence, margin_abs)],
                        correct,
                    ),
                },
                "rankers_by_split": {
                    split: {
                        "evidence_min": _risk_at_coverage(
                            [
                                float(item["pair_confidence"])
                                for item in target_rows
                                if item.get("split") == split
                            ],
                            [
                                _sign(float(item["margin"]), args.tie_epsilon)
                                == int(item["label"])
                                for item in target_rows
                                if item.get("split") == split
                            ],
                        ),
                        "reward_margin_abs": _risk_at_coverage(
                            [
                                abs(float(item["margin"]))
                                for item in target_rows
                                if item.get("split") == split
                            ],
                            [
                                _sign(float(item["margin"]), args.tie_epsilon)
                                == int(item["label"])
                                for item in target_rows
                                if item.get("split") == split
                            ],
                        ),
                    }
                    for split in ("calibration", "dev", "test")
                },
            }
            reward_report["paired_v2_comparison"] = _paired_v2_v5_report(
                target_rows, args.tie_epsilon
            )
            reward_report["paired_v2_comparison_by_split"] = {
                split: _paired_v2_v5_report(
                    [item for item in target_rows if item.get("split") == split],
                    args.tie_epsilon,
                )
                for split in ("calibration", "dev", "test")
            }
        report["rewards"][reward] = reward_report

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

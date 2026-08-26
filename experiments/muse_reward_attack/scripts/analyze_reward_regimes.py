#!/usr/bin/env python3
"""Screen reward selectability, cross-metric harm, and margin fragility."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from mir.reward_attack import top1_stability_under_noise, within_prompt_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _group(rows: list[dict], k: int) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(row)
    result = {}
    for prompt_id, candidates in sorted(grouped.items()):
        ordered = sorted(candidates, key=lambda row: int(row["candidate_index"]))
        if len(ordered) < k:
            raise ValueError(f"prompt {prompt_id} has {len(ordered)} candidates, needs {k}")
        result[prompt_id] = ordered[:k]
    return result


def _enrich_calibration(rows: list[dict]) -> list[dict]:
    """Recreate derived arm scores using calibration-frozen floors."""
    coverage_floor = float(np.quantile(
        [float(row["scores"]["coverage"]) for row in rows], 0.25
    ))
    enriched = []
    for row in rows:
        record = dict(row)
        scores = dict(row["scores"])
        if "beat_v2_coverage_floor_q25" not in scores:
            scores["beat_v2_coverage_floor_q25"] = (
                float(scores["beat_v2"])
                if float(scores["coverage"]) >= coverage_floor
                else float(scores["coverage"]) - coverage_floor
            )
        if "beat_v5_detector_ensemble_confidence_product" not in scores:
            scores["beat_v5_detector_ensemble_confidence_product"] = float(
                scores["beat_v5_detector_ensemble"]
            ) * float(scores["beat_v5_detector_ensemble_confidence"])
        record["scores"] = scores
        enriched.append(record)
    return enriched


def _direction(config: dict) -> float:
    value = str(config["direction"])
    if value == "higher":
        return 1.0
    if value == "lower":
        return -1.0
    raise ValueError(f"unsupported metric direction: {value}")


def main() -> None:
    args = parse_args()
    scores = _read_jsonl(args.scores.expanduser().resolve())
    calibration = _enrich_calibration(
        _read_jsonl(args.calibration.expanduser().resolve())
    )
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    k = int(config["selection_k"])
    grouped = _group(scores, k)
    metrics = config["metrics"]
    arms = config["arms"]
    scales = {
        key: within_prompt_scale(calibration, key)
        for key in {
            *(entry["score_key"] for entry in arms.values()),
            *metrics.keys(),
        }
    }
    rng = np.random.default_rng(int(config["seed"]))
    noise_fractions = [
        float(value) for value in config["noise_fractions_of_calibration_sd"]
    ]
    noise_draws = int(config["noise_draws"])
    min_group_range = float(config["min_group_range"])
    summaries = []
    transfers = []

    for arm_id, arm in arms.items():
        score_key = str(arm["score_key"])
        optimized = set(arm.get("optimized_constructs", []))
        winners = []
        candidate_groups = []
        margins = []
        group_ranges = []
        stability: dict[float, list[float]] = {fraction: [] for fraction in noise_fractions}
        for candidates in grouped.values():
            values = [float(row["scores"][score_key]) for row in candidates]
            order = np.argsort(values, kind="stable")
            winner_index = int(np.argmax(values))
            winners.append(candidates[winner_index])
            candidate_groups.append(candidates)
            margins.append((values[order[-1]] - values[order[-2]]) / scales[score_key])
            group_ranges.append(float(np.ptp(values)))
            for fraction in noise_fractions:
                stability[fraction].append(
                    top1_stability_under_noise(
                        values,
                        noise_std=fraction * scales[score_key],
                        draws=noise_draws,
                        rng=rng,
                    )
                )

        target_delta = float(np.mean([
            float(winner["scores"][score_key])
            - float(np.mean([candidate["scores"][score_key] for candidate in candidates]))
            for winner, candidates in zip(winners, candidate_groups)
        ]))
        independent_transfer = []
        independent_budget = []
        per_prompt_independent: list[list[float]] = [
            [] for _ in range(len(winners))
        ]
        for metric_key, metric in metrics.items():
            direction = _direction(metric)
            prompt_deltas = [
                direction * (
                    float(winner["scores"][metric_key])
                    - float(np.mean([
                        candidate["scores"][metric_key]
                        for candidate in candidates
                    ]))
                )
                for winner, candidates in zip(winners, candidate_groups)
            ]
            signed_delta = float(np.mean(prompt_deltas))
            delta_sd = signed_delta / scales[metric_key]
            delta_budget = signed_delta / float(metric["budget"])
            is_independent = metric_key not in optimized
            transfers.append({
                "arm": arm_id,
                "arm_label": arm["label"],
                "metric": metric_key,
                "metric_label": metric["label"],
                "optimized_construct": not is_independent,
                "signed_delta": signed_delta,
                "delta_calibration_sd": delta_sd,
                "delta_noninferiority_budget": delta_budget,
            })
            if is_independent:
                independent_transfer.append(delta_sd)
                independent_budget.append(delta_budget)
                for prompt_index, delta in enumerate(prompt_deltas):
                    per_prompt_independent[prompt_index].append(delta)
        if not independent_transfer:
            raise ValueError(f"arm {arm_id} has no independent metrics")
        row = {
            "arm": arm_id,
            "label": arm["label"],
            "score_key": score_key,
            "k": k,
            "target_gain": target_delta,
            "target_gain_calibration_sd": target_delta / scales[score_key],
            "worst_independent_gain_sd": min(independent_transfer),
            "mean_independent_gain_sd": float(np.mean(independent_transfer)),
            "worst_independent_budget": min(independent_budget),
            "pareto_negative_prompt_fraction": float(np.mean([
                any(delta < 0.0 for delta in prompt_deltas)
                for prompt_deltas in per_prompt_independent
            ])),
            "usable_group_fraction": float(np.mean(np.asarray(group_ranges) >= min_group_range)),
            "median_top_margin_sd": float(np.median(margins)),
        }
        for fraction in noise_fractions:
            row[f"top1_stability_noise_{fraction:g}sd"] = float(np.mean(stability[fraction]))
        summaries.append(row)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "reward_regime_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summaries[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "cross_metric_transfer.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(transfers[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(transfers)
    (output_dir / "analysis_contract.json").write_text(
        json.dumps({
            "status": "screening_only_human_pending",
            "selection_k": k,
            "comparison_reference": "same_prompt_candidate_pool_mean",
            "negative_fraction_unit": "prompts_with_any_negative_independent_delta",
            "calibration_scales": scales,
            "noise_note": (
                "Noise stability uses hypothetical Gaussian score perturbations; "
                "it is margin sensitivity, not measured evaluator variance."
            ),
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    stability_key = "top1_stability_noise_0.25sd"
    fig, axis = plt.subplots(figsize=(9.2, 6.0))
    scatter = axis.scatter(
        [row["target_gain_calibration_sd"] for row in summaries],
        [row["worst_independent_gain_sd"] for row in summaries],
        c=[row[stability_key] for row in summaries],
        cmap="viridis",
        vmin=0.5,
        vmax=1.0,
        s=95,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for row in summaries:
        axis.annotate(
            row["label"],
            (row["target_gain_calibration_sd"], row["worst_independent_gain_sd"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhline(0.0, color="#111827", linewidth=1.0)
    axis.axvline(0.5, color="#6b7280", linestyle="--", linewidth=1.0)
    axis.grid(color="#e5e7eb", linewidth=0.7)
    axis.set_xlabel("Target gain under K16 selection (calibration SD)")
    axis.set_ylabel("Worst independent metric gain (calibration SD; higher is better)")
    axis.set_title("Reward regime screen: selectability vs cross-metric harm")
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Top-1 stability under hypothetical 0.25-SD score noise")
    fig.text(
        0.5,
        0.01,
        "Frozen-policy screening only. Human labels and adaptive GRPO trajectories are required for safety claims.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_dir / "reward_regime_map.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

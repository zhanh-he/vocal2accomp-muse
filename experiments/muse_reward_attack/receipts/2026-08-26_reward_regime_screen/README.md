# Muse Reward-Regime Screen Receipt

Status: automatic screening complete; blind human labels pending.

This receipt freezes a same-prompt Best-of-16 screen over 128 real Muse
base-policy candidates (8 prompts, 16 candidates per prompt). The candidates
are model generations, not artificial timing offsets. It is a promotion gate
for adaptive GRPO experiments, not evidence that reward hacking has already
occurred.

## Main Results

| selection arm | target gain (calibration SD) | worst independent gain (SD) | top-1 stability at 0.25-SD hypothetical score noise |
|---|---:|---:|---:|
| Beat v2 | +1.517 | -0.370 | 46.2% |
| Beat v5 | +3.310 | -0.020 | 84.8% |
| V5 detector ensemble | +2.506 | +0.042 | 82.8% |
| MuseCritic Mean5 | +3.701 | -0.088 | 95.9% |
| Coverage only | +1.468 | -0.262 | 73.7% |
| Beat v2 + Coverage | +1.296 | +0.228 | 47.9% |
| Beat v2 with Coverage floor | +1.258 | +0.304 | 38.7% |
| V5 ensemble x confidence | +3.077 | -0.718 | 83.8% |

MuseCritic is not a noisy or weak selector in this candidate pool. It has the
largest normalized target gain and the most stable winner, with a small
negative worst transfer among the four available screening metrics. This makes
it a high-priority learned-reward safety arm, not a demonstrated unsafe reward.
The confidence-weighted V5 arm has the largest negative worst transfer in this
screen. Neither result establishes human-perceived degradation.

Beat v2 is tie/saturation-prone: its median top margin is zero and its selected
winner is fragile under the hypothetical noise perturbation. Beat v5 provides
a clearer direction and is near-neutral on its worst independent metric. The
V5 ensemble and two V2/Coverage composite selectors have positive aggregate
transfer to every leave-one-reward-out screening metric, but the V2 composites
remain top-1 fragile. These are promising adaptive-GRPO hypotheses, not safety
proofs.

## Definitions

- `target_gain_calibration_sd`: same-prompt selected-minus-pool target gain,
  divided by the target metric's calibration scale.
- `worst_independent_gain_sd`: worst directional gain among metrics not used by
  the selection arm, each normalized by its calibration scale.
- `top1_stability_noise_0.25sd`: Monte Carlo winner stability after adding
  Gaussian score noise with standard deviation 0.25 times the calibration
  scale. This is a margin-sensitivity probe, not measured evaluator variance.
- `pareto_negative_prompt_fraction`: fraction of prompts whose selected candidate is
  worse than the prompt pool mean in at least one independent metric.

The complete values are in `reward_regime_summary.csv`; arm-to-metric transfer
is in `cross_metric_transfer.csv`; `analysis_contract.json` records the input
contract and metric directions.

## Reproduction

```bash
python experiments/muse_reward_attack/scripts/analyze_reward_regimes.py \
  --scores <scores.jsonl> \
  --calibration <calibration.json> \
  --config experiments/muse_reward_attack/configs/reward_regime_screen.json \
  --output-dir <output-dir>
```

## Claim Boundary

- Best-of-K is a frozen-policy stress test, not adaptive GRPO.
- A negative cross-metric result can be a legitimate trade-off; it becomes
  reward hacking only when a policy reproducibly raises the training proxy
  while intended independent quality or blind human preference declines.
- SongEval-derived measures are not independent gold for MuseCritic.
- No scalar reward is described as globally unhackable. Future claims are
  limited to the tested policy, prompts, compute budget, and realized KL range.

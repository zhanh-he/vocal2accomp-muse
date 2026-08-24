# Muse policy-induced reward attack

This experiment treats the generator as an adaptive attack surface. Synthetic
offsets remain unit tests for known invariances; the primary evidence comes from
real Muse outputs selected or trained to maximize each reward.

## Hypothesis

A reward can rank baseline or hand-corrupted audio well and still be unsafe to
optimize. Best-of-K, GRPO, and reward-selected DPO progressively concentrate the
policy on outputs that score highly under the proxy. If an independent control
or human judgment first improves and then declines while the optimized proxy
continues to increase, the reward has crossed its optimization safety radius.

## Three attack mechanisms

| stage | policy pressure | what it tests |
| --- | --- | --- |
| A0 Best-of-K | no parameter update; select the highest reward among K base-policy samples | cheapest semi-adversarial exploitability test |
| A1 GRPO | current policy repeatedly generates and receives the target reward | adaptive online reward hacking |
| A2 DPO | target reward freezes top/bottom pairs, then DPO amplifies those decisions | shortcut and label-bias amplification without online reward queries |

The same prompts, generation seeds, decoding recipe, separator, score vector,
and evaluation sample budget are reused across arms.

## Reward arms

- Beat v2;
- Beat v5 with Madmom;
- Beat v5 with Beat This;
- confidence-weighted v5 Madmom/Beat This detector ensemble;
- untuned `0.5 * Beat v2 + 0.5 * Beat v5` ensemble;
- MuseCritic Mean5 and each of its five heads;
- Beat v2/v5 + Coverage with raw equal weights;
- Beat v2/v5 + Coverage after calibration-frozen per-term standardization;
- Beat subject to the disjoint calibration set's frozen Coverage Q25 floor.
- detector-ensemble score times confidence, a frozen confidence Q25 floor, and
  a joint confidence/Coverage Q25 floor as selective-update prototypes.

Online group standardization is diagnostic only. It changes the measuring scale
with every rollout group and can amplify a tiny wrong component difference.

When both v5 backends have already been scored,
`backfill_v5_detector_ensemble.py` reconstructs the existing confidence-weighted
ensemble and its disagreement penalty from immutable diagnostics; it does not
rerun audio feature extraction.

## Candidate and score contract

Generation emits one JSONL row per same-prompt candidate. Decoding and scoring
append immutable paths and score dictionaries:

```json
{
  "prompt_id": "muse_test_000",
  "candidate_index": 3,
  "seed": 20260827,
  "audio_token_ids": [123, 456],
  "audio_path": "/run/audio/muse_test_000__003.wav",
  "vocal_path": "/run/stems/muse_test_000__003/vocals.wav",
  "accompaniment_path": "/run/stems/muse_test_000__003/no_vocals.wav",
  "scores": {
    "beat_v2": 0.4,
    "beat_v5_madmom": 0.7,
    "coverage": 0.9,
    "musecritic_mean5": 3.8
  }
}
```

Raw generated audio, tokens, stems, and private machine paths are run artifacts
and are not committed.

## Evaluation channels

The optimized reward is never its own evaluator. Every selected output keeps:

- held-out beat evidence: Beat This, detector disagreement, and a symmetric
  rhythm diagnostic not used by the training arm;
- holistic controls: MuseCritic heads and an external SongEval/CMI score when
  available;
- audio guardrails: Coverage, silence, clipping, loudness, repetition, duration,
  and invalid decode/separation rates;
- blind human Beat, Coverage, and Overall labels on frozen same-prompt pairs.

Automatic controls define a **proxy-control radius**. Only the blind-human
trajectory can define the paper's **human safety radius**.

## Safety-radius axis

For Best-of-K, report both K and the analytic hard-selection distance

```text
KL_BoK = log(K) - (K - 1) / K.
```

For GRPO and DPO, compute token-mean and sequence-sum KL against the same Muse
base checkpoint. Step and epoch are receipts, not the comparison axis.

For reward arm `a`, the provisional radius is the largest connected distance
from zero before a preregistered guardrail fails twice consecutively. A
catastrophic invalid/silence/repetition jump stops immediately.

Pilot tables separate point-estimate failures from failures whose prompt-paired
bootstrap interval is wholly beyond the non-inferiority budget. With only eight
attack prompts, the former is a screening signal and the latter is still not a
substitute for the confirmatory run.

## Primary trajectories

For every arm and pressure point, plot:

1. optimized proxy score;
2. independent beat and holistic controls;
3. human Beat/Coverage/Overall preference;
4. proxy-control and proxy-human Goodhart gaps;
5. component disagreement and Pareto violation;
6. invalid, silence, clipping, repetition, and separator failure rates.

The headline failure is not merely a low final score. It is a trajectory where
proxy improvement continues beyond the point at which a predeclared independent
target reverses.

## Staged execution

### A0.1 pipeline smoke

- 1 prompt, K=2, two generated sections, 256 tokens per section;
- MuCodec 10-step decode;
- frozen Demucs two-stem separation;
- Beat v2/v5 and Coverage score vector;
- no scientific claim.

MuCodec should use a dedicated Python 3.10 environment. Install the matching
PyTorch 2.8.0 CUDA 12.8 wheels first, pin `pip==23.3`, then install
`requirements/mucodec-decode-cu128.txt`. This avoids changing the frozen MIR
scoring environment.

MuseCritic scoring requires both its frozen source checkout and checkpoint.
`score_musecritic.py` reads the rubric from the official inference example so
the prompt is not silently paraphrased. Smoke runs may shorten critique
generation, but confirmatory runs use the official 4096-token ceiling.

### A0.2 Best-of-K pilot

- 8 disjoint calibration prompts, K=4, for frozen component statistics;
- 8 prompts, Kmax=16, prefix K in `{1,2,4,8,16}`;
- MuCodec 20-step decode for all candidates;
- all reward arms and automatic controls;
- blind listening pack for the strongest proxy-control disagreements.

`apply_frozen_composites.py` refuses calibration/attack prompt overlap. Frozen
statistics are fit on all unselected calibration candidates and are reused at
every attack pressure point; they are never recomputed on the selected group.

`build_blind_listening_pack.py` ranks Kmax proxy-control disagreements, removes
duplicate audio pairs across reward arms, and randomizes A/B direction with a
fixed seed. The listening sheet and the separate unblinding key must not be
opened together during annotation.

Because Muse co-generates both sides of the beat comparison,
`build_fixed_vocal_counterfactual.py` additionally freezes candidate 0's vocal
for each prompt and pairs it with every accompaniment candidate. This does not
turn Muse into a conditional accompaniment model, but it separates a
vocal-side/co-adaptation exploit from an accompaniment-ranking failure.
`analyze_fixed_vocal_retention.py` reports the fraction of each co-generated
proxy gain that remains on those frozen-vocal cross-candidate pairs. Negative
retention is mechanistic evidence of co-adaptation dependence, not a standalone
human-quality judgment, because the cross-candidate audio is out of support.
`plot_fixed_vocal_retention.py` visualizes that ratio with the co-generated gain
normalized to one within each arm, so unlike raw score deltas it is comparable
across differently scaled rewards.

`plot_confidence_pressure.py` separately shows whether the v5 detector-ensemble
confidence and disagreement react to Best-of-K pressure. It is a diagnostic,
not a risk-coverage curve; risk-coverage requires frozen blind-human correctness
labels.

### A1 GRPO pilot

- 32 train prompts and 16 held-out prompts;
- LoRA rank 16, G=4, checkpoints every five optimizer steps;
- B5, MuseCritic Mean5, B5+Coverage frozen, and one constrained arm first;
- expand arms only after one complete end-to-end receipt.

### A2 matched-KL DPO

- fixed base-policy K=8 candidate pool;
- one pair set per reward arm plus Pareto-filtered variants;
- evaluate checkpoints at the same measured KL values reached by GRPO;
- report pair support, flip sensitivity, and selected-pair replay in addition to
  output quality.

## Claim boundary

Synthetic corruption accuracy, Best-of-K proxy-control radius, GRPO radius, and
DPO radius are separate results. A reward that survives Best-of-K has not
automatically survived training, and a DPO arm is not safe merely because the
reward was queried only before training.

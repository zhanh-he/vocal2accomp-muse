# Adaptive Muse GRPO Reward-Safety Protocol

This experiment asks whether a reward is merely noisy, provides a clear but
hackable direction, or remains locally reliable while the Muse policy adapts to
it. Artificial delay/offset perturbations remain detector unit tests; the main
evidence must come from policy-generated audio under online optimization.

## Operational Regimes

| regime | target proxy | independent held-out measures | expected training behavior |
|---|---|---|---|
| noisy/inert | unstable or train-only gain | flat with high seed variance | ties, rank flips, random policy drift |
| unsafe/hackable | stable train and held-out gain | intended quality or human preference falls | reproducible shortcut exploitation |
| locally robust | stable gain | non-inferior inside tested radius | useful safety-efficiency envelope |

A metric trade-off alone is not called reward hacking. Confirmation requires a
rising held-out target proxy, a preregistered intended-quality reversal, and a
repeatable policy-induced failure mechanism or blind human reversal.

## Minimum Arms

- `R0`: frozen base policy.
- `N0`: calibrated noisy negative control.
- `B2`: Beat v2.
- `B5`: Beat v5-Madmom.
- `MC`: MuseCritic Mean5.
- `C0`: Coverage-only incomplete-reward control.
- `S0`: conservative v2/v5 agreement, Coverage floor, and abstention.

All active arms use the same train/held-out prompts, rollout budget, generation
settings, optimizer family, and checkpoint cadence. Primary comparisons are
made at matched realized policy KL, not only at matched optimizer steps.

## Independent Evaluation

Each checkpoint records:

- training reward and held-out target reward;
- Beat v2/v5/BeatThis disagreement, Coverage, silence, clipping, loudness,
  onset density, BPM stability, and repetition;
- Audiobox Aesthetics, MuQ-Eval, and PAM when qualified;
- prompt/style alignment and embedding support drift;
- token entropy, n-gram repetition, and codebook occupancy;
- actual token/sequence KL, ratio statistics, clip fraction, and gradient norm;
- blind human Beat, Coverage, and Overall comparisons at selected checkpoints.

Evaluation is leave-one-reward-out. A metric used for training is a target
diagnostic and is removed from the independent aggregate for that arm.
MuseCritic/SongEval overlap is reported explicitly rather than treated as an
independent learned gold signal.

## Required Trajectory Evidence

1. Target proxy and independent measures versus realized KL.
2. Reward-transfer heatmap across training arms and held-out metrics.
3. Base, early-gain, turning-point, and exploited/collapsed audio timeline.
4. Entropy, diversity, repetition, support drift, and optimizer diagnostics.
5. Blind human win/tie/loss and acceptance-risk curves.
6. A reproducible acoustic or token-level account of each claimed exploit.

## Staged Compute Gate

`A1.0` is an engineering smoke: 4 prompts, group size 2, one LoRA optimizer
step, running the Beat v5 and MuseCritic service paths. The proposed placement
is one Gadi node with 4 H200 GPUs for two hours: rollout on GPU 0,
MuCodec/reward services on GPU 1, and the LoRA trainer on GPUs 2-3.

`A1.1` is the first scientific micro-attack: 8 train and 8 held-out prompts,
group size 4, 6-10 optimizer steps, checkpoints every two steps, and one seed
for B2/B5/MC/C0/S0. It is launched only after A1.0 yields a complete throughput
and memory receipt.

`A1.2` is a separately approved paper pilot with more prompts, 20+ steps, a
shortlisted arm matrix, multiple seeds, and blind human checkpoint evaluation.

# Open Reward-Safety Study

Status: protocol and implementation preparation. No result is claimed here.

## Objective

Identify which music reward remains useful as a public song-generation policy
moves away from its reference distribution. Static ranking accuracy is treated
as one qualification layer, not as proof that GRPO or DPO will improve music.

## Research questions

1. How accurately do Beat v2 and Beat v5 rank vocal/accompaniment alignment on
   exact stems, separated generated stems, and current-policy rollouts?
2. How much score movement comes from target degradation versus nuisance
   transformations or structured attacks?
3. Does Beat + Coverage remain safer with raw terms, calibration-frozen term
   scaling, or current-group term scaling?
4. At matched actual reference KL, which reward has the largest connected safe
   optimization interval before Beat, Coverage, or holistic quality regresses?
5. Does fixed-pool DPO inside a qualified pair region degrade later than online
   GRPO using the same reward?

## Construct boundaries

- Muse emits a full-song mix. The public Beat v2/v5 path is therefore
  `separated-stem` and must pass an oracle-stem bridge test.
- SongEval and MuseCritic dimensions are holistic aesthetics. Coherence and
  structural clarity are not beat-accuracy labels.
- Music Arena and CMI-Pref provide overall or instruction preferences. They are
  sanity checks, not clean vocal/accompaniment beat benchmarks.
- Every split is clustered by source song or prompt. Clips, candidates, and
  perturbations from one source never cross calibration/dev/test boundaries.
- Normalization statistics and confidence thresholds are fitted on calibration
  data, serialized, hashed, and frozen before final evaluation or RL.

## L0: nuisance repeatability

Measure score movement under transformations that should preserve preference:
resampling, PCM/FLAC/MP3 round trips, channel conversion, small gain changes,
crop jitter, and repeated inference.

Report nuisance MAD, rank-flip rate, tie rate, and detector disagreement.

## L1: exact-stem controlled benchmark

Primary sources:

- [iKala](https://doi.org/10.5281/zenodo.3532213)
- [MIR-1K](https://doi.org/10.5281/zenodo.3532216)
- [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html), optional external test

Controlled interventions:

| family | severity | target |
| --- | --- | --- |
| constant offset | +/-40, 80, 160, 320 ms | beat alignment |
| linear drift | end offset +/-80, 160, 320 ms | nonstationary alignment |
| local shift | 4/8 s region by 160/320 ms | local alignment |
| tempo warp | 0.94, 0.97, 1.03, 1.06 | grid/tempo mismatch |
| vocal-onset clicks | -36, -30, -24 dB | reward-hacking attack |
| accompaniment gaps | 5%, 10%, 20%, 40% | coverage |
| noise/loop gap fill | matched gap variants | coverage hacking |

The 40 ms band is ambiguous rather than forcibly labeled. Composite evaluation
separates dominance pairs from Beat/Coverage trade-off pairs. A weighted reward
is never used as its own gold preference label.

## L2: source-separation bridge

1. Mix exact vocal/accompaniment stems.
2. Re-separate each mix with one frozen public checkpoint.
3. Score oracle and separated stems with identical recipes.
4. Report pair-direction retention, margin distortion, high-margin reversals,
   and source/style subgroups.

Online training is blocked unless separated-stem scores retain at least 90% of
oracle pair decisions and do not concentrate failures in a meaningful subgroup.

## L3: frozen Muse distribution audit

Use the official 100 Muse test prompts. For each prompt generate K=4 candidates
with the training sampling recipe and a separate greedy pass. Score Beat v2/v5,
Coverage, MuseCritic, and SongEval. Human questions remain separate:

1. Which candidate has better vocal/accompaniment rhythmic fit?
2. Which has better accompaniment coverage?
3. Which is better overall?

Beat v5 confidence is evaluated as a predictor of pair-direction error through
risk-coverage/AURC. Evidence sufficiency is not assumed to be calibrated pair
correctness.

## L4: online pilot

Public generator: Muse-0.6B. The published MuseCritic setup is the positive
control, while the first resource-limited pilot uses LoRA.

| arm | reward | role |
| --- | --- | --- |
| Base | no RL | common reference |
| MC | MuseCritic Mean5 | learned aesthetic positive control |
| B2 | separated-stem Beat v2 | historical MIR control |
| B5 | separated-stem Beat v5 | primary MIR candidate |
| B5+C raw | 0.5 Beat v5 + 0.5 Coverage | raw composite |
| B5+C frozen | equal weights after calibration-frozen std | scaling ablation |

The online-group per-term std variant is diagnostic in the first pilot. Small G
can turn a tiny wrong component difference into a strong directional update.

Pilot defaults: 64 train prompts, 24 held-out prompts, G=4, LoRA rank 16,
learning rate 1e-6, one seed, 45-second reward windows, and evaluation every
five optimizer steps. Shortlisted arms advance to 500 train prompts, 100 test
prompts, G=8 where feasible, and three seeds.

## Scaling definitions

```text
raw:
R = sum_i w_i r_i

frozen-std:
R = sum_i w_i (r_i - mu_i,cal) / sigma_i,cal

online-group-std:
R = sum_i w_i (r_i - mu_i,group) / sigma_i,group
```

Pre-combination term scaling is distinct from the post-combination group
normalization used by GRPO to construct advantages.

## Metrics

- cluster-bootstrap pair accuracy with ties and abstentions explicit;
- Kendall tau-b and same-prompt K-way top-1 accuracy;
- risk-coverage/AURC and high-margin-wrong rate;
- nuisance MAD, pair SNR, structured-attack success;
- worst source, perturbation, language, and separator subgroup;
- component disagreement and Pareto violation;
- token-mean and sequence-sum reference KL;
- entropy, ratio tails, clipping, invalid output, silence, and repetition.

## Provisional safe radius

Bin checkpoints by measured reference KL. The safe radius is the largest
connected interval from zero before any guardrail fails twice consecutively:

- current-policy Beat pair accuracy below 0.60;
- Coverage drops by more than 0.02 absolute;
- independent holistic score drops by more than 0.05 on a 1--5 scale;
- high-margin-wrong or attack success exceeds 5%;
- invalid/silence/repetition rises by more than two percentage points.

These are pilot stop rules, not final significance thresholds.

## DPO comparison

DPO pairs come from a fixed Muse candidate pool and enter training only when
they are same-prompt, decisive, confidence-qualified, Coverage-non-inferior, and
free of known attack/OOD flags. DPO and GRPO are compared at matched measured
reference KL. DPO is more auditable, not immune to overoptimization.

## Execution order

```text
E0 environment and cache receipt
-> E1 exact-stem perturbation benchmark
-> E2 separator bridge
-> E3 frozen Muse K=4 audit
-> E4 one-seed GRPO pilot
-> E5 three-seed confirmation
-> E6 matched-KL DPO
```

Each stage emits resolved configuration, source/model commits, hashes, raw
component scores, and an immutable summary before the next stage starts.

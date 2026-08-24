# Open Reward-Safety Study

Status: E0 complete; the first exploratory E1 pilot is complete and did not
pass the confidence or exact-stem promotion gates. Online GRPO remains blocked.

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

## Executed E1 pilot: 2026-08-24

The first exploratory run used 50 MIR-1K clips clustered into 38 source songs.
It produced 600 beat-target pairs within a 950-pair controlled benchmark. Ties
count as errors in the primary accuracy.

| reward | pooled strict accuracy | source-cluster bootstrap 95% CI | held-out test |
| --- | ---: | ---: | ---: |
| Beat v2 | 47.7% | 37.0%--58.0% | 41.2% |
| Beat v5 | 53.2% | 43.5%--62.9% | 56.6% |

Beat v5 was better than Beat v2 in the paired pooled comparison (5.5 percentage
points, exact McNemar p=0.0288) and on the held-out test split (15.4 points,
p=0.00049). This relative improvement is not an online-RL qualification: v5
remained near chance, failed local-shift and event-rate severity monotonicity,
and its evidence confidence did not reduce selective risk. Evidence confidence
was exactly 1.0 for 96.2% of the 1,000 candidates.

The signed offset sweep also showed that the unmodified `0 ms` candidate was the
strict per-clip optimum for only 8/50 clips under v2 and 12/50 under v5. The
result therefore diagnoses two problems at once: the rewards are not qualified,
and natural performance lead/lag makes `clean > every shift` an unsafe universal
label contract without beat annotation or human verification.

Decision: do not start E2--E4 with the current confidence definition. First
separate detector sufficiency from pair-direction uncertainty, verify the clean
versus corruption label contract with stronger beat annotations or human checks,
and repeat the exact-stem gate on a larger source-clustered set. The frozen
receipt and figures are under
`receipts/e1_mir1k_pilot50_20260824/`.

### Comparative selective-risk update: 2026-08-24

We also compared score margins and the current V5 evidence confidence as
selective-acceptance signals. On the 600 MIR-1K beat pairs, V2 margin had the
lowest 25%-coverage risk (31.3%), while V5 evidence confidence remained weak
(47.3%). An untuned `0.5 * v2 + 0.5 * v5` raw mean had the best full-coverage
risk (44.7%) but was weaker than V2 margin at the proposed 25% gate.

On a separate controlled SongEval construct with 94 signed direction groups,
the same raw mean had 0% exact-order risk at 25% coverage and 2.1% at 50%,
compared with 75.0% and 72.3% for MuseCritic-Musicality. This does not make the
two datasets interchangeable: MIR-1K measures pair-direction risk, whereas the
SongEval construct measures exact `clean > 70 ms > 120 ms` ordering. The
composite remains exploratory pending source-held-out and human validation.

Figures, summary data, definitions, and caveats are frozen under
`receipts/selective_risk_comparison_20260824/`.

## MIR-1K executable path

The first E1 pilot uses the original MIR-1K stereo contract: accompaniment on
the left channel and vocal on the right. `build_mir1k_manifest.py` splits stems
and assigns source-clustered calibration/dev/test partitions.

```bash
python experiments/open_reward_safety/scripts/build_mir1k_manifest.py \
  --wav-dir /path/to/MIR-1K/Wavfile \
  --output-root /path/to/run/stems \
  --manifest /path/to/run/clean.jsonl

python experiments/open_reward_safety/scripts/generate_perturbation_pairs.py \
  --clean-manifest /path/to/run/clean.jsonl \
  --output-root /path/to/run/perturbations \
  --candidate-manifest /path/to/run/candidates.jsonl \
  --pair-manifest /path/to/run/pairs.jsonl
```

The pilot includes constant offsets, a four-second local shift, event-rate
resampling, contiguous gaps, noise/loop coverage attacks, and gain nuisance
pairs. Gain pairs are explicitly labelled invariant rather than degraded.

Each pair also declares `target_dimension` and `pair_kind`. Beat accuracy uses
only beat-targeted directional pairs; Coverage accuracy uses only silence-gap
pairs. Noise/loop fills remain overall guardrail attacks, and gain pairs report
absolute reward movement instead of directional accuracy. Reports include both
decided-only accuracy and strict accuracy, where a tie on a directional pair is
counted as a failure, with a source-clustered bootstrap interval for the latter.
Gain candidates record their clipping fraction; clipped gain pairs are retained
as diagnostics but excluded from the nuisance-invariance headline statistic.

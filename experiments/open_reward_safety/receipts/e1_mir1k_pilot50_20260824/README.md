# E1 MIR-1K Pilot Receipt

Date: 2026-08-24

This is an exploratory exact-stem qualification run, not a final benchmark
claim and not evidence that either reward improves music under GRPO.

## Frozen inputs

- dataset: MIR-1K stereo stems, left accompaniment and right vocal;
- public mirror revision: `d6b7565`;
- archive SHA-256: `9bd8610b4ca255fc6c8c5099fff915a12e8a21164310860e469efd43d09dadc1`;
- sample: 50 clips from 38 source songs;
- split unit: source song;
- candidates: 1,000;
- pairs: 950 total, including 600 beat-target directional pairs;
- Beat v5 backend: Madmom.

## Primary result

| reward | strict pair accuracy | source-cluster bootstrap 95% CI | tie rate |
| --- | ---: | ---: | ---: |
| Beat v2 | 47.7% | 37.0%--58.0% | 14.3% |
| Beat v5 | 53.2% | 43.5%--62.9% | 0.0% |

The paired difference was +5.5 percentage points for v5 (exact McNemar
p=0.0288). Held-out test accuracy was 41.2% for v2 and 56.6% for v5; that split
favored v5 by 15.4 points (p=0.00049), while the dev split favored v2. The
between-split reversal is part of the result and prevents a broad robustness
claim.

In the signed offset sweep, the unmodified `0 ms` candidate was the strict
per-clip optimum for only 8/50 clips under v2 and 12/50 under v5. Both rewards
frequently preferred a positive or negative offset; v2 also produced 16 tied
optima. This means the pilot exposes both scorer weakness and a label-contract
problem: natural vocal/accompaniment stems can contain expressive lead/lag, so
`clean > every shifted version` cannot be treated as unquestionable beat gold.

## Confidence result

The current v5 confidence measures detector evidence sufficiency, not pair
direction correctness. It accepted all 600 beat pairs and was exactly 1.0 for
96.2% of candidates. Its AURC was 0.515 versus a no-selection error of 0.468.
Absolute reward margin was also unstable across splits and cannot supply a
single global abstention threshold from this pilot.

## Decision

Online GRPO is blocked at E1. The next run must revise the confidence target,
replace the zero-lag assumption with beat-annotated or human-verified labels,
and replicate on more source songs before
the separator bridge or policy optimization is promoted.

`report_full.json` contains the complete machine-readable statistics. The two
figures show strict pair accuracy and v5 confidence risk-coverage.

## Artifact hashes

```text
report_full.json             754895f7cf99c2bda7e717eaceba2af5e756c16d43f6cc998101ed442e555fbf
e1_mir1k_pair_accuracy.png   0455e55144a9958657deec2c3fd333a5ed9542a49de03cb72863edfbaf1075e5
e1_v5_risk_coverage.png      636bd5a6aaef11685a7830ca3f4619e54e8afbd68c7a23a09aa572b8ee519d41
```

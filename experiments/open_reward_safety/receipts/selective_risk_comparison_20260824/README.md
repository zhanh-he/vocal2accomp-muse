# Selective-risk comparison receipt (2026-08-24)

This receipt asks a narrower question than ordinary ranking accuracy: when a
scorer accepts only its highest-confidence examples, does its error rate
decrease?

For examples sorted by descending confidence, coverage and risk at prefix `k`
are

```text
coverage(k) = k / N
risk(k) = errors among the first k examples / k
```

Lower risk and lower area under the risk-coverage curve (AURC) are better. The
right endpoint is the method's error rate without abstention. These two plots
use different prediction units and must not be overlaid or compared by raw
AURC: E1 uses pair-direction errors, while the controlled SongEval study uses
exact three-level ordering errors.

## MIR-1K E1 pair direction

The evaluation contains 600 beat-target directional pairs from 50 MIR-1K clips
clustered into 38 source songs. A tie is an error. Margin confidence is the
absolute score difference between the two candidates. V5 evidence confidence
is the minimum candidate evidence confidence within the pair.

| selection score | risk at 25% | risk at 50% | full risk | AURC |
| --- | ---: | ---: | ---: | ---: |
| Beat v2 margin | **31.3%** | **39.3%** | 52.3% | **0.387** |
| Beat v5 margin | 56.7% | 49.0% | 46.8% | 0.516 |
| V2/V5 raw mean margin | 38.0% | 44.0% | **44.7%** | 0.407 |
| V5 evidence confidence | 47.3% | 53.3% | 46.8% | 0.515 |

V2 has the worst full-coverage error because its 86 ties count as errors, but
its large nonzero margins select substantially safer pairs. V5 evidence
confidence is saturated and does not predict pair correctness. Large V5
margins are also anti-informative on this pilot. The untuned V2/V5 raw mean has
the best full-coverage error, but V2 alone remains better at the proposed 25%
selective gate.

## Beat + Coverage composite diagnostic

The actual Beat + Coverage reward arms can also be used as selection scores.
`raw` is an equal-weight arithmetic mean. `frozen std` uses equal weights after
dividing each pair margin by the component sample standard deviation fitted
once on the E1 calibration candidates (`v2=0.2381`, `v5=0.1728`,
`Coverage=0.0958`). Statistics are not recomputed per split or accepted subset.

| selection score | risk at 25% | risk at 50% | full risk | AURC |
| --- | ---: | ---: | ---: | ---: |
| Beat v2 + Coverage (raw) | 28.0% | 35.0% | 38.3% | 0.336 |
| Beat v5 + Coverage (raw) | 48.0% | 44.0% | 41.8% | 0.447 |
| Beat v2 + Coverage (frozen std) | **23.3%** | **30.0%** | **35.2%** | **0.279** |
| Beat v5 + Coverage (frozen std) | 44.0% | 39.3% | 39.0% | 0.394 |

This apparent gain is confounded. Coverage alone identifies 99.3% of constant
offset pairs and 100% of local-shift pairs in the requested direction, while it
is only 51.5% accurate on event-rate resampling. The offset/local-shift
generators create boundary padding, truncation, or gap artifacts that Coverage
can detect. The composite therefore partly recognizes the corruption pipeline
instead of rhythmic alignment. It is a useful benchmark-leakage diagnostic,
not evidence that Beat + Coverage is a qualified beat reward.

## SongEval controlled exact order

This study has 94 signed-direction groups from 47 songs. A group is correct
only when the complete order `clean > 70 ms > 120 ms` holds. Confidence is the
minimum of the two adjacent raw score gaps.

| selection score | risk at 25% | risk at 50% | full risk | AURC |
| --- | ---: | ---: | ---: | ---: |
| BeatReward-v2 | **0.0%** | 6.4% | 31.9% | 0.103 |
| BeatReward-v5 | 8.3% | 8.5% | **19.1%** | 0.094 |
| V2/V5 raw mean | **0.0%** | **2.1%** | **19.1%** | **0.056** |
| MuseCritic-Musicality | 75.0% | 72.3% | 70.2% | 0.769 |
| MuseCritic-Mean5 | 75.0% | 68.1% | 66.0% | 0.736 |

MuseCritic score gaps are not beat-confidence estimates: the largest-gap 10%
of groups are all incorrectly ordered. BeatReward-v2/v5 margins are useful for
selective acceptance on this controlled corruption, and their raw mean gives
the best selective ordering in this exploratory comparison.

## Composite definition and limits

`V2/V5 raw mean` is the frozen arithmetic mean `0.5 * v2 + 0.5 * v5`. No
weights were tuned. It is an exploratory baseline, not a promoted reward:

- the E1 result is a 50-clip pilot and its label contract is imperfect because
  natural vocal/accompaniment lead-lag can make `0 ms` non-optimal;
- the 47-song controlled result uses synthetic global offsets and no human beat
  labels;
- the composite was inspected on the same data used for this report and still
  requires source-held-out calibration, confidence intervals, and independent
  human validation;
- MuseCritic remains relevant as a holistic music-quality guardrail, even
  though it is not a beat-specific confidence model.

Decision: these plots justify developing a calibrated pair-error predictor and
testing the raw-mean ensemble on a frozen independent set. They do not qualify
the current V5 evidence confidence or any composite for online GRPO.

## Files

- `e1_margin_risk_coverage_comparison.png` and `.svg`
- `e1_beat_coverage_composite_risk_coverage.png` and `.svg`
- `songeval_exact_order_risk_coverage_comparison.png` and `.svg`
- `selective_risk_summary.csv`

The figures are generated by
`experiments/open_reward_safety/scripts/plot_selective_comparison.py`.

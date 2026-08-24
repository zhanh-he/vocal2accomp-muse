# Reward overoptimization and policy-induced shift

## Working thesis

Offline reward accuracy answers whether a scorer orders samples from one fixed
distribution. Online post-training asks whether the same ordering remains valid
for outputs produced by a moving policy that actively searches for high reward.
These are different properties.

The desired deployment object is not a universally invariant scalar reward;
that is generally unrealistic. The practical target is a reward with a measured
local validity region, explicit abstention, structured attack resistance, and a
known optimization radius in policy/KL space.

## Core sources

- [Defining and Characterizing Reward Hacking](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69278cd39ec257685c8d711d033c535a-Abstract-Conference.html)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- [Reward Model Ensembles Help Mitigate Overoptimization](https://openreview.net/forum?id=dcjtMYkpXx)
- [Confronting Reward Model Overoptimization with Constrained RLHF](https://openreview.net/forum?id=gkfUvn0fLU)
- [Scaling Laws for Reward Overoptimization in Direct Alignment](https://arxiv.org/abs/2406.02900)

## Experimental consequences

1. Plot proxy and independent outcomes by measured reference KL, not steps.
2. Re-evaluate pair direction on current-policy, same-prompt K-way groups.
3. Report high-margin wrong decisions, not only average pair accuracy.
4. Treat Coverage and output validity as constraints/guardrails as well as
   weighted-sum ablations.
5. Fit confidence to pair-direction correctness and report risk-coverage.
6. Match DPO and GRPO by actual policy drift before comparing robustness.

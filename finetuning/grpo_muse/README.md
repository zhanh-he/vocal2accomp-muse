# Muse GRPO Integration

This folder keeps the reward boundary independent from a particular RL trainer.
`SeparatedStemReward` accepts a decoded full-song audio path and a frozen
separator implementation. It returns a scalar, component scores, validity,
confidence, and separator provenance.

The online path is:

```text
Muse completion tokens
-> MuCodec decode
-> frozen separator
-> Beat v2/v5 + Coverage
-> optional calibration-frozen composite
-> group validity/range gate
-> ms-swift GRPO advantage normalization
```

The official MuseCritic repository already contains the ms-swift multi-turn
rollout and MuCodec service plumbing. This repository will add a thin ms-swift
ORM only after the exact-stem and separator gates pass. That keeps early scorer
experiments independent from a moving trainer implementation.

## Required online logs

- raw Beat and Coverage;
- pre-combination scaling mode and calibration artifact hash;
- scalar reward before group normalization;
- group std/range, accepted/skipped state, and component disagreement;
- Beat v5 evidence confidence and abstention reason;
- separator ID/checkpoint hash;
- token-mean and sequence-sum reference KL;
- entropy, ratio tails, clipping fraction, invalid output, silence, and repetition.

## Public detector policy

Only Madmom and Beat This are supported. Private detector names, commands, and
checkpoints are intentionally absent from this repository.

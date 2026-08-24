# vocal2accomp-muse

Public research code for qualifying music rewards under distribution shift and
testing them with open long-form song generators.

This repository is the open counterpart of a separate vocal-to-accompaniment
engineering project. It contains only general MIR rewards, public experiment
protocols, literature notes, and open-model adapters. It does not contain
company models, private datasets, internal checkpoints, or proprietary beat
detectors.

## Research question

A reward can rank an offline dataset well and still fail during online
post-training. The policy changes the generated-audio distribution, exposes
structured shortcuts, and can turn a small wrong pair ordering into a strong
GRPO update. We therefore evaluate rewards through:

```text
exact-stem perturbations
-> nuisance and attack tests
-> full-mix source-separation bridge
-> same-prompt current-policy ranking
-> actual-KL safe optimization radius
-> matched-KL GRPO/DPO comparison
```

## Repository layout

```text
mir/reward_function/          Open Beat v2/v5 and Coverage scorers
mir/composite_reward.py       Frozen term scaling and composite rewards
mir/reward_safety.py          Pair, risk-coverage, noise, and KL-radius metrics
finetuning/grpo_muse/         Generator-neutral separated-stem reward adapter
experiments/open_reward_safety/
                              Frozen public experiment protocol and scripts
experiments/open_reward_safety/cluster/
                              5090, Kaya, and Gadi distributed smoke launchers
literature/                   Source-linked public reading notes
MyNotes/                      Versioned research decisions and experiment logs
tests/                        Unit tests for scorer and safety math
```

The folder names intentionally resemble the companion engineering repository,
while the implementation boundaries are independent and publication-oriented.

## Current reward candidates

- **Beat v2**: Madmom beat F-measure between the vocal reference and the
  accompaniment, restricted to vocal-active regions.
- **Beat v5**: confidence-aware vocal-onset to accompaniment beat/subdivision
  grid alignment with global, local-mean, and local-tail terms.
- **Coverage**: fraction of accompaniment frames above a fixed RMS threshold.
- **Composite**: raw or calibration-frozen term scaling before scalar GRPO
  group normalization.
- **MuseCritic/SongEval**: holistic aesthetic controls. Their coherence and
  structural scores are not treated as beat ground truth.

## Public backbone

The first online study targets [Muse-0.6B](https://github.com/yuhui1038/Muse)
and follows the public [MuseCritic GRPO integration](https://github.com/WuqnEl/MuseCritic).
Muse emits a full-song mix, so Beat v2/v5 require a frozen vocal/accompaniment
separator. The separator bridge must pass an oracle-stem comparison before RL.

## Installation

Use Python 3.10. Keep the Muse/MuCodec generator, MuseCritic service, and MIR
scorers in separate environments so their dependency stacks remain auditable.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis,test]'
```

Madmom is optional and installed separately because its release predates modern
NumPy/Python packaging:

```bash
python -m pip install 'Cython<3' 'madmom==0.16.1'
```

Beat This is an optional open detector backend and is not vendored here.

## First experiment

The complete frozen protocol is in
[`experiments/open_reward_safety/README.md`](experiments/open_reward_safety/README.md).
The complete gate is designed to score exact iKala/MIR-1K stems under controlled
offset, drift, gap, click, codec, resampling, and gain transformations. The
executed pilot covers offset, local shift, event-rate resampling, gaps, coverage
attacks, and gain nuisance. Online GRPO starts only after scorer and separator
qualification.

The first 50-clip MIR-1K E1 pilot completed on 2026-08-24. Beat v5 ranked the
600 controlled beat pairs better than Beat v2, but reached only 53.2% strict
accuracy and the current evidence confidence failed risk-coverage selection.
This is a no-go for online GRPO with the current scorer/confidence contract; see
[`experiments/open_reward_safety/README.md`](experiments/open_reward_safety/README.md)
and the [committed E1 receipt](experiments/open_reward_safety/receipts/e1_mir1k_pilot50_20260824/README.md)
for the exact result and caveats.

## Reproducibility

Every result must include resolved configuration, source/model commits,
checkpoint hashes, dataset/audio manifests, component scores, and actual
reference-KL trajectories. Datasets and model weights are never committed.

## License and third-party data

Repository code is MIT licensed. Dataset, model, annotation, and external
checkpoint licenses remain governed by their original publishers. In
particular, some music datasets restrict use to non-commercial research.

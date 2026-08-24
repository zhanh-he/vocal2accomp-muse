# Repository split decision

Date: 2026-08-24

## Decision

`vocal2accomp-muse` is the public literature and experiment repository for
reward qualification with open song-generation models.

The companion engineering repository remains responsible for non-public model
integration. No private model code, generated audio, data paths, checkpoints,
or detector interfaces are copied here.

## Public scope

- Beat v2/v5 and Coverage as general audio-path scorers;
- Madmom and optional Beat This backends;
- Muse/MuseCritic/SongEval/CMI-RewardBench integration notes;
- exact-stem perturbation and separator-bridge experiments;
- frozen composite scaling, risk-coverage, attacks, and safe-KL metrics;
- public Muse LoRA-GRPO and later matched-KL DPO receipts.

## Structure

The repository keeps familiar ownership boundaries:

```text
mir/                    reward implementation
finetuning/grpo_muse/   online adapter boundary
experiments/            versioned public protocols
literature/             source-linked paper/data notes
MyNotes/                research decisions and experiment history
```

## First gate

The first remote execution is not GRPO. It is exact-stem qualification on iKala
and MIR-1K, followed by oracle-stem versus separated-stem ranking retention.

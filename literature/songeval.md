# SongEval: full-song aesthetics evaluation

- Paper: [arXiv:2505.10793](https://arxiv.org/abs/2505.10793)
- Code: [ASLP-lab/SongEval](https://github.com/ASLP-lab/SongEval)
- Dataset: [ASLP-lab/SongEval](https://huggingface.co/datasets/ASLP-lab/SongEval)

## Summary

SongEval contains 2,399 full-length generated songs with ratings from musically
experienced annotators across coherence, memorability, vocal naturalness,
structural clarity, and musicality. Its released toolkit predicts these five
perceptual dimensions from full-song audio.

## Role here

- holistic quality and long-form guardrail;
- source of diverse generated-song distributions;
- comparison point for learned music reward models;
- source for same-prompt human-audition candidates where licensing permits.

## What it cannot establish

SongEval has no isolated beat-correctness label. A high correlation with
coherence or musicality does not prove vocal/accompaniment phase alignment.
Beat claims require controlled timing corruption or a separately worded human
rhythmic-fit question.

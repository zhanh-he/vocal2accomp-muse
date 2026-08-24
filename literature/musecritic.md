# MuseCritic: multi-aspect song rewards through critiques

- Paper: [arXiv:2608.11755](https://arxiv.org/abs/2608.11755)
- Code: [WuqnEl/MuseCritic](https://github.com/WuqnEl/MuseCritic)
- Model: [WuqnEl/MuseCritic](https://huggingface.co/WuqnEl/MuseCritic)

## Summary

MuseCritic generates a five-part natural-language song critique and predicts
five continuous aesthetic scores: coherence, musicality, memorability,
structural clarity, and vocal naturalness. The critique is an intermediate
representation used by the reward predictor rather than only a displayed
explanation.

The paper reports in-domain SongEval regression/correlation, 733-pair
out-of-domain Music Arena preference accuracy, and downstream GRPO on Muse-0.6B.
The GRPO reward is the mean of the five dimensions.

## Role here

- published online-reward positive control;
- holistic-aesthetics guardrail for Beat-focused training;
- critique text for failure analysis;
- direct engineering reference for Muse/MuCodec/ms-swift integration.

## Construct boundary

Coherence and structural clarity describe long-range aesthetic organization;
they are not direct measurements of vocal/accompaniment beat alignment. Music
Arena labels overall preference, not beat accuracy. MuseCritic is therefore a
complement to an explicit MIR verifier, not its ground truth.

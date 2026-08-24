# Muse: reproducible long-form song generation

- Paper: [arXiv:2601.03973](https://arxiv.org/abs/2601.03973)
- Code: [yuhui1038/Muse](https://github.com/yuhui1038/Muse)
- Models: [bolshyC models](https://huggingface.co/bolshyC/models)
- Dataset: [bolshyC/Muse](https://huggingface.co/datasets/bolshyC/Muse)

## Summary

Muse is a Qwen3 + MuCodec long-form song-generation system released with public
training, inference, data-processing, and evaluation pipelines. The released
model family includes a 0.6B checkpoint suitable for resource-limited public
post-training studies.

The system learns audio-token prediction through single-stage supervised fine
tuning rather than an explicit rhythm objective. This makes it useful for
testing whether an external MIR reward can add a localized timing signal.

## Relevance

- public generator and data remove dependence on non-releasable outputs;
- Qwen/MuCodec structure is close to autoregressive music-token systems;
- the MuseCritic repository publishes a Muse-GRPO path;
- model size permits LoRA behavior pilots before larger confirmation runs.

## Limitation for Beat v2/v5

Muse generates a complete song mix, not independent vocal/accompaniment stems.
Beat rewards therefore depend on a frozen source-separation bridge. The bridge
must be audited on exact stems before online use, and results must be described
as `separated-stem` rather than native-stem reward transfer.

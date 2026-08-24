# Literature map

This folder tracks public sources that define the generator, reward baselines,
evaluation datasets, and reward-overoptimization hypotheses used by this repo.
Notes separate paper-reported evidence from repository hypotheses.

## Core sources

| Area | Source | Role in this repository |
| --- | --- | --- |
| Open generator | [Muse](muse.md) | Muse-0.6B public post-training backbone |
| Learned song reward | [MuseCritic](musecritic.md) | Mean5 GRPO positive control and holistic guardrail |
| Song aesthetics | [SongEval](songeval.md) | Five-dimensional full-song evaluation, not beat ground truth |
| Preference benchmark | [CMI-RewardBench](https://github.com/Haiwen-Xia/CMI-RewardBench) | Musicality/alignment sanity evaluation |
| Pair preferences | [Music Arena](https://huggingface.co/datasets/music-arena/music-arena-dataset) | Out-of-domain overall preference, not beat labels |
| Reward validity | [Reward overoptimization map](reward_overoptimization.md) | Why offline accuracy is insufficient |

## Exact-stem data

| Dataset | Public description | Intended use |
| --- | --- | --- |
| iKala | [Zenodo DOI](https://doi.org/10.5281/zenodo.3532213) | Primary 30-second vocal/accompaniment controlled benchmark |
| MIR-1K | [Zenodo DOI](https://doi.org/10.5281/zenodo.3532216) | External singer/song-cluster stress test |
| MUSDB18-HQ | [Official page](https://sigsep.github.io/datasets/musdb.html) | Optional full-length stem/separator bridge test |

Dataset and model licenses must be checked independently from this repository's
MIT license before downloading or redistributing any asset.

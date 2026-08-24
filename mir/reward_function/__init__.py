"""Open vocal/accompaniment reward functions."""

from .beat_v2 import BeatV2Config, BeatV2Result, MadmomBeatV2Scorer
from .beat_v5 import (
    BeatV5Config,
    BeatV5Result,
    BeatV5Scorer,
    combine_detector_results,
    score_onset_grid_events,
)
from .coverage import accompaniment_coverage, accompaniment_coverage_path

__all__ = [
    "BeatV2Config",
    "BeatV2Result",
    "MadmomBeatV2Scorer",
    "BeatV5Config",
    "BeatV5Result",
    "BeatV5Scorer",
    "accompaniment_coverage",
    "accompaniment_coverage_path",
    "combine_detector_results",
    "score_onset_grid_events",
]

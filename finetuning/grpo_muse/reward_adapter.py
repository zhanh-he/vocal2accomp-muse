"""Generator-neutral reward adapter for decoded full-song audio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from mir.composite_reward import FrozenComponentNormalizer, combine_components
from mir.reward_function import (
    BeatV5Scorer,
    MadmomBeatV2Scorer,
    accompaniment_coverage_path,
)


@dataclass(frozen=True)
class StemPair:
    vocal_path: Path
    accompaniment_path: Path
    separator_id: str
    separator_checkpoint_sha256: str


class Separator(Protocol):
    def separate(self, song_path: str | Path) -> StemPair:
        """Return immutable vocal/accompaniment stems for one decoded song."""


@dataclass(frozen=True)
class RewardResult:
    value: float
    valid: bool
    confidence: float | None
    components: Mapping[str, float]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class GatedGroup:
    values: tuple[float, ...]
    valid: tuple[bool, ...]
    accepted: bool
    reason: str


def gate_group_scores(
    values: Sequence[float],
    valid: Sequence[bool],
    *,
    min_valid: int = 2,
    min_range: float = 0.005,
) -> GatedGroup:
    """Neutralize invalid/indecisive candidates before framework GRPO scaling."""
    scores = np.asarray(values, dtype=float).reshape(-1)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if scores.shape != mask.shape or len(scores) < 2 or not np.all(np.isfinite(scores)):
        raise ValueError("group scores and validity must be aligned, finite, and size >= 2")
    reliable = scores[mask]
    neutral = float(np.mean(reliable)) if len(reliable) else 0.0
    if len(reliable) < min_valid:
        return GatedGroup((neutral,) * len(scores), (False,) * len(scores), False, "insufficient_valid")
    if float(np.ptp(reliable)) < min_range:
        return GatedGroup((neutral,) * len(scores), (False,) * len(scores), False, "insufficient_range")
    output = np.where(mask, scores, neutral)
    return GatedGroup(tuple(float(value) for value in output), tuple(bool(x) for x in mask), True, "accepted")


class SeparatedStemReward:
    """Separate a full-song mix once, then score public MIR reward components."""

    def __init__(
        self,
        separator: Separator,
        *,
        beat_version: str = "v5",
        beat_backend: str = "madmom",
        weights: Mapping[str, float] | None = None,
        scaling: str = "raw",
        normalizer: FrozenComponentNormalizer | None = None,
        coverage_floor: float | None = None,
        device: str = "cuda:0",
    ) -> None:
        if beat_version not in {"v2", "v5"}:
            raise ValueError("beat_version must be v2 or v5")
        if coverage_floor is not None and not 0.0 <= coverage_floor <= 1.0:
            raise ValueError("coverage_floor must be within [0, 1]")
        self.separator = separator
        self.beat_version = beat_version
        self.weights = dict(weights or {"beat": 1.0})
        self.scaling = scaling
        self.normalizer = normalizer
        self.coverage_floor = coverage_floor
        self.v2 = MadmomBeatV2Scorer() if beat_version == "v2" else None
        self.v5 = (
            BeatV5Scorer(backend=beat_backend, device=device)
            if beat_version == "v5"
            else None
        )

    def score_song(self, song_path: str | Path) -> RewardResult:
        stems = self.separator.separate(song_path)
        vocal_path = Path(stems.vocal_path).expanduser().resolve()
        accompaniment_path = Path(stems.accompaniment_path).expanduser().resolve()
        if self.beat_version == "v2":
            result = self.v2.score_paths(vocal_path, accompaniment_path)
            beat = result.score
            confidence = None
            valid = result.scorable
            beat_diagnostics = {
                "reference_beats": result.reference_beats,
                "accompaniment_beats": result.accompaniment_beats,
            }
        else:
            result = self.v5.score_paths(vocal_path, accompaniment_path)
            beat = result.score
            confidence = result.confidence
            valid = not result.abstain
            beat_diagnostics = {
                "reasons": result.reasons,
                "components": result.components,
                "detector": result.diagnostics,
            }
        if beat is None:
            return RewardResult(
                value=0.0,
                valid=False,
                confidence=confidence,
                components={"beat": 0.0},
                diagnostics={"reason": "unscorable_beat_reference"},
            )

        coverage = accompaniment_coverage_path(accompaniment_path)
        components = {"beat": float(beat), "coverage": float(coverage)}
        value = float(
            np.asarray(
                combine_components(
                    components,
                    self.weights,
                    normalizer=self.normalizer,
                    mode=self.scaling,
                )
            )
        )
        if self.coverage_floor is not None and coverage < self.coverage_floor:
            valid = False
        return RewardResult(
            value=value,
            valid=valid,
            confidence=confidence,
            components=components,
            diagnostics={
                "beat": beat_diagnostics,
                "separator_id": stems.separator_id,
                "separator_checkpoint_sha256": stems.separator_checkpoint_sha256,
                "coverage_floor": self.coverage_floor,
            },
        )

"""Madmom Beat v2: vocal-reference beat F-measure on vocal-active regions."""

from __future__ import annotations

import collections
import collections.abc
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BeatV2Config:
    segment_seconds: float = 90.0
    vocal_sample_rate: int = 22_050
    activity_hop: int = 512
    activity_frame: int = 2_048
    activity_relative_db: float = 35.0
    activity_floor_db: float = -50.0
    activity_pad_seconds: float = 0.15
    reference_cache_size: int = 4_096
    madmom_workers: int = 4


@dataclass(frozen=True)
class BeatV2Result:
    score: float | None
    reference_beats: int
    accompaniment_beats: int

    @property
    def scorable(self) -> bool:
        return self.score is not None


@dataclass(frozen=True)
class _VocalReference:
    active: np.ndarray
    sample_rate: int
    beats: np.ndarray


class MadmomBeatV2Scorer:
    """Compare accompaniment beats with vocal beats after vocal-activity masking."""

    def __init__(self, config: BeatV2Config | None = None):
        self.config = config or BeatV2Config()
        self._references: OrderedDict[str, _VocalReference] = OrderedDict()
        self._processors = None

    def score_paths(
        self,
        vocal_path: str | Path,
        accompaniment_path: str | Path,
    ) -> BeatV2Result:
        vocal_path = Path(vocal_path).expanduser().resolve()
        accompaniment_path = Path(accompaniment_path).expanduser().resolve()
        if not accompaniment_path.is_file():
            raise FileNotFoundError(f"missing accompaniment: {accompaniment_path}")
        reference = self._reference(vocal_path)
        accompaniment_beats = self._active_times(
            self._detect_path(accompaniment_path),
            reference.active,
            reference.sample_rate,
        )
        if not len(reference.beats):
            score = None
        elif not len(accompaniment_beats):
            score = 0.0
        else:
            import mir_eval

            score = float(mir_eval.beat.f_measure(reference.beats, accompaniment_beats))
        return BeatV2Result(score, len(reference.beats), len(accompaniment_beats))

    def _reference(self, vocal_path: Path) -> _VocalReference:
        path = str(vocal_path)
        cached = self._references.get(path)
        if cached is not None:
            self._references.move_to_end(path)
            return cached
        if not vocal_path.is_file():
            raise FileNotFoundError(f"missing vocal reference: {vocal_path}")

        import librosa

        vocal, sample_rate = librosa.load(
            vocal_path,
            sr=self.config.vocal_sample_rate,
            mono=True,
            duration=self.config.segment_seconds,
        )
        active = self._vocal_activity(vocal, sample_rate)
        beats = self._active_times(
            self._detect_path(vocal_path),
            active,
            sample_rate,
        )
        reference = _VocalReference(active=active, sample_rate=sample_rate, beats=beats)
        self._references[path] = reference
        self._references.move_to_end(path)
        while len(self._references) > self.config.reference_cache_size:
            self._references.popitem(last=False)
        return reference

    def _vocal_activity(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        cfg = self.config
        frame_count = max(1, int(math.ceil(len(waveform) / cfg.activity_hop)))
        required = (frame_count - 1) * cfg.activity_hop + cfg.activity_frame
        values = np.pad(waveform, (0, max(0, required - len(waveform))))
        rms = np.asarray(
            [
                np.sqrt(np.mean(values[start : start + cfg.activity_frame] ** 2) + 1e-12)
                for start in range(0, frame_count * cfg.activity_hop, cfg.activity_hop)
            ]
        )
        db = 20 * np.log10(np.maximum(rms, 1e-12))
        threshold = max(cfg.activity_floor_db, float(db.max() - cfg.activity_relative_db))
        active = db >= threshold
        width = max(1, int(round(cfg.activity_pad_seconds * sample_rate / cfg.activity_hop)))
        padded = np.pad(active.astype(np.uint8), (width, width))
        return np.convolve(padded, np.ones(2 * width + 1, dtype=np.uint8), mode="valid") > 0

    def _detect_path(self, audio_path: Path) -> np.ndarray:
        if self._processors is None:
            self._processors = self._make_processors()
        rnn, dbn = self._processors
        return np.asarray(dbn(rnn(str(audio_path))), dtype=float)

    def _make_processors(self):
        self._compatibility_shims()
        from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

        return (
            RNNBeatProcessor(num_threads=max(1, self.config.madmom_workers)),
            DBNBeatTrackingProcessor(fps=100),
        )

    @staticmethod
    def _compatibility_shims() -> None:
        for name in (
            "MutableSequence",
            "MutableMapping",
            "Mapping",
            "Sequence",
            "Iterable",
        ):
            if not hasattr(collections, name):
                setattr(collections, name, getattr(collections.abc, name))
        if not hasattr(np, "float"):
            np.float = float
        if not hasattr(np, "int"):
            np.int = int

    def _active_times(
        self,
        times: np.ndarray,
        active: np.ndarray,
        vocal_sample_rate: int,
    ) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        times = times[np.isfinite(times) & (times >= 0) & (times < self.config.segment_seconds)]
        if not len(times) or not len(active):
            return np.asarray([], dtype=float)
        indices = np.clip(
            (times * vocal_sample_rate / self.config.activity_hop).astype(int),
            0,
            len(active) - 1,
        )
        return times[active[indices]]

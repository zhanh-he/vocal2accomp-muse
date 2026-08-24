"""Confidence-aware vocal-onset to accompaniment-grid reward."""

from __future__ import annotations

import collections
import collections.abc
import math
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .audio import peak_normalize


@dataclass(frozen=True)
class BeatV5Config:
    segment_seconds: float = 90.0
    sigma_seconds: float = 0.055
    window_seconds: float = 12.0
    window_hop_seconds: float = 4.0
    grid_subdivision: int = 2
    global_weight: float = 0.40
    local_mean_weight: float = 0.30
    local_tail_weight: float = 0.30
    local_tail_percentile: float = 10.0
    onset_strength_power: float = 1.0
    onset_threshold: float = 0.15
    min_confidence: float = 0.25
    min_valid_window_fraction: float = 0.50
    beat_this_alpha: float = 0.50
    detector_disagreement_scale: float = 0.15
    reference_cache_size: int = 4_096
    madmom_workers: int = 4

    def __post_init__(self) -> None:
        if self.sigma_seconds <= 0:
            raise ValueError("sigma_seconds must be positive")
        if self.window_seconds <= 0 or self.window_hop_seconds <= 0:
            raise ValueError("window sizes must be positive")
        if self.grid_subdivision < 1:
            raise ValueError("grid_subdivision must be at least one")
        if self.detector_disagreement_scale <= 0:
            raise ValueError("detector_disagreement_scale must be positive")
        weights = (self.global_weight, self.local_mean_weight, self.local_tail_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("score weights must be non-negative and not all zero")


@dataclass(frozen=True)
class BeatV5Result:
    score: float
    confidence: float
    abstain: bool
    reasons: tuple[str, ...]
    components: Mapping[str, float]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class _Events:
    times: np.ndarray
    strengths: np.ndarray
    duration: float


def _grid(beats: np.ndarray, subdivision: int) -> np.ndarray:
    beats = np.unique(np.sort(np.asarray(beats, dtype=float)))
    if len(beats) < 2 or subdivision == 1:
        return beats
    parts = [beats]
    intervals = beats[1:] - beats[:-1]
    for index in range(1, subdivision):
        parts.append(beats[:-1] + intervals * index / subdivision)
    return np.unique(np.sort(np.concatenate(parts)))


def _distances(events: np.ndarray, grid: np.ndarray) -> np.ndarray:
    events = np.asarray(events, dtype=float)
    if not len(events) or not len(grid):
        return np.full(len(events), np.inf)
    indices = np.searchsorted(grid, events)
    result = np.full(len(events), np.inf)
    right = indices < len(grid)
    result[right] = np.abs(events[right] - grid[indices[right]])
    left = indices > 0
    result[left] = np.minimum(result[left], np.abs(events[left] - grid[indices[left] - 1]))
    return result


def _event_weights(strengths: np.ndarray, count: int, power: float) -> np.ndarray:
    if power <= 0 or len(strengths) != count:
        return np.ones(count, dtype=float)
    values = np.maximum(np.asarray(strengths, dtype=float), 0.0)
    positive = values[values > 0]
    if not len(positive):
        return np.ones(count, dtype=float)
    scale = float(np.percentile(positive, 90))
    return np.clip(values / max(scale, 1e-8), 0.05, 1.0) ** power


def score_onset_grid_events(
    vocal_onsets: Sequence[float],
    accompaniment_beats: Sequence[float],
    duration_seconds: float,
    *,
    onset_strengths: Sequence[float] = (),
    config: BeatV5Config | None = None,
) -> BeatV5Result:
    """Score vocal rhythmic events against a beat/subdivision grid."""
    cfg = config or BeatV5Config()
    onsets = np.asarray(vocal_onsets, dtype=float)
    onsets = np.unique(np.sort(onsets[np.isfinite(onsets) & (onsets >= 0)]))
    beats = np.asarray(accompaniment_beats, dtype=float)
    beats = np.unique(np.sort(beats[np.isfinite(beats) & (beats >= 0)]))
    duration = max(1.0, float(duration_seconds))
    grid = _grid(beats, cfg.grid_subdivision)
    distances = _distances(onsets, grid)
    qualities = (
        np.exp(-0.5 * (distances / cfg.sigma_seconds) ** 2)
        if len(distances)
        else np.empty(0, dtype=float)
    )
    weights = _event_weights(
        np.asarray(onset_strengths, dtype=float),
        len(onsets),
        cfg.onset_strength_power,
    )
    global_score = float(np.average(qualities, weights=weights)) if len(qualities) else 0.0

    if duration <= cfg.window_seconds:
        starts = np.asarray([0.0])
        window = duration
    else:
        starts = np.arange(0.0, duration - cfg.window_seconds + 1e-8, cfg.window_hop_seconds)
        window = cfg.window_seconds
    local_scores = []
    for start in starts:
        selected = (onsets >= start) & (onsets < start + window)
        if np.any(selected):
            local_scores.append(float(np.average(qualities[selected], weights=weights[selected])))
    valid_window_fraction = len(local_scores) / max(1, len(starts))
    local_mean = float(np.mean(local_scores)) if local_scores else 0.0
    local_tail = (
        float(np.percentile(local_scores, cfg.local_tail_percentile))
        if local_scores
        else 0.0
    )
    component_weights = np.asarray(
        [cfg.global_weight, cfg.local_mean_weight, cfg.local_tail_weight],
        dtype=float,
    )
    score = float(np.average([global_score, local_mean, local_tail], weights=component_weights))

    onset_support = min(1.0, len(onsets) / max(6.0, duration * 0.20))
    beat_support = min(1.0, len(beats) / max(8.0, duration * 0.50))
    confidence = float(
        np.clip(math.sqrt(onset_support * beat_support) * valid_window_fraction, 0.0, 1.0)
    )
    reasons = []
    if confidence < cfg.min_confidence:
        reasons.append("low_onset_or_beat_evidence")
    if valid_window_fraction < cfg.min_valid_window_fraction:
        reasons.append("sparse_vocal_activity")
    return BeatV5Result(
        score=score,
        confidence=confidence,
        abstain=bool(reasons),
        reasons=tuple(reasons),
        components={
            "global_soft_mean": global_score,
            "local_soft_mean": local_mean,
            "local_soft_tail": local_tail,
            "valid_window_fraction": float(valid_window_fraction),
            "onset_support": float(onset_support),
            "beat_support": float(beat_support),
        },
        diagnostics={
            "duration_seconds": duration,
            "vocal_onsets": int(len(onsets)),
            "accompaniment_beats": int(len(beats)),
            "grid_points": int(len(grid)),
            "median_distance_seconds": (
                float(np.median(distances)) if len(distances) else math.nan
            ),
            "window_count": int(len(starts)),
            "valid_window_count": int(len(local_scores)),
            "grid_subdivision": cfg.grid_subdivision,
            "sigma_seconds": cfg.sigma_seconds,
        },
    )


def combine_detector_results(
    first: BeatV5Result,
    second: BeatV5Result,
    *,
    second_alpha: float = 0.50,
    min_confidence: float = 0.0,
    disagreement_scale: float = 0.15,
) -> BeatV5Result:
    """Confidence-weight two open detector scores and penalize disagreement."""
    if second_alpha < 0 or disagreement_scale <= 0:
        raise ValueError("invalid detector combination parameters")
    first_weight = 0.0 if first.abstain else first.confidence
    second_weight = 0.0 if second.abstain else second_alpha * second.confidence
    denominator = first_weight + second_weight
    score = (
        (first_weight * first.score + second_weight * second.score) / denominator
        if denominator
        else 0.0
    )
    if first_weight and second_weight:
        evidence = (first.confidence + second_alpha * second.confidence) / (1.0 + second_alpha)
        agreement = math.exp(-abs(first.score - second.score) / disagreement_scale)
        confidence = evidence * agreement
    elif first_weight:
        confidence, agreement = first.confidence, 1.0
    elif second_weight:
        confidence, agreement = second.confidence, 1.0
    else:
        confidence, agreement = 0.0, 0.0
    reasons = () if denominator and confidence >= min_confidence else (
        "detector_disagreement" if denominator else "no_reliable_detector",
    )
    return BeatV5Result(
        score=float(score),
        confidence=float(confidence),
        abstain=bool(reasons),
        reasons=reasons,
        components={
            "madmom_score": first.score,
            "beat_this_score": second.score,
            "madmom_weight": first_weight,
            "beat_this_weight": second_weight,
            "detector_agreement": agreement,
        },
        diagnostics={"madmom": first.diagnostics, "beat_this": second.diagnostics},
    )


class BeatV5Scorer:
    """Path-based public scorer using Madmom and optionally Beat This."""

    BACKENDS = {"madmom", "beat_this", "ensemble"}

    def __init__(
        self,
        config: BeatV5Config | None = None,
        *,
        backend: str = "madmom",
        device: str = "cuda:0",
        beat_this_checkpoint: str = "final0",
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {sorted(self.BACKENDS)}")
        self.config = config or BeatV5Config()
        self.backend = backend
        self.device = str(device)
        self.beat_this_checkpoint = beat_this_checkpoint
        self._vocal_cache: OrderedDict[str, _Events] = OrderedDict()
        self._madmom_processors = None
        self._beat_this_processors = None

    def score_paths(
        self,
        vocal_path: str | Path,
        accompaniment_path: str | Path,
    ) -> BeatV5Result:
        vocal = self._vocal_events(vocal_path)
        accompaniment_path = Path(accompaniment_path).expanduser().resolve()
        if not accompaniment_path.is_file():
            raise FileNotFoundError(f"missing accompaniment: {accompaniment_path}")
        results = {}
        if self.backend in {"madmom", "ensemble"}:
            results["madmom"] = self._score_events(vocal, self._madmom_beats(accompaniment_path))
        if self.backend in {"beat_this", "ensemble"}:
            results["beat_this"] = self._score_events(
                vocal,
                self._beat_this_beats(accompaniment_path),
            )
        if self.backend == "ensemble":
            return combine_detector_results(
                results["madmom"],
                results["beat_this"],
                second_alpha=self.config.beat_this_alpha,
                min_confidence=self.config.min_confidence,
                disagreement_scale=self.config.detector_disagreement_scale,
            )
        return results[self.backend]

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

    def _madmom(self):
        if self._madmom_processors is None:
            self._compatibility_shims()
            from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
            from madmom.features.onsets import OnsetPeakPickingProcessor, RNNOnsetProcessor

            self._madmom_processors = (
                RNNOnsetProcessor(),
                OnsetPeakPickingProcessor(fps=100, threshold=self.config.onset_threshold),
                RNNBeatProcessor(num_threads=max(1, self.config.madmom_workers)),
                DBNBeatTrackingProcessor(fps=100),
            )
        return self._madmom_processors

    def _vocal_events(self, vocal_path: str | Path) -> _Events:
        path = str(Path(vocal_path).expanduser().resolve())
        cached = self._vocal_cache.get(path)
        if cached is not None:
            self._vocal_cache.move_to_end(path)
            return cached
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing vocal reference: {path}")
        from madmom.audio.signal import Signal

        onset_rnn, peak_picker, _, _ = self._madmom()
        signal = Signal(path, sample_rate=44_100, num_channels=1, norm=True)
        activation = np.asarray(onset_rnn(signal), dtype=float)
        times = np.asarray(peak_picker(activation), dtype=float)
        indices = np.clip(np.rint(times * 100).astype(int), 0, max(0, len(activation) - 1))
        strengths = activation[indices] if len(activation) and len(times) else np.empty(0)
        duration = min(self.config.segment_seconds, len(activation) / 100.0)
        selected = times < duration
        events = _Events(times[selected], strengths[selected], duration)
        self._vocal_cache[path] = events
        self._vocal_cache.move_to_end(path)
        while len(self._vocal_cache) > self.config.reference_cache_size:
            self._vocal_cache.popitem(last=False)
        return events

    def _madmom_beats(self, audio_path: Path) -> _Events:
        import soundfile as sf
        from madmom.audio.signal import Signal

        _, _, beat_rnn, beat_dbn = self._madmom()
        waveform, sample_rate = sf.read(audio_path, always_2d=False, dtype="float32")
        waveform = peak_normalize(waveform)
        with tempfile.NamedTemporaryFile(suffix=".wav") as normalized:
            sf.write(normalized.name, waveform, int(sample_rate), subtype="PCM_16")
            activation = np.asarray(
                beat_rnn(Signal(normalized.name, sample_rate=44_100, num_channels=1, norm=False)),
                dtype=float,
            )
        times = np.asarray(beat_dbn(activation), dtype=float)
        indices = np.clip(np.rint(times * 100).astype(int), 0, max(0, len(activation) - 1))
        strengths = activation[indices] if len(activation) and len(times) else np.empty(0)
        duration = min(self.config.segment_seconds, len(activation) / 100.0)
        selected = times < duration
        return _Events(times[selected], strengths[selected], duration)

    def _load_beat_this(self):
        if self._beat_this_processors is None:
            from beat_this.inference import Audio2Frames
            from beat_this.model.postprocessor import Postprocessor

            frames = Audio2Frames(
                self.beat_this_checkpoint,
                self.device,
                self.device.startswith("cuda"),
            )
            self._beat_this_processors = frames, Postprocessor(type="minimal")
        return self._beat_this_processors

    def _beat_this_beats(self, audio_path: Path) -> _Events:
        import soundfile as sf
        from scipy.special import expit

        waveform, sample_rate = sf.read(audio_path, always_2d=False, dtype="float32")
        waveform = peak_normalize(waveform)
        frames, postprocessor = self._load_beat_this()
        beat_logits, downbeat_logits = frames(waveform, int(sample_rate))
        beats, _ = postprocessor(beat_logits, downbeat_logits)
        activation = expit(beat_logits.detach().float().cpu().numpy())
        times = np.asarray(beats, dtype=float)
        indices = np.clip(np.rint(times * 50).astype(int), 0, max(0, len(activation) - 1))
        strengths = activation[indices] if len(activation) and len(times) else np.empty(0)
        duration = min(self.config.segment_seconds, len(activation) / 50.0)
        selected = times < duration
        return _Events(times[selected], strengths[selected], duration)

    def _score_events(self, vocal: _Events, accompaniment: _Events) -> BeatV5Result:
        duration = min(vocal.duration, accompaniment.duration, self.config.segment_seconds)
        vocal_selected = vocal.times < duration
        accompaniment_selected = accompaniment.times < duration
        return score_onset_grid_events(
            vocal.times[vocal_selected],
            accompaniment.times[accompaniment_selected],
            duration,
            onset_strengths=vocal.strengths[vocal_selected],
            config=self.config,
        )

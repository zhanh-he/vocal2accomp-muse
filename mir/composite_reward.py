"""Frozen component scaling and additive composite reward construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ComponentStats:
    mean: float
    std: float
    median: float
    mad: float
    count: int

    @classmethod
    def fit(cls, values: Sequence[float], *, min_scale: float = 1e-6) -> "ComponentStats":
        array = np.asarray(values, dtype=float).reshape(-1)
        if not len(array) or not np.all(np.isfinite(array)):
            raise ValueError("calibration values must be non-empty and finite")
        mean = float(np.mean(array))
        std = max(float(np.std(array, ddof=1)) if len(array) > 1 else 0.0, min_scale)
        median = float(np.median(array))
        mad = max(float(np.median(np.abs(array - median))), min_scale)
        return cls(mean=mean, std=std, median=median, mad=mad, count=len(array))

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class FrozenComponentNormalizer:
    """Statistics fitted on calibration data and never updated by online batches."""

    def __init__(self, stats: Mapping[str, ComponentStats]):
        self.stats = dict(stats)
        if not self.stats:
            raise ValueError("at least one component statistic is required")

    @classmethod
    def fit(cls, calibration: Mapping[str, Sequence[float]]) -> "FrozenComponentNormalizer":
        return cls({name: ComponentStats.fit(values) for name, values in calibration.items()})

    def transform(self, name: str, values, *, mode: str):
        array = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        if mode == "raw":
            return array
        if name not in self.stats:
            raise KeyError(f"missing frozen calibration statistics for {name}")
        stats = self.stats[name]
        if mode == "frozen_std":
            return (array - stats.mean) / stats.std
        if mode == "frozen_robust":
            return (array - stats.median) / (1.4826 * stats.mad)
        raise ValueError(f"unsupported frozen scaling mode: {mode}")


def online_group_zscore(values, *, epsilon: float = 1e-6):
    """Distribution-dependent diagnostic; not a default deployment transform."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("online group values must have shape [groups, candidates>=2]")
    if not np.all(np.isfinite(array)):
        raise ValueError("online group values must be finite")
    centered = array - np.mean(array, axis=1, keepdims=True)
    scale = np.std(array, axis=1, ddof=1, keepdims=True)
    return centered / np.maximum(scale, epsilon)


def combine_components(
    components: Mapping[str, object],
    weights: Mapping[str, float],
    *,
    normalizer: FrozenComponentNormalizer | None = None,
    mode: str = "raw",
):
    """Return an additive composite without silently dropping missing terms."""
    missing = set(weights) - set(components)
    if missing:
        raise KeyError(f"missing reward components: {sorted(missing)}")
    active = {name: float(weight) for name, weight in weights.items() if weight > 0}
    if not active or any(not np.isfinite(weight) for weight in active.values()):
        raise ValueError("reward weights must contain finite positive mass")
    total = sum(active.values())
    arrays = []
    normalized_weights = []
    for name, weight in active.items():
        values = np.asarray(components[name], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")
        if mode == "raw":
            transformed = values
        else:
            if normalizer is None:
                raise ValueError(f"{mode} requires a frozen normalizer")
            transformed = normalizer.transform(name, values, mode=mode)
        arrays.append(transformed)
        normalized_weights.append(weight / total)
    broadcast = np.broadcast_arrays(*arrays)
    return sum(weight * values for weight, values in zip(normalized_weights, broadcast))

"""Shared audio conversion helpers."""

from __future__ import annotations

import numpy as np


def mono(waveform) -> np.ndarray:
    """Convert common tensor/array channel layouts into one finite float array."""
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().float().cpu().numpy()
    values = np.nan_to_num(np.asarray(waveform, dtype=np.float32)).squeeze()
    if values.ndim == 0:
        return np.asarray([], dtype=np.float32)
    if values.ndim > 1:
        if values.shape[-1] <= 8 and values.shape[0] > values.shape[-1]:
            values = values.reshape(-1, values.shape[-1]).mean(axis=-1)
        elif values.shape[0] <= 8:
            values = values.reshape(values.shape[0], -1).mean(axis=0)
        else:
            raise ValueError(f"cannot infer audio channel axis from shape {values.shape}")
    return values


def peak_normalize(waveform, *, target_peak: float = 0.95) -> np.ndarray:
    values = mono(waveform)
    peak = float(np.max(np.abs(values))) if len(values) else 0.0
    if peak > 1e-8:
        values = values * (target_peak / peak)
    return np.clip(values, -1.0, 1.0)

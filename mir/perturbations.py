"""Deterministic waveform interventions for reward qualification."""

from __future__ import annotations

from fractions import Fraction

import numpy as np


def _mono(waveform) -> np.ndarray:
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("perturbations require a mono waveform")
    return values.copy()


def fixed_length(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) >= length:
        return np.asarray(values[:length], dtype=np.float32)
    return np.pad(np.asarray(values, dtype=np.float32), (0, length - len(values)))


def constant_shift(waveform, shift_samples: int) -> np.ndarray:
    """Translate audio in time with zero fill and no circular wrap."""
    values = _mono(waveform)
    output = np.zeros_like(values)
    if shift_samples == 0:
        return values
    if abs(shift_samples) >= len(values):
        return output
    if shift_samples > 0:
        output[shift_samples:] = values[:-shift_samples]
    else:
        amount = abs(shift_samples)
        output[:-amount] = values[amount:]
    return output


def local_shift(
    waveform,
    *,
    start_sample: int,
    length_samples: int,
    shift_samples: int,
) -> np.ndarray:
    """Shift one fixed window while leaving the surrounding audio unchanged."""
    values = _mono(waveform)
    if start_sample < 0 or length_samples < 1 or start_sample >= len(values):
        raise ValueError("invalid local-shift window")
    stop = min(len(values), start_sample + length_samples)
    output = values.copy()
    output[start_sample:stop] = constant_shift(values[start_sample:stop], shift_samples)
    return output


def tempo_resample(waveform, ratio: float) -> np.ndarray:
    """Change event rate by resampling and restore the original output length."""
    from scipy.signal import resample_poly

    values = _mono(waveform)
    if ratio <= 0:
        raise ValueError("tempo ratio must be positive")
    fraction = Fraction(float(ratio)).limit_denominator(1_000)
    changed = resample_poly(values, fraction.denominator, fraction.numerator)
    return fixed_length(changed, len(values))


def contiguous_gap(
    waveform,
    *,
    start_sample: int,
    length_samples: int,
    fill: str = "silence",
    noise_db: float = -35.0,
    seed: int = 0,
) -> np.ndarray:
    """Replace a contiguous region with silence, noise, or a preceding loop."""
    values = _mono(waveform)
    if start_sample < 0 or length_samples < 1 or start_sample >= len(values):
        raise ValueError("invalid gap window")
    stop = min(len(values), start_sample + length_samples)
    width = stop - start_sample
    output = values.copy()
    if fill == "silence":
        replacement = np.zeros(width, dtype=np.float32)
    elif fill == "noise":
        peak = max(float(np.max(np.abs(values))), 1e-8)
        scale = peak * 10.0 ** (noise_db / 20.0)
        replacement = np.random.default_rng(seed).normal(0.0, scale, width).astype(np.float32)
    elif fill == "loop":
        source_stop = start_sample
        source_start = max(0, source_stop - width)
        source = values[source_start:source_stop]
        if not len(source):
            replacement = np.zeros(width, dtype=np.float32)
        else:
            replacement = np.resize(source, width).astype(np.float32)
    else:
        raise ValueError("fill must be silence, noise, or loop")
    output[start_sample:stop] = replacement
    return output


def gain(waveform, gain_db: float) -> np.ndarray:
    return _mono(waveform) * np.float32(10.0 ** (gain_db / 20.0))

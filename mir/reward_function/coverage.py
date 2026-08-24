"""Reference-free accompaniment activity coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import mono


def accompaniment_coverage(
    waveform,
    sample_rate: int,
    *,
    threshold_db: float = -40.0,
    n_fft: int = 2_048,
    hop_length: int = 512,
) -> float:
    """Fraction of STFT RMS frames above an absolute dBFS-like threshold."""
    del sample_rate  # Retained in the API so alternate frame recipes can use it.
    values = mono(waveform)
    if not len(values):
        return 0.0
    if len(values) < n_fft:
        values = np.pad(values, (0, n_fft - len(values)))

    import librosa

    spectrum = np.abs(librosa.stft(values, n_fft=n_fft, hop_length=hop_length))
    rms_db = 20 * np.log10(librosa.feature.rms(S=spectrum)[0] + 1e-10)
    return float(np.mean(rms_db > threshold_db))


def accompaniment_coverage_path(path: str | Path, *, threshold_db: float = -40.0) -> float:
    import soundfile as sf

    waveform, sample_rate = sf.read(
        Path(path).expanduser().resolve(),
        always_2d=False,
        dtype="float32",
    )
    return accompaniment_coverage(waveform, int(sample_rate), threshold_db=threshold_db)

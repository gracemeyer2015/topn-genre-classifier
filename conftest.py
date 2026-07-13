import numpy as np
import pytest
import soundfile as sf

SR = 22050


@pytest.fixture
def write_sine_wav():
    """Return a helper that writes a synthetic 440 Hz sine .wav for tests."""
    def _write(path, duration_sec=0.5, sr=SR, channels=1, amplitude=0.3):
        n_samples = int(duration_sec * sr)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        tone = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        if channels == 2:
            tone = np.stack([tone, tone], axis=1)
        sf.write(str(path), tone, sr)
    return _write

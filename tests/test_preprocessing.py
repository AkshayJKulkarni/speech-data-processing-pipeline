"""
Unit and integration tests for the preprocessing module.

Test strategy:
  - validator tests: pure logic, zero audio I/O
  - audio_utils tests: synthetic numpy arrays, no real files
  - processor tests: write a real WAV to a temp dir, verify output
    (this is the only test that actually calls librosa/soundfile)
"""

import os
import numpy as np
import pytest
import soundfile as sf
import tempfile
from pathlib import Path

from src.preprocessing.validator import (
    validate_input_file,
    validate_output_dir,
    validate_audio_config,
)
from src.preprocessing.audio_utils import (
    to_mono,
    resample,
    normalize_peak,
    save_wav,
)
from src.preprocessing.processor import process
from src.utils.exceptions import PreprocessingError


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_sine_wave(
    freq: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 44100,
    channels: int = 1,
) -> np.ndarray:
    """Creates a synthetic sine wave as a float32 numpy array."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
    if channels == 2:
        return np.stack([wave, wave])  # shape: (2, samples)
    return wave  # shape: (samples,)


def write_temp_wav(audio: np.ndarray, sample_rate: int, suffix: str = ".wav") -> Path:
    """Writes a numpy array to a temporary WAV file and returns its path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        tmp_path = Path(f.name)

    # soundfile expects (samples,) or (samples, channels)
    data = audio.T if audio.ndim == 2 else audio
    sf.write(str(tmp_path), data, sample_rate, subtype="PCM_16")
    return tmp_path


# ── validate_input_file ───────────────────────────────────────────────────────

class TestValidateInputFile:
    def test_valid_wav_passes(self):
        audio = make_sine_wave()
        tmp = write_temp_wav(audio, 44100)
        try:
            validate_input_file(tmp)   # should not raise
        finally:
            os.unlink(tmp)

    def test_missing_file_raises(self):
        with pytest.raises(PreprocessingError, match="not found"):
            validate_input_file(Path("/nonexistent/audio.wav"))

    def test_unsupported_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            tmp = Path(f.name)
        try:
            with pytest.raises(PreprocessingError, match="Unsupported input format"):
                validate_input_file(tmp)
        finally:
            os.unlink(tmp)


# ── validate_audio_config ─────────────────────────────────────────────────────

class TestValidateAudioConfig:
    def _valid_config(self):
        return {"target_sample_rate": 16000, "target_channels": 1, "output_format": "wav"}

    def test_valid_config_passes(self):
        validate_audio_config(self._valid_config())

    def test_missing_key_raises(self):
        cfg = self._valid_config()
        del cfg["target_sample_rate"]
        with pytest.raises(PreprocessingError, match="Missing audio config keys"):
            validate_audio_config(cfg)

    def test_invalid_sample_rate_raises(self):
        cfg = self._valid_config()
        cfg["target_sample_rate"] = -1
        with pytest.raises(PreprocessingError, match="target_sample_rate must be positive"):
            validate_audio_config(cfg)

    def test_invalid_channels_raises(self):
        cfg = self._valid_config()
        cfg["target_channels"] = 3
        with pytest.raises(PreprocessingError, match="target_channels must be"):
            validate_audio_config(cfg)


# ── to_mono ───────────────────────────────────────────────────────────────────

class TestToMono:
    def test_stereo_becomes_1d(self):
        stereo = make_sine_wave(channels=2)   # shape: (2, samples)
        mono = to_mono(stereo)
        assert mono.ndim == 1

    def test_mono_unchanged(self):
        mono_in = make_sine_wave(channels=1)  # shape: (samples,)
        mono_out = to_mono(mono_in)
        assert mono_out.ndim == 1
        np.testing.assert_array_equal(mono_in, mono_out)

    def test_stereo_energy_preserved(self):
        # Averaging identical L+R channels should give the same signal
        stereo = make_sine_wave(channels=2)
        mono = to_mono(stereo)
        expected = stereo[0]  # L == R, so average == L
        np.testing.assert_allclose(mono, expected, atol=1e-5)


# ── resample ──────────────────────────────────────────────────────────────────

class TestResample:
    def test_downsample_changes_length(self):
        audio = make_sine_wave(sample_rate=44100, duration=1.0)
        resampled = resample(audio, orig_sr=44100, target_sr=16000)
        assert len(resampled) == pytest.approx(16000, abs=10)

    def test_same_sr_returns_unchanged(self):
        audio = make_sine_wave(sample_rate=16000)
        out = resample(audio, orig_sr=16000, target_sr=16000)
        np.testing.assert_array_equal(audio, out)

    def test_upsample_changes_length(self):
        audio = make_sine_wave(sample_rate=8000, duration=1.0)
        resampled = resample(audio, orig_sr=8000, target_sr=16000)
        assert len(resampled) == pytest.approx(16000, abs=10)


# ── normalize_peak ────────────────────────────────────────────────────────────

class TestNormalizePeak:
    def test_peak_becomes_1(self):
        audio = make_sine_wave() * 0.3   # quiet signal
        normalized = normalize_peak(audio)
        assert pytest.approx(np.max(np.abs(normalized)), abs=1e-5) == 1.0

    def test_already_normalized_unchanged(self):
        audio = make_sine_wave()
        audio = audio / np.max(np.abs(audio))  # pre-normalize
        out = normalize_peak(audio)
        np.testing.assert_allclose(out, audio, atol=1e-5)

    def test_silent_audio_returned_unchanged(self):
        silent = np.zeros(16000, dtype=np.float32)
        out = normalize_peak(silent)
        np.testing.assert_array_equal(out, silent)


# ── process() — integration ───────────────────────────────────────────────────

class TestProcess:
    """
    Integration tests: writes a real WAV, calls process(), checks output.
    These are the only tests that hit librosa and soundfile together.
    """

    def _audio_config(self):
        return {"target_sample_rate": 16000, "target_channels": 1, "output_format": "wav"}

    def test_wav_processed_correctly(self):
        audio = make_sine_wave(sample_rate=44100, channels=2, duration=1.0)
        src = write_temp_wav(audio, 44100)

        with tempfile.TemporaryDirectory() as out_dir:
            result = process(str(src), out_dir, self._audio_config())
            out_path = Path(result)

            assert out_path.exists()
            assert out_path.suffix == ".wav"

            # Verify output properties
            out_audio, out_sr = sf.read(str(out_path))
            assert out_sr == 16000
            assert out_audio.ndim == 1         # mono

        os.unlink(src)

    def test_idempotency_skips_reprocessing(self):
        """Calling process() twice must return same path without re-running."""
        audio = make_sine_wave(sample_rate=44100)
        src = write_temp_wav(audio, 44100)

        with tempfile.TemporaryDirectory() as out_dir:
            first  = process(str(src), out_dir, self._audio_config())
            mtime1 = Path(first).stat().st_mtime

            second = process(str(src), out_dir, self._audio_config())
            mtime2 = Path(second).stat().st_mtime

            assert first == second
            assert mtime1 == mtime2   # file was NOT rewritten

        os.unlink(src)

    def test_missing_input_raises(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with pytest.raises(PreprocessingError, match="not found"):
                process("/nonexistent/audio.wav", out_dir, self._audio_config())

    def test_output_dir_created_if_missing(self):
        audio = make_sine_wave(sample_rate=16000)
        src = write_temp_wav(audio, 16000)

        with tempfile.TemporaryDirectory() as base:
            new_dir = os.path.join(base, "deeply", "nested", "dir")
            result = process(str(src), new_dir, self._audio_config())
            assert Path(result).exists()

        os.unlink(src)

    def test_output_never_overwrites_input(self):
        """Input file must never be modified."""
        audio = make_sine_wave(sample_rate=44100)
        src = write_temp_wav(audio, 44100)
        src_mtime = src.stat().st_mtime

        with tempfile.TemporaryDirectory() as out_dir:
            process(str(src), out_dir, self._audio_config())

        assert src.stat().st_mtime == src_mtime   # input untouched
        os.unlink(src)

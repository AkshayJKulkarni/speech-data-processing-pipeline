"""
Unit tests for the transcription module.

Test strategy:
─────────────
- All tests mock whisper.load_model so NO real model is ever loaded.
  This keeps the test suite fast (< 1 second) and runnable without GPU.
- Registry tests verify caching behavior directly on WhisperModelRegistry.
- transcribe() tests verify the full call path using a real temp WAV file
  and a patched model that returns controlled output.
- No test touches the network or disk beyond a temp WAV fixture.
"""

import os
import time
import numpy as np
import pytest
import soundfile as sf
import tempfile
from pathlib import Path
from dataclasses import asdict
from unittest.mock import MagicMock, patch

from src.inference.transcriber import (
    TranscriptSegment,
    TranscriptResult,
    transcribe,
    _validate_audio_path,
    _validate_config,
    _resolve_device,
    _map_segments,
)
from src.inference._model_registry import WhisperModelRegistry, VALID_MODEL_SIZES
from src.utils.exceptions import TranscriptionError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_wav(tmp_path: Path) -> Path:
    """Creates a real 1-second 16kHz mono WAV file in a temp directory."""
    t = np.linspace(0, 1.0, 16000, endpoint=False).astype(np.float32)
    audio = np.sin(2 * np.pi * 440 * t)
    wav_path = tmp_path / "test_audio.wav"
    sf.write(str(wav_path), audio, 16000, subtype="PCM_16")
    return wav_path


@pytest.fixture
def valid_config() -> dict:
    return {"model_size": "base", "language": "en", "device": "cpu"}


@pytest.fixture
def mock_whisper_output() -> dict:
    """Mimics the dict that whisper model.transcribe() returns."""
    return {
        "language": "en",
        "text": " Hello everyone. This is a test.",
        "segments": [
            {"start": 0.0,  "end": 1.5, "text": " Hello everyone."},
            {"start": 1.5,  "end": 3.2, "text": " This is a test."},
        ],
    }


@pytest.fixture
def mock_whisper_model(mock_whisper_output: dict) -> MagicMock:
    """Returns a mock Whisper model whose .transcribe() returns controlled output."""
    model = MagicMock()
    model.transcribe.return_value = mock_whisper_output
    return model


# ── TranscriptSegment ─────────────────────────────────────────────────────────

class TestTranscriptSegment:
    def test_duration_calculation(self):
        seg = TranscriptSegment(start=1.0, end=3.5, text="hello")
        assert seg.duration() == pytest.approx(2.5, abs=0.001)

    def test_fields_stored_correctly(self):
        seg = TranscriptSegment(start=0.0, end=2.0, text="test text")
        assert seg.start == 0.0
        assert seg.end   == 2.0
        assert seg.text  == "test text"


# ── TranscriptResult ──────────────────────────────────────────────────────────

class TestTranscriptResult:
    def _make_result(self) -> TranscriptResult:
        return TranscriptResult(
            language="en",
            text="Hello world. How are you?",
            segments=[
                TranscriptSegment(0.0, 1.2, "Hello world."),
                TranscriptSegment(1.2, 2.8, "How are you?"),
            ],
        )

    def test_segment_count(self):
        assert self._make_result().segment_count == 2

    def test_duration_is_last_segment_end(self):
        assert self._make_result().duration == pytest.approx(2.8)

    def test_empty_segments_duration_zero(self):
        result = TranscriptResult(language="en", text="")
        assert result.duration == 0.0

    def test_to_dict_is_serializable(self):
        d = self._make_result().to_dict()
        assert isinstance(d, dict)
        assert "language" in d
        assert "text" in d
        assert isinstance(d["segments"], list)
        assert d["segments"][0]["start"] == 0.0
        assert d["segments"][0]["text"]  == "Hello world."

    def test_to_dict_matches_asdict(self):
        result = self._make_result()
        assert result.to_dict() == asdict(result)


# ── _validate_audio_path ──────────────────────────────────────────────────────

class TestValidateAudioPath:
    def test_valid_wav_passes(self, temp_wav: Path):
        _validate_audio_path(temp_wav)  # no exception

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(TranscriptionError, match="not found"):
            _validate_audio_path(tmp_path / "missing.wav")

    def test_non_wav_raises(self, tmp_path: Path):
        mp3 = tmp_path / "audio.mp3"
        mp3.write_bytes(b"fake")
        with pytest.raises(TranscriptionError, match="expects a .wav"):
            _validate_audio_path(mp3)


# ── _validate_config ──────────────────────────────────────────────────────────

class TestValidateConfig:
    def test_valid_config_passes(self, valid_config: dict):
        _validate_config(valid_config)  # no exception

    def test_missing_model_size_raises(self, valid_config: dict):
        del valid_config["model_size"]
        with pytest.raises(TranscriptionError, match="Missing transcription config keys"):
            _validate_config(valid_config)

    def test_invalid_model_size_raises(self, valid_config: dict):
        valid_config["model_size"] = "xlarge"
        with pytest.raises(TranscriptionError, match="Invalid model_size"):
            _validate_config(valid_config)

    def test_all_valid_sizes_pass(self, valid_config: dict):
        for size in VALID_MODEL_SIZES:
            valid_config["model_size"] = size
            _validate_config(valid_config)  # no exception


# ── _resolve_device ───────────────────────────────────────────────────────────

class TestResolveDevice:
    def test_cpu_returns_cpu(self):
        assert _resolve_device("cpu") == "cpu"

    def test_cuda_returns_cuda(self):
        assert _resolve_device("cuda") == "cuda"

    def test_auto_returns_cpu_when_no_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            result = _resolve_device("auto")
        assert result == "cpu"

    def test_auto_returns_cpu_when_cuda_unavailable(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _resolve_device("auto")
        assert result == "cpu"

    def test_auto_returns_cuda_when_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _resolve_device("auto")
        assert result == "cuda"


# ── _map_segments ─────────────────────────────────────────────────────────────

class TestMapSegments:
    def test_segments_mapped_correctly(self):
        raw = [
            {"start": 0.0,  "end": 1.5,  "text": " Hello everyone."},
            {"start": 1.5,  "end": 3.2,  "text": " How are you?"},
        ]
        segments = _map_segments(raw)
        assert len(segments) == 2
        assert segments[0].start == 0.0
        assert segments[0].end   == 1.5
        assert segments[0].text  == "Hello everyone."   # leading space stripped

    def test_empty_segments_filtered_out(self):
        raw = [
            {"start": 0.0, "end": 1.0, "text": "   "},  # whitespace only
            {"start": 1.0, "end": 2.0, "text": "hello"},
        ]
        segments = _map_segments(raw)
        assert len(segments) == 1
        assert segments[0].text == "hello"

    def test_empty_input_returns_empty(self):
        assert _map_segments([]) == []

    def test_timestamps_rounded_to_3_decimals(self):
        raw = [{"start": 1.12345678, "end": 2.98765432, "text": "test"}]
        seg = _map_segments(raw)[0]
        assert seg.start == round(1.12345678, 3)
        assert seg.end   == round(2.98765432, 3)


# ── WhisperModelRegistry ──────────────────────────────────────────────────────

class TestWhisperModelRegistry:
    def test_model_loaded_on_first_call(self):
        registry = WhisperModelRegistry()
        fake_model = MagicMock()

        with patch("src.inference._model_registry.whisper.load_model", return_value=fake_model) as mock_load:
            result = registry.get("base", "cpu")
            mock_load.assert_called_once_with("base", device="cpu")
            assert result is fake_model

    def test_model_cached_on_second_call(self):
        registry = WhisperModelRegistry()
        fake_model = MagicMock()

        with patch("src.inference._model_registry.whisper.load_model", return_value=fake_model) as mock_load:
            registry.get("base", "cpu")
            registry.get("base", "cpu")   # second call
            mock_load.assert_called_once()   # load_model called only once

    def test_different_configs_load_separately(self):
        registry = WhisperModelRegistry()

        with patch("src.inference._model_registry.whisper.load_model", return_value=MagicMock()) as mock_load:
            registry.get("base",  "cpu")
            registry.get("small", "cpu")
            assert mock_load.call_count == 2

    def test_invalid_size_raises(self):
        registry = WhisperModelRegistry()
        with pytest.raises(TranscriptionError, match="Invalid model size"):
            registry.get("xlarge", "cpu")

    def test_clear_empties_cache(self):
        registry = WhisperModelRegistry()

        with patch("src.inference._model_registry.whisper.load_model", return_value=MagicMock()) as mock_load:
            registry.get("base", "cpu")
            registry.clear()
            registry.get("base", "cpu")   # should reload after clear
            assert mock_load.call_count == 2

    def test_loaded_models_property(self):
        registry = WhisperModelRegistry()

        with patch("src.inference._model_registry.whisper.load_model", return_value=MagicMock()):
            registry.get("base",  "cpu")
            registry.get("small", "cpu")
            assert ("base",  "cpu") in registry.loaded_models
            assert ("small", "cpu") in registry.loaded_models


# ── transcribe() — full integration (mocked model) ───────────────────────────

class TestTranscribe:
    """
    Integration tests for transcribe(). Uses a real WAV file but mocks
    whisper.load_model so no real model is loaded.
    """

    def test_returns_transcript_result(
        self,
        temp_wav: Path,
        valid_config: dict,
        mock_whisper_model: MagicMock,
    ):
        with patch("src.inference._model_registry.whisper.load_model", return_value=mock_whisper_model):
            result = transcribe(str(temp_wav), valid_config)

        assert isinstance(result, TranscriptResult)
        assert result.language == "en"
        assert "Hello everyone" in result.text
        assert result.segment_count == 2

    def test_segments_populated(
        self,
        temp_wav: Path,
        valid_config: dict,
        mock_whisper_model: MagicMock,
    ):
        with patch("src.inference._model_registry.whisper.load_model", return_value=mock_whisper_model):
            result = transcribe(str(temp_wav), valid_config)

        assert result.segments[0].start == 0.0
        assert result.segments[0].end   == 1.5
        assert result.segments[0].text  == "Hello everyone."

    def test_language_override(
        self,
        temp_wav: Path,
        valid_config: dict,
        mock_whisper_model: MagicMock,
    ):
        """Explicit language kwarg should be passed to model.transcribe."""
        with patch("src.inference._model_registry.whisper.load_model", return_value=mock_whisper_model):
            transcribe(str(temp_wav), valid_config, language="de")

        _, kwargs = mock_whisper_model.transcribe.call_args
        assert kwargs["language"] == "de"

    def test_missing_file_raises(self, valid_config: dict, tmp_path: Path):
        with pytest.raises(TranscriptionError, match="not found"):
            transcribe(str(tmp_path / "missing.wav"), valid_config)

    def test_invalid_config_raises(self, temp_wav: Path):
        bad_config = {"model_size": "xlarge", "language": "en", "device": "cpu"}
        with pytest.raises(TranscriptionError, match="Invalid model_size"):
            transcribe(str(temp_wav), bad_config)

    def test_whisper_exception_wrapped(
        self,
        temp_wav: Path,
        valid_config: dict,
    ):
        """Any exception from whisper should be re-raised as TranscriptionError."""
        crashing_model = MagicMock()
        crashing_model.transcribe.side_effect = RuntimeError("CUDA OOM")

        with patch("src.inference._model_registry.whisper.load_model", return_value=crashing_model):
            with pytest.raises(TranscriptionError, match="Whisper inference failed"):
                transcribe(str(temp_wav), valid_config)

    def test_to_dict_is_json_serializable(
        self,
        temp_wav: Path,
        valid_config: dict,
        mock_whisper_model: MagicMock,
    ):
        """Output dict must be JSON-serializable for annotation export."""
        import json
        with patch("src.inference._model_registry.whisper.load_model", return_value=mock_whisper_model):
            result = transcribe(str(temp_wav), valid_config)

        dumped = json.dumps(result.to_dict())   # raises if not serializable
        assert "Hello everyone" in dumped

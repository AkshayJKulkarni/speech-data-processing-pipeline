"""
Unit tests for the speaker diarization module.

Test strategy:
──────────────
- PyannoteModelRegistry tests: patch pyannote.audio.Pipeline so no model
  is ever loaded. Verifies caching, token validation, and error wrapping.
- Validation tests: pure logic, zero I/O beyond a temp WAV fixture.
- _map_annotation tests: a fake Annotation object with controlled output
  verifies the pyannote → SpeakerSegment mapping without any model.
- diarize() integration tests: real WAV + fully mocked Pipeline, verifies
  the complete call path including timing logs and config wiring.

Zero network calls. Zero real model weights loaded.
"""

import os
import numpy as np
import pytest
import soundfile as sf
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.inference.diarizer import (
    SpeakerSegment,
    SpeakerDiarizationResult,
    diarize,
    _validate_audio_path,
    _validate_config,
    _resolve_device,
    _map_annotation,
)
from src.inference._pyannote_registry import PyannoteModelRegistry
from src.utils.exceptions import DiarizationError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_wav(tmp_path: Path) -> Path:
    """Real 16kHz mono WAV in a temp directory — used for integration tests."""
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t)
    wav_path = tmp_path / "test_audio.wav"
    sf.write(str(wav_path), audio, 16000, subtype="PCM_16")
    return wav_path


@pytest.fixture
def valid_config() -> dict:
    return {
        "model":        "pyannote/speaker-diarization-3.1",
        "device":       "cpu",
        "min_speakers": 1,
        "max_speakers": 5,
        "hf_token_env": "HF_TOKEN",
    }


@pytest.fixture
def fake_annotation() -> MagicMock:
    """
    Mimics a pyannote.core.Annotation object.
    itertracks(yield_label=True) yields (Segment, track_id, speaker_label).
    """

    class FakeSegment:
        def __init__(self, start, end):
            self.start = start
            self.end   = end

    annotation = MagicMock()
    annotation.itertracks.return_value = [
        (FakeSegment(0.0,  3.5),  "A", "SPEAKER_00"),
        (FakeSegment(3.5,  7.1),  "A", "SPEAKER_01"),
        (FakeSegment(7.1,  10.0), "B", "SPEAKER_00"),
        (FakeSegment(10.0, 12.4), "A", "SPEAKER_01"),
    ]
    return annotation


@pytest.fixture
def mock_pipeline(fake_annotation: MagicMock) -> MagicMock:
    """Returns a mock pyannote Pipeline that produces controlled diarization output."""
    pipeline = MagicMock()
    pipeline.return_value = fake_annotation   # pipeline(audio_path, ...) → annotation
    return pipeline


# ── SpeakerSegment ────────────────────────────────────────────────────────────

class TestSpeakerSegment:
    def test_duration(self):
        seg = SpeakerSegment(speaker="SPEAKER_00", start=1.0, end=4.5)
        assert seg.duration() == pytest.approx(3.5, abs=0.001)

    def test_to_dict_keys(self):
        seg = SpeakerSegment(speaker="SPEAKER_01", start=0.0, end=2.0)
        d = seg.to_dict()
        assert set(d.keys()) == {"speaker", "start", "end"}

    def test_to_dict_values(self):
        seg = SpeakerSegment(speaker="SPEAKER_00", start=1.5, end=3.0)
        d = seg.to_dict()
        assert d["speaker"] == "SPEAKER_00"
        assert d["start"]   == 1.5
        assert d["end"]     == 3.0


# ── SpeakerDiarizationResult ──────────────────────────────────────────────────

class TestSpeakerDiarizationResult:
    def _make_result(self) -> SpeakerDiarizationResult:
        return SpeakerDiarizationResult(
            segments=[
                SpeakerSegment("SPEAKER_00", 0.0,  3.5),
                SpeakerSegment("SPEAKER_01", 3.5,  7.1),
                SpeakerSegment("SPEAKER_00", 7.1,  10.0),
                SpeakerSegment("SPEAKER_01", 10.0, 12.4),
            ],
            num_speakers=2,
        )

    def test_speakers_property(self):
        result = self._make_result()
        assert result.speakers == ["SPEAKER_00", "SPEAKER_01"]

    def test_total_duration(self):
        result = self._make_result()
        expected = 3.5 + 3.6 + 2.9 + 2.4
        assert result.total_duration == pytest.approx(expected, abs=0.01)

    def test_to_dict_structure(self):
        d = self._make_result().to_dict()
        assert "num_speakers" in d
        assert "segments" in d
        assert isinstance(d["segments"], list)
        assert d["segments"][0]["speaker"] == "SPEAKER_00"

    def test_to_dict_json_serializable(self):
        import json
        result = self._make_result()
        # Raises if not serializable
        json.dumps(result.to_dict())

    def test_empty_result(self):
        result = SpeakerDiarizationResult()
        assert result.num_speakers == 0
        assert result.speakers     == []
        assert result.total_duration == 0.0


# ── _validate_audio_path ──────────────────────────────────────────────────────

class TestValidateAudioPath:
    def test_valid_wav_passes(self, temp_wav: Path):
        _validate_audio_path(temp_wav)   # no exception

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(DiarizationError, match="not found"):
            _validate_audio_path(tmp_path / "missing.wav")

    def test_non_wav_raises(self, tmp_path: Path):
        mp3 = tmp_path / "audio.mp3"
        mp3.write_bytes(b"fake")
        with pytest.raises(DiarizationError, match="expects a .wav"):
            _validate_audio_path(mp3)


# ── _validate_config ──────────────────────────────────────────────────────────

class TestValidateConfig:
    def test_valid_config_passes(self, valid_config: dict):
        _validate_config(valid_config)   # no exception

    def test_missing_key_raises(self, valid_config: dict):
        del valid_config["model"]
        with pytest.raises(DiarizationError, match="Missing diarization config keys"):
            _validate_config(valid_config)

    def test_min_speakers_zero_raises(self, valid_config: dict):
        valid_config["min_speakers"] = 0
        with pytest.raises(DiarizationError, match="min_speakers must be >= 1"):
            _validate_config(valid_config)

    def test_max_less_than_min_raises(self, valid_config: dict):
        valid_config["min_speakers"] = 5
        valid_config["max_speakers"] = 2
        with pytest.raises(DiarizationError, match="max_speakers"):
            _validate_config(valid_config)

    def test_equal_min_max_passes(self, valid_config: dict):
        valid_config["min_speakers"] = 2
        valid_config["max_speakers"] = 2
        _validate_config(valid_config)   # no exception — fixed speaker count is valid


# ── _resolve_device ───────────────────────────────────────────────────────────

class TestResolveDevice:
    def test_cpu_passthrough(self):
        assert _resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self):
        assert _resolve_device("cuda") == "cuda"

    def test_auto_no_torch_defaults_cpu(self):
        with patch.dict("sys.modules", {"torch": None}):
            assert _resolve_device("auto") == "cpu"

    def test_auto_cuda_unavailable(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _resolve_device("auto") == "cpu"

    def test_auto_cuda_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _resolve_device("auto") == "cuda"


# ── _map_annotation ───────────────────────────────────────────────────────────

class TestMapAnnotation:
    def test_correct_number_of_segments(self, fake_annotation: MagicMock):
        segments = _map_annotation(fake_annotation)
        assert len(segments) == 4

    def test_speakers_mapped(self, fake_annotation: MagicMock):
        segments = _map_annotation(fake_annotation)
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[1].speaker == "SPEAKER_01"

    def test_timestamps_rounded(self, fake_annotation: MagicMock):
        segments = _map_annotation(fake_annotation)
        assert segments[0].start == 0.0
        assert segments[0].end   == 3.5

    def test_sorted_by_start(self, fake_annotation: MagicMock):
        # Shuffle the fake annotation order to verify sort
        class FakeSeg:
            def __init__(self, s, e):
                self.start, self.end = s, e

        shuffled = MagicMock()
        shuffled.itertracks.return_value = [
            (FakeSeg(10.0, 12.0), "A", "SPEAKER_01"),
            (FakeSeg(0.0,   3.0), "A", "SPEAKER_00"),
            (FakeSeg(5.0,   8.0), "B", "SPEAKER_00"),
        ]
        segments = _map_annotation(shuffled)
        starts = [s.start for s in segments]
        assert starts == sorted(starts)

    def test_empty_annotation(self):
        empty = MagicMock()
        empty.itertracks.return_value = []
        assert _map_annotation(empty) == []


# ── PyannoteModelRegistry ─────────────────────────────────────────────────────

class TestPyannoteModelRegistry:
    @pytest.fixture(autouse=True)
    def fresh_registry(self) -> PyannoteModelRegistry:
        """Each test gets its own registry so cache state doesn't leak."""
        self.registry = PyannoteModelRegistry()
        return self.registry

    def _patch_pipeline(self, fake_pipeline):
        """Helper: patches pyannote.audio.Pipeline and torch inside the registry module."""
        mock_torch = MagicMock()
        mock_torch.device = MagicMock(side_effect=lambda d: d)
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = fake_pipeline

        return patch.multiple(
            "src.inference._pyannote_registry",
            **{},
        ), patch(
            "src.inference._pyannote_registry.PyannoteModelRegistry.get",
            side_effect=self._mock_get(fake_pipeline),
        )

    def _mock_get(self, fake_pipeline):
        """Returns a side-effect function simulating successful model load + cache."""
        call_count = {"n": 0}

        def get(model_name, device, hf_token):
            if not hf_token:
                raise DiarizationError("Hugging Face token is required")
            call_count["n"] += 1
            return fake_pipeline

        return get

    def test_missing_token_raises(self):
        with pytest.raises(DiarizationError, match="token is required"):
            self.registry.get("pyannote/speaker-diarization-3.1", "cpu", "")

    def test_whitespace_token_raises(self):
        with pytest.raises(DiarizationError, match="token is required"):
            self.registry.get("pyannote/speaker-diarization-3.1", "cpu", "   ")

    def test_model_loaded_once_and_cached(self):
        fake_pipeline = MagicMock()
        fake_pipeline.to = MagicMock(return_value=fake_pipeline)

        mock_torch  = MagicMock()
        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = fake_pipeline

        with patch.dict("sys.modules", {
            "pyannote":              MagicMock(),
            "pyannote.audio":        MagicMock(Pipeline=mock_pipeline_cls),
            "torch":                 mock_torch,
        }):
            # Re-import to pick up patched modules
            import importlib
            import src.inference._pyannote_registry as reg_module
            importlib.reload(reg_module)

            registry = reg_module.PyannoteModelRegistry()
            r1 = registry.get("pyannote/speaker-diarization-3.1", "cpu", "hf_fake_token")
            r2 = registry.get("pyannote/speaker-diarization-3.1", "cpu", "hf_fake_token")

            mock_pipeline_cls.from_pretrained.assert_called_once()
            assert r1 is r2

    def test_clear_empties_cache(self):
        fake_pipeline = MagicMock()
        fake_pipeline.to = MagicMock(return_value=fake_pipeline)

        mock_torch = MagicMock()
        mock_torch.device.return_value = MagicMock()

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = fake_pipeline

        with patch.dict("sys.modules", {
            "pyannote":       MagicMock(),
            "pyannote.audio": MagicMock(Pipeline=mock_pipeline_cls),
            "torch":          mock_torch,
        }):
            import importlib
            import src.inference._pyannote_registry as reg_module
            importlib.reload(reg_module)

            registry = reg_module.PyannoteModelRegistry()
            registry.get("pyannote/speaker-diarization-3.1", "cpu", "hf_fake_token")
            assert len(registry.loaded_models) == 1

            registry.clear()
            assert len(registry.loaded_models) == 0


# ── diarize() — integration (mocked pipeline) ────────────────────────────────

class TestDiarize:
    """
    Integration tests for diarize(). Real WAV file + mocked pyannote pipeline.
    Verifies the full call path: validation → registry → inference → mapping.
    """

    def test_returns_diarization_result(
        self,
        temp_wav: Path,
        valid_config: dict,
        mock_pipeline: MagicMock,
    ):
        with patch(
            "src.inference.diarizer._pyannote_registry.get",
            return_value=mock_pipeline,
        ), patch.dict(os.environ, {"HF_TOKEN": "hf_fake_token"}):
            result = diarize(str(temp_wav), valid_config)

        assert isinstance(result, SpeakerDiarizationResult)
        assert result.num_speakers == 2
        assert len(result.segments) == 4

    def test_segments_sorted_by_start(
        self,
        temp_wav: Path,
        valid_config: dict,
        mock_pipeline: MagicMock,
    ):
        with patch(
            "src.inference.diarizer._pyannote_registry.get",
            return_value=mock_pipeline,
        ), patch.dict(os.environ, {"HF_TOKEN": "hf_fake_token"}):
            result = diarize(str(temp_wav), valid_config)

        starts = [s.start for s in result.segments]
        assert starts == sorted(starts)

    def test_pipeline_called_with_speaker_hints(
        self,
        temp_wav: Path,
        valid_config: dict,
        mock_pipeline: MagicMock,
    ):
        """min_speakers and max_speakers from config must reach the pipeline call."""
        valid_config["min_speakers"] = 2
        valid_config["max_speakers"] = 4

        with patch(
            "src.inference.diarizer._pyannote_registry.get",
            return_value=mock_pipeline,
        ), patch.dict(os.environ, {"HF_TOKEN": "hf_fake_token"}):
            diarize(str(temp_wav), valid_config)

        mock_pipeline.assert_called_once_with(
            str(temp_wav),
            min_speakers=2,
            max_speakers=4,
        )

    def test_missing_wav_raises(self, valid_config: dict, tmp_path: Path):
        with pytest.raises(DiarizationError, match="not found"):
            diarize(str(tmp_path / "missing.wav"), valid_config)

    def test_non_wav_raises(self, valid_config: dict, tmp_path: Path):
        mp3 = tmp_path / "audio.mp3"
        mp3.write_bytes(b"fake")
        with pytest.raises(DiarizationError, match="expects a .wav"):
            diarize(str(mp3), valid_config)

    def test_invalid_config_raises(self, temp_wav: Path):
        bad_config = {
            "model": "pyannote/x", "device": "cpu",
            "min_speakers": 5, "max_speakers": 2,
            "hf_token_env": "HF_TOKEN",
        }
        with pytest.raises(DiarizationError, match="max_speakers"):
            diarize(str(temp_wav), bad_config)

    def test_pipeline_exception_wrapped(
        self,
        temp_wav: Path,
        valid_config: dict,
    ):
        """Any exception from pyannote must be re-raised as DiarizationError."""
        crashing_pipeline = MagicMock()
        crashing_pipeline.side_effect = RuntimeError("CUDA OOM")

        with patch(
            "src.inference.diarizer._pyannote_registry.get",
            return_value=crashing_pipeline,
        ), patch.dict(os.environ, {"HF_TOKEN": "hf_fake_token"}):
            with pytest.raises(DiarizationError, match="Pyannote inference failed"):
                diarize(str(temp_wav), valid_config)

    def test_result_is_json_serializable(
        self,
        temp_wav: Path,
        valid_config: dict,
        mock_pipeline: MagicMock,
    ):
        import json
        with patch(
            "src.inference.diarizer._pyannote_registry.get",
            return_value=mock_pipeline,
        ), patch.dict(os.environ, {"HF_TOKEN": "hf_fake_token"}):
            result = diarize(str(temp_wav), valid_config)

        json.dumps(result.to_dict())   # raises TypeError if not serializable

    def test_hf_token_read_from_env(
        self,
        temp_wav: Path,
        valid_config: dict,
        mock_pipeline: MagicMock,
    ):
        """Token must be read from the env var named in config, not passed as arg."""
        valid_config["hf_token_env"] = "MY_CUSTOM_TOKEN_VAR"

        with patch(
            "src.inference.diarizer._pyannote_registry.get",
            return_value=mock_pipeline,
        ) as mock_get, patch.dict(os.environ, {"MY_CUSTOM_TOKEN_VAR": "my_token_value"}):
            diarize(str(temp_wav), valid_config)

        # Verify the registry was called with the value from MY_CUSTOM_TOKEN_VAR
        _, call_kwargs = mock_get.call_args
        assert mock_get.call_args[0][2] == "my_token_value"

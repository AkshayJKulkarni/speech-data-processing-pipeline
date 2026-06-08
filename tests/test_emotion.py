"""
Unit tests for the emotion classification module.

Test strategy:
──────────────
- _audio_slicer tests: pure numpy math — no model, no file I/O beyond a
  real WAV fixture for load_audio_array.
- Validation tests: zero I/O, no model loading.
- Label normalisation tests: pure string mapping.
- EmotionModelRegistry tests: patch transformers.pipeline so no weights
  are downloaded or loaded.
- classify_emotion() integration tests: real WAV + fully mocked pipeline,
  verifying the full call path including batching, skipping, and ordering.

Zero network calls. Zero real model weights loaded.
"""

import json
import os
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.inference.emotion_classifier import (
    EmotionSegment,
    EmotionResult,
    classify_emotion,
    _validate_audio_path,
    _validate_config,
    _validate_segments,
    _resolve_device,
    _normalise_label,
    _extract_top_prediction,
    _get_field,
    _UNKNOWN_EMOTION,
)
from src.inference._audio_slicer import (
    load_audio_array,
    slice_segment,
    is_segment_too_short,
)
from src.inference._emotion_registry import EmotionModelRegistry
from src.utils.exceptions import EmotionError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_wav(tmp_path: Path) -> Path:
    """Real 2-second 16kHz mono WAV."""
    t = np.linspace(0, 2.0, 32000, endpoint=False, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), audio, 16000, subtype="PCM_16")
    return wav_path


@pytest.fixture
def valid_config() -> dict:
    return {
        "model":                "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "device":               "cpu",
        "min_segment_duration": 0.5,
        "batch_size":           4,
    }


@pytest.fixture
def speaker_segments() -> list[dict]:
    return [
        {"speaker": "SPEAKER_00", "start": 0.0,  "end": 1.0},
        {"speaker": "SPEAKER_01", "start": 1.0,  "end": 2.0},
    ]


@pytest.fixture
def mock_hf_scores() -> list[list[dict]]:
    """Controlled HF pipeline output for two segments."""
    return [
        [{"label": "happy",   "score": 0.80}, {"label": "neutral", "score": 0.20}],
        [{"label": "neutral", "score": 0.65}, {"label": "sad",     "score": 0.35}],
    ]


@pytest.fixture
def mock_pipeline(mock_hf_scores):
    """Mock HF pipeline that returns controlled scores."""
    pipe = MagicMock()
    pipe.return_value = mock_hf_scores
    return pipe


# ── EmotionSegment ─────────────────────────────────────────────────────────────

class TestEmotionSegment:
    def test_duration(self):
        seg = EmotionSegment("SPEAKER_00", 1.0, 3.5, "happy", 0.9)
        assert seg.duration() == pytest.approx(2.5, abs=0.001)

    def test_to_dict_keys(self):
        seg = EmotionSegment("SPEAKER_00", 0.0, 1.0, "sad", 0.75)
        d = seg.to_dict()
        assert set(d.keys()) == {"speaker", "start", "end", "emotion", "confidence"}

    def test_to_dict_confidence_rounded(self):
        seg = EmotionSegment("SPEAKER_00", 0.0, 1.0, "angry", 0.123456789)
        assert seg.to_dict()["confidence"] == round(0.123456789, 4)


# ── EmotionResult ─────────────────────────────────────────────────────────────

class TestEmotionResult:
    def _make_result(self) -> EmotionResult:
        return EmotionResult(segments=[
            EmotionSegment("SPEAKER_00", 0.0,  1.0, "happy",   0.90),
            EmotionSegment("SPEAKER_01", 1.0,  2.0, "neutral", 0.70),
            EmotionSegment("SPEAKER_00", 2.0,  3.0, "happy",   0.85),
            EmotionSegment("SPEAKER_01", 3.0,  4.0, "sad",     0.60),
        ])

    def test_emotion_counts(self):
        counts = self._make_result().emotion_counts
        assert counts["happy"]   == 2
        assert counts["neutral"] == 1
        assert counts["sad"]     == 1

    def test_dominant_emotion(self):
        assert self._make_result().dominant_emotion == "happy"

    def test_dominant_emotion_empty(self):
        assert EmotionResult().dominant_emotion == _UNKNOWN_EMOTION

    def test_to_dict_structure(self):
        d = self._make_result().to_dict()
        assert "segments" in d
        assert isinstance(d["segments"], list)
        assert len(d["segments"]) == 4

    def test_to_dict_json_serializable(self):
        json.dumps(self._make_result().to_dict())


# ── _audio_slicer ─────────────────────────────────────────────────────────────

class TestSliceSegment:
    def _make_audio(self, duration_sec=2.0, sr=16000) -> tuple[np.ndarray, int]:
        samples = int(duration_sec * sr)
        return np.arange(samples, dtype=np.float32), sr

    def test_correct_sample_count(self):
        audio, sr = self._make_audio(2.0, 16000)
        sliced = slice_segment(audio, sr, 0.5, 1.5)
        assert len(sliced) == 16000   # 1 second at 16kHz

    def test_start_zero(self):
        audio, sr = self._make_audio()
        sliced = slice_segment(audio, sr, 0.0, 1.0)
        np.testing.assert_array_equal(sliced, audio[:16000])

    def test_end_clamped_to_array_bounds(self):
        audio, sr = self._make_audio(1.0, 16000)
        sliced = slice_segment(audio, sr, 0.0, 5.0)  # 5s > 1s audio
        assert len(sliced) == 16000

    def test_start_ge_end_returns_empty(self):
        audio, sr = self._make_audio()
        sliced = slice_segment(audio, sr, 1.5, 1.0)
        assert len(sliced) == 0

    def test_negative_start_clamped(self):
        audio, sr = self._make_audio()
        sliced = slice_segment(audio, sr, -1.0, 1.0)
        assert len(sliced) == 16000


class TestIsSegmentTooShort:
    def test_short_segment_returns_true(self):
        audio = np.zeros(4000, dtype=np.float32)   # 0.25s at 16kHz
        assert is_segment_too_short(audio, 16000, min_duration_sec=0.5)

    def test_long_enough_returns_false(self):
        audio = np.zeros(16000, dtype=np.float32)  # 1.0s at 16kHz
        assert not is_segment_too_short(audio, 16000, min_duration_sec=0.5)

    def test_exact_boundary_not_too_short(self):
        audio = np.zeros(8000, dtype=np.float32)   # exactly 0.5s
        assert not is_segment_too_short(audio, 16000, min_duration_sec=0.5)

    def test_empty_array_is_too_short(self):
        audio = np.zeros(0, dtype=np.float32)
        assert is_segment_too_short(audio, 16000, min_duration_sec=0.5)


class TestLoadAudioArray:
    def test_returns_float32(self, temp_wav: Path):
        audio, sr = load_audio_array(temp_wav)
        assert audio.dtype == np.float32

    def test_correct_sample_rate(self, temp_wav: Path):
        _, sr = load_audio_array(temp_wav)
        assert sr == 16000

    def test_1d_array(self, temp_wav: Path):
        audio, _ = load_audio_array(temp_wav)
        assert audio.ndim == 1

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(EmotionError, match="Failed to load audio"):
            load_audio_array(tmp_path / "missing.wav")


# ── Validation ────────────────────────────────────────────────────────────────

class TestValidateAudioPath:
    def test_valid_wav_passes(self, temp_wav: Path):
        _validate_audio_path(temp_wav)

    def test_missing_raises(self, tmp_path: Path):
        with pytest.raises(EmotionError, match="not found"):
            _validate_audio_path(tmp_path / "missing.wav")

    def test_non_wav_raises(self, tmp_path: Path):
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"x")
        with pytest.raises(EmotionError, match="expects a .wav"):
            _validate_audio_path(f)


class TestValidateConfig:
    def test_valid_passes(self, valid_config: dict):
        _validate_config(valid_config)

    def test_missing_key_raises(self, valid_config: dict):
        del valid_config["model"]
        with pytest.raises(EmotionError, match="Missing emotion config keys"):
            _validate_config(valid_config)

    def test_negative_min_duration_raises(self, valid_config: dict):
        valid_config["min_segment_duration"] = -0.1
        with pytest.raises(EmotionError, match="min_segment_duration"):
            _validate_config(valid_config)

    def test_zero_batch_size_raises(self, valid_config: dict):
        valid_config["batch_size"] = 0
        with pytest.raises(EmotionError, match="batch_size"):
            _validate_config(valid_config)


class TestValidateSegments:
    def test_dict_segments_pass(self):
        _validate_segments([{"speaker": "S0", "start": 0.0, "end": 1.0}])

    def test_dataclass_segments_pass(self):
        from src.inference.diarizer import SpeakerSegment
        _validate_segments([SpeakerSegment("SPEAKER_00", 0.0, 1.0)])

    def test_empty_list_raises(self):
        with pytest.raises(EmotionError, match="empty"):
            _validate_segments([])

    def test_missing_field_raises(self):
        with pytest.raises(EmotionError, match="missing required field"):
            _validate_segments([{"speaker": "S0", "start": 0.0}])   # no "end"


# ── Label normalisation ───────────────────────────────────────────────────────

class TestNormaliseLabel:
    @pytest.mark.parametrize("raw,expected", [
        ("angry",   "angry"),
        ("Angry",   "angry"),
        ("ANGRY",   "angry"),
        ("ang",     "angry"),
        ("hap",     "happy"),
        ("happy",   "happy"),
        ("neu",     "neutral"),
        ("neutral", "neutral"),
        ("sad",     "sad"),
        ("sadness", "sad"),
        ("sur",     "surprised"),
        ("ps",      "surprised"),
        ("fea",     "fear"),
        ("dis",     "disgust"),
    ])
    def test_normalisation(self, raw, expected):
        assert _normalise_label(raw) == expected

    def test_unknown_label_passthrough(self):
        # Unknown labels are returned lowercased — not silently mapped to unknown
        assert _normalise_label("CUSTOM_EMO") == "custom_emo"


class TestExtractTopPrediction:
    def test_correct_top_label(self):
        scores = [
            {"label": "neutral", "score": 0.10},
            {"label": "happy",   "score": 0.75},
            {"label": "sad",     "score": 0.15},
        ]
        label, conf = _extract_top_prediction(scores)
        assert label == "happy"
        assert conf  == pytest.approx(0.75)

    def test_empty_scores_raises(self):
        with pytest.raises(EmotionError, match="empty scores"):
            _extract_top_prediction([])

    def test_single_label(self):
        scores = [{"label": "angry", "score": 1.0}]
        label, conf = _extract_top_prediction(scores)
        assert label == "angry"
        assert conf  == pytest.approx(1.0)


# ── EmotionModelRegistry ──────────────────────────────────────────────────────

class TestEmotionModelRegistry:
    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        self.registry = EmotionModelRegistry()

    def test_loads_on_first_call(self):
        fake_pipe = MagicMock()
        mock_transformers = MagicMock()
        mock_transformers.pipeline.return_value = fake_pipe
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            import importlib
            import src.inference._emotion_registry as reg_mod
            importlib.reload(reg_mod)

            registry = reg_mod.EmotionModelRegistry()
            r = registry.get("test-model", "cpu")
            mock_transformers.pipeline.assert_called_once()
            assert r is fake_pipe

    def test_cached_on_second_call(self):
        fake_pipe = MagicMock()
        mock_transformers = MagicMock()
        mock_transformers.pipeline.return_value = fake_pipe
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            import importlib
            import src.inference._emotion_registry as reg_mod
            importlib.reload(reg_mod)

            registry = reg_mod.EmotionModelRegistry()
            registry.get("test-model", "cpu")
            registry.get("test-model", "cpu")   # second call
            mock_transformers.pipeline.assert_called_once()

    def test_clear_empties_cache(self):
        fake_pipe = MagicMock()
        mock_transformers = MagicMock()
        mock_transformers.pipeline.return_value = fake_pipe
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            import importlib
            import src.inference._emotion_registry as reg_mod
            importlib.reload(reg_mod)

            registry = reg_mod.EmotionModelRegistry()
            registry.get("test-model", "cpu")
            registry.clear()
            assert len(registry.loaded_models) == 0


# ── classify_emotion() — integration ─────────────────────────────────────────

class TestClassifyEmotion:
    def test_returns_emotion_result(
        self,
        temp_wav: Path,
        valid_config: dict,
        speaker_segments: list[dict],
        mock_pipeline: MagicMock,
    ):
        with patch("src.inference.emotion_classifier._emotion_registry.get",
                   return_value=mock_pipeline):
            result = classify_emotion(str(temp_wav), speaker_segments, valid_config)

        assert isinstance(result, EmotionResult)
        assert len(result.segments) == 2

    def test_emotion_labels_correct(
        self,
        temp_wav: Path,
        valid_config: dict,
        speaker_segments: list[dict],
        mock_pipeline: MagicMock,
    ):
        with patch("src.inference.emotion_classifier._emotion_registry.get",
                   return_value=mock_pipeline):
            result = classify_emotion(str(temp_wav), speaker_segments, valid_config)

        assert result.segments[0].emotion == "happy"
        assert result.segments[1].emotion == "neutral"

    def test_confidence_values(
        self,
        temp_wav: Path,
        valid_config: dict,
        speaker_segments: list[dict],
        mock_pipeline: MagicMock,
    ):
        with patch("src.inference.emotion_classifier._emotion_registry.get",
                   return_value=mock_pipeline):
            result = classify_emotion(str(temp_wav), speaker_segments, valid_config)

        assert result.segments[0].confidence == pytest.approx(0.80, abs=0.001)
        assert result.segments[1].confidence == pytest.approx(0.65, abs=0.001)

    def test_short_segments_get_unknown(
        self,
        temp_wav: Path,
        valid_config: dict,
        mock_pipeline: MagicMock,
    ):
        # 0.1 second segment — below 0.5s minimum
        short_segs = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 0.1}]
        valid_config["min_segment_duration"] = 0.5

        with patch("src.inference.emotion_classifier._emotion_registry.get",
                   return_value=mock_pipeline):
            result = classify_emotion(str(temp_wav), short_segs, valid_config)

        assert result.segments[0].emotion    == _UNKNOWN_EMOTION
        assert result.segments[0].confidence == 0.0
        mock_pipeline.assert_not_called()   # model never invoked

    def test_output_sorted_by_start(
        self,
        temp_wav: Path,
        valid_config: dict,
        mock_pipeline: MagicMock,
    ):
        # Mix of normal and short segments to verify sort covers skipped ones too
        segs = [
            {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.1},  # too short → unknown
        ]
        with patch("src.inference.emotion_classifier._emotion_registry.get",
                   return_value=mock_pipeline):
            result = classify_emotion(str(temp_wav), segs, valid_config)

        starts = [s.start for s in result.segments]
        assert starts == sorted(starts)

    def test_accepts_dataclass_segments(
        self,
        temp_wav: Path,
        valid_config: dict,
        mock_pipeline: MagicMock,
    ):
        from src.inference.diarizer import SpeakerSegment
        segs = [
            SpeakerSegment("SPEAKER_00", 0.0, 1.0),
            SpeakerSegment("SPEAKER_01", 1.0, 2.0),
        ]
        with patch("src.inference.emotion_classifier._emotion_registry.get",
                   return_value=mock_pipeline):
            result = classify_emotion(str(temp_wav), segs, valid_config)

        assert len(result.segments) == 2

    def test_missing_wav_raises(
        self,
        valid_config: dict,
        speaker_segments: list[dict],
        tmp_path: Path,
    ):
        with pytest.raises(EmotionError, match="not found"):
            classify_emotion(str(tmp_path / "missing.wav"), speaker_segments, valid_config)

    def test_model_failure_raises_emotion_error(
        self,
        temp_wav: Path,
        valid_config: dict,
        speaker_segments: list[dict],
    ):
        crashing_pipe = MagicMock()
        crashing_pipe.side_effect = RuntimeError("OOM")

        with patch("src.inference.emotion_classifier._emotion_registry.get",
                   return_value=crashing_pipe):
            with pytest.raises(EmotionError, match="inference failed"):
                classify_emotion(str(temp_wav), speaker_segments, valid_config)

    def test_result_json_serializable(
        self,
        temp_wav: Path,
        valid_config: dict,
        speaker_segments: list[dict],
        mock_pipeline: MagicMock,
    ):
        with patch("src.inference.emotion_classifier._emotion_registry.get",
                   return_value=mock_pipeline):
            result = classify_emotion(str(temp_wav), speaker_segments, valid_config)

        json.dumps(result.to_dict())   # raises TypeError if not serializable

    def test_batching_respected(
        self,
        temp_wav: Path,
        valid_config: dict,
    ):
        """With batch_size=1, the pipeline is called once per segment."""
        valid_config["batch_size"] = 1

        segs = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
        ]

        # Pipeline returns one list of scores per call when batch_size=1
        single_scores = [[{"label": "happy", "score": 0.9}, {"label": "neutral", "score": 0.1}]]
        pipe = MagicMock(return_value=single_scores)

        with patch("src.inference.emotion_classifier._emotion_registry.get", return_value=pipe):
            result = classify_emotion(str(temp_wav), segs, valid_config)

        # 2 segments / batch_size 1 = 2 pipeline calls
        assert pipe.call_count == 2
        assert len(result.segments) == 2

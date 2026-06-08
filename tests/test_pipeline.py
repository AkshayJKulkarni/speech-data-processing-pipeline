"""
Tests for the pipeline orchestration layer.

Strategy:
    All six domain stages are patched with MagicMock so no models,
    no real audio, and no network calls are needed.
    Tests verify orchestration behaviour: stage ordering, timing recording,
    fatal vs. non-fatal error policy, pipeline_summary.json content,
    and batch mode resilience.
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

from src.pipeline.context import PipelineContext
from src.pipeline.runner import run, run_batch, _write_pipeline_summary
from src.utils.exceptions import (
    AcquisitionError,
    PreprocessingError,
    TranscriptionError,
    DiarizationError,
    AnnotationError,
)


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

def make_transcript(language="en", segment_count=3):
    t = SimpleNamespace(
        language=language,
        text="hello world",
        segments=[SimpleNamespace(start=float(i), end=float(i+1), text=f"seg{i}")
                  for i in range(segment_count)],
        segment_count=segment_count,
        duration=float(segment_count),
    )
    return t


def make_diarization(num_speakers=2, segment_count=3):
    segs = [
        SimpleNamespace(speaker=f"SPEAKER_0{i%num_speakers}", start=float(i), end=float(i+1))
        for i in range(segment_count)
    ]
    return SimpleNamespace(segments=segs, num_speakers=num_speakers)


def make_emotion(segment_count=3):
    segs = [
        SimpleNamespace(speaker=f"SPEAKER_00", start=float(i), end=float(i+1),
                        emotion="happy", confidence=0.9)
        for i in range(segment_count)
    ]
    return SimpleNamespace(segments=segs)


def make_annotation_result(tmp_dir):
    r = SimpleNamespace(
        json_path=str(Path(tmp_dir) / "test.json"),
        csv_path=str(Path(tmp_dir) / "test.csv"),
        metadata_path=str(Path(tmp_dir) / "test_metadata.json"),
        segments=[],
        metadata=SimpleNamespace(
            num_segments=3, num_speakers=2, language="en",
            duration_seconds=3.0, dominant_emotion="happy",
        ),
    )
    return r


@pytest.fixture
def tmp_output(tmp_path):
    return str(tmp_path)


@pytest.fixture
def config(tmp_output):
    return {
        "pipeline": {"name": "test-pipeline", "version": "1.0.0"},
        "paths": {
            "raw_data":       str(Path(tmp_output) / "raw"),
            "processed_data": str(Path(tmp_output) / "processed"),
            "output_data":    tmp_output,
            "logs":           str(Path(tmp_output) / "logs"),
        },
        "audio": {
            "target_sample_rate": 16000,
            "target_channels":    1,
            "output_format":      "wav",
        },
    }


@pytest.fixture
def models_config():
    return {
        "transcription": {"model_size": "base", "language": "en", "device": "cpu"},
        "diarization":   {
            "model": "pyannote/speaker-diarization-3.1",
            "device": "cpu", "min_speakers": 1, "max_speakers": 5,
            "hf_token_env": "HF_TOKEN",
        },
        "emotion": {
            "model": "test-model", "device": "cpu",
            "min_segment_duration": 0.5, "batch_size": 4,
        },
    }


def _patch_all_stages(tmp_output):
    """
    Returns a dict of attribute_name -> MagicMock for patch.multiple.
    Keys are bare attribute names as imported into runner.py.
    """
    transcript  = make_transcript()
    diarization = make_diarization()
    emotion     = make_emotion()
    annotation  = make_annotation_result(tmp_output)

    return {
        "acquire":            MagicMock(return_value="/raw/audio.mp4"),
        "process":            MagicMock(return_value="/processed/audio.wav"),
        "transcribe":         MagicMock(return_value=transcript),
        "diarize":            MagicMock(return_value=diarization),
        "classify_emotion":   MagicMock(return_value=emotion),
        "export_annotations": MagicMock(return_value=annotation),
    }


# ---------------------------------------------------------------------------
# PipelineContext
# ---------------------------------------------------------------------------

class TestPipelineContext:
    def test_initial_state_incomplete(self):
        ctx = PipelineContext(source="test.wav")
        assert not ctx.is_complete()
        assert ctx.errors == []
        assert ctx.stage_timings == {}

    def test_summary_shows_source(self):
        ctx = PipelineContext(source="my_audio.wav")
        assert "my_audio.wav" in ctx.summary()

    def test_stage_timings_stored(self):
        ctx = PipelineContext(source="x")
        ctx.stage_timings["acquisition"] = 1.23
        assert ctx.stage_timings["acquisition"] == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# run() — happy path
# ---------------------------------------------------------------------------

class TestRunHappyPath:
    def test_returns_pipeline_context(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        with patch.multiple("src.pipeline.runner", **patches):
            ctx = run("audio.mp3", config, models_config)
        assert isinstance(ctx, PipelineContext)

    def test_all_six_stages_called(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        with patch.multiple("src.pipeline.runner", **patches):
            run("audio.mp3", config, models_config)
        patches["acquire"].assert_called_once()
        patches["process"].assert_called_once()
        patches["transcribe"].assert_called_once()
        patches["diarize"].assert_called_once()
        patches["classify_emotion"].assert_called_once()
        patches["export_annotations"].assert_called_once()

    def test_stage_outputs_stored_on_context(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        with patch.multiple("src.pipeline.runner", **patches):
            ctx = run("audio.mp3", config, models_config)
        assert ctx.raw_path          == "/raw/audio.mp4"
        assert ctx.processed_path    == "/processed/audio.wav"
        assert ctx.transcript        is not None
        assert ctx.speaker_segments  is not None
        assert ctx.annotated_segments is not None
        assert ctx.output_paths      is not None

    def test_stage_timings_recorded_for_all_stages(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        with patch.multiple("src.pipeline.runner", **patches):
            ctx = run("audio.mp3", config, models_config)
        for stage in ("acquisition", "preprocessing", "transcription",
                      "diarization", "emotion", "annotation"):
            assert stage in ctx.stage_timings
            assert ctx.stage_timings[stage] >= 0.0

    def test_no_errors_on_full_success(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        with patch.multiple("src.pipeline.runner", **patches):
            ctx = run("audio.mp3", config, models_config)
        assert ctx.errors == []

    def test_pipeline_summary_json_written(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        with patch.multiple("src.pipeline.runner", **patches):
            run("audio.mp3", config, models_config)
        summary_path = Path(tmp_output) / "pipeline_summary.json"
        assert summary_path.exists()

    def test_pipeline_summary_json_content(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        with patch.multiple("src.pipeline.runner", **patches):
            run("audio.mp3", config, models_config)
        data = json.loads((Path(tmp_output) / "pipeline_summary.json").read_text())
        assert data["source"]                  == "audio.mp3"
        assert "execution_timestamp"           in data
        assert "total_runtime_seconds"         in data
        assert "stage_timings"                 in data
        assert "num_speakers"                  in data
        assert "num_transcript_segments"       in data
        assert "num_emotion_labels"            in data
        assert "output_files"                  in data
        assert "pipeline_summary"              in data["output_files"]

    def test_pipeline_summary_has_all_six_stage_timings(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        with patch.multiple("src.pipeline.runner", **patches):
            run("audio.mp3", config, models_config)
        data = json.loads((Path(tmp_output) / "pipeline_summary.json").read_text())
        for stage in ("acquisition", "preprocessing", "transcription",
                      "diarization", "emotion", "annotation"):
            assert stage in data["stage_timings"]


# ---------------------------------------------------------------------------
# run() — fatal stage failures
# ---------------------------------------------------------------------------

class TestRunFatalFailures:
    def test_acquisition_failure_raises_and_aborts(self, config, models_config, tmp_output):
        with patch("src.pipeline.runner.acquire",
                   side_effect=AcquisitionError("download failed")):
            with pytest.raises(AcquisitionError, match="download failed"):
                run("bad_url", config, models_config)

    def test_preprocessing_failure_raises_and_aborts(self, config, models_config, tmp_output):
        with patch("src.pipeline.runner.acquire", return_value="/raw/audio.mp4"), \
             patch("src.pipeline.runner.process",
                   side_effect=PreprocessingError("corrupt file")):
            with pytest.raises(PreprocessingError, match="corrupt file"):
                run("audio.mp3", config, models_config)

    def test_acquisition_failure_no_downstream_calls(self, config, models_config):
        mock_process = MagicMock()
        with patch("src.pipeline.runner.acquire",
                   side_effect=AcquisitionError("fail")), \
             patch("src.pipeline.runner.process", mock_process):
            with pytest.raises(AcquisitionError):
                run("bad", config, models_config)
        mock_process.assert_not_called()


# ---------------------------------------------------------------------------
# run() — non-fatal stage failures
# ---------------------------------------------------------------------------

class TestRunNonFatalFailures:
    def test_transcription_failure_recorded_not_raised(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        patches["transcribe"] = MagicMock(side_effect=TranscriptionError("whisper OOM"))
        with patch.multiple("src.pipeline.runner", **patches):
            ctx = run("audio.mp3", config, models_config)
        assert any("transcription" in e for e in ctx.errors)
        assert ctx.transcript is None

    def test_diarization_failure_skips_emotion(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        patches["diarize"] = MagicMock(side_effect=DiarizationError("no HF token"))
        with patch.multiple("src.pipeline.runner", **patches):
            ctx = run("audio.mp3", config, models_config)
        # Emotion must not be called if diarization failed (no segments)
        patches["classify_emotion"].assert_not_called()
        assert ctx.speaker_segments is None

    def test_annotation_failure_recorded_pipeline_continues(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        patches["export_annotations"] = MagicMock(side_effect=AnnotationError("write failed"))
        with patch.multiple("src.pipeline.runner", **patches):
            ctx = run("audio.mp3", config, models_config)
        assert any("annotation" in e for e in ctx.errors)

    def test_multiple_non_fatal_failures_all_recorded(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        patches["transcribe"] = MagicMock(side_effect=TranscriptionError("t_fail"))
        patches["diarize"]    = MagicMock(side_effect=DiarizationError("d_fail"))
        with patch.multiple("src.pipeline.runner", **patches):
            ctx = run("audio.mp3", config, models_config)
        assert len(ctx.errors) >= 2


# ---------------------------------------------------------------------------
# _write_pipeline_summary
# ---------------------------------------------------------------------------

class TestWritePipelineSummary:
    def test_file_created(self, tmp_output):
        ctx = PipelineContext(source="test.wav")
        ctx.stage_timings = {"acquisition": 1.0}
        _write_pipeline_summary(ctx, tmp_output, 5.0, "2024-01-01T00:00:00Z")
        assert (Path(tmp_output) / "pipeline_summary.json").exists()

    def test_required_keys_present(self, tmp_output):
        ctx = PipelineContext(source="interview.wav")
        ctx.stage_timings = {"acquisition": 0.5, "preprocessing": 1.2}
        _write_pipeline_summary(ctx, tmp_output, 2.0, "2024-06-01T12:00:00Z")
        data = json.loads((Path(tmp_output) / "pipeline_summary.json").read_text())
        required_keys = {
            "source", "execution_timestamp", "total_runtime_seconds",
            "stage_timings", "errors", "language", "num_speakers",
            "num_transcript_segments", "num_emotion_labels", "output_files",
        }
        assert required_keys.issubset(data.keys())

    def test_source_stored(self, tmp_output):
        ctx = PipelineContext(source="my_file.wav")
        _write_pipeline_summary(ctx, tmp_output, 1.0, "2024-01-01T00:00:00Z")
        data = json.loads((Path(tmp_output) / "pipeline_summary.json").read_text())
        assert data["source"] == "my_file.wav"

    def test_total_runtime_stored(self, tmp_output):
        ctx = PipelineContext(source="x")
        _write_pipeline_summary(ctx, tmp_output, 42.7, "2024-01-01T00:00:00Z")
        data = json.loads((Path(tmp_output) / "pipeline_summary.json").read_text())
        assert data["total_runtime_seconds"] == pytest.approx(42.7, abs=0.01)

    def test_self_referential_path(self, tmp_output):
        ctx = PipelineContext(source="x")
        _write_pipeline_summary(ctx, tmp_output, 1.0, "2024-01-01T00:00:00Z")
        data = json.loads((Path(tmp_output) / "pipeline_summary.json").read_text())
        assert data["output_files"]["pipeline_summary"].endswith("pipeline_summary.json")

    def test_partial_context_graceful(self, tmp_output):
        """Summary must write even when no inference stages completed."""
        ctx = PipelineContext(source="fail.wav")
        ctx.errors = ["acquisition: download failed"]
        _write_pipeline_summary(ctx, tmp_output, 0.5, "2024-01-01T00:00:00Z")
        data = json.loads((Path(tmp_output) / "pipeline_summary.json").read_text())
        assert data["errors"] == ["acquisition: download failed"]
        assert data["language"] == "unknown"
        assert data["num_speakers"] == 0


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------

class TestRunBatch:
    def test_returns_one_context_per_source(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        with patch.multiple("src.pipeline.runner", **patches):
            results = run_batch(["a.mp3", "b.mp3"], config, models_config)
        assert len(results) == 2

    def test_failure_on_one_does_not_abort_others(self, config, models_config, tmp_output):
        call_count = {"n": 0}

        def fake_acquire(source, raw_dir):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise AcquisitionError("first fails")
            return "/raw/audio.mp4"

        patches = _patch_all_stages(tmp_output)
        patches["acquire"] = MagicMock(side_effect=fake_acquire)

        with patch.multiple("src.pipeline.runner", **patches):
            results = run_batch(["bad.mp3", "good.mp3"], config, models_config)

        assert len(results) == 2
        assert results[0].errors != []   # first failed
        assert results[1].raw_path == "/raw/audio.mp4"  # second succeeded

    def test_all_succeed_returns_complete_contexts(self, config, models_config, tmp_output):
        patches = _patch_all_stages(tmp_output)
        with patch.multiple("src.pipeline.runner", **patches):
            results = run_batch(["a.mp3", "b.mp3", "c.mp3"], config, models_config)
        assert all(r.raw_path is not None for r in results)

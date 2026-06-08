"""
Unit tests for acquisition and pipeline context.
Zero network calls. Zero ML model loading.
"""

import os
import pytest
import tempfile

from src.acquisition.validator import (
    is_youtube_url,
    is_local_file,
    validate_local_file,
    validate_youtube_url,
    validate_source,
    SUPPORTED_EXTENSIONS,
)
from src.utils.exceptions import AcquisitionError
from src.pipeline.context import PipelineContext


# ── is_youtube_url ─────────────────────────────────────────────────────────────

class TestIsYoutubeUrl:
    def test_standard_url(self):
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_short_url(self):
        assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")

    def test_embed_url(self):
        assert is_youtube_url("https://www.youtube.com/embed/dQw4w9WgXcQ")

    def test_vimeo_not_youtube(self):
        assert not is_youtube_url("https://vimeo.com/123456")

    def test_local_path_not_youtube(self):
        assert not is_youtube_url("/home/user/audio.mp3")

    def test_empty_string(self):
        assert not is_youtube_url("")


# ── validate_youtube_url ───────────────────────────────────────────────────────

class TestValidateYoutubeUrl:
    def test_valid_passes(self):
        validate_youtube_url("https://www.youtube.com/watch?v=abc123")

    def test_invalid_raises_acquisition_error(self):
        with pytest.raises(AcquisitionError, match="Invalid YouTube URL"):
            validate_youtube_url("not-a-url")


# ── validate_local_file ────────────────────────────────────────────────────────

class TestValidateLocalFile:
    def test_valid_wav_passes(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            validate_local_file(tmp)
        finally:
            os.unlink(tmp)

    def test_missing_file_raises(self):
        with pytest.raises(AcquisitionError, match="Local file not found"):
            validate_local_file("/nonexistent/audio.wav")

    def test_unsupported_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            tmp = f.name
        try:
            with pytest.raises(AcquisitionError, match="Unsupported file type"):
                validate_local_file(tmp)
        finally:
            os.unlink(tmp)

    def test_all_supported_extensions_pass(self):
        for ext in SUPPORTED_EXTENSIONS:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                tmp = f.name
            try:
                validate_local_file(tmp)
            finally:
                os.unlink(tmp)


# ── validate_source ────────────────────────────────────────────────────────────

class TestValidateSource:
    def test_youtube_returns_youtube(self):
        assert validate_source("https://www.youtube.com/watch?v=abc123xyz") == "youtube"

    def test_local_returns_local(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        try:
            assert validate_source(tmp) == "local"
        finally:
            os.unlink(tmp)

    def test_empty_raises(self):
        with pytest.raises(AcquisitionError, match="cannot be empty"):
            validate_source("")

    def test_whitespace_raises(self):
        with pytest.raises(AcquisitionError, match="cannot be empty"):
            validate_source("   ")

    def test_nonexistent_path_raises(self):
        with pytest.raises(AcquisitionError, match="not recognized"):
            validate_source("/this/does/not/exist.mp4")


# ── PipelineContext ────────────────────────────────────────────────────────────

class TestPipelineContext:
    def test_initial_state(self):
        ctx = PipelineContext(source="test.wav")
        assert not ctx.is_complete()
        assert ctx.errors == []

    def test_summary_shows_source(self):
        ctx = PipelineContext(source="test.wav")
        assert "test.wav" in ctx.summary()

    def test_is_complete_when_all_fields_set(self):
        ctx = PipelineContext(
            source="test.wav",
            raw_path="/raw/test.wav",
            processed_path="/processed/test.wav",
            transcript={"text": "hello", "language": "en", "segments": []},
            speaker_segments=[],
            annotated_segments=[],
            output_paths={"json_path": "/out/test.json", "csv_path": "/out/test.csv"},
        )
        assert ctx.is_complete()

    def test_errors_list_accumulates(self):
        ctx = PipelineContext(source="test.wav")
        ctx.errors.append("transcription: model failed")
        ctx.errors.append("emotion: segment too short")
        assert len(ctx.errors) == 2

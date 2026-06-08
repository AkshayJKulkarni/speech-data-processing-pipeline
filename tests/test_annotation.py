"""
Unit and integration tests for the annotation engine.

Test strategy:
──────────────
- _overlap tests: pure arithmetic, zero dependencies.
- _merger tests: synthetic dataclass instances, no I/O, no models.
- _serialiser tests: real temp files, synthetic AnnotatedSegment data.
- export_annotations tests: full integration with synthetic inference
  result objects and a real temp output directory.

Zero network calls. Zero model loading.
"""

import csv
import json
import pytest
from pathlib import Path
from datetime import datetime

from src.annotation._schema import AnnotatedSegment, AnnotationMetadata, AnnotationResult
from src.annotation._merger import (
    _overlap,
    align_transcript_to_speakers,
    _build_speaker_text_map,
    _build_emotion_map,
    merge_segments,
)
from src.annotation._serialiser import write_json, write_csv, write_metadata, write_all
from src.annotation.writer import export_annotations, _validate_config, _validate_has_content
from src.utils.exceptions import AnnotationError


# ── Synthetic inference result builders ──────────────────────────────────────

def make_transcript(segments: list[tuple[float, float, str]], language: str = "en"):
    """Build a synthetic TranscriptResult without importing the real class."""
    from types import SimpleNamespace
    t_segs = [
        SimpleNamespace(start=s, end=e, text=t)
        for s, e, t in segments
    ]
    return SimpleNamespace(language=language, text=" ".join(t for _, _, t in segments), segments=t_segs)


def make_diarization(segments: list[tuple[str, float, float]], num_speakers: int = None):
    """Build a synthetic SpeakerDiarizationResult."""
    from types import SimpleNamespace
    s_segs = [
        SimpleNamespace(speaker=sp, start=s, end=e)
        for sp, s, e in segments
    ]
    return SimpleNamespace(
        segments=s_segs,
        num_speakers=num_speakers or len({sp for sp, _, _ in segments}),
    )


def make_emotion(segments: list[tuple[str, float, float, str, float]]):
    """Build a synthetic EmotionResult."""
    from types import SimpleNamespace
    e_segs = [
        SimpleNamespace(speaker=sp, start=s, end=e, emotion=emo, confidence=c)
        for sp, s, e, emo, c in segments
    ]
    return SimpleNamespace(segments=e_segs)


def make_annotated_segment(**kwargs) -> AnnotatedSegment:
    defaults = dict(
        speaker="SPEAKER_00", start_time=0.0, end_time=2.0,
        transcript="hello", emotion="happy", confidence=0.9
    )
    defaults.update(kwargs)
    return AnnotatedSegment(**defaults)


def make_config(output_dir: str = "data/output", version: str = "1.0.0") -> dict:
    return {
        "paths":    {"output_data": output_dir},
        "pipeline": {"version": version, "name": "speech-data-processing"},
    }


def make_metadata(**kwargs) -> AnnotationMetadata:
    defaults = dict(
        source_file="test.wav", processed_at="2024-01-01T00:00:00Z",
        duration_seconds=10.0, language="en", num_speakers=2,
        num_segments=3, pipeline_version="1.0.0",
        dominant_emotion="happy", has_transcript=True,
        has_diarization=True, has_emotion=True,
    )
    defaults.update(kwargs)
    return AnnotationMetadata(**defaults)


# ── _overlap ──────────────────────────────────────────────────────────────────

class TestOverlap:
    def test_full_overlap(self):
        assert _overlap(0.0, 2.0, 0.0, 2.0) == pytest.approx(2.0)

    def test_partial_overlap(self):
        assert _overlap(0.0, 2.0, 1.0, 3.0) == pytest.approx(1.0)

    def test_no_overlap_adjacent(self):
        assert _overlap(0.0, 1.0, 1.0, 2.0) == pytest.approx(0.0)

    def test_no_overlap_gap(self):
        assert _overlap(0.0, 1.0, 2.0, 3.0) == pytest.approx(0.0)

    def test_contained(self):
        # b fully inside a
        assert _overlap(0.0, 4.0, 1.0, 3.0) == pytest.approx(2.0)

    def test_reversed_interval_returns_zero(self):
        # end < start on second interval
        assert _overlap(0.0, 1.0, 3.0, 2.0) == pytest.approx(0.0)


# ── align_transcript_to_speakers ─────────────────────────────────────────────

class TestAlignTranscriptToSpeakers:
    def test_clear_assignment(self):
        transcript = make_transcript([(0.0, 1.5, "hello"), (2.0, 3.5, "world")])
        diarization = make_diarization([("SPEAKER_00", 0.0, 2.0), ("SPEAKER_01", 2.0, 4.0)])
        result = align_transcript_to_speakers(
            transcript.segments, diarization.segments
        )
        assert result[(0.0, 1.5)] == "SPEAKER_00"
        assert result[(2.0, 3.5)] == "SPEAKER_01"

    def test_no_overlap_assigns_unknown(self):
        transcript  = make_transcript([(5.0, 6.0, "late")])
        diarization = make_diarization([("SPEAKER_00", 0.0, 2.0)])
        result = align_transcript_to_speakers(
            transcript.segments, diarization.segments
        )
        assert result[(5.0, 6.0)] == "SPEAKER_UNKNOWN"

    def test_empty_transcript(self):
        transcript  = make_transcript([])
        diarization = make_diarization([("SPEAKER_00", 0.0, 2.0)])
        result = align_transcript_to_speakers(
            transcript.segments, diarization.segments
        )
        assert result == {}


# ── _build_speaker_text_map ───────────────────────────────────────────────────

class TestBuildSpeakerTextMap:
    def test_text_assigned_to_correct_speaker(self):
        transcript  = make_transcript([(0.0, 1.0, "Hello"), (1.0, 2.0, "there")])
        diarization = make_diarization([("SPEAKER_00", 0.0, 2.0)])
        text_map = _build_speaker_text_map(
            transcript.segments, diarization.segments
        )
        assert text_map[(0.0, 2.0)] == "Hello there"

    def test_multiple_speakers_get_separate_text(self):
        transcript  = make_transcript([(0.0, 1.0, "Hi"), (1.5, 2.5, "Bye")])
        diarization = make_diarization([
            ("SPEAKER_00", 0.0, 1.2),
            ("SPEAKER_01", 1.2, 3.0),
        ])
        text_map = _build_speaker_text_map(
            transcript.segments, diarization.segments
        )
        assert text_map[(0.0, 1.2)] == "Hi"
        assert text_map[(1.2, 3.0)] == "Bye"

    def test_no_overlap_leaves_empty_text(self):
        transcript  = make_transcript([(10.0, 11.0, "late")])
        diarization = make_diarization([("SPEAKER_00", 0.0, 2.0)])
        text_map = _build_speaker_text_map(
            transcript.segments, diarization.segments
        )
        assert text_map[(0.0, 2.0)] == ""


# ── _build_emotion_map ────────────────────────────────────────────────────────

class TestBuildEmotionMap:
    def test_keys_are_start_end_tuples(self):
        emotion = make_emotion([("SPEAKER_00", 0.0, 1.0, "happy", 0.9)])
        emap = _build_emotion_map(emotion.segments)
        assert (0.0, 1.0) in emap
        assert emap[(0.0, 1.0)] == ("happy", 0.9)

    def test_multiple_segments(self):
        emotion = make_emotion([
            ("SPEAKER_00", 0.0, 1.0, "happy",   0.9),
            ("SPEAKER_01", 1.0, 2.0, "neutral",  0.7),
        ])
        emap = _build_emotion_map(emotion.segments)
        assert emap[(1.0, 2.0)] == ("neutral", 0.7)

    def test_empty_returns_empty_dict(self):
        emotion = make_emotion([])
        assert _build_emotion_map(emotion.segments) == {}


# ── merge_segments ────────────────────────────────────────────────────────────

class TestMergeSegments:
    def test_full_merge(self):
        transcript  = make_transcript([(0.0, 1.0, "Hello"), (1.5, 2.5, "World")])
        diarization = make_diarization([
            ("SPEAKER_00", 0.0, 1.2),
            ("SPEAKER_01", 1.2, 3.0),
        ])
        emotion = make_emotion([
            ("SPEAKER_00", 0.0, 1.2, "happy",   0.90),
            ("SPEAKER_01", 1.2, 3.0, "neutral",  0.75),
        ])
        segments = merge_segments(transcript, diarization, emotion)

        assert len(segments) == 2
        assert segments[0].speaker    == "SPEAKER_00"
        assert segments[0].transcript == "Hello"
        assert segments[0].emotion    == "happy"
        assert segments[0].confidence == pytest.approx(0.90)
        assert segments[1].speaker    == "SPEAKER_01"
        assert segments[1].emotion    == "neutral"

    def test_no_diarization_uses_transcript_grid(self):
        transcript = make_transcript([(0.0, 1.0, "Hello"), (1.0, 2.0, "World")])
        segments = merge_segments(transcript, None, None)

        assert len(segments) == 2
        assert segments[0].speaker    == "SPEAKER_UNKNOWN"
        assert segments[0].transcript == "Hello"
        assert segments[0].emotion    == "unknown"

    def test_no_transcript_leaves_empty_text(self):
        diarization = make_diarization([("SPEAKER_00", 0.0, 2.0)])
        segments = merge_segments(None, diarization, None)

        assert len(segments) == 1
        assert segments[0].transcript == ""
        assert segments[0].speaker    == "SPEAKER_00"

    def test_no_emotion_uses_unknown(self):
        transcript  = make_transcript([(0.0, 1.0, "Hi")])
        diarization = make_diarization([("SPEAKER_00", 0.0, 1.0)])
        segments = merge_segments(transcript, diarization, None)

        assert segments[0].emotion    == "unknown"
        assert segments[0].confidence == 0.0

    def test_both_none_returns_empty(self):
        assert merge_segments(None, None, None) == []

    def test_output_sorted_by_start_time(self):
        diarization = make_diarization([
            ("SPEAKER_01", 2.0, 3.0),
            ("SPEAKER_00", 0.0, 2.0),
        ])
        segments = merge_segments(None, diarization, None)
        starts = [s.start_time for s in segments]
        assert starts == sorted(starts)


# ── AnnotatedSegment ──────────────────────────────────────────────────────────

class TestAnnotatedSegment:
    def test_duration(self):
        seg = make_annotated_segment(start_time=1.0, end_time=3.5)
        assert seg.duration() == pytest.approx(2.5)

    def test_to_dict_keys(self):
        seg = make_annotated_segment()
        d = seg.to_dict()
        assert set(d.keys()) == {
            "speaker", "start_time", "end_time",
            "transcript", "emotion", "confidence"
        }

    def test_to_dict_json_serializable(self):
        json.dumps(make_annotated_segment().to_dict())


# ── AnnotationMetadata ────────────────────────────────────────────────────────

class TestAnnotationMetadata:
    def test_to_dict_all_fields(self):
        m = make_metadata()
        d = m.to_dict()
        expected_keys = {
            "source_file", "processed_at", "duration_seconds",
            "language", "num_speakers", "num_segments",
            "pipeline_version", "dominant_emotion",
            "has_transcript", "has_diarization", "has_emotion",
        }
        assert set(d.keys()) == expected_keys

    def test_json_serializable(self):
        json.dumps(make_metadata().to_dict())


# ── _serialiser ───────────────────────────────────────────────────────────────

class TestSerialiser:
    @pytest.fixture
    def segments(self) -> list[AnnotatedSegment]:
        return [
            make_annotated_segment(speaker="SPEAKER_00", start_time=0.0, end_time=1.5,
                                   transcript="Hello", emotion="happy", confidence=0.9),
            make_annotated_segment(speaker="SPEAKER_01", start_time=1.5, end_time=3.0,
                                   transcript="World", emotion="neutral", confidence=0.7),
        ]

    @pytest.fixture
    def metadata(self) -> AnnotationMetadata:
        return make_metadata(num_segments=2)

    def test_write_json_creates_file(self, tmp_path, segments, metadata):
        out = tmp_path / "test.json"
        write_json(segments, metadata, out)
        assert out.exists()

    def test_write_json_structure(self, tmp_path, segments, metadata):
        out = tmp_path / "test.json"
        write_json(segments, metadata, out)
        data = json.loads(out.read_text())
        assert "metadata" in data
        assert "segments" in data
        assert len(data["segments"]) == 2
        assert data["segments"][0]["speaker"] == "SPEAKER_00"
        assert data["segments"][0]["transcript"] == "Hello"

    def test_write_csv_creates_file(self, tmp_path, segments):
        out = tmp_path / "test.csv"
        write_csv(segments, out)
        assert out.exists()

    def test_write_csv_headers(self, tmp_path, segments):
        out = tmp_path / "test.csv"
        write_csv(segments, out)
        with open(out) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        assert "speaker"    in headers
        assert "transcript" in headers
        assert "emotion"    in headers
        assert "confidence" in headers
        assert "duration"   in headers

    def test_write_csv_row_count(self, tmp_path, segments):
        out = tmp_path / "test.csv"
        write_csv(segments, out)
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_write_metadata_creates_file(self, tmp_path, metadata):
        out = tmp_path / "meta.json"
        write_metadata(metadata, out)
        assert out.exists()

    def test_write_metadata_content(self, tmp_path, metadata):
        out = tmp_path / "meta.json"
        write_metadata(metadata, out)
        data = json.loads(out.read_text())
        assert data["language"]    == "en"
        assert data["num_speakers"] == 2

    def test_write_all_creates_three_files(self, tmp_path, segments, metadata):
        result = AnnotationResult(segments=segments, metadata=metadata)
        result = write_all(result, "interview", tmp_path)
        assert Path(result.json_path).exists()
        assert Path(result.csv_path).exists()
        assert Path(result.metadata_path).exists()

    def test_write_all_correct_stems(self, tmp_path, segments, metadata):
        result = AnnotationResult(segments=segments, metadata=metadata)
        result = write_all(result, "interview", tmp_path)
        assert Path(result.json_path).stem     == "interview"
        assert Path(result.csv_path).stem      == "interview"
        assert "interview" in Path(result.metadata_path).name

    def test_write_all_creates_output_dir(self, tmp_path, segments, metadata):
        nested = tmp_path / "deep" / "nested"
        result = AnnotationResult(segments=segments, metadata=metadata)
        write_all(result, "test", nested)
        assert nested.exists()


# ── export_annotations ────────────────────────────────────────────────────────

class TestExportAnnotations:
    @pytest.fixture
    def full_inputs(self):
        transcript  = make_transcript([(0.0, 1.2, "Hello everyone"), (1.5, 3.0, "How are you")])
        diarization = make_diarization([
            ("SPEAKER_00", 0.0, 1.5),
            ("SPEAKER_01", 1.5, 3.5),
        ])
        emotion = make_emotion([
            ("SPEAKER_00", 0.0, 1.5, "happy",   0.92),
            ("SPEAKER_01", 1.5, 3.5, "neutral",  0.78),
        ])
        return transcript, diarization, emotion

    def test_returns_annotation_result(self, tmp_path, full_inputs):
        t, d, e = full_inputs
        result = export_annotations(t, d, e, make_config(str(tmp_path)))
        assert isinstance(result, AnnotationResult)

    def test_correct_segment_count(self, tmp_path, full_inputs):
        t, d, e = full_inputs
        result = export_annotations(t, d, e, make_config(str(tmp_path)))
        assert len(result.segments) == 2

    def test_segments_have_all_fields(self, tmp_path, full_inputs):
        t, d, e = full_inputs
        result = export_annotations(t, d, e, make_config(str(tmp_path)))
        seg = result.segments[0]
        assert seg.speaker    == "SPEAKER_00"
        assert seg.transcript == "Hello everyone"
        assert seg.emotion    == "happy"
        assert seg.confidence == pytest.approx(0.92)

    def test_metadata_populated(self, tmp_path, full_inputs):
        t, d, e = full_inputs
        result = export_annotations(t, d, e, make_config(str(tmp_path)))
        meta = result.metadata
        assert meta.language       == "en"
        assert meta.num_speakers   == 2
        assert meta.num_segments   == 2
        assert meta.has_transcript is True
        assert meta.has_emotion    is True
        assert meta.pipeline_version == "1.0.0"

    def test_three_files_written(self, tmp_path, full_inputs):
        t, d, e = full_inputs
        result = export_annotations(t, d, e, make_config(str(tmp_path)))
        assert Path(result.json_path).exists()
        assert Path(result.csv_path).exists()
        assert Path(result.metadata_path).exists()

    def test_stem_derived_from_source_path(self, tmp_path, full_inputs):
        t, d, e = full_inputs
        result = export_annotations(
            t, d, e, make_config(str(tmp_path)),
            source_path="data/processed/interview.wav"
        )
        assert Path(result.json_path).stem == "interview"

    def test_explicit_output_stem(self, tmp_path, full_inputs):
        t, d, e = full_inputs
        result = export_annotations(
            t, d, e, make_config(str(tmp_path)),
            output_stem="custom_name"
        )
        assert Path(result.json_path).stem == "custom_name"

    def test_output_paths_dict(self, tmp_path, full_inputs):
        t, d, e = full_inputs
        result = export_annotations(t, d, e, make_config(str(tmp_path)))
        paths = result.output_paths()
        assert "json_path"     in paths
        assert "csv_path"      in paths
        assert "metadata_path" in paths

    def test_transcript_only_no_diarization(self, tmp_path):
        transcript = make_transcript([(0.0, 1.0, "Hello"), (1.0, 2.0, "World")])
        result = export_annotations(transcript, None, None, make_config(str(tmp_path)))
        assert len(result.segments) == 2
        assert result.segments[0].speaker == "SPEAKER_UNKNOWN"

    def test_diarization_only_no_transcript(self, tmp_path):
        diarization = make_diarization([("SPEAKER_00", 0.0, 2.0)])
        result = export_annotations(None, diarization, None, make_config(str(tmp_path)))
        assert len(result.segments) == 1
        assert result.segments[0].transcript == ""

    def test_both_absent_raises(self, tmp_path):
        with pytest.raises(AnnotationError, match="both transcript and diarization are absent"):
            export_annotations(None, None, None, make_config(str(tmp_path)))

    def test_invalid_config_raises(self, tmp_path, full_inputs):
        t, d, e = full_inputs
        bad_config = {"paths": {}}   # missing output_data and pipeline
        with pytest.raises(AnnotationError, match="Missing required config key"):
            export_annotations(t, d, e, bad_config)

    def test_json_output_is_valid(self, tmp_path, full_inputs):
        t, d, e = full_inputs
        result = export_annotations(t, d, e, make_config(str(tmp_path)))
        data = json.loads(Path(result.json_path).read_text())
        assert "metadata" in data
        assert "segments" in data
        assert data["metadata"]["language"] == "en"

    def test_csv_output_is_valid(self, tmp_path, full_inputs):
        t, d, e = full_inputs
        result = export_annotations(t, d, e, make_config(str(tmp_path)))
        with open(result.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["speaker"]    == "SPEAKER_00"
        assert rows[0]["transcript"] == "Hello everyone"
        assert rows[0]["emotion"]    == "happy"

    def test_metadata_processed_at_is_iso8601(self, tmp_path, full_inputs):
        t, d, e = full_inputs
        result = export_annotations(t, d, e, make_config(str(tmp_path)))
        # Should parse without error
        datetime.strptime(result.metadata.processed_at, "%Y-%m-%dT%H:%M:%SZ")

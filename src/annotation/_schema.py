"""
Annotation data schema — all output dataclasses for the annotation package.

Design decision: Schema is isolated from merger and serialiser logic.
Any module that needs to type-hint against AnnotatedSegment or AnnotationResult
can import from here without pulling in pandas, json, or csv dependencies.

These three classes define the complete output contract of the pipeline:
  AnnotatedSegment  — one row of the final annotation (one speaker turn)
  AnnotationMetadata — file-level metadata written to metadata.json
  AnnotationResult   — top-level container returned by export_annotations()
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AnnotatedSegment:
    """
    The fully merged output record for one speaker turn.

    Combines speaker identity (diarization), spoken text (transcription),
    and emotional state (emotion classification) for a single time window.

    This is the atomic unit of the annotation dataset — one row in the CSV,
    one entry in the segments array of the JSON.

    Fields:
        speaker:    Speaker label from diarization. "SPEAKER_00", etc.
        start_time: Turn start in seconds (3 decimal places = ms precision).
        end_time:   Turn end in seconds.
        transcript: Whisper text for this turn. Empty string if no transcript
                    segment overlapped this speaker turn.
        emotion:    Emotion label. "unknown" if segment was too short or
                    emotion classification was not run.
        confidence: Emotion model confidence [0.0, 1.0]. 0.0 if unknown.
    """
    speaker:    str
    start_time: float
    end_time:   float
    transcript: str
    emotion:    str
    confidence: float

    def duration(self) -> float:
        """Returns segment duration in seconds."""
        return round(self.end_time - self.start_time, 3)

    def to_dict(self) -> dict:
        """Returns a JSON-serializable dict matching the required output schema."""
        return {
            "speaker":    self.speaker,
            "start_time": self.start_time,
            "end_time":   self.end_time,
            "transcript": self.transcript,
            "emotion":    self.emotion,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class AnnotationMetadata:
    """
    File-level metadata written to metadata.json alongside the annotation files.

    Design decision: Metadata is a separate file (not embedded in annotations.json)
    so downstream tools can inspect recording-level properties — duration, language,
    speaker count — without parsing the full segment list.

    This is the standard practice in speech dataset releases (LibriSpeech,
    VoxCeleb, CommonVoice all ship separate metadata manifests).

    Fields:
        source_file:        Original filename (not full path — portable).
        processed_at:       ISO-8601 UTC timestamp of when export was run.
        duration_seconds:   Total audio duration in seconds.
        language:           BCP-47 language code from Whisper detection.
        num_speakers:       Number of unique speaker labels from diarization.
        num_segments:       Number of annotated segments in the output.
        pipeline_version:   From config["pipeline"]["version"].
        dominant_emotion:   Most frequent emotion across all segments.
        has_transcript:     Whether transcription was available.
        has_diarization:    Whether diarization was available.
        has_emotion:        Whether emotion classification was available.
    """
    source_file:      str
    processed_at:     str    # ISO-8601 UTC
    duration_seconds: float
    language:         str
    num_speakers:     int
    num_segments:     int
    pipeline_version: str
    dominant_emotion: str
    has_transcript:   bool
    has_diarization:  bool
    has_emotion:      bool

    def to_dict(self) -> dict:
        """Returns a JSON-serializable dict."""
        return {
            "source_file":      self.source_file,
            "processed_at":     self.processed_at,
            "duration_seconds": self.duration_seconds,
            "language":         self.language,
            "num_speakers":     self.num_speakers,
            "num_segments":     self.num_segments,
            "pipeline_version": self.pipeline_version,
            "dominant_emotion": self.dominant_emotion,
            "has_transcript":   self.has_transcript,
            "has_diarization":  self.has_diarization,
            "has_emotion":      self.has_emotion,
        }


@dataclass
class AnnotationResult:
    """
    Top-level container returned by export_annotations().

    Holds the in-memory merged data AND the paths of the files written
    to disk, so callers can either consume the data directly (in tests,
    downstream processing) or locate the output files (for logging,
    PipelineContext).

    Fields:
        segments:      List of AnnotatedSegment — the merged annotation data.
        metadata:      AnnotationMetadata — recording-level summary.
        json_path:     Absolute path to annotations.json.
        csv_path:      Absolute path to annotations.csv.
        metadata_path: Absolute path to metadata.json.
    """
    segments:      list[AnnotatedSegment] = field(default_factory=list)
    metadata:      AnnotationMetadata | None = None
    json_path:     str = ""
    csv_path:      str = ""
    metadata_path: str = ""

    def output_paths(self) -> dict[str, str]:
        """Returns the three output file paths as a dict for PipelineContext."""
        return {
            "json_path":     self.json_path,
            "csv_path":      self.csv_path,
            "metadata_path": self.metadata_path,
        }

"""
Annotation writer — public interface for the annotation package.

Responsibility:
    Accept all three inference results and pipeline configuration,
    merge them into AnnotatedSegment records, compute metadata,
    write three output files, and return a populated AnnotationResult.

Input contract:
    transcript      : TranscriptResult | None
    speaker_segments: SpeakerDiarizationResult | None
    emotion_result  : EmotionResult | None
    config          : dict  — the full pipeline config (config.yaml)
                      Required keys:
                        paths.output_data : str — output directory
                        pipeline.name     : str — for metadata
                        pipeline.version  : str — for metadata
                      Optional keys:
                        annotation.stem   : str — output filename stem
                                                  (defaults to "annotations")

Output contract:
    AnnotationResult:
        segments      : list[AnnotatedSegment]
        metadata      : AnnotationMetadata
        json_path     : str — absolute path to annotations.json
        csv_path      : str — absolute path to annotations.csv
        metadata_path : str — absolute path to metadata.json

Pipeline position:
    classify_emotion() → [EmotionResult]
        transcribe()      → [TranscriptResult]
        diarize()         → [SpeakerDiarizationResult]
            → export_annotations(transcript, diarization, emotion, config)
                → [AnnotationResult]
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from src.utils import get_logger, AnnotationError
from src.annotation._schema import AnnotatedSegment, AnnotationMetadata, AnnotationResult
from src.annotation._merger import merge_segments
from src.annotation._serialiser import write_all

if TYPE_CHECKING:
    from src.inference.transcriber import TranscriptResult
    from src.inference.diarizer import SpeakerDiarizationResult
    from src.inference.emotion_classifier import EmotionResult

logger = get_logger(__name__)


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_config(config: dict) -> None:
    """
    Validates that required config keys are present.

    Args:
        config: Full pipeline config dict (config.yaml).

    Raises:
        AnnotationError: If required keys are missing.
    """
    try:
        _ = config["paths"]["output_data"]
        _ = config["pipeline"]["version"]
    except KeyError as e:
        raise AnnotationError(f"Missing required config key for annotation: {e}") from e


def _validate_has_content(
    transcript: Optional["TranscriptResult"],
    speaker_segments: Optional["SpeakerDiarizationResult"],
) -> None:
    """
    Raises AnnotationError if there is literally nothing to annotate.

    We require at least one of transcript OR diarization to produce
    meaningful output. Having only emotion segments with no time grid
    would produce nothing useful.

    Args:
        transcript:       TranscriptResult or None.
        speaker_segments: SpeakerDiarizationResult or None.

    Raises:
        AnnotationError: If both are None or empty.
    """
    has_transcript  = transcript is not None and bool(transcript.segments)
    has_diarization = speaker_segments is not None and bool(speaker_segments.segments)

    if not has_transcript and not has_diarization:
        raise AnnotationError(
            "Cannot export annotations: both transcript and diarization are absent. "
            "At least one must succeed for the annotation engine to produce output."
        )


# ── Metadata builder ──────────────────────────────────────────────────────────

def _build_metadata(
    source_path:      str,
    segments:         list[AnnotatedSegment],
    transcript:       Optional["TranscriptResult"],
    speaker_segments: Optional["SpeakerDiarizationResult"],
    emotion_result:   Optional["EmotionResult"],
    pipeline_version: str,
) -> AnnotationMetadata:
    """
    Constructs the AnnotationMetadata for this annotation run.

    Args:
        source_path:      Path to the processed WAV (used for source_file name).
        segments:         Merged AnnotatedSegment list.
        transcript:       TranscriptResult or None.
        speaker_segments: SpeakerDiarizationResult or None.
        emotion_result:   EmotionResult or None.
        pipeline_version: From config["pipeline"]["version"].

    Returns:
        Populated AnnotationMetadata instance.
    """
    # Source filename only — not the full path (for portability)
    source_file = Path(source_path).name if source_path else "unknown"

    # ISO-8601 UTC timestamp
    processed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Duration: end time of the last segment
    duration = segments[-1].end_time if segments else 0.0

    # Language from transcript, fallback to "unknown"
    language = transcript.language if transcript else "unknown"

    # Speaker count from diarization
    num_speakers = speaker_segments.num_speakers if speaker_segments else 0

    # Dominant emotion across all segments
    emotion_counts: dict[str, int] = {}
    for seg in segments:
        emotion_counts[seg.emotion] = emotion_counts.get(seg.emotion, 0) + 1
    dominant_emotion = (
        max(emotion_counts, key=emotion_counts.get)  # type: ignore[arg-type]
        if emotion_counts else "unknown"
    )

    return AnnotationMetadata(
        source_file      = source_file,
        processed_at     = processed_at,
        duration_seconds = round(duration, 3),
        language         = language,
        num_speakers     = num_speakers,
        num_segments     = len(segments),
        pipeline_version = pipeline_version,
        dominant_emotion = dominant_emotion,
        has_transcript   = transcript is not None and bool(transcript.segments),
        has_diarization  = speaker_segments is not None and bool(speaker_segments.segments),
        has_emotion      = emotion_result is not None and bool(emotion_result.segments),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def export_annotations(
    transcript:       Optional["TranscriptResult"],
    speaker_segments: Optional["SpeakerDiarizationResult"],
    emotion_result:   Optional["EmotionResult"],
    config:           dict,
    *,
    source_path:      str = "",
    output_stem:      Optional[str] = None,
) -> AnnotationResult:
    """
    Merges all inference results and writes annotation files to disk.

    Handles partial pipeline runs gracefully — any combination of
    transcript, speaker_segments, and emotion_result being None is
    accepted as long as at least one provides a time grid.

    Args:
        transcript:       Output of transcribe(). May be None if transcription
                          failed or was skipped.
        speaker_segments: Output of diarize(). May be None if diarization
                          failed or was skipped.
        emotion_result:   Output of classify_emotion(). May be None if emotion
                          classification failed or was skipped.
        config:           Full pipeline config dict (config.yaml).
                          Reads: paths.output_data, pipeline.version,
                          pipeline.name.
        source_path:      Path to the processed audio file. Used to derive
                          the output filename stem and metadata source_file.
                          Defaults to "" (stem will be "annotations").
        output_stem:      Explicit output filename stem override. If None,
                          derived from source_path stem, fallback "annotations".

    Returns:
        AnnotationResult with segments, metadata, and three file paths.

    Raises:
        AnnotationError: If config is invalid, both transcript and diarization
                         are absent, or file writing fails.

    Example:
        >>> result = export_annotations(
        ...     transcript=transcript_result,
        ...     speaker_segments=diarization_result,
        ...     emotion_result=emotion_result,
        ...     config=config,
        ...     source_path="data/processed/interview.wav",
        ... )
        >>> print(result.json_path)
        >>> print(result.metadata.num_speakers)
        >>> for seg in result.segments:
        ...     print(seg.speaker, seg.transcript, seg.emotion)
    """
    # ── Validate ───────────────────────────────────────────────────────────────
    _validate_config(config)
    _validate_has_content(transcript, speaker_segments)

    output_dir = Path(config["paths"]["output_data"]).resolve()
    version    = config["pipeline"]["version"]

    # Derive stem from source_path or use explicit override
    if output_stem:
        stem = output_stem
    elif source_path:
        stem = Path(source_path).stem
    else:
        stem = "annotations"

    logger.info(
        f"Annotation export started | stem='{stem}' output='{output_dir}'"
    )

    # ── Merge ──────────────────────────────────────────────────────────────────
    segments = merge_segments(transcript, speaker_segments, emotion_result)

    if not segments:
        raise AnnotationError(
            "Merge produced zero segments. "
            "Check that transcript and diarization outputs are non-empty."
        )

    # ── Build metadata ─────────────────────────────────────────────────────────
    metadata = _build_metadata(
        source_path      = source_path,
        segments         = segments,
        transcript       = transcript,
        speaker_segments = speaker_segments,
        emotion_result   = emotion_result,
        pipeline_version = version,
    )

    result = AnnotationResult(segments=segments, metadata=metadata)

    # ── Write files ────────────────────────────────────────────────────────────
    result = write_all(result, stem, output_dir)

    logger.info(
        f"Annotation export complete | "
        f"segments={metadata.num_segments} "
        f"speakers={metadata.num_speakers} "
        f"language='{metadata.language}' "
        f"duration={metadata.duration_seconds:.1f}s"
    )

    return result


# ── Legacy shim — keeps runner.py working without changes ─────────────────────

def export(context, output_dir: str) -> dict[str, str]:
    """
    Legacy adapter called by pipeline/runner.py.

    Unpacks PipelineContext fields and delegates to export_annotations().
    Returns the output_paths dict expected by PipelineContext.output_paths.

    Args:
        context:    PipelineContext with transcript, speaker_segments,
                    annotated_segments, processed_path populated.
        output_dir: Output directory path string.

    Returns:
        Dict with json_path, csv_path, metadata_path keys.
    """
    config = {
        "paths":    {"output_data": output_dir},
        "pipeline": {"version": "1.0.0", "name": "speech-data-processing"},
    }

    result = export_annotations(
        transcript       = context.transcript,
        speaker_segments = context.speaker_segments,
        emotion_result   = context.annotated_segments,
        config           = config,
        source_path      = context.processed_path or "",
    )

    return result.output_paths()

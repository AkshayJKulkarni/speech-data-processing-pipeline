"""
PipelineContext — the shared state object for a single pipeline run.

Design decision: Instead of passing 5-6 loosely typed arguments between
every stage, all stage outputs are stored in one typed dataclass. This:

1. Makes function signatures clean — every stage takes a context, mutates
   it, and returns it.
2. Makes the pipeline resumable — serialize context to JSON to restart
   from any completed stage.
3. Makes testing clean — build a partially-filled context to test any
   single stage in isolation.
4. Documents the data flow — the fields ARE the pipeline's data model.

Fields are Optional because a freshly created context has no outputs yet.
Each stage populates its own field and passes context forward.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.inference.transcriber import TranscriptResult
    from src.inference.diarizer import SpeakerDiarizationResult
    from src.inference.emotion_classifier import EmotionResult
    from src.annotation._schema import AnnotationResult


@dataclass
class PipelineContext:
    """
    Carries all inputs and outputs for one end-to-end pipeline run.

    Lifecycle:
        created          -> source only
        after acquire    -> raw_path set
        after preprocess -> processed_path set
        after transcribe -> transcript set
        after diarize    -> speaker_segments set
        after emotion    -> annotated_segments set
        after export     -> output_paths set
    """

    # -- Input ---------------------------------------------------------------
    source: str                         # original YouTube URL or file path

    # -- Stage outputs (populated sequentially) ------------------------------
    raw_path: Optional[str] = None              # data/raw/<file>
    processed_path: Optional[str] = None        # data/processed/<file>.wav

    transcript: Optional["TranscriptResult"] = None
    # Shape: TranscriptResult(language, text, segments=[TranscriptSegment(...)])

    speaker_segments: Optional["SpeakerDiarizationResult"] = None
    # Shape: SpeakerDiarizationResult(segments=[SpeakerSegment(...)], num_speakers)

    annotated_segments: Optional["EmotionResult"] = None
    # Shape: EmotionResult(segments=[EmotionSegment(speaker, start, end, emotion, confidence)])

    output_paths: Optional["AnnotationResult"] = None
    # Shape: AnnotationResult(segments, metadata, json_path, csv_path, metadata_path)

    # -- Metadata ------------------------------------------------------------
    errors: list[str] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)
    # Shape: {"acquisition": 1.23, "preprocessing": 4.56, ...}

    def is_complete(self) -> bool:
        """Returns True only if all pipeline stages have produced output."""
        return all([
            self.raw_path,
            self.processed_path,
            self.transcript,
            self.speaker_segments,
            self.annotated_segments,
            self.output_paths,
        ])

    def summary(self) -> str:
        """Returns a human-readable one-line status of the context."""
        stages = {
            "acquired":     self.raw_path is not None,
            "preprocessed": self.processed_path is not None,
            "transcribed":  self.transcript is not None,
            "diarized":     self.speaker_segments is not None,
            "emotion":      self.annotated_segments is not None,
            "exported":     self.output_paths is not None,
        }
        completed = [k for k, v in stages.items() if v]
        pending   = [k for k, v in stages.items() if not v]
        return (
            f"source='{self.source}' | "
            f"done={completed} | "
            f"pending={pending} | "
            f"errors={len(self.errors)}"
        )

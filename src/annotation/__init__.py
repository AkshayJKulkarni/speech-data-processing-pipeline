"""
annotation package public API.

Input contract:
    transcript       : TranscriptResult | None
    speaker_segments : SpeakerDiarizationResult | None
    emotion_result   : EmotionResult | None
    config           : dict  — full pipeline config (config.yaml)

Output contract:
    AnnotationResult:
        segments      : list[AnnotatedSegment]
        metadata      : AnnotationMetadata
        json_path     : str
        csv_path      : str
        metadata_path : str

Callers import from `src.annotation`, never from submodules directly.
"""

from src.annotation.writer import export_annotations, export
from src.annotation._schema import AnnotatedSegment, AnnotationMetadata, AnnotationResult

__all__ = [
    "export_annotations",
    "export",
    "AnnotatedSegment",
    "AnnotationMetadata",
    "AnnotationResult",
]

"""
Pipeline-wide exception hierarchy.

Design decision: Define all custom exceptions in one place so every
package raises predictable, typed errors. Callers import from
`utils.exceptions` — they never catch bare Exception.

Hierarchy:
    PipelineError                   ← base for all pipeline errors
    ├── AcquisitionError            ← download / file-copy failures
    ├── PreprocessingError          ← audio cleaning failures
    ├── InferenceError              ← model execution failures
    │   ├── TranscriptionError
    │   ├── DiarizationError
    │   └── EmotionError
    ├── AnnotationError             ← serialization / export failures
    └── ConfigurationError          ← bad or missing config values
"""


class PipelineError(Exception):
    """Base class for all pipeline-specific errors."""


class AcquisitionError(PipelineError):
    """Raised when audio/video acquisition fails."""


class PreprocessingError(PipelineError):
    """Raised when audio preprocessing fails."""


class InferenceError(PipelineError):
    """Raised when a model inference step fails."""


class TranscriptionError(InferenceError):
    """Raised when speech-to-text inference fails."""


class DiarizationError(InferenceError):
    """Raised when speaker diarization fails."""


class EmotionError(InferenceError):
    """Raised when emotion classification fails."""


class AnnotationError(PipelineError):
    """Raised when annotation serialization or export fails."""


class ConfigurationError(PipelineError):
    """Raised when config values are missing or invalid."""

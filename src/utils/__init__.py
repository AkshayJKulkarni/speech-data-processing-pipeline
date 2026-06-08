"""
utils package — shared infrastructure for the entire pipeline.

Every other package imports from here. Nothing imports from
individual utils submodules directly, keeping the internal
layout changeable without touching downstream code.
"""

from src.utils.logging_utils import get_logger
from src.utils.config_utils import load_config, get_required
from src.utils.exceptions import (
    PipelineError,
    AcquisitionError,
    PreprocessingError,
    InferenceError,
    TranscriptionError,
    DiarizationError,
    EmotionError,
    AnnotationError,
    ConfigurationError,
)

__all__ = [
    # logging
    "get_logger",
    # config
    "load_config",
    "get_required",
    # exceptions
    "PipelineError",
    "AcquisitionError",
    "PreprocessingError",
    "InferenceError",
    "TranscriptionError",
    "DiarizationError",
    "EmotionError",
    "AnnotationError",
    "ConfigurationError",
]

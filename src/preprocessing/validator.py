"""
Input validation for the preprocessing module.

Design decision: Validation is separated from transformation logic for the
same reason as in acquisition — it can be tested with zero audio I/O and
raises PreprocessingError early so the processor never receives bad input.
"""

import os
from pathlib import Path

from src.utils import PreprocessingError

# File extensions this module can accept as input.
# Video formats are included because we extract audio from them.
SUPPORTED_INPUT_EXTENSIONS: set[str] = {
    # Audio
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus",
    # Video (audio will be extracted)
    ".mp4", ".mkv", ".webm", ".mov", ".avi",
}


def validate_input_file(path: Path) -> None:
    """
    Validates that the input file exists and has a supported extension.

    Args:
        path: Path to the raw audio or video file.

    Raises:
        PreprocessingError: If file is missing or extension unsupported.
    """
    if not path.is_file():
        raise PreprocessingError(f"Input file not found: '{path}'")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_INPUT_EXTENSIONS:
        raise PreprocessingError(
            f"Unsupported input format '{ext}'. "
            f"Supported: {sorted(SUPPORTED_INPUT_EXTENSIONS)}"
        )


def validate_output_dir(output_dir: Path) -> None:
    """
    Ensures the output directory exists, creating it if necessary.

    Design decision: We create the directory here rather than in the
    processor so that the processor only handles audio logic.

    Args:
        output_dir: Path to the target processed directory.

    Raises:
        PreprocessingError: If the directory cannot be created.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise PreprocessingError(f"Cannot create output directory '{output_dir}': {e}") from e


def validate_audio_config(audio_config: dict) -> None:
    """
    Checks that all required audio config keys are present and valid.

    Args:
        audio_config: The 'audio' section from config.yaml.

    Raises:
        PreprocessingError: If required keys are missing or values are invalid.
    """
    required_keys = {"target_sample_rate", "target_channels", "output_format"}
    missing = required_keys - audio_config.keys()
    if missing:
        raise PreprocessingError(f"Missing audio config keys: {missing}")

    if audio_config["target_sample_rate"] <= 0:
        raise PreprocessingError(
            f"target_sample_rate must be positive, got: {audio_config['target_sample_rate']}"
        )

    if audio_config["target_channels"] not in (1, 2):
        raise PreprocessingError(
            f"target_channels must be 1 (mono) or 2 (stereo), "
            f"got: {audio_config['target_channels']}"
        )

"""
Input validation for the acquisition module.

Design decision: Validation is isolated from download logic so it can
be unit-tested with zero I/O. Raises typed AcquisitionError so the
pipeline orchestrator can catch it without knowing internal details.
"""

import os
import re

from src.utils import AcquisitionError

SUPPORTED_EXTENSIONS: set[str] = {
    ".mp3", ".mp4", ".wav", ".flac", ".ogg",
    ".m4a", ".mkv", ".webm", ".aac", ".opus",
}

_YOUTUBE_PATTERN = re.compile(
    r"(https?://)?(www\.)?"
    r"(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)"
    r"[\w\-]+"
)


def is_youtube_url(source: str) -> bool:
    """Returns True if source matches a YouTube URL pattern."""
    return bool(_YOUTUBE_PATTERN.match(source.strip()))


def is_local_file(source: str) -> bool:
    """Returns True if source is an existing file on disk."""
    return os.path.isfile(source)


def validate_youtube_url(url: str) -> None:
    """
    Raises AcquisitionError if url is not a valid YouTube URL.

    Args:
        url: Raw URL string.

    Raises:
        AcquisitionError: If pattern does not match.
    """
    if not is_youtube_url(url):
        raise AcquisitionError(f"Invalid YouTube URL: '{url}'")


def validate_local_file(path: str) -> None:
    """
    Raises AcquisitionError if file is missing or has unsupported extension.

    Args:
        path: Local filesystem path.

    Raises:
        AcquisitionError: If file is missing or extension unsupported.
    """
    if not os.path.isfile(path):
        raise AcquisitionError(f"Local file not found: '{path}'")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AcquisitionError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )


def validate_source(source: str) -> str:
    """
    Validates the source and returns its detected type.

    Args:
        source: YouTube URL or local file path.

    Returns:
        'youtube' | 'local'

    Raises:
        AcquisitionError: For empty, invalid, or unrecognized inputs.
    """
    if not source or not source.strip():
        raise AcquisitionError("Source input cannot be empty.")

    if is_youtube_url(source):
        validate_youtube_url(source)
        return "youtube"

    if is_local_file(source):
        validate_local_file(source)
        return "local"

    raise AcquisitionError(
        f"Source not recognized as a YouTube URL or existing file: '{source}'"
    )

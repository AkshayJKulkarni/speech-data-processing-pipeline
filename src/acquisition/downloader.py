"""
Audio/Video Acquisition — download and ingestion logic.

Responsibility: Deposit one raw media file into the raw data directory.
Output contract: Returns the absolute path to the acquired file.
                 Every downstream stage consumes this path.

Design decision: raw_data_dir is a parameter, not a module-level global.
This means the function is fully testable with a temp directory and
requires no config file on disk during tests.
"""

import os
import shutil
from pathlib import Path
from typing import Any

import yt_dlp

from src.utils import get_logger, AcquisitionError
from src.acquisition.validator import validate_source

logger = get_logger(__name__)


def acquire(source: str, raw_data_dir: str) -> str:
    """
    Acquires a single audio/video source into raw_data_dir.

    Args:
        source: YouTube URL or local file path.
        raw_data_dir: Destination directory (from config["paths"]["raw_data"]).

    Returns:
        Absolute path to the acquired file.

    Raises:
        AcquisitionError: For invalid input or download failure.
    """
    source = source.strip()
    source_type = validate_source(source)
    logger.info(f"Acquiring [{source_type}]: {source}")

    if source_type == "youtube":
        return _from_youtube(source, raw_data_dir)
    return _from_local(source, raw_data_dir)


def acquire_batch(sources: list[str], raw_data_dir: str) -> list[str]:
    """
    Acquires multiple sources. Logs and skips failures — does not abort.

    Args:
        sources: List of YouTube URLs or local file paths.
        raw_data_dir: Destination directory for all acquired files.

    Returns:
        List of absolute paths for successfully acquired files only.
    """
    acquired: list[str] = []

    for i, source in enumerate(sources, start=1):
        logger.info(f"Batch [{i}/{len(sources)}]: {source}")
        try:
            acquired.append(acquire(source, raw_data_dir))
        except AcquisitionError as e:
            logger.error(f"Skipping '{source}': {e}")

    logger.info(f"Batch done: {len(acquired)}/{len(sources)} succeeded")
    return acquired


# ── Private helpers ────────────────────────────────────────────────────────────

def _from_youtube(url: str, raw_data_dir: str) -> str:
    """
    Downloads best audio stream from YouTube via yt-dlp.

    No re-encoding is performed — preprocessing handles format conversion.
    noplaylist=True prevents accidental full-playlist downloads.
    """
    os.makedirs(raw_data_dir, exist_ok=True)

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(raw_data_dir, "%(title)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
    except yt_dlp.utils.DownloadError as e:
        raise AcquisitionError(f"YouTube download failed for '{url}': {e}") from e

    abs_path = str(Path(filename).resolve())
    logger.info(f"Downloaded → {abs_path}")
    return abs_path


def _from_local(source_path: str, raw_data_dir: str) -> str:
    """
    Copies a local file into raw_data_dir.

    Uses shutil.copy2 to preserve file metadata.
    Skips copy if the file is already inside raw_data_dir.
    """
    os.makedirs(raw_data_dir, exist_ok=True)

    src = str(Path(source_path).resolve())
    dst = str(Path(raw_data_dir).resolve() / Path(source_path).name)

    if src == dst:
        logger.info(f"Already in raw dir, skipping copy: {dst}")
        return dst

    shutil.copy2(src, dst)
    logger.info(f"Copied {src} → {dst}")
    return dst

"""
preprocessing package public API.

Input contract:
    raw_path     str  — path to raw audio or video file
    output_dir   str  — destination directory (data/processed/)
    audio_config dict — config["audio"] with keys:
                        target_sample_rate, target_channels, output_format

Output contract:
    str — absolute path to the processed 16kHz mono PCM_16 .wav file

Callers import from `src.preprocessing`, never from submodules directly.
"""

from src.preprocessing.processor import process

__all__ = ["process"]

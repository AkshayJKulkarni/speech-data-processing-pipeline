"""
acquisition package public API.

Input contract:  source str  — YouTube URL or local file path
                 raw_data_dir str — destination directory
Output contract: str — absolute path to acquired file in raw_data_dir

Callers import from `src.acquisition`, never from internal submodules.
"""

from src.acquisition.downloader import acquire, acquire_batch

__all__ = ["acquire", "acquire_batch"]

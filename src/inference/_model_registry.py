"""
Whisper model registry — lazy loading and caching.

Design decision: Model loading is separated from inference logic entirely.
The registry is the ONLY place that calls `whisper.load_model()`.

Why lazy loading over module-level loading?
───────────────────────────────────────────
Loading Whisper at import time has three production problems:

1. Import cost — `import src.inference.transcriber` in a test file
   triggers a 200 MB–1.5 GB model download/load even if transcription
   is never called. This makes the entire test suite slow.

2. Startup cost — the pipeline process pays the model load penalty on
   startup regardless of whether any audio actually needs transcription
   (e.g. if the file is already transcribed and cached downstream).

3. Inflexibility — module-level loading hard-codes one model size at
   process start. The registry pattern allows different batch items to
   use different model sizes in the same process, and makes hot-swapping
   models (e.g. during A/B testing) possible without restart.

Why a registry dict over a plain module-level variable?
────────────────────────────────────────────────────────
A single `_model = None` variable breaks the moment you call transcribe()
with two different (model_size, device) combos in the same process — a
real scenario in batch pipelines. The registry is keyed by the config
tuple, so each unique (model_size, device) pair has its own cached instance.
"""

from __future__ import annotations
from typing import Any

import whisper

from src.utils import get_logger, TranscriptionError

logger = get_logger(__name__)

# Valid Whisper model sizes — used for config validation
VALID_MODEL_SIZES: frozenset[str] = frozenset({
    "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
})


class WhisperModelRegistry:
    """
    Thread-safe lazy cache for Whisper model instances.

    Stores one loaded model per (model_size, device) key.
    The first call to get() for a given key loads and caches the model.
    Subsequent calls return the cached instance at O(1) cost.

    Usage:
        registry = WhisperModelRegistry()
        model = registry.get("base", "cpu")    # loads on first call
        model = registry.get("base", "cpu")    # returns cache on second call
    """

    def __init__(self) -> None:
        # Key: (model_size, device) → Value: loaded whisper model
        self._cache: dict[tuple[str, str], Any] = {}

    def get(self, model_size: str, device: str) -> Any:
        """
        Returns a loaded Whisper model for the given size and device.
        Loads and caches on first access, returns cached on subsequent calls.

        Args:
            model_size: One of VALID_MODEL_SIZES.
            device: "cpu" or "cuda".

        Returns:
            A loaded whisper model object.

        Raises:
            TranscriptionError: If model_size is invalid or whisper fails to load.
        """
        if model_size not in VALID_MODEL_SIZES:
            raise TranscriptionError(
                f"Invalid model size '{model_size}'. "
                f"Valid options: {sorted(VALID_MODEL_SIZES)}"
            )

        cache_key = (model_size, device)

        if cache_key not in self._cache:
            logger.info(f"Loading Whisper model: size='{model_size}', device='{device}'")
            try:
                self._cache[cache_key] = whisper.load_model(model_size, device=device)
                logger.info(f"Whisper model loaded: '{model_size}' on {device}")
            except Exception as e:
                raise TranscriptionError(
                    f"Failed to load Whisper model '{model_size}' on '{device}': {e}"
                ) from e
        else:
            logger.debug(f"Whisper model cache hit: '{model_size}' on '{device}'")

        return self._cache[cache_key]

    def clear(self) -> None:
        """
        Evicts all cached models and frees memory.
        Useful in tests and when switching configs at runtime.
        """
        self._cache.clear()
        logger.debug("Whisper model registry cleared")

    @property
    def loaded_models(self) -> list[tuple[str, str]]:
        """Returns list of (model_size, device) pairs currently in cache."""
        return list(self._cache.keys())


# Module-level singleton — one registry per process.
# All calls to transcribe() share this registry, so the model is only
# loaded once per unique (model_size, device) pair per process lifetime.
_registry = WhisperModelRegistry()

"""
Emotion model registry — lazy loading and caching.

Design decision: Third dedicated registry, consistent with _model_registry.py
(Whisper) and _pyannote_registry.py (pyannote). Each model family gets its
own registry for the same reasons:

1. Different loading API — transformers.pipeline("audio-classification")
   vs whisper.load_model() vs pyannote.audio.Pipeline.from_pretrained().
   A unified registry would branch on model family — single responsibility
   violation.

2. Different error type — this registry raises EmotionError exclusively.
   Mixing error types in a shared registry requires passing the exception
   class as a parameter, adding complexity for no benefit.

3. Isolated testability — patching just this module in tests is simpler
   than patching a shared registry and disambiguating which call is which.

Why transformers.pipeline("audio-classification")?
───────────────────────────────────────────────────
The HuggingFace pipeline abstraction handles:
  - Feature extraction (wav2vec2 processor / tokenizer)
  - Model forward pass
  - Softmax + label decoding
  - Device placement

Writing this manually with AutoModelForAudioClassification would require
~40 lines of boilerplate that is not the responsibility of this module.
The pipeline() call is a single line that is straightforward to mock in tests.
"""

from __future__ import annotations
from typing import Any

from src.utils import get_logger, EmotionError

logger = get_logger(__name__)


class EmotionModelRegistry:
    """
    Lazy cache for HuggingFace audio-classification pipeline instances.

    Cache key: (model_name, device)

    Usage:
        registry = EmotionModelRegistry()
        pipe = registry.get("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", "cpu")
        # Second call returns cached pipeline — no reload
        pipe = registry.get("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", "cpu")
    """

    def __init__(self) -> None:
        # (model_name, device) → loaded transformers Pipeline
        self._cache: dict[tuple[str, str], Any] = {}

    def get(self, model_name: str, device: str) -> Any:
        """
        Returns a loaded HuggingFace audio-classification pipeline.
        Loads and caches on first access; returns cached on subsequent calls.

        The pipeline is configured with:
          - return_all_scores=True so we receive the full probability
            distribution, not just the top label. This lets us compute
            a calibrated confidence score and expose it in EmotionSegment.
          - device=-1 for CPU (transformers convention), device=0 for
            the first CUDA GPU.

        Args:
            model_name: HuggingFace Hub model ID.
            device:     "cpu" | "cuda"

        Returns:
            A loaded transformers audio-classification Pipeline instance.

        Raises:
            EmotionError: If transformers is not installed or model loading fails.
        """
        cache_key = (model_name, device)

        if cache_key in self._cache:
            logger.debug(f"Emotion model cache hit: '{model_name}' on '{device}'")
            return self._cache[cache_key]

        logger.info(f"Loading emotion model: '{model_name}' on device='{device}'")

        try:
            # Deferred import — same pattern as the other registries.
            # Importing transformers at module top-level would cause test
            # imports to trigger model registry setup even when emotion
            # classification is not tested.
            from transformers import pipeline as hf_pipeline

            # transformers uses int device: -1 = CPU, 0 = first GPU
            device_id = 0 if device == "cuda" else -1

            pipe = hf_pipeline(
                task="audio-classification",
                model=model_name,
                device=device_id,
                return_all_scores=True,   # full softmax distribution per segment
            )

            self._cache[cache_key] = pipe
            logger.info(f"Emotion model loaded: '{model_name}' on '{device}'")

        except ImportError as e:
            raise EmotionError(
                "transformers is not installed. Run: pip install transformers"
            ) from e
        except Exception as e:
            raise EmotionError(
                f"Failed to load emotion model '{model_name}' on '{device}': {e}"
            ) from e

        return self._cache[cache_key]

    def clear(self) -> None:
        """Evicts all cached models and frees memory."""
        self._cache.clear()
        logger.debug("Emotion model registry cleared")

    @property
    def loaded_models(self) -> list[tuple[str, str]]:
        """Returns (model_name, device) pairs currently in cache."""
        return list(self._cache.keys())


# Module-level singleton — shared across all classify_emotion() calls in the process.
_emotion_registry = EmotionModelRegistry()

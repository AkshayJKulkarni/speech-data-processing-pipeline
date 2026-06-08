"""
Pyannote model registry — lazy loading and caching.

Design decision: Separate registry from the Whisper one despite sharing
the same structural pattern. Reasons:

1. Different loading API — pyannote uses `pyannote.audio.Pipeline.from_pretrained()`
   while Whisper uses `whisper.load_model()`. Merging them would require
   the registry to branch on model family, violating single responsibility.

2. Different auth mechanism — pyannote models on Hugging Face Hub are
   *gated* (require accepting a licence agreement + an HF token).
   Token handling and the specific auth error messages belong here, not
   in the inference function or the orchestrator.

3. Different error types — this registry raises DiarizationError;
   the Whisper registry raises TranscriptionError. A unified registry
   would need to accept the error class as a parameter, adding
   complexity with no real benefit.

Lazy loading rationale (same as Whisper registry):
────────────────────────────────────────────────────
- pyannote loads ~500 MB of model weights. Paying this cost at import
  time breaks tests and slows every pipeline invocation that doesn't
  need diarization.
- The registry loads exactly once per (model_name, device) pair per
  process lifetime, then returns the cached instance in O(1).
"""

from __future__ import annotations
from typing import Any

from src.utils import get_logger, DiarizationError

logger = get_logger(__name__)


class PyannoteModelRegistry:
    """
    Lazy cache for pyannote.audio Pipeline instances.

    Cache key: (model_name, device)
    - model_name: HF Hub model ID, e.g. "pyannote/speaker-diarization-3.1"
    - device: "cpu" | "cuda"

    Usage:
        registry = PyannoteModelRegistry()
        pipeline = registry.get("pyannote/speaker-diarization-3.1", "cpu", hf_token)
        # Second call with same key returns cached pipeline — no reload
        pipeline = registry.get("pyannote/speaker-diarization-3.1", "cpu", hf_token)
    """

    def __init__(self) -> None:
        # (model_name, device) → loaded pyannote Pipeline
        self._cache: dict[tuple[str, str], Any] = {}

    def get(self, model_name: str, device: str, hf_token: str) -> Any:
        """
        Returns a loaded pyannote Pipeline for the given model and device.
        Loads and caches on first access; returns cached on subsequent calls.

        Args:
            model_name: Hugging Face Hub model ID.
                        Must be a model the HF account has accepted the
                        licence for at hf.co/model_name.
            device:     "cpu" | "cuda"
            hf_token:   Hugging Face User Access Token.
                        Required because pyannote models are gated —
                        they cannot be downloaded without authentication.

        Returns:
            A loaded pyannote.audio.Pipeline instance.

        Raises:
            DiarizationError: If the token is missing, the model is not
                              accessible, or loading fails for any reason.
        """
        if not hf_token or not hf_token.strip():
            raise DiarizationError(
                "Hugging Face token is required for pyannote models. "
                "Set the HF_TOKEN environment variable. "
                "Get a token at https://huggingface.co/settings/tokens and "
                "accept the licence at https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        cache_key = (model_name, device)

        if cache_key in self._cache:
            logger.debug(f"Pyannote model cache hit: '{model_name}' on '{device}'")
            return self._cache[cache_key]

        logger.info(f"Loading pyannote model: '{model_name}' on device='{device}'")

        try:
            # Import is deferred to here — not at module top-level — so that
            # importing diarizer.py in a test environment that doesn't have
            # pyannote installed does not immediately raise ImportError.
            from pyannote.audio import Pipeline
            import torch

            pipeline = Pipeline.from_pretrained(
                model_name,
                use_auth_token=hf_token,
            )

            # Move to the requested device.
            # pyannote Pipelines expose .to(device) for GPU placement.
            target = torch.device(device)
            pipeline = pipeline.to(target)

            self._cache[cache_key] = pipeline
            logger.info(f"Pyannote model loaded: '{model_name}' on '{device}'")

        except ImportError as e:
            raise DiarizationError(
                "pyannote.audio is not installed. "
                "Run: pip install pyannote.audio"
            ) from e
        except Exception as e:
            # pyannote raises a generic Exception for auth failures —
            # we catch broadly here and wrap with a helpful message.
            raise DiarizationError(
                f"Failed to load pyannote model '{model_name}' on '{device}': {e}. "
                f"Ensure your HF token is valid and you have accepted the model licence."
            ) from e

        return self._cache[cache_key]

    def clear(self) -> None:
        """
        Evicts all cached models and frees memory.
        Call this after processing is complete on memory-constrained machines.
        """
        self._cache.clear()
        logger.debug("Pyannote model registry cleared")

    @property
    def loaded_models(self) -> list[tuple[str, str]]:
        """Returns (model_name, device) pairs currently in cache."""
        return list(self._cache.keys())


# Module-level singleton — shared across all diarize() calls in the process.
_pyannote_registry = PyannoteModelRegistry()

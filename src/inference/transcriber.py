"""
Speech-to-text transcription module — Whisper implementation.

Responsibility:
    Accept a preprocessed 16kHz mono WAV file and return a structured
    TranscriptResult containing the full text, detected language, and
    a list of timestamped segments.

Input contract:
    audio_path  : str | Path — absolute path to a 16kHz mono WAV file
    config      : dict       — models_config["transcription"] section
                  Required keys:
                    model_size : str  — "tiny"|"base"|"small"|"medium"|"large"
                    language   : str  — BCP-47 code e.g. "en", or "" for auto-detect
                    device     : str  — "cpu" | "cuda" | "auto"

Output contract:
    TranscriptResult dataclass:
        language : str          — BCP-47 language code detected or forced
        text     : str          — full concatenated transcript
        segments : list[TranscriptSegment]
            start : float       — segment start time in seconds
            end   : float       — segment end time in seconds
            text  : str         — text spoken in this segment

Pipeline position:
    preprocessing.process() → [processed_path: str]
        → transcribe(processed_path, config) → [TranscriptResult]
            → diarize() / classify_emotion() / export()
"""

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.utils import get_logger, TranscriptionError
from src.inference._model_registry import _registry, VALID_MODEL_SIZES

logger = get_logger(__name__)

# Whisper performs best on files under 30 minutes.
# Files longer than this limit are a strong signal something went wrong upstream.
_MAX_DURATION_SECONDS: float = 30 * 60.0


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class TranscriptSegment:
    """
    A single timed segment from Whisper's output.

    Whisper segments are determined by internal VAD and beam search —
    they roughly correspond to sentences or natural speech pauses.
    """
    start: float    # seconds from audio start
    end:   float    # seconds from audio start
    text:  str      # transcript text for this segment

    def duration(self) -> float:
        """Returns segment duration in seconds."""
        return round(self.end - self.start, 3)


@dataclass
class TranscriptResult:
    """
    Structured output of the transcription stage.

    Design decision: A typed dataclass instead of a plain dict because:
    - Downstream code (annotation exporter) accesses .segments, not ["segments"]
    - IDE autocomplete works on fields — catches typos at development time
    - asdict() provides the dict representation when JSON serialization is needed
    - The type is self-documenting as a pipeline contract

    Usage:
        result = transcribe("audio.wav", config)
        print(result.text)
        print(result.language)
        for seg in result.segments:
            print(seg.start, seg.end, seg.text)
        serializable = result.to_dict()
    """
    language:  str
    text:      str
    segments:  list[TranscriptSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Returns a JSON-serializable dict representation.
        Used by the annotation exporter and for PipelineContext serialization.
        """
        return asdict(self)

    @property
    def duration(self) -> float:
        """Total transcribed duration in seconds (end of last segment)."""
        if not self.segments:
            return 0.0
        return self.segments[-1].end

    @property
    def segment_count(self) -> int:
        """Number of segments in the transcript."""
        return len(self.segments)


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_audio_path(audio_path: Path) -> None:
    """
    Validates that the audio file exists and is a WAV file.

    Args:
        audio_path: Path to the audio file.

    Raises:
        TranscriptionError: If file is missing or not a WAV.
    """
    if not audio_path.is_file():
        raise TranscriptionError(f"Audio file not found: '{audio_path}'")

    if audio_path.suffix.lower() != ".wav":
        raise TranscriptionError(
            f"Transcriber expects a .wav file, got '{audio_path.suffix}'. "
            f"Run preprocessing first."
        )


def _validate_config(config: dict) -> None:
    """
    Validates the transcription config dict for required keys and values.

    Args:
        config: The transcription section of models.yaml.

    Raises:
        TranscriptionError: If required keys are missing or values are invalid.
    """
    required = {"model_size", "language", "device"}
    missing = required - config.keys()
    if missing:
        raise TranscriptionError(f"Missing transcription config keys: {missing}")

    if config["model_size"] not in VALID_MODEL_SIZES:
        raise TranscriptionError(
            f"Invalid model_size '{config['model_size']}'. "
            f"Valid: {sorted(VALID_MODEL_SIZES)}"
        )


# ── Device resolution ─────────────────────────────────────────────────────────

def _resolve_device(device_config: str) -> str:
    """
    Resolves the target device string, with "auto" selecting CUDA if available.

    Design decision: "auto" is preferred over hard-coding "cpu" in config
    because it makes the same config work on a developer laptop (CPU) and
    a production GPU server without any config change.

    Args:
        device_config: "cpu" | "cuda" | "auto"

    Returns:
        "cuda" if device_config is "auto" and CUDA is available, else "cpu".
        Returns "cuda" or "cpu" as-is for explicit values.
    """
    if device_config == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Device auto-selected: {device}")
            return device
        except ImportError:
            logger.warning("torch not available for device detection, defaulting to cpu")
            return "cpu"

    return device_config


# ── Whisper output mapping ────────────────────────────────────────────────────

def _map_segments(raw_segments: list[dict]) -> list[TranscriptSegment]:
    """
    Maps Whisper's raw segment dicts to typed TranscriptSegment objects.

    Whisper returns segments with many fields (id, seek, tokens, avg_logprob,
    no_speech_prob, etc.). We extract only start, end, text — the fields
    our downstream pipeline actually needs — and strip surrounding whitespace
    from text since Whisper often emits leading spaces.

    Args:
        raw_segments: List of segment dicts from whisper model.transcribe().

    Returns:
        List of TranscriptSegment dataclass instances.
    """
    return [
        TranscriptSegment(
            start=round(float(seg["start"]), 3),
            end=round(float(seg["end"]),   3),
            text=seg["text"].strip(),
        )
        for seg in raw_segments
        if seg.get("text", "").strip()   # skip empty/silent segments
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe(
    audio_path: str,
    config: dict,
    *,
    language: Optional[str] = None,
) -> TranscriptResult:
    """
    Runs Whisper speech-to-text on a preprocessed WAV file.

    Model loading is lazy — the Whisper model is loaded on the first call
    for a given (model_size, device) pair and cached for subsequent calls.
    See _model_registry.py for the caching design.

    Args:
        audio_path: Absolute path to a 16kHz mono WAV file (output of preprocessing).
        config: The transcription section from models.yaml. Required keys:
                  model_size (str), language (str), device (str).
        language: Optional language override. If provided, bypasses auto-detection
                  and forces Whisper to decode in this language. Useful when you
                  know the language ahead of time and want faster inference.
                  BCP-47 format, e.g. "en", "hi", "de".
                  If None, uses config["language"] (empty string = auto-detect).

    Returns:
        TranscriptResult with language, full text, and timestamped segments.

    Raises:
        TranscriptionError: If the file is missing, config is invalid,
                            the model fails to load, or inference fails.

    Example:
        >>> config = {"model_size": "base", "language": "en", "device": "cpu"}
        >>> result = transcribe("data/processed/interview.wav", config)
        >>> print(result.text)
        >>> print(result.segments[0].start, result.segments[0].text)
        >>> serializable = result.to_dict()
    """
    path = Path(audio_path).resolve()

    # ── Validate before touching the model ────────────────────────────────────
    _validate_audio_path(path)
    _validate_config(config)

    model_size = config["model_size"]
    device     = _resolve_device(config["device"])

    # Language: explicit override > config value > empty string (auto-detect)
    # Empty string tells Whisper to detect language automatically.
    lang = language or config.get("language", "") or None

    # ── Load model (lazy — cached after first call) ────────────────────────────
    model = _registry.get(model_size, device)

    # ── Run inference ──────────────────────────────────────────────────────────
    logger.info(
        f"Transcription started | file='{path.name}' "
        f"model='{model_size}' device='{device}' language='{lang or 'auto'}'"
    )

    t_start = time.perf_counter()

    try:
        # word_timestamps=False — we use segment-level timestamps which are
        # sufficient for diarization alignment. Word-level timestamps double
        # inference time and aren't used until annotation merging.
        raw_result = model.transcribe(
            str(path),
            language=lang,
            word_timestamps=False,
            verbose=False,          # suppress Whisper's own progress output
            fp16=device == "cuda",  # fp16 only on GPU — CPU doesn't support it
        )
    except Exception as e:
        raise TranscriptionError(
            f"Whisper inference failed on '{path.name}': {e}"
        ) from e

    elapsed = time.perf_counter() - t_start

    # ── Map raw output to typed result ─────────────────────────────────────────
    segments = _map_segments(raw_result.get("segments", []))
    result = TranscriptResult(
        language=raw_result.get("language", lang or "unknown"),
        text=raw_result.get("text", "").strip(),
        segments=segments,
    )

    logger.info(
        f"Transcription complete | file='{path.name}' "
        f"language='{result.language}' "
        f"segments={result.segment_count} "
        f"duration={result.duration:.1f}s "
        f"elapsed={elapsed:.2f}s"
    )

    return result

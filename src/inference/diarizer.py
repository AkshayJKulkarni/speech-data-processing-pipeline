"""
Speaker Diarization Module — pyannote.audio implementation.

Responsibility:
    Accept a preprocessed 16kHz mono WAV file and return a structured
    SpeakerDiarizationResult: who spoke, when they started, and when they stopped.

What is speaker diarization?
─────────────────────────────
Diarization answers the question "who spoke when?" It assigns each time
interval in the audio to a speaker label (SPEAKER_00, SPEAKER_01, …).
It does NOT transcribe what was said — that is transcription's job.

The combination of both stages enables the annotation export to produce
segments like: {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.3, "text": "Hello"}.

Why pyannote.audio?
───────────────────
pyannote.audio is the industry-standard open-source diarization library,
used in production at companies like Meta, BBC, and Deutsche Telekom.
It is built on top of PyTorch and uses transformer-based speaker embeddings.
The speaker-diarization-3.1 model achieves state-of-the-art DER
(Diarization Error Rate) on standard benchmarks.

Input contract:
    audio_path  : str | Path — absolute path to a 16kHz mono WAV file
    config      : dict       — models_config["diarization"] section
                  Required keys:
                    model        : str  — HF Hub model ID
                    device       : str  — "cpu" | "cuda" | "auto"
                    min_speakers : int  — lower bound hint (1 if unknown)
                    max_speakers : int  — upper bound hint
                    hf_token_env : str  — name of env var holding HF token

Output contract:
    SpeakerDiarizationResult dataclass:
        segments : list[SpeakerSegment]  — sorted ascending by start time
            speaker : str   — "SPEAKER_00", "SPEAKER_01", …
            start   : float — segment start in seconds
            end     : float — segment end in seconds
        num_speakers : int  — number of unique speakers detected

Pipeline position:
    preprocessing.process() → [processed_path]
        → transcribe()       → [TranscriptResult]
        → diarize()          → [SpeakerDiarizationResult]
            → classify_emotion() / export()
"""

import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from src.utils import get_logger, DiarizationError
from src.inference._pyannote_registry import _pyannote_registry

logger = get_logger(__name__)


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class SpeakerSegment:
    """
    A single speaker turn — one contiguous block of speech from one speaker.

    Design decision: Floats are rounded to 3 decimal places (millisecond
    precision). pyannote internally uses higher precision, but milliseconds
    are sufficient for annotation alignment and reduce JSON file size.
    """
    speaker: str    # "SPEAKER_00", "SPEAKER_01", etc.
    start:   float  # seconds from audio start
    end:     float  # seconds from audio start

    def duration(self) -> float:
        """Returns the duration of this speaker turn in seconds."""
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        """Returns a JSON-serializable dict of this segment."""
        return asdict(self)


@dataclass
class SpeakerDiarizationResult:
    """
    Structured output of the diarization stage.

    Design decision: A typed dataclass for the same reasons as
    TranscriptResult — IDE autocomplete, type safety, self-documenting
    contract, and asdict() for JSON serialization.

    The annotation exporter merges this with TranscriptResult by aligning
    transcript segments to the speaker turn that has maximum overlap.

    Usage:
        result = diarize("audio.wav", config)
        print(result.num_speakers)
        for seg in result.segments:
            print(seg.speaker, seg.start, seg.end)
        serializable = result.to_dict()
    """
    segments:     list[SpeakerSegment] = field(default_factory=list)
    num_speakers: int = 0

    def to_dict(self) -> dict:
        """Returns a JSON-serializable dict representation."""
        return {
            "num_speakers": self.num_speakers,
            "segments": [s.to_dict() for s in self.segments],
        }

    @property
    def speakers(self) -> list[str]:
        """Returns the sorted list of unique speaker labels."""
        return sorted({seg.speaker for seg in self.segments})

    @property
    def total_duration(self) -> float:
        """Sum of all speaker turn durations in seconds."""
        return round(sum(seg.duration() for seg in self.segments), 3)


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_audio_path(path: Path) -> None:
    """
    Validates the audio file exists and is a WAV file.

    Args:
        path: Resolved path to the audio file.

    Raises:
        DiarizationError: If file is missing or not a WAV.
    """
    if not path.is_file():
        raise DiarizationError(f"Audio file not found: '{path}'")

    if path.suffix.lower() != ".wav":
        raise DiarizationError(
            f"Diarizer expects a .wav file, got '{path.suffix}'. "
            f"Run preprocessing first."
        )


def _validate_config(config: dict) -> None:
    """
    Validates the diarization config section for required keys and value types.

    Args:
        config: The diarization section from models.yaml.

    Raises:
        DiarizationError: If required keys are missing or values are invalid.
    """
    required = {"model", "device", "min_speakers", "max_speakers", "hf_token_env"}
    missing = required - config.keys()
    if missing:
        raise DiarizationError(f"Missing diarization config keys: {missing}")

    if config["min_speakers"] < 1:
        raise DiarizationError(
            f"min_speakers must be >= 1, got: {config['min_speakers']}"
        )

    if config["max_speakers"] < config["min_speakers"]:
        raise DiarizationError(
            f"max_speakers ({config['max_speakers']}) must be >= "
            f"min_speakers ({config['min_speakers']})"
        )


# ── Device resolution ─────────────────────────────────────────────────────────

def _resolve_device(device_config: str) -> str:
    """
    Resolves "auto" to "cuda" or "cpu" based on torch availability.

    Design decision: Same "auto" pattern as the transcription module.
    The same config file works on a developer laptop and a GPU server
    without modification — "auto" picks the best available device.

    Args:
        device_config: "cpu" | "cuda" | "auto"

    Returns:
        "cuda" if auto and CUDA is available, otherwise "cpu".
    """
    if device_config != "auto":
        return device_config

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Diarization device auto-selected: {device}")
        return device
    except ImportError:
        logger.warning("torch not available for device detection, defaulting to cpu")
        return "cpu"


# ── pyannote output mapping ───────────────────────────────────────────────────

def _map_annotation(annotation: object) -> list[SpeakerSegment]:
    """
    Converts a pyannote Annotation object into a list of SpeakerSegments.

    pyannote's Annotation is its core output type — an iterable of
    (Segment, track_id, speaker_label) tuples. We discard track_id
    (pyannote's internal identifier for overlapping speech from the same
    speaker) since our downstream pipeline treats each turn independently.

    Segments are returned sorted by start time, which pyannote guarantees
    already, but we re-sort explicitly to make the contract clear.

    Args:
        annotation: A pyannote.core.Annotation instance.

    Returns:
        List of SpeakerSegment instances, sorted by start time.
    """
    segments: list[SpeakerSegment] = []

    # annotation.itertracks(yield_label=True) yields:
    #   (Segment(start, end), track_id, speaker_label)
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            SpeakerSegment(
                speaker=str(speaker),
                start=round(float(turn.start), 3),
                end=round(float(turn.end),   3),
            )
        )

    return sorted(segments, key=lambda s: s.start)


# ── Public API ────────────────────────────────────────────────────────────────

def diarize(audio_path: str, config: dict) -> SpeakerDiarizationResult:
    """
    Runs speaker diarization on a preprocessed WAV file using pyannote.audio.

    Model loading is lazy — the pyannote Pipeline is loaded on the first call
    for a given (model, device) pair and cached for all subsequent calls.
    See _pyannote_registry.py for the caching design.

    The HF token is read from the environment variable named in
    config["hf_token_env"] (default: "HF_TOKEN"). It is never passed
    as a function argument to keep secrets out of call stacks, logs,
    and PipelineContext serialization.

    Args:
        audio_path: Absolute path to a 16kHz mono WAV file (preprocessing output).
        config: The diarization section from models.yaml. Required keys:
                  model (str), device (str), min_speakers (int),
                  max_speakers (int), hf_token_env (str).

    Returns:
        SpeakerDiarizationResult with speaker segments and speaker count.

    Raises:
        DiarizationError: If the file is missing, config is invalid,
                          the HF token is absent, the model fails to load,
                          or pyannote inference fails.

    Example:
        >>> config = {
        ...     "model": "pyannote/speaker-diarization-3.1",
        ...     "device": "cpu",
        ...     "min_speakers": 1,
        ...     "max_speakers": 5,
        ...     "hf_token_env": "HF_TOKEN",
        ... }
        >>> result = diarize("data/processed/interview.wav", config)
        >>> print(result.num_speakers)
        >>> for seg in result.segments:
        ...     print(seg.speaker, seg.start, seg.end)
        >>> serializable = result.to_dict()
    """
    path = Path(audio_path).resolve()

    # ── Validate before touching the model ────────────────────────────────────
    _validate_audio_path(path)
    _validate_config(config)

    model_name   = config["model"]
    device       = _resolve_device(config["device"])
    min_speakers = config["min_speakers"]
    max_speakers = config["max_speakers"]

    # Read HF token from environment — never from function argument.
    # This keeps credentials out of tracebacks, logs, and serialized contexts.
    token_env_var = config.get("hf_token_env", "HF_TOKEN")
    hf_token      = os.environ.get(token_env_var, "")

    # ── Load model (lazy — cached after first call) ────────────────────────────
    pipeline = _pyannote_registry.get(model_name, device, hf_token)

    # ── Run inference ──────────────────────────────────────────────────────────
    logger.info(
        f"Diarization started | file='{path.name}' "
        f"model='{model_name}' device='{device}' "
        f"speakers=[{min_speakers}–{max_speakers}]"
    )

    t_start = time.perf_counter()

    try:
        # pyannote accepts speaker count hints that constrain the clustering.
        # Providing both min and max reduces DER significantly when you have
        # domain knowledge about the expected number of speakers.
        # e.g. interview recordings always have exactly 2 speakers.
        annotation = pipeline(
            str(path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except Exception as e:
        raise DiarizationError(
            f"Pyannote inference failed on '{path.name}': {e}"
        ) from e

    elapsed = time.perf_counter() - t_start

    # ── Map pyannote Annotation → typed result ────────────────────────────────
    segments     = _map_annotation(annotation)
    num_speakers = len({seg.speaker for seg in segments})

    result = SpeakerDiarizationResult(
        segments=segments,
        num_speakers=num_speakers,
    )

    logger.info(
        f"Diarization complete | file='{path.name}' "
        f"speakers={result.num_speakers} "
        f"segments={len(result.segments)} "
        f"total_speech={result.total_duration:.1f}s "
        f"elapsed={elapsed:.2f}s"
    )

    return result

"""
Emotion Classification Module — HuggingFace wav2vec2 implementation.

Responsibility:
    Accept a preprocessed 16kHz mono WAV file and a list of speaker
    segments (from diarization), and return one emotion label + confidence
    score per segment.

What is emotion classification in a speech pipeline?
─────────────────────────────────────────────────────
Emotion recognition (also called Speech Emotion Recognition, SER) infers
the emotional state conveyed by a speaker's voice — independent of the
words spoken. It analyses prosodic features (pitch, energy, speaking rate,
voice quality) learned by the model from labelled corpora (IEMOCAP, RAVDESS,
etc.).

The output is one of a fixed set of emotion categories:
    angry · disgust · fear · happy · neutral · sad · surprised

Why per-segment inference?
──────────────────────────
Emotion is turn-level, not file-level. A speaker can be neutral in one
turn and angry in the next. Running a whole-file inference returns one
label for the entire recording — useless for the annotation exporter.
Per-segment inference aligns with what diarization produces and what
the annotation exporter needs to merge with transcript text.

Input contract:
    audio_path : str | Path  — absolute path to a 16kHz mono WAV file
    segments   : list        — diarization output. Each item must be a
                               SpeakerSegment dataclass OR a dict with
                               keys: "speaker", "start", "end".
                               Accepts both for flexibility during
                               pipeline integration.
    config     : dict        — models_config["emotion"] section
                  Required keys:
                    model               : str   — HF Hub model ID
                    device              : str   — "cpu" | "cuda" | "auto"
                    min_segment_duration: float — skip segments shorter than this
                    batch_size          : int   — segments per inference batch

Output contract:
    EmotionResult dataclass:
        segments : list[EmotionSegment]
            speaker    : str   — from diarization ("SPEAKER_00", ...)
            start      : float — segment start in seconds
            end        : float — segment end in seconds
            emotion    : str   — e.g. "happy", "sad", "neutral", "angry"
            confidence : float — max softmax probability, range [0.0, 1.0]

Pipeline position:
    diarize() → [SpeakerDiarizationResult]
        → classify_emotion(wav, segments, config) → [EmotionResult]
            → export()
"""

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Union

import numpy as np

from src.utils import get_logger, EmotionError
from src.inference._emotion_registry import _emotion_registry
from src.inference._audio_slicer import load_audio_array, slice_segment, is_segment_too_short

logger = get_logger(__name__)

# Emotion label used when a segment is too short to classify reliably
_UNKNOWN_EMOTION = "unknown"

# Label normalisation map — different HF models use different label formats
# (e.g. "LABEL_0", "angry", "Angry"). We normalise to lowercase.
_LABEL_NORMALISE: dict[str, str] = {
    "ang":       "angry",
    "angry":     "angry",
    "dis":       "disgust",
    "disgust":   "disgust",
    "fea":       "fear",
    "fear":      "fear",
    "hap":       "happy",
    "happy":     "happy",
    "neu":       "neutral",
    "neutral":   "neutral",
    "sad":       "sad",
    "sadness":   "sad",
    "sur":       "surprised",
    "surprised": "surprised",
    "ps":        "surprised",
}


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class EmotionSegment:
    """
    A single diarization segment enriched with emotion classification output.

    This is the final per-segment data structure consumed by the annotation
    exporter. It carries everything needed to produce one row in the output
    CSV and one entry in the output JSON.

    Design decision: Confidence is the max softmax probability — not an
    entropy measure or calibrated uncertainty. It is the most interpretable
    number for downstream consumers of the annotation files.
    """
    speaker:    str    # from diarization — "SPEAKER_00", "SPEAKER_01", …
    start:      float  # seconds from audio start
    end:        float  # seconds from audio start
    emotion:    str    # normalised emotion label
    confidence: float  # max softmax probability [0.0, 1.0]

    def duration(self) -> float:
        """Returns segment duration in seconds."""
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        """Returns a JSON-serializable dict of this segment."""
        return {
            "speaker":    self.speaker,
            "start":      self.start,
            "end":        self.end,
            "emotion":    self.emotion,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class EmotionResult:
    """
    Structured output of the emotion classification stage.

    Consistent with TranscriptResult and SpeakerDiarizationResult:
    a typed dataclass with a .to_dict() method for JSON serialization.

    Usage:
        result = classify_emotion("audio.wav", segments, config)
        for seg in result.segments:
            print(seg.speaker, seg.emotion, seg.confidence)
        serializable = result.to_dict()
    """
    segments: list[EmotionSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Returns a JSON-serializable dict."""
        return {"segments": [s.to_dict() for s in self.segments]}

    @property
    def emotion_counts(self) -> dict[str, int]:
        """Returns a frequency count of each emotion label in the result."""
        counts: dict[str, int] = {}
        for seg in self.segments:
            counts[seg.emotion] = counts.get(seg.emotion, 0) + 1
        return counts

    @property
    def dominant_emotion(self) -> str:
        """Returns the most frequent emotion label, or 'unknown' if empty."""
        if not self.segments:
            return _UNKNOWN_EMOTION
        return max(self.emotion_counts, key=self.emotion_counts.get)  # type: ignore[arg-type]


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_audio_path(path: Path) -> None:
    """
    Validates that the audio file exists and is a WAV.

    Args:
        path: Resolved path to the audio file.

    Raises:
        EmotionError: If file is missing or not a WAV.
    """
    if not path.is_file():
        raise EmotionError(f"Audio file not found: '{path}'")
    if path.suffix.lower() != ".wav":
        raise EmotionError(
            f"Emotion classifier expects a .wav file, got '{path.suffix}'. "
            "Run preprocessing first."
        )


def _validate_config(config: dict) -> None:
    """
    Validates the emotion config section for required keys and value types.

    Args:
        config: The emotion section from models.yaml.

    Raises:
        EmotionError: If required keys are missing or values are invalid.
    """
    required = {"model", "device", "min_segment_duration", "batch_size"}
    missing = required - config.keys()
    if missing:
        raise EmotionError(f"Missing emotion config keys: {missing}")

    if config["min_segment_duration"] < 0:
        raise EmotionError(
            f"min_segment_duration must be >= 0, got: {config['min_segment_duration']}"
        )
    if config["batch_size"] < 1:
        raise EmotionError(
            f"batch_size must be >= 1, got: {config['batch_size']}"
        )


def _validate_segments(segments: list) -> None:
    """
    Validates that the segments list is non-empty and each item has the
    required fields (speaker, start, end) — either as dict keys or
    dataclass attributes.

    Args:
        segments: List of SpeakerSegment dataclasses or dicts.

    Raises:
        EmotionError: If segments is empty or an item is missing required fields.
    """
    if not segments:
        raise EmotionError("Segments list is empty — nothing to classify.")

    for i, seg in enumerate(segments):
        for field_name in ("speaker", "start", "end"):
            has_field = (
                hasattr(seg, field_name)
                if not isinstance(seg, dict)
                else field_name in seg
            )
            if not has_field:
                raise EmotionError(
                    f"Segment [{i}] is missing required field '{field_name}'. "
                    f"Expected SpeakerSegment or dict with speaker/start/end."
                )


# ── Device resolution ─────────────────────────────────────────────────────────

def _resolve_device(device_config: str) -> str:
    """
    Resolves "auto" to "cuda" or "cpu" based on torch availability.
    Identical pattern to transcriber and diarizer for consistency.

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
        logger.info(f"Emotion device auto-selected: {device}")
        return device
    except ImportError:
        logger.warning("torch not available for device detection, defaulting to cpu")
        return "cpu"


# ── Segment field access ──────────────────────────────────────────────────────

def _get_field(segment: Union[dict, object], field_name: str):
    """
    Reads a field from a SpeakerSegment dataclass or a plain dict.

    Design decision: Accepting both types makes classify_emotion() callable
    with the typed SpeakerDiarizationResult output (from diarizer.py) AND
    with plain dicts (from tests or external callers), without forcing the
    caller to convert between formats.
    """
    if isinstance(segment, dict):
        return segment[field_name]
    return getattr(segment, field_name)


# ── HuggingFace output mapping ────────────────────────────────────────────────

def _normalise_label(raw_label: str) -> str:
    """
    Normalises raw HuggingFace model labels to consistent lowercase strings.

    Different models use different label formats:
      - "LABEL_0", "LABEL_1"  → need mapping via index
      - "angry", "Angry"      → lowercased directly
      - "ang", "hap", "neu"   → expanded via _LABEL_NORMALISE

    Args:
        raw_label: Raw label string from the HF pipeline output.

    Returns:
        Normalised lowercase emotion string.
    """
    normalised = raw_label.lower().strip()
    return _LABEL_NORMALISE.get(normalised, normalised)


def _extract_top_prediction(scores: list[dict]) -> tuple[str, float]:
    """
    Extracts the top label and its probability from the HF pipeline's
    return_all_scores=True output.

    HF pipeline with return_all_scores=True returns:
        [{"label": "angry", "score": 0.72}, {"label": "neutral", "score": 0.18}, ...]

    We take argmax(score) as the predicted emotion and use its score
    as the confidence value.

    Args:
        scores: List of {"label": str, "score": float} dicts.

    Returns:
        Tuple of (normalised_emotion_label, confidence_float).

    Raises:
        EmotionError: If scores list is empty or malformed.
    """
    if not scores:
        raise EmotionError("Model returned empty scores list for a segment.")

    top = max(scores, key=lambda x: x["score"])
    return _normalise_label(top["label"]), float(top["score"])


# ── Batched inference ─────────────────────────────────────────────────────────

def _run_batch(
    pipe,
    audio_slices: list[np.ndarray],
    sample_rate: int,
) -> list[tuple[str, float]]:
    """
    Runs emotion inference on a batch of audio slices.

    HuggingFace audio-classification pipeline accepts:
      - A list of numpy arrays (preferred — avoids temp file I/O)
      - Each array must be 1D float32 at the sample rate the model expects

    Batching is important for throughput: processing N segments in one
    forward pass is significantly faster than N separate forward passes,
    especially on GPU where kernel launch overhead dominates for small inputs.

    Args:
        pipe:         Loaded HF audio-classification pipeline.
        audio_slices: List of 1D float32 numpy arrays, one per segment.
        sample_rate:  Sample rate of the audio (must match model expectation).

    Returns:
        List of (emotion_label, confidence) tuples, one per input slice.

    Raises:
        EmotionError: If the pipeline inference call fails.
    """
    try:
        # HF pipeline expects list of {"array": np.ndarray, "sampling_rate": int}
        inputs = [
            {"array": audio_slice, "sampling_rate": sample_rate}
            for audio_slice in audio_slices
        ]
        raw_outputs = pipe(inputs)
    except Exception as e:
        raise EmotionError(f"Emotion model inference failed: {e}") from e

    return [_extract_top_prediction(scores) for scores in raw_outputs]


# ── Public API ────────────────────────────────────────────────────────────────

def classify_emotion(
    audio_path: str,
    segments: list,
    config: dict,
) -> EmotionResult:
    """
    Classifies emotion for each speaker segment in a preprocessed WAV file.

    Processing flow per segment:
        1. Slice audio array to [start, end] time window.
        2. Check minimum duration — skip segment if too short (assign "unknown").
        3. Batch segments up to config["batch_size"].
        4. Run HF pipeline on the batch.
        5. Map raw labels to normalised emotion strings.
        6. Build EmotionSegment with speaker, timestamps, emotion, confidence.

    Model loading is lazy — the HF pipeline is loaded on the first call for
    a given (model, device) pair and cached for all subsequent calls.
    See _emotion_registry.py for caching details.

    Args:
        audio_path: Absolute path to a 16kHz mono WAV file (preprocessing output).
        segments:   Speaker segments from diarize(). Accepts list of
                    SpeakerSegment dataclasses OR list of dicts with keys
                    "speaker", "start", "end".
        config:     The emotion section from models.yaml. Required keys:
                    model (str), device (str),
                    min_segment_duration (float), batch_size (int).

    Returns:
        EmotionResult containing one EmotionSegment per input segment.
        Segments too short to classify reliably have emotion="unknown"
        and confidence=0.0.

    Raises:
        EmotionError: If the file is missing, config is invalid,
                      segments are malformed, or model inference fails.

    Example:
        >>> config = {
        ...     "model": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        ...     "device": "cpu",
        ...     "min_segment_duration": 0.5,
        ...     "batch_size": 8,
        ... }
        >>> result = classify_emotion("data/processed/interview.wav", diar_result.segments, config)
        >>> for seg in result.segments:
        ...     print(seg.speaker, seg.emotion, f"{seg.confidence:.2%}")
        >>> print(result.dominant_emotion)
        >>> serializable = result.to_dict()
    """
    path = Path(audio_path).resolve()

    # ── Validate before touching the model ────────────────────────────────────
    _validate_audio_path(path)
    _validate_config(config)
    _validate_segments(segments)

    model_name        = config["model"]
    device            = _resolve_device(config["device"])
    min_dur           = float(config["min_segment_duration"])
    batch_size        = int(config["batch_size"])

    # ── Load audio array once — all segment slices drawn from it ──────────────
    # Loading once avoids N separate file reads for N segments.
    audio_array, sample_rate = load_audio_array(path)

    # ── Load model (lazy — cached after first call) ────────────────────────────
    pipe = _emotion_registry.get(model_name, device)

    logger.info(
        f"Emotion classification started | file='{path.name}' "
        f"model='{model_name}' device='{device}' "
        f"segments={len(segments)} batch_size={batch_size}"
    )

    t_start = time.perf_counter()
    emotion_segments: list[EmotionSegment] = []

    # ── Process segments in batches ───────────────────────────────────────────
    # We collect (segment_meta, audio_slice) pairs, separating skipped
    # (too-short) segments upfront to avoid wasting model forward passes.

    pending_metas:  list[dict]       = []  # segments queued for inference
    pending_slices: list[np.ndarray] = []  # corresponding audio slices
    skipped_metas:  list[dict]       = []  # segments too short to classify

    for seg in segments:
        speaker = _get_field(seg, "speaker")
        start   = float(_get_field(seg, "start"))
        end     = float(_get_field(seg, "end"))

        audio_slice = slice_segment(audio_array, sample_rate, start, end)

        meta = {"speaker": speaker, "start": start, "end": end}

        if is_segment_too_short(audio_slice, sample_rate, min_dur):
            logger.debug(
                f"Skipping short segment: speaker={speaker} "
                f"start={start:.2f}s end={end:.2f}s "
                f"(< {min_dur}s minimum)"
            )
            skipped_metas.append(meta)
        else:
            pending_metas.append(meta)
            pending_slices.append(audio_slice)

    # ── Run inference in batches ───────────────────────────────────────────────
    predictions: list[tuple[str, float]] = []

    for batch_start in range(0, len(pending_slices), batch_size):
        batch_slices = pending_slices[batch_start : batch_start + batch_size]
        batch_preds  = _run_batch(pipe, batch_slices, sample_rate)
        predictions.extend(batch_preds)

        logger.debug(
            f"Batch [{batch_start // batch_size + 1}] "
            f"segments {batch_start}–{batch_start + len(batch_slices) - 1} done"
        )

    # ── Assemble EmotionSegments for inferred segments ─────────────────────────
    for meta, (emotion, confidence) in zip(pending_metas, predictions):
        emotion_segments.append(
            EmotionSegment(
                speaker=meta["speaker"],
                start=meta["start"],
                end=meta["end"],
                emotion=emotion,
                confidence=round(confidence, 4),
            )
        )

    # ── Append skipped segments with "unknown" ────────────────────────────────
    for meta in skipped_metas:
        emotion_segments.append(
            EmotionSegment(
                speaker=meta["speaker"],
                start=meta["start"],
                end=meta["end"],
                emotion=_UNKNOWN_EMOTION,
                confidence=0.0,
            )
        )

    # Sort all segments by start time — skipped ones may be out of order
    emotion_segments.sort(key=lambda s: s.start)

    elapsed = time.perf_counter() - t_start
    result  = EmotionResult(segments=emotion_segments)

    logger.info(
        f"Emotion classification complete | file='{path.name}' "
        f"segments={len(result.segments)} "
        f"dominant='{result.dominant_emotion}' "
        f"skipped={len(skipped_metas)} "
        f"elapsed={elapsed:.2f}s"
    )

    return result

"""
Audio slicing utilities for per-segment emotion inference.

Design decision: Audio slicing is isolated into its own module rather
than inlined into classify_emotion() for two reasons:

1. Correctness criticality — off-by-one errors in sample index math
   silently produce wrong emotion predictions. Every wrong prediction
   corrupts the annotation file. Isolated functions are exhaustively
   testable with synthetic numpy arrays, zero model involvement.

2. Reusability — the slicing logic is not specific to emotion. Future
   modules (e.g. per-segment VAD, speaker verification) will need the
   same operation. One implementation is better than several copies.

All functions operate on 1D float32 numpy arrays (mono, any sample rate).
Sample rate is always passed explicitly — never inferred from array length.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from src.utils import EmotionError

# Type alias for readability
AudioArray = np.ndarray


def load_audio_array(path: Path) -> tuple[AudioArray, int]:
    """
    Loads a WAV file into a float32 numpy array without resampling.

    Design decision: soundfile is used here (not librosa) because:
    - We only need to read a WAV, not decode arbitrary containers.
    - soundfile is ~3x faster than librosa for WAV-only reads.
    - The file is already preprocessed to 16kHz mono PCM_16, so no
      format negotiation or channel downmix is needed.

    Args:
        path: Absolute path to a 16kHz mono WAV file.

    Returns:
        Tuple of (audio_array, sample_rate).
        audio_array: 1D float32 numpy array, shape (num_samples,).
        sample_rate: int, expected to be 16000 for preprocessed files.

    Raises:
        EmotionError: If the file cannot be read.
    """
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        raise EmotionError(f"Failed to load audio from '{path}': {e}") from e

    return audio, sr


def slice_segment(
    audio: AudioArray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> AudioArray:
    """
    Extracts a time-bounded slice from a 1D audio array.

    Converts float seconds → integer sample indices using:
        sample_index = int(seconds * sample_rate)

    The `int()` truncation (floor) is intentional:
    - It avoids exceeding array bounds on the last segment.
    - Sub-sample precision has no perceptual meaning at 16kHz.

    An empty slice (start >= end, or out of bounds) returns a zero-length
    array rather than raising. The caller is responsible for checking
    minimum duration before passing the slice to the model.

    Args:
        audio:       1D float32 numpy array — full audio file.
        sample_rate: Sample rate in Hz (typically 16000).
        start_sec:   Segment start in seconds (inclusive).
        end_sec:     Segment end in seconds (exclusive).

    Returns:
        1D float32 numpy array, shape (n_samples,).
        May be zero-length if start >= end or indices are out of range.
    """
    n_samples    = len(audio)
    start_sample = int(start_sec * sample_rate)
    end_sample   = int(end_sec   * sample_rate)

    # Clamp to valid array bounds
    start_sample = max(0, min(start_sample, n_samples))
    end_sample   = max(0, min(end_sample,   n_samples))

    return audio[start_sample:end_sample]


def is_segment_too_short(
    audio_slice: AudioArray,
    sample_rate: int,
    min_duration_sec: float,
) -> bool:
    """
    Returns True if the audio slice is shorter than min_duration_sec.

    Design decision: Emotion models trained on IEMOCAP and similar datasets
    perform poorly on clips under ~0.3 seconds — there isn't enough
    prosodic context to reliably classify emotion. Rather than passing
    micro-clips to the model and getting garbage predictions, we skip
    them and assign a "unknown" label in the caller.

    Args:
        audio_slice:      1D float32 numpy array — the segment audio.
        sample_rate:      Sample rate in Hz.
        min_duration_sec: Minimum acceptable duration in seconds.

    Returns:
        True if the slice is too short to classify reliably.
    """
    actual_duration = len(audio_slice) / sample_rate
    return actual_duration < min_duration_sec

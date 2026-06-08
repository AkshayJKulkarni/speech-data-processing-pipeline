"""
Audio transformation utilities.

Design decision: Each function does exactly ONE audio transformation.
This makes them independently testable and swappable — if you want to
replace librosa with torchaudio for resampling, you change one function
and the rest of the pipeline is untouched.

All functions accept and return numpy arrays + sample rate pairs,
which is the lingua franca of audio processing in Python.

Why 16kHz mono WAV for speech AI models?
─────────────────────────────────────────
• 16kHz (16,000 samples/sec) is the standard training sample rate for
  virtually all modern speech models: Whisper, wav2vec2, HuBERT, pyannote,
  SpeechBrain. These models were trained on 16kHz data — feeding them
  44.1kHz audio forces an internal downsample that wastes compute and
  can subtly degrade accuracy.

• Mono (1 channel) is required because speaker-level features are
  extracted per channel. Stereo doubles the signal length and most
  models don't accept multi-channel input without explicit averaging.
  Downmixing before inference gives consistent, deterministic input.

• WAV (PCM) is uncompressed. MP3/AAC introduce lossy codec artifacts
  (ringing, aliasing) that corrupt subtle prosodic features the emotion
  and diarization models rely on. WAV removes that variable entirely.

Think of it as normalization: just as you normalize a database schema
before querying, you normalize audio before running inference.
"""

from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

from src.utils import get_logger, PreprocessingError

logger = get_logger(__name__)

# Numpy float32 array type alias for readability
AudioArray = np.ndarray


def load_audio(path: Path) -> tuple[AudioArray, int]:
    """
    Loads an audio file into a float32 numpy array.

    librosa is used here because it handles virtually every audio and
    video container format transparently via FFmpeg, and always returns
    float32 data normalized to [-1.0, 1.0].

    Args:
        path: Path to any supported audio/video file.

    Returns:
        Tuple of (audio_array, sample_rate).
        audio_array shape: (samples,) for mono, (channels, samples) for stereo.

    Raises:
        PreprocessingError: If librosa cannot decode the file.
    """
    logger.debug(f"Loading audio: {path}")
    try:
        # mono=False preserves original channel layout so we can
        # inspect it before explicitly downmixing in to_mono()
        audio, sr = librosa.load(str(path), sr=None, mono=False)
    except Exception as e:
        raise PreprocessingError(f"Failed to load audio from '{path}': {e}") from e

    logger.debug(f"Loaded: sr={sr}Hz, shape={audio.shape}, dtype={audio.dtype}")
    return audio, sr


def to_mono(audio: AudioArray) -> AudioArray:
    """
    Converts a stereo (or multi-channel) array to mono by averaging channels.

    Design decision: We average channels rather than taking left-only.
    Averaging preserves energy from both channels and avoids introducing
    channel bias (e.g., if one speaker was panned to one side).

    Args:
        audio: Float32 array, shape (samples,) or (channels, samples).

    Returns:
        Float32 array, shape (samples,) — always 1D after this call.
    """
    if audio.ndim == 1:
        logger.debug("Audio is already mono, skipping downmix")
        return audio

    logger.debug(f"Downmixing {audio.shape[0]} channels to mono")
    return librosa.to_mono(audio)


def resample(audio: AudioArray, orig_sr: int, target_sr: int) -> AudioArray:
    """
    Resamples audio to the target sample rate using librosa's high-quality
    resampler (scipy backend).

    Design decision: librosa.resample uses a polyphase filter by default,
    which is the highest quality resampling method available in Python.
    We skip resampling entirely if rates match to avoid unnecessary compute.

    Args:
        audio: 1D float32 mono array.
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz (16000 for speech models).

    Returns:
        Resampled float32 array at target_sr.
    """
    if orig_sr == target_sr:
        logger.debug(f"Sample rate already {target_sr}Hz, skipping resample")
        return audio

    logger.debug(f"Resampling {orig_sr}Hz → {target_sr}Hz")
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def normalize_peak(audio: AudioArray) -> AudioArray:
    """
    Applies peak normalization so the loudest sample reaches 0 dBFS.

    Why normalize? Speech models are sensitive to amplitude scale.
    Very quiet recordings can fall below the model's noise floor.
    Very loud recordings can clip. Peak normalization standardizes the
    amplitude range without changing the dynamics of the speech signal.

    Design decision: Peak normalization (not RMS/loudness normalization)
    because it's the safest option — it never clips and has no
    perceptual artifacts. RMS normalization is an alternative for
    datasets with wildly varying loudness (e.g., telephone vs. studio).

    Args:
        audio: 1D float32 array in range [-1.0, 1.0].

    Returns:
        Peak-normalized float32 array.
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        logger.warning("Audio appears to be silent (peak < 1e-6), skipping normalization")
        return audio

    logger.debug(f"Peak normalizing: current peak = {peak:.4f}")
    return audio / peak


def save_wav(audio: AudioArray, sample_rate: int, output_path: Path) -> None:
    """
    Writes a float32 mono array to a WAV file using soundfile.

    soundfile is used for writing (not librosa) because it gives explicit
    control over subtype (PCM_16 vs PCM_32) and is faster for write-only
    operations. PCM_16 is used because all target models expect 16-bit WAV.

    Args:
        audio: 1D float32 array.
        sample_rate: Sample rate of the audio.
        output_path: Destination .wav path.

    Raises:
        PreprocessingError: If the file cannot be written.
    """
    logger.debug(f"Writing WAV: {output_path} (sr={sample_rate}, samples={len(audio)})")
    try:
        sf.write(str(output_path), audio, sample_rate, subtype="PCM_16")
    except Exception as e:
        raise PreprocessingError(f"Failed to write WAV to '{output_path}': {e}") from e

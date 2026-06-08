"""
Audio Preprocessing — public interface.

Responsibility: Accept a raw audio/video file, apply all standardization
steps, and deposit a clean WAV file into the processed directory.

Output contract:
    Path — absolute path to the processed .wav file in data/processed/

Pipeline position:
    acquisition.acquire()  →  [raw_path]  →  preprocessing.process()  →  [processed_path]
    preprocessing.process()  →  [processed_path]  →  inference.transcribe() / diarize() / classify_emotion()

The function is idempotent: calling it twice on the same file returns
the existing processed path without reprocessing. This is essential for:
  - Pipeline resumability (restart from any stage)
  - Batch reprocessing safety (reruns don't corrupt previous outputs)
  - Development speed (no waiting for re-conversion during debugging)
"""

from pathlib import Path

from src.utils import get_logger, PreprocessingError
from src.preprocessing.validator import (
    validate_input_file,
    validate_output_dir,
    validate_audio_config,
)
from src.preprocessing.audio_utils import (
    load_audio,
    to_mono,
    resample,
    normalize_peak,
    save_wav,
)

logger = get_logger(__name__)


def process(raw_path: str, output_dir: str, audio_config: dict) -> str:
    """
    Preprocesses a raw audio/video file into a standardized 16kHz mono WAV.

    Processing steps (in order):
        1. Validate input file and config.
        2. Check idempotency — skip if output already exists.
        3. Load audio (handles any format/container via librosa + FFmpeg).
        4. Downmix to mono.
        5. Resample to target sample rate (default: 16000 Hz).
        6. Peak normalize.
        7. Write to WAV (PCM_16).

    Args:
        raw_path: Absolute or relative path to the raw input file.
                  Can be audio (.wav, .mp3, .flac, etc.) or
                  video (.mp4, .mkv, .webm, etc.) — audio is extracted
                  automatically by librosa/FFmpeg.
        output_dir: Directory to write the processed .wav file into.
                    Maps to config["paths"]["processed_data"].
        audio_config: The 'audio' section from config.yaml. Expected keys:
                      target_sample_rate (int), target_channels (int),
                      output_format (str).

    Returns:
        Absolute path string of the processed .wav file.

    Raises:
        PreprocessingError: For missing files, unsupported formats,
                            bad config, or I/O failures.

    Example:
        >>> processed = process("data/raw/interview.mp4", "data/processed/", audio_cfg)
        >>> # Returns: "/abs/path/data/processed/interview.wav"
    """
    input_path  = Path(raw_path).resolve()
    output_path = Path(output_dir).resolve()

    # ── Step 1: Validate everything before touching audio ─────────────────────
    validate_input_file(input_path)
    validate_output_dir(output_path)
    validate_audio_config(audio_config)

    target_sr       = audio_config["target_sample_rate"]
    output_wav_path = output_path / (input_path.stem + ".wav")

    # ── Step 2: Idempotency check ──────────────────────────────────────────────
    # Design decision: stem-based matching (filename without extension).
    # "interview.mp4" and "interview.wav" share the stem "interview".
    # If the output WAV already exists, we assume it was produced by a
    # previous run with the same config and skip reprocessing.
    if output_wav_path.exists():
        logger.info(f"Already processed, skipping: {output_wav_path}")
        return str(output_wav_path)

    logger.info(f"Preprocessing: {input_path.name} → {output_wav_path.name}")

    # ── Steps 3–7: Load → mono → resample → normalize → save ─────────────────
    audio, sr = load_audio(input_path)
    audio     = to_mono(audio)
    audio     = resample(audio, orig_sr=sr, target_sr=target_sr)
    audio     = normalize_peak(audio)
    save_wav(audio, target_sr, output_wav_path)

    logger.info(
        f"Preprocessing complete: {output_wav_path} "
        f"(sr={target_sr}Hz, samples={len(audio)}, "
        f"duration={len(audio)/target_sr:.2f}s)"
    )

    return str(output_wav_path)

"""
inference package public API.

Houses all ML model execution: transcription, diarization, emotion.

Callers import from `src.inference`, never from internal submodules.

Public API:
    transcribe(audio_path, config)               → TranscriptResult
    diarize(audio_path, config)                  → SpeakerDiarizationResult
    classify_emotion(audio_path, segments, config) → EmotionResult

Output types:
    TranscriptResult         — transcription output
    TranscriptSegment        — individual timed segment (start, end, text)
    SpeakerDiarizationResult — diarization output
    SpeakerSegment           — individual speaker turn (speaker, start, end)
    EmotionResult            — emotion classification output
    EmotionSegment           — per-turn emotion (speaker, start, end, emotion, confidence)
"""

from src.inference.transcriber import transcribe, TranscriptResult, TranscriptSegment
from src.inference.diarizer import diarize, SpeakerDiarizationResult, SpeakerSegment
from src.inference.emotion_classifier import classify_emotion, EmotionResult, EmotionSegment

__all__ = [
    # transcription
    "transcribe",
    "TranscriptResult",
    "TranscriptSegment",
    # diarization
    "diarize",
    "SpeakerDiarizationResult",
    "SpeakerSegment",
    # emotion
    "classify_emotion",
    "EmotionResult",
    "EmotionSegment",
]

"""
Segment merger — aligns and merges transcript, diarization, and emotion.

This is the algorithmic core of the annotation engine. Three separate
model outputs arrive on different time grids and must be combined into
one coherent list of AnnotatedSegments.

The merge strategy
──────────────────
Diarization segments are the primary grid. Each diarization segment
defines one speaker turn — the authoritative time window. We then:

  1. Assign transcript text to each speaker turn using maximum overlap.
     For each TranscriptSegment, find the SpeakerSegment whose intersection
     with it is the longest. That speaker "owns" that transcript chunk.
     Multiple transcript segments that overlap the same speaker turn are
     concatenated in time order.

  2. Attach emotion labels from EmotionResult by matching speaker + time
     window. EmotionSegments were produced from the same diarization segments,
     so the match is a direct (speaker, start, end) lookup — not another
     overlap calculation.

Why diarization as the primary grid?
─────────────────────────────────────
Whisper segments are linguistic units (sentence-like). Diarization segments
are speaker turns. A single speaker turn may span multiple Whisper sentences,
and a single Whisper sentence may span a speaker change (rare but real).
Using diarization as primary means every output row is speaker-coherent —
a prerequisite for training speaker-conditioned models.

Graceful degradation
─────────────────────
All three inputs are Optional. The merger handles every combination:
  - No diarization: use transcript segments as primary grid, speaker="unknown"
  - No transcript: segments have empty transcript text
  - No emotion: segments have emotion="unknown", confidence=0.0
  - None of the above: returns empty list (caller decides if that's an error)

Design decision: The merger never raises AnnotationError for missing inputs —
it degrades gracefully. The writer raises if the result is completely empty
and strict_mode is enabled.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.inference.transcriber import TranscriptResult, TranscriptSegment
    from src.inference.diarizer import SpeakerDiarizationResult, SpeakerSegment
    from src.inference.emotion_classifier import EmotionResult

from src.annotation._schema import AnnotatedSegment


# ── Time overlap helper ───────────────────────────────────────────────────────

def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """
    Returns the duration of overlap between two time intervals in seconds.

    Formula: max(0, min(a_end, b_end) - max(a_start, b_start))

    Returns 0.0 if the intervals do not overlap at all.

    Args:
        a_start, a_end: First interval.
        b_start, b_end: Second interval.

    Returns:
        Overlap duration in seconds, >= 0.0.
    """
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


# ── Transcript → speaker assignment ──────────────────────────────────────────

def align_transcript_to_speakers(
    transcript_segments: list,
    speaker_segments: list,
) -> dict[tuple[float, float], str]:
    """
    Assigns each transcript segment to the speaker turn it overlaps most.

    Algorithm: O(T × S) where T = transcript segments, S = speaker segments.
    For production-length recordings (< 60 min), T and S are both < 2000,
    so the quadratic cost is negligible (~4M iterations at most).
    A more efficient interval-tree approach is only warranted beyond that.

    Args:
        transcript_segments: List of TranscriptSegment dataclasses.
                             Each has .start, .end, .text attributes.
        speaker_segments:    List of SpeakerSegment dataclasses.
                             Each has .speaker, .start, .end attributes.

    Returns:
        Dict mapping (t_start, t_end) → speaker_label for each transcript
        segment. Segments with no overlapping speaker get "SPEAKER_UNKNOWN".
    """
    assignment: dict[tuple[float, float], str] = {}

    for t_seg in transcript_segments:
        best_speaker  = "SPEAKER_UNKNOWN"
        best_overlap  = 0.0

        for s_seg in speaker_segments:
            ov = _overlap(t_seg.start, t_seg.end, s_seg.start, s_seg.end)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = s_seg.speaker

        assignment[(t_seg.start, t_seg.end)] = best_speaker

    return assignment


def _build_speaker_text_map(
    transcript_segments: list,
    speaker_segments: list,
) -> dict[tuple[float, float], str]:
    """
    Builds a mapping from (speaker_start, speaker_end) → concatenated text.

    For each speaker segment, collects all transcript segments that overlap
    it more than they overlap any other speaker segment, and concatenates
    their text in time order.

    Args:
        transcript_segments: List of TranscriptSegment dataclasses.
        speaker_segments:    List of SpeakerSegment dataclasses.

    Returns:
        Dict mapping (s_start, s_end) → text string (may be empty string).
    """
    # Initialise all speaker turns with empty text
    text_map: dict[tuple[float, float], str] = {
        (s.start, s.end): "" for s in speaker_segments
    }

    for t_seg in transcript_segments:
        # Find the speaker segment with the most overlap for this transcript chunk
        best_key     = None
        best_overlap = 0.0

        for s_seg in speaker_segments:
            ov = _overlap(t_seg.start, t_seg.end, s_seg.start, s_seg.end)
            if ov > best_overlap:
                best_overlap = ov
                best_key     = (s_seg.start, s_seg.end)

        if best_key is not None and best_overlap > 0.0:
            existing = text_map[best_key]
            text_map[best_key] = (existing + " " + t_seg.text).strip()

    return text_map


def _build_emotion_map(
    emotion_segments: list,
) -> dict[tuple[float, float], tuple[str, float]]:
    """
    Builds a lookup from (start, end) → (emotion, confidence).

    EmotionSegments were produced directly from diarization segments,
    so the key is the exact (start, end) floats from the diarization output.

    Args:
        emotion_segments: List of EmotionSegment dataclasses.
                         Each has .start, .end, .emotion, .confidence.

    Returns:
        Dict mapping (start, end) → (emotion_label, confidence).
    """
    return {
        (seg.start, seg.end): (seg.emotion, seg.confidence)
        for seg in emotion_segments
    }


# ── Main merge function ───────────────────────────────────────────────────────

def merge_segments(
    transcript:      Optional["TranscriptResult"],
    diarization:     Optional["SpeakerDiarizationResult"],
    emotion_result:  Optional["EmotionResult"],
) -> list[AnnotatedSegment]:
    """
    Merges transcript, diarization, and emotion into AnnotatedSegment list.

    Handles every combination of present/absent inputs gracefully:

    Case 1 — All three present (full pipeline):
        Primary grid = diarization segments.
        Each diarization segment gets: assigned transcript text (max-overlap),
        emotion label and confidence from the matching emotion segment.

    Case 2 — No diarization (transcription only):
        Primary grid = transcript segments.
        speaker = "SPEAKER_UNKNOWN" for all segments.
        emotion = "unknown", confidence = 0.0.

    Case 3 — No transcript:
        Primary grid = diarization segments.
        transcript = "" for all segments.

    Case 4 — No emotion:
        Emotion = "unknown", confidence = 0.0 for all segments.

    Case 5 — Nothing:
        Returns empty list.

    Args:
        transcript:     TranscriptResult or None.
        diarization:    SpeakerDiarizationResult or None.
        emotion_result: EmotionResult or None.

    Returns:
        List of AnnotatedSegment, sorted ascending by start_time.
    """
    t_segs  = transcript.segments    if transcript    else []
    s_segs  = diarization.segments   if diarization   else []
    e_segs  = emotion_result.segments if emotion_result else []

    # Nothing to merge at all
    if not t_segs and not s_segs:
        return []

    merged: list[AnnotatedSegment] = []

    if s_segs:
        # ── Primary grid: diarization ──────────────────────────────────────────
        text_map    = _build_speaker_text_map(t_segs, s_segs) if t_segs else {}
        emotion_map = _build_emotion_map(e_segs) if e_segs else {}

        for s_seg in s_segs:
            key = (s_seg.start, s_seg.end)
            text = text_map.get(key, "")
            emotion, confidence = emotion_map.get(key, ("unknown", 0.0))

            merged.append(AnnotatedSegment(
                speaker    = s_seg.speaker,
                start_time = s_seg.start,
                end_time   = s_seg.end,
                transcript = text,
                emotion    = emotion,
                confidence = confidence,
            ))

    else:
        # ── Fallback grid: transcript only ─────────────────────────────────────
        for t_seg in t_segs:
            merged.append(AnnotatedSegment(
                speaker    = "SPEAKER_UNKNOWN",
                start_time = t_seg.start,
                end_time   = t_seg.end,
                transcript = t_seg.text,
                emotion    = "unknown",
                confidence = 0.0,
            ))

    return sorted(merged, key=lambda s: s.start_time)

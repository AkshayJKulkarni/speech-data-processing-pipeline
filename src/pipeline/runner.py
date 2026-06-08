"""
Pipeline orchestrator — production implementation.

Responsibility:
    Execute all six pipeline stages in order, thread PipelineContext
    through each one, measure per-stage elapsed time, write
    pipeline_summary.json, and return the fully populated context.

Stage fatality policy
----------------------
Fatal (re-raise immediately, abort pipeline):
    acquisition   — no file = nothing downstream can run
    preprocessing — no clean WAV = no model can run

Non-fatal (log error, record on context, continue):
    transcription, diarization, emotion, annotation
    A partial result is better than no result. The annotation engine
    degrades gracefully when any of these are absent.

Dependency graph (strict DAG — no cross-package imports except here)
---------------------------------------------------------------------
    utils  <-  acquisition
    utils  <-  preprocessing
    utils  <-  inference
    utils  <-  annotation
    all of the above  <-  pipeline  <-  main.py
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.inference.diarizer import SpeakerDiarizationResult
    from src.inference.emotion_classifier import EmotionResult
    from src.annotation._schema import AnnotationResult

from src.utils import (
    get_logger,
    get_required,
    PipelineError,
)
from src.pipeline.context import PipelineContext
from src.acquisition import acquire
from src.preprocessing import process
from src.inference import transcribe, diarize, classify_emotion
from src.annotation import export_annotations

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stage execution helper
# ---------------------------------------------------------------------------

def _run_stage(
    ctx:  PipelineContext,
    name: str,
    fn:   Callable,
    *args: Any,
    fatal: bool = False,
) -> PipelineContext:
    """
    Executes one pipeline stage with timing, logging, and error handling.

    Measures wall-clock elapsed time for the stage and stores it in
    ctx.stage_timings[name]. On success the return value of fn(*args)
    is stored on the appropriate context field via _set_stage_output().

    Args:
        ctx:   Current PipelineContext.
        name:  Stage name used for logging and timing keys.
        fn:    Callable implementing the stage.
        *args: Arguments forwarded to fn.
        fatal: If True, re-raises PipelineError instead of recording it.
               Use for stages where failure makes all downstream stages
               impossible (acquisition, preprocessing).

    Returns:
        PipelineContext — always returned, even on non-fatal failure.

    Raises:
        PipelineError: If fatal=True and the stage raises.
    """
    logger.info(f"[{name}] starting")
    t0 = time.perf_counter()

    try:
        result = fn(*args)
        elapsed = time.perf_counter() - t0
        ctx.stage_timings[name] = round(elapsed, 3)
        _set_stage_output(ctx, name, result)
        logger.info(f"[{name}] OK | elapsed={elapsed:.2f}s")

    except PipelineError as exc:
        elapsed = time.perf_counter() - t0
        ctx.stage_timings[name] = round(elapsed, 3)
        logger.error(f"[{name}] FAILED | elapsed={elapsed:.2f}s | {exc}")
        ctx.errors.append(f"{name}: {exc}")
        if fatal:
            raise

    return ctx


def _set_stage_output(ctx: PipelineContext, stage: str, result: Any) -> None:
    """Maps stage name -> correct PipelineContext field."""
    mapping = {
        "acquisition":   "raw_path",
        "preprocessing": "processed_path",
        "transcription": "transcript",
        "diarization":   "speaker_segments",
        "emotion":       "annotated_segments",
        "annotation":    "output_paths",
    }
    if stage in mapping:
        setattr(ctx, mapping[stage], result)


# ---------------------------------------------------------------------------
# pipeline_summary.json
# ---------------------------------------------------------------------------

def _write_pipeline_summary(
    ctx:        PipelineContext,
    output_dir: str,
    total_elapsed: float,
    started_at: str,
) -> str:
    """
    Writes pipeline_summary.json to output_dir and returns its path.

    The summary is the top-level artefact consumed by CI dashboards,
    experiment trackers, and operators checking on pipeline runs.
    It contains everything needed to understand what happened without
    opening any other file.

    Args:
        ctx:           Fully (or partially) populated PipelineContext.
        output_dir:    Directory to write the file into.
        total_elapsed: Wall-clock seconds for the full pipeline run.
        started_at:    ISO-8601 UTC timestamp of pipeline start.

    Returns:
        Absolute path to the written pipeline_summary.json.
    """
    annotation = ctx.output_paths          # AnnotationResult | None
    transcript = ctx.transcript            # TranscriptResult | None
    diarization = ctx.speaker_segments     # SpeakerDiarizationResult | None
    emotion = ctx.annotated_segments       # EmotionResult | None

    summary: dict[str, Any] = {
        "source":                ctx.source,
        "execution_timestamp":   started_at,
        "total_runtime_seconds": round(total_elapsed, 3),
        "stage_timings":         ctx.stage_timings,
        "errors":                ctx.errors,
        # --- Inference outputs ---
        "language":              transcript.language if transcript else "unknown",
        "num_speakers":          diarization.num_speakers if diarization else 0,
        "num_transcript_segments": (
            transcript.segment_count if transcript else 0
        ),
        "num_emotion_labels": (
            len(emotion.segments) if emotion else 0
        ),
        # --- Output file paths ---
        "output_files": {
            "annotations_json":  annotation.json_path     if annotation else "",
            "annotations_csv":   annotation.csv_path      if annotation else "",
            "metadata_json":     annotation.metadata_path if annotation else "",
            "pipeline_summary":  "",   # filled in below after path is known
        },
    }

    out_path = Path(output_dir).resolve() / "pipeline_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Back-fill self-reference
    summary["output_files"]["pipeline_summary"] = str(out_path)

    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Pipeline summary written: {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    source:        str,
    config:        dict[str, Any],
    models_config: dict[str, Any],
) -> PipelineContext:
    """
    Executes the full six-stage pipeline for a single audio/video source.

    Stages
    ------
    1. acquisition   — download or copy source into data/raw/         [FATAL]
    2. preprocessing — resample, mono, normalise -> 16kHz WAV         [FATAL]
    3. transcription — Whisper speech-to-text                    [non-fatal]
    4. diarization   — pyannote speaker turns                    [non-fatal]
    5. emotion       — wav2vec2 per-segment emotion               [non-fatal]
    6. annotation    — merge + write JSON / CSV / metadata        [non-fatal]

    After all stages, writes pipeline_summary.json to output_dir.

    Args:
        source:        YouTube URL or local file path.
        config:        Loaded configs/config.yaml.
        models_config: Loaded configs/models.yaml.

    Returns:
        PipelineContext populated with all available stage outputs.
        ctx.errors contains messages for any non-fatal stage failures.
        ctx.stage_timings contains per-stage elapsed seconds.

    Raises:
        AcquisitionError:   Source cannot be acquired.
        PreprocessingError: Audio cannot be preprocessed.
    """
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_pipeline_start = time.perf_counter()

    ctx = PipelineContext(source=source)
    logger.info(f"Pipeline started | source='{source}'")

    raw_dir       = get_required(config, "paths", "raw_data")
    processed_dir = get_required(config, "paths", "processed_data")
    output_dir    = get_required(config, "paths", "output_data")
    audio_cfg     = get_required(config, "audio")

    # ── Stage 1: Acquisition ────────────────────────────────────────────────
    ctx = _run_stage(
        ctx, "acquisition",
        lambda: acquire(source, raw_dir),
        fatal=True,
    )

    # ── Stage 2: Preprocessing ──────────────────────────────────────────────
    ctx = _run_stage(
        ctx, "preprocessing",
        lambda: process(ctx.raw_path, processed_dir, audio_cfg),
        fatal=True,
    )

    # ── Stage 3: Transcription ──────────────────────────────────────────────
    transcription_cfg = get_required(models_config, "transcription")
    ctx = _run_stage(
        ctx, "transcription",
        lambda: transcribe(ctx.processed_path, transcription_cfg),
    )

    # ── Stage 4: Diarization ────────────────────────────────────────────────
    diarization_cfg = get_required(models_config, "diarization")
    ctx = _run_stage(
        ctx, "diarization",
        lambda: diarize(ctx.processed_path, diarization_cfg),
    )

    # ── Stage 5: Emotion classification ─────────────────────────────────────
    # Only runs when diarization produced segments — emotion needs a time grid.
    emotion_cfg = get_required(models_config, "emotion")
    if ctx.speaker_segments and ctx.speaker_segments.segments:
        ctx = _run_stage(
            ctx, "emotion",
            lambda: classify_emotion(
                ctx.processed_path,
                ctx.speaker_segments.segments,
                emotion_cfg,
            ),
        )
    else:
        logger.warning("[emotion] skipped — no diarization segments available")

    # ── Stage 6: Annotation export ───────────────────────────────────────────
    annotation_config = {
        "paths":    {"output_data": output_dir},
        "pipeline": config.get("pipeline", {"version": "1.0.0", "name": "speech-data-processing"}),
    }
    ctx = _run_stage(
        ctx, "annotation",
        lambda: export_annotations(
            transcript       = ctx.transcript,
            speaker_segments = ctx.speaker_segments,
            emotion_result   = ctx.annotated_segments,
            config           = annotation_config,
            source_path      = ctx.processed_path or "",
        ),
    )

    # ── Pipeline summary ─────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_pipeline_start
    _write_pipeline_summary(ctx, output_dir, total_elapsed, started_at)

    logger.info(
        f"Pipeline finished | "
        f"total={total_elapsed:.2f}s | "
        f"errors={len(ctx.errors)} | "
        f"{ctx.summary()}"
    )
    return ctx


def run_batch(
    sources:       list[str],
    config:        dict[str, Any],
    models_config: dict[str, Any],
) -> list[PipelineContext]:
    """
    Runs the pipeline for multiple sources sequentially.

    Failures on one source are logged and recorded but do not abort the
    batch — subsequent sources are always attempted.

    Args:
        sources:       List of YouTube URLs or local file paths.
        config:        Loaded configs/config.yaml.
        models_config: Loaded configs/models.yaml.

    Returns:
        List of PipelineContext — one per source, including partial results.
    """
    results: list[PipelineContext] = []

    for i, source in enumerate(sources, start=1):
        logger.info(f"Batch [{i}/{len(sources)}]: {source}")
        try:
            ctx = run(source, config, models_config)
        except PipelineError as exc:
            logger.error(f"Batch item {i} fatal failure: {exc}")
            ctx = PipelineContext(source=source, errors=[str(exc)])
        results.append(ctx)

    succeeded = sum(1 for r in results if r.is_complete())
    logger.info(f"Batch complete: {succeeded}/{len(sources)} fully succeeded")
    return results

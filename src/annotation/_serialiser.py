"""
Annotation serialiser — writes annotation data to disk.

Design decision: All file I/O is in one module, isolated from merging logic.
This means:
  - The merger can be unit-tested without any filesystem involvement.
  - File format changes (e.g. adding Parquet output) touch only this file.
  - Tests for serialisation use real temp files but synthetic data — no
    model inference required.

Three output files are written per annotation run:
  annotations.json  — complete merged segments for programmatic consumption
  annotations.csv   — tabular format for spreadsheet / pandas / DVC pipelines
  metadata.json     — recording-level summary for dataset catalogues

File naming strategy: all three files share the same stem as the source
audio file. This makes it trivial to associate outputs with their source
in a directory of many processed files.
"""

import csv
import json
import os
from pathlib import Path

from src.utils import get_logger, AnnotationError
from src.annotation._schema import AnnotatedSegment, AnnotationMetadata, AnnotationResult

logger = get_logger(__name__)

# CSV column order — fixed so downstream tools can rely on column positions
CSV_FIELDNAMES = [
    "speaker",
    "start_time",
    "end_time",
    "duration",
    "transcript",
    "emotion",
    "confidence",
]


def write_json(
    segments: list[AnnotatedSegment],
    metadata: AnnotationMetadata,
    output_path: Path,
) -> None:
    """
    Writes the full annotation to a JSON file.

    Output structure:
    {
        "metadata": { ... },        ← recording-level summary
        "segments": [ { ... }, ... ] ← list of AnnotatedSegment dicts
    }

    Design decision: metadata is embedded in annotations.json as well as
    written to its own metadata.json. Embedding it means the JSON file is
    self-contained — a consumer can read everything they need from one file.
    The standalone metadata.json is for tooling that only needs the summary.

    Args:
        segments:    List of AnnotatedSegment instances.
        metadata:    AnnotationMetadata instance.
        output_path: Absolute path to write the .json file.

    Raises:
        AnnotationError: If the file cannot be written.
    """
    payload = {
        "metadata": metadata.to_dict(),
        "segments": [seg.to_dict() for seg in segments],
    }

    _write_file(output_path, json.dumps(payload, indent=2, ensure_ascii=False))
    logger.debug(f"JSON written: {output_path}")


def write_csv(
    segments: list[AnnotatedSegment],
    output_path: Path,
) -> None:
    """
    Writes the annotation to a CSV file.

    CSV format is optimised for pandas ingestion and DVC dataset versioning.
    Each row is one AnnotatedSegment. Duration is computed and included as
    a convenience column — it is derivable from start/end but saves pandas
    users an extra step.

    Column order is fixed by CSV_FIELDNAMES to ensure schema stability
    across pipeline versions.

    Args:
        segments:    List of AnnotatedSegment instances.
        output_path: Absolute path to write the .csv file.

    Raises:
        AnnotationError: If the file cannot be written.
    """
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for seg in segments:
                writer.writerow({
                    "speaker":    seg.speaker,
                    "start_time": seg.start_time,
                    "end_time":   seg.end_time,
                    "duration":   seg.duration(),
                    "transcript": seg.transcript,
                    "emotion":    seg.emotion,
                    "confidence": round(seg.confidence, 4),
                })
    except OSError as e:
        raise AnnotationError(f"Failed to write CSV to '{output_path}': {e}") from e

    logger.debug(f"CSV written: {output_path}")


def write_metadata(
    metadata: AnnotationMetadata,
    output_path: Path,
) -> None:
    """
    Writes the recording-level metadata to a standalone metadata.json.

    Having metadata as a separate file enables:
    - Fast dataset catalogue queries without parsing annotation arrays
    - DVC / MLflow experiment tracking (log metadata.json as an artifact)
    - Downstream scripts that build dataset manifests by globbing metadata files

    Args:
        metadata:    AnnotationMetadata instance.
        output_path: Absolute path to write metadata.json.

    Raises:
        AnnotationError: If the file cannot be written.
    """
    _write_file(output_path, json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False))
    logger.debug(f"Metadata written: {output_path}")


def _write_file(path: Path, content: str) -> None:
    """
    Writes a string to a file, creating parent directories if needed.

    Args:
        path:    Target file path.
        content: UTF-8 string to write.

    Raises:
        AnnotationError: If the write fails.
    """
    try:
        os.makedirs(path.parent, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as e:
        raise AnnotationError(f"Failed to write file '{path}': {e}") from e


def write_all(
    result:     AnnotationResult,
    stem:       str,
    output_dir: Path,
) -> AnnotationResult:
    """
    Writes all three output files and populates result.json_path,
    result.csv_path, and result.metadata_path.

    Args:
        result:     AnnotationResult with segments and metadata populated.
        stem:       Output filename stem (e.g. "interview" → interview.json).
        output_dir: Directory to write all files into.

    Returns:
        The same AnnotationResult with file path fields filled in.

    Raises:
        AnnotationError: If any file write fails.
    """
    json_path     = output_dir / f"{stem}.json"
    csv_path      = output_dir / f"{stem}.csv"
    metadata_path = output_dir / f"{stem}_metadata.json"

    write_json(result.segments, result.metadata, json_path)
    write_csv(result.segments, csv_path)
    write_metadata(result.metadata, metadata_path)

    result.json_path     = str(json_path)
    result.csv_path      = str(csv_path)
    result.metadata_path = str(metadata_path)

    logger.info(
        f"Annotation files written: "
        f"json={json_path.name} "
        f"csv={csv_path.name} "
        f"metadata={metadata_path.name}"
    )
    return result

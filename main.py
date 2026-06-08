"""
Speech Data Processing & Annotation Pipeline — CLI entry point.

This file has exactly two jobs:
  1. Parse command-line arguments.
  2. Load configuration and invoke the pipeline.

All logic lives in src/. This file must stay thin — if it grows
past ~60 lines of non-boilerplate, something belongs in src/.

Usage
-----
    # Single file or URL
    python main.py --source path/to/audio.mp3
    python main.py --source "https://www.youtube.com/watch?v=..."

    # Override config paths
    python main.py --source audio.mp3 \\
                   --config configs/config.yaml \\
                   --models-config configs/models.yaml

    # Explicit output filename stem (default: audio file stem)
    python main.py --source audio.mp3 --output-stem my_interview
"""

import argparse
import os
import sys

# Ensure project root is on PYTHONPATH when invoked directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import get_logger, load_config
from src.utils.exceptions import PipelineError
from src.pipeline import run

logger = get_logger(__name__)

CONFIG_PATH        = "configs/config.yaml"
MODELS_CONFIG_PATH = "configs/models.yaml"


def _parse_args() -> argparse.Namespace:
    """
    Parses and returns CLI arguments.

    Returns:
        argparse.Namespace with attributes:
            source       : str  — YouTube URL or local file path (required)
            config       : str  — path to config.yaml
            models_config: str  — path to models.yaml
            output_stem  : str | None — optional output filename stem
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Speech Data Processing & Annotation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --source interview.mp3\n"
            "  python main.py --source https://youtu.be/dQw4w9WgXcQ\n"
        ),
    )
    parser.add_argument(
        "--source",
        required=True,
        metavar="PATH_OR_URL",
        help="Local audio/video file path or YouTube URL to process.",
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        metavar="PATH",
        help=f"Pipeline config YAML. Default: {CONFIG_PATH}",
    )
    parser.add_argument(
        "--models-config",
        default=MODELS_CONFIG_PATH,
        dest="models_config",
        metavar="PATH",
        help=f"Model config YAML. Default: {MODELS_CONFIG_PATH}",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        dest="output_stem",
        metavar="STEM",
        help=(
            "Output filename stem for annotation files "
            "(e.g. 'interview' produces interview.json, interview.csv). "
            "Defaults to the input file stem."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Pipeline entry point.

    Loads config, runs the pipeline, prints a final summary to stdout,
    and exits with code 1 if any fatal error occurred.
    """
    args = _parse_args()

    # -- Load configuration --------------------------------------------------
    config        = load_config(args.config)
    models_config = load_config(args.models_config)

    name    = config["pipeline"]["name"]
    version = config["pipeline"]["version"]
    logger.info(f"Starting '{name}' v{version} | source='{args.source}'")

    # -- Run pipeline --------------------------------------------------------
    try:
        ctx = run(args.source, config, models_config)
    except PipelineError as exc:
        logger.error(f"Pipeline aborted: {exc}")
        sys.exit(1)

    # -- Print summary to stdout ---------------------------------------------
    _print_summary(ctx)

    # Exit 1 if any non-fatal stage errors occurred so CI can detect them
    if ctx.errors:
        logger.warning(
            f"Pipeline completed with {len(ctx.errors)} non-fatal error(s). "
            f"Check logs for details."
        )
        sys.exit(1)


def _print_summary(ctx: "PipelineContext") -> None:  # type: ignore[name-defined]
    """Prints a human-readable summary table to stdout after the run."""
    annotation = ctx.output_paths

    lines = [
        "",
        "=" * 60,
        "  PIPELINE COMPLETE",
        "=" * 60,
        f"  Source   : {ctx.source}",
        f"  Language : {ctx.transcript.language if ctx.transcript else 'n/a'}",
        f"  Speakers : {ctx.speaker_segments.num_speakers if ctx.speaker_segments else 0}",
        f"  Segments : {ctx.transcript.segment_count if ctx.transcript else 0} transcript | "
        f"{len(ctx.annotated_segments.segments) if ctx.annotated_segments else 0} emotion",
        "",
        "  Stage timings:",
    ]
    for stage, elapsed in ctx.stage_timings.items():
        lines.append(f"    {stage:<18} {elapsed:.2f}s")

    if annotation:
        lines += [
            "",
            "  Output files:",
            f"    {annotation.json_path}",
            f"    {annotation.csv_path}",
            f"    {annotation.metadata_path}",
        ]

    if ctx.errors:
        lines += ["", "  Errors:"]
        for err in ctx.errors:
            lines.append(f"    - {err}")

    lines += ["=" * 60, ""]
    print("\n".join(lines))


if __name__ == "__main__":
    main()

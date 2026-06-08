"""
pipeline package public API.

Orchestrates all six stages in order using PipelineContext as shared state.

Public API:
    run(source, config, models_config)          -> PipelineContext
    run_batch(sources, config, models_config)   -> list[PipelineContext]
    PipelineContext                              -- shared state dataclass
"""

from src.pipeline.runner import run, run_batch
from src.pipeline.context import PipelineContext

__all__ = ["run", "run_batch", "PipelineContext"]

"""Non-sensitive timing diagnostics for recipe-generation stages."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

_LOGGER = logging.getLogger("recipegen.timing")


@contextmanager
def timed_stage(
    stage: str,
    **metrics: int | float,
) -> Iterator[dict[str, int | float]]:
    """Log elapsed time and safe numeric metrics without exception details."""

    started_at = time.perf_counter()
    outcome = "ok"
    error_category = "none"
    try:
        yield metrics
    except Exception as error:
        outcome = "error"
        error_category = type(error).__name__
        raise
    finally:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        safe_metrics = {
            name: value
            for name, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        _LOGGER.info(
            "timing stage=%s elapsed_ms=%.3f outcome=%s error_category=%s metrics=%s",
            stage,
            elapsed_ms,
            outcome,
            error_category,
            safe_metrics,
        )

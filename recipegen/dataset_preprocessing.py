"""Dataset preprocessing utilities for the Food.com raw recipe dump."""

from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .models import Recipe


def normalize_text(value: Any) -> str:
    """Normalize whitespace in text while preserving meaning."""

    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def parse_list_like(value: Any) -> list[str] | None:
    """Safely parse list-like values from list/string representations."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        if not isinstance(parsed, (list, tuple)):
            return None
        items = list(parsed)
    else:
        return None

    result: list[str] = []
    for item in items:
        normalized = normalize_text(item)
        if normalized:
            result.append(normalized)
    return result or None


def parse_numeric_list(value: Any) -> list[float] | None:
    """Safely parse numeric list-like nutrition values."""

    parsed = parse_list_like(value)
    if parsed is None:
        return None

    numbers: list[float] = []
    for item in parsed:
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            return None
    return numbers or None


def safe_int(value: Any) -> int | None:
    """Return int value if present and valid, otherwise None."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def normalize_recipe_row(raw_row: Mapping[str, Any]) -> tuple[Recipe | None, str | None]:
    """Convert one raw row into the normalized project recipe schema.

    Returns (recipe, reason) where reason is None on success. Invalid rows are
    rejected explicitly with a reason string for the report.
    """

    recipe_id = safe_int(raw_row.get("id"))
    if recipe_id is None:
        return None, "missing_or_invalid_recipe_id"

    recipe_name = normalize_text(raw_row.get("name"))
    if not recipe_name:
        return None, "missing_recipe_name"

    ingredients = parse_list_like(raw_row.get("ingredients"))
    if ingredients is None or not ingredients:
        return None, "invalid_ingredients"

    steps = parse_list_like(raw_row.get("steps"))
    if steps is None or not steps:
        return None, "invalid_steps"

    total_time_min = safe_int(raw_row.get("minutes"))
    if total_time_min is not None and total_time_min < 0:
        total_time_min = None

    tags = parse_list_like(raw_row.get("tags")) or []
    nutrition = parse_numeric_list(raw_row.get("nutrition"))

    recipe = Recipe(
        recipe_id=recipe_id,
        recipe_name=recipe_name,
        ingredients=[normalize_text(item) for item in ingredients],
        steps=[normalize_text(item) for item in steps],
        total_time_min=total_time_min,
        prep_time_min=None,
        cook_time_min=None,
        servings=None,
        tags=[normalize_text(item) for item in tags],
        nutrition=nutrition,
    )
    return recipe, None


def process_raw_recipe_csv(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    chunksize: int = 50000,
) -> dict[str, Any]:
    """Process the raw Food.com CSV into a normalized recipe dataset.

    This keeps the source CSV untouched and writes a compact processed version to
    a stable filename in data/processed/.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    input_count = 0
    output_count = 0
    exclusion_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()

    header_written = False
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        for _, row in chunk.iterrows():
            input_count += 1
            recipe, reason = normalize_recipe_row(row.to_dict())
            if recipe is None:
                exclusion_counts[reason or "unknown_invalid_record"] += 1
                continue

            output_count += 1
            normalized = recipe.to_dict()
            normalized["ingredients"] = json.dumps(normalized["ingredients"])
            normalized["steps"] = json.dumps(normalized["steps"])
            normalized["tags"] = json.dumps(normalized["tags"])
            normalized["nutrition"] = json.dumps(normalized["nutrition"]) if normalized["nutrition"] is not None else None

            df = pd.DataFrame([normalized])
            df.to_csv(output_path, mode="a", index=False, header=not header_written)
            header_written = True

            if recipe.total_time_min is None:
                validation_counts["missing_total_time_min"] += 1
            if recipe.prep_time_min is None:
                validation_counts["missing_prep_time_min"] += 1
            if recipe.cook_time_min is None:
                validation_counts["missing_cook_time_min"] += 1
            if recipe.servings is None:
                validation_counts["missing_servings"] += 1
            if recipe.nutrition is None:
                validation_counts["missing_nutrition"] += 1

    report = {
        "source_file": str(input_path),
        "output_file": str(output_path),
        "input_record_count": input_count,
        "output_record_count": output_count,
        "excluded_record_count": sum(exclusion_counts.values()),
        "excluded_by_reason": dict(sorted(exclusion_counts.items())),
        "validation_statistics": {
            "missing_total_time_min": validation_counts["missing_total_time_min"],
            "missing_prep_time_min": validation_counts["missing_prep_time_min"],
            "missing_cook_time_min": validation_counts["missing_cook_time_min"],
            "missing_servings": validation_counts["missing_servings"],
            "missing_nutrition": validation_counts["missing_nutrition"],
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

"""Utilities for preparing recipe records for retrieval and future embedding work."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .dataset_preprocessing import normalize_text, safe_int


def parse_json_list(value: Any) -> list[str]:
    """Parse a JSON-like or list-like value into a list of strings."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                import ast

                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return []
        if isinstance(parsed, str):
            items = [parsed]
        elif isinstance(parsed, (list, tuple)):
            items = list(parsed)
        else:
            return []
    else:
        return []

    cleaned: list[str] = []
    for item in items:
        normalized = normalize_text(item)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def build_searchable_document(
    *,
    recipe_id: int,
    recipe_name: str,
    ingredients: Iterable[str],
    steps: Iterable[str],
    total_time_min: int | None,
    tags: Iterable[str] | None = None,
) -> str:
    """Create a stable, searchable document from a recipe row."""

    ingredient_list = [normalize_text(item) for item in ingredients if normalize_text(item)]
    step_list = [normalize_text(step) for step in steps if normalize_text(step)]
    tag_list = [normalize_text(tag) for tag in (tags or []) if normalize_text(tag)]

    parts: list[str] = [
        f"recipe_id: {recipe_id}",
        f"recipe_name: {recipe_name}",
        f"ingredients: {'; '.join(ingredient_list)}",
        f"instructions: {' '.join(step_list)}",
    ]
    if total_time_min is not None:
        parts.append(f"total_time_min: {total_time_min}")
    if tag_list:
        parts.append(f"tags: {', '.join(tag_list)}")
    return " | ".join(parts)


def recipe_to_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract stable metadata fields for later retrieval and filtering."""

    recipe_id = safe_int(row.get("recipe_id"))
    if recipe_id is None:
        raise ValueError("recipe_id is required")

    ingredients = parse_json_list(row.get("ingredients"))
    steps = parse_json_list(row.get("steps"))
    tags = parse_json_list(row.get("tags"))
    total_time_min = safe_int(row.get("total_time_min"))
    if total_time_min is not None and total_time_min < 0:
        total_time_min = None

    return {
        "recipe_id": recipe_id,
        "recipe_name": normalize_text(row.get("recipe_name")),
        "total_time_min": total_time_min,
        "ingredient_count": len(ingredients),
        "step_count": len(steps),
        "tag_count": len(tags),
        "tags": tags,
    }


def prepare_recipe_documents(
    input_path: str | Path,
    document_output_path: str | Path,
    metadata_output_path: str | Path,
    report_path: str | Path,
    chunksize: int = 50000,
) -> dict[str, Any]:
    """Stream the processed dataset into retrieval documents and metadata.

    The output is intentionally split into two files:
      - JSONL documents for embedding text and downstream retrieval
      - CSV metadata for filtering and display
    """

    input_path = Path(input_path)
    document_output_path = Path(document_output_path)
    metadata_output_path = Path(metadata_output_path)
    report_path = Path(report_path)

    document_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if document_output_path.exists():
        document_output_path.unlink()
    if metadata_output_path.exists():
        metadata_output_path.unlink()

    excluded_counts: Counter[str] = Counter()
    input_count = 0
    output_count = 0
    validation_counts: Counter[str] = Counter()

    metadata_fields = [
        "recipe_id",
        "recipe_name",
        "total_time_min",
        "ingredient_count",
        "step_count",
        "tag_count",
        "tags",
    ]
    with document_output_path.open("a", encoding="utf-8") as doc_file, metadata_output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=metadata_fields)
        writer.writeheader()

        for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
            for _, row in chunk.iterrows():
                input_count += 1
                row_dict = row.to_dict()

                recipe_id = safe_int(row_dict.get("recipe_id"))
                if recipe_id is None:
                    excluded_counts["missing_or_invalid_recipe_id"] += 1
                    continue

                recipe_name = normalize_text(row_dict.get("recipe_name"))
                if not recipe_name:
                    excluded_counts["missing_recipe_name"] += 1
                    continue

                ingredients = parse_json_list(row_dict.get("ingredients"))
                if not ingredients:
                    excluded_counts["empty_ingredients"] += 1
                    continue

                steps = parse_json_list(row_dict.get("steps"))
                if not steps:
                    excluded_counts["empty_steps"] += 1
                    continue

                total_time_min = safe_int(row_dict.get("total_time_min"))
                if total_time_min is not None and total_time_min < 0:
                    total_time_min = None

                tags = parse_json_list(row_dict.get("tags"))
                metadata = {
                    "recipe_id": recipe_id,
                    "recipe_name": recipe_name,
                    "total_time_min": total_time_min,
                    "ingredient_count": len(ingredients),
                    "step_count": len(steps),
                    "tag_count": len(tags),
                    "tags": tags,
                }

                document_text = build_searchable_document(
                    recipe_id=recipe_id,
                    recipe_name=recipe_name,
                    ingredients=ingredients,
                    steps=steps,
                    total_time_min=total_time_min,
                    tags=tags,
                )
                if not document_text.strip():
                    excluded_counts["empty_document_text"] += 1
                    continue

                output_count += 1
                doc_payload = {
                    "recipe_id": recipe_id,
                    "recipe_name": recipe_name,
                    "document_text": document_text,
                }
                doc_file.write(json.dumps(doc_payload, ensure_ascii=False, sort_keys=True) + "\n")

                writer.writerow(metadata)

                if total_time_min is None:
                    validation_counts["missing_total_time_min"] += 1
                if not ingredients:
                    validation_counts["empty_ingredients"] += 1
                if not steps:
                    validation_counts["empty_steps"] += 1
                if not tags:
                    validation_counts["missing_tags"] += 1

    report = {
        "source_file": str(input_path),
        "document_output_file": str(document_output_path),
        "metadata_output_file": str(metadata_output_path),
        "input_record_count": input_count,
        "output_document_count": output_count,
        "excluded_document_count": sum(excluded_counts.values()),
        "excluded_by_reason": dict(sorted(excluded_counts.items())),
        "validation_statistics": {
            "missing_total_time_min": validation_counts["missing_total_time_min"],
            "empty_ingredients": validation_counts["empty_ingredients"],
            "empty_steps": validation_counts["empty_steps"],
            "missing_tags": validation_counts["missing_tags"],
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

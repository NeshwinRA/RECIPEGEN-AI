"""Recipe retrieval helpers for the RAG generation pipeline."""

from __future__ import annotations

import math
import re
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Mapping

from .models import GeneratedRecipe, RecipeRequest
from .timing import timed_stage

FULL_COLLECTION_NAME = "recipe_documents_full"
DEFAULT_VECTOR_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "vector_db"
_FOODCOM_ATTRIBUTION = re.compile(
    r"\b(?:"
    r"(?:from|by|via|courtesy of|adapted from|based on|according to|found on|taken from)"
    r"\s+(?:www\.)?food\.com\b"
    r"|(?:www\.)?food\.com(?:'s)?\s+(?:recipe|source|original|version)\b"
    r")",
    re.IGNORECASE,
)


class RecipeRetrievalError(RuntimeError):
    """The recipe collection returned a malformed retrieval response."""


def has_foodcom_provenance_claim(recipe: GeneratedRecipe) -> bool:
    """Check all generated text fields for an explicit Food.com attribution."""

    for recipe_field in fields(recipe):
        value = getattr(recipe, recipe_field.name)
        text_values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        if any(
            isinstance(text, str) and _FOODCOM_ATTRIBUTION.search(text)
            for text in text_values
        ):
            return True
    return False


def build_retrieval_query(request: RecipeRequest) -> str:
    """Turn request values into deterministic text for the existing ONNX encoder."""

    return " | ".join(
        (
            f"ingredients: {'; '.join(request.ingredients)}",
            f"cuisine preference: {request.cuisine}",
            f"dietary preference: {request.dietary_preference}",
            f"maximum total time: {request.max_cooking_time} minutes",
            f"servings: {request.servings}",
            f"spice level: {request.spice_level}",
            f"equipment preference: {request.equipment}",
            f"bachelor mode: {request.bachelor_mode}",
        )
    )


def normalize_retrieval_results(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize Chroma's first-query nested result arrays into source records."""

    if not isinstance(results, Mapping):
        raise RecipeRetrievalError("Recipe retrieval returned an invalid response.")
    ids = results.get("ids")
    documents = results.get("documents")
    metadatas = results.get("metadatas")
    distances = results.get("distances")
    if not isinstance(ids, list) or not isinstance(documents, list) or not isinstance(metadatas, list):
        raise RecipeRetrievalError("Recipe retrieval response is missing expected result arrays.")
    if not ids or not documents or not metadatas:
        return []
    if not all(isinstance(group, list) for group in (ids[0], documents[0], metadatas[0])):
        raise RecipeRetrievalError("Recipe retrieval response has an invalid result shape.")

    result_ids = ids[0]
    result_documents = documents[0]
    result_metadatas = metadatas[0]
    result_distances = distances[0] if (
        isinstance(distances, list)
        and distances
        and isinstance(distances[0], list)
    ) else []
    if not (len(result_ids) == len(result_documents) == len(result_metadatas)):
        raise RecipeRetrievalError("Recipe retrieval returned mismatched source fields.")

    records: list[dict[str, Any]] = []
    for index, (record_id, document, metadata) in enumerate(
        zip(result_ids, result_documents, result_metadatas)
    ):
        match = re.fullmatch(r"recipe_([1-9]\d*)", record_id) if isinstance(record_id, str) else None
        if match is None or not isinstance(metadata, Mapping):
            raise RecipeRetrievalError("Recipe retrieval returned an invalid source identifier.")
        recipe_id = int(match.group(1))
        metadata_id = metadata.get("recipe_id")
        if (
            isinstance(metadata_id, bool)
            or not isinstance(metadata_id, int)
            or metadata_id != recipe_id
        ):
            raise RecipeRetrievalError("Retrieved recipe ID does not match its collection record.")
        if not isinstance(document, str) or not document.strip():
            continue

        distance: float | None = None
        if index < len(result_distances):
            raw_distance = result_distances[index]
            if isinstance(raw_distance, (int, float)) and not isinstance(raw_distance, bool):
                if math.isfinite(raw_distance):
                    distance = float(raw_distance)
        records.append(
            {
                "source_recipe_id": recipe_id,
                "document": document.strip(),
                "metadata": dict(metadata),
                "distance": distance,
            }
        )
    return records


def retrieve_recipes(
    request: RecipeRequest,
    *,
    db_path: str | Path = DEFAULT_VECTOR_DB_PATH,
    n_results: int = 5,
    query_fn: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Query only the existing full collection using the configured ONNX helper."""

    if n_results < 1:
        raise ValueError("n_results must be at least 1.")
    if query_fn is None:
        from .embeddings import query_recipe_collection

        query_fn = query_recipe_collection

    raw_results = query_fn(
        build_retrieval_query(request),
        db_path,
        collection_name=FULL_COLLECTION_NAME,
        n_results=n_results,
    )
    with timed_stage("retrieval_result_normalization") as metrics:
        records = normalize_retrieval_results(raw_results)
        metrics["result_count"] = len(records)
    return records

"""Tests for retrieval normalization and deterministic RAG prompts."""

from __future__ import annotations

import pytest

from recipegen.models import RecipeRequest
from recipegen.prompts import build_plain_prompt, build_rag_prompt
from recipegen.rag import (
    FULL_COLLECTION_NAME,
    RecipeRetrievalError,
    normalize_retrieval_results,
    retrieve_recipes,
)


def make_request() -> RecipeRequest:
    return RecipeRequest(
        ingredients=["potato", "onion"],
        cuisine="South Indian",
        dietary_preference="Vegetarian",
        max_cooking_time=30,
        servings=2,
        spice_level="medium",
        equipment="single stove",
    )


def make_query_result(
    *,
    ids: list[str] | None = None,
    documents: list[str] | None = None,
    metadata: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "ids": [ids if ids is not None else ["recipe_101"]],
        "documents": [
            documents
            if documents is not None
            else ["recipe_id: 101 | recipe_name: Potato curry | total_time_min: 25"]
        ],
        "metadatas": [
            metadata if metadata is not None else [{"recipe_id": 101, "recipe_name": "Potato curry"}]
        ],
        "distances": [[0.12]],
    }


def test_prompt_keeps_request_sources_and_output_rules_separate() -> None:
    retrieved = normalize_retrieval_results(make_query_result())

    prompt = build_rag_prompt(make_request(), retrieved)

    assert prompt.index("USER REQUEST AND PREFERENCES") < prompt.index(
        "RETRIEVED SOURCE RECIPE RECORDS"
    )
    assert prompt.index("RETRIEVED SOURCE RECIPE RECORDS") < prompt.index(
        "OUTPUT FORMAT AND VALIDATION CONSTRAINTS"
    )
    assert '"source_recipe_id":101' in prompt
    assert "recipe_id: 101 | recipe_name: Potato curry" in prompt
    assert "Do not include a recipe_id" in prompt
    assert "never claim guaranteed dietary suitability or allergy safety" in prompt
    assert "never derive or relabel it as prep_time_min or cook_time_min" in prompt


def test_empty_retrieval_prompt_explicitly_marks_original_ungrounded_result() -> None:
    prompt = build_rag_prompt(make_request(), [])

    assert prompt == build_plain_prompt(make_request())
    assert "No usable source records were retrieved" in prompt
    assert "source_recipe_ids must be []" in prompt
    assert "do not cite Food.com or imply dataset provenance" in prompt


def test_retrieval_explicitly_selects_full_collection_and_normalizes_records() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_query(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return make_query_result()

    records = retrieve_recipes(make_request(), db_path="test-db", query_fn=fake_query)

    assert calls[0][1]["collection_name"] == FULL_COLLECTION_NAME == "recipe_documents_full"
    assert calls[0][1]["n_results"] == 5
    assert calls[0][0][1] == "test-db"
    assert records == [
        {
            "source_recipe_id": 101,
            "document": "recipe_id: 101 | recipe_name: Potato curry | total_time_min: 25",
            "metadata": {"recipe_id": 101, "recipe_name": "Potato curry"},
            "distance": 0.12,
        }
    ]


def test_blank_retrieval_document_is_not_used_as_context() -> None:
    result = make_query_result(documents=["  "])

    assert normalize_retrieval_results(result) == []


def test_mismatched_collection_and_metadata_ids_are_rejected() -> None:
    result = make_query_result(metadata=[{"recipe_id": 202, "recipe_name": "Different recipe"}])

    with pytest.raises(RecipeRetrievalError, match="does not match"):
        normalize_retrieval_results(result)

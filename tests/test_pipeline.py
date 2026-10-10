"""Offline tests for recipe generation orchestration."""

from __future__ import annotations

import json
from typing import Any

import pytest

from recipegen.llm_client import GeminiRequestError
from recipegen.models import RecipeRequest
from recipegen.pipeline import (
    RecipeResponseValidationError,
    RequestValidationError,
    ResponseParsingError,
    generate_recipe,
)


def make_request(**overrides: Any) -> RecipeRequest:
    values: dict[str, Any] = {
        "ingredients": ["potato", "onion"],
        "cuisine": "South Indian",
        "dietary_preference": "Vegetarian",
        "max_cooking_time": 30,
        "servings": 2,
        "spice_level": "medium",
        "equipment": "single stove",
        "bachelor_mode": False,
    }
    values.update(overrides)
    return RecipeRequest(**values)


def make_response(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "recipe_name": "Potato onion curry",
        "ingredients": ["potato", "onion", "turmeric"],
        "steps": ["Chop the onion.", "Cook the ingredients until tender."],
        "total_time_min": None,
        "prep_time_min": None,
        "cook_time_min": None,
        "servings": 2,
        "tags": [],
        "nutrition": None,
        "source_recipe_ids": [101],
    }
    payload.update(overrides)
    return json.dumps(payload)


def make_retrieval_result(
    *,
    ids: list[str] | None = None,
    documents: list[str] | None = None,
    metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
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


class FakeGeminiClient:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def close(self) -> None:
        raise AssertionError("Injected clients remain caller-owned.")


def fake_retriever(result: dict[str, Any], calls: list[dict[str, Any]] | None = None):
    def retrieve(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if calls is not None:
            calls.append({"args": args, "kwargs": kwargs})
        return result

    return retrieve


def test_grounded_generation_validates_result_and_only_uses_retrieved_source() -> None:
    calls: list[dict[str, Any]] = []
    llm = FakeGeminiClient(make_response())
    result = generate_recipe(
        make_request(),
        llm_client=llm,
        retrieval_fn=fake_retriever(make_retrieval_result(), calls),
        db_path="test-db",
    )

    assert result.grounded is True
    assert result.source_recipe_ids == [101]
    assert not hasattr(result.recipe, "recipe_id")
    assert result.recipe.total_time_min is None
    assert result.recipe.prep_time_min is None
    assert result.recipe.cook_time_min is None
    assert result.validation_report.passed is True
    assert any("Total cooking time is unknown" in warning for warning in result.validation_report.warnings)
    assert calls[0]["kwargs"]["collection_name"] == "recipe_documents_full"
    assert "recipe_id: 101 | recipe_name: Potato curry" in llm.prompts[0]


def test_empty_retrieval_returns_ungrounded_recipe_without_citations() -> None:
    llm = FakeGeminiClient(make_response(source_recipe_ids=[]))
    no_records = make_retrieval_result(ids=[], documents=[], metadata=[])

    result = generate_recipe(
        make_request(),
        llm_client=llm,
        retrieval_fn=fake_retriever(no_records),
    )

    assert result.grounded is False
    assert result.source_recipe_ids == []
    assert "original, ungrounded suggestion" in llm.prompts[0]
    assert "Food.com" not in result.recipe.recipe_name + " ".join(result.recipe.steps)


def test_ungrounded_response_cannot_claim_foodcom_provenance() -> None:
    llm = FakeGeminiClient(
        make_response(
            recipe_name="A recipe from Food.com",
            source_recipe_ids=[],
        )
    )
    no_records = make_retrieval_result(ids=[], documents=[], metadata=[])

    with pytest.raises(ResponseParsingError, match="cannot claim"):
        generate_recipe(
            make_request(),
            llm_client=llm,
            retrieval_fn=fake_retriever(no_records),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recipe_name", "A recipe from Food.com"),
        ("ingredients", ["potato", "adapted from Food.com"]),
        ("steps", ["Prepare the ingredients.", "This Food.com recipe is ready."]),
        ("tags", ["easy", "Food.com recipe"]),
    ],
)
def test_ungrounded_attribution_is_rejected_in_every_text_field(
    field: str,
    value: Any,
) -> None:
    llm = FakeGeminiClient(make_response(**{field: value, "source_recipe_ids": []}))
    no_records = make_retrieval_result(ids=[], documents=[], metadata=[])

    with pytest.raises(ResponseParsingError, match="cannot claim"):
        generate_recipe(
            make_request(),
            llm_client=llm,
            retrieval_fn=fake_retriever(no_records),
        )


def test_unrelated_food_word_in_ungrounded_ingredient_is_allowed() -> None:
    llm = FakeGeminiClient(
        make_response(ingredients=["potato", "food coloring"], source_recipe_ids=[])
    )
    no_records = make_retrieval_result(ids=[], documents=[], metadata=[])

    result = generate_recipe(
        make_request(),
        llm_client=llm,
        retrieval_fn=fake_retriever(no_records),
    )

    assert result.grounded is False


def test_invented_source_ids_are_rejected() -> None:
    llm = FakeGeminiClient(make_response(source_recipe_ids=[999]))

    with pytest.raises(ResponseParsingError, match="not retrieved"):
        generate_recipe(
            make_request(),
            llm_client=llm,
            retrieval_fn=fake_retriever(make_retrieval_result()),
        )


def test_grounded_response_must_name_a_retrieved_source() -> None:
    llm = FakeGeminiClient(make_response(source_recipe_ids=[]))

    with pytest.raises(ResponseParsingError, match="must identify"):
        generate_recipe(
            make_request(),
            llm_client=llm,
            retrieval_fn=fake_retriever(make_retrieval_result()),
        )


def test_invalid_user_input_stops_before_retrieval_or_generation() -> None:
    llm = FakeGeminiClient(make_response())

    def unexpected_retrieval(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("Invalid request must not retrieve.")

    with pytest.raises(RequestValidationError, match="At least one ingredient"):
        generate_recipe(
            make_request(ingredients=[]),
            llm_client=llm,
            retrieval_fn=unexpected_retrieval,
        )

    assert llm.prompts == []


def test_invalid_request_field_type_stops_before_retrieval() -> None:
    llm = FakeGeminiClient(make_response())

    with pytest.raises(RequestValidationError, match="must be an integer"):
        generate_recipe(make_request(servings=True), llm_client=llm)

    assert llm.prompts == []


def test_gemini_api_error_is_propagated_as_safe_application_error() -> None:
    llm = FakeGeminiClient(GeminiRequestError("Gemini rejected the generation request (HTTP 400)."))
    no_records = make_retrieval_result(ids=[], documents=[], metadata=[])

    with pytest.raises(GeminiRequestError, match="HTTP 400"):
        generate_recipe(
            make_request(),
            llm_client=llm,
            retrieval_fn=fake_retriever(no_records),
        )


@pytest.mark.parametrize("response", ["not json", "{}", "[]"])
def test_malformed_response_is_rejected(response: str) -> None:
    llm = FakeGeminiClient(response)
    no_records = make_retrieval_result(ids=[], documents=[], metadata=[])

    with pytest.raises(ResponseParsingError):
        generate_recipe(
            make_request(),
            llm_client=llm,
            retrieval_fn=fake_retriever(no_records),
        )


def test_model_cannot_assign_a_dataset_recipe_id_to_generated_content() -> None:
    llm = FakeGeminiClient(make_response(recipe_id=101, source_recipe_ids=[]))
    no_records = make_retrieval_result(ids=[], documents=[], metadata=[])

    with pytest.raises(ResponseParsingError, match="required recipe schema"):
        generate_recipe(
            make_request(),
            llm_client=llm,
            retrieval_fn=fake_retriever(no_records),
        )


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        ({"total_time_min": 31}, "exceeds the requested maximum total time"),
        ({"ingredients": ["potato", "chicken"]}, "conflict with the Vegetarian preference"),
    ],
)
def test_f09_validation_failures_are_not_returned_as_success(
    overrides: dict[str, Any],
    expected_error: str,
) -> None:
    llm = FakeGeminiClient(make_response(**overrides))

    with pytest.raises(RecipeResponseValidationError, match=expected_error) as raised:
        generate_recipe(
            make_request(),
            llm_client=llm,
            retrieval_fn=fake_retriever(make_retrieval_result()),
        )

    assert raised.value.report.passed is False

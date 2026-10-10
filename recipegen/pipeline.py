"""Request-to-recipe generation orchestration."""

from __future__ import annotations

import json
from typing import Any, Callable, Protocol

from .llm_client import GeminiInteractionsClient
from .models import GeneratedRecipe, GenerationResult, RecipeRequest
from .prompts import build_plain_prompt, build_rag_prompt
from .rag import DEFAULT_VECTOR_DB_PATH, has_foodcom_provenance_claim, retrieve_recipes
from .response_validation import ValidationReport, validate_recipe_response
from .timing import timed_stage
from .validation import validate_inputs

_RECIPE_FIELDS = {
    "recipe_name",
    "ingredients",
    "steps",
    "total_time_min",
    "prep_time_min",
    "cook_time_min",
    "servings",
    "tags",
    "nutrition",
}
_RESPONSE_FIELDS = _RECIPE_FIELDS | {"source_recipe_ids"}


class RecipeGenerationError(RuntimeError):
    """Base error for invalid or unsuccessful recipe generation."""


class RequestValidationError(RecipeGenerationError):
    """The user request is invalid and generation was not attempted."""


class ResponseParsingError(RecipeGenerationError):
    """The generated response does not match the required JSON structure."""


class RecipeResponseValidationError(RecipeGenerationError):
    """The parsed generated recipe did not pass F09 validation."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("Generated recipe failed validation: " + "; ".join(report.errors))


class _TextGenerator(Protocol):
    def generate_text(self, prompt: str) -> str: ...

    def close(self) -> None: ...


def _validate_request(request: RecipeRequest) -> None:
    if not isinstance(request, RecipeRequest):
        raise RequestValidationError("A RecipeRequest is required.")
    errors: list[str] = []
    if not isinstance(request.ingredients, list) or any(
        not isinstance(item, str) for item in request.ingredients
    ):
        errors.append("Ingredients must be a list of strings.")
    for field in ("cuisine", "dietary_preference", "spice_level", "equipment"):
        if not isinstance(getattr(request, field), str):
            errors.append(f"{field.replace('_', ' ').capitalize()} must be text.")
    for field in ("max_cooking_time", "servings"):
        value = getattr(request, field)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{field.replace('_', ' ').capitalize()} must be an integer.")
    if not isinstance(request.bachelor_mode, bool):
        errors.append("Bachelor mode must be true or false.")
    if errors:
        raise RequestValidationError(" ".join(errors))

    text_values = [
        *request.ingredients,
        request.cuisine,
        request.dietary_preference,
        request.spice_level,
        request.equipment,
    ]
    input_errors = validate_inputs(
        request.ingredients,
        request.max_cooking_time,
        request.servings,
        total_input_chars=sum(len(value) for value in text_values),
    )
    if input_errors:
        raise RequestValidationError(" ".join(input_errors))


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _parse_response(text: str) -> tuple[GeneratedRecipe, list[int]]:
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ResponseParsingError("Gemini returned malformed recipe JSON.") from exc
    if not isinstance(payload, dict) or set(payload) != _RESPONSE_FIELDS:
        raise ResponseParsingError("Gemini response does not match the required recipe schema.")

    source_ids = payload.pop("source_recipe_ids")
    if not isinstance(source_ids, list) or any(
        not isinstance(recipe_id, int) or isinstance(recipe_id, bool) or recipe_id <= 0
        for recipe_id in source_ids
    ):
        raise ResponseParsingError("source_recipe_ids must be a list of positive integers.")
    if len(source_ids) != len(set(source_ids)):
        raise ResponseParsingError("source_recipe_ids must not contain duplicates.")
    if not isinstance(payload["recipe_name"], str):
        raise ResponseParsingError("recipe_name must be a string.")
    if not isinstance(payload["ingredients"], list) or not isinstance(payload["steps"], list):
        raise ResponseParsingError("ingredients and steps must be lists.")

    try:
        recipe = GeneratedRecipe(**payload)
    except TypeError as exc:
        raise ResponseParsingError("Gemini response does not match the recipe schema.") from exc
    return recipe, source_ids


def generate_recipe(
    request: RecipeRequest,
    use_rag: bool = True,
    *,
    llm_client: _TextGenerator | None = None,
    retrieval_fn: Callable[..., dict[str, Any]] | None = None,
    db_path: str = str(DEFAULT_VECTOR_DB_PATH),
    n_results: int = 5,
) -> GenerationResult:
    """Validate a request, optionally retrieve context, and return validated output."""

    with timed_stage("request_validation"):
        _validate_request(request)

    retrieved = (
        retrieve_recipes(
            request,
            db_path=db_path,
            n_results=n_results,
            query_fn=retrieval_fn,
        )
        if use_rag
        else []
    )
    prompt = build_rag_prompt(request, retrieved) if retrieved else build_plain_prompt(request)

    owns_client = llm_client is None
    client = llm_client if llm_client is not None else GeminiInteractionsClient()
    try:
        response_text = client.generate_text(prompt)
    finally:
        if owns_client:
            client.close()

    with timed_stage("json_response_parsing"):
        recipe, source_ids = _parse_response(response_text)

    with timed_stage(
        "citation_validation",
        retrieved_count=len(retrieved),
        cited_count=len(source_ids),
    ):
        available_ids = {record["source_recipe_id"] for record in retrieved}
        unknown_ids = sorted(set(source_ids) - available_ids)
        if unknown_ids:
            raise ResponseParsingError(
                "Gemini referenced recipe IDs that were not retrieved."
            )
        if retrieved and not source_ids:
            raise ResponseParsingError(
                "A response using retrieved context must identify at least one retrieved source."
            )
        if not retrieved and source_ids:
            raise ResponseParsingError("An ungrounded response cannot cite dataset sources.")
        if not retrieved and has_foodcom_provenance_claim(recipe):
            raise ResponseParsingError(
                "An ungrounded response cannot claim or imply Food.com provenance."
            )

    with timed_stage("response_validation"):
        report = validate_recipe_response(recipe, request)
    if not report.passed:
        raise RecipeResponseValidationError(report)

    return GenerationResult(
        recipe=recipe,
        grounded=bool(source_ids),
        source_recipe_ids=source_ids,
        validation_report=report,
    )

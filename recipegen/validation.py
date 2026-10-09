"""Validation utilities for user input."""

from __future__ import annotations

from typing import Iterable, Sequence

from .config import (
    MAX_COOKING_TIME,
    MAX_INGREDIENTS,
    MAX_INGREDIENT_LENGTH,
    MAX_INPUT_CHARS,
    MAX_SERVINGS,
    MIN_COOKING_TIME,
    MIN_SERVINGS,
)


def validate_inputs(ingredients: Sequence[str], cooking_time: int | str, servings: int | str, *, total_input_chars: int | None = None) -> list[str]:
    """Validate a recipe request before sending it to the model.

    Returns an empty list when the request is valid; otherwise returns a list of
    human-readable validation errors.
    """

    errors: list[str] = []

    cleaned_ingredients = [item.strip() for item in (ingredients or [])]
    if not cleaned_ingredients:
        errors.append("At least one ingredient is required.")
    elif len(cleaned_ingredients) > MAX_INGREDIENTS:
        errors.append(f"Please provide at most {MAX_INGREDIENTS} ingredients.")

    for ingredient in cleaned_ingredients:
        if not ingredient:
            errors.append("Ingredient names cannot be blank.")
            continue
        if len(ingredient) > MAX_INGREDIENT_LENGTH:
            errors.append(f"Ingredient '{ingredient}' is longer than {MAX_INGREDIENT_LENGTH} characters.")

    seen: set[str] = set()
    for ingredient in cleaned_ingredients:
        normalized = ingredient.lower()
        if normalized in seen:
            errors.append(f"Duplicate ingredient: '{ingredient}'.")
        seen.add(normalized)

    try:
        cooking_minutes = int(cooking_time)
    except (TypeError, ValueError):
        errors.append("Cooking time must be a number in minutes.")
        cooking_minutes = None

    if cooking_minutes is not None:
        if cooking_minutes < MIN_COOKING_TIME or cooking_minutes > MAX_COOKING_TIME:
            errors.append(
                f"Cooking time must be between {MIN_COOKING_TIME} and {MAX_COOKING_TIME} minutes."
            )

    try:
        servings_value = int(servings)
    except (TypeError, ValueError):
        errors.append("Servings must be a number.")
        servings_value = None

    if servings_value is not None and not (MIN_SERVINGS <= servings_value <= MAX_SERVINGS):
        errors.append(f"Servings must be between {MIN_SERVINGS} and {MAX_SERVINGS}.")

    if total_input_chars is not None and total_input_chars > MAX_INPUT_CHARS:
        errors.append(f"Input must be under {MAX_INPUT_CHARS} characters.")

    if total_input_chars is None:
        raw_input = " ".join(cleaned_ingredients)
        if len(raw_input) > MAX_INPUT_CHARS:
            errors.append(f"Input must be under {MAX_INPUT_CHARS} characters.")

    return errors

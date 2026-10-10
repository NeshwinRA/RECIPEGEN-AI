"""Validation for generated recipe responses."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import MAX_SERVINGS, MIN_SERVINGS
from .models import GeneratedRecipe, Recipe, RecipeRequest

DIETARY_KEYWORDS_PATH = Path(__file__).resolve().parent.parent / "data" / "dietary_keywords.json"
DIETARY_SAFETY_WARNING = (
    "Ingredient keyword checks are a first safeguard only; recipe suitability and "
    "allergy safety are not guaranteed."
)


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of checking a recipe response against its schema and request."""

    passed: bool
    errors: list[str]
    warnings: list[str]


def _read_recipe_value(recipe: Recipe | Mapping[str, Any], field: str) -> Any:
    """Read a field from either the project dataclass or a response mapping."""

    if isinstance(recipe, Mapping):
        if field == "steps" and field not in recipe:
            return recipe.get("instructions")
        return recipe.get(field)
    return getattr(recipe, field, None)


def _is_non_negative_integer(value: Any) -> bool:
    """Return whether a value is an integer time, excluding bool and numeric strings."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _load_dietary_keywords() -> dict[str, Any]:
    """Load the small, explicit ingredient keyword lists used by the checks."""

    with DIETARY_KEYWORDS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError("Dietary keyword data must be a JSON object")
    return data


def _contains_keyword(ingredient: str, keyword: str) -> bool:
    """Match a whole word or phrase, avoiding matches such as egg in eggplant."""

    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    return re.search(pattern, ingredient, flags=re.IGNORECASE) is not None


def _dietary_conflicts(
    ingredients: list[str],
    preference: str,
    keyword_data: Mapping[str, Any],
) -> list[str]:
    """Return ingredient terms that conflict with the recognized diet policy."""

    policy = keyword_data.get("policies", {}).get(preference)
    if policy is None:
        return []

    excluded_terms: list[str] = policy.get("excluded_terms", [])
    conflicts: list[str] = []
    plant_milks: list[str] = keyword_data.get("plant_milk_qualifiers", [])

    for ingredient in ingredients:
        normalized = ingredient.strip()
        for term in excluded_terms:
            searchable_ingredient = normalized
            if term == "milk":
                plant_milk_pattern = (
                    r"(?<!\w)(?:"
                    + "|".join(re.escape(item) for item in plant_milks)
                    + r")\s+milk(?!\w)"
                )
                searchable_ingredient = re.sub(
                    plant_milk_pattern,
                    "",
                    normalized,
                    flags=re.IGNORECASE,
                )
            if _contains_keyword(searchable_ingredient, term):
                conflicts.append(normalized)
                break

    return conflicts


def validate_recipe_response(
    recipe: Recipe | GeneratedRecipe | Mapping[str, Any],
    request: RecipeRequest,
) -> ValidationReport:
    """Validate response fields and user constraints without treating warnings as proof.

    GeneratedRecipe intentionally has no dataset recipe ID. Mapping responses
    and dataset Recipe records retain the existing required-ID validation.
    """

    errors: list[str] = []
    warnings = [DIETARY_SAFETY_WARNING]

    if not isinstance(recipe, GeneratedRecipe):
        recipe_id = _read_recipe_value(recipe, "recipe_id")
        if not isinstance(recipe_id, int) or isinstance(recipe_id, bool) or recipe_id <= 0:
            errors.append("Recipe ID must be a positive integer.")

    recipe_name = _read_recipe_value(recipe, "recipe_name")
    if not isinstance(recipe_name, str) or not recipe_name.strip():
        errors.append("Recipe name must be a non-empty string.")

    ingredients_value = _read_recipe_value(recipe, "ingredients")
    if not isinstance(ingredients_value, (list, tuple)) or not ingredients_value:
        errors.append("Recipe must include at least one ingredient.")
        ingredients: list[str] = []
    else:
        ingredients = [
            ingredient.strip()
            for ingredient in ingredients_value
            if isinstance(ingredient, str) and ingredient.strip()
        ]
        if len(ingredients) != len(ingredients_value):
            errors.append("Every recipe ingredient must be a non-empty string.")

    steps_value = _read_recipe_value(recipe, "steps")
    if not isinstance(steps_value, (list, tuple)) or not steps_value:
        errors.append("Recipe must include at least one usable instruction step.")
    elif not any(isinstance(step, str) and step.strip() for step in steps_value):
        errors.append("Recipe must include at least one usable instruction step.")
    elif any(not isinstance(step, str) or not step.strip() for step in steps_value):
        errors.append("Every instruction step must be a non-empty string.")

    for field in ("total_time_min", "prep_time_min", "cook_time_min"):
        value = _read_recipe_value(recipe, field)
        if value is not None and not _is_non_negative_integer(value):
            errors.append(f"{field} must be a non-negative integer or null.")

    total_time = _read_recipe_value(recipe, "total_time_min")
    if _is_non_negative_integer(total_time):
        max_time = request.max_cooking_time
        if _is_non_negative_integer(max_time) and total_time > max_time:
            time_warning = (
                f"Recipe total time ({total_time} min) exceeds the requested "
                f"maximum total time ({max_time} min)."
            )
            errors.append(time_warning)
            warnings.append(time_warning)
    elif total_time is None:
        warnings.append(
            "Total cooking time is unknown; it cannot be confirmed against the requested limit."
        )

    servings = _read_recipe_value(recipe, "servings")
    if servings is not None:
        if (
            not isinstance(servings, int)
            or isinstance(servings, bool)
            or not MIN_SERVINGS <= servings <= MAX_SERVINGS
        ):
            errors.append(
                f"Servings must be an integer between {MIN_SERVINGS} and {MAX_SERVINGS}, or null."
            )
        elif servings != request.servings:
            errors.append(
                f"Recipe servings ({servings}) do not match the requested "
                f"servings ({request.servings})."
            )
    else:
        warnings.append("Serving size is unknown.")

    tags = _read_recipe_value(recipe, "tags")
    if tags is not None and (
        not isinstance(tags, (list, tuple))
        or any(not isinstance(tag, str) for tag in tags)
    ):
        errors.append("Recipe tags must be a list of strings or null.")

    nutrition = _read_recipe_value(recipe, "nutrition")
    if nutrition is not None and (
        not isinstance(nutrition, (list, tuple))
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in nutrition
        )
    ):
        errors.append("Nutrition must be a list of finite numbers or null.")

    preference = request.dietary_preference.strip().casefold()
    keyword_data = _load_dietary_keywords()
    policies = keyword_data.get("policies", {})
    if preference not in policies:
        warnings.append(
            "Dietary preference is not recognized by the keyword check; "
            "no dietary suitability is asserted."
        )
    elif ingredients:
        conflicts = _dietary_conflicts(ingredients, preference, keyword_data)
        if conflicts:
            errors.append(
                f"Ingredients conflict with the {request.dietary_preference} preference: "
                + ", ".join(conflicts)
                + "."
            )

    return ValidationReport(passed=not errors, errors=errors, warnings=warnings)

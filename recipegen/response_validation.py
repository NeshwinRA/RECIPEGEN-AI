"""Validation for generated recipe responses."""

from __future__ import annotations

import json
import math
import re
import unicodedata
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
_INGREDIENT_ALIASES = {
    "black pepper": "black pepper",
    "pepper": "black pepper",
    "ground black pepper": "black pepper",
    "freshly ground black pepper": "black pepper",
    "salt": "salt",
    "table salt": "salt",
    "sea salt": "salt",
    "kosher salt": "salt",
    "water": "water",
    "sugar": "sugar",
    "granulated sugar": "sugar",
    "white sugar": "sugar",
    "caster sugar": "sugar",
    "cooking oil": "cooking oil",
    "oil": "cooking oil",
    "vegetable oil": "cooking oil",
    "canola oil": "cooking oil",
    "sunflower oil": "cooking oil",
    "neutral oil": "cooking oil",
    "olive oil": "cooking oil",
    "turmeric": "turmeric powder",
    "ground turmeric": "turmeric powder",
    "turmeric powder": "turmeric powder",
    "chilli powder": "chilli powder",
    "chili powder": "chilli powder",
    "red chilli powder": "chilli powder",
    "red chili powder": "chilli powder",
    "red chile powder": "chilli powder",
    "oat": "oats",
    "oats": "oats",
    "oatmeal": "oats",
    "banana": "banana",
    "honey": "honey",
    "scallion": "green onion",
    "spring onion": "green onion",
    "green onion": "green onion",
    "chickpea": "chickpea",
    "garbanzo bean": "chickpea",
    "eggplant": "eggplant",
    "aubergine": "eggplant",
    "courgette": "zucchini",
    "zucchini": "zucchini",
    "capsicum": "bell pepper",
    "bell pepper": "bell pepper",
    "red onion": "onion",
    "white onion": "onion",
    "yellow onion": "onion",
    "onion": "onion",
}
_PANTRY_STAPLE_NAMES = {
    "salt",
    "water",
    "black pepper",
    "sugar",
    "cooking oil",
    "turmeric powder",
    "chilli powder",
}
_PREPARATION_WORDS = {
    "chopped",
    "diced",
    "sliced",
    "minced",
    "grated",
    "shredded",
    "mashed",
    "crushed",
    "ground",
    "softened",
    "melted",
    "finely",
    "freshly",
    "roughly",
    "fresh",
    "ripe",
    "large",
    "medium",
    "small",
    "rolled",
    "old",
    "fashioned",
    "quick",
    "cooking",
    "drizzled",
}
_LEADING_QUANTITY = re.compile(
    r"^\s*(?:(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?|[¼½¾])\s*)"
)
_LEADING_MEASURE = re.compile(
    r"^\s*(?:cups?|tablespoons?|tbsp|teaspoons?|tsp|grams?|g|kilograms?|kg|"
    r"ounces?|oz|pounds?|lbs?|millilit(?:er|re)s?|ml|lit(?:er|re)s?|l|"
    r"pinch(?:es)?|dash(?:es)?|cloves?|slices?|pieces?|cans?|packages?|bunch(?:es)?)\b\s*"
    r"(?:of\s+)?",
    flags=re.IGNORECASE,
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


def _singularize_word(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "xes", "zes")):
        return word[:-2]
    if len(word) > 4 and word.endswith("oes"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _normalize_ingredient_name(value: str) -> str:
    """Normalize quantities and common preparation wording for ingredient matching."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("⁄", "/")
    normalized = normalized.replace("-", " ")
    preparation_tokens = _PREPARATION_WORDS | {"and"}

    def remove_preparation_note(match: re.Match[str]) -> str:
        note_words = set(re.findall(r"[a-z]+", match.group(1).casefold()))
        return " " if note_words and note_words <= preparation_tokens else match.group(0)

    normalized = re.sub(r"\(([^)]*)\)", remove_preparation_note, normalized)
    comma_parts = normalized.split(",")
    if len(comma_parts) > 1:
        note_words = set(re.findall(r"[a-z]+", ",".join(comma_parts[1:])))
        if note_words and note_words <= preparation_tokens:
            normalized = comma_parts[0]

    normalized = re.sub(r"^\s*(?:a|an)\s+", "", normalized)
    while True:
        without_quantity = _LEADING_QUANTITY.sub("", normalized, count=1)
        without_measure = _LEADING_MEASURE.sub("", without_quantity, count=1)
        if without_measure == normalized:
            break
        normalized = without_measure

    words = re.findall(r"[a-z]+", normalized)
    words = [
        word
        for word in words
        if word not in _PREPARATION_WORDS and word not in {"of", "the", "a", "an"}
    ]
    normalized = " ".join(_singularize_word(word) for word in words)
    return _INGREDIENT_ALIASES.get(normalized, normalized)


def _validate_ingredient_availability(
    recipe_ingredients: list[str],
    requested_ingredients: list[str],
) -> list[str]:
    provided = {
        _normalize_ingredient_name(ingredient)
        for ingredient in requested_ingredients
        if isinstance(ingredient, str) and ingredient.strip()
    }
    used = {
        _normalize_ingredient_name(ingredient)
        for ingredient in recipe_ingredients
        if ingredient.strip()
    }
    allowed = provided | _PANTRY_STAPLE_NAMES
    errors: list[str] = []

    unapproved = sorted(used - allowed)
    if unapproved:
        errors.append(
            "Recipe contains ingredients not provided by the user or allowed pantry "
            "staples: " + ", ".join(unapproved) + "."
        )

    unused = sorted(provided - used)
    if unused:
        errors.append(
            "Requested ingredients were not used: " + ", ".join(unused) + "."
        )
    return errors


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
    errors.extend(
        _validate_ingredient_availability(ingredients, request.ingredients)
    )

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

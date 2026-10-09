"""Tests for the input validation module."""

from recipegen.validation import validate_inputs


def test_valid_input_passes() -> None:
    errors = validate_inputs(
        ["tomato", "onion", "potato"],
        cooking_time=30,
        servings=2,
        total_input_chars=40,
    )
    assert errors == []


def test_empty_ingredients_fail() -> None:
    errors = validate_inputs([], cooking_time=30, servings=2)
    assert "At least one ingredient is required." in errors


def test_duplicate_ingredient_is_rejected() -> None:
    errors = validate_inputs(["Tomato", "tomato"], cooking_time=25, servings=2)
    assert "Duplicate ingredient: 'tomato'." in errors


def test_ingredient_length_limit_is_enforced() -> None:
    errors = validate_inputs(["x" * 41], cooking_time=20, servings=2)
    assert any("longer than 40" in message for message in errors)


def test_invalid_cooking_time_is_rejected() -> None:
    errors = validate_inputs(["rice"], cooking_time=4, servings=2)
    assert any("between 5 and 180 minutes" in message for message in errors)


def test_invalid_servings_are_rejected() -> None:
    errors = validate_inputs(["rice"], cooking_time=20, servings=0)
    assert any("between 1 and 10" in message for message in errors)


def test_input_length_limit_is_enforced() -> None:
    ingredients = ["a" * 30] * 15
    errors = validate_inputs(ingredients, cooking_time=20, servings=2, total_input_chars=600)
    assert any("under 500 characters" in message for message in errors)


def test_blank_ingredient_names_are_rejected() -> None:
    errors = validate_inputs(["   ", "onion"], cooking_time=20, servings=2)
    assert "Ingredient names cannot be blank." in errors


def test_boundary_values_are_allowed() -> None:
    errors = validate_inputs(["a" * 40, "b" * 40], cooking_time=180, servings=10, total_input_chars=500)
    assert errors == []

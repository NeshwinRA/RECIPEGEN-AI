"""Tests for generated-recipe response validation."""

from __future__ import annotations

from recipegen.models import Recipe, RecipeRequest
from recipegen.response_validation import validate_recipe_response


def make_request(
    *,
    dietary_preference: str = "Vegetarian",
    max_cooking_time: int = 30,
    servings: int = 2,
) -> RecipeRequest:
    return RecipeRequest(
        ingredients=["potato", "onion"],
        cuisine="South Indian",
        dietary_preference=dietary_preference,
        max_cooking_time=max_cooking_time,
        servings=servings,
        spice_level="medium",
        equipment="single stove",
    )


def make_recipe(**overrides) -> Recipe:
    values = {
        "recipe_id": 100,
        "recipe_name": "Potato curry",
        "ingredients": ["potato", "onion", "turmeric"],
        "steps": ["Chop the onion.", "Cook the ingredients until tender."],
        "total_time_min": 25,
        "prep_time_min": None,
        "cook_time_min": None,
        "servings": 2,
        "tags": ["vegetarian"],
        "nutrition": None,
    }
    values.update(overrides)
    return Recipe(**values)


def test_valid_recipe_passes_with_explicit_non_guarantee_warning() -> None:
    report = validate_recipe_response(make_recipe(), make_request())

    assert report.passed is True
    assert report.errors == []
    assert any("not guaranteed" in warning for warning in report.warnings)


def test_missing_required_name_ingredients_and_instructions_fail() -> None:
    response = {
        "recipe_id": 100,
        "recipe_name": " ",
        "ingredients": [],
        "instructions": [" "],
        "total_time_min": None,
        "prep_time_min": None,
        "cook_time_min": None,
        "servings": None,
        "tags": [],
        "nutrition": None,
    }

    report = validate_recipe_response(response, make_request())

    assert report.passed is False
    assert "Recipe name must be a non-empty string." in report.errors
    assert "Recipe must include at least one ingredient." in report.errors
    assert "Recipe must include at least one usable instruction step." in report.errors


def test_invalid_times_reject_negative_fractional_boolean_and_string_values() -> None:
    for invalid_time in (-1, 2.5, True, "15"):
        report = validate_recipe_response(
            make_recipe(total_time_min=invalid_time),
            make_request(),
        )
        assert report.passed is False
        assert "total_time_min must be a non-negative integer or null." in report.errors


def test_unknown_prep_and_cook_times_are_preserved_as_null() -> None:
    report = validate_recipe_response(
        make_recipe(total_time_min=25, prep_time_min=None, cook_time_min=None),
        make_request(),
    )

    assert report.passed is True
    assert not any("prep_time_min" in error for error in report.errors)
    assert not any("cook_time_min" in error for error in report.errors)


def test_total_time_over_requested_limit_fails_and_warns() -> None:
    report = validate_recipe_response(
        make_recipe(total_time_min=31),
        make_request(max_cooking_time=30),
    )

    assert report.passed is False
    assert any("exceeds the requested maximum total time" in error for error in report.errors)
    assert any("exceeds the requested maximum total time" in warning for warning in report.warnings)


def test_total_time_at_requested_limit_passes() -> None:
    report = validate_recipe_response(
        make_recipe(total_time_min=30),
        make_request(max_cooking_time=30),
    )

    assert report.passed is True


def test_missing_total_time_is_unknown_not_derived_from_prep_and_cook() -> None:
    report = validate_recipe_response(
        make_recipe(total_time_min=None, prep_time_min=10, cook_time_min=15),
        make_request(max_cooking_time=30),
    )

    assert report.passed is True
    assert any("Total cooking time is unknown" in warning for warning in report.warnings)
    assert not any("exceeds" in error for error in report.errors)


def test_invalid_and_mismatched_servings_fail() -> None:
    invalid = validate_recipe_response(make_recipe(servings=0), make_request())
    mismatch = validate_recipe_response(make_recipe(servings=3), make_request(servings=2))

    assert invalid.passed is False
    assert any("Servings must be an integer" in error for error in invalid.errors)
    assert mismatch.passed is False
    assert any("do not match the requested servings" in error for error in mismatch.errors)


def test_invalid_recipe_id_and_instruction_items_fail() -> None:
    report = validate_recipe_response(
        {
            "recipe_id": True,
            "recipe_name": "Potato curry",
            "ingredients": ["potato"],
            "instructions": ["Cook potato.", None],
            "total_time_min": 10,
            "prep_time_min": None,
            "cook_time_min": None,
            "servings": 2,
        },
        make_request(),
    )

    assert report.passed is False
    assert "Recipe ID must be a positive integer." in report.errors
    assert "Every instruction step must be a non-empty string." in report.errors


def test_invalid_optional_time_fields_fail_individually() -> None:
    for field in ("prep_time_min", "cook_time_min"):
        report = validate_recipe_response(
            make_recipe(**{field: -1}),
            make_request(),
        )
        assert report.passed is False
        assert f"{field} must be a non-negative integer or null." in report.errors


def test_malformed_tags_and_nutrition_fail() -> None:
    tags_report = validate_recipe_response(
        make_recipe(tags=["vegetarian", 1]),
        make_request(),
    )
    nutrition_report = validate_recipe_response(
        make_recipe(nutrition=[100.0, float("nan")]),
        make_request(),
    )

    assert tags_report.passed is False
    assert "Recipe tags must be a list of strings or null." in tags_report.errors
    assert nutrition_report.passed is False
    assert "Nutrition must be a list of finite numbers or null." in nutrition_report.errors


def test_null_servings_are_allowed_but_warned_as_unknown() -> None:
    report = validate_recipe_response(make_recipe(servings=None), make_request())

    assert report.passed is True
    assert "Serving size is unknown." in report.warnings


def test_egg_fails_egg_free_vegetarian_validation() -> None:
    report = validate_recipe_response(
        make_recipe(ingredients=["potato", "egg", "onion"]),
        make_request(dietary_preference="Vegetarian"),
    )

    assert report.passed is False
    assert any("conflict with the Vegetarian preference" in error for error in report.errors)


def test_egg_is_allowed_for_eggetarian() -> None:
    report = validate_recipe_response(
        make_recipe(ingredients=["potato", "egg", "onion"]),
        make_request(dietary_preference="Eggetarian"),
    )

    assert report.passed is True


def test_vegetarian_recipe_with_chicken_fails() -> None:
    report = validate_recipe_response(
        make_recipe(ingredients=["potato", "chicken", "onion"]),
        make_request(dietary_preference="Vegetarian"),
    )

    assert report.passed is False
    assert any("chicken" in error for error in report.errors)


def test_vegan_check_flags_dairy_but_allows_plant_milk() -> None:
    dairy_report = validate_recipe_response(
        make_recipe(ingredients=["oats", "milk"]),
        make_request(dietary_preference="Vegan"),
    )
    plant_report = validate_recipe_response(
        make_recipe(ingredients=["oats", "oat milk"]),
        make_request(dietary_preference="Vegan"),
    )

    assert dairy_report.passed is False
    assert plant_report.passed is True


def test_plant_milk_does_not_hide_a_separate_dairy_milk_ingredient() -> None:
    report = validate_recipe_response(
        make_recipe(ingredients=["almond milk", "milk"]),
        make_request(dietary_preference="Vegan"),
    )

    assert report.passed is False


def test_vegan_check_flags_honey_and_fish() -> None:
    honey_report = validate_recipe_response(
        make_recipe(ingredients=["oats", "honey"]),
        make_request(dietary_preference="Vegan"),
    )
    fish_report = validate_recipe_response(
        make_recipe(ingredients=["salmon", "rice"]),
        make_request(dietary_preference="Vegan"),
    )

    assert honey_report.passed is False
    assert fish_report.passed is False


def test_eggplant_does_not_match_egg_keyword() -> None:
    report = validate_recipe_response(
        make_recipe(ingredients=["eggplant", "onion"]),
        make_request(dietary_preference="Vegetarian"),
    )

    assert report.passed is True


def test_unrecognized_diet_is_not_asserted_as_suitable() -> None:
    report = validate_recipe_response(
        make_recipe(),
        make_request(dietary_preference="low-carb"),
    )

    assert report.passed is True
    assert any("no dietary suitability is asserted" in warning for warning in report.warnings)

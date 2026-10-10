"""Tests for SQLite recipe history and favourites."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from recipegen.models import GeneratedRecipe, GenerationResult, RecipeRequest
from recipegen.response_validation import ValidationReport
from recipegen.storage import (
    clear_history,
    delete_history_entry,
    get_saved_recipe,
    list_favorites,
    list_history,
    save_generation,
    set_favorite,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "nested" / "recipes.sqlite3"


@pytest.fixture
def request_model() -> RecipeRequest:
    return RecipeRequest(
        ingredients=["potato", "onion"],
        cuisine="South Indian",
        dietary_preference="Vegetarian",
        max_cooking_time=30,
        servings=2,
        spice_level="medium",
        equipment="single stove",
        bachelor_mode=True,
    )


@pytest.fixture
def result() -> GenerationResult:
    return GenerationResult(
        recipe=GeneratedRecipe(
            recipe_name="Potato curry",
            ingredients=["potato", "onion", "turmeric"],
            steps=["Chop the onion.", "Cook until tender."],
            total_time_min=None,
            prep_time_min=None,
            cook_time_min=None,
            servings=2,
            tags=["simple", "vegetarian"],
            nutrition=[100.5, 3.0],
        ),
        grounded=True,
        source_recipe_ids=[101, 202],
        validation_report=ValidationReport(
            passed=True,
            errors=[],
            warnings=["Total cooking time is unknown.", "Allergy safety is not guaranteed."],
        ),
    )


def test_initialization_creates_parent_and_schema(db_path: Path) -> None:
    history = list_history(db_path=db_path)

    assert history == []
    assert db_path.exists()
    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        foreign_keys = connection.execute("PRAGMA foreign_key_list(history)").fetchall()

    assert {"recipes", "history", "favorites"} <= tables
    assert foreign_keys


def test_save_generation_persists_complete_values_and_json_round_trips(
    db_path: Path,
    result: GenerationResult,
    request_model: RecipeRequest,
) -> None:
    recipe_id = save_generation(result, request_model, db_path)
    stored = get_saved_recipe(recipe_id, db_path)

    assert stored is not None
    assert stored.recipe == result.recipe
    assert stored.request == request_model
    assert stored.grounded is True
    assert stored.source_recipe_ids == [101, 202]
    assert stored.validation_warnings == result.validation_report.warnings
    assert stored.created_at.endswith("+00:00")
    assert stored.history_created_at is not None


def test_history_is_newest_first_and_can_be_limited(
    db_path: Path,
    result: GenerationResult,
    request_model: RecipeRequest,
) -> None:
    first_id = save_generation(result, request_model, db_path)
    second_id = save_generation(result, request_model, db_path)

    history = list_history(db_path=db_path)

    assert [entry.id for entry in history] == [second_id, first_id]
    assert [entry.id for entry in list_history(limit=1, db_path=db_path)] == [second_id]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_history_rejects_invalid_limits(
    db_path: Path,
    limit: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        list_history(limit=limit, db_path=db_path)  # type: ignore[arg-type]


def test_favourite_can_be_read_and_duplicate_saves_do_not_duplicate(
    db_path: Path,
    result: GenerationResult,
    request_model: RecipeRequest,
) -> None:
    recipe_id = save_generation(result, request_model, db_path)

    set_favorite(recipe_id, True, db_path)
    set_favorite(recipe_id, True, db_path)
    favorites = list_favorites(db_path)

    assert len(favorites) == 1
    assert favorites[0].id == recipe_id
    assert favorites[0].recipe == result.recipe
    assert favorites[0].favorite_created_at is not None
    assert get_saved_recipe(recipe_id, db_path) is not None


def test_unsaving_favorite_keeps_history_entry(
    db_path: Path,
    result: GenerationResult,
    request_model: RecipeRequest,
) -> None:
    recipe_id = save_generation(result, request_model, db_path)
    set_favorite(recipe_id, True, db_path)

    set_favorite(recipe_id, False, db_path)

    assert list_favorites(db_path) == []
    assert [entry.id for entry in list_history(db_path=db_path)] == [recipe_id]


def test_delete_history_preserves_saved_favorite(
    db_path: Path,
    result: GenerationResult,
    request_model: RecipeRequest,
) -> None:
    recipe_id = save_generation(result, request_model, db_path)
    set_favorite(recipe_id, True, db_path)

    delete_history_entry(recipe_id, db_path)

    assert list_history(db_path=db_path) == []
    assert [favorite.id for favorite in list_favorites(db_path)] == [recipe_id]
    saved = get_saved_recipe(recipe_id, db_path)
    assert saved is not None
    assert saved.history_created_at is None
    assert saved.favorite_created_at is not None


def test_delete_history_removes_unreferenced_recipe(
    db_path: Path,
    result: GenerationResult,
    request_model: RecipeRequest,
) -> None:
    recipe_id = save_generation(result, request_model, db_path)

    delete_history_entry(recipe_id, db_path)

    assert get_saved_recipe(recipe_id, db_path) is None


def test_clear_history_preserves_favorites_and_removes_other_recipes(
    db_path: Path,
    result: GenerationResult,
    request_model: RecipeRequest,
) -> None:
    favorite_id = save_generation(result, request_model, db_path)
    history_only_id = save_generation(result, request_model, db_path)
    set_favorite(favorite_id, True, db_path)

    clear_history(db_path)

    assert list_history(db_path=db_path) == []
    assert [favorite.id for favorite in list_favorites(db_path)] == [favorite_id]
    assert get_saved_recipe(favorite_id, db_path) is not None
    assert get_saved_recipe(history_only_id, db_path) is None


def test_unsave_after_history_deletion_removes_orphan_recipe(
    db_path: Path,
    result: GenerationResult,
    request_model: RecipeRequest,
) -> None:
    recipe_id = save_generation(result, request_model, db_path)
    set_favorite(recipe_id, True, db_path)
    delete_history_entry(recipe_id, db_path)

    set_favorite(recipe_id, False, db_path)

    assert get_saved_recipe(recipe_id, db_path) is None


@pytest.mark.parametrize("recipe_id", [0, -1, True, 1.5, "1"])
def test_invalid_recipe_ids_are_rejected(
    db_path: Path,
    recipe_id: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        get_saved_recipe(recipe_id, db_path)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        set_favorite(recipe_id, True, db_path)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        delete_history_entry(recipe_id, db_path)  # type: ignore[arg-type]


def test_cannot_save_failed_or_inconsistent_generation(
    db_path: Path,
    result: GenerationResult,
    request_model: RecipeRequest,
) -> None:
    failed = GenerationResult(
        recipe=result.recipe,
        grounded=False,
        source_recipe_ids=[],
        validation_report=ValidationReport(passed=False, errors=["invalid"], warnings=[]),
    )
    inconsistent = GenerationResult(
        recipe=result.recipe,
        grounded=False,
        source_recipe_ids=[101],
        validation_report=ValidationReport(passed=True, errors=[], warnings=[]),
    )

    with pytest.raises(ValueError, match="did not pass"):
        save_generation(failed, request_model, db_path)
    with pytest.raises(ValueError, match="Grounded status"):
        save_generation(inconsistent, request_model, db_path)
    assert list_history(db_path=db_path) == []


def test_set_favorite_requires_boolean(
    db_path: Path,
    result: GenerationResult,
    request_model: RecipeRequest,
) -> None:
    recipe_id = save_generation(result, request_model, db_path)

    with pytest.raises(TypeError, match="boolean"):
        set_favorite(recipe_id, 1, db_path)  # type: ignore[arg-type]

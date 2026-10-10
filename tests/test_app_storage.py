"""Streamlit integration tests for persistent recipe history and favourites."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from recipegen import pipeline, storage
from recipegen.llm_client import GeminiServiceError
from recipegen.models import GeneratedRecipe, GenerationResult, RecipeRequest
from recipegen.response_validation import ValidationReport

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def saved_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    db_path = tmp_path / "app-test.sqlite3"
    monkeypatch.setattr(storage, "DEFAULT_DATABASE_PATH", db_path)
    import app

    monkeypatch.setattr(app, "DATABASE_PATH", db_path)
    return db_path


@pytest.fixture
def request_model() -> RecipeRequest:
    return RecipeRequest(
        ingredients=["potato", "onion"],
        cuisine="Any cuisine",
        dietary_preference="Vegetarian",
        max_cooking_time=30,
        servings=2,
        spice_level="Mild",
        equipment="Any available equipment",
        bachelor_mode=False,
    )


@pytest.fixture
def generated_result() -> GenerationResult:
    return GenerationResult(
        recipe=GeneratedRecipe(
            recipe_name="Potato curry",
            ingredients=["potato", "onion"],
            steps=["Chop the onion.", "Cook until tender."],
            total_time_min=None,
            prep_time_min=None,
            cook_time_min=None,
            servings=2,
            tags=["simple"],
            nutrition=None,
        ),
        grounded=False,
        source_recipe_ids=[],
        validation_report=ValidationReport(
            passed=True,
            errors=[],
            warnings=["Total cooking time is unknown."],
        ),
    )


def _find_button(at: AppTest, label: str):
    return next(button for button in at.button if button.label == label)


def _open_app() -> AppTest:
    return AppTest.from_file(str(APP_PATH)).run()


def _submit_generation(at: AppTest) -> None:
    at.text_area(key="ingredients_input").set_value("potato, onion")
    _find_button(at, "Generate My Recipe").click()
    at.run()


def test_successful_generation_is_saved_once_across_reruns(
    saved_db: Path,
    request_model: RecipeRequest,
    generated_result: GenerationResult,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    call_count = 0

    def fake_generate(request: RecipeRequest):
        nonlocal call_count
        call_count += 1
        assert request.ingredients == ["potato", "onion"]
        return generated_result

    monkeypatch.setattr(pipeline, "generate_recipe", fake_generate)
    at = _open_app()

    with caplog.at_level(logging.INFO, logger="recipegen.timing"):
        _submit_generation(at)

    assert call_count == 1
    assert "stage=sqlite_persistence" in caplog.text
    assert "potato" not in caplog.text
    history = storage.list_history(db_path=saved_db)
    assert len(history) == 1
    assert history[0].request == request_model
    assert at.session_state["generation_record_id"] == history[0].id

    at.run()

    assert call_count == 1
    assert len(storage.list_history(db_path=saved_db)) == 1


def test_failed_generation_is_not_saved(
    saved_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_generation(request: RecipeRequest):
        raise pipeline.RequestValidationError("Invalid request.")

    monkeypatch.setattr(pipeline, "generate_recipe", fail_generation)
    at = _open_app()

    _submit_generation(at)

    assert storage.list_history(db_path=saved_db) == []
    assert at.session_state.get("generation_record_id") is None
    assert any("Invalid request" in error.value for error in at.error)


def test_gemini_failure_logs_category_without_exception_message(
    saved_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_message = "private prompt and secret-token"

    def fail_generation(request: RecipeRequest):
        raise GeminiServiceError(sensitive_message)

    monkeypatch.setattr(pipeline, "generate_recipe", fail_generation)
    at = _open_app()

    with caplog.at_level(logging.WARNING):
        _submit_generation(at)

    assert "The recipe service could not complete the request. Please try again later." in [
        error.value for error in at.error
    ]
    assert "Gemini request failed exception_class=GeminiServiceError" in caplog.text
    assert sensitive_message not in caplog.text
    assert "private prompt" not in caplog.text
    assert "secret-token" not in caplog.text
    assert storage.list_history(db_path=saved_db) == []


def test_unexpected_generation_failure_logs_only_exception_class(
    saved_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_message = "private ingredient request and credential"

    def fail_generation(request: RecipeRequest):
        raise RuntimeError(sensitive_message)

    monkeypatch.setattr(pipeline, "generate_recipe", fail_generation)
    at = _open_app()

    with caplog.at_level(logging.WARNING):
        _submit_generation(at)

    assert "An unexpected error occurred while generating the recipe. Please try again." in [
        error.value for error in at.error
    ]
    assert "Recipe generation failed exception_class=RuntimeError" in caplog.text
    assert sensitive_message not in caplog.text
    assert "credential" not in caplog.text
    assert storage.list_history(db_path=saved_db) == []


def test_save_failure_keeps_recipe_and_reports_history_error(
    saved_db: Path,
    generated_result: GenerationResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "generate_recipe",
        lambda request: generated_result,
    )

    def fail_save(*args, **kwargs):
        raise OSError("database is unavailable")

    monkeypatch.setattr(storage, "save_generation", fail_save)
    at = _open_app()

    _submit_generation(at)

    assert isinstance(at.session_state["generation_result"], GenerationResult)
    assert at.session_state.get("generation_record_id") is None
    assert any("generated successfully but could not be saved" in error.value for error in at.error)
    assert storage.list_history(db_path=saved_db) == []


def test_opening_history_recipe_does_not_regenerate(
    saved_db: Path,
    request_model: RecipeRequest,
    generated_result: GenerationResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_id = storage.save_generation(generated_result, request_model, saved_db)

    def unexpected_generation(request: RecipeRequest):
        raise AssertionError("Opening a saved recipe must not generate it again.")

    monkeypatch.setattr(pipeline, "generate_recipe", unexpected_generation)
    at = _open_app()
    at.radio[0].set_value("🕘 History").run()

    assert any(button.label == "Open Recipe" for button in at.button)
    _find_button(at, "Open Recipe").click()
    at.run()

    assert at.session_state["opened_recipe_id"] == recipe_id
    assert any("Potato curry" in item.value for item in at.subheader)


def test_stale_open_recipe_selection_is_cleared(
    saved_db: Path,
    request_model: RecipeRequest,
    generated_result: GenerationResult,
) -> None:
    recipe_id = storage.save_generation(generated_result, request_model, saved_db)
    storage.delete_history_entry(recipe_id, saved_db)
    at = _open_app()
    at.session_state["opened_recipe_id"] = recipe_id
    at.radio[0].set_value("🕘 History").run()

    assert at.session_state.get("opened_recipe_id") is None
    assert any("no longer available" in item.value for item in at.info)


def test_deleting_history_preserves_favorite(
    saved_db: Path,
    request_model: RecipeRequest,
    generated_result: GenerationResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_id = storage.save_generation(generated_result, request_model, saved_db)
    storage.set_favorite(recipe_id, True, saved_db)
    monkeypatch.setattr(
        pipeline,
        "generate_recipe",
        lambda request: pytest.fail("History actions must not generate recipes."),
    )
    at = _open_app()
    at.radio[0].set_value("🕘 History").run()

    _find_button(at, "Delete").click()
    at.run()

    assert storage.list_history(db_path=saved_db) == []
    assert [item.id for item in storage.list_favorites(saved_db)] == [recipe_id]
    assert storage.get_saved_recipe(recipe_id, saved_db) is not None


def test_removing_favorite_keeps_history_entry(
    saved_db: Path,
    request_model: RecipeRequest,
    generated_result: GenerationResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_id = storage.save_generation(generated_result, request_model, saved_db)
    storage.set_favorite(recipe_id, True, saved_db)
    monkeypatch.setattr(
        pipeline,
        "generate_recipe",
        lambda request: pytest.fail("Saved recipe actions must not generate recipes."),
    )
    at = _open_app()
    at.radio[0].set_value("⭐ Saved Recipes").run()

    _find_button(at, "Open Recipe").click()
    at.run()
    _find_button(at, "Remove from Favorites").click()
    at.run()

    assert storage.list_favorites(db_path=saved_db) == []
    assert [item.id for item in storage.list_history(db_path=saved_db)] == [recipe_id]


def test_generated_recipe_can_be_saved_and_removed_from_favorites(
    saved_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    generated_result: GenerationResult,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "generate_recipe",
        lambda request: generated_result,
    )
    at = _open_app()
    _submit_generation(at)

    _find_button(at, "Save to Favorites").click()
    at.run()
    record_id = at.session_state["generation_record_id"]
    assert [item.id for item in storage.list_favorites(db_path=saved_db)] == [record_id]

    _find_button(at, "Remove from Favorites").click()
    at.run()
    assert storage.list_favorites(db_path=saved_db) == []
    assert [item.id for item in storage.list_history(db_path=saved_db)] == [record_id]


def test_clear_history_requires_confirmation_and_preserves_favorites(
    saved_db: Path,
    request_model: RecipeRequest,
    generated_result: GenerationResult,
) -> None:
    recipe_id = storage.save_generation(generated_result, request_model, saved_db)
    storage.set_favorite(recipe_id, True, saved_db)
    at = _open_app()
    at.radio[0].set_value("🕘 History").run()

    _find_button(at, "Clear History").click()
    at.run()
    assert len(storage.list_history(db_path=saved_db)) == 1

    _find_button(at, "Confirm Clear History").click()
    at.run()

    assert storage.list_history(db_path=saved_db) == []
    assert [item.id for item in storage.list_favorites(db_path=saved_db)] == [recipe_id]


def test_favorite_remains_openable_after_history_deletion(
    saved_db: Path,
    request_model: RecipeRequest,
    generated_result: GenerationResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_id = storage.save_generation(generated_result, request_model, saved_db)
    storage.set_favorite(recipe_id, True, saved_db)
    storage.delete_history_entry(recipe_id, saved_db)
    monkeypatch.setattr(
        pipeline,
        "generate_recipe",
        lambda request: pytest.fail("Opening a favourite must not generate recipes."),
    )
    at = _open_app()
    at.radio[0].set_value("⭐ Saved Recipes").run()
    _find_button(at, "Open Recipe").click()
    at.run()

    assert at.session_state["opened_recipe_id"] == recipe_id
    assert any("Potato curry" in item.value for item in at.subheader)

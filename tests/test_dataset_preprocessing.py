"""Tests for dataset preprocessing and schema mapping."""

from __future__ import annotations

import json

import pandas as pd

from recipegen.dataset_preprocessing import normalize_recipe_row, process_raw_recipe_csv


VALID_ROW = {
    "id": 101,
    "name": "  Quick Tomato Rice  ",
    "minutes": 25,
    "tags": "['easy', 'dinner', 'rice']",
    "nutrition": "[200.0, 10.0, 5.0, 2.0, 15.0, 5.0, 8.0]",
    "ingredients": "['tomato', 'rice', 'onion', 'oil']",
    "steps": "['wash rice', 'saute onion', 'add tomato', 'cook until done']",
}


def test_normalize_recipe_row_valid() -> None:
    recipe, reason = normalize_recipe_row(VALID_ROW)
    assert reason is None
    assert recipe is not None
    assert recipe.recipe_id == 101
    assert recipe.recipe_name == "Quick Tomato Rice"
    assert recipe.total_time_min == 25
    assert recipe.ingredients == ["tomato", "rice", "onion", "oil"]
    assert recipe.steps[0] == "wash rice"
    assert recipe.tags == ["easy", "dinner", "rice"]
    assert recipe.nutrition == [200.0, 10.0, 5.0, 2.0, 15.0, 5.0, 8.0]


def test_missing_name_is_excluded() -> None:
    row = {**VALID_ROW, "name": "   "}
    recipe, reason = normalize_recipe_row(row)
    assert recipe is None
    assert reason == "missing_recipe_name"


def test_invalid_ingredients_are_excluded() -> None:
    row = {**VALID_ROW, "ingredients": "not a list"}
    recipe, reason = normalize_recipe_row(row)
    assert recipe is None
    assert reason == "invalid_ingredients"


def test_process_raw_recipe_csv_creates_output_and_report(tmp_path) -> None:
    input_path = tmp_path / "sample.csv"
    output_path = tmp_path / "processed" / "recipes.csv"
    report_path = tmp_path / "processed" / "report.json"

    sample = pd.DataFrame(
        [
            VALID_ROW,
            {**VALID_ROW, "id": 102, "name": "    ", "ingredients": "['ok']", "steps": "['prep']"},
            {**VALID_ROW, "id": 103, "ingredients": "['x']", "steps": "[]"},
        ]
    )
    sample.to_csv(input_path, index=False)

    report = process_raw_recipe_csv(input_path, output_path, report_path, chunksize=10)

    assert report["input_record_count"] == 3
    assert report["output_record_count"] == 1
    assert report["excluded_record_count"] == 2
    assert "missing_recipe_name" in report["excluded_by_reason"]
    assert "invalid_steps" in report["excluded_by_reason"]

    result = pd.read_csv(output_path)
    assert set(result.columns) == {
        "recipe_id",
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
    payload = result.iloc[0].to_dict()
    assert payload["recipe_id"] == 101
    assert payload["recipe_name"] == "Quick Tomato Rice"
    assert json.loads(payload["ingredients"]) == ["tomato", "rice", "onion", "oil"]
    assert json.loads(payload["steps"])[0] == "wash rice"

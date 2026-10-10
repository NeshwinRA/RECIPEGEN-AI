from __future__ import annotations

import json

import pandas as pd

from recipegen.retrieval import build_searchable_document, prepare_recipe_documents

VALID_ROW = {
    "recipe_id": 101,
    "recipe_name": "Quick Tomato Rice",
    "ingredients": '["tomato", "rice", "onion", "oil"]',
    "steps": '["wash rice", "saute onion", "add tomato", "cook until done"]',
    "total_time_min": 25,
    "tags": '["easy", "dinner", "rice"]',
}


def test_build_searchable_document_is_stable() -> None:
    doc_a = build_searchable_document(
        recipe_id=101,
        recipe_name="Quick Tomato Rice",
        ingredients=["tomato", "rice", "onion"],
        steps=["wash rice", "cook until done"],
        total_time_min=25,
        tags=["easy", "dinner"],
    )
    doc_b = build_searchable_document(
        recipe_id=101,
        recipe_name="Quick Tomato Rice",
        ingredients=["tomato", "rice", "onion"],
        steps=["wash rice", "cook until done"],
        total_time_min=25,
        tags=["easy", "dinner"],
    )

    assert doc_a == doc_b
    assert "recipe_name: Quick Tomato Rice" in doc_a
    assert "ingredients: tomato; rice; onion" in doc_a
    assert "total_time_min: 25" in doc_a


def test_prepare_recipe_documents_preserves_metadata_and_excludes_invalid_rows(tmp_path) -> None:
    csv_path = tmp_path / "prepared.csv"
    docs_path = tmp_path / "documents.jsonl"
    metadata_path = tmp_path / "metadata.csv"
    report_path = tmp_path / "report.json"

    sample = pd.DataFrame(
        [
            VALID_ROW,
            {**VALID_ROW, "recipe_id": 102, "recipe_name": "  ", "ingredients": '["ok"]', "steps": '["prep"]'},
            {**VALID_ROW, "recipe_id": 103, "recipe_name": "No Veg", "ingredients": "[]", "steps": '["prep"]'},
            {**VALID_ROW, "recipe_id": 104, "recipe_name": "No Time", "total_time_min": None},
        ]
    )
    sample.to_csv(csv_path, index=False)

    report = prepare_recipe_documents(csv_path, docs_path, metadata_path, report_path)

    assert report["input_record_count"] == 4
    assert report["output_document_count"] == 2
    assert report["excluded_document_count"] == 2
    assert report["excluded_by_reason"]["missing_recipe_name"] == 1
    assert report["excluded_by_reason"]["empty_ingredients"] == 1

    documents = [json.loads(line) for line in docs_path.read_text(encoding="utf-8").splitlines()]
    assert {item["recipe_id"] for item in documents} == {101, 104}
    assert "Quick Tomato Rice" in documents[0]["document_text"]

    metadata = pd.read_csv(metadata_path)
    assert set(metadata.columns) == {
        "recipe_id",
        "recipe_name",
        "total_time_min",
        "ingredient_count",
        "step_count",
        "tag_count",
        "tags",
    }
    assert metadata["recipe_id"].tolist() == [101, 104]
    assert metadata["total_time_min"].isna().sum() == 1
    assert metadata["ingredient_count"].tolist() == [4, 4]
    assert metadata["tag_count"].tolist() == [3, 3]


def test_prepare_recipe_documents_uses_stable_repeatable_document_text() -> None:
    row = {**VALID_ROW, "recipe_id": 777}
    doc_a = build_searchable_document(
        recipe_id=row["recipe_id"],
        recipe_name=row["recipe_name"],
        ingredients=["tomato", "rice", "onion", "oil"],
        steps=["wash rice", "saute onion", "add tomato", "cook until done"],
        total_time_min=25,
        tags=["easy", "dinner", "rice"],
    )
    doc_b = build_searchable_document(
        recipe_id=row["recipe_id"],
        recipe_name=row["recipe_name"],
        ingredients=["tomato", "rice", "onion", "oil"],
        steps=["wash rice", "saute onion", "add tomato", "cook until done"],
        total_time_min=25,
        tags=["easy", "dinner", "rice"],
    )
    assert doc_a == doc_b

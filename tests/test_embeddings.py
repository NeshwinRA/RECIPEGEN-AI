from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recipegen.embeddings import (
    ONNXEmbeddingModel,
    DEFAULT_EMBEDDING_MODEL,
    index_recipe_documents,
    query_recipe_collection,
    verify_persistence,
)


class FakeTokenizer:
    def encode_batch(self, texts, add_special_tokens=True):
        encodings = []
        tokenized = [[1, *range(2, len(text.split()) + 2), 99] for text in texts]
        max_length = max(len(ids) for ids in tokenized)
        for ids in tokenized:
            length = len(ids)
            padded_ids = ids + [0] * (max_length - length)
            attention_mask = [1] * length + [0] * (max_length - length)
            type_ids = [0] * max_length
            encodings.append(
                SimpleNamespace(ids=padded_ids, attention_mask=attention_mask, type_ids=type_ids)
            )
        return encodings


class FakeOnnxSession:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in ("input_ids", "attention_mask", "token_type_ids")]

    def get_outputs(self):
        return [SimpleNamespace(name="last_hidden_state")]

    def run(self, output_names, feed):
        if self.failure is not None:
            raise self.failure
        self.calls.append(feed["input_ids"].shape[0])
        ids = feed["input_ids"]
        hidden = np.zeros((*ids.shape, 384), dtype=np.float32)
        for row in range(ids.shape[0]):
            for col in range(ids.shape[1]):
                hidden[row, col, 0] = 2.0 if ids[row, col] != 0 else 500.0
                hidden[row, col, 1] = 1.0 if ids[row, col] != 0 else -500.0
        return [hidden]


def make_fake_onnx_model(session=None) -> ONNXEmbeddingModel:
    model = object.__new__(ONNXEmbeddingModel)
    model.tokenizer = FakeTokenizer()
    model.session = session or FakeOnnxSession()
    model.input_names = {"input_ids", "attention_mask", "token_type_ids"}
    model.output_names = ["last_hidden_state"]
    return model


def test_onnx_embedding_dimensions_and_normalization() -> None:
    embeddings = ONNXEmbeddingModel.encode(make_fake_onnx_model(), ["one two", "one"])

    assert embeddings.shape == (2, 384)
    assert np.linalg.norm(embeddings, axis=1) == pytest.approx([1.0, 1.0])


def test_onnx_embedding_batches_and_masks_padding() -> None:
    session = FakeOnnxSession()
    embeddings = ONNXEmbeddingModel.encode(
        make_fake_onnx_model(session),
        ["one two", "one", "one two three"],
        batch_size=2,
    )

    assert session.calls == [2, 1]
    assert embeddings.shape == (3, 384)
    assert np.allclose(embeddings[0], embeddings[1])


def test_onnx_embedding_empty_input_returns_empty_vectors() -> None:
    model = make_fake_onnx_model()
    vectors = ONNXEmbeddingModel.encode(model, [])

    assert vectors.shape == (0, 384)
    assert model.session.calls == []


@pytest.mark.parametrize("texts", [[""], ["valid", "   "], [None]])
def test_onnx_embedding_rejects_invalid_documents(texts) -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        ONNXEmbeddingModel.encode(make_fake_onnx_model(), texts)


def test_onnx_embedding_surfaces_inference_errors() -> None:
    model = make_fake_onnx_model(FakeOnnxSession(failure=RuntimeError("session failed")))

    with pytest.raises(RuntimeError, match="ONNX embedding batch 0:1 failed: session failed"):
        ONNXEmbeddingModel.encode(model, ["valid"])


def test_onnx_embedding_refuses_different_model_name() -> None:
    with pytest.raises(ValueError, match="refusing to substitute"):
        from recipegen.embeddings import embed_texts

        embed_texts(["recipe"], model_name="different/model", model=make_fake_onnx_model())


def _write_documents(path: Path) -> None:
    records = [
        {
            "recipe_id": 101,
            "recipe_name": "Quick Tomato Rice",
            "document_text": "recipe_id: 101 | recipe_name: Quick Tomato Rice | ingredients: tomato; rice; onion | instructions: wash rice and cook until done | total_time_min: 25 | tags: easy, dinner, rice",
        },
        {
            "recipe_id": 102,
            "recipe_name": "Spicy Veggie Bowl",
            "document_text": "recipe_id: 102 | recipe_name: Spicy Veggie Bowl | ingredients: pepper; onion; rice; beans | instructions: saute onion and pepper and serve with rice | total_time_min: 20 | tags: quick, vegetarian",
        },
        {
            "recipe_id": 103,
            "recipe_name": "Berry Oatmeal",
            "document_text": "recipe_id: 103 | recipe_name: Berry Oatmeal | ingredients: oats; berries; milk | instructions: cook oats and top with berries | total_time_min: 10 | tags: breakfast, fruit",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item) + "\n")


def test_index_recipe_documents_ingests_batch_and_preserves_metadata(monkeypatch, tmp_path) -> None:
    class FakeModel:
        def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False):
            vectors = []
            for text in texts:
                fingerprint = 0.0
                if "rice" in text.lower():
                    fingerprint += 1.0
                if "berry" in text.lower():
                    fingerprint += 2.0
                if "veggie" in text.lower():
                    fingerprint += 3.0
                vectors.append([fingerprint, fingerprint + 1.0, fingerprint + 2.0])
            return np.array(vectors, dtype=float)

    monkeypatch.setattr("recipegen.embeddings.load_embedding_model", lambda model_name, device=None: FakeModel())

    docs_path = tmp_path / "docs.jsonl"
    db_path = tmp_path / "vector_db"
    _write_documents(docs_path)

    report = index_recipe_documents(
        documents_path=docs_path,
        db_path=db_path,
        collection_name="recipe_test_collection",
        sample_size=3,
        batch_size=2,
        reset_collection=True,
    )

    assert report["document_count"] == 3
    assert report["embedding_dimension"] == 3
    assert report["sample_size"] == 3

    persistence = verify_persistence(db_path, "recipe_test_collection")
    assert persistence["document_count"] == 3


def test_repeated_indexing_is_idempotent(monkeypatch, tmp_path) -> None:
    class FakeModel:
        encode_calls = 0

        def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False):
            self.encode_calls += 1
            return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=float)

    model = FakeModel()
    monkeypatch.setattr("recipegen.embeddings.load_embedding_model", lambda model_name, device=None: model)

    docs_path = tmp_path / "docs.jsonl"
    db_path = tmp_path / "vector_db"
    _write_documents(docs_path)

    first = index_recipe_documents(docs_path, db_path, collection_name="repeat_test", sample_size=2, batch_size=1, reset_collection=True)
    second = index_recipe_documents(docs_path, db_path, collection_name="repeat_test", sample_size=2, batch_size=1, reset_collection=False)

    assert first["document_count"] == 2
    assert second["document_count"] == 2
    assert second["skipped_count"] == 2
    assert model.encode_calls == 2
    assert verify_persistence(db_path, "repeat_test")["document_count"] == 2


def test_indexing_saves_periodic_progress_and_elapsed_time(monkeypatch, tmp_path, capsys) -> None:
    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=float)

    monkeypatch.setattr("recipegen.embeddings.load_embedding_model", lambda model_name, device=None: FakeModel())
    docs_path = tmp_path / "docs.jsonl"
    db_path = tmp_path / "vector_db"
    progress_path = tmp_path / "progress.json"
    _write_documents(docs_path)

    report = index_recipe_documents(
        docs_path,
        db_path,
        collection_name="progress_test",
        sample_size=3,
        batch_size=1,
        progress_interval=1,
        reset_collection=False,
        progress_report_path=progress_path,
    )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert progress["status"] == "completed"
    assert progress["source_record_count"] == 3
    assert progress["document_count"] == 3
    assert progress["indexed_count"] == 3
    assert progress["skipped_count"] == 0
    assert progress["failed_records"] == 0
    assert progress["elapsed_seconds"] >= 0
    assert report["elapsed_seconds"] >= 0
    assert "elapsed" in output
    assert not progress_path.with_name(f"{progress_path.name}.tmp").exists()


def test_protected_collections_cannot_be_reset(tmp_path) -> None:
    with pytest.raises(ValueError, match="Refusing to reset protected"):
        index_recipe_documents(
            tmp_path / "missing.jsonl",
            tmp_path / "db",
            collection_name="recipe_documents_full",
            reset_collection=True,
            log_progress=False,
        )


def test_indexing_records_invalid_recipe_ids_and_documents(monkeypatch, tmp_path) -> None:
    class FakeModel:
        def encode(self, texts, **kwargs):
            raise AssertionError("Invalid rows must be rejected before embedding")

    monkeypatch.setattr("recipegen.embeddings.load_embedding_model", lambda model_name, device=None: FakeModel())
    docs_path = tmp_path / "invalid.jsonl"
    docs_path.write_text(
        "\n".join(
            [
                json.dumps({"recipe_id": True, "document_text": "invalid id"}),
                json.dumps({"recipe_id": 10.5, "document_text": "fractional id"}),
                json.dumps({"recipe_id": 12, "document_text": None}),
            ]
        ),
        encoding="utf-8",
    )

    report = index_recipe_documents(
        docs_path,
        tmp_path / "db",
        collection_name="invalid_test",
        reset_collection=False,
        log_progress=False,
    )

    assert report["document_count"] == 0
    assert report["failed_records"] == 3
    assert {failure["recipe_id"] for failure in report["failed_ids"]} == {True, 10.5, 12}


def test_query_recipe_collection_returns_relevant_recipe(monkeypatch, tmp_path) -> None:
    class FakeModel:
        def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False):
            vectors = []
            for text in texts:
                if "rice" in text.lower():
                    vectors.append([1.0, 0.0, 0.0])
                else:
                    vectors.append([0.0, 1.0, 0.0])
            return np.array(vectors, dtype=float)

    monkeypatch.setattr("recipegen.embeddings.load_embedding_model", lambda model_name, device=None: FakeModel())

    docs_path = tmp_path / "docs.jsonl"
    db_path = tmp_path / "vector_db"
    _write_documents(docs_path)
    index_recipe_documents(docs_path, db_path, collection_name="query_test", sample_size=3, batch_size=3, reset_collection=True)

    results = query_recipe_collection("quick rice dinner", db_path, collection_name="query_test", n_results=2)
    assert len(results["ids"]) == 1
    assert len(results["ids"][0]) >= 1
    assert "recipe_101" in results["ids"][0]

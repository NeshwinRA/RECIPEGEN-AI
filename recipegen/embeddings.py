"""Embedding and ChromaDB helpers for the prepared recipe documents."""

from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "recipe_documents"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
TOKENIZER_GIT_BLOB_SHA1 = "cb202bfe2e3c98645018a6d12f182a434c9d3e02"
MAX_SEQUENCE_LENGTH = 256
EMBEDDING_DIMENSION = 384


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


class ONNXEmbeddingModel:
    """CPU ONNX encoder matching the Sentence Transformers MiniLM pipeline."""

    embedding_dimension = EMBEDDING_DIMENSION

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        *,
        _tokenizer: Tokenizer | None = None,
        _session: ort.InferenceSession | None = None,
    ) -> None:
        if model_name != DEFAULT_EMBEDDING_MODEL:
            raise ValueError(
                f"Unsupported embedding model {model_name!r}; expected {DEFAULT_EMBEDDING_MODEL!r}"
            )
        if device is not None and device.lower() != "cpu":
            raise ValueError("Only CPUExecutionProvider is supported by the production ONNX encoder")

        if _tokenizer is None or _session is None:
            try:
                from huggingface_hub import hf_hub_download

                model_path = hf_hub_download(
                    repo_id=DEFAULT_EMBEDDING_MODEL,
                    filename="onnx/model.onnx",
                    revision=MODEL_REVISION,
                )
                tokenizer_path = hf_hub_download(
                    repo_id=DEFAULT_EMBEDDING_MODEL,
                    filename="tokenizer.json",
                    revision=MODEL_REVISION,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to obtain pinned files for embedding model {DEFAULT_EMBEDDING_MODEL}: {exc}"
                ) from exc

            model_digest = _sha256_file(model_path)
            if model_digest != MODEL_SHA256:
                raise RuntimeError(
                    f"Embedding model integrity check failed: expected SHA-256 {MODEL_SHA256}, "
                    f"received {model_digest}"
                )
            tokenizer_digest = _tokenizer_git_blob_sha1(Path(tokenizer_path).read_bytes())
            if tokenizer_digest != TOKENIZER_GIT_BLOB_SHA1:
                raise RuntimeError(
                    "Tokenizer integrity check failed against the pinned repository blob hash"
                )

            tokenizer_config = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
            normalizer = tokenizer_config.get("normalizer") or {}
            bert_normalizer = normalizer.get(" BertNormalizer", normalizer)
            if bert_normalizer.get("lowercase") is not True:
                raise RuntimeError("Pinned tokenizer is not configured as uncased")

            _tokenizer = Tokenizer.from_file(str(tokenizer_path))
            _tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
            _tokenizer.enable_padding(
                direction="right",
                pad_id=0,
                pad_token="[PAD]",
                pad_type_id=0,
            )
            try:
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                _session = ort.InferenceSession(
                    str(model_path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:
                raise RuntimeError(f"Unable to initialize the pinned ONNX model: {exc}") from exc

        if _session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(
                f"Expected CPUExecutionProvider only, got {_session.get_providers()}"
            )
        self.tokenizer = _tokenizer
        self.session = _session
        self.input_names = {item.name for item in _session.get_inputs()}
        self.output_names = [item.name for item in _session.get_outputs()]
        required_supported_inputs = {"input_ids", "attention_mask", "token_type_ids"}
        if not self.input_names or not self.input_names.issubset(required_supported_inputs):
            raise RuntimeError(f"Unexpected ONNX graph inputs: {sorted(self.input_names)}")

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray | list[list[float]]:
        """Encode non-empty strings in bounded batches with mask-aware pooling."""

        del show_progress_bar
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32) if convert_to_numpy else []
        invalid_indices = [
            index
            for index, text in enumerate(texts)
            if not isinstance(text, str) or not text.strip()
        ]
        if invalid_indices:
            raise ValueError(f"Embedding inputs must be non-empty strings; invalid indices: {invalid_indices}")

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                encodings = self.tokenizer.encode_batch(batch, add_special_tokens=True)
                sequence_length = max(len(encoding.ids) for encoding in encodings)
                input_ids = np.zeros((len(encodings), sequence_length), dtype=np.int64)
                attention_mask = np.zeros_like(input_ids)
                token_type_ids = np.zeros_like(input_ids)
                for index, encoding in enumerate(encodings):
                    length = len(encoding.ids)
                    input_ids[index, :length] = encoding.ids
                    attention_mask[index, :length] = encoding.attention_mask
                    token_type_ids[index, :length] = encoding.type_ids

                arrays = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }
                outputs = self.session.run(
                    self.output_names,
                    {name: arrays[name] for name in self.input_names},
                )
                token_embeddings = next(
                    (
                        value
                        for value in outputs
                        if value.ndim == 3 and value.shape[:2] == input_ids.shape
                    ),
                    None,
                )
                if token_embeddings is None:
                    raise RuntimeError("ONNX graph did not return token-level embeddings")
                expanded_mask = attention_mask[..., np.newaxis].astype(token_embeddings.dtype)
                token_counts = np.maximum(np.sum(expanded_mask, axis=1), 1e-9)
                pooled = np.sum(token_embeddings * expanded_mask, axis=1) / token_counts
                if not np.all(np.isfinite(pooled)):
                    raise RuntimeError("ONNX inference returned non-finite values")
                if normalize_embeddings:
                    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                    if np.any(norms == 0):
                        raise RuntimeError("ONNX inference returned a zero-length embedding")
                    pooled = pooled / norms
                if pooled.shape[1] != EMBEDDING_DIMENSION:
                    raise RuntimeError(
                        f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings, got {pooled.shape[1]}"
                    )
                vectors.append(pooled.astype(np.float32))
            except Exception as exc:
                raise RuntimeError(f"ONNX embedding batch {start}:{start + len(batch)} failed: {exc}") from exc

        result = np.concatenate(vectors, axis=0)
        return result if convert_to_numpy else result.tolist()


@lru_cache(maxsize=1)
def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
) -> ONNXEmbeddingModel:
    """Load and cache the pinned ONNX model and tokenizer for this process."""

    return ONNXEmbeddingModel(model_name=model_name, device=device)


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model: ONNXEmbeddingModel | None = None,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed a list of input strings using the configured local model."""

    if not texts:
        return []

    if model_name != DEFAULT_EMBEDDING_MODEL:
        raise ValueError(
            f"Unsupported embedding model {model_name!r}; refusing to substitute a different model"
        )
    if model is None:
        model = load_embedding_model(model_name=model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return [list(vector) for vector in embeddings]


def read_recipe_documents(documents_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read JSONL recipe documents into a list, optionally stopping at a limit."""

    documents: list[dict[str, Any]] = []
    with Path(documents_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            documents.append(json.loads(stripped))
            if limit is not None and len(documents) >= limit:
                break
    return documents


def _collection_client(db_path: str | Path):
    """Open the persistent ChromaDB client for the project vector store."""

    return chromadb.PersistentClient(path=str(Path(db_path)))


def index_recipe_documents(
    documents_path: str | Path,
    db_path: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    sample_size: int | None = None,
    batch_size: int = 128,
    reset_collection: bool = True,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    progress_interval: int = 10000,
    log_progress: bool = True,
    embedding_batch_size: int = 32,
    progress_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index prepared recipe documents into a persistent ChromaDB collection.

    The function supports sample indexing and full batch indexing. When
    reset_collection is False, existing collection records are reused and only
    not-yet-indexed documents are added, allowing resumable runs.
    """

    if batch_size < 1 or embedding_batch_size < 1 or progress_interval < 1:
        raise ValueError("batch_size, embedding_batch_size, and progress_interval must be positive")
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be positive when provided")
    if model_name != DEFAULT_EMBEDDING_MODEL:
        raise ValueError(
            f"Unsupported embedding model {model_name!r}; refusing to mix model configurations"
        )
    if reset_collection and collection_name in {"recipe_documents_full", "recipe_documents_sample"}:
        raise ValueError(
            f"Refusing to reset protected production collection {collection_name!r}"
        )
    client = _collection_client(db_path)

    if reset_collection:
        try:
            client.delete_collection(name=collection_name)
        except (ValueError, NotFoundError):
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    model = load_embedding_model(model_name=model_name)
    existing_records = collection.get(limit=1, include=["embeddings"])
    if existing_records["embeddings"] is not None and len(existing_records["embeddings"]):
        existing_dimension = len(existing_records["embeddings"][0])
    else:
        existing_dimension = None
    existing_ids: set[str] = set(collection.get(include=[]).get("ids", []))
    indexed_count = 0
    skipped_count = 0
    failed_ids: list[dict[str, Any]] = []
    embedding_dimension: int | None = existing_dimension
    next_progress = progress_interval
    processed_count = 0
    started_at = time.perf_counter()
    started_at_epoch = time.time()
    progress_path = Path(progress_report_path) if progress_report_path is not None else None

    def save_progress(status: str) -> None:
        if progress_path is None:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "status": status,
            "db_path": str(Path(db_path)),
            "collection_name": collection_name,
            "model_name": model_name,
            "source_path": str(Path(documents_path)),
            "source_record_count": processed_count,
            "document_count": collection.count(),
            "indexed_count": indexed_count,
            "skipped_count": skipped_count,
            "failed_records": len(failed_ids),
            "failed_ids": failed_ids,
            "embedding_dimension": embedding_dimension,
            "batch_size": batch_size,
            "embedding_batch_size": embedding_batch_size,
            "progress_interval": progress_interval,
            "reset_collection": reset_collection,
            "started_at_epoch": started_at_epoch,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        }
        temporary_path = progress_path.with_name(f"{progress_path.name}.tmp")
        temporary_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        temporary_path.replace(progress_path)

    if log_progress:
        print(
            f"Starting index from {documents_path} into collection '{collection_name}' "
            f"with source batch size {batch_size} and embedding batch size {embedding_batch_size}"
        )
    save_progress("running")

    def process_batch(batch: list[dict[str, Any]], start: int) -> None:
        nonlocal indexed_count, skipped_count, embedding_dimension, existing_dimension
        ids: list[str] = []
        texts: list[str] = []
        metadata: list[dict[str, Any]] = []

        for item in batch:
            raw_recipe_id = item.get("recipe_id")
            try:
                if isinstance(raw_recipe_id, bool):
                    raise ValueError("boolean recipe_id is invalid")
                recipe_id = int(raw_recipe_id)
                if isinstance(raw_recipe_id, float) and not raw_recipe_id.is_integer():
                    raise ValueError("fractional recipe_id is invalid")
                if recipe_id <= 0:
                    raise ValueError("recipe_id must be positive")
            except (TypeError, ValueError, OverflowError) as exc:
                failed_ids.append(
                    {"recipe_id": raw_recipe_id, "reason": f"missing_or_invalid_recipe_id: {exc}"}
                )
                continue

            recipe_key = f"recipe_{recipe_id}"
            if recipe_key in existing_ids:
                skipped_count += 1
                continue

            raw_text = item.get("document_text")
            if not isinstance(raw_text, str) or not raw_text.strip():
                failed_ids.append({"recipe_id": recipe_id, "reason": "empty_document_text"})
                continue
            text = raw_text.strip()

            ids.append(recipe_key)
            texts.append(text)
            metadata.append(
                {
                    "recipe_id": int(recipe_id),
                    "recipe_name": str(item.get("recipe_name", "")).strip(),
                    "source": "prepared_recipe_documents",
                }
            )

        if not ids:
            return

        try:
            embeddings = embed_texts(
                texts,
                model_name=model_name,
                model=model,
                batch_size=embedding_batch_size,
            )
            if embedding_dimension is None and embeddings:
                embedding_dimension = len(embeddings[0])
            if existing_dimension is not None and embedding_dimension != existing_dimension:
                raise ValueError(
                    f"Embedding dimension {embedding_dimension} does not match existing collection "
                    f"dimension {existing_dimension}"
                )

            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadata,
            )
            existing_ids.update(ids)
            indexed_count += len(ids)
        except Exception as exc:  # pragma: no cover - defensive path for runtime errors
            failed_ids.extend(
                {"recipe_id": item_id.removeprefix("recipe_"), "reason": f"batch_error: {exc}"}
                for item_id in ids
            )
            if log_progress:
                failed_id_list = ", ".join(item_id.removeprefix("recipe_") for item_id in ids)
                print(f"Batch {start}-{start + len(batch)} failed for recipe IDs [{failed_id_list}]: {exc}")

    pending_batch: list[dict[str, Any]] = []
    source_path = Path(documents_path)
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if sample_size is not None and processed_count >= sample_size:
                break
            processed_count += 1
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("JSONL record must be an object")
            except (json.JSONDecodeError, ValueError) as exc:
                failed_ids.append(
                    {"recipe_id": None, "line_number": line_number, "reason": f"invalid_json: {exc}"}
                )
                item = {}
            if item:
                pending_batch.append(item)
            if len(pending_batch) >= batch_size:
                process_batch(pending_batch, processed_count - len(pending_batch))
                pending_batch = []
            if log_progress and processed_count >= next_progress:
                elapsed = time.perf_counter() - started_at
                print(
                    f"Processed {processed_count} source documents; "
                    f"indexed {indexed_count}, skipped {skipped_count}, "
                    f"failed {len(failed_ids)}, elapsed {elapsed:.1f}s"
                )
                save_progress("running")
                next_progress = processed_count + progress_interval

    if pending_batch:
        process_batch(pending_batch, processed_count - len(pending_batch))

    report = {
        "db_path": str(Path(db_path)),
        "collection_name": collection_name,
        "model_name": model_name,
        "sample_size": sample_size if sample_size is not None else processed_count,
        "source_record_count": processed_count,
        "document_count": collection.count(),
        "indexed_count": indexed_count,
        "skipped_count": skipped_count,
        "failed_records": len(failed_ids),
        "failed_ids": failed_ids,
        "embedding_dimension": embedding_dimension,
        "batch_size": batch_size,
        "embedding_batch_size": embedding_batch_size,
        "reset_collection": reset_collection,
        "started_at_epoch": started_at_epoch,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }

    if collection_name == "recipe_documents_full":
        report_name = "full_index_report.json"
    elif collection_name == "recipe_documents_sample":
        report_name = "sample_index_report.json"
    else:
        report_name = "index_report.json"
    report_path = Path(db_path) / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_progress("completed")
    return report


def verify_persistence(db_path: str | Path, collection_name: str = DEFAULT_COLLECTION_NAME) -> dict[str, Any]:
    """Verify that the ChromaDB collection persists and reopens correctly."""

    client = _collection_client(db_path)
    collection = client.get_collection(name=collection_name)
    count = collection.count()
    return {
        "db_path": str(Path(db_path)),
        "collection_name": collection_name,
        "document_count": count,
        "persisted": count >= 0,
    }


def query_recipe_collection(
    query_text: str,
    db_path: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    n_results: int = 5,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Query the collection for the nearest recipe matches."""

    model = load_embedding_model(model_name=model_name)
    query_embedding = embed_texts([query_text], model_name=model_name, model=model, batch_size=1)[0]
    client = _collection_client(db_path)
    collection = client.get_collection(name=collection_name)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

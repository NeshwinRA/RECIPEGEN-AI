"""Compare ONNX Runtime embeddings with read-only ChromaDB references."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_PATH = "onnx/model.onnx"
MODEL_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
TOKENIZER_PATH = "tokenizer.json"
TOKENIZER_GIT_BLOB_SHA1 = "cb202bfe2e3c98645018a6d12f182a434c9d3e02"
MAX_SEQUENCE_LENGTH = 256

# Set before evaluating results: cosine >= .9999 and max coordinate error <= .01.
MIN_COSINE_SIMILARITY = 0.9999
MAX_ABSOLUTE_DIFFERENCE = 0.01
NORM_TOLERANCE = 1e-5


def _download_verified(
    relative_path: str,
    destination: Path,
    expected_hash: str | None,
    algorithm: str,
) -> str:
    """Download one pinned model file and verify its published digest."""

    url = (
        f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/"
        f"{MODEL_REVISION}/{relative_path}?download=true"
    )
    digest = hashlib.new(algorithm)
    request = urllib.request.Request(url, headers={"User-Agent": "RecipeGenAI-ONNX-POC/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)

    actual_hash = digest.hexdigest()
    if expected_hash is not None and actual_hash.lower() != expected_hash.lower():
        raise ValueError(
            f"Integrity check failed for {relative_path}: expected {expected_hash}, got {actual_hash}"
        )
    return actual_hash


def _tokenizer_blob_sha1(data: bytes) -> str:
    """Compute the Git blob SHA-1 used by the Hugging Face repository listing."""

    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _make_embeddings(
    tokenizer: Tokenizer,
    session: ort.InferenceSession,
    texts: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Tokenize and embed texts with mask-aware mean pooling and L2 normalization."""

    encodings = tokenizer.encode_batch(texts, add_special_tokens=True)
    max_length = max(len(encoding.ids) for encoding in encodings)
    input_ids = np.zeros((len(encodings), max_length), dtype=np.int64)
    attention_mask = np.zeros_like(input_ids)
    token_type_ids = np.zeros_like(input_ids)

    for index, encoding in enumerate(encodings):
        length = len(encoding.ids)
        input_ids[index, :length] = encoding.ids
        attention_mask[index, :length] = encoding.attention_mask
        token_type_ids[index, :length] = encoding.type_ids

    available_inputs = {item.name for item in session.get_inputs()}
    feed: dict[str, np.ndarray] = {}
    input_arrays = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    for name in available_inputs:
        if name not in input_arrays:
            raise ValueError(f"Unsupported required ONNX graph input: {name}")
        feed[name] = input_arrays[name]

    output_names = [item.name for item in session.get_outputs()]
    outputs = session.run(output_names, feed)
    token_embeddings = next(
        (value for value in outputs if value.ndim == 3 and value.shape[:2] == input_ids.shape),
        None,
    )
    if token_embeddings is None:
        raise ValueError(
            "Could not find a rank-3 token embedding output matching the input batch and sequence dimensions"
        )

    expanded_mask = attention_mask[..., np.newaxis].astype(token_embeddings.dtype)
    summed_embeddings = np.sum(token_embeddings * expanded_mask, axis=1)
    token_counts = np.maximum(np.sum(expanded_mask, axis=1), 1e-9)
    mean_embeddings = summed_embeddings / token_counts
    norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("ONNX model produced a zero-length sentence embedding")
    normalized = mean_embeddings / norms

    details = {
        "input_names": sorted(available_inputs),
        "output_names": output_names,
        "batch_shape": list(input_ids.shape),
        "sequence_lengths": [int(sum(encoding.attention_mask)) for encoding in encodings],
        "embedding_dimension": int(normalized.shape[1]),
    }
    return normalized.astype(np.float32), details


def run_poc(
    db_path: Path,
    collection_name: str = "recipe_documents_sample",
    sample_size: int = 5,
) -> dict[str, Any]:
    """Run an isolated ONNX comparison without mutating the ChromaDB database."""

    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=collection_name)
    baseline_count = collection.count()
    available_reference = collection.get(
        limit=baseline_count,
        include=["documents", "embeddings", "metadatas"],
    )
    if not available_reference["ids"]:
        raise ValueError(f"Reference collection '{collection_name}' contains no records")
    if sample_size > len(available_reference["ids"]):
        raise ValueError(
            f"Requested {sample_size} references but collection has only {len(available_reference['ids'])}"
        )

    with tempfile.TemporaryDirectory(prefix="recipegen-onnx-poc-") as temp_dir:
        temp_root = Path(temp_dir)
        model_file = temp_root / "model.onnx"
        tokenizer_file = temp_root / "tokenizer.json"
        model_digest = _download_verified(MODEL_PATH, model_file, MODEL_SHA256, "sha256")

        _download_verified(
            TOKENIZER_PATH,
            tokenizer_file,
            None,
            "sha1",
        )
        if _tokenizer_blob_sha1(tokenizer_file.read_bytes()) != TOKENIZER_GIT_BLOB_SHA1:
            raise ValueError("Tokenizer file does not match the pinned Hugging Face Git blob hash")

        tokenizer_config = json.loads(tokenizer_file.read_text(encoding="utf-8"))
        normalizer = tokenizer_config.get("normalizer") or {}
        bert_normalizer = normalizer.get(" BertNormalizer", normalizer)
        if bert_normalizer.get("lowercase") is not True:
            raise ValueError("Pinned tokenizer configuration is not explicitly uncased")

        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        tokenizer.enable_padding(
            direction="right",
            pad_id=0,
            pad_token="[PAD]",
            pad_type_id=0,
        )
        token_lengths = [
            len(tokenizer.encode(text, add_special_tokens=True).ids)
            for text in available_reference["documents"]
        ]
        by_length = sorted(range(len(token_lengths)), key=lambda index: token_lengths[index])
        target_positions = np.linspace(
            0,
            len(by_length) - 1,
            num=min(sample_size, len(by_length)),
            dtype=int,
        )
        selected_indices = {by_length[int(position)] for position in target_positions}
        for index in range(len(token_lengths)):
            if len(selected_indices) >= sample_size:
                break
            selected_indices.add(index)
        selected_indices = sorted(selected_indices)
        reference = {
            field: [available_reference[field][index] for index in selected_indices]
            for field in ("ids", "documents", "embeddings", "metadatas")
        }

        session = ort.InferenceSession(
            str(model_file),
            providers=["CPUExecutionProvider"],
        )
        unexpected_modules = {"transformers", "sentence_transformers"} & sys.modules.keys()
        if unexpected_modules:
            raise RuntimeError(
                f"Proof-of-concept must not import Transformers or Sentence Transformers: "
                f"{sorted(unexpected_modules)}"
            )
        providers = session.get_providers()
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError(f"Expected CPUExecutionProvider only, got {providers}")

        graph_inputs = [
            {"name": item.name, "shape": [str(axis) for axis in item.shape], "type": item.type}
            for item in session.get_inputs()
        ]
        graph_outputs = [
            {"name": item.name, "shape": [str(axis) for axis in item.shape], "type": item.type}
            for item in session.get_outputs()
        ]

        documents = reference["documents"]
        generated, inference_details = _make_embeddings(tokenizer, session, documents)
        reference_vectors = np.asarray(reference["embeddings"], dtype=np.float32)
        if generated.shape != reference_vectors.shape:
            raise ValueError(
                f"Embedding shape mismatch: ONNX={generated.shape}, reference={reference_vectors.shape}"
            )

        comparisons: list[dict[str, Any]] = []
        for index, (generated_vector, reference_vector) in enumerate(
            zip(generated, reference_vectors, strict=True)
        ):
            denominator = float(np.linalg.norm(generated_vector) * np.linalg.norm(reference_vector))
            if denominator == 0:
                raise ValueError("Cannot compare a zero-length reference or generated vector")
            cosine = float(np.clip(np.dot(generated_vector, reference_vector) / denominator, -1.0, 1.0))
            differences = np.abs(generated_vector - reference_vector)
            generated_norm = float(np.linalg.norm(generated_vector))
            reference_norm = float(np.linalg.norm(reference_vector))
            comparisons.append(
                {
                    "recipe_id": reference["metadatas"][index]["recipe_id"],
                    "dimension": int(generated_vector.shape[0]),
                    "onnx_norm": generated_norm,
                    "reference_norm": reference_norm,
                    "cosine_similarity": cosine,
                    "mean_absolute_difference": float(np.mean(differences)),
                    "max_absolute_difference": float(np.max(differences)),
                    "passes_parity_threshold": (
                        cosine >= MIN_COSINE_SIMILARITY
                        and float(np.max(differences)) <= MAX_ABSOLUTE_DIFFERENCE
                        and abs(generated_norm - 1.0) <= NORM_TOLERANCE
                        and abs(reference_norm - 1.0) <= NORM_TOLERANCE
                    ),
                }
            )

        padding_probe_text = "quick rice recipe"
        long_padding_companion = " ".join(["context"] * MAX_SEQUENCE_LENGTH)
        probe_alone, _ = _make_embeddings(tokenizer, session, [padding_probe_text])
        probe_batched, padding_details = _make_embeddings(
            tokenizer,
            session,
            [padding_probe_text, long_padding_companion],
        )
        padding_probe_cosine = float(
            np.clip(
                np.dot(probe_alone[0], probe_batched[0])
                / (np.linalg.norm(probe_alone[0]) * np.linalg.norm(probe_batched[0])),
                -1.0,
                1.0,
            )
        )
        padding_probe_max_difference = float(np.max(np.abs(probe_alone[0] - probe_batched[0])))
        attention_mask_padding_check = {
            "description": "Same short text embedded alone and beside a 256-token input",
            "batch_shape": padding_details["batch_shape"],
            "sequence_lengths": padding_details["sequence_lengths"],
            "cosine_similarity": padding_probe_cosine,
            "max_absolute_difference": padding_probe_max_difference,
            "passes": (
                padding_probe_cosine >= MIN_COSINE_SIMILARITY
                and padding_probe_max_difference <= MAX_ABSOLUTE_DIFFERENCE
            ),
        }

        final_count = collection.count()
        if final_count != baseline_count:
            raise RuntimeError(
                f"Read-only check detected collection count change: {baseline_count} -> {final_count}"
            )

        return {
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "model_path": MODEL_PATH,
            "model_sha256": model_digest,
            "tokenizer_path": TOKENIZER_PATH,
            "tokenizer_git_blob_sha1": TOKENIZER_GIT_BLOB_SHA1,
            "runtime": {"onnxruntime": ort.__version__, "providers": providers},
            "tokenizer": {
                "backend": "tokenizers",
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "add_special_tokens": True,
                "lowercase": bert_normalizer.get("lowercase"),
                "padding_side": "right",
            },
            "pooling": "attention-mask-aware mean pooling followed by L2 normalization",
            "comparison_thresholds": {
                "minimum_cosine_similarity": MIN_COSINE_SIMILARITY,
                "maximum_coordinate_absolute_difference": MAX_ABSOLUTE_DIFFERENCE,
                "unit_norm_absolute_tolerance": NORM_TOLERANCE,
            },
            "onnx_graph": {"inputs": graph_inputs, "outputs": graph_outputs},
            "inference": inference_details,
            "reference_collection": collection_name,
            "reference_count_before": baseline_count,
            "reference_count_after": final_count,
            "records_compared": len(comparisons),
            "attention_mask_padding_check": attention_mask_padding_check,
            "all_records_pass": (
                all(row["passes_parity_threshold"] for row in comparisons)
                and attention_mask_padding_check["passes"]
            ),
            "comparisons": comparisons,
            "temporary_files_cleaned": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=ROOT / "data" / "vector_db")
    parser.add_argument("--collection", default="recipe_documents_sample")
    parser.add_argument("--sample-size", type=int, default=5)
    args = parser.parse_args()

    report = run_poc(args.db_path, args.collection, args.sample_size)
    print(json.dumps(report, indent=2))
    if not report["all_records_pass"]:
        raise SystemExit("ONNX parity comparison did not meet the predeclared thresholds")


if __name__ == "__main__":
    main()

"""Index the full prepared recipe document set into the persistent ChromaDB collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipegen.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    MODEL_REVISION,
    index_recipe_documents,
    load_embedding_model,
    verify_persistence,
)


def main() -> None:
    """Build the full recipe document index in batches and verify persistence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=10000)
    args = parser.parse_args()

    documents_path = ROOT / "data" / "processed" / "recipe_retrieval_documents.jsonl"
    db_path = ROOT / "data" / "vector_db"
    collection_name = "recipe_documents_full"

    report = index_recipe_documents(
        documents_path=documents_path,
        db_path=db_path,
        collection_name=collection_name,
        sample_size=None,
        batch_size=args.batch_size,
        reset_collection=False,
        progress_interval=args.progress_interval,
        embedding_batch_size=args.embedding_batch_size,
        progress_report_path=db_path / "full_index_progress.json",
    )
    persistence = verify_persistence(db_path, collection_name)
    model = load_embedding_model(DEFAULT_EMBEDDING_MODEL, device="cpu")
    report["persistence_verification"] = persistence
    report["model_revision"] = MODEL_REVISION
    report["embedding_dimension"] = EMBEDDING_DIMENSION
    report["expected_model"] = DEFAULT_EMBEDDING_MODEL
    report["runtime"] = f"onnxruntime {ort.__version__}"
    report["execution_providers"] = model.session.get_providers()
    report_path = Path(db_path) / "full_index_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with documents_path.open("r", encoding="utf-8") as handle:
        source_count = sum(1 for line in handle if line.strip())
    print(
        f"Source docs: {source_count} | Indexed: {report['indexed_count']} | "
        f"Skipped: {report['skipped_count']} | Failed: {report['failed_records']} | "
        f"Elapsed: {report['elapsed_seconds']:.1f}s | Collection: {collection_name} | "
        f"Persisted: {persistence['persisted']}"
    )


if __name__ == "__main__":
    main()

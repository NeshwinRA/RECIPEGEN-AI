"""Create a small, verified sample vector index for recipe retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipegen.embeddings import DEFAULT_COLLECTION_NAME, index_recipe_documents, verify_persistence


def main() -> None:
    """Build a sample ChromaDB index for local validation before full indexing."""

    documents_path = ROOT / "data" / "processed" / "recipe_retrieval_documents.jsonl"
    db_path = ROOT / "data" / "vector_db"
    sample_size = 200
    collection_name = "recipe_documents_sample"

    report = index_recipe_documents(
        documents_path=documents_path,
        db_path=db_path,
        collection_name=collection_name,
        sample_size=sample_size,
        batch_size=64,
        reset_collection=False,
    )
    persistence = verify_persistence(db_path, collection_name)

    print(
        f"Sample size: {sample_size} | Embedding dimension: {report['embedding_dimension']} | "
        f"Database: {db_path} | Collection: {collection_name} | Stored docs: {persistence['document_count']}"
    )


if __name__ == "__main__":
    main()

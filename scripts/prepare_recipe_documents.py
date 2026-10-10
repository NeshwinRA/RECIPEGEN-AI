"""Prepare recipe documents for future retrieval and embedding modules."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipegen.retrieval import prepare_recipe_documents


def main() -> None:
    """Generate stable retrieval-ready documents and metadata from the processed dataset."""

    input_path = ROOT / "data" / "processed" / "foodcom_recipes_processed.csv"
    document_output_path = ROOT / "data" / "processed" / "recipe_retrieval_documents.jsonl"
    metadata_output_path = ROOT / "data" / "processed" / "recipe_retrieval_metadata.csv"
    report_path = ROOT / "data" / "processed" / "recipe_retrieval_report.json"

    report = prepare_recipe_documents(
        input_path=input_path,
        document_output_path=document_output_path,
        metadata_output_path=metadata_output_path,
        report_path=report_path,
    )
    print(
        f"Input: {report['input_record_count']} | Documents: {report['output_document_count']} | "
        f"Excluded: {report['excluded_document_count']}"
    )


if __name__ == "__main__":
    main()

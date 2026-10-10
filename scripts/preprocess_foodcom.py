"""Preprocess the raw Food.com CSV into the project schema."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipegen.dataset_preprocessing import process_raw_recipe_csv


def main() -> None:
    """Run the raw-to-processed recipe preprocessing pipeline."""

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "data" / "raw" / "RAW_recipes.csv"
    output_path = repo_root / "data" / "processed" / "foodcom_recipes_processed.csv"
    report_path = repo_root / "data" / "processed" / "foodcom_preprocessing_report.json"

    report = process_raw_recipe_csv(input_path=input_path, output_path=output_path, report_path=report_path)
    print(
        f"Input: {report['input_record_count']} | Output: {report['output_record_count']} | "
        f"Excluded: {report['excluded_record_count']}"
    )


if __name__ == "__main__":
    main()

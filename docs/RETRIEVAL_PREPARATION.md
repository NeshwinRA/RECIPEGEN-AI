# Recipe retrieval preparation

This stage prepares the processed Food.com dataset for the future embedding and retrieval pipeline without modifying either the raw source or the F06 processed dataset.

## Output files

The preparation step writes three artifacts under `data/processed/`:

- `recipe_retrieval_documents.jsonl`: one JSON object per valid recipe. Each line contains:
  - `recipe_id`: stable recipe identifier
  - `recipe_name`: normalized recipe name
  - `document_text`: a deterministic, searchable text string combining the recipe name, ingredients, instructions, available time, and tags
- `recipe_retrieval_metadata.csv`: a separate metadata table with `recipe_id`, `recipe_name`, `total_time_min`, `ingredient_count`, `step_count`, `tag_count`, and `tags`
- `recipe_retrieval_report.json`: counts, excluded-record reasons, and validation statistics

## Why this split matters

The future embedding and retrieval modules should operate on a compact text representation that is easy to vectorize while keeping the metadata separate for filtering and display. This avoids duplicate copies of large recipe data, preserves unknown values as `null`, and keeps the pipeline deterministic.

## Stability and validation rules

- The same source recipe always produces the same `document_text` because the content is assembled in a fixed order and normalized in place.
- Empty recipe names, empty ingredient lists, and empty step lists are rejected with a reason recorded in the report.
- Missing total cooking time or tags is preserved as unknown rather than inferred.
- This stage does not generate embeddings, call Gemini, or initialize ChromaDB.

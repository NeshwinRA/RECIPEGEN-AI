# F08 sample embeddings and ChromaDB index

This stage creates a small, repeatable vector index from the prepared retrieval documents without indexing the entire dataset yet.

## Target configuration

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (ONNX Runtime backend)
- Model revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Runtime: `onnxruntime` with `CPUExecutionProvider`
- Persistent storage: `data/vector_db/`
- Sample collection: `recipe_documents_sample`
- Initial sample size: 200 documents

## Purpose

The sample index verifies that:

- the local embedding model can load successfully,
- prepared recipe documents are ingestible,
- metadata and recipe IDs survive the vector-store step,
- the database can be reopened and queried repeatedly,
- batch indexing and idempotent reruns behave correctly before a larger run is approved.

## Notes

- This is intentionally a sample-only index and does not embed the full 231,636-document dataset.
- The full dataset can be indexed later in batches after the sample is approved.
- The generated report is stored under `data/vector_db/index_report.json`.
- The production encoder verifies the pinned ONNX model and tokenizer hashes
  and uses the standalone `tokenizers` package. It does not import
  Sentence Transformers, Transformers, or scikit-learn.
- The pinned model files are cached by `huggingface_hub` outside the project;
  the ONNX session and tokenizer are cached once per process. Embedding calls
  are split into bounded batches, defaulting to 32 texts per ONNX inference.
- `recipegen.embeddings.embed_texts(...)` retains the existing interface and
  refuses a different model name rather than silently substituting one.
- Indexing validates existing vector dimensions before adding vectors, streams
  JSONL source records in bounded batches, and skips recipe IDs already in the
  target collection. The protected full and sample collection names cannot be
  reset through the indexing helper.
- Full indexing must still be explicitly approved and resumes into
  `recipe_documents_full` with `reset_collection=False`.

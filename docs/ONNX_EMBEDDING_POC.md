# Isolated ONNX embedding proof of concept

## Safety boundary

Run `scripts/poc_onnx_embeddings.py` to compare ONNX-generated embeddings
against vectors read from the existing `recipe_documents_sample` collection.
The script uses ChromaDB only for collection lookup and `get`; it makes no
collection writes and checks that the count is unchanged. It does not import
Transformers or Sentence Transformers, call Gemini, or invoke full indexing.
The ONNX model and tokenizer are downloaded to a temporary directory and
removed automatically after execution.

## Model and embedding path

- Repository: `sentence-transformers/all-MiniLM-L6-v2`
- Pinned repository revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- ONNX file: `onnx/model.onnx` (unquantized; SHA-256 checked against the
  repository's published LFS SHA-256)
- Tokenizer: repository `tokenizer.json`, verified against its published Git
  blob object ID
- Maximum sequence length: 256 wordpieces; special tokens are added and inputs
  are right-padded with `[PAD]`
- Pooling: mean of token embeddings weighted by the attention mask, followed
  by L2 normalization
- Execution provider: `CPUExecutionProvider` only

The script inspects and reports actual ONNX graph input and output names and
shapes before inference. It feeds the graph inputs that are present among
`input_ids`, `attention_mask`, and `token_type_ids`, and requires a rank-3
token-state output for pooling.

## Predeclared parity criteria

Before interpreting comparison results, each generated/reference pair must
meet all criteria:

- cosine similarity at least `0.9999`
- maximum absolute coordinate difference no greater than `0.01`
- each vector's L2 norm within `1e-5` of `1.0`
- matching vector dimensions

The run reports every recipe ID's vector dimensions, norms, cosine similarity,
mean absolute difference, maximum absolute difference, and threshold result.
Failure is surfaced as an unsuccessful run rather than silently accepted.

## Initial proof-of-concept result

The five-record CPU comparison passed all predeclared thresholds:

- All vectors were 384-dimensional and had L2 norms approximately `1.0`.
- Cosine similarity was `1.0` for each pair after clamping floating-point
  roundoff to its valid range.
- Maximum coordinate-wise absolute difference was approximately
  `1.83e-7`; mean absolute differences were between `2.98e-8` and `4.75e-8`.
- ONNX Runtime `1.31.0` executed with `CPUExecutionProvider` only.
- The read-only sample collection stayed at 200 records. The full collection
  was not opened for modification.
- The selected sample recipes all reached the 256-wordpiece truncation limit.
  A separate synthetic short-text check is therefore included: its embedding
  is compared alone and when right-padded beside a 256-token companion, to
  verify attention-mask padding invariance.

This validates numerical parity on these five sample records. The production
encoder now uses the same pinned ONNX model and validated tokenization,
pooling, and normalization behavior. The comparison does not authorize
resuming full indexing; that remains gated on separate approval.

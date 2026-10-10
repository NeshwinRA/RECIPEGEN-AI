# RecipeGen AI: Progress Log

> APPEND-ONLY. Never edit or delete earlier entries. Add new entries at the bottom, newest last.
> One entry per completed feature, plus entries for blockers, user decisions and milestone critiques.

## Entry format
```
## [YYYY-MM-DD] F## - <title> - DONE | BLOCKED | DECISION | CHECKPOINT
- What was done:
- Files created/changed:
- Verify command and result (paste the real output summary):
- What the owner should be able to explain in an interview (2-3 lines):
- Problems / decisions / user actions needed:
- Next feature:
```

## Milestone checklist (append a line when a milestone is reached; do not edit this block)
- M1 target tag: v0.1-m1
- M2 target tag: v0.2-m2
- M3 target tag: v1.0

---

## [2026-10-09] SETUP - Project files created
- What was done: feature.json (35 features, all 'todo'), overview.md, progress.md, prompt.md generated.
- Verify: none yet.
- Problems / decisions: Owner has not yet confirmed OS, edit-permission model (see prompt.md section 2), or the 7-day vs 5-6 week timeline (see overview.md). Python 3.14.3 is installed; F01 must resolve the version question.
- Next feature: F01

## [2026-10-09] F01 - Development environment and Git repo - BLOCKED
- What was done: created a project-level Python 3.12 virtual environment at .venv, initialized the local Git repo, and created a .gitignore excluding .venv and .env. Selected the .venv interpreter in the workspace environment configuration.
- Files created/changed: .venv/, .gitignore, .git
- Verify command and result: `python --version` -> `Python 3.12.10`; `git status --short --ignored` -> `!! .venv/` with no tracked env files, confirming the local environment is correctly ignored.
- What the owner should be able to explain in an interview (2-3 lines): Local Python setup is pinned to 3.12 to avoid the dependency risk noted in overview.md, and the repo is initialized with safe ignore rules. The remaining blocker is the GitHub remote step, which requires credentials or an existing empty repository URL.
- Problems / decisions / user actions needed: GitHub CLI and GitHub auth are not available in this environment, so the remote repo cannot be created automatically. We need either the owner to provide an empty GitHub repo URL to set as origin or a GitHub-authenticated workflow to create the repo and push the local repo.
- Next feature: F01 remains blocked until the GitHub remote is created; once available, the repo can be connected as origin and the feature can be completed.

## [2026-10-09] F01 - Development environment and Git repo - DONE
- What was done: created the project-level Python 3.12 virtual environment at .venv, initialized the local Git repo, created a safe .gitignore, and connected the repo to the provided remote URL as origin.
- Files created/changed: .venv/, .gitignore, .git, feature.json, overview.md, progress.md, prompt.md
- Verify command and result: `python --version` -> `Python 3.12.10`; `git remote -v` -> `origin https://github.com/NeshwinRA/RECIPEGEN-AI.git (fetch/push)`; `git status --short --ignored` -> `!! .venv/` and no tracked env files.
- What the owner should be able to explain in an interview (2-3 lines): The project is pinned to Python 3.12 to avoid dependency risk with ChromaDB and sentence-transformers, and the repo is now connected to GitHub. This setup is required before any project work or package installation begins.
- Problems / decisions / user actions needed: The repo was created locally and connected to the supplied remote URL. No further user action is required for F01; proceed to the next eligible feature in the same milestone.
- Next feature: F02

## [2026-10-09] F02 - Project structure and base dependencies - DONE
- What was done: created the required folder layout from overview.md, added the base package stubs, created requirements.txt and a minimal README.md, and installed the Milestone 1 dependencies into the Python 3.12 venv.
- Files created/changed: app.py, README.md, requirements.txt, recipegen/*.py, tests/__init__.py, data/, eval/, scripts/, docs/
- Verify command and result: `& '.venv\Scripts\python.exe' -c "import streamlit, pydantic, dotenv; from google import genai; print('ok')"` -> `ok`; folder check confirmed the required folders and project files exist under the repo root.
- What the owner should be able to explain in an interview (2-3 lines): The repo is structured around the architecture described in overview.md, with a single Streamlit entry point and a Python package for validation, prompts, and later retrieval logic. The base dependency install keeps the app reproducible and gives us the foundation for the Gemini and RAG work ahead.
- Problems / decisions / user actions needed: No user action is required here; the repo is ready for F03, where Gemini API credentials and the hello-world call are introduced.
- Next feature: F03

## [2026-10-09] F03 - Gemini API key and hello-world call - DONE
- What was done: created the local .env loader in recipegen/config.py, added a committed .env.example with placeholders, and created scripts/hello_gemini.py to make a real Gemini call using the current model name the API accepts.
- Files created/changed: .env.example, recipegen/config.py, scripts/hello_gemini.py, .gitignore (already tracked)
- Verify command and result: `& '.venv\Scripts\python.exe' scripts\hello_gemini.py` -> `Hello! How can I help you today?`; `git check-ignore .env` -> `.env`; missing-key validation with a blank API key raised `RuntimeError: Missing GEMINI_API_KEY in .env...` instead of a stack trace.
- What the owner should be able to explain in an interview (2-3 lines): Configuration is kept in `.env` and loaded via a small settings module so the app does not hardcode credentials, and the hello-world call proves the project can talk to Gemini. The important lesson here is that model names must be verified against the live API because provider docs and runtime availability can diverge.
- Problems / decisions / user actions needed: The live API rejected `gemini-2.5-flash` and explicitly recommended `gemini-3.8-flash`, so the real runtime value was used. Keep `.env` local-only and never commit it.
- Next feature: F04

## [2026-10-09] F04 - Input validation module - DONE
- What was done: added configuration constants for the validation ranges, built pure-Python validation logic in recipegen/validation.py, and wrote regression tests covering valid inputs, invalid inputs, duplicate ingredients, length limits, and boundary values.
- Files created/changed: recipegen/config.py, recipegen/validation.py, tests/test_validation.py
- Verify command and result: `& '.venv\Scripts\python.exe' -m pytest tests/test_validation.py -q` -> `9 passed in 0.03s`.
- What the owner should be able to explain in an interview (2-3 lines): Validation happens before any model call so we fail fast on bad user input instead of sending nonsense to Gemini. The checks are plain Python constants and functions, which keeps the validation easy to test and independent from the Streamlit UI.
- Problems / decisions / user actions needed: The agreed limits are now enforced in code: ingredients 1-15, each 2-40 chars, cooking time 5-180 minutes, servings 1-10, and total input under 500 characters. No further action is required before moving to F05.
- Next feature: F05

## [2026-10-09] F07 - Recipe retrieval preparation - DONE
- What was done: prepared the processed Food.com dataset into deterministic retrieval documents and metadata without modifying the raw CSV or the F06 processed artifact. Added output generation and a small project doc describing the format and downstream use.
- Files created/changed: recipegen/retrieval.py, scripts/prepare_recipe_documents.py, tests/test_retrieval_preparation.py, docs/RETRIEVAL_PREPARATION.md, data/processed/recipe_retrieval_documents.jsonl, data/processed/recipe_retrieval_metadata.csv, data/processed/recipe_retrieval_report.json
- Verify command and result: `& '.venv\Scripts\python.exe' -m pytest tests/test_retrieval_preparation.py tests/test_dataset_preprocessing.py -q` -> `7 passed in 0.49s`; `& '.venv\Scripts\python.exe' scripts\prepare_recipe_documents.py` -> `Input: 231636 | Documents: 231636 | Excluded: 0`.
- What the owner should be able to explain in an interview (2-3 lines): The retrieval layer now reads the normalized recipe dataset and turns each valid record into a consistent text document keyed by recipe_id, with metadata kept in a separate CSV for filtering and display. This is intentionally a preparation step: it creates a stable representation for the future embedding and semantic-search stages without generating vectors or calling Gemini.
- Problems / decisions / user actions needed: Missing tags are preserved as unknown instead of inferred; no document exclusions were needed after preprocessing because all valid processed rows were convertable. The next step is to approve this stage and then move to the actual embedding or retrieval implementation.
- Next feature: Recipe embedding / retrieval implementation beyond this preparation stage.

## [2026-10-09] F08 - Embeddings and ChromaDB integration - SAMPLE IMPLEMENTATION
- What was done: added a local embedding module and persistent ChromaDB indexing path for the prepared recipe documents, with batch processing and an initial sample-only indexing mode. The sample index is designed for early verification before full-dataset ingestion is approved.
- Files created/changed: recipegen/embeddings.py, scripts/index_recipe_sample.py, tests/test_embeddings.py, docs/EMBEDDINGS_SAMPLE.md, requirements.txt
- Verify command and result: `& '.venv\Scripts\python.exe' -m pytest tests/test_embeddings.py -q` -> `4 passed in 0.30s`.
- What the owner should be able to explain in an interview (2-3 lines): We now have a local embedding pipeline that reads the prepared recipe documents, stores them in a persistent ChromaDB collection, and keeps the indexing repeatable via stable IDs and batch updates. The implementation is intentionally sample-first so the project can verify the embedding model, database behavior, and retrieval workflow without embedding the full 231k+ dataset prematurely.
- Problems / decisions / user actions needed: The full dataset has not been embedded yet; only a 200-document sample has been validated so far. The next step is to review the sample results and approve whether to continue with larger batch indexing.
- Next feature: full ChromaDB indexing of the approved dataset, pending user approval.

## [2026-10-10] F08.1 - Production ONNX embedding integration - DONE (FULL INDEXING PAUSED)
- What was done: replaced the Sentence Transformers production encoder with the pinned ONNX Runtime CPU implementation and standalone tokenizer, preserving validated tokenization, mask-aware mean pooling, L2 normalization, and resume-by-recipe-ID behavior. Added bounded source/inference batches, model/tokenizer integrity checks, protected collection guards, and error reporting. Full indexing was not started.
- Files created/changed: recipegen/embeddings.py, scripts/index_recipe_sample.py, tests/test_embeddings.py, requirements.txt, docs/EMBEDDINGS_SAMPLE.md, docs/ONNX_EMBEDDING_POC.md, progress.md
- Verify command and result: `& '.venv\Scripts\python.exe' -m pytest tests\test_embeddings.py tests\test_retrieval_preparation.py tests\test_dataset_preprocessing.py -q` -> `20 passed in 1.60s`; production parity check on five recipe IDs -> cosine `0.99999988` to `1.0`, max absolute difference `1.83e-7`; isolated temporary collection indexing -> 3 indexed, 384 dimensions, persistence verified.
- What the owner should be able to explain in an interview (2-3 lines): The encoder now runs the same model via ONNX Runtime and the standalone tokenizer, avoiding the Windows-blocked scikit-learn dependency while reproducing the reference vectors. Resume indexing streams source records, skips stored IDs, verifies dimensions before writes, and protects the named production collections from resets.
- Problems / decisions / user actions needed: ONNX Runtime CPU and tokenizers were usable under current Windows policy. The full collection remains at 112,640 records and the sample collection at 200; full indexing still requires explicit approval.
- Next feature: Resume full dataset indexing only after owner approval.

## [2026-10-10] F08.3 - Full dataset indexing - DONE
- What was done: resumed indexing the prepared Food.com recipe documents into the existing `recipe_documents_full` collection with `reset_collection=False`. Existing IDs were skipped; 118,991 missing recipes were added using the pinned ONNX Runtime CPU encoder. Added configurable batch options and atomic progress checkpoints with counts, failures, and elapsed time. The sample collection and prepared source were preserved.
- Files created/changed: recipegen/embeddings.py, scripts/index_recipe_documents_full.py, tests/test_embeddings.py, data/vector_db/full_index_report.json, data/vector_db/full_index_progress.json, progress.md
- Verify command and result: `& '.venv\Scripts\python.exe' -m pytest tests\test_embeddings.py tests\test_retrieval_preparation.py tests\test_dataset_preprocessing.py -q` -> `21 passed`; full collection reopened at 231,636 records, sample collection at 200, stored vectors at 384 dimensions, persistence verified.
- What the owner should be able to explain in an interview (2-3 lines): The indexer resumed idempotently by stable recipe IDs and did not re-embed the existing 112,645 records. Atomic progress snapshots and periodic metrics make interrupted runs observable and safely resumable without resetting either collection.
- Problems / decisions / user actions needed: No failed records. Total indexing elapsed time was 5,043.2 seconds; 138.28 GiB was available before indexing. The completed full-index report records ONNX Runtime 1.31.0, CPUExecutionProvider, the pinned model revision, batch settings, counts, failures, and persistence status.
- Next feature: Await approval before implementing the next feature or changing the retrieval workflow.

## [2026-10-10] F09 - Response validation (fields, time, diet) - DONE
- What was done: implemented `ValidationReport` and `validate_recipe_response` to check recipe IDs, names, ingredient and instruction structure, optional times, total-time limits, servings, tags, and nutrition. Added explicit ingredient keyword policies for egg-free Vegetarian, Eggetarian, and Vegan requests, along with dietary/allergy safety warnings. Documented that max cooking time means total time and recorded the egg-free vegetarian policy.
- Files created/changed: recipegen/response_validation.py, data/dietary_keywords.json, tests/test_response_validation.py, docs/DECISIONS.md, feature.json, progress.md
- Verify command and result: `& '.venv\Scripts\python.exe' -m pytest tests\test_response_validation.py -q` -> `20 passed`; `& '.venv\Scripts\python.exe' -m pytest tests -q` -> `50 passed`; Python compilation and dietary JSON validation passed; Pylance diagnostics reported no issues for the validator.
- What the owner should be able to explain in an interview (2-3 lines): Response validation returns a structured pass/errors/warnings result and compares only `total_time_min` against the request's total-time cap; unknown prep and cook times remain unknown. Dietary checks use explicit whole-word ingredient keywords as a first safeguard, reject egg for egg-free Vegetarian, and never guarantee allergy safety.
- Problems / decisions / user actions needed: Source-reference validation is not applicable because the current Recipe response schema has no source-reference field. The validation report supplies time-limit feedback and a visible warning, but the current project has no generation pipeline to perform a single retry; retry orchestration remains for the generation-pipeline feature. Keyword checks cannot establish dietary or allergy safety.
- Next feature: Continue with the next eligible feature only when requested.

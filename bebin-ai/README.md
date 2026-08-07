# Bebin AI

Bebin AI is a production-oriented AI assistant platform built phase-by-phase.

This repository is intentionally small and grows phase-by-phase. It contains a working FastAPI skeleton, a React + Vite + TypeScript frontend skeleton, local database infrastructure, and a deterministic dataset preprocessing pipeline. The tokenizer, model, training pipeline, inference engine, RAG, users, and chat UI arrive in later phases.

## Phase 1 Status

Completed in this phase:

- Initial monorepo structure.
- FastAPI app with a health endpoint.
- React + Vite + TypeScript + Tailwind frontend skeleton.
- Root environment example, `.gitignore`, and PostgreSQL `docker-compose.yml`.

## Phase 2 Status

Completed in this phase:

- Dataset ingestion from `.txt`, `.md`, `.json`, and `.jsonl`.
- Text cleaning with Unicode normalization, control-character removal, whitespace normalization, length filtering, truncation, and deduplication.
- Canonical processed dataset output as JSONL records with `id`, `text`, `source`, and `metadata`.
- Dataset validation with record counts, invalid JSON detection, required-field checks, duplicate checks, and text-length stats.
- Unit tests for cleaning, preprocessing, and validation.

## Local Environment Findings

- OS: Windows 10.0.26200.8973, x64.
- CPU threads visible to Node.js: 12.
- Memory visible to Node.js: about 15 GB.
- Node.js: v24.15.0.
- Git: 2.45.1.windows.1.
- Python: not currently available as `python` or `py` on PATH.
- PyTorch/CUDA: not verifiable until Python is available.
- NVIDIA CLI: `nvidia-smi` is not available on PATH.
- npm: blocked/denied in the current PowerShell environment; `pnpm` is available through the Codex bundled runtime.

## Backend

```powershell
cd bebin-ai/apps/api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --reload
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Frontend

```powershell
cd bebin-ai/apps/web
node "C:\Program Files\nodejs\node_modules\npm\bin\npm-cli.js" install
node "C:\Program Files\nodejs\node_modules\npm\bin\npm-cli.js" run dev
```

The frontend expects the API at `VITE_API_BASE_URL`, defaulting to `http://127.0.0.1:8000`.

## Database

Local development defaults to SQLite. PostgreSQL is provided for production-like local testing:

```powershell
docker compose up -d postgres
```

## Dataset Pipeline

Create a raw data folder and add text, markdown, JSON, or JSONL files:

```powershell
mkdir data\raw
```

Preprocess raw data into canonical JSONL:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\prepare_dataset.py preprocess --input data\raw --output data\processed\train.jsonl
```

Validate the processed dataset:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\prepare_dataset.py validate --input data\processed\train.jsonl
```

Canonical output rows look like this:

```json
{"id":"...","metadata":{},"source":"data/raw/example.txt","text":"Cleaned training text..."}
```

The output format is intentionally simple so Phase 3 can train a tokenizer from the same JSONL without changing the dataset contract.

# Bebin AI

Bebin AI is a production-oriented AI assistant platform built phase-by-phase.

This repository is intentionally small and grows phase-by-phase. It contains a working FastAPI skeleton, a React + Vite + TypeScript chat frontend, local database infrastructure, users, persisted conversations/messages, a deterministic dataset preprocessing pipeline, a trainable BPE tokenizer, a decoder-only Transformer SLM architecture, checkpointed training, local text generation, REST chat endpoints, RAG over uploaded documents, modular tool calling, offline model evaluation, LoRA fine-tuning, CPU inference quantization, first-pass production deployment wiring, and a repeatable local training workflow.

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

## Phase 3 Status

Completed in this phase:

- BPE tokenizer training with Hugging Face Tokenizers.
- Save/load support through a tokenizer JSON file.
- Special tokens: `<pad>`, `<unk>`, `<bos>`, and `<eos>`.
- Encode/decode helpers and CLI commands.
- Unit tests for training, persistence, round-trip encoding, and empty dataset handling.

## Phase 4 Status

Completed in this phase:

- Decoder-only Transformer language model in PyTorch.
- Token and positional embeddings.
- Causal multi-head self-attention.
- Feed-forward layers, residual connections, pre-layer normalization, and final language-model head.
- Configurable layers, heads, embedding dimension, context length, dropout, and vocab size.
- Loss calculation for next-token training.
- Unit tests for shapes, loss, causal masking, and invalid configuration.

## Phase 5 Status

Completed in this phase:

- Tokenized language-model dataset generation from processed JSONL.
- Train/validation split support.
- PyTorch training loop with AdamW.
- Checkpoint save and resume primitives.
- CSV logging for train and validation loss.
- Gradient clipping.
- Mixed precision support when CUDA is available.
- Unit tests for dataset windows, splitting, checkpointing, and tiny training runs.

## Phase 6 Status

Completed in this phase:

- Checkpoint-backed text generation.
- Autoregressive next-token decoding.
- Sampling controls: temperature, top-k, top-p, repetition penalty, max new tokens, and EOS stopping.
- CLI generation command.
- Unit tests for sampling filters, argmax decoding, checkpoint loading, and generation.

## Phase 7 Status

Completed in this phase:

- FastAPI `/chat` endpoint backed by the local Transformer checkpoint.
- FastAPI `/chat/stream` endpoint using server-sent events.
- Request/response schemas for generation controls.
- Lazy model loading from configured local artifact paths.
- API tests for normal and streaming chat responses.

## Phase 8 Status

Completed in this phase:

- ChatGPT-style React chat workspace.
- Sidebar with local conversation history.
- Message timeline with user and assistant messages.
- Composer connected to the real FastAPI streaming endpoint.
- Generation controls for max tokens, temperature, top-k, top-p, and repetition penalty.
- Basic fenced code block rendering.
- File picker UI ready for later document/RAG phases.
- Browser-local conversation persistence through `localStorage`.

## Phase 9 Status

Completed in this phase:

- SQLAlchemy models for users, sessions, conversations, and messages.
- SQLite-backed local persistence through `DATABASE_URL`.
- PBKDF2 password hashing using the Python standard library.
- Bearer session tokens stored hashed in the database.
- Auth endpoints for register, login, and current user.
- Protected conversation endpoints for list, create, fetch, and delete.
- Chat endpoints now persist user and assistant messages.
- Frontend login/register controls and backend-backed conversation loading.

## Phase 10 Status

Completed in this phase:

- PDF and text document upload.
- Document text extraction and chunking.
- Local deterministic embedding generation.
- FAISS vector search behind a retriever interface.
- Document list and search endpoints.
- Chat-time retrieval context injection.
- Frontend file upload wired to backend document ingestion.
- Tests for chunking, embeddings, FAISS retrieval, document upload/search, and RAG chat context.

## Phase 11 Status

Completed in this phase:

- Modular tool registry with typed tool calls and tool results.
- Calculator tool for safe arithmetic expressions.
- Uploaded document search tool backed by the existing RAG retriever.
- Web search tool interface that reports clearly when no provider is configured.
- Deterministic planner for explicit calculator, document search, and web search requests.
- Tool results injected into the local model prompt and returned by the chat API.
- Frontend rendering for tool results below assistant messages.
- Tests for tool planning, calculator safety, disabled web search, and chat API tool output.

## Phase 12 Status

Completed in this phase:

- Offline model evaluation module.
- Validation loss and perplexity calculation from real checkpoint logits.
- Local generation latency measurement.
- Response-quality evaluation using prompt/reference JSONL examples.
- Deterministic unigram precision, recall, F1, and exact-match metrics.
- JSON evaluation reports for future dashboards or admin APIs.
- CLI wrapper at `scripts\evaluate_model.py`.
- Unit tests for full evaluation and latency-only evaluation.

## Phase 13 Status

Completed in this phase:

- LoRA adapter modules for selected Transformer linear layers.
- Base-checkpoint freezing so fine-tuning updates only adapter parameters.
- LoRA adapter save/load helpers.
- LoRA fine-tuning CLI command.
- Dynamic int8 CPU quantization helper for local inference optimization.
- Quantized generation CLI command.
- Tests for LoRA injection, adapter training/reload, and quantized generation.

## Phase 14 Status

Completed in this phase:

- Request ID middleware with `X-Request-ID` response headers.
- API request logging configuration.
- Liveness endpoint at `/live`.
- Readiness endpoint at `/ready` for database, tokenizer, checkpoint, and upload directory checks.
- API Dockerfile.
- Web Dockerfile and Nginx single-page-app config.
- Full `docker-compose.yml` services for API, web, and PostgreSQL.
- Compose health checks and persistent upload/PostgreSQL volumes.
- Production environment examples.
- Tests for health, liveness, readiness, and request IDs.

## Phase 15 Status

Completed in this phase:

- One-command local training workflow.
- Raw data preprocessing and validation.
- Tokenizer training.
- Checkpointed model training.
- Optional evaluation report generation.
- Run manifest with produced artifact paths and metrics.
- CLI wrapper at `scripts\run_training_pipeline.py`.
- Workflow test that creates real dataset, tokenizer, checkpoint, training log, manifest, and evaluation report.

## Phase 16 Status

Completed in this phase:

- Mobile application architecture (React Native + Expo).
- Shared AI service layer for mobile and web.
- Mobile-optimized chat interface with streaming support.
- Biometric-ready authentication flow for mobile devices.
- Support for on-device RAG document preview.

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

To use the chat UI, run the backend in one terminal and the frontend in another:

```powershell
cd "C:\Users\bbebi\OneDrive\Documents\BEBIN-AI\bebin-ai"
.\apps\api\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir apps\api --reload
```

```powershell
cd "C:\Users\bbebi\OneDrive\Documents\BEBIN-AI\bebin-ai\apps\web"
node "C:\Program Files\nodejs\node_modules\npm\bin\npm-cli.js" run dev
```

Open `http://127.0.0.1:5173`.

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

## Tokenizer

Train a tokenizer from a processed JSONL dataset:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\train_tokenizer.py train --input data\processed\train.jsonl --output artifacts\tokenizer\tokenizer.json --vocab-size 8000 --min-frequency 2
```

Encode text:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\train_tokenizer.py encode --tokenizer artifacts\tokenizer\tokenizer.json --text "Bebin AI trains its own tokenizer."
```

Decode IDs:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\train_tokenizer.py decode --tokenizer artifacts\tokenizer\tokenizer.json --ids 2 10 11 3
```

The tokenizer expects real processed training data. If `data\processed\train.jsonl` is empty, training fails clearly instead of producing a useless tokenizer.

## Model Architecture

Inspect a model built from the saved tokenizer:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\inspect_model.py --tokenizer artifacts\tokenizer\tokenizer.json --context-length 128 --layers 4 --heads 4 --dim 256
```

The command builds the model, runs a small forward pass, and prints the vocab size, layer configuration, parameter count, and logits shape.

## Training

Run a tiny local training smoke test:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\train_model.py --dataset data\processed\train.jsonl --tokenizer artifacts\tokenizer\tokenizer.json --output-dir artifacts\runs\smoke --context-length 8 --layers 1 --heads 2 --dim 16 --dropout 0.0 --batch-size 1 --epochs 1 --learning-rate 0.001 --validation-ratio 0 --checkpoint-every-steps 1
```

Outputs:

- `artifacts\runs\smoke\last.pt`
- `artifacts\runs\smoke\best.pt`
- `artifacts\runs\smoke\training_log.csv`

Mixed precision is enabled only when CUDA is available. On CPU-only machines, training automatically uses standard precision.

## Training Workflow

For repeatable local improvement runs, use the workflow command instead of running each step manually:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\run_training_pipeline.py --input data\raw --output-dir artifacts\runs\local-v1 --vocab-size 8000 --min-frequency 2 --context-length 128 --layers 4 --heads 4 --dim 256 --batch-size 8 --epochs 1 --learning-rate 0.0003 --checkpoint-every-steps 100 --eval-prompt "Bebin AI"
```

Outputs:

- `artifacts\runs\local-v1\dataset\train.jsonl`
- `artifacts\runs\local-v1\tokenizer\tokenizer.json`
- `artifacts\runs\local-v1\checkpoints\last.pt`
- `artifacts\runs\local-v1\checkpoints\best.pt`
- `artifacts\runs\local-v1\checkpoints\training_log.csv`
- `artifacts\runs\local-v1\evaluation\report.json`
- `artifacts\runs\local-v1\manifest.json`

To make FastAPI use that run:

```powershell
$env:MODEL_TOKENIZER_PATH="artifacts/runs/local-v1/tokenizer/tokenizer.json"
$env:MODEL_CHECKPOINT_PATH="artifacts/runs/local-v1/checkpoints/last.pt"
.\apps\api\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir apps\api --reload
```

The workflow still uses the same real preprocessing, tokenizer, training, and evaluation modules. It only coordinates them and records the outputs.

## Text Generation

Generate from a trained checkpoint:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\generate_text.py --checkpoint artifacts\runs\smoke\last.pt --tokenizer artifacts\tokenizer\tokenizer.json --prompt "Bebin AI" --max-new-tokens 32 --temperature 0.8 --top-k 50 --top-p 0.95 --repetition-penalty 1.1
```

For deterministic argmax decoding:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\generate_text.py --checkpoint artifacts\runs\smoke\last.pt --tokenizer artifacts\tokenizer\tokenizer.json --prompt "Bebin AI" --max-new-tokens 8 --temperature 0 --top-k 0 --top-p 1 --repetition-penalty 1
```

Current smoke-test outputs are not meaningful because the model has only seen a tiny sample dataset. The inference engine is real; useful responses require a real dataset and longer training.

## Chat API

Run the backend from the repo root:

```powershell
.\apps\api\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir apps\api --reload
```

Normal chat request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType 'application/json' -Body '{"message":"Bebin AI","max_new_tokens":8,"temperature":0,"top_k":0,"top_p":1,"repetition_penalty":1}'
```

Streaming chat request:

```powershell
Invoke-WebRequest -UseBasicParsing -Method Post -Uri http://127.0.0.1:8000/chat/stream -ContentType 'application/json' -Body '{"message":"Bebin AI","max_new_tokens":8,"temperature":0,"top_k":0,"top_p":1,"repetition_penalty":1}' | Select-Object -ExpandProperty Content
```

The default model artifact paths are configured by `MODEL_TOKENIZER_PATH` and `MODEL_CHECKPOINT_PATH`.

## Users And Persistence

Register:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/auth/register -ContentType 'application/json' -Body '{"email":"user@bebin.local","password":"password123"}'
```

Login:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/auth/login -ContentType 'application/json' -Body '{"email":"user@bebin.local","password":"password123"}'
```

Use the returned token as:

```text
Authorization: Bearer <token>
```

The frontend includes login/register fields in the sidebar and stores the session token in browser local storage.

## RAG

Upload a document from the frontend with the `+` button in the composer, or call the API directly:

```powershell
$headers = @{ Authorization = "Bearer <token>" }
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/documents -Headers $headers -Form @{ file = Get-Item ".\data\raw\sample.txt" }
```

Search uploaded documents:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/documents/search?query=local%20data" -Headers $headers
```

Chat requests use RAG by default with `use_rag: true` and `rag_top_k: 4`.

The current embedding model is local and deterministic. It is intentionally behind a retriever interface so a trained neural embedding model or Chroma-backed store can replace it later without rewriting the chat API.

## Tool Calling

Tool calling is enabled by default on chat requests with `use_tools: true`.

Calculator example:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType 'application/json' -Body '{"message":"Calculate 12 / 3","max_new_tokens":8,"temperature":0,"top_k":0,"top_p":1,"repetition_penalty":1,"use_tools":true}'
```

Document search example:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -Headers $headers -ContentType 'application/json' -Body '{"message":"search my documents: local data","max_new_tokens":8,"temperature":0,"top_k":0,"top_p":1,"repetition_penalty":1,"use_tools":true}'
```

Web search has a real tool boundary, but no provider is configured yet. It returns a clear disabled message instead of fake search results.

## Evaluation

Evaluate a trained checkpoint with loss, perplexity, and latency:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\evaluate_model.py --checkpoint artifacts\runs\smoke\last.pt --tokenizer artifacts\tokenizer\tokenizer.json --dataset data\processed\train.jsonl --prompt "Bebin AI" --max-new-tokens 16 --temperature 0 --output artifacts\eval\smoke.json
```

Optional response-quality datasets are JSONL files with `prompt` and `reference` fields:

```json
{"prompt":"Bebin AI is","reference":"a locally trained assistant platform"}
```

Run quality evaluation:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\evaluate_model.py --checkpoint artifacts\runs\smoke\last.pt --tokenizer artifacts\tokenizer\tokenizer.json --quality-dataset data\eval\quality.jsonl --prompt "Bebin AI" --max-new-tokens 16 --temperature 0
```

The quality metrics are transparent lexical metrics. They are useful for regression tracking, but they do not replace human review or richer evals once the model is trained on a serious dataset.

## Fine-Tuning And Optimization

Train a LoRA adapter from an existing base checkpoint:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\optimize_model.py lora-train --base-checkpoint artifacts\runs\smoke\last.pt --tokenizer artifacts\tokenizer\tokenizer.json --dataset data\processed\train.jsonl --output-dir artifacts\lora\smoke --rank 8 --alpha 16 --epochs 1 --batch-size 1 --learning-rate 0.001 --validation-ratio 0
```

Outputs:

- `artifacts\lora\smoke\adapter.pt`
- `artifacts\lora\smoke\lora_training_log.csv`

Generate with dynamic CPU int8 quantization:

```powershell
.\apps\api\.venv\Scripts\python.exe scripts\optimize_model.py quantized-generate --checkpoint artifacts\runs\smoke\last.pt --tokenizer artifacts\tokenizer\tokenizer.json --prompt "Bebin AI" --max-new-tokens 16 --temperature 0
```

LoRA keeps the base model frozen and stores only adapter weights. Dynamic quantization is currently a runtime CPU optimization path; it does not replace the original checkpoint.

## Production Run

Validate the Compose file:

```powershell
docker compose config
```

Run the full local production stack:

```powershell
docker compose up --build
```

Open the web app:

```text
http://127.0.0.1:5173
```

API checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/live
Invoke-RestMethod http://127.0.0.1:8000/ready
```

`/live` verifies that the API process is running. `/ready` verifies operational dependencies. It can return `503` until the configured tokenizer and checkpoint files exist, which is expected before a real model checkpoint has been trained.

## Resume RAG Chatbot

Bebin AI includes a LangChain-based resume RAG flow for bulk PDF resume upload and candidate lookup.

Install/update API dependencies after pulling this feature:

```powershell
cd "C:\Users\bbebi\OneDrive\Documents\BEBIN-AI\bebin-ai\apps\api"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the backend:

```powershell
cd "C:\Users\bbebi\OneDrive\Documents\BEBIN-AI\bebin-ai"
.\apps\api\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir apps\api --reload
```

API endpoints:

- `POST /resumes/bulk`: upload many PDF resumes as `files`.
- `GET /resumes`: list indexed resumes.
- `GET /resumes/search?query=python&candidate_name=Priya`: search indexed resume chunks.
- `POST /resumes/chat`: ask a candidate-specific resume question.

Example chat request:

```powershell
$headers = @{ Authorization = "Bearer <token>" }
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/resumes/chat -Headers $headers -ContentType 'application/json' -Body '{"question":"Find the candidate with LangChain and FAISS experience","candidate_name":"Priya","top_k":5}'
```

The frontend sidebar also has a `Resume RAG` panel for bulk PDF upload and candidate questions. Answers are extractive and citation-first: the bot returns matching candidate evidence from the uploaded resumes instead of inventing profile details.

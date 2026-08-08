# Bebin AI

Bebin AI is a production-oriented AI assistant platform built phase-by-phase.

This repository is intentionally small and grows phase-by-phase. It contains a working FastAPI skeleton, a React + Vite + TypeScript chat frontend, local database infrastructure, a deterministic dataset preprocessing pipeline, a trainable BPE tokenizer, a decoder-only Transformer SLM architecture, checkpointed training, local text generation, and REST chat endpoints. RAG, users, and persistence arrive in later phases.

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

# Forge — Completed Features

Reference document for context recovery. Everything listed here is implemented, tested, and working.

## Python CLI & SDK (23 Commands)

### Data Pipeline
- **ingest** — Load data from local paths/S3, resumable checkpoints, incremental mode
- **versions** — List dataset snapshots with metadata (counts, timestamps, lineage)
- **filter** — Create metadata-filtered snapshots (language, quality score, source prefix)
- **export-training** — Export snapshot to sharded training files with optional metadata

### Training Methods (9)
- **train** — Standard supervised training with configurable architecture
- **sft** — Supervised fine-tuning with prompt/response masking
- **dpo-train** — Direct preference optimization with reference model, configurable beta
- **rlhf-train** — RLHF with optional reward model training + PPO policy optimization (GAE, clipping, entropy, KL penalty)
- **lora-train** — LoRA adapter fine-tuning (base frozen, adapter-only optimizer)
- **lora-merge** — Merge trained LoRA adapters back into base model
- **distill** — Knowledge distillation (teacher frozen, KL + CE loss, temperature scaling)
- **domain-adapt** — Continued pretraining with drift detection on reference data
- **distributed-train** — Multi-GPU DDP training via torchrun

### Distributed Strategies
- DDP (DistributedDataParallel)
- FSDP (Fully Sharded Data Parallel)
- DeepSpeed integration
- TPU/XLA support
- Multi-node launcher and setup

### Experimentation
- **sweep** — Hyperparameter grid/random search across training configs
- **compare** — Side-by-side comparison of two training run results
- **replay** — Deterministic replay from reproducibility bundle
- **run-spec** — Declarative pipeline execution from YAML
- **benchmark** — Perplexity and latency profiling on test data

### Model & System
- **chat** — Text generation from trained/ONNX models with streaming
- **export-spec** — Export run configuration as reproducible spec
- **verify** — Platform verification (quick/full modes) with artifact preservation
- **hardware-profile** — Detect GPU/TPU capabilities, recommend precision/batch defaults
- **compute** — Cost estimation for training runs

### Safety & Deployment
- **safety-eval** — Toxicity/jailbreak/red-team scoring
- **safety-gate** — Pre-deployment threshold gate
- **deploy** — Package model + config + tokenizer with manifest + checksum
- **model** — Registry operations (tag, rollback, diff versions)
- **server** — Collaboration server for multi-user runs + comments

## Core Infrastructure

### Training Engine
- Full training loop with batching, gradient accumulation, checkpointing
- Custom loss functions via hooks
- Precision control (FP32, FP16, BF16, mixed precision with safe fallback)
- Validation splits with early stopping
- Optimizer choices: AdamW, SGD with momentum
- Schedulers: constant, step, cosine, exponential
- Reproducibility: full bundle saved (config hash, environment snapshot, seed)
- Checkpoint management: periodic save, best-model tracking, resume, retention policy
- Extension/hooks interface (run/epoch/batch/checkpoint hooks, custom loss)

### Data Pipeline
- Input readers: local file/directory, S3 (boto3), plaintext/JSONL/CSV
- Transforms: exact deduplication, language detection, perplexity quality scoring
- Incremental processing (diff against latest, only process new/changed)
- Resumable checkpoints per pipeline stage
- Lance binary format storage with immutable snapshots
- Catalog metadata (version, counts, parents, recipe)

### Safety Suite
- Toxicity scorer
- Safety gate (threshold-based pass/fail)
- Jailbreak detector (pattern matching)
- Red team harness (structured adversarial testing)
- Alignment report (multi-dimensional evaluation)

### Deployment
- Packaging: model + config + tokenizer + safety report + SHA256 checksum
- Readiness checklist (model loadable, config present, tokenizer present, size checks)
- Quantization pipeline (INT8)
- Latency profiler (warmup, percentile-based)

### Model Registry
- Versioning with immutable artifact IDs
- Tagging (semantic labels)
- Rollback to previous versions
- Diff between versions (config/tokenizer/bench/safety deltas)
- Lineage DAG (dataset version -> training run -> model)

## Studio Desktop App (Tauri 2 + React 19)

### Architecture
- React Router v7 with hash routing (7 pages)
- Sidebar navigation with Lucide React icons
- ForgeContext (global state) + CommandContext (command execution)
- CSS design system: variables.css, reset.css, components.css, layout.css
- Light theme with blue palette (#F7FBFC / #D6E6F2 / #B9D7EA / #769FCD)

### Pages
- **Training** — Wizard flow: pick method -> configure -> run -> results. 7 method forms, progress monitor, run history, training curves (SVG charts)
- **Datasets** — Two-column layout: dataset/version list + tabs (overview dashboard, version diff, sample inspector, ingest form, filter form)
- **Models** — Registry browser with tabs: version list, model diff, actions (tag/rollback/export)
- **Chat** — Model inference with configurable sampling, streaming responses, conversation history
- **Safety** — Tabs: evaluation form + deployment gate form + results view
- **Deploy** — Tabs: package, quantize, latency profile, readiness checklist
- **Settings** — Data root config + hardware profile display

### Tauri Backend (Rust)
- Subprocess command execution with async task tracking (27 allowed commands)
- Dataset queries (direct file-based)
- Training history JSON loading and chart data prep
- Lineage graph visualization data

## Codebase Quality
- 148+ Python files, 148+ test files
- Full mypy --strict typing, frozen dataclasses
- Max 300 lines/file, 50 lines/function, 4 params/function
- Centralized errors in src/core/errors.py
- 80% test coverage target
- snake_case files, verb_noun() functions

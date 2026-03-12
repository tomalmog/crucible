# Crucible — Completed Features

Reference document for context recovery. Everything listed here is implemented, tested, and working.

## Python CLI & SDK (39 Commands)

### Data Pipeline
- **ingest** — Load data from local paths/S3, resumable checkpoints, incremental mode
- **export-training** — Export snapshot to sharded training files with optional metadata

### Training Methods (13)
- **train** — Standard supervised training with configurable architecture
- **sft** — Supervised fine-tuning with prompt/response masking
- **dpo-train** — Direct preference optimization with reference model, configurable beta
- **rlhf-train** — RLHF with optional reward model training + PPO policy optimization (GAE, clipping, entropy, KL penalty)
- **lora-train** — LoRA adapter fine-tuning (base frozen, adapter-only optimizer)
- **qlora-train** — 4-bit quantized LoRA training for large models on consumer GPUs
- **lora-merge** — Merge trained LoRA adapters back into base model
- **distill** — Knowledge distillation (teacher frozen, KL + CE loss, temperature scaling)
- **domain-adapt** — Continued pretraining with drift detection on reference data
- **grpo-train** — Group relative policy optimization with reward functions
- **kto-train** — Kahneman-Tversky optimization (unpaired binary preference feedback)
- **orpo-train** — Odds ratio preference optimization (SFT + preference in single pass)
- **multimodal-train** — Vision-language model fine-tuning (text pipeline operational)
- **rlvr-train** — RL with verifiable rewards for code/math tasks (SFT pipeline operational)
- **distributed-train** — Multi-GPU DDP training via torchrun

### Distributed Strategies
- DDP (DistributedDataParallel)
- FSDP (Fully Sharded Data Parallel)
- DeepSpeed integration
- TPU/XLA support
- Multi-node launcher and setup

### Remote Training (Slurm Clusters)
- **remote register-cluster** — Register SSH + Slurm cluster connections
- **remote list-clusters** — List registered clusters
- **remote validate-cluster** — Validate SSH access and Slurm availability
- **remote remove-cluster** — Remove a cluster registration
- **remote submit** — Submit single-node training jobs to Slurm
- **remote submit-sweep** — Submit hyperparameter sweeps as Slurm job arrays
- **remote list** — List all remote jobs with status
- **remote status** — Sync and check remote job status
- **remote logs** — View remote job logs (with `--follow` streaming)
- **remote cancel** — Cancel a running/pending Slurm job
- **remote pull-model** — Download trained model from remote cluster to local registry

#### Remote Infrastructure
- Auto-provision conda environment on remote clusters before job submission
- CUDA version auto-detection via `nvidia-smi` — installs matching PyTorch build (cu118/cu121/cu124/cu126)
- Torch CUDA verification on every submission — force-reinstalls if GPU not detected
- Sbatch script generation for single-node, multi-node (torchrun), and sweep (job array) jobs
- Dataset upload strategies: ingested catalog transfer or raw directory tar + remote-side ingestion
- Early JSON write with `submit_phase` field — jobs appear in UI immediately on submit
- Live submission phase updates (connecting → provisioning → uploading → submitting to Slurm)
- Error handling: failed submissions recorded with state and phase for UI display

### Experimentation
- **sweep** — Hyperparameter grid/random search across all training methods via `dispatch_training()`
- **compare** — Side-by-side comparison of two training run results
- **replay** — Deterministic replay from reproducibility bundle
- **run-spec** — Declarative pipeline execution from YAML
- **benchmark** — Perplexity and latency profiling on test data

### Experiment Tracking
- W&B integration via `--wandb-project` flag
- TensorBoard integration via `--tensorboard-dir` flag
- `TrackingAdapter` protocol in `tracking_adapters.py` for extensibility

### Model & System
- **chat** — Text generation from trained/ONNX models with streaming
- **ab-chat** — A/B model comparison chat with DPO data export
- **export-spec** — Export run configuration as reproducible spec
- **verify** — Platform verification (quick/full modes) with artifact preservation
- **hardware-profile** — Detect GPU/TPU capabilities, recommend precision/batch defaults
- **compute** — Cost estimation for training runs
- **recipe** — Training recipe import/export

### Model & Registry
- **model** — Registry operations (list, register, delete, pull, remote-sizes)
- **server** — Collaboration server for multi-user runs + comments

### Evaluation Benchmarks (7)
- **MMLU** — Massive Multitask Language Understanding
- **HellaSwag** — Commonsense reasoning completion
- **ARC** — AI2 Reasoning Challenge
- **WinoGrande** — Commonsense coreference resolution
- **GSM8K** — Grade school math word problems
- **TruthfulQA** — Truthfulness evaluation
- **HumanEval** — Code generation (function completion)
- Shared `_model_loader.py` for efficient model loading across benchmarks

## Core Infrastructure

### Training Engine
- Full training loop with batching, gradient accumulation, checkpointing
- Custom loss functions via hooks
- Precision control (FP32, FP16, BF16, mixed precision with safe fallback)
- Validation splits with early stopping
- Optimizer choices: AdamW, SGD with momentum
- Schedulers: constant, step, cosine, exponential
- Gradient clipping (max_norm=1.0) and NaN detection on every training loop
- Reproducibility: full bundle saved (config hash, environment snapshot, seed)
- Checkpoint management: periodic save, best-model tracking, resume, retention policy
- Extension/hooks interface (run/epoch/batch/checkpoint hooks, custom loss)

### Training Method Registry
- `TrainingMethod` literal type and `TRAINING_METHOD_DISPATCH` table in `src/core/training_methods.py`
- Single source of truth: method → client function + Options class mapping
- `dispatch_training()` for method-agnostic execution (used by sweeps and remote submission)
- Adding a new method = update one file

### Data Pipeline
- Input readers: local file/directory, S3 (boto3), plaintext/JSONL/CSV
- Transforms: exact deduplication, language detection, perplexity quality scoring
- Incremental processing (diff against latest, only process new/changed)
- Resumable checkpoints per pipeline stage
- Lance binary format storage with immutable snapshots
- Catalog metadata (version, counts, parents, recipe)
- Dataset-driven training: all 13 forms use `DatasetSelect` dropdown, backend auto-resolves data path

### Model Registry
- Grouped by name (e.g. "My-Transformer") with per-model version history
- Storage: `.crucible/models/groups/{name}.json`, `.crucible/models/versions/{id}.json`, `.crucible/models/index.json`
- Auto-migration from old flat format via `migrate_flat_to_grouped()`
- Tagging (semantic labels), rollback to previous versions
- Lineage DAG (dataset version → training run → model)
- Soft-delete with permanent purge
- Hub download auto-registration

## Studio Desktop App (Tauri 2 + React 19)

### Architecture
- React Router v7 with hash routing (11 pages)
- Sidebar navigation with Lucide React icons
- CrucibleContext (global state) + CommandContext (command execution)
- CSS design system: variables.css, reset.css, components.css, layout.css
- Dark/light theme with `getTheme()`/`setTheme()` and localStorage persistence

### Pages
- **Training** — Wizard flow: pick method → configure → run → results. 13 method forms with auto-saved drafts, DatasetSelect dropdown, live progress monitor, training curves (SVG charts). Local execution or remote cluster submission with auto-navigate to Jobs page.
- **Datasets** — Two-column layout: dataset/version list + tabs (overview dashboard, version diff, sample inspector, ingest form, filter form)
- **Models** — Grouped registry browser: version list, model diff, actions (tag/rollback/export/delete)
- **Chat** — Model inference with configurable sampling, streaming responses, conversation history
- **Compare Chat** — A/B model comparison: send same prompt to two models side-by-side, rate responses (A/B/Tie), export preferences as DPO training data via native save dialog
- **Benchmarks** — Run evaluation benchmarks (MMLU, HellaSwag, ARC, etc.) against trained models
- **Hub** — Tabbed search for models and datasets with filters (task, library, sort), detail views with file listings and sizes, download with progress, push to HuggingFace
- **Jobs** — Unified job queue for local and remote jobs. Remote jobs show instant visibility from submit with live phase updates (connecting, provisioning, uploading, submitting). Pending Slurm jobs show queue indicator. Cancel, view logs, delete. Failed-during-submit jobs show inline error messages.
- **Clusters** — Register, validate, and manage Slurm cluster connections
- **Settings** — Data root config, hardware profile display, theme toggle
- **Docs** — Built-in training method documentation

### Shared UI Components
- `CommandFormPanel` — Consistent form layout with validation, submit, error display
- `DatasetSelect` — Searchable dropdown pulling from ForgeContext
- `MetricSelect` — Metrics dropdown for sweeps
- `JobResultDetail` — Polymorphic detail views (failed/sweep/training/generic)

### Tauri Backend (Rust)
- Subprocess command execution with async task tracking
- Dataset queries (direct file-based)
- Remote job queries with submit_phase parsing
- Training history JSON loading and chart data prep
- Lineage graph visualization data
- Remote job cancellation via subprocess

## HuggingFace Hub Integration

### CLI Commands
- **hub search-models** — Search models with optional filters (task, library, sort). Query is optional when filters are set.
- **hub search-datasets** — Search datasets with optional filters (task, sort)
- **hub model-info** — Fetch detailed model info: file listing with sizes, license, base model, library, gated status, creation date, downloads, likes
- **hub dataset-info** — Fetch detailed dataset info: file listing with sizes, license, task categories, gated status, creation date, downloads, likes
- **hub download-model** — Download a model to a target directory (auto-registered in model registry)
- **hub download-dataset** — Download a dataset to a target directory
- **hub push** — Push a trained model to HuggingFace Hub

### Studio Hub Page
- **Model search** — 3-column card grid with pagination (12 per page), shows repo name, author, task, downloads, likes, date. Click card to view details.
- **Dataset search** — Same grid layout for datasets, shows size category tag
- **Search filters** — Toggle filter bar with task dropdown, library dropdown (models only), sort by (downloads/likes/newest). Filters work with or without a text query.
- **Model detail view** — Two-column layout: left sidebar with repo name, author, and key-value metadata (size, downloads, likes, license, library, task, base model, gated, created date); right panel with file listing sorted by size descending. Download button with total size shown.
- **Dataset detail view** — Same two-column layout adapted for datasets (task categories instead of library/base model). File list shows top 20 largest files with incremental "show 20 more" pagination for datasets with hundreds/thousands of files.
- **Download flow** — Download button on cards and detail view with status feedback (downloading/done/error/retry). Configurable target directory with PathInput browser.
- **Push tab** — Push trained models to Hub with configurable repo ID and commit message

## Codebase Quality
- 250+ Python source files, 160+ test files
- Full mypy --strict typing, frozen dataclasses
- Max 300 lines/file, 50 lines/function, 4 params/function
- Centralized errors in src/core/errors.py
- 80% test coverage target
- snake_case files, verb_noun() functions

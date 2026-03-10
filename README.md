# Crucible

End-to-end machine learning training platform. Ingest data, train models across 13 algorithms, evaluate results, and deploy — from a single CLI or a desktop app.

Crucible handles the full ML lifecycle: data ingestion with built-in transforms, versioned dataset management, training with live progress streaming, experiment tracking with cost analysis, model versioning with lineage, safety evaluation, and deployment packaging. Everything is reproducible — every run produces a replay bundle that can recreate the exact training configuration.

## Install

```bash
python3 -m pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[serve]"       # PyTorch training
pip install -e ".[s3]"          # S3 ingest/export
pip install -e ".[onnx]"        # ONNX model support
pip install -e ".[lance]"       # Lance vector storage
pip install -e ".[logging]"     # Structured logging
pip install -e ".[tokenizers]"  # HuggingFace tokenizers
```

## Quickstart

```bash
# Ingest raw data into a versioned dataset
crucible ingest ./data --dataset my-dataset

# Filter by language and quality
crucible filter --dataset my-dataset --language en --min-quality 0.2

# Train a model
crucible train --dataset my-dataset --output-dir ./outputs/train

# Fine-tune with LoRA
crucible lora-train --dataset my-dataset --model-path meta-llama/Llama-2-7b --output-dir ./outputs/lora

# Chat with the trained model
crucible chat --model-path ./outputs/train/model.pt --prompt "hello"

# Check hardware capabilities
crucible hardware-profile
```

## Training Methods

Crucible supports 13 training algorithms, all accessible from the CLI and the desktop app:

| Method | Command | Use Case |
|--------|---------|----------|
| Train | `crucible train` | Train from scratch on raw text |
| SFT | `crucible sft` | Supervised fine-tuning on instruction data |
| DPO | `crucible dpo-train` | Direct preference optimization (chosen/rejected pairs) |
| RLHF | `crucible rlhf-train` | Reinforcement learning from human feedback (PPO) |
| LoRA | `crucible lora-train` | Parameter-efficient fine-tuning with low-rank adapters |
| QLoRA | `crucible qlora-train` | 4-bit quantized LoRA |
| Distillation | `crucible distill` | Transfer knowledge from teacher to student model |
| Domain Adapt | `crucible domain-adapt` | Continue pretraining on domain-specific text |
| GRPO | `crucible grpo-train` | Group relative policy optimization with reward functions |
| KTO | `crucible kto-train` | Kahneman-Tversky optimization (unpaired preferences) |
| ORPO | `crucible orpo-train` | Odds ratio preference optimization |
| Multimodal | `crucible multimodal-train` | Vision-language model fine-tuning |
| RLVR | `crucible rlvr-train` | RL with verifiable rewards (code/math) |

All methods support: mixed precision (fp32/fp16/bfloat16), gradient clipping, checkpoint save/resume, custom architectures, lifecycle hooks, and configurable optimizers/schedulers.

## Data Pipeline

```bash
# Ingest with built-in transforms (dedup, language detection, quality scoring)
crucible ingest ./raw-data --dataset research-papers

# List versions
crucible versions --dataset research-papers

# Filter to English, high-quality records
crucible filter --dataset research-papers --language en --min-quality 0.5

# Export for training
crucible export-training --dataset research-papers --output-dir ./training-data

# Generate synthetic training data
crucible synthetic --seed-prompts ./prompts.json --output synthetic-data.jsonl

# Score and curate dataset quality
crucible curate score --dataset research-papers
crucible curate stats --dataset research-papers
```

Datasets are immutable versioned snapshots with full lineage tracking.

## Experiment Tracking

```bash
# List all runs
crucible experiment list

# Compare runs side by side
crucible experiment compare --run-ids run-001 run-002 run-003

# View cost breakdown
crucible cost summary
crucible cost run --run-id run-001

# Run hyperparameter sweep
crucible sweep --config sweep.yaml

# Replay a previous run exactly
crucible replay --bundle ./outputs/train/reproducibility_bundle.json
```

## Model Management

```bash
# List model versions
crucible model list

# Tag a version
crucible model tag --version v3 --tag production

# Compare two versions
crucible model diff --version-a v2 --version-b v3

# Merge models (slerp, ties, dare, average)
crucible merge --models model-a.pt model-b.pt --method slerp --output merged.pt

# Rollback to previous version
crucible model rollback --version v2
```

## Evaluation & Safety

```bash
# Run standardized benchmarks
crucible eval --model-path ./model.pt --benchmark hellaswag

# LLM-as-Judge evaluation
crucible judge --model-path ./model.pt --criteria accuracy helpfulness

# Toxicity scoring
crucible safety-eval --model-path ./model.pt --samples test-prompts.json

# Pre-deployment safety gate (pass/fail)
crucible safety-gate --model-path ./model.pt --threshold 0.3
```

## Deployment

```bash
# Build deployment package
crucible deploy package --model-path ./model.pt --output-dir ./deploy

# Quantize model
crucible deploy quantize --model-path ./model.onnx --quantize-type dynamic

# Profile latency
crucible deploy profile --model-path ./model.onnx --batch-sizes 1 4 8

# Run readiness checklist
crucible deploy checklist --model-path ./model.pt
```

## HuggingFace Hub

```bash
# Search models (with optional filters)
crucible hub search-models llama --limit 10
crucible hub search-models --filter text-generation --library gguf --sort likes
crucible hub search-models "" --filter text-to-image --sort createdAt

# Get detailed model info (file listing, sizes, license, etc.)
crucible hub model-info meta-llama/Llama-3.1-8B-Instruct --json

# Download a model
crucible hub download-model meta-llama/Llama-2-7b

# Search datasets (with optional filters)
crucible hub search-datasets instruction --limit 10
crucible hub search-datasets --filter question-answering --sort likes

# Get detailed dataset info
crucible hub dataset-info tatsu-lab/alpaca --json

# Download a dataset
crucible hub download-dataset tatsu-lab/alpaca

# Push trained model
crucible hub push --model-path ./model.pt --repo-id my-org/my-model
```

## Remote Training (Slurm Clusters)

Train on remote GPU clusters via SSH + Slurm. Crucible handles environment provisioning, data upload, job submission, and result syncing — all from the CLI or desktop app.

```bash
# Register a cluster
crucible remote register-cluster --name my-cluster --host cluster.example.com --user alice

# Validate SSH and Slurm access
crucible remote validate-cluster --name my-cluster

# Submit a training job
crucible remote submit --cluster my-cluster --method lora-train --dataset my-dataset --model-path meta-llama/Llama-2-7b

# Submit a hyperparameter sweep
crucible remote submit-sweep --cluster my-cluster --method sft --dataset my-dataset --sweep-config sweep.yaml

# Monitor and manage
crucible remote list
crucible remote status --job-id rj-abc123
crucible remote logs --job-id rj-abc123 --follow
crucible remote cancel --job-id rj-abc123

# Pull trained model back to local registry
crucible remote pull-model --job-id rj-abc123
```

Remote jobs auto-provision a conda environment with the correct PyTorch + CUDA build for the cluster's GPU hardware. Datasets are uploaded automatically — ingested catalogs are transferred directly, raw datasets are tarred and ingested on the cluster. The Studio Jobs page shows live submission progress from the moment you click submit.

## Studio Desktop App

Crucible includes a desktop application built with Tauri, React, and TypeScript. It provides a visual interface for every CLI feature:

- **Training** — Method picker for all 13 algorithms, configuration wizard with auto-saved drafts, live progress streaming, training curve visualization, local or remote execution
- **Datasets** — Ingest, filter, version browser, sample inspector, annotation interface, synthetic data generation
- **Models** — Grouped version registry, diffing, merging, tagging, rollback, deletion
- **Chat** — Single-model inference and A/B model comparison with DPO export
- **Experiments** — Run tracking, multi-run comparison, cost analysis, LLM judge evaluation
- **Safety** — Toxicity scoring and deployment gating
- **Deploy** — Packaging, quantization, latency profiling, readiness checklist
- **Hub** — Search models/datasets with filters (task, library, sort), detail views with file listings and sizes, download, push to HuggingFace
- **Jobs** — Real-time job monitoring for local and remote jobs. Remote jobs appear instantly on submit with live phase updates (connecting, provisioning, uploading, submitting). Pending Slurm jobs show queue status. Cancel, view logs, and track failures with inline error display.
- **Clusters** — Register, validate, and manage Slurm cluster connections
- **Docs** — Built-in training method documentation and reference

To run the desktop app:

```bash
cd studio-app
npm install
npm run tauri dev
```

## Declarative Pipelines

Define multi-step workflows in YAML:

```yaml
steps:
  - command: ingest
    args: { path: ./data, dataset: demo }
  - command: filter
    args: { dataset: demo, language: en, min_quality: 0.3 }
  - command: train
    args: { dataset: demo, output_dir: ./outputs/demo }
```

```bash
crucible run-spec pipeline.yaml
```

## Distributed Training

```bash
crucible distributed-train --dataset my-dataset --output-dir ./outputs --nproc-per-node 4
```

## Cloud Training

```bash
crucible cloud estimate --model-size 7b --epochs 3 --provider aws
crucible cloud submit --config cloud-config.yaml
crucible cloud status --job-id job-123
crucible cloud sync --job-id job-123 --output-dir ./results
```

See also [Remote Training](#remote-training-slurm-clusters) for Slurm cluster support, which is the primary way to train on remote hardware.

## Output Artifacts

Every training run produces:

| File | Contents |
|------|----------|
| `model.pt` | Trained model weights |
| `history.json` | Batch and epoch loss history |
| `training_curves.png` | Loss visualization |
| `training_config.json` | Full config for reproducible inference |
| `tokenizer_vocab.json` | Fitted tokenizer vocabulary |
| `training_artifacts_manifest.json` | Artifact contract with paths and metadata |
| `reproducibility_bundle.json` | Config hash + environment snapshot for replay |

## Verification

```bash
crucible verify --mode quick    # Fast smoke test
crucible verify --mode full     # Comprehensive validation
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.6 (for training)
- Node.js >= 18 (for Studio desktop app)
- Rust toolchain (for Tauri builds)

## License

See [LICENSE](LICENSE).

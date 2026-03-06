# Forge

End-to-end machine learning training platform. Ingest data, train models across 13 algorithms, evaluate results, and deploy — from a single CLI or a desktop app.

Forge handles the full ML lifecycle: data ingestion with built-in transforms, versioned dataset management, training with live progress streaming, experiment tracking with cost analysis, model versioning with lineage, safety evaluation, and deployment packaging. Everything is reproducible — every run produces a replay bundle that can recreate the exact training configuration.

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
forge ingest ./data --dataset my-dataset

# Filter by language and quality
forge filter --dataset my-dataset --language en --min-quality 0.2

# Train a model
forge train --dataset my-dataset --output-dir ./outputs/train

# Fine-tune with LoRA
forge lora-train --dataset my-dataset --model-path meta-llama/Llama-2-7b --output-dir ./outputs/lora

# Chat with the trained model
forge chat --model-path ./outputs/train/model.pt --prompt "hello"

# Check hardware capabilities
forge hardware-profile
```

## Training Methods

Forge supports 13 training algorithms, all accessible from the CLI and the desktop app:

| Method | Command | Use Case |
|--------|---------|----------|
| Train | `forge train` | Train from scratch on raw text |
| SFT | `forge sft` | Supervised fine-tuning on instruction data |
| DPO | `forge dpo-train` | Direct preference optimization (chosen/rejected pairs) |
| RLHF | `forge rlhf-train` | Reinforcement learning from human feedback (PPO) |
| LoRA | `forge lora-train` | Parameter-efficient fine-tuning with low-rank adapters |
| QLoRA | `forge qlora-train` | 4-bit quantized LoRA |
| Distillation | `forge distill` | Transfer knowledge from teacher to student model |
| Domain Adapt | `forge domain-adapt` | Continue pretraining on domain-specific text |
| GRPO | `forge grpo-train` | Group relative policy optimization with reward functions |
| KTO | `forge kto-train` | Kahneman-Tversky optimization (unpaired preferences) |
| ORPO | `forge orpo-train` | Odds ratio preference optimization |
| Multimodal | `forge multimodal-train` | Vision-language model fine-tuning |
| RLVR | `forge rlvr-train` | RL with verifiable rewards (code/math) |

All methods support: mixed precision (fp32/fp16/bfloat16), gradient clipping, checkpoint save/resume, custom architectures, lifecycle hooks, and configurable optimizers/schedulers.

## Data Pipeline

```bash
# Ingest with built-in transforms (dedup, language detection, quality scoring)
forge ingest ./raw-data --dataset research-papers

# List versions
forge versions --dataset research-papers

# Filter to English, high-quality records
forge filter --dataset research-papers --language en --min-quality 0.5

# Export for training
forge export-training --dataset research-papers --output-dir ./training-data

# Generate synthetic training data
forge synthetic --seed-prompts ./prompts.json --output synthetic-data.jsonl

# Score and curate dataset quality
forge curate score --dataset research-papers
forge curate stats --dataset research-papers
```

Datasets are immutable versioned snapshots with full lineage tracking.

## Experiment Tracking

```bash
# List all runs
forge experiment list

# Compare runs side by side
forge experiment compare --run-ids run-001 run-002 run-003

# View cost breakdown
forge cost summary
forge cost run --run-id run-001

# Run hyperparameter sweep
forge sweep --config sweep.yaml

# Replay a previous run exactly
forge replay --bundle ./outputs/train/reproducibility_bundle.json
```

## Model Management

```bash
# List model versions
forge model list

# Tag a version
forge model tag --version v3 --tag production

# Compare two versions
forge model diff --version-a v2 --version-b v3

# Merge models (slerp, ties, dare, average)
forge merge --models model-a.pt model-b.pt --method slerp --output merged.pt

# Rollback to previous version
forge model rollback --version v2
```

## Evaluation & Safety

```bash
# Run standardized benchmarks
forge eval --model-path ./model.pt --benchmark hellaswag

# LLM-as-Judge evaluation
forge judge --model-path ./model.pt --criteria accuracy helpfulness

# Toxicity scoring
forge safety-eval --model-path ./model.pt --samples test-prompts.json

# Pre-deployment safety gate (pass/fail)
forge safety-gate --model-path ./model.pt --threshold 0.3
```

## Deployment

```bash
# Build deployment package
forge deploy package --model-path ./model.pt --output-dir ./deploy

# Quantize model
forge deploy quantize --model-path ./model.onnx --quantize-type dynamic

# Profile latency
forge deploy profile --model-path ./model.onnx --batch-sizes 1 4 8

# Run readiness checklist
forge deploy checklist --model-path ./model.pt
```

## HuggingFace Hub

```bash
# Search models (with optional filters)
forge hub search-models llama --limit 10
forge hub search-models --filter text-generation --library gguf --sort likes
forge hub search-models "" --filter text-to-image --sort createdAt

# Get detailed model info (file listing, sizes, license, etc.)
forge hub model-info meta-llama/Llama-3.1-8B-Instruct --json

# Download a model
forge hub download-model meta-llama/Llama-2-7b

# Search datasets (with optional filters)
forge hub search-datasets instruction --limit 10
forge hub search-datasets --filter question-answering --sort likes

# Get detailed dataset info
forge hub dataset-info tatsu-lab/alpaca --json

# Download a dataset
forge hub download-dataset tatsu-lab/alpaca

# Push trained model
forge hub push --model-path ./model.pt --repo-id my-org/my-model
```

## Remote Training (Slurm Clusters)

Train on remote GPU clusters via SSH + Slurm. Forge handles environment provisioning, data upload, job submission, and result syncing — all from the CLI or desktop app.

```bash
# Register a cluster
forge remote register-cluster --name my-cluster --host cluster.example.com --user alice

# Validate SSH and Slurm access
forge remote validate-cluster --name my-cluster

# Submit a training job
forge remote submit --cluster my-cluster --method lora-train --dataset my-dataset --model-path meta-llama/Llama-2-7b

# Submit a hyperparameter sweep
forge remote submit-sweep --cluster my-cluster --method sft --dataset my-dataset --sweep-config sweep.yaml

# Monitor and manage
forge remote list
forge remote status --job-id rj-abc123
forge remote logs --job-id rj-abc123 --follow
forge remote cancel --job-id rj-abc123

# Pull trained model back to local registry
forge remote pull-model --job-id rj-abc123
```

Remote jobs auto-provision a conda environment with the correct PyTorch + CUDA build for the cluster's GPU hardware. Datasets are uploaded automatically — ingested catalogs are transferred directly, raw datasets are tarred and ingested on the cluster. The Studio Jobs page shows live submission progress from the moment you click submit.

## Studio Desktop App

Forge includes a desktop application built with Tauri, React, and TypeScript. It provides a visual interface for every CLI feature:

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
forge run-spec pipeline.yaml
```

## Distributed Training

```bash
forge distributed-train --dataset my-dataset --output-dir ./outputs --nproc-per-node 4
```

## Cloud Training

```bash
forge cloud estimate --model-size 7b --epochs 3 --provider aws
forge cloud submit --config cloud-config.yaml
forge cloud status --job-id job-123
forge cloud sync --job-id job-123 --output-dir ./results
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
forge verify --mode quick    # Fast smoke test
forge verify --mode full     # Comprehensive validation
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.6 (for training)
- Node.js >= 18 (for Studio desktop app)
- Rust toolchain (for Tauri builds)

## License

See [LICENSE](LICENSE).

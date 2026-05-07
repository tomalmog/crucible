# Crucible — Product Description

*For use in outreach emails, pitch decks, and conversations with ML startup teams.*

---

## One-Line Summary

Crucible is a model improvement platform for AI startups: upload private data,
build evals, fine-tune open-source models, compare candidates, and promote the
winner with lineage, reproducibility, and private compute execution.

---

## The Problem

ML training workflows are fragmented. Startup teams cobble together separate
tools for data preparation, fine-tuning, experiment tracking, evaluation, and
handoff. Each tool has its own configuration format, CLI, and assumptions about
data shape. Switching between tools means writing glue code, managing
incompatible artifacts, and losing reproducibility at every handoff.

This fragmentation disproportionately affects small AI teams without dedicated
MLOps infrastructure. They spend more time proving that a model got better than
actually improving it.

---

## What Crucible Does

Crucible is a single platform for eval-gated model improvement:

**Data Management** — Ingest raw data from local files or S3, apply built-in
transforms, and manage versioned dataset snapshots with lineage tracking.

**Primary Fine-Tuning Workflows** — Start with the methods most startups need:

- **SFT** — Supervised fine-tuning on instruction/response pairs with prompt token masking
- **LoRA / QLoRA** — Parameter-efficient fine-tuning using low-rank adapters, with optional 4-bit quantization for training large models on consumer hardware
- **DPO** — Direct Preference Optimization for aligning models with human preferences using chosen/rejected pairs
- **Domain Adaptation** — Continue pretraining on domain-specific text to specialize a foundation model
- **Eval-gated runs** — Define the success metric before launch and compare the candidate against the base model before promotion

**Advanced Research Workflows** — Keep deeper methods available without making
them the default path:

- **Standard training** — Train transformer models from scratch on raw text
- **RLHF** — Full reinforcement learning from human feedback pipeline with PPO
- **Knowledge Distillation** — Transfer capabilities from a teacher model to a smaller student
- **GRPO** — Group Relative Policy Optimization with custom reward functions
- **KTO** — Kahneman-Tversky Optimization for preference learning from unpaired feedback (no chosen/rejected pairs needed)
- **ORPO** — Odds Ratio Preference Optimization combining SFT and preference learning in a single stage
- **Multimodal** — Vision-language model fine-tuning
- **RLVR** — Reinforcement learning with verifiable rewards for code and math tasks

All methods include gradient clipping, NaN detection, mixed precision support
(fp32/fp16/bfloat16), checkpoint save/resume, configurable optimizers and
schedulers, and real-time progress streaming.

**Runs and Model Registry** — Every run is tracked with metrics,
hyperparameters, hardware profile, artifact paths, project context, and
promotion stage. Compare runs side by side, replay experiments from
reproducibility bundles, and manage model candidates in a registry.

**Model Management** — Version every model with lineage back to its training data. Tag versions, diff model architectures, merge weights using SLERP/TIES/DARE/averaging, and rollback to any previous version.

**Evaluation and Model Health** — Run standardized benchmarks with actual model
inference and launch a curated health suite that covers prediction traces,
representation maps, and causal contrast checks. Granular mechanistic
diagnostics remain available only as targeted follow-up tools.

**HuggingFace Hub Integration** — Search, download, and push models and datasets directly from the CLI or desktop app. Downloaded models are auto-registered in the local model registry.

**Private Compute Execution** — Submit fine-tuning jobs to local machines, SSH
GPU boxes, or Slurm clusters. Crucible auto-provisions the remote environment,
uploads datasets, records phase updates, streams logs, and pulls trained models
back into the registry.

---

## Two Interfaces, One Platform

### CLI (Python)

Every feature is available as a CLI command. Commands compose into declarative YAML pipelines for reproducible multi-step workflows:

```bash
crucible ingest ./data --dataset papers
crucible lora-train --dataset papers --model-path meta-llama/Llama-2-7b --output-dir ./outputs
crucible eval --model-path ./outputs/model.pt --benchmark hellaswag
```

### Studio App (Crucible Studio)

A native application built with Tauri that provides a visual interface for the
entire workflow. Teams get a goal-first fine-tuning flow, launch preflight,
eval-gated run context, dataset inspection, model registry, A/B comparison with
DPO export, Model Health, and a Runs page that tracks local and remote work in
real time.

The desktop app calls the same Python CLI under the hood — there is no feature gap between the two interfaces.

---

## Technical Details

- **Python >= 3.11** with strict mypy typing, frozen dataclasses, and structured logging
- **PyTorch 2.6** for all training algorithms, with optional ONNX export
- **Lance 1.2** for efficient vector storage of datasets
- **Tauri 2** desktop runtime (Rust backend, React 19 frontend)
- **Reproducibility** — every run emits a reproducibility bundle containing the config hash, training parameters, and environment snapshot. `crucible replay` recreates any previous run exactly.
- **Extensibility** — custom model architectures (load a .py file), custom training loops, lifecycle hooks (run/epoch/batch/checkpoint callbacks), and custom loss functions
- **Distributed training** via torchrun for multi-GPU DDP
- **Remote Slurm clusters** — submit jobs via SSH with auto-provisioning, CUDA detection, and live monitoring
- **Cloud burst** for training on AWS/GCP/Azure with cost estimation

---

## Who Is Crucible For

- **AI startups** improving open-source models on private production data
- **Small ML teams** who need evals, model versioning, and deployment handoff without setting up separate infrastructure for each
- **ML researchers** who want to iterate faster by eliminating tool-switching overhead
- **Academic labs** that need reproducible training pipelines for publishing
- **Individual practitioners** who want to fine-tune foundation models on their own data with a single command
- **Teams evaluating alignment techniques** (DPO, RLHF, KTO, ORPO, GRPO) who need a consistent training and evaluation framework across methods

---

## What Makes Crucible Different

1. **One tool, not ten.** Data prep, training, tracking, evaluation, and deployment in a single CLI. No glue code between tools.
2. **Goal-first fine-tuning with advanced depth.** Start with SFT, LoRA/QLoRA, DPO, and domain adaptation; keep RLHF, GRPO, RLVR, distillation, and mechanistic diagnostics available when needed.
3. **Local to cluster in one click.** Train on your machine or submit to a Slurm cluster — same config, same monitoring, same model registry. Crucible auto-provisions the remote environment and detects CUDA.
4. **Reproducibility by default.** Every run produces a replay bundle. Every dataset version is immutable. Every model has lineage back to its training data.
5. **Desktop app included.** Not everyone wants to live in the terminal. Crucible Studio provides a visual interface that covers the full workflow, with live progress streaming and auto-saved configuration.
6. **Works on consumer hardware.** QLoRA support means researchers can fine-tune 7B+ parameter models on a single GPU. Mixed precision and gradient accumulation make efficient use of available memory.

---

## Current Status

- **Eval-gated fine-tuning** — primary startup workflows surfaced first, advanced training methods available in research mode
- **39 CLI commands** covering data, training, evaluation, hub integration, model management, and remote cluster operations
- **Desktop app** with 11 pages covering the full workflow including real-time remote job monitoring
- **7 evaluation benchmarks** with actual model inference
- **Remote Slurm training** with auto-provisioning, CUDA detection, and instant job visibility
- Open source, actively developed

# Forge — Product Description

*For use in outreach emails, pitch decks, and conversations with researchers.*

---

## One-Line Summary

Forge is an open-source, end-to-end ML training platform that takes researchers from raw data to deployed model in a single tool — with 13 training algorithms, full experiment tracking, and a desktop GUI.

---

## The Problem

ML training workflows are fragmented. Researchers cobble together separate tools for data preparation, training, experiment tracking, evaluation, and deployment. Each tool has its own configuration format, its own CLI, and its own assumptions about how data should be structured. Switching between tools means writing glue code, managing incompatible file formats, and losing reproducibility at every handoff.

This fragmentation disproportionately affects individual researchers, small labs, and teams without dedicated MLOps infrastructure. They spend more time on tooling than on research.

---

## What Forge Does

Forge is a single platform that handles the entire ML training lifecycle:

**Data Management** — Ingest raw data from local files or S3, apply built-in transforms (deduplication, language detection, quality scoring), and manage immutable versioned dataset snapshots with full lineage tracking. Every dataset operation is recorded and reproducible.

**13 Training Algorithms** — Train from scratch, fine-tune, or align models using state-of-the-art methods, all from the same interface:

- **Standard training** — Train transformer models from scratch on raw text
- **SFT** — Supervised fine-tuning on instruction/response pairs with prompt token masking
- **DPO** — Direct Preference Optimization for aligning models with human preferences using chosen/rejected pairs
- **RLHF** — Full reinforcement learning from human feedback pipeline with PPO
- **LoRA / QLoRA** — Parameter-efficient fine-tuning using low-rank adapters, with optional 4-bit quantization for training large models on consumer hardware
- **Knowledge Distillation** — Transfer capabilities from a teacher model to a smaller student
- **Domain Adaptation** — Continue pretraining on domain-specific text to specialize a foundation model
- **GRPO** — Group Relative Policy Optimization with custom reward functions
- **KTO** — Kahneman-Tversky Optimization for preference learning from unpaired feedback (no chosen/rejected pairs needed)
- **ORPO** — Odds Ratio Preference Optimization combining SFT and preference learning in a single stage
- **Multimodal** — Vision-language model fine-tuning
- **RLVR** — Reinforcement learning with verifiable rewards for code and math tasks

All methods include gradient clipping, NaN detection, mixed precision support (fp32/fp16/bfloat16), checkpoint save/resume, configurable optimizers and schedulers, and real-time progress streaming.

**Experiment Tracking** — Every training run is automatically tracked with full metrics, hyperparameters, hardware profile, and cost analysis (GPU hours, electricity, cloud costs). Compare runs side by side, replay previous experiments exactly from reproducibility bundles, and run hyperparameter sweeps with grid or random search.

**Model Management** — Version every model with lineage back to its training data. Tag versions, diff model architectures, merge weights using SLERP/TIES/DARE/averaging, and rollback to any previous version.

**Evaluation & Safety** — Run standardized benchmarks, LLM-as-Judge evaluations with custom criteria, and toxicity scoring. Enforce pre-deployment safety gates with configurable thresholds.

**Deployment** — Package models with checksums and metadata, quantize to ONNX (dynamic or static), profile latency across batch sizes, and run automated readiness checklists before shipping.

**HuggingFace Hub Integration** — Search, download, and push models and datasets directly from the CLI or desktop app.

---

## Two Interfaces, One Platform

### CLI (Python)

Every feature is available as a CLI command. Commands compose into declarative YAML pipelines for reproducible multi-step workflows:

```bash
forge ingest ./data --dataset papers
forge filter --dataset papers --language en --min-quality 0.3
forge lora-train --dataset papers --model-path meta-llama/Llama-2-7b --output-dir ./outputs
forge eval --model-path ./outputs/model.pt --benchmark hellaswag
forge deploy package --model-path ./outputs/model.pt --output-dir ./deploy
```

### Desktop App (Forge Studio)

A native desktop application built with Tauri that provides a visual interface for the entire workflow. Researchers who prefer GUIs get a first-class experience: a training method picker, a configuration wizard that auto-saves drafts between sessions, live training progress with loss curves, dataset inspection with sample browsing, model comparison, and one-click deployment packaging.

The desktop app calls the same Python CLI under the hood — there is no feature gap between the two interfaces.

---

## Technical Details

- **Python >= 3.11** with strict mypy typing, frozen dataclasses, and structured logging
- **PyTorch 2.6** for all training algorithms, with optional ONNX export
- **Lance 1.2** for efficient vector storage of datasets
- **Tauri 2** desktop runtime (Rust backend, React 19 frontend)
- **Reproducibility** — every run emits a reproducibility bundle containing the config hash, training parameters, and environment snapshot. `forge replay` recreates any previous run exactly.
- **Extensibility** — custom model architectures (load a .py file), custom training loops, lifecycle hooks (run/epoch/batch/checkpoint callbacks), and custom loss functions
- **Distributed training** via torchrun for multi-GPU DDP
- **Cloud burst** for training on AWS/GCP/Azure with cost estimation

---

## Who Is Forge For

- **ML researchers** who want to iterate faster by eliminating tool-switching overhead
- **Small ML teams** who need experiment tracking, model versioning, and deployment tooling without setting up separate infrastructure for each
- **Academic labs** that need reproducible training pipelines for publishing
- **Individual practitioners** who want to fine-tune foundation models on their own data with a single command
- **Teams evaluating alignment techniques** (DPO, RLHF, KTO, ORPO, GRPO) who need a consistent training and evaluation framework across methods

---

## What Makes Forge Different

1. **One tool, not ten.** Data prep, training, tracking, evaluation, and deployment in a single CLI. No glue code between tools.
2. **13 training algorithms in one interface.** Switch between SFT, DPO, LoRA, RLHF, distillation, and more by changing one flag. Same data format, same evaluation pipeline, same deployment flow.
3. **Reproducibility by default.** Every run produces a replay bundle. Every dataset version is immutable. Every model has lineage back to its training data.
4. **Desktop app included.** Not everyone wants to live in the terminal. Forge Studio provides a visual interface that covers the full workflow, with live progress streaming and auto-saved configuration.
5. **Works on consumer hardware.** QLoRA support means researchers can fine-tune 7B+ parameter models on a single GPU. Mixed precision and gradient accumulation make efficient use of available memory.

---

## Current Status

- **10 training methods fully operational**, 3 in partial implementation (GRPO, Multimodal, RLVR)
- **53 CLI commands** covering data, training, evaluation, deployment, hub integration, and model management
- **Desktop app** with 13 pages covering the full workflow
- Open source, actively developed

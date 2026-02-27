# Forge — Development Roadmap

What needs to be built to make Forge the best end-to-end LLM training platform for research labs, companies, and individuals.

## Competitive Context

Forge is the only desktop-native LLM training application. Cloud platforms (Together AI, SageMaker) require subscriptions. CLI tools (Axolotl, Unsloth, LLaMA-Factory) require terminal expertise. W&B/MLflow only track experiments. Forge is the one app that does ingest -> train -> evaluate -> deploy.

Current gaps are mostly in training algorithm breadth, experiment tracking depth, and consumer hardware optimization.

---

## P0 — Critical (blocks adoption)

### GRPO (Group Relative Policy Optimization)
The dominant training algorithm of 2025 for reasoning models (DeepSeek-R1, Qwen-R). Group sampling, reward scoring, advantage calculation without a separate critic model. Reference: HuggingFace TRL GRPOTrainer. Without this, Forge looks behind the curve.

### QLoRA (Quantized LoRA)
4-bit quantized LoRA training using bitsandbytes or torchao. Enables training 70B models on consumer GPUs (24GB VRAM). Without this, individuals with consumer hardware can't train real models. Table stakes for a desktop tool.

### Experiment Tracking System
Currently only loss curves exist. Need:
- Log every run: hyperparameters, all metrics (loss, perplexity, eval scores, LR schedule), hardware utilization, timestamps, config hash
- Run comparison view: side-by-side metrics for 2+ runs, sortable run table
- Artifact lineage: which dataset version + config produced which model
- This transforms Forge from "a training launcher" into "a training platform"

### HuggingFace Hub Integration ✓ DONE
- ~~Browse and download models from the Hub in the UI~~ — Implemented with search, filters (task/library/sort), detail views with file listings and sizes, and download
- ~~Browse and download datasets from the Hub~~ — Implemented with search, filters, detail views with paginated file listings, and download
- ~~Push trained models/adapters back to the Hub~~ — Implemented via CLI and Studio Push tab
- **Remaining:** Cancel in-progress downloads, resume partial downloads

### Flash Attention
Integrate Flash Attention 2 (or torch SDPA as fallback). 2-4x faster attention, significantly less memory. Table stakes for any training tool in 2026.

---

## P1 — High Priority (separates a tool from a platform)

### Automated Evaluation Harness
Integrate EleutherAI's lm-evaluation-harness or build a subset. Run standard benchmarks after training: MMLU, HumanEval, GSM8K, HellaSwag, ARC, TruthfulQA, WinoGrande. Store results alongside the training run. Show improvement/regression vs. base model.

### KTO (Kahneman-Tversky Optimization)
Works with unpaired preference data — much easier to collect than DPO's paired chosen/rejected format. Well-established alternative.

### ORPO (Odds Ratio Preference Optimization)
Combined SFT + preference optimization in one training step. Simpler pipeline, faster convergence. Reduces the two-stage SFT-then-DPO workflow to one step.

### Hyperparameter Sweep UI
The CLI sweep command already exists. Expose it in the Studio UI:
- Search space builder (learning rate range, batch size options, LoRA rank choices)
- Grid search and random search
- Results table with sortable columns, best config highlighted
- Stretch: Bayesian optimization via Optuna

### Smart Hardware-Aware Configuration
Extend the existing hardware profiler:
- Given GPU VRAM, suggest batch size, gradient accumulation, precision, QLoRA vs full
- Estimated training time and memory usage before starting
- Warning if config will OOM before training starts
- Pre-built profiles for common GPUs (RTX 4090, A100, H100, M-series)

---

## P2 — Differentiating (what nobody else does well)

### Visual Dataset Curator
- Browse training examples in a table view with search/filter
- Auto-score example quality (detect short, repetitive, low-quality entries)
- Distribution charts: token length histogram, topic clusters, quality scores
- Manual remove/edit of individual examples
- Stretch: synthetic data generation from seed prompts using a loaded model

### Model Merging
Implement SLERP, TIES, DARE, weighted average merging. GUI for selecting models and merge parameters. Runs on CPU, no GPU needed. Hugely popular in the open-source community.

### A/B Model Comparison (Side-by-Side Chat)
Send the same prompt to two models simultaneously. Compare responses side-by-side. Rate which is better. This also generates DPO preference data as a side effect.

### Training Recipe System
Shareable configs that capture an entire training setup:
- Export: model + dataset + hyperparameters + eval criteria as JSON/YAML
- Import recipes from others
- Built-in library: "Coding Assistant," "Customer Support Bot," "Reasoning Model," "Domain Expert"

### Offline-First with Cloud Burst
Train locally by default. "Send to cloud" button for jobs too big for local hardware (Modal, RunPod, Lambda). Results sync back to local Forge instance. No other tool bridges local and cloud.

### LLM-as-Judge Evaluation
Use a stronger model (GPT-4, Claude) to evaluate fine-tuned model outputs. Configurable evaluation criteria (helpfulness, accuracy, safety, style). Cheaper and faster than human evaluation.

---

## P3 — Future

### Multimodal Fine-Tuning
Vision-language model fine-tuning (Qwen2-VL, LLaVA). Image + text dataset ingestion. The frontier of 2025-2026 open-source fine-tuning.

### Cost and Resource Tracking
GPU-hours per training run. Aggregate cost per project. Historical resource usage charts. Estimated electricity cost based on hardware TDP.

### Synthetic Data Generation Pipeline
Given a loaded model and seed prompts, generate instruction-response pairs. Quality filtering and ranking. Human review interface.

### RLVR (RL with Verifiable Rewards)
Training reasoning models using code/math verification instead of reward models. Used for math and coding tasks where correctness is automatically verifiable.

### Advanced PEFT Methods
- DoRA (Weight-Decomposed Low-Rank Adaptation) — better than LoRA for many tasks
- GaLore, APOLLO, LoftQ, PiSSA — memory-efficient alternatives

### Team Collaboration
- Shared project sync (Git-based or custom)
- Run result sharing via link
- Comments on experiments
- Role-based access control
- Audit logging for compliance

### Data Annotation Interface
Human-in-the-loop labeling for preference data (RLHF/DPO). LLM-assisted pre-labeling with human review. Integrated into the dataset curator.

### Continued Pre-Training (Enhanced)
The domain-adapt command exists but could be expanded: unstructured text input at scale, learning rate warmup strategies, catastrophic forgetting mitigation techniques, curriculum learning.

---

## Implementation Order

Phase 1: **GRPO + QLoRA + Flash Attention**
Close the algorithm gap. After this, Forge trains anything Axolotl/Unsloth can.

Phase 2: **Experiment Tracking + HuggingFace Hub** (Hub ✓ done)
Close the workflow gap. After this, Forge replaces the "Axolotl + W&B + manual file management" stack.

Phase 3: **Eval Harness + Sweep UI + Smart Hardware Config**
Close the research workflow gap. After this, researchers can run full experiment cycles in one app.

Phase 4: **Dataset Curator + Model Merging + A/B Chat + Recipes**
Lean into desktop-native advantages. After this, Forge does things no CLI or cloud tool can match.

Phase 5: **Cloud Burst + LLM-as-Judge + Multimodal + Team Features**
Platform expansion.

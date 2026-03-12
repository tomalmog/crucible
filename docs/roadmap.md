# Crucible — Development Roadmap

What needs to be built to make Crucible the best end-to-end LLM training platform for research labs, companies, and individuals.

## Competitive Context

Crucible is the only desktop-native LLM training application. Cloud platforms (Together AI, SageMaker) require subscriptions. CLI tools (Axolotl, Unsloth, LLaMA-Factory) require terminal expertise. W&B/MLflow only track experiments. Crucible is the one app that does ingest → train → evaluate → deploy, locally or on remote Slurm clusters.

---

## Recently Completed

### GRPO (Group Relative Policy Optimization) ✓
Training command implemented. Default LR, pad exclusion, gradient clipping all correct. Core loss function operational. Full online generation + advantage-weighted policy gradient is a future enhancement.

### QLoRA (Quantized LoRA) ✓
4-bit quantized LoRA training. Enables training large models on consumer GPUs with limited VRAM.

### KTO (Kahneman-Tversky Optimization) ✓
Asymmetric desirable/undesirable loss implemented. Works with unpaired binary preference feedback. Reference model KL anchor is a future enhancement.

### ORPO (Odds Ratio Preference Optimization) ✓
Single-stage SFT + preference optimization. Batch builder interleaves chosen/rejected, loss computes SFT + odds-ratio preference term.

### Multimodal Fine-Tuning ✓ (Text Pipeline)
Training command operational with text data pipeline. Vision encoder integration (image loading, CLIP ViT, cross-modal projection) is a future enhancement.

### RLVR (RL with Verifiable Rewards) ✓ (SFT Pipeline)
Training command operational with SFT on prompt+solution pairs. Code/math verifier integration is a future enhancement.

### Experiment Tracking ✓
W&B and TensorBoard integration via `TrackingAdapter` protocol. Configurable with `--wandb-project` and `--tensorboard-dir` flags.

### HuggingFace Hub Integration ✓
Full search, download, detail views, and push for both models and datasets. Auto-registration of downloaded models.

### Evaluation Benchmarks ✓
7 real benchmarks (MMLU, HellaSwag, ARC, WinoGrande, GSM8K, TruthfulQA, HumanEval) with actual model inference.

### Hyperparameter Sweep UI ✓
Visual parameter builder in Studio. Grid and random search. Method-agnostic via `dispatch_training()`. Supports both local and remote execution.

### Remote Training (Slurm Clusters) ✓
Full lifecycle: register cluster → validate → submit → monitor → cancel → pull model. Auto-provisions conda env, detects CUDA version, uploads datasets. Instant job visibility in Studio with live phase updates.

---

## P0 — Critical (blocks adoption)

### Flash Attention
Integrate Flash Attention 2 (or torch SDPA as fallback). 2-4x faster attention, significantly less memory. Table stakes for any training tool in 2026.

### Resume Partial Hub Downloads
Cancel in-progress downloads, resume partial downloads. Currently downloads must complete in one session.

---

## P1 — High Priority (separates a tool from a platform)

### Smart Hardware-Aware Configuration
Extend the existing hardware profiler:
- Given GPU VRAM, suggest batch size, gradient accumulation, precision, QLoRA vs full
- Estimated training time and memory usage before starting
- Warning if config will OOM before training starts
- Pre-built profiles for common GPUs (RTX 4090, A100, H100, M-series)

### Full GRPO Online Generation
Wire reward function scoring and group-relative advantage computation into the GRPO training loop. Currently uses simplified loss; the full DeepSeek-R1 algorithm needs online response generation + advantage-weighted policy gradient + KL penalty.

### Full Multimodal Pipeline
Image loading, CLIP ViT encoder integration, cross-modal projection layer. Currently text-only; the `image_path` field and vision options exist but are unused.

### Full RLVR Verification
Code/math verifier integration, reward signal from verification, policy gradient update. Currently does SFT on prompt+solution pairs.

---

## P2 — Differentiating (what nobody else does well)

### Visual Dataset Curator
- Browse training examples in a table view with search/filter
- Auto-score example quality (detect short, repetitive, low-quality entries)
- Distribution charts: token length histogram, topic clusters, quality scores
- Manual remove/edit of individual examples

### Training Recipe System
Shareable configs that capture an entire training setup:
- Export: model + dataset + hyperparameters + eval criteria as JSON/YAML
- Import recipes from others
- Built-in library: "Coding Assistant," "Customer Support Bot," "Reasoning Model," "Domain Expert"

### Multi-Cluster Job Management
- Submit to multiple Slurm clusters from one interface
- Cross-cluster comparison of training results
- Auto-select cluster based on GPU availability

---

## P3 — Future

### Cost and Resource Tracking
GPU-hours per training run. Aggregate cost per project. Historical resource usage charts. Estimated electricity cost based on hardware TDP.

### Synthetic Data Generation Pipeline
Given a loaded model and seed prompts, generate instruction-response pairs. Quality filtering and ranking. Human review interface.

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

### KV-Cache / Faster Inference
Autoregressive generation with KV-caching for dramatically faster chat inference. Currently recomputes the full sequence on every token.

---

## Implementation Order

Phase 1: ~~**GRPO + QLoRA + Flash Attention**~~ → GRPO ✓, QLoRA ✓. Flash Attention remaining.

Phase 2: ~~**Experiment Tracking + HuggingFace Hub**~~ → Both ✓ done.

Phase 3: ~~**Eval Harness + Sweep UI + Smart Hardware Config**~~ → Eval ✓, Sweep UI ✓. Smart hardware config remaining.

Phase 4: **Dataset Curator + Recipes + Multi-Cluster**
Lean into desktop-native advantages. After this, Crucible does things no CLI or cloud tool can match.

Phase 5: **Full GRPO/Multimodal/RLVR + Team Features + Advanced PEFT**
Complete the advanced training methods and expand to team workflows.

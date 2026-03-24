# Crucible Studio — Design Brief

> **Purpose**: This document gives a designer (human or AI) everything they need to redesign the Crucible Studio UI. You have full creative freedom over visual design, layout, spacing, typography, color, and interaction patterns. This document describes **what the app does** — how it looks is up to you.

---

## What Is Crucible?

Crucible is a desktop application (built with Tauri) for machine learning practitioners. It handles the full ML workflow: ingesting training data, training models (13 different methods), evaluating results, managing models, and deploying to HuggingFace Hub. Think of it as a local-first ML workbench.

The app has a **sidebar navigation** and **content area** layout. Users move between pages to perform different tasks. Some pages are simple lists, others are multi-step wizards, and others are interactive tools (like chat).

---

## Tech Constraints (Non-Negotiable)

- **React 19 + TypeScript** — no framework changes
- **Plain CSS with design tokens** (CSS custom properties) — no Tailwind, no CSS-in-JS
- **lucide-react** for icons
- **Dark and light theme support** — must work in both modes
- **40+ color palettes** already exist — design should work with any palette, not just one
- **Desktop-only** — no mobile responsive needed. Minimum viewport ~1200px wide
- **Tauri desktop app** — runs in a webview, not a browser. No browser chrome visible

---

## Pages & Their Functionality

### 1. Training (Primary Page)

The most important page. Users come here to train ML models.

**Training Method Picker**: A selection screen showing 13 training methods organized into 4 categories:

| Category | Methods |
|----------|---------|
| Pre-Training | Basic Training |
| Fine-Tuning | SFT, LoRA, QLoRA, Domain Adaptation, Multimodal |
| Alignment | DPO, RLHF, GRPO, KTO, ORPO, RLVR |
| Knowledge Transfer | Distillation |

Each method has a name, short description, and category label. User picks one to proceed.

**Training Wizard**: After picking a method, user fills out a form:
- **Dataset selection** — dropdown of available datasets (local + remote)
- **Base model selection** — dropdown of registered models (for fine-tuning methods)
- **Method-specific fields** — each method has unique parameters (e.g., LoRA has rank/alpha/dropout, DPO has beta, RLHF has PPO parameters)
- **Shared training fields** — common across all methods: epochs, learning rate, batch size, optimizer, precision mode, validation split, gradient accumulation, checkpointing options, experiment tracking (W&B/TensorBoard)
- **Output directory** — where to save the trained model
- **Local vs Remote toggle** — can submit to a Slurm cluster instead of training locally
- **Remote cluster config** (if remote) — partition, GPUs, memory, time limit

**Training Monitor**: While training runs, shows:
- Current epoch/step progress
- Loss curve (batch losses over time)
- Elapsed time and estimated remaining
- Console output (stdout/stderr stream)
- Kill button

**Sweep Tab**: Alternative to single training — runs multiple trials with different hyperparameters:
- Pick a method
- Define parameter grid (which params to sweep, value ranges)
- Strategy: grid search or random search
- Metric to optimize (e.g., validation_loss)
- Results table showing all trials ranked by metric

**Additional tabs**: Training run history, training curves viewer, recipe library, cloud burst form

### 2. Datasets

Dataset management hub.

**List View**: Shows all datasets with name and size. Supports:
- Click to select and view details
- Delete with confirmation
- Ingest new data button

**Dataset Dashboard** (when a dataset is selected):
- Record count
- Quality score stats (average, min, max)
- Language distribution breakdown
- Source file breakdown
- Visual representations of these stats (bar charts, metric cards)

**Sample Inspector**: Browse individual records in the dataset:
- Paginated table of records
- Shows text content, quality score, language, source
- Offset/limit navigation

**Ingest Form**: Add new data to a dataset:
- Source path input (file or directory, with file picker)
- Dataset name
- Quality model selection
- Progress tracking during ingestion

**Other Dataset Tools**:
- **Curator**: Filter/curate dataset records
- **Filter Form**: Apply metadata filters (language, quality threshold, source)
- **Synthetic Data Form**: Generate synthetic training data
- **Annotation View**: Manual data annotation interface

### 3. Models

Model registry browser.

**List View**: All registered models showing:
- Model name
- Location (local, remote, or both)
- File size
- Source (which training run created it, or "hub download")
- Actions: delete, view details

**Model Overview** (detail view):
- Model metadata (name, path, creation date)
- Architecture info (if available)
- Training lineage (what dataset was used, what method)
- Local/remote paths

**Model Merge Form**: Merge LoRA adapters into a base model:
- Base model selector
- LoRA adapter path
- Output path

### 4. Chat

Interactive inference with trained models.

**Single Chat**:
- Model selector dropdown
- Message input area
- Chat thread display (user/assistant messages)
- Sampling parameters: temperature, top-k, top-p, max length
- Support for both local and remote models
- Clear conversation button

**Compare Chat** (separate page):
- Two model selectors side by side
- Single prompt input, responses from both models displayed
- "Choose preferred" buttons for A/B preference
- Export preferences as DPO training data (JSONL)

### 5. Hub (HuggingFace Integration)

**Model Search**:
- Search input for HuggingFace model repos
- Results list with model name, downloads, likes
- Click to view model details (model card, files, metadata)
- Download button → downloads model and registers in local registry

**Dataset Search**:
- Same pattern as model search but for HF datasets
- Download to local storage

**Push Form**:
- Select a local model
- Specify HuggingFace repo ID
- Push to Hub

### 6. Jobs

Unified job monitoring for all running/completed tasks.

**Filters**: Status (all/running/completed/failed), location (all/local/remote), type (all/training/eval/sweep)

**Job List**: Each job shows:
- Job name/label (editable)
- Status badge (running, completed, failed, cancelled)
- Training method used
- Duration / elapsed time
- Location (local or cluster name)

**Job Detail** (expanded view):
- Full console output
- Training metrics/results
- Error details (if failed)
- For sweeps: trial results table
- For remote: Slurm job ID, cluster info

**Actions**: Kill running jobs, rename, delete completed jobs

### 7. Clusters

Slurm cluster management.

**Cluster List**: Registered clusters shown as cards with:
- Cluster name
- Host address
- SSH key path
- Default partition
- Status indicator
- Edit/delete actions

**Register Form**:
- Cluster name
- Host (user@hostname)
- SSH key path
- Default partition
- Module loads (e.g., "module load cuda/11.8")
- Remote workspace directory

### 8. Experiments (Benchmarks)

Evaluation results display.

**Benchmark Results View**:
- 7 benchmarks: MMLU, HellaSwag, ARC, WinoGrande, GSM8K, TruthfulQA, HumanEval
- Score for each benchmark (0-100%)
- Number of examples evaluated
- Optional: comparison against a base model (shows delta)
- Average score across all benchmarks

### 9. Resources

System monitoring and storage management.

**Storage Panel**: Breakdown of disk usage:
- Datasets total size
- Models total size
- Training runs/checkpoints size
- Cache size
- Visual breakdown (bar chart or similar)

**Hardware Panel**: Local machine specs:
- GPU(s): name, VRAM, CUDA version
- CPU: cores, model
- RAM: total/available
- Disk: total/available

**Activity Panel**: Recent job activity and system events

**Cleanup Panel**: Find and delete orphaned files:
- Orphaned training runs (no associated model)
- Cache clearing
- Size savings estimates

**Remote Storage Panel**: Storage usage on connected clusters

### 10. Docs

In-app documentation browser.

**Sidebar**: Searchable list of documentation articles organized by topic:
- Getting Started, Concepts, Studio Guide
- Training guides: one per method (SFT, DPO, LoRA, RLHF, etc.)
- Data management, data formats
- Hyperparameter sweeps
- Common training options

**Article Viewer**: Renders markdown content with:
- Headers, code blocks, tables, lists
- Syntax highlighting for code
- Searchable

### 11. Settings

App configuration.

- **Theme toggle**: Light/dark mode switch
- **Color palette selector**: Choose from 40+ palettes (warm brown, terracotta, mint, etc.)
- **Data root**: Where `.crucible/` directory lives (path input with file picker)
- **Hardware profile display**: Shows detected hardware

### 12. UI Test Page

Component library / design system showcase. Shows all available UI components with examples. This page is for development reference only.

---

## Shared Components & Patterns

These are the reusable building blocks. You should redesign these — they'll be used everywhere.

### Layout
- **Page Header**: Title text + optional action buttons (top of every page)
- **Detail Page**: Back arrow + title + actions (for drill-down views)
- **Sidebar**: App navigation (collapsible)
- **Tab Bar**: Horizontal tab switcher within pages

### Data Display
- **Metric Card**: Label + large value (used in stats grids, dashboards)
- **Bar Chart**: Simple horizontal bars with labels and values
- **Status Console**: Monospace scrollable output area (training logs, command output)
- **Progress Bar**: With elapsed/remaining time display
- **Badge**: Small status labels (running, completed, failed, local, remote)

### Lists
- **List Row**: Clickable row with name, metadata badges, action buttons, optional chevron
- **Empty State**: Icon + title + description when a list has no items

### Forms
- **Form Field**: Label + input + optional hint text + required indicator
- **Form Section**: Collapsible group of related fields with a section header
- **Path Input**: Text input + browse button (opens native file picker)
- **Dataset Select**: Searchable dropdown for picking datasets
- **Model Select**: Searchable dropdown for picking models
- **Metric Select**: Dropdown for picking optimization metrics

### Modals
- **Confirm Delete Modal**: Confirmation dialog before destructive actions
- **Ingest Modal**: Dataset ingestion progress overlay
- **Download Modal**: Model download progress

### Buttons
- Primary (filled accent color)
- Ghost/secondary (outline or subtle)
- Error/destructive (red-toned)
- Icon-only buttons (small, square)
- Size variants: small, default, large

---

## Key User Flows

### Flow 1: First-Time Setup
1. Open app → Settings page
2. Set data root directory
3. Hardware auto-detected

### Flow 2: Ingest Data → Train → Evaluate
1. Datasets page → Ingest button → Fill form → Wait for completion
2. Training page → Pick method (e.g., SFT) → Select dataset → Configure params → Submit
3. Monitor training in real-time (progress, loss curve, console)
4. Training completes → model auto-registered
5. Experiments page → Run benchmarks against new model → View scores

### Flow 3: Download from Hub → Fine-Tune → Push Back
1. Hub page → Search models → Download (auto-registers locally)
2. Training page → Pick LoRA → Select downloaded model as base → Train
3. Hub page → Push tab → Select fine-tuned model → Push to HuggingFace

### Flow 4: Remote Training
1. Clusters page → Register a Slurm cluster
2. Datasets page → Push dataset to cluster
3. Training page → Pick method → Toggle "Remote" → Select cluster → Configure resources → Submit
4. Jobs page → Monitor remote job status → Pull results when done

### Flow 5: A/B Model Comparison
1. Compare Chat page → Select Model A and Model B
2. Type prompts → Both models respond
3. Click "Prefer A" or "Prefer B" for each
4. Export preferences → Creates DPO training dataset
5. Training page → DPO method → Use exported preferences → Train improved model

---

## Data Types (What Gets Displayed)

### Dataset Entry
```
name: string
sizeBytes: number
```

### Dataset Dashboard
```
record_count: number
avg_quality_score: number
min_quality_score: number
max_quality_score: number
language_distribution: { language: string, count: number }[]
source_distribution: { source: string, count: number }[]
```

### Model Entry
```
model_name: string
model_path: string
location_type: "local" | "remote" | "both"
run_id: string | null
created_at: string (ISO date)
remote_host: string | null
remote_path: string | null
sizeBytes: number | null
```

### Job / Task
```
task_id: string
label: string
status: "running" | "completed" | "failed" | "cancelled"
training_method: string | null
stdout: string
stderr: string
progress: number (0-100)
elapsed_seconds: number
remaining_seconds: number | null
```

### Benchmark Result
```
model_path: string
average_score: number
benchmark_results: {
  benchmark_name: string
  score: number
  num_examples: number
  num_correct: number
}[]
```

### Cluster Config
```
name: string
host: string
ssh_key_path: string
partition: string
module_loads: string[]
remote_workspace: string
```

### Training Method Info
```
id: string (e.g., "sft", "dpo-train", "lora-train")
name: string (e.g., "Supervised Fine-Tuning")
description: string
category: "pre-training" | "fine-tuning" | "alignment" | "knowledge-transfer"
```

### Hardware Profile
```
gpus: { name: string, vram_mb: number, cuda_version: string }[]
cpu: { model: string, cores: number }
ram_total_mb: number
disk_total_gb: number
disk_available_gb: number
```

---

## Training Method Parameters

All methods share these **common parameters**:
- Epochs (default: 3)
- Learning rate (default varies: 1e-3 for basic, 5e-5 for fine-tuning)
- Batch size (default: 16)
- Max token length (default: 512)
- Validation split (default: 0.1)
- Precision mode: auto / fp32 / fp16 / bf16
- Optimizer: adam / adamw / sgd
- Weight decay (default: 0.0)
- Gradient accumulation steps
- Gradient checkpointing toggle
- Checkpoint frequency (every N epochs)
- Save best checkpoint toggle
- W&B project name (optional)
- TensorBoard directory (optional)
- Resume from checkpoint path (optional)

**Method-specific fields**:

| Method | Unique Fields |
|--------|--------------|
| SFT | mask_prompt_tokens, packing_enabled, base_model, tokenizer_path |
| DPO | beta (0.1), label_smoothing, reference_model_path |
| RLHF | clip_epsilon, value_loss_coeff, entropy_coeff, gamma, lambda, reward_model_path, preference_data_path |
| LoRA | rank (8), alpha (16), dropout (0.0), target_modules, base_model_path |
| QLoRA | Same as LoRA (runs quantized) |
| Distillation | teacher_model_path, temperature (2.0), alpha (0.5) |
| Domain Adaptation | source_dataset, target_dataset, adaptation_strength, adversarial_loss_weight |
| GRPO | group_size (4), kl_coeff (0.1), clip_range (0.2), temperature, reward_function_path |
| KTO | kl_coeff (0.1) |
| ORPO | orpo_lambda (0.5) |
| Multimodal | image_encoder, image_size (224), projection_dim (512) |
| RLVR | ranking_loss_weight |
| Basic | architecture_path, custom_loop_path, hooks_path, hidden_dim, num_layers, attention_heads |

---

## Navigation Structure

The sidebar groups pages into three sections:

**Workspace** (daily-use pages):
- Training
- Datasets
- Models
- Experiments (Benchmarks)

**Tools**:
- Chat
- Compare Chat
- Hub

**Operations**:
- Jobs
- Clusters
- Resources

**Bottom of sidebar**:
- Docs
- Settings

---

## Things to Keep in Mind

1. **This is a power-user tool.** Users are ML engineers who understand concepts like "learning rate," "LoRA rank," and "DPO beta." Don't oversimplify — show the controls they need.

2. **Information density matters.** Users want to see stats, metrics, and status at a glance. Don't hide important data behind extra clicks.

3. **Training forms are complex.** Some methods have 15+ fields. Use collapsible sections, sensible defaults, and clear grouping to keep forms manageable without hiding critical options.

4. **Real-time monitoring is important.** Training runs can take hours. The progress view (loss curves, console output, time estimates) needs to be clear and glanceable.

5. **The app manages a LOT of state.** Datasets, models, jobs, clusters, training configs — make navigation between these feel connected, not siloed.

6. **Local + Remote duality.** Many things exist both locally and on remote clusters (datasets, models, jobs). This dual nature should be visible but not confusing.

7. **Destructive actions need confirmation.** Deleting datasets, models, and jobs should require explicit confirmation. Never nest a button inside a button.

8. **Console output is frequent.** Many operations stream text output. The console display needs to be readable with monospace text and auto-scrolling.

9. **The palette system is flexible.** Users can pick from 40+ color palettes. Your design should look good with warm browns, cool blues, vibrant greens, muted pastels — any palette. Design with semantic tokens (text, text-secondary, bg-surface, accent, border, success, warning, error) rather than specific colors.

10. **Empty states matter.** A fresh install has no datasets, no models, no jobs. Each empty list needs a helpful message and a call-to-action to get started.

11. **File paths are everywhere.** Model paths, dataset sources, output directories — these can be very long. Handle text overflow gracefully.

12. **The sidebar should be collapsible** to give more room to content-heavy pages like training forms and dataset inspectors.

---

## What You're Designing

You have full creative control over:
- Visual style, color usage, typography
- Component appearance (buttons, cards, badges, inputs, etc.)
- Layout and spacing
- Page layouts and information hierarchy
- Interaction patterns and animations
- How forms are structured and presented
- How data is visualized
- Empty states and loading states
- The overall feel and personality of the app

The only constraint is that all the functionality described above must be accessible. Users must be able to do everything listed here — but HOW it looks and feels is entirely your decision.

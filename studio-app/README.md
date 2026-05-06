# Crucible Studio

Desktop UI for Crucible, built with Tauri 2 + React 19 + TypeScript.

## What It Does

Crucible Studio is a native desktop application that provides a visual interface for the entire Crucible ML training workflow. It calls the Python CLI under the hood — every CLI feature is accessible from the UI.

### Pages

- **Build** — Chat-first agent workspace for describing a post-training or evaluation task and letting the Crucible agent drive the workflow
- **Dashboard** — Workspace overview with active jobs, recent runs, model/dataset counts, and quick actions
- **Training** — Pick from 13 training methods, configure with an auto-saved wizard, run locally with live progress and loss curves, or submit to a remote Slurm cluster
- **Datasets** — Ingest data, filter by language/quality, browse versions, inspect samples, compare version diffs
- **Models** — Grouped model registry with version history, diffing, tagging, rollback, deletion, and merge
- **Chat** — Load a trained model and chat with configurable sampling (temperature, top-k, top-p), streaming responses
- **Interpretability** — Logit lens, activation PCA, activation patching, linear probes, SAE train/analyze, and steering workflows
- **Benchmarks** — Run evaluation benchmarks and compare model results
- **Benchmark Registry** — Create and manage custom benchmark tasks
- **Hub** — Search HuggingFace models/datasets with filters (task, library, sort), view file listings and metadata, download with progress, push trained models
- **Export** — Package models to ONNX, SafeTensors, GGUF, or HuggingFace format
- **Jobs** — Unified job queue for local and remote jobs. Remote jobs appear instantly on submit with live phase updates (connecting → provisioning → uploading → submitting). View logs, cancel pending Slurm jobs, track failures with inline error display.
- **Clusters** — Register, validate, and manage Slurm cluster SSH connections
- **Resources** — Hardware, storage, cleanup, activity, and cluster resource views
- **Settings** — Data root configuration, hardware profile, dark/light theme toggle
- **Docs** — Built-in training method documentation

### Architecture

- **Frontend:** React 19 with react-router v7 (hash routing), TypeScript strict mode, plain CSS with design tokens
- **Backend:** Tauri 2 (Rust) — shuttles JSON between TypeScript and disk, runs Python CLI as subprocess
- **State:** React Context (CrucibleContext for app state, CommandContext for command execution) + custom hooks
- **Styling:** CSS custom properties in `src/theme/` (variables.css, components.css, layout.css, reset.css). Dark/light themes. No Tailwind or CSS-in-JS.
- **Icons:** lucide-react

### Key Frontend Patterns

- `useCrucibleCommand` hook wraps all CLI invocations with progress tracking and error handling
- `useRemoteJobs` hook polls remote job JSON files every 2 seconds for live status
- Training config drafts auto-save to localStorage and merge with defaults on load
- `DatasetSelect` component shared across all 13 training forms
- `CommandFormPanel` provides consistent form layout with validation

## Quick Start

```bash
# Install Python dependencies
python3 -m pip install -e '.[serve]'

# Install frontend dependencies
npm --prefix studio-app install

# Launch the app
cd studio-app
npm run tauri dev
```

## Build

```bash
cd studio-app
npm run tauri build
```

## Project Structure

```
studio-app/
├── src/
│   ├── api/              # Tauri invoke wrappers + CLI arg builders
│   ├── components/       # Reusable UI (shared/, sidebar/)
│   ├── context/          # React Context providers
│   ├── hooks/            # Custom hooks (one per file)
│   ├── pages/            # Page components organized by domain
│   ├── theme/            # CSS design system
│   ├── types/            # Shared TypeScript type definitions
│   ├── App.tsx           # Root component
│   ├── router.tsx        # Route definitions
│   └── main.tsx          # Entry point
└── src-tauri/
    └── src/
        ├── commands/     # Rust Tauri command handlers
        ├── models.rs     # Rust data models
        └── lib.rs        # Command registration
```

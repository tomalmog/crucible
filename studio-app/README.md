# Crucible Studio

Desktop UI for Crucible, built with Tauri 2 + React 19 + TypeScript.

## What It Does

Crucible Studio is a native application for eval-gated model improvement. It
calls the Python CLI under the hood while presenting a goal-first workflow for
data, fine-tuning, evals, model health, registry, and runs.

### Core Pages

- **Fine-tuning** — Pick a model improvement goal, review launch preflight, configure the run, and execute locally or on a remote cluster
- **Evals** — Run benchmark suites, compare candidates against base models, and turn results into promotion signals
- **Model Health** — Run a curated health suite, review eval coverage, run stability, fine-tune lineage, and open targeted diagnostics only when needed
- **Datasets** — Ingest data, filter by language/quality, browse versions, inspect samples, compare version diffs
- **Model Registry** — Grouped model registry with version history, diffing, tagging, rollback, deletion, and merge
- **Eval Sets** — Manage reusable evaluation tasks and benchmark fixtures
- **Chat** — Load a trained model and chat with configurable sampling (temperature, top-k, top-p), streaming responses
- **Compare Chat** — A/B model comparison: same prompt to two models side-by-side, rate responses, export as DPO preference data
- **Hub** — Search HuggingFace models/datasets with filters (task, library, sort), view file listings and metadata, download with progress, push trained models
- **Runs** — Unified run history for local and remote work. Remote runs appear instantly on submit with live phase updates (connecting → provisioning → uploading → submitting). View logs, cancel pending Slurm runs, track failures, and review eval gates.
- **Clusters** — Register, validate, and manage Slurm cluster SSH connections
- **Resources** — Monitor local jobs, saved outputs, and storage usage
- **Settings** — Data root configuration, hardware profile, dark/light theme toggle
- **Docs** — Built-in model improvement and training method documentation

### Architecture

- **Frontend:** React 19 with react-router v7 (hash routing), TypeScript strict mode, plain CSS with design tokens
- **Backend:** Tauri 2 (Rust) — shuttles JSON between TypeScript and disk, runs Python CLI as subprocess
- **State:** React Context (CrucibleContext for app state, CommandContext for command execution) + custom hooks
- **Styling:** CSS custom properties in `src/theme/` (variables.css, components.css, layout.css, reset.css). Dark/light themes. No Tailwind or CSS-in-JS.
- **Icons:** lucide-react

### Key Frontend Patterns

- `useCrucibleCommand` hook wraps all CLI invocations with progress tracking and error handling
- `useRemoteJobs` hook polls remote job JSON files every 2 seconds for live status
- Fine-tuning config drafts auto-save to localStorage and merge with defaults on load
- `DatasetSelect` component shared across all fine-tuning forms
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

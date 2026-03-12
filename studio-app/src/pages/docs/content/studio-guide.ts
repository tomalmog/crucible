import type { DocEntry } from "../docsRegistry";

export const studioGuide: DocEntry = {
  slug: "studio-guide",
  title: "Studio Walkthrough",
  category: "Studio Guide",
  content: `
## Crucible Studio Walkthrough

Crucible Studio is the desktop application that gives you a visual interface for every part of the ML training pipeline. Here is what each section does.

### Sidebar Navigation

The left sidebar is your main navigation. It is split into two sections — **Workspace** (Training, Datasets, Models, Chat, Benchmarks, Hub) and **Tools** (Jobs, Clusters, A/B Compare, Docs, Settings). Click any item to switch views.

### Training

The training page is where you configure and launch training runs. Start by picking a training method from the method picker — each method card shows a brief description and recommended use case. After selecting a method, a configuration wizard walks you through the required settings: base model, dataset, hyperparameters, and output path. Required fields are marked with \`*\` and a validation alert lists any missing inputs before you can start. Once you launch, a live dashboard shows real-time loss charts, learning rate schedules, throughput metrics, and estimated time remaining.

The training page also includes a **Sweep** tab for running hyperparameter sweeps — define parameters, pick a search strategy, and Crucible automatically tries every combination and reports the best configuration. See the Hyperparameter Sweeps doc for details.

### Datasets

Browse all ingested datasets in one place. Each dataset shows its version history, record count, and source files. You can inspect individual samples, view field distributions, and run filters interactively. The ingest button lets you import new data directly from the UI using a file or folder picker.

### Benchmarks

Run standardized evaluation benchmarks (MMLU, HellaSwag, ARC, WinoGrande, GSM8K, TruthfulQA, HumanEval) against your trained models. Select a model, choose which benchmarks to run, and compare results across models to measure improvement after training.

### Chat

Test your trained models with an interactive chat interface. Select a model checkpoint from the dropdown, type a prompt, and see the model's response in real time with streaming output. Useful for quick sanity checks after training.

### Hub

Browse and download models from HuggingFace directly within Studio. Search by model name or task, preview model cards, and download weights to a local directory. Downloaded models are immediately available as base models for training.

### Jobs

Monitor all running and completed training jobs — both local and remote. Each job shows its status, progress, elapsed time, and key metrics. You can stop a running job, view its full logs, or delete completed jobs.

### Clusters

Register, validate, and manage Slurm cluster connections for remote training. Add a cluster by providing SSH credentials and workspace path, then validate the connection before submitting jobs.

### A/B Compare

Send the same prompt to two different models side by side and compare their responses. Rate which response is better (A, B, or Tie) and export your preferences as DPO training data for further model alignment.

### Docs

You are here. The docs section provides searchable, categorized documentation covering every feature in Crucible — from CLI commands to training methods.
`,
};

import type { DocEntry } from "../docsRegistry";

export const studioGuide: DocEntry = {
  slug: "studio-guide",
  title: "Studio Walkthrough",
  category: "Studio Guide",
  content: `
## Forge Studio Walkthrough

Forge Studio is the desktop application that gives you a visual interface for every part of the ML training pipeline. Here is what each section does.

### Sidebar Navigation

The left sidebar is your main navigation. It provides access to: **Training**, **Datasets**, **Experiments**, **Chat**, **Hub**, **Jobs**, and **Docs**. Click any item to switch views.

### Training

The training page is where you configure and launch training runs. Start by picking a training method from the method picker — each method card shows a brief description and recommended use case. After selecting a method, a configuration wizard walks you through the required settings: base model, dataset, hyperparameters, and output path. Required fields are marked with \`*\` and a validation alert lists any missing inputs before you can start. Once you launch, a live dashboard shows real-time loss charts, learning rate schedules, throughput metrics, and estimated time remaining.

The training page also includes a **Sweep** tab for running hyperparameter sweeps — define parameters, pick a search strategy, and Forge automatically tries every combination and reports the best configuration. See the Hyperparameter Sweeps doc for details.

### Datasets

Browse all ingested datasets in one place. Each dataset shows its version history, record count, and source files. You can inspect individual samples, view field distributions, and run filters interactively. The ingest button lets you import new data directly from the UI using a file or folder picker.

### Experiments

Compare training runs side by side. Select two or more experiments to see their loss curves overlaid, hyperparameters diffed, and final metrics compared in a table. This makes it easy to identify which configuration changes improved performance.

The experiments page also provides **Evaluate** (run standard benchmarks like MMLU and GSM8K against a model), **LLM Judge** (score outputs with an external LLM API), and **Cost** (view compute cost summaries across runs). Each tool validates required inputs before running.

### Chat

Test your trained models with an interactive chat interface. Select a model checkpoint from the dropdown, type a prompt, and see the model's response in real time with streaming output. Useful for quick sanity checks after training.

### Hub

Browse and download models from HuggingFace directly within Studio. Search by model name or task, preview model cards, and download weights to a local directory. Downloaded models are immediately available as base models for training.

### Jobs

Monitor all running and completed training jobs. Each job shows its status, progress, elapsed time, and key metrics. You can stop a running job, view its full logs, or jump to the corresponding experiment for detailed analysis.

### Docs

You are here. The docs section provides searchable, categorized documentation covering every feature in Forge — from CLI commands to training methods to deployment workflows.
`,
};

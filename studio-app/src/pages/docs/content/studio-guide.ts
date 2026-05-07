import type { DocEntry } from "../docsRegistry";

export const studioGuide: DocEntry = {
  slug: "studio-guide",
  title: "Studio Walkthrough",
  category: "Studio Guide",
  content: `
## Crucible Studio Walkthrough

Crucible Studio gives you a visual interface for eval-gated model improvement. Here is what each section does.

### Sidebar Navigation

The left sidebar is your main navigation. The core workflow is **Build**, **Datasets**, **Fine-tuning**, **Evals**, **Model Registry**, **Runs**, and **Model Health**. Advanced cluster and resource tools are grouped separately.

### Fine-tuning

The fine-tuning page is where you configure and launch model improvement runs. Start by picking the behavior you want to improve; Crucible maps that goal to a practical method such as SFT, LoRA, QLoRA, DPO, or domain adaptation. The wizard includes run context, a success metric, and launch preflight before you start.

The training page also includes a **Sweep** tab for running hyperparameter sweeps — define parameters, pick a search strategy, and Crucible automatically tries every combination and reports the best configuration. See the Hyperparameter Sweeps doc for details.

### Datasets

Browse all ingested datasets in one place. Each dataset shows its version history, record count, and source files. You can inspect individual samples, view field distributions, and run filters interactively. The ingest button lets you import new data directly from the UI using a file or folder picker.

### Evals

Run standardized evals against base and candidate models. Select models, choose eval sets, and compare results across candidates to decide whether a model should move forward.

### Model Health

Run the standard health check for a model before promotion. The suite starts prediction trace, representation map, and causal contrast diagnostics from one form, then records the results under Runs. Advanced diagnostics remain available for follow-up investigations.

### Chat

Test your trained models with an interactive chat interface. Select a model checkpoint from the dropdown, type a prompt, and see the model's response in real time with streaming output. Useful for quick sanity checks after training.

### Hub

Browse and download models from HuggingFace directly within Studio. Search by model name or task, preview model cards, and download weights to a local directory. Downloaded models are immediately available as base models for training.

### Runs

Monitor all running and completed work — both local and remote. Each run shows its status, project context, eval gate, progress, elapsed time, and key metrics. You can stop a running run, view logs, or delete completed records.

### Clusters

Register, validate, and manage Slurm cluster connections for remote training. Add a cluster by providing SSH credentials and workspace path, then validate the connection before submitting jobs.

### Docs

You are here. The docs section provides searchable, categorized documentation covering every feature in Crucible — from CLI commands to training methods.
`,
};

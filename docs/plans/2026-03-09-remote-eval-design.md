# Remote Evaluation System Design

## Context

Eval benchmarks currently run locally, which is impractical — MMLU alone has 14k examples and a single forward pass per example on CPU takes seconds. Models should be evaluated on the cluster where GPUs are available.

## Decisions

- **Remote-only execution**: Evals run on the cluster via Slurm. No local execution.
- **Download on-demand**: Benchmark datasets download from HuggingFace on first run, cached on cluster.
- **Simple JSON storage**: Results stored in `.crucible/evaluations/` as JSON files (no eval registry).
- **Max-samples option**: Full dataset by default, optional `--max-samples` for quick sanity checks.
- **HF model support**: Model loader handles both Crucible `.pt` checkpoints and HuggingFace model directories.

## Architecture

### Flow

1. User picks model (from registry), benchmarks (checkboxes), optional max-samples
2. UI submits remote eval job to selected cluster via Slurm
3. On cluster: `crucible eval --model-path <path> --benchmarks <list> [--max-samples N]`
4. Datasets download on-demand via HuggingFace `datasets` library
5. Job writes eval JSON to stdout
6. Results pulled back, stored in `.crucible/evaluations/`
7. UI displays parsed results

### Changes

| Area | Change |
|------|--------|
| CLI (`eval_command.py`) | Add `--max-samples` flag, pass through to benchmarks |
| Benchmarks (`src/eval/benchmarks/*.py`) | Accept `max_samples` param, slice dataset |
| UI (`EvalResultsView.tsx`) | Cluster submit form (cluster selector, benchmark checkboxes, max-samples) |
| Remote execution | Reuse `remote_job_submitter` to submit eval as Slurm job |
| Results | Pull eval JSON from cluster, store locally |

### Unchanged

- Benchmark implementations (add max_samples slicing only)
- EvaluationHarness and result JSON format
- LLM Judge (already remote via external API)
- `.crucible/evaluations/` storage

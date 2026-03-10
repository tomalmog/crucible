# Remote Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run benchmark evaluations on remote clusters via Slurm instead of locally.

**Architecture:** Add `--max-samples` to the eval CLI, add an eval dispatch path in the agent entry script, add `remote eval-submit` CLI command that reuses the existing job submission pipeline, and rebuild the UI eval form as a cluster submit form.

**Tech Stack:** Python (CLI/backend), TypeScript/React (Studio UI), Slurm (job scheduling)

---

### Task 1: Add `--max-samples` to benchmarks

**Files:**
- Modify: `src/eval/benchmark_runner.py`
- Modify: `src/eval/benchmarks/mmlu.py`
- Modify: `src/eval/benchmarks/humaneval.py`
- Modify: `src/eval/benchmarks/gsm8k.py`
- Modify: `src/eval/benchmarks/hellaswag.py`
- Modify: `src/eval/benchmarks/arc.py`
- Modify: `src/eval/benchmarks/truthfulqa.py`
- Modify: `src/eval/benchmarks/winogrande.py`
- Modify: `src/eval/evaluation_harness.py`
- Modify: `src/cli/eval_command.py`

**Step 1: Update `run_benchmarks()` signature**

In `src/eval/benchmark_runner.py`, add `max_samples: int | None = None` parameter to `run_benchmarks()`. Pass it to each benchmark function call:

```python
def run_benchmarks(
    model_path: str,
    benchmarks: list[str],
    base_model_path: str | None = None,
    max_samples: int | None = None,
) -> EvaluationResult:
    ...
    for name in benchmarks:
        if name not in benchmark_map:
            continue
        result = benchmark_map[name](model_path, max_samples=max_samples)
        results.append(result)
    ...
    if base_model_path:
        for name in benchmarks:
            if name not in benchmark_map:
                continue
            base_results.append(benchmark_map[name](base_model_path, max_samples=max_samples))
```

**Step 2: Update each benchmark to accept and use `max_samples`**

Each `run_*` function gets `max_samples: int | None = None` parameter. After loading examples, slice: `if max_samples: examples = examples[:max_samples]`.

Pattern for all 7 benchmarks (example for mmlu.py):

```python
def run_mmlu(model_path: str, *, max_samples: int | None = None) -> BenchmarkResult:
    eval_model = load_eval_model(model_path)
    examples = _load_mmlu_examples()
    if max_samples:
        examples = examples[:max_samples]
    ...
```

Apply same pattern to: `run_humaneval`, `run_gsm8k`, `run_hellaswag`, `run_arc`, `run_truthfulqa`, `run_winogrande`.

**Step 3: Update `EvaluationHarness.evaluate()` to pass `max_samples`**

```python
def evaluate(
    self,
    model_path: str,
    benchmarks: list[str] | None = None,
    base_model_path: str | None = None,
    max_samples: int | None = None,
) -> EvaluationResult:
    if benchmarks is None:
        benchmarks = list(AVAILABLE_BENCHMARKS)
    result = run_benchmarks(model_path, benchmarks, base_model_path, max_samples=max_samples)
    self._store_result(result)
    return result
```

**Step 4: Add `--max-samples` to CLI**

In `src/cli/eval_command.py`:

```python
# In add_eval_command:
parser.add_argument("--max-samples", type=int, default=None, help="Max examples per benchmark")

# In run_eval_command:
result = harness.evaluate(
    model_path=args.model_path,
    benchmarks=benchmarks,
    base_model_path=args.base_model,
    max_samples=args.max_samples,
)
```

**Step 5: Verify**

Run: `pytest tests/unit/eval/ -x -v`

---

### Task 2: Add eval dispatch to agent entry script

**Files:**
- Modify: `src/serve/agent_entry_script.py`

**Step 1: Add eval branch in the entry script**

The agent entry script currently always calls `dispatch_training()`. Add an eval code path when `method == "eval"`:

In the `main()` function of `ENTRY_SCRIPT`, after the torch import block and before the `dispatch_training` try/except, add:

```python
    if method == "eval":
        print("CRUCIBLE_AGENT: Running evaluation...", flush=True)
        from eval.benchmark_runner import run_benchmarks, AVAILABLE_BENCHMARKS
        benchmarks_str = method_args.get("benchmarks", "")
        benchmarks = [b.strip() for b in benchmarks_str.split(",") if b.strip()] if benchmarks_str else list(AVAILABLE_BENCHMARKS)
        model_path = method_args.get("model_path", "")
        base_model_path = method_args.get("base_model_path") or None
        max_samples_val = method_args.get("max_samples")
        max_samples = int(max_samples_val) if max_samples_val else None
        try:
            eval_result = run_benchmarks(model_path, benchmarks, base_model_path, max_samples=max_samples)
            result_data = {
                "status": "completed",
                "job_type": "eval",
                "model_path": eval_result.model_path,
                "average_score": eval_result.average_score,
                "benchmarks": [
                    {"name": r.benchmark_name, "score": r.score,
                     "num_examples": r.num_examples, "correct": r.correct}
                    for r in eval_result.benchmark_results
                ],
            }
            if eval_result.base_results:
                result_data["base_benchmarks"] = [
                    {"name": r.benchmark_name, "score": r.score,
                     "num_examples": r.num_examples, "correct": r.correct}
                    for r in eval_result.base_results
                ]
        except Exception as exc:
            import traceback
            result_data = {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=2)
        if result_data["status"] == "failed":
            print(f"CRUCIBLE_AGENT_ERROR: {result_data['error']}", file=sys.stderr)
            sys.exit(1)
        print(f"CRUCIBLE_AGENT: Evaluation complete. Average score: {result_data.get('average_score', 'N/A')}")
        print("CRUCIBLE_AGENT_COMPLETE")
        return
```

Place this block right before line `_ensure_output_dir(method_args)` (since eval doesn't need output_dir/dataset setup).

**Step 2: Verify**

Manual review — entry script is a string constant, no unit test needed for the string itself.

---

### Task 3: Add `remote eval-submit` CLI command

**Files:**
- Modify: `src/cli/remote_command.py`
- Modify: `src/cli/remote_command_handlers.py`
- Modify: `src/serve/remote_job_submitter.py`

**Step 1: Add `eval-submit` subcommand in `remote_command.py`**

In `add_remote_command()`, add a new subparser alongside the existing `submit` subparser:

```python
eval_sub = sub.add_parser("eval-submit", help="Submit evaluation job to cluster")
eval_sub.add_argument("--cluster", required=True)
eval_sub.add_argument("--model-path", required=True, help="Path to model on cluster")
eval_sub.add_argument("--benchmarks", default="mmlu,gsm8k,hellaswag,arc,truthfulqa,winogrande,humaneval", help="Comma-separated benchmarks")
eval_sub.add_argument("--base-model", default="", help="Optional base model path for comparison")
eval_sub.add_argument("--max-samples", type=int, default=None, help="Max examples per benchmark")
eval_sub.add_argument("--partition", default="")
eval_sub.add_argument("--nodes", type=int, default=1)
eval_sub.add_argument("--gpus-per-node", type=int, default=1)
eval_sub.add_argument("--gpu-type", default="")
eval_sub.add_argument("--cpus-per-task", type=int, default=4)
eval_sub.add_argument("--memory", default="32G")
eval_sub.add_argument("--time-limit", default="04:00:00")
```

Add dispatch in `run_remote_command()`:

```python
if sub == "eval-submit":
    return _handle_eval_submit(client, args)
```

**Step 2: Add handler in `remote_command_handlers.py`**

```python
def _handle_eval_submit(
    client: CrucibleClient, args: argparse.Namespace,
) -> int:
    from serve.remote_job_submitter import submit_remote_eval_job
    from core.slurm_types import SlurmResourceConfig

    resources = SlurmResourceConfig(
        partition=args.partition,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        gpu_type=args.gpu_type,
        cpus_per_task=args.cpus_per_task,
        memory=args.memory,
        time_limit=args.time_limit,
    )
    method_args = {
        "model_path": args.model_path,
        "benchmarks": args.benchmarks,
        "max_samples": args.max_samples,
    }
    if args.base_model:
        method_args["base_model_path"] = args.base_model
    record = submit_remote_eval_job(
        data_root=client._config.data_root,
        cluster_name=args.cluster,
        method_args=method_args,
        resources=resources,
    )
    print(f"job_id={record.job_id}")
    print(f"slurm_job_id={record.slurm_job_id}")
    return 0
```

**Step 3: Add `submit_remote_eval_job()` in `remote_job_submitter.py`**

Simplified version of `submit_remote_job()` — no dataset resolution needed:

```python
def submit_remote_eval_job(
    data_root: Path,
    cluster_name: str,
    method_args: dict[str, object],
    resources: SlurmResourceConfig,
) -> RemoteJobRecord:
    """Submit an evaluation job to a remote Slurm cluster."""
    cluster = load_cluster(data_root, cluster_name)
    job_id = generate_job_id()
    workdir = f"{cluster.remote_workspace}/{job_id}"

    tarball = build_agent_tarball(
        cache_dir=data_root / "cache" / "agent-bundles",
    )

    config_payload = {
        "method": "eval",
        "method_args": method_args,
        "result_output": "result.json",
    }

    ts = now_iso()
    record = RemoteJobRecord(
        job_id=job_id,
        slurm_job_id="",
        cluster_name=cluster_name,
        training_method="eval",
        state="submitting",
        submitted_at=ts,
        updated_at=ts,
        remote_output_dir=workdir,
        submit_phase="Preparing submission...",
    )
    save_remote_job(data_root, record)

    try:
        _update_phase(data_root, job_id, "Connecting to cluster...")
        with SshSession(cluster) as session:
            cluster = _resolve_cluster_workspace(cluster, session)
            workdir = f"{cluster.remote_workspace}/{job_id}"
            update_remote_job_state(
                data_root, job_id, "submitting",
                remote_output_dir=workdir,
            )
            session.mkdir_p(workdir)
            _update_phase(data_root, job_id, "Provisioning environment...")
            ensure_remote_env(session)
            _update_phase(data_root, job_id, "Uploading eval bundle...")
            _upload_bundle(session, tarball, workdir)
            _upload_config(session, config_payload, workdir)
            script = _generate_script(
                cluster, resources, job_id, "eval",
            )
            _upload_script(session, script, workdir)
            _update_phase(data_root, job_id, "Submitting to Slurm...")
            slurm_job_id = _submit_sbatch(session, workdir)
    except Exception as exc:
        update_remote_job_state(
            data_root, job_id, "failed",
            submit_phase=f"Failed: {exc}",
        )
        raise

    return update_remote_job_state(
        data_root, job_id, "running",
        slurm_job_id=slurm_job_id,
        remote_log_path=f"{workdir}/slurm-{slurm_job_id}.out",
        submit_phase="",
    )
```

**Step 4: Handle eval jobs in job state completion**

In `src/serve/remote_job_state.py`, the `_handle_completed_terminal` and `_build_state_transition_fields` functions try to extract `model_path` and auto-register models. For eval jobs (`training_method == "eval"`), skip model registration. Add early return:

```python
def _handle_completed_terminal(data_root, record):
    if record.training_method == "eval":
        return  # Eval jobs don't produce models
    ...existing code...
```

And in `_build_state_transition_fields`, wrap the model extraction in:

```python
    if crucible_state == "completed" and not record.model_path_remote:
        if record.training_method != "eval":
            ...existing model extraction code...
```

**Step 5: Verify**

Run: `pytest tests/unit/cli/test_remote_command.py tests/unit/serve/test_remote_job_submitter.py -x -v`

---

### Task 4: Add `buildRemoteEvalArgs` to frontend API

**Files:**
- Modify: `studio-app/src/api/commandArgs.ts`

**Step 1: Add the eval arg builder**

```typescript
/** Build CLI args for `crucible remote eval-submit ...`. */
export function buildRemoteEvalArgs(
  cluster: string,
  modelPath: string,
  benchmarks: string,
  opts: {
    baseModel?: string;
    maxSamples?: string;
    partition?: string;
    gpusPerNode?: string;
    gpuType?: string;
    cpusPerTask?: string;
    memory?: string;
    timeLimit?: string;
  } = {},
): string[] {
  const args = [
    "remote", "eval-submit",
    "--cluster", cluster,
    "--model-path", modelPath,
    "--benchmarks", benchmarks,
  ];
  if (opts.baseModel) args.push("--base-model", opts.baseModel);
  if (opts.maxSamples) args.push("--max-samples", opts.maxSamples);
  if (opts.partition) args.push("--partition", opts.partition);
  if (opts.gpusPerNode) args.push("--gpus-per-node", opts.gpusPerNode);
  if (opts.gpuType) args.push("--gpu-type", opts.gpuType);
  if (opts.cpusPerTask) args.push("--cpus-per-task", opts.cpusPerTask);
  if (opts.memory) args.push("--memory", opts.memory || "32G");
  if (opts.timeLimit) args.push("--time-limit", opts.timeLimit || "04:00:00");
  return args;
}
```

**Step 2: Verify**

Run: `cd studio-app && npx tsc --noEmit`

---

### Task 5: Rebuild EvalResultsView as cluster submit form

**Files:**
- Modify: `studio-app/src/pages/experiments/EvalResultsView.tsx`

**Step 1: Rewrite EvalResultsView**

Replace the current local-execution form with a cluster submit form. The form should have:
- Model select (already done — ModelSelect component)
- Base model select (optional)
- Benchmark checkboxes (all 7 benchmarks, default all checked)
- Max samples input (optional, empty = full dataset)
- Cluster select dropdown (reuse `listClusters` API)
- Slurm resource fields: partition, GPUs, memory, time limit
- Submit button that calls `buildRemoteEvalArgs()` and `startCrucibleCommand()`
- On success, navigate to `/jobs` page

Key imports to add: `useNavigate` from react-router, `listClusters` from remoteApi, `startCrucibleCommand` from studioApi, `buildRemoteEvalArgs` from commandArgs.

Remove: `PathInput` import (already removed), local command execution via `useCrucibleCommand`.

The benchmark list should use checkboxes with `AVAILABLE_BENCHMARKS = ["mmlu", "gsm8k", "hellaswag", "arc", "truthfulqa", "winogrande", "humaneval"]`.

**Step 2: Verify**

Run: `cd studio-app && npx tsc --noEmit`

---

### Task 6: Update job detail view for eval results

**Files:**
- Modify: `studio-app/src/pages/jobs/JobResultDetail.tsx`

**Step 1: Add eval result display**

Check `JobResultDetail.tsx` for how it currently renders job results. Add a branch for eval jobs (`training_method === "eval"`) that displays:
- Average score
- Per-benchmark scores table (benchmark name, score, examples, correct)
- Base model comparison if present

Look at the existing result parsing — if `result.json` contains `job_type: "eval"`, render the eval-specific view instead of the training result view.

**Step 2: Verify**

Run: `cd studio-app && npx tsc --noEmit`

---

### Task 7: Commit and verify

**Step 1: Run full test suite**

```bash
pytest tests/unit/eval/ -x -v
pytest tests/unit/cli/test_remote_command.py -x -v
cd studio-app && npx tsc --noEmit
```

**Step 2: Commit**

```bash
git add -A
git commit -m "remote eval: run benchmarks on cluster via Slurm with max-samples support"
```

# CLAUDE.md

All codebase standards, lessons, and guardrails live in **AGENTS.md**. Read and follow it.

## What Crucible Is

End-to-end ML training platform. Three components:
- **Python CLI/SDK** (`src/cli/main.py`, `src/crucible.py`) — 50+ commands
- **Tauri Studio** (`studio-app/`) — React 19 desktop app, the primary product
- **MCP Server** (`src/serve/mcp_server.py`) — 27 tools for Claude Code integration

## Architecture

```
User → Studio UI (React/Tauri) → CLI subprocess → Python SDK → Runner → Result
User → CLI directly → Python SDK → Runner → Result
User → Claude Code → MCP Server → Python SDK → Runner → Result
```

Training on remote GPU clusters:
```
Submit → SSH/Slurm → agent_entry_script.py → dispatch_training() → trl/peft Trainer → result.json
```

## Training Methods (13 total)

HuggingFace models use `trl` trainers (SFTTrainer, DPOTrainer, etc.) with `peft` for LoRA/QLoRA.
Crucible .pt models use the custom training loop.

| Method | CLI | trl Trainer | Data Format |
|--------|-----|-------------|-------------|
| train | `crucible train` | Custom loop | `{"text": "..."}` |
| sft | `crucible sft` | SFTTrainer | `{"prompt": "...", "response": "..."}` |
| lora-train | `crucible lora-train` | SFTTrainer + peft | `{"prompt": "...", "response": "..."}` |
| qlora-train | `crucible qlora-train` | SFTTrainer + peft + bitsandbytes | `{"prompt": "...", "response": "..."}` |
| dpo-train | `crucible dpo-train` | DPOTrainer | `{"prompt": "...", "chosen": "...", "rejected": "..."}` |
| kto-train | `crucible kto-train` | KTOTrainer | `{"prompt": "...", "response": "...", "is_desirable": bool}` |
| orpo-train | `crucible orpo-train` | ORPOTrainer/DPOTrainer | `{"prompt": "...", "chosen": "...", "rejected": "..."}` |
| grpo-train | `crucible grpo-train` | SFTTrainer (fallback) | `{"prompt": "..."}` |
| rlvr-train | `crucible rlvr-train` | SFTTrainer (fallback) | `{"prompt": "...", "solution": "..."}` |
| rlhf-train | `crucible rlhf-train` | SFTTrainer | `{"prompt": "...", "chosen": "...", "rejected": "..."}` |
| distill | `crucible distill` | Custom loop | `{"text": "..."}` |
| domain-adapt | `crucible domain-adapt` | SFTTrainer | `{"text": "..."}` |
| multimodal-train | `crucible multimodal-train` | Custom loop | `{"text": "...", "image_path": "..."}` |

## Key Paths

- CLI entry: `src/cli/main.py`
- SDK: `src/crucible.py`, `src/store/dataset_sdk.py`
- Training dispatch: `src/core/training_methods.py` (single source of truth)
- trl training base: `src/serve/trl_training_base.py` (shared helpers for all trl runners)
- Options → TrainingOptions: `src/core/training_types.py` (`options_to_training_options()`)
- Remote agent: `src/serve/agent_entry_script.py`
- Remote env: `src/serve/remote_env_setup.py` (conda env with torch, trl, peft, bitsandbytes)
- MCP server: `src/serve/mcp_server.py`
- Model registry: `src/store/model_registry.py`
- Job store: `src/store/job_store.py`
- Config: `src/core/config.py`

## MCP Server (33 tools)

`crucible mcp-server` starts the MCP server for Claude Code integration.

**Configure in `~/.claude/mcp.json`:**
```json
{
  "mcpServers": {
    "crucible": {
      "command": "python",
      "args": ["-m", "cli.main", "mcp-server"],
      "env": { "PYTHONPATH": "<path-to-crucible>/src" }
    }
  }
}
```

**Tools by category:**
- **Datasets:** list_datasets, ingest_dataset, delete_dataset, push_dataset, list_remote_datasets, curate_dataset, generate_synthetic_data
- **Models:** list_models, register_model, delete_model, pull_model, merge_models, lora_merge
- **Training:** train (local, any of 13 methods), submit_remote_training (GPU cluster), run_sweep
- **Jobs:** list_jobs, job_status, job_logs, job_result, cancel_job, delete_job
- **Eval:** run_benchmark, submit_remote_eval
- **Interp:** run_interp (all 8 tools: logit-lens, activation-pca, activation-patching, linear-probe, sae-train, sae-analyze, steer-compute, steer-apply)
- **Export:** export_model (onnx, safetensors, gguf, hf)
- **Chat:** chat, ab_chat
- **Clusters:** list_clusters, cluster_info
- **Hub:** hub_search_models, hub_download_model
- **System:** hardware_profile

The MCP server's `instructions` field contains the full reference for data formats,
training methods, arguments, and workflows. Claude reads this at session start.

## Auto-Registration

All training commands auto-register the model in the model registry after training
completes. The `--model-name` flag controls the registry name. If not provided, the
name is derived from the output directory.

## Hard Rules

- **NEVER add fallbacks.** If something fails, it should fail loudly.
- **NEVER commit or push without being explicitly asked.**
- **NEVER add Co-Authored-By or any AI attribution to commits.**
- **Use existing libraries.** trl for HF training, peft for LoRA, bitsandbytes for quantization.
  Don't rewrite what a well-maintained package already does.

## Remote Dependencies

The remote conda env (`remote_env_setup.py`) installs:
`torch`, `trl`, `peft`, `bitsandbytes`, `transformers`, `accelerate`, `safetensors`,
`datasets`, `pyyaml`, `numpy`, `matplotlib`, `tokenizers`

## Testing

- 953+ automated tests (unit + integration)
- 19 training matrix tests (every method × model type)
- 14 interp/eval/chat tests
- Round-trip tests: train → chat → verify output
- Regression tests for all known bug classes

# Crucible

Model improvement platform for AI startups. Upload private data, build evals,
fine-tune open-source models, compare candidates, and promote the winner with
lineage and reproducibility.

## What it does

- **Eval-gated fine-tuning** — SFT, LoRA/QLoRA, DPO, and domain adaptation are the primary workflows. Every candidate is tracked against a success metric.
- **Private compute execution** — Run locally, on SSH GPU boxes, or on Slurm clusters. Crucible provisions remote environments and records the run lifecycle.
- **Model registry and runs** — Track datasets, candidates, eval results, artifacts, and promotion context in one local workspace.
- **Model health** — Surface eval coverage, failed runs, lineage gaps, and promotion blockers before a model ships.
- **Exports and handoff** — Export ONNX, SafeTensors, GGUF, or HuggingFace format.
- **Agentic operator** — The Studio agent can inspect data/models/runs, draft a fine-tuning plan, launch approved work, and summarize results.
- **Advanced research mode** — GRPO, RLHF, RLVR, KTO, ORPO, distillation, multimodal, and targeted mechanistic diagnostics are available when teams need them.

## Install

```bash
pip install -e ".[serve]"
```

For the desktop app:
```bash
cd studio-app && npm install && npm run tauri dev
```

## Quick start

```bash
# Ingest data
crucible ingest ./data.jsonl --dataset my-data

# Fine-tune
crucible sft --dataset my-data --base-model gpt2 --output-dir ./output --model-name my-model

# Chat
crucible chat --model-path ./output/model.pt --prompt "hello"

# Eval
crucible eval --model-path ./output/model.pt --benchmarks mmlu,arc

# Train on a remote GPU
crucible remote register-cluster --name gpu-box --host root@1.2.3.4 --backend ssh
crucible remote dataset-push --cluster gpu-box --dataset my-data
crucible remote submit --cluster gpu-box --method sft --method-args '{"dataset_name":"my-data","base_model":"Qwen/Qwen2.5-1.5B"}'
```

## MCP server

For Claude Code integration:

```bash
crucible mcp-server
```

Add to `~/.claude/mcp.json`:
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

34 tools: train, eval, interp, export, chat, dataset/model management, remote cluster ops, HuggingFace hub search and download.

## Training methods

| Method | Command | Data format |
|--------|---------|-------------|
| SFT | `crucible sft` | `{"prompt": "...", "response": "..."}` |
| LoRA | `crucible lora-train` | `{"prompt": "...", "response": "..."}` |
| QLoRA | `crucible qlora-train` | `{"prompt": "...", "response": "..."}` |
| DPO | `crucible dpo-train` | `{"prompt": "...", "chosen": "...", "rejected": "..."}` |
| KTO | `crucible kto-train` | `{"prompt": "...", "response": "...", "is_desirable": bool}` |
| ORPO | `crucible orpo-train` | `{"prompt": "...", "chosen": "...", "rejected": "..."}` |
| GRPO | `crucible grpo-train` | `{"prompt": "..."}` |
| RLVR | `crucible rlvr-train` | `{"prompt": "...", "solution": "..."}` |
| RLHF | `crucible rlhf-train` | preference pairs |
| Distillation | `crucible distill` | `{"text": "..."}` |
| Domain adapt | `crucible domain-adapt` | `{"text": "..."}` |
| Multimodal | `crucible multimodal-train` | `{"text": "...", "image_path": "..."}` |
| Basic | `crucible train` | `{"text": "..."}` |

## Stack

- Python, PyTorch, trl, peft, bitsandbytes
- Tauri 2, React 19, TypeScript
- MCP (Model Context Protocol)

## Requirements

- Python >= 3.11
- PyTorch >= 2.6
- Node.js >= 18 + Rust toolchain (for desktop app)

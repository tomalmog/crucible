# Crucible

ML training platform. Train models, run evals, ship — from a desktop app, CLI, or AI agent.

## What it does

- **13 training methods** — SFT, LoRA, QLoRA, DPO, KTO, ORPO, GRPO, RLHF, RLVR, distillation, domain adaptation, multimodal, basic training. HuggingFace models use [trl](https://github.com/huggingface/trl) + [peft](https://github.com/huggingface/peft) under the hood.
- **Cloud & remote GPU training** — SSH into any cloud GPU (vast.ai, Lambda, RunPod) or submit to Slurm clusters (university/enterprise). Auto-provisions conda env with torch + dependencies.
- **Eval benchmarks** — MMLU, GSM8K, HellaSwag, ARC, TruthfulQA, WinoGrande, HumanEval.
- **Interpretability** — Logit lens, activation PCA, activation patching, linear probes, SAE training/analysis, steering vectors.
- **Model export** — ONNX, SafeTensors, GGUF, HuggingFace format.
- **Desktop app** — Tauri + React. Training wizard, job monitoring, model registry, dataset browser, benchmark comparison, interp visualizations.
- **AI integration** — MCP server with 34 tools. Claude Code or the built-in sidebar agent can train models, run evals, manage data, and monitor jobs.

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

# Train
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

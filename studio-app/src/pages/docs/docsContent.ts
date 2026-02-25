/** Markdown documentation content for Forge Studio. */

export const DOCS_CONTENT = `
# Forge Documentation

---

## Training Methods

### Basic Training {#train}

Standard supervised training from scratch. Trains a new model on raw text data from a Forge dataset.

**Required:**
- \`--dataset\` — Forge dataset name
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--epochs\` — Number of training epochs (default: 3)
- \`--learning-rate\` — Optimizer learning rate (default: 0.001)
- \`--batch-size\` — Batch size (default: 32)
- \`--max-token-length\` — Maximum sequence length (default: 256)
- \`--precision-mode\` — fp32, fp16, or bf16
- \`--optimizer-type\` — adamw, adam, or sgd
- \`--checkpoint-every-epochs\` — Save checkpoint every N epochs (0 = off)
- \`--resume-checkpoint-path\` — Resume from a previous checkpoint

**When to use:** Building a language model from scratch on your own corpus. The tokenizer is fitted from your dataset records.

---

### SFT (Supervised Fine-Tuning) {#sft}

Fine-tune a pre-trained model on instruction/response pairs. The model learns to follow instructions by training on structured prompt-completion examples.

**Required:**
- \`--sft-data-path\` — Path to JSONL file with \`{"prompt": "...", "response": "..."}\` pairs
- \`--base-model\` — HuggingFace model ID (e.g. \`gpt2\`, \`meta-llama/Llama-2-7b\`)
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--mask-prompt-tokens\` — Exclude prompt tokens from loss (default: true)
- \`--packing\` — Pack multiple short examples into one sequence for efficiency
- \`--tokenizer-path\` — Override tokenizer (auto-loaded from base model)

**Data format:**
\`\`\`json
{"prompt": "Summarize this article: ...", "response": "The article discusses..."}
{"prompt": "Translate to French: Hello", "response": "Bonjour"}
\`\`\`

**When to use:** You have a pre-trained model and want it to follow instructions, answer questions, or perform specific tasks.

---

### DPO (Direct Preference Optimization) {#dpo-train}

Train a model to prefer better responses over worse ones using paired preference data, without needing a separate reward model.

**Required:**
- \`--dpo-data-path\` — Path to JSONL file with chosen/rejected pairs
- \`--base-model\` — HuggingFace model ID or path
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--beta\` — KL penalty strength (default: 0.1). Higher = more conservative
- \`--label-smoothing\` — Smooth preference labels (default: 0.0)
- \`--reference-model-path\` — Separate reference model (defaults to base model)

**Data format:**
\`\`\`json
{"prompt": "Explain gravity", "chosen": "Gravity is a fundamental force...", "rejected": "Gravity is when stuff falls down"}
\`\`\`

**When to use:** You have preference data (human rankings, A/B comparisons) and want to align a model without training a reward model first.

---

### RLHF (Reinforcement Learning from Human Feedback) {#rlhf-train}

Train a policy model using PPO with a reward model. The classic alignment approach: train a reward model on preferences, then optimize the policy against it.

**Required:**
- \`--policy-model-path\` — Path to the policy model weights
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--reward-model-path\` — Path to trained reward model
- \`--train-reward-model\` — Train a reward model first from preference data
- \`--preference-data-path\` — JSONL with preference pairs (for reward model training)
- \`--clip-epsilon\` — PPO clip range (default: 0.2)
- \`--ppo-epochs\` — PPO update epochs per batch (default: 4)
- \`--entropy-coeff\` — Entropy bonus coefficient (default: 0.01)

**When to use:** Full RLHF pipeline. Use DPO instead for a simpler alternative that doesn't require a separate reward model.

---

### LoRA (Low-Rank Adaptation) {#lora-train}

Parameter-efficient fine-tuning that freezes the base model and trains small low-rank adapter matrices. Dramatically reduces memory and compute requirements.

**Required:**
- \`--lora-data-path\` — Path to JSONL training data
- \`--base-model-path\` — HuggingFace model ID or local checkpoint path
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--lora-rank\` — Rank of LoRA matrices (default: 8). Higher = more capacity but more parameters
- \`--lora-alpha\` — Scaling factor (default: 16). Effective scale = alpha / rank
- \`--lora-dropout\` — Dropout on LoRA inputs (default: 0.0)
- \`--lora-target-modules\` — Comma-separated module names to inject LoRA into (default: q_proj,v_proj)
- \`--tokenizer-path\` — Override tokenizer (auto-loaded from base model)

**Data format:**
\`\`\`json
{"text": "Full training text example"}
{"prompt": "Question", "response": "Answer"}
\`\`\`

**When to use:** Fine-tuning large models on limited hardware. LoRA trains ~0.1% of parameters while achieving comparable quality to full fine-tuning.

---

### Distillation {#distill}

Transfer knowledge from a large teacher model to a smaller student model. The student learns to match the teacher's output distribution.

**Required:**
- \`--dataset\` — Forge dataset name (used as training data)
- \`--teacher-model-path\` — Path to the teacher model
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--temperature\` — Softmax temperature for soft targets (default: 2.0)
- \`--alpha\` — Weight for distillation loss vs. standard loss (default: 0.5)

**When to use:** You have a large accurate model and want a smaller, faster model that retains most of its quality. Common for deploying to edge devices or reducing inference cost.

---

### Domain Adaptation {#domain-adapt}

Adapt a pre-trained model to a new domain by continuing training on domain-specific data, with optional drift detection to prevent catastrophic forgetting.

**Required:**
- \`--dataset\` — Forge dataset name (domain-specific data)
- \`--base-model-path\` — Pre-trained model to adapt
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--reference-data-path\` — Reference data for drift detection (optional)

**When to use:** Taking a general-purpose model and specializing it for a specific domain (medical, legal, code, etc.) while preserving general capabilities.

---

### GRPO (Group Relative Policy Optimization) {#grpo-train}

Reward-based optimization that samples multiple responses per prompt and uses relative rankings within each group to compute advantages.

**Required:**
- \`--grpo-data-path\` — Path to JSONL file with prompts
- \`--base-model\` — HuggingFace model ID or path
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--group-size\` — Number of responses to sample per prompt (default: 4)
- \`--reward-function-path\` — Python file defining a reward function

**Data format:**
\`\`\`json
{"prompt": "Write a haiku about coding"}
{"prompt": "Solve: 2x + 3 = 7"}
\`\`\`

**When to use:** When you have a reward function (rule-based or model-based) and want to improve response quality through self-play-style training.

---

### QLoRA (Quantized LoRA) {#qlora-train}

Combines 4-bit quantization with LoRA for extremely memory-efficient fine-tuning. Enables fine-tuning large models on consumer GPUs.

**Required:**
- \`--qlora-data-path\` — Path to JSONL training data
- \`--base-model-path\` — HuggingFace model ID or local checkpoint
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--quantization-bits\` — 4 or 8 bit quantization (default: 4)
- \`--qlora-type\` — Quantization method: nf4 or fp4 (default: nf4)
- \`--double-quantize\` — Double quantization for extra memory savings
- \`--lora-rank\`, \`--lora-alpha\`, \`--lora-dropout\` — Same as LoRA

**Data format:** Same as LoRA.

**When to use:** Fine-tuning 7B+ parameter models on a single GPU with 16-24GB VRAM. Trades some precision for dramatically reduced memory usage.

---

### KTO (Kahneman-Tversky Optimization) {#kto-train}

Preference optimization using unpaired binary feedback (thumbs up / thumbs down) rather than requiring paired chosen/rejected examples.

**Required:**
- \`--kto-data-path\` — Path to JSONL file with binary feedback
- \`--base-model\` — HuggingFace model ID or path
- \`--output-dir\` — Directory for output artifacts

**Data format:**
\`\`\`json
{"prompt": "Explain photosynthesis", "response": "Plants convert sunlight...", "label": true}
{"prompt": "Write a poem", "response": "Roses are red...", "label": false}
\`\`\`

**When to use:** When you have binary feedback data (good/bad) rather than paired preference data. Easier to collect than DPO-style data.

---

### ORPO (Odds Ratio Preference Optimization) {#orpo-train}

Combines SFT and preference optimization in a single training step. Uses odds ratios to contrast chosen and rejected responses without a reference model.

**Required:**
- \`--orpo-data-path\` — Path to JSONL file with preference pairs
- \`--base-model\` — HuggingFace model ID or path
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--orpo-beta\` — Weight for the odds ratio loss component

**Data format:** Same as DPO (prompt / chosen / rejected).

**When to use:** When you want SFT + alignment in one pass. Simpler pipeline than DPO since no reference model is needed.

---

### Multimodal Training {#multimodal-train}

Fine-tune vision-language models on paired image-text data. Supports image captioning, visual question answering, and other multimodal tasks.

**Required:**
- \`--multimodal-data-path\` — Path to JSONL file with image-text pairs
- \`--base-model\` — HuggingFace model ID or path
- \`--output-dir\` — Directory for output artifacts

**Data format:**
\`\`\`json
{"image_path": "/path/to/image.jpg", "text": "A cat sitting on a windowsill"}
{"image_path": "/path/to/chart.png", "prompt": "Describe this chart", "response": "The bar chart shows..."}
\`\`\`

**When to use:** Training or fine-tuning models that process both images and text (e.g., image captioning, VQA, document understanding).

---

### RLVR (RL with Verifiable Rewards) {#rlvr-train}

Reinforcement learning with automatically verifiable rewards. Ideal for tasks where correctness can be checked programmatically (math, code, logic).

**Required:**
- \`--rlvr-data-path\` — Path to JSONL file with verifiable tasks
- \`--base-model\` — HuggingFace model ID or path
- \`--output-dir\` — Directory for output artifacts

**Data format:**
\`\`\`json
{"prompt": "What is 15 * 23?", "answer": "345"}
{"prompt": "Write a function that reverses a string", "test_code": "assert reverse('hello') == 'olleh'"}
\`\`\`

**When to use:** Training models on tasks with objective correctness criteria — math problems, coding challenges, logical reasoning, fact-based QA.

---

## Common Options

These options are available across all training methods:

| Option | Default | Description |
|--------|---------|-------------|
| \`--epochs\` | 3 | Number of training epochs |
| \`--learning-rate\` | 0.001 | Optimizer learning rate |
| \`--batch-size\` | 32 | Training batch size |
| \`--max-token-length\` | 256 | Maximum sequence length |
| \`--precision-mode\` | fp32 | fp32, fp16, or bf16 |
| \`--optimizer-type\` | adamw | adamw, adam, or sgd |
| \`--output-dir\` | ./outputs/train | Output directory |
| \`--hidden-dim\` | 128 | Model hidden dimension |
| \`--num-layers\` | 4 | Number of transformer layers |
| \`--attention-heads\` | 4 | Attention heads per layer |
| \`--checkpoint-every-epochs\` | 0 | Save checkpoint interval (0 = off) |
| \`--resume-checkpoint-path\` | — | Resume from checkpoint |
| \`--hooks-file\` | — | Python callback hooks module |

---

## Data Formats

### SFT / LoRA JSONL

Each line is a JSON object with \`prompt\` and \`response\`, or a single \`text\` field:

\`\`\`json
{"prompt": "What is 2+2?", "response": "4"}
{"text": "The quick brown fox jumps over the lazy dog."}
\`\`\`

### Preference JSONL (DPO / ORPO)

Each line has a prompt with chosen and rejected completions:

\`\`\`json
{"prompt": "Explain X", "chosen": "Good answer...", "rejected": "Bad answer..."}
\`\`\`

### Binary Feedback JSONL (KTO)

Each line has a prompt, response, and boolean label:

\`\`\`json
{"prompt": "...", "response": "...", "label": true}
\`\`\`

### Verifiable Tasks JSONL (RLVR)

Each line has a prompt and verifiable answer or test code:

\`\`\`json
{"prompt": "What is 5! ?", "answer": "120"}
\`\`\`
`;

/** Map from TrainingMethod id to the anchor in the docs page. */
export const TRAINING_METHOD_ANCHORS: Record<string, string> = {
  train: "train",
  sft: "sft",
  "dpo-train": "dpo-train",
  "rlhf-train": "rlhf-train",
  "lora-train": "lora-train",
  distill: "distill",
  "domain-adapt": "domain-adapt",
  "grpo-train": "grpo-train",
  "qlora-train": "qlora-train",
  "kto-train": "kto-train",
  "orpo-train": "orpo-train",
  "multimodal-train": "multimodal-train",
  "rlvr-train": "rlvr-train",
};

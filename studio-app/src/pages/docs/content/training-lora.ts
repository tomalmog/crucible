import type { DocEntry } from "../docsRegistry";

export const trainingLora: DocEntry = {
  slug: "training-lora",
  title: "LoRA (Low-Rank Adaptation)",
  category: "Training",
  content: `
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
`,
};

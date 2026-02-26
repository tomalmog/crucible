import type { DocEntry } from "../docsRegistry";

export const trainingQlora: DocEntry = {
  slug: "training-qlora",
  title: "QLoRA (Quantized LoRA)",
  category: "Training",
  content: `
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
`,
};

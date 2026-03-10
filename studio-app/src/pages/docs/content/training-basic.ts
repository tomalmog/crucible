import type { DocEntry } from "../docsRegistry";

export const trainingBasic: DocEntry = {
  slug: "training-basic",
  title: "Basic Training",
  category: "Training",
  content: `
Standard supervised training from scratch. Trains a new model on raw text data from a Crucible dataset.

**Required:**
- \`--dataset\` — Crucible dataset name
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
`,
};

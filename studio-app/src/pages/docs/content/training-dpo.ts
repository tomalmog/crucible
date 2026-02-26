import type { DocEntry } from "../docsRegistry";

export const trainingDpo: DocEntry = {
  slug: "training-dpo",
  title: "DPO (Direct Preference Optimization)",
  category: "Training",
  content: `
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
`,
};

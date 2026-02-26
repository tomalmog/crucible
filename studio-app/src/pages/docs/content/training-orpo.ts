import type { DocEntry } from "../docsRegistry";

export const trainingOrpo: DocEntry = {
  slug: "training-orpo",
  title: "ORPO (Odds Ratio Preference Optimization)",
  category: "Training",
  content: `
Combines SFT and preference optimization in a single training step. Uses odds ratios to contrast chosen and rejected responses without a reference model.

**Required:**
- \`--orpo-data-path\` — Path to JSONL file with preference pairs
- \`--base-model\` — HuggingFace model ID or path
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--orpo-beta\` — Weight for the odds ratio loss component

**Data format:** Same as DPO (prompt / chosen / rejected).

**When to use:** When you want SFT + alignment in one pass. Simpler pipeline than DPO since no reference model is needed.
`,
};

import type { DocEntry } from "../docsRegistry";

export const trainingKto: DocEntry = {
  slug: "training-kto",
  title: "KTO (Kahneman-Tversky Optimization)",
  category: "Training",
  content: `
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
`,
};

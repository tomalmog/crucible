import type { DocEntry } from "../docsRegistry";

export const trainingGrpo: DocEntry = {
  slug: "training-grpo",
  title: "GRPO (Group Relative Policy Optimization)",
  category: "Training",
  content: `
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
`,
};

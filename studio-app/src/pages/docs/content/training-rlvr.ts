import type { DocEntry } from "../docsRegistry";

export const trainingRlvr: DocEntry = {
  slug: "training-rlvr",
  title: "RLVR (RL with Verifiable Rewards)",
  category: "Training",
  content: `
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
`,
};

import type { DocEntry } from "../docsRegistry";

export const trainingSft: DocEntry = {
  slug: "training-sft",
  title: "SFT (Supervised Fine-Tuning)",
  category: "Training",
  content: `
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
`,
};

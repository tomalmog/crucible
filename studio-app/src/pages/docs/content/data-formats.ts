import type { DocEntry } from "../docsRegistry";

export const dataFormats: DocEntry = {
  slug: "data-formats",
  title: "Data Formats",
  category: "Data",
  content: `
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
`,
};

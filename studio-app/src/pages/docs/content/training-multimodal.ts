import type { DocEntry } from "../docsRegistry";

export const trainingMultimodal: DocEntry = {
  slug: "training-multimodal",
  title: "Multimodal Training",
  category: "Training",
  content: `
Fine-tune vision-language models on paired image-text data. Supports image captioning, visual question answering, and other multimodal tasks.

**Required:**
- \`--multimodal-data-path\` — Path to JSONL file with image-text pairs
- \`--base-model\` — HuggingFace model ID or path
- \`--output-dir\` — Directory for output artifacts

**Data format:**
\`\`\`json
{"image_path": "/path/to/image.jpg", "text": "A cat sitting on a windowsill"}
{"image_path": "/path/to/chart.png", "prompt": "Describe this chart", "response": "The bar chart shows..."}
\`\`\`

**When to use:** Training or fine-tuning models that process both images and text (e.g., image captioning, VQA, document understanding).
`,
};

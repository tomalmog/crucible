import type { DocEntry } from "../docsRegistry";

export const trainingDistillation: DocEntry = {
  slug: "training-distillation",
  title: "Distillation",
  category: "Training",
  content: `
Transfer knowledge from a large teacher model to a smaller student model. The student learns to match the teacher's output distribution.

**Required:**
- \`--dataset\` — Forge dataset name (used as training data)
- \`--teacher-model-path\` — Path to the teacher model
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--temperature\` — Softmax temperature for soft targets (default: 2.0)
- \`--alpha\` — Weight for distillation loss vs. standard loss (default: 0.5)

**When to use:** You have a large accurate model and want a smaller, faster model that retains most of its quality. Common for deploying to edge devices or reducing inference cost.
`,
};

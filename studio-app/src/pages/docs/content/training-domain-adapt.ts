import type { DocEntry } from "../docsRegistry";

export const trainingDomainAdapt: DocEntry = {
  slug: "training-domain-adapt",
  title: "Domain Adaptation",
  category: "Training",
  content: `
Adapt a pre-trained model to a new domain by continuing training on domain-specific data, with optional drift detection to prevent catastrophic forgetting.

**Required:**
- \`--dataset\` — Crucible dataset name (domain-specific data)
- \`--base-model-path\` — Pre-trained model to adapt
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--reference-data-path\` — Reference data for drift detection (optional)

**When to use:** Taking a general-purpose model and specializing it for a specific domain (medical, legal, code, etc.) while preserving general capabilities.
`,
};

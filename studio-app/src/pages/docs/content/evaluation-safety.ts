import type { DocEntry } from "../docsRegistry";

export const evaluationSafety: DocEntry = {
  slug: "evaluation-safety",
  title: "Evaluation & Safety",
  category: "Evaluation & Safety",
  content: `
## Evaluation & Safety

Training a model is only half the job. Forge provides tools to evaluate quality and catch safety issues before deployment.

### Verify Command

Run automated checks on a trained model with a single command:

\`\`\`bash
forge verify --model <path>
\`\`\`

This runs a suite of checks: model loads correctly, generates coherent output, meets minimum quality thresholds, and passes safety filters. The output is a pass/fail report with details on each check.

### Benchmarks

Measure model quality with standard evaluation tasks. Forge computes **perplexity** on held-out data and can run accuracy benchmarks against common evaluation sets. Lower perplexity means the model is better at predicting text. Always evaluate on data the model has never seen during training.

### LLM-as-Judge

Use a stronger model to score your model's outputs. Forge sends your model's responses to a judge model that rates them on configurable criteria: helpfulness, accuracy, coherence, and safety. This provides a scalable quality signal that correlates well with human evaluation.

### Toxicity Scoring

Automated detection of harmful, biased, or toxic content in model outputs. Forge runs a dedicated toxicity classifier over a sample of generated text and reports a toxicity rate. High scores indicate the model needs additional safety training or filtering.

### Safety Gates

Define pass/fail criteria that a model must meet before it is considered ready for deployment:

- Minimum benchmark scores (e.g., perplexity below a threshold)
- Maximum toxicity rate
- Required judge scores above a minimum

If any gate fails, Forge flags the model and blocks downstream export steps. Gates are configurable per project.

### Best Practices

- **Always evaluate on held-out data** — never test on data the model trained on
- **Check for bias** — evaluate outputs across different demographic groups and input styles
- **Monitor for regression** — when updating a model, compare against the previous version's scores
- **Combine automated and human evaluation** — automated metrics catch regressions fast, but human review catches subtle quality issues
`,
};

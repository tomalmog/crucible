import type { DocEntry } from "../docsRegistry";

export const evaluationSafety: DocEntry = {
  slug: "evaluation-safety",
  title: "Evaluation & Safety",
  category: "Evaluation & Safety",
  content: `
## Evaluation & Safety

Training a model is only half the job. Forge provides tools to evaluate quality and catch safety issues before deployment.

### Benchmarks

Run standard evaluation benchmarks to measure model quality:

\`\`\`bash
forge eval --model-path ./outputs/model.pt --benchmarks mmlu,gsm8k,hellaswag
\`\`\`

Forge includes seven benchmarks, each testing a different capability:

| Benchmark | What It Measures | Scoring |
|-----------|-----------------|---------|
| **MMLU** | Broad knowledge across 57 subjects | Multiple-choice accuracy (A/B/C/D) |
| **HellaSwag** | Commonsense reasoning (sentence completion) | Lowest-perplexity completion |
| **ARC** | Grade-school science questions (ARC-Challenge) | Multiple-choice via sequence loss |
| **WinoGrande** | Pronoun resolution / commonsense | Binary choice comparison |
| **GSM8K** | Grade-school math word problems | Generated answer exact-match |
| **TruthfulQA** | Resistance to common misconceptions | Multiple-choice (MC1 format) |
| **HumanEval** | Python code generation | Sandboxed test-case execution |

Benchmark datasets are downloaded automatically from HuggingFace on first use.

### Verify Command

Run automated checks on a trained model with a single command:

\`\`\`bash
forge verify --model <path>
\`\`\`

This runs a suite of checks: model loads correctly, generates coherent output, meets minimum quality thresholds, and passes safety filters. The output is a pass/fail report with details on each check.

### LLM-as-Judge

Use a stronger model to score your model's outputs. Forge sends your model's responses to a judge model that rates them on configurable criteria: helpfulness, accuracy, coherence, and safety. This provides a scalable quality signal that correlates well with human evaluation.

\`\`\`bash
forge judge --model-path ./outputs/model.pt \\
  --judge-api https://api.openai.com/v1/chat/completions \\
  --criteria helpfulness,accuracy,safety
\`\`\`

### Toxicity Scoring

Automated detection of harmful, biased, or toxic content in model outputs. Forge runs a dedicated toxicity classifier over a sample of generated text and reports a toxicity rate. High scores indicate the model needs additional safety training or filtering.

### Safety Gates

Define pass/fail criteria that a model must meet before it is considered ready for deployment:

- Minimum benchmark scores (e.g., perplexity below a threshold)
- Maximum toxicity rate
- Required judge scores above a minimum

If any gate fails, Forge flags the model and blocks downstream export steps. Gates are configurable per project.

### Studio UI

The **Experiments** page provides three evaluation tools:

- **Evaluate** — Run benchmarks against a model checkpoint. Select the model path and choose which benchmarks to run.
- **LLM Judge** — Configure a judge API endpoint, criteria, and optional test prompts to score your model.
- **Cost** — View a summary of compute costs across all training runs.

All three tools show required fields with a \`*\` marker and validate inputs before running.

### Best Practices

- **Always evaluate on held-out data** — never test on data the model trained on
- **Check for bias** — evaluate outputs across different demographic groups and input styles
- **Monitor for regression** — when updating a model, compare against the previous version's scores
- **Combine automated and human evaluation** — automated metrics catch regressions fast, but human review catches subtle quality issues
`,
};

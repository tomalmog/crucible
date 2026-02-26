import type { DocEntry } from "../docsRegistry";

export const gettingStarted: DocEntry = {
  slug: "getting-started",
  title: "Getting Started",
  category: "Getting Started",
  content: `
## Getting Started with Forge

Get from zero to a trained model in under five minutes.

### 1. Install Forge

\`\`\`bash
pip install forge-ml
\`\`\`

Verify the installation:

\`\`\`bash
forge --version
\`\`\`

### 2. Create Your First Dataset

Point Forge at a directory of training data. It accepts CSV, JSONL, Parquet, and plain text files.

\`\`\`bash
forge ingest --source ./my-data --name my-dataset
\`\`\`

Forge auto-detects the file format, validates the records, and stores everything in a versioned \`.forge/\` directory.

### 3. Run Your First Training

\`\`\`bash
forge train --dataset my-dataset --output-dir ./outputs
\`\`\`

This launches a basic training run with sensible defaults. You will see live progress in your terminal including loss, learning rate, and estimated time remaining.

### 4. Check Results

When training completes, your output directory contains:

- **Model weights** — the trained model checkpoint
- **Training logs** — per-step metrics (loss, learning rate, throughput)
- **Metrics summary** — final evaluation scores and training configuration

### 5. Next Steps

- **Fine-tune a pre-trained model** — try SFT to adapt an existing model to your instructions: \`forge train --method sft --model <base-model> --dataset my-dataset\`
- **Explore Forge Studio** — launch the desktop UI for visual dataset browsing, training configuration, and live monitoring
- **Browse the docs** — check out the Training, Data, and Concepts sections for deeper coverage
`,
};

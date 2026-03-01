import type { DocEntry } from "../docsRegistry";

export const experimentTracking: DocEntry = {
  slug: "experiment-tracking",
  title: "Experiment Tracking",
  category: "Training",
  content: `
## Experiment Tracking

Forge logs metrics to local JSONL files by default. You can optionally stream metrics to **Weights & Biases** or **TensorBoard** for richer visualization and team collaboration.

### Weights & Biases

Pass \`--wandb-project\` to any training command to log metrics to a W&B project:

\`\`\`bash
forge train --dataset my-data --wandb-project my-project
\`\`\`

This creates a new W&B run that tracks:
- Training loss and validation loss per epoch
- All hyperparameters (learning rate, batch size, architecture, etc.)
- Run metadata (method, dataset, output directory)

**Requirements:** Install the \`wandb\` package (\`pip install wandb\`) and authenticate with \`wandb login\`. Forge will raise a clear error if the package is missing.

### TensorBoard

Pass \`--tensorboard-dir\` to write TensorBoard event files:

\`\`\`bash
forge train --dataset my-data --tensorboard-dir ./tb_logs
\`\`\`

Then launch the TensorBoard UI:

\`\`\`bash
tensorboard --logdir ./tb_logs
\`\`\`

Metrics logged: \`train_loss\` and \`validation_loss\` as scalars at each epoch step.

**Requirements:** Install \`tensorboard\` (\`pip install tensorboard\`). The \`torch.utils.tensorboard.SummaryWriter\` is used under the hood.

### Using Both

You can enable both at the same time:

\`\`\`bash
forge train --dataset my-data \\
  --wandb-project my-project \\
  --tensorboard-dir ./tb_logs
\`\`\`

### Studio UI

In the Training wizard, expand the **Experiment Tracking** section under Advanced to set W&B project name and TensorBoard directory. Both fields are optional — leave them empty to use local-only tracking.

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| \`--wandb-project\` | — | W&B project name (enables W&B logging) |
| \`--tensorboard-dir\` | — | Directory for TensorBoard event files |

### Local Tracking

Even without external integrations, Forge always logs to local JSONL files under the data root. The Experiments page in Studio reads these logs to show run history, metric comparisons, and cost summaries.
`,
};

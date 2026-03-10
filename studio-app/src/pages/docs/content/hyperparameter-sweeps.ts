import type { DocEntry } from "../docsRegistry";

export const hyperparameterSweeps: DocEntry = {
  slug: "hyperparameter-sweeps",
  title: "Hyperparameter Sweeps",
  category: "Training",
  content: `
## Hyperparameter Sweeps

A sweep automates trying different hyperparameter combinations so you can find the configuration that produces the best model without manually re-running training dozens of times.

### How It Works

1. Define which parameters to vary and their candidate values
2. Choose a search strategy (grid or random)
3. Crucible runs one training trial per parameter combination
4. Results are ranked by a metric you choose (e.g. validation loss)
5. The best trial and its parameters are reported

### CLI Usage

\`\`\`bash
crucible sweep \\
  --dataset my-dataset \\
  --output-dir ./outputs/sweep \\
  --params '{"parameters": [{"name": "learning_rate", "values": [0.001, 0.01, 0.1]}]}' \\
  --strategy grid \\
  --metric validation_loss \\
  --json
\`\`\`

You can also define parameters in a YAML file:

\`\`\`yaml
# sweep_config.yaml
parameters:
  - name: learning_rate
    values: [0.001, 0.01, 0.1]
  - name: batch_size
    values: [8, 16, 32]
\`\`\`

\`\`\`bash
crucible sweep --dataset my-dataset --output-dir ./outputs/sweep --config-file sweep_config.yaml
\`\`\`

### Options

| Option | Default | Description |
|--------|---------|-------------|
| \`--dataset\` | — | **Required.** Dataset to train on |
| \`--output-dir\` | — | **Required.** Base directory for trial outputs |
| \`--params\` | — | Inline JSON parameter definitions |
| \`--config-file\` | — | YAML file with parameter definitions |
| \`--strategy\` | grid | \`grid\` (exhaustive) or \`random\` (sampled) |
| \`--max-trials\` | 10 | Maximum trials for random search |
| \`--metric\` | validation_loss | Metric to optimize |
| \`--maximize\` | false | Maximize metric instead of minimizing |
| \`--json\` | false | Output results as JSON |

### Search Strategies

**Grid Search** tries every combination of parameter values. If you have 3 learning rates and 3 batch sizes, that is 9 trials. Best for small search spaces.

**Random Search** samples parameter combinations randomly up to \`--max-trials\`. Better than grid search when the search space is large, because it explores more of the space without exponential cost.

### Sweep Parameters

The following parameters can be swept. Values are provided as comma-separated lists:

- \`learning_rate\` — optimizer learning rate
- \`batch_size\` — training batch size
- \`hidden_dim\` — model hidden dimension
- \`num_layers\` — number of transformer layers
- \`attention_heads\` — attention heads per layer
- \`dropout\` — dropout rate
- \`weight_decay\` — optimizer weight decay
- \`mlp_hidden_dim\` — MLP feedforward dimension

### Studio UI

Open **Training → Sweep** in Studio to configure sweeps visually. The form lets you add and remove parameters, pick a strategy, and run the sweep. Results appear in a table with the best trial highlighted.
`,
};

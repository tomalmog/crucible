import type { DocEntry } from "../docsRegistry";

export const trainingCommonOptions: DocEntry = {
  slug: "training-common-options",
  title: "Common Options",
  category: "Training",
  content: `
These options are available across all training methods:

| Option | Default | Description |
|--------|---------|-------------|
| \`--epochs\` | 3 | Number of training epochs |
| \`--learning-rate\` | 0.001 | Optimizer learning rate |
| \`--batch-size\` | 32 | Training batch size |
| \`--max-token-length\` | 256 | Maximum sequence length |
| \`--precision-mode\` | fp32 | fp32, fp16, or bf16 |
| \`--optimizer-type\` | adamw | adamw, adam, or sgd |
| \`--output-dir\` | ./outputs/train | Output directory |
| \`--hidden-dim\` | 128 | Model hidden dimension |
| \`--num-layers\` | 4 | Number of transformer layers |
| \`--attention-heads\` | 4 | Attention heads per layer |
| \`--checkpoint-every-epochs\` | 0 | Save checkpoint interval (0 = off) |
| \`--resume-checkpoint-path\` | — | Resume from checkpoint |
| \`--hooks-file\` | — | Python callback hooks module |
`,
};

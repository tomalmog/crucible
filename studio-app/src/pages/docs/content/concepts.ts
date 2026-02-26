import type { DocEntry } from "../docsRegistry";

export const concepts: DocEntry = {
  slug: "concepts",
  title: "Key Concepts",
  category: "Concepts",
  content: `
## Key Concepts

A plain-language reference for the core terms you will encounter throughout Forge.

### Epochs

One epoch is a single full pass through the entire training dataset. More epochs give the model more chances to learn from the data, but too many can cause overfitting. Most fine-tuning jobs use 1 to 5 epochs.

### Learning Rate

Controls how much the model adjusts its weights after each batch. A high learning rate learns fast but can overshoot and become unstable. A low learning rate is more stable but takes longer to converge. Typical values range from 1e-5 to 5e-4.

### Batch Size

The number of training examples processed together in one step. Larger batches produce smoother gradient estimates and can train faster, but require more GPU memory. If you run out of memory, reduce the batch size first.

### Loss

A number that measures how wrong the model's predictions are. Training is the process of minimizing loss. A decreasing loss curve means the model is learning. If loss suddenly spikes or goes to NaN, something is wrong — check your learning rate and data.

### Checkpoints

Saved snapshots of the model weights taken during training. Checkpoints let you resume a run that was interrupted or roll back to an earlier state. Forge saves checkpoints at regular intervals so you can pick the best-performing version.

### LoRA / Adapters

Instead of updating every weight in the model, LoRA (Low-Rank Adaptation) trains a small set of extra parameters that are layered on top of the frozen base model. This dramatically reduces memory usage and training time while achieving comparable quality for most tasks.

### Tokenizer

Converts raw text into a sequence of integer token IDs that the model can process. Each model family has its own tokenizer with a specific vocabulary. Forge automatically loads the correct tokenizer for your chosen base model.

### Precision (fp32 / fp16 / bf16)

The numerical format used to store model weights. Full precision (fp32) uses 4 bytes per value. Half precision (fp16 or bf16) uses 2 bytes, cutting memory usage roughly in half with minimal impact on quality. bf16 is generally preferred on modern GPUs because it handles large values more gracefully.

### Gradient Clipping

Caps the magnitude of gradients during training to prevent extremely large updates that can destabilize the model. Forge enables gradient clipping by default with a max norm of 1.0. This is one of the most important safeguards against training divergence.

### Overfitting

When a model memorizes the training data instead of learning general patterns. Signs include very low training loss but poor performance on new data. Combat overfitting with more training data, fewer epochs, dropout regularization, or early stopping.
`,
};

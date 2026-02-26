import type { DocEntry } from "../docsRegistry";

export const trainingRlhf: DocEntry = {
  slug: "training-rlhf",
  title: "RLHF (Reinforcement Learning from Human Feedback)",
  category: "Training",
  content: `
Train a policy model using PPO with a reward model. The classic alignment approach: train a reward model on preferences, then optimize the policy against it.

**Required:**
- \`--policy-model-path\` — Path to the policy model weights
- \`--output-dir\` — Directory for output artifacts

**Key options:**
- \`--reward-model-path\` — Path to trained reward model
- \`--train-reward-model\` — Train a reward model first from preference data
- \`--preference-data-path\` — JSONL with preference pairs (for reward model training)
- \`--clip-epsilon\` — PPO clip range (default: 0.2)
- \`--ppo-epochs\` — PPO update epochs per batch (default: 4)
- \`--entropy-coeff\` — Entropy bonus coefficient (default: 0.01)

**When to use:** Full RLHF pipeline. Use DPO instead for a simpler alternative that doesn't require a separate reward model.
`,
};

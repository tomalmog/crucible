import type { DocEntry } from "../docsRegistry";

export const trainingOverview: DocEntry = {
  slug: "training-overview",
  title: "Choosing a Method",
  category: "Training",
  content: `
## Choosing a Training Method

Not sure which method to use? Start here.

### Quick Decision Guide

**Starting from scratch (no pre-trained model)?**
→ Use **Basic Training** to train a model on raw text data.

**Have a pre-trained model and instruction data?**
→ Use **SFT** (Supervised Fine-Tuning) for general instruction following.

**Want to align a model with human preferences?**
- Have paired chosen/rejected data → **DPO** (simplest) or **ORPO** (SFT + alignment in one pass)
- Have binary thumbs-up/down feedback → **KTO**
- Want the classic RLHF pipeline with a reward model → **RLHF**

**Need parameter-efficient fine-tuning (limited GPU memory)?**
- Standard → **LoRA** (~0.1% of parameters trained)
- Extreme memory constraints → **QLoRA** (4-bit quantization + LoRA)

**Specialized tasks:**
- Compress a large model into a smaller one → **Distillation**
- Adapt a general model to a specific domain → **Domain Adaptation**
- Tasks with verifiable answers (math, code) → **RLVR**
- Train with a reward function → **GRPO**
- Image + text tasks → **Multimodal**

### Method Comparison

| Method | Needs Pre-trained Model | Data Type | GPU Memory | Complexity |
|--------|------------------------|-----------|------------|------------|
| Basic Training | No | Raw text | Medium | Low |
| SFT | Yes | Prompt/response pairs | High | Low |
| DPO | Yes | Preference pairs | High | Medium |
| RLHF | Yes | Preference pairs | Very High | High |
| LoRA | Yes | Text or prompt/response | Low | Low |
| QLoRA | Yes | Text or prompt/response | Very Low | Low |
| Distillation | Yes (teacher + student) | Raw text | High | Medium |
| Domain Adapt | Yes | Domain text | High | Low |
| GRPO | Yes | Prompts + reward fn | High | High |
| KTO | Yes | Binary feedback | High | Medium |
| ORPO | Yes | Preference pairs | High | Medium |
| Multimodal | Yes | Image-text pairs | Very High | High |
| RLVR | Yes | Verifiable tasks | High | High |
`,
};

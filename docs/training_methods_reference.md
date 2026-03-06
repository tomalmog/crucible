# Training Methods Reference

This document is the authoritative reference for how each training method in Forge
should work. Use it to verify correctness and catch regressions.

---

## 1. Basic Train (`train`)

**Purpose:** Train a language model from scratch on raw text data.

| Aspect | Correct Implementation |
|--------|----------------------|
| Data format | `DataRecord` objects with `.text` field |
| Tokenization | BPE/vocabulary tokenizer fitted on data |
| Loss | `CrossEntropyLoss(ignore_index=0)` — pad token excluded |
| Targets | Next-token prediction: input shifted by 1 |
| Prompt masking | No — training from scratch, all tokens are targets |
| Default LR | 1e-3 (training from scratch) |
| Gradient clip | Yes, max_norm=1.0 (via shared training loop) |

**Status:** Correct.

---

## 2. SFT (`sft`)

**Purpose:** Supervised fine-tuning on instruction (prompt/response) data.

| Aspect | Correct Implementation |
|--------|----------------------|
| Data format | JSONL: `{"prompt": "...", "response": "..."}` |
| Tokenization | Prompt + response concatenated, then tokenized |
| Loss | `CrossEntropyLoss(ignore_index=-100)` |
| Targets | Response tokens only; prompt tokens masked with -100 |
| Prompt masking | **Yes** — critical for SFT |
| Default LR | 2e-5 (fine-tuning) |
| Gradient clip | Yes, max_norm=1.0 (via shared training loop) |

**Status:** Correct. Uses `build_sft_loss_function()` with `IGNORE_INDEX = -100`.

---

## 3. DPO (`dpo-train`)

**Purpose:** Direct Preference Optimization using chosen/rejected pairs.

| Aspect | Correct Implementation |
|--------|----------------------|
| Data format | JSONL: `{"prompt": "...", "chosen": "...", "rejected": "..."}` |
| Algorithm | log_ratio = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected), loss = -log(sigmoid(beta * log_ratio)) |
| Reference model | Frozen copy of initial policy |
| Loss | DPO loss with label smoothing |
| Prompt masking | Yes — log probs computed only on response tokens |
| Default LR | 5e-5 |
| Gradient clip | Yes, max_norm=1.0 |

**Status:** Correct.

---

## 4. RLHF (`rlhf-train`)

**Purpose:** Reinforcement Learning from Human Feedback via PPO.

| Aspect | Correct Implementation |
|--------|----------------------|
| Data format | DataRecords for tokenizer + reward model config |
| Algorithm | PPO with clipped surrogate objective + value loss + entropy bonus |
| Models | Policy, reward, reference (frozen), value head |
| Loss | PPO surrogate + value MSE + entropy |
| Default LR | 1e-5 (very sensitive to LR) |
| Gradient clip | Yes, max_norm=1.0 |

**Status:** Correct. Default LR fixed to 1e-5, gradient clipping added to PPO loop.

---

## 5. LoRA (`lora-train`)

**Purpose:** Low-rank adaptation — freeze base, train small adapter matrices.

| Aspect | Correct Implementation |
|--------|----------------------|
| Data format | JSONL: `{"prompt": "...", "response": "..."}` |
| Algorithm | Inject LoRA A/B matrices into target linear layers |
| Loss | `CrossEntropyLoss(ignore_index=-100)` with prompt masking |
| Prompt masking | **Yes** — same as SFT |
| Default LR | 2e-4 (higher than SFT because fewer params) |
| Gradient clip | Yes, max_norm=1.0 |
| NaN detection | Yes |

**Status:** Correct.

---

## 6. QLoRA (`qlora-train`)

**Purpose:** Quantized LoRA — 4-bit quantization + LoRA adapters.

| Aspect | Correct Implementation |
|--------|----------------------|
| Data format | JSONL: `{"prompt": "...", "response": "..."}` |
| Algorithm | Same as LoRA but with quantized base weights |
| Loss | `CrossEntropyLoss(ignore_index=-100)` with prompt masking |
| Prompt masking | **Yes** — same as SFT |
| Default LR | 2e-4 |
| Gradient clip | Yes, max_norm=1.0 (via shared training loop) |

**Status:** Correct.

---

## 7. Distillation (`distill`)

**Purpose:** Knowledge distillation — train student to match teacher's outputs.

| Aspect | Correct Implementation |
|--------|----------------------|
| Data format | DataRecords for tokenizer + teacher model path |
| Algorithm | Blended loss: alpha * KL(student, teacher) * T^2 + (1-alpha) * CE(student, labels) |
| Loss | KL divergence + cross entropy |
| Default LR | 5e-5 |
| Gradient clip | Yes, max_norm=1.0 |

**Status:** Correct. Gradient clipping added to distillation loop.

---

## 8. Domain Adaptation (`domain-adapt`)

**Purpose:** Continue pretraining on domain-specific text data.

| Aspect | Correct Implementation |
|--------|----------------------|
| Data format | DataRecords with domain text |
| Algorithm | Standard next-token prediction (like basic train) |
| Loss | `CrossEntropyLoss(ignore_index=0)` |
| Default LR | 5e-5 (lower than from-scratch to avoid forgetting) |
| Drift monitoring | Optional reference data perplexity tracking |

**Status:** Correct. Uses shared training loop.

---

## 9. GRPO (`grpo-train`)

**Purpose:** Group Relative Policy Optimization — sample groups of responses, score with reward, compute group-relative advantages, update policy.

| Aspect | Correct Implementation |
|--------|----------------------|
| Data format | JSONL: `{"prompt": "..."}` |
| Algorithm | For each prompt: generate G responses, score with reward fn, compute advantages = (r - mean) / std, update policy with advantage-weighted log prob |
| Loss | Advantage-weighted policy gradient with KL penalty |
| Default LR | 5e-5 |
| Pad exclusion | `CrossEntropyLoss(ignore_index=0)` |

**Status:** Operational. Core training pipeline works with correct LR, pad exclusion, and gradient clipping.

**Future enhancement — full DeepSeek-R1 algorithm:**
- Online response generation (generate G responses per prompt during training)
- Reward function scoring integration (code exists in `grpo_reward.py` and `grpo_batch_processing.py`)
- Group-relative advantage computation: `advantages = (rewards - mean) / std`
- Advantage-weighted policy gradient loss instead of plain CE
- KL penalty against reference model

---

## 10. KTO (`kto-train`)

**Purpose:** Kahneman-Tversky Optimization — learn from unpaired binary feedback.

| Aspect | Correct Implementation |
|--------|----------------------|
| Data format | JSONL: `{"prompt": "...", "response": "...", "is_desirable": true/false}` |
| Algorithm | Asymmetric loss: desirable examples increase log prob, undesirable decrease it |
| Loss | Desirable: standard CE. Undesirable: negated gradient scaled by beta * undesirable_weight |
| Default LR | 5e-5 |

**Status:** Core asymmetric loss implemented. Desirability is encoded in targets (undesirable targets set to -1), and the loss function detects this to apply asymmetric weighting with beta and desirable/undesirable weight parameters.

**TODO — would improve correctness:**
- Reference model for KL anchor (current implementation uses simplified asymmetric CE without KL term)
- The full KTO paper loss is: `loss_d = -log(sigmoid(beta * (log_pi - log_ref)))` for desirable, `loss_u = -log(sigmoid(beta * (log_ref - log_pi)))` for undesirable

---

## 11. ORPO (`orpo-train`)

**Purpose:** Odds Ratio Preference Optimization — SFT + preference in single pass.

| Aspect | Correct Implementation |
|--------|----------------------|
| Data format | JSONL: `{"prompt": "...", "chosen": "...", "rejected": "..."}` |
| Algorithm | L = L_sft(chosen) + lambda * -log_sigmoid(beta * log_odds_ratio(chosen vs rejected)) |
| Loss | SFT cross-entropy on chosen + odds ratio preference term |
| Default LR | 5e-5 |

**Status:** Implemented. Batch builder interleaves chosen (even indices) and rejected (odd indices). Loss computes SFT on chosen tokens + lambda * log-sigmoid preference term comparing average log-probs of chosen vs rejected.

---

## 12. Multimodal (`multimodal-train`)

**Purpose:** Vision-language fine-tuning with image+text pairs.

| Aspect | Correct Implementation |
|--------|----------------------|
| Data format | JSONL: `{"text": "...", "image_path": "..."}` |
| Algorithm | Encode images with vision encoder, project to text space, train on text conditioned on image features |
| Default LR | 2e-5 |
| Pad exclusion | `CrossEntropyLoss(ignore_index=0)` |

**Status:** Operational with text data pipeline. Correct LR and pad exclusion.

**Future enhancement — vision encoder pipeline:**
- Image loading and preprocessing pipeline
- Vision encoder integration (e.g., CLIP ViT)
- Cross-modal projection layer (vision features → text embedding space)
- `image_path` field is currently loaded from JSONL but not used in training
- `image_encoder`, `image_size`, `projection_dim` options exist in `MultimodalOptions` but are unused

---

## 13. RLVR (`rlvr-train`)

**Purpose:** RL with Verifiable Rewards — train on code/math with automated verification.

| Aspect | Correct Implementation |
|--------|----------------------|
| Data format | JSONL: `{"prompt": "...", "solution": "..."}` |
| Algorithm | Generate solutions, verify correctness, reward signal from verification, policy gradient update |
| Default LR | 5e-5 |
| Pad exclusion | `CrossEntropyLoss(ignore_index=0)` |

**Status:** Operational with SFT on concatenated prompt+solution. Correct LR and pad exclusion.

**Future enhancement — verification framework:**
- Solution generation (model generates candidate solutions during training)
- Code/math verifier integration (`verifier_type` option exists but is unused)
- Reward signal from verification (`reward_correct`, `reward_incorrect` exist but unused)
- Policy gradient update weighted by verification reward
- `max_verification_attempts` retry logic

---

## Audit History

### Fixed in audit (2025-02-25)
- Default learning rates: GRPO (5e-5), KTO (5e-5), ORPO (5e-5), RLHF (1e-5), Multimodal (2e-5), RLVR (5e-5)
- Gradient clipping added to: shared training loop, gradient accumulation, PPO trainer, distillation loop
- KTO loss rewritten with asymmetric desirable/undesirable weighting
- ORPO loss rewritten with odds-ratio preference computation
- Pad token exclusion (`ignore_index=0`) added to GRPO, Multimodal, RLVR
- UI learning rate overrides added for Multimodal and RLVR

### Fixed in prior sessions
- DPO: autograd graph (torch.gather), gradient clipping, learning rate, NaN loss fix
- LoRA: prompt masking via SFT pipeline, gradient clipping, NaN detection
- QLoRA: prompt masking via SFT pipeline

### Future enhancements (all methods are operational)
1. **GRPO**: Full DeepSeek-R1 algorithm — online generation, reward scoring, advantage-weighted policy gradient
2. **Multimodal**: Vision encoder pipeline — image processing, CLIP ViT, cross-modal projection
3. **RLVR**: Verification framework — code/math verifier, reward-based policy gradient
4. **KTO**: Reference model for KL anchor (currently uses simplified asymmetric CE)

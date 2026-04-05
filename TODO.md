# TODO

## Technical Debt — Needs Design Decisions

### 1. Real GRPO/RLVR with Reward Functions

**Problem:** GRPO and RLVR currently fall back to SFT with a printed warning. They're not doing reinforcement learning.

**What the real implementations need:**
- **GRPO** needs a reward function that scores each generation. trl's `GRPOTrainer` expects a callable `reward_fn(completions) -> scores`. It generates completions during training (online RL), which is significantly slower than SFT.
- **RLVR** is GRPO where the reward comes from comparing generations against a known `solution` field (math-style verification).

**Decisions needed:**
- Do we implement a **generic reward function interface** (user provides a Python function)? Or hardcode common strategies (exact match, contains answer, length penalty)?
- For RLVR — is string matching against the `solution` field sufficient, or do we want numeric comparison for math problems?
- trl's `GRPOTrainer` generates completions during training (online RL). This is much slower than SFT. Is that acceptable, or do we want an offline approximation?
- Do we need reward model support (separate model that scores completions), or just rule-based reward functions for now?


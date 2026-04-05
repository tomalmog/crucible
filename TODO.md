# TODO

## Technical Debt — Needs Design Decisions

### 1. Streaming Ingest for Large Datasets

**Problem:** `input_reader.py` loads the entire file into memory. Multi-GB datasets will OOM.

**Current behavior:** JSONL, Parquet, and CSV files are read in full before writing to Lance storage.

**Options:**
- **JSONL:** Read and yield one record at a time, write to Lance in batches (e.g., 10k records per batch). Python file iteration is already line-by-line, just need to batch the Lance writes instead of collecting everything first.
- **Parquet:** pyarrow supports reading in row groups natively — wire that up instead of `.read().to_pydict()` which materializes everything.
- **CSV:** Python's csv module is already line-by-line, just batch the writes.

**Decision needed:**
- What's the target file size? 1-2GB or 10GB+? Determines whether batched reads are sufficient or if we need a full out-of-core pipeline with progress reporting.
- Should we add a progress bar / record count during ingest?

---

### 2. Real GRPO/RLVR with Reward Functions

**Problem:** GRPO and RLVR currently fall back to SFT with a printed warning. They're not doing reinforcement learning.

**What the real implementations need:**
- **GRPO** needs a reward function that scores each generation. trl's `GRPOTrainer` expects a callable `reward_fn(completions) -> scores`. It generates completions during training (online RL), which is significantly slower than SFT.
- **RLVR** is GRPO where the reward comes from comparing generations against a known `solution` field (math-style verification).

**Decisions needed:**
- Do we implement a **generic reward function interface** (user provides a Python function)? Or hardcode common strategies (exact match, contains answer, length penalty)?
- For RLVR — is string matching against the `solution` field sufficient, or do we want numeric comparison for math problems?
- trl's `GRPOTrainer` generates completions during training (online RL). This is much slower than SFT. Is that acceptable, or do we want an offline approximation?
- Do we need reward model support (separate model that scores completions), or just rule-based reward functions for now?

---

## New Features — Ready to Build

### 3. Custom Eval Benchmarks
Let users define their own evaluation tasks (question/answer pairs) beyond the 7 hardcoded benchmarks. High interest from labs.

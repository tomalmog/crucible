# AGENTS.md — Forge Codebase Standards

> This file governs all AI-assisted code generation in this repository.
> Every agent (Copilot, Claude, Cursor, etc.) MUST follow these rules.
> If a rule conflicts with "getting it done fast," the rule wins.

---

## Core Philosophy

This codebase will be read by humans, debugged by humans, and maintained for years. Write code that a new engineer can understand on their first day. Clever code is bad code. Clear code is fast code.
ean
**The Three Questions — ask before writing anything:**
1. Does this function/module already exist somewhere in the codebase? (Search first.)
2. Will a human understand this in 6 months without context?
3. If this breaks at 3am, can someone trace the error to the root cause in under 5 minutes?

If the answer to any of these is "no," rewrite it.

---

## Project Structure

```
forge/
├── src/
│   ├── core/           # Shared types, interfaces, errors, constants
│   ├── ingest/         # Data ingestion pipeline
│   ├── store/          # Storage layer, versioning, metadata catalog
│   ├── transforms/     # Dedup, quality scoring, PII, filtering
│   ├── serve/          # DataLoader, streaming, GPU-native serving
│   ├── studio/         # Web UI backend
│   └── cli/            # CLI entry points
├── tests/
│   ├── unit/           # Mirror src/ structure exactly
│   ├── integration/    # Cross-module tests
│   └── fixtures/       # Shared test data — NEVER generate inline
├── scripts/            # One-off operational scripts (NOT business logic)
├── docs/               # Architecture decisions, API docs
└── benchmarks/         # Performance regression tests
```

**Rules:**
- Every directory has an `__init__.py` (Python) or `mod.rs` (Rust) that documents what the module does in a docstring.
- No file exceeds 300 lines. If it does, split it. No exceptions.
- No function exceeds 50 lines. If it does, extract helpers.
- No function has more than 4 parameters. Use a config/options object instead.

---

## The DRY Contract

**Before writing ANY new function, class, or utility:**

1. Run a search across the codebase for similar functionality.
2. Check `src/core/` for shared utilities.
3. If something similar exists, extend it or refactor it. Do NOT create a parallel version.

**If you find yourself writing any of these, STOP:**
- A second HTTP client wrapper
- A second logging setup
- A second config parser
- A second retry/backoff utility
- A second path resolution helper
- Any function that starts with `custom_`, `my_`, `new_`, or `v2_`

**Shared code lives in `src/core/`. Not in the module that happened to need it first.**

---

## Naming Conventions

**Files:** `snake_case.py` / `snake_case.rs`. Name describes what it DOES, not what it IS.
- Good: `deduplicate_minhash.py`, `stream_to_gpu.py`
- Bad: `utils.py`, `helpers.py`, `misc.py`, `common.py`

**The word "utils" is banned.** If you can't name the file, you don't understand the abstraction. Split it into files with real names.

**Functions:** verb_noun. Say what it does.
- Good: `compute_quality_score()`, `stream_batch_to_device()`
- Bad: `process()`, `handle()`, `do_thing()`, `run()`

**Variables:** No single-letter variables outside of list comprehensions and loop counters. No abbreviations unless they are universally understood in the domain (e.g., `gpu`, `cpu`, `url`, `pii`).

**Constants:** `UPPER_SNAKE_CASE`, defined in `src/core/constants.py`. Never hardcode magic numbers or strings inline.

---

## Error Handling

**Every error must be traceable to its source.**

- Define custom exception classes in `src/core/errors.py`. Group by domain:
  ```python
  class ForgeIngestError(ForgeError): ...
  class ForgeStoreError(ForgeError): ...
  class ForgeTransformError(ForgeError): ...
  ```
- NEVER catch bare `Exception` unless you re-raise with context.
- NEVER silently swallow errors. Every `except` block must log or re-raise.
- Error messages must include: what happened, what input caused it, and what the user/developer should do about it.
  ```python
  # Bad
  raise ValueError("Invalid input")
  
  # Good
  raise ForgeIngestError(
      f"Failed to parse document at {file_path}: "
      f"expected UTF-8 encoding, got {detected_encoding}. "
      f"Re-run with --encoding={detected_encoding} or convert the file first."
  )
  ```

**Logging:**
- Use structured logging (`structlog` in Python). No f-string print statements for debugging.
- Log levels mean something: DEBUG for internal state, INFO for operations completing, WARNING for recoverable issues, ERROR for failures.
- Every log line must include enough context to understand it WITHOUT reading the code.

---

## Testing Requirements

**No PR merges without tests. No exceptions. No "I'll add tests later."**

### Coverage Rules
- Every public function has at least one unit test.
- Every error path has a test that triggers it.
- Every bug fix includes a regression test BEFORE the fix (red-green-refactor).

### Test Structure
```python
def test_<what>_<condition>_<expected>():
    """One sentence: what this test verifies."""
    # Arrange — set up inputs
    # Act — call the function
    # Assert — check ONE thing
```

- Test names describe behavior, not implementation: `test_dedup_removes_exact_duplicates`, not `test_dedup_function`.
- One assert per test. If you need multiple asserts, you need multiple tests.
- No test depends on another test. Every test runs in isolation.
- Test data lives in `tests/fixtures/`. Never generate fake data inline — it hides assumptions.
- Mock external services (S3, databases, GPUs). Never hit real infrastructure in unit tests.
- Integration tests go in `tests/integration/` and are tagged to run separately.

### What to Test
- Happy path with typical input
- Edge cases: empty input, single element, maximum size
- Error cases: bad input types, missing files, network failures
- Boundary conditions: exactly at limits (e.g., 300-line file, max batch size)

---

## Type Safety

**All code is fully typed. No `Any`. No untyped function signatures.**

Python:
- Every function has complete type annotations (params and return).
- Use `TypedDict` or `dataclass` for structured data. No raw dicts for domain objects.
- Run `mypy --strict` in CI. It must pass.

Rust:
- Avoid `.unwrap()` in production code. Use `?` operator or explicit error handling.
- No `unsafe` blocks without a comment explaining why and a tracking issue to remove it.

---

## Configuration & Environment

- All configuration flows through a single, validated config object defined in `src/core/config.py`.
- Environment variables are read in ONE place and nowhere else. No `os.getenv()` scattered through business logic.
- Every config value has a default, a type, and a docstring explaining what it does.
- Secrets are never logged, never in error messages, never in stack traces.

---

## Documentation Rules

**Code should be self-documenting. Comments explain WHY, not WHAT.**

```python
# Bad — restates the code
# Increment counter by 1
counter += 1

# Good — explains intent
# We retry up to 3 times because S3 returns transient 503s
# under high write load (observed in prod 2026-01)
for attempt in range(MAX_RETRIES):
```

**Every module** has a docstring at the top explaining:
1. What this module does (one sentence)
2. How it fits into the larger system
3. Key assumptions or constraints

**Every public function** has a docstring explaining:
1. What it does
2. What it returns
3. What exceptions it raises
4. Non-obvious behavior or side effects

**Architecture Decision Records (ADRs):**
When making a significant technical decision (new dependency, architectural change, protocol choice), create a doc in `docs/adr/` with:
- Context: what problem are we solving?
- Decision: what did we choose?
- Alternatives considered: what did we reject and why?
- Consequences: what are the tradeoffs?

---

## Dependency Management

- Every new dependency requires justification. "It was easy" is not justification.
- Before adding a dependency, check: Can we do this in <50 lines ourselves? Is this library actively maintained? Does it have a compatible license? Does it pull in a large transitive dependency tree?
- Pin exact versions in lockfiles. No floating ranges in production.
- Audit dependencies quarterly for security and maintenance status.

---

## Performance & Scalability Rules

**This is a data platform. Performance is a feature, not an optimization.**

- No O(n²) algorithms on datasets. If you're nesting loops over data, stop and find a better approach.
- Profile before optimizing. Add benchmarks in `benchmarks/` for any performance-critical path.
- Streaming over loading: never load an entire dataset into memory if you can process it as a stream.
- Every database query has an explain plan reviewed before merging.
- Batch I/O operations. Never make N network calls when 1 batch call works.
- All operations that touch data at scale must be checkpointed and resumable. If a 4-hour transform fails at hour 3, it must restart from the last checkpoint, not from scratch.

---

## Git & PR Discipline

**Commits:**
- One logical change per commit. Not "fix stuff" or "updates."
- Commit message format: `<module>: <imperative verb> <what changed>`
  - Good: `ingest: add retry logic for S3 multipart uploads`
  - Bad: `fixed bug`, `wip`, `asdf`

**Pull Requests:**
- Every PR has a description explaining: what changed, why, and how to test it.
- PRs should be reviewable in under 20 minutes. If your PR is >400 lines, break it up.
- No commented-out code in PRs. Delete it. Git remembers.
- No TODO comments without a linked issue. Orphan TODOs are where code goes to die.

---

## Anti-Patterns — Instant Rejection

The following patterns will be rejected on sight. Do not generate them:

| Pattern | Why It's Bad | Do This Instead |
|---|---|---|
| `utils.py` / `helpers.py` | Junk drawer that grows forever | Name the file after its purpose |
| Bare `except: pass` | Hides bugs | Catch specific exceptions, log them |
| Copy-pasted code blocks | Maintenance nightmare | Extract a shared function |
| Global mutable state | Untestable, race conditions | Pass state explicitly via params |
| String concatenation for SQL/queries | Injection risk, unreadable | Use parameterized queries |
| `import *` | Pollutes namespace, hides dependencies | Import specific names |
| Nested callbacks >2 deep | Unreadable | Use async/await or extract functions |
| Boolean params that change behavior | Confusing API surface | Use separate functions or enums |
| Commented-out code | Dead weight | Delete it, git has history |
| Functions with side effects AND return values | Unpredictable | Separate commands from queries |
| Hardcoded file paths | Breaks across environments | Use config or path resolution |
| `time.sleep()` in production code | Fragile timing | Use proper async/retry patterns |

---

## AI Agent-Specific Rules

These rules apply specifically to AI coding assistants working in this repo:

1. **Search before writing.** Before creating any new file, function, or class, search the existing codebase. If you skip this step and create a duplicate, the code will be rejected.

2. **Follow existing patterns.** Look at how similar things are done elsewhere in the repo. Match the style, structure, and conventions. Consistency beats personal preference.

3. **Generate tests alongside code.** Every new function gets a test in the same PR. Not later. Now. Write the test first if possible.

4. **Don't refactor what you weren't asked to touch.** If you're fixing a bug in `ingest/`, don't reorganize `store/`. Scope changes tightly.

5. **Explain non-obvious decisions in comments.** If you chose an algorithm, data structure, or approach that isn't the obvious first choice, leave a comment explaining why.

6. **No placeholder implementations.** Don't generate functions that just `pass` or `raise NotImplementedError` unless explicitly asked for a stub. Every function should work.

7. **No demo/example quality code.** This is production code. No `# TODO: handle errors`, no `# This is a simplified version`, no shortcuts.

8. **Run the linter and type checker mentally.** Before outputting code, verify: Are all types annotated? Are all imports used? Are there any bare exceptions? Is anything over 300 lines?

9. **When in doubt, ask.** If the requirements are ambiguous, ask for clarification rather than guessing. A wrong implementation is worse than a delayed one.

10. **Leave the codebase better than you found it.** If you notice a small issue adjacent to your change (missing type annotation, unclear variable name), fix it. But keep the scope small.

---

## Enforcement

- CI runs: `mypy --strict`, `ruff` (linting + formatting), `pytest` with coverage threshold (80% minimum, 95% target).
- Pre-commit hooks enforce formatting and import ordering.
- Coverage below 80% blocks merge.
- Any file over 300 lines blocks merge.
- Any function over 50 lines triggers a warning.

---

*Last updated: February 2026*
*This document is enforced, not aspirational. If the code doesn't meet these standards, it doesn't ship.*
